"""Known tensile curves test the criterion; short CPU runs test its plumbing."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import openmmpolymer.elongation as elongation
from openmmpolymer.elongation import (
    DEFORM_STEM,
    WORKFLOW_NAME,
    ElongationError,
    ElongationSpec,
    analyse_elongation,
    elongation_protocol,
    elongation_scan,
    elongation_schedule,
    elongation_stages,
    run_elongation_scan,
    write_elongation_report,
)
from openmmpolymer.protocols import RunManifest, standard_melt_equilibration
from openmmpolymer.trajectory import AnalysisError

from .helpers import QUICK_EQUILIBRATION, write_deformation

QUICK = ElongationSpec(
    temperature_k=120.0,
    strain_increment=0.002,
    max_strain=0.012,
    relax_ps=0.05,
    n_replicas=2,
    samples_per_step=2,
    stage_ps=0.1,
)
PLANTED = ElongationSpec(
    strain_increment=0.1,
    max_strain=1.1,
    relax_ps=1.0,
    stage_ps=4.0,
    n_replicas=2,
)


def _plant(directory: Path, *, spec: ElongationSpec = PLANTED) -> None:
    """Write two chunked curves whose nominal peaks are 100 and 120, with different stress-loss endpoints."""
    directory.mkdir(parents=True, exist_ok=True)
    strains = (1.0 + spec.strain_increment) ** np.arange(1, 9) - 1.0
    lateral_stretch = 1.0 - 0.2 * strains
    nominal = np.asarray([0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0])
    stages: dict[str, dict[str, Any]] = {}
    for replica in range(spec.n_replicas):
        stresses_bar = (
            nominal * (1.0 + 0.2 * replica) / lateral_stretch**2 - 0.1
        ) / 0.1
        if replica == 1:
            # Cross the threshold a full hold earlier in the second replica.
            stresses_bar[4] = (45.0 / lateral_stretch[4] ** 2 - 0.1) / 0.1
        for chunk in range(2):
            part = slice(chunk * 4, (chunk + 1) * 4)
            stages[f"{DEFORM_STEM}_r{replica}_{chunk:03d}"] = {
                "mean_temperature_k": spec.temperature_k,
                "samples": {
                    "segment_strain": strains[part].tolist(),
                    "segment_stress_xx_bar": [-1.0] * 4,
                    "segment_stress_yy_bar": [-1.0] * 4,
                    "segment_stress_zz_bar": stresses_bar[part].tolist(),
                    "segment_box_x_nm": (5.0 * lateral_stretch[part]).tolist(),
                    "segment_box_y_nm": (5.0 * lateral_stretch[part]).tolist(),
                    "segment_box_z_nm": (5.0 * (1.0 + strains[part])).tolist(),
                    "segment_duration_ps": [spec.relax_ps] * 4,
                    "reference_box_nm": [5.0] * 3,
                    "deform_axis": [float(spec.axis)],
                },
            }
    RunManifest(protocol="elongation", seed=11, stages=stages).save(directory)
    record = {
        "request": {"spec": asdict(spec)},
        "reference_box_nm": [5.0] * 3,
        "replica_stages": [
            [f"{DEFORM_STEM}_r{replica}_{chunk:03d}" for chunk in range(2)]
            for replica in range(spec.n_replicas)
        ],
        "steps_per_replica": 8,
        "timestep_fs": 2.0,
    }
    (directory / WORKFLOW_NAME).write_text(json.dumps(record))


def test_compounded_ladder_reaches_its_target_and_reports_its_rate() -> None:
    spec = replace(QUICK, max_strain=0.05, relax_ps=50.0, stage_ps=500.0)
    schedule = elongation_schedule(spec)
    assert schedule.n_steps == 25
    assert (1.002) ** 24 - 1.0 < spec.max_strain <= schedule.max_strain
    assert schedule.total_ps == 1250.0
    assert schedule.strain_rate_per_ns == pytest.approx(schedule.max_strain / 1.25)


def test_chunks_preserve_the_strain_reference_and_only_initialize_velocity_once() -> (
    None
):
    spec = replace(QUICK, max_strain=0.05, relax_ps=50.0, stage_ps=500.0)
    reference = (4.0, 5.0, 6.0)
    stages = elongation_protocol(spec, replica=2, reference_box_nm=reference).stages
    assert [stage.name for stage in stages] == [
        "06_elongation_r2_000",
        "06_elongation_r2_001",
        "06_elongation_r2_002",
    ]
    assert [stage.options["n_steps"] for stage in stages] == [10, 10, 5]
    assert [stage.options["new_velocities"] for stage in stages] == [True, False, False]
    assert [stage.options["strain_start"] for stage in stages] == pytest.approx(
        [(1.002) ** done - 1.0 for done in (0, 10, 20)]
    )
    assert all(stage.options["reference_box_nm"] == list(reference) for stage in stages)
    other = elongation_protocol(spec, replica=3, reference_box_nm=reference).stages
    assert [stage.options for stage in other] == [stage.options for stage in stages]
    assert [stage.name for stage in other] != [stage.name for stage in stages]


def test_full_schedule_cost_includes_equilibration_and_every_replica() -> None:
    protocol = elongation_scan(QUICK, **QUICK_EQUILIBRATION)
    equilibration = standard_melt_equilibration(
        target_temperature_k=QUICK.temperature_k,
        pressure_bar=QUICK.pressure_bar,
        **QUICK_EQUILIBRATION,
    )
    assert protocol.total_duration_ps == pytest.approx(
        equilibration.total_duration_ps
        + QUICK.n_replicas * elongation_schedule(QUICK).total_ps
    )
    assert len(protocol.stages) == len(equilibration.stages) + 6


def test_trajectory_recording_is_explicitly_optional() -> None:
    assert "trajectory" not in elongation_protocol(QUICK).stages[0].options
    stage = elongation_protocol(replace(QUICK, trajectory_ps=0.01)).stages[0]
    assert stage.options["trajectory"].format == "xtc"
    assert stage.options["trajectory"].interval_ps == 0.01


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("temperature_k", 0.0),
        ("temperature_k", np.nan),
        ("pressure_bar", -1.0),
        ("strain_increment", 0.0),
        ("strain_increment", 0.012),
        ("max_strain", 0.001),
        ("relax_ps", np.inf),
        ("stage_ps", 0.01),
        ("n_replicas", 0),
        ("n_replicas", 1.5),
        ("axis", -1),
        ("axis", 3),
        ("axis", True),
        ("samples_per_step", 1),
        ("samples_per_step", 2.5),
        ("confirmation_steps", 1),
        ("confirmation_steps", True),
        ("failure_fraction", 0.0),
        ("failure_fraction", 1.0),
        ("failure_fraction", np.nan),
        ("trajectory_ps", 0.0),
        ("max_total_ns", -1.0),
    ],
)
def test_invalid_controls_are_rejected_before_a_protocol_is_built(
    name: str, value: Any
) -> None:
    with pytest.raises((ValueError, TypeError), match=name):
        replace(QUICK, **{name: value})


def test_planted_replicas_are_grouped_and_elongation_is_averaged(
    tmp_path: Path,
) -> None:
    _plant(tmp_path)
    # An unrelated modulus extension must never enter the strength analysis.
    write_deformation(tmp_path, stage="06_deform_r0_00")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    report = analyse_elongation(tmp_path)
    assert report.resolved
    assert len(report.curves) == len(report.replicas) == 2
    assert [curve.n_points for curve in report.curves] == [8, 8]
    assert [fit.elongation_percent for fit in report.replicas] == pytest.approx(
        [100.0 * (1.1**6 - 1.0), 100.0 * (1.1**5 - 1.0)]
    )
    expected = [100.0 * (1.1**6 - 1.0), 100.0 * (1.1**5 - 1.0)]
    assert report.elongation_percent == pytest.approx(np.mean(expected))
    assert report.replica_spread_percent == pytest.approx(np.std(expected, ddof=1))
    assert report.replicas[0].break_stress_mpa == pytest.approx(35.0)
    assert report.replicas[0].strain_rate_per_ns == pytest.approx(
        elongation_schedule(PLANTED).strain_rate_per_ns
    )
    assert len(elongation_stages(tmp_path)) == 4
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("remove", ["replica", "first_chunk", "last_chunk", "workflow"])
def test_incomplete_scans_keep_peaks_but_do_not_report_a_headline_elongation(
    tmp_path: Path, remove: str
) -> None:
    _plant(tmp_path)
    if remove == "workflow":
        (tmp_path / WORKFLOW_NAME).unlink()
    else:
        manifest = RunManifest.load(tmp_path)
        assert manifest is not None
        names = (
            ["06_elongation_r1_000", "06_elongation_r1_001"]
            if remove == "replica"
            else [f"06_elongation_r1_{'000' if remove == 'first_chunk' else '001'}"]
        )
        for name in names:
            del manifest.stages[name]
        manifest.save(tmp_path)
    report = analyse_elongation(tmp_path)
    assert not report.resolved
    assert report.elongation_percent is None
    assert report.replica_spread_percent is None
    assert report.replicas[0].peak_stress_mpa == pytest.approx(100.0)
    assert any("incomplete" in note for note in report.notes)
    if remove == "workflow":
        assert all(not fit.resolved for fit in report.replicas)
        assert all(fit.elongation_percent is None for fit in report.replicas)
        assert all(fit.break_bracket is None for fit in report.replicas)


def test_a_truncated_replica_cannot_resolve_even_after_a_sufficient_stress_drop(
    tmp_path: Path,
) -> None:
    _plant(tmp_path, spec=replace(PLANTED, n_replicas=1, confirmation_steps=2))
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    samples = manifest.stages["06_elongation_r0_001"]["samples"]
    for key in samples:
        if key.startswith("segment_"):
            samples[key] = samples[key][:-1]
    manifest.save(tmp_path)
    report = analyse_elongation(tmp_path)
    fit = report.replicas[0]
    assert report.curves[0].n_points == 7
    assert not fit.resolved
    assert fit.peak_stress_mpa == pytest.approx(100.0)
    assert fit.elongation_percent is None
    assert fit.strain_at_break is None
    assert fit.break_stress_mpa is None
    assert fit.break_bracket is None
    assert any("Incomplete replica" in note for note in fit.notes)


def test_single_replica_has_no_estimated_replica_spread(tmp_path: Path) -> None:
    _plant(tmp_path, spec=replace(PLANTED, n_replicas=1))
    report = analyse_elongation(tmp_path)
    assert report.resolved
    assert report.elongation_percent == pytest.approx(100.0 * (1.1**6 - 1.0))
    assert report.replica_spread_percent is None


def test_saved_criterion_controls_reanalysis(tmp_path: Path) -> None:
    _plant(tmp_path, spec=replace(PLANTED, failure_fraction=0.25))
    report = analyse_elongation(tmp_path)
    assert report.failure_fraction == 0.25
    assert not report.resolved
    assert all(not fit.resolved for fit in report.replicas)


@pytest.mark.parametrize(
    "damage", ["request", "spec", "criterion", "extent", "steps", "chunks"]
)
def test_partial_or_inconsistent_workflow_cannot_invent_a_resolved_break(
    tmp_path: Path, damage: str
) -> None:
    _plant(tmp_path, spec=replace(PLANTED, n_replicas=3, failure_fraction=0.25))
    assert not analyse_elongation(tmp_path).resolved
    path = tmp_path / WORKFLOW_NAME
    record = json.loads(path.read_text())
    if damage == "request":
        del record["request"]
    elif damage == "spec":
        del record["request"]["spec"]
    elif damage == "criterion":
        del record["request"]["spec"]["failure_fraction"]
    elif damage == "extent":
        record["request"]["spec"]["max_strain"] = 4.0
    elif damage == "steps":
        record["steps_per_replica"] = 7
    else:
        record["replica_stages"][0].pop()
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="workflow record"):
        analyse_elongation(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_box_nm", [5.0, 5.0, 6.0]),
        ("deform_axis", [0.0]),
        ("segment_strain", [0.4, 0.3, 0.2, 0.1]),
        ("segment_stress_zz_bar", [1.0, 2.0]),
        ("segment_box_x_nm", [5.0, 5.0, 5.0, -5.0]),
    ],
)
def test_inconsistent_recorded_chunks_are_rejected(
    tmp_path: Path, field: str, value: list[float]
) -> None:
    _plant(tmp_path)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    manifest.stages["06_elongation_r0_001"]["samples"][field] = value
    manifest.save(tmp_path)
    with pytest.raises(AnalysisError):
        analyse_elongation(tmp_path)


def test_no_elongation_data_has_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="No elongation-at-break"):
        analyse_elongation(tmp_path)
    write_deformation(tmp_path)
    with pytest.raises(AnalysisError, match="No elongation-at-break"):
        analyse_elongation(tmp_path)


def test_report_writes_strict_json_and_a_figure_for_each_replica(
    tmp_path: Path,
) -> None:
    _plant(tmp_path)
    report = analyse_elongation(tmp_path)
    files = write_elongation_report(report, formats=("png", "svg"))
    record = json.loads(Path(files.json).read_text())
    assert json.dumps(record, allow_nan=False)
    assert record["elongation_percent"] == pytest.approx(50.0 * (1.1**6 + 1.1**5 - 2.0))
    assert record["replicas"][0]["nominal_stress_mpa"] == pytest.approx(
        [0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0]
    )
    assert record["replicas"][0]["strain_rate_per_ns"] > 0.0
    assert record["curves"][0]["temperature_k"] == PLANTED.temperature_k
    assert len(files.figures) == 4
    assert all(Path(path).stat().st_size > 0 for path in files.figures)


def test_unresolved_report_uses_json_null_and_can_skip_figures(tmp_path: Path) -> None:
    _plant(tmp_path, spec=replace(PLANTED, failure_fraction=0.25))
    files = write_elongation_report(analyse_elongation(tmp_path), formats=())
    record = json.loads(Path(files.json).read_text())
    assert json.dumps(record, allow_nan=False)
    assert record["elongation_percent"] is None
    assert record["replica_spread_percent"] is None
    assert record["replicas"][0]["strain_at_break"] is None
    assert files.figures == ()


def test_budget_refuses_a_scan_before_creating_any_output(
    argon_scan_run: Any,
) -> None:
    with pytest.raises(ElongationError, match="max_total_ns"):
        run_elongation_scan(
            argon_scan_run,
            "run",
            spec=replace(QUICK, max_total_ns=1e-6),
            **QUICK_EQUILIBRATION,
        )
    assert not Path("run").exists()


def test_an_interrupted_request_cannot_resume_with_different_settings(
    argon_scan_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic interruption before first stage")

    monkeypatch.setattr(elongation, "run_protocol", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_elongation_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    assert Path("run", WORKFLOW_NAME).is_file()
    assert not Path("run/manifest.json").exists()
    with pytest.raises(ElongationError, match="different settings"):
        run_elongation_scan(
            argon_scan_run,
            "run",
            spec=replace(QUICK, n_replicas=1),
            **QUICK_EQUILIBRATION,
        )
    assert calls == 1


@pytest.mark.parametrize("change", ["system", "coordinates", "box"])
def test_resume_fingerprints_the_physical_system_and_initial_configuration(
    argon_scan_run: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(elongation, "run_protocol", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_elongation_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    if change == "system":
        argon_scan_run.system_xml += "\n"
    elif change == "coordinates":
        argon_scan_run.box.positions_nm[0, 0] += 0.001
    else:
        argon_scan_run.box.box_nm = (2.81, 2.8, 2.8)
    with pytest.raises(ElongationError, match="different settings"):
        run_elongation_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)


@pytest.mark.parametrize("protocol", ["foreign", "elongation"])
def test_foreign_stages_require_a_fresh_directory(
    argon_scan_run: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
) -> None:
    write_deformation(tmp_path)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    manifest.protocol = protocol
    manifest.save(tmp_path)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A foreign manifest must be rejected before dynamics.")

    monkeypatch.setattr(elongation, "run_protocol", unexpected)
    with pytest.raises(ElongationError, match="fresh directory"):
        run_elongation_scan(argon_scan_run, tmp_path, spec=QUICK, **QUICK_EQUILIBRATION)
    assert not (tmp_path / WORKFLOW_NAME).exists()


def test_missing_completed_state_is_refused_before_new_dynamics(
    argon_scan_run: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("interrupted")

    monkeypatch.setattr(elongation, "run_protocol", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_elongation_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    RunManifest(
        protocol="elongation",
        seed=11,
        stages={"00_minimise": {"final_state": "run/missing.xml"}},
    ).save("run")
    before = Path("run/manifest.json").read_bytes()
    with pytest.raises(ElongationError, match="missing state files"):
        run_elongation_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    assert calls == 1
    assert Path("run/manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "damage", ["equilibration_gap", "replica_gap", "early_replica", "unexpected"]
)
def test_resume_cannot_mix_new_predecessors_with_saved_descendants(
    argon_scan_run: Any, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("interrupted")

    monkeypatch.setattr(elongation, "run_protocol", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_elongation_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    settled = [
        stage.name
        for stage in elongation_scan(QUICK, **QUICK_EQUILIBRATION).stages
        if not stage.name.startswith(DEFORM_STEM)
    ]
    replica = [stage.name for stage in elongation_protocol(QUICK).stages]
    names = {
        "equilibration_gap": settled[1:2],
        "replica_gap": [*settled, replica[1]],
        "early_replica": [settled[0], replica[0]],
        "unexpected": ["foreign_stage"],
    }[damage]
    state = Path("run/existing.xml")
    state.write_text("Readable state whose contents must never reach dynamics.")
    RunManifest(
        protocol="elongation",
        seed=11,
        stages={name: {"final_state": str(state)} for name in names},
    ).save("run")
    before = {path: path.read_bytes() for path in Path("run").iterdir()}
    with pytest.raises(ElongationError, match="Cannot resume"):
        run_elongation_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    assert calls == 1
    assert {path: path.read_bytes() for path in Path("run").iterdir()} == before


def test_real_scan_preserves_every_stage_on_explicit_rerun_and_resume(
    argon_scan_run: Any,
) -> None:
    first = run_elongation_scan(
        argon_scan_run, "run", spec=QUICK, resume=False, **QUICK_EQUILIBRATION
    )
    manifest = RunManifest.load("run")
    assert manifest is not None
    assert len(manifest.stages) == 12
    assert "00_minimise" in manifest.stages
    assert "05_npt" in manifest.stages
    assert len(first.curves) == len(first.replicas) == 2
    assert [curve.n_points for curve in first.curves] == [6, 6]
    record = json.loads(Path("run", WORKFLOW_NAME).read_text())
    assert Path(record["start_state"]).name.startswith("05_npt")
    expected = (1.002) ** np.arange(1, 7) - 1.0
    for curve in first.curves:
        assert curve.strain == pytest.approx(expected)
    for name, stage in manifest.stages.items():
        if not name.startswith(DEFORM_STEM):
            continue
        samples = stage["samples"]
        assert samples["reference_box_nm"] == pytest.approx(record["reference_box_nm"])
        assert np.asarray(samples["segment_box_z_nm"]) / samples["reference_box_nm"][
            2
        ] - 1.0 == pytest.approx(samples["segment_strain"])
    assert not np.array_equal(first.curves[0].stress_mpa, first.curves[1].stress_mpa)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in Path("run").iterdir()
        if path.is_file()
    }
    resumed = run_elongation_scan(
        argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION
    )
    assert [fit.peak_stress_mpa for fit in resumed.replicas] == pytest.approx(
        [fit.peak_stress_mpa for fit in first.replicas]
    )
    for path, (content, modified) in before.items():
        assert path.read_bytes() == content
        if path.name not in (WORKFLOW_NAME, "manifest.json"):
            assert path.stat().st_mtime_ns == modified


def test_missing_first_replica_preserves_ids_in_json_and_figures(
    tmp_path: Path,
) -> None:
    _plant(tmp_path)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    for name in list(manifest.stages):
        if name.startswith("06_elongation_r0_"):
            del manifest.stages[name]
    manifest.save(tmp_path)
    report = analyse_elongation(tmp_path)
    assert report.replica_indices == (1,)
    assert not report.resolved
    files = write_elongation_report(report)
    assert [Path(path).name for path in files.figures] == ["elongation_r1.png"]
    assert json.loads(Path(files.json).read_text())["replica_indices"] == [1]


def test_public_workflow_api_is_exported() -> None:
    import openmmpolymer as polymer

    for name in (
        "ElongationAtBreak",
        "ElongationError",
        "ElongationSpec",
        "ElongationSchedule",
        "ElongationReport",
        "elongation_at_break",
        "elongation_schedule",
        "elongation_protocol",
        "elongation_scan",
        "elongation_stages",
        "run_elongation_scan",
        "analyse_elongation",
        "write_elongation_report",
        "plot_elongation_at_break",
    ):
        assert name in polymer.__all__
        assert getattr(polymer, name) is not None
