"""Known proof-stress curves and resumable CPU scans exercise the yield workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import openmmpolymer.yielding as yielding
from openmmpolymer.protocols import RunManifest, standard_melt_equilibration
from openmmpolymer.trajectory import AnalysisError
from openmmpolymer.yielding import (
    DEFORM_STEM,
    WORKFLOW_NAME,
    YieldError,
    YieldSpec,
    analyse_yield,
    run_yield_scan,
    write_yield_report,
    yield_protocol,
    yield_scan,
    yield_schedule,
    yield_stages,
)

from .helpers import QUICK_EQUILIBRATION, write_deformation

QUICK = YieldSpec(
    temperature_k=120.0,
    max_strain=0.024,
    relax_ps=0.05,
    n_replicas=2,
    samples_per_step=2,
    stage_ps=0.2,
    fit_max_strain=0.0125,
)
PLANTED = replace(QUICK, temperature_k=298.15, relax_ps=1.0, stage_ps=4.0)


def _plant(directory: Path, *, spec: YieldSpec = PLANTED) -> None:
    """Two elastic/plateau curves cross at strain 0.018 and stresses 18, 21.6."""
    directory.mkdir(parents=True, exist_ok=True)
    strains = (1.0 + spec.strain_increment) ** np.arange(1, 13) - 1.0
    lateral_stretch = 1.0 - 0.2 * strains
    nominal = np.minimum(1000.0 * strains + 2.0, 18.0)
    stages: dict[str, dict[str, Any]] = {}
    for replica in range(spec.n_replicas):
        stresses_bar = (
            nominal * (1.0 + 0.2 * replica) / lateral_stretch**2 - 0.1
        ) / 0.1
        for chunk in range(3):
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
    RunManifest(protocol="yield", seed=11, stages=stages).save(directory)
    record = {
        "request": {"spec": asdict(spec)},
        "reference_box_nm": [5.0] * 3,
        "replica_stages": [
            [f"{DEFORM_STEM}_r{replica}_{chunk:03d}" for chunk in range(3)]
            for replica in range(spec.n_replicas)
        ],
        "steps_per_replica": 12,
        "timestep_fs": 2.0,
    }
    (directory / WORKFLOW_NAME).write_text(json.dumps(record))


def test_compounded_schedule_cost_and_chunk_references() -> None:
    spec = replace(QUICK, max_strain=0.05, relax_ps=50.0, stage_ps=500.0)
    schedule = yield_schedule(spec)
    assert schedule.n_steps == 25
    assert 1.002**24 - 1.0 < spec.max_strain <= schedule.max_strain
    assert schedule.total_ps == 1250.0
    assert schedule.strain_rate_per_ns == pytest.approx(schedule.max_strain / 1.25)
    stages = yield_protocol(spec, replica=2, reference_box_nm=(4.0, 5.0, 6.0)).stages
    assert [stage.name for stage in stages] == [
        "06_yield_r2_000",
        "06_yield_r2_001",
        "06_yield_r2_002",
    ]
    assert [stage.options["n_steps"] for stage in stages] == [10, 10, 5]
    assert [stage.options["new_velocities"] for stage in stages] == [True, False, False]
    assert [stage.options["strain_start"] for stage in stages] == pytest.approx(
        [1.002**done - 1.0 for done in (0, 10, 20)]
    )
    assert all(stage.options["reference_box_nm"] == [4.0, 5.0, 6.0] for stage in stages)
    settle = standard_melt_equilibration(
        target_temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        **QUICK_EQUILIBRATION,
    )
    assert yield_scan(spec, **QUICK_EQUILIBRATION).total_duration_ps == pytest.approx(
        settle.total_duration_ps + spec.n_replicas * schedule.total_ps
    )


def test_trajectory_recording_is_optional() -> None:
    assert "trajectory" not in yield_protocol(QUICK).stages[0].options
    stage = yield_protocol(replace(QUICK, trajectory_ps=0.01)).stages[0]
    assert stage.options["trajectory"].format == "xtc"
    assert stage.options["trajectory"].interval_ps == 0.01


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("temperature_k", 0.0),
        ("temperature_k", np.nan),
        ("pressure_bar", -1.0),
        ("strain_increment", 0.0),
        ("strain_increment", 0.024),
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
        ("offset_strain", 0.0),
        ("offset_strain", np.nan),
        ("offset_strain", 0.024),
        ("fit_min_strain", -0.001),
        ("fit_min_strain", np.inf),
        ("fit_min_strain", 0.0125),
        ("fit_max_strain", 0.0),
        ("fit_max_strain", 0.024),
        ("trajectory_ps", 0.0),
        ("max_total_ns", -1.0),
    ],
)
def test_invalid_controls_fail_before_building_a_protocol(
    name: str, value: Any
) -> None:
    with pytest.raises((ValueError, TypeError), match=name):
        replace(QUICK, **{name: value})


def test_replica_yield_strengths_are_fitted_separately_and_averaged(
    tmp_path: Path,
) -> None:
    _plant(tmp_path)
    write_deformation(tmp_path, stage="06_deform_r0_00")
    write_deformation(tmp_path, stage="06_breaking_r0_000")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    report = analyse_yield(tmp_path)
    assert report.resolved
    assert [curve.n_points for curve in report.curves] == [12, 12]
    assert [fit.strength_mpa for fit in report.replicas] == pytest.approx([18.0, 21.6])
    assert [fit.yield_strain for fit in report.replicas] == pytest.approx([0.018] * 2)
    assert report.strength_mpa == pytest.approx(19.8)
    assert report.replica_spread_mpa == pytest.approx(np.std([18.0, 21.6], ddof=1))
    assert report.replicas[0].strain_rate_per_ns == pytest.approx(
        yield_schedule(PLANTED).strain_rate_per_ns
    )
    assert len(yield_stages(tmp_path)) == 6
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("remove", ["replica", "first_chunk", "last_chunk", "workflow"])
def test_incomplete_scans_keep_curves_without_a_headline(
    tmp_path: Path, remove: str
) -> None:
    _plant(tmp_path)
    if remove == "workflow":
        (tmp_path / WORKFLOW_NAME).unlink()
    else:
        manifest = RunManifest.load(tmp_path)
        assert manifest is not None
        names = (
            [f"06_yield_r1_{i:03d}" for i in range(3)]
            if remove == "replica"
            else [f"06_yield_r1_{'000' if remove == 'first_chunk' else '002'}"]
        )
        for name in names:
            del manifest.stages[name]
        manifest.save(tmp_path)
    report = analyse_yield(tmp_path)
    assert not report.resolved
    assert report.strength_mpa is None
    assert report.replica_spread_mpa is None
    assert report.curves[0].n_points == 12
    assert any("incomplete" in note for note in report.notes)
    if remove != "replica":
        assert not report.replicas[-1].resolved
        assert report.replicas[-1].strength_mpa is None
        assert report.replicas[-1].yield_bracket is None


def test_truncated_replica_cannot_resolve_even_after_crossing(tmp_path: Path) -> None:
    _plant(tmp_path, spec=replace(PLANTED, n_replicas=1))
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    samples = manifest.stages["06_yield_r0_002"]["samples"]
    for key in samples:
        if key.startswith("segment_"):
            samples[key] = samples[key][:-1]
    manifest.save(tmp_path)
    report = analyse_yield(tmp_path)
    assert report.curves[0].n_points == 11
    assert not report.replicas[0].resolved
    assert report.replicas[0].strength_mpa is None
    assert report.replicas[0].yield_strain is None
    assert report.replicas[0].yield_bracket is None


def test_single_replica_has_no_replica_spread(tmp_path: Path) -> None:
    _plant(tmp_path, spec=replace(PLANTED, n_replicas=1))
    report = analyse_yield(tmp_path)
    assert report.resolved
    assert report.strength_mpa == pytest.approx(18.0)
    assert report.replica_spread_mpa is None


def test_missing_first_replica_keeps_the_recorded_replica_number(
    tmp_path: Path,
) -> None:
    _plant(tmp_path)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    for name in list(manifest.stages):
        if name.startswith("06_yield_r0_"):
            del manifest.stages[name]
    manifest.save(tmp_path)
    report = analyse_yield(tmp_path)
    assert report.replica_indices == (1,)
    assert not report.resolved
    files = write_yield_report(report)
    assert [Path(path).name for path in files.figures] == ["yield_r1.png"]
    assert json.loads(Path(files.json).read_text())["replica_indices"] == [1]


def test_saved_offset_controls_reanalysis_and_unresolved_json(tmp_path: Path) -> None:
    _plant(tmp_path, spec=replace(PLANTED, offset_strain=0.02))
    report = analyse_yield(tmp_path)
    assert report.offset_strain == 0.02
    assert not report.resolved
    files = write_yield_report(report, formats=())
    record = json.loads(Path(files.json).read_text())
    assert json.dumps(record, allow_nan=False)
    assert record["strength_mpa"] is None
    assert record["replicas"][0]["yield_strain"] is None
    assert files.figures == ()


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
def test_inconsistent_chunks_are_rejected(
    tmp_path: Path, field: str, value: list[float]
) -> None:
    _plant(tmp_path)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    manifest.stages["06_yield_r0_001"]["samples"][field] = value
    manifest.save(tmp_path)
    with pytest.raises(AnalysisError):
        analyse_yield(tmp_path)


def test_no_yield_data_has_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="No yield-strength"):
        analyse_yield(tmp_path)
    write_deformation(tmp_path)
    with pytest.raises(AnalysisError, match="No yield-strength"):
        analyse_yield(tmp_path)


def test_report_writes_strict_json_and_per_replica_figures(tmp_path: Path) -> None:
    _plant(tmp_path)
    files = write_yield_report(analyse_yield(tmp_path), formats=("png", "svg"))
    record = json.loads(Path(files.json).read_text())
    assert json.dumps(record, allow_nan=False)
    assert record["strength_mpa"] == pytest.approx(19.8)
    assert record["replicas"][0]["modulus_mpa"] == pytest.approx(1000.0)
    assert record["replicas"][0]["nominal_stress_mpa"][-1] == pytest.approx(18.0)
    assert record["curves"][0]["temperature_k"] == PLANTED.temperature_k
    assert len(files.figures) == 4
    assert all(Path(path).stat().st_size > 0 for path in files.figures)


def test_budget_refuses_before_creating_output(argon_scan_run: Any) -> None:
    with pytest.raises(YieldError, match="max_total_ns"):
        run_yield_scan(
            argon_scan_run,
            "run",
            spec=replace(QUICK, max_total_ns=1e-6),
            **QUICK_EQUILIBRATION,
        )
    assert not Path("run").exists()


@pytest.mark.parametrize(
    "change", ["criterion", "replicas", "system", "coordinates", "box"]
)
def test_interrupted_resume_fingerprints_request_and_physical_system(
    argon_scan_run: Any,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("interrupted")

    monkeypatch.setattr(yielding, "run_protocol", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_yield_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    assert Path("run", WORKFLOW_NAME).is_file()
    assert not Path("run/manifest.json").exists()
    changed = QUICK
    if change == "criterion":
        changed = replace(QUICK, offset_strain=0.003)
    elif change == "replicas":
        changed = replace(QUICK, n_replicas=1)
    elif change == "system":
        argon_scan_run.system_xml += "\n"
    elif change == "coordinates":
        argon_scan_run.box.positions_nm[0, 0] += 0.001
    else:
        argon_scan_run.box.box_nm = (2.81, 2.8, 2.8)
    with pytest.raises(YieldError, match="different settings"):
        run_yield_scan(argon_scan_run, "run", spec=changed, **QUICK_EQUILIBRATION)
    assert calls == 1


@pytest.mark.parametrize("protocol", ["foreign", "yield"])
def test_foreign_stages_require_a_fresh_directory(
    argon_scan_run: Any,
    tmp_path: Path,
    protocol: str,
) -> None:
    write_deformation(tmp_path)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    manifest.protocol = protocol
    manifest.save(tmp_path)
    with pytest.raises(YieldError, match="fresh directory"):
        run_yield_scan(argon_scan_run, tmp_path, spec=QUICK, **QUICK_EQUILIBRATION)
    assert not (tmp_path / WORKFLOW_NAME).exists()


def test_missing_state_is_refused_before_new_dynamics(
    argon_scan_run: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(yielding, "run_protocol", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_yield_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    RunManifest(
        protocol="yield",
        seed=11,
        stages={
            "00_minimise": {"final_state": "run/missing.xml"},
        },
    ).save("run")
    before = Path("run/manifest.json").read_bytes()
    with pytest.raises(YieldError, match="missing state files"):
        run_yield_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    assert Path("run/manifest.json").read_bytes() == before


def test_real_scan_branches_replicas_and_preserves_stages_on_resume(
    argon_scan_run: Any,
) -> None:
    first = run_yield_scan(
        argon_scan_run, "run", spec=QUICK, resume=False, **QUICK_EQUILIBRATION
    )
    manifest = RunManifest.load("run")
    assert manifest is not None
    assert len(manifest.stages) == 12
    assert "00_minimise" in manifest.stages and "05_npt" in manifest.stages
    assert [curve.n_points for curve in first.curves] == [12, 12]
    record = json.loads(Path("run", WORKFLOW_NAME).read_text())
    assert Path(record["start_state"]).name.startswith("05_npt")
    for curve in first.curves:
        assert curve.strain == pytest.approx(1.002 ** np.arange(1, 13) - 1.0)
    for name, stage in manifest.stages.items():
        if name.startswith(DEFORM_STEM):
            samples = stage["samples"]
            assert samples["reference_box_nm"] == pytest.approx(
                record["reference_box_nm"]
            )
            assert np.asarray(samples["segment_box_z_nm"]) / samples[
                "reference_box_nm"
            ][2] - 1.0 == pytest.approx(samples["segment_strain"])
    assert not np.array_equal(first.curves[0].stress_mpa, first.curves[1].stress_mpa)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in Path("run").iterdir()
        if path.is_file()
    }
    resumed = run_yield_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    for old, new in zip(first.curves, resumed.curves, strict=True):
        assert new.stress_mpa == pytest.approx(old.stress_mpa)
    for path, (content, modified) in before.items():
        assert path.read_bytes() == content
        if path.name not in (WORKFLOW_NAME, "manifest.json"):
            assert path.stat().st_mtime_ns == modified
