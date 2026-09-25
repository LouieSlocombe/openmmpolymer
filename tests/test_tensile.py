"""Planted scans test how the tensile engine reads; short CPU runs its plumbing.

The engine is shared, so its behaviour is checked for every measurement; what
each one reads off its curves is checked against the curves planted for it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import openmmpolymer
import openmmpolymer.tensile as tensile
from openmmpolymer.protocols import RunManifest, standard_melt_equilibration
from openmmpolymer.tensile import (
    BreakingError,
    BreakingSpec,
    ElongationError,
    ElongationSpec,
    TensileSpec,
    YieldError,
    YieldSpec,
    analyse_breaking,
    analyse_elongation,
    analyse_yield,
    breaking_stages,
    elongation_stages,
    run_breaking_scan,
    run_elongation_scan,
    run_yield_scan,
    tensile_protocol,
    tensile_scan,
    tensile_schedule,
    write_breaking_report,
    write_elongation_report,
    write_yield_report,
    yield_stages,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    PLANTED_TENSILE,
    QUICK_EQUILIBRATION,
    snapshot_files,
    write_deformation,
    write_tensile_scan,
)

MEASUREMENTS = ("breaking", "elongation", "yield")

#: Each measurement's entry points, and the report fields its headline uses.
API = {
    "breaking": SimpleNamespace(
        run=run_breaking_scan,
        stages=breaking_stages,
        analyse=analyse_breaking,
        write=write_breaking_report,
        error=BreakingError,
        value="strength_mpa",
        spread="replica_spread_mpa",
        event=(
            "strength_mpa",
            "failure_strain",
            "failure_stress_mpa",
            "failure_bracket",
        ),
        label="breaking-strength",
    ),
    "elongation": SimpleNamespace(
        run=run_elongation_scan,
        stages=elongation_stages,
        analyse=analyse_elongation,
        write=write_elongation_report,
        error=ElongationError,
        value="elongation_percent",
        spread="replica_spread_percent",
        event=(
            "elongation_percent",
            "strain_at_break",
            "break_stress_mpa",
            "break_bracket",
        ),
        label="elongation-at-break",
    ),
    "yield": SimpleNamespace(
        run=run_yield_scan,
        stages=yield_stages,
        analyse=analyse_yield,
        write=write_yield_report,
        error=YieldError,
        value="strength_mpa",
        spread="replica_spread_mpa",
        event=("strength_mpa", "yield_strain", "yield_bracket"),
        label="yield-strength",
    ),
}

_QUICK_LADDER: dict[str, Any] = {
    "temperature_k": 120.0,
    "relax_ps": 0.05,
    "n_replicas": 2,
    "samples_per_step": 2,
}

#: Ladders short enough to run: six holds in three chunks, or twelve for yield.
QUICK: dict[str, Any] = {
    "breaking": BreakingSpec(
        **_QUICK_LADDER, strain_increment=0.002, max_strain=0.012, stage_ps=0.1
    ),
    "elongation": ElongationSpec(
        **_QUICK_LADDER, strain_increment=0.002, max_strain=0.012, stage_ps=0.1
    ),
    "yield": YieldSpec(
        **_QUICK_LADDER, max_strain=0.024, stage_ps=0.2, fit_max_strain=0.0125
    ),
}

#: A criterion each planted scan fails, where the default criterion passes it.
STRICTER: dict[str, dict[str, Any]] = {
    "breaking": {"failure_fraction": 0.25},
    "elongation": {"failure_fraction": 0.25},
    "yield": {"offset_strain": 0.02},
}

#: What each planted scan reads back: the replicas' own values of the
#: headline, and further fields that locate their events.
PLANTED = {
    "breaking": {
        "strength_mpa": [100.0, 120.0],
        "failure_stress_mpa": [35.0, 45.0],
        "failure_strain": [1.1**6 - 1.0, 1.1**5 - 1.0],
    },
    "elongation": {
        "elongation_percent": [100.0 * (1.1**6 - 1.0), 100.0 * (1.1**5 - 1.0)],
        "break_stress_mpa": [35.0, 45.0],
        "strain_at_break": [1.1**6 - 1.0, 1.1**5 - 1.0],
    },
    "yield": {
        "strength_mpa": [18.0, 21.6],
        "yield_strain": [0.018, 0.018],
        "modulus_mpa": [1000.0, 1200.0],
    },
}


def _interrupt(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Stop every scan at its first stage, counting the attempts."""
    calls: list[Any] = []

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise RuntimeError("interrupted before the first stage")

    monkeypatch.setattr(tensile, "run_protocol", interrupt)
    return calls


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_chunks_share_a_strain_reference_and_draw_velocities_once(name: str) -> None:
    spec = replace(QUICK[name], max_strain=0.05, relax_ps=50.0, stage_ps=500.0)
    schedule = tensile_schedule(spec)
    assert schedule.n_steps == 25
    assert 1.002**24 - 1.0 < spec.max_strain <= schedule.max_strain
    assert schedule.total_ps == 1250.0
    assert schedule.strain_rate_per_ns == pytest.approx(schedule.max_strain / 1.25)
    reference = (4.0, 5.0, 6.0)
    ladder = tensile_protocol(spec, replica=2, reference_box_nm=reference)
    assert ladder.name == name
    assert [stage.name for stage in ladder.stages] == [
        f"06_{name}_r2_000",
        f"06_{name}_r2_001",
        f"06_{name}_r2_002",
    ]
    assert [stage.options["n_steps"] for stage in ladder.stages] == [10, 10, 5]
    assert [stage.options["new_velocities"] for stage in ladder.stages] == [
        True,
        False,
        False,
    ]
    assert [stage.options["strain_start"] for stage in ladder.stages] == pytest.approx(
        [1.002**done - 1.0 for done in (0, 10, 20)]
    )
    assert all(
        stage.options["reference_box_nm"] == list(reference) for stage in ladder.stages
    )
    other = tensile_protocol(spec, replica=3, reference_box_nm=reference).stages
    assert [stage.options for stage in other] == [
        stage.options for stage in ladder.stages
    ]


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_the_listed_scan_prices_equilibration_and_every_replica(name: str) -> None:
    spec = QUICK[name]
    protocol = tensile_scan(spec, **QUICK_EQUILIBRATION)
    equilibration = standard_melt_equilibration(
        target_temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        **QUICK_EQUILIBRATION,
    )
    assert protocol.name == name
    assert protocol.total_duration_ps == pytest.approx(
        equilibration.total_duration_ps
        + spec.n_replicas * tensile_schedule(spec).total_ps
    )
    assert len(protocol.stages) == len(equilibration.stages) + 6


def test_trajectory_recording_is_explicitly_optional() -> None:
    spec = QUICK["breaking"]
    assert "trajectory" not in tensile_protocol(spec).stages[0].options
    stage = tensile_protocol(replace(spec, trajectory_ps=0.01)).stages[0]
    assert stage.options["trajectory"].format == "xtc"
    assert stage.options["trajectory"].interval_ps == 0.01


def test_a_spec_selects_its_measurement_by_exact_type(argon_scan_run: Any) -> None:
    assert isinstance(ElongationSpec(), BreakingSpec)
    assert tensile_protocol(ElongationSpec()).name == "elongation"
    with pytest.raises(TypeError, match="TensileSpec"):
        tensile_protocol(TensileSpec())
    with pytest.raises(TypeError, match="BreakingSpec"):
        run_breaking_scan(argon_scan_run, "run", spec=ElongationSpec())
    assert not Path("run").exists()


@pytest.mark.parametrize("name", MEASUREMENTS)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature_k", 0.0),
        ("temperature_k", np.nan),
        ("pressure_bar", -1.0),
        ("strain_increment", 0.0),
        ("strain_increment", 0.5),
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
        ("trajectory_ps", 0.0),
        ("max_total_ns", -1.0),
    ],
)
def test_an_unusable_ladder_is_refused_before_any_protocol(
    name: str, field: str, value: Any
) -> None:
    with pytest.raises((ValueError, TypeError), match=field):
        replace(QUICK[name], **{field: value})


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("breaking", "confirmation_steps", 1),
        ("breaking", "confirmation_steps", True),
        ("breaking", "failure_fraction", 0.0),
        ("breaking", "failure_fraction", 1.0),
        ("breaking", "failure_fraction", np.nan),
        ("yield", "offset_strain", 0.0),
        ("yield", "offset_strain", np.nan),
        ("yield", "offset_strain", 0.024),
        ("yield", "fit_min_strain", -0.001),
        ("yield", "fit_min_strain", np.inf),
        ("yield", "fit_min_strain", 0.0125),
        ("yield", "fit_max_strain", 0.0),
        ("yield", "fit_max_strain", 0.024),
    ],
)
def test_an_unusable_criterion_is_refused_before_any_protocol(
    name: str, field: str, value: Any
) -> None:
    with pytest.raises((ValueError, TypeError), match=field):
        replace(QUICK[name], **{field: value})


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_planted_replicas_are_read_separately_and_averaged(
    tmp_path: Path, name: str
) -> None:
    spec = PLANTED_TENSILE[name]
    write_tensile_scan(tmp_path, spec)
    # Neither a modulus extension nor another tensile ladder may leak in.
    write_deformation(tmp_path, stage="06_deform_r0_00")
    other = "breaking" if name == "yield" else "yield"
    write_deformation(tmp_path, stage=f"06_{other}_r0_000")
    before = snapshot_files(tmp_path)
    report = API[name].analyse(tmp_path)
    assert report.resolved
    assert report.replica_indices == (0, 1)
    assert len(API[name].stages(tmp_path)) == 2 * len(tensile_protocol(spec).stages)
    for field, values in PLANTED[name].items():
        assert [getattr(fit, field) for fit in report.replicas] == pytest.approx(values)
    values = PLANTED[name][API[name].value]
    assert getattr(report, API[name].value) == pytest.approx(np.mean(values))
    assert getattr(report, API[name].spread) == pytest.approx(np.std(values, ddof=1))
    assert report.replicas[0].strain_rate_per_ns == pytest.approx(
        tensile_schedule(spec).strain_rate_per_ns
    )
    assert snapshot_files(tmp_path) == before


@pytest.mark.parametrize("name", MEASUREMENTS)
@pytest.mark.parametrize("remove", ["replica", "first_chunk", "last_chunk", "workflow"])
def test_an_incomplete_scan_keeps_its_curves_without_a_headline(
    tmp_path: Path, name: str, remove: str
) -> None:
    record = write_tensile_scan(tmp_path, PLANTED_TENSILE[name])
    if remove == "workflow":
        record.unlink()
    else:
        manifest = RunManifest.load(tmp_path)
        assert manifest is not None
        chunks = [stage for stage in manifest.stages if f"_{name}_r1_" in stage]
        removed = {
            "replica": chunks,
            "first_chunk": chunks[:1],
            "last_chunk": chunks[-1:],
        }[remove]
        for stage in removed:
            del manifest.stages[stage]
        manifest.save(tmp_path)
    report = API[name].analyse(tmp_path)
    assert not report.resolved
    assert getattr(report, API[name].value) is None
    assert getattr(report, API[name].spread) is None
    assert report.curves[0].n_points == tensile_schedule(PLANTED_TENSILE[name]).n_steps
    assert any("incomplete" in note for note in report.notes)
    if remove != "replica":
        damaged = report.replicas[-1]
        assert not damaged.resolved
        assert all(getattr(damaged, field) is None for field in API[name].event)
        assert any("Incomplete replica" in note for note in damaged.notes)
    if remove == "workflow":
        assert not any(fit.resolved for fit in report.replicas)
        assert any("No workflow record" in note for note in report.notes)


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_a_truncated_replica_cannot_resolve_even_after_its_event(
    tmp_path: Path, name: str
) -> None:
    spec = replace(PLANTED_TENSILE[name], n_replicas=1)
    if name != "yield":
        # Two confirming holds, so the stress loss survives the truncation.
        spec = replace(spec, confirmation_steps=2)
    write_tensile_scan(tmp_path, spec)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    samples = manifest.stages[list(manifest.stages)[-1]]["samples"]
    for key in samples:
        if key.startswith("segment_"):
            samples[key] = samples[key][:-1]
    manifest.save(tmp_path)
    report = API[name].analyse(tmp_path)
    fit = report.replicas[0]
    assert report.curves[0].n_points == tensile_schedule(spec).n_steps - 1
    assert not fit.resolved
    assert all(getattr(fit, field) is None for field in API[name].event)
    assert any("Incomplete replica" in note for note in fit.notes)


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_one_replica_has_no_estimated_spread(tmp_path: Path, name: str) -> None:
    write_tensile_scan(tmp_path, replace(PLANTED_TENSILE[name], n_replicas=1))
    report = API[name].analyse(tmp_path)
    assert report.resolved
    value = PLANTED[name][API[name].value][0]
    assert getattr(report, API[name].value) == pytest.approx(value)
    assert getattr(report, API[name].spread) is None


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_the_saved_criterion_controls_reanalysis(tmp_path: Path, name: str) -> None:
    write_tensile_scan(tmp_path, replace(PLANTED_TENSILE[name], **STRICTER[name]))
    report = API[name].analyse(tmp_path)
    for field, value in STRICTER[name].items():
        assert getattr(report, field) == value
    assert not report.resolved
    assert not any(fit.resolved for fit in report.replicas)


@pytest.mark.parametrize("name", MEASUREMENTS)
@pytest.mark.parametrize(
    "damage",
    [
        "request",
        "spec",
        "not_a_spec",
        "criterion",
        "extent",
        "steps",
        "no_steps",
        "chunks",
        "no_chunks",
    ],
)
def test_a_partial_or_inconsistent_record_cannot_invent_a_resolved_event(
    tmp_path: Path, name: str, damage: str
) -> None:
    # Three replicas under a criterion they fail. The default criterion and
    # replica count would resolve this scan, so no gap may fall back on them.
    spec = replace(PLANTED_TENSILE[name], n_replicas=3, **STRICTER[name])
    path = write_tensile_scan(tmp_path, spec)
    assert not API[name].analyse(tmp_path).resolved
    record = json.loads(path.read_text())
    if damage == "request":
        del record["request"]
    elif damage == "spec":
        del record["request"]["spec"]
    elif damage == "not_a_spec":
        record["request"]["spec"] = list(record["request"]["spec"].values())
    elif damage == "criterion":
        for field in STRICTER[name]:
            del record["request"]["spec"][field]
    elif damage == "extent":
        record["request"]["spec"]["max_strain"] = 4.0
    elif damage == "steps":
        record["steps_per_replica"] -= 1
    elif damage == "no_steps":
        del record["steps_per_replica"]
    elif damage == "chunks":
        record["replica_stages"][0].pop()
    else:
        del record["replica_stages"]
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="workflow record"):
        API[name].analyse(tmp_path)


@pytest.mark.parametrize("name", MEASUREMENTS)
@pytest.mark.parametrize(
    ("text", "message"),
    [("not json", "Cannot read"), ("[]", "must contain a workflow record")],
)
def test_an_unreadable_record_is_refused(
    tmp_path: Path, name: str, text: str, message: str
) -> None:
    write_tensile_scan(tmp_path, PLANTED_TENSILE[name]).write_text(text)
    with pytest.raises(API[name].error, match=message):
        API[name].analyse(tmp_path)


@pytest.mark.parametrize("name", MEASUREMENTS)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_box_nm", [5.0, 5.0, 6.0]),
        ("reference_box_nm", [5.0, 5.0]),
        ("deform_axis", [0.0]),
        ("deform_axis", [2.0, 2.0]),
        ("segment_strain", [0.4, 0.3, 0.2, 0.1]),
        ("segment_stress_zz_bar", [1.0, 2.0]),
        ("segment_box_x_nm", [5.0, 5.0, 5.0, -5.0]),
    ],
)
def test_chunks_that_do_not_continue_one_ladder_are_refused(
    tmp_path: Path, name: str, field: str, value: list[float]
) -> None:
    write_tensile_scan(tmp_path, PLANTED_TENSILE[name])
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    manifest.stages[f"06_{name}_r0_001"]["samples"][field] = value
    manifest.save(tmp_path)
    with pytest.raises(AnalysisError, match=f"Malformed {name} replica 0"):
        API[name].analyse(tmp_path)


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_a_run_without_the_measurements_stages_has_a_clear_error(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(AnalysisError, match=f"No {API[name].label} stages"):
        API[name].analyse(tmp_path)
    write_deformation(tmp_path)
    with pytest.raises(AnalysisError, match=f"No {API[name].label} stages"):
        API[name].stages(tmp_path)


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_the_report_is_strict_json_with_a_figure_per_replica(
    tmp_path: Path, name: str
) -> None:
    spec = PLANTED_TENSILE[name]
    write_tensile_scan(tmp_path, spec)
    report = API[name].analyse(tmp_path)
    files = API[name].write(report, figure_format="svg")
    assert files.json == str(tmp_path / "analysis" / f"{name}.json")
    record = json.loads(Path(files.json).read_text())
    assert json.dumps(record, allow_nan=False)
    values = PLANTED[name][API[name].value]
    assert record[API[name].value] == pytest.approx(np.mean(values))
    assert record["replica_indices"] == [0, 1]
    assert record["curves"][0]["temperature_k"] == spec.temperature_k
    assert record["curves"][1]["lateral_strain"] == pytest.approx(
        report.curves[1].lateral_strain
    )
    assert record["replicas"][1]["nominal_stress_mpa"] == pytest.approx(
        report.replicas[1].nominal_stress_mpa
    )
    assert [Path(path).name for path in files.figures] == [
        f"{name}_r0.svg",
        f"{name}_r1.svg",
    ]
    assert all(Path(path).stat().st_size > 0 for path in files.figures)


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_an_unresolved_report_writes_json_null_and_can_skip_figures(
    tmp_path: Path, name: str
) -> None:
    write_tensile_scan(tmp_path, replace(PLANTED_TENSILE[name], **STRICTER[name]))
    report = API[name].analyse(tmp_path)
    files = API[name].write(report, tmp_path / "elsewhere", figures=False)
    record = json.loads(Path(files.json).read_text())
    assert Path(files.json).parent == tmp_path / "elsewhere"
    assert record[API[name].value] is None
    assert record[API[name].spread] is None
    assert all(record["replicas"][0][field] is None for field in API[name].event)
    assert files.figures == ()


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_a_missing_replica_does_not_renumber_the_others(
    tmp_path: Path, name: str
) -> None:
    write_tensile_scan(tmp_path, PLANTED_TENSILE[name])
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    for stage in list(manifest.stages):
        if stage.startswith(f"06_{name}_r0_"):
            del manifest.stages[stage]
    manifest.save(tmp_path)
    report = API[name].analyse(tmp_path)
    assert report.replica_indices == (1,)
    assert not report.resolved
    files = API[name].write(report)
    assert [Path(path).name for path in files.figures] == [f"{name}_r1.png"]
    assert json.loads(Path(files.json).read_text())["replica_indices"] == [1]


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_the_budget_refuses_a_scan_before_any_output(
    argon_scan_run: Any, name: str
) -> None:
    with pytest.raises(API[name].error, match="max_total_ns"):
        API[name].run(
            argon_scan_run,
            "run",
            spec=replace(QUICK[name], max_total_ns=1e-6),
            **QUICK_EQUILIBRATION,
        )
    assert not Path("run").exists()


@pytest.mark.parametrize("name", MEASUREMENTS)
@pytest.mark.parametrize(
    "change", ["criterion", "replicas", "system", "coordinates", "box"]
)
def test_an_interrupted_scan_resumes_only_with_the_same_request(
    argon_scan_run: Any, monkeypatch: pytest.MonkeyPatch, name: str, change: str
) -> None:
    calls = _interrupt(monkeypatch)
    spec = QUICK[name]
    with pytest.raises(RuntimeError, match="interrupted"):
        API[name].run(argon_scan_run, "run", spec=spec, **QUICK_EQUILIBRATION)
    assert Path("run", f"{name}_workflow.json").is_file()
    assert not Path("run/manifest.json").exists()
    if change == "criterion":
        spec = replace(spec, **STRICTER[name])
    elif change == "replicas":
        spec = replace(spec, n_replicas=1)
    elif change == "system":
        argon_scan_run.system_xml += "\n"
    elif change == "coordinates":
        argon_scan_run.box.positions_nm[0, 0] += 0.001
    else:
        argon_scan_run.box.box_nm = (2.81, 2.8, 2.8)
    with pytest.raises(API[name].error, match="different settings"):
        API[name].run(argon_scan_run, "run", spec=spec, **QUICK_EQUILIBRATION)
    assert len(calls) == 1


@pytest.mark.parametrize("name", MEASUREMENTS)
@pytest.mark.parametrize("foreign", [True, False])
def test_stages_without_the_scans_record_need_a_fresh_directory(
    argon_scan_run: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    foreign: bool,
) -> None:
    write_deformation(tmp_path)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    manifest.protocol = "foreign" if foreign else name
    manifest.save(tmp_path)
    calls = _interrupt(monkeypatch)
    with pytest.raises(API[name].error, match="fresh directory"):
        API[name].run(argon_scan_run, tmp_path, spec=QUICK[name], **QUICK_EQUILIBRATION)
    assert not calls
    assert not (tmp_path / f"{name}_workflow.json").exists()


@pytest.mark.parametrize("name", MEASUREMENTS)
@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing_state", "missing state files"),
        ("equilibration_gap", "missing predecessors"),
        ("replica_gap", "missing predecessors"),
        ("early_replica", "before equilibration is complete"),
        ("unexpected", "unexpected stages"),
    ],
)
def test_a_resume_cannot_mix_new_predecessors_with_saved_descendants(
    argon_scan_run: Any,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    damage: str,
    message: str,
) -> None:
    calls = _interrupt(monkeypatch)
    spec = QUICK[name]
    with pytest.raises(RuntimeError, match="interrupted"):
        API[name].run(argon_scan_run, "run", spec=spec, **QUICK_EQUILIBRATION)
    settled = [
        stage.name
        for stage in tensile_scan(spec, **QUICK_EQUILIBRATION).stages
        if not stage.name.startswith(f"06_{name}_")
    ]
    replica = [stage.name for stage in tensile_protocol(spec).stages]
    names = {
        "missing_state": settled[:1],
        "equilibration_gap": settled[1:2],
        "replica_gap": [*settled, replica[1]],
        "early_replica": [settled[0], replica[0]],
        "unexpected": ["foreign_stage"],
    }[damage]
    state = Path("run/existing.xml")
    if damage != "missing_state":
        state.write_text("Readable state whose contents must never reach dynamics.")
    RunManifest(
        protocol=name,
        seed=11,
        stages={stage: {"final_state": str(state)} for stage in names},
    ).save("run")
    before = snapshot_files(Path("run"))
    with pytest.raises(API[name].error, match=f"Cannot resume: .*{message}"):
        API[name].run(argon_scan_run, "run", spec=spec, **QUICK_EQUILIBRATION)
    assert len(calls) == 1
    assert snapshot_files(Path("run")) == before


@pytest.mark.slow
def test_a_real_scan_branches_its_replicas_and_survives_rerun_and_resume(
    argon_scan_run: Any,
) -> None:
    spec = QUICK["breaking"]
    first = run_breaking_scan(
        argon_scan_run, "run", spec=spec, resume=False, **QUICK_EQUILIBRATION
    )
    manifest = RunManifest.load("run")
    assert manifest is not None
    assert len(manifest.stages) == 12
    assert "00_minimise" in manifest.stages and "05_npt" in manifest.stages
    assert first.replica_indices == (0, 1)
    assert [curve.n_points for curve in first.curves] == [6, 6]
    record = json.loads(Path("run/breaking_workflow.json").read_text())
    assert Path(record["start_state"]).name.startswith("05_npt")
    for curve in first.curves:
        assert curve.strain == pytest.approx(1.002 ** np.arange(1, 7) - 1.0)
    for stage_name, stage in manifest.stages.items():
        if stage_name.startswith("06_breaking_"):
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
    resumed = run_breaking_scan(argon_scan_run, "run", spec=spec, **QUICK_EQUILIBRATION)
    for old, new in zip(first.curves, resumed.curves, strict=True):
        assert new.stress_mpa == pytest.approx(old.stress_mpa)
    for path, (content, modified) in before.items():
        assert path.read_bytes() == content
        if path.name not in ("breaking_workflow.json", "manifest.json"):
            assert path.stat().st_mtime_ns == modified


def test_every_tensile_entry_point_is_exported() -> None:
    for name in MEASUREMENTS:
        for entry in ("run", "stages", "analyse", "write", "error"):
            assert getattr(API[name], entry).__name__ in openmmpolymer.__all__
    for name in (
        "TensileSpec",
        "BreakingSpec",
        "ElongationSpec",
        "YieldSpec",
        "TensileSchedule",
        "BreakingReport",
        "ElongationReport",
        "YieldReport",
        "tensile_schedule",
        "tensile_protocol",
        "tensile_scan",
    ):
        assert name in openmmpolymer.__all__
