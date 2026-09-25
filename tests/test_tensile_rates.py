"""Event-specific measurements keep their criteria, replicas and censoring."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmmpolymer import tensile_rates
from openmmpolymer.protocols import Protocol, RunManifest
from openmmpolymer.tensile import BreakingSpec, ElongationSpec, YieldSpec
from openmmpolymer.tensile_rates import (
    WORKFLOW_NAME,
    analyse_tensile_rates,
    run_tensile_rate_scan,
    validate_tensile_rate_scan,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    PLANTED_TENSILE,
    QUICK_EQUILIBRATION,
    fake_scan_dynamics,
    write_manifest,
    write_tensile_rate_series,
)

BREAKING_SPEC = PLANTED_TENSILE["breaking"]


@pytest.mark.parametrize(
    "property_name",
    ["yield_strength", "yield_strain", "breaking_strength", "elongation_at_break"],
)
def test_each_event_has_correct_units_and_preserves_replicas(
    tmp_path: Path, property_name: str
) -> None:
    directories = write_tensile_rate_series(tmp_path, property_name)
    report = analyse_tensile_rates(
        directories, property_name=property_name, target_rate=0.1
    )
    assert report.property.name == property_name
    assert report.property.rate_unit == "strain/ns"
    assert report.property.value_unit == (
        "%"
        if property_name == "elongation_at_break"
        else "strain"
        if property_name == "yield_strain"
        else "MPa"
    )
    assert len(report.observations) == 6
    assert report.log_linear is not None
    assert report.log_linear.n_rates == 3
    assert report.log_linear.sensitivity_per_decade == pytest.approx(0, abs=1e-8)
    assert any(
        "not a standard error" in note
        for observation in report.observations
        for note in observation.notes
    )


def test_missing_yield_replica_is_not_silently_dropped_from_rate_fit(
    tmp_path: Path,
) -> None:
    directories = write_tensile_rate_series(tmp_path, "yield_strength")
    path = directories[1] / "manifest.json"
    record = json.loads(path.read_text())
    record["stages"] = {
        name: stage for name, stage in record["stages"].items() if "_r1_" not in name
    }
    path.write_text(json.dumps(record))
    report = analyse_tensile_rates(
        directories, property_name="yield_strength", target_rate=0.1
    )
    assert any(not item.resolved for item in report.observations)
    assert report.log_linear is None or not report.log_linear.resolved
    assert report.power_law is None or not report.power_law.resolved


def test_surviving_yield_replica_retains_its_recorded_index(tmp_path: Path) -> None:
    directories = write_tensile_rate_series(tmp_path, "yield_strength")
    path = directories[1] / "manifest.json"
    record = json.loads(path.read_text())
    record["stages"] = {
        name: stage for name, stage in record["stages"].items() if "_r0_" not in name
    }
    path.write_text(json.dumps(record))
    report = analyse_tensile_rates(
        directories, property_name="yield_strength", target_rate=0.1
    )
    surviving = [
        item for item in report.observations if str(directories[1]) in item.source
    ]
    assert len(surviving) == 1
    assert surviving[0].source.endswith(":replica_1")
    assert not surviving[0].resolved


@pytest.mark.parametrize("criterion", ["offset_strain", "fit_max_strain"])
def test_different_yield_criteria_cannot_be_combined(
    tmp_path: Path, criterion: str
) -> None:
    directories = write_tensile_rate_series(tmp_path, "yield_strength")
    path = directories[1] / "yield_workflow.json"
    record = json.loads(path.read_text())
    record["request"]["spec"][criterion] *= 0.9
    path.write_text(json.dumps(record))
    report = analyse_tensile_rates(
        directories, property_name="yield_strength", target_rate=0.1
    )
    assert report.log_linear is None and report.power_law is None
    assert any("condition" in note for note in report.notes)


def test_duplicate_directories_are_not_independent_replicas(tmp_path: Path) -> None:
    directories = write_tensile_rate_series(tmp_path, "yield_strength")
    with pytest.raises(AnalysisError, match="more than once"):
        analyse_tensile_rates(
            [*directories, directories[0]],
            property_name="yield_strength",
            target_rate=0.1,
        )


def test_budget_covers_common_preparation_and_every_replica() -> None:
    spec = BreakingSpec(n_replicas=2)
    plan = validate_tensile_rate_scan(
        spec,
        (10, 20, 30),
        property_name="breaking_strength",
        target_rate=0.1,
        **QUICK_EQUILIBRATION,
    )
    with pytest.raises(ValueError, match="max_total_ns"):
        validate_tensile_rate_scan(
            replace(spec, max_total_ns=plan.total_ns - 0.001),
            (10, 20, 30),
            property_name="breaking_strength",
            target_rate=0.1,
            **QUICK_EQUILIBRATION,
        )
    with pytest.raises(ValueError, match="YieldSpec"):
        validate_tensile_rate_scan(
            spec, (10, 20, 30), property_name="yield_strength", target_rate=0.1
        )


def test_shared_start_independent_rate_seeds_and_resume_guard(
    tmp_path: Path, argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def dynamics(protocol: Protocol, run: Any, directory: Path, **kwargs: Any) -> Any:
        calls.append({"protocol": protocol, "seed": run.seed, **kwargs})
        directory.mkdir(parents=True, exist_ok=True)
        state = directory / "state.xml"
        state.write_text("test state")
        return SimpleNamespace(final_state=str(state))

    fake_scan_dynamics(monkeypatch, tensile_rates, dynamics, box_nm=(2.4,) * 3)
    monkeypatch.setattr(
        tensile_rates, "analyse_tensile_rates", lambda *args, **kwargs: None
    )
    spec = replace(BREAKING_SPEC, n_replicas=2)
    options: dict[str, Any] = dict(
        spec=spec,
        property_name="breaking_strength",
        hold_times_ps=(1, 2, 3),
        target_rate=0.1,
    )
    run_tensile_rate_scan(argon_run, tmp_path, **options)
    assert len(calls) == 7
    assert len({item["seed"] for item in calls[1:]}) == 3
    assert {item["state_in"] for item in calls[1:]} == {
        str(tmp_path / "equilibration/state.xml")
    }
    assert all(
        item["protocol"].stages[0].options["new_velocities"] for item in calls[1:]
    )
    before = len(calls)
    with pytest.raises(ValueError, match="different settings"):
        run_tensile_rate_scan(
            argon_run, tmp_path, **{**options, "hold_times_ps": (1, 2, 4)}
        )
    assert len(calls) == before
    assert (tmp_path / WORKFLOW_NAME).is_file()


def test_unconfirmed_terminal_event_is_preserved_as_missing(tmp_path: Path) -> None:
    directories = write_tensile_rate_series(tmp_path, "breaking_strength")
    for directory in directories:
        path = directory / "manifest.json"
        record = json.loads(path.read_text())
        for stage in record["stages"].values():
            stage["samples"]["segment_stress_zz_bar"] = [100.0] * 4
        write_manifest(directory, record["stages"])
    report = analyse_tensile_rates(
        directories, property_name="breaking_strength", target_rate=0.1
    )
    assert any(item.value is None for item in report.observations)
    assert report.log_linear is None
    assert report.power_law is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("segment_duration_ps", [0.5, 1.5, 1, 1]),
        ("deform_axis", [0]),
        ("segment_temperature_k", [400] * 4),
    ],
)
def test_recorded_physics_must_match_saved_settings(
    tmp_path: Path, key: str, value: list[float]
) -> None:
    directories = write_tensile_rate_series(tmp_path, "breaking_strength")
    path = directories[0] / "manifest.json"
    record = json.loads(path.read_text())
    first = next(iter(record["stages"].values()))
    first["samples"][key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match=r"different|changes"):
        analyse_tensile_rates(
            directories, property_name="breaking_strength", target_rate=0.1
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "property_name", ["breaking_strength", "elongation_at_break", "yield_strength"]
)
def test_real_rate_workflow_round_trip_and_resume(
    tmp_path: Path, argon_run: Any, property_name: str
) -> None:
    settings: dict[str, Any] = {
        "temperature_k": 120.0,
        "strain_increment": 0.002,
        "max_strain": 0.012,
        "relax_ps": 0.02,
        "stage_ps": 0.08,
        "samples_per_step": 2,
        "n_replicas": 2,
    }
    spec = (
        YieldSpec(**settings, fit_max_strain=0.007)
        if property_name == "yield_strength"
        else ElongationSpec(**settings)
        if property_name == "elongation_at_break"
        else BreakingSpec(**settings)
    )
    options: dict[str, Any] = {
        "property_name": property_name,
        "spec": spec,
        "hold_times_ps": (0.02, 0.04, 0.06),
        "target_rate": 1.0,
        "nvt_ps": 0.02,
        "compress_ps_each": 0.02,
        "npt_ps": 0.03,
        "anneal_cycles": 1,
        "anneal_window_ps": 0.01,
        "anneal_hold_ps": 0.01,
        "compress_pressures_bar": (1.0, 5.0, 1.0),
    }
    report = run_tensile_rate_scan(argon_run, tmp_path, resume=False, **options)
    assert len(report.observations) == 6
    assert len({round(item.rate, 7) for item in report.observations}) == 3
    # These very short liquid trajectories exercise recording, not a claim
    # that argon has a resolved polymer failure response.
    manifests = [Path(path) / "manifest.json" for path in report.run_dirs]
    before = [json.loads(path.read_text())["stages"] for path in manifests]
    assert all(len(stages) >= 4 for stages in before)
    resumed = run_tensile_rate_scan(argon_run, tmp_path, **options)
    assert len(resumed.observations) == 6
    assert before == [json.loads(path.read_text())["stages"] for path in manifests]


@pytest.mark.parametrize(
    "metadata", ["equilibration", "system", "composition", "preparation_state_sha256"]
)
def test_saved_tensile_physics_and_preparation_must_match(
    tmp_path: Path, metadata: str
) -> None:
    directories = write_tensile_rate_series(tmp_path, "yield_strength")
    for index, directory in enumerate(directories):
        path = directory / "yield_workflow.json"
        record = json.loads(path.read_text())
        if metadata == "equilibration":
            record["request"][metadata] = [
                {
                    "kind": "npt",
                    "options": {"temperature_k": 600 if index == 1 else 450},
                }
            ]
        elif metadata == "system":
            record["request"][metadata] = {"hydrogen_mass_amu": 3 if index == 1 else 1}
        elif metadata == "preparation_state_sha256":
            record["request"][metadata] = "different" if index == 1 else "shared"
        else:
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["box"] = {
                "n_molecules": 5 if index == 1 else 10,
                "atoms_per_chain": 100,
            }
            manifest_path.write_text(json.dumps(manifest))
        path.write_text(json.dumps(record))
    report = analyse_tensile_rates(
        directories, property_name="yield_strength", target_rate=0.1
    )
    assert report.log_linear is None and report.power_law is None
    assert any("condition" in note for note in report.notes)


def test_legacy_tensile_preparation_absence_is_reported(tmp_path: Path) -> None:
    report = analyse_tensile_rates(
        write_tensile_rate_series(tmp_path, "yield_strength"),
        property_name="yield_strength",
        target_rate=0.1,
    )
    assert any(
        "preparation" in note and "unavailable" in note
        for item in report.observations
        for note in item.notes
    )


@pytest.mark.parametrize("damage", ["changed", "missing", "missing_fingerprint"])
def test_tensile_shared_state_is_checked_before_any_resume_write(
    tmp_path: Path,
    argon_run: Any,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    calls: list[str] = []

    def dynamics(protocol: Protocol, run: Any, directory: Path, **kwargs: Any) -> Any:
        calls.append(protocol.name)
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / "state.xml"
        final.write_text("original common state")
        RunManifest(protocol=protocol.name, seed=run.seed).save(directory)
        return SimpleNamespace(final_state=str(final))

    fake_scan_dynamics(monkeypatch, tensile_rates, dynamics, box_nm=(2.4,) * 3)
    monkeypatch.setattr(
        tensile_rates, "analyse_tensile_rates", lambda *args, **kwargs: None
    )
    options: dict[str, Any] = {
        "property_name": "breaking_strength",
        "spec": BREAKING_SPEC,
        "hold_times_ps": (1, 2, 3),
        "target_rate": 0.1,
    }
    run_tensile_rate_scan(argon_run, tmp_path, **options)
    workflow = tmp_path / WORKFLOW_NAME
    record = json.loads(workflow.read_text())
    assert record["start_state_sha256"]
    for rate in range(3):
        child = json.loads(
            (tmp_path / f"rate_{rate:02d}/breaking_workflow.json").read_text()
        )
        assert (
            child["request"]["equilibration"]
            == record["request"]["equilibration"]["stages"]
        )
        assert (
            child["request"]["preparation_state_sha256"] == record["start_state_sha256"]
        )
    if damage == "changed":
        Path(record["start_state"]).write_text("different common state")
    elif damage == "missing":
        Path(record["start_state"]).unlink()
    else:
        del record["start_state_sha256"]
        workflow.write_text(json.dumps(record))
    before = workflow.read_text()
    count = len(calls)
    with pytest.raises(ValueError, match="fingerprint"):
        run_tensile_rate_scan(argon_run, tmp_path, **options)
    assert len(calls) == count
    assert workflow.read_text() == before
