"""Elastic hold-time sweeps retain their control variable and uncertainty.

Young's modulus shares the extension ladder with Poisson's ratio, but keeps
the record its original rate scan kept and pools each rate's replicas into
one fit; its tests are in their own section.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmmpolymer import elastic_rates
from openmmpolymer.elastic_rates import (
    ELASTIC_RATE_PROPERTIES,
    WORKFLOW_NAME,
    YOUNGS_WORKFLOW_NAME,
    analyse_elastic_rates,
    run_elastic_rate_scan,
    validate_elastic_rate_scan,
)
from openmmpolymer.mdsystem import SystemSpec
from openmmpolymer.mechanical import MechanicalError, ModulusSpec, deform_protocol
from openmmpolymer.protocols import Protocol
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    deformation_rate_per_ns,
    fake_scan_dynamics,
    planted_extension_runner,
    planted_modulus_mpa,
    write_bulk,
    write_deformation,
    write_modulus_rate_series,
    write_shear,
)

HOLDS = (20.0, 60.0, 200.0)
SPEC = ModulusSpec(n_replicas=2, max_strain=0.02)
#: Every elastic property; all but Young's modulus observe each replica.
PROPERTIES = tuple(ELASTIC_RATE_PROPERTIES)
REPLICA_PROPERTIES = tuple(name for name in PROPERTIES if name != "youngs_modulus")


def _stamp(directory: Path, hold: float) -> None:
    manifest = directory / "manifest.json"
    record = json.loads(manifest.read_text())
    for stage in record["stages"].values():
        samples = stage["samples"]
        key = next(key for key in elastic_rates._PATH_KEYS.values() if key in samples)
        n = len(samples[key])
        samples["segment_duration_ps"] = [hold] * n
        samples["segment_temperature_k"] = [298.15] * n
        samples["lateral_pressure_bar"] = [1.0]
        if key == "segment_shear_strain":
            samples["shear_plane"] = [0.0, 2.0]
    manifest.write_text(json.dumps(record))


def _series(root: Path, name: str, *, replicas: int = 1) -> list[Path]:
    directories = []
    for i, hold in enumerate(HOLDS):
        directory = root / f"rate_{i:02d}"
        for replica in range(replicas):
            if name in ("youngs_modulus", "poisson_ratio"):
                rate = deformation_rate_per_ns(hold)
                write_deformation(
                    directory,
                    modulus_mpa=planted_modulus_mpa(rate),
                    poisson=0.3 + 0.01 * math.log10(rate / 0.01),
                    relax_ps=hold,
                    stage=f"06_deform_r{i * replicas + replica}_00",
                )
            elif name == "shear_modulus":
                rate = 0.02 / (4 * hold) * 1000
                write_shear(
                    directory,
                    modulus_mpa=700 + 20 * math.log10(rate / 0.01),
                    stage=f"09_shear_r{i * replicas + replica}",
                )
            elif name == "bulk_modulus":
                rate = 598 / (7 * hold) * 1000
                write_bulk(
                    directory,
                    modulus_mpa=1500 + 50 * math.log10(rate / 100) + replica * 5,
                    stage=f"08_bulk_r{i * replicas + replica}",
                )
            else:
                stresses = [0.0, 50.0, 100.0, 150.0]
                rate = 150 / (4 * hold) * 1000
                modulus = 2000 + 50 * math.log10(rate / 100)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "manifest.json"
                record: dict[str, Any] = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"stages": {}, "protocol": "test", "seed": 17}
                )
                record["stages"][f"07_load_r{i * replicas + replica}"] = {
                    "samples": {
                        "segment_applied_stress_bar": stresses,
                        "load_axis": [2],
                        "segment_box_x_nm": [5.0] * 4,
                        "segment_box_y_nm": [5.0] * 4,
                        "segment_box_z_nm": [
                            5 * (1 + p * 0.1 / modulus) for p in stresses
                        ],
                    },
                    "mean_temperature_k": 298.15,
                }
                path.write_text(json.dumps(record))
        _stamp(directory, hold)
        directories.append(directory)
    return directories


def _run() -> Any:
    return SimpleNamespace(
        spec=SystemSpec(),
        seed=17,
        system_xml="system",
        box=SimpleNamespace(positions_nm=np.zeros((2, 3)), box_nm=(5.0, 5.0, 5.0)),
    )


def _fail(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("interrupted")


@pytest.mark.parametrize("name", PROPERTIES)
def test_recovers_property_log_law_and_retains_units(tmp_path: Path, name: str) -> None:
    target = (
        0.001 if name in ("youngs_modulus", "poisson_ratio", "shear_modulus") else 10.0
    )
    expected = {
        "youngs_modulus": 1700.0,
        "poisson_ratio": 0.29,
        "shear_modulus": 680.0,
        "bulk_modulus": 1450.0,
        "load_modulus": 1950.0,
    }[name]
    report = analyse_elastic_rates(
        _series(tmp_path, name), property_name=name, target_rate=target
    )
    assert report.log_linear is not None
    assert report.log_linear.value == pytest.approx(expected)
    assert len(report.observations) == 3
    assert report.property.rate_unit == ("strain/ns" if target == 0.001 else "bar/ns")
    if name == "bulk_modulus":
        assert all(o.standard_error is not None for o in report.observations)
    assert report.log_linear.resolved


def test_bulk_replicas_supply_uncertainty_beyond_the_single_fit_error(
    tmp_path: Path,
) -> None:
    report = analyse_elastic_rates(
        _series(tmp_path, "bulk_modulus", replicas=2),
        property_name="bulk_modulus",
        target_rate=10.0,
    )
    assert report.log_linear is not None
    assert report.log_linear.resolved
    assert all(o.standard_error is not None for o in report.observations)
    assert report.log_linear.standard_errors == pytest.approx(
        [np.std([0, 5], ddof=1)] * 3
    )
    assert report.log_linear.n_rates == 3


def test_bulk_rate_observations_carry_fit_uncertainty_and_reject_noise(
    tmp_path: Path,
) -> None:
    directories = _series(tmp_path, "bulk_modulus")
    for directory in directories:
        path = directory / "manifest.json"
        record = json.loads(path.read_text())
        samples = next(iter(record["stages"].values()))["samples"]
        density = np.asarray(samples["segment_density_g_cm3"])
        samples["segment_density_g_cm3"] = (
            density * np.exp([0.0, -0.4, 0.4, 0.0, 0.4, -0.4, 0.0])
        ).tolist()
        path.write_text(json.dumps(record))
    report = analyse_elastic_rates(
        directories, property_name="bulk_modulus", target_rate=10.0
    )
    assert all(o.standard_error is not None for o in report.observations)
    assert all(not o.resolved for o in report.observations)
    assert report.log_linear is not None
    assert not report.log_linear.resolved


@pytest.mark.parametrize("name", ("shear_modulus", "bulk_modulus", "load_modulus"))
def test_nominal_rate_counts_both_branches_and_every_hold(
    tmp_path: Path, name: str
) -> None:
    directories = _series(tmp_path, name)
    for directory in directories:
        file = directory / "manifest.json"
        record = json.loads(file.read_text())
        samples = next(iter(record["stages"].values()))["samples"]
        key = elastic_rates._PATH_KEYS[name]
        if name == "shear_modulus":
            samples[key] = [0.01, 0.02, 0.01, 0.0]
            expected_path = 0.04
        elif name == "bulk_modulus":
            expected_path = 598.0
        else:
            samples[key] = [0.0, 100.0, 200.0, 0.0]
            expected_path = 400.0
        file.write_text(json.dumps(record))
    report = analyse_elastic_rates(directories, property_name=name, target_rate=1.0)
    for observation, hold in zip(report.observations, HOLDS, strict=True):
        count = len(observation.conditions["loading_path"])
        assert observation.rate == pytest.approx(expected_path / (count * hold) * 1000)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("segment_temperature_k", [310.0] * 4),
        ("shear_plane", [1.0, 2.0]),
        ("lateral_pressure_bar", [5.0]),
        ("segment_shear_strain", [0.004, 0.008, 0.012, 0.016]),
    ],
)
def test_incompatible_measurements_refuse_rate_fit(
    tmp_path: Path, field: str, replacement: list[float]
) -> None:
    directories = _series(tmp_path, "shear_modulus")
    file = directories[1] / "manifest.json"
    record = json.loads(file.read_text())
    next(iter(record["stages"].values()))["samples"][field] = replacement
    file.write_text(json.dumps(record))
    report = analyse_elastic_rates(
        directories, property_name="shear_modulus", target_rate=0.001
    )
    assert report.log_linear is None
    assert report.power_law is None
    assert any("same" in note for note in report.notes)


@pytest.mark.parametrize(
    "field", ["segment_temperature_k", "lateral_pressure_bar", "shear_plane"]
)
def test_unknown_conditions_stay_unresolved(tmp_path: Path, field: str) -> None:
    directories = _series(tmp_path, "shear_modulus")
    for directory in directories:
        file = directory / "manifest.json"
        record = json.loads(file.read_text())
        next(iter(record["stages"].values()))["samples"].pop(field)
        file.write_text(json.dumps(record))
    report = analyse_elastic_rates(
        directories, property_name="shear_modulus", target_rate=0.001
    )
    assert all(not o.resolved for o in report.observations)
    assert report.log_linear is not None and not report.log_linear.resolved
    assert any("missing" in note for note in report.notes)


def test_missing_measurement_stays_present_and_blocks_extrapolation(
    tmp_path: Path,
) -> None:
    directories = _series(tmp_path, "shear_modulus")
    file = directories[1] / "manifest.json"
    record = json.loads(file.read_text())
    next(iter(record["stages"].values()))["samples"].pop("segment_shear_stress_bar")
    file.write_text(json.dumps(record))
    report = analyse_elastic_rates(
        directories, property_name="shear_modulus", target_rate=0.001
    )
    assert len(report.observations) == 3
    assert report.observations[1].value is None
    assert report.observations[1].standard_error is None
    assert report.log_linear is None


@pytest.mark.parametrize("name", PROPERTIES)
def test_plan_counts_only_selected_pass_and_all_replicas(name: str) -> None:
    plan = validate_elastic_rate_scan(
        SPEC, HOLDS, property_name=name, target_rate=0.001
    )
    assert len(plan.protocols) == 3
    assert all(len(group) == SPEC.n_replicas for group in plan.protocols)
    expected = (
        plan.equilibration.total_duration_ps
        + sum(p.total_duration_ps for group in plan.protocols for p in group)
    ) / 1000
    assert plan.total_ns == pytest.approx(expected)
    names = [
        stage.name for group in plan.protocols for p in group for stage in p.stages
    ]
    assert len(names) == len(set(names))
    assert all(group[0].stages[0].options["new_velocities"] for group in plan.protocols)
    with pytest.raises(MechanicalError, match="budget"):
        validate_elastic_rate_scan(
            replace(SPEC, max_total_ns=expected * 0.999),
            HOLDS,
            property_name=name,
            target_rate=0.001,
        )


def test_disabled_and_invalid_ladders_rejected() -> None:
    for spec in (
        replace(SPEC, shear_strains=None),
        replace(SPEC, shear_strains=(0.01, 0.01)),
        replace(SPEC, shear_strains=(0.01, math.nan)),
    ):
        with pytest.raises(ValueError):
            validate_elastic_rate_scan(
                spec, HOLDS, property_name="shear_modulus", target_rate=1.0
            )
    with pytest.raises(ValueError, match="Unknown"):
        validate_elastic_rate_scan(SPEC, HOLDS, property_name="bogus", target_rate=1.0)


def _workflow(root: Path, name: str) -> list[Path]:
    directories = _series(root, name, replicas=2)
    spec = replace(SPEC, load_stresses_bar=(0.0, 50.0, 100.0, 150.0))
    (root / WORKFLOW_NAME).write_text(
        json.dumps(
            {
                "request": {
                    "property_name": name,
                    "spec": asdict(spec),
                    "hold_times_ps": HOLDS,
                },
                "run_dirs": [d.name for d in directories],
            }
        )
    )
    return directories


@pytest.mark.parametrize("name", REPLICA_PROPERTIES)
def test_workflow_expansion_verifies_every_replica_and_hold(
    tmp_path: Path, name: str
) -> None:
    directories = _workflow(tmp_path, name)
    report = analyse_elastic_rates([tmp_path], property_name=name, target_rate=0.001)
    assert len(report.observations) == 6
    path = directories[-1] / "manifest.json"
    record = json.loads(path.read_text())
    record["stages"].pop(next(iter(record["stages"])))
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="incomplete"):
        analyse_elastic_rates([tmp_path], property_name=name, target_rate=0.001)


def test_resume_different_request_rejected_even_when_preparation_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_scan_dynamics(monkeypatch, elastic_rates, _fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_elastic_rate_scan(
            _run(),
            tmp_path,
            property_name="bulk_modulus",
            hold_times_ps=HOLDS,
            target_rate=1.0,
        )
    assert (tmp_path / WORKFLOW_NAME).is_file()
    with pytest.raises(MechanicalError, match="different settings"):
        run_elastic_rate_scan(
            _run(),
            tmp_path,
            property_name="bulk_modulus",
            hold_times_ps=(20.0, 60.0, 201.0),
            target_rate=1.0,
        )


def test_runs_without_a_workflow_record_cannot_be_adopted(tmp_path: Path) -> None:
    write_deformation(tmp_path / "rate_00")
    with pytest.raises(MechanicalError, match="settings cannot be verified"):
        run_elastic_rate_scan(
            _run(),
            tmp_path,
            property_name="poisson_ratio",
            hold_times_ps=HOLDS,
            target_rate=1.0,
        )


@pytest.mark.parametrize("name", PROPERTIES)
def test_all_passes_start_from_common_state_and_force_rerun_keeps_replicas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    calls: list[tuple[Protocol, dict[str, Any]]] = []
    Path("equilibrated.xml").write_text("original prepared state")

    def fake(protocol: Protocol, run: Any, directory: Path, **kwargs: Any) -> Any:
        calls.append((protocol, kwargs))
        return SimpleNamespace(final_state="equilibrated.xml")

    fake_scan_dynamics(monkeypatch, elastic_rates, fake)
    monkeypatch.setattr(
        elastic_rates, "analyse_elastic_rates", lambda *args, **kwargs: None
    )
    run_elastic_rate_scan(
        _run(),
        tmp_path,
        property_name=name,
        hold_times_ps=HOLDS,
        target_rate=1.0,
        spec=SPEC,
        resume=False,
    )
    assert len(calls) == 1 + 3 * SPEC.n_replicas
    assert calls[0][1]["resume"] is False
    assert all(options["state_in"] == "equilibrated.xml" for _, options in calls[1:])
    assert [options["resume"] for _, options in calls[1:]] == [False, True] * 3
    assert all(
        protocol.stages[0].options.get("reference_box_nm", [5.0] * 3) == [5.0] * 3
        for protocol, _ in calls[1:]
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ("shear_modulus", "bulk_modulus", "load_modulus"))
def test_real_elastic_scan_records_metadata_and_resumes(
    tmp_path: Path, argon_run: Any, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # Tiny argon runs exercise dynamics and resume without asserting solid-like response.
    from openmmpolymer import simulate

    initialisations: list[tuple[str, bool]] = []
    original_initialise = simulate._initialise

    def tracked_initialise(*args: Any, **kwargs: Any) -> None:
        initialisations.append((args[2], kwargs.get("reuse_velocities", True)))
        original_initialise(*args, **kwargs)

    monkeypatch.setattr(simulate, "_initialise", tracked_initialise)
    monkeypatch.setattr(
        elastic_rates, "analyse_elastic_rates", lambda *args, **kwargs: None
    )
    spec = ModulusSpec(
        temperature_k=120.0,
        n_replicas=2,
        samples_per_step=4,
        shear_strains=(0.001, 0.002, 0.003),
        bulk_pressures_bar=(1.0, 5.0, 1.0),
        load_stresses_bar=(0.0, 1.0, 2.0),
    )
    options: dict[str, Any] = dict(
        property_name=name,
        hold_times_ps=(0.02, 0.04, 0.06),
        target_rate=1.0,
        spec=spec,
        nvt_ps=0.02,
        compress_ps_each=0.02,
        npt_ps=0.03,
        anneal_cycles=1,
        anneal_window_ps=0.01,
        anneal_hold_ps=0.01,
        compress_pressures_bar=(1.0, 5.0, 1.0),
    )
    run_elastic_rate_scan(argon_run, tmp_path, **options)
    record = json.loads((tmp_path / WORKFLOW_NAME).read_text())
    elastic_rates._check_recorded_scan(tmp_path, record, name)
    replicas = [reuse for label, reuse in initialisations if "_r" in label]
    assert replicas == [False] * 6
    manifests = [tmp_path / d / "manifest.json" for d in record["run_dirs"]]
    before = [json.loads(path.read_text())["stages"] for path in manifests]
    assert all(len(stages) == 2 for stages in before)
    for stages in before:
        for stage in stages.values():
            assert Path(stage["final_state"]).is_file()
    run_elastic_rate_scan(argon_run, tmp_path, **options)
    assert before == [json.loads(path.read_text())["stages"] for path in manifests]


@pytest.mark.parametrize(
    "field, values",
    [
        ("segment_duration_ps", [10.0] * 4),
        ("segment_temperature_k", [310.0] * 4),
        ("shear_plane", [1.0, 2.0]),
        ("lateral_pressure_bar", [2.0]),
        ("segment_shear_strain", [0.001, 0.002, 0.003, 0.004]),
    ],
)
def test_workflow_refuses_changed_recorded_settings(
    tmp_path: Path, field: str, values: list[float]
) -> None:
    directories = _workflow(tmp_path, "shear_modulus")
    path = directories[0] / "manifest.json"
    record = json.loads(path.read_text())
    next(iter(record["stages"].values()))["samples"][field] = values
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match=r"different|differs"):
        analyse_elastic_rates(
            [tmp_path], property_name="shear_modulus", target_rate=0.001
        )


def test_missing_duration_or_duplicate_directory_never_guesses_rate(
    tmp_path: Path,
) -> None:
    directories = _series(tmp_path, "shear_modulus")
    with pytest.raises(AnalysisError, match="more than once"):
        analyse_elastic_rates(
            [directories[0]] * 2, property_name="shear_modulus", target_rate=0.001
        )
    with pytest.raises(AnalysisError, match="Supply run directories"):
        analyse_elastic_rates([], property_name="shear_modulus", target_rate=0.001)
    file = directories[0] / "manifest.json"
    record = json.loads(file.read_text())
    next(iter(record["stages"].values()))["samples"].pop("segment_duration_ps")
    file.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="recorded positive durations"):
        analyse_elastic_rates(
            directories, property_name="shear_modulus", target_rate=0.001
        )


def test_resume_refuses_missing_states_and_changed_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run()
    fake_scan_dynamics(monkeypatch, elastic_rates, _fail)
    options: dict[str, Any] = dict(
        property_name="bulk_modulus", hold_times_ps=HOLDS, target_rate=1.0
    )
    with pytest.raises(RuntimeError):
        run_elastic_rate_scan(run, tmp_path, **options)
    child = tmp_path / "equilibration"
    child.mkdir()
    (child / "manifest.json").write_text(
        json.dumps({"stages": {"00_minimise": {"final_state": "/no/such/state.xml"}}})
    )
    with pytest.raises(MechanicalError, match="missing states"):
        run_elastic_rate_scan(run, tmp_path, **options)
    run.box.positions_nm[0, 0] = 1.0
    with pytest.raises(MechanicalError, match="different settings"):
        run_elastic_rate_scan(run, tmp_path, **options)


def _interrupted_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "shear_modulus"
) -> tuple[Any, dict[str, Any], Path, list[str]]:
    state = tmp_path / "prepared.xml"
    state.write_text("original preparation state")
    calls: list[str] = []

    def fake(protocol: Protocol, run: Any, directory: Path, **kwargs: Any) -> Any:
        calls.append(directory.name)
        if directory.name == "equilibration":
            return SimpleNamespace(final_state=str(state))
        raise RuntimeError("interrupted measurement")

    fake_scan_dynamics(monkeypatch, elastic_rates, fake)
    run = _run()
    options: dict[str, Any] = dict(
        property_name=name, hold_times_ps=HOLDS, target_rate=0.001
    )
    with pytest.raises(RuntimeError, match="interrupted measurement"):
        run_elastic_rate_scan(run, tmp_path, **options)
    return run, options, state, calls


@pytest.mark.parametrize("name", ["shear_modulus", "youngs_modulus"])
def test_resume_rejects_changed_prepared_state_before_running_any_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    run, options, state, calls = _interrupted_branch(tmp_path, monkeypatch, name)
    workflow = tmp_path / (
        YOUNGS_WORKFLOW_NAME if name == "youngs_modulus" else WORKFLOW_NAME
    )
    record_before = workflow.read_text()
    prior_calls = list(calls)
    state.write_text("different preparation with the same system and coordinates")
    with pytest.raises(MechanicalError, match="no longer matches"):
        run_elastic_rate_scan(run, tmp_path, **options)
    assert calls == prior_calls
    assert workflow.read_text() == record_before


def test_prepared_state_fingerprint_survives_interrupted_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, options, state, _ = _interrupted_branch(tmp_path, monkeypatch)
    record_before = json.loads((tmp_path / WORKFLOW_NAME).read_text())
    fake_scan_dynamics(monkeypatch, elastic_rates, _fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_elastic_rate_scan(run, tmp_path, **options)
    record_after = json.loads((tmp_path / WORKFLOW_NAME).read_text())
    assert record_after["start_state_sha256"] == record_before["start_state_sha256"]
    assert record_after["start_state"] == str(state)


def test_resume_refuses_a_different_returned_preparation_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, options, _, calls = _interrupted_branch(tmp_path, monkeypatch)
    changed = tmp_path / "other-preparation.xml"
    changed.write_text("another state")
    monkeypatch.setattr(
        "openmmpolymer._workflow.settled_state", lambda *args, **kwargs: str(changed)
    )
    prior_count = len(calls)
    with pytest.raises(MechanicalError, match="preparation state changed"):
        run_elastic_rate_scan(run, tmp_path, **options)
    assert calls[prior_count:] == ["equilibration"]


def test_existing_branches_without_preparation_fingerprint_cannot_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, options, _, _ = _interrupted_branch(tmp_path, monkeypatch)
    path = tmp_path / WORKFLOW_NAME
    record = json.loads(path.read_text())
    record.pop("start_state_sha256")
    path.write_text(json.dumps(record))
    child = tmp_path / "rate_00"
    child.mkdir()
    (child / "manifest.json").write_text(json.dumps({"stages": {}}))
    with pytest.raises(MechanicalError, match="no preparation-state fingerprint"):
        run_elastic_rate_scan(run, tmp_path, **options)


@pytest.mark.parametrize(
    "changed", ["preparation", "system_sha256", "preparation_state_sha256"]
)
@pytest.mark.parametrize("children", [False, True])
def test_saved_elastic_rates_compare_preparation_and_system_provenance(
    tmp_path: Path, changed: str, children: bool
) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    paths: list[Path] = []
    for index, root in enumerate(roots):
        directories = _workflow(root, "shear_modulus")
        paths.extend(directories if children else [root])
        path = root / WORKFLOW_NAME
        record = json.loads(path.read_text())
        record["request"]["equilibration"] = {
            "name": "prepare",
            "stages": [{"temperature_k": 300.0}],
        }
        record["request"]["system_sha256"] = "same system"
        record["start_state_sha256"] = "same prepared state"
        if index == 1:
            if changed == "preparation":
                record["request"]["equilibration"]["stages"][0]["temperature_k"] = 600.0
            elif changed == "system_sha256":
                record["request"]["system_sha256"] = "other system"
            else:
                record["start_state_sha256"] = "other prepared state"
        path.write_text(json.dumps(record))
    report = analyse_elastic_rates(
        paths, property_name="shear_modulus", target_rate=0.001
    )
    assert report.log_linear is None and report.power_law is None
    assert any("same measurement conditions" in note for note in report.notes)


def test_poisson_child_analysis_inherits_original_young_scan_preparation(
    tmp_path: Path,
) -> None:
    directories = _workflow(tmp_path, "poisson_ratio")
    source = tmp_path / WORKFLOW_NAME
    record = json.loads(source.read_text())
    record["request"].pop("property_name")
    record["request"]["relax_ps"] = record["request"].pop("hold_times_ps")
    preparation = {"name": "prepare", "stages": [{"temperature_k": 300.0}]}
    record["request"]["equilibration"] = {"protocol": preparation}
    (tmp_path / YOUNGS_WORKFLOW_NAME).write_text(json.dumps(record))
    source.unlink()
    report = analyse_elastic_rates(
        directories, property_name="poisson_ratio", target_rate=0.001
    )
    assert all(
        item.conditions["preparation"] == preparation for item in report.observations
    )
    assert report.log_linear is not None and report.log_linear.resolved


# --------------------------------------------------------------------------
# Young's modulus: its own scan record, one pooled fit per rate
# --------------------------------------------------------------------------

YOUNGS_HOLDS = (50.0, 150.0, 500.0)
YOUNGS_SPEC = ModulusSpec(n_replicas=2, max_strain=0.02)


def _youngs(directories: list[Path], **options: Any) -> Any:
    return analyse_elastic_rates(
        directories,
        property_name="youngs_modulus",
        target_rate=options.pop("target_rate", 0.001),
        **options,
    )


def test_saved_rates_recover_a_planted_logarithmic_law(tmp_path: Path) -> None:
    report = _youngs(write_modulus_rate_series(tmp_path))
    assert report.log_linear is not None
    assert report.log_linear.value == pytest.approx(1700.0)
    assert report.log_linear.sensitivity_per_decade == pytest.approx(300.0)
    assert report.log_linear.resolved
    assert len(report.observations) == 3
    assert all(item.resolved for item in report.observations)
    rates = [item.rate for item in report.observations]
    assert rates == sorted(rates)


def test_each_rate_keeps_only_its_own_notes(tmp_path: Path) -> None:
    """Report notes were copied onto every rate, where they read as its own."""
    directories = write_modulus_rate_series(tmp_path)
    report = _youngs(directories)
    for item in report.observations:
        assert item.notes == (
            f"Rate {item.rate:.6g} /ns has one replica; no replica spread is available.",
        )
    assert {item.source for item in report.observations} == {
        f"{directory.resolve()}: 06_deform_r0_00" for directory in directories
    }
    per_observation = [note for note in report.notes if note.startswith("Observation")]
    assert len(per_observation) == 3


def test_same_rate_directories_are_replicas_not_extra_rate_observations(
    tmp_path: Path,
) -> None:
    directories = write_modulus_rate_series(tmp_path)
    duplicate = tmp_path / "replica"
    write_deformation(duplicate, relax_ps=YOUNGS_HOLDS[-1], modulus_mpa=2000.0)
    report = _youngs([*directories, duplicate])
    assert len(report.observations) == 3
    assert report.log_linear is not None and report.log_linear.n_rates == 3
    slower = planted_modulus_mpa(deformation_rate_per_ns(500.0))
    assert report.observations[0].value == pytest.approx((slower + 2000) / 2)
    error = report.observations[0].standard_error
    assert error is not None
    assert error >= np.std([slower, 2000], ddof=1) - 1.0e-8
    assert not any("one replica" in note for note in report.observations[0].notes)


def test_excessive_replica_spread_remains_unresolved(tmp_path: Path) -> None:
    directories = write_modulus_rate_series(tmp_path)
    write_deformation(
        directories[0],
        stage="06_deform_r1_00",
        modulus_mpa=6000.0,
        relax_ps=YOUNGS_HOLDS[0],
    )
    report = _youngs(directories)
    slowest = report.observations[-1]
    assert not slowest.resolved
    assert slowest.standard_error is not None and slowest.standard_error > 2000.0
    assert any("replica quality and spread" in note for note in slowest.notes)
    assert report.log_linear is not None and not report.log_linear.resolved
    assert report.power_law is not None and not report.power_law.resolved


def test_a_nonpositive_pooled_modulus_leaves_the_fits_unavailable(
    tmp_path: Path,
) -> None:
    """It is a measurement outside the property's bounds, not a broken input."""
    directories = write_modulus_rate_series(
        tmp_path, modulus=lambda rate: -50.0 if rate > 0.03 else 1000.0
    )
    report = _youngs(directories)
    assert report.log_linear is None and report.power_law is None
    assert len(report.observations) == 3
    assert any("physical bounds" in note for note in report.notes)


def test_fewer_than_three_rates_leave_the_fits_unavailable(tmp_path: Path) -> None:
    report = _youngs(write_modulus_rate_series(tmp_path, (50.0, 500.0)))
    assert report.log_linear is None and report.power_law is None
    assert any("three or more distinct" in note for note in report.notes)


@pytest.mark.parametrize(
    "changed",
    [{"axis": 0}, {"increment": 0.001}, {"temperature_k": 310.0}],
)
def test_saved_rates_must_measure_comparable_deformations(
    tmp_path: Path, changed: dict[str, Any]
) -> None:
    directories = write_modulus_rate_series(tmp_path)
    write_deformation(directories[-1], relax_ps=YOUNGS_HOLDS[-1], **changed)
    with pytest.raises(AnalysisError, match="same"):
        _youngs(directories)


def test_missing_recorded_rate_and_repeated_directories_are_refused(
    tmp_path: Path,
) -> None:
    directories = write_modulus_rate_series(tmp_path)
    path = directories[0] / "manifest.json"
    record = json.loads(path.read_text())
    del record["stages"]["06_deform_r0_00"]["samples"]["segment_duration_ps"]
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="positive strain rate"):
        _youngs(directories)
    with pytest.raises(AnalysisError, match="more than once"):
        _youngs([directories[1], directories[1]])
    write_shear(tmp_path / "sheared")
    with pytest.raises(AnalysisError, match="no strain-controlled extension"):
        _youngs([tmp_path / "sheared"])


def test_scan_root_expands_only_its_recorded_directories(tmp_path: Path) -> None:
    directories = write_modulus_rate_series(tmp_path)
    (tmp_path / YOUNGS_WORKFLOW_NAME).write_text(
        json.dumps({"run_dirs": [directory.name for directory in directories]})
    )
    write_deformation(tmp_path / "unrelated", relax_ps=123.0)
    assert len(_youngs([tmp_path]).run_dirs) == 3
    (directories[1] / "manifest.json").unlink()
    with pytest.raises(AnalysisError, match="No completed rate manifest"):
        _youngs([tmp_path])
    (tmp_path / YOUNGS_WORKFLOW_NAME).write_text(json.dumps({"run_dirs": []}))
    with pytest.raises(AnalysisError, match="records no rate run directories"):
        _youngs([tmp_path])


def _youngs_dynamics(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    fake_scan_dynamics(monkeypatch, elastic_rates, planted_extension_runner(calls))
    return calls


def _youngs_scan(run: Any, directory: Path, **options: Any) -> Any:
    return run_elastic_rate_scan(
        run,
        directory,
        property_name="youngs_modulus",
        hold_times_ps=options.pop("hold_times_ps", YOUNGS_HOLDS),
        target_rate=0.001,
        spec=options.pop("spec", YOUNGS_SPEC),
        **options,
    )


def test_youngs_scans_keep_the_original_record_and_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing scans resume only if new ones record exactly what they did."""
    calls = _youngs_dynamics(monkeypatch)
    run = _run()
    report = _youngs_scan(run, tmp_path)
    assert not (tmp_path / WORKFLOW_NAME).exists()
    record = json.loads((tmp_path / YOUNGS_WORKFLOW_NAME).read_text())
    request = record["request"]
    assert set(request) == {
        "spec",
        "relax_ps",
        "equilibration",
        "target_rate_per_ns",
        "max_extrapolation_decades",
        "system",
        "seed",
    }
    assert request["relax_ps"] == list(YOUNGS_HOLDS)
    assert set(request["equilibration"]) == {"protocol"}
    assert request["target_rate_per_ns"] == 0.001
    assert request["system"] == asdict(run.spec) and request["seed"] == 17
    assert record["run_dirs"] == ["rate_00", "rate_01", "rate_02"]
    assert {"start_state", "reference_box_nm", "timestep_fs"} <= set(record)
    branches = calls[1:]
    assert [call["directory"].name for call in branches] == [
        name for name in record["run_dirs"] for _ in range(YOUNGS_SPEC.n_replicas)
    ]
    assert [call["protocol"].stages[0].name for call in branches] == [
        stage.name
        for index, hold in enumerate(YOUNGS_HOLDS)
        for replica in range(YOUNGS_SPEC.n_replicas)
        for stage in deform_protocol(
            replace(YOUNGS_SPEC, relax_ps=hold),
            timestep_fs=2.0,
            replica=index * YOUNGS_SPEC.n_replicas + replica,
        ).stages
    ]
    assert {str(call["state_in"]) for call in branches} == {record["start_state"]}
    assert report.property.name == "youngs_modulus"
    assert len(report.observations) == 3


def test_a_record_from_before_fingerprints_still_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Young's scans predate the common-state fingerprint the others keep."""
    _youngs_dynamics(monkeypatch)
    _youngs_scan(_run(), tmp_path)
    path = tmp_path / YOUNGS_WORKFLOW_NAME
    record = json.loads(path.read_text())
    del record["start_state_sha256"]
    path.write_text(json.dumps(record))
    _youngs_scan(_run(), tmp_path)
    assert "start_state_sha256" in json.loads(path.read_text())


@pytest.mark.parametrize("interrupted_at", ["equilibration", "rate_01"])
def test_interrupted_forced_rerun_does_not_reuse_removed_replicas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted_at: str
) -> None:
    calls: list[dict[str, Any]] = []
    dynamics = planted_extension_runner(calls)
    fake_scan_dynamics(monkeypatch, elastic_rates, dynamics)
    run = _run()
    _youngs_scan(run, tmp_path)
    manifests = [tmp_path / f"rate_{index:02d}/manifest.json" for index in range(3)]
    assert all(len(json.loads(path.read_text())["stages"]) == 2 for path in manifests)

    def interrupted(
        protocol: Protocol, run: Any, directory: Path, **options: Any
    ) -> Any:
        if directory.name == interrupted_at:
            raise RuntimeError("interrupted forced rerun")
        return dynamics(protocol, run, directory, **options)

    fake_scan_dynamics(monkeypatch, elastic_rates, interrupted)
    smaller = replace(YOUNGS_SPEC, n_replicas=1)
    with pytest.raises(RuntimeError, match="interrupted forced rerun"):
        _youngs_scan(run, tmp_path, spec=smaller, resume=False)
    record = json.loads((tmp_path / YOUNGS_WORKFLOW_NAME).read_text())
    assert record["request"]["spec"]["n_replicas"] == 1
    assert all(not path.exists() for path in manifests[1:])

    # Preparation deliberately produces the same digest on every attempt.
    # Resuming must still have no old replica available for analysis.
    fake_scan_dynamics(monkeypatch, elastic_rates, dynamics)
    report = _youngs_scan(run, tmp_path, spec=smaller)
    assert len(report.observations) == 3
    for index, path in enumerate(manifests):
        assert set(json.loads(path.read_text())["stages"]) == {f"06_deform_r{index}_00"}


def test_changed_hamiltonian_is_rejected_before_workflow_metadata_is_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argon_run: Any
) -> None:
    from openmmpolymer.protocols import ProtocolError, Stage
    from openmmpolymer.protocols import run_protocol as real_run_protocol

    _youngs_dynamics(monkeypatch)
    _youngs_scan(argon_run, tmp_path)
    # Record real input provenance without running the expensive rate scan.
    real_run_protocol(
        Protocol("initial", (Stage("00_minimise", "minimise"),)),
        argon_run,
        tmp_path / "equilibration",
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    changed = replace(argon_run, system_xml=argon_run.system_xml + "\n")
    with pytest.raises(ProtocolError, match="starting inputs changed"):
        _youngs_scan(changed, tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "changed", [{"hold_times_ps": (50.0, 200.0, 500.0)}, {"npt_ps": 50.0}]
)
def test_changed_rates_or_equilibration_cannot_resume_an_interrupted_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: dict[str, Any]
) -> None:
    fake_scan_dynamics(monkeypatch, elastic_rates, _fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        _youngs_scan(_run(), tmp_path)
    assert (tmp_path / YOUNGS_WORKFLOW_NAME).is_file()
    with pytest.raises(MechanicalError, match="different settings"):
        _youngs_scan(_run(), tmp_path, **changed)


def test_budget_failure_creates_no_directory(tmp_path: Path) -> None:
    target = tmp_path / "unstarted"
    with pytest.raises(MechanicalError, match="budget"):
        _youngs_scan(_run(), target, spec=replace(YOUNGS_SPEC, max_total_ns=0.001))
    assert not target.exists()


def test_recorded_scan_refuses_an_incomplete_replica_or_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _youngs_dynamics(monkeypatch)
    _youngs_scan(_run(), tmp_path)
    manifest = tmp_path / "rate_00" / "manifest.json"
    pristine = manifest.read_text()
    record = json.loads(pristine)
    record["stages"]["06_deform_r0_00"]["samples"]["segment_duration_ps"][0] += 1.0
    manifest.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="different rate holds"):
        _youngs([tmp_path])
    record = json.loads(pristine)
    del record["stages"]["06_deform_r1_00"]
    manifest.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="incomplete"):
        _youngs([tmp_path])


@pytest.mark.slow
def test_real_youngs_scan_retains_replicas_common_reference_and_resume(
    tmp_path: Path, argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Argon is not an elastic solid. Inspect real output rather than asking
    # its noisy extension slopes to support a positive modulus extrapolation.
    monkeypatch.setattr(
        elastic_rates, "analyse_elastic_rates", lambda *args, **kwargs: None
    )
    spec = ModulusSpec(
        temperature_k=120.0,
        strain_increment=0.004,
        max_strain=0.012,
        elastic_strain_limit=0.012,
        n_replicas=2,
        samples_per_step=4,
        stage_ps=0.3,
    )
    holds = (0.1, 0.2, 0.3)
    options: dict[str, Any] = {
        "spec": spec,
        "hold_times_ps": holds,
        "nvt_ps": 0.2,
        "compress_ps_each": 0.2,
        "npt_ps": 0.3,
        "anneal_cycles": 1,
        "anneal_window_ps": 0.1,
        "anneal_hold_ps": 0.1,
        "compress_pressures_bar": (1.0, 20.0, 1.0),
    }
    _youngs_scan(argon_run, tmp_path, **dict(options))
    workflow = json.loads((tmp_path / YOUNGS_WORKFLOW_NAME).read_text())
    reference = workflow["reference_box_nm"]
    manifests = [tmp_path / name / "manifest.json" for name in workflow["run_dirs"]]
    before = [json.loads(path.read_text())["stages"] for path in manifests]
    for rate_index, (stages, hold) in enumerate(zip(before, holds, strict=True)):
        for replica in range(spec.n_replicas):
            replica_index = rate_index * spec.n_replicas + replica
            entries = [
                entry for name, entry in stages.items() if f"_r{replica_index}_" in name
            ]
            assert entries
            assert (
                sum(len(entry["samples"]["segment_strain"]) for entry in entries) == 3
            )
            for entry in entries:
                samples = entry["samples"]
                assert samples["reference_box_nm"] == pytest.approx(reference)
                assert samples["segment_duration_ps"] == pytest.approx(
                    [hold] * len(samples["segment_strain"])
                )
                assert Path(entry["final_state"]).is_file()
    _youngs_scan(argon_run, tmp_path, **dict(options))
    after = [json.loads(path.read_text())["stages"] for path in manifests]
    assert after == before
