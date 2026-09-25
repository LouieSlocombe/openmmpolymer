"""Thermal rate parity preserves histories, brackets and missing transitions."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmmpolymer import thermal_rates
from openmmpolymer.protocols import Protocol, RunManifest
from openmmpolymer.tg import TgSpec
from openmmpolymer.thermal_rates import (
    WORKFLOW_NAME,
    ThermalRateError,
    analyse_thermal_rates,
    run_thermal_rate_scan,
    validate_thermal_rate_scan,
)
from openmmpolymer.tm import TmSpec
from openmmpolymer.trajectory import AnalysisError

from .helpers import planted_curve, write_heating, write_quench

HOLDS = (100.0, 1000.0, 10000.0)


def _tg_history(directory: Path, hold: float, transition: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    temperatures = np.arange(600.0, 199.0, -10.0)
    volume = (
        0.9
        + 0.00015 * (temperatures - transition)
        + 0.00045 * np.maximum(temperatures - transition, 0)
    )
    return write_quench(
        directory,
        temperatures,
        1 / volume,
        segment_duration_ps=[hold] * len(temperatures),
    )


def _tg_series(root: Path, replicas: int = 1) -> list[Path]:
    return [
        _tg_history(
            root / f"rate_{rate}_replica_{replica}",
            hold,
            350.0 + 20.0 * math.log10(10000 / hold) + 4 * replica,
        )
        for rate, hold in enumerate(HOLDS)
        for replica in range(replicas)
    ]


def test_saved_tg_rates_recover_log_law_without_inventing_single_fit_errors(
    tmp_path: Path,
) -> None:
    report = analyse_thermal_rates(
        _tg_series(tmp_path), property_name="glass_transition", target_rate=0.1
    )
    assert report.property.rate_unit == "K/ns"
    assert report.log_linear is not None
    assert report.log_linear.value == pytest.approx(330)
    assert report.log_linear.sensitivity_per_decade == pytest.approx(20)
    assert all(
        item.standard_error is None and item.temperature_k is None
        for item in report.observations
    )
    assert not report.log_linear.resolved
    assert any("preparation_state" in note for note in report.notes)


def test_replicate_thermal_temperatures_propagate_rate_uncertainty(
    tmp_path: Path,
) -> None:
    report = analyse_thermal_rates(
        _tg_series(tmp_path, replicas=2),
        property_name="glass_transition",
        target_rate=0.1,
    )
    assert len(report.observations) == 6
    assert report.log_linear is not None and report.log_linear.resolved
    assert report.log_linear.value == pytest.approx(332)
    assert report.log_linear.standard_error > 0
    assert report.log_linear.n_rates == 3


def test_saved_melting_brackets_are_preserved_but_not_standard_errors(
    tmp_path: Path,
) -> None:
    directories = []
    for index, hold in enumerate(HOLDS):
        curve = planted_curve(volume_split=12 - index, enthalpy_split=12 - index)
        directories.append(
            write_heating(
                tmp_path / str(index), replace(curve, hold_ps=(hold,) * curve.n_points)
            )
        )
    report = analyse_thermal_rates(
        directories, property_name="melting_temperature", target_rate=0.1
    )
    assert [item.value for item in report.observations] == [415, 405, 395]
    assert all(item.standard_error is None for item in report.observations)
    assert all(
        any("Finite-grid" in note for note in item.notes)
        for item in report.observations
    )
    assert report.log_linear is not None and not report.log_linear.resolved


def test_missing_melting_event_blocks_models_without_dropping_history(
    tmp_path: Path,
) -> None:
    directories = []
    for index, hold in enumerate(HOLDS):
        curve = planted_curve(volume_jump=0 if index == 1 else 0.1)
        directories.append(
            write_heating(
                tmp_path / str(index), replace(curve, hold_ps=(hold,) * curve.n_points)
            )
        )
    report = analyse_thermal_rates(
        directories, property_name="melting_temperature", target_rate=0.1
    )
    assert len(report.observations) == 3
    assert report.observations[1].value is None
    assert not report.observations[1].resolved
    assert report.log_linear is None and report.power_law is None


def test_too_short_cooling_histories_remain_unresolved_observations(
    tmp_path: Path,
) -> None:
    directories = []
    for index, hold in enumerate(HOLDS):
        directory = tmp_path / str(index)
        directory.mkdir()
        directories.append(
            write_quench(
                directory,
                [450, 400, 350, 300],
                [0.8, 0.81, 0.82, 0.83],
                segment_duration_ps=[hold] * 4,
            )
        )
    report = analyse_thermal_rates(
        directories, property_name="glass_transition", target_rate=0.1
    )
    assert len(report.observations) == 3
    assert all(item.value is None for item in report.observations)
    assert report.log_linear is None


@pytest.mark.parametrize("change", ["ladder", "pressure", "system"])
def test_different_saved_thermal_conditions_are_refused(
    tmp_path: Path, change: str
) -> None:
    directories = []
    for index, hold in enumerate(HOLDS):
        curve = planted_curve()
        if index == 1 and change == "ladder":
            curve = replace(
                curve, temperature_k=tuple(t + 5 for t in curve.temperature_k)
            )
        if index == 1 and change == "pressure":
            curve = replace(curve, pressure_bar=(2.0,) * curve.n_points)
        directory = write_heating(
            tmp_path / str(index), replace(curve, hold_ps=(hold,) * curve.n_points)
        )
        if index == 1 and change == "system":
            manifest = RunManifest.load(directory)
            assert manifest is not None
            manifest.system = {"hydrogen_mass_amu": 3}
            manifest.save(directory)
        directories.append(directory)
    with pytest.raises(AnalysisError, match="different"):
        analyse_thermal_rates(
            directories, property_name="melting_temperature", target_rate=0.1
        )


@pytest.mark.parametrize(
    "spec,property_name",
    [(TgSpec(), "glass_transition"), (TmSpec(), "melting_temperature")],
)
def test_thermal_budget_counts_common_preparation_and_every_replica(
    spec: TgSpec | TmSpec, property_name: str
) -> None:
    plan = validate_thermal_rate_scan(
        spec, HOLDS, property_name=property_name, target_rate=0.1, n_replicas=2
    )
    expected = (
        plan.equilibration.total_duration_ps + 2 * len(plan.temperatures_k) * sum(HOLDS)
    ) / 1000
    assert plan.total_ns == expected
    with pytest.raises(ThermalRateError, match="all rates and replicas"):
        validate_thermal_rate_scan(
            replace(spec, max_total_ns=expected - 1),
            HOLDS,
            property_name=property_name,
            target_rate=0.1,
            n_replicas=2,
        )
    assert all(
        stage.kind != ("quench" if property_name == "glass_transition" else "heat")
        for stage in plan.equilibration.stages
    )


@pytest.mark.parametrize(
    "spec,property_name",
    [(TgSpec(), "melting_temperature"), (TmSpec(), "glass_transition")],
)
def test_property_and_thermal_spec_must_agree(
    spec: TgSpec | TmSpec, property_name: str
) -> None:
    with pytest.raises(ValueError, match="requires"):
        validate_thermal_rate_scan(
            spec, HOLDS, property_name=property_name, target_rate=0.1
        )


@pytest.fixture
def fake_dynamics(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def runner(protocol: Protocol, run: Any, directory: Path, **kwargs: Any) -> Any:
        calls.append({"protocol": protocol, "directory": directory, **kwargs})
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / "state.xml"
        final.write_text("common configuration")
        if protocol.stages[0].kind in ("quench", "heat"):
            stages: dict[str, Any] = {}
            for stage in protocol.stages:
                options = stage.options
                temperatures = np.array(options["temperatures_k"])
                threshold = 400 + 10 * math.log10(1000 / options["hold_ps"])
                if stage.kind == "quench":
                    volume = (
                        0.9
                        + 0.00015 * (temperatures - threshold)
                        + 0.00045 * np.maximum(temperatures - threshold, 0)
                    )
                else:
                    volume = (
                        0.9
                        + 0.00015 * (temperatures - threshold)
                        + 0.1 * (temperatures > threshold)
                    )
                samples = {
                    "segment_temperature_k": list(temperatures),
                    "segment_density_g_cm3": list(1 / volume),
                    "segment_duration_ps": [options["hold_ps"]] * len(temperatures),
                }
                if stage.kind == "heat":
                    samples["segment_enthalpy_kj_mol"] = list(
                        -4000
                        + 5 * (temperatures - 300)
                        + 400 * (temperatures > threshold)
                    )
                    samples["segment_pressure_bar"] = [options["pressure_bar"]] * len(
                        temperatures
                    )
                stages[stage.name] = {"name": stage.name, "samples": samples}
            RunManifest(protocol=protocol.name, seed=run.seed, stages=stages).save(
                directory
            )
        else:
            RunManifest(protocol=protocol.name, seed=run.seed).save(directory)
        return SimpleNamespace(final_state=str(final))

    monkeypatch.setattr(thermal_rates, "run_protocol", runner)
    return calls


@pytest.mark.parametrize(
    "property_name,spec",
    [("glass_transition", TgSpec(coarse_step_k=10)), ("melting_temperature", TmSpec())],
)
def test_all_rates_and_replicas_branch_from_same_state_with_fresh_streams(
    tmp_path: Path,
    argon_run: Any,
    fake_dynamics: list[dict[str, Any]],
    property_name: str,
    spec: TgSpec | TmSpec,
) -> None:
    report = run_thermal_rate_scan(
        argon_run,
        tmp_path,
        property_name=property_name,
        spec=spec,
        hold_times_ps=HOLDS,
        target_rate=0.1,
        n_replicas=2,
        crystalline=True,
    )
    assert len(report.observations) == 6
    assert len(fake_dynamics) == 7
    branches = fake_dynamics[1:]
    assert {str(call["state_in"]) for call in branches} == {
        str(tmp_path / "equilibration/state.xml")
    }
    names = [call["protocol"].stages[0].name for call in branches]
    assert len(set(names)) == 6
    ladders = []
    for call in branches:
        stages = call["protocol"].stages
        assert stages[0].options["new_velocities"]
        assert all(not stage.options["new_velocities"] for stage in stages[1:])
        ladders.append([t for stage in stages for t in stage.options["temperatures_k"]])
    assert all(ladder == ladders[0] for ladder in ladders)
    assert (
        "melt" not in fake_dynamics[0]["protocol"].name
        if property_name == "melting_temperature"
        else True
    )


def test_melting_needs_explicit_crystal_assertion_before_writing(
    tmp_path: Path, argon_run: Any
) -> None:
    with pytest.raises(ThermalRateError, match="crystalline=True"):
        run_thermal_rate_scan(
            argon_run,
            tmp_path / "no",
            property_name="melting_temperature",
            hold_times_ps=HOLDS,
            target_rate=0.1,
        )
    assert not (tmp_path / "no").exists()


@pytest.mark.parametrize(
    "controls",
    [("isotropic",), ("flexible",), ("isotropic", "anisotropic"), ("andersen",)],
)
def test_a_system_that_controls_its_own_state_is_refused_before_writing(
    tmp_path: Path, argon_run: Any, controls: tuple[str, ...]
) -> None:
    """The stages attach their own barostat; a second would act beside it."""
    import openmm as mm

    forces = {
        "isotropic": lambda: mm.MonteCarloBarostat(1.0, 300.0),
        "anisotropic": lambda: mm.MonteCarloAnisotropicBarostat(
            mm.Vec3(1.0, 1.0, 1.0), 300.0
        ),
        "flexible": lambda: mm.MonteCarloFlexibleBarostat(1.0, 300.0),
        "membrane": lambda: mm.MonteCarloMembraneBarostat(
            1.0,
            0.0,
            300.0,
            mm.MonteCarloMembraneBarostat.XYIsotropic,
            mm.MonteCarloMembraneBarostat.ZFree,
        ),
        "andersen": lambda: mm.AndersenThermostat(300.0, 1.0),
    }
    system = mm.XmlSerializer.deserialize(argon_run.system_xml)
    for control in controls:
        system.addForce(forces[control]())
    argon_run.system_xml = mm.XmlSerializer.serialize(system)
    with pytest.raises(ThermalRateError, match="barostat or Andersen thermostat"):
        run_thermal_rate_scan(
            argon_run,
            tmp_path / "no",
            property_name="melting_temperature",
            hold_times_ps=HOLDS,
            target_rate=0.1,
            crystalline=True,
        )
    assert not (tmp_path / "no").exists()


@pytest.mark.parametrize("change", ["coordinates", "hold", "state", "preparation"])
def test_changed_thermal_inputs_are_refused_on_resume(
    tmp_path: Path,
    argon_run: Any,
    fake_dynamics: list[dict[str, Any]],
    change: str,
) -> None:
    state = tmp_path / "input.xml"
    state.write_text("initial coordinates")
    options: dict[str, Any] = {
        "property_name": "glass_transition",
        "hold_times_ps": HOLDS,
        "target_rate": 0.1,
        "n_replicas": 1,
        "state_in": state,
    }
    run_thermal_rate_scan(argon_run, tmp_path / "scan", **options)
    before = len(fake_dynamics)
    if change == "coordinates":
        argon_run.box.positions_nm[0, 0] += 0.01
    elif change == "state":
        state.write_text("different input")
    elif change == "hold":
        options["hold_times_ps"] = (100, 500, 10000)
    else:
        options["npt_ps"] = 1234
    with pytest.raises(ThermalRateError, match="different settings"):
        run_thermal_rate_scan(argon_run, tmp_path / "scan", **options)
    assert len(fake_dynamics) == before


@pytest.mark.parametrize("damage", ["directory", "stage", "hold", "temperature"])
def test_incomplete_or_mismatched_thermal_series_cannot_be_analysed(
    tmp_path: Path,
    argon_run: Any,
    fake_dynamics: list[dict[str, Any]],
    damage: str,
) -> None:
    run_thermal_rate_scan(
        argon_run,
        tmp_path,
        property_name="glass_transition",
        hold_times_ps=HOLDS,
        target_rate=0.1,
        n_replicas=1,
    )
    manifest_path = tmp_path / "rate_01/replica_00/manifest.json"
    payload = json.loads(manifest_path.read_text())
    if damage == "directory":
        manifest_path.unlink()
    else:
        name = next(iter(payload["stages"]))
        if damage == "stage":
            del payload["stages"][name]
        else:
            column = (
                "segment_duration_ps" if damage == "hold" else "segment_temperature_k"
            )
            payload["stages"][name]["samples"][column][0] += 1
        manifest_path.write_text(json.dumps(payload))
    with pytest.raises(AnalysisError, match=r"[Ii]ncomplete|Missing"):
        analyse_thermal_rates(
            [tmp_path], property_name="glass_transition", target_rate=0.1
        )


def test_workflow_is_saved_before_preparation_failure(
    tmp_path: Path, argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(thermal_rates, "run_protocol", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_thermal_rate_scan(
            argon_run,
            tmp_path,
            property_name="glass_transition",
            hold_times_ps=HOLDS,
            target_rate=0.1,
        )
    saved = json.loads((tmp_path / WORKFLOW_NAME).read_text())
    assert saved["request"]["property_name"] == "glass_transition"
    assert len(saved["entries"]) == 9
    assert saved["total_ns"] > 0


@pytest.mark.slow
def test_real_tiny_heating_series_records_three_rates_and_resumes(
    tmp_path: Path, argon_run: Any
) -> None:
    spec = TmSpec(
        t_start_k=100,
        t_end_k=120,
        step_k=4,
        hold_ps=0.04,
        equilibration_ps=0.04,
        stage_ps=0.2,
        samples_per_segment=4,
    )
    options: dict[str, Any] = {
        "property_name": "melting_temperature",
        "spec": spec,
        "hold_times_ps": (0.04, 0.08, 0.12),
        "target_rate": 10000,
        "n_replicas": 1,
        "crystalline": True,
    }
    report = run_thermal_rate_scan(argon_run, tmp_path, **options)
    assert len(report.observations) == 3
    assert sorted(item.rate for item in report.observations) == pytest.approx(
        [4 / 0.12 * 1000, 50000, 100000]
    )
    assert all(not item.resolved for item in report.observations)
    manifests = [Path(directory) / "manifest.json" for directory in report.run_dirs]
    before = [json.loads(path.read_text())["stages"] for path in manifests]
    resumed = run_thermal_rate_scan(argon_run, tmp_path, **options)
    assert len(resumed.observations) == 3
    assert [json.loads(path.read_text())["stages"] for path in manifests] == before
