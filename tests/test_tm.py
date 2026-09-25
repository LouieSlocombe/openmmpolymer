"""Melting scans preserve a crystal and distinguish a jump from a Tg corner.

The transition estimator is exercised against independent planted curves.
One tiny CPU run checks the saved measurement, resume and input provenance;
argon is a plumbing fixture, not a polymer melting-temperature benchmark.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.protocols import RunManifest, run_protocol
from openmmpolymer.simulate import run_nvt
from openmmpolymer.tm import (
    HeatingCurve,
    TmError,
    TmSpec,
    analyse_melting,
    heating_curve,
    heating_stages,
    melting_scan,
    melting_temperature,
    run_tm_scan,
    write_melting_report,
)
from openmmpolymer.trajectory import AnalysisError


def planted_curve(
    *,
    volume_jump: float = 0.1,
    enthalpy_jump: float = 400.0,
    volume_split: int = 10,
    enthalpy_split: int = 10,
    noise: float = 1.0,
) -> HeatingCurve:
    """Two expanding branches separated by a known first-order jump."""
    temperatures = np.arange(300.0, 500.0, 10.0)
    index = np.arange(len(temperatures))
    # Noise is much smaller than a real jump, and differs between observables.
    volume = (
        1.0
        + 0.0003 * (temperatures - 300.0)
        + volume_jump * (index >= volume_split)
        + noise * 0.0002 * np.sin(index * 2.1)
    )
    enthalpy = (
        -4000.0
        + 5.0 * (temperatures - 300.0)
        + enthalpy_jump * (index >= enthalpy_split)
        + noise * 0.7 * np.cos(index * 1.7)
    )
    return HeatingCurve(
        temperature_k=tuple(float(t) for t in temperatures),
        density_g_cm3=tuple(float(1.0 / v) for v in volume),
        enthalpy_kj_mol=tuple(float(h) for h in enthalpy),
        hold_ps=(1000.0,) * len(temperatures),
        pressure_bar=(1.0,) * len(temperatures),
        stages=("heating",),
    )


def write_heating(
    directory: Path,
    curve: HeatingCurve,
    *,
    chunks: tuple[int, ...] = (9, 10, 1),
) -> Path:
    """Store the public stage-result schema without running dynamics."""
    directory.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}
    start = 0
    for number, length in enumerate(chunks):
        stop = start + length
        name = f"heat_{number:02d}"
        stages[name] = {
            "name": name,
            "samples": {
                "segment_temperature_k": list(curve.temperature_k[start:stop]),
                "segment_density_g_cm3": list(curve.density_g_cm3[start:stop]),
                "segment_enthalpy_kj_mol": list(curve.enthalpy_kj_mol[start:stop]),
                "segment_duration_ps": list(curve.hold_ps[start:stop]),
                "segment_pressure_bar": list(curve.pressure_bar[start:stop]),
            },
        }
        start = stop
    assert start == curve.n_points
    RunManifest(protocol="tm_heating", seed=11, stages=stages).save(directory)
    return directory


def test_matching_enthalpy_and_volume_jumps_resolve_a_temperature_bracket() -> None:
    transition = melting_temperature(planted_curve())

    assert transition.resolved
    assert transition.bracket_k == (390.0, 400.0)
    assert transition.temperature_k == pytest.approx(395.0)
    assert transition.volume_jump_cm3_g == pytest.approx(0.1, abs=0.001)
    assert transition.enthalpy_jump_kj_mol == pytest.approx(400.0, abs=3.0)


def test_exact_piecewise_lines_are_resolved_without_dividing_by_zero() -> None:
    transition = melting_temperature(planted_curve(noise=0.0))
    assert transition.resolved
    assert transition.temperature_k == pytest.approx(395.0)


@pytest.mark.parametrize("noise", [0.0, 1.0, 10.0])
def test_linear_thermal_expansion_does_not_get_a_melting_temperature(
    noise: float,
) -> None:
    transition = melting_temperature(
        planted_curve(volume_jump=0.0, enthalpy_jump=0.0, noise=noise)
    )
    assert not transition.resolved
    assert transition.temperature_k is None
    assert transition.bracket_k is None


def test_a_continuous_glass_transition_is_not_reported_as_melting() -> None:
    curve = planted_curve(volume_jump=0.0, enthalpy_jump=0.0)
    temperatures = np.asarray(curve.temperature_k)
    volume = np.asarray(curve.specific_volume_cm3_g) + 0.0008 * np.maximum(
        temperatures - 395.0, 0.0
    )
    enthalpy = np.asarray(curve.enthalpy_kj_mol) + 3.0 * np.maximum(
        temperatures - 395.0, 0.0
    )
    transition = melting_temperature(
        replace(
            curve,
            density_g_cm3=tuple(float(1.0 / value) for value in volume),
            enthalpy_kj_mol=tuple(float(value) for value in enthalpy),
        )
    )
    assert not transition.resolved
    assert transition.temperature_k is None


@pytest.mark.parametrize(
    ("volume_jump", "enthalpy_jump"),
    [(-0.1, 400.0), (0.1, -400.0), (0.0, 400.0), (0.1, 0.0)],
)
def test_both_observables_must_show_the_expected_positive_jump(
    volume_jump: float, enthalpy_jump: float
) -> None:
    transition = melting_temperature(
        planted_curve(volume_jump=volume_jump, enthalpy_jump=enthalpy_jump)
    )
    assert not transition.resolved
    assert transition.temperature_k is None
    assert transition.notes


def test_jumps_at_different_temperatures_do_not_agree_on_a_transition() -> None:
    transition = melting_temperature(planted_curve(volume_split=7, enthalpy_split=13))
    assert not transition.resolved
    assert transition.temperature_k is None


def test_a_broad_smooth_crossover_is_not_given_a_one_step_melting_bracket() -> None:
    curve = planted_curve()
    temperature = np.asarray(curve.temperature_k)
    x = (temperature - 300.0) / 190.0
    crossover = 1.0 / (1.0 + np.exp(-(temperature - 395.0) / 20.0))
    transition = melting_temperature(
        replace(
            curve,
            density_g_cm3=tuple(1.0 / (1.0 + 0.06 * x + 0.1 * crossover)),
            enthalpy_kj_mol=tuple(-4000.0 + 950.0 * x + 400.0 * crossover),
        )
    )
    assert not transition.resolved
    assert transition.bracket_k is None


def test_two_comparable_transitions_do_not_get_reduced_to_one_melting_point() -> None:
    curve = planted_curve()
    temperature = np.asarray(curve.temperature_k)
    x = (temperature - 300.0) / 190.0
    transitions = (temperature >= 360.0) + 0.75 * (temperature >= 440.0)
    transition = melting_temperature(
        replace(
            curve,
            density_g_cm3=tuple(1.0 / (1.0 + 0.06 * x + 0.1 * transitions)),
            enthalpy_kj_mol=tuple(-4000.0 + 950.0 * x + 400.0 * transitions),
        )
    )
    assert not transition.resolved
    assert transition.temperature_k is None


def test_shared_thermal_noise_is_rejected_in_most_repeated_scans() -> None:
    """Correlated density and energy noise are not independent confirmations.

    A statistical decision can have false positives; across these fixed seeds
    fewer than five per cent of transition-free curves may resolve.
    """
    curve = planted_curve()
    temperature = np.asarray(curve.temperature_k)
    x = (temperature - 300.0) / 190.0
    resolved = 0
    for seed in range(100):
        noise = np.random.default_rng(seed).normal(size=curve.n_points)
        transition = melting_temperature(
            replace(
                curve,
                density_g_cm3=tuple(1.0 / (1.0 + 0.06 * x + 0.003 * noise)),
                enthalpy_kj_mol=tuple(-4000.0 + 950.0 * x + 12.0 * noise),
            )
        )
        resolved += transition.resolved
    assert resolved < 5


def test_a_transition_at_the_end_has_no_measured_high_temperature_branch() -> None:
    transition = melting_temperature(planted_curve(volume_split=19, enthalpy_split=19))
    assert not transition.resolved
    assert transition.temperature_k is None


def test_the_curve_retains_rate_pressure_and_specific_volume() -> None:
    curve = planted_curve()
    assert curve.n_points == 20
    assert curve.heating_rate_k_per_ns == pytest.approx(10.0)
    assert curve.specific_volume_cm3_g[0] == pytest.approx(1.0)
    assert curve.pressure_bar == (1.0,) * 20


def test_a_mixed_pressure_history_cannot_be_a_constant_pressure_melting_curve() -> None:
    with pytest.raises(ValueError, match="pressure"):
        replace(planted_curve(), pressure_bar=(1.0,) * 10 + (2.0,) * 10)


def test_an_irregular_schedule_has_no_single_nominal_heating_rate() -> None:
    curve = planted_curve()
    assert (
        replace(curve, hold_ps=(1000.0,) * 19 + (2000.0,)).heating_rate_k_per_ns is None
    )
    irregular = (*curve.temperature_k[:10], 405.0, *curve.temperature_k[11:])
    assert replace(curve, temperature_k=irregular).heating_rate_k_per_ns is None


def test_too_few_temperatures_remain_an_unresolved_measurement() -> None:
    curve = planted_curve()
    short = HeatingCurve(
        curve.temperature_k[:4],
        curve.density_g_cm3[:4],
        curve.enthalpy_kj_mol[:4],
        curve.hold_ps[:4],
        curve.pressure_bar[:4],
    )
    result = melting_temperature(short)
    assert not result.resolved
    assert result.temperature_k is None
    assert result.notes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("t_start_k", 0.0),
        ("t_end_k", float("nan")),
        ("step_k", 0.0),
        ("hold_ps", -1.0),
        ("equilibration_ps", -1.0),
        ("pressure_bar", 0.0),
        ("stage_ps", 0.0),
        ("samples_per_segment", 0),
        ("min_points_per_branch", 1),
        ("max_total_ns", 0.0),
        ("barostat", "none"),
        ("trajectory_ps", -1.0),
    ],
)
def test_invalid_scan_settings_are_rejected_at_construction(
    field: str, value: Any
) -> None:
    with pytest.raises(ValueError):
        TmSpec(**{field: value})


def test_a_melting_scan_heats_instead_of_cooling() -> None:
    with pytest.raises(ValueError):
        TmSpec(t_start_k=500.0, t_end_k=300.0)


def test_protocol_preserves_the_crystal_and_visits_every_heating_point_once() -> None:
    spec = TmSpec(t_start_k=300.0, t_end_k=355.0, step_k=10.0, stage_ps=2000.0)
    protocol = melting_scan(spec)

    assert [stage.kind for stage in protocol.stages[:2]] == ["minimise", "npt"]
    assert protocol.stages[1].options["temperature_k"] == 300.0
    heating = protocol.stages[2:]
    assert all(stage.kind == "heat" for stage in heating)
    ladder = [t for stage in heating for t in stage.options["temperatures_k"]]
    assert ladder == [300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 355.0]
    # Unlike a quench, a trailing one-temperature chunk is kept as it is.
    assert [len(stage.options["temperatures_k"]) for stage in heating] == [2, 2, 2, 1]
    assert all(stage.options["barostat"] == "anisotropic" for stage in heating)
    assert protocol.total_duration_ps == pytest.approx(8000.0)


def test_an_entire_scan_is_costed_before_dynamics() -> None:
    # 41 temperatures at 1000 ps plus the 1000 ps initial hold.
    assert melting_scan(TmSpec()).total_duration_ps == pytest.approx(42_000.0)


def test_chunks_including_a_single_temperature_are_read_as_one_history(
    tmp_path: Path,
) -> None:
    source = planted_curve()
    write_heating(tmp_path, source)
    assert heating_stages(tmp_path) == ("heat_00", "heat_01", "heat_02")
    restored = heating_curve(tmp_path)
    assert restored.temperature_k == source.temperature_k
    assert restored.density_g_cm3 == source.density_g_cm3
    assert restored.enthalpy_kj_mol == source.enthalpy_kj_mol
    assert restored.hold_ps == source.hold_ps
    assert restored.pressure_bar == source.pressure_bar
    assert restored.stages == ("heat_00", "heat_01", "heat_02")


def test_cooling_and_unmeasured_stages_are_not_heating_data(tmp_path: Path) -> None:
    write_heating(tmp_path, planted_curve())
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    manifest.stages["a_quench"] = {
        "samples": {
            "segment_temperature_k": [500.0, 490.0],
            "segment_density_g_cm3": [0.8, 0.9],
            "segment_enthalpy_kj_mol": [1000.0, 900.0],
            "segment_duration_ps": [1000.0, 1000.0],
            "segment_pressure_bar": [1.0, 1.0],
        }
    }
    manifest.stages["ordinary_npt"] = {
        "samples": {
            "segment_temperature_k": [300.0],
            "segment_density_g_cm3": [1.0],
        }
    }
    manifest.save(tmp_path)
    assert heating_stages(tmp_path) == ("heat_00", "heat_01", "heat_02")


@pytest.mark.parametrize(
    ("sample", "values"),
    [
        ("segment_enthalpy_kj_mol", [1.0]),
        ("segment_density_g_cm3", [0.0] * 9),
        ("segment_enthalpy_kj_mol", [float("nan")] * 9),
        ("segment_duration_ps", [-1.0] * 9),
        ("segment_pressure_bar", [1.0] * 8),
    ],
)
def test_incomplete_or_nonphysical_samples_are_not_silently_fitted(
    tmp_path: Path, sample: str, values: list[float]
) -> None:
    write_heating(tmp_path, planted_curve())
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    manifest.stages["heat_00"]["samples"][sample] = values
    manifest.save(tmp_path)
    with pytest.raises(AnalysisError):
        heating_curve(tmp_path, stages=("heat_00", "heat_01", "heat_02"))


def test_reordered_chunks_cannot_be_sorted_into_a_different_thermal_history(
    tmp_path: Path,
) -> None:
    write_heating(tmp_path, planted_curve())
    with pytest.raises(AnalysisError):
        heating_curve(tmp_path, stages=("heat_01", "heat_00", "heat_02"))


def test_an_explicit_stage_name_can_select_one_recorded_heating_curve(
    tmp_path: Path,
) -> None:
    write_heating(tmp_path, planted_curve(), chunks=(20,))
    assert heating_curve(tmp_path, stages="heat_00").n_points == 20
    with pytest.raises(AnalysisError, match="missing"):
        heating_curve(tmp_path, stages="missing")


def test_analysis_needs_a_manifest_and_measured_heating_data(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError):
        analyse_melting(tmp_path)
    RunManifest(protocol="empty", seed=11).save(tmp_path)
    with pytest.raises(AnalysisError):
        analyse_melting(tmp_path)


def test_finished_scan_report_and_figures_keep_the_temperature_bracket(
    tmp_path: Path,
) -> None:
    write_heating(tmp_path, planted_curve())
    report = analyse_melting(tmp_path)
    assert report.resolved
    assert report.temperature_k == pytest.approx(395.0)
    files = write_melting_report(report)
    payload = json.loads(Path(files.json).read_text())
    assert Path(files.json).name == "tm.json"
    assert "395" in json.dumps(payload)
    assert len(files.figures) == 1
    assert Path(files.figures[0]).name == "melting.png"
    assert Path(files.figures[0]).stat().st_size > 1000


def test_report_can_write_json_without_figures(tmp_path: Path) -> None:
    write_heating(tmp_path, planted_curve(volume_jump=0.0, enthalpy_jump=0.0))
    report = analyse_melting(tmp_path)
    assert not report.resolved
    assert report.temperature_k is None
    files = write_melting_report(report, tmp_path / "report", figures=False)
    assert Path(files.json).parent == tmp_path / "report"
    assert files.figures == ()

    # Reject JavaScript's permissive NaN/Infinity JSON extensions as well.
    def reject_constant(value: str) -> None:
        raise AssertionError(f"Non-finite JSON value: {value}")

    json.loads(Path(files.json).read_text(), parse_constant=reject_constant)


def test_an_unsupported_figure_format_is_refused(tmp_path: Path) -> None:
    write_heating(tmp_path, planted_curve())
    with pytest.raises(ValueError, match="figure_format"):
        write_melting_report(analyse_melting(tmp_path), figure_format="not-a-format")


def test_unconfirmed_amorphous_coordinates_are_refused_before_dynamics(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_run(*args: Any, **kwargs: Any) -> None:
        pytest.fail("The crystalline-start guard must run before dynamics")

    monkeypatch.setattr("openmmpolymer.tm.run_protocol", unexpected_run)
    with pytest.raises(TmError, match="crystall"):
        run_tm_scan(argon_run, crystalline=False)


@pytest.mark.parametrize("control", ["thermostat", "barostat"])
def test_imported_ensemble_controls_are_refused_before_heating(
    control: str,
    argon_run: Any,
    tmp_path: Path,
) -> None:
    import openmm as mm

    system = mm.XmlSerializer.deserialize(argon_run.system_xml)
    force = (
        mm.AndersenThermostat(100.0, 100.0)
        if control == "thermostat"
        else mm.MonteCarloBarostat(1.0, 100.0)
    )
    system.addForce(force)
    argon_run.system_xml = mm.XmlSerializer.serialize(system)
    directory = tmp_path / "conflicting_controls"
    with pytest.raises(TmError, match="barostat or Andersen thermostat"):
        run_tm_scan(argon_run, directory, crystalline=True)
    assert not directory.exists()


def test_the_budget_guard_includes_low_temperature_equilibration(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_run(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An over-budget scan must not run dynamics")

    monkeypatch.setattr("openmmpolymer.tm.run_protocol", unexpected_run)
    with pytest.raises(TmError):
        run_tm_scan(argon_run, crystalline=True, spec=TmSpec(max_total_ns=41.5))


def test_a_different_workflow_manifest_is_not_relabelled_as_a_melting_scan(
    argon_run: Any, tmp_path: Path
) -> None:
    directory = tmp_path / "existing"
    directory.mkdir()
    RunManifest(protocol="melt_quench", seed=11).save(directory)
    before = (directory / "manifest.json").read_text()
    with pytest.raises(TmError):
        run_tm_scan(argon_run, directory, crystalline=True)
    assert (directory / "manifest.json").read_text() == before


def test_cpu_scan_records_heating_and_resumes_only_identical_inputs(
    argon_run: Any, tmp_path: Path
) -> None:
    spec = TmSpec(
        t_start_k=120.0,
        t_end_k=125.0,
        step_k=1.0,
        hold_ps=0.2,
        equilibration_ps=0.2,
        stage_ps=0.4,
        samples_per_segment=4,
    )
    directory = tmp_path / "real"
    first = run_tm_scan(argon_run, directory, spec=spec, crystalline=True)
    assert Path(first.manifest_path).is_file()
    assert (directory / "tm_workflow.json").is_file()
    assert first.report.curve.temperature_k == (
        120.0,
        121.0,
        122.0,
        123.0,
        124.0,
        125.0,
    )
    assert np.all(np.isfinite(first.report.curve.enthalpy_kj_mol))
    assert first.report.curve.pressure_bar == (1.0,) * 6

    second = run_tm_scan(argon_run, directory, spec=spec, crystalline=True)
    assert not second.summary.results
    assert len(second.summary.skipped) == len(first.summary.results)
    assert second.report.curve == first.report.curve

    with pytest.raises(TmError):
        run_tm_scan(
            argon_run,
            directory,
            spec=replace(spec, hold_ps=0.3),
            crystalline=True,
        )
    argon_run.box.positions_nm[0, 0] += 0.01
    with pytest.raises(TmError):
        run_tm_scan(argon_run, directory, spec=spec, crystalline=True)
    argon_run.box.positions_nm[0, 0] -= 0.01
    (directory / "tm_workflow.json").unlink()
    with pytest.raises(TmError, match=r"record.*missing"):
        run_tm_scan(argon_run, directory, spec=spec, crystalline=True)


def test_an_explicit_starting_state_is_used_and_fingerprinted(
    argon_run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openmmpolymer import tm

    prepared = run_nvt(
        argon_run, tmp_path / "crystal", temperature_k=120.0, duration_ps=0.1
    )
    state_inputs: list[str | Path | None] = []

    def record_start(*args: Any, **kwargs: Any) -> Any:
        state_inputs.append(kwargs.get("state_in"))
        return run_protocol(*args, **kwargs)

    monkeypatch.setattr(tm, "run_protocol", record_start)
    spec = TmSpec(
        t_start_k=120.0,
        t_end_k=125.0,
        step_k=1.0,
        hold_ps=0.1,
        equilibration_ps=0.1,
        stage_ps=0.6,
        samples_per_segment=4,
    )
    directory = tmp_path / "from_state"
    result = run_tm_scan(
        argon_run,
        directory,
        spec=spec,
        state_in=prepared.final_state,
        crystalline=True,
    )
    assert state_inputs == [prepared.final_state]
    assert Path(result.summary.final_state).is_file()
    state_path = Path(prepared.final_state)
    state_path.write_text(state_path.read_text() + "\n")
    with pytest.raises(TmError, match="inputs changed"):
        run_tm_scan(
            argon_run,
            directory,
            spec=spec,
            state_in=prepared.final_state,
            crystalline=True,
        )
    assert state_inputs == [prepared.final_state]
