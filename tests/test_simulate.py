"""Tests for the stages, run against a real but tiny argon cell.

No force-field file appears here. The cell is a real periodic
``NonbondedForce``, so the barostat has something to do and a density is a real
number, and everything the stage layer can get wrong is reachable in
milliseconds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmm as mm
import pytest
from openmm import unit

import openmmpolymer.simulate as simulate
from openmmpolymer.forcefield import PolymerForceField
from openmmpolymer.mdsystem import SystemSpec
from openmmpolymer.packing import AVOGADRO, NM3_PER_CM3
from openmmpolymer.simulate import (
    Segment,
    SimulationError,
    StageResult,
    _apply_positions,
    _check_deformed_box,
    _check_molecules,
    _density_g_cm3,
    _enthalpy_kj_mol,
    _Live,
    _open,
    _positions_nm,
    heating_temperatures,
    nonbonded_cutoff_nm,
    prepare_run,
    quench_temperatures,
    run_anneal,
    run_compress,
    run_deform,
    run_heat,
    run_load,
    run_minimise,
    run_npt,
    run_nvt,
    run_production,
    run_pushoff,
    run_quench,
    run_relax,
    run_segments,
    run_shear,
    safe_timestep_fs,
    set_pressures,
    set_temperature,
    temperature_k_of,
)
from openmmpolymer.stress import STRESS_ESTIMATOR_VERSION, StressError, affine_scale

from .helpers import argon_context, bare_simulation, rigid_rotor_system


@pytest.fixture(scope="module")
def minimised(tmp_path_factory: pytest.TempPathFactory) -> StageResult:
    """The argon cell minimised once, for every stage here to start from.

    At 120 K, near where these stages run, so the velocities the state carries
    are close to the ones they ask for.
    """
    return run_minimise(
        argon_context(64, 2.4),
        tmp_path_factory.mktemp("minimised") / "00_minimise",
        temperature_k=120.0,
    )


def _probe(
    run: Any,
    barostat: str | None = None,
    *,
    state_in: str | None = None,
    new_velocities: bool = False,
) -> Any:
    """A stage's Simulation, built and placed the way every stage's is, unrun."""
    return _open(
        run,
        "probe",
        temperature_k=120.0,
        timestep_fs=2.0,
        friction_ps=1.0,
        state_in=state_in,
        new_velocities=new_velocities,
        barostat=barostat,
    ).simulation


def test_prepare_run_measures_the_cell_mass(argon_run: Any) -> None:
    """64 argon atoms, and the density arithmetic depends on it."""
    assert argon_run.total_mass_g_mol == pytest.approx(64 * 39.948, rel=1e-3)


def test_safe_timestep_is_quantised_and_derated() -> None:
    """A derated step should be a number someone can read."""
    spec = SystemSpec()
    assert safe_timestep_fs(300.0, spec) == 2.0
    assert safe_timestep_fs(600.0, spec) == 1.25
    assert safe_timestep_fs(1200.0, spec) == 1.0


def test_minimise_leaves_a_state_and_a_structure(minimised: StageResult) -> None:
    """The first thing a packed cell needs."""
    assert Path(minimised.final_state).is_file()
    assert Path(minimised.final_pdb or "").is_file()
    assert np.isfinite(minimised.samples["potential_energy_kj_mol"][0])
    assert minimised.mean_density_g_cm3 is not None


def test_minimise_refuses_a_cell_it_could_not_rescue(argon_box: Any) -> None:
    """A cell with atoms on top of each other is not worth running."""
    box, system = argon_box
    box.positions_nm = np.zeros_like(box.positions_nm)
    run = prepare_run(
        box,
        PolymerForceField("unused.xml", (), "AR", "smirnoff"),
        platform="CPU",
        seed=3,
        system=system,
    )
    with pytest.raises(SimulationError, match=r"forces|finite"):
        run_minimise(run, "00_minimise", max_iterations=1)


def test_nvt_holds_the_volume_and_reaches_the_temperature(
    argon_run: Any, minimised: StageResult
) -> None:
    """No barostat means no volume move, whatever else happens."""
    before = argon_run.box.box_nm[0] ** 3
    result = run_nvt(
        argon_run,
        "01_nvt",
        temperature_k=120.0,
        duration_ps=2.0,
        state_in=minimised.final_state,
    )
    volume = (
        argon_run.total_mass_g_mol
        * NM3_PER_CM3
        / (AVOGADRO * (result.mean_density_g_cm3 or 1.0))
    )
    assert volume == pytest.approx(before, rel=1e-6)
    assert result.mean_temperature_k == pytest.approx(120.0, abs=40.0)


def test_npt_actually_moves_the_volume(argon_run: Any, minimised: StageResult) -> None:
    """Adding a barostat to a System that already has a Context does nothing.

    This is the test for that: the stage builds its own Simulation precisely so
    the barostat is there before the Context is. Forty trial moves suffice;
    sustained compression would shrink this tiny cell below twice its cutoff.
    """
    result = run_npt(
        argon_run,
        "01_npt",
        temperature_k=120.0,
        pressure_bar=500.0,
        duration_ps=0.4,
        barostat_frequency=5,
        state_in=minimised.final_state,
    )
    assert result.mean_density_g_cm3 != pytest.approx(
        minimised.mean_density_g_cm3, rel=1e-4
    )


@pytest.mark.parametrize("barostat", [None, "isotropic"])
def test_a_temperature_change_moves_the_thermostat_not_just_the_barostat(
    argon_run: Any, barostat: str | None
) -> None:
    """The failure this guards against produces a plausible, wrong density.

    ``context.setParameter("MonteCarloTemperature", T)`` feeds only the
    barostat's Metropolis test; without the integrator being set too, every
    window integrates at the starting temperature. Without a barostat there
    is no parameter, and setting one would raise.
    """
    simulation = _probe(argon_run, barostat)
    set_temperature(simulation, 400.0, barostat)
    assert simulation.integrator.getTemperature().value_in_unit(
        unit.kelvin
    ) == pytest.approx(400.0)
    if barostat is not None:
        assert simulation.context.getParameter(
            "MonteCarloTemperature"
        ) == pytest.approx(400.0)


def test_an_npt_state_loads_into_an_nvt_stage(
    argon_run: Any, minimised: StageResult
) -> None:
    """A state saved under NPT carries the barostat's global parameters.

    Restoring it with ``loadState`` into a Context that has no barostat raises,
    which would make every NPT-to-NVT transition a failure.
    """
    npt = run_npt(
        argon_run,
        "01_npt",
        temperature_k=100.0,
        duration_ps=1.0,
        state_in=minimised.final_state,
    )
    nvt = run_nvt(
        argon_run,
        "02_nvt",
        temperature_k=100.0,
        duration_ps=1.0,
        state_in=npt.final_state,
    )
    assert nvt.steps > 0


def test_pushoff_climbs_a_ladder_of_timesteps(
    argon_run: Any, minimised: StageResult
) -> None:
    """Energy is drained rather than turned into velocity."""
    result = run_pushoff(
        argon_run,
        "01_pushoff",
        duration_ps=0.6,
        temperature_k=100.0,
        state_in=minimised.final_state,
    )
    assert result.samples["timestep_fs"] == [0.1, 0.25, 0.5]
    assert result.steps == 2000 + 800 + 400
    assert Path(result.final_state).is_file()


def test_compress_walks_the_pressure_ladder(
    argon_run: Any, minimised: StageResult
) -> None:
    """A density is recorded at every rung, so the run can be read afterwards."""
    result = run_compress(
        argon_run,
        "01_compress",
        temperature_k=100.0,
        pressures_bar=(1.0, 200.0, 1.0),
        duration_ps_each=1.0,
        barostat_frequency=5,
        state_in=minimised.final_state,
    )
    assert result.samples["segment_pressure_bar"] == [1.0, 200.0, 1.0]
    assert len(result.samples["segment_density_g_cm3"]) == 3


def test_anneal_visits_the_top_and_the_bottom_of_every_cycle(
    argon_run: Any, minimised: StageResult
) -> None:
    """What 'melt it' means: repeated excursions above and back below."""
    result = run_anneal(
        argon_run,
        "01_anneal",
        t_low=80.0,
        t_high=160.0,
        n_cycles=2,
        ramp_windows=2,
        window_ps=0.2,
        hold_ps=0.2,
        state_in=minimised.final_state,
    )
    visited = result.samples["segment_temperature_k"]
    assert visited.count(160.0) == 2 * 2
    assert visited.count(80.0) == 2 * 2
    assert result.temperature_k == pytest.approx(80.0)


def test_quench_records_a_density_and_a_hold_per_temperature(
    argon_run: Any, minimised: StageResult
) -> None:
    """The specific-volume curve a glass transition is read off.

    Each hold's length is recorded because a stage's CSV knows only its total
    time, so a ladder split across stages or resumed part-way through would
    otherwise have its cooling rate worked out wrong rather than reported as
    unknown. Waypoints are off unless asked for: one serialised state per
    temperature is megabytes for a real cell.
    """
    result = run_quench(
        argon_run,
        "01_quench",
        t_start=150.0,
        t_end=90.0,
        step_k=30.0,
        hold_ps=0.4,
        barostat_frequency=5,
        state_in=minimised.final_state,
    )
    assert result.samples["segment_temperature_k"] == [150.0, 120.0, 90.0]
    assert len(result.samples["segment_density_g_cm3"]) == 3
    assert result.samples["segment_duration_ps"] == [0.4, 0.4, 0.4]
    assert result.waypoints == ()
    assert list(Path().glob("*waypoint*")) == []


def test_a_quench_waypoint_restarts_a_stage_where_it_was_written(
    argon_run: Any, minimised: StageResult
) -> None:
    """What lets a finer second pass carry on from the middle of the first.

    Written under one barostat and loaded into another, as a scan does.
    """
    quenched = run_quench(
        argon_run,
        "01_quench",
        t_start=150.0,
        t_end=90.0,
        step_k=30.0,
        hold_ps=0.4,
        barostat_frequency=5,
        waypoints=True,
        state_in=minimised.final_state,
    )
    assert [Path(path).name for path in quenched.waypoints] == [
        "01_quench_waypoint00_150K.state.xml",
        "01_quench_waypoint01_120K.state.xml",
        "01_quench_waypoint02_90K.state.xml",
    ]
    assert all(Path(path).is_file() for path in quenched.waypoints)

    resumed = run_quench(
        argon_run,
        "02_quench",
        temperatures_k=[120.0, 110.0],
        hold_ps=0.4,
        barostat_frequency=5,
        state_in=quenched.waypoints[1],
    )
    assert resumed.samples["segment_temperature_k"] == [120.0, 110.0]


def test_production_writes_a_trajectory_and_its_topology(
    argon_run: Any, minimised: StageResult
) -> None:
    """Neither XTC nor DCD carries a topology, so one is written beside it."""
    run_production(
        argon_run,
        "01_production",
        temperature_k=100.0,
        duration_ps=1.0,
        pressure_bar=None,
        trajectory="xtc",
        report_interval_ps=0.2,
        state_in=minimised.final_state,
    )
    assert Path("01_production.xtc").is_file()
    assert Path("01_production_topology.pdb").is_file()


def test_production_holds_the_pressure_asked_for_even_zero(
    argon_run: Any, minimised: StageResult
) -> None:
    """``pressure_bar or 1.0`` quietly ran a requested 0 bar at 1 bar."""
    result = run_production(
        argon_run,
        "01_production",
        temperature_k=100.0,
        duration_ps=0.2,
        pressure_bar=0.0,
        trajectory="none",
        state_in=minimised.final_state,
    )
    saved = mm.XmlSerializer.deserialize(Path(result.final_state).read_text())
    assert saved.getParameters()["MonteCarloPressure"] == 0.0


def test_run_segments_needs_something_to_run(argon_run: Any) -> None:
    """An empty stage is a mistake, not a no-op."""
    with pytest.raises(ValueError, match="no segments"):
        run_segments(argon_run, "empty", [], "01_empty")


def test_run_segments_checks_the_timestep_against_the_hottest_segment(
    argon_run: Any,
) -> None:
    """A stage that starts cold and anneals hot is integrated for the hot part."""
    with pytest.raises(ValueError, match="too long for 900 K"):
        run_segments(
            argon_run,
            "ramp",
            [Segment(300.0, 0.2), Segment(900.0, 0.2)],
            "01_ramp",
            timestep_fs=2.0,
        )


@pytest.mark.parametrize(
    "runner", [run_nvt, run_deform, run_load, run_shear, run_relax]
)
@pytest.mark.parametrize("timestep_fs", [0.0, -1.0, float("nan")])
def test_every_stage_refuses_a_timestep_that_is_not_positive(
    argon_run: Any, runner: Callable[..., StageResult], timestep_fs: float
) -> None:
    """Checked before a Context is built, by every stage and not only some."""
    with pytest.raises(ValueError, match="timestep_fs"):
        runner(argon_run, timestep_fs=timestep_fs)


def test_the_numeric_csv_is_all_numbers_and_the_log_has_the_rest(
    argon_run: Any, minimised: StageResult
) -> None:
    """Progress renders as '20.0%' and an unknown remaining time as '--'.

    Either in the data would stop the file being a table of numbers, so they
    go in the human log instead.
    """
    result = run_nvt(
        argon_run,
        "01_nvt",
        temperature_k=100.0,
        duration_ps=1.0,
        report_interval_ps=0.2,
        state_in=minimised.final_state,
    )
    table = np.genfromtxt(result.csv or "", delimiter=",", names=True)
    assert table.size > 0
    assert not np.isnan(np.asarray(table.tolist(), dtype=float)).any()
    assert "Density_gmL" in (table.dtype.names or ())
    header = Path("01_nvt.log").read_text().splitlines()[0]
    assert "Progress" in header
    assert "Speed" in header


def test_a_run_is_reproducible_from_its_seed(argon_box: Any) -> None:
    """Everything stochastic derives from one number."""

    def temperatures(seed: int) -> list[float]:
        box, system = argon_box
        run = prepare_run(
            box,
            PolymerForceField("unused.xml", (), "AR", "smirnoff"),
            platform="Reference",
            seed=seed,
            system=system,
        )
        minimised = run_minimise(run, f"min{seed}")
        result = run_nvt(
            run,
            f"nvt{seed}",
            temperature_k=100.0,
            duration_ps=0.2,
            state_in=minimised.final_state,
        )
        return result.samples["segment_mean_temperature_k"]

    assert temperatures(5) == temperatures(5)


def test_density_and_temperature_helpers_agree_with_openmm(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both are computed here rather than read off a reporter."""
    simulation = _probe(argon_run)
    expected = argon_run.total_mass_g_mol * NM3_PER_CM3 / (AVOGADRO * 2.4**3)
    assert _density_g_cm3(simulation, argon_run.total_mass_g_mol) == pytest.approx(
        expected
    )
    temperature = temperature_k_of(simulation)
    assert temperature == pytest.approx(120.0, rel=0.35)

    state = simulation.context.getState(getEnergy=True)

    def unexpected_fetch(**kwargs: Any) -> Any:
        pytest.fail("An existing state must not trigger another state fetch.")

    monkeypatch.setattr(simulation.context, "getState", unexpected_fetch)
    assert _density_g_cm3(
        simulation, argon_run.total_mass_g_mol, state=state
    ) == pytest.approx(expected)
    assert temperature_k_of(simulation, state=state) == pytest.approx(temperature)


# --------------------------------------------------------------------------
# The ladder, its waypoints, and what a segment records
# --------------------------------------------------------------------------


def test_the_ladder_helper_reproduces_the_one_a_quench_runs() -> None:
    """The cost estimate, the chunker and the stage all count from this."""
    ladder = quench_temperatures(650.0, 150.0, 25.0)

    assert len(ladder) == 21
    assert ladder[0] == 650.0
    assert ladder[-1] == 150.0
    assert ladder == sorted(ladder, reverse=True)


def test_a_ladder_that_does_not_divide_evenly_still_reaches_the_bottom() -> None:
    """The floor is a temperature someone chose, not a rounding artefact."""
    assert quench_temperatures(100.0, 30.0, 90.0) == [100.0, 30.0]


@pytest.mark.parametrize(
    ("start", "end", "step", "match"),
    [
        (100.0, 100.0, 20.0, "a quench cools"),
        (100.0, 200.0, 20.0, "a quench cools"),
        (float("inf"), 100.0, 20.0, "finite"),
        (float("nan"), 100.0, 20.0, "finite"),
        (200.0, float("nan"), 20.0, "finite"),
        (200.0, float("-inf"), 20.0, "finite"),
        (200.0, 100.0, float("nan"), "finite"),
        (200.0, 100.0, float("inf"), "finite"),
        (200.0, 100.0, 0.0, "greater than zero"),
        (200.0, 100.0, -1.0, "greater than zero"),
        (200.0, 100.0, 1e-30, "too small"),
        (-2.0 + 2.0**-52, -3.0, 2.0**-52, "too small"),
    ],
)
def test_quench_ladder_rejects_invalid_arguments(
    start: float, end: float, step: float, match: str
) -> None:
    """Invalid endpoints and steps must fail instead of hanging a quench."""
    with pytest.raises(ValueError, match=match):
        quench_temperatures(start, end, step)


def test_quench_ladder_accepts_offsets_for_nominal_schedule_costs() -> None:
    """A nominal fine schedule counts relative rungs before its window is known."""
    assert quench_temperatures(10.0, -10.0, 5.0) == [10.0, 5.0, 0.0, -5.0, -10.0]


@pytest.mark.parametrize("endpoint", ["t_start", "t_end"])
@pytest.mark.parametrize("value", [0.0, -100.0, -1e100])
def test_quench_rejects_nonpositive_physical_temperatures(
    argon_run: Any, endpoint: str, value: float
) -> None:
    """Offset ladders are useful for cost estimates, but cannot run dynamics."""
    options: dict[str, Any] = {endpoint: value}
    with pytest.raises(ValueError):
        run_quench(argon_run, **options)


@pytest.mark.parametrize(
    ("temperatures", "match"),
    [
        ([], "nothing to hold"),
        ([100.0, 120.0], "has to descend"),
        ([100.0, 100.0], "has to descend"),
        ([float("nan")], "finite"),
        ([float("inf")], "finite"),
        ([0.0], "greater than zero"),
        ([-1.0], "greater than zero"),
        ([120.0, float("nan")], "finite"),
        ([float("inf"), 120.0], "finite"),
    ],
)
def test_quench_rejects_an_invalid_explicit_ladder(
    argon_run: Any, temperatures: list[float], match: str
) -> None:
    """A one-window chunk needs the same valid temperatures as a full ladder."""
    with pytest.raises(ValueError, match=match):
        run_quench(argon_run, temperatures_k=temperatures)


def test_heating_ladder_includes_both_endpoints() -> None:
    """The cost and stage must agree even when the final interval is short."""
    assert heating_temperatures(100.0, 175.0, 30.0) == [100.0, 130.0, 160.0, 175.0]
    assert heating_temperatures(100.0, 110.0, 30.0) == [100.0, 110.0]
    assert heating_temperatures(100.0, 160.0, 30.0) == [100.0, 130.0, 160.0]


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [
        (100.0, 100.0, 20.0),
        (200.0, 100.0, 20.0),
        (0.0, 100.0, 20.0),
        (100.0, float("inf"), 20.0),
        (float("nan"), 200.0, 20.0),
        (100.0, 200.0, float("nan")),
        (100.0, 200.0, 0.0),
        (100.0, 200.0, 1e-30),
    ],
)
def test_heating_ladder_rejects_invalid_arguments(
    start: float, end: float, step: float
) -> None:
    """Nonfinite or nonprogressing input must fail before entering dynamics."""
    with pytest.raises(ValueError):
        heating_temperatures(start, end, step)


@pytest.mark.parametrize(
    "temperatures",
    [[], [120.0, 100.0], [100.0, 100.0], [100.0, float("nan")], [0.0, 100.0]],
)
def test_heat_rejects_an_invalid_explicit_ladder(
    argon_run: Any, temperatures: list[float]
) -> None:
    """An explicit chunk has the same finite, positive, ascending rules."""
    with pytest.raises(ValueError):
        run_heat(argon_run, temperatures_k=temperatures)


@pytest.mark.parametrize("option", ["hold_ps", "pressure_bar"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_heat_rejects_invalid_holds_and_pressures(
    argon_run: Any, option: str, value: float
) -> None:
    """Fail before a context or any output is created."""
    options: dict[str, Any] = {option: value}
    with pytest.raises(ValueError, match=option):
        run_heat(argon_run, **options)


def test_heat_uses_the_hot_endpoint_to_validate_timestep(argon_run: Any) -> None:
    """A cool start does not make a large step safe at the hot end."""
    with pytest.raises(ValueError, match="too long for 900 K"):
        run_heat(argon_run, t_start=100.0, t_end=900.0, timestep_fs=2.0)


def test_heat_records_density_enthalpy_and_anisotropic_pressure(
    argon_run: Any, minimised: StageResult
) -> None:
    """The CPU integration exercises the complete heating and state path."""
    result = run_heat(
        argon_run,
        "01_heat",
        temperatures_k=[90.0, 120.0, 150.0],
        hold_ps=0.4,
        pressure_bar=2.0,
        barostat_frequency=5,
        waypoints=True,
        state_in=minimised.final_state,
    )
    assert result.samples["segment_temperature_k"] == [90.0, 120.0, 150.0]
    assert result.samples["segment_pressure_bar"] == [2.0, 2.0, 2.0]
    assert result.samples["segment_duration_ps"] == [0.4, 0.4, 0.4]
    assert len(result.samples["segment_enthalpy_kj_mol"]) == 3
    assert np.isfinite(result.samples["segment_enthalpy_kj_mol"]).all()
    assert np.all(np.asarray(result.samples["segment_density_g_cm3"]) > 0.0)
    assert len(result.waypoints) == 3
    saved = mm.XmlSerializer.deserialize(Path(result.final_state).read_text())
    assert saved.getParameters()["MonteCarloPressureX"] == pytest.approx(2.0)
    assert saved.getParameters()["MonteCarloPressureY"] == pytest.approx(2.0)
    assert saved.getParameters()["MonteCarloPressureZ"] == pytest.approx(2.0)
    resumed = run_heat(
        argon_run,
        "02_heat",
        temperatures_k=[180.0],
        hold_ps=0.1,
        state_in=result.final_state,
    )
    assert resumed.samples["segment_temperature_k"] == [180.0]
    assert len(resumed.samples["segment_enthalpy_kj_mol"]) == 1


@pytest.mark.parametrize("remove_cm", [False, True])
@pytest.mark.parametrize("measure_enthalpy", [False, True])
def test_a_hold_keeps_every_observable_from_the_same_second_half(
    monkeypatch: pytest.MonkeyPatch, remove_cm: bool, measure_enthalpy: bool
) -> None:
    """The transient is discarded for every observable at the same times.

    One state supplies each reading, and the degrees of freedom are counted
    once across holds. Enthalpy includes the kinetic energy and the pV work.
    """
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addConstraint(0, 1, 0.1)
    if remove_cm:
        system.addForce(mm.CMMotionRemover())
    simulation = SimpleNamespace(index=-1, system=system)
    state_calls = 0
    force_calls = 0
    get_forces = system.getForces

    def counted_forces() -> Any:
        nonlocal force_calls
        force_calls += 1
        return get_forces()

    monkeypatch.setattr(system, "getForces", counted_forces)

    def step(steps: int) -> None:
        assert steps == 1
        simulation.index += 1

    def state(**kwargs: Any) -> Any:
        nonlocal state_calls
        state_calls += 1
        assert kwargs == {"getEnergy": True}
        index = simulation.index % 4
        return SimpleNamespace(
            getPotentialEnergy=lambda: (
                [1000, 500, 20, 40][index] * unit.kilojoule_per_mole
            ),
            getKineticEnergy=lambda: [1000, 500, 3, 4][index] * unit.kilojoule_per_mole,
            getPeriodicBoxVolume=lambda: [1000, 500, 5, 7][index] * unit.nanometer**3,
        )

    simulation.step = step
    simulation.context = SimpleNamespace(getState=state)
    run: Any = SimpleNamespace(total_mass_g_mol=1.0)
    live = _Live(run, Path("fake"), simulation, 1.0, 0.0, "fake")
    samples: dict[str, list[float]] = {
        "segment_temperature_k": [],
        "segment_mean_temperature_k": [],
        "segment_density_g_cm3": [],
        "segment_duration_ps": [],
    }
    expected_density = NM3_PER_CM3 / AVOGADRO * (1 / 5 + 1 / 7) / 2
    degrees = 2 if remove_cm else 5
    gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    )
    expected_temperature = 2 * 3.5 / (degrees * gas_constant)
    for _ in range(2):
        enthalpies, density, temperature = live.hold(
            samples,
            300.0,
            0.004,
            4,
            "blew up",
            partial(_enthalpy_kj_mol, pressure_bar=2.0) if measure_enthalpy else None,
        )
        assert density == pytest.approx(expected_density)
        assert temperature == pytest.approx(expected_temperature)
        if measure_enthalpy:
            assert np.mean(enthalpies) == pytest.approx(33.5 + 12.0 * 0.0602214076)
        else:
            assert enthalpies == []
    assert state_calls == 8
    assert force_calls == 1
    assert samples["segment_temperature_k"] == [300.0, 300.0]
    assert samples["segment_duration_ps"] == [0.004, 0.004]
    assert samples["segment_density_g_cm3"] == pytest.approx([expected_density] * 2)
    assert samples["segment_mean_temperature_k"] == pytest.approx(
        [expected_temperature] * 2
    )


def test_a_quench_can_be_given_its_temperatures_outright(
    argon_run: Any, minimised: StageResult
) -> None:
    """A chunked pass hands each stage a slice of one ladder, not endpoints.

    Deriving each chunk's endpoints from the grid is exactly the off-by-one
    that repeats or skips a temperature at every boundary.
    """
    result = run_quench(
        argon_run,
        "01_quench",
        temperatures_k=[140.0, 125.0, 110.0],
        hold_ps=0.4,
        barostat_frequency=5,
        state_in=minimised.final_state,
    )
    assert result.samples["segment_temperature_k"] == [140.0, 125.0, 110.0]


def test_the_readings_behind_a_segment_average_can_be_raised(
    argon_run: Any, minimised: StageResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ten readings, halved, leaves five behind each point on a curve.

    Thin for a segment whose whole purpose is a low-noise point, so a fine
    pass asks for more.
    """
    calls = 0
    real = simulate._density_g_cm3

    def counted(
        simulation: Any, total_mass_g_mol: float, *, state: Any | None = None
    ) -> float:
        nonlocal calls
        calls += 1
        return real(simulation, total_mass_g_mol, state=state)

    monkeypatch.setattr(simulate, "_density_g_cm3", counted)
    run_nvt(
        argon_run,
        "01_nvt",
        temperature_k=120.0,
        duration_ps=1.0,
        samples_per_segment=25,
        state_in=minimised.final_state,
    )
    # A step count that does not divide ends on a shorter chunk, which is one
    # more reading than was asked for.
    assert calls == pytest.approx(25, abs=1)


def test_a_sample_count_below_one_is_refused(argon_run: Any) -> None:
    """Zero readings is a segment that measures nothing."""
    with pytest.raises(ValueError, match="samples_per_segment"):
        run_nvt(argon_run, "01_nvt", duration_ps=0.2, samples_per_segment=0)


# --------------------------------------------------------------------------
# Deformation plumbing
# --------------------------------------------------------------------------


def test_a_stage_can_be_told_to_draw_fresh_velocities(argon_run: Any) -> None:
    """Which is what makes two runs from one configuration independent.

    Without it a replica inherits the saved state's velocities along with
    its positions, repeats the same trajectory, and the spread over the
    replicas is zero pretending to be an error bar.
    """
    settled = run_npt(argon_run, "npt", temperature_k=120.0, duration_ps=0.4)

    def velocities(new: bool) -> Any:
        simulation = _probe(argon_run, state_in=settled.final_state, new_velocities=new)
        return (
            simulation.context.getState(getVelocities=True)
            .getVelocities(asNumpy=True)
            .value_in_unit(unit.nanometer / unit.picosecond)
        )

    assert not np.allclose(velocities(False), velocities(True))


def test_a_state_saved_without_velocities_is_given_fresh_ones(
    argon_run: Any, minimised: StageResult
) -> None:
    """A state from elsewhere need not carry any, and is not refused for it."""
    context = _probe(argon_run, state_in=minimised.final_state).context
    state = context.getState(getPositions=True)
    Path("positions.xml").write_text(mm.XmlSerializer.serialize(state))
    simulation = _probe(argon_run, state_in="positions.xml")
    assert temperature_k_of(simulation) == pytest.approx(120.0, rel=0.35)


def test_pressures_can_be_set_per_axis(argon_run: Any) -> None:
    """A uniaxial load is one axis held somewhere the other two are not."""
    simulation = _probe(argon_run, "anisotropic")
    set_pressures(simulation, (1.0, 1.0, -20.0), "anisotropic")
    assert simulation.context.getParameter("MonteCarloPressureZ") == pytest.approx(
        -20.0
    )
    assert simulation.context.getParameter("MonteCarloPressureX") == pytest.approx(1.0)


def test_three_different_pressures_are_refused_by_a_barostat_that_holds_one(
    argon_run: Any,
) -> None:
    """Applying the first of three to all of them would not be the run asked for."""
    simulation = _probe(argon_run, "isotropic")
    with pytest.raises(ValueError, match="one pressure"):
        set_pressures(simulation, (1.0, 1.0, -20.0), "isotropic")
    set_pressures(simulation, (5.0, 5.0, 5.0), "isotropic")
    assert simulation.context.getParameter("MonteCarloPressure") == pytest.approx(5.0)


def test_a_cell_that_has_shrunk_below_the_cutoff_is_refused(
    argon_run: Any,
) -> None:
    """Stretching one axis contracts the other two, and OpenMM has a floor.

    Checked every step, because OpenMM's own complaint arrives at the next
    force evaluation rather than at the line that caused it - several steps
    away from the strain that produced it, and about a box nobody set.

    Tested on the guard rather than through a run: making a barostat
    actually contract a cell past its cutoff takes far more dynamics than a
    fast test can spend, and what is being checked here is the arithmetic.
    """
    simulation = _probe(argon_run)
    cutoff = nonbonded_cutoff_nm(simulation.system)
    assert cutoff > 0.0
    assert nonbonded_cutoff_nm(mm.System()) == 0.0

    # The fixture's cell is comfortably above twice the cutoff.
    _check_deformed_box("probe", simulation, cutoff, 0.0)
    # Pretending the cutoff is most of the box is the same arithmetic.
    with pytest.raises(SimulationError, match="cutoff"):
        _check_deformed_box("probe", simulation, 10.0, 0.05)
    # Zero means "nothing here has one", which is not a box of zero size.
    _check_deformed_box("probe", simulation, 0.0, 0.0)


def test_a_cell_openmm_groups_differently_from_the_packing_is_reported(
    argon_run: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A merged pair of chains is a wrong molecular virial, silently."""
    simulation = _probe(argon_run)
    with caplog.at_level(logging.WARNING, logger="openmmpolymer.simulate"):
        _check_molecules("probe", simulation, 64)
        assert not caplog.records
        _check_molecules("probe", simulation, 63)
    assert "into 64 molecules, but 63 chains" in caplog.text


def test_a_strain_leaves_every_constraint_satisfied() -> None:
    """Scaling atoms stretches the bonds they share, and the repair undoes it.

    Positions go back onto the constraint, and velocities lose the component
    along it that the constraint does not allow - drawn at random here, so
    they start with one.
    """
    simulation = bare_simulation(*rigid_rotor_system(30, 3.0))
    drawn = np.random.default_rng(3).normal(0.0, 0.5, (60, 3))
    simulation.context.setVelocities(drawn * unit.nanometer / unit.picosecond)
    factors = (1.0, 1.0, 1.05)
    vectors = simulation.context.getState().getPeriodicBoxVectors()
    _apply_positions(
        simulation,
        affine_scale(_positions_nm(simulation), factors),
        [vector * factor for vector, factor in zip(vectors, factors, strict=True)],
    )
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    velocities = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    bonds = positions[1::2] - positions[0::2]
    along = np.einsum("ij,ij->i", velocities[1::2] - velocities[0::2], bonds)
    assert np.linalg.norm(bonds, axis=1) == pytest.approx(0.109, rel=1e-6)
    assert along == pytest.approx(0.0, abs=1e-6)


def test_a_shear_past_the_reduced_form_is_refused_before_the_ladder_runs(
    argon_run: Any, minimised: StageResult
) -> None:
    """Checked against every rung up front, not discovered on the last one."""
    with pytest.raises((SimulationError, StressError), match="reduced form"):
        run_shear(
            argon_run,
            "shear",
            temperature_k=120.0,
            strains=(0.01, 0.9),
            duration_ps_each=0.1,
            samples_per_step=2,
            state_in=minimised.final_state,
        )


def test_a_shear_records_the_corrected_stress_estimator(
    argon_run: Any, minimised: StageResult
) -> None:
    """Analysis must distinguish new physical stresses from old box derivatives."""
    result = run_shear(
        argon_run,
        "shear",
        temperature_k=120.0,
        strains=(0.01, 0.02),
        duration_ps_each=0.1,
        samples_per_step=2,
        state_in=minimised.final_state,
    )
    assert result.samples["stress_estimator_version"] == [STRESS_ESTIMATOR_VERSION]
    assert np.isfinite(result.samples["segment_shear_stress_bar"]).all()


def test_a_deformation_records_a_reference_cell_and_an_axis(
    argon_run: Any, minimised: StageResult
) -> None:
    """Recorded rather than recomputed, so a resumed chunk shares the origin."""
    result = run_deform(
        argon_run,
        "deform",
        temperature_k=120.0,
        n_steps=2,
        strain_increment=0.002,
        relax_ps=0.1,
        samples_per_step=2,
        axis=1,
        state_in=minimised.final_state,
    )
    assert len(result.samples["reference_box_nm"]) == 3
    assert result.samples["deform_axis"] == [1.0]
    assert len(result.samples["segment_strain"]) == 2


def test_a_deformation_resumed_mid_ladder_keeps_the_original_origin(
    argon_run: Any, minimised: StageResult
) -> None:
    """The strain a second chunk reports is measured from the unstrained cell."""
    first = run_deform(
        argon_run,
        "deform_a",
        temperature_k=120.0,
        n_steps=2,
        strain_increment=0.004,
        relax_ps=0.1,
        samples_per_step=2,
        state_in=minimised.final_state,
    )
    origin = first.samples["reference_box_nm"]
    second = run_deform(
        argon_run,
        "deform_b",
        temperature_k=120.0,
        n_steps=2,
        strain_increment=0.004,
        relax_ps=0.1,
        samples_per_step=2,
        strain_start=first.samples["segment_strain"][-1],
        reference_box_nm=origin,
        state_in=first.final_state,
    )
    assert second.samples["reference_box_nm"] == origin
    assert second.samples["segment_strain"][0] > first.samples["segment_strain"][-1]
    # And the recorded strain still describes the cell it was measured in.
    assert second.samples["segment_box_z_nm"][-1] / origin[2] - 1.0 == pytest.approx(
        second.samples["segment_strain"][-1]
    )


@pytest.mark.parametrize("option", ["baseline_ps", "ramp_ps", "time_offset_ps"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_relax_refuses_a_time_that_is_negative_or_not_finite(
    argon_run: Any, option: str, value: float
) -> None:
    """``< 0.0`` let NaN through, and an endless baseline is not a baseline."""
    options: dict[str, Any] = {option: value}
    with pytest.raises(ValueError, match="finite and zero or more"):
        run_relax(argon_run, **options)
