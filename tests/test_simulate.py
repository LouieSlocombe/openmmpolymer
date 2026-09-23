"""Tests for the stages, run against a real but tiny argon cell.

No force-field file appears here. The cell is a real periodic
``NonbondedForce``, so the barostat has something to do and a density is a real
number, and everything the stage layer can get wrong is reachable in
milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmmpolymer.mdsystem import SystemSpec, make_barostat
from openmmpolymer.simulate import (
    Segment,
    SimulationError,
    StageResult,
    density_g_cm3,
    heating_temperatures,
    prepare_run,
    quench_temperatures,
    run_anneal,
    run_compress,
    run_deform,
    run_heat,
    run_minimise,
    run_npt,
    run_nvt,
    run_production,
    run_pushoff,
    run_quench,
    run_segments,
    run_shear,
    safe_timestep_fs,
    set_temperature,
    temperature_k_of,
)


def test_prepare_run_measures_the_cell_mass(argon_run: Any) -> None:
    """64 argon atoms, and the density arithmetic depends on it."""
    assert argon_run.total_mass_g_mol == pytest.approx(64 * 39.948, rel=1e-3)


def test_safe_timestep_is_quantised_and_derated() -> None:
    """A derated step should be a number someone can read."""
    spec = SystemSpec()
    assert safe_timestep_fs(300.0, spec) == 2.0
    assert safe_timestep_fs(600.0, spec) == 1.25
    assert safe_timestep_fs(1200.0, spec) == 1.0


def test_minimise_lowers_the_energy_and_leaves_a_state(argon_run: Any) -> None:
    """The first thing a packed cell needs."""
    result = run_minimise(argon_run, "00_minimise")
    assert isinstance(result, StageResult)
    assert Path(result.final_state).is_file()
    assert Path(result.final_pdb or "").is_file()
    assert np.isfinite(result.samples["potential_energy_kj_mol"][0])
    assert result.mean_density_g_cm3 is not None


def test_minimise_refuses_a_cell_it_could_not_rescue(
    argon_box: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell with atoms on top of each other is not worth running."""
    from openmmpolymer.forcefield import PolymerForceField

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


def test_nvt_holds_the_volume_and_reaches_the_temperature(argon_run: Any) -> None:
    """No barostat means no volume move, whatever else happens."""
    minimised = run_minimise(argon_run, "00_minimise")
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
        * 1.0e21
        / (6.02214076e23 * (result.mean_density_g_cm3 or 1.0))
    )
    assert volume == pytest.approx(before, rel=1e-6)
    assert result.mean_temperature_k == pytest.approx(120.0, abs=40.0)


def test_npt_actually_moves_the_volume(argon_run: Any) -> None:
    """Adding a barostat to a System that already has a Context does nothing.

    This is the test for that: the stage builds its own Simulation precisely so
    the barostat is there before the Context is.
    """
    minimised = run_minimise(argon_run, "00_minimise")
    result = run_npt(
        argon_run,
        "01_npt",
        temperature_k=120.0,
        pressure_bar=500.0,
        duration_ps=4.0,
        barostat_frequency=5,
        state_in=minimised.final_state,
    )
    assert result.mean_density_g_cm3 != pytest.approx(
        minimised.mean_density_g_cm3, rel=1e-4
    )


def test_a_temperature_ramp_moves_the_thermostat_not_just_the_barostat(
    argon_run: Any,
) -> None:
    """The failure this guards against produces a plausible, wrong density.

    ``context.setParameter("MonteCarloTemperature", T)`` feeds only the
    barostat's Metropolis test. Without the integrator being set too, every
    window integrates at the starting temperature.
    """
    import openmm as mm
    from openmm import app, unit

    system = mm.XmlSerializer.deserialize(argon_run.system_xml)
    system.addForce(make_barostat("isotropic", 100.0, 1.0, 25, 9))
    integrator = mm.LangevinMiddleIntegrator(
        100.0 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds
    )
    simulation = app.Simulation(
        argon_run.box.topology,
        system,
        integrator,
        mm.Platform.getPlatformByName("CPU"),
    )
    simulation.context.setPositions(argon_run.box.positions)

    set_temperature(simulation, 400.0, "isotropic")
    assert simulation.integrator.getTemperature().value_in_unit(
        unit.kelvin
    ) == pytest.approx(400.0)
    assert simulation.context.getParameter("MonteCarloTemperature") == pytest.approx(
        400.0
    )


def test_set_temperature_without_a_barostat_touches_only_the_integrator(
    argon_run: Any,
) -> None:
    """Setting the barostat parameter on a Context with none raises."""
    import openmm as mm
    from openmm import app, unit

    system = mm.XmlSerializer.deserialize(argon_run.system_xml)
    integrator = mm.LangevinMiddleIntegrator(
        100.0 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds
    )
    simulation = app.Simulation(
        argon_run.box.topology,
        system,
        integrator,
        mm.Platform.getPlatformByName("CPU"),
    )
    simulation.context.setPositions(argon_run.box.positions)
    set_temperature(simulation, 250.0, None)
    assert simulation.integrator.getTemperature().value_in_unit(
        unit.kelvin
    ) == pytest.approx(250.0)


def test_an_npt_state_loads_into_an_nvt_stage(argon_run: Any) -> None:
    """A state saved under NPT carries the barostat's global parameters.

    Restoring it with ``loadState`` into a Context that has no barostat raises,
    which would make every NPT-to-NVT transition a failure.
    """
    minimised = run_minimise(argon_run, "00_minimise")
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


def test_pushoff_climbs_a_ladder_of_timesteps(argon_run: Any) -> None:
    """Energy is drained rather than turned into velocity."""
    minimised = run_minimise(argon_run, "00_minimise")
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


def test_compress_walks_the_pressure_ladder(argon_run: Any) -> None:
    """A density is recorded at every rung, so the run can be read afterwards."""
    minimised = run_minimise(argon_run, "00_minimise")
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
    argon_run: Any,
) -> None:
    """What 'melt it' means: repeated excursions above and back below."""
    minimised = run_minimise(argon_run, "00_minimise")
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


def test_quench_descends_and_records_a_density_per_temperature(
    argon_run: Any,
) -> None:
    """The specific-volume curve a glass transition is read off."""
    minimised = run_minimise(argon_run, "00_minimise")
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
    temperatures = result.samples["segment_temperature_k"]
    assert temperatures == [150.0, 120.0, 90.0]
    assert len(result.samples["segment_density_g_cm3"]) == 3
    assert temperatures == sorted(temperatures, reverse=True)


def test_quench_refuses_to_go_upwards(argon_run: Any) -> None:
    """A quench cools; the other direction is an anneal."""
    with pytest.raises(ValueError, match="a quench cools"):
        run_quench(argon_run, "01_quench", t_start=100.0, t_end=200.0)


def test_production_writes_a_trajectory_and_its_topology(argon_run: Any) -> None:
    """Neither XTC nor DCD carries a topology, so one is written beside it."""
    minimised = run_minimise(argon_run, "00_minimise")
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


def test_the_numeric_csv_is_all_numbers(argon_run: Any) -> None:
    """Progress renders as '20.0%' and an unknown remaining time as '--'.

    Either in the data would stop the file being a table of numbers, so they
    go in the human log instead.
    """
    minimised = run_minimise(argon_run, "00_minimise")
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


def test_the_human_log_carries_the_progress_columns(argon_run: Any) -> None:
    """The other half of the split."""
    minimised = run_minimise(argon_run, "00_minimise")
    run_nvt(
        argon_run,
        "01_nvt",
        temperature_k=100.0,
        duration_ps=1.0,
        report_interval_ps=0.2,
        state_in=minimised.final_state,
    )
    header = Path("01_nvt.log").read_text().splitlines()[0]
    assert "Progress" in header
    assert "Speed" in header


def test_a_run_is_reproducible_from_its_seed(argon_box: Any) -> None:
    """Everything stochastic derives from one number."""
    from openmmpolymer.forcefield import PolymerForceField

    def densities(seed: int) -> list[float]:
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

    assert densities(5) == densities(5)


def test_density_and_temperature_helpers_agree_with_openmm(argon_run: Any) -> None:
    """Both are computed here rather than read off a reporter."""
    import openmm as mm
    from openmm import app, unit

    system = mm.XmlSerializer.deserialize(argon_run.system_xml)
    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtoseconds
    )
    simulation = app.Simulation(
        argon_run.box.topology,
        system,
        integrator,
        mm.Platform.getPlatformByName("CPU"),
    )
    simulation.context.setPositions(argon_run.box.positions)
    simulation.context.setVelocitiesToTemperature(300.0, 1)

    expected = argon_run.total_mass_g_mol * 1.0e21 / (6.02214076e23 * 2.4**3)
    assert density_g_cm3(simulation, argon_run.total_mass_g_mol) == pytest.approx(
        expected
    )
    assert temperature_k_of(simulation) == pytest.approx(300.0, rel=0.35)


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


def test_heat_records_density_enthalpy_and_anisotropic_pressure(argon_run: Any) -> None:
    """The CPU integration exercises the complete heating and state path."""
    import openmm as mm

    minimised = run_minimise(argon_run, "00_minimise")
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


def test_enthalpy_includes_kinetic_energy_and_pv_over_retained_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transient is discarded for every observable at the same times."""
    from openmm import unit

    import openmmpolymer.simulate as simulate

    simulation = SimpleNamespace(index=-1)

    def step(steps: int) -> None:
        assert steps == 1
        simulation.index += 1

    def state(**kwargs: Any) -> Any:
        index = simulation.index
        return SimpleNamespace(
            getPotentialEnergy=lambda: (
                [1000, 500, 20, 40][index] * unit.kilojoule_per_mole
            ),
            getKineticEnergy=lambda: [1000, 500, 3, 4][index] * unit.kilojoule_per_mole,
            getPeriodicBoxVolume=lambda: [1000, 500, 5, 7][index] * unit.nanometer**3,
        )

    simulation.step = step
    simulation.context = SimpleNamespace(getState=state)
    monkeypatch.setattr(
        simulate, "density_g_cm3", lambda sim, mass: float(sim.index + 1)
    )
    monkeypatch.setattr(
        simulate, "temperature_k_of", lambda sim: 100.0 * (sim.index + 1)
    )
    density, temperature, enthalpy = simulate._sample_segment(
        simulation, 4, 1.0, 4, pressure_bar=2.0
    )
    assert density == pytest.approx(3.5)
    assert temperature == pytest.approx(350.0)
    assert enthalpy == pytest.approx(33.5 + 12.0 * 0.0602214076)


def test_a_quench_can_be_given_its_temperatures_outright(argon_run: Any) -> None:
    """A chunked pass hands each stage a slice of one ladder, not endpoints.

    Deriving each chunk's endpoints from the grid is exactly the off-by-one
    that repeats or skips a temperature at every boundary.
    """
    minimised = run_minimise(argon_run, "00_minimise")
    result = run_quench(
        argon_run,
        "01_quench",
        temperatures_k=[140.0, 125.0, 110.0],
        hold_ps=0.4,
        barostat_frequency=5,
        state_in=minimised.final_state,
    )
    assert result.samples["segment_temperature_k"] == [140.0, 125.0, 110.0]


def test_a_ladder_that_does_not_descend_is_refused(argon_run: Any) -> None:
    """Handed a list, the guard still has to be there."""
    with pytest.raises(ValueError, match="has to descend"):
        run_quench(argon_run, "01_quench", temperatures_k=[100.0, 120.0])


def test_an_empty_ladder_is_refused(argon_run: Any) -> None:
    """There is nothing to hold, and it says what to give instead."""
    with pytest.raises(ValueError, match="nothing to hold"):
        run_quench(argon_run, "01_quench", temperatures_k=[])


def test_a_stage_records_how_long_each_segment_was_held(argon_run: Any) -> None:
    """A stage's CSV knows only its total time.

    So a ladder split across stages, or resumed part-way through, would have
    its cooling rate worked out wrong rather than reported as unknown.
    """
    minimised = run_minimise(argon_run, "00_minimise")
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
    assert result.samples["segment_duration_ps"] == [0.4, 0.4, 0.4]


def test_a_quench_can_leave_a_waypoint_at_every_temperature(
    argon_run: Any,
) -> None:
    """What lets a finer second pass carry on from the middle of the first."""
    minimised = run_minimise(argon_run, "00_minimise")
    result = run_quench(
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
    assert len(result.waypoints) == len(result.samples["segment_temperature_k"])
    assert [Path(str(path)).name for path in result.waypoints] == [
        "01_quench_waypoint00_150K.state.xml",
        "01_quench_waypoint01_120K.state.xml",
        "01_quench_waypoint02_90K.state.xml",
    ]
    assert all(Path(str(path)).is_file() for path in result.waypoints)


def test_waypoints_are_off_unless_they_are_asked_for(argon_run: Any) -> None:
    """One serialised state per temperature is megabytes for a real cell."""
    minimised = run_minimise(argon_run, "00_minimise")
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
    assert result.waypoints == ()
    assert list(Path().glob("*waypoint*")) == []


def test_a_waypoint_restarts_a_stage_where_it_was_written(argon_run: Any) -> None:
    """Written under a barostat and loaded into another one, as a scan does."""
    import openmm as mm

    minimised = run_minimise(argon_run, "00_minimise")
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
    waypoint = str(quenched.waypoints[1])
    saved = mm.XmlSerializer.deserialize(Path(waypoint).read_text())

    resumed = run_quench(
        argon_run,
        "02_quench",
        temperatures_k=[120.0, 110.0],
        hold_ps=0.4,
        barostat_frequency=5,
        state_in=waypoint,
    )
    started = mm.XmlSerializer.deserialize(Path("02_quench.state.xml").read_text())

    assert resumed.samples["segment_temperature_k"] == [120.0, 110.0]
    assert saved.getPeriodicBoxVectors() is not None
    assert started.getPeriodicBoxVectors() is not None


def test_the_readings_behind_a_segment_average_can_be_raised(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ten readings, halved, leaves five behind each point on a curve.

    Thin for a segment whose whole purpose is a low-noise point, so a fine
    pass asks for more.
    """
    import openmmpolymer.simulate as simulate

    calls = 0
    real = simulate.density_g_cm3

    def counted(simulation: Any, total_mass_g_mol: float) -> float:
        nonlocal calls
        calls += 1
        return float(real(simulation, total_mass_g_mol))

    monkeypatch.setattr(simulate, "density_g_cm3", counted)
    minimised = run_minimise(argon_run, "00_minimise")
    run_nvt(
        argon_run,
        "01_nvt",
        temperature_k=120.0,
        duration_ps=1.0,
        samples_per_segment=25,
        state_in=minimised.final_state,
    )
    # The last chunk is short whenever the step count does not divide, so
    # the loop takes one more reading than it was asked for rather than a
    # shorter final one.
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
    from openmm import unit

    from openmmpolymer.simulate import _build_simulation, _initialise

    settled = run_npt(argon_run, "npt", temperature_k=120.0, duration_ps=0.4)

    def velocities(reuse: bool) -> Any:
        simulation = _build_simulation(
            argon_run,
            "probe" if reuse else "probe_fresh",
            temperature_k=120.0,
            timestep_fs=2.0,
            friction_ps=1.0,
            barostat=None,
            pressure_bar=1.0,
            barostat_frequency=25,
        )
        _initialise(
            argon_run,
            simulation,
            "probe" if reuse else "probe_fresh",
            settled.final_state,
            120.0,
            reuse_velocities=reuse,
        )
        return np.asarray(
            simulation.context.getState(getVelocities=True)
            .getVelocities(asNumpy=True)
            .value_in_unit(unit.nanometer / unit.picosecond)
        )

    inherited, fresh = velocities(True), velocities(False)
    assert not np.allclose(inherited, fresh)


def test_pressures_can_be_set_per_axis(argon_run: Any) -> None:
    """A uniaxial load is one axis held somewhere the other two are not."""
    from openmmpolymer.simulate import _build_simulation, set_pressures

    simulation = _build_simulation(
        argon_run,
        "aniso",
        temperature_k=120.0,
        timestep_fs=2.0,
        friction_ps=1.0,
        barostat="anisotropic",
        pressure_bar=1.0,
        barostat_frequency=25,
    )
    set_pressures(simulation, (1.0, 1.0, -20.0), "anisotropic")
    assert simulation.context.getParameter("MonteCarloPressureZ") == pytest.approx(
        -20.0
    )
    assert simulation.context.getParameter("MonteCarloPressureX") == pytest.approx(1.0)


def test_three_different_pressures_are_refused_by_a_barostat_that_holds_one(
    argon_run: Any,
) -> None:
    """Applying the first of three to all of them would not be the run asked for."""
    from openmmpolymer.simulate import _build_simulation, set_pressures

    simulation = _build_simulation(
        argon_run,
        "iso",
        temperature_k=120.0,
        timestep_fs=2.0,
        friction_ps=1.0,
        barostat="isotropic",
        pressure_bar=1.0,
        barostat_frequency=25,
    )
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
    from openmmpolymer.simulate import (
        _build_simulation,
        _check_deformed_box,
        _initialise,
        nonbonded_cutoff_nm,
    )

    simulation = _build_simulation(
        argon_run,
        "probe",
        temperature_k=120.0,
        timestep_fs=2.0,
        friction_ps=1.0,
        barostat=None,
        pressure_bar=1.0,
        barostat_frequency=25,
    )
    _initialise(argon_run, simulation, "probe", None, 120.0)
    cutoff = nonbonded_cutoff_nm(simulation.system)
    assert cutoff > 0.0

    # The fixture's cell is comfortably above twice the cutoff.
    _check_deformed_box("probe", simulation, cutoff, 0.0)
    # Pretending the cutoff is most of the box is the same arithmetic.
    with pytest.raises(SimulationError, match="cutoff"):
        _check_deformed_box("probe", simulation, 10.0, 0.05)


def test_a_system_with_no_cutoff_is_not_checked_against_one(
    argon_run: Any,
) -> None:
    """Zero means "nothing here has one", which is not a box of zero size."""
    from openmmpolymer.simulate import _build_simulation, _check_deformed_box

    simulation = _build_simulation(
        argon_run,
        "probe",
        temperature_k=120.0,
        timestep_fs=2.0,
        friction_ps=1.0,
        barostat=None,
        pressure_bar=1.0,
        barostat_frequency=25,
    )
    _check_deformed_box("probe", simulation, 0.0, 0.0)


def test_a_shear_past_the_reduced_form_is_refused_before_the_ladder_runs(
    argon_run: Any,
) -> None:
    """Checked against every rung up front, not discovered on the last one."""
    from openmmpolymer.stress import StressError

    minimised = run_minimise(argon_run, "min", temperature_k=120.0)
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


def test_a_deformation_records_a_reference_cell_and_an_axis(
    argon_run: Any,
) -> None:
    """Recorded rather than recomputed, so a resumed chunk shares the origin."""
    minimised = run_minimise(argon_run, "min", temperature_k=120.0)
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
    argon_run: Any,
) -> None:
    """The strain a second chunk reports is measured from the unstrained cell."""
    minimised = run_minimise(argon_run, "min", temperature_k=120.0)
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
