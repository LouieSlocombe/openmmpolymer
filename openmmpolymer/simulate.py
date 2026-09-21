"""The stages a polymer melt goes through, and the machinery they share.

Each stage builds its own ``Simulation``. That is deliberate: a barostat cannot
be added to, removed from or swapped in a ``System`` that already has a
``Context`` - OpenMM accepts the call and silently does nothing - so the
ensemble is decided when the Context is built and not after. Rebuilding from a
serialised System costs a fraction of a second and removes a whole class of
runs that look like NPT and are not.

The other trap these stages exist to avoid is the temperature one. A barostat
reads its temperature from a global parameter, and the integrator reads its own
from itself. Setting one leaves the other where it was, which gives a melt
whose thermostat and barostat disagree, a plausible-looking density and no
error anywhere. :func:`set_temperature` sets both, and every stage reports the
temperature it actually ran at.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._seeds import derive_seed, seed_random_stream
from ._validation import require_choice, require_integer, require_positive
from .forcefield import PolymerForceField
from .mdsystem import (
    BAROSTAT_TEMPERATURE_PARAMETER,
    PackedBox,
    SystemSpec,
    build_system,
    check_timestep,
    make_barostat,
    select_platform,
)
from .reporters import TrajectoryOptions, reporting, rotate_existing, steps_for

log = logging.getLogger(__name__)

#: Barostat pressure parameters, per kind. The anisotropic barostat needs
#: three. The flexible one shares the isotropic one's name - observed from
#: ``MonteCarloFlexibleBarostat.Pressure()``, not assumed from the class
#: hierarchy, which has none in common.
_PRESSURE_PARAMETERS = {
    "isotropic": ("MonteCarloPressure",),
    "anisotropic": (
        "MonteCarloPressureX",
        "MonteCarloPressureY",
        "MonteCarloPressureZ",
    ),
    "flexible": ("MonteCarloPressure",),
}

#: Largest force, in kJ/mol/nm, a minimised cell may still carry. Above this
#: something is wrong with the geometry or the parameters and the first steps
#: of dynamics will not survive.
MAX_FORCE_AFTER_MINIMISATION = 1.0e5

#: Grams per mole in a gram, for the density arithmetic.
_AVOGADRO = 6.02214076e23


class SimulationError(RuntimeError):
    """A stage could not run, or produced something unphysical."""


@dataclass(frozen=True)
class StageResult:
    """What one stage did.

    Args:
        name: The stage's name.
        steps: Steps run.
        wall_seconds: How long it took.
        final_state: The portable state it left behind, which the next stage
            starts from.
        temperature_k: The temperature asked for, where there was one.
        mean_temperature_k: The temperature it ran at. These differing by more
            than a few kelvin means the thermostat and the barostat disagree.
        mean_density_g_cm3: The density it settled at.
        final_pdb: The structure it left.
        csv: Its numeric state data.
        samples: Anything the stage measured along the way - the quench records
            a density per temperature here.
        waypoints: The state written at the end of each segment, one entry per
            segment and in the same order as ``samples``. Empty unless
            ``run_segments`` was given ``waypoints=True``, so an ordinary
            stage carries no list of nulls into the manifest. A two-pass quench restarts its fine ladder from
            one of these, which is the only way to continue a cooling history
            rather than start a second one.
    """

    name: str
    steps: int
    wall_seconds: float
    final_state: str
    temperature_k: float | None = None
    mean_temperature_k: float | None = None
    mean_density_g_cm3: float | None = None
    final_pdb: str | None = None
    csv: str | None = None
    samples: dict[str, list[float]] = field(default_factory=dict)
    waypoints: tuple[str, ...] = ()


@dataclass
class RunContext:
    """Everything a stage needs that does not change between stages.

    Args:
        box: The packed cell.
        forcefield: Its force field.
        system_xml: The System, serialised once. Each stage deserialises it and
            adds whatever barostat it needs, because a barostat cannot be
            changed once a Context exists.
        spec: How the System was built.
        platform_name: Platform to use, or None for the fastest available.
        precision: Precision for the GPU platforms.
        seed: Master seed. Every stream derives from this.
        total_mass_g_mol: The cell's total mass, for the density arithmetic.
    """

    box: PackedBox
    forcefield: PolymerForceField
    system_xml: str
    spec: SystemSpec
    platform_name: str | None = None
    precision: str = "mixed"
    seed: int = 0xF0
    total_mass_g_mol: float = 0.0


def prepare_run(
    box: PackedBox,
    forcefield: PolymerForceField,
    spec: SystemSpec | None = None,
    *,
    platform: str | None = None,
    precision: str = "mixed",
    seed: int = 0xF0,
    system: Any | None = None,
) -> RunContext:
    """Build the System once and wrap it up for the stages to use.

    Args:
        box: The packed cell, already through
            :func:`openmmpolymer.mdsystem.prepare_box`.
        forcefield: Its force field.
        spec: How to build the System.
        platform: Platform name, or None for the fastest available.
        precision: Precision for the GPU platforms.
        seed: Master seed.
        system: A System to use instead of building one. For a System prepared
            some other way - and for tests, which can then exercise every stage
            without a force-field file.

    Returns:
        The context every stage takes.
    """
    import openmm as mm

    settings = spec or SystemSpec()
    if system is None:
        system = build_system(box, forcefield, settings)
    total_mass = sum(
        system.getParticleMass(index).value_in_unit_system(mm.unit.md_unit_system)
        for index in range(system.getNumParticles())
    )
    return RunContext(
        box=box,
        forcefield=forcefield,
        system_xml=mm.XmlSerializer.serialize(system),
        spec=settings,
        platform_name=platform,
        precision=precision,
        seed=seed,
        total_mass_g_mol=float(total_mass),
    )


def _build_simulation(
    run: RunContext,
    label: str,
    *,
    temperature_k: float,
    timestep_fs: float,
    friction_ps: float,
    barostat: str | None,
    pressure_bar: float,
    barostat_frequency: int,
    pressures_bar: Sequence[float] | None = None,
    scale_axes: Sequence[bool] = (True, True, True),
) -> Any:
    """Build a Simulation for one stage, with the ensemble it needs."""
    import openmm as mm
    from openmm import app, unit

    system = mm.XmlSerializer.deserialize(run.system_xml)
    if barostat is not None:
        system.addForce(
            make_barostat(
                barostat,
                temperature_k,
                pressure_bar,
                barostat_frequency,
                derive_seed(run.seed, "barostat", label),
                scale_molecules_as_rigid=run.spec.scale_molecules_as_rigid,
                pressures_bar=pressures_bar,
                scale_axes=scale_axes,
            )
        )
    integrator = mm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        friction_ps / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    seed_random_stream(integrator, derive_seed(run.seed, "thermostat", label))
    platform, properties = select_platform(run.platform_name, run.precision)
    return app.Simulation(run.box.topology, system, integrator, platform, properties)


def _initialise(
    run: RunContext,
    simulation: Any,
    label: str,
    state_in: str | Path | None,
    temperature_k: float | None,
    *,
    reuse_velocities: bool = True,
) -> None:
    """Put the simulation at its starting point.

    Positions, velocities and box vectors are transferred individually rather
    than through ``loadState``. A state saved under NPT carries the barostat's
    global parameters, and setting one of those on a Context that has no
    barostat raises - which would make every NPT-to-NVT transition a failure.

    Args:
        run: The run context.
        simulation: The simulation to place.
        label: The stage label, which every derived seed hangs off.
        state_in: A previous stage's state, or None for the packed cell.
        temperature_k: The temperature to draw velocities at, where any are
            drawn.
        reuse_velocities: Whether to carry the saved state's velocities over.
            False re-draws them from the Maxwell-Boltzmann distribution at
            *temperature_k*, seeded off *label*. That is what makes several
            runs from one equilibrated configuration into independent
            samples: inheriting the velocities as well as the positions gives
            the same trajectory every time, and a spread computed over those
            replicas would be a spread of zero dressed up as an error bar.
    """
    import openmm as mm

    if state_in is None:
        simulation.context.setPositions(run.box.positions)
        simulation.context.computeVirtualSites()
        if temperature_k is not None:
            simulation.context.setVelocitiesToTemperature(
                temperature_k, derive_seed(run.seed, "velocities", label)
            )
        return

    state = mm.XmlSerializer.deserialize(Path(state_in).read_text())
    simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    simulation.context.setPositions(state.getPositions())
    simulation.context.computeVirtualSites()
    if not reuse_velocities:
        if temperature_k is None:
            raise SimulationError(
                f"Stage {label!r} asked for fresh velocities without a "
                "temperature to draw them at."
            )
        simulation.context.setVelocitiesToTemperature(
            temperature_k, derive_seed(run.seed, "velocities", label)
        )
        return
    try:
        simulation.context.setVelocities(state.getVelocities())
    except Exception:  # pragma: no cover - a state saved without velocities
        if temperature_k is not None:
            simulation.context.setVelocitiesToTemperature(
                temperature_k, derive_seed(run.seed, "velocities", label)
            )


def set_temperature(
    simulation: Any, temperature_k: float, barostat: str | None
) -> None:
    """Set the temperature everywhere it is held.

    The integrator keeps its own, the barostat reads a global parameter, and
    they are not the same setting. Changing only the parameter leaves the
    thermostat where it was: the quench then integrates every window at the
    starting temperature while the barostat accepts volume moves as though it
    were at the ramp temperature, and the density curve that comes out looks
    entirely reasonable.

    Args:
        simulation: The running simulation.
        temperature_k: The temperature to set.
        barostat: Which barostat is attached, or None.
    """
    from openmm import unit

    simulation.integrator.setTemperature(temperature_k * unit.kelvin)
    if barostat is not None:
        simulation.context.setParameter(
            BAROSTAT_TEMPERATURE_PARAMETER[barostat], temperature_k
        )


def set_pressure(simulation: Any, pressure_bar: float, barostat: str) -> None:
    """Set the barostat's pressure, in bar.

    Args:
        simulation: The running simulation.
        pressure_bar: The pressure to set.
        barostat: Which barostat is attached.
    """
    for name in _PRESSURE_PARAMETERS[barostat]:
        simulation.context.setParameter(name, pressure_bar)


def set_pressures(
    simulation: Any, pressures_bar: Sequence[float], barostat: str
) -> None:
    """Set a different pressure on each axis, in bar.

    Only the anisotropic barostat has three; for the other two this is
    :func:`set_pressure` and every entry must agree, because silently
    applying the first of three to all of them is how a uniaxial load becomes
    a hydrostatic one without anything saying so.

    Args:
        simulation: The running simulation.
        pressures_bar: One pressure per axis.
        barostat: Which barostat is attached.

    Raises:
        ValueError: Three different pressures were given to a barostat that
            holds one.
    """
    values = tuple(float(value) for value in pressures_bar)
    if len(values) != 3:
        raise ValueError(f"pressures_bar={pressures_bar!r} must have three entries.")
    names = _PRESSURE_PARAMETERS[barostat]
    if len(names) == 1:
        if len(set(values)) != 1:
            raise ValueError(
                f"A {barostat} barostat holds one pressure, but "
                f"{values} are three different ones."
            )
        simulation.context.setParameter(names[0], values[0])
        return
    for name, value in zip(names, values, strict=True):
        simulation.context.setParameter(name, value)


def density_g_cm3(simulation: Any, total_mass_g_mol: float) -> float:
    """Return the cell's current density.

    Args:
        simulation: The running simulation.
        total_mass_g_mol: The cell's total mass.

    Returns:
        The density in g/cm3.
    """
    from openmm import unit

    volume_nm3 = (
        simulation.context.getState()
        .getPeriodicBoxVolume()
        .value_in_unit(unit.nanometer**3)
    )
    return float(total_mass_g_mol * 1.0e21 / (_AVOGADRO * volume_nm3))


def temperature_k_of(simulation: Any) -> float:
    """Return the cell's current instantaneous temperature, in kelvin."""
    from openmm import unit

    state = simulation.context.getState(getEnergy=True)
    system = simulation.system
    degrees = (
        3 * system.getNumParticles()
        - system.getNumConstraints()
        - (3 if _has_cm_remover(system) else 0)
    )
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    )
    return float(2.0 * kinetic / (degrees * gas_constant))


def _has_cm_remover(system: Any) -> bool:
    """Whether the System removes centre-of-mass motion."""
    import openmm as mm

    return any(isinstance(force, mm.CMMotionRemover) for force in system.getForces())


@dataclass(frozen=True)
class Segment:
    """One stretch of dynamics at a fixed temperature and pressure.

    Args:
        temperature_k: The temperature to hold.
        duration_ps: How long to hold it.
        pressure_bar: The pressure, for a stage that has a barostat.
        label: What to call it in the log and in the samples.
    """

    temperature_k: float
    duration_ps: float
    pressure_bar: float = 1.0
    label: str = ""


#: Samples taken per segment. The mean is over the second half, so a segment
#: that is still relaxing does not drag its own average.
_SAMPLES_PER_SEGMENT = 10


def _save_final(simulation: Any, prefix: Path) -> tuple[str, str]:
    """Write the stage's end state and structure, returning both paths."""
    from openmm import app

    prefix.parent.mkdir(parents=True, exist_ok=True)
    state_path = prefix.with_suffix(".state.xml")
    pdb_path = prefix.with_suffix(".pdb")
    simulation.saveState(str(state_path))

    state = simulation.context.getState(getPositions=True)
    simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with pdb_path.open("w") as handle:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), handle)
    return str(state_path), str(pdb_path)


def _save_waypoint(
    simulation: Any, prefix: Path, index: int, temperature_k: float
) -> str:
    """Write the state at the end of one segment, and return where it went.

    Named by index and temperature so the directory both sorts into run order
    and says what each file is. No structure is written beside it:
    :func:`_initialise` restarts from the serialised state alone, and a PDB per
    temperature would double the cost of an already bulky option.
    """
    path = prefix.parent / (
        f"{prefix.name}_waypoint{index:02d}_{temperature_k:.0f}K.state.xml"
    )
    simulation.saveState(str(path))
    return str(path)


def run_minimise(
    run: RunContext,
    output_prefix: str | Path = "00_minimise",
    *,
    tolerance_kj_per_nm: float = 10.0,
    max_iterations: int = 10_000,
    temperature_k: float = 300.0,
    state_in: str | Path | None = None,
) -> StageResult:
    """Minimise the cell's energy.

    A packed cell always needs this. packmol guarantees no two atoms of
    different molecules are closer than the tolerance and nothing at all about
    how strained the result is.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        tolerance_kj_per_nm: Force tolerance to stop at.
        max_iterations: Iteration cap; zero means run to the tolerance.
        temperature_k: Temperature for the integrator, which does not step.
        state_in: A previous stage's state, or None to start from the packed
            coordinates.

    Returns:
        What the stage did.

    Raises:
        SimulationError: The minimised cell still carries impossible forces.
    """
    from openmm import unit

    prefix = Path(output_prefix)
    started = time.monotonic()
    simulation = _build_simulation(
        run,
        prefix.name,
        temperature_k=temperature_k,
        timestep_fs=1.0,
        friction_ps=1.0,
        barostat=None,
        pressure_bar=1.0,
        barostat_frequency=25,
    )
    _initialise(run, simulation, prefix.name, state_in, temperature_k)

    before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    if not np.isfinite(before.value_in_unit(unit.kilojoule_per_mole)):
        raise SimulationError(
            "The starting energy is not finite, so there is nothing for the "
            "minimiser to descend. Atoms are on top of each other: check that "
            "openmmpolymer.packing.check_packing passes on this cell, and "
            "that the packed coordinates belong to this topology."
        )
    try:
        simulation.minimizeEnergy(
            tolerance=tolerance_kj_per_nm * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max_iterations,
        )
    except Exception as error:
        # OpenMM's own message for this points at a FAQ; the cause here is
        # almost always the packing, and saying so saves the round trip.
        raise SimulationError(
            f"Minimisation failed: {error}. A packed cell that does this has "
            "overlapping molecules - repack at a lower density or with a "
            "larger tolerance, and check openmmpolymer.packing.check_packing "
            "passes."
        ) from error
    after = simulation.context.getState(getEnergy=True, getForces=True)
    energy = after.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = after.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    max_force = float(np.sqrt((np.asarray(forces) ** 2).sum(axis=1)).max())
    _check_minimised(energy, max_force)

    log.info(
        "Minimised: %.4g -> %.4g kJ/mol, largest force %.3g kJ/mol/nm.",
        before.value_in_unit(unit.kilojoule_per_mole),
        energy,
        max_force,
    )
    state_path, pdb_path = _save_final(simulation, prefix)
    return StageResult(
        name=prefix.name,
        steps=0,
        wall_seconds=time.monotonic() - started,
        final_state=state_path,
        final_pdb=pdb_path,
        mean_density_g_cm3=density_g_cm3(simulation, run.total_mass_g_mol),
        samples={"potential_energy_kj_mol": [energy], "max_force": [max_force]},
    )


def _check_minimised(energy: float, max_force: float) -> None:
    """Raise unless the minimised cell could survive the first step.

    Args:
        energy: The minimised potential energy.
        max_force: The largest force on any particle.

    Raises:
        SimulationError: The energy is not finite, or a force is impossible.
    """
    if not np.isfinite(energy):
        raise SimulationError(
            "The minimised energy is not finite. The packed cell has atoms on "
            "top of each other, or the force field does not describe it."
        )
    if max_force > MAX_FORCE_AFTER_MINIMISATION:
        raise SimulationError(
            f"The largest force after minimisation is {max_force:.3g} "
            f"kJ/mol/nm, against a limit of {MAX_FORCE_AFTER_MINIMISATION:.3g}. "
            "Dynamics will not survive this. Repack with a larger tolerance or "
            "at a lower density, and check openmmpolymer.packing.check_packing "
            "passed."
        )


def run_segments(
    run: RunContext,
    name: str,
    segments: Sequence[Segment],
    output_prefix: str | Path,
    *,
    barostat: str | None = None,
    timestep_fs: float | None = None,
    friction_ps: float = 1.0,
    barostat_frequency: int = 25,
    trajectory: TrajectoryOptions | str = "none",
    report_interval_ps: float = 10.0,
    state_in: str | Path | None = None,
    waypoints: bool = False,
    samples_per_segment: int = _SAMPLES_PER_SEGMENT,
) -> StageResult:
    """Run a sequence of segments in one ensemble.

    Every stage below is this function with a different segment list. The
    ensemble is fixed for the whole stage because it is fixed when the Context
    is built.

    Args:
        run: The run context.
        name: The stage's name.
        segments: What to run, in order.
        output_prefix: Stem for this stage's files.
        barostat: ``"isotropic"``, ``"anisotropic"`` or None for NVT.
        timestep_fs: The integration timestep, or None to take the longest
            step that is safe for the hottest segment. Checked against the
            hottest segment and not the first one: a stage that starts at 300 K
            and anneals to 600 K has to be integrated for the 600 K.
        friction_ps: Langevin friction.
        barostat_frequency: Steps between volume moves.
        trajectory: Trajectory settings.
        report_interval_ps: Time between state-data rows.
        state_in: The previous stage's state.
        waypoints: Write a state at the end of every segment, not just at the
            end of the stage. A quench's waypoints are what let a second,
            finer pass carry on from the middle of the first one instead of
            reheating a glass. Off by default: it costs one serialised state
            per segment, which for a cell of tens of thousands of atoms is
            megabytes each.
        samples_per_segment: How many times each segment is interrupted to
            measure its density and temperature. The mean is over the second
            half of them, so this sets how many readings that mean rests on -
            the default of ten leaves five, which is thin for a segment whose
            whole purpose is a low-noise point on a curve.

    Returns:
        What the stage did, including a density and a temperature per segment.

    Raises:
        SimulationError: The run produced a non-finite energy.
        ValueError: The timestep is too long for the hottest segment.
    """
    if not segments:
        raise ValueError(f"Stage {name!r} has no segments to run.")

    require_integer(samples_per_segment, minimum=1, name="samples_per_segment")
    hottest = max(segment.temperature_k for segment in segments)
    if timestep_fs is None:
        timestep_fs = safe_timestep_fs(hottest, run.spec)
    require_positive(timestep_fs, None, name="timestep_fs")
    check_timestep(timestep_fs, hottest, run.spec)

    prefix = Path(output_prefix)
    started = time.monotonic()
    first = segments[0]
    simulation = _build_simulation(
        run,
        prefix.name,
        temperature_k=first.temperature_k,
        timestep_fs=timestep_fs,
        friction_ps=friction_ps,
        barostat=barostat,
        pressure_bar=first.pressure_bar,
        barostat_frequency=barostat_frequency,
    )
    _initialise(run, simulation, prefix.name, state_in, first.temperature_k)

    per_segment = [steps_for(segment.duration_ps, timestep_fs) for segment in segments]
    total_steps = sum(per_segment)
    report_interval = steps_for(report_interval_ps, timestep_fs)
    trajectory_interval = _frame_interval(trajectory, timestep_fs)
    if trajectory_interval is not None and trajectory_interval > total_steps:
        log.warning(
            "%s: a frame every %d steps, but the stage is only %d steps long, so "
            "the trajectory will have no frames in it. Lower interval_ps below "
            "%.4g ps to get one.",
            name,
            trajectory_interval,
            total_steps,
            total_steps * timestep_fs / 1000.0,
        )

    samples: dict[str, list[float]] = {
        "segment_temperature_k": [],
        "segment_density_g_cm3": [],
        "segment_mean_temperature_k": [],
        # Recorded rather than inferred: a stage's CSV knows only its total
        # time, so a ladder split across stages or resumed part-way through
        # would otherwise have its cooling rate worked out wrong rather than
        # reported as unknown.
        "segment_duration_ps": [],
    }
    waypoint_paths: list[str] = []
    log.info(
        "%s: %d segments, %.1f ps at %.1f fs (%d steps)%s.",
        name,
        len(segments),
        sum(segment.duration_ps for segment in segments),
        timestep_fs,
        total_steps,
        "" if barostat is None else f", {barostat} barostat",
    )

    with reporting(
        simulation,
        prefix,
        total_steps=total_steps,
        report_interval=report_interval,
        trajectory=trajectory,
        trajectory_interval=trajectory_interval,
    ) as paths:
        for index, (segment, steps) in enumerate(
            zip(segments, per_segment, strict=True)
        ):
            set_temperature(simulation, segment.temperature_k, barostat)
            if barostat is not None:
                set_pressure(simulation, segment.pressure_bar, barostat)
            density, temperature = _run_segment(
                simulation, steps, run.total_mass_g_mol, samples_per_segment
            )
            samples["segment_temperature_k"].append(segment.temperature_k)
            samples["segment_density_g_cm3"].append(density)
            samples["segment_mean_temperature_k"].append(temperature)
            samples["segment_duration_ps"].append(segment.duration_ps)
            if waypoints:
                waypoint_paths.append(
                    _save_waypoint(simulation, prefix, index, segment.temperature_k)
                )
            log.info(
                "  %s%.0f K: %.4f g/cm3, ran at %.0f K.",
                f"{segment.label} " if segment.label else "",
                segment.temperature_k,
                density,
                temperature,
            )
        state_path, pdb_path = _save_final(simulation, prefix)

    mean_temperature = float(np.mean(samples["segment_mean_temperature_k"]))
    _check_temperature(name, segments, samples["segment_mean_temperature_k"])
    return StageResult(
        name=name,
        steps=total_steps,
        wall_seconds=time.monotonic() - started,
        final_state=state_path,
        temperature_k=segments[-1].temperature_k,
        mean_temperature_k=mean_temperature,
        mean_density_g_cm3=samples["segment_density_g_cm3"][-1],
        final_pdb=pdb_path,
        csv=paths.csv,
        samples=samples,
        waypoints=tuple(waypoint_paths),
    )


def _run_segment(
    simulation: Any,
    steps: int,
    total_mass_g_mol: float,
    samples_per_segment: int = _SAMPLES_PER_SEGMENT,
) -> tuple[float, float]:
    """Run one segment, returning its settled density and temperature.

    Sampled in chunks and averaged over the second half, so a segment that
    spends its first part relaxing does not drag its own average.
    """
    from openmm import unit

    chunk = max(1, steps // samples_per_segment)
    densities: list[float] = []
    temperatures: list[float] = []
    remaining = steps
    while remaining > 0:
        simulation.step(min(chunk, remaining))
        remaining -= chunk
        energy = (
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        if not np.isfinite(energy):
            raise SimulationError(
                "The potential energy went to NaN. The timestep is too long "
                "for this temperature, or the starting geometry was strained. "
                "Minimise again, or shorten the timestep."
            )
        densities.append(density_g_cm3(simulation, total_mass_g_mol))
        temperatures.append(temperature_k_of(simulation))

    half = max(1, len(densities) // 2)
    return float(np.mean(densities[-half:])), float(np.mean(temperatures[-half:]))


#: How far a stage's realised temperature may sit from the one it asked for.
TEMPERATURE_TOLERANCE_K = 25.0


def _check_temperature(
    name: str, segments: Sequence[Segment], measured: Sequence[float]
) -> None:
    """Say so when a segment did not run at the temperature it was given.

    This is the cheap check that catches the thermostat and the barostat
    disagreeing, which otherwise produces a perfectly plausible density at
    entirely the wrong temperature.
    """
    for segment, actual in zip(segments, measured, strict=True):
        drift = abs(actual - segment.temperature_k)
        if drift > TEMPERATURE_TOLERANCE_K:
            log.warning(
                "%s ran a %.0f K segment at %.0f K. Either it had not "
                "equilibrated, or the thermostat and the barostat are set to "
                "different temperatures.",
                name,
                segment.temperature_k,
                actual,
            )


#: Timesteps are rounded down to a multiple of this, so a derated step is a
#: round number rather than 1.4142 fs.
_TIMESTEP_QUANTUM_FS = 0.25


def _frame_interval(
    trajectory: TrajectoryOptions | str, timestep_fs: float
) -> int | None:
    """Steps between trajectory frames, or None to take the default.

    ``TrajectoryOptions.interval_ps`` is a time because that is what a caller
    knows; the reporter needs a step count, and only the stage knows the
    timestep it settled on. None is passed straight through so that
    :func:`openmmpolymer.reporters.reporting` keeps its own default of ten
    times the state-data interval, which is what a stage naming a bare format
    string has always got.

    Args:
        trajectory: Trajectory settings, or just a format name.
        timestep_fs: The timestep the stage is running at.

    Returns:
        Steps between frames, or None for the reporter's default.
    """
    if isinstance(trajectory, str) or trajectory.interval_ps is None:
        return None
    return steps_for(trajectory.interval_ps, timestep_fs)


def safe_timestep_fs(temperature_k: float, spec: SystemSpec) -> float:
    """Return the longest sensible timestep for this temperature.

    The limit derated for temperature, rounded down to a quarter of a
    femtosecond so that what ends up in the manifest is a number someone can
    read.

    Args:
        temperature_k: The hottest temperature the stage reaches.
        spec: The System settings, for constraints and hydrogen mass.

    Returns:
        The timestep in femtoseconds.
    """
    from .mdsystem import max_timestep_fs

    limit = max_timestep_fs(temperature_k, spec)
    quantised = int(limit / _TIMESTEP_QUANTUM_FS) * _TIMESTEP_QUANTUM_FS
    return max(_TIMESTEP_QUANTUM_FS, quantised)


def run_pushoff(
    run: RunContext,
    output_prefix: str | Path = "01_pushoff",
    *,
    temperature_k: float = 300.0,
    duration_ps: float = 10.0,
    timesteps_fs: Sequence[float] = (0.1, 0.25, 0.5),
    friction_ps: float = 10.0,
    state_in: str | Path | None = None,
) -> StageResult:
    """Relieve what is left of the packing before real dynamics start.

    Minimisation takes the worst of a packed cell out; it does not take out the
    strain, and the first step at a normal timestep turns that strain into
    velocity. This runs the same total time at a sequence of increasing
    timesteps under heavy friction, which drains the energy instead. It is
    cheap, it is boring, and without it a rigid-placement melt regularly dies
    on its first picosecond.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: Temperature to hold.
        duration_ps: Total time, split evenly between the timesteps.
        timesteps_fs: The ladder of timesteps to climb.
        friction_ps: Langevin friction. High on purpose.
        state_in: The previous stage's state.

    Returns:
        What the stage did.
    """
    prefix = Path(output_prefix)
    started = time.monotonic()
    state: str | Path | None = state_in
    steps = 0
    results: list[StageResult] = []
    share = duration_ps / len(timesteps_fs)

    for index, timestep in enumerate(timesteps_fs):
        result = run_segments(
            run,
            f"{prefix.name}[{timestep:g} fs]",
            [Segment(temperature_k, share, label=f"{timestep:g} fs")],
            prefix.with_name(f"{prefix.name}_{index}"),
            barostat=None,
            timestep_fs=timestep,
            friction_ps=friction_ps,
            state_in=state,
        )
        state = result.final_state
        steps += result.steps
        results.append(result)

    last = results[-1]
    return StageResult(
        name=prefix.name,
        steps=steps,
        wall_seconds=time.monotonic() - started,
        final_state=last.final_state,
        temperature_k=temperature_k,
        mean_temperature_k=last.mean_temperature_k,
        mean_density_g_cm3=last.mean_density_g_cm3,
        final_pdb=last.final_pdb,
        csv=last.csv,
        samples={"timestep_fs": list(timesteps_fs)},
    )


def run_nvt(
    run: RunContext,
    output_prefix: str | Path = "02_nvt",
    *,
    temperature_k: float = 600.0,
    duration_ps: float = 500.0,
    **kwargs: Any,
) -> StageResult:
    """Hold the cell at constant volume and temperature.

    Well above the glass transition, this is where the packing's memory starts
    to come out: packmol placed rigid coils that do not interpenetrate, and
    only dynamics fixes that.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: Temperature to hold.
        duration_ps: How long to hold it.
        **kwargs: Passed to :func:`run_segments`.

    Returns:
        What the stage did.
    """
    return run_segments(
        run,
        Path(output_prefix).name,
        [Segment(temperature_k, duration_ps)],
        output_prefix,
        barostat=None,
        **kwargs,
    )


def run_npt(
    run: RunContext,
    output_prefix: str | Path = "04_npt",
    *,
    temperature_k: float = 600.0,
    pressure_bar: float = 1.0,
    duration_ps: float = 2000.0,
    barostat: str = "isotropic",
    **kwargs: Any,
) -> StageResult:
    """Hold the cell at constant pressure and temperature.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: Temperature to hold.
        pressure_bar: Pressure to hold.
        duration_ps: How long to hold it.
        barostat: Which barostat to attach.
        **kwargs: Passed to :func:`run_segments`.

    Returns:
        What the stage did.
    """
    return run_segments(
        run,
        Path(output_prefix).name,
        [Segment(temperature_k, duration_ps, pressure_bar)],
        output_prefix,
        barostat=barostat,
        **kwargs,
    )


#: The pressure ladder :func:`run_compress` climbs and comes back down.
DEFAULT_COMPRESSION_BAR = (1.0, 100.0, 500.0, 1000.0, 500.0, 100.0, 1.0)


def run_compress(
    run: RunContext,
    output_prefix: str | Path = "03_compress",
    *,
    temperature_k: float = 600.0,
    pressures_bar: Sequence[float] = DEFAULT_COMPRESSION_BAR,
    duration_ps_each: float = 100.0,
    barostat: str = "isotropic",
    **kwargs: Any,
) -> StageResult:
    """Drive the loose packing down to something like a melt density.

    The cell was packed at around a third of its final density, because packmol
    cannot reliably do better. Raising the pressure and bringing it back down
    closes that gap far faster than waiting at one bar would, and the excursion
    back down leaves the cell at the pressure the run actually wants.

    Note what this does not do. Compression rescales coordinates; it does not
    make chains interpenetrate. That is what the high-temperature stages are
    for, and they need far longer than this.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: Temperature to hold throughout.
        pressures_bar: The pressure ladder.
        duration_ps_each: Time at each rung.
        barostat: Which barostat to attach.
        **kwargs: Passed to :func:`run_segments`.

    Returns:
        What the stage did, with a density per rung.
    """
    segments = [
        Segment(temperature_k, duration_ps_each, pressure, label=f"{pressure:g} bar")
        for pressure in pressures_bar
    ]
    result = run_segments(
        run,
        Path(output_prefix).name,
        segments,
        output_prefix,
        barostat=barostat,
        **kwargs,
    )
    result.samples["segment_pressure_bar"] = list(pressures_bar)
    return result


def run_anneal(
    run: RunContext,
    output_prefix: str | Path = "05_anneal",
    *,
    t_low: float = 300.0,
    t_high: float = 600.0,
    n_cycles: int = 3,
    ramp_windows: int = 5,
    window_ps: float = 20.0,
    hold_ps: float = 50.0,
    pressure_bar: float = 1.0,
    barostat: str = "isotropic",
    **kwargs: Any,
) -> StageResult:
    """Melt the cell and let it set, repeatedly.

    Cycling well above and back below the glass transition is what "melt it"
    means in practice: each excursion upwards gives the chains the mobility to
    forget a little more of how they were placed, and each one downwards lets
    the cell find a density at the temperature of interest.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        t_low: The bottom of each cycle.
        t_high: The top of each cycle.
        n_cycles: How many times round.
        ramp_windows: Steps in each ramp.
        window_ps: Time at each ramp step.
        hold_ps: Time at the top and at the bottom.
        pressure_bar: Pressure to hold throughout.
        barostat: Which barostat to attach.
        **kwargs: Passed to :func:`run_segments`.

    Returns:
        What the stage did, with a density per temperature visited.
    """
    segments: list[Segment] = []
    for cycle in range(n_cycles):
        for direction, label in ((1, "heat"), (-1, "cool")):
            ends = (t_low, t_high) if direction == 1 else (t_high, t_low)
            for window in range(1, ramp_windows + 1):
                temperature = ends[0] + (ends[1] - ends[0]) * window / ramp_windows
                segments.append(
                    Segment(
                        temperature,
                        window_ps,
                        pressure_bar,
                        label=f"cycle {cycle + 1} {label}",
                    )
                )
            segments.append(
                Segment(
                    ends[1],
                    hold_ps,
                    pressure_bar,
                    label=f"cycle {cycle + 1} hold {ends[1]:.0f} K",
                )
            )
    return run_segments(
        run,
        Path(output_prefix).name,
        segments,
        output_prefix,
        barostat=barostat,
        **kwargs,
    )


def quench_temperatures(t_start: float, t_end: float, step_k: float) -> list[float]:
    """Return the ladder of temperatures a quench visits, hottest first.

    Split out of :func:`run_quench` so that everything which needs to know how
    long a quench is - the cost reported before it starts, the chunker that
    splits it into resumable stages, and the stage that runs it - counts the
    same temperatures. A ladder that does not divide evenly still ends exactly
    at *t_end*, because the bottom of the curve is a point someone chose.

    Args:
        t_start: Where cooling starts.
        t_end: Where it stops, always visited.
        step_k: How far it drops at each step.

    Returns:
        The temperatures, descending.

    Raises:
        ValueError: The ramp does not descend, or the step is not positive.
    """
    if t_end >= t_start:
        raise ValueError(
            f"t_end={t_end} must be below t_start={t_start}: a quench cools."
        )
    require_positive(step_k, None, name="step_k")

    temperatures: list[float] = []
    temperature = t_start
    while temperature > t_end + 1e-9:
        temperatures.append(temperature)
        temperature -= step_k
    temperatures.append(t_end)
    return temperatures


def _descending_ladder(temperatures_k: Sequence[float]) -> list[float]:
    """Check a caller-supplied ladder is one a quench could run."""
    ladder = [float(value) for value in temperatures_k]
    if not ladder:
        raise ValueError(
            "temperatures_k is empty, so there is nothing to hold. Give the "
            "temperatures to visit, or leave it out and name t_start, t_end "
            "and step_k instead."
        )
    for hotter, cooler in itertools.pairwise(ladder):
        if cooler >= hotter:
            raise ValueError(
                f"temperatures_k goes {hotter} -> {cooler}: a quench cools, so "
                "the ladder has to descend."
            )
    return ladder


def run_quench(
    run: RunContext,
    output_prefix: str | Path = "06_quench",
    *,
    t_start: float = 600.0,
    t_end: float = 200.0,
    step_k: float = 20.0,
    hold_ps: float = 200.0,
    pressure_bar: float = 1.0,
    barostat: str = "isotropic",
    temperatures_k: Sequence[float] | None = None,
    **kwargs: Any,
) -> StageResult:
    """Cool the cell in steps, recording the density at each temperature.

    The result is a specific-volume-against-temperature curve, which is how a
    glass transition is located in a simulation. Read it knowing what it is:
    every all-atom cooling rate is many orders of magnitude faster than any
    experiment, so the transition it shows sits well above the measured one.
    The shape is informative, the number is not directly comparable.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        t_start: Where to start cooling.
        t_end: Where to stop.
        step_k: How far to drop at each step.
        hold_ps: Time held at each temperature.
        pressure_bar: Pressure to hold throughout.
        barostat: Which barostat to attach.
        temperatures_k: The exact ladder to visit, replacing *t_start*,
            *t_end* and *step_k*. A long quench is split into several stages so
            that an interrupted run resumes at the stage it stopped in rather
            than at the top of the ramp, and handing each piece its own slice
            of one ladder is what keeps a temperature from being repeated or
            skipped at every boundary.
        **kwargs: Passed to :func:`run_segments`, including ``waypoints``.

    Returns:
        What the stage did. ``samples`` carries the temperatures and the
        densities they settled at.

    Raises:
        ValueError: The ramp does not descend.
    """
    temperatures = (
        quench_temperatures(t_start, t_end, step_k)
        if temperatures_k is None
        else _descending_ladder(temperatures_k)
    )

    segments = [
        Segment(value, hold_ps, pressure_bar, label=f"{value:.0f} K")
        for value in temperatures
    ]
    return run_segments(
        run,
        Path(output_prefix).name,
        segments,
        output_prefix,
        barostat=barostat,
        **kwargs,
    )


def run_production(
    run: RunContext,
    output_prefix: str | Path = "07_production",
    *,
    temperature_k: float = 450.0,
    pressure_bar: float | None = 1.0,
    duration_ps: float = 10_000.0,
    trajectory: TrajectoryOptions | str = "xtc",
    **kwargs: Any,
) -> StageResult:
    """Run the production trajectory.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: Temperature to hold.
        pressure_bar: Pressure to hold, or None to run at constant volume.
        duration_ps: How long to run.
        trajectory: Trajectory settings.
        **kwargs: Passed to :func:`run_segments`.

    Returns:
        What the stage did.
    """
    return run_segments(
        run,
        Path(output_prefix).name,
        [Segment(temperature_k, duration_ps, pressure_bar or 1.0)],
        output_prefix,
        barostat=None if pressure_bar is None else "isotropic",
        trajectory=trajectory,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Deformation
# --------------------------------------------------------------------------


#: How many standard errors the realised lateral pressure may sit from the
#: one the barostat was set to before the stage says so. Scaled to the
#: window's own noise rather than fixed, because a constant in bar is either
#: never reached or reached every step depending on the cell.
LATERAL_SIGMA_TOLERANCE = 4.0

#: A floor under that, so a window that happened to be very quiet does not
#: warn about a difference of no consequence.
LATERAL_PRESSURE_FLOOR_BAR = 100.0


def _box_lengths_nm(simulation: Any) -> npt.NDArray[np.float64]:
    """The three cell edge lengths, in nanometres."""
    from openmm import unit

    vectors = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
    lengths = np.asarray(vectors.value_in_unit(unit.nanometer), dtype=np.float64)
    return np.asarray([lengths[axis][axis] for axis in range(3)], dtype=np.float64)


def _positions_nm(simulation: Any) -> npt.NDArray[np.float64]:
    """Whole-molecule positions, in nanometres.

    Deliberately without ``enforcePeriodicBox``: wrapping puts the two halves
    of a molecule that straddles a face on opposite sides of the cell, and its
    centre of mass somewhere in the middle of neither.
    """
    from openmm import unit

    state = simulation.context.getState(getPositions=True)
    return np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )


def _apply_positions(
    simulation: Any,
    positions_nm: npt.NDArray[np.float64],
    vectors_nm: Any,
    *,
    tolerance: float = 1.0e-8,
) -> None:
    """Put a new box and new positions onto a live Context, constraints intact.

    An affine strain moves every atom, which stretches every constrained
    bond it is not parallel to. ``applyConstraints`` puts them back before
    anything reads an energy, and ``applyVelocityConstraints`` does the same
    for the velocity components along them - without it the thermostat spends
    the next picosecond removing motion the constraint does not allow, and
    the temperature reads high.

    Measured with the usual ``constraints="hbonds"``: a strain increment of
    0.002 stretches the longest constrained bond by 0.2 pm, and the repair
    moves no atom further than 2e-4 nm.
    """
    from openmm import unit

    simulation.context.setPeriodicBoxVectors(*vectors_nm)
    simulation.context.setPositions(positions_nm * unit.nanometer)
    simulation.context.computeVirtualSites()
    if simulation.system.getNumConstraints():
        simulation.context.applyConstraints(tolerance)
        simulation.context.applyVelocityConstraints(tolerance)


def _sample_mechanics(
    simulation: Any,
    steps: int,
    total_mass_g_mol: float,
    samples: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
    """Run one window: its mean stress, that mean's error, density and temperature.

    Averaged over the second half, as every other stage here averages, because
    the first half is the cell responding to whatever was just done to it. The
    stress is the noisy one: OpenMM warns that the instantaneous pressure
    fluctuates enormously, so the number that matters is this mean and the
    spread behind it.
    """
    from openmm import unit

    from .stress import stress_tensor_bar

    chunk = max(1, steps // samples)
    stresses: list[npt.NDArray[np.float64]] = []
    densities: list[float] = []
    temperatures: list[float] = []
    remaining = steps
    while remaining > 0:
        simulation.step(min(chunk, remaining))
        remaining -= chunk
        energy = (
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        if not np.isfinite(energy):
            raise SimulationError(
                "The potential energy went to NaN during a deformation. The "
                "strain increment is too large for the relaxation time, or "
                "the timestep is too long. Lower strain_increment, raise "
                "relax_ps, or minimise after each step."
            )
        stresses.append(stress_tensor_bar(simulation))
        densities.append(density_g_cm3(simulation, total_mass_g_mol))
        temperatures.append(temperature_k_of(simulation))

    half = max(1, len(densities) // 2)
    window = np.stack(stresses[-half:])
    # Averaged column by column over the components that were measured. A
    # plain nanmean over an all-NaN off-diagonal - which is every off-diagonal
    # under an anisotropic barostat - warns "mean of empty slice", and this
    # package runs its tests with warnings as errors.
    measured = np.isfinite(window).all(axis=0)
    mean_stress = np.full((3, 3), np.nan, dtype=np.float64)
    mean_stress[measured] = window[:, measured].mean(axis=0)
    spread = np.full((3, 3), np.nan, dtype=np.float64)
    if window.shape[0] > 1:
        spread[measured] = window[:, measured].std(axis=0, ddof=1) / math.sqrt(
            window.shape[0]
        )
    return (
        mean_stress,
        spread,
        float(np.mean(densities[-half:])),
        float(np.mean(temperatures[-half:])),
    )


def run_deform(
    run: RunContext,
    output_prefix: str | Path = "06_deform",
    *,
    temperature_k: float = 298.15,
    pressure_bar: float = 1.0,
    axis: int = 2,
    strain_increment: float = 0.002,
    n_steps: int = 25,
    relax_ps: float = 50.0,
    strain_start: float = 0.0,
    reference_box_nm: Sequence[float] | None = None,
    minimise_each_step: bool = False,
    new_velocities: bool = False,
    samples_per_step: int = 20,
    barostat_frequency: int = 25,
    timestep_fs: float | None = None,
    friction_ps: float = 1.0,
    trajectory: TrajectoryOptions | str = "none",
    report_interval_ps: float = 10.0,
    state_in: str | Path | None = None,
    waypoints: bool = False,
) -> StageResult:
    """Stretch the cell along one axis in steps, measuring the stress at each.

    OpenMM has no continuous deformation, so a strain rate is a staircase:
    scale the box by one increment, let the cell relax under a barostat that
    holds the other two axes at pressure and leaves the driven one alone, and
    read the stress. The driven axis is held by giving the anisotropic
    barostat ``scale_axes`` with that entry False, which stops it undoing the
    strain while still reporting the pressure along it.

    The strain is applied by translating whole molecules, not by scaling atoms
    - see :func:`openmmpolymer.stress.scale_molecules` for why that is the
    only option with constraints on, and why it is also what makes the
    measured stress mean what it says.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: The temperature to deform at. Above the polymer's glass
            transition this measures a rubber, which is a real number about a
            different material state; nothing here knows which side of it you
            are on.
        pressure_bar: The pressure the two lateral axes are held at.
        axis: The axis to stretch, 0, 1 or 2.
        strain_increment: Engineering strain added per step.
        n_steps: How many increments to apply.
        relax_ps: Time to relax after each increment. The mean is over the
            second half, so this is twice the averaging window.
        strain_start: The strain the cell is already carrying. Non-zero when a
            ladder has been split across stages so it can be resumed.
        reference_box_nm: The unstrained cell edges, all three. None takes
            the current ones, which is right only when *strain_start* is
            zero - a resumed chunk starts from an already-stretched cell and
            would otherwise call that stretch the origin. All three are
            needed and not just the driven one, because the lateral
            contraction measured against them is Poisson's ratio.
        minimise_each_step: Minimise briefly after each increment. Off by
            default: rigid-molecule scaling at a strain increment of a couple
            of parts in a thousand does not create the overlaps that per-atom
            scaling would, and a minimisation between samples costs a
            thermalised cell its velocities.
        new_velocities: Draw fresh velocities rather than inheriting the
            starting state's. This is what makes replicas independent.
        samples_per_step: Stress readings per increment. The mean is over
            the second half of them. Worth setting high: the instantaneous
            pressure decorrelates in well under a picosecond, so readings
            spaced picoseconds apart leave most of the window's statistics
            unused, and each reading is only about six energy evaluations.
        barostat_frequency: Steps between lateral volume moves.
        timestep_fs: The integration timestep, or None for the longest safe
            one.
        friction_ps: Langevin friction.
        trajectory: Trajectory settings.
        report_interval_ps: Time between state-data rows.
        state_in: The previous stage's state.
        waypoints: Write a state at the end of every increment.

    Returns:
        What the stage did, with a strain, a stress tensor and the three box
        lengths recorded per increment.

    Raises:
        SimulationError: The cell blew up, or stretching it took an edge below
            what the cutoff allows.
        ValueError: The axis or the increment is not usable.
    """
    from .stress import affine_scale

    if axis not in (0, 1, 2):
        raise ValueError(f"axis={axis!r} must be 0, 1 or 2.")
    require_integer(n_steps, minimum=1, name="n_steps")
    require_integer(samples_per_step, minimum=1, name="samples_per_step")
    require_positive(relax_ps, None, name="relax_ps")
    require_positive(abs(strain_increment), None, name="strain_increment")

    prefix = Path(output_prefix)
    started = time.monotonic()
    if timestep_fs is None:
        timestep_fs = safe_timestep_fs(temperature_k, run.spec)
    check_timestep(timestep_fs, temperature_k, run.spec)

    scale_flags = [True, True, True]
    scale_flags[axis] = False
    simulation = _build_simulation(
        run,
        prefix.name,
        temperature_k=temperature_k,
        timestep_fs=timestep_fs,
        friction_ps=friction_ps,
        barostat="anisotropic",
        pressure_bar=pressure_bar,
        barostat_frequency=barostat_frequency,
        scale_axes=scale_flags,
    )
    _initialise(
        run,
        simulation,
        prefix.name,
        state_in,
        temperature_k,
        reuse_velocities=not new_velocities,
    )

    _check_molecules(prefix.name, simulation, run.box.n_molecules)
    cutoff_nm = nonbonded_cutoff_nm(simulation.system)
    reference = (
        _box_lengths_nm(simulation)
        if reference_box_nm is None
        else np.asarray([float(value) for value in reference_box_nm], dtype=np.float64)
    )
    if reference.shape != (3,) or not np.all(reference > 0.0):
        raise SimulationError(
            f"reference_box_nm={reference.tolist()} is not three positive cell edges."
        )

    steps_each = steps_for(relax_ps, timestep_fs)
    total_steps = steps_each * n_steps
    samples: dict[str, list[float]] = {
        "segment_strain": [],
        "segment_stress_xx_bar": [],
        "segment_stress_yy_bar": [],
        "segment_stress_zz_bar": [],
        "segment_box_x_nm": [],
        "segment_box_y_nm": [],
        "segment_box_z_nm": [],
        "segment_temperature_k": [],
        "segment_mean_temperature_k": [],
        "segment_density_g_cm3": [],
        "segment_duration_ps": [],
    }
    waypoint_paths: list[str] = []
    log.info(
        "%s: %d steps of %+.4f strain on axis %d, %.1f ps each at %.2f fs "
        "(%d steps), lateral %g bar, L0 = %.4f nm.",
        prefix.name,
        n_steps,
        strain_increment,
        axis,
        relax_ps,
        timestep_fs,
        total_steps,
        pressure_bar,
        reference[axis],
    )

    with reporting(
        simulation,
        prefix,
        total_steps=total_steps,
        report_interval=steps_for(report_interval_ps, timestep_fs),
        trajectory=trajectory,
        trajectory_interval=_frame_interval(trajectory, timestep_fs),
    ) as paths:
        strain = float(strain_start)
        for index in range(n_steps):
            factors = [1.0, 1.0, 1.0]
            factors[axis] = 1.0 + strain_increment
            vectors = simulation.context.getState().getPeriodicBoxVectors()
            scaled_vectors = [
                vector * factors[axis] if which == axis else vector
                for which, vector in enumerate(vectors)
            ]
            _apply_positions(
                simulation,
                affine_scale(_positions_nm(simulation), factors),
                scaled_vectors,
            )
            strain = (1.0 + strain) * (1.0 + strain_increment) - 1.0

            _check_deformed_box(prefix.name, simulation, cutoff_nm, strain)

            if minimise_each_step:
                simulation.minimizeEnergy(maxIterations=200)

            stress, error, density, realised = _sample_mechanics(
                simulation, steps_each, run.total_mass_g_mol, samples_per_step
            )
            edges = _box_lengths_nm(simulation)
            samples["segment_strain"].append(strain)
            for which, name in enumerate("xyz"):
                samples[f"segment_stress_{name}{name}_bar"].append(
                    float(stress[which, which])
                )
                samples[f"segment_box_{name}_nm"].append(float(edges[which]))
            samples["segment_temperature_k"].append(temperature_k)
            samples["segment_mean_temperature_k"].append(realised)
            samples["segment_density_g_cm3"].append(density)
            samples["segment_duration_ps"].append(relax_ps)
            if waypoints:
                waypoint_paths.append(
                    _save_waypoint(simulation, prefix, index, temperature_k)
                )
            _check_lateral(prefix.name, stress, error, axis, pressure_bar, strain)
            log.info(
                "  strain %+.4f: sigma_%s%s = %.1f +/- %.1f bar, %.4f g/cm3, "
                "ran at %.0f K.",
                strain,
                "xyz"[axis],
                "xyz"[axis],
                float(stress[axis, axis]),
                float(error[axis, axis]),
                density,
                realised,
            )
        state_path, pdb_path = _save_final(simulation, prefix)

    samples["reference_box_nm"] = reference.tolist()
    samples["deform_axis"] = [float(axis)]
    return StageResult(
        name=prefix.name,
        steps=total_steps,
        wall_seconds=time.monotonic() - started,
        final_state=state_path,
        temperature_k=temperature_k,
        mean_temperature_k=float(np.mean(samples["segment_mean_temperature_k"])),
        mean_density_g_cm3=samples["segment_density_g_cm3"][-1],
        final_pdb=pdb_path,
        csv=paths.csv,
        samples=samples,
        waypoints=tuple(waypoint_paths),
    )


def nonbonded_cutoff_nm(system: Any) -> float:
    """The longest nonbonded cutoff actually in the System, in nanometres.

    Read off the forces rather than taken from
    :class:`~openmmpolymer.mdsystem.SystemSpec`, because the two can differ:
    ``prepare_run`` accepts a ready-made System, and then the spec records
    what a System *would* have been built with rather than what this one has.
    A deformation is checked against the cutoff OpenMM will actually enforce.

    Args:
        system: The System to inspect.

    Returns:
        The cutoff in nanometres, or 0.0 when nothing in the System has one.
    """
    from openmm import unit

    cutoffs = []
    for force in system.getForces():
        getter = getattr(force, "getCutoffDistance", None)
        if getter is None:
            continue
        try:
            cutoffs.append(float(getter().value_in_unit(unit.nanometer)))
        except Exception:  # pragma: no cover - a force with no periodic cutoff
            continue
    return max(cutoffs, default=0.0)


def _check_deformed_box(
    name: str, simulation: Any, cutoff_nm: float, strain: float
) -> None:
    """Refuse a strain that has taken an edge below twice the cutoff.

    Stretching one axis contracts the other two, so a cell that was
    comfortable unstrained need not stay so. OpenMM refuses a cutoff over
    half the box, and it refuses it when the next force evaluation happens
    rather than when the box was set - which puts the error several steps
    away from the line that caused it unless something looks here first.

    Args:
        name: The stage name, for the message.
        simulation: The running simulation.
        cutoff_nm: The cutoff to check against.
        strain: The strain or applied stress reached, for the message.

    Raises:
        SimulationError: An edge is below twice the cutoff.
    """
    if cutoff_nm <= 0.0:
        return
    edges = _box_lengths_nm(simulation)
    shortest = float(edges.min())
    if shortest < 2.0 * cutoff_nm:
        raise SimulationError(
            f"{name}: at {strain:+.4g} the cell is "
            f"{edges.round(3).tolist()} nm, and its shortest edge is below "
            f"twice the {cutoff_nm:.2f} nm cutoff. Stretching one axis "
            "contracts the other two, so a cell that was comfortable "
            "unstrained need not stay so. Pack more chains, shorten the "
            "strain ladder, or lower the cutoff."
        )


def _check_molecules(name: str, simulation: Any, expected: int) -> None:
    """Say so when OpenMM groups the cell into a different number of molecules.

    The molecular virial is a sum over molecules, so if two chains ended up
    bonded to each other there are fewer of them than were packed and the
    stress is being computed over the wrong units. In the limit where
    everything is one molecule, a rigid volume move barely changes the
    energy and the reported pressure is close to meaningless. This package
    has seen packmol merge molecules before, so it is worth one line.
    """
    found = len(simulation.context.getMolecules())
    if expected and found != expected:
        log.warning(
            "%s: OpenMM groups this cell into %d molecules, but %d chains "
            "were packed. The stress is a sum over molecules, so if chains "
            "have become bonded to each other it is being computed over the "
            "wrong units.",
            name,
            found,
            expected,
        )


def _check_lateral(
    name: str,
    stress: npt.NDArray[np.float64],
    error: npt.NDArray[np.float64],
    axis: int,
    pressure_bar: float,
    strain: float,
) -> None:
    """Say so when the lateral axes are not at the pressure they were set to.

    Free, because the stress tensor is already in hand, and it checks the
    barostat and the pressure readout against each other at once: the two
    come from different places, and both being wrong in the same direction
    is unlikely.

    The threshold is the window's own standard error rather than a constant,
    because the instantaneous pressure of a small cell is enormously noisy
    and how noisy depends on the cell, the temperature and how long the
    window was. A fixed number is either never reached or reached every
    step; :data:`LATERAL_SIGMA_TOLERANCE` standard errors is the same
    statement at any of them.
    """
    lateral = [index for index in range(3) if index != axis]
    realised = [-float(stress[index, index]) for index in lateral]
    errors = [float(error[index, index]) for index in lateral]
    limits = [
        max(LATERAL_SIGMA_TOLERANCE * value, LATERAL_PRESSURE_FLOOR_BAR)
        if math.isfinite(value)
        else math.inf
        for value in errors
    ]
    if any(
        abs(value - pressure_bar) > limit
        for value, limit in zip(realised, limits, strict=True)
    ):
        log.warning(
            "%s at strain %+.4f: the lateral axes read %s bar (+/- %s) "
            "against a set point of %g. Either they have not relaxed at this "
            "strain, or the barostat is not acting on them.",
            name,
            strain,
            [round(value) for value in realised],
            [round(value) for value in errors],
            pressure_bar,
        )


def run_load(
    run: RunContext,
    output_prefix: str | Path = "07_load",
    *,
    temperature_k: float = 298.15,
    pressure_bar: float = 1.0,
    axis: int = 2,
    stresses_bar: Sequence[float] = (0.0, 100.0, 200.0, 300.0),
    duration_ps_each: float = 1000.0,
    new_velocities: bool = False,
    samples_per_step: int = 20,
    barostat_frequency: int = 25,
    timestep_fs: float | None = None,
    friction_ps: float = 1.0,
    trajectory: TrajectoryOptions | str = "none",
    report_interval_ps: float = 10.0,
    state_in: str | Path | None = None,
) -> StageResult:
    """Pull on one axis at a known stress and measure how far the cell gives.

    The mirror image of :func:`run_deform`, and worth having precisely
    because it shares none of its machinery. Here the stress is *imposed* -
    the barostat is set to ``pressure - sigma`` along the driven axis and to
    ``pressure`` on the other two - so the only thing measured is the box
    length, which is the one quantity OpenMM reports exactly. No virial is
    involved anywhere. Two methods that agree is evidence; one method that
    looks reasonable is not.

    What it costs is time. The cell's length is its slowest coordinate, so
    each rung needs long enough for the box to actually settle, and a rung
    that has not settled is a point on the wrong curve.
    :func:`openmmpolymer.elasticity.load_curve` checks that per rung rather
    than assuming it.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: The temperature to hold.
        pressure_bar: The pressure on the two lateral axes, and the baseline
            the applied stress is measured from.
        axis: The axis to pull on.
        stresses_bar: The applied tensile stresses to step through. Starting
            at zero gives the unstrained length the rest are measured
            against, in the same ensemble rather than a different one.
        duration_ps_each: Time at each rung.
        new_velocities: Draw fresh velocities rather than inheriting them.
        samples_per_step: Box readings per rung.
        barostat_frequency: Steps between volume moves.
        timestep_fs: The integration timestep, or None for the longest safe
            one.
        friction_ps: Langevin friction.
        trajectory: Trajectory settings.
        report_interval_ps: Time between state-data rows.
        state_in: The previous stage's state.

    Returns:
        What the stage did, with the applied stress and the three box lengths
        recorded per rung.

    Raises:
        ValueError: The axis is not 0, 1 or 2, or no stresses were given.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis={axis!r} must be 0, 1 or 2.")
    rungs = [float(value) for value in stresses_bar]
    if not rungs:
        raise ValueError("stresses_bar is empty, so there is nothing to pull with.")
    require_integer(samples_per_step, minimum=1, name="samples_per_step")
    require_positive(duration_ps_each, None, name="duration_ps_each")

    prefix = Path(output_prefix)
    started = time.monotonic()
    if timestep_fs is None:
        timestep_fs = safe_timestep_fs(temperature_k, run.spec)
    check_timestep(timestep_fs, temperature_k, run.spec)

    simulation = _build_simulation(
        run,
        prefix.name,
        temperature_k=temperature_k,
        timestep_fs=timestep_fs,
        friction_ps=friction_ps,
        barostat="anisotropic",
        pressure_bar=pressure_bar,
        barostat_frequency=barostat_frequency,
    )
    _initialise(
        run,
        simulation,
        prefix.name,
        state_in,
        temperature_k,
        reuse_velocities=not new_velocities,
    )

    steps_each = steps_for(duration_ps_each, timestep_fs)
    total_steps = steps_each * len(rungs)
    samples: dict[str, list[float]] = {
        "segment_applied_stress_bar": [],
        "segment_box_x_nm": [],
        "segment_box_y_nm": [],
        "segment_box_z_nm": [],
        "segment_temperature_k": [],
        "segment_mean_temperature_k": [],
        "segment_density_g_cm3": [],
        "segment_duration_ps": [],
    }
    log.info(
        "%s: %d applied stresses on axis %d, %.1f ps each at %.2f fs (%d steps).",
        prefix.name,
        len(rungs),
        axis,
        duration_ps_each,
        timestep_fs,
        total_steps,
    )

    with reporting(
        simulation,
        prefix,
        total_steps=total_steps,
        report_interval=steps_for(report_interval_ps, timestep_fs),
        trajectory=trajectory,
        trajectory_interval=_frame_interval(trajectory, timestep_fs),
    ) as paths:
        cutoff_nm = nonbonded_cutoff_nm(simulation.system)
        for applied in rungs:
            targets = [pressure_bar, pressure_bar, pressure_bar]
            # Tension is a negative pressure: pulling at sigma means holding
            # the axis below the ambient pressure by exactly that much.
            targets[axis] = pressure_bar - applied
            set_pressures(simulation, targets, "anisotropic")
            edges, density, realised = _sample_box(
                simulation, steps_each, run.total_mass_g_mol, samples_per_step
            )
            # A fluid cannot hold a deviatoric stress: rather than settling
            # at a new length it creeps, and keeps creeping until the cell
            # is thinner than the cutoff. Saying that here beats OpenMM
            # saying it several steps later about a box nobody set.
            _check_deformed_box(prefix.name, simulation, cutoff_nm, applied)
            samples["segment_applied_stress_bar"].append(applied)
            for which, name in enumerate("xyz"):
                samples[f"segment_box_{name}_nm"].append(float(edges[which]))
            samples["segment_temperature_k"].append(temperature_k)
            samples["segment_mean_temperature_k"].append(realised)
            samples["segment_density_g_cm3"].append(density)
            samples["segment_duration_ps"].append(duration_ps_each)
            log.info(
                "  %g bar: box %s nm, %.4f g/cm3, ran at %.0f K.",
                applied,
                edges.round(4).tolist(),
                density,
                realised,
            )
        state_path, pdb_path = _save_final(simulation, prefix)

    samples["load_axis"] = [float(axis)]
    return StageResult(
        name=prefix.name,
        steps=total_steps,
        wall_seconds=time.monotonic() - started,
        final_state=state_path,
        temperature_k=temperature_k,
        mean_temperature_k=float(np.mean(samples["segment_mean_temperature_k"])),
        mean_density_g_cm3=samples["segment_density_g_cm3"][-1],
        final_pdb=pdb_path,
        csv=paths.csv,
        samples=samples,
    )


def _sample_box(
    simulation: Any,
    steps: int,
    total_mass_g_mol: float,
    samples: int,
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Run one window, returning its mean box edges, density and temperature."""
    from openmm import unit

    chunk = max(1, steps // samples)
    edges: list[npt.NDArray[np.float64]] = []
    densities: list[float] = []
    temperatures: list[float] = []
    remaining = steps
    while remaining > 0:
        simulation.step(min(chunk, remaining))
        remaining -= chunk
        energy = (
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        if not np.isfinite(energy):
            raise SimulationError(
                "The potential energy went to NaN under load. The applied "
                "stress is large enough to be pulling the cell apart rather "
                "than straining it elastically."
            )
        edges.append(_box_lengths_nm(simulation))
        densities.append(density_g_cm3(simulation, total_mass_g_mol))
        temperatures.append(temperature_k_of(simulation))

    half = max(1, len(densities) // 2)
    return (
        np.mean(np.stack(edges[-half:]), axis=0),
        float(np.mean(densities[-half:])),
        float(np.mean(temperatures[-half:])),
    )


def run_shear(
    run: RunContext,
    output_prefix: str | Path = "09_shear",
    *,
    temperature_k: float = 298.15,
    strains: Sequence[float] = (0.005, 0.010, 0.015, 0.020),
    duration_ps_each: float = 200.0,
    plane: tuple[int, int] = (0, 2),
    new_velocities: bool = False,
    samples_per_step: int = 20,
    timestep_fs: float | None = None,
    friction_ps: float = 1.0,
    trajectory: TrajectoryOptions | str = "none",
    report_interval_ps: float = 10.0,
    state_in: str | Path | None = None,
) -> StageResult:
    """Shear the cell in steps, measuring the shear stress at each.

    Run at constant volume, and that is not a shortcut. An anisotropic
    barostat scales only the diagonal, so it would change ``cz`` underneath
    the measurement and with it the shear strain ``gamma = cx / cz``; a live
    flexible barostat would relax the shear away, which is the one thing
    being held. So the barostat here is a flexible one at ``frequency=0``: it
    never moves the box, and exists because ``computeCurrentPressure`` is a
    method on a barostat and OpenMM refuses it for a force that is not in the
    Context. It is the only barostat that reports off-diagonal components.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: The temperature to hold.
        strains: The shear strains to step through, ascending.
        duration_ps_each: Time held at each strain.
        plane: ``(driven, gradient)`` axes - the displaced direction and the
            direction it varies along.
        new_velocities: Draw fresh velocities rather than inheriting them.
        samples_per_step: Stress readings per strain.
        timestep_fs: The integration timestep, or None for the longest safe
            one.
        friction_ps: Langevin friction.
        trajectory: Trajectory settings.
        report_interval_ps: Time between state-data rows.
        state_in: The previous stage's state.

    Returns:
        What the stage did, with the shear strain and shear stress recorded
        per step.

    Raises:
        SimulationError: A strain would tilt the box past what OpenMM's
            reduced form allows.
        ValueError: The plane or the strain ladder is not usable.
    """
    from .stress import affine_shear, shear_box_vectors

    driven, gradient = plane
    if driven == gradient or not {driven, gradient} <= {0, 1, 2}:
        raise ValueError(f"plane={plane!r} must be two different axes of 0, 1, 2.")
    ladder = [float(value) for value in strains]
    if not ladder:
        raise ValueError("strains is empty, so there is nothing to shear.")
    require_integer(samples_per_step, minimum=1, name="samples_per_step")
    require_positive(duration_ps_each, None, name="duration_ps_each")

    prefix = Path(output_prefix)
    started = time.monotonic()
    if timestep_fs is None:
        timestep_fs = safe_timestep_fs(temperature_k, run.spec)
    check_timestep(timestep_fs, temperature_k, run.spec)

    simulation = _build_simulation(
        run,
        prefix.name,
        temperature_k=temperature_k,
        timestep_fs=timestep_fs,
        friction_ps=friction_ps,
        barostat="flexible",
        pressure_bar=1.0,
        barostat_frequency=0,
    )
    _initialise(
        run,
        simulation,
        prefix.name,
        state_in,
        temperature_k,
        reuse_velocities=not new_velocities,
    )

    _check_molecules(prefix.name, simulation, run.box.n_molecules)
    original = simulation.context.getState().getPeriodicBoxVectors()
    for gamma in ladder:
        # Checked against every rung before the first one runs, so a ladder
        # that cannot finish does not spend an hour finding out.
        shear_box_vectors(original, gamma, plane)

    steps_each = steps_for(duration_ps_each, timestep_fs)
    total_steps = steps_each * len(ladder)
    samples: dict[str, list[float]] = {
        "segment_shear_strain": [],
        "segment_shear_stress_bar": [],
        "segment_temperature_k": [],
        "segment_mean_temperature_k": [],
        "segment_density_g_cm3": [],
        "segment_duration_ps": [],
    }
    log.info(
        "%s: %d shear strains in the %s%s plane, %.1f ps each at %.2f fs "
        "(%d steps), at constant volume.",
        prefix.name,
        len(ladder),
        "xyz"[driven],
        "xyz"[gradient],
        duration_ps_each,
        timestep_fs,
        total_steps,
    )

    reference = _positions_nm(simulation)
    with reporting(
        simulation,
        prefix,
        total_steps=total_steps,
        report_interval=steps_for(report_interval_ps, timestep_fs),
        trajectory=trajectory,
        trajectory_interval=_frame_interval(trajectory, timestep_fs),
    ) as paths:
        for gamma in ladder:
            # Each strain is applied to the same starting configuration
            # rather than added to the last one, so a ladder is a set of
            # independent measurements of one cell rather than a history.
            _apply_positions(
                simulation,
                affine_shear(reference, gamma, plane),
                shear_box_vectors(original, gamma, plane),
            )
            stress, error, density, realised = _sample_mechanics(
                simulation, steps_each, run.total_mass_g_mol, samples_per_step
            )
            samples["segment_shear_strain"].append(gamma)
            samples["segment_shear_stress_bar"].append(float(stress[driven, gradient]))
            samples["segment_temperature_k"].append(temperature_k)
            samples["segment_mean_temperature_k"].append(realised)
            samples["segment_density_g_cm3"].append(density)
            samples["segment_duration_ps"].append(duration_ps_each)
            log.info(
                "  gamma %.4f: sigma_%s%s = %.1f +/- %.1f bar, ran at %.0f K.",
                gamma,
                "xyz"[driven],
                "xyz"[gradient],
                float(stress[driven, gradient]),
                float(error[driven, gradient]),
                realised,
            )
        state_path, pdb_path = _save_final(simulation, prefix)

    samples["shear_plane"] = [float(driven), float(gradient)]
    return StageResult(
        name=prefix.name,
        steps=total_steps,
        wall_seconds=time.monotonic() - started,
        final_state=state_path,
        temperature_k=temperature_k,
        mean_temperature_k=float(np.mean(samples["segment_mean_temperature_k"])),
        mean_density_g_cm3=samples["segment_density_g_cm3"][-1],
        final_pdb=pdb_path,
        csv=paths.csv,
        samples=samples,
    )


# --------------------------------------------------------------------------
# Stress relaxation
# --------------------------------------------------------------------------

#: The deformations a relaxation stage knows how to apply. Both measure the
#: shear relaxation modulus ``G(t)``: a shear step reads it straight off the
#: off-diagonal stress, and a tensile step reads it off the differential
#: stress, which for an isotropic solid is exactly ``2 G (e_axial -
#: e_lateral)`` with the Lame constant cancelling - see
#: :func:`openmmpolymer.stress.deviatoric_strain`. Young's relaxation modulus
#: is derived from it afterwards, with the material's own Poisson ratio.
RELAX_MODES = ("tensile", "shear")

#: How many standard errors the pre-strain deviatoric stress may sit from zero
#: before the stage says so. An equilibrated isotropic cell has none; one that
#: does is carrying a residual stress the measurement cannot undo.
BASELINE_SIGMA_TOLERANCE = 4.0

#: A floor under that, in bar, so a very quiet baseline does not warn about a
#: difference of no consequence.
BASELINE_FLOOR_BAR = 50.0


def relax_bin_edges_ps(
    first_sample_ps: float, total_ps: float, bins_per_decade: int
) -> npt.NDArray[np.float64]:
    """Return the logarithmic time grid a relaxation stage bins its stress into.

    Derived from the settings and never from the data, and that is the whole
    point of it. A relaxation modulus is read on a log time axis, so the
    readings have to be pooled into log-spaced bins somewhere; doing it here,
    from numbers every chunk and every replica were given, means they all land
    on the same grid. Their curves can then be merged bin by bin and averaged
    element by element. A grid fitted to whatever each chunk happened to
    sample would turn both of those into a regridding problem.

    The bins also do the averaging where it is needed. Early bins are narrow
    and hold a reading or two, which keeps the fast part of the decay
    resolved; late bins are wide and hold thousands, which is exactly where
    the stress has decayed towards a noise floor that only averaging gets
    through.

    Args:
        first_sample_ps: The earliest time a reading can land at, which is the
            sampling cadence.
        total_ps: The whole relaxation, not just this chunk of it.
        bins_per_decade: Bins per decade of time.

    Returns:
        The ``n + 1`` bin edges, ascending.

    Raises:
        ValueError: The span is not positive, or *bins_per_decade* is not.
    """
    require_positive(first_sample_ps, None, name="first_sample_ps")
    require_integer(bins_per_decade, minimum=1, name="bins_per_decade")
    if total_ps <= first_sample_ps:
        raise ValueError(
            f"total_ps={total_ps} is not above first_sample_ps="
            f"{first_sample_ps}, so there is no time axis to bin."
        )
    decades = math.log10(total_ps / first_sample_ps)
    count = max(1, round(bins_per_decade * decades))
    return np.geomspace(first_sample_ps, total_ps, count + 1)


class _LogBins:
    """Stress readings accumulated into fixed logarithmic time bins.

    Sums rather than running means, and the sum of squares beside them,
    because that is what merges. Two chunks of one relaxation can share a bin
    at the boundary between them, and a count-weighted mean of their means
    with a count-weighted mean of their mean-squares recovers exactly the
    numbers one unbroken run would have written. A stored standard error does
    not merge at all.
    """

    def __init__(self, edges_ps: npt.NDArray[np.float64]) -> None:
        """Set up empty bins between *edges_ps*."""
        self._edges = edges_ps
        count = edges_ps.size - 1
        self._n = np.zeros(count, dtype=np.int64)
        self._sum = np.zeros(count, dtype=np.float64)
        self._sum_sq = np.zeros(count, dtype=np.float64)
        self._sum_time = np.zeros(count, dtype=np.float64)
        self._diagonal = np.zeros((count, 3), dtype=np.float64)

    def add(
        self, time_ps: float, measure_bar: float, diagonal_bar: npt.NDArray[np.float64]
    ) -> None:
        """Add one reading, ignoring one that falls off either end of the grid.

        The top edge is the exception, and it is closed rather than open. A
        run ends exactly on it, and floating-point time accumulated a step at
        a time lands a hair either side, so an open edge would drop the very
        last reading of every stage - and the rule "every reading is in a bin"
        is worth more than the bin boundary being uniformly half-open.
        """
        index = int(np.searchsorted(self._edges, time_ps, side="right")) - 1
        if index == self._n.size and time_ps <= self._edges[-1] * (1.0 + 1.0e-9):
            index -= 1
        if index < 0 or index >= self._n.size:
            return
        self._n[index] += 1
        self._sum[index] += measure_bar
        self._sum_sq[index] += measure_bar * measure_bar
        self._sum_time[index] += time_ps
        self._diagonal[index] += diagonal_bar

    @property
    def populated(self) -> npt.NDArray[np.bool_]:
        """Which bins got at least one reading."""
        return self._n > 0

    def samples(self) -> dict[str, list[float]]:
        """The populated bins, as the lists a StageResult records.

        Means rather than sums, because a mean is the number a reader wants
        and the count is recorded beside it, so nothing is lost.
        """
        keep = self.populated
        counts = self._n[keep].astype(np.float64)
        recorded = {
            # The mean time of the readings in the bin, not its geometric
            # centre. They differ when a bin is not sampled uniformly across
            # its width - which is every bin the sampling cadence changes
            # inside, and every bin holding a single reading - and the mean
            # stress belongs to the mean time whatever the cadence did.
            "segment_relax_time_ps": self._sum_time[keep] / counts,
            # The index is the merge key. Two chunks of one relaxation share
            # a grid by construction, so adding them bin for bin is exact;
            # matching float times would be the same thing done fragilely.
            "segment_bin": np.flatnonzero(keep).astype(np.float64),
            "segment_stress_bar": self._sum[keep] / counts,
            "segment_stress_sq_bar2": self._sum_sq[keep] / counts,
            "segment_samples": counts,
        }
        for axis, name in enumerate("xyz"):
            recorded[f"segment_stress_{name}{name}_bar"] = (
                self._diagonal[keep, axis] / counts
            )
        return {
            key: [float(value) for value in values] for key, values in recorded.items()
        }


def _relax_measure(
    stress: npt.NDArray[np.float64], mode: str, axis: int, plane: tuple[int, int]
) -> float:
    """The one stress component a relaxation stage is watching.

    Tensile: the differential stress ``sigma_zz - (sigma_xx + sigma_yy) / 2``,
    which is what filters the isotropic background out of the decay.
    Shear: the off-diagonal component the step displaced.
    """
    from .stress import tensile_stress_bar

    if mode == "tensile":
        return tensile_stress_bar(stress, axis)
    return float(stress[plane[0], plane[1]])


def _strain_increment(
    simulation: Any,
    *,
    mode: str,
    axis: int,
    plane: tuple[int, int],
    increment: float,
    poisson: float,
) -> None:
    """Strain the cell by one more increment, affinely, from where it is now.

    Composes: applying this twice with half the strain each time leaves the
    cell where applying it once with the whole strain would, which is what
    lets a ramp be a loop. For the tensile case the three scale factors
    multiply; for the shear case the tilts add, because the gradient axis is
    the one the displacement is read off and the increment never touches it.
    """
    from .stress import affine_scale, affine_shear, shear_box_vectors

    vectors = simulation.context.getState().getPeriodicBoxVectors()
    if mode == "shear":
        _apply_positions(
            simulation,
            affine_shear(_positions_nm(simulation), increment, plane),
            shear_box_vectors(vectors, increment, plane),
        )
        return

    # Lateral contraction by (1 + e) ** -nu: at nu = 0.5 the volume is
    # preserved exactly, which is the incompressible limit a melt is usually
    # deformed in. The parameter exists because a glass is nearer 0.35, and
    # the differential stress this stage measures is deviatoric and so barely
    # notices the hydrostatic part either choice leaves behind.
    lateral = (1.0 + increment) ** -poisson
    factors = [lateral, lateral, lateral]
    factors[axis] = 1.0 + increment
    scaled = [vector * factors[which] for which, vector in enumerate(vectors)]
    _apply_positions(
        simulation, affine_scale(_positions_nm(simulation), factors), scaled
    )


def _check_baseline(name: str, mean_bar: float, error_bar: float, samples: int) -> None:
    """Say so when the cell was already carrying a deviatoric stress.

    The differential stress cancels an isotropic background - which is the
    whole reason a relaxation is read off it rather than off ``sigma_zz`` -
    but it cannot cancel a deviatoric one, and a cell frozen at an NPT
    snapshot can be carrying one. Everything downstream subtracts this mean,
    so a large one is not fatal; it is a warning that the cell the modulus
    belongs to was not the isotropic one it is supposed to be.

    The mean pressure is deliberately not checked. Locking the box at an NPT
    snapshot leaves it wherever that fluctuation happened to be, which is
    expected and harmless.
    """
    if samples < 2:
        return
    limit = max(BASELINE_SIGMA_TOLERANCE * error_bar, BASELINE_FLOOR_BAR)
    if abs(mean_bar) > limit:
        log.warning(
            "%s: before straining, the cell already carried a deviatoric "
            "stress of %.1f +/- %.1f bar over %d readings. It is subtracted, "
            "but a cell that is not isotropic to begin with is not the one "
            "this measurement assumes.",
            name,
            mean_bar,
            error_bar,
            samples,
        )


def run_relax(
    run: RunContext,
    output_prefix: str | Path = "06_relax",
    *,
    temperature_k: float = 298.15,
    mode: str = "tensile",
    axis: int = 2,
    plane: tuple[int, int] = (0, 2),
    step_strain: float = 0.03,
    poisson: float = 0.5,
    ramp_ps: float = 0.0,
    baseline_ps: float = 1000.0,
    duration_ps: float = 10_000.0,
    total_ps: float | None = None,
    time_offset_ps: float = 0.0,
    strain_applied: bool = False,
    reference_box_nm: Sequence[float] | None = None,
    sample_every_ps: float = 0.05,
    late_sample_every_ps: float = 5.0,
    late_after_ps: float = 200.0,
    bins_per_decade: int = 20,
    new_velocities: bool = False,
    timestep_fs: float | None = None,
    friction_ps: float = 1.0,
    trajectory: TrajectoryOptions | str = "none",
    report_interval_ps: float = 10.0,
    write_raw: bool = True,
    state_in: str | Path | None = None,
) -> StageResult:
    """Strain the cell once, hold it there, and watch the stress decay.

    The measurement a relaxation modulus is defined by. Everything else here
    deforms a cell and asks how hard it pushed back; this one deforms it once
    and asks how long it keeps pushing. What comes out is the shear relaxation
    modulus ``G(t)``, whichever step was applied, which is what a Prony series
    or a stretched exponential is fitted to. The stage records the strain to
    divide the stress by rather than the modulus itself, because turning one
    into the other is arithmetic and belongs in
    :mod:`openmmpolymer.relaxation` with the rest of it.

    The box is locked for the whole production run and a barostat is attached
    anyway, at ``frequency=0``. That is not a contradiction: the applied strain
    *is* the measurement, so nothing may relax it away, but OpenMM reports a
    pressure only through ``Barostat.computeCurrentPressure`` and refuses it
    for a force that is not in the Context. A barostat that never moves the box
    exists purely to be asked - the same trick :func:`run_shear` uses, and the
    one ``make_barostat`` documents ``frequency=0`` for.

    Before straining, the stage measures the stress it is about to strain from.
    The differential stress filters out an isotropic background but not a
    deviatoric one, and a cell frozen at an NPT snapshot can carry one. That
    baseline is recorded rather than subtracted: subtraction is arithmetic over
    recorded numbers, and that belongs in
    :mod:`openmmpolymer.relaxation` with the rest of it.

    Readings are pooled into logarithmic time bins as they are taken, so the
    cost is bounded and the fast part of the decay keeps its resolution while
    the slow part gets the heavy averaging it needs. The grid comes from the
    settings and not from the data, so every replica and every resumed chunk
    lands on the same bins - see :func:`relax_bin_edges_ps`.

    Args:
        run: The run context.
        output_prefix: Stem for this stage's files.
        temperature_k: The temperature to hold. Which side of the polymer's
            glass transition this falls on decides what is being measured, and
            nothing here knows which: below it the decay is local and fast,
            above it the chains themselves have to move and a 10 ns window
            sees the beginning of it at best.
        mode: ``"tensile"`` or ``"shear"``. Both measure ``G(t)``; a shear
            step does it without imposing a lateral contraction at all, which
            above the glass transition is the cleaner deformation.
        axis: The axis to stretch, for a tensile step.
        plane: ``(driven, gradient)`` axes, for a shear step.
        step_strain: The strain applied, all at once. Small enough to stay
            inside the linear viscoelastic region, where the modulus does not
            depend on it - which nothing here can check from one run, and
            which ``linearity_strains`` on a
            :class:`~openmmpolymer.viscoelastic.RelaxationSpec` exists to test.
        poisson: The lateral contraction, as ``(1 + e) ** -poisson``. The
            default of 0.5 preserves the volume exactly, which is the usual
            assumption for a melt.
        ramp_ps: Apply the strain over this long instead of instantaneously.
            Zero by default, because an instantaneous step is what ``E(t)`` is
            defined against; a short ramp trades a sharper time origin for a
            gentler perturbation. The relaxation clock starts when the ramp
            ends either way.
        baseline_ps: Time held at the locked box before straining, to measure
            what the cell was already carrying.
        duration_ps: How much relaxation *this stage* runs.
        total_ps: The whole relaxation the time grid spans, when this stage is
            one chunk of a longer one. None means *duration_ps*.
        time_offset_ps: Where this chunk starts on the relaxation clock. A
            resumed chunk is handed this because a state file does not carry
            it, and a chunk that called its own start time zero would fold the
            late part of the decay back onto the early part.
        strain_applied: This chunk opens an already-strained cell, so it
            neither measures a baseline nor strains again.
        reference_box_nm: The unstrained cell edges. None reads them from the
            cell, which is right only when the strain has not been applied yet.
        sample_every_ps: Time between stress readings, early on. Each reading
            costs OpenMM about six energy evaluations, so this is the knob that
            sets the stage's overhead.
        late_sample_every_ps: Time between readings after *late_after_ps*. The
            decay is slow by then and the bins are wide, so a dense cadence
            buys very little.
        late_after_ps: When to change down.
        bins_per_decade: Logarithmic bins per decade of time.
        new_velocities: Draw fresh velocities rather than inheriting them.
            This is what makes replicas of one configuration independent.
        timestep_fs: The integration timestep, or None for the longest safe one.
        friction_ps: Langevin friction.
        trajectory: Trajectory settings.
        report_interval_ps: Time between state-data rows.
        write_raw: Write every reading to ``<stem>_stress.csv`` beside the
            binned curve, so it can be re-binned or analysed some other way
            without running the whole thing again.
        state_in: The previous stage's state.

    Returns:
        What the stage did, with the binned decay, the baseline and the
        deformation recorded in its samples.

    Raises:
        SimulationError: The cell blew up, or the strain took an edge below
            what the cutoff allows.
        ValueError: The mode, the axes or one of the times is not usable.
    """
    from .stress import deviatoric_strain

    require_choice(mode, RELAX_MODES, name="mode")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis={axis!r} must be 0, 1 or 2.")
    driven, gradient = plane
    if driven == gradient or not {driven, gradient} <= {0, 1, 2}:
        raise ValueError(f"plane={plane!r} must be two different axes of 0, 1, 2.")
    require_positive(duration_ps, None, name="duration_ps")
    require_positive(sample_every_ps, None, name="sample_every_ps")
    require_positive(late_sample_every_ps, None, name="late_sample_every_ps")
    require_positive(abs(step_strain), None, name="step_strain")
    if baseline_ps < 0.0 or ramp_ps < 0.0 or time_offset_ps < 0.0:
        raise ValueError(
            f"baseline_ps={baseline_ps}, ramp_ps={ramp_ps} and "
            f"time_offset_ps={time_offset_ps} must all be zero or more."
        )

    prefix = Path(output_prefix)
    started = time.monotonic()
    if timestep_fs is None:
        timestep_fs = safe_timestep_fs(temperature_k, run.spec)
    check_timestep(timestep_fs, temperature_k, run.spec)

    # A shear step measures G directly: sigma_xz = G gamma. A tensile step
    # measures it through the deviator, which is why the factor of two and
    # the lateral strain are here and no assumption about the material is.
    measure_strain = (
        float(step_strain)
        if mode == "shear"
        else 2.0 * deviatoric_strain(step_strain, poisson)
    )
    span_ps = duration_ps if total_ps is None else float(total_ps)
    edges = relax_bin_edges_ps(sample_every_ps, span_ps, bins_per_decade)
    bins = _LogBins(edges)

    # Anisotropic reports the diagonal, which is what a tensile step needs;
    # only the flexible one reports shear. Neither moves anything at
    # frequency=0 - they are here to be asked, not to hold a pressure.
    simulation = _build_simulation(
        run,
        prefix.name,
        temperature_k=temperature_k,
        timestep_fs=timestep_fs,
        friction_ps=friction_ps,
        barostat="anisotropic" if mode == "tensile" else "flexible",
        pressure_bar=1.0,
        barostat_frequency=0,
    )
    _initialise(
        run,
        simulation,
        prefix.name,
        state_in,
        temperature_k,
        reuse_velocities=not new_velocities,
    )
    _check_molecules(prefix.name, simulation, run.box.n_molecules)
    cutoff_nm = nonbonded_cutoff_nm(simulation.system)
    reference = (
        _box_lengths_nm(simulation)
        if reference_box_nm is None
        else np.asarray([float(value) for value in reference_box_nm], dtype=np.float64)
    )
    if reference.shape != (3,) or not np.all(reference > 0.0):
        raise SimulationError(
            f"reference_box_nm={reference.tolist()} is not three positive cell edges."
        )

    # Guarded rather than left to steps_for, which floors at one step: a
    # baseline of zero or an instantaneous strain would otherwise each run a
    # single step of dynamics, and the second of those puts the relaxation
    # clock's origin one step after the strain it is supposed to start at.
    skip = strain_applied
    baseline_steps = (
        0 if skip or baseline_ps <= 0.0 else steps_for(baseline_ps, timestep_fs)
    )
    ramp_steps = 0 if skip or ramp_ps <= 0.0 else steps_for(ramp_ps, timestep_fs)
    production_steps = steps_for(duration_ps, timestep_fs)
    total_steps = baseline_steps + ramp_steps + production_steps
    log.info(
        "%s: %s step of %+.4f%s, %.1f ps baseline then %.1f ps held at a "
        "locked box from t = %.1f ps, at %.2f fs (%d steps), into %d log bins.",
        prefix.name,
        mode,
        step_strain,
        ""
        if strain_applied
        else f" on {'xyz'[axis] if mode == 'tensile' else ''.join('xyz'[i] for i in plane)}",
        0.0 if strain_applied else baseline_ps,
        duration_ps,
        time_offset_ps,
        timestep_fs,
        total_steps,
        edges.size - 1,
    )
    from .stress import stress_tensor_bar

    instant_bar: float | None = None

    def read() -> tuple[npt.NDArray[np.float64], float]:
        """One reading: the diagonal, and the component being watched.

        Finiteness is checked here rather than by reading the potential
        energy separately, as the windowed stages do. A cell that has blown
        up reports a non-finite stress, and the stress is already in hand, so
        the check costs nothing. The off-diagonals an anisotropic barostat
        leaves NaN are deliberately not looked at.
        """
        stress = stress_tensor_bar(simulation)
        diagonal = np.asarray(
            [stress[index, index] for index in range(3)], dtype=np.float64
        )
        measure = _relax_measure(stress, mode, axis, plane)
        if not (bool(np.all(np.isfinite(diagonal))) and math.isfinite(measure)):
            raise SimulationError(
                f"{prefix.name}: the stress went to NaN. The step strain is "
                "too large for this cell, or the timestep is too long for "
                "this temperature. Lower step_strain, or ramp it in over "
                "ramp_ps instead of applying it at once."
            )
        return diagonal, measure

    raw_path = prefix.parent / f"{prefix.name}_stress.csv"
    baseline_n = 0
    baseline_sum = 0.0
    baseline_sum_sq = 0.0
    temperatures: list[float] = []
    elapsed_ps = float(time_offset_ps)
    ran_steps = 0

    with ExitStack() as stack:
        paths = stack.enter_context(
            reporting(
                simulation,
                prefix,
                total_steps=max(1, total_steps),
                report_interval=steps_for(report_interval_ps, timestep_fs),
                trajectory=trajectory,
                trajectory_interval=_frame_interval(trajectory, timestep_fs),
            )
        )
        raw = None
        if write_raw:
            rotate_existing(raw_path)
            raw = stack.enter_context(raw_path.open("w"))
            raw.write("time_ps,sigma_xx_bar,sigma_yy_bar,sigma_zz_bar,sigma_bar\n")

        chunk = max(1, steps_for(sample_every_ps, timestep_fs))
        remaining = baseline_steps
        while remaining > 0:
            taken = min(chunk, remaining)
            simulation.step(taken)
            remaining -= taken
            ran_steps += taken
            _, measure = read()
            baseline_n += 1
            baseline_sum += measure
            baseline_sum_sq += measure * measure

        baseline_mean = baseline_sum / baseline_n if baseline_n else 0.0
        baseline_error = 0.0
        if baseline_n > 1:
            variance = max(0.0, baseline_sum_sq / baseline_n - baseline_mean**2)
            baseline_error = math.sqrt(variance / baseline_n)
        _check_baseline(prefix.name, baseline_mean, baseline_error, baseline_n)

        if not strain_applied:
            # Composed from increments so that a ramp and a step are the same
            # code path. Engineering strain compounds, so a tensile increment
            # is the root of one plus the total; a shear tilt simply adds.
            increments = max(1, round(ramp_ps / sample_every_ps)) if ramp_ps > 0 else 1
            each = (
                step_strain / increments
                if mode == "shear"
                else (1.0 + step_strain) ** (1.0 / increments) - 1.0
            )
            per_increment = ramp_steps // increments
            for _ in range(increments):
                _strain_increment(
                    simulation,
                    mode=mode,
                    axis=axis,
                    plane=plane,
                    increment=each,
                    poisson=poisson,
                )
                if per_increment:
                    simulation.step(per_increment)
                    ran_steps += per_increment
            _check_deformed_box(prefix.name, simulation, cutoff_nm, step_strain)
            # The response before anything has moved: the affine part of the
            # modulus, and the one point of the decay no amount of dynamics
            # can give back, since every later reading is already relaxing.
            # It has no time to be binned at, so it is recorded on its own.
            _, instant_bar = read()

        # The box is locked from here, so the density cannot change and is
        # worth one reading rather than one per sample.
        density = density_g_cm3(simulation, run.total_mass_g_mol)
        locked = _box_lengths_nm(simulation)

        remaining = production_steps
        since_temperature = 0
        rows = 0
        while remaining > 0:
            cadence = (
                sample_every_ps if elapsed_ps < late_after_ps else late_sample_every_ps
            )
            taken = min(max(1, steps_for(cadence, timestep_fs)), remaining)
            simulation.step(taken)
            remaining -= taken
            ran_steps += taken
            elapsed_ps += taken * timestep_fs / 1000.0
            diagonal, measure = read()
            bins.add(elapsed_ps, measure, diagonal)
            if raw is not None:
                raw.write(
                    f"{elapsed_ps:.8g},{diagonal[0]:.8g},{diagonal[1]:.8g},"
                    f"{diagonal[2]:.8g},{measure:.8g}\n"
                )
                rows += 1
                if rows % 1000 == 0:
                    # Flushed as it goes: the run this matters for is the one
                    # that gets killed, and a buffer is lost with the process.
                    raw.flush()
            since_temperature += 1
            if since_temperature >= 50:
                temperatures.append(temperature_k_of(simulation))
                since_temperature = 0

        if not temperatures:
            temperatures.append(temperature_k_of(simulation))
        moved = _box_lengths_nm(simulation)
        if not np.allclose(moved, locked, rtol=0.0, atol=1.0e-9):
            raise SimulationError(
                f"{prefix.name}: the cell moved from {locked.round(6).tolist()} "
                f"to {moved.round(6).tolist()} nm during the hold. The strain "
                "is the measurement, so a box that relaxes is a barostat that "
                "is not at frequency=0."
            )
        state_path, pdb_path = _save_final(simulation, prefix)

    samples = bins.samples()
    samples["segment_duration_ps"] = [float(duration_ps)]
    # What the modulus is divided by. Recorded rather than left to be worked
    # out downstream, because it is the one number that says what the stress
    # means: the differential stress of an isotropic solid is exactly
    # 2 G (e_axial - e_lateral), with the Lame constant cancelling, so
    # dividing by this gives G whatever Poisson's ratio the step imposed and
    # whatever the material's own turns out to be.
    samples["relax_strain_measure"] = [measure_strain]
    samples["relax_volume_ratio"] = [
        float(np.prod(locked) / np.prod(reference)) if not strain_applied else math.nan
    ]
    samples["relax_window_ps"] = [float(time_offset_ps), float(elapsed_ps)]
    samples["step_strain"] = [float(step_strain)]
    samples["relax_ramp_ps"] = [float(ramp_ps)]
    samples["reference_box_nm"] = [float(value) for value in reference]
    # Which key is here says which deformation ran, the way a stage's kind is
    # everywhere else read off what it recorded rather than off its name.
    if mode == "shear":
        samples["relax_plane"] = [float(driven), float(gradient)]
    else:
        samples["relax_axis"] = [float(axis)]
        samples["relax_poisson"] = [float(poisson)]
    if instant_bar is not None:
        samples["instant_stress_bar"] = [instant_bar]
    if baseline_n:
        samples["baseline_stress_bar"] = [baseline_mean]
        samples["baseline_stress_sq_bar2"] = [baseline_sum_sq / baseline_n]
        samples["baseline_samples"] = [float(baseline_n)]

    realised = float(np.mean(temperatures))
    log.info(
        "  %d of %d bins filled from %.4g to %.4g ps, ran at %.0f K, %.4f g/cm3%s.",
        int(np.count_nonzero(bins.populated)),
        edges.size - 1,
        float(edges[0]),
        float(edges[-1]),
        realised,
        density,
        f", baseline {baseline_mean:+.1f} +/- {baseline_error:.1f} bar"
        if baseline_n
        else "",
    )
    return StageResult(
        name=prefix.name,
        steps=ran_steps,
        wall_seconds=time.monotonic() - started,
        final_state=state_path,
        temperature_k=temperature_k,
        mean_temperature_k=realised,
        mean_density_g_cm3=density,
        final_pdb=pdb_path,
        csv=paths.csv,
        samples=samples,
    )
