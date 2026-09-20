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

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._seeds import derive_seed, seed_random_stream
from ._validation import require_positive
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
from .reporters import TrajectoryOptions, reporting, steps_for

log = logging.getLogger(__name__)

#: Barostat pressure parameters, per kind. The anisotropic barostat needs three.
_PRESSURE_PARAMETERS = {
    "isotropic": ("MonteCarloPressure",),
    "anisotropic": (
        "MonteCarloPressureX",
        "MonteCarloPressureY",
        "MonteCarloPressureZ",
    ),
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
) -> None:
    """Put the simulation at its starting point.

    Positions, velocities and box vectors are transferred individually rather
    than through ``loadState``. A state saved under NPT carries the barostat's
    global parameters, and setting one of those on a Context that has no
    barostat raises - which would make every NPT-to-NVT transition a failure.
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
    simulation.minimizeEnergy(
        tolerance=tolerance_kj_per_nm * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )
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

    Returns:
        What the stage did, including a density and a temperature per segment.

    Raises:
        SimulationError: The run produced a non-finite energy.
        ValueError: The timestep is too long for the hottest segment.
    """
    if not segments:
        raise ValueError(f"Stage {name!r} has no segments to run.")

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

    samples: dict[str, list[float]] = {
        "segment_temperature_k": [],
        "segment_density_g_cm3": [],
        "segment_mean_temperature_k": [],
    }
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
    ) as paths:
        for segment, steps in zip(segments, per_segment, strict=True):
            set_temperature(simulation, segment.temperature_k, barostat)
            if barostat is not None:
                set_pressure(simulation, segment.pressure_bar, barostat)
            density, temperature = _run_segment(simulation, steps, run.total_mass_g_mol)
            samples["segment_temperature_k"].append(segment.temperature_k)
            samples["segment_density_g_cm3"].append(density)
            samples["segment_mean_temperature_k"].append(temperature)
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
    )


def _run_segment(
    simulation: Any, steps: int, total_mass_g_mol: float
) -> tuple[float, float]:
    """Run one segment, returning its settled density and temperature.

    Sampled in chunks and averaged over the second half, so a segment that
    spends its first part relaxing does not drag its own average.
    """
    from openmm import unit

    chunk = max(1, steps // _SAMPLES_PER_SEGMENT)
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
        **kwargs: Passed to :func:`run_segments`.

    Returns:
        What the stage did. ``samples`` carries the temperatures and the
        densities they settled at.

    Raises:
        ValueError: The ramp does not descend.
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
