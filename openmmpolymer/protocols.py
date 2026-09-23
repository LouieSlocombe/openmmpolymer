"""Named stage sequences, and the runner that sees one through.

A polymer melt run is days of wall time in a queue that will interrupt it, so
the runner's real job is not ordering the stages - that part is a list - but
being able to pick the run up again. Every stage writes a portable serialised
state and the manifest records which stages finished; a resumed run skips those
and starts the next one from the last state written.

The manifest is also where the run says what it actually did, as opposed to
what it was asked to do: the temperature each stage ran at, the density it
settled to, the timestep that was used after derating, and the chain dimensions
at the end. That last one is the honest part. packmol places chains that do not
interpenetrate, the Rouse time of a melt is tens of nanoseconds, and no
protocol here runs for that long, so the measurement is reported and the
interpretation is left to whoever reads it.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .reporters import TrajectoryOptions
from .simulate import (
    RunContext,
    StageResult,
    heating_temperatures,
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
    run_shear,
)

log = logging.getLogger(__name__)

#: The stage kinds a protocol may name, and what runs them. Typed loosely on
#: purpose: the runners differ in their keyword arguments, and a Stage's
#: options are validated by the runner it names rather than here.
STAGE_RUNNERS: dict[str, Callable[..., StageResult]] = {
    "minimise": run_minimise,
    "pushoff": run_pushoff,
    "nvt": run_nvt,
    "npt": run_npt,
    "compress": run_compress,
    "anneal": run_anneal,
    "quench": run_quench,
    "heat": run_heat,
    "production": run_production,
    "deform": run_deform,
    "load": run_load,
    "shear": run_shear,
    "relax": run_relax,
}

#: What the manifest is called.
MANIFEST_NAME = "manifest.json"

#: How far the measured chain dimensions may sit from the expected ones before
#: the run stops claiming they agree.
CHAIN_DIMENSION_TOLERANCE = 0.25


class ProtocolError(RuntimeError):
    """A protocol could not be run."""


@dataclass(frozen=True)
class Stage:
    """One step of a protocol.

    Args:
        name: What this stage is called, and the stem of everything it writes.
            Numbered, so the run directory sorts into the order it ran.
        kind: One of the keys of :data:`STAGE_RUNNERS`.
        options: Keyword arguments for that runner.
    """

    name: str
    kind: str
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a stage kind nothing can run."""
        if self.kind not in STAGE_RUNNERS:
            raise ValueError(
                f"Stage {self.name!r} has kind {self.kind!r}; it must be one "
                f"of {', '.join(sorted(STAGE_RUNNERS))}."
            )


@dataclass(frozen=True)
class Protocol:
    """An ordered list of stages.

    Args:
        name: What the protocol is called.
        stages: What to run, in order.
    """

    name: str
    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        """Reject an empty protocol or a repeated stage name."""
        if not self.stages:
            raise ValueError(f"Protocol {self.name!r} has no stages.")
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Protocol {self.name!r} repeats a stage name; each one names "
                "the files it writes, so they have to differ."
            )

    @property
    def total_duration_ps(self) -> float:
        """How much dynamics this protocol asks for, in picoseconds.

        Approximate on purpose - it is for telling a user whether they asked
        for nanoseconds or microseconds, not for scheduling - but it counts
        every stage. Three kinds state their time as a ladder rather than a
        duration, and a budget that silently omits the most expensive stage in
        a protocol is worse than no budget, because it gets believed.
        """
        return sum(_stage_duration_ps(stage) for stage in self.stages)


def _stage_options(stage: Stage) -> dict[str, Any]:
    """A stage's options, filled in with its runner's own defaults.

    Read off the runner's signature rather than repeated here, so the numbers
    a cost estimate is built from cannot drift away from the numbers the run
    actually uses.
    """
    parameters = inspect.signature(STAGE_RUNNERS[stage.kind]).parameters
    defaults = {
        name: parameter.default
        for name, parameter in parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    return {**defaults, **stage.options}


def _stage_duration_ps(stage: Stage) -> float:
    """How much dynamics one stage asks for, from the options it was given."""
    options = _stage_options(stage)
    if stage.kind == "heat":
        ladder = options.get("temperatures_k")
        if ladder is None:
            ladder = heating_temperatures(
                float(options["t_start"]),
                float(options["t_end"]),
                float(options["step_k"]),
            )
        return float(options["hold_ps"]) * len(ladder)
    if stage.kind == "quench":
        ladder = options.get("temperatures_k") or quench_temperatures(
            float(options["t_start"]),
            float(options["t_end"]),
            float(options["step_k"]),
        )
        return float(options["hold_ps"]) * len(ladder)
    if stage.kind == "anneal":
        ramp_ps = int(options["ramp_windows"]) * float(options["window_ps"])
        return int(options["n_cycles"]) * 2.0 * (ramp_ps + float(options["hold_ps"]))
    if stage.kind == "compress":
        return float(options["duration_ps_each"]) * len(options["pressures_bar"])
    if stage.kind == "deform":
        return float(options["relax_ps"]) * int(options["n_steps"])
    if stage.kind == "load":
        return float(options["duration_ps_each"]) * len(options["stresses_bar"])
    if stage.kind == "shear":
        return float(options["duration_ps_each"]) * len(options["strains"])
    if stage.kind == "relax":
        # A chunk that opens an already-strained cell repeats neither the
        # baseline nor the ramp, so counting them would price a resumed
        # ladder as several first chunks.
        held = 0.0 if options["strain_applied"] else float(options["baseline_ps"])
        ramp = 0.0 if options["strain_applied"] else float(options["ramp_ps"])
        return held + ramp + float(options["duration_ps"])
    duration = options.get("duration_ps")
    return 0.0 if duration is None else float(duration)


def standard_melt_equilibration(
    *,
    target_temperature_k: float = 450.0,
    melt_temperature_k: float = 600.0,
    pressure_bar: float = 1.0,
    nvt_ps: float = 500.0,
    compress_ps_each: float = 100.0,
    npt_ps: float = 2000.0,
    anneal_cycles: int = 3,
    anneal_t_low_k: float | None = None,
    anneal_window_ps: float = 20.0,
    anneal_hold_ps: float = 50.0,
    compress_pressures_bar: Sequence[float] | None = None,
    npt_trajectory: TrajectoryOptions | str = "none",
) -> Protocol:
    """The default recipe: from a packed cell to an equilibrated melt.

    Minimise, relieve the packing, randomise at high temperature, compress to a
    melt density, anneal, then settle at the temperature of interest.

    Args:
        target_temperature_k: Where the run ends up.
        melt_temperature_k: The temperature the chains are mobilised at.
        pressure_bar: The pressure held once there is a barostat.
        nvt_ps: Time spent at constant volume at the melt temperature.
        compress_ps_each: Time at each rung of the pressure ladder.
        npt_ps: Time spent settling at constant pressure.
        anneal_cycles: How many melt-and-set cycles.
        anneal_window_ps: Time at each step of an annealing ramp.
        anneal_hold_ps: Time at the top and bottom of each cycle.
        anneal_t_low_k: The bottom of each annealing cycle, when it should not
            be *target_temperature_k*. A run that settles at the temperature
            it will start cooling from needs the two separated, or the anneal
            has nothing to cycle between.
        compress_pressures_bar: The pressure ladder, when
            :data:`~openmmpolymer.simulate.DEFAULT_COMPRESSION_BAR` is wrong
            for this cell. A kilobar squeezes a sparse cell past twice the
            nonbonded cutoff, which OpenMM refuses outright.
        npt_trajectory: Trajectory settings for the final NPT stage. ``"none"``
            by default, which is what every run has always written. A melt
            cannot be shown to have equilibrated without one - the chains'
            mean-squared displacement is the only evidence that says so - but
            it is frames of the whole cell, so it is asked for rather than
            assumed.

    Returns:
        The protocol.
    """
    return Protocol(
        name="standard_melt_equilibration",
        stages=(
            Stage("00_minimise", "minimise"),
            Stage("01_pushoff", "pushoff", {"temperature_k": 300.0}),
            Stage(
                "02_nvt",
                "nvt",
                {"temperature_k": melt_temperature_k, "duration_ps": nvt_ps},
            ),
            Stage(
                "03_compress",
                "compress",
                {
                    "temperature_k": melt_temperature_k,
                    "duration_ps_each": compress_ps_each,
                    **(
                        {}
                        if compress_pressures_bar is None
                        else {"pressures_bar": tuple(compress_pressures_bar)}
                    ),
                },
            ),
            Stage(
                "04_anneal",
                "anneal",
                {
                    "t_low": (
                        target_temperature_k
                        if anneal_t_low_k is None
                        else anneal_t_low_k
                    ),
                    "t_high": melt_temperature_k,
                    "n_cycles": anneal_cycles,
                    "window_ps": anneal_window_ps,
                    "hold_ps": anneal_hold_ps,
                    "pressure_bar": pressure_bar,
                },
            ),
            Stage(
                "05_npt",
                "npt",
                {
                    "temperature_k": target_temperature_k,
                    "pressure_bar": pressure_bar,
                    "duration_ps": npt_ps,
                    "trajectory": npt_trajectory,
                },
            ),
        ),
    )


def melt_quench(
    *,
    melt_temperature_k: float = 600.0,
    t_start: float | None = None,
    t_end: float = 200.0,
    step_k: float = 20.0,
    hold_ps: float = 200.0,
    pressure_bar: float = 1.0,
    **equilibration: Any,
) -> Protocol:
    """Equilibrate the melt, then cool it in steps.

    The quench records a density at every temperature on the way down, which is
    the specific-volume curve a glass transition is read off. Cooling rates
    here are many orders of magnitude faster than any experiment, so the
    transition comes out well above the measured one; the shape is what is
    worth looking at.

    Args:
        melt_temperature_k: The temperature the chains are mobilised at, and
            where cooling starts unless *t_start* says otherwise.
        t_start: Where the cooling starts, when that should not be the melt
            temperature.
        t_end: Where it stops.
        step_k: How far it drops at each step.
        hold_ps: Time held at each temperature.
        pressure_bar: The pressure held throughout.
        **equilibration: Passed to :func:`standard_melt_equilibration`.

    Returns:
        The protocol.
    """
    base = standard_melt_equilibration(
        melt_temperature_k=melt_temperature_k,
        pressure_bar=pressure_bar,
        **equilibration,
    )
    return Protocol(
        name="melt_quench",
        stages=(
            *base.stages,
            Stage(
                "06_quench",
                "quench",
                {
                    "t_start": (melt_temperature_k if t_start is None else t_start),
                    "t_end": t_end,
                    "step_k": step_k,
                    "hold_ps": hold_ps,
                    "pressure_bar": pressure_bar,
                },
            ),
        ),
    )


@dataclass(frozen=True)
class ChainDimensions:
    """What the chains in a cell actually look like.

    Args:
        mean_squared_end_to_end_nm2: Mean over molecules.
        mean_radius_of_gyration_nm: Mean over molecules.
        ratio_of_squares: <R^2> / <Rg^2>. An ideal chain gives six; well below
            it means the chains are still collapsed.
        characteristic_ratio: <R^2> / (n l^2), measured.
        expected_characteristic_ratio: What the polymer's should be.
        consistent: Whether the two agree within
            :data:`CHAIN_DIMENSION_TOLERANCE`. This is a snapshot, so it says
            the dimensions are consistent with an equilibrated melt, not that
            the melt is equilibrated - nothing measurable in one frame can say
            that.
    """

    mean_squared_end_to_end_nm2: float
    mean_radius_of_gyration_nm: float
    ratio_of_squares: float
    characteristic_ratio: float
    expected_characteristic_ratio: float
    consistent: bool


def chain_dimensions(
    positions_nm: npt.NDArray[np.float64],
    backbone: Sequence[int],
    atoms_per_chain: int,
    n_chains: int,
    *,
    expected_characteristic_ratio: float,
    masses: npt.NDArray[np.float64] | None = None,
) -> ChainDimensions:
    """Measure the chain dimensions in a packed cell.

    Every molecule is a copy of the same chain, so one backbone path shifted by
    the chain's atom count covers all of them.

    Args:
        positions_nm: The cell's positions, unwrapped.
        backbone: Backbone atom indices within one chain, in order.
        atoms_per_chain: Atoms per molecule.
        n_chains: How many molecules.
        expected_characteristic_ratio: The polymer's C-infinity.
        masses: Per-atom masses within one chain, for the radius of gyration.
            Unweighted when None.

    Returns:
        The measurement.
    """
    path = np.asarray(backbone, dtype=int)
    squares: list[float] = []
    radii: list[float] = []
    bonds: list[float] = []

    for index in range(n_chains):
        offset = index * atoms_per_chain
        chain = positions_nm[offset : offset + atoms_per_chain]
        ends = chain[path[-1]] - chain[path[0]]
        squares.append(float(ends @ ends))

        weights = np.ones(len(chain)) if masses is None else masses
        centre = (weights[:, None] * chain).sum(axis=0) / weights.sum()
        offsets = chain - centre
        radii.append(
            float(
                np.sqrt(
                    (weights * (offsets * offsets).sum(axis=1)).sum() / weights.sum()
                )
            )
        )

        steps = chain[path[1:]] - chain[path[:-1]]
        bonds.append(float(np.sqrt((steps * steps).sum(axis=1)).mean()))

    mean_square = float(np.mean(squares))
    mean_radius = float(np.mean(radii))
    bond_length = float(np.mean(bonds))
    measured = mean_square / ((len(path) - 1) * bond_length**2)
    drift = (
        abs(measured - expected_characteristic_ratio) / expected_characteristic_ratio
    )
    return ChainDimensions(
        mean_squared_end_to_end_nm2=mean_square,
        mean_radius_of_gyration_nm=mean_radius,
        ratio_of_squares=mean_square / float(np.mean(np.square(radii))),
        characteristic_ratio=measured,
        expected_characteristic_ratio=expected_characteristic_ratio,
        consistent=drift <= CHAIN_DIMENSION_TOLERANCE,
    )


def _write_atomically(path: Path, text: str) -> None:
    """Write *text* to *path* without ever leaving it half-written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


@dataclass
class RunManifest:
    """The record of one protocol run, and the basis of resuming it.

    Args:
        protocol: Which protocol ran.
        seed: The master seed every stream derived from.
        versions: What produced this.
        system: The System settings, as recorded floats.
        stages: One entry per completed stage, keyed by name.
        chains: The chain dimensions measured at the end.
        box: What was in the cell - ``n_molecules``, ``atoms_per_chain`` and
            ``box_nm``. :class:`~openmmpolymer.mdsystem.SystemSpec` records the
            settings a run was given but not the thing it was given them for,
            and every molecule being a copy of the same chain is the invariant
            the whole package indexes by, so analysis of a finished run should
            be able to read it rather than infer it. None for a manifest
            written before this was recorded.
    """

    protocol: str
    seed: int
    versions: dict[str, str] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    chains: dict[str, Any] | None = None
    box: dict[str, Any] | None = None

    def save(self, run_dir: str | Path) -> str:
        """Write the manifest into *run_dir*, atomically."""
        path = Path(run_dir) / MANIFEST_NAME
        _write_atomically(path, json.dumps(asdict(self), indent=2, default=str) + "\n")
        return str(path)

    @classmethod
    def load(cls, run_dir: str | Path) -> RunManifest | None:
        """Read the manifest from *run_dir*, or None if there is not one."""
        path = Path(run_dir) / MANIFEST_NAME
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text()))


def _versions() -> dict[str, str]:
    """Record what produced a run, so a surprising result can be placed."""
    from importlib.metadata import PackageNotFoundError, version

    import openmm as mm

    # Read from the installed metadata rather than the package namespace: this
    # module is imported while that namespace is still being built.
    try:
        own = version("openmmpolymer")
    except PackageNotFoundError:  # pragma: no cover - uninstalled checkout
        own = "0.0.0+unknown"
    versions = {"openmmpolymer": own, "openmm": mm.version.version}
    try:
        import forcefill

        versions["forcefill"] = getattr(forcefill, "__version__", "unknown")
    except ImportError:  # pragma: no cover - forcefill is a hard dependency
        pass
    return versions


@dataclass(frozen=True)
class RunSummary:
    """What a whole protocol run did.

    Args:
        protocol: Which protocol ran.
        run_dir: Where it wrote.
        manifest_path: The manifest.
        results: Each stage's result, in order.
        skipped: Stages that were already complete and were not rerun.
        wall_seconds: Total wall time of the stages that actually ran.
        final_state: The last state written, which a later run starts from.
        chains: The chain dimensions at the end, if they were measured.
    """

    protocol: str
    run_dir: str
    manifest_path: str
    results: tuple[StageResult, ...]
    skipped: tuple[str, ...]
    wall_seconds: float
    final_state: str
    chains: ChainDimensions | None = None


def run_protocol(
    protocol: Protocol,
    run: RunContext,
    run_dir: str | Path = "run",
    *,
    resume: bool = True,
    state_in: str | Path | None = None,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
) -> RunSummary:
    """Run a protocol's stages in order, skipping any already finished.

    Args:
        protocol: What to run.
        run: The run context.
        run_dir: Where everything is written.
        resume: Skip stages the manifest records as complete and whose state
            is still on disk. Turn this off to force a rerun.
        state_in: A state to start the first stage from, instead of the packed
            coordinates.
        chain_backbone: Backbone atom indices within one chain, from
            :func:`openmmpolymer.chain.backbone_path`. Given these, the run
            measures its chain dimensions at the end and records them.
        atoms_per_chain: Atoms per molecule, needed with *chain_backbone*.
        expected_characteristic_ratio: The polymer's C-infinity, for that
            measurement.

    Returns:
        What the run did.

    Raises:
        ProtocolError: A stage failed.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = (RunManifest.load(directory) if resume else None) or RunManifest(
        protocol=protocol.name, seed=run.seed
    )
    manifest.protocol = protocol.name
    manifest.seed = run.seed
    manifest.versions = _versions()
    manifest.system = asdict(run.spec)
    manifest.box = {
        "n_molecules": run.box.n_molecules,
        "atoms_per_chain": run.box.topology.getNumAtoms() // run.box.n_molecules,
        "box_nm": list(run.box.box_nm),
    }

    results: list[StageResult] = []
    skipped: list[str] = []
    state: str | Path | None = state_in
    started = time.monotonic()

    for stage in protocol.stages:
        recorded = manifest.stages.get(stage.name)
        if resume and recorded and Path(recorded.get("final_state", "")).is_file():
            log.info("Skipping %s: already complete.", stage.name)
            skipped.append(stage.name)
            state = recorded["final_state"]
            continue

        log.info("Running %s (%s).", stage.name, stage.kind)
        runner = STAGE_RUNNERS[stage.kind]
        try:
            result = runner(
                run,
                directory / stage.name,
                state_in=state,
                **stage.options,
            )
        except Exception as error:
            manifest.save(directory)
            raise ProtocolError(
                f"Stage {stage.name!r} of {protocol.name!r} failed: {error}. "
                f"The manifest in {directory} records what completed; fix the "
                "cause and run again to pick up from there."
            ) from error

        results.append(result)
        state = result.final_state
        manifest.stages[stage.name] = asdict(result)
        # Saved after every stage, not at the end: the point of the manifest is
        # to survive whatever stops the run.
        manifest.save(directory)

    dimensions = _measure_chains(
        run,
        state,
        chain_backbone,
        atoms_per_chain,
        expected_characteristic_ratio,
    )
    if dimensions is not None:
        manifest.chains = asdict(dimensions)
        # The backbone beside the dimensions it produced, so a later analysis
        # can measure the same thing without having to guess the path.
        assert chain_backbone is not None  # _measure_chains returned dimensions
        manifest.chains["backbone"] = [int(index) for index in chain_backbone]
    manifest_path = manifest.save(directory)

    return RunSummary(
        protocol=protocol.name,
        run_dir=str(directory),
        manifest_path=manifest_path,
        results=tuple(results),
        skipped=tuple(skipped),
        wall_seconds=time.monotonic() - started,
        final_state=str(state),
        chains=dimensions,
    )


def _measure_chains(
    run: RunContext,
    state_path: str | Path | None,
    backbone: Sequence[int] | None,
    atoms_per_chain: int | None,
    expected_ratio: float,
) -> ChainDimensions | None:
    """Measure the final chain dimensions, when there is enough to do it with."""
    if backbone is None or atoms_per_chain is None or state_path is None:
        return None

    import openmm as mm
    from openmm import unit

    state = mm.XmlSerializer.deserialize(Path(state_path).read_text())
    positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )
    system = mm.XmlSerializer.deserialize(run.system_xml)
    masses = np.array(
        [
            system.getParticleMass(index).value_in_unit(unit.dalton)
            for index in range(atoms_per_chain)
        ]
    )
    dimensions = chain_dimensions(
        positions,
        backbone,
        atoms_per_chain,
        run.box.n_molecules,
        expected_characteristic_ratio=expected_ratio,
        masses=masses,
    )
    log.info(
        "Chain dimensions: Rg %.3f nm, <R^2> %.3f nm^2, C %.2f against an "
        "expected %.2f - %s.",
        dimensions.mean_radius_of_gyration_nm,
        dimensions.mean_squared_end_to_end_nm2,
        dimensions.characteristic_ratio,
        dimensions.expected_characteristic_ratio,
        "consistent"
        if dimensions.consistent
        else "not consistent, so the chains have not relaxed at their own scale",
    )
    return dimensions
