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

import functools
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import openmm as mm
from openmm import unit

from ._files import file_sha256, write_json
from .reporters import TrajectoryOptions
from .simulate import (
    RunContext,
    StageResult,
    _anneal_segments,
    _compress_segments,
    _heat_segments,
    _quench_segments,
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

    @property
    def duration_ps(self) -> float:
        """How much dynamics this stage asks for, from the options it was given."""
        options = _stage_options(self)
        if self.kind in {"heat", "quench"}:
            build = _heat_segments if self.kind == "heat" else _quench_segments
            segments = build(
                t_start=options["t_start"],
                t_end=options["t_end"],
                step_k=options["step_k"],
                hold_ps=options["hold_ps"],
                pressure_bar=options["pressure_bar"],
                temperatures_k=options["temperatures_k"],
            )
        elif self.kind == "anneal":
            segments = _anneal_segments(
                t_low=options["t_low"],
                t_high=options["t_high"],
                n_cycles=options["n_cycles"],
                ramp_windows=options["ramp_windows"],
                window_ps=options["window_ps"],
                hold_ps=options["hold_ps"],
                pressure_bar=options["pressure_bar"],
            )
        elif self.kind == "compress":
            segments = _compress_segments(
                temperature_k=options["temperature_k"],
                pressures_bar=options["pressures_bar"],
                duration_ps_each=options["duration_ps_each"],
            )
        else:
            segments = None
        if segments is not None:
            return sum((segment.duration_ps for segment in segments), 0.0)
        if self.kind == "deform":
            return float(options["relax_ps"]) * int(options["n_steps"])
        if self.kind == "load":
            return float(options["duration_ps_each"]) * len(options["stresses_bar"])
        if self.kind == "shear":
            return float(options["duration_ps_each"]) * len(options["strains"])
        if self.kind == "relax":
            # A chunk that opens an already-strained cell repeats neither the
            # baseline nor the ramp, so counting them would price a resumed
            # ladder as several first chunks.
            held = 0.0 if options["strain_applied"] else float(options["baseline_ps"])
            ramp = 0.0 if options["strain_applied"] else float(options["ramp_ps"])
            return held + ramp + float(options["duration_ps"])
        duration = options.get("duration_ps")
        return 0.0 if duration is None else float(duration)


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
        every stage. Ladder stages state their time as individual holds, and
        a budget that silently omits the most expensive stage in
        a protocol is worse than no budget, because it gets believed.
        """
        return sum(stage.duration_ps for stage in self.stages)


def _stage_options(stage: Stage) -> dict[str, Any]:
    """A stage's options, filled in with its runner's own defaults.

    Read off the runner's signature rather than repeated here, so the numbers
    a cost estimate is built from cannot drift away from the numbers the run
    actually uses.
    """
    parameters = inspect.signature(STAGE_RUNNERS[stage.kind]).parameters
    forwarded = (
        inspect.signature(run_segments).parameters
        if any(
            item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        )
        else {}
    )
    defaults = {
        name: parameter.default
        for name, parameter in {**forwarded, **parameters}.items()
        if parameter.default is not inspect.Parameter.empty
    }
    options = {**defaults, **stage.options}
    if stage.kind == "heat":
        options["measure_enthalpy"] = True
    if stage.kind == "production":
        options["barostat"] = None if options["pressure_bar"] is None else "isotropic"
    return options


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


@dataclass
class _ChainDimensionSums:
    """Sufficient statistics to pool chains without retaining coordinates."""

    n_chains: int = 0
    sum_squared_end_to_end_nm2: float = 0.0
    sum_radius_of_gyration_nm: float = 0.0
    sum_squared_radius_of_gyration_nm2: float = 0.0
    sum_bond_length_nm: float = 0.0

    def add(self, other: _ChainDimensionSums) -> None:
        """Include another cell or frame's chains in the pooled sums."""
        self.n_chains += other.n_chains
        self.sum_squared_end_to_end_nm2 += other.sum_squared_end_to_end_nm2
        self.sum_radius_of_gyration_nm += other.sum_radius_of_gyration_nm
        self.sum_squared_radius_of_gyration_nm2 += (
            other.sum_squared_radius_of_gyration_nm2
        )
        self.sum_bond_length_nm += other.sum_bond_length_nm

    def dimensions(
        self, n_bonds: int, *, expected_characteristic_ratio: float
    ) -> ChainDimensions:
        """Form ratios from pooled means, including the square of mean bond length."""
        mean_square = self.sum_squared_end_to_end_nm2 / self.n_chains
        mean_radius = self.sum_radius_of_gyration_nm / self.n_chains
        mean_squared_radius = self.sum_squared_radius_of_gyration_nm2 / self.n_chains
        bond_length = self.sum_bond_length_nm / self.n_chains
        measured = mean_square / (n_bonds * bond_length**2)
        drift = (
            abs(measured - expected_characteristic_ratio)
            / expected_characteristic_ratio
        )
        return ChainDimensions(
            mean_squared_end_to_end_nm2=mean_square,
            mean_radius_of_gyration_nm=mean_radius,
            ratio_of_squares=mean_square / mean_squared_radius,
            characteristic_ratio=measured,
            expected_characteristic_ratio=expected_characteristic_ratio,
            consistent=drift <= CHAIN_DIMENSION_TOLERANCE,
        )


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
    return _chain_dimension_sums(
        positions_nm, backbone, atoms_per_chain, n_chains, masses=masses
    ).dimensions(
        len(backbone) - 1, expected_characteristic_ratio=expected_characteristic_ratio
    )


def _chain_dimension_sums(
    positions_nm: npt.NDArray[np.float64],
    backbone: Sequence[int],
    atoms_per_chain: int,
    n_chains: int,
    *,
    masses: npt.NDArray[np.float64] | None = None,
) -> _ChainDimensionSums:
    """Measure each chain once, keeping the sums needed by both dimension ratios."""
    path = np.asarray(backbone, dtype=int)
    sums = _ChainDimensionSums()

    for index in range(n_chains):
        offset = index * atoms_per_chain
        chain = positions_nm[offset : offset + atoms_per_chain]
        ends = chain[path[-1]] - chain[path[0]]
        sums.n_chains += 1
        sums.sum_squared_end_to_end_nm2 += float(ends @ ends)

        weights = np.ones(len(chain)) if masses is None else masses
        centre = (weights[:, None] * chain).sum(axis=0) / weights.sum()
        offsets = chain - centre
        radius = float(
            np.sqrt((weights * (offsets * offsets).sum(axis=1)).sum() / weights.sum())
        )
        sums.sum_radius_of_gyration_nm += radius
        sums.sum_squared_radius_of_gyration_nm2 += radius * radius

        steps = chain[path[1:]] - chain[path[:-1]]
        sums.sum_bond_length_nm += float(np.sqrt((steps * steps).sum(axis=1)).mean())

    return sums


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
        provenance: Versioned fingerprints of the original cell and each
            stage's request and state dependencies. Legacy manifests without
            this remain readable for analysis, but cannot safely be resumed.
    """

    protocol: str
    seed: int
    versions: dict[str, str] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    chains: dict[str, Any] | None = None
    box: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    def save(self, run_dir: str | Path) -> str:
        """Write the manifest into *run_dir*, atomically."""
        return write_json(Path(run_dir) / MANIFEST_NAME, asdict(self), strict=False)

    @classmethod
    def load(cls, run_dir: str | Path) -> RunManifest | None:
        """Read the manifest from *run_dir*, or None if there is not one."""
        path = Path(run_dir) / MANIFEST_NAME
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text()))


def _versions() -> dict[str, str]:
    """Record what produced a run, so a surprising result can be placed."""
    # Imported here rather than with the module: it brings the whole
    # parameterisation stack with it, and only a run that writes a manifest
    # needs its version.
    import forcefill

    # Read from the installed metadata rather than the package namespace: this
    # module is imported while that namespace is still being built.
    try:
        own = version("openmmpolymer")
    except PackageNotFoundError:  # pragma: no cover - uninstalled checkout
        own = "0.0.0+unknown"
    return {
        "openmmpolymer": own,
        "openmm": mm.version.version,
        "forcefill": forcefill.__version__,
    }


def _canonical(value: Any) -> Any:
    """Represent scientific inputs consistently in memory and in JSON."""
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    # Fail closed for unsupported objects and nonfinite settings, rather than
    # giving unrelated inputs the same uninformative string representation.
    json.dumps(value, allow_nan=False)
    return value


def _digest(value: bytes) -> str:
    """The SHA-256 of *value*, in hex."""
    return hashlib.sha256(value).hexdigest()


def _run_identity(run: RunContext) -> dict[str, Any]:
    """Fingerprint the actual cell and Hamiltonian, not just build settings."""
    topology = run.box.topology
    # Reporters update topology box vectors to the final cell. The immutable
    # initial vectors used by dynamics are already in system_xml and box_nm.
    description = {
        "atoms": [
            [
                atom.name,
                None if atom.element is None else atom.element.atomic_number,
                atom.residue.index,
                atom.residue.name,
                atom.residue.chain.index,
            ]
            for atom in topology.atoms()
        ],
        "bonds": [
            [bond[0].index, bond[1].index, str(bond.type), bond.order]
            for bond in topology.bonds()
        ],
    }
    positions = np.asarray(run.box.positions_nm, dtype="<f8")
    return cast(
        "dict[str, Any]",
        _canonical(
            {
                "seed": run.seed,
                "system": asdict(run.spec),
                "system_sha256": _digest(run.system_xml.encode()),
                "topology_sha256": _digest(
                    json.dumps(description, sort_keys=True, allow_nan=False).encode()
                ),
                "positions_sha256": _digest(positions.tobytes()),
                "positions_shape": positions.shape,
                "box_nm": run.box.box_nm,
                "n_molecules": run.box.n_molecules,
                "total_mass_g_mol": run.total_mass_g_mol,
                "precision": run.precision,
            }
        ),
    )


def _state_source(state: str | Path | None, manifest: RunManifest) -> dict[str, Any]:
    """Say where a stage's starting state came from, as its request records it.

    A state the manifest itself produced is named by the stage and the artifact
    - its final state, or a waypoint by index - so that invalidating that stage
    reaches the request too. Anything from outside is named by its content.
    """
    if state is None:
        return {"packed": True}
    path = Path(state).resolve()
    for name, recorded in manifest.stages.items():
        if Path(recorded["final_state"]).resolve() == path:
            return {"stage": name, "artifact": "final_state"}
        for index, waypoint in enumerate(recorded.get("waypoints", ())):
            if Path(waypoint).resolve() == path:
                return {"stage": name, "artifact": index}
    return {"external_state_sha256": file_sha256(path)}


def _source_path(source: dict[str, Any], manifest: RunManifest) -> str | None:
    """The file a recorded stage source names, or None if it is not recorded."""
    parent = manifest.stages.get(source.get("stage", ""))
    if parent is None:
        return None
    artifact = source["artifact"]
    if artifact == "final_state":
        return str(parent["final_state"])
    waypoints = parent.get("waypoints", ())
    return str(waypoints[artifact]) if artifact < len(waypoints) else None


def _validate_identity(manifest: RunManifest, identity: dict[str, Any]) -> None:
    """Refuse to resume a manifest written for other starting inputs, or none.

    A legacy manifest, written before provenance was recorded, still loads for
    analysis; it cannot be resumed, because nothing says what it started from.
    """
    if manifest.provenance is None:
        raise ProtocolError(
            "Cannot safely resume a legacy manifest without input provenance. "
            "Use a fresh run directory or rerun with resume=False."
        )
    if (
        manifest.provenance.get("version") != 1
        or manifest.provenance.get("run") != identity
    ):
        raise ProtocolError(
            "Cannot resume: starting inputs changed (system, seed, coordinates, "
            "topology or settings). Use a fresh run directory or rerun with resume=False."
        )


def validate_run_inputs(run: RunContext, run_dir: str | Path) -> None:
    """Read-only input check for workflows that save metadata before stages."""
    manifest = RunManifest.load(run_dir)
    if manifest is not None:
        _validate_identity(manifest, _run_identity(run))


def _prepare_manifest(
    protocol: Protocol,
    run: RunContext,
    directory: Path,
    *,
    resume: bool,
    state_in: str | Path | None,
) -> RunManifest:
    """Validate all requested stages before modifying any run artifacts.

    Stage dependencies support prefix extensions and independent branches in
    one manifest. Invalidating a parent also invalidates every descendant,
    including branches absent from the current protocol invocation. Nothing is
    written here, so each file's digest is taken once however many stages it
    feeds.
    """
    sha256 = functools.cache(file_sha256)
    identity = _run_identity(run)
    manifest = RunManifest.load(directory) if resume else None
    if manifest is not None:
        _validate_identity(manifest, identity)
        if manifest.protocol != protocol.name:
            raise ProtocolError(
                "Cannot resume: protocol changed. Use a fresh run directory "
                "or rerun with resume=False."
            )
    else:
        manifest = RunManifest(
            protocol=protocol.name,
            seed=run.seed,
            versions=_versions(),
            system=asdict(run.spec),
            box={
                "n_molecules": run.box.n_molecules,
                "atoms_per_chain": run.box.topology.getNumAtoms()
                // run.box.n_molecules,
                "box_nm": list(run.box.box_nm),
            },
            provenance={"version": 1, "run": identity, "stages": {}},
        )
    assert manifest.provenance is not None
    recorded_inputs = manifest.provenance["stages"]
    requested: dict[str, Any] = {}
    source = _state_source(state_in, manifest)
    for stage in protocol.stages:
        options = _stage_options(stage)
        options.pop("state_in", None)
        options.pop("output_prefix", None)
        signature = {
            "kind": stage.kind,
            "options": _canonical(options),
            "input": source,
        }
        previous = recorded_inputs.get(stage.name)
        if previous is not None and previous["request"] != signature:
            raise ProtocolError(
                f"Cannot resume stage {stage.name!r}: its settings or starting "
                "state changed. Use a fresh run directory or rerun with resume=False."
            )
        requested[stage.name] = {"request": signature}
        source = {"stage": stage.name, "artifact": "final_state"}

    invalid: set[str] = set()
    for name, recorded in manifest.stages.items():
        provenance = recorded_inputs.get(name)
        final = str(recorded.get("final_state", ""))
        if (
            provenance is None
            or not Path(final).is_file()
            or provenance.get("output_sha256") != sha256(final)
        ):
            invalid.add(name)
            continue
        parent = provenance["request"]["input"]
        if "stage" in parent:
            path = _source_path(parent, manifest)
            if (
                path is None
                or not Path(path).is_file()
                or provenance.get("input_sha256") != sha256(path)
            ):
                invalid.add(name)
                # A consumed waypoint is an upstream output too. Recreate
                # its producer instead of accepting the altered waypoint as
                # a fresh input to an otherwise unchanged branch.
                invalid.add(parent["stage"])
    while True:
        descendants = {
            name
            for name, item in recorded_inputs.items()
            if item["request"]["input"].get("stage") in invalid
        }
        if descendants <= invalid:
            break
        invalid.update(descendants)
    initial_parent = requested[protocol.stages[0].name]["request"]["input"].get("stage")
    if initial_parent in invalid:
        raise ProtocolError(
            f"Cannot start from stale stage {initial_parent!r}. Rerun the upstream "
            "preparation protocol first, so this branch uses a verified state."
        )
    for name in invalid:
        manifest.stages.pop(name, None)
        recorded_inputs.pop(name, None)
    if invalid:
        manifest.chains = None
        log.info("Invalidated stages with stale upstream states: %s", sorted(invalid))
    for name, item in requested.items():
        recorded_inputs.setdefault(name, item)
    return manifest


BUILD_REQUEST_NAME = "build_request.json"


def check_build_request(run_dir: str | Path, request: dict[str, Any]) -> None:
    """Check CLI inputs before rebuilding assets in an existing run directory.

    Callers pass all chemistry, packing, force-field and simulation arguments;
    output locations and presentation-only arguments should be omitted.
    """
    directory = Path(run_dir)
    path = directory / BUILD_REQUEST_NAME
    if path.is_file():
        if json.loads(path.read_text()) != _canonical(request):
            raise ProtocolError(
                "Build or simulation inputs changed; use a fresh output directory "
                "to preserve the existing run and its build artifacts."
            )
    elif directory.exists() and (
        (directory / "build").exists() or any(directory.rglob(MANIFEST_NAME))
    ):
        raise ProtocolError(
            "Existing build or run artifacts lack input provenance; use a fresh "
            "output directory to preserve them."
        )


def record_build_request(run_dir: str | Path, request: dict[str, Any]) -> None:
    """Record a checked CLI request atomically, before build work starts."""
    check_build_request(run_dir, request)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / BUILD_REQUEST_NAME, _canonical(request))


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
        resume: Skip completed stages only when their inputs and saved states
            still match. Missing or modified states invalidate their downstream
            stages too. Turn this off to force a rerun with changed inputs.
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
        ProtocolError: A stage failed, inputs changed, or a legacy manifest
            lacks the provenance needed to resume safely.
    """
    directory = Path(run_dir)
    manifest = _prepare_manifest(
        protocol, run, directory, resume=resume, state_in=state_in
    )
    directory.mkdir(parents=True, exist_ok=True)
    # Persist invalidations before any runner can overwrite an upstream file.
    # An interruption must never leave old descendants marked complete.
    manifest.save(directory)
    assert manifest.provenance is not None

    results: list[StageResult] = []
    skipped: list[str] = []
    # A state's digest travels with it, so none is read twice: a completed
    # stage's was checked a moment ago, a new one's is taken as it is written,
    # and only a state handed in from outside is read when a stage needs it.
    state: str | Path | None = state_in
    digest: str | None = None
    started = time.monotonic()

    for stage in protocol.stages:
        recorded = manifest.stages.get(stage.name)
        provenance = manifest.provenance["stages"][stage.name]
        if resume and recorded and Path(recorded.get("final_state", "")).is_file():
            log.info("Skipping %s: already complete.", stage.name)
            skipped.append(stage.name)
            state = recorded["final_state"]
            digest = provenance["output_sha256"]
            continue

        log.info("Running %s (%s).", stage.name, stage.kind)
        manifest.chains = None
        runner = STAGE_RUNNERS[stage.kind]
        if state is not None and digest is None:
            digest = file_sha256(state)
        provenance["input_sha256"] = digest
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
        digest = file_sha256(state)
        manifest.stages[stage.name] = asdict(result)
        provenance["output_sha256"] = digest
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
