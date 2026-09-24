"""Resumable tensile scans: offset yield, breaking strength, elongation at break.

The three measurements share one engine and differ only in how they read a
curve. Each scan equilibrates the cell once and stretches every replica from
that equilibrated state with fresh velocities, in compounding strain
increments while the two transverse dimensions relax at the set pressure. A
replica's ladder is split into resumable chunks that share one unstrained
reference box, so an interrupted scan resumes without resetting its strain
origin. The complete ladder is always sampled: stopping at the first stress dip
would confuse yield or a fluctuation with the ultimate strength.

The request is saved atomically before the first stage. A scan resumes only
with the same request, and only where its completed stages are unbroken
prefixes of the equilibration and of each ladder with their states intact;
``resume=False`` reruns the complete scan. A directory holding another
protocol, or stages without a record, needs a fresh directory. Engine failures
propagate: they are never read as material failure.

Analysis never runs dynamics or modifies recorded results. Replicas are fitted
separately - pooling their points would mix different event strains - with the
criterion the scan recorded, and that record must be complete and agree with
the ladder it lists. A missing or incomplete replica keeps its curve and fit
diagnostics but leaves the headline ``None``. The headline is the mean over
replicas and the spread their sample standard deviation, which measures
trajectory variability at one starting structure, not morphology uncertainty.
Reports record each curve's differential Cauchy stress beside its nominal
stress, so every event is auditable.

Fixed-topology force fields cannot break bonds, so these are apparent
properties of the simulated cell rather than chemical fracture tests, and an
offset-line crossing without an unloading measurement does not establish
permanent deformation.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._files import ReportFiles, json_value, write_json
from ._validation import require_axis, require_finite, require_integer, require_positive
from ._workflow import (
    equilibrated_box_nm,
    run_fingerprint,
    sample_spread,
    settled_state,
)
from .elasticity import StressStrain, stress_strain
from .plots import plot_breaking_strength, plot_elongation_at_break, plot_yield_strength
from .protocols import (
    Protocol,
    RunManifest,
    Stage,
    run_protocol,
    standard_melt_equilibration,
    validate_run_inputs,
)
from .reporters import TrajectoryOptions
from .simulate import RunContext, safe_timestep_fs
from .strength import (
    BreakingStrength,
    ElongationAtBreak,
    YieldStrength,
    breaking_strength,
    elongation_at_break,
    yield_strength,
)
from .trajectory import AnalysisError

log = logging.getLogger(__name__)


class BreakingError(RuntimeError):
    """A tensile-strength scan cannot safely start or resume."""


class ElongationError(RuntimeError):
    """An elongation-at-break scan cannot safely start or resume."""


class YieldError(RuntimeError):
    """A yield-strength scan cannot safely start or resume."""


@dataclass(frozen=True)
class TensileSpec:
    """The strain ladder every tensile measurement shares.

    Increments compound; ``max_strain`` is the target engineering strain,
    reached or slightly exceeded by the last increment. ``relax_ps`` is the
    duration of each hold, whose mean stress comes from the second half of
    its ``samples_per_step`` readings, and ``stage_ps`` bounds each resumable
    chunk. Each of the ``n_replicas`` ladders starts from the same
    equilibrated cell with fresh velocities. ``trajectory_ps`` optionally
    saves XTC frames for structural inspection. ``max_total_ns`` budgets the
    equilibration and every replica; a scan resumes under a changed budget,
    because the budget does not change the dynamics.

    A ladder alone selects no measurement: :class:`BreakingSpec`,
    :class:`ElongationSpec` and :class:`YieldSpec` add their criteria.
    """

    temperature_k: float = 298.15
    pressure_bar: float = 1.0
    axis: int = 2
    strain_increment: float = 0.01
    max_strain: float = 1.0
    relax_ps: float = 50.0
    n_replicas: int = 3
    samples_per_step: int = 250
    stage_ps: float = 1000.0
    trajectory_ps: float | None = None
    max_total_ns: float | None = None

    def __post_init__(self) -> None:
        """Reject unusable settings before any dynamics or output."""
        for name in (
            "temperature_k",
            "pressure_bar",
            "strain_increment",
            "max_strain",
            "relax_ps",
            "stage_ps",
        ):
            require_positive(getattr(self, name), None, name=name)
        require_axis(self.axis)
        require_integer(self.n_replicas, name="n_replicas")
        require_integer(self.samples_per_step, minimum=2, name="samples_per_step")
        if self.strain_increment >= self.max_strain:
            raise ValueError("strain_increment must be below max_strain.")
        if self.stage_ps < self.relax_ps:
            raise ValueError("stage_ps must hold at least one relax_ps increment.")
        for name in ("trajectory_ps", "max_total_ns"):
            value = getattr(self, name)
            if value is not None:
                require_positive(value, None, name=name)


@dataclass(frozen=True)
class BreakingSpec(TensileSpec):
    """A tensile ladder and the stress-loss criterion that confirms its peak.

    At least ``confirmation_steps`` terminal holds must stay at or below
    ``failure_fraction`` of the peak nominal stress. The criterion records a
    loss of stress, not covalent fracture; a ladder that ends while still
    strengthening stays unresolved.
    """

    failure_fraction: float = 0.5
    confirmation_steps: int = 3

    def __post_init__(self) -> None:
        """Reject an unusable criterion as well as an unusable ladder."""
        super().__post_init__()
        require_positive(self.failure_fraction, None, name="failure_fraction")
        if self.failure_fraction >= 1.0:
            raise ValueError("failure_fraction must be strictly between zero and one.")
        require_integer(self.confirmation_steps, minimum=2, name="confirmation_steps")


@dataclass(frozen=True)
class ElongationSpec(BreakingSpec):
    """The breaking ladder and criterion, read for elongation at break.

    The endpoint is the first hold of the confirmed terminal loss of stress,
    not the strain at the peak. Strains are fractions of the reference length:
    ``max_strain=1.0`` requests at least 100% extension.
    """


@dataclass(frozen=True)
class YieldSpec(TensileSpec):
    """A small-strain tensile ladder and its offset proof-stress criterion.

    An elastic line is fitted to nominal stress between ``fit_min_strain`` and
    ``fit_max_strain`` and shifted by ``offset_strain`` - the default is a 0.2%
    offset - and the proof stress is its first later crossing with the curve.
    A ladder with an unreliable elastic fit or no crossing stays unresolved.
    """

    strain_increment: float = 0.002
    max_strain: float = 0.3
    offset_strain: float = 0.002
    fit_min_strain: float = 0.0
    fit_max_strain: float = 0.02

    def __post_init__(self) -> None:
        """Reject an unusable criterion as well as an unusable ladder."""
        super().__post_init__()
        require_positive(self.offset_strain, None, name="offset_strain")
        require_positive(self.fit_max_strain, None, name="fit_max_strain")
        require_finite(self.fit_min_strain, None, name="fit_min_strain")
        if not 0.0 <= self.fit_min_strain < self.fit_max_strain:
            raise ValueError(
                "fit_min_strain must be nonnegative and below fit_max_strain."
            )
        if self.fit_max_strain >= self.max_strain:
            raise ValueError("fit_max_strain must be below max_strain.")
        if self.offset_strain >= self.max_strain:
            raise ValueError("offset_strain must be below max_strain.")


@dataclass(frozen=True)
class TensileSchedule:
    """One replica's compounded strain ladder and simulated duration."""

    n_steps: int
    increment: float
    relax_ps: float

    @property
    def max_strain(self) -> float:
        """Actual final engineering strain, including the last full increment."""
        return float((1.0 + self.increment) ** self.n_steps - 1.0)

    @property
    def total_ps(self) -> float:
        """Total duration of the holds."""
        return self.n_steps * self.relax_ps

    @property
    def strain_rate_per_ns(self) -> float:
        """Average engineering strain rate; the logarithmic rate is constant."""
        return self.max_strain / self.total_ps * 1000.0


@dataclass(frozen=True)
class BreakingReport:
    """Separate replica peaks, criterion status, and all measured curves.

    ``strength_mpa`` is the mean of the replicas' confirmed peaks.
    ``replica_indices`` keeps each replica's recorded number when some are
    missing.
    """

    run_dir: str
    manifest_path: str
    curves: tuple[StressStrain, ...]
    replicas: tuple[BreakingStrength, ...]
    replica_indices: tuple[int, ...]
    strength_mpa: float | None
    replica_spread_mpa: float | None
    resolved: bool
    failure_fraction: float
    confirmation_steps: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ElongationReport:
    """Apparent elongation at break for independent velocity replicas.

    ``elongation_percent`` is the mean of the replicas' percentages, and
    ``replica_spread_percent`` their spread in percentage points.
    ``replica_indices`` keeps each replica's recorded number when some are
    missing.
    """

    run_dir: str
    manifest_path: str
    curves: tuple[StressStrain, ...]
    replicas: tuple[ElongationAtBreak, ...]
    replica_indices: tuple[int, ...]
    elongation_percent: float | None
    replica_spread_percent: float | None
    resolved: bool
    failure_fraction: float
    confirmation_steps: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class YieldReport:
    """Separate replica proof stresses and all measured curves.

    ``strength_mpa`` is the mean of the replicas' proof stresses.
    ``replica_indices`` keeps each replica's recorded number when some are
    missing.
    """

    run_dir: str
    manifest_path: str
    curves: tuple[StressStrain, ...]
    replicas: tuple[YieldStrength, ...]
    replica_indices: tuple[int, ...]
    strength_mpa: float | None
    replica_spread_mpa: float | None
    resolved: bool
    offset_strain: float
    fit_min_strain: float
    fit_max_strain: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class TensileMeasurement[R: BreakingReport | ElongationReport | YieldReport]:
    """What sets one tensile measurement apart from the other two.

    The ladder, its record, the resume rules and the replica reader are
    shared. ``name`` names a measurement's protocol, its stages
    (``06_{name}_r{replica}_{chunk:03d}``), its workflow record, its report
    and its figures; the rest says how it reads a curve and reports it.

    Args:
        name: One of ``breaking``, ``elongation`` and ``yield``.
        label: The measured property, as a missing-stages error names it.
        error: What a scan that cannot safely start or resume raises.
        spec: The settings, whose exact type selects this measurement.
        fit: Reads one replica's curve, given the ``criterion`` settings.
        criterion: The spec fields the fit takes, which the report records.
        event: The fit fields locating the event, cleared for an incomplete
            replica, whose event can only be provisional.
        report: The report type.
        value: The fit field whose replica mean is the report field of the
            same name.
        spread: The report field for the replicas' sample standard deviation.
        plot: Draws one replica's curve and fit.
        notes: The caveats every report carries.
        provisional: The note an incomplete replica's fit gains.
        incomplete: The note an incomplete scan's report gains.
    """

    name: str
    label: str
    error: type[RuntimeError]
    spec: type[TensileSpec]
    fit: Callable[..., Any]
    criterion: tuple[str, ...]
    event: tuple[str, ...]
    report: Callable[..., R]
    value: str
    spread: str
    plot: Callable[[StressStrain, Any], Any]
    notes: tuple[str, ...]
    provisional: str
    incomplete: str

    @property
    def stem(self) -> str:
        """The prefix of every stage this measurement's ladders run."""
        return f"06_{self.name}"

    @property
    def workflow_name(self) -> str:
        """The record a scan saves before its first stage."""
        return f"{self.name}_workflow.json"


_NOMINAL_STRESS_NOTE = (
    "Nominal stress uses each hold's mean Cauchy stress and final transverse "
    "area; it approximates the mean force when the lateral area fluctuates."
)

BREAKING = TensileMeasurement(
    name="breaking",
    label="breaking-strength",
    error=BreakingError,
    spec=BreakingSpec,
    fit=breaking_strength,
    criterion=("failure_fraction", "confirmation_steps"),
    event=("strength_mpa", "failure_strain", "failure_stress_mpa", "failure_bracket"),
    report=BreakingReport,
    value="strength_mpa",
    spread="replica_spread_mpa",
    plot=plot_breaking_strength,
    notes=(
        "Apparent ultimate nominal tensile strength from a terminal stress-loss "
        "criterion; this is not proof of fracture or covalent bond scission.",
        "Strength depends on strain rate, temperature, cell size, morphology and "
        "force field. Velocity replicas do not sample independent morphologies.",
        _NOMINAL_STRESS_NOTE,
    ),
    provisional="Incomplete replica: the observed peak is provisional.",
    incomplete="The requested scan is incomplete; observed peaks are provisional.",
)

ELONGATION = TensileMeasurement(
    name="elongation",
    label="elongation-at-break",
    error=ElongationError,
    spec=ElongationSpec,
    fit=elongation_at_break,
    criterion=("failure_fraction", "confirmation_steps"),
    event=(
        "elongation_percent",
        "strain_at_break",
        "break_stress_mpa",
        "break_bracket",
    ),
    report=ElongationReport,
    value="elongation_percent",
    spread="replica_spread_percent",
    plot=plot_elongation_at_break,
    notes=(
        "Apparent engineering elongation at break from a terminal stress-loss "
        "criterion; this is not proof of fracture or covalent bond scission.",
        "Elongation depends on strain rate, temperature, cell size, morphology and "
        "force field. Velocity replicas do not sample independent morphologies.",
        _NOMINAL_STRESS_NOTE,
    ),
    provisional="Incomplete replica: a stress-loss endpoint is provisional.",
    incomplete="The requested scan is incomplete; elongation at break is unresolved.",
)

YIELD = TensileMeasurement(
    name="yield",
    label="yield-strength",
    error=YieldError,
    spec=YieldSpec,
    fit=yield_strength,
    criterion=("offset_strain", "fit_min_strain", "fit_max_strain"),
    event=("strength_mpa", "yield_strain", "yield_bracket"),
    report=YieldReport,
    value="strength_mpa",
    spread="replica_spread_mpa",
    plot=plot_yield_strength,
    notes=(
        "Apparent nominal yield strength from an offset proof-stress criterion. "
        "An unloading measurement is needed to establish permanent strain.",
        "Strength depends on strain rate, temperature, cell size, morphology and "
        "force field. Velocity replicas do not sample independent morphologies.",
        _NOMINAL_STRESS_NOTE,
    ),
    provisional="Incomplete replica: an offset-line crossing is provisional.",
    incomplete="The requested scan is incomplete; yield strengths are provisional.",
)


_MEASUREMENTS: tuple[TensileMeasurement[Any], ...] = (BREAKING, ELONGATION, YIELD)


def _measurement(spec: TensileSpec) -> TensileMeasurement[Any]:
    """The measurement a spec's exact type selects.

    Exact, because an :class:`ElongationSpec` is a :class:`BreakingSpec` with
    the same settings but names a different scan.
    """
    for measurement in _MEASUREMENTS:
        if type(spec) is measurement.spec:
            return measurement
    raise TypeError(
        f"{type(spec).__name__} selects no tensile measurement; use "
        "BreakingSpec, ElongationSpec or YieldSpec."
    )


def tensile_schedule(spec: TensileSpec) -> TensileSchedule:
    """Price one replica's ladder with the increments the engine applies."""
    steps = math.ceil(math.log1p(spec.max_strain) / math.log1p(spec.strain_increment))
    return TensileSchedule(steps, spec.strain_increment, spec.relax_ps)


def tensile_protocol(
    spec: TensileSpec,
    *,
    timestep_fs: float = 2.0,
    replica: int = 0,
    reference_box_nm: Sequence[float] | None = None,
) -> Protocol:
    """One replica's ladder, split into resumable chunks sharing a strain reference.

    The spec's type names the stages. A scan's ``run_*_scan`` supplies the
    equilibrated cell and its box to every chunk, and branches each replica
    from that same configuration.
    """
    measurement = _measurement(spec)
    require_integer(replica, minimum=0, name="replica")
    require_positive(timestep_fs, None, name="timestep_fs")
    n_steps = tensile_schedule(spec).n_steps
    per_chunk = int(spec.stage_ps // spec.relax_ps)
    stages: list[Stage] = []
    for chunk, done in enumerate(range(0, n_steps, per_chunk)):
        options: dict[str, Any] = {
            "temperature_k": spec.temperature_k,
            "pressure_bar": spec.pressure_bar,
            "axis": spec.axis,
            "strain_increment": spec.strain_increment,
            "n_steps": min(per_chunk, n_steps - done),
            "relax_ps": spec.relax_ps,
            "strain_start": (1.0 + spec.strain_increment) ** done - 1.0,
            "samples_per_step": spec.samples_per_step,
            "timestep_fs": timestep_fs,
            "new_velocities": chunk == 0,
        }
        if reference_box_nm is not None:
            options["reference_box_nm"] = list(reference_box_nm)
        if spec.trajectory_ps is not None:
            options["trajectory"] = TrajectoryOptions("xtc", spec.trajectory_ps)
        stages.append(
            Stage(f"{measurement.stem}_r{replica}_{chunk:03d}", "deform", options)
        )
    return Protocol(measurement.name, tuple(stages))


def _equilibration(spec: TensileSpec, options: dict[str, Any]) -> Protocol:
    base = standard_melt_equilibration(
        target_temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        **options,
    )
    return Protocol(_measurement(spec).name, base.stages)


def tensile_scan(spec: TensileSpec, **equilibration: Any) -> Protocol:
    """The equilibration and every replica's ladder, for inspection and pricing.

    Run a scan through its ``run_*_scan`` instead: replicas branch from the
    equilibrated cell rather than following each other as this flat listing
    of the cost has them.
    """
    settle = _equilibration(spec, equilibration)
    return Protocol(
        settle.name,
        (
            *settle.stages,
            *(
                stage
                for replica in range(spec.n_replicas)
                for stage in tensile_protocol(spec, replica=replica).stages
            ),
        ),
    )


def _read_record(
    measurement: TensileMeasurement[Any], directory: Path
) -> dict[str, Any]:
    path = directory / measurement.workflow_name
    if not path.is_file():
        return {}
    try:
        record = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise measurement.error(f"Cannot read {path}: {error}") from error
    if not isinstance(record, dict):
        raise measurement.error(f"{path} must contain a workflow record.")
    return record


def _check_resume(
    measurement: TensileMeasurement[Any],
    directory: Path,
    previous: dict[str, Any],
    manifest: RunManifest | None,
    request: dict[str, Any],
    settle: Protocol,
    ladders: Sequence[Protocol],
) -> None:
    """Refuse a resume that would mix new dynamics with what is recorded.

    The equilibration and each ladder have to be complete up to some stage
    and absent after it, with no ladder begun before the equilibration
    finished and no completed state missing: new predecessors would otherwise
    feed old descendants.
    """
    if manifest is not None and manifest.protocol != measurement.name:
        raise measurement.error(
            f"{directory} contains a different protocol; use a fresh directory."
        )
    if previous and previous.get("request") != request:
        raise measurement.error(
            f"Cannot resume {measurement.name} scan with different settings. Restore "
            "the original request, use a fresh directory, or rerun with resume=False."
        )
    if manifest is None:
        return
    if manifest.stages and not previous:
        raise measurement.error(
            f"{directory} already contains stages without a matching "
            f"{measurement.name} workflow record. Use a fresh directory or rerun "
            "with resume=False."
        )
    plans = (settle, *ladders)
    completed = set(manifest.stages)
    if completed - {stage.name for plan in plans for stage in plan.stages}:
        raise measurement.error(
            "Cannot resume: the manifest contains unexpected stages."
        )
    for plan in plans:
        names = [stage.name for stage in plan.stages]
        present = [name for name in names if name in completed]
        if present != names[: len(present)]:
            raise measurement.error(
                "Cannot resume: completed stages have missing predecessors. "
                "Restore them or rerun with resume=False."
            )
    settled = {stage.name for stage in settle.stages}
    if completed - settled and not settled <= completed:
        raise measurement.error(
            "Cannot resume: tensile stages exist before equilibration is complete. "
            "Restore missing stages or rerun with resume=False."
        )
    missing = [
        name
        for name, stage in manifest.stages.items()
        if not Path(stage.get("final_state", "")).is_file()
    ]
    if missing:
        raise measurement.error(
            "Cannot resume: completed stages have missing state files "
            f"({', '.join(missing)}). Restore them or rerun with resume=False so "
            "old descendants are not mixed with new dynamics."
        )


def _run_scan[R: BreakingReport | ElongationReport | YieldReport](
    measurement: TensileMeasurement[R],
    run: RunContext,
    run_dir: str | Path,
    spec: TensileSpec | None,
    *,
    resume: bool,
    chain_backbone: Sequence[int] | None,
    atoms_per_chain: int | None,
    expected_characteristic_ratio: float,
    **equilibration: Any,
) -> R:
    """Budget, record and run a scan, then analyse what it recorded."""
    spec = measurement.spec() if spec is None else spec
    if type(spec) is not measurement.spec:
        raise TypeError(
            f"The {measurement.name} scan requires {measurement.spec.__name__}, "
            f"not {type(spec).__name__}."
        )
    directory = Path(run_dir)
    settle = _equilibration(spec, equilibration)
    timestep = safe_timestep_fs(spec.temperature_k, run.spec)
    ladders = tuple(
        tensile_protocol(spec, timestep_fs=timestep, replica=replica)
        for replica in range(spec.n_replicas)
    )
    total_ns = (
        settle.total_duration_ps + sum(ladder.total_duration_ps for ladder in ladders)
    ) / 1000.0
    if spec.max_total_ns is not None and total_ns > spec.max_total_ns:
        raise measurement.error(
            f"The {measurement.name} scan costs {total_ns:.3g} ns, above "
            f"max_total_ns={spec.max_total_ns:g}; shorten the ladder or raise the "
            "budget."
        )
    settings = asdict(spec)
    settings.pop("max_total_ns")
    request = json.loads(
        json.dumps(
            {
                "spec": settings,
                "equilibration": [asdict(stage) for stage in settle.stages],
                **run_fingerprint(run),
            }
        )
    )
    previous = _read_record(measurement, directory) if resume else {}
    manifest = RunManifest.load(directory) if resume else None
    _check_resume(measurement, directory, previous, manifest, request, settle, ladders)
    schedule = tensile_schedule(spec)
    record = {
        "request": request,
        "replica_stages": [
            [stage.name for stage in ladder.stages] for ladder in ladders
        ],
        "steps_per_replica": schedule.n_steps,
        "timestep_fs": timestep,
    }
    if resume:
        validate_run_inputs(run, directory)
    directory.mkdir(parents=True, exist_ok=True)
    workflow = directory / measurement.workflow_name
    write_json(workflow, record)
    log.info(
        "%s scan: %d replicas, %.3g ns total, %.3g average strain/ns.",
        measurement.name.capitalize(),
        spec.n_replicas,
        total_ns,
        schedule.strain_rate_per_ns,
    )
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    start_state = settled_state(
        run_protocol(settle, run, directory, resume=resume, **chains),
        directory,
        error=measurement.error,
        verb="stretch",
    )
    origin = equilibrated_box_nm(start_state)
    record.update({"reference_box_nm": origin, "start_state": start_state})
    write_json(workflow, record)
    for replica in range(spec.n_replicas):
        run_protocol(
            tensile_protocol(
                spec, timestep_fs=timestep, replica=replica, reference_box_nm=origin
            ),
            run,
            directory,
            # The equilibration alone resets a manifest for an explicit rerun.
            # The replicas preserve it and each other.
            resume=True,
            state_in=start_state,
            **chains,
        )
    return _analyse(measurement, directory)


def run_breaking_scan(
    run: RunContext,
    run_dir: str | Path = "breaking",
    *,
    spec: BreakingSpec | None = None,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> BreakingReport:
    """Equilibrate, stretch every replica and read the apparent tensile strength.

    ``spec`` defaults to :class:`BreakingSpec`'s settings. Remaining keywords
    configure :func:`~openmmpolymer.protocols.standard_melt_equilibration`.
    """
    return _run_scan(
        BREAKING,
        run,
        run_dir,
        spec,
        resume=resume,
        chain_backbone=chain_backbone,
        atoms_per_chain=atoms_per_chain,
        expected_characteristic_ratio=expected_characteristic_ratio,
        **equilibration,
    )


def run_elongation_scan(
    run: RunContext,
    run_dir: str | Path = "elongation",
    *,
    spec: ElongationSpec | None = None,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> ElongationReport:
    """Equilibrate, stretch every replica and read the apparent elongation at break.

    ``spec`` defaults to :class:`ElongationSpec`'s settings. Remaining keywords
    configure :func:`~openmmpolymer.protocols.standard_melt_equilibration`.
    """
    return _run_scan(
        ELONGATION,
        run,
        run_dir,
        spec,
        resume=resume,
        chain_backbone=chain_backbone,
        atoms_per_chain=atoms_per_chain,
        expected_characteristic_ratio=expected_characteristic_ratio,
        **equilibration,
    )


def run_yield_scan(
    run: RunContext,
    run_dir: str | Path = "yield",
    *,
    spec: YieldSpec | None = None,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> YieldReport:
    """Equilibrate, stretch every replica and read the apparent yield strength.

    ``spec`` defaults to :class:`YieldSpec`'s settings. Remaining keywords
    configure :func:`~openmmpolymer.protocols.standard_melt_equilibration`.
    """
    return _run_scan(
        YIELD,
        run,
        run_dir,
        spec,
        resume=resume,
        chain_backbone=chain_backbone,
        atoms_per_chain=atoms_per_chain,
        expected_characteristic_ratio=expected_characteristic_ratio,
        **equilibration,
    )


def _replica_chunks(
    measurement: TensileMeasurement[Any],
    manifest: RunManifest | None,
    run_dir: str | Path,
) -> dict[int, list[str]]:
    """Each replica's stage names in chunk order, keyed in replica order."""
    found: dict[int, list[tuple[int, str]]] = {}
    for name in manifest.stages if manifest is not None else ():
        match = re.fullmatch(rf"{measurement.stem}_r(\d+)_(\d+)", name)
        if match is not None:
            replica, chunk = map(int, match.groups())
            found.setdefault(replica, []).append((chunk, name))
    if not found:
        raise AnalysisError(f"No {measurement.label} stages in {run_dir}.")
    return {
        replica: [name for _, name in sorted(found[replica])]
        for replica in sorted(found)
    }


def _stages(
    measurement: TensileMeasurement[Any], run_dir: str | Path
) -> tuple[str, ...]:
    """Name only this measurement's stages, not other ladders or extensions."""
    chunks = _replica_chunks(measurement, RunManifest.load(run_dir), run_dir)
    return tuple(name for names in chunks.values() for name in names)


def breaking_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name a run's breaking-strength stages, by replica and then chunk."""
    return _stages(BREAKING, run_dir)


def elongation_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name a run's elongation-at-break stages, by replica and then chunk."""
    return _stages(ELONGATION, run_dir)


def yield_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name a run's yield-strength stages, by replica and then chunk."""
    return _stages(YIELD, run_dir)


def _recorded_spec(
    measurement: TensileMeasurement[Any], record: dict[str, Any]
) -> TensileSpec:
    """The saved settings, which must be complete and agree with the saved ladder.

    A missing setting is refused rather than defaulted: a default criterion
    could resolve a scan its own criterion left unresolved.
    """
    if not record:
        # Unrecorded curves keep their diagnostics, but cannot resolve a headline.
        return measurement.spec()
    try:
        settings = record["request"]["spec"]
        if not isinstance(settings, dict):
            raise ValueError("spec must contain the saved settings")
        required = {field.name for field in fields(measurement.spec)}
        missing = required - {"max_total_ns"} - settings.keys()
        if missing:
            raise ValueError(f"missing settings: {', '.join(sorted(missing))}")
        spec = measurement.spec(**settings)
        expected = [
            [stage.name for stage in tensile_protocol(spec, replica=index).stages]
            for index in range(spec.n_replicas)
        ]
        if (
            record.get("replica_stages") != expected
            or record.get("steps_per_replica") != tensile_schedule(spec).n_steps
        ):
            raise ValueError("the recorded ladder does not match the saved settings")
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisError(
            f"Malformed {measurement.name} workflow record: {error}"
        ) from error
    return spec


def _read_replica(
    directory: Path, stages: dict[str, dict[str, Any]], names: Sequence[str]
) -> StressStrain:
    """Check that the chunks continue one ladder, then read them as one curve.

    The common reader sorts the samples by strain, which would splice chunks
    that disagree about their reference box or axis, or overlap in strain.
    """
    origin: list[float] | None = None
    axis: int | None = None
    last_strain = -1.0
    for name in names:
        samples = stages[name]["samples"]
        reference = np.asarray(samples["reference_box_nm"], dtype=np.float64)
        recorded_axis = samples["deform_axis"]
        if (
            reference.shape != (3,)
            or not np.all(np.isfinite(reference))
            or np.any(reference <= 0.0)
        ):
            raise ValueError(f"{name} has an invalid reference box.")
        if len(recorded_axis) != 1 or recorded_axis[0] not in (0, 1, 2):
            raise ValueError(f"{name} has an invalid deformation axis.")
        if origin is None:
            origin, axis = reference.tolist(), int(recorded_axis[0])
        elif reference.tolist() != origin or int(recorded_axis[0]) != axis:
            raise ValueError(f"{name} changes the reference box or deformation axis.")
        strain = np.asarray(samples["segment_strain"], dtype=np.float64)
        if (
            strain.ndim != 1
            or strain.size == 0
            or not np.all(np.isfinite(strain))
            or strain[0] <= last_strain
            or np.any(np.diff(strain) <= 0.0)
        ):
            raise ValueError(f"{name} has non-increasing or invalid strains.")
        for key in (
            "segment_duration_ps",
            "segment_box_x_nm",
            "segment_box_y_nm",
            "segment_box_z_nm",
            "segment_stress_xx_bar",
            "segment_stress_yy_bar",
            "segment_stress_zz_bar",
        ):
            values = np.asarray(samples[key], dtype=np.float64)
            if values.shape != strain.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} has invalid {key} samples.")
            if (key.endswith("_nm") or key == "segment_duration_ps") and np.any(
                values <= 0.0
            ):
                raise ValueError(f"{name} has nonpositive {key} samples.")
        last_strain = float(strain[-1])
    return stress_strain(directory, names)


def _analyse[R: BreakingReport | ElongationReport | YieldReport](
    measurement: TensileMeasurement[R], run_dir: str | Path
) -> R:
    directory = Path(run_dir)
    manifest = RunManifest.load(directory)
    chunks = _replica_chunks(measurement, manifest, directory)
    assert manifest is not None  # the replicas above were found in it
    record = _read_record(measurement, directory)
    spec = _recorded_spec(measurement, record)
    criterion = {name: getattr(spec, name) for name in measurement.criterion}
    expected = record.get("replica_stages", [])
    complete = bool(record) and len(chunks) == spec.n_replicas
    curves: list[StressStrain] = []
    fits: list[Any] = []
    for replica, names in chunks.items():
        try:
            curve = _read_replica(directory, manifest.stages, names)
            fit = measurement.fit(curve, **criterion)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise AnalysisError(
                f"Malformed {measurement.name} replica {replica}: {error}"
            ) from error
        if not (
            replica < len(expected)
            and names == expected[replica]
            and curve.n_points == record["steps_per_replica"]
        ):
            complete = False
            fit = replace(
                fit,
                resolved=False,
                notes=(*fit.notes, measurement.provisional),
                **dict.fromkeys(measurement.event),
            )
        curves.append(curve)
        fits.append(fit)
    notes = list(measurement.notes)
    if not record:
        notes.append(
            "No workflow record: the requested scan extent cannot be verified."
        )
    if not complete:
        notes.append(measurement.incomplete)
    resolved = complete and all(fit.resolved for fit in fits)
    values = [getattr(fit, measurement.value) for fit in fits]
    return measurement.report(
        run_dir=str(directory),
        manifest_path=str(directory / "manifest.json"),
        curves=tuple(curves),
        replicas=tuple(fits),
        replica_indices=tuple(chunks),
        resolved=resolved,
        notes=tuple(notes),
        **{
            measurement.value: float(np.mean(values)) if resolved else None,
            measurement.spread: sample_spread(values) if resolved else None,
        },
        **criterion,
    )


def analyse_breaking(run_dir: str | Path) -> BreakingReport:
    """Read a breaking scan's apparent tensile strength under its saved criterion."""
    return _analyse(BREAKING, run_dir)


def analyse_elongation(run_dir: str | Path) -> ElongationReport:
    """Read an elongation scan's apparent elongation at break, as a percentage."""
    return _analyse(ELONGATION, run_dir)


def analyse_yield(run_dir: str | Path) -> YieldReport:
    """Read a yield scan's offset proof stress under its saved fit window."""
    return _analyse(YIELD, run_dir)


def _write_report[R: BreakingReport | ElongationReport | YieldReport](
    measurement: TensileMeasurement[R],
    report: R,
    output_dir: str | Path | None,
    *,
    figures: bool,
    figure_format: str,
) -> ReportFiles:
    directory = (
        Path(report.run_dir) / "analysis" if output_dir is None else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = write_json(
        directory / f"{measurement.name}.json", json_value(asdict(report))
    )
    written: list[str] = []
    if figures:
        for index, curve, fit in zip(
            report.replica_indices, report.curves, report.replicas, strict=True
        ):
            figure_path = directory / f"{measurement.name}_r{index}.{figure_format}"
            measurement.plot(curve, fit).savefig(figure_path, bbox_inches="tight")
            written.append(str(figure_path))
    return ReportFiles(json=path, figures=tuple(written))


def write_breaking_report(
    report: BreakingReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``breaking.json`` and a figure per replica, ``breaking_r{i}``.

    They go to ``<run_dir>/analysis`` unless *output_dir* is given.
    """
    return _write_report(
        BREAKING, report, output_dir, figures=figures, figure_format=figure_format
    )


def write_elongation_report(
    report: ElongationReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``elongation.json`` and a figure per replica, ``elongation_r{i}``.

    They go to ``<run_dir>/analysis`` unless *output_dir* is given.
    """
    return _write_report(
        ELONGATION, report, output_dir, figures=figures, figure_format=figure_format
    )


def write_yield_report(
    report: YieldReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``yield.json`` and a figure per replica, ``yield_r{i}``.

    They go to ``<run_dir>/analysis`` unless *output_dir* is given.
    """
    return _write_report(
        YIELD, report, output_dir, figures=figures, figure_format=figure_format
    )
