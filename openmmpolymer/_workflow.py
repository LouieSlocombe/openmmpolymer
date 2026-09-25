"""The machinery the measurement workflows share.

Every scan equilibrates once, records what it was asked for so that a resume
can be checked against it, and measures from the cell that equilibration
left - most of them by branching every replica and pass from it. The pieces of
that which are not specific to one measurement live here.

So do the conventions. A workflow runs every protocol under its own name, so
an interrupted run's manifest still says which workflow it belongs to. Its
stage names are numbered, so a run directory sorts into run order, and free of
dots, because a name becomes a file stem and ``Path.with_suffix`` would read a
dot as an extension. And what it derives goes in its own ``*_workflow.json``
beside the manifest rather than in it: the manifest is the record of what ran,
and a derived field in it would be stale after the next resume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt

from ._files import ReportFiles, file_sha256, write_json, write_report
from ._validation import require_positive
from .protocols import (
    MANIFEST_NAME,
    Protocol,
    RunManifest,
    RunSummary,
    Stage,
    run_protocol,
    standard_melt_equilibration,
    validate_run_inputs,
)
from .reporters import TrajectoryOptions
from .trajectory import AnalysisError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .simulate import RunContext

log = logging.getLogger(__name__)


def run_fingerprint(run: RunContext, *, spec_key: str = "system") -> dict[str, Any]:
    """The inputs a resumed scan has to share with the one that recorded it.

    *spec_key* names the System settings as each workflow's record already
    spells them, because those records are compared as saved.
    """
    return {
        spec_key: asdict(run.spec),
        "seed": run.seed,
        "system_sha256": hashlib.sha256(run.system_xml.encode()).hexdigest(),
        "coordinates_sha256": hashlib.sha256(
            np.asarray(run.box.positions_nm, dtype=np.float64).tobytes()
        ).hexdigest(),
        "box_nm": list(run.box.box_nm),
    }


def settled_state(
    summary: RunSummary, directory: Path, *, error: type[Exception], verb: str
) -> str:
    """The state the equilibration finished at, whether it ran or resumed.

    Taken from the summary, which threads the state through skipped stages as
    well as run ones, rather than from the last manifest entry with a state:
    on a resume that entry is whatever the previous attempt got furthest
    through, and every branch has to start from the *equilibrated* cell.
    """
    state = summary.final_state
    if state and state != "None" and Path(state).is_file():
        return str(state)
    raise error(
        f"{directory} has no finished equilibration stage to {verb} from. Run "
        "the equilibration first, or delete the manifest and start over."
    )


def equilibrated_box_nm(state_path: str | Path) -> list[float]:
    """The cell edges a saved state carries, in nanometres.

    Read from the state rather than from ``run.box``, which is the *packed*
    cell: strain measured against the packed edges would be measured against
    a cell that stopped existing at the first barostat move.
    """
    import openmm as mm
    from openmm import unit

    state = mm.XmlSerializer.deserialize(Path(state_path).read_text())
    vectors = state.getPeriodicBoxVectors()
    return [
        float(vectors[axis][axis].value_in_unit(unit.nanometer)) for axis in range(3)
    ]


def group_by_stem(names: Sequence[str]) -> list[tuple[str, ...]]:
    """Group stage names into one tuple of chunks per replica.

    Grouped on the stem a numeric chunk suffix hangs off, so a ladder split for
    resume comes back as one curve and two replicas do not come back as one. A
    name with no such suffix is a group of its own.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        stem, _, tail = name.rpartition("_")
        groups.setdefault(stem if stem and tail.isdigit() else name, []).append(name)
    return [tuple(group) for group in groups.values()]


def sample_spread(values: Sequence[float]) -> float | None:
    """The sample standard deviation of the finite values, or None below two.

    None rather than zero: one replica has no spread to report, and a zero
    would read as several runs that agreed perfectly.
    """
    usable = [value for value in values if math.isfinite(value)]
    if len(usable) < 2:
        return None
    return float(np.std(usable, ddof=1))


def validate_hold_times(
    values: Sequence[float], *, name: str = "hold_times_ps"
) -> tuple[float, ...]:
    """Require three distinct positive holds, preserving their requested order."""
    holds = tuple(require_positive(value, None, name=name) for value in values)
    if len(holds) < 3 or any(
        math.isclose(first, second, rel_tol=1e-8)
        for first, second in pairwise(sorted(holds))
    ):
        raise ValueError(f"{name} needs at least three distinct positive holds.")
    return holds


def require_positive_fields(
    spec: object, names: Iterable[str], *, optional: Iterable[str] = ()
) -> None:
    """Require each named field of *spec* to be a finite positive number.

    The *optional* ones may be None instead, which is how a spec says "skip
    this" or "no limit".
    """
    for name in names:
        require_positive(getattr(spec, name), None, name=name)
    for name in optional:
        value = getattr(spec, name)
        if value is not None:
            require_positive(value, None, name=name)


def equilibration_at(
    name: str, temperature_k: float, pressure_bar: float, **options: Any
) -> Protocol:
    """The standard melt equilibration, settled where a measurement is made."""
    base = standard_melt_equilibration(
        target_temperature_k=temperature_k, pressure_bar=pressure_bar, **options
    )
    return Protocol(name, base.stages)


def resume_chunks(n_items: int, item_ps: float, stage_ps: float) -> list[range]:
    """Split a ladder of *n_items* holds into stages of at most *stage_ps* each.

    The split is bookkeeping, not physics: each stage starts from the state
    the one before it left, so the history is continuous. What it buys is
    resume granularity - a stage is the unit a run picks itself back up at,
    and a hundred nanoseconds in one stage is a hundred nanoseconds to repeat.
    Every stage holds at least one item, however short *stage_ps* is.
    """
    size = max(1, int(stage_ps // item_ps))
    return [
        range(start, min(start + size, n_items)) for start in range(0, n_items, size)
    ]


@dataclass(frozen=True)
class StrainSchedule:
    """A compounded extension's increments, holds and nominal strain rate."""

    n_steps: int
    increment: float
    relax_ps: float

    @classmethod
    def reaching(cls, max_strain: float, increment: float, relax_ps: float) -> Self:
        """Reach a target with full increments, including any final overshoot."""
        steps = math.ceil(math.log1p(max_strain) / math.log1p(increment))
        return cls(max(1, steps), increment, relax_ps)

    def strain_after(self, n_steps: int) -> float:
        """Engineering strain after *n_steps* increments from the reference cell."""
        return float((1.0 + self.increment) ** n_steps - 1.0)

    @property
    def max_strain(self) -> float:
        """The strain reached by the last full increment."""
        return self.strain_after(self.n_steps)

    @property
    def total_ps(self) -> float:
        """Total duration of the holds."""
        return self.relax_ps * self.n_steps

    @property
    def strain_rate_per_ns(self) -> float:
        """Average engineering strain rate, in strain per nanosecond."""
        return self.max_strain / self.total_ps * 1000.0


def deformation_stages(
    schedule: StrainSchedule,
    *,
    stem: str,
    chunk_digits: int,
    stage_ps: float,
    temperature_k: float,
    pressure_bar: float,
    axis: int,
    samples_per_step: int,
    timestep_fs: float,
    reference_box_nm: Sequence[float] | None = None,
    trajectory_ps: float | None = None,
) -> tuple[Stage, ...]:
    """Split one extension into chunks with continuous strain and velocities.

    The caller supplies its existing stage stem and suffix width: names seed
    the random streams and identify resumable output. Only the first chunk
    draws velocities; every chunk measures strain from the same reference.
    """
    stages: list[Stage] = []
    for index, steps in enumerate(
        resume_chunks(schedule.n_steps, schedule.relax_ps, stage_ps)
    ):
        options: dict[str, Any] = {
            "temperature_k": temperature_k,
            "pressure_bar": pressure_bar,
            "axis": axis,
            "strain_increment": schedule.increment,
            "n_steps": len(steps),
            "relax_ps": schedule.relax_ps,
            "strain_start": schedule.strain_after(steps.start),
            "samples_per_step": samples_per_step,
            "timestep_fs": timestep_fs,
            "new_velocities": index == 0,
        }
        if reference_box_nm is not None:
            options["reference_box_nm"] = list(reference_box_nm)
        if trajectory_ps is not None:
            options["trajectory"] = TrajectoryOptions("xtc", trajectory_ps)
        stages.append(Stage(f"{stem}_{index:0{chunk_digits}d}", "deform", options))
    return tuple(stages)


def with_reference_box(
    stages: Iterable[Stage], reference_box_nm: Sequence[float] | None
) -> tuple[Stage, ...]:
    """Tell every chunk of a strained branch the cell its strain is measured from.

    A state file carries the cell but not its origin, so a resumed chunk would
    otherwise open an already-strained cell and call that the unstrained one.
    """
    if reference_box_nm is None:
        return tuple(stages)
    origin = [float(value) for value in reference_box_nm]
    return tuple(
        Stage(stage.name, stage.kind, {**stage.options, "reference_box_nm": origin})
        for stage in stages
    )


def scan_listing(name: str, protocols: Iterable[Protocol]) -> Protocol:
    """Every stage a branched scan runs, as one protocol to price or inspect.

    Not one to run: run as a single protocol, the branches would follow each
    other instead of each starting from the equilibrated cell.
    """
    return Protocol(
        name, tuple(stage for protocol in protocols for stage in protocol.stages)
    )


def spec_request(
    spec: Any, *, drop: Iterable[str] = (), **extra: Any
) -> dict[str, Any]:
    """What a scan was asked for, in the form its workflow record stores it.

    Round-tripped through JSON before it is compared with anything, because a
    tuple comes back from a file as a list, and comparing the two directly
    would make every resume look like a change of settings. *drop* leaves out
    settings that do not change the dynamics, such as a budget.
    """
    dropped = set(drop)
    settings = {key: value for key, value in asdict(spec).items() if key not in dropped}
    stored: dict[str, Any] = json.loads(
        json.dumps({"spec": settings, **extra}, default=str)
    )
    return stored


def check_request(
    path: Path, request: dict[str, Any], *, error: type[Exception]
) -> dict[str, Any]:
    """The workflow record at *path*, refusing one made under another request.

    The manifest's provenance already refuses to resume a stage whose own
    options changed, but only once that stage is asked for again, and only for
    what a stage is told. A request is wider. It holds the settings the
    analysis reads, and the passes and replicas that a smaller request would
    simply not ask for again - leaving the old ones in the manifest for the
    analysis to find. So the whole request is compared here, before any
    dynamics, and the refusal names what changed.
    """
    if not path.is_file():
        return {}
    record: dict[str, Any] = json.loads(path.read_text())
    previous = record.get("request")
    if previous is not None and previous != request:
        raise error(
            f"{path} records a scan run with different settings "
            f"({', '.join(_changed(previous, request)) or 'unknown'}), and "
            "resuming would keep results measured under the old ones. Run into "
            "a fresh directory, put the settings back, or rerun with resume=False."
        )
    return record


def _changed(previous: dict[str, Any], request: dict[str, Any]) -> list[str]:
    """The settings two requests disagree on, a spec's by field name."""
    before, after = previous.get("spec", {}), request.get("spec", {})
    changed = {
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    }
    changed |= {
        key
        for key in (previous.keys() | request.keys()) - {"spec"}
        if previous.get(key) != request.get(key)
    }
    return sorted(changed)


def remaining_ps(stages: Iterable[Stage], manifest: RunManifest | None) -> float:
    """How much of *stages* the manifest does not already record as done."""
    done = set() if manifest is None else set(manifest.stages)
    return sum(stage.duration_ps for stage in stages if stage.name not in done)


def equilibrate(
    settle: Protocol,
    run: RunContext,
    directory: Path,
    *,
    resume: bool,
    error: type[Exception],
    verb: str,
    **chains: Any,
) -> tuple[str, list[float]]:
    """Run the equilibration a scan branches from; return its state and cell.

    *chains* goes to :func:`~openmmpolymer.protocols.run_protocol`, and
    *error* and *verb* to :func:`settled_state`.
    """
    summary = run_protocol(settle, run, directory, resume=resume, **chains)
    state = settled_state(summary, directory, error=error, verb=verb)
    box_nm = equilibrated_box_nm(state)
    log.info(
        "Equilibrated cell is %s nm; every branch starts from %s.",
        [round(value, 4) for value in box_nm],
        Path(state).name,
    )
    return state, box_nm


def run_branches(
    branches: Iterable[Protocol],
    run: RunContext,
    directory: Path,
    state_in: str,
    **chains: Any,
) -> None:
    """Run each branch from the equilibrated state, keeping what is recorded.

    Always resuming: a forced rerun resets the manifest once, in the
    equilibration, and every branch after that has to keep what the
    equilibration and the branches before it recorded.
    """
    for protocol in branches:
        run_protocol(protocol, run, directory, resume=True, state_in=state_in, **chains)


def record_scan_request(
    workflow: Path,
    record: dict[str, Any],
    request: dict[str, Any],
    runs: Sequence[Path],
    *,
    resume: bool,
    **metadata: Any,
) -> None:
    """Save a checked request before dynamics, clearing runs it replaces.

    Call after request, input and budget checks. A forced rerun clears every
    participating manifest before saving its new request: interruption before
    any branch starts must not pair new settings with old completed replicas.
    Only the supplied manifests are removed; states and unrelated runs remain.
    """
    if not resume:
        for directory in runs:
            (directory / MANIFEST_NAME).unlink(missing_ok=True)
    workflow.parent.mkdir(parents=True, exist_ok=True)
    record.update(request=request, **metadata)
    write_json(workflow, record, strict=False)


def scan_request(
    run: RunContext, spec: Any, settle: Protocol, **chains: Any
) -> dict[str, Any]:
    """The measurement, preparation and starting inputs of a complete scan."""
    return spec_request(
        spec,
        equilibration=[asdict(stage) for stage in settle.stages],
        **run_fingerprint(run),
        **chains,
    )


def run_branched_scan(
    run: RunContext,
    workflow: Path,
    request: dict[str, Any],
    settle: Protocol,
    branches: Callable[[Sequence[float]], Iterable[Protocol]],
    *,
    resume: bool,
    error: type[Exception],
    verb: str,
    metadata: dict[str, Any],
    **chains: Any,
) -> None:
    """Record, equilibrate and run a budgeted scan in one manifest.

    Check the complete request and starting inputs before writing anything.
    Save the request before equilibration and the reference cell before any
    branch, so interruptions during dynamics or analysis cannot leave stages
    without their settings. A forced rerun discards the old manifest before
    replacing its request; only equilibration resets it, and the branches
    preserve each other's results.
    """
    directory = workflow.parent
    record = check_request(workflow, request, error=error) if resume else {}
    if resume:
        manifest = RunManifest.load(directory)
        if manifest is not None:
            if manifest.protocol != settle.name:
                raise error(
                    f"{directory} contains a different protocol; use a fresh directory."
                )
            if record.get("request") is None:
                raise error(
                    f"{directory} already holds runs without a request in "
                    f"{workflow.name}, so their settings cannot be verified. "
                    "Use a fresh directory or rerun with resume=False."
                )
            validate_run_inputs(run, directory)
    record_scan_request(
        workflow, record, request, [directory], resume=resume, **metadata
    )
    start, origin = equilibrate(
        settle, run, directory, resume=resume, error=error, verb=verb, **chains
    )
    record.update(start_state=start, reference_box_nm=origin)
    write_json(workflow, record, strict=False)
    run_branches(branches(origin), run, directory, start, **chains)


def optional[T](read: Callable[[], T], notes: list[str], what: str) -> T | None:
    """Run a reader, turning "there is nothing to read" into a note.

    A measurement that was skipped and one that broke look the same from
    outside, and neither is a reason to fail the rest of a report: the note
    says which one is missing and why, and the report goes on without it.
    """
    try:
        return read()
    except AnalysisError as error:
        notes.append(f"{what}: {error}")
        return None


def write_report_files(
    run_dir: str,
    output_dir: str | Path | None,
    name: str,
    fields: dict[str, Any],
    figures: Iterable[tuple[str, Figure]],
    figure_format: str,
) -> ReportFiles:
    """Write a report's record and figures into ``<run_dir>/analysis``.

    Or into *output_dir*, for a run directory that should not be touched. The
    record opens with the version that wrote it and the versions that produced
    the run, so a surprising number can be placed. Its *fields* are spelled
    out by each workflow rather than taken from ``asdict``, which drops
    properties, renders arrays as strings and would tie what is on disk to how
    the dataclasses happen to be laid out. An undefined diagnostic is written
    as null, as every report writes it. Each ``(stem, figure)`` pair is saved
    as ``<stem>.<figure_format>``, in the order *figures* yields them.
    """
    from . import __version__

    directory = Path(run_dir) / "analysis" if output_dir is None else Path(output_dir)
    manifest = RunManifest.load(run_dir)
    record = {
        "openmmpolymer": __version__,
        "run_dir": run_dir,
        "versions": {} if manifest is None else manifest.versions,
        **fields,
    }
    return write_report(directory, name, record, figures, figure_format)


# --------------------------------------------------------------------------
# Rate scans: one equilibration, every rate and replica branched from it
# --------------------------------------------------------------------------


def chain_options(
    chain_backbone: Sequence[int] | None,
    atoms_per_chain: int | None,
    expected_characteristic_ratio: float,
) -> dict[str, Any]:
    """The chain-measurement keywords a scan passes to every protocol it runs."""
    return {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }


def resumable_record(
    run: RunContext,
    workflow: Path,
    request: dict[str, Any],
    branches: Sequence[str],
    *,
    resume: bool,
    error: type[Exception],
    fingerprinted: bool = True,
) -> dict[str, Any]:
    """The saved record a branched scan goes on from, once it is safe to.

    A forced rerun returns an empty record without checking the previous run;
    :func:`record_scan_request` then clears its manifests before saving it.
    On a resume, the manifests say what each run did, not that the runs belong
    together, so this refuses before anything is written: a record of another
    request; runs in the directory that no record accounts for;
    completed stages whose saved states are gone, *branches* whose common
    starting state is missing, changed or was never fingerprinted, or a
    *run* other than the one the equilibration recorded. Resuming any of them
    would mix branches that did not all start from one verified cell.
    *fingerprinted* False admits a record that predates the fingerprint: each
    branch stage's own provenance still ties it to the state it started from.

    Returns the record to extend: empty, unless resuming.
    """
    if not resume:
        return {}
    directory = workflow.parent
    record = check_request(workflow, request, error=error)
    if not record and directory.is_dir() and any(directory.rglob(MANIFEST_NAME)):
        raise error(
            f"{directory} already holds runs but no {workflow.name}, so their "
            "settings cannot be verified. Use a fresh directory."
        )
    runs = [directory / "equilibration", *(directory / name for name in branches)]
    for path in runs:
        manifest = path / MANIFEST_NAME
        if manifest.is_file() and any(
            not Path(entry.get("final_state", "")).is_file()
            for entry in json.loads(manifest.read_text()).get("stages", {}).values()
        ):
            raise error(
                f"{path} has completed stages with missing states; restore them "
                "or rerun with resume=False."
            )
    fingerprint = record.get("start_state_sha256")
    if fingerprint is not None:
        start = Path(record.get("start_state", ""))
        if not start.is_file() or file_sha256(start) != fingerprint:
            raise error(
                "The common preparation state is missing or no longer matches its "
                "recorded fingerprint; restore it or rerun with resume=False."
            )
    elif fingerprinted and any((path / MANIFEST_NAME).is_file() for path in runs[1:]):
        raise error(
            "Existing rate branches have no preparation-state fingerprint; rerun "
            "with resume=False so every branch starts from one verified state."
        )
    validate_run_inputs(run, runs[0])
    return record


def start_fingerprint(
    state: str, record: dict[str, Any], *, error: type[Exception]
) -> str:
    """The digest of the state every branch starts from, refusing a changed one.

    An equilibration that had to run again on a resume would otherwise hand
    the branches still to come a different cell from the ones measured.
    """
    digest = file_sha256(state)
    if record.get("start_state_sha256", digest) != digest:
        raise error(
            "The common preparation state changed; restore it or rerun with "
            "resume=False."
        )
    return digest


def strain_ladder(
    strain_start: float, strain_increment: float, n_steps: int
) -> npt.NDArray[np.float64]:
    """The strains a deformation stage records, compounding from *strain_start*.

    Each increment scales a cell the one before it already scaled, so step
    *i* of the stage ends at ``(1 + start) (1 + increment)**i - 1``.
    """
    steps = np.arange(1, n_steps + 1)
    return np.asarray(
        (1.0 + strain_start) * (1.0 + strain_increment) ** steps - 1.0,
        dtype=np.float64,
    )


def require_distinct(directories: Sequence[Path], *, what: str) -> None:
    """Refuse an empty series, or one that names a run directory twice.

    Named twice - directly, or once directly and once through its scan's
    record - a run's replicas would count twice, as if independent.
    """
    if not directories:
        raise AnalysisError(f"Supply run directories containing {what}.")
    seen: set[Path] = set()
    for directory in directories:
        if directory in seen:
            raise AnalysisError(f"{directory} was supplied more than once.")
        seen.add(directory)
