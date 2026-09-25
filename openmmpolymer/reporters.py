"""What a stage writes while it runs.

Two state-data reporters rather than one, because the two audiences want
incompatible files. ``StateDataReporter`` renders progress as ``20.0%`` and a
not-yet-known remaining time as ``--``, so a file carrying those columns is not
a table of numbers however it is parsed - and it refuses to write progress at
all without ``totalSteps``. The machine-readable CSV therefore carries only
numeric columns, and the human log carries the rest. Restart states are
written through :class:`AtomicStateReporter`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Self

from ._validation import require_choice

log = logging.getLogger(__name__)

#: Trajectory formats. ``xtc`` is the default: it is compressed, and a melt run
#: long enough to be interesting writes tens of gigabytes as DCD.
TRAJECTORY_FORMATS = ("xtc", "dcd", "pdb", "none")

#: The numeric CSV's columns: the ``StateDataReporter`` flags that are switched
#: on for it. Every one is a number, so
#: ``numpy.genfromtxt(..., delimiter=",", names=True)`` reads the file.
CSV_COLUMNS = (
    "step",
    "time",
    "potentialEnergy",
    "kineticEnergy",
    "totalEnergy",
    "temperature",
    "volume",
    "density",
)


class _TrajectoryFields(NamedTuple):
    """The fields of :class:`TrajectoryOptions`, which checks them."""

    format: str = "xtc"
    interval_ps: float | None = None
    enforce_periodic_box: bool = False


class TrajectoryOptions(_TrajectoryFields):
    """How a stage writes its trajectory.

    A tuple and not a dataclass because a stage's options are fingerprinted
    into its manifest, and a tuple is what every run already on disk recorded.

    Args:
        format: One of :data:`TRAJECTORY_FORMATS`.
        interval_ps: Time between frames. None means ten times the state-data
            interval, which is also what a stage naming a bare format string
            gets.
        enforce_periodic_box: Whether to wrap molecules into the cell. Off by
            default: wrapping splits a chain that straddles a face, and a split
            chain has a meaningless radius of gyration.

    Raises:
        ValueError: *format* is not one of :data:`TRAJECTORY_FORMATS`.
    """

    __slots__ = ()

    def __new__(
        cls,
        format: str = "xtc",
        interval_ps: float | None = None,
        enforce_periodic_box: bool = False,
    ) -> Self:
        """Refuse a format nothing writes, before a stage writes the wrong one."""
        require_choice(format, TRAJECTORY_FORMATS, name="format")
        return super().__new__(cls, format, interval_ps, enforce_periodic_box)


@dataclass(frozen=True)
class ReporterPaths:
    """Where a stage's output went.

    Args:
        csv: The numeric state data.
        log: The human-readable progress log.
        trajectory: The trajectory, if one was written.
        topology: The topology written beside a binary trajectory. None for a
            PDB trajectory, which carries its own.
        state: The portable restart state.
    """

    csv: str
    log: str
    trajectory: str | None
    topology: str | None
    state: str


class AtomicStateReporter:
    """Writes a portable restart state, alternating between two files.

    ``CheckpointReporter`` overwrites in place, so a crash during a write
    destroys the only checkpoint there was - a poor way to end a three-day
    run. Alternating between two files and recording which is current only
    after the write completes means there is always one good state on disk.

    Serialised state rather than a binary checkpoint because a checkpoint is
    tied to the platform that wrote it, and a run that has to move from a GPU
    queue to a CPU one should not have to start again.
    """

    def __init__(self, prefix: str | Path, interval: int) -> None:
        """Set up the reporter.

        Args:
            prefix: Stem for the two state files and the pointer beside them.
            interval: Steps between writes.
        """
        self._prefix = Path(prefix)
        self._interval = interval
        self._next = "a"

    @property
    def pointer_path(self) -> Path:
        """The file naming whichever state file is current."""
        return self._prefix.with_suffix(".state.which")

    def state_path(self, slot: str) -> Path:
        """Return the path of state file *slot*."""
        return self._prefix.with_suffix(f".state.{slot}.xml")

    def current_state(self) -> Path | None:
        """Return the last state fully written, or None if there is none."""
        if not self.pointer_path.is_file():
            return None
        candidate = self.state_path(self.pointer_path.read_text().strip())
        return candidate if candidate.is_file() else None

    def describeNextReport(self, simulation: Any) -> dict[str, Any]:
        """Tell OpenMM when this reporter next wants to be called."""
        steps = self._interval - simulation.currentStep % self._interval
        return {"steps": steps, "periodic": None, "include": []}

    def report(self, simulation: Any, state: Any) -> None:
        """Write the state, then point at it."""
        del state
        slot = self._next
        simulation.saveState(str(self.state_path(slot)))
        # Only now is the file complete, so only now does the pointer move.
        self.pointer_path.write_text(slot)
        self._next = "b" if slot == "a" else "a"


def steps_for(duration_ps: float, timestep_fs: float) -> int:
    """Return the number of steps covering *duration_ps*.

    Args:
        duration_ps: How long the stage should run.
        timestep_fs: The integration timestep.

    Returns:
        At least one step.
    """
    return max(1, round(duration_ps * 1000.0 / timestep_fs))


def rotate_existing(path: str | Path) -> str | None:
    """Move an existing file aside, returning where it went.

    ``XTCReporter`` refuses to open a file that already exists and is not
    empty, so a stage restarted after a crash dies while building its
    reporters rather than while running. Nothing is deleted: a partial
    trajectory is still evidence.

    Args:
        path: The file that is about to be written.

    Returns:
        Where the old file was moved, or None if there was none.
    """
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return None
    attempt = 1
    while True:
        moved = target.with_suffix(f".attempt{attempt}{target.suffix}")
        if not moved.exists():
            os.replace(target, moved)
            log.info("Moved the existing %s aside to %s.", target.name, moved.name)
            return str(moved)
        attempt += 1


def _path_for(output_prefix: str | Path, trajectory_format: str) -> Path:
    """Where a stage's trajectory goes, for *trajectory_format*.

    ``<stem>.<format>``, except that a PDB trajectory goes to
    ``<stem>_trajectory.pdb``: ``<stem>.pdb`` is the stage's closing
    structure, and two writers sharing it left a file that was neither a
    trajectory nor reliably a snapshot. The suffix follows
    ``<stem>_topology.pdb``, written beside a binary trajectory.

    Args:
        output_prefix: Stem for every file a stage writes.
        trajectory_format: One of :data:`TRAJECTORY_FORMATS`.

    Returns:
        The path to write the trajectory to.
    """
    prefix = Path(output_prefix)
    if trajectory_format == "pdb":
        return prefix.with_name(f"{prefix.name}_trajectory.pdb")
    return prefix.with_suffix(f".{trajectory_format}")


@contextmanager
def reporting(
    simulation: Any,
    output_prefix: str | Path,
    *,
    total_steps: int,
    report_interval: int,
    trajectory: TrajectoryOptions | str = "xtc",
    trajectory_interval: int | None = None,
) -> Iterator[ReporterPaths]:
    """Attach a stage's reporters, and take them down again afterwards.

    Args:
        simulation: The simulation to report on.
        output_prefix: Stem for every file this stage writes.
        total_steps: How many steps the stage will run. Required: without it
            ``StateDataReporter`` raises rather than omitting the progress
            column. A tenth of it is the interval between restart states, so a
            stage always leaves several.
        report_interval: Steps between state-data rows.
        trajectory: Trajectory settings, or just a format name.
        trajectory_interval: Steps between frames. Defaults to ten times
            *report_interval*.

    Yields:
        Where everything was written.

    Raises:
        ValueError: The trajectory format is not one of
            :data:`TRAJECTORY_FORMATS`.
    """
    from openmm import app

    options = (
        TrajectoryOptions(format=trajectory)
        if isinstance(trajectory, str)
        else trajectory
    )
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frames = trajectory_interval or report_interval * 10

    csv_path = prefix.with_suffix(".csv")
    log_path = prefix.with_suffix(".log")
    trajectory_path: Path | None = None
    topology_path: Path | None = None
    state_reporter = AtomicStateReporter(prefix, max(1, total_steps // 10))

    with ExitStack() as stack:
        # The file objects are opened here rather than inside the reporters so
        # that closing them is this context manager's job and not the garbage
        # collector's.
        csv_handle = stack.enter_context(csv_path.open("w"))
        log_handle = stack.enter_context(log_path.open("w"))

        simulation.reporters.append(
            app.StateDataReporter(
                csv_handle, report_interval, **dict.fromkeys(CSV_COLUMNS, True)
            )
        )
        simulation.reporters.append(
            app.StateDataReporter(
                log_handle,
                report_interval,
                step=True,
                time=True,
                temperature=True,
                density=True,
                speed=True,
                progress=True,
                remainingTime=True,
                totalSteps=total_steps,
            )
        )

        if options.format != "none":
            trajectory_path = _path_for(prefix, options.format)
            rotate_existing(trajectory_path)
            if options.format in {"xtc", "dcd"}:
                # Neither format carries a topology, and this is written now
                # rather than at the end because a run that crashes is exactly
                # the one whose trajectory has to stay readable.
                topology_path = prefix.with_name(f"{prefix.name}_topology.pdb")
                with topology_path.open("w") as handle:
                    app.PDBFile.writeFile(
                        simulation.topology,
                        simulation.context.getState(getPositions=True).getPositions(),
                        handle,
                    )
            writer = {
                "xtc": app.XTCReporter,
                "dcd": app.DCDReporter,
                "pdb": app.PDBReporter,
            }[options.format]
            simulation.reporters.append(
                writer(
                    str(trajectory_path),
                    frames,
                    enforcePeriodicBox=options.enforce_periodic_box,
                )
            )

        simulation.reporters.append(state_reporter)
        try:
            yield ReporterPaths(
                csv=str(csv_path),
                log=str(log_path),
                trajectory=None if trajectory_path is None else str(trajectory_path),
                topology=None if topology_path is None else str(topology_path),
                state=str(state_reporter.pointer_path),
            )
        finally:
            simulation.reporters.clear()
