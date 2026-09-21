"""What a stage writes while it runs.

Two state-data reporters rather than one, because the two audiences want
incompatible files. ``StateDataReporter`` renders progress as ``20.0%`` and a
not-yet-known remaining time as ``--``, so a file carrying those columns is not
a table of numbers however it is parsed - and it refuses to write progress at
all without ``totalSteps``. The machine-readable CSV therefore carries only
numeric columns, and the human log carries the rest.

The checkpoint is written through :class:`AtomicStateReporter` rather than
``CheckpointReporter``, which overwrites its target in place: a crash during
the write leaves no usable checkpoint at all, which is a poor way to end a
three-day run.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

log = logging.getLogger(__name__)

#: Trajectory formats. ``xtc`` is the default: it is compressed, and a melt run
#: long enough to be interesting writes tens of gigabytes as DCD.
TRAJECTORY_FORMATS = ("xtc", "dcd", "pdb", "none")

#: The numeric CSV's columns, in order. Every one is a number, so
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


class TrajectoryOptions(NamedTuple):
    """How a stage writes its trajectory.

    Args:
        format: One of :data:`TRAJECTORY_FORMATS`.
        interval_ps: Time between frames. None means ten times the state-data
            interval, which is what a stage gets when it names a format as a
            bare string rather than building one of these.
        enforce_periodic_box: Whether to wrap molecules into the cell. Off by
            default: wrapping splits a chain that straddles a face, and a split
            chain has a meaningless radius of gyration.
    """

    format: str = "xtc"
    interval_ps: float | None = None
    enforce_periodic_box: bool = False


@dataclass(frozen=True)
class ReporterPaths:
    """Where a stage's output went.

    Args:
        csv: The numeric state data.
        log: The human-readable progress log.
        trajectory: The trajectory, if one was written.
        topology: The topology written beside a binary trajectory.
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
    destroys the only checkpoint there was. Alternating between two files and
    recording which is current only after the write completes means there is
    always one good state on disk.

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


def _trajectory_reporter(path: Path, interval: int, options: TrajectoryOptions) -> Any:
    """Build the trajectory reporter for *options*."""
    from openmm import app

    if options.format == "xtc":
        return app.XTCReporter(
            str(path), interval, enforcePeriodicBox=options.enforce_periodic_box
        )
    if options.format == "dcd":
        return app.DCDReporter(
            str(path), interval, enforcePeriodicBox=options.enforce_periodic_box
        )
    return app.PDBReporter(
        str(path), interval, enforcePeriodicBox=options.enforce_periodic_box
    )


@contextmanager
def reporting(
    simulation: Any,
    output_prefix: str | Path,
    *,
    total_steps: int,
    report_interval: int,
    trajectory: TrajectoryOptions | str = "xtc",
    trajectory_interval: int | None = None,
    state_interval: int | None = None,
) -> Iterator[ReporterPaths]:
    """Attach a stage's reporters, and take them down again afterwards.

    Args:
        simulation: The simulation to report on.
        output_prefix: Stem for every file this stage writes.
        total_steps: How many steps the stage will run. Required: without it
            ``StateDataReporter`` raises rather than omitting the progress
            column.
        report_interval: Steps between state-data rows.
        trajectory: Trajectory settings, or just a format name.
        trajectory_interval: Steps between frames. Defaults to ten times
            *report_interval*.
        state_interval: Steps between restart states. Defaults to
            *total_steps* divided by ten, so a stage always leaves several.

    Yields:
        Where everything was written.
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
    states = state_interval or max(1, total_steps // 10)

    csv_path = prefix.with_suffix(".csv")
    log_path = prefix.with_suffix(".log")
    trajectory_path: Path | None = None
    topology_path: Path | None = None
    state_reporter = AtomicStateReporter(prefix, states)

    with ExitStack() as stack:
        # The file objects are opened here rather than inside the reporters so
        # that closing them is this context manager's job and not the garbage
        # collector's.
        csv_handle = stack.enter_context(csv_path.open("w"))
        log_handle = stack.enter_context(log_path.open("w"))

        simulation.reporters.append(
            app.StateDataReporter(
                csv_handle,
                report_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
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
            trajectory_path = prefix.with_suffix(f".{options.format}")
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
            simulation.reporters.append(
                _trajectory_reporter(trajectory_path, frames, options)
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
