"""The one place a finished run becomes something measurable.

Five facts about what this package writes shape everything here.

A stage's file stem is not its name. :func:`~openmmpolymer.simulate.run_pushoff`
runs three sub-stages into ``01_pushoff_0..2`` and reports them as one
``01_pushoff`` whose files are the last one's, so ``run/01_pushoff.csv`` does not
exist. The stem is therefore read out of the manifest rather than built by
appending a suffix to a stage name.

Most runs have no trajectory. ``run_segments`` defaults to ``trajectory="none"``
and neither shipped protocol names a production stage, so the ordinary case is a
single frame - and every stage writes ``<stem>.pdb``, unwrapped, with the box in
its ``CRYST1`` record. A one-frame :class:`Ensemble` off that snapshot is the
fallback, not an error.

Only the binary formats are read back. A stage asked for ``pdb`` writes a real
multi-frame trajectory to ``<stem>_trajectory.pdb``, and it is deliberately not
read here: the format records no time at all, so every lag in a displacement or
a relaxation would be a fabrication. It is a format for looking at a run in a
viewer; ``xtc`` is the one for measuring it. A stage that wrote one is pointed
at ``xtc`` rather than quietly analysed as a single frame.

MDAnalysis supplies coordinates and box vectors; everything else comes from
OpenMM. The molecule partition, the atom count, the masses and the elements are
read from the topology PDB through :func:`~openmmpolymer.packing.read_pdb`, the
same reader the rest of the package uses, and the Universe is built with
``to_guess=()`` so that MDAnalysis never guesses a mass. Guessed masses would be
wrong anyway wherever :attr:`~openmmpolymer.mdsystem.SystemSpec.hydrogen_mass_amu`
has repartitioned them, and wrong for virtual sites.

A chain is index arithmetic, not connectivity. Every molecule is a copy of the
same chain, so atom ``i`` of molecule ``m`` is at ``m * atoms_per_chain + i``;
that is the invariant :func:`~openmmpolymer.protocols.chain_dimensions` already
indexes by, and what makes a per-chain ``backbone`` from
:func:`~openmmpolymer.chain.backbone_path` mean anything. Deriving molecules from
bonds instead would also break above 99 999 atoms, where OpenMM's PDB writer
switches atom serials to hexadecimal and CONECT records stop parsing.

Two side effects worth knowing. MDAnalysis writes hidden
``.<name>.xtc_offsets.npz`` and ``.<name>.xtc_offsets.lock`` files beside a
trajectory the first time it reads one, so opening a run modifies its directory.
And each stage builds its own ``Simulation``, so a trajectory's time origin is
the stage start, not the run start.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from openmm import unit

from ._validation import require_integer
from .packing import read_pdb
from .protocols import RunManifest

log = logging.getLogger(__name__)

#: Trajectory extensions this package reads back, in the order they are looked
#: for. ``pdb`` is deliberately absent: it carries no frame times, so nothing
#: here could put a lag on an axis. See this module's docstring.
READABLE_FORMATS = ("xtc", "dcd")

#: Ångström per nanometre. MDAnalysis reports both coordinates and box lengths
#: in Ångström; this module is the only place in the package that sees them.
_ANGSTROM_PER_NM = 10.0

#: Biggest position array a load will build without being asked twice, in
#: gibibytes. A long run of a large cell will not fit in memory, and saying so
#: with the arithmetic is kinder than being killed by the OOM reaper.
MAX_POSITIONS_GIB = 4.0


class AnalysisError(RuntimeError):
    """A run did not write what an analysis needs."""


@dataclass(frozen=True)
class StageFiles:
    """Where one stage's output actually went.

    Args:
        stage: The stage name, as the manifest records it.
        prefix: The file stem every path below is built on. Not always the
            stage name - see this module's docstring.
        csv: The numeric state-data CSV, if the stage wrote one.
        final_pdb: The end-of-stage structure.
        final_state: The end-of-stage serialised state.
        trajectory: The trajectory, if one was asked for and is readable.
        topology: The topology to read *trajectory* against.
        n_molecules: Molecules in the cell, as the run recorded them, or None
            for a manifest written before that was recorded.
        atoms_per_chain: Atoms in each one, likewise. Preferred over counting
            residues, because the run knows and the topology only implies.
    """

    stage: str
    prefix: str
    csv: str | None
    final_pdb: str | None
    final_state: str | None
    trajectory: str | None
    topology: str | None
    n_molecules: int | None = None
    atoms_per_chain: int | None = None


@dataclass(frozen=True)
class Frame:
    """One frame's coordinates, in this package's units.

    Args:
        index: Position in the trajectory, counting from zero.
        time_ps: Time within the stage.
        positions_nm: ``(n_atoms, 3)``, unwrapped.
        box_nm: The three cell edge lengths.
    """

    index: int
    time_ps: float
    positions_nm: npt.NDArray[np.float64]
    box_nm: npt.NDArray[np.float64]


@dataclass(frozen=True)
class Ensemble:
    """A stage's coordinates, and the block structure that makes them per-chain.

    Args:
        stage: Which stage this came from.
        topology: The OpenMM ``Topology``.
        universe: The MDAnalysis ``Universe`` the coordinates come from.
        n_chains: How many molecules.
        atoms_per_chain: Atoms in each one.
        masses_amu: One chain's per-atom masses, length *atoms_per_chain*.
        is_hydrogen: One chain's hydrogen mask, length *atoms_per_chain*.
        n_frames: How many frames there are. One for a snapshot.
        interval_ps: Time between frames. Zero for a snapshot.
        topology_path: Where the topology was read from.
        trajectory_path: Where the coordinates were read from, or None when
            they came from the topology file itself.
    """

    stage: str
    topology: Any
    universe: Any
    n_chains: int
    atoms_per_chain: int
    masses_amu: npt.NDArray[np.float64]
    is_hydrogen: npt.NDArray[np.bool_]
    n_frames: int
    interval_ps: float
    topology_path: str
    trajectory_path: str | None

    @property
    def is_snapshot(self) -> bool:
        """Whether this is one frame rather than a trajectory."""
        return self.trajectory_path is None or self.n_frames == 1

    @property
    def n_atoms(self) -> int:
        """Atoms in the whole cell."""
        return self.n_chains * self.atoms_per_chain

    def frames(
        self, *, start: int = 0, stop: int | None = None, stride: int = 1
    ) -> Iterator[Frame]:
        """Iterate over frames, converting to nanometres.

        Args:
            start: First frame to yield.
            stop: One past the last, or None for all of them.
            stride: Yield every *stride*-th frame.

        Yields:
            Each frame in turn.

        Raises:
            ValueError: *stride* is not a positive integer.
        """
        require_integer(stride, name="stride")
        last = self.n_frames if stop is None else min(stop, self.n_frames)
        for index, step in enumerate(self.universe.trajectory[start:last:stride]):
            yield Frame(
                index=start + index * stride,
                time_ps=self._time_ps(start + index * stride),
                positions_nm=np.asarray(self.universe.atoms.positions, dtype=np.float64)
                / _ANGSTROM_PER_NM,
                box_nm=np.asarray(step.dimensions[:3], dtype=np.float64)
                / _ANGSTROM_PER_NM,
            )

    def per_chain(
        self, positions_nm: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Reshape whole-cell positions to ``(n_chains, atoms_per_chain, 3)``.

        Args:
            positions_nm: ``(n_atoms, 3)`` for the whole cell.

        Returns:
            The same data, split by molecule.

        Raises:
            AnalysisError: The array does not hold this cell's atoms.
        """
        expected = self.n_atoms
        if positions_nm.shape[0] != expected:
            raise AnalysisError(
                f"Positions hold {positions_nm.shape[0]} atoms, but this cell "
                f"has {self.n_chains} chains of {self.atoms_per_chain} "
                f"({expected})."
            )
        return positions_nm.reshape(self.n_chains, self.atoms_per_chain, 3)

    def _time_ps(self, index: int) -> float:
        """Time of frame *index*, from the interval rather than the reader.

        Read off ``interval_ps`` rather than the frame's own ``time``, because
        a single-frame reader has no ``dt`` and warns when asked for one.
        """
        return (index + 1) * self.interval_ps


def load_manifest(run_dir: str | Path) -> RunManifest:
    """Read the manifest that makes a directory a run directory.

    Raises:
        AnalysisError: There is none.
    """
    directory = Path(run_dir)
    manifest = RunManifest.load(directory)
    if manifest is None:
        raise AnalysisError(
            f"No manifest in {directory}, so there is nothing to say what the "
            "run wrote. A run directory is one that holds manifest.json."
        )
    return manifest


def stage_record(manifest: RunManifest, name: str, directory: Path) -> dict[str, Any]:
    """What the manifest in *directory* recorded for one stage.

    Raises:
        AnalysisError: It recorded no stage of that name.
    """
    recorded = manifest.stages.get(name)
    if recorded is None:
        raise AnalysisError(
            f"The manifest in {directory} has no stage {name!r}. It records: "
            f"{', '.join(manifest.stages) or 'nothing'}."
        )
    return recorded


def stage_names(stage: str | Sequence[str]) -> tuple[str, ...]:
    """One stage name, or several read as one pass, as a tuple.

    Raises:
        AnalysisError: *stage* names nothing.
    """
    names = (stage,) if isinstance(stage, str) else tuple(stage)
    if not names:
        raise AnalysisError("No stage was named, so there is nothing to read.")
    return names


def stages_holding(
    run_dir: str | Path, holds: Callable[[dict[str, Any]], bool], what: str
) -> tuple[str, ...]:
    """Name every stage whose recorded samples *holds* accepts, in manifest order.

    Stages are found by what they recorded rather than by what they were
    called, so a ladder split into chunks for resume - or repeated as several
    replicas - is read without anything having to agree on names in advance.

    Args:
        run_dir: A directory a run wrote to.
        holds: Whether one stage's samples are the kind being looked for.
        what: What that kind is, for the refusal.

    Raises:
        AnalysisError: There is no manifest, or no stage in it qualifies.
    """
    stages = load_manifest(run_dir).stages
    found = tuple(
        name
        for name, recorded in stages.items()
        if holds(recorded.get("samples") or {})
    )
    if not found:
        raise AnalysisError(
            f"No stage in {Path(run_dir)} recorded {what}. It records: "
            f"{', '.join(stages) or 'nothing'}."
        )
    return found


def stage_files(run_dir: str | Path, stage: str | None = None) -> StageFiles:
    """Find what one stage of a run left on disk.

    Resolved through ``run/manifest.json`` rather than by appending suffixes to
    a stage name, because a stage's file stem is not always its name.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        stage: Which stage, or None for the last one the manifest records.

    Returns:
        Where that stage's output went.

    Raises:
        AnalysisError: There is no manifest, it records no stages, or it does
            not record the stage asked for.
    """
    directory = Path(run_dir)
    manifest = load_manifest(directory)
    if not manifest.stages:
        raise AnalysisError(
            f"The manifest in {directory} records no completed stages, so the "
            "run did not get far enough to leave anything to measure."
        )
    name = stage if stage is not None else list(manifest.stages)[-1]
    recorded = stage_record(manifest, name, directory)

    final_pdb = recorded.get("final_pdb")
    csv = recorded.get("csv")
    final_state = recorded.get("final_state")
    prefix = _prefix_of(name, final_pdb, csv, final_state, directory)
    trajectory, topology = _coordinate_paths(prefix)
    recorded_box = manifest.box or {}
    return StageFiles(
        stage=name,
        prefix=str(prefix),
        csv=csv,
        final_pdb=final_pdb,
        final_state=final_state,
        trajectory=trajectory,
        topology=topology,
        n_molecules=_recorded_int(recorded_box, "n_molecules"),
        atoms_per_chain=_recorded_int(recorded_box, "atoms_per_chain"),
    )


def open_run(
    run_dir: str | Path,
    stage: str | None = None,
    *,
    atoms_per_chain: int | None = None,
) -> Ensemble:
    """Open a stage of a run for analysis.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        stage: Which stage, or None for the last one.
        atoms_per_chain: Override the discovered block size, for a topology
            whose residues do not describe its molecules.

    Returns:
        The stage's coordinates and block structure.

    Raises:
        AnalysisError: The stage wrote no coordinates, the topology does not
            divide into equal molecules, the trajectory is empty, it has been
            wrapped into the cell, or it is too large to load.
    """
    files = stage_files(run_dir, stage)
    if files.topology is None:
        raise AnalysisError(
            f"Stage {files.stage!r} wrote no structure to read. Expected "
            f"{files.prefix}.pdb, which every stage writes at its end."
        )
    return _open_stage(files, atoms_per_chain=atoms_per_chain)


def _prefix_of(
    stage: str,
    final_pdb: str | None,
    csv: str | None,
    final_state: str | None,
    directory: Path,
) -> Path:
    """Work out a stage's file stem from whatever paths it recorded.

    Taken from a recorded path rather than from the stage name, because
    ``run_pushoff`` reports sub-stage files under the parent's name. Falling
    back to the name covers a manifest that recorded no paths at all.
    """
    for path, suffix in ((final_pdb, ".pdb"), (csv, ".csv"), (final_state, ".xml")):
        if path:
            candidate = Path(path)
            if suffix == ".xml":
                # <stem>.state.xml, so two suffixes come off.
                return candidate.with_suffix("").with_suffix("")
            return candidate.with_suffix("")
    return directory / stage


def _recorded_int(box: dict[str, Any], key: str) -> int | None:
    """Read one integer out of the manifest's box record, if it is there.

    A manifest is a file, and a file can be edited, so a value that is not a
    usable count is treated as absent rather than trusted.
    """
    value = box.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _coordinate_paths(prefix: Path) -> tuple[str | None, str | None]:
    """Find a stage's trajectory and the topology to read it against.

    A binary trajectory is read against the ``<stem>_topology.pdb`` written at
    stage start; with no trajectory, the end-of-stage ``<stem>.pdb`` is both.
    """
    for extension in READABLE_FORMATS:
        candidate = prefix.with_suffix(f".{extension}")
        if candidate.is_file():
            topology = prefix.with_name(f"{prefix.name}_topology.pdb")
            if topology.is_file():
                return str(candidate), str(topology)
            snapshot = prefix.with_suffix(".pdb")
            if snapshot.is_file():
                log.warning(
                    "%s has no %s beside it, so the final snapshot is being "
                    "used as its topology instead.",
                    candidate.name,
                    topology.name,
                )
                return str(candidate), str(snapshot)
            raise AnalysisError(
                f"{candidate} has no topology beside it. A trajectory carries "
                f"no topology of its own, so {topology.name} is needed to read "
                "it."
            )
    written_as_pdb = prefix.with_name(f"{prefix.name}_trajectory.pdb")
    if written_as_pdb.is_file():
        log.info(
            "%s holds this stage's frames, but a PDB trajectory records no "
            "frame times, so it is being read as the closing snapshot only. "
            "Run the stage with trajectory='xtc' to measure anything "
            "time-dependent.",
            written_as_pdb.name,
        )
    snapshot = prefix.with_suffix(".pdb")
    return None, str(snapshot) if snapshot.is_file() else None


def _open_stage(files: StageFiles, *, atoms_per_chain: int | None = None) -> Ensemble:
    """Build an :class:`Ensemble` from resolved paths."""
    assert files.topology is not None  # open_run checked this
    structure = read_pdb(files.topology)
    topology = structure.topology
    declared = atoms_per_chain if atoms_per_chain is not None else files.atoms_per_chain
    n_chains, block = _blocks(topology, declared)
    if declared is not None and files.n_molecules not in (None, n_chains):
        log.info(
            "%s: the manifest recorded %s molecules and the blocks come to %d.",
            files.stage,
            files.n_molecules,
            n_chains,
        )

    universe = _universe(files.topology, files.trajectory)
    n_frames = len(universe.trajectory)
    _check_size(universe.atoms.n_atoms, n_frames)

    masses, hydrogen = _chain_atoms(topology, block)
    interval_ps = _interval_ps(universe, files.trajectory)
    ensemble = Ensemble(
        stage=files.stage,
        topology=topology,
        universe=universe,
        n_chains=n_chains,
        atoms_per_chain=block,
        masses_amu=masses,
        is_hydrogen=hydrogen,
        n_frames=n_frames,
        interval_ps=interval_ps,
        topology_path=files.topology,
        trajectory_path=files.trajectory,
    )
    if not ensemble.is_snapshot:
        _check_unwrapped(ensemble)
    return ensemble


def _universe(topology_path: str, trajectory_path: str | None) -> Any:
    """Open a Universe for coordinates only.

    ``to_guess=()`` because every attribute worth having is read off the
    OpenMM topology instead, and guessing is where MDAnalysis warns. The
    warning filter is scoped to this call rather than left to the caller's
    environment, so a library user under ``-W error`` is not tripped by a
    message about a file this package wrote and they never asked about.
    """
    with warnings.catch_warnings():
        # Message-matched, never a blanket ignore: the diagnostics this module
        # relies on - an empty trajectory, a wrapped one - must still surface.
        #
        # The import is inside the block, not above it, because the first one
        # is where the noise is: MDAnalysis.coordinates imports every reader
        # eagerly, so `import MDAnalysis` reaches netCDF4 through TRJ and
        # emits
        #   RuntimeWarning: numpy.ndarray size changed, may indicate binary
        #   incompatibility. Expected 16 from C header, got 96 from PyObject
        # which is the same mismatch pyproject.toml already ignores for
        # forcefill, two levels down in a dependency and nothing this package
        # can act on. Held here as well as there so that a library user
        # running under -W error is not stopped by it.
        warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
        warnings.filterwarnings("ignore", message="Element information is missing")
        warnings.filterwarnings("ignore", message="Unit cell dimensions not found")
        # Observed on a run directory copied off a cluster, because the frame
        # offsets MDAnalysis cached beside the trajectory record its ctime:
        #   UserWarning: Reload offsets from trajectory
        #    ctime or size or n_atoms did not match
        # It recomputes them and carries on, so the only effect of letting this
        # through would be to break analysis of any run that had been moved.
        warnings.filterwarnings("ignore", message="Reload offsets from trajectory")

        import MDAnalysis as mda

        if trajectory_path is None:
            return mda.Universe(topology_path, to_guess=())
        try:
            return mda.Universe(topology_path, trajectory_path, to_guess=())
        except (OSError, ValueError) as error:
            # Both the failures that matter surface here rather than from any
            # check of our own. A trajectory with no frames in it fails while
            # the reader opens the file, reporting "XDR read error =
            # endoffile" - a true statement about the file and no help about
            # why it is empty. And a trajectory read against the wrong
            # topology raises a ValueError naming both atom counts, which is
            # better than anything worth writing here; it just should not
            # reach a caller as a bare ValueError.
            raise AnalysisError(
                f"{trajectory_path} could not be read: {error}. The usual "
                "cause is a trajectory with no frames in it, which is what a "
                "stage shorter than its own frame interval writes - lower "
                "interval_ps on its TrajectoryOptions."
            ) from error


def _blocks(topology: Any, atoms_per_chain: int | None) -> tuple[int, int]:
    """Return ``(n_chains, atoms_per_chain)``, checked against the residues.

    A block size from the caller or from the manifest is used as given and
    only checked for divisibility: the run knows what it packed, and a
    topology standing in for something else - a cell of single atoms used as
    dimers, say - would otherwise be partitioned wrongly.

    With nothing declared, the residues are the partition, since one chain is
    one molecule is one residue. That is checked rather than trusted, because
    the failure mode if it ever stopped holding is not an exception but every
    measurement silently averaging over the wrong atoms.
    """
    residues = list(topology.residues())
    n_atoms = topology.getNumAtoms()
    if not residues or n_atoms == 0:
        raise AnalysisError("The topology holds no atoms.")

    if atoms_per_chain is not None:
        block = require_integer(atoms_per_chain, name="atoms_per_chain")
        if n_atoms % block:
            raise AnalysisError(
                f"atoms_per_chain={block} does not divide the cell's {n_atoms} atoms."
            )
        return n_atoms // block, block

    block = n_atoms // len(residues)
    if n_atoms % len(residues):
        raise AnalysisError(
            f"{n_atoms} atoms do not divide evenly into {len(residues)} "
            "residues, so the cell is not copies of one chain. Pass "
            "atoms_per_chain to say what the blocks are."
        )
    # Only the atom count is checked. That each residue's atoms are one
    # contiguous ascending run - which is what makes the index arithmetic
    # valid - is guaranteed upstream: openmm.app.Topology.addAtom raises
    # "All atoms within a residue must be contiguous" rather than allowing it,
    # so a topology that reached here cannot violate it.
    for index, residue in enumerate(residues):
        count = sum(1 for _ in residue.atoms())
        if count != block:
            raise AnalysisError(
                f"Residue {index} has {count} atoms but residue 0 has "
                f"{block}. Every molecule should be a copy of the same chain."
            )
    return len(residues), block


def _chain_atoms(
    topology: Any, atoms_per_chain: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """One chain's masses and hydrogen mask, from the topology's elements."""
    masses: list[float] = []
    hydrogen: list[bool] = []
    for atom in list(topology.atoms())[:atoms_per_chain]:
        element = atom.element
        if element is None:
            raise AnalysisError(
                f"Atom {atom.index} has no element, so it has no mass. The "
                "topology PDB is missing its element column."
            )
        masses.append(float(element.mass.value_in_unit(unit.dalton)))
        hydrogen.append(element.symbol == "H")
    return (
        np.asarray(masses, dtype=np.float64),
        np.asarray(hydrogen, dtype=np.bool_),
    )


def _interval_ps(universe: Any, trajectory_path: str | None) -> float:
    """Time between frames, or zero when there is only a snapshot.

    A reader with one frame has no interval, and asking it for ``dt`` warns
    and answers 1.0 ps - a plausible-looking number that would make every
    time axis wrong. So it is never asked.
    """
    if trajectory_path is None or len(universe.trajectory) < 2:
        return 0.0
    return float(universe.trajectory.dt)


def _check_size(n_atoms: int, n_frames: int) -> None:
    """Refuse a load that will not fit, with the arithmetic in the message."""
    gib = n_atoms * n_frames * 3 * 8 / 1024**3
    if gib > MAX_POSITIONS_GIB:
        raise AnalysisError(
            f"{n_frames} frames of {n_atoms} atoms is {gib:.1f} GiB of "
            f"positions, over the {MAX_POSITIONS_GIB} GiB this will load. Pass "
            "a stride to the analysis, which reads frames one at a time."
        )


def _check_unwrapped(ensemble: Ensemble) -> None:
    """Warn if the trajectory looks wrapped into the cell.

    Stages write unwrapped coordinates on purpose, because wrapping splits a
    chain across a face and a split chain has a meaningless radius of
    gyration. A run made with ``enforce_periodic_box=True`` is still readable,
    and every conformational number off it is nonsense, so it is worth saying
    so once rather than returning a quiet wrong answer.
    """
    frames = list(ensemble.frames(start=0, stop=2))
    if len(frames) < 2:
        return
    step = np.abs(frames[1].positions_nm - frames[0].positions_nm).max()
    half = float(frames[0].box_nm.min()) / 2.0
    if step > half:
        log.warning(
            "%s: an atom moved %.2f nm between the first two frames, more than "
            "half the %.2f nm cell. The trajectory looks wrapped, and a "
            "wrapped chain has no meaningful radius of gyration or "
            "displacement.",
            ensemble.stage,
            step,
            half * 2.0,
        )


def chain_positions(
    ensemble: Ensemble, *, stride: int = 1, heavy_atoms_only: bool = False
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read every frame into one array, split by chain.

    Provided because most analyses want all the frames at once and the
    conversion, the reshape and the time axis are the same three lines each
    time.

    Args:
        ensemble: What to read.
        stride: Take every *stride*-th frame.
        heavy_atoms_only: Drop hydrogens from each chain.

    Returns:
        ``(n_frames, n_chains, atoms, 3)`` positions in nanometres, and the
        matching ``(n_frames,)`` times in picoseconds.
    """
    positions, times, _ = _load_chain_frames(
        ensemble, stride=stride, heavy_atoms_only=heavy_atoms_only
    )
    return positions, times


def _load_chain_frames(
    ensemble: Ensemble, *, stride: int = 1, heavy_atoms_only: bool = False
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read chain positions, times and cell edges in one trajectory pass.

    Returns:
        ``(n_frames, n_chains, atoms, 3)`` positions in nanometres,
        ``(n_frames,)`` times in picoseconds, and ``(n_frames, 3)`` cell edges
        in nanometres, all sampled at the same stride. Only positions are
        filtered by *heavy_atoms_only*.
    """
    keep = ~ensemble.is_hydrogen if heavy_atoms_only else None
    positions: list[npt.NDArray[np.float64]] = []
    times: list[float] = []
    boxes: list[npt.NDArray[np.float64]] = []
    for frame in ensemble.frames(stride=stride):
        chains = ensemble.per_chain(frame.positions_nm)
        positions.append(chains if keep is None else chains[:, keep])
        times.append(frame.time_ps)
        boxes.append(frame.box_nm)
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        np.asarray(boxes, dtype=np.float64),
    )


def boxes_nm(ensemble: Ensemble, *, stride: int = 1) -> npt.NDArray[np.float64]:
    """Return every frame's cell edges, ``(n_frames, 3)`` in nanometres.

    Args:
        ensemble: What to read.
        stride: Take every *stride*-th frame.

    Returns:
        The cell edges, frame by frame.
    """
    return np.asarray(
        [frame.box_nm for frame in ensemble.frames(stride=stride)], dtype=np.float64
    )


def capped_stride(n_frames: int, stride: int, cap: int | None) -> int:
    """The stride that keeps a measurement to at most *cap* of *n_frames*.

    Never finer than *stride*, and *stride* itself when there is no cap.

    Raises:
        TypeError: *cap* is not an integer.
        ValueError: *cap* is not positive.
    """
    if cap is None:
        return stride
    return max(stride, -(-n_frames // require_integer(cap, name="frame cap")))


def require_trajectory(ensemble: Ensemble, what: str) -> None:
    """Refuse a measurement that needs more than one frame.

    Args:
        ensemble: What was passed.
        what: The measurement's name, for the message.

    Raises:
        AnalysisError: *ensemble* is a single snapshot.
    """
    if ensemble.is_snapshot:
        raise AnalysisError(
            f"{what} needs a trajectory, and stage {ensemble.stage!r} has only "
            "a single frame. No shipped protocol writes a trajectory: add a "
            "production stage, or pass trajectory= to the stage's options."
        )


def backbone_indices(
    backbone: Sequence[int], atoms_per_chain: int
) -> npt.NDArray[np.int64]:
    """Validate chain-local backbone indices and return them as an array.

    Args:
        backbone: Backbone atom indices within one chain, in order.
        atoms_per_chain: Atoms per molecule.

    Returns:
        The indices.

    Raises:
        AnalysisError: The path is too short to measure, or indexes an atom
            the chain does not have.
    """
    path = np.asarray(backbone, dtype=np.int64)
    if path.size < 2:
        raise AnalysisError(
            f"A backbone of {path.size} atoms has no end-to-end vector. Pass "
            "the path from ChainResult.backbone."
        )
    if path.min() < 0 or path.max() >= atoms_per_chain:
        raise AnalysisError(
            f"Backbone indices run {path.min()}..{path.max()}, outside a chain "
            f"of {atoms_per_chain} atoms. They are indices within one chain, "
            "not into the whole cell."
        )
    return path
