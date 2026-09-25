"""Filling a periodic cell with polymer chains, using packmol.

Two decisions here are worth stating rather than discovering.

**Pack loose, compress later.** packmol places each structure as a rigid body
and will grind for hours trying to reach a melt density it cannot reach. The
default packing density is 0.3 g/cm3 and the compression stage takes the cell
the rest of the way; that is faster and far more reliable than asking packmol
for 0.9 directly.

**Inset the packing region from the periodic cell.** packmol's ``pbc`` keyword
packs across the boundary, which leaves molecules split between opposite faces
- and OpenMM's ``HarmonicBondForce`` does not use the minimum image, so a split
chain has one bond the length of the box and the first force evaluation
diverges. Keeping every molecule whole inside an inset region costs a little
usable volume and cannot produce that failure.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._validation import require_integer, require_positive

log = logging.getLogger(__name__)

#: Avogadro's number, for the density-to-box arithmetic.
AVOGADRO = 6.02214076e23

#: 1 cm^3 in nm^3.
NM3_PER_CM3 = 1.0e21

#: Angstrom per nanometre. RDKit, packmol and MDAnalysis work in angstrom;
#: OpenMM and this package work in nanometres.
ANGSTROM_PER_NM = 10.0

#: Default closest approach packmol enforces between atoms of different
#: molecules, in nanometres.
DEFAULT_TOLERANCE_NM = 0.2

#: Density to pack at, in g/cm3. See the module docstring.
DEFAULT_PACKING_DENSITY = 0.3

#: Seconds packmol is allowed by default.
PACKMOL_TIMEOUT_S = 3600.0

#: Environment variable naming the packmol executable, consulted before PATH.
PACKMOL_ENV_VAR = "PACKMOL"

#: packmol's own exit codes start here.
_PACKMOL_ERROR_BASE = 170

#: The longest bond :func:`check_packing` tolerates, in nanometres.
_MAX_BOND_NM = 0.25

#: The closest approach :func:`check_packing` tolerates between atoms of
#: different molecules, in nanometres: between heavy atoms, and where either
#: atom is a hydrogen.
_MIN_HEAVY_NM = 0.20
_MIN_HYDROGEN_NM = 0.15

#: The largest ring the ring-spearing check looks for.
_MAX_RING_SIZE = 8


class PackmolError(RuntimeError):
    """packmol could not be found, or could not pack the cell."""


@dataclass(frozen=True)
class PackedComponent:
    """One structure to place, and how many copies.

    Args:
        pdb_path: A single-molecule PDB. One conformer of a chain.
        count: How many copies to place.
    """

    pdb_path: str
    count: int = 1

    def __post_init__(self) -> None:
        """Reject a count that would make an empty or negative block."""
        require_integer(self.count, name="count", minimum=0)


@dataclass(frozen=True)
class PackResult:
    """What :func:`pack_box` produced.

    Args:
        packed_pdb: The packed cell. Carries coordinates and nothing else
            trustworthy: packmol writes no bonds, so the topology has to come
            from replicating the single-chain one.
        box_nm: The periodic cell edges.
        n_molecules: How many molecules were placed, in placement order.
        input_path: The generated packmol input, kept for inspection.
        log_path: packmol's output.
        seed: The seed packmol was given.
    """

    packed_pdb: str
    box_nm: tuple[float, float, float]
    n_molecules: int
    input_path: str
    log_path: str
    seed: int


def box_edge_nm(
    counts: Sequence[int],
    molar_masses_g_mol: Sequence[float],
    density_g_cm3: float,
) -> float:
    """Return the cubic cell edge holding this much material at this density.

    ``V = sum(N_i M_i) / (rho N_A)`` in cm3, converted to nm3.

    Args:
        counts: How many of each component.
        molar_masses_g_mol: Each component's molar mass.
        density_g_cm3: The target density.

    Returns:
        The cell edge in nanometres.

    Raises:
        ValueError: The density is not positive, or there is nothing to pack.
    """
    require_positive(density_g_cm3, None, name="density_g_cm3")
    total_mass = sum(
        count * mass for count, mass in zip(counts, molar_masses_g_mol, strict=True)
    )
    if total_mass <= 0.0:
        raise ValueError("Nothing to pack: every component count is zero.")
    volume_nm3 = total_mass * NM3_PER_CM3 / (density_g_cm3 * AVOGADRO)
    return float(volume_nm3 ** (1.0 / 3.0))


def distribute_conformers(
    pdb_paths: Sequence[str], n_molecules: int
) -> list[PackedComponent]:
    """Spread *n_molecules* copies over the conformers available.

    One conformer per molecule is the right thing and what
    :func:`openmmpolymer.chain.build_chain` is set up for. It is not always
    affordable: packmol takes a separate ``structure`` block per conformer,
    and several hundred of them is a long wait. Given fewer conformers than
    molecules, this repeats them as evenly as it can, which still beats
    packing one conformation many times over.

    Args:
        pdb_paths: The conformers, in any order.
        n_molecules: How many molecules the cell holds.

    Returns:
        One component per conformer, with counts summing to *n_molecules*.

    Raises:
        ValueError: There are no conformers, or nothing to place.
    """
    if not pdb_paths:
        raise ValueError("No conformers to pack.")
    require_integer(n_molecules, name="n_molecules")

    base, extra = divmod(n_molecules, len(pdb_paths))
    return [
        PackedComponent(path, count)
        for index, path in enumerate(pdb_paths)
        if (count := base + (1 if index < extra else 0))
    ]


def _find_packmol() -> str:
    """Return the packmol executable: ``$PACKMOL`` if it is set, else PATH's."""
    override = os.environ.get(PACKMOL_ENV_VAR)
    if override:
        resolved = shutil.which(override) or override
        if Path(resolved).is_file():
            return resolved
        raise PackmolError(f"{PACKMOL_ENV_VAR}={override!r} is not an executable file.")

    found = shutil.which("packmol")
    if found is None:
        raise PackmolError(
            "packmol is not on PATH. conda-forge ships it, both as its own "
            "package and inside ambertools: "
            "conda install -c conda-forge packmol. Or point at a build with "
            f"the {PACKMOL_ENV_VAR} environment variable."
        )
    return found


def _render_packmol_input(
    components: Sequence[PackedComponent],
    box_angstrom: tuple[float, float, float],
    output_pdb: str,
    *,
    tolerance_angstrom: float,
    inset_angstrom: float,
    seed: int,
) -> str:
    """Render a packmol input file.

    Everything here is in angstrom, and says so, because packmol works in the
    units of the PDB it is given while the rest of this package works in
    nanometres. A ten-fold slip in the tolerance packs atoms on top of each
    other; the same slip in the box is a cell a thousand times the wrong size.
    """
    lines = [
        f"tolerance {tolerance_angstrom:.4f}",
        "filetype pdb",
        f"output {output_pdb}",
        f"seed {seed}",
        "",
    ]
    low = inset_angstrom
    high = tuple(edge - inset_angstrom for edge in box_angstrom)
    for component in components:
        if component.count == 0:
            continue
        lines.extend(
            [
                f"structure {component.pdb_path}",
                f"  number {component.count}",
                # Number residues across the whole output rather than per
                # structure. This package writes one structure block per
                # conformer, and packmol's default restarts residue numbering
                # in each: past the twenty-sixth block the chain identifiers
                # run out too, and molecules start sharing a chain and residue
                # number. OpenMM then merges them - measured at 40 conformers,
                # 37 residues came back instead of 40 - after one warning that
                # is easy to miss.
                "  resnumbers 3",
                f"  inside box {low:.4f} {low:.4f} {low:.4f} "
                f"{high[0]:.4f} {high[1]:.4f} {high[2]:.4f}",
                "end structure",
                "",
            ]
        )
    return "\n".join(lines)


def pack_box(
    components: Sequence[PackedComponent],
    box_nm: float | tuple[float, float, float],
    output_pdb: str | Path = "packed.pdb",
    *,
    tolerance_nm: float = DEFAULT_TOLERANCE_NM,
    seed: int = 0xF0,
    timeout_s: float | None = PACKMOL_TIMEOUT_S,
    workdir: str | Path | None = None,
) -> PackResult:
    """Pack *components* into a periodic cell of *box_nm*.

    The binary is ``$PACKMOL`` if that is set, and otherwise the ``packmol``
    on PATH.

    Args:
        components: What to place, and how many of each.
        box_nm: Cubic edge, or the three edges.
        output_pdb: Where the packed cell is written.
        tolerance_nm: Closest approach between different molecules. The
            packing region is held back from each face by half of it, the
            least that keeps molecules on opposite faces a full tolerance
            apart across the periodic boundary.
        seed: packmol's random seed.
        timeout_s: Seconds to allow, or None for no limit.
        workdir: Where the input file and log are written. Defaults to the
            output's directory.

    Returns:
        What was packed.

    Raises:
        PackmolError: packmol is missing, failed, or did not converge.
        ValueError: The tolerance or a box edge is not positive.
    """
    require_positive(tolerance_nm, None, name="tolerance_nm")
    edges = (box_nm, box_nm, box_nm) if isinstance(box_nm, int | float) else box_nm
    for edge in edges:
        require_positive(edge, None, name="box_nm")
    inset = tolerance_nm / 2.0

    destination = Path(output_pdb)
    directory = Path(workdir) if workdir is not None else destination.parent
    directory.mkdir(parents=True, exist_ok=True)

    binary = _find_packmol()
    text = _render_packmol_input(
        [
            PackedComponent(str(Path(item.pdb_path).resolve()), item.count)
            for item in components
        ],
        tuple(edge * ANGSTROM_PER_NM for edge in edges),  # type: ignore[arg-type]
        str(destination.resolve()),
        tolerance_angstrom=tolerance_nm * ANGSTROM_PER_NM,
        inset_angstrom=inset * ANGSTROM_PER_NM,
        seed=seed,
    )
    input_path = directory / f"{destination.stem}.inp"
    log_path = directory / f"{destination.stem}.packmol.log"
    input_path.write_text(text)

    total = sum(item.count for item in components)
    log.info(
        "Packing %d molecules into a %.2f x %.2f x %.2f nm cell.",
        total,
        *edges,
    )
    output = _run_packmol(binary, input_path, timeout_s)
    log_path.write_text(output)

    if "Success!" not in output:
        raise PackmolError(
            f"packmol exited cleanly but did not report success. Its output is "
            f"in {log_path}. The usual cause is a cell too small for the "
            "molecules: pack at a lower density, or use fewer chains."
        )
    if not destination.is_file():  # pragma: no cover - packmol said Success
        raise PackmolError(f"packmol reported success but wrote no {destination}.")

    return PackResult(
        packed_pdb=str(destination),
        box_nm=(float(edges[0]), float(edges[1]), float(edges[2])),
        n_molecules=total,
        input_path=str(input_path),
        log_path=str(log_path),
        seed=seed,
    )


def _run_packmol(binary: str, input_path: Path, timeout_s: float | None) -> str:
    """Run packmol on *input_path*, which it reads from stdin.

    Returns:
        packmol's combined output.

    Raises:
        PackmolError: packmol failed or timed out.
    """
    with input_path.open() as handle:
        try:
            completed = subprocess.run(
                [binary],
                stdin=handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise PackmolError(
                f"packmol did not finish within {timeout_s} s. Pack at a lower "
                "density, or allow it longer."
            ) from error

    if completed.returncode != 0:
        detail = (
            " That is one of packmol's own failure codes, which mean it could "
            "not satisfy the constraints."
            if completed.returncode >= _PACKMOL_ERROR_BASE
            else ""
        )
        raise PackmolError(
            f"packmol exited with code {completed.returncode}.{detail}\n"
            f"{completed.stdout[-2000:]}"
        )
    return completed.stdout


def read_pdb(path: str | Path) -> Any:
    """Read a PDB through ``openmm.app.PDBFile``.

    ``PDBFile`` opens a file only when handed a ``str``; anything else it takes
    for an open file, so a :class:`~pathlib.Path` fails inside the parser.

    Args:
        path: The file to read.

    Returns:
        The parsed ``PDBFile``.
    """
    from openmm import app

    return app.PDBFile(str(path))


def read_packed_pdb(packed_pdb: str | Path) -> tuple[Any, npt.NDArray[np.float64]]:
    """Read packmol's output once, returning its topology and its positions.

    Read through ``openmm.app.PDBFile`` rather than by slicing columns: the
    coordinate fields are not where a naive slice puts them, and getting the z
    column off by one is a silent error of a few tenths of an angstrom.

    Args:
        packed_pdb: packmol's output.

    Returns:
        The parsed topology - bondless, since packmol writes no CONECT
        records - and an ``(n_atoms, 3)`` array of positions in nanometres.
    """
    from openmm import unit

    pdb = read_pdb(packed_pdb)
    positions = np.asarray(
        pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )
    return pdb.topology, positions


def check_packing(topology: Any, positions_nm: npt.NDArray[np.float64]) -> None:
    """Check a packed cell for the three faults that survive minimisation.

    A molecule split across the periodic boundary, two molecules on top of each
    other, and a bond threaded through a ring.

    Args:
        topology: The assembled box topology, with bonds.
        positions_nm: Its positions.

    Raises:
        PackmolError: The cell has a fault that would not survive a run.
    """
    _check_bond_lengths(topology, positions_nm)
    _check_contacts(topology, positions_nm)
    _check_ring_spearing(topology, positions_nm)


def _check_bond_lengths(topology: Any, positions_nm: npt.NDArray[np.float64]) -> None:
    """Raise if any bond is longer than :data:`_MAX_BOND_NM`.

    This is the test for a molecule split across the periodic boundary. OpenMM
    computes bonded terms without the minimum image, so a split molecule has
    one bond roughly the width of the cell and the first force evaluation
    produces an energy no minimiser recovers from.
    """
    worst = 0.0
    worst_pair: tuple[int, int] = (0, 0)
    for first, second in topology.bonds():
        delta = positions_nm[first.index] - positions_nm[second.index]
        length = float(np.sqrt(delta @ delta))
        if length > worst:
            worst, worst_pair = length, (first.index, second.index)
    if worst > _MAX_BOND_NM:
        raise PackmolError(
            f"Atoms {worst_pair[0]} and {worst_pair[1]} are bonded but "
            f"{worst:.3f} nm apart, against a limit of {_MAX_BOND_NM} nm. A "
            "molecule is split across the periodic boundary, or the packed "
            "coordinates do not correspond to this topology."
        )


class CellList:
    """Points bucketed on a uniform grid, for near-neighbour queries.

    Not periodic, which suits both users: a packed cell holds whole molecules,
    and a chain being grown has no boundary at all.
    """

    def __init__(self, spacing: float) -> None:
        self._spacing = spacing
        self._cells: dict[tuple[int, int, int], list[int]] = {}
        self.points: list[npt.NDArray[np.float64]] = []

    def _cell(self, point: npt.NDArray[np.float64]) -> tuple[int, int, int]:
        x, y, z = np.floor(point / self._spacing).astype(int)
        return int(x), int(y), int(z)

    def add(self, points: npt.NDArray[np.float64]) -> None:
        """Record *points*, indexed on from those already recorded."""
        for point in points:
            self._cells.setdefault(self._cell(point), []).append(len(self.points))
            self.points.append(point)

    def neighbours(self, point: npt.NDArray[np.float64]) -> list[int]:
        """Return the index of every recorded point in the 27 cells around *point*."""
        cx, cy, cz = self._cell(point)
        found: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    found.extend(self._cells.get((cx + dx, cy + dy, cz + dz), ()))
        return found


def _check_contacts(topology: Any, positions_nm: npt.NDArray[np.float64]) -> None:
    """Raise if two different molecules are closer than they should be.

    Only pairs in *different* molecules are considered. packmol's tolerance is
    an intermolecular constraint and says nothing about bonded neighbours, so a
    test that included them would fail on every C-H bond at 0.11 nm.
    """
    molecule = np.empty(topology.getNumAtoms(), dtype=np.int64)
    is_hydrogen = np.zeros(topology.getNumAtoms(), dtype=bool)
    for index, residue in enumerate(topology.residues()):
        for atom in residue.atoms():
            molecule[atom.index] = index
            is_hydrogen[atom.index] = atom.element is not None and (
                atom.element.atomic_number == 1
            )

    cells = CellList(max(_MIN_HEAVY_NM, _MIN_HYDROGEN_NM))
    cells.add(positions_nm)
    for index, point in enumerate(positions_nm):
        for other in cells.neighbours(point):
            if other <= index or molecule[other] == molecule[index]:
                continue
            delta = point - positions_nm[other]
            distance = float(np.sqrt(delta @ delta))
            limit = (
                _MIN_HYDROGEN_NM
                if is_hydrogen[index] or is_hydrogen[other]
                else _MIN_HEAVY_NM
            )
            if distance < limit:
                raise PackmolError(
                    f"Atoms {index} and {other} are in different molecules but "
                    f"{distance:.3f} nm apart, against a limit of {limit} nm. "
                    "Raise the packmol tolerance, or pack at a lower density."
                )


def _find_rings(
    topology: Any, neighbours: dict[int, list[int]]
) -> list[tuple[int, ...]]:
    """Return the rings in *topology* up to :data:`_MAX_RING_SIZE` atoms.

    Every molecule in a packed cell is a copy of the same chain, so the rings
    are found once on the first residue and shifted onto the rest. Falling back
    to a per-residue search when the residues differ in size keeps this honest
    for a mixed cell.
    """
    residues = [
        [atom.index for atom in residue.atoms()] for residue in topology.residues()
    ]
    counts = {len(indices) for indices in residues}
    if len(counts) != 1:
        return [
            ring for indices in residues for ring in _rings_within(neighbours, indices)
        ]

    base = _rings_within(neighbours, residues[0])
    stride = counts.pop()
    return [
        tuple(index + offset * stride for index in ring)
        for offset in range(len(residues))
        for ring in base
    ]


def _rings_within(
    neighbours: dict[int, list[int]], atoms: Sequence[int]
) -> list[tuple[int, ...]]:
    """Return the rings among *atoms*, by shortest-cycle search."""
    members = set(atoms)
    seen: set[frozenset[int]] = set()
    rings: list[tuple[int, ...]] = []
    for start in atoms:
        for neighbour in neighbours.get(start, ()):
            if neighbour not in members or neighbour < start:
                continue
            path = _shortest_path_avoiding(
                neighbours, neighbour, start, _MAX_RING_SIZE - 1, members
            )
            if path is None:
                continue
            ring = (start, *path)
            key = frozenset(ring)
            if key not in seen:
                seen.add(key)
                rings.append(ring)
    return rings


def _shortest_path_avoiding(
    neighbours: dict[int, list[int]],
    start: int,
    goal: int,
    max_length: int,
    members: set[int],
) -> tuple[int, ...] | None:
    """Return the shortest path start -> goal not using the direct bond.

    *goal* is never entered as an intermediate node, only reached as the last
    step. Letting the search pass through it instead returns paths that leave
    the goal and come back, which read as rings and are not.
    """
    queue: deque[tuple[int, tuple[int, ...]]] = deque([(start, (start,))])
    visited = {start, goal}
    while queue:
        node, path = queue.popleft()
        if len(path) > max_length:
            continue
        for neighbour in neighbours.get(node, ()):
            if neighbour == goal:
                if node != start:
                    return path
                continue
            if neighbour in visited or neighbour not in members:
                continue
            visited.add(neighbour)
            queue.append((neighbour, (*path, neighbour)))
    return None


def _check_ring_spearing(topology: Any, positions_nm: npt.NDArray[np.float64]) -> None:
    """Raise if a bond passes through a ring.

    The classic packmol failure for anything with an aromatic or aliphatic
    ring. A 2 angstrom tolerance guarantees no two atoms are on top of each
    other and says nothing about a chain segment threaded through a ring; the
    knot survives minimisation and every stage after it, permanently.
    """
    neighbours: dict[int, list[int]] = {}
    for first, second in topology.bonds():
        neighbours.setdefault(first.index, []).append(second.index)
        neighbours.setdefault(second.index, []).append(first.index)
    rings = _find_rings(topology, neighbours)
    if not rings:
        return

    centres: list[npt.NDArray[np.float64]] = []
    normals: list[npt.NDArray[np.float64]] = []
    radii: list[float] = []
    excluded: list[set[int]] = []
    for ring in rings:
        points = positions_nm[list(ring)]
        centre = points.mean(axis=0)
        # The plane normal is the direction of least spread, which for a ring
        # is the axis through it.
        normal = np.linalg.svd(points - centre)[2][2]
        centres.append(centre)
        normals.append(normal)
        radii.append(float(np.linalg.norm(points - centre, axis=1).mean()))
        nearby = set(ring)
        for index in ring:
            nearby.update(neighbours.get(index, ()))
        excluded.append(nearby)

    grid = CellList(max(radii) + 0.2)
    grid.add(np.asarray(centres, dtype=np.float64))
    for first, second in topology.bonds():
        start, end = positions_nm[first.index], positions_nm[second.index]
        for ring_index in grid.neighbours((start + end) / 2.0):
            if (
                first.index in excluded[ring_index]
                or second.index in excluded[ring_index]
            ):
                continue
            if _segment_pierces_ring(
                start, end, centres[ring_index], normals[ring_index], radii[ring_index]
            ):
                raise PackmolError(
                    f"The bond between atoms {first.index} and {second.index} "
                    f"passes through the ring at {rings[ring_index]}. A "
                    "threaded ring is a permanent knot: minimisation does not "
                    "undo it. Repack with a different seed, or at a lower "
                    "density."
                )


def _segment_pierces_ring(
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    centre: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    radius: float,
) -> bool:
    """Whether the segment crosses the ring's plane inside the ring."""
    direction = end - start
    denominator = float(direction @ normal)
    if abs(denominator) < 1e-12:
        return False
    fraction = float((centre - start) @ normal) / denominator
    if not 0.0 <= fraction <= 1.0:
        return False
    crossing = start + fraction * direction
    return bool(np.linalg.norm(crossing - centre) < radius)
