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
import re
import shutil
import subprocess
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

#: Angstrom per nanometre. packmol works in angstrom, OpenMM in nanometres, and
#: this module is the one place the two meet.
ANGSTROM_PER_NM = 10.0

#: Default closest approach packmol enforces between atoms of different
#: molecules, in nanometres.
DEFAULT_TOLERANCE_NM = 0.2

#: Density to pack at, in g/cm3. See the module docstring.
DEFAULT_PACKING_DENSITY = 0.3

#: Environment variable consulted when the binary is not passed explicitly.
PACKMOL_ENV_VAR = "PACKMOL"

#: packmol's own exit codes start here.
_PACKMOL_ERROR_BASE = 170


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


def density_g_cm3(
    counts: Sequence[int],
    molar_masses_g_mol: Sequence[float],
    volume_nm3: float,
) -> float:
    """Return the density of this much material in this volume.

    Args:
        counts: How many of each component.
        molar_masses_g_mol: Each component's molar mass.
        volume_nm3: The cell volume.

    Returns:
        The density in g/cm3.
    """
    require_positive(volume_nm3, None, name="volume_nm3")
    total_mass = sum(
        count * mass for count, mass in zip(counts, molar_masses_g_mol, strict=True)
    )
    return float(total_mass * NM3_PER_CM3 / (volume_nm3 * AVOGADRO))


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
        PackedComponent(path, base + (1 if index < extra else 0))
        for index, path in enumerate(pdb_paths)
        if base + (1 if index < extra else 0) > 0
    ]


def find_packmol(packmol: str | Path | None = None) -> str:
    """Locate the packmol executable.

    Looks at the explicit argument, then ``$PACKMOL``, then ``PATH``.

    Args:
        packmol: An explicit path, or None to search.

    Returns:
        The path to the executable.

    Raises:
        PackmolError: No executable was found.
    """
    import os

    for candidate in (packmol, os.environ.get(PACKMOL_ENV_VAR)):
        if candidate:
            resolved = shutil.which(str(candidate)) or str(candidate)
            if Path(resolved).is_file():
                return resolved
            raise PackmolError(f"packmol={candidate!r} is not an executable file.")

    found = shutil.which("packmol")
    if found is None:
        raise PackmolError(
            "packmol is not on PATH. conda-forge ships it, both as its own "
            "package and inside ambertools: "
            "conda install -c conda-forge packmol. Or point at a build with "
            f"the {PACKMOL_ENV_VAR} environment variable."
        )
    return found


def packmol_version(packmol: str | Path | None = None) -> tuple[int, ...]:
    """Return packmol's version.

    packmol prints a banner and then asks for an input file, so running it with
    nothing on stdin is how it is asked.

    Args:
        packmol: An explicit path, or None to search.

    Returns:
        The version as a tuple of integers, e.g. ``(21, 0, 1)``.

    Raises:
        PackmolError: The banner could not be read.
    """
    binary = find_packmol(packmol)
    completed = subprocess.run(
        [binary],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    match = re.search(r"Version\s+([0-9]+(?:\.[0-9]+)*)", completed.stdout)
    if match is None:
        raise PackmolError(f"{binary} did not print a recognisable version banner.")
    return tuple(int(part) for part in match.group(1).split("."))


def render_packmol_input(
    components: Sequence[PackedComponent],
    box_angstrom: tuple[float, float, float],
    output_pdb: str,
    *,
    tolerance_angstrom: float,
    inset_angstrom: float,
    seed: int,
    nloop: int | None = None,
) -> str:
    """Render a packmol input file.

    Everything here is in angstrom, and says so, because packmol works in the
    units of the PDB it is given while the rest of this package works in
    nanometres. A ten-fold slip in the tolerance packs atoms on top of each
    other; the same slip in the box is a cell a thousand times the wrong size.

    Args:
        components: What to place.
        box_angstrom: The periodic cell edges.
        output_pdb: Where packmol writes the packed cell.
        tolerance_angstrom: Closest approach between different molecules.
        inset_angstrom: How far the packing region is held back from each face.
        seed: packmol's random seed.
        nloop: Optimisation loops per structure; packmol's default when None.

    Returns:
        The input file's text.
    """
    lines = [
        f"tolerance {tolerance_angstrom:.4f}",
        "filetype pdb",
        f"output {output_pdb}",
        f"seed {seed}",
    ]
    if nloop is not None:
        lines.append(f"nloop {nloop}")
    lines.append("")

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
    inset_nm: float | None = None,
    seed: int = 0xF0,
    nloop: int | None = None,
    packmol: str | Path | None = None,
    timeout: float | None = 3600.0,
    workdir: str | Path | None = None,
) -> PackResult:
    """Pack *components* into a periodic cell of *box_nm*.

    Args:
        components: What to place, and how many of each.
        box_nm: Cubic edge, or the three edges.
        output_pdb: Where the packed cell is written.
        tolerance_nm: Closest approach between different molecules.
        inset_nm: How far the packing region is held back from each face.
            Defaults to half the tolerance, which is the least that keeps two
            molecules on opposite faces a full tolerance apart across the
            periodic boundary.
        seed: packmol's random seed.
        nloop: Optimisation loops per structure.
        packmol: Path to the executable, or None to search.
        timeout: Seconds to allow.
        workdir: Where the input file and log are written. Defaults to the
            output's directory.

    Returns:
        What was packed.

    Raises:
        PackmolError: packmol is missing, failed, or did not converge.
        ValueError: A component would not fit in the cell.
    """
    require_positive(tolerance_nm, None, name="tolerance_nm")
    edges = (box_nm, box_nm, box_nm) if isinstance(box_nm, int | float) else box_nm
    for edge in edges:
        require_positive(edge, None, name="box_nm")
    inset = tolerance_nm / 2.0 if inset_nm is None else inset_nm

    destination = Path(output_pdb)
    directory = Path(workdir) if workdir is not None else destination.parent
    directory = directory if str(directory) else Path()
    directory.mkdir(parents=True, exist_ok=True)

    binary = find_packmol(packmol)
    text = render_packmol_input(
        [
            PackedComponent(str(Path(item.pdb_path).resolve()), item.count)
            for item in components
        ],
        tuple(edge * ANGSTROM_PER_NM for edge in edges),  # type: ignore[arg-type]
        str(destination.resolve()),
        tolerance_angstrom=tolerance_nm * ANGSTROM_PER_NM,
        inset_angstrom=inset * ANGSTROM_PER_NM,
        seed=seed,
        nloop=nloop,
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
    output = _run_packmol(binary, input_path, timeout)
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


def _run_packmol(binary: str, input_path: Path, timeout: float | None) -> str:
    """Run packmol on *input_path*, which it reads from stdin.

    Args:
        binary: The executable.
        input_path: The generated input file.
        timeout: Seconds to allow.

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
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise PackmolError(
                f"packmol did not finish within {timeout} s. Pack at a lower "
                "density, or lower nloop."
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
    """Read a PDB and close the file.

    Handed a path, ``openmm.app.PDBFile`` opens it and leaves the handle to
    the garbage collector, which under ``-W error`` is a ResourceWarning and
    in a long build is a slow leak of descriptors. Handed an open file it
    reads and returns, so opening it here is all it takes.

    Args:
        path: The file to read.

    Returns:
        The parsed ``PDBFile``.
    """
    from openmm import app

    with Path(path).open() as handle:
        return app.PDBFile(handle)


def read_packed_pdb(packed_pdb: str | Path) -> tuple[Any, npt.NDArray[np.float64]]:
    """Read packmol's output once, returning its topology and its positions.

    Read through ``openmm.app.PDBFile`` rather than by slicing columns: the
    coordinate fields are not where a naive slice puts them, and getting the z
    column off by one is a silent error of a few tenths of an angstrom. The
    file is opened here rather than by path so that closing it is this
    function's job: handed a path, ``PDBFile`` leaves the handle to the
    garbage collector.

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


def load_positions_nm(packed_pdb: str | Path) -> npt.NDArray[np.float64]:
    """Read the packed coordinates, in nanometres.

    Args:
        packed_pdb: packmol's output.

    Returns:
        An ``(n_atoms, 3)`` array in nanometres.
    """
    return read_packed_pdb(packed_pdb)[1]


def check_packing(
    topology: Any,
    positions_nm: npt.NDArray[np.float64],
    *,
    max_bond_nm: float = 0.25,
    min_heavy_nm: float = 0.20,
    min_hydrogen_nm: float = 0.15,
    check_rings: bool = True,
) -> None:
    """Check a packed cell for the three faults that survive minimisation.

    Args:
        topology: The assembled box topology, with bonds.
        positions_nm: Its positions.
        max_bond_nm: Longest bond tolerated. A molecule split across the
            periodic boundary shows up here as a bond the length of the cell.
        min_heavy_nm: Closest approach allowed between heavy atoms of different
            molecules.
        min_hydrogen_nm: The same where either atom is a hydrogen.
        check_rings: Whether to test for bonds threaded through rings.

    Raises:
        PackmolError: The cell has a fault that would not survive a run.
    """
    _check_bond_lengths(topology, positions_nm, max_bond_nm)
    _check_contacts(topology, positions_nm, min_heavy_nm, min_hydrogen_nm)
    if check_rings:
        _check_ring_spearing(topology, positions_nm)


def _box_lengths_nm(topology: Any) -> npt.NDArray[np.float64] | None:
    """Return the periodic cell edges in nanometres, or None if unset."""
    from openmm import unit

    vectors = topology.getPeriodicBoxVectors()
    if vectors is None:
        return None
    lengths = [vectors[axis][axis].value_in_unit(unit.nanometer) for axis in range(3)]
    return np.asarray(lengths, dtype=np.float64)


def _minimum_image(
    delta: npt.NDArray[np.float64], box: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.float64]:
    """Wrap displacement *delta* into the periodic cell, if there is one."""
    if box is None:
        return delta
    return delta - box * np.round(delta / box)


def _check_bond_lengths(
    topology: Any, positions_nm: npt.NDArray[np.float64], max_bond_nm: float
) -> None:
    """Raise if any bond is implausibly long.

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
    if worst > max_bond_nm:
        raise PackmolError(
            f"Atoms {worst_pair[0]} and {worst_pair[1]} are bonded but "
            f"{worst:.3f} nm apart, against a limit of {max_bond_nm} nm. A "
            "molecule is split across the periodic boundary, or the packed "
            "coordinates do not correspond to this topology."
        )


class _CellList:
    """Atoms bucketed by position, for near-neighbour queries."""

    def __init__(self, positions: npt.NDArray[np.float64], spacing: float) -> None:
        self._spacing = spacing
        self._cells: dict[tuple[int, int, int], list[int]] = {}
        for index, point in enumerate(positions):
            self._cells.setdefault(self.cell(point), []).append(index)

    def cell(self, point: npt.NDArray[np.float64]) -> tuple[int, int, int]:
        """Return the grid cell *point* falls in."""
        x, y, z = np.floor(point / self._spacing).astype(int)
        return int(x), int(y), int(z)

    def neighbours(self, point: npt.NDArray[np.float64]) -> list[int]:
        """Return every atom in the 27 cells around *point*."""
        cx, cy, cz = self.cell(point)
        found: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    found.extend(self._cells.get((cx + dx, cy + dy, cz + dz), ()))
        return found


def _check_contacts(
    topology: Any,
    positions_nm: npt.NDArray[np.float64],
    min_heavy_nm: float,
    min_hydrogen_nm: float,
) -> None:
    """Raise if two different molecules are closer than they should be.

    Only pairs in *different* molecules are considered. packmol's tolerance is
    an intermolecular constraint and says nothing about bonded neighbours, so a
    test that included them would fail on every C-H bond at 0.11 nm.
    """
    box = _box_lengths_nm(topology)
    molecule = np.empty(topology.getNumAtoms(), dtype=np.int64)
    is_hydrogen = np.zeros(topology.getNumAtoms(), dtype=bool)
    for index, residue in enumerate(topology.residues()):
        for atom in residue.atoms():
            molecule[atom.index] = index
            is_hydrogen[atom.index] = atom.element is not None and (
                atom.element.atomic_number == 1
            )

    spacing = max(min_heavy_nm, min_hydrogen_nm)
    cells = _CellList(positions_nm, spacing)
    for index, point in enumerate(positions_nm):
        for other in cells.neighbours(point):
            if other <= index or molecule[other] == molecule[index]:
                continue
            delta = _minimum_image(point - positions_nm[other], box)
            distance = float(np.sqrt(delta @ delta))
            limit = (
                min_hydrogen_nm
                if is_hydrogen[index] or is_hydrogen[other]
                else min_heavy_nm
            )
            if distance < limit:
                raise PackmolError(
                    f"Atoms {index} and {other} are in different molecules but "
                    f"{distance:.3f} nm apart, against a limit of {limit} nm. "
                    "Raise the packmol tolerance, or pack at a lower density."
                )


def find_rings(topology: Any, max_size: int = 8) -> list[tuple[int, ...]]:
    """Return the small rings in *topology*, as tuples of atom indices.

    Every molecule in a packed cell is a copy of the same chain, so the rings
    are found once on the first residue and shifted onto the rest. Falling back
    to a per-residue search when the residues differ in size keeps this honest
    for a mixed cell.

    Args:
        topology: The topology to search.
        max_size: Largest ring to look for.

    Returns:
        One tuple of atom indices per ring.
    """
    neighbours: dict[int, list[int]] = {}
    for first, second in topology.bonds():
        neighbours.setdefault(first.index, []).append(second.index)
        neighbours.setdefault(second.index, []).append(first.index)

    residues = [
        [atom.index for atom in residue.atoms()] for residue in topology.residues()
    ]
    counts = {len(indices) for indices in residues}
    if len(counts) != 1:
        return [
            ring
            for indices in residues
            for ring in _rings_within(neighbours, indices, max_size)
        ]

    base = _rings_within(neighbours, residues[0], max_size)
    stride = counts.pop()
    return [
        tuple(index + offset * stride for index in ring)
        for offset in range(len(residues))
        for ring in base
    ]


def _rings_within(
    neighbours: dict[int, list[int]], atoms: Sequence[int], max_size: int
) -> list[tuple[int, ...]]:
    """Return the rings up to *max_size* among *atoms*, by shortest-cycle search."""
    members = set(atoms)
    seen: set[frozenset[int]] = set()
    rings: list[tuple[int, ...]] = []
    for start in atoms:
        for neighbour in neighbours.get(start, ()):
            if neighbour not in members or neighbour < start:
                continue
            path = _shortest_path_avoiding(
                neighbours, neighbour, start, max_size - 1, members
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
    queue: list[tuple[int, tuple[int, ...]]] = [(start, (start,))]
    visited = {start, goal}
    while queue:
        node, path = queue.pop(0)
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
    rings = find_rings(topology)
    if not rings:
        return

    centres: list[npt.NDArray[np.float64]] = []
    normals: list[npt.NDArray[np.float64]] = []
    radii: list[float] = []
    excluded: list[set[int]] = []
    neighbours: dict[int, list[int]] = {}
    for first, second in topology.bonds():
        neighbours.setdefault(first.index, []).append(second.index)
        neighbours.setdefault(second.index, []).append(first.index)

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

    reach = max(radii) + 0.2
    grid = _CellList(np.asarray(centres, dtype=np.float64), reach)
    for first, second in topology.bonds():
        start, end = positions_nm[first.index], positions_nm[second.index]
        for ring_index in grid.neighbours((start + end) / 2.0):
            if (
                first.index in excluded[ring_index]
                or second.index in (excluded[ring_index])
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
