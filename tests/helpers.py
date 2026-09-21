"""Builders the tests share.

Topologies are built in code rather than committed as files: a hand-aligned PDB
in the repository is a column-counting exercise that goes wrong silently, and
what these tests need is small enough to read.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

#: A force field for a two-atom "dimer" residue: enough to exercise the real
#: ForceField path, small enough to read. The atoms are argon-like so nothing
#: depends on a real chemistry.
DIMER_FFXML = """<ForceField>
 <AtomTypes>
  <Type name="DIM-C" class="DC" element="C" mass="12.011"/>
 </AtomTypes>
 <Residues>
  <Residue name="DIM">
   <Atom name="C1" type="DIM-C"/>
   <Atom name="C2" type="DIM-C"/>
   <Bond atomName1="C1" atomName2="C2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="DC" class2="DC" length="0.153" k="250000.0"/>
 </HarmonicBondForce>
 <NonbondedForce coulomb14scale="0.8333333333333334" lj14scale="0.5">
  <Atom type="DIM-C" charge="0.0" sigma="0.34" epsilon="0.36"/>
 </NonbondedForce>
</ForceField>
"""


def build_dimer_pdb(path: Path, *, separation_nm: float = 0.153) -> str:
    """Write a one-residue, two-atom PDB with the bond recorded.

    Built in code rather than committed: a hand-aligned PDB in the repository
    is a column-counting exercise that goes wrong silently.
    """
    from openmm import app, unit

    topology = app.Topology()
    residue = topology.addResidue("DIM", topology.addChain())
    carbon = app.Element.getBySymbol("C")
    first = topology.addAtom("C1", carbon, residue)
    second = topology.addAtom("C2", carbon, residue)
    topology.addBond(first, second)
    positions = [[0.0, 0.0, 0.0], [separation_nm, 0.0, 0.0]] * unit.nanometer
    with path.open("w") as handle:
        app.PDBFile.writeFile(topology, positions, handle)
    return str(path)


def argon_system(
    n_atoms: int,
    box_nm: float,
    *,
    cutoff_nm: float = 0.8,
    atoms_per_molecule: int = 1,
) -> tuple[Any, Any, np.ndarray]:
    """Build an argon cell: a System, a Topology and positions on a lattice.

    A real periodic ``NonbondedForce``, so the barostat has something to do and
    the density is a real number, with no force-field file anywhere.
    """
    import openmm as mm
    from openmm import app, unit

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_nm, 0, 0) * unit.nanometer,
        mm.Vec3(0, box_nm, 0) * unit.nanometer,
        mm.Vec3(0, 0, box_nm) * unit.nanometer,
    )
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(cutoff_nm * unit.nanometer)
    nonbonded.setUseDispersionCorrection(True)

    topology = app.Topology()
    chain = topology.addChain()
    argon = app.Element.getBySymbol("Ar")
    for _ in range(n_atoms):
        system.addParticle(39.948 * unit.dalton)
        nonbonded.addParticle(
            0.0, 0.34 * unit.nanometer, 0.996 * unit.kilojoule_per_mole
        )
        residue = topology.addResidue("AR", chain)
        topology.addAtom("AR", argon, residue)
    system.addForce(nonbonded)
    system.addForce(mm.CMMotionRemover())

    topology.setPeriodicBoxVectors(
        [
            mm.Vec3(box_nm, 0, 0),
            mm.Vec3(0, box_nm, 0),
            mm.Vec3(0, 0, box_nm),
        ]
        * unit.nanometer
    )
    return system, topology, _lattice(n_atoms, box_nm)


def _lattice(n_atoms: int, box_nm: float) -> np.ndarray:
    """Return *n_atoms* positions on a cubic lattice inside the cell."""
    per_side = math.ceil(n_atoms ** (1 / 3))
    spacing = box_nm / per_side
    points = [
        (
            (i + 0.5) * spacing,
            (j + 0.5) * spacing,
            (k + 0.5) * spacing,
        )
        for i in range(per_side)
        for j in range(per_side)
        for k in range(per_side)
    ]
    return np.asarray(points[:n_atoms], dtype=np.float64)


def requires(module: str) -> Any:
    """Skip the importing test module unless *module* is available.

    Called at module scope. Collection imports a test module before any
    per-test hook runs, so a capability that a module's imports depend on has
    to be checked here rather than in ``pytest_runtest_setup``.
    """
    return pytest.importorskip(module)


def rod_positions(n_beads: int, spacing_nm: float, *, axis: int = 0) -> np.ndarray:
    """A straight chain of *n_beads*, so its radius of gyration is exact.

    For equal masses on a line at spacing ``l``,
    ``Rg = l * sqrt((n^2 - 1) / 12)`` - a closed form to check a measurement
    against, with no simulation involved.
    """
    positions = np.zeros((n_beads, 3), dtype=np.float64)
    positions[:, axis] = np.arange(n_beads, dtype=np.float64) * spacing_nm
    return positions


def drifting_frames(
    positions_nm: np.ndarray,
    velocity_nm_ps: tuple[float, float, float],
    n_frames: int,
    interval_ps: float,
) -> np.ndarray:
    """Frames of one cell translating at constant velocity.

    Uniform drift gives ``MSD(tau) = |v|^2 tau^2`` exactly, which is the
    ballistic answer a displacement measurement has to reproduce - and the
    log-log slope of two that stops it reporting a diffusion coefficient.
    """
    velocity = np.asarray(velocity_nm_ps, dtype=np.float64)
    times = np.arange(n_frames, dtype=np.float64) * interval_ps
    return positions_nm[None, :, :] + times[:, None, None] * velocity[None, None, :]


def state_data_csv(rows: Sequence[Sequence[float]]) -> str:
    """The numeric CSV a stage writes, header and all.

    The header is OpenMM's, quoted and commented exactly as
    ``StateDataReporter`` emits it, because what the analysis has to cope with
    is how ``numpy.genfromtxt`` mangles *that* into field names.
    """
    header = (
        '#"Step","Time (ps)","Potential Energy (kJ/mole)",'
        '"Kinetic Energy (kJ/mole)","Total Energy (kJ/mole)","Temperature (K)",'
        '"Box Volume (nm^3)","Density (g/mL)"'
    )
    body = "\n".join(",".join(repr(float(value)) for value in row) for row in rows)
    return f"{header}\n{body}\n"


class _StubFrame:
    """One frame's box, standing in for an MDAnalysis timestep."""

    def __init__(self, dimensions: npt.NDArray[np.float64]) -> None:
        self.dimensions: npt.NDArray[np.float64] = dimensions


class _StubAtoms:
    """The atom group of a :class:`StubUniverse`."""

    def __init__(self, universe: StubUniverse) -> None:
        self._universe = universe

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """The current frame's coordinates, in Angstrom."""
        return self._universe.current

    @property
    def n_atoms(self) -> int:
        """Atoms in the cell."""
        return int(self._universe.positions_angstrom.shape[1])


class _StubTrajectory:
    """The frame reader of a :class:`StubUniverse`."""

    def __init__(self, universe: StubUniverse) -> None:
        self._universe = universe

    def __len__(self) -> int:
        return int(self._universe.positions_angstrom.shape[0])

    @property
    def dt(self) -> float:
        """Time between frames."""
        return self._universe.interval_ps

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))

            def frames() -> Any:
                for index in indices:
                    yield self._universe.seek(index)

            return frames()
        return self._universe.seek(int(key))


class StubUniverse:
    """A Universe made of arrays, for testing the arithmetic without files.

    The analysis functions take an :class:`~openmmpolymer.trajectory.Ensemble`,
    and an Ensemble holds a reader. Standing a reader up from arrays is what
    lets a rigid rod or a chain drifting at constant velocity - cases with
    answers that can be written down - drive the same code path a real
    trajectory does, with no MDAnalysis, no file and no simulation.
    """

    def __init__(
        self, positions_nm: npt.NDArray[np.float64], box_nm: npt.NDArray[np.float64]
    ) -> None:
        self.positions_angstrom: npt.NDArray[np.float64] = (
            np.asarray(positions_nm, dtype=np.float64) * 10.0
        )
        boxes = np.asarray(box_nm, dtype=np.float64) * 10.0
        self.dimensions_angstrom: npt.NDArray[np.float64] = np.concatenate(
            [boxes, np.full((boxes.shape[0], 3), 90.0)], axis=1
        )
        self.interval_ps = 1.0
        self.current: npt.NDArray[np.float64] = self.positions_angstrom[0]
        self.atoms = _StubAtoms(self)
        self.trajectory = _StubTrajectory(self)

    def seek(self, index: int) -> _StubFrame:
        """Make frame *index* current, and describe its box."""
        self.current = self.positions_angstrom[index]
        return _StubFrame(self.dimensions_angstrom[index])


def synthetic_ensemble(
    positions_nm: np.ndarray,
    *,
    n_chains: int,
    box_nm: float | np.ndarray = 10.0,
    interval_ps: float = 1.0,
    masses_amu: np.ndarray | None = None,
    is_hydrogen: np.ndarray | None = None,
    stage: str = "synthetic",
) -> Any:
    """Build an Ensemble from ``(n_frames, n_atoms, 3)`` positions.

    Args:
        positions_nm: The frames. A single ``(n_atoms, 3)`` frame is accepted
            and read as a snapshot.
        n_chains: How many equal blocks the atoms divide into.
        box_nm: One cube edge for every frame, or per-frame edges.
        interval_ps: Time between frames.
        masses_amu: One chain's masses. Equal masses when None.
        is_hydrogen: One chain's hydrogen mask. All heavy when None.
        stage: Name to record on the Ensemble.

    Returns:
        An Ensemble reading from arrays rather than from a file.
    """
    from openmmpolymer.trajectory import Ensemble

    frames = np.asarray(positions_nm, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, :, :]
    n_frames, n_atoms, _ = frames.shape
    if isinstance(box_nm, (int, float)):
        boxes = np.full((n_frames, 3), float(box_nm))
    else:
        boxes = np.asarray(box_nm, dtype=np.float64)
        if boxes.ndim == 1:
            boxes = np.tile(boxes, (n_frames, 1))
    atoms_per_chain = n_atoms // n_chains
    universe = StubUniverse(frames, boxes)
    universe.interval_ps = interval_ps
    return Ensemble(
        stage=stage,
        topology=None,
        universe=universe,
        n_chains=n_chains,
        atoms_per_chain=atoms_per_chain,
        masses_amu=(
            np.ones(atoms_per_chain, dtype=np.float64)
            if masses_amu is None
            else np.asarray(masses_amu, dtype=np.float64)
        ),
        is_hydrogen=(
            np.zeros(atoms_per_chain, dtype=np.bool_)
            if is_hydrogen is None
            else np.asarray(is_hydrogen, dtype=np.bool_)
        ),
        n_frames=n_frames,
        interval_ps=interval_ps if n_frames > 1 else 0.0,
        topology_path="synthetic.pdb",
        trajectory_path=None if n_frames == 1 else "synthetic.xtc",
    )


def freely_rotating_chain(
    n_bonds: int,
    bond_length_nm: float,
    persistence_length_nm: float,
    *,
    n_chains: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Chains of known stiffness, stacked into one cell's positions.

    Each bond turns from the last by a fixed angle about a uniformly random
    torsion, which makes the bond-direction correlation ``cos(a)^s`` exactly in
    expectation. Choosing ``a`` so that ``cos(a) = exp(-l / lp)`` therefore
    gives a chain whose persistence length is *persistence_length_nm* by
    construction - an answer to check a measurement against, rather than a
    measurement to compare with another measurement.
    """
    angle = math.acos(math.exp(-bond_length_nm / persistence_length_nm))
    generator = np.random.default_rng(seed)
    chains = []
    for _ in range(n_chains):
        directions = np.zeros((n_bonds, 3), dtype=np.float64)
        first = generator.normal(size=3)
        directions[0] = first / np.linalg.norm(first)
        for index in range(1, n_bonds):
            previous = directions[index - 1]
            away = (
                np.array([1.0, 0.0, 0.0])
                if abs(previous[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            first_axis = np.cross(previous, away)
            first_axis /= np.linalg.norm(first_axis)
            second_axis = np.cross(previous, first_axis)
            torsion = generator.uniform(0.0, 2.0 * math.pi)
            step = math.cos(angle) * previous + math.sin(angle) * (
                math.cos(torsion) * first_axis + math.sin(torsion) * second_axis
            )
            directions[index] = step / np.linalg.norm(step)
        chains.append(
            np.concatenate(
                [
                    np.zeros((1, 3), dtype=np.float64),
                    np.cumsum(directions * bond_length_nm, axis=0),
                ]
            )
        )
    return np.concatenate(chains, axis=0)


def random_walk_frames(
    n_frames: int,
    n_chains: int,
    diffusion_nm2_ps: float,
    interval_ps: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Frames of dimers whose centres of mass random-walk with known D.

    Each step is drawn so that ``<|dr|^2> = 6 D dt`` in three dimensions, which
    makes ``MSD(tau) = 6 D tau`` and the diffusion coefficient recoverable
    exactly - the arithmetic a measurement's unit conversion has to reproduce.
    """
    generator = np.random.default_rng(seed)
    spread = math.sqrt(2.0 * diffusion_nm2_ps * interval_ps)
    steps = generator.normal(0.0, spread, size=(n_frames, n_chains, 3))
    walk = np.cumsum(steps, axis=0)
    frames = np.zeros((n_frames, n_chains * 2, 3), dtype=np.float64)
    frames[:, 0::2, :] = walk
    frames[:, 1::2, :] = walk + np.array([0.0, 0.0, 0.6])
    return frames
