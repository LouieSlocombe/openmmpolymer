"""Builders the tests share.

Topologies are built in code rather than committed as files: a hand-aligned PDB
in the repository is a column-counting exercise that goes wrong silently, and
what these tests need is small enough to read.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from openmmpolymer.elasticity import StressStrain
from openmmpolymer.rate_dependence import (
    RateObservation,
    RateProperty,
    RateReport,
    analyse_rate_observations,
)
from openmmpolymer.relaxation import RelaxationCurve
from openmmpolymer.tensile import BreakingSpec, ElongationSpec, TensileSpec, YieldSpec
from openmmpolymer.timeseries import GlassTransition
from openmmpolymer.tm import HeatingCurve

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


#: An equilibration short enough for a scan test, and gentle: a kilobar
#: squeezes a small argon cell past twice its cutoff.
QUICK_EQUILIBRATION: dict[str, Any] = {
    "nvt_ps": 0.2,
    "compress_ps_each": 0.2,
    "npt_ps": 0.3,
    "anneal_cycles": 1,
    "anneal_window_ps": 0.1,
    "anneal_hold_ps": 0.1,
    "compress_pressures_bar": (1.0, 20.0, 1.0),
}


def argon_context(
    n_atoms: int = 64, box_nm: float = 2.4, *, atoms_per_molecule: int = 1
) -> Any:
    """A run context over an argon cell, on the deterministic CPU platform."""
    from openmmpolymer.forcefield import PolymerForceField
    from openmmpolymer.mdsystem import PackedBox
    from openmmpolymer.simulate import prepare_run

    system, topology, positions = argon_system(
        n_atoms, box_nm, atoms_per_molecule=atoms_per_molecule
    )
    box = PackedBox(topology, positions, (box_nm,) * 3, n_atoms // atoms_per_molecule)
    return prepare_run(
        box,
        PolymerForceField("unused.xml", (), "AR", "smirnoff"),
        platform="CPU",
        seed=11,
        system=system,
    )


def argon_system(
    n_atoms: int,
    box_nm: float,
    *,
    cutoff_nm: float = 0.8,
    atoms_per_molecule: int = 1,
) -> tuple[Any, Any, np.ndarray]:
    """Build an argon cell: a System, a Topology and positions on a lattice.

    A real periodic ``NonbondedForce``, so the barostat has something to do and
    the density is a real number, with no force-field file anywhere. With
    *atoms_per_molecule* above one, consecutive atoms are bonded into chains at
    the lattice spacing, so each molecule starts at its bonds' rest length.
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
    bonds = mm.HarmonicBondForce()
    spacing_nm = box_nm / math.ceil(n_atoms ** (1 / 3))

    topology = app.Topology()
    chain = topology.addChain()
    argon = app.Element.getBySymbol("Ar")
    for _ in range(0, n_atoms, atoms_per_molecule):
        residue = topology.addResidue("AR", chain)
        names = (
            ["AR"]
            if atoms_per_molecule == 1
            else [f"AR{index + 1}" for index in range(atoms_per_molecule)]
        )
        atoms = [topology.addAtom(name, argon, residue) for name in names]
        for _ in atoms:
            system.addParticle(39.948 * unit.dalton)
            nonbonded.addParticle(
                0.0, 0.34 * unit.nanometer, 0.996 * unit.kilojoule_per_mole
            )
        for first, second in pairwise(atoms):
            topology.addBond(first, second)
            bonds.addBond(first.index, second.index, spacing_nm, 1000.0)
            nonbonded.addException(first.index, second.index, 0.0, 0.34, 0.0)
    system.addForce(nonbonded)
    if bonds.getNumBonds():
        system.addForce(bonds)
    system.addForce(mm.CMMotionRemover())

    topology.setPeriodicBoxVectors(
        [
            mm.Vec3(box_nm, 0, 0),
            mm.Vec3(0, box_nm, 0),
            mm.Vec3(0, 0, box_nm),
        ]
        * unit.nanometer
    )
    return system, topology, lattice(n_atoms, box_nm)


def lattice(n_atoms: int, box_nm: float) -> np.ndarray:
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


def write_quenches(
    directory: Path,
    stages: Mapping[str, Mapping[str, object]],
) -> Path:
    """Write a manifest holding several quench stages at once.

    Each entry gives ``temperature_k`` and ``density_g_cm3``, and optionally
    ``total_ps`` (the stage CSV's last time, which is what fabricates a chosen
    cooling rate), ``segment_duration_ps`` and ``waypoints``.
    """
    recorded: dict[str, object] = {}
    for stage, fields in stages.items():
        temperature = list(cast(Sequence[float], fields["temperature_k"]))
        density = list(cast(Sequence[float], fields["density_g_cm3"]))
        total_ps = cast(float | None, fields.get("total_ps", 4200.0))
        csv: str | None = None
        if total_ps is not None:
            csv = str(directory / f"{stage}.csv")
            rows = [
                [
                    index * 1000,
                    index * total_ps / 100.0,
                    -1.0,
                    1.0,
                    0.0,
                    300.0,
                    13.8,
                    0.9,
                ]
                for index in range(1, 101)
            ]
            Path(csv).write_text(state_data_csv(rows))
        samples: dict[str, list[float]] = {
            "segment_temperature_k": temperature,
            "segment_density_g_cm3": density,
        }
        holds = fields.get("segment_duration_ps")
        if holds is not None:
            samples["segment_duration_ps"] = list(cast(Sequence[float], holds))
        entry: dict[str, object] = {
            "name": stage,
            "csv": csv,
            "samples": samples,
        }
        waypoints = fields.get("waypoints")
        if waypoints is not None:
            entry["waypoints"] = list(cast(Sequence[object], waypoints))
        recorded[stage] = entry

    payload = {
        "protocol": "melt-quench",
        "seed": 1,
        "versions": {},
        "system": {},
        "stages": recorded,
        "chains": None,
        "box": None,
    }
    (directory / "manifest.json").write_text(json.dumps(payload))
    return directory


def write_quench(
    directory: Path,
    temperature_k: npt.NDArray[np.float64] | Sequence[float],
    density_g_cm3: npt.NDArray[np.float64] | Sequence[float],
    *,
    with_csv: bool = True,
    stage: str = "06_quench",
    total_ps: float = 4200.0,
    segment_duration_ps: Sequence[float] | None = None,
    waypoints: Sequence[object] | None = None,
) -> Path:
    """Write a manifest holding one quench's samples, and optionally its CSV.

    *total_ps* is the knob that fabricates a chosen cooling rate: a ladder of
    *n* points stepping *dT* and cooled at *R* K/ns needs
    ``total_ps = n * dT * 1000 / R``.
    """
    return write_quenches(
        directory,
        {
            stage: {
                "temperature_k": list(temperature_k),
                "density_g_cm3": list(density_g_cm3),
                "total_ps": total_ps if with_csv else None,
                "segment_duration_ps": segment_duration_ps,
                "waypoints": waypoints,
            }
        },
    )


def two_line_curve(
    transition_k: float = 350.0,
    n_points: int = 21,
    glass_slope: float = 2.0e-4,
    melt_slope: float = 8.0e-4,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A specific-volume curve made of two exact straight lines.

    The grid is ``linspace(200, 600, n_points)``. Put *transition_k* on a grid
    point and the fit recovers it exactly: both branches are exactly linear,
    the best break is the knot, and both lines pass through it.
    """
    temperature = np.linspace(200.0, 600.0, n_points)
    volume = np.where(
        temperature <= transition_k,
        1.0 + glass_slope * (temperature - transition_k),
        1.0 + melt_slope * (temperature - transition_k),
    )
    return temperature, 1.0 / volume


def transition_at(
    cooling_rate_k_per_ns: float | None,
    temperature_k: float,
    *,
    resolved: bool = True,
    specific_volume_cm3_g: float = 1.0,
) -> GlassTransition:
    """A GlassTransition built by hand, for the pure rate-fit tests.

    Going through a quench curve to make one would obscure what is being
    tested: the rate fit is a function of (rate, transition) pairs and nothing
    else.
    """
    return GlassTransition(
        temperature_k=temperature_k,
        specific_volume_cm3_g=specific_volume_cm3_g,
        melt_expansion_per_k=8.0e-4,
        glass_expansion_per_k=2.0e-4,
        residual_cm3_g=0.0,
        n_points_melt=10,
        n_points_glass=11,
        cooling_rate_k_per_ns=cooling_rate_k_per_ns,
        resolved=resolved,
    )


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


def ideal_gas_system(
    n_atoms: int, box_nm: float, *, mass_amu: float = 40.0
) -> tuple[Any, Any, np.ndarray]:
    """A cell of particles with mass and no forces between them.

    The zero-parameter ``NonbondedForce`` is not decoration: OpenMM refuses a
    barostat in a System that uses no periodic boundary conditions, and the
    pressure is only readable through a barostat. With every charge and every
    epsilon zero the potential is identically zero, so the pressure is purely
    kinetic and exactly ``sum(m v_a^2) / V`` on each axis - which makes it the
    one test of a pressure readout that involves no statistics at all.

    No ``CMMotionRemover``, so the answer is over all ``n_atoms`` rather than
    ``n_atoms - 1``.
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
    nonbonded.setCutoffDistance(min(1.0, box_nm / 2.5) * unit.nanometer)

    topology = app.Topology()
    chain = topology.addChain()
    argon = app.Element.getBySymbol("Ar")
    for _ in range(n_atoms):
        system.addParticle(mass_amu * unit.dalton)
        nonbonded.addParticle(0.0, 0.3 * unit.nanometer, 0.0)
        topology.addAtom("AR", argon, topology.addResidue("AR", chain))
    system.addForce(nonbonded)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())
    return system, topology, lattice(n_atoms, box_nm)


def rigid_rotor_system(
    n_molecules: int, box_nm: float, *, bond_nm: float = 0.109
) -> tuple[Any, Any, np.ndarray]:
    """A cell of constrained diatomics with no forces between them.

    The polyatomic, constrained counterpart of :func:`ideal_gas_system`, and
    the only fixture here that can tell the molecular virial from the atomic
    one. With no interactions the exact pressure is ``N_molecules k T / V``:
    a rigid rotor's rotational kinetic energy does not contribute, because
    the molecule translates as one under a volume move. An atomic virial
    counts it and comes out 5/3 too high, which is what makes this the test
    that pins the convention.
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
    nonbonded.setCutoffDistance(min(1.0, box_nm / 2.5) * unit.nanometer)

    topology = app.Topology()
    chain = topology.addChain()
    carbon = app.Element.getBySymbol("C")
    hydrogen = app.Element.getBySymbol("H")
    for index in range(n_molecules):
        system.addParticle(12.011 * unit.dalton)
        system.addParticle(1.008 * unit.dalton)
        nonbonded.addParticle(0.0, 0.34 * unit.nanometer, 0.0)
        nonbonded.addParticle(0.0, 0.24 * unit.nanometer, 0.0)
        nonbonded.addException(2 * index, 2 * index + 1, 0.0, 0.3, 0.0)
        system.addConstraint(2 * index, 2 * index + 1, bond_nm * unit.nanometer)
        residue = topology.addResidue("CH", chain)
        first = topology.addAtom("C", carbon, residue)
        second = topology.addAtom("H", hydrogen, residue)
        topology.addBond(first, second)
    system.addForce(nonbonded)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

    centres = lattice(n_molecules, box_nm)
    positions = np.empty((2 * n_molecules, 3), dtype=np.float64)
    offsets = np.asarray(
        [[bond_nm, 0.0, 0.0], [0.0, bond_nm, 0.0], [0.0, 0.0, bond_nm]],
        dtype=np.float64,
    )
    for index, centre in enumerate(centres):
        positions[2 * index] = centre
        positions[2 * index + 1] = centre + offsets[index % 3]
    return system, topology, positions


def write_deformation(
    run_dir: Path,
    *,
    modulus_mpa: float = 2000.0,
    poisson: float = 0.35,
    reference_nm: float = 5.0,
    n_steps: int = 10,
    increment: float = 0.002,
    relax_ps: float = 50.0,
    stage: str = "06_deform_r0_00",
    temperature_k: float = 298.15,
    axis: int = 2,
) -> Path:
    """Write a manifest holding an exactly linear stress-strain curve.

    Both the modulus and the ratio are planted, so a correct fit recovers
    them to machine precision and an assertion can be an equality rather than
    a tolerance - the trick :func:`two_line_curve` uses for a quench.
    """
    lateral = [index for index in range(3) if index != axis]
    strains = [(1.0 + increment) ** (step + 1) - 1.0 for step in range(n_steps)]
    boxes = {
        name: [reference_nm * (1.0 + value) for value in strains]
        if index == axis
        else [reference_nm * (1.0 - poisson * value) for value in strains]
        for index, name in enumerate("xyz")
    }
    samples: dict[str, list[float]] = {
        "segment_strain": list(strains),
        f"segment_stress_{'xyz'[axis]}{'xyz'[axis]}_bar": [
            modulus_mpa * value / 0.1 for value in strains
        ],
        "segment_duration_ps": [relax_ps] * n_steps,
        "reference_box_nm": [reference_nm] * 3,
        "deform_axis": [float(axis)],
    }
    for index in lateral:
        samples[f"segment_stress_{'xyz'[index]}{'xyz'[index]}_bar"] = [0.0] * n_steps
    for name, values in boxes.items():
        samples[f"segment_box_{name}_nm"] = values
    return write_manifest(
        run_dir, {stage: {"samples": samples, "mean_temperature_k": temperature_k}}
    )


def write_bulk(
    run_dir: Path,
    *,
    modulus_mpa: float = 1500.0,
    pressures_bar: Sequence[float] = (1.0, 100.0, 200.0, 300.0, 200.0, 100.0, 1.0),
    density_g_cm3: float = 0.9,
    stage: str = "08_bulk",
    temperature_k: float = 298.15,
    merge: dict[str, Any] | None = None,
) -> Path:
    """Write a manifest holding an exactly log-linear pressure ladder."""
    densities = [
        density_g_cm3 * math.exp(pressure * 0.1 / modulus_mpa)
        for pressure in pressures_bar
    ]
    stages = dict(merge or {})
    stages[stage] = {
        "samples": {
            "segment_pressure_bar": list(pressures_bar),
            "segment_density_g_cm3": densities,
        },
        "mean_temperature_k": temperature_k,
    }
    return write_manifest(run_dir, stages)


def write_shear(
    run_dir: Path,
    *,
    modulus_mpa: float = 700.0,
    strains: Sequence[float] = (0.005, 0.010, 0.015, 0.020),
    stage: str = "09_shear",
    temperature_k: float = 298.15,
    merge: dict[str, Any] | None = None,
) -> Path:
    """Write a manifest holding an exactly linear shear ladder."""
    from openmmpolymer.stress import STRESS_ESTIMATOR_VERSION

    stages = dict(merge or {})
    stages[stage] = {
        "samples": {
            "stress_estimator_version": [float(STRESS_ESTIMATOR_VERSION)],
            "segment_shear_strain": list(strains),
            "segment_shear_stress_bar": [
                modulus_mpa * value / 0.1 for value in strains
            ],
        },
        "mean_temperature_k": temperature_k,
    }
    return write_manifest(run_dir, stages)


def write_manifest(run_dir: Path, stages: dict[str, Any]) -> Path:
    """Write a minimal manifest holding *stages*, merging with any already there."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    record: dict[str, Any] = (
        json.loads(path.read_text())
        if path.is_file()
        else {
            "protocol": "test",
            "seed": 1,
            "versions": {},
            "system": {},
            "stages": {},
            "chains": None,
            "box": None,
        }
    )
    record["stages"].update(stages)
    path.write_text(json.dumps(record, indent=2))
    return path


def write_relaxation(
    run_dir: Path,
    *,
    modulus_mpa: float = 1000.0,
    tau_ps: float = 100.0,
    beta: float = 0.5,
    equilibrium_mpa: float = 0.0,
    step_strain: float = 0.03,
    strain_measure: float | None = None,
    first_ps: float = 0.1,
    total_ps: float = 1.0e4,
    bins_per_decade: int = 20,
    baseline_bar: float = 0.0,
    samples_per_bin: float = 100.0,
    stem: str = "06_relax_r0",
    chunks: int = 1,
    temperature_k: float = 298.15,
    mode: str = "tensile",
    poisson: float = 0.5,
    merge: dict[str, Any] | None = None,
) -> Path:
    """Write a manifest holding an exact stretched-exponential relaxation.

    ``G(t) = equilibrium + modulus exp[-(t/tau)^beta]`` planted on the real
    logarithmic grid the stage would have used, so a correct reader recovers
    it to machine precision and an assertion can be an equality rather than a
    tolerance - the trick :func:`write_deformation` uses for a modulus.

    Every bin is given zero scatter, so its standard error is zero and the
    whole curve is inside any signal window. *chunks* splits the bins across
    that many stages, which is what a resumed relaxation looks like on disk
    and the only way to exercise the merge.
    """
    from openmmpolymer.elasticity import MPA_PER_BAR
    from openmmpolymer.simulate import relax_bin_edges_ps
    from openmmpolymer.stress import STRESS_ESTIMATOR_VERSION, deviatoric_strain

    if strain_measure is None:
        strain_measure = (
            step_strain
            if mode == "shear"
            else 2.0 * deviatoric_strain(step_strain, poisson)
        )
    edges = relax_bin_edges_ps(first_ps, total_ps, bins_per_decade)
    centres = np.sqrt(edges[:-1] * edges[1:])
    moduli = equilibrium_mpa + modulus_mpa * np.exp(-((centres / tau_ps) ** beta))
    # The reader subtracts the baseline, scales to MPa and divides by the
    # strain, so plant the stress that comes back out as exactly `moduli`.
    stresses = moduli * strain_measure / MPA_PER_BAR + baseline_bar

    stages = dict(merge or {})
    groups = np.array_split(np.arange(centres.size), max(1, chunks))
    for index, group in enumerate(groups):
        samples: dict[str, Any] = {
            "segment_bin": [float(value) for value in group],
            "segment_relax_time_ps": [float(centres[value]) for value in group],
            "segment_stress_bar": [float(stresses[value]) for value in group],
            "segment_stress_sq_bar2": [float(stresses[value] ** 2) for value in group],
            "segment_samples": [samples_per_bin] * len(group),
            "relax_strain_measure": [float(strain_measure)],
            "step_strain": [float(step_strain)],
        }
        if mode == "shear":
            samples["stress_estimator_version"] = [float(STRESS_ESTIMATOR_VERSION)]
            samples["relax_plane"] = [0.0, 2.0]
        else:
            samples["relax_axis"] = [2.0]
            samples["relax_poisson"] = [float(poisson)]
        if index == 0:
            samples["baseline_stress_bar"] = [float(baseline_bar)]
            samples["baseline_stress_sq_bar2"] = [float(baseline_bar) ** 2]
            samples["baseline_samples"] = [samples_per_bin]
            samples["instant_stress_bar"] = [
                float((equilibrium_mpa + modulus_mpa) * strain_measure / MPA_PER_BAR)
                + baseline_bar
            ]
        stages[f"{stem}_{index:02d}"] = {
            "samples": samples,
            "mean_temperature_k": temperature_k,
        }
    return write_manifest(run_dir, stages)


def write_polymer_snapshot(
    run_dir: Path,
    *,
    stage: str = "05_npt",
    n_chains: int = 4,
    box_nm: float = 4.0,
    side_group: bool = True,
) -> tuple[int, ...]:
    """Write a closing structure of straight five-carbon chains, bonds recorded.

    Written through OpenMM so the CONECT records are the ones a real stage
    leaves. Each chain is a rod of five carbons 0.153 nm apart along x, a side
    carbon on the middle one when *side_group* is set, and one hydrogen on the
    last - so a backbone inferred from the bond graph has something to leave
    out, and ``<R^2>`` is exactly ``(4 * 0.153)^2``.

    Returns:
        The true backbone, ``(0, 1, 2, 3, 4)``.
    """
    import openmm as mm
    from openmm import app, unit

    spacing = 0.153
    topology = app.Topology()
    carbon = app.Element.getBySymbol("C")
    hydrogen = app.Element.getBySymbol("H")
    positions: list[Any] = []
    for chain_index in range(n_chains):
        residue = topology.addResidue("POL", topology.addChain())
        offset = box_nm * (chain_index + 0.5) / n_chains
        carbons = [topology.addAtom(f"C{i}", carbon, residue) for i in range(5)]
        for first, second in pairwise(carbons):
            topology.addBond(first, second)
        positions.extend(mm.Vec3(0.5 + spacing * i, offset, 0.5) for i in range(5))
        if side_group:
            side = topology.addAtom("C5", carbon, residue)
            topology.addBond(carbons[2], side)
            positions.append(mm.Vec3(0.5 + 2.0 * spacing, offset + 0.15, 0.5))
        cap = topology.addAtom("H1", hydrogen, residue)
        topology.addBond(carbons[4], cap)
        positions.append(mm.Vec3(0.5 + 4.0 * spacing, offset, 0.61))
    topology.setPeriodicBoxVectors(
        [mm.Vec3(box_nm, 0, 0), mm.Vec3(0, box_nm, 0), mm.Vec3(0, 0, box_nm)]
        * unit.nanometer
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    pdb = run_dir / f"{stage}.pdb"
    with pdb.open("w") as handle:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, handle)
    path = write_manifest(
        run_dir,
        {
            stage: {
                "name": stage,
                "final_pdb": str(pdb),
                "final_state": None,
                "csv": None,
                "samples": {},
            }
        },
    )
    record = json.loads(path.read_text())
    record["box"] = {
        "n_molecules": n_chains,
        "atoms_per_chain": topology.getNumAtoms() // n_chains,
        "box_nm": [box_nm, box_nm, box_nm],
    }
    path.write_text(json.dumps(record, indent=2))
    return (0, 1, 2, 3, 4)


#: The tensile ladders :func:`write_tensile_scan` plants curves on, two
#: replicas each: eight holds of compounding 10% strain in two chunks for
#: breaking and elongation, twelve of 0.2% in three chunks for yield.
PLANTED_TENSILE: dict[str, Any] = {
    "breaking": BreakingSpec(
        strain_increment=0.1, max_strain=1.1, relax_ps=1.0, stage_ps=4.0, n_replicas=2
    ),
    "elongation": ElongationSpec(
        strain_increment=0.1, max_strain=1.1, relax_ps=1.0, stage_ps=4.0, n_replicas=2
    ),
    "yield": YieldSpec(
        max_strain=0.024,
        relax_ps=1.0,
        n_replicas=2,
        samples_per_step=2,
        stage_ps=4.0,
        fit_max_strain=0.0125,
    ),
}


def write_tensile_scan(run_dir: Path, spec: TensileSpec) -> Path:
    """Write the manifest and workflow record a finished tensile scan leaves.

    The nominal stresses are planted, so every result has an answer written
    down. Yield replica ``r`` rises as ``(1 + 0.2 r)(1000 strain + 2)`` MPa to
    a plateau at ``18 (1 + 0.2 r)``, which the 0.2% offset line meets at
    strain 0.018. A breaking or elongation replica peaks at ``100 (1 + 0.2 r)``
    MPa on the fourth hold and then loses stress for good; replica 1 falls to
    45 MPa on the fifth, so it crosses half its peak a hold before replica 0.
    The cell narrows as ``1 - 0.2 strain`` on both lateral axes under -1 bar,
    and the axial stress is whatever makes the nominal stress exactly that.
    *spec* must keep its planted ladder; its criterion and replica count are
    free. Returns the workflow record's path.
    """
    from openmmpolymer.protocols import RunManifest
    from openmmpolymer.tensile import tensile_protocol, tensile_schedule

    n_steps = tensile_schedule(spec).n_steps
    strains = (1.0 + spec.strain_increment) ** np.arange(1, n_steps + 1) - 1.0
    lateral = 1.0 - 0.2 * strains
    ladders = [
        tensile_protocol(spec, replica=replica) for replica in range(spec.n_replicas)
    ]
    stages: dict[str, Any] = {}
    for replica, ladder in enumerate(ladders):
        scale = 1.0 + 0.2 * replica
        if isinstance(spec, YieldSpec):
            nominal = np.minimum(1000.0 * strains + 2.0, 18.0) * scale
        else:
            nominal = np.asarray([0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0])
            nominal *= scale
            if replica == 1:
                nominal[4] = 45.0
        axial_bar = (nominal / lateral**2 - 0.1) / 0.1
        done = 0
        for stage in ladder.stages:
            count = stage.options["n_steps"]
            part = slice(done, done + count)
            done += count
            stages[stage.name] = {
                "mean_temperature_k": spec.temperature_k,
                "samples": {
                    "segment_strain": strains[part].tolist(),
                    "segment_stress_xx_bar": [-1.0] * count,
                    "segment_stress_yy_bar": [-1.0] * count,
                    "segment_stress_zz_bar": axial_bar[part].tolist(),
                    "segment_box_x_nm": (5.0 * lateral[part]).tolist(),
                    "segment_box_y_nm": (5.0 * lateral[part]).tolist(),
                    "segment_box_z_nm": (5.0 * (1.0 + strains[part])).tolist(),
                    "segment_duration_ps": [spec.relax_ps] * count,
                    "reference_box_nm": [5.0] * 3,
                    "deform_axis": [float(spec.axis)],
                },
            }
    run_dir.mkdir(parents=True, exist_ok=True)
    name = ladders[0].name
    RunManifest(protocol=name, seed=11, stages=stages).save(run_dir)
    record = {
        "request": {"spec": asdict(spec)},
        "reference_box_nm": [5.0] * 3,
        "replica_stages": [
            [stage.name for stage in ladder.stages] for ladder in ladders
        ],
        "steps_per_replica": n_steps,
        "timestep_fs": 2.0,
    }
    path = run_dir / f"{name}_workflow.json"
    path.write_text(json.dumps(record))
    return path


def write_two_molecule_sdf(path: Path) -> str:
    """Write methanol and ethane, with conformers, into one SDF.

    What a chain SDF must never be: the chain readers refuse it rather than
    quietly taking the first molecule.
    """
    from openff.toolkit import Molecule

    blocks = []
    for index, smiles in enumerate(("CO", "CC")):
        molecule = Molecule.from_smiles(smiles)
        molecule.generate_conformers(n_conformers=1)
        single = path.with_name(f"{path.stem}_{index}.sdf")
        molecule.to_file(str(single), file_format="SDF")
        blocks.append(single.read_text())
    path.write_text("".join(blocks))
    return str(path)


def bare_simulation(
    system: Any,
    topology: Any,
    positions_nm: npt.ArrayLike,
    *,
    barostat: Any = None,
    platform: str = "Reference",
) -> Any:
    """A Simulation of *system* with none of a stage's machinery around it.

    For tests of what a stage calls rather than of a stage: a 1 fs Langevin
    integrator at 300 K, *barostat* (a ``Force``) added when there is one, and
    the positions set. Reference by default, because the stress tests compare
    numbers to the last digit.
    """
    import openmm as mm
    from openmm import app, unit

    if barostat is not None:
        system.addForce(barostat)
    simulation = app.Simulation(
        topology,
        system,
        mm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtoseconds
        ),
        mm.Platform.getPlatformByName(platform),
    )
    simulation.context.setPositions(np.asarray(positions_nm) * unit.nanometer)
    return simulation


def nominal_curve(
    strain: npt.ArrayLike,
    nominal_stress_mpa: npt.ArrayLike,
    *,
    stage: str = "06_tensile_r0_00",
    rate_per_ns: float | None = 0.2,
    poisson: float = 0.2,
    lateral_stress_mpa: float = 0.7,
) -> StressStrain:
    """A stress-strain curve whose nominal tensile stress is known exactly.

    The lateral axes contract by *poisson* times the strain and carry a
    constant *lateral_stress_mpa*, so the recorded stress is the nominal one
    divided by the shrinking area plus that offset - which is what a strength
    analysis has to undo, and what a figure of it should not show.
    """
    strains = np.asarray(strain, dtype=np.float64)
    lateral = np.column_stack([-poisson * strains, -poisson * strains])
    area_ratio = np.prod(1.0 + lateral, axis=1)
    return StressStrain(
        stage=stage,
        axis=2,
        strain=strains,
        stress_mpa=np.asarray(nominal_stress_mpa, dtype=np.float64) / area_ratio
        + lateral_stress_mpa,
        lateral_strain=lateral,
        lateral_stress_mpa=np.full((strains.size, 2), lateral_stress_mpa),
        temperature_k=298.15,
        strain_rate_per_ns=rate_per_ns,
    )


def planted_relaxation(
    time_ps: npt.NDArray[np.float64],
    modulus_mpa: npt.NDArray[np.float64],
    *,
    error_mpa: npt.NDArray[np.float64] | None = None,
    floor: float = 0.0,
    mode: str = "shear",
    poisson: float = 0.5,
) -> RelaxationCurve:
    """A relaxation curve built straight from arrays, for testing a fit alone."""
    return RelaxationCurve(
        stage="planted",
        mode=mode,
        bin_index=np.arange(time_ps.size),
        time_ps=time_ps,
        modulus_mpa=modulus_mpa,
        standard_error_mpa=np.zeros(time_ps.size) if error_mpa is None else error_mpa,
        n_samples=np.full(time_ps.size, 100.0),
        step_strain=0.03,
        strain_measure=0.03,
        temperature_k=298.15,
        poisson=poisson,
        baseline_mpa=0.0,
        noise_floor_mpa=floor,
    )


def stationary_trace(n_samples: int = 6000) -> npt.NDArray[np.float64]:
    """Uncorrelated noise about ten, the trace every window test starts from."""
    return 10.0 + np.random.default_rng(7).normal(0.0, 0.5, n_samples)


#: Structural-window settings small enough for a test cell: coarse grids, no
#: frame caps, and a wavevector floor a four-rod cell can meet.
STRUCTURAL_OPTIONS: dict[str, Any] = {
    "q_max_per_nm": 8,
    "q_bins": 8,
    "rdf_bins": 20,
    "min_frames": 3,
    "min_vectors_per_bin": 1,
    "max_distribution_frames": None,
    "max_structure_factor_frames": None,
}


def frozen_rods(n_frames: int = 72) -> Any:
    """Four straight five-bead rods, repeated unchanged over *n_frames* frames."""
    positions = np.concatenate(
        [rod_positions(5, 0.15) + np.array([0, index * 0.8, 0]) for index in range(4)]
    )
    frames = np.repeat(positions[None, :, :], n_frames, axis=0)
    return synthetic_ensemble(frames, n_chains=4, box_nm=4)


def log_linear_transitions(
    rates: Sequence[float] = (1.0, 10.0, 100.0), **options: Any
) -> list[GlassTransition]:
    """Transitions exactly on ``Tg = 340 + 20 log10(R)``, one per rate."""
    return [
        transition_at(rate, 340.0 + 20.0 * math.log10(rate), **options)
        for rate in rates
    ]


#: The VFT melt the rate fits are checked against: T0, B and R0.
VFT_PLANTED = (300.0, 400.0, 1.0e4)


def vft_transitions(rates: Sequence[float] = (2.0, 5.0, 10.0)) -> list[GlassTransition]:
    """Transitions exactly on the :data:`VFT_PLANTED` relation, one per rate."""
    t0_k, b_k, r0 = VFT_PLANTED
    return [transition_at(rate, t0_k + b_k / math.log(r0 / rate)) for rate in rates]


def dimer_cell(
    n_chains: int = 32, separation_nm: float = 0.6, *, spacing_nm: float = 1.0
) -> np.ndarray:
    """One frame of *n_chains* two-atom molecules at a known separation.

    The molecules sit on a lattice rather than on top of each other, so that
    their centres of mass are distinct. Coincident chains make every
    displacement identically zero, which passes a test without exercising it.
    """
    per_side = math.ceil(n_chains ** (1 / 3))
    origins = lattice(n_chains, spacing_nm * per_side)
    positions = np.zeros((n_chains * 2, 3), dtype=np.float64)
    positions[0::2, :] = origins
    positions[1::2, :] = origins + np.array([0.0, 0.0, separation_nm])
    return positions


def rotating_dimer(
    n_frames: int, radians_per_frame: float, *, interval_ps: float
) -> Any:
    """One 0.6 nm dimer turning at a constant rate in the xy plane.

    Its end-to-end correlation is ``cos(rate * lag)`` exactly, so it crosses
    1/e at a known lag; at a rate of zero it never decorrelates at all.
    """
    angles = np.arange(n_frames, dtype=np.float64) * radians_per_frame
    frames = np.zeros((n_frames, 2, 3), dtype=np.float64)
    frames[:, 1, 0] = np.cos(angles) * 0.6
    frames[:, 1, 1] = np.sin(angles) * 0.6
    return synthetic_ensemble(frames, n_chains=1, interval_ps=interval_ps)


# --------------------------------------------------------------------------
# Rate series
# --------------------------------------------------------------------------


def planted_rate_report(*, target: float = 0.01, unknown: bool = False) -> RateReport:
    """A yield strength exactly logarithmic in rate, 100 MPa per decade from 0.1 /ns.

    Each measurement is known to 10 MPa, or not at all with *unknown*.
    """
    property = RateProperty("yield_strength", "Yield strength", "MPa", "strain/ns")
    source = [
        RateObservation(
            rate,
            value,
            None if unknown else 10.0,
            True,
            source=f"run-{index}",
            notes=("Finite rate.",),
        )
        for index, (rate, value) in enumerate([(0.1, 900), (1.0, 1000), (10.0, 1100)])
    ]
    return analyse_rate_observations(source, property=property, target_rate=target)


def deformation_rate_per_ns(
    hold_ps: float, *, n_steps: int = 10, increment: float = 0.002
) -> float:
    """The nominal strain rate of :func:`write_deformation`'s compounding ladder."""
    return ((1.0 + increment) ** n_steps - 1.0) / (n_steps * hold_ps) * 1000.0


def planted_modulus_mpa(rate_per_ns: float) -> float:
    """Young's modulus rising 300 MPa per decade of strain rate: 1700 at 0.001 /ns."""
    return 2000.0 + 300.0 * math.log10(rate_per_ns / 0.01)


def write_modulus_rate_series(
    root: Path,
    holds_ps: Sequence[float] = (50.0, 150.0, 500.0),
    *,
    modulus: Any = planted_modulus_mpa,
) -> list[Path]:
    """One extension run per hold, ``rate_00`` upwards, each on ``modulus(rate)``."""
    directories = []
    for index, hold in enumerate(holds_ps):
        directory = root / f"rate_{index:02d}"
        write_deformation(
            directory,
            modulus_mpa=modulus(deformation_rate_per_ns(hold)),
            relax_ps=hold,
        )
        directories.append(directory)
    return directories


def planted_extension_runner(calls: list[dict[str, Any]]) -> Any:
    """A ``run_protocol`` stand-in whose extensions record :func:`planted_modulus_mpa`.

    Each call is appended to *calls*. An extension leaves a linear curve at the
    rate its own ladder runs, stamped with the state it finished at, as a real
    stage's manifest entry is; any other protocol leaves only a state.
    """

    def run(protocol: Any, run: Any, directory: Path, **options: Any) -> Any:
        calls.append({"protocol": protocol, "directory": directory, **options})
        directory.mkdir(parents=True, exist_ok=True)
        final_state = directory / "state.xml"
        final_state.write_text("state")
        if protocol.stages[0].kind == "deform":
            (stage,) = protocol.stages
            settings = stage.options
            rate = deformation_rate_per_ns(
                settings["relax_ps"],
                n_steps=settings["n_steps"],
                increment=settings["strain_increment"],
            )
            manifest = write_deformation(
                directory,
                modulus_mpa=planted_modulus_mpa(rate),
                n_steps=settings["n_steps"],
                increment=settings["strain_increment"],
                relax_ps=settings["relax_ps"],
                stage=stage.name,
            )
            record = json.loads(manifest.read_text())
            record["stages"][stage.name]["final_state"] = str(final_state)
            manifest.write_text(json.dumps(record))
        return SimpleNamespace(final_state=str(final_state))

    return run


def write_tensile_rate_series(root: Path, property_name: str) -> list[Path]:
    """Planted tensile scans at holds of 1, 10 and 100 ps, measuring *property_name*.

    Every hold carries the same planted curves, so no event depends on rate.
    """
    measurement = {
        "yield_strength": "yield",
        "yield_strain": "yield",
        "breaking_strength": "breaking",
        "elongation_at_break": "elongation",
    }[property_name]
    spec = PLANTED_TENSILE[measurement]
    directories = []
    for index, hold in enumerate((1.0, 10.0, 100.0)):
        directory = root / f"rate_{index}"
        write_tensile_scan(directory, replace(spec, relax_ps=hold, stage_ps=4 * hold))
        directories.append(directory)
    return directories


def planted_curve(
    *,
    volume_jump: float = 0.1,
    enthalpy_jump: float = 400.0,
    volume_split: int = 10,
    enthalpy_split: int = 10,
    noise: float = 1.0,
) -> HeatingCurve:
    """Two expanding branches separated by a known first-order jump."""
    temperatures = np.arange(300.0, 500.0, 10.0)
    index = np.arange(len(temperatures))
    # Noise is much smaller than a real jump, and differs between observables.
    volume = (
        1.0
        + 0.0003 * (temperatures - 300.0)
        + volume_jump * (index >= volume_split)
        + noise * 0.0002 * np.sin(index * 2.1)
    )
    enthalpy = (
        -4000.0
        + 5.0 * (temperatures - 300.0)
        + enthalpy_jump * (index >= enthalpy_split)
        + noise * 0.7 * np.cos(index * 1.7)
    )
    return HeatingCurve(
        temperature_k=tuple(float(t) for t in temperatures),
        density_g_cm3=tuple(float(1.0 / v) for v in volume),
        enthalpy_kj_mol=tuple(float(h) for h in enthalpy),
        hold_ps=(1000.0,) * len(temperatures),
        pressure_bar=(1.0,) * len(temperatures),
        stages=("heating",),
    )


def write_heating(
    directory: Path,
    curve: HeatingCurve,
    *,
    chunks: tuple[int, ...] = (9, 10, 1),
) -> Path:
    """Store the public stage-result schema without running dynamics."""
    from openmmpolymer.protocols import RunManifest

    directory.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}
    start = 0
    for number, length in enumerate(chunks):
        stop = start + length
        name = f"heat_{number:02d}"
        stages[name] = {
            "name": name,
            "samples": {
                "segment_temperature_k": list(curve.temperature_k[start:stop]),
                "segment_density_g_cm3": list(curve.density_g_cm3[start:stop]),
                "segment_enthalpy_kj_mol": list(curve.enthalpy_kj_mol[start:stop]),
                "segment_duration_ps": list(curve.hold_ps[start:stop]),
                "segment_pressure_bar": list(curve.pressure_bar[start:stop]),
            },
        }
        start = stop
    assert start == curve.n_points
    RunManifest(protocol="tm_heating", seed=11, stages=stages).save(directory)
    return directory


def fake_scan_dynamics(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    runner: Any,
    *,
    box_nm: Sequence[float] = (5.0, 5.0, 5.0),
) -> None:
    """Route a rate scan's equilibration and branches through *runner*.

    The shared workflow helpers run the equilibration and the scan's own
    module its branches, so both are replaced; the equilibrated cell is
    *box_nm* rather than whatever a state file holds.
    """
    from openmmpolymer import _workflow

    monkeypatch.setattr(_workflow, "run_protocol", runner)
    monkeypatch.setattr(module, "run_protocol", runner)
    monkeypatch.setattr(_workflow, "equilibrated_box_nm", lambda state: list(box_nm))


class BuildReached(Exception):
    """Where a stand-in for the melt build stops a command-line run."""


def write_crystal(box: Any, system: Any, directory: Path) -> tuple[Path, Path]:
    """Write a prepared periodic cell as a PDB, and exactly its System."""
    import openmm as mm
    from openmm import app

    pdb = directory / "crystal.pdb"
    xml = directory / "system.xml"
    with pdb.open("w") as stream:
        app.PDBFile.writeFile(box.topology, box.positions, stream)
    xml.write_text(mm.XmlSerializer.serialize(system))
    return pdb, xml
