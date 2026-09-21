"""Builders the tests share.

Topologies are built in code rather than committed as files: a hand-aligned PDB
in the repository is a column-counting exercise that goes wrong silently, and
what these tests need is small enough to read.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
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
