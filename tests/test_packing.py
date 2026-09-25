"""Tests for the box arithmetic, the packmol input, and the packing checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.mdsystem import assemble_box
from openmmpolymer.packing import (
    PackedComponent,
    PackmolError,
    _render_packmol_input,
    box_edge_nm,
    check_packing,
    distribute_conformers,
    pack_box,
    read_packed_pdb,
)

from .helpers import build_dimer_pdb


def test_box_edge_against_a_hand_computed_volume() -> None:
    """A hundred waters at 1 g/cm3 fill 2.989 nm3, a cube 1.4405 nm on edge."""
    assert box_edge_nm([100], [18.0], 1.0) == pytest.approx(1.44048, abs=1e-5)


def test_box_edge_scales_as_the_cube_root_of_the_count() -> None:
    """Eight times the chains is twice the edge."""
    assert box_edge_nm([80], [500.0], 0.5) == pytest.approx(
        2.0 * box_edge_nm([10], [500.0], 0.5)
    )


@pytest.mark.parametrize(
    ("count", "density", "message"),
    [(0, 0.5, "Nothing to pack"), (10, 0.0, "density_g_cm3")],
)
def test_box_edge_refuses_a_cell_it_cannot_size(
    count: int, density: float, message: str
) -> None:
    """Nothing to pack is an error, and so is a density that asks for infinity."""
    with pytest.raises(ValueError, match=message):
        box_edge_nm([count], [500.0], density)


@pytest.mark.parametrize(
    ("components", "fragments", "absent"),
    [
        (
            [PackedComponent("a.pdb", 3)],
            [
                "tolerance 2.0000",
                "filetype pdb",
                "output packed.pdb",
                "seed 7",
                "structure a.pdb",
                "  number 3",
            ],
            [],
        ),
        # A zero count writes no block rather than an empty one.
        (
            [PackedComponent("a.pdb", 0), PackedComponent("b.pdb", 2)],
            ["structure b.pdb", "  number 2"],
            ["a.pdb"],
        ),
        (
            [PackedComponent(f"c{index}.pdb", 1) for index in range(3)],
            ["structure c0.pdb", "structure c2.pdb"],
            [],
        ),
    ],
)
def test_the_packmol_input_has_one_block_per_placed_structure(
    components: list[PackedComponent], fragments: list[str], absent: list[str]
) -> None:
    """And numbers residues across the whole output in every block.

    packmol restarts residue numbering in each block by default, and past the
    twenty-sixth the chain identifiers run out as well, so molecules start
    sharing a chain and residue number and OpenMM merges them - measured at 40
    conformers, 37 residues came back instead of 40.
    """
    text = _render_packmol_input(
        components,
        (24.0, 24.0, 24.0),
        "packed.pdb",
        tolerance_angstrom=2.0,
        inset_angstrom=1.0,
        seed=7,
    )
    assert all(fragment in text for fragment in fragments)
    assert not any(name in text for name in absent)
    placed = sum(1 for component in components if component.count)
    assert text.count("  resnumbers 3") == text.count("end structure") == placed


@pytest.mark.parametrize("named_by_environment", [False, True])
def test_pack_box_converts_to_angstrom_and_insets_the_region(
    fake_packmol: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    named_by_environment: bool,
) -> None:
    """The region is held back from each face by half the tolerance.

    An atom at the inset and one at the far face are then a full tolerance
    apart across the periodic boundary. The input stays on disk, because it is
    the thing to inspect.
    """
    if named_by_environment:
        monkeypatch.setenv("PACKMOL", str(fake_packmol))
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    monkeypatch.setenv("PACKMOL_FAKE_OUTPUT", source)
    result = pack_box([PackedComponent(source, 1)], 3.0, "packed.pdb")
    text = Path(result.input_path).read_text()
    assert "tolerance 2.0000" in text
    assert "inside box 1.0000 1.0000 1.0000 29.0000 29.0000 29.0000" in text
    assert result.box_nm == (3.0, 3.0, 3.0)
    assert result.n_molecules == 1


def test_a_missing_packmol_says_how_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rather than failing obscurely."""
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("PACKMOL", raising=False)
    with pytest.raises(PackmolError, match="conda install"):
        pack_box([PackedComponent("dimer.pdb", 1)], 3.0)


def test_a_packmol_named_by_the_environment_has_to_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It fails at once, by name, rather than falling back to PATH."""
    monkeypatch.setenv("PACKMOL", str(tmp_path / "nowhere"))
    with pytest.raises(PackmolError, match="not an executable file"):
        pack_box([PackedComponent("dimer.pdb", 1)], 3.0)


@pytest.mark.parametrize(
    ("script", "timeout_s", "message"),
    [
        # packmol can exit zero without converging, so the output is read.
        ("cat > /dev/null\necho 'not converged'", None, "did not report success"),
        # Its own failure codes start at 170 and are worth naming.
        ("cat > /dev/null\necho boom\nexit 171", None, r"171.*could not satisfy"),
        ("sleep 5", 0.1, "did not finish within 0.1 s"),
    ],
)
def test_a_packmol_failure_is_reported_for_what_it_is(
    fake_packmol: Path,
    tmp_path: Path,
    script: str,
    timeout_s: float | None,
    message: str,
) -> None:
    fake_packmol.write_text(f"#!/bin/sh\n{script}\n")
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    with pytest.raises(PackmolError, match=message):
        pack_box([PackedComponent(source, 2)], 3.0, timeout_s=timeout_s)


def test_packed_coordinates_are_read_in_nanometres(tmp_path: Path) -> None:
    """Through the PDB parser, in file order."""
    topology, positions = read_packed_pdb(build_dimer_pdb(tmp_path / "dimer.pdb"))
    assert topology.getNumAtoms() == 2
    assert positions.shape == (2, 3)
    assert positions[1][0] == pytest.approx(0.153, abs=1e-3)


def _two_dimers(separation_nm: float, symbol: str = "C") -> tuple[Any, np.ndarray]:
    """Two bonded pairs of *symbol* atoms, the second *separation_nm* along x."""
    from openmm import app

    topology = app.Topology()
    chain = topology.addChain()
    element = app.Element.getBySymbol(symbol)
    positions = []
    for index in range(2):
        residue = topology.addResidue("DIM", chain)
        first = topology.addAtom("A1", element, residue)
        second = topology.addAtom("A2", element, residue)
        topology.addBond(first, second)
        offset = index * separation_nm
        positions.extend([[offset, 0.0, 0.0], [offset + 0.153, 0.0, 0.0]])
    return topology, np.asarray(positions, dtype=np.float64)


def test_check_packing_passes_a_well_separated_cell() -> None:
    """Including the bonds, which are shorter than the intermolecular limit.

    packmol's tolerance is intermolecular, so bonded neighbours must not trip
    it: a C-C bond is 0.153 nm against a 0.20 nm limit.
    """
    check_packing(*_two_dimers(1.0))


def test_check_packing_catches_a_molecule_split_across_the_boundary() -> None:
    """A bond the width of the cell is the signature, and it is fatal."""
    topology, positions = _two_dimers(1.0)
    positions[1] = [5.0, 0.0, 0.0]
    with pytest.raises(PackmolError, match="bonded but"):
        check_packing(topology, positions)


def test_check_packing_catches_molecules_on_top_of_each_other() -> None:
    """Heavy atoms of different molecules 0.18 nm apart are too close."""
    with pytest.raises(PackmolError, match="different molecules"):
        check_packing(*_two_dimers(0.33))


def test_check_packing_lets_hydrogens_come_closer() -> None:
    """Where either atom is a hydrogen the limit is 0.15 nm, not 0.20."""
    check_packing(*_two_dimers(0.33, "H"))


def _ring_and_rod(
    *, rod_x_nm: float, rod_in_own_residue: bool
) -> tuple[Any, np.ndarray]:
    """A flat six-ring of radius 0.2 nm at the origin, and a rod along z.

    The ring is wide enough, and the 0.24 nm rod long enough, that a rod
    through the centre clears every contact and bond limit, so threading is the
    only fault. On its own the rod is a molecule of a different size from the
    ring's. Otherwise there are two identical molecules, each a ring and a rod,
    the second's rod through the first's ring: then the rings are found once
    and shifted onto the copy.
    """
    from openmm import app

    topology = app.Topology()
    chain = topology.addChain()
    carbon = app.Element.getBySymbol("C")
    hexagon = [
        [0.2 * math.cos(i * math.pi / 3), 0.2 * math.sin(i * math.pi / 3), 0.0]
        for i in range(6)
    ]
    rod = [[rod_x_nm, 0.0, -0.12], [rod_x_nm, 0.0, 0.12]]
    if rod_in_own_residue:
        molecules = [(hexagon, []), ([], rod)]
    else:
        far_ring = [[x + 10.0, y, z] for x, y, z in hexagon]
        far_rod = [[x + 5.0, y, z] for x, y, z in rod]
        molecules = [(hexagon, far_rod), (far_ring, rod)]

    positions: list[list[float]] = []
    for ring, pair in molecules:
        residue = topology.addResidue("MOL", chain)
        atoms = [topology.addAtom(f"C{i}", carbon, residue) for i in range(len(ring))]
        for index, atom in enumerate(atoms):
            topology.addBond(atom, atoms[(index + 1) % len(atoms)])
        if pair:
            ends = [topology.addAtom(f"R{i}", carbon, residue) for i in range(2)]
            topology.addBond(*ends)
        positions.extend([*ring, *pair])
    return topology, np.asarray(positions, dtype=np.float64)


@pytest.mark.parametrize("rod_in_own_residue", [True, False])
def test_check_packing_catches_a_bond_threaded_through_a_ring(
    rod_in_own_residue: bool,
) -> None:
    """The classic packmol failure for anything with a ring in it."""
    topology, positions = _ring_and_rod(
        rod_x_nm=0.0, rod_in_own_residue=rod_in_own_residue
    )
    with pytest.raises(PackmolError, match="passes through the ring"):
        check_packing(topology, positions)


def test_check_packing_allows_a_bond_that_misses_the_ring() -> None:
    """A segment crossing the ring's plane outside it is not threaded."""
    check_packing(*_ring_and_rod(rod_x_nm=1.0, rod_in_own_residue=False))


@pytest.mark.parametrize(
    ("n_conformers", "n_molecules", "counts"),
    [
        # Fewer conformers than molecules still beats one conformation repeated.
        (3, 10, [4, 3, 3]),
        # The default, and what build_chain is set up for.
        (4, 4, [1, 1, 1, 1]),
        # A conformer with nothing to place is not given to packmol at all.
        (3, 2, [1, 1]),
    ],
)
def test_conformers_are_spread_evenly_over_the_molecules(
    n_conformers: int, n_molecules: int, counts: list[int]
) -> None:
    paths = [f"{index}.pdb" for index in range(n_conformers)]
    components = distribute_conformers(paths, n_molecules)
    assert [component.count for component in components] == counts
    assert [component.pdb_path for component in components] == paths[: len(counts)]


def test_distributing_over_no_conformers_is_refused() -> None:
    """A clearer failure than dividing by zero."""
    with pytest.raises(ValueError, match="No conformers"):
        distribute_conformers([], 5)


@pytest.mark.packmol
def test_real_packmol_packs_a_cell_that_passes_the_checks(tmp_path: Path) -> None:
    """Every molecule whole, none overlapping, and in the order they were given."""
    components = [PackedComponent(build_dimer_pdb(tmp_path / "dimer.pdb"), 40)]
    packed = pack_box(components, 3.0, tmp_path / "packed.pdb", seed=3)
    box = assemble_box(components, packed.packed_pdb, packed.box_nm)
    check_packing(box.topology, box.positions_nm)
    assert box.n_molecules == 40
