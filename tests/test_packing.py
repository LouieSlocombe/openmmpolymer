"""Tests for the box arithmetic, the packmol input, and the packing checks."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from openmmpolymer.packing import (
    AVOGADRO,
    PackedComponent,
    PackmolError,
    box_edge_nm,
    check_packing,
    density_g_cm3,
    find_packmol,
    find_rings,
    load_positions_nm,
    pack_box,
    packmol_version,
    render_packmol_input,
)

from .helpers import build_dimer_pdb


def test_box_edge_matches_the_density_it_was_asked_for() -> None:
    """The cell holds exactly the material it was sized for."""
    edge = box_edge_nm([30], [563.1], 0.3)
    assert density_g_cm3([30], [563.1], edge**3) == pytest.approx(0.3)


def test_box_edge_against_a_hand_computed_volume() -> None:
    """The conversion between g/mol, g/cm3 and nm3 is the documented one."""
    expected = (100 * 18.0 * 1.0e21 / (1.0 * AVOGADRO)) ** (1 / 3)
    assert box_edge_nm([100], [18.0], 1.0) == pytest.approx(expected)


def test_box_edge_scales_as_the_cube_root_of_the_count() -> None:
    """Eight times the chains is twice the edge."""
    assert box_edge_nm([80], [500.0], 0.5) == pytest.approx(
        2.0 * box_edge_nm([10], [500.0], 0.5)
    )


def test_box_edge_refuses_an_empty_cell() -> None:
    """Nothing to pack is an error, not a zero-sized box."""
    with pytest.raises(ValueError, match="Nothing to pack"):
        box_edge_nm([0], [500.0], 0.5)


def test_box_edge_refuses_a_non_positive_density() -> None:
    """A density of zero would ask for an infinite cell."""
    with pytest.raises(ValueError, match="density_g_cm3"):
        box_edge_nm([10], [500.0], 0.0)


def test_packmol_input_converts_nanometres_to_angstrom_once() -> None:
    """The rendered input is in angstrom, which is what packmol reads."""
    text = render_packmol_input(
        [PackedComponent("a.pdb", 3)],
        (24.0, 24.0, 24.0),
        "packed.pdb",
        tolerance_angstrom=2.0,
        inset_angstrom=1.0,
        seed=7,
    )
    assert "tolerance 2.0000" in text
    assert "seed 7" in text
    assert "filetype pdb" in text
    assert "output packed.pdb" in text
    assert "structure a.pdb" in text
    assert "  number 3" in text


def test_packmol_region_is_inset_from_every_face() -> None:
    """Molecules stay a full tolerance apart across the periodic boundary.

    An atom at the inset and one at the far face are two insets apart across
    the boundary, so the inset has to be at least half the tolerance.
    """
    text = render_packmol_input(
        [PackedComponent("a.pdb", 1)],
        (24.0, 24.0, 24.0),
        "packed.pdb",
        tolerance_angstrom=2.0,
        inset_angstrom=1.0,
        seed=1,
    )
    assert "inside box 1.0000 1.0000 1.0000 23.0000 23.0000 23.0000" in text


def test_packmol_input_skips_a_component_with_no_copies() -> None:
    """A zero count writes no block rather than an empty one."""
    text = render_packmol_input(
        [PackedComponent("a.pdb", 0), PackedComponent("b.pdb", 2)],
        (24.0, 24.0, 24.0),
        "packed.pdb",
        tolerance_angstrom=2.0,
        inset_angstrom=1.0,
        seed=1,
    )
    assert "a.pdb" not in text
    assert "b.pdb" in text


def test_packmol_input_includes_nloop_only_when_asked() -> None:
    """Left out, packmol uses its own default rather than ours."""
    common = {
        "box_angstrom": (24.0, 24.0, 24.0),
        "output_pdb": "packed.pdb",
        "tolerance_angstrom": 2.0,
        "inset_angstrom": 1.0,
        "seed": 1,
    }
    components = [PackedComponent("a.pdb", 1)]
    assert "nloop" not in render_packmol_input(components, **common)  # type: ignore[arg-type]
    assert "nloop 50" in render_packmol_input(components, nloop=50, **common)  # type: ignore[arg-type]


def test_find_packmol_reports_how_to_get_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing binary says where to get one rather than failing obscurely."""
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("PACKMOL", raising=False)
    with pytest.raises(PackmolError, match="conda install"):
        find_packmol()


def test_find_packmol_rejects_a_path_that_is_not_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit path that does not exist fails at once, by name."""
    monkeypatch.delenv("PACKMOL", raising=False)
    with pytest.raises(PackmolError, match="not an executable file"):
        find_packmol(tmp_path / "nowhere")


def test_find_packmol_honours_the_environment_override(fake_packmol: Path) -> None:
    """The stub on PATH is what gets found."""
    assert Path(find_packmol()).name == "packmol"


def test_pack_box_raises_when_packmol_does_not_report_success(
    fake_packmol: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """packmol can exit zero without converging, so the output is read."""
    fake_packmol.write_text("#!/bin/sh\ncat > /dev/null\necho 'not converged'\n")
    fake_packmol.chmod(0o755)
    build_dimer_pdb(tmp_path / "dimer.pdb")
    with pytest.raises(PackmolError, match="did not report success"):
        pack_box([PackedComponent(str(tmp_path / "dimer.pdb"), 2)], 3.0)


def test_pack_box_reports_the_exit_code_and_what_it_means(
    fake_packmol: Path, tmp_path: Path
) -> None:
    """packmol's own failure codes start at 170 and are worth naming."""
    fake_packmol.write_text("#!/bin/sh\ncat > /dev/null\necho boom\nexit 171\n")
    fake_packmol.chmod(0o755)
    build_dimer_pdb(tmp_path / "dimer.pdb")
    with pytest.raises(PackmolError, match=r"171.*could not satisfy"):
        pack_box([PackedComponent(str(tmp_path / "dimer.pdb"), 2)], 3.0)


def test_pack_box_writes_the_input_it_used(
    fake_packmol: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The generated input stays on disk, because it is the thing to inspect."""
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    monkeypatch.setenv("PACKMOL_FAKE_OUTPUT", source)
    result = pack_box([PackedComponent(source, 1)], 3.0, "packed.pdb")
    text = Path(result.input_path).read_text()
    assert "tolerance 2.0000" in text
    assert result.box_nm == (3.0, 3.0, 3.0)
    assert result.n_molecules == 1


def test_load_positions_reads_through_the_pdb_parser(tmp_path: Path) -> None:
    """Coordinates come back in nanometres, in file order."""
    path = Path(build_dimer_pdb(tmp_path / "dimer.pdb"))
    positions = load_positions_nm(path)
    assert positions.shape == (2, 3)
    assert positions[1][0] == pytest.approx(0.153, abs=1e-3)


def _two_molecule_topology(separation_nm: float) -> tuple[object, np.ndarray]:
    """Two dimers, *separation_nm* apart, with their bonds recorded."""
    from openmm import app

    topology = app.Topology()
    chain = topology.addChain()
    carbon = app.Element.getBySymbol("C")
    positions = []
    for index in range(2):
        residue = topology.addResidue("DIM", chain)
        first = topology.addAtom("C1", carbon, residue)
        second = topology.addAtom("C2", carbon, residue)
        topology.addBond(first, second)
        offset = index * separation_nm
        positions.extend([[offset, 0.0, 0.0], [offset + 0.153, 0.0, 0.0]])
    return topology, np.asarray(positions, dtype=np.float64)


def test_check_packing_passes_a_well_separated_cell() -> None:
    """Two dimers a nanometre apart are fine."""
    topology, positions = _two_molecule_topology(1.0)
    check_packing(topology, positions, check_rings=False)


def test_check_packing_catches_a_molecule_split_across_the_boundary() -> None:
    """A bond the width of the cell is the signature, and it is fatal."""
    topology, positions = _two_molecule_topology(1.0)
    positions[1] = [5.0, 0.0, 0.0]
    with pytest.raises(PackmolError, match="bonded but"):
        check_packing(topology, positions, check_rings=False)


def test_check_packing_catches_two_molecules_on_top_of_each_other() -> None:
    """Overlap between molecules is fatal; the C-H distance inside one is not."""
    topology, positions = _two_molecule_topology(0.05)
    with pytest.raises(PackmolError, match="different molecules"):
        check_packing(topology, positions, check_rings=False)


def test_check_packing_ignores_bonded_neighbours() -> None:
    """packmol's tolerance is intermolecular; a 0.153 nm bond must not trip it."""
    topology, positions = _two_molecule_topology(2.0)
    check_packing(topology, positions, min_heavy_nm=0.30, check_rings=False)


def _benzene_topology() -> tuple[object, np.ndarray]:
    """A flat six-ring in the xy plane, plus a separate two-atom rod."""
    from openmm import app

    topology = app.Topology()
    chain = topology.addChain()
    carbon = app.Element.getBySymbol("C")
    ring_residue = topology.addResidue("BEN", chain)
    atoms = [topology.addAtom(f"C{i}", carbon, ring_residue) for i in range(6)]
    for index in range(6):
        topology.addBond(atoms[index], atoms[(index + 1) % 6])

    radius = 0.14
    positions = [
        [radius * math.cos(i * math.pi / 3), radius * math.sin(i * math.pi / 3), 0.0]
        for i in range(6)
    ]
    rod_residue = topology.addResidue("ROD", chain)
    first = topology.addAtom("C1", carbon, rod_residue)
    second = topology.addAtom("C2", carbon, rod_residue)
    topology.addBond(first, second)
    return topology, np.asarray(positions, dtype=np.float64)


def test_find_rings_finds_the_six_ring() -> None:
    """The ring finder sees a benzene-shaped cycle."""
    topology, _ = _benzene_topology()
    rings = find_rings(topology)
    assert [len(ring) for ring in rings] == [6]


def test_check_packing_catches_a_bond_threaded_through_a_ring() -> None:
    """The classic packmol failure for anything with a ring in it."""
    topology, ring = _benzene_topology()
    speared = np.vstack([ring, [[0.0, 0.0, -0.1], [0.0, 0.0, 0.1]]])
    with pytest.raises(PackmolError, match="passes through the ring"):
        check_packing(topology, speared, min_heavy_nm=0.05)


def test_check_packing_allows_a_bond_that_misses_the_ring() -> None:
    """A segment crossing the ring's plane outside it is not threaded."""
    topology, ring = _benzene_topology()
    clear = np.vstack([ring, [[1.0, 0.0, -0.1], [1.0, 0.0, 0.1]]])
    check_packing(topology, clear, min_heavy_nm=0.05)


@pytest.mark.packmol
def test_real_packmol_reports_a_version() -> None:
    """The version probe works against the real binary."""
    if shutil.which("packmol") is None:
        pytest.skip("packmol is not on PATH")
    assert packmol_version()[0] >= 20
