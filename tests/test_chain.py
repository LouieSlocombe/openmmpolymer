"""Tests for building a polymer chain as one molecule."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rdkit import Chem

from openmmpolymer.chain import (
    ChainError,
    ChainResult,
    ChainSpec,
    _atom_names,
    _characteristic_ratio,
    _trans_fraction,
    assemble_chain,
    build_chain,
)

PE = "[*]CC[*]"
PS = "[*]CC([*])c1ccccc1"
PLA = "[*]OC(C)C(=O)[*]"

PE12 = ChainSpec(
    monomer_smiles=PE, degree_of_polymerization=12, residue_name="PE", seed=5
)


@pytest.fixture(scope="module")
def pe12(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    """Four grown conformers of a twelve-unit polyethylene, built once."""
    return build_chain(
        PE12, "chain", n_conformers=4, output_dir=tmp_path_factory.mktemp("pe12")
    )


def _conformer(sdf_path: str) -> Any:
    """A written conformer, hydrogens and all."""
    return Chem.MolFromMolFile(sdf_path, removeHs=False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        # forcefill truncates to three characters, and a collision is fatal.
        ({"residue_name": "POLY"}, "one to three characters"),
        # Residue names go into a PDB column and an XML attribute.
        ({"residue_name": "P-E"}, "alphanumeric"),
        ({"degree_of_polymerization": 0}, "degree_of_polymerization"),
        # A typo fails at construction rather than silently doing nothing.
        ({"tacticity": "isotatic"}, "tacticity"),
    ],
)
def test_a_malformed_spec_is_refused_at_construction(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ChainSpec(monomer_smiles=PE, **changes)


def test_units_are_joined_head_to_tail_and_capped() -> None:
    """A ten-unit polyethylene is icosane: every attachment point consumed."""
    molecule = assemble_chain(ChainSpec(monomer_smiles=PE, degree_of_polymerization=10))
    assert Chem.MolToSmiles(molecule) == "C" * 20
    styrene = assemble_chain(ChainSpec(monomer_smiles=PS, degree_of_polymerization=4))
    assert "*" not in Chem.MolToSmiles(styrene)


@pytest.mark.parametrize(
    ("smiles", "message"),
    [
        # One or three is not a linear repeat unit.
        ("CC", "exactly two"),
        ("[*]C(C[*]", "not valid SMILES"),
        # Both on one atom leaves no backbone bond to build a frame on.
        ("[*]C(C)(C)[*]", "same atom"),
    ],
)
def test_a_malformed_monomer_is_refused_by_name(smiles: str, message: str) -> None:
    with pytest.raises(ChainError, match=message):
        assemble_chain(ChainSpec(monomer_smiles=smiles))


def test_hydrogen_cap_on_a_carbonyl_is_refused() -> None:
    """Capping a polyester with H makes an aldehyde, not an acid."""
    with pytest.raises(ChainError, match="carbonyl carbon"):
        assemble_chain(ChainSpec(monomer_smiles=PLA, degree_of_polymerization=3))


def test_an_explicit_cap_gets_the_polyester_built() -> None:
    """Naming the cap is all that is needed once the default is refused."""
    molecule = assemble_chain(
        ChainSpec(monomer_smiles=PLA, degree_of_polymerization=3, tail_cap="[*]O")
    )
    assert Chem.MolToSmiles(molecule).count("O") == 7


@pytest.mark.parametrize(
    ("cap", "message"),
    [
        # A cap with two would extend the chain rather than close it.
        ("[*]C[*]", "exactly one"),
        ("[*]C(", "not valid SMILES"),
    ],
)
def test_a_malformed_cap_is_refused(cap: str, message: str) -> None:
    with pytest.raises(ChainError, match=message):
        assemble_chain(
            ChainSpec(monomer_smiles=PE, degree_of_polymerization=2, head_cap=cap)
        )


@pytest.mark.parametrize(
    ("monomer", "tacticity", "distinct_tags"),
    [
        (PS, "isotactic", 1),
        (PS, "syndiotactic", 2),
        # Polyethylene has nothing to set, and that is not an error.
        (PE, "isotactic", 0),
    ],
)
def test_tacticity_sets_the_backbone_stereocentres(
    monomer: str, tacticity: str, distinct_tags: int
) -> None:
    molecule = assemble_chain(
        ChainSpec(
            monomer_smiles=monomer, degree_of_polymerization=6, tacticity=tacticity
        )
    )
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    tags = {
        atom.GetChiralTag()
        for atom in molecule.GetAtoms()  # type: ignore[no-untyped-call]
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    }
    assert len(tags) == distinct_tags


def test_atom_names_are_unique_and_fit_four_columns() -> None:
    """PDB atom names are truncated to four without complaint, so they must fit."""
    molecule = Chem.AddHs(
        assemble_chain(ChainSpec(monomer_smiles=PE, degree_of_polymerization=60))
    )
    names = _atom_names(molecule)
    assert len(set(names)) == len(names) == molecule.GetNumAtoms()
    assert max(len(name) for name in names) <= 4


def test_the_trans_fraction_inverts_the_characteristic_ratio() -> None:
    """The documented worked values; a stiffer chain wants more trans."""
    assert _trans_fraction(7.0) == pytest.approx(0.681, abs=0.002)
    assert _trans_fraction(4.31) == pytest.approx(0.550, abs=0.002)
    assert _trans_fraction(4.0) < _trans_fraction(7.0) < _trans_fraction(10.0)


def test_the_trans_fraction_stays_a_probability() -> None:
    """Absurd targets clamp rather than producing a probability outside [0, 1]."""
    assert 0.0 < _trans_fraction(1.0) < _trans_fraction(1000.0) < 1.0


def test_characteristic_ratio_of_a_straight_chain() -> None:
    """A rod of n bonds of length l has R^2 = (n l)^2, so C = n."""
    positions = np.array([[i * 0.15, 0.0, 0.0] for i in range(11)])
    assert _characteristic_ratio(positions, range(11), 0.15) == pytest.approx(10.0)


def test_a_chain_is_written_as_one_file_pair_per_conformer(pe12: ChainResult) -> None:
    """Packing needs the PDBs; parameterisation needs an SDF's bond orders."""
    assert len(pe12.sdf_paths) == len(pe12.pdb_paths) == 4
    assert all(Path(path).is_file() for path in (*pe12.sdf_paths, *pe12.pdb_paths))
    assert pe12.embedder == "grown"


def test_molar_mass_counts_the_explicit_hydrogens_once(pe12: ChainResult) -> None:
    """A twelve-unit polyethylene capped with hydrogen is C24H50."""
    assert pe12.molar_mass_g_mol == pytest.approx(338.66, abs=0.05)
    assert pe12.n_atoms == 74


def test_conformers_differ_from_one_another(pe12: ChainResult) -> None:
    """Packing one conformer N times makes a cell of N identical coils."""
    assert len(set(pe12.radius_of_gyration_nm)) == 4


def test_the_seed_decides_the_conformers(pe12: ChainResult, tmp_path: Path) -> None:
    """The same seed gives the same coordinates, and another seed other ones.

    Otherwise every run of a study would pack the same cell.
    """
    again = build_chain(PE12, "again", n_conformers=2, output_dir=tmp_path)
    other = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=12, seed=6),
        "other",
        n_conformers=2,
        output_dir=tmp_path,
    )
    assert again.radius_of_gyration_nm == pe12.radius_of_gyration_nm[:2]
    assert other.radius_of_gyration_nm != again.radius_of_gyration_nm


def test_the_backbone_runs_bonded_from_head_to_tail(pe12: ChainResult) -> None:
    """Nothing downstream can recover it: the caps consumed the attachment points."""
    molecule = _conformer(pe12.sdf_paths[0])
    assert len(set(pe12.backbone)) == len(pe12.backbone) == 24
    assert all(
        molecule.GetAtomWithIdx(index).GetSymbol() == "C" for index in pe12.backbone
    )
    assert all(
        molecule.GetBondBetweenAtoms(first, second) is not None
        for first, second in pairwise(pe12.backbone)
    )


def test_the_extent_bound_never_understates_the_real_extent(pe12: ChainResult) -> None:
    """packmol cannot place a conformer longer than the box."""
    positions = np.asarray(_conformer(pe12.sdf_paths[0]).GetConformer().GetPositions())
    deltas = (positions[:, None, :] - positions[None, :, :]) / 10.0
    real = float(np.sqrt((deltas**2).sum(axis=-1)).max())
    assert pe12.max_extent_nm[0] >= real - 1e-9


def test_short_chains_are_embedded_with_etkdg(tmp_path: Path) -> None:
    """The grown embedder needs four units to have a template to stamp."""
    spec = ChainSpec(monomer_smiles=PE, degree_of_polymerization=2)
    assert build_chain(spec, "short", output_dir=tmp_path).embedder == "etkdg"
    with pytest.raises(ChainError, match="at least 4 units"):
        build_chain(spec, "grown", output_dir=tmp_path, embedder="grown")


def test_the_pdb_is_one_residue_and_matches_the_sdf(tmp_path: Path) -> None:
    """The topology comes from the PDB and the parameters from the SDF.

    They have to describe the same atoms in the same order, or the System ends
    up with every parameter on the wrong atom and nothing raises.
    """
    from openmm import app

    result = build_chain(
        ChainSpec(monomer_smiles=PS, degree_of_polymerization=4, residue_name="PS"),
        "chain",
        output_dir=tmp_path,
    )
    pdb = app.PDBFile(result.pdb_paths[0])
    sdf = _conformer(result.sdf_paths[0])

    assert pdb.topology.getNumResidues() == 1
    assert next(pdb.topology.residues()).name == "PS"
    assert pdb.topology.getNumAtoms() == sdf.GetNumAtoms() == result.n_atoms
    assert sum(1 for _ in pdb.topology.bonds()) == sdf.GetNumBonds()
    assert [atom.element.symbol for atom in pdb.topology.atoms()] == [
        atom.GetSymbol() for atom in sdf.GetAtoms()
    ]


def test_the_grown_embedder_reproduces_the_expected_chain_dimensions(
    tmp_path: Path,
) -> None:
    """The point of sampling torsions rather than tabulating them.

    A single chain's end-to-end distance has the scatter of a random walk, so
    this averages over a sample.
    """
    result = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=40, residue_name="PE"),
        "chain",
        n_conformers=24,
        output_dir=tmp_path,
    )
    assert result.characteristic_ratio == pytest.approx(7.0, rel=0.35)


def test_etkdg_collapses_a_long_chain(tmp_path: Path) -> None:
    """Recorded because it is the reason the default is what it is."""
    result = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=40, residue_name="PE"),
        "chain",
        n_conformers=2,
        output_dir=tmp_path,
        embedder="etkdg",
    )
    assert result.characteristic_ratio is not None
    assert result.characteristic_ratio < 4.0
