"""Tests for building a polymer chain as one molecule."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from openmmpolymer.chain import (
    ChainError,
    ChainResult,
    ChainSpec,
    assemble_chain,
    atom_names,
    backbone_path,
    build_chain,
    characteristic_ratio,
    molar_mass_g_mol,
    trans_fraction,
)

PE = "[*]CC[*]"
PS = "[*]CC([*])c1ccccc1"
PLA = "[*]OC(C)C(=O)[*]"


def test_spec_rejects_a_residue_name_forcefill_would_truncate() -> None:
    """forcefill truncates to three characters, and a collision there is fatal."""
    with pytest.raises(ValueError, match="one to three characters"):
        ChainSpec(monomer_smiles=PE, residue_name="POLY")


def test_spec_rejects_a_non_alphanumeric_residue_name() -> None:
    """Residue names go into a PDB column and an XML attribute."""
    with pytest.raises(ValueError, match="alphanumeric"):
        ChainSpec(monomer_smiles=PE, residue_name="P-E")


def test_spec_rejects_a_zero_degree_of_polymerization() -> None:
    """A chain of no units is not a chain."""
    with pytest.raises(ValueError, match="degree_of_polymerization"):
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=0)


def test_spec_rejects_an_unknown_tacticity() -> None:
    """A typo fails at construction rather than silently doing nothing."""
    with pytest.raises(ValueError, match="tacticity"):
        ChainSpec(monomer_smiles=PE, tacticity="isotatic")


def test_assemble_joins_units_head_to_tail() -> None:
    """A ten-unit polyethylene is decane: twenty carbons, all in one chain."""
    molecule = assemble_chain(ChainSpec(monomer_smiles=PE, degree_of_polymerization=10))
    carbons = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6]
    assert len(carbons) == 20
    assert not [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]


def test_assemble_leaves_no_dummy_atoms_behind() -> None:
    """Every attachment point is consumed by a bond or a cap."""
    molecule = assemble_chain(ChainSpec(monomer_smiles=PS, degree_of_polymerization=4))
    assert all(atom.GetAtomicNum() != 0 for atom in molecule.GetAtoms())


def test_molar_mass_counts_explicit_hydrogens_once() -> None:
    """A five-unit polyethylene capped with hydrogen is decane, 142 g/mol."""
    from rdkit import Chem

    molecule = Chem.AddHs(
        assemble_chain(ChainSpec(monomer_smiles=PE, degree_of_polymerization=5))
    )
    assert molar_mass_g_mol(molecule) == pytest.approx(142.3, abs=0.2)


def test_monomer_needs_exactly_two_attachment_points() -> None:
    """One or three is not a linear repeat unit, and the message says so."""
    with pytest.raises(ChainError, match="exactly two"):
        assemble_chain(ChainSpec(monomer_smiles="CC"))


def test_monomer_smiles_must_parse() -> None:
    """A malformed SMILES is caught at the front door."""
    with pytest.raises(ChainError, match="not valid SMILES"):
        assemble_chain(ChainSpec(monomer_smiles="[*]C(C[*]"))


def test_monomer_needs_two_distinct_backbone_atoms() -> None:
    """Both attachment points on one atom leaves no bond to build a frame on."""
    with pytest.raises(ChainError, match="same atom"):
        assemble_chain(ChainSpec(monomer_smiles="[*]C(C)(C)[*]"))


def test_hydrogen_cap_on_a_carbonyl_is_refused() -> None:
    """Capping a polyester with H makes an aldehyde, not an acid."""
    with pytest.raises(ChainError, match="carbonyl carbon"):
        assemble_chain(ChainSpec(monomer_smiles=PLA, degree_of_polymerization=3))


def test_an_explicit_cap_gets_the_polyester_built() -> None:
    """Naming the cap is all that is needed once the default is refused."""
    molecule = assemble_chain(
        ChainSpec(monomer_smiles=PLA, degree_of_polymerization=3, tail_cap="[*]O")
    )
    assert sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 8) == 7


def test_a_cap_needs_exactly_one_attachment_point() -> None:
    """A cap with two would extend the chain rather than close it."""
    with pytest.raises(ChainError, match="exactly one"):
        assemble_chain(
            ChainSpec(monomer_smiles=PE, degree_of_polymerization=2, head_cap="[*]C[*]")
        )


def test_backbone_path_runs_the_length_of_the_chain() -> None:
    """Polyethylene has two backbone atoms per unit."""
    from rdkit import Chem

    molecule = Chem.AddHs(
        assemble_chain(ChainSpec(monomer_smiles=PE, degree_of_polymerization=8))
    )
    assert len(backbone_path(molecule, 8)) == 16


def test_atom_names_are_unique_and_fit_four_columns() -> None:
    """PDB atom names are truncated to four without complaint, so they must fit."""
    from rdkit import Chem

    molecule = Chem.AddHs(
        assemble_chain(ChainSpec(monomer_smiles=PE, degree_of_polymerization=60))
    )
    names = atom_names(molecule)
    assert len(set(names)) == len(names)
    assert max(len(name) for name in names) <= 4


def test_trans_fraction_inverts_the_characteristic_ratio() -> None:
    """The documented worked values, both ways round."""
    assert trans_fraction(7.0) == pytest.approx(0.681, abs=0.002)
    assert trans_fraction(4.31) == pytest.approx(0.550, abs=0.002)


def test_trans_fraction_is_monotonic_in_the_target() -> None:
    """A stiffer chain wants more trans."""
    assert trans_fraction(4.0) < trans_fraction(7.0) < trans_fraction(10.0)


def test_trans_fraction_stays_a_probability() -> None:
    """Absurd targets clamp rather than producing a probability outside [0, 1]."""
    assert 0.0 < trans_fraction(1.0) < 1.0
    assert 0.0 < trans_fraction(1000.0) < 1.0


def test_characteristic_ratio_of_a_straight_chain() -> None:
    """A rod of n bonds of length l has R^2 = (n l)^2, so C = n."""
    positions = np.array([[i * 0.15, 0.0, 0.0] for i in range(11)])
    assert characteristic_ratio(positions, range(11), 0.15) == pytest.approx(10.0)


def test_build_chain_writes_one_file_pair_per_conformer(tmp_path: Path) -> None:
    """Packing needs the PDBs; parameterisation needs an SDF's bond orders."""
    result = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=5, residue_name="PE"),
        "chain",
        n_conformers=3,
        output_dir=tmp_path,
    )
    assert len(result.sdf_paths) == len(result.pdb_paths) == 3
    assert all(Path(path).is_file() for path in result.sdf_paths)
    assert all(Path(path).is_file() for path in result.pdb_paths)


def test_conformers_differ_from_one_another(tmp_path: Path) -> None:
    """Packing one conformer N times makes a cell of N identical coils."""
    result = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=12, residue_name="PE"),
        "chain",
        n_conformers=4,
        output_dir=tmp_path,
    )
    assert len(set(result.radius_of_gyration_nm)) == 4


def test_build_chain_is_reproducible(tmp_path: Path) -> None:
    """The same seed gives the same coordinates."""
    spec = ChainSpec(
        monomer_smiles=PE, degree_of_polymerization=12, residue_name="PE", seed=5
    )
    first = build_chain(spec, "a", n_conformers=2, output_dir=tmp_path)
    second = build_chain(spec, "b", n_conformers=2, output_dir=tmp_path)
    assert first.radius_of_gyration_nm == second.radius_of_gyration_nm


def test_a_different_seed_gives_different_chains(tmp_path: Path) -> None:
    """Otherwise every run of a study would pack the same cell."""

    def make(seed: int, stem: str) -> ChainResult:
        return build_chain(
            ChainSpec(
                monomer_smiles=PE,
                degree_of_polymerization=12,
                residue_name="PE",
                seed=seed,
            ),
            stem,
            n_conformers=2,
            output_dir=tmp_path,
        )

    assert make(1, "a").radius_of_gyration_nm != make(2, "b").radius_of_gyration_nm


def test_short_chains_use_etkdg_and_long_ones_are_grown(tmp_path: Path) -> None:
    """The grown embedder needs four units to have a template to stamp."""
    short = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=2, residue_name="PE"),
        "short",
        output_dir=tmp_path,
    )
    long = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=8, residue_name="PE"),
        "long",
        output_dir=tmp_path,
    )
    assert short.embedder == "etkdg"
    assert long.embedder == "grown"


def test_growing_a_chain_too_short_to_grow_is_refused(tmp_path: Path) -> None:
    """Asking for the impossible says which embedder to use instead."""
    with pytest.raises(ChainError, match="at least 4 units"):
        build_chain(
            ChainSpec(monomer_smiles=PE, degree_of_polymerization=2),
            "x",
            output_dir=tmp_path,
            embedder="grown",
        )


def test_the_grown_embedder_reproduces_the_expected_chain_dimensions(
    tmp_path: Path,
) -> None:
    """The point of sampling torsions rather than tabulating them.

    A single chain's end-to-end distance has the scatter of a random walk, so
    this averages over a sample. ETKDG on the same chain comes out around a
    quarter of this, which is why it is not the default.
    """
    result = build_chain(
        ChainSpec(
            monomer_smiles=PE,
            degree_of_polymerization=40,
            residue_name="PE",
            characteristic_ratio=7.0,
        ),
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
        n_conformers=4,
        output_dir=tmp_path,
        embedder="etkdg",
    )
    assert result.characteristic_ratio is not None
    assert result.characteristic_ratio < 4.0


def test_the_pdb_is_one_residue_and_matches_the_sdf(tmp_path: Path) -> None:
    """The topology comes from the PDB and the parameters from the SDF.

    They have to describe the same atoms in the same order, or the System ends
    up with every parameter on the wrong atom and nothing raises.
    """
    from openmm import app
    from rdkit import Chem

    result = build_chain(
        ChainSpec(monomer_smiles=PS, degree_of_polymerization=4, residue_name="PS"),
        "chain",
        output_dir=tmp_path,
    )
    pdb = app.PDBFile(result.pdb_paths[0])
    sdf = Chem.MolFromMolFile(result.sdf_paths[0], removeHs=False)

    assert pdb.topology.getNumResidues() == 1
    assert next(pdb.topology.residues()).name == "PS"
    assert pdb.topology.getNumAtoms() == sdf.GetNumAtoms() == result.n_atoms
    assert sum(1 for _ in pdb.topology.bonds()) == sdf.GetNumBonds()
    sdf_elements = [atom.GetSymbol() for atom in sdf.GetAtoms()]  # type: ignore[no-untyped-call]
    assert [atom.element.symbol for atom in pdb.topology.atoms()] == sdf_elements


def test_tacticity_sets_alternating_stereocentres(tmp_path: Path) -> None:
    """Syndiotactic polystyrene alternates; isotactic does not."""
    from rdkit import Chem

    def tags(tacticity: str) -> list[str]:
        molecule = assemble_chain(
            ChainSpec(
                monomer_smiles=PS,
                degree_of_polymerization=6,
                residue_name="PS",
                tacticity=tacticity,
            )
        )
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        return [
            str(atom.GetChiralTag())
            for atom in molecule.GetAtoms()
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ]

    isotactic = tags("isotactic")
    syndiotactic = tags("syndiotactic")
    assert len(set(isotactic)) == 1
    assert len(set(syndiotactic)) == 2


def test_tacticity_on_a_monomer_with_no_stereocentre_is_a_no_op(
    tmp_path: Path,
) -> None:
    """Polyethylene has nothing to set, and that is not an error."""
    result = build_chain(
        ChainSpec(
            monomer_smiles=PE,
            degree_of_polymerization=6,
            residue_name="PE",
            tacticity="isotactic",
        ),
        "chain",
        output_dir=tmp_path,
    )
    assert result.n_atoms > 0


def test_extent_bound_never_understates_the_real_extent(tmp_path: Path) -> None:
    """It gates whether a conformer fits the packing cell, so it must not."""
    from rdkit import Chem

    result = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=10, residue_name="PE"),
        "chain",
        output_dir=tmp_path,
    )
    molecule = Chem.MolFromMolFile(result.sdf_paths[0], removeHs=False)
    positions = np.asarray(molecule.GetConformer().GetPositions()) / 10.0
    deltas = positions[:, None, :] - positions[None, :, :]
    real = float(np.sqrt((deltas**2).sum(axis=-1)).max())
    assert result.max_extent_nm[0] >= real - 1e-9


def test_backbone_is_reported_for_the_manifest(tmp_path: Path) -> None:
    """Nothing downstream can recover it: the caps consumed the attachment points."""
    result = build_chain(
        ChainSpec(monomer_smiles=PE, degree_of_polymerization=6, residue_name="PE"),
        "chain",
        output_dir=tmp_path,
    )
    assert len(result.backbone) == 12
    assert math.isclose(len(set(result.backbone)), len(result.backbone))
