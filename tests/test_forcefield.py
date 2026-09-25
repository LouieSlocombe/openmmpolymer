"""Tests for turning a chain into an OpenMM force field through forcefill."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer.chain import ChainResult, ChainSpec, build_chain
from openmmpolymer.charges import assign_charges, read_chain_molecule
from openmmpolymer.forcefield import (
    DEFAULT_BASE_FORCEFIELD,
    ForceFieldError,
    PolymerForceField,
    build_polymer_forcefield,
    cache_key,
)

from .helpers import write_two_molecule_sdf

PE4 = ChainSpec(
    monomer_smiles="[*]CC[*]", degree_of_polymerization=4, residue_name="PE"
)


def test_the_files_property_puts_the_new_xml_last() -> None:
    """OpenMM reads them in order and the polymer template has to win."""
    forcefield = PolymerForceField("poly.xml", DEFAULT_BASE_FORCEFIELD, "POL", "gaff")
    assert forcefield.files == (*DEFAULT_BASE_FORCEFIELD, "poly.xml")


@pytest.mark.parametrize("backend", ["smirnof", "charmm"])
def test_an_unknown_backend_is_refused_before_anything_expensive(backend: str) -> None:
    """A typo should not cost an AM1-BCC run to discover.

    Nor should charmm, which needs a CGenFF stream for the whole chain that
    nothing here can write.
    """
    with pytest.raises(ValueError, match="backend"):
        build_polymer_forcefield("chain.sdf", backend=backend)


@pytest.fixture(scope="module")
def charged_chain(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    """A short polyethylene chain, both conformers charged.

    Both, because the cache is keyed on the molecule and its charges rather
    than on the file: two conformers of one chain should hit the same entry,
    and an uncharged one legitimately would not.
    """
    result = build_chain(
        PE4, "chain", n_conformers=2, output_dir=tmp_path_factory.mktemp("pe4")
    )
    for path in result.sdf_paths:
        assign_charges(path, "gasteiger")
    return result


def _molecule(sdf_path: str) -> Any:
    return read_chain_molecule(Path(sdf_path))


def test_the_cache_key_ignores_coordinates(charged_chain: ChainResult) -> None:
    """Two conformers of one chain share their parameters, so share a key.

    Keying on the file would make every conformer a fresh parameterisation of
    the same molecule.
    """
    first, second = (_molecule(path) for path in charged_chain.sdf_paths)
    options = {"backend": "smirnoff"}
    assert cache_key(first, options) == cache_key(second, options)


def test_the_cache_key_changes_with_the_charges_and_the_options(
    charged_chain: ChainResult, tmp_path: Path
) -> None:
    """Charges are parameters, and a different backend is a different force field."""
    charged = _molecule(charged_chain.sdf_paths[0])
    uncharged = _molecule(build_chain(PE4, "bare", output_dir=tmp_path).sdf_paths[0])
    key = cache_key(charged, {"backend": "smirnoff"})
    assert cache_key(charged, {"backend": "gaff"}) != key
    assert cache_key(uncharged, {"backend": "smirnoff"}) != key


def test_an_sdf_holding_more_than_one_molecule_is_refused(tmp_path: Path) -> None:
    """Parameterising the first of them would be a guess about which is the chain."""
    with pytest.raises(ForceFieldError, match="holds 2 molecules"):
        build_polymer_forcefield(write_two_molecule_sdf(tmp_path / "two.sdf"))


def test_an_uncharged_chain_is_warned_about_before_forcefill_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """AM1-BCC on a chain is slow at best, and unavailable without AmberTools.

    forcefill is stopped short: the warning is the thing under test, and what
    forcefill says when it fails comes back as a ForceFieldError.
    """
    import forcefill

    def decline(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("declined")

    monkeypatch.setattr(forcefill, "build_ligand_xml", decline)
    chain = build_chain(PE4, "bare", output_dir=tmp_path)
    with (
        caplog.at_level("WARNING"),
        pytest.raises(ForceFieldError, match=r"could not parameterise PE.*declined"),
    ):
        build_polymer_forcefield(chain.sdf_paths[0], residue_name="PE")
    assert "carries no partial charges" in caplog.text


@pytest.fixture(scope="module")
def parameterised(
    charged_chain: ChainResult, tmp_path_factory: pytest.TempPathFactory
) -> tuple[PolymerForceField, Path]:
    """The first conformer through forcefill for real, once, into a cache."""
    directory = tmp_path_factory.mktemp("forcefield")
    forcefield = build_polymer_forcefield(
        charged_chain.sdf_paths[0],
        directory / "pe.xml",
        residue_name="PE",
        cache_dir=directory / "cache",
        workdir=directory / "work",
    )
    return forcefield, directory / "cache"


@pytest.mark.forcefield
def test_parameterising_a_chain_gives_a_loadable_force_field(
    parameterised: tuple[PolymerForceField, Path],
) -> None:
    """The handoff: an ordinary ffxml, loaded next to amber14."""
    from openmm import app

    forcefield, _ = parameterised
    assert forcefield.cached is False
    assert forcefield.virtual_site_residues == ()
    assert app.ForceField(*forcefield.files) is not None


@pytest.mark.forcefield
def test_the_force_field_carries_the_charges_in_the_sdf(
    charged_chain: ChainResult, parameterised: tuple[PolymerForceField, Path]
) -> None:
    """The mechanism the whole chain-length story rests on.

    openmmforcefields passes a molecule's own charges through instead of
    running AM1-BCC, so the Gasteiger charges written into the SDF are the ones
    in the template. Without that, nothing longer than a short oligomer is
    reachable.
    """
    forcefield, _ = parameterised
    template = ET.parse(forcefield.forcefield_xml).find("./Residues/Residue")
    assert template is not None
    written = [float(atom.attrib["charge"]) for atom in template.findall("Atom")]
    expected = _molecule(charged_chain.sdf_paths[0]).partial_charges.m
    assert written == pytest.approx(list(expected), abs=1e-6)


@pytest.mark.forcefield
def test_another_conformer_comes_from_the_cache(
    charged_chain: ChainResult,
    parameterised: tuple[PolymerForceField, Path],
    tmp_path: Path,
) -> None:
    """Parameterising a long chain is minutes; nothing about it is per-run."""
    forcefield, cache = parameterised
    second = build_polymer_forcefield(
        charged_chain.sdf_paths[1],
        tmp_path / "second.xml",
        residue_name="PE",
        cache_dir=cache,
    )
    assert second.cached is True
    assert Path(second.forcefield_xml).read_bytes() == (
        Path(forcefield.forcefield_xml).read_bytes()
    )


@pytest.mark.forcefield
def test_the_cache_records_what_it_was_keyed_on(
    charged_chain: ChainResult, parameterised: tuple[PolymerForceField, Path]
) -> None:
    """So a stale entry can be explained rather than guessed at."""
    _, cache = parameterised
    (entry,) = cache.glob("*.xml")
    recorded = json.loads(entry.with_suffix(".json").read_text())
    assert recorded["backend"] == "smirnoff"
    assert recorded["residue_name"] == "PE"
    assert entry.stem == cache_key(_molecule(charged_chain.sdf_paths[0]), recorded)
