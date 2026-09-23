"""Tests for turning a chain into an OpenMM force field through forcefill."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmmpolymer.chain import ChainSpec, build_chain
from openmmpolymer.charges import assign_charges
from openmmpolymer.forcefield import (
    BACKENDS,
    DEFAULT_BASE_FORCEFIELD,
    ForceFieldError,
    PolymerForceField,
    build_polymer_forcefield,
    cache_key,
    check_forcefield,
)

from .helpers import build_dimer_pdb

# Collection imports this module before any per-test hook runs, so a
# capability the module's own imports need has to be checked here.
pytest.importorskip("openff.toolkit")


def test_the_files_property_puts_the_new_xml_last() -> None:
    """OpenMM reads them in order and the polymer template has to win."""
    forcefield = PolymerForceField("poly.xml", DEFAULT_BASE_FORCEFIELD, "POL", "gaff")
    assert forcefield.files == (*DEFAULT_BASE_FORCEFIELD, "poly.xml")


def test_an_unknown_backend_is_refused_before_anything_expensive() -> None:
    """A typo should not cost an AM1-BCC run to discover."""
    with pytest.raises(ValueError, match="backend"):
        build_polymer_forcefield("chain.sdf", backend="smirnof")


def test_espaloma_is_not_offered() -> None:
    """It pulls in PyTorch, and NAGL charges cover what it would be used for."""
    assert "espaloma" not in BACKENDS
    assert set(BACKENDS) == {"smirnoff", "gaff", "charmm"}


@pytest.fixture
def charged_chain(tmp_path: Path) -> Any:
    """A short polyethylene chain, charged, ready to parameterise."""
    result = build_chain(
        ChainSpec(
            monomer_smiles="[*]CC[*]", degree_of_polymerization=4, residue_name="PE"
        ),
        "chain",
        n_conformers=2,
        output_dir=tmp_path,
    )
    # Both conformers, because the cache is keyed on the molecule and its
    # charges rather than on the file: two conformers of one chain should hit
    # the same entry, and an uncharged one legitimately would not.
    for path in result.sdf_paths:
        assign_charges(path, "gasteiger")
    return result


def test_the_cache_key_ignores_coordinates(charged_chain: Any) -> None:
    """Two conformers of one chain share their parameters, so share a key.

    Keying on the file would make every conformer a fresh parameterisation of
    the same molecule.
    """
    options = {"backend": "smirnoff"}
    assert cache_key(charged_chain.sdf_paths[0], options) == cache_key(
        charged_chain.sdf_paths[1], options
    )


def test_the_cache_key_changes_with_the_options(charged_chain: Any) -> None:
    """A different backend is a different force field."""
    assert cache_key(charged_chain.sdf_paths[0], {"backend": "smirnoff"}) != cache_key(
        charged_chain.sdf_paths[0], {"backend": "gaff"}
    )


def test_the_cache_key_changes_with_the_charges(
    charged_chain: Any, tmp_path: Path
) -> None:
    """Charges are parameters, so a different charge set is a different entry."""
    options = {"backend": "smirnoff"}
    before = cache_key(charged_chain.sdf_paths[0], options)
    uncharged = build_chain(
        ChainSpec(
            monomer_smiles="[*]CC[*]", degree_of_polymerization=4, residue_name="PE"
        ),
        "bare",
        output_dir=tmp_path,
    )
    assert cache_key(uncharged.sdf_paths[0], options) != before


def test_the_cache_key_is_stable_across_calls(charged_chain: Any) -> None:
    """It has to be, or the cache never hits."""
    options = {"backend": "smirnoff", "residue_name": "PE"}
    assert cache_key(charged_chain.sdf_paths[0], options) == cache_key(
        charged_chain.sdf_paths[0], options
    )


@pytest.fixture
def recorded_charmm_builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Record stream loading without running an unrelated parameterisation."""
    builds: list[tuple[str, ...]] = []

    def build(ligands: dict[str, Any], output: str, **kwargs: Any) -> Any:
        streams = next(iter(ligands.values())).charmm_files
        contents = tuple(Path(path).read_text() for path in streams)
        builds.append(contents)
        Path(output).write_text("\n".join(contents))
        return SimpleNamespace(forcefield_xml=output)

    monkeypatch.setitem(
        sys.modules,
        "forcefill",
        SimpleNamespace(
            __version__="test",
            LigandSpec=lambda **kwargs: SimpleNamespace(**kwargs),
            build_ligand_xml=build,
            residue_templates_with_virtual_sites=lambda path: (),
        ),
    )
    monkeypatch.setattr(
        "openmmpolymer.forcefield._molecule_fingerprint", lambda path: ("CC", (0.0,))
    )
    return builds


def test_charmm_cache_tracks_stream_contents_paths_and_order(
    tmp_path: Path, recorded_charmm_builds: list[tuple[str, ...]]
) -> None:
    """An in-place edit must not silently retain the previous parameters."""
    first, second = tmp_path / "first.str", tmp_path / "second.str"
    first.write_text("parameters A")
    second.write_text("parameters B")
    options: dict[str, Any] = {
        "backend": "charmm",
        "cache_dir": tmp_path / "cache",
        "charmm_files": [first, second],
    }

    def build() -> PolymerForceField:
        return build_polymer_forcefield(
            "chain.sdf", tmp_path / "polymer.xml", **options
        )

    assert not build().cached
    assert build().cached
    first.write_text("parameters A edited")
    edited = build()
    assert not edited.cached
    assert (
        Path(edited.forcefield_xml).read_text() == "parameters A edited\nparameters B"
    )
    assert build().cached
    options["charmm_files"] = [second, first]
    assert not build().cached
    options["charmm_files"] = [second]
    assert not build().cached
    options["charmm_files"] = [first]
    assert not build().cached
    assert len(recorded_charmm_builds) == 5
    records = [
        json.loads(path.read_text()) for path in (tmp_path / "cache").glob("*.json")
    ]
    assert any(
        record["charmm_files"]
        == [
            {
                "path": str(first),
                "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            }
        ]
        for record in records
    )


def test_charmm_cache_does_not_hide_a_missing_parameter_stream(
    tmp_path: Path, recorded_charmm_builds: list[tuple[str, ...]]
) -> None:
    stream = tmp_path / "parameters.str"
    stream.write_text("parameters")
    options: dict[str, Any] = {
        "backend": "charmm",
        "cache_dir": tmp_path / "cache",
        "charmm_files": [stream],
    }
    build_polymer_forcefield("chain.sdf", tmp_path / "polymer.xml", **options)
    stream.unlink()
    with pytest.raises(ForceFieldError, match="Cannot read CHARMM parameter stream"):
        build_polymer_forcefield("chain.sdf", tmp_path / "polymer.xml", **options)
    assert len(recorded_charmm_builds) == 1


@pytest.mark.forcefield
def test_parameterising_a_chain_gives_a_loadable_force_field(
    charged_chain: Any, tmp_path: Path
) -> None:
    """The handoff: an ordinary ffxml, loaded next to amber14."""
    pytest.importorskip("forcefill")
    from openmm import app

    forcefield = build_polymer_forcefield(
        charged_chain.sdf_paths[0],
        tmp_path / "pe.xml",
        residue_name="PE",
        workdir=tmp_path / "wd",
    )
    assert Path(forcefield.forcefield_xml).is_file()
    assert forcefield.cached is False
    loaded = app.ForceField(*forcefield.files)
    assert loaded is not None


@pytest.mark.forcefield
def test_a_second_call_comes_from_the_cache(charged_chain: Any, tmp_path: Path) -> None:
    """Parameterising a long chain is minutes; nothing about it is per-run."""
    pytest.importorskip("forcefill")
    first = build_polymer_forcefield(
        charged_chain.sdf_paths[0],
        tmp_path / "a.xml",
        residue_name="PE",
        cache_dir=tmp_path / "cache",
        workdir=tmp_path / "wd",
    )
    second = build_polymer_forcefield(
        charged_chain.sdf_paths[1],
        tmp_path / "b.xml",
        residue_name="PE",
        cache_dir=tmp_path / "cache",
        workdir=tmp_path / "wd",
    )
    assert first.cached is False
    assert second.cached is True
    assert Path(second.forcefield_xml).is_file()


@pytest.mark.forcefield
def test_the_cache_records_what_it_was_keyed_on(
    charged_chain: Any, tmp_path: Path
) -> None:
    """So a stale entry can be explained rather than guessed at."""
    pytest.importorskip("forcefill")
    build_polymer_forcefield(
        charged_chain.sdf_paths[0],
        tmp_path / "a.xml",
        residue_name="PE",
        cache_dir=tmp_path / "cache",
        workdir=tmp_path / "wd",
    )
    sidecar = next((tmp_path / "cache").glob("*.json"))
    recorded = json.loads(sidecar.read_text())
    assert recorded["backend"] == "smirnoff"
    assert recorded["residue_name"] == "PE"


@pytest.mark.forcefield
def test_an_uncharged_chain_is_warned_about(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AM1-BCC on a chain is slow at best and unavailable at worst."""
    pytest.importorskip("forcefill")
    chain = build_chain(
        ChainSpec(
            monomer_smiles="[*]CC[*]", degree_of_polymerization=3, residue_name="PE"
        ),
        "chain",
        output_dir=tmp_path,
    )
    with caplog.at_level("WARNING"), contextlib.suppress(ForceFieldError):
        build_polymer_forcefield(
            chain.sdf_paths[0],
            tmp_path / "pe.xml",
            residue_name="PE",
            workdir=tmp_path / "wd",
        )
    assert "carries no partial charges" in caplog.text


@pytest.mark.forcefield
def test_the_force_field_is_checked_against_a_single_chain(
    charged_chain: Any, tmp_path: Path
) -> None:
    """Milliseconds, and it catches what only shows up hours into a run."""
    pytest.importorskip("forcefill")
    forcefield = build_polymer_forcefield(
        charged_chain.sdf_paths[0],
        tmp_path / "pe.xml",
        residue_name="PE",
        workdir=tmp_path / "wd",
    )
    report = check_forcefield(forcefield, charged_chain.pdb_paths[0])
    assert report is not None
    assert report.n_atoms == charged_chain.n_atoms


@pytest.mark.forcefield
def test_a_force_field_that_does_not_fit_the_structure_is_reported(
    charged_chain: Any, tmp_path: Path
) -> None:
    """A template for one residue will not build a System for another."""
    pytest.importorskip("forcefill")
    forcefield = build_polymer_forcefield(
        charged_chain.sdf_paths[0],
        tmp_path / "pe.xml",
        residue_name="PE",
        workdir=tmp_path / "wd",
    )
    other = build_dimer_pdb(tmp_path / "dimer.pdb")
    with pytest.raises(ForceFieldError, match="does not build a working System"):
        check_forcefield(forcefield, other)
