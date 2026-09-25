"""Tests for building a melt in one call, and for rebuilding it without harm.

The staging - scratch rebuilds, the record, the refusals - is tested against a
stand-in for the chemistry, so it runs anywhere and in milliseconds. One real
build of a tiny melt checks that the layers join up and that a real rebuild
reproduces itself.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer.chain import ChainSpec
from openmmpolymer.melt import build_melt
from openmmpolymer.protocols import Protocol, ProtocolError, Stage, run_protocol

PE = ChainSpec(monomer_smiles="[*]CC[*]", residue_name="PE")


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _build(directory: Path = Path("run"), **options: Any) -> Any:
    return build_melt(PE, 30, directory, target_density_g_cm3=0.85, **options)


def test_a_first_build_records_what_it_made(staged_melt: dict[str, Any]) -> None:
    """The record is what a rebuild is later held to."""
    chain, run = _build()
    record = json.loads(Path("run/build/inputs.json").read_text())
    assert staged_melt["builds"] == [Path("run/build")]
    assert set(record["artifacts"]) == {
        "chain_0.sdf",
        "chain_0.pdb",
        "polymer_ff.xml",
        "packed.pdb",
    }
    assert record["inputs"]["chain"]["n_atoms"] == chain.n_atoms
    assert record["inputs"]["run"]["seed"] == run.seed
    assert Path(run.forcefield.forcefield_xml) == Path("run/build/polymer_ff.xml")


@pytest.mark.parametrize(("n_conformers", "built"), [(None, 30), (5, 5), (50, 30)])
def test_one_conformer_per_chain_at_most(
    staged_melt: dict[str, Any], n_conformers: int | None, built: int
) -> None:
    """None means one per chain; more than that would never be packed."""
    _build(n_conformers=n_conformers)
    assert staged_melt["options"]["n_conformers"] == built


def test_a_rebuild_is_checked_in_scratch_and_the_originals_are_used(
    staged_melt: dict[str, Any],
) -> None:
    """New dependency versions can change the Hamiltonian, so a rerun rebuilds.

    It does so beside the originals, from a copy of the cache, and hands back
    references to the originals it was checked against - the real cache
    included, which the rebuild must not write to.
    """
    _build()
    before = _files(Path("run"))
    chain, run = _build(platform="CPU")
    assert _files(Path("run")) == before
    scratch = staged_melt["builds"][1]
    assert scratch.name == "build"
    assert scratch.parent.name.startswith(".build-check-")
    assert not scratch.exists()
    assert chain.sdf_paths == (str(Path("run/build/chain_0.sdf")),)
    assert Path(run.forcefield.forcefield_xml) == Path("run/build/polymer_ff.xml")


@pytest.mark.parametrize("failure", ["different_system", "build_error"])
def test_a_changed_or_failed_rebuild_cannot_overwrite_the_originals(
    staged_melt: dict[str, Any], failure: str
) -> None:
    _build()
    before = _files(Path("run"))
    if failure == "different_system":
        staged_melt["system_suffix"] = "\n"
        expected: type[Exception] = ProtocolError
    else:
        staged_melt["fail"] = True
        expected = RuntimeError
    with pytest.raises(expected):
        _build()
    assert _files(Path("run")) == before
    assert not list(Path("run").glob(".build-check-*"))


def test_a_changed_original_is_refused_before_rebuilding(
    staged_melt: dict[str, Any],
) -> None:
    """A run resting on an edited asset cannot be vouched for."""
    _build()
    Path("run/build/polymer_ff.xml").write_text("modified original")
    with pytest.raises(ProtocolError, match=r"'polymer_ff\.xml' changed or is missing"):
        _build()
    assert len(staged_melt["builds"]) == 1


def test_a_build_with_no_record_is_refused(staged_melt: dict[str, Any]) -> None:
    """There is nothing to check a rebuild against, so nothing is overwritten."""
    Path("run/build").mkdir(parents=True)
    Path("run/build/polymer_ff.xml").write_text("unrecorded parameters")
    with pytest.raises(ProtocolError, match="lacks verified inputs"):
        _build()
    assert Path("run/build/polymer_ff.xml").read_text() == "unrecorded parameters"


@pytest.mark.parametrize("run_dir", ["run", "run/equilibration"])
def test_a_rebuild_must_match_the_runs_already_started(
    staged_melt: dict[str, Any], argon_run: Any, run_dir: str
) -> None:
    """Including a rate scan's shared preparation, kept in ``equilibration``."""
    _build()
    run_protocol(
        Protocol("other", (Stage("00_minimise", "minimise"),)),
        replace(argon_run, seed=99),
        run_dir,
    )
    before = _files(Path("run"))
    with pytest.raises(ProtocolError, match="starting inputs changed"):
        _build()
    assert _files(Path("run")) == before


@pytest.mark.forcefield
@pytest.mark.packmol
def test_a_real_melt_builds_and_rebuilds_to_the_same_cell(tmp_path: Path) -> None:
    """Eight tetramers at a density low enough for the cutoff check to pass."""
    spec = ChainSpec(
        monomer_smiles="[*]CC[*]",
        degree_of_polymerization=4,
        residue_name="PE",
        seed=11,
    )
    settings: dict[str, Any] = {
        "target_density_g_cm3": 0.05,
        "charge_method": "gasteiger",
        "pack_density_g_cm3": 0.03,
        "platform": "CPU",
    }
    lines: list[str] = []
    chain, run = build_melt(spec, 8, tmp_path, progress=lines.append, **settings)
    assert [line.split(":")[0] for line in lines] == [
        "chain",
        "cell",
        "force field",
        "packed",
    ]
    assert chain.n_atoms == 26
    assert run.box.n_molecules == 8
    assert run.box.topology.getNumAtoms() == 8 * 26

    before = _files(tmp_path)
    again, rerun = build_melt(spec, 8, tmp_path, **settings)
    assert _files(tmp_path) == before
    assert again == chain
    assert rerun.system_xml == run.system_xml

    with pytest.raises(ProtocolError, match="changed"):
        build_melt(replace(spec, seed=12), 8, tmp_path, **settings)
    assert _files(tmp_path) == before
