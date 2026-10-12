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

from openmmpolymer import melt
from openmmpolymer.chain import ChainSpec
from openmmpolymer.melt import build_melt
from openmmpolymer.protocols import Protocol, ProtocolError, Stage, run_protocol

from .helpers import snapshot_files

PE = ChainSpec(monomer_smiles="[*]CC[*]", residue_name="PE")


def _build(directory: Path = Path("run"), **options: Any) -> Any:
    return build_melt(PE, 30, directory, target_density_g_cm3=0.85, **options)


def test_a_first_build_records_what_it_made(staged_melt: dict[str, Any]) -> None:
    """The record is what a rebuild is later held to."""
    chain, run = _build()
    record = json.loads(Path("run/build/inputs.json").read_text())
    (working,) = staged_melt["builds"]
    assert working.name == "build"
    assert working.parent.name.startswith(".build-check-")
    assert not working.parent.exists()
    assert set(record["artifacts"]) == {
        "chain_0.sdf",
        "chain_0.pdb",
        "polymer_ff.xml",
        "packed.pdb",
    }
    assert record["inputs"]["chain"]["n_atoms"] == chain.n_atoms
    assert record["inputs"]["run"]["seed"] == run.seed
    assert ".build-check-" not in json.dumps(record)
    assert chain.sdf_paths == (str(Path("run/build/chain_0.sdf")),)
    assert chain.pdb_paths == (str(Path("run/build/chain_0.pdb")),)
    assert Path(run.forcefield.forcefield_xml) == Path("run/build/polymer_ff.xml")
    assert all(
        Path(path).is_file()
        for path in (*chain.sdf_paths, *chain.pdb_paths, run.forcefield.forcefield_xml)
    )


@pytest.mark.parametrize("failure", ["preparation", "record", "publication"])
def test_a_failed_first_build_can_be_retried_without_manual_cleanup(
    staged_melt: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Assets and their verification record must become visible together."""
    with monkeypatch.context() as patch:
        if failure == "preparation":
            staged_melt["fail"] = True
            expected: type[Exception] = RuntimeError
        elif failure == "record":

            def fail_record(path: Path, record: Any) -> None:
                path.with_suffix(".json.tmp").write_text("partial record")
                raise OSError("record failed")

            patch.setattr(melt, "write_json", fail_record)
            expected = OSError
        else:

            def fail_publish(source: Path, target: Path) -> None:
                assert (source / "inputs.json").is_file()
                raise OSError("publication failed")

            patch.setattr(Path, "rename", fail_publish)
            expected = OSError
        with pytest.raises(expected, match="failed"):
            _build()

    assert not Path("run/build").exists()
    assert not list(Path("run").glob(".build-check-*"))
    # Parameterisation can still be reused after a later build layer fails.
    assert Path("run/cache/entry_1.xml").is_file()
    staged_melt["fail"] = False
    chain, run = _build()
    assert Path("run/build/inputs.json").is_file()
    assert Path(chain.sdf_paths[0]).is_file()
    assert Path(run.forcefield.forcefield_xml).is_file()
    assert not list(Path("run").glob(".build-check-*"))


@pytest.mark.parametrize(("n_conformers", "built"), [(None, 30), (5, 5), (50, 30)])
def test_one_conformer_per_chain_at_most(
    staged_melt: dict[str, Any], n_conformers: int | None, built: int
) -> None:
    """None means one per chain; more than that would never be packed."""
    _build(n_conformers=n_conformers)
    assert staged_melt["options"]["n_conformers"] == built


@pytest.mark.parametrize("cache", [Path("run/build/cache"), Path("run/build")])
def test_a_cache_inside_the_build_is_staged_with_its_assets(
    staged_melt: dict[str, Any], cache: Path
) -> None:
    staged_melt["fail"] = True
    with pytest.raises(RuntimeError, match="preparation failed"):
        _build(cache_dir=cache)
    assert not Path("run/build").exists()

    staged_melt["fail"] = False
    _build(cache_dir=cache)
    assert (cache / "entry_2.xml").is_file()
    before = snapshot_files(Path("run"))
    _build(cache_dir=cache)
    assert snapshot_files(Path("run")) == before


def test_a_rebuild_is_checked_in_scratch_and_the_originals_are_used(
    staged_melt: dict[str, Any],
) -> None:
    """New dependency versions can change the Hamiltonian, so a rerun rebuilds.

    It does so beside the originals, from a copy of the cache, and hands back
    references to the originals it was checked against - the real cache
    included, which the rebuild must not write to.
    """
    _build()
    before = snapshot_files(Path("run"))
    chain, run = _build(platform="CPU")
    assert snapshot_files(Path("run")) == before
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
    before = snapshot_files(Path("run"))
    if failure == "different_system":
        staged_melt["system_suffix"] = "\n"
        expected: type[Exception] = ProtocolError
    else:
        staged_melt["fail"] = True
        expected = RuntimeError
    with pytest.raises(expected):
        _build()
    assert snapshot_files(Path("run")) == before
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


def test_a_build_with_no_record_is_refused_before_rebuilding(
    staged_melt: dict[str, Any],
) -> None:
    """There is nothing to check a rebuild against, so nothing is overwritten."""
    Path("run/build").mkdir(parents=True)
    Path("run/build/polymer_ff.xml").write_text("unrecorded parameters")
    before = snapshot_files(Path("run"))
    with pytest.raises(ProtocolError, match="lacks verified inputs"):
        _build()
    assert Path("run/build/polymer_ff.xml").read_text() == "unrecorded parameters"
    assert snapshot_files(Path("run")) == before
    assert not staged_melt["builds"]


@pytest.mark.parametrize("run_dir", ["run", "run/equilibration"])
def test_a_first_build_is_not_published_when_existing_run_inputs_conflict(
    staged_melt: dict[str, Any], argon_run: Any, run_dir: str
) -> None:
    run_protocol(
        Protocol("other", (Stage("00_minimise", "minimise"),)),
        replace(argon_run, seed=99),
        run_dir,
    )
    before = snapshot_files(Path(run_dir))
    with pytest.raises(ProtocolError, match="starting inputs changed"):
        _build()
    assert not Path("run/build").exists()
    assert not list(Path("run").glob(".build-check-*"))
    after = snapshot_files(Path(run_dir))
    assert all(after[path] == contents for path, contents in before.items())


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
    before = snapshot_files(Path("run"))
    with pytest.raises(ProtocolError, match="starting inputs changed"):
        _build()
    assert snapshot_files(Path("run")) == before


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

    before = snapshot_files(tmp_path)
    again, rerun = build_melt(spec, 8, tmp_path, **settings)
    assert snapshot_files(tmp_path) == before
    assert again == chain
    assert rerun.system_xml == run.system_xml

    with pytest.raises(ProtocolError, match="changed"):
        build_melt(replace(spec, seed=12), 8, tmp_path, **settings)
    assert snapshot_files(tmp_path) == before
