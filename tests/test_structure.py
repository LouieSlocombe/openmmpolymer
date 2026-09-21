"""Tests for reading a run's structure and dynamics back."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.structure import (
    MAX_STRUCTURE_FACTOR_FRAMES,
    analyse_structure,
    infer_backbone,
    structure_stages,
    write_structure_report,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import _write_manifest, synthetic_ensemble, write_polymer_snapshot

ROD_SQUARE_NM2 = (4 * 0.153) ** 2


def _topology(n_atoms: int, bonds: list[tuple[int, int]], hydrogens: set[int]) -> Any:
    """One residue of *n_atoms*, bonded as listed, some of them hydrogens."""
    from openmm import app

    topology = app.Topology()
    residue = topology.addResidue("POL", topology.addChain())
    atoms = [
        topology.addAtom(
            f"A{index}",
            app.Element.getBySymbol("H" if index in hydrogens else "C"),
            residue,
        )
        for index in range(n_atoms)
    ]
    for first, second in bonds:
        topology.addBond(atoms[first], atoms[second])
    return topology


def _bonded_ensemble(
    n_atoms: int, bonds: list[tuple[int, int]], hydrogens: set[int] | None = None
) -> Any:
    """A one-chain snapshot carrying a real OpenMM topology for its bonds."""
    hydrogens = hydrogens or set()
    frames = np.zeros((1, n_atoms, 3), dtype=np.float64)
    frames[0, :, 0] = np.arange(n_atoms) * 0.15
    is_hydrogen = np.zeros(n_atoms, dtype=np.bool_)
    is_hydrogen[list(hydrogens)] = True
    return dataclasses.replace(
        synthetic_ensemble(frames, n_chains=1, is_hydrogen=is_hydrogen),
        topology=_topology(n_atoms, bonds, hydrogens),
    )


def _add_snapshot_stage(run_directory: Path, name: str) -> None:
    """Add a snapshot-only stage after the dimer fixture's trajectory stage."""
    manifest = json.loads((run_directory / "manifest.json").read_text())
    source = Path(str(manifest["stages"]["02_nvt"]["final_pdb"]))
    copy = run_directory / f"{name}.pdb"
    shutil.copy(source, copy)
    manifest["stages"][name] = {
        "name": name,
        "final_pdb": str(copy),
        "csv": None,
        "samples": {},
    }
    (run_directory / "manifest.json").write_text(json.dumps(manifest))


# --------------------------------------------------------------------------
# Which stage
# --------------------------------------------------------------------------


def test_a_stage_that_wrote_its_closing_structure_is_readable(tmp_path: Path) -> None:
    """Every finished stage leaves one, so every finished stage qualifies."""
    write_polymer_snapshot(tmp_path)
    assert structure_stages(tmp_path) == ("05_npt",)


def test_a_stage_with_a_trajectory_is_readable(dimer_run_directory: Path) -> None:
    """The real fixture: an xtc with its topology beside it."""
    assert structure_stages(dimer_run_directory) == ("02_nvt",)


def test_a_stage_that_left_no_files_is_not_readable(tmp_path: Path) -> None:
    """A manifest entry with nothing on disk behind it is not coordinates."""
    _write_manifest(tmp_path, {"05_npt": {"name": "05_npt", "samples": {}}})
    with pytest.raises(AnalysisError, match="left coordinates"):
        structure_stages(tmp_path)


def test_a_directory_with_no_manifest_is_refused_in_the_usual_words(
    tmp_path: Path,
) -> None:
    """The refusal is the one stage_files already gives, not a new one."""
    with pytest.raises(AnalysisError, match="No manifest"):
        structure_stages(tmp_path)


def test_the_last_trajectory_beats_a_later_snapshot(
    dimer_run_directory: Path,
) -> None:
    """A trajectory can answer the dynamic questions and a snapshot cannot."""
    _add_snapshot_stage(dimer_run_directory, "03_alone")
    report = analyse_structure(dimer_run_directory, backbone=(0, 1))
    assert report.stage == "02_nvt"
    assert report.stage_source == "last_trajectory"
    assert not report.is_snapshot
    assert report.n_frames == 10


def test_a_requested_stage_is_measured_even_when_it_is_a_snapshot(
    dimer_run_directory: Path,
) -> None:
    """Asking for a stage is asking for that stage."""
    _add_snapshot_stage(dimer_run_directory, "03_alone")
    report = analyse_structure(dimer_run_directory, stage="03_alone", backbone=(0, 1))
    assert report.stage == "03_alone"
    assert report.stage_source == "requested"
    assert report.is_snapshot
    assert report.displacement is None
    assert report.relaxation is None
    assert any("single snapshot" in note for note in report.notes)


def test_a_snapshot_only_directory_says_so_once(tmp_path: Path) -> None:
    """One note for the whole dynamic half, not one per measurement skipped."""
    write_polymer_snapshot(tmp_path)
    report = analyse_structure(tmp_path, backbone=(0, 1, 2, 3, 4))
    assert report.stage_source == "last_snapshot"
    assert report.is_snapshot
    assert report.displacement is None
    assert report.relaxation is None
    snapshot_notes = [note for note in report.notes if "wrote a trajectory" in note]
    assert len(snapshot_notes) == 1
    assert len(report.notes) == 1


def test_an_unknown_stage_is_refused_naming_the_readable_ones(tmp_path: Path) -> None:
    """So the caller can see what to ask for instead."""
    write_polymer_snapshot(tmp_path)
    with pytest.raises(AnalysisError, match="05_npt"):
        analyse_structure(tmp_path, stage="99_nothing")


# --------------------------------------------------------------------------
# Where the backbone comes from
# --------------------------------------------------------------------------


def test_a_backbone_given_is_the_backbone_used(tmp_path: Path) -> None:
    write_polymer_snapshot(tmp_path)
    report = analyse_structure(tmp_path, backbone=(0, 4))
    assert report.backbone == (0, 4)
    assert report.backbone_source == "argument"
    assert report.backbone_file is None


def test_a_workflow_record_supplies_the_backbone(tmp_path: Path) -> None:
    """The scans write chain_backbone beside the manifest; it is read back."""
    write_polymer_snapshot(tmp_path)
    (tmp_path / "tg_workflow.json").write_text(
        json.dumps({"chain_backbone": [4, 3, 2, 1, 0]})
    )
    report = analyse_structure(tmp_path)
    assert report.backbone == (4, 3, 2, 1, 0)
    assert report.backbone_source == "workflow"
    assert report.backbone_file == "tg_workflow.json"


def test_the_manifest_supplies_the_backbone_when_it_records_one(
    tmp_path: Path,
) -> None:
    """And the dimensions recorded beside it come back for cross-reference."""
    write_polymer_snapshot(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["chains"] = {
        "mean_squared_end_to_end_nm2": 0.37,
        "mean_radius_of_gyration_nm": 0.2,
        "ratio_of_squares": 6.0,
        "characteristic_ratio": 6.5,
        "expected_characteristic_ratio": 7.0,
        "consistent": True,
        "backbone": [0, 1, 2, 3],
    }
    path.write_text(json.dumps(manifest))
    report = analyse_structure(tmp_path)
    assert report.backbone == (0, 1, 2, 3)
    assert report.backbone_source == "manifest"
    assert report.backbone_file == "manifest.json"
    assert report.recorded_chains is not None
    assert report.recorded_chains.characteristic_ratio == 6.5


def test_with_nothing_recorded_the_backbone_is_inferred_from_the_bonds(
    tmp_path: Path,
) -> None:
    """The side carbon and the hydrogen are left out, and the note says how."""
    write_polymer_snapshot(tmp_path)
    report = analyse_structure(tmp_path)
    assert report.backbone == (0, 1, 2, 3, 4)
    assert report.backbone_source == "inferred"
    assert any("side group" in note for note in report.notes)
    assert report.conformation is not None
    assert report.conformation.mean.mean_squared_end_to_end_nm2 == pytest.approx(
        ROD_SQUARE_NM2
    )


def test_inference_can_be_turned_off(tmp_path: Path) -> None:
    """Then an unknown backbone is unknown, and the chain half is a note."""
    write_polymer_snapshot(tmp_path)
    report = analyse_structure(tmp_path, infer_backbone=False)
    assert report.backbone is None
    assert report.backbone_source is None
    assert report.conformation is None
    assert report.persistence is None
    assert any("No backbone is recorded" in note for note in report.notes)


def test_a_backbone_given_wrongly_is_not_guessed_past(tmp_path: Path) -> None:
    """An explicit answer that does not fit is a question, not a fallback."""
    write_polymer_snapshot(tmp_path)
    report = analyse_structure(tmp_path, backbone=(0, 99))
    assert report.backbone is None
    assert report.backbone_source is None
    assert any("was not used" in note for note in report.notes)
    assert not any("inferred" in note for note in report.notes)


def test_a_bad_workflow_record_is_noted_and_inference_carries_on(
    tmp_path: Path,
) -> None:
    write_polymer_snapshot(tmp_path)
    (tmp_path / "mechanical_workflow.json").write_text(
        json.dumps({"chain_backbone": [0, 99]})
    )
    report = analyse_structure(tmp_path)
    assert report.backbone_source == "inferred"
    assert any("mechanical_workflow.json was not used" in note for note in report.notes)


def test_a_workflow_record_that_is_not_a_list_is_noted(tmp_path: Path) -> None:
    write_polymer_snapshot(tmp_path)
    (tmp_path / "tg_workflow.json").write_text(json.dumps({"chain_backbone": "abc"}))
    report = analyse_structure(tmp_path)
    assert report.backbone_source == "inferred"
    assert any("not a list of atom indices" in note for note in report.notes)


def test_a_workflow_file_that_cannot_be_read_is_skipped(tmp_path: Path) -> None:
    """A half-written record is not a reason to stop reading the run."""
    write_polymer_snapshot(tmp_path)
    (tmp_path / "tg_workflow.json").write_text("{not json")
    (tmp_path / "viscoelastic_workflow.json").write_text(json.dumps({"other": 1}))
    report = analyse_structure(tmp_path)
    assert report.backbone_source == "inferred"


def test_a_structure_without_bond_records_leaves_the_backbone_unknown(
    tmp_path: Path,
) -> None:
    """A packmol PDB carries no CONECT lines, and then there is no graph."""
    write_polymer_snapshot(tmp_path)
    pdb = tmp_path / "05_npt.pdb"
    pdb.write_text(
        "".join(
            line
            for line in pdb.read_text().splitlines(keepends=True)
            if not line.startswith("CONECT")
        )
    )
    report = analyse_structure(tmp_path)
    assert report.backbone is None
    assert any("No backbone could be inferred" in note for note in report.notes)
    assert any("records no bonds" in note for note in report.notes)
    assert report.distribution is not None


def test_the_inferred_path_is_the_diameter_of_the_heavy_atom_tree() -> None:
    """A side branch shorter than the main chain is not the backbone."""
    ensemble = _bonded_ensemble(7, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6)])
    assert infer_backbone(ensemble) == (0, 1, 2, 3, 4, 5)


def test_hydrogens_are_not_part_of_the_inferred_backbone() -> None:
    ensemble = _bonded_ensemble(4, [(0, 1), (1, 2), (2, 3)], hydrogens={3})
    assert infer_backbone(ensemble) == (0, 1, 2)


def test_a_disconnected_chain_cannot_have_a_backbone_inferred() -> None:
    ensemble = _bonded_ensemble(4, [(0, 1), (2, 3)])
    with pytest.raises(AnalysisError, match="not one connected molecule"):
        infer_backbone(ensemble)


def test_a_topology_without_bonds_cannot_have_a_backbone_inferred() -> None:
    ensemble = _bonded_ensemble(4, [])
    with pytest.raises(AnalysisError, match="records no bonds"):
        infer_backbone(ensemble)


def test_an_ensemble_without_a_topology_cannot_have_a_backbone_inferred() -> None:
    frames = np.zeros((1, 4, 3), dtype=np.float64)
    with pytest.raises(AnalysisError, match="no topology"):
        infer_backbone(synthetic_ensemble(frames, n_chains=1))


# --------------------------------------------------------------------------
# Each measurement stands on its own
# --------------------------------------------------------------------------


def test_one_molecule_has_no_pair_distribution_but_still_a_structure_factor(
    tmp_path: Path,
) -> None:
    write_polymer_snapshot(tmp_path, n_chains=1)
    report = analyse_structure(tmp_path, backbone=(0, 1, 2, 3, 4))
    assert report.distribution is None
    assert any("No pair distribution" in note for note in report.notes)
    assert report.structure is not None


def test_a_structure_factor_the_cell_cannot_afford_is_a_note(tmp_path: Path) -> None:
    write_polymer_snapshot(tmp_path)
    report = analyse_structure(tmp_path, backbone=(0, 1, 2, 3, 4), q_max_per_nm=5000.0)
    assert report.structure is None
    assert any("No structure factor" in note for note in report.notes)
    assert report.distribution is not None


def test_a_trajectory_gets_the_dynamic_half(dimer_run_directory: Path) -> None:
    """Dimers have one bond, too few for a persistence length, and that is a
    note beside four measurements that went ahead."""
    report = analyse_structure(dimer_run_directory, backbone=(0, 1))
    assert report.persistence is None
    assert any("No persistence length" in note for note in report.notes)
    assert report.conformation is not None
    assert report.displacement is not None
    assert report.relaxation is not None
    assert report.distribution is not None
    assert report.distribution.n_frames == 10
    assert report.structure is not None
    assert report.structure.n_frames <= MAX_STRUCTURE_FACTOR_FRAMES


def test_a_stride_that_is_not_a_count_is_refused(tmp_path: Path) -> None:
    write_polymer_snapshot(tmp_path)
    with pytest.raises(ValueError, match="stride"):
        analyse_structure(tmp_path, stride=0)


# --------------------------------------------------------------------------
# Reading writes nothing; reporting writes the record
# --------------------------------------------------------------------------


def test_analysing_a_snapshot_writes_nothing(tmp_path: Path) -> None:
    write_polymer_snapshot(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    analyse_structure(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_the_report_writes_a_record_and_its_figures(tmp_path: Path) -> None:
    """One JSON, pinned field by field against the planted rod."""
    write_polymer_snapshot(tmp_path)
    report = analyse_structure(tmp_path, backbone=(0, 1, 2, 3, 4))
    files = write_structure_report(report, tmp_path / "analysis")

    record = json.loads(Path(files.json).read_text())
    assert record["openmmpolymer"]
    assert record["stage"] == "05_npt"
    assert record["stage_source"] == "last_snapshot"
    assert record["is_snapshot"] is True
    assert record["n_frames"] == 1
    assert record["n_chains"] == 4
    assert record["backbone"] == [0, 1, 2, 3, 4]
    assert record["backbone_source"] == "argument"
    assert record["backbone_file"] is None
    assert record["conformation"]["mean"]["mean_squared_end_to_end_nm2"] == (
        pytest.approx(ROD_SQUARE_NM2)
    )
    assert record["conformation"]["settled"] is None
    assert record["persistence"]["decayed"] is False
    assert math.isinf(record["persistence"]["persistence_length_nm"])
    assert record["radial_distribution"]["n_frames"] == 1
    assert record["structure_factor"]["n_frames"] == 1
    assert record["displacement"] is None
    assert record["relaxation"] is None
    assert record["recorded_chains"] is None
    assert isinstance(record["notes"], list)

    assert [Path(path).name for path in files.figures] == [
        "correlations.png",
        "conformation.png",
        "persistence.png",
    ]
    assert all(Path(path).is_file() for path in files.figures)


def test_the_report_can_leave_the_figures_out(tmp_path: Path) -> None:
    write_polymer_snapshot(tmp_path)
    report = analyse_structure(tmp_path)
    files = write_structure_report(report, figures=False)
    assert files.figures == ()
    assert Path(files.json) == tmp_path / "analysis" / "structure.json"


def test_the_report_can_be_written_outside_the_run_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_polymer_snapshot(run)
    report = analyse_structure(run)
    files = write_structure_report(report, tmp_path / "elsewhere", figures=False)
    assert Path(files.json).parent == tmp_path / "elsewhere"
    assert not (run / "analysis").exists()


def test_a_trajectory_report_draws_the_dynamics(dimer_run_directory: Path) -> None:
    """End to end on the real fixture: the record and three figures."""
    report = analyse_structure(dimer_run_directory, backbone=(0, 1))
    files = write_structure_report(report, dimer_run_directory / "analysis")
    assert Path(files.json).is_file()
    assert [Path(path).name for path in files.figures] == [
        "correlations.png",
        "conformation.png",
        "dynamics.png",
    ]
    record = json.loads(Path(files.json).read_text())
    assert record["displacement"]["n_origins"] == 10
    assert record["relaxation"]["decorrelated"] in (True, False)
