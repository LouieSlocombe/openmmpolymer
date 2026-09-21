"""Tests for turning a finished run into something measurable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.trajectory import (
    AnalysisError,
    backbone_indices,
    boxes_nm,
    chain_positions,
    open_run,
    require_trajectory,
    stage_files,
)

from .helpers import synthetic_ensemble


def write_manifest(directory: Path, stages: dict[str, Any], **extra: Any) -> Path:
    """Write a manifest holding *stages*, as ``run_protocol`` would."""
    payload = {
        "protocol": "test",
        "seed": 1,
        "versions": {},
        "system": {},
        "stages": stages,
        "chains": None,
        **extra,
    }
    (directory / "manifest.json").write_text(json.dumps(payload))
    return directory


def test_a_directory_without_a_manifest_is_not_a_run_directory(tmp_path: Path) -> None:
    """The manifest is what says a run happened, so its absence is the error."""
    with pytest.raises(AnalysisError, match="No manifest"):
        stage_files(tmp_path)


def test_a_manifest_with_no_stages_says_the_run_got_nowhere(tmp_path: Path) -> None:
    """A run killed during its first stage leaves a manifest and nothing else."""
    write_manifest(tmp_path, {})
    with pytest.raises(AnalysisError, match="no completed stages"):
        stage_files(tmp_path)


def test_asking_for_a_stage_that_is_not_there_lists_the_ones_that_are(
    tmp_path: Path,
) -> None:
    """Naming the alternatives saves a guess at the numbering."""
    write_manifest(tmp_path, {"02_nvt": {"final_pdb": str(tmp_path / "02_nvt.pdb")}})
    with pytest.raises(AnalysisError, match="It records: 02_nvt"):
        stage_files(tmp_path, "04_npt")


def test_the_default_stage_is_the_last_one_the_manifest_recorded(
    tmp_path: Path,
) -> None:
    """The end of a run is what you almost always want to look at."""
    write_manifest(
        tmp_path,
        {
            "02_nvt": {"final_pdb": str(tmp_path / "02_nvt.pdb")},
            "05_npt": {"final_pdb": str(tmp_path / "05_npt.pdb")},
        },
    )
    assert stage_files(tmp_path).stage == "05_npt"


def test_the_file_stem_comes_from_the_manifest_not_from_the_stage_name(
    tmp_path: Path,
) -> None:
    """run_pushoff reports three sub-stages as one, under the last one's files.

    Building the stem by appending a suffix to the stage name would look for
    ``01_pushoff.csv``, which never exists.
    """
    write_manifest(
        tmp_path,
        {
            "01_pushoff": {
                "final_pdb": str(tmp_path / "01_pushoff_2.pdb"),
                "csv": str(tmp_path / "01_pushoff_2.csv"),
            }
        },
    )
    assert stage_files(tmp_path, "01_pushoff").prefix.endswith("01_pushoff_2")


def test_the_stem_survives_a_stage_that_recorded_only_a_state(
    tmp_path: Path,
) -> None:
    """``<stem>.state.xml`` carries two suffixes, so one strip is not enough."""
    write_manifest(
        tmp_path,
        {"02_nvt": {"final_state": str(tmp_path / "02_nvt.state.xml")}},
    )
    assert stage_files(tmp_path, "02_nvt").prefix.endswith("02_nvt")


def test_a_stage_that_recorded_no_paths_falls_back_to_its_name(
    tmp_path: Path,
) -> None:
    """Better to look in the obvious place than to refuse outright."""
    write_manifest(tmp_path, {"02_nvt": {}})
    assert stage_files(tmp_path, "02_nvt").prefix.endswith("02_nvt")


@pytest.mark.parametrize("extension", ["xtc", "dcd"])
def test_both_readable_trajectory_formats_are_found(
    tmp_path: Path, extension: str
) -> None:
    """A run may have been written in either, and analysis should not care."""
    (tmp_path / f"02_nvt.{extension}").write_bytes(b"")
    (tmp_path / "02_nvt_topology.pdb").write_text("")
    write_manifest(tmp_path, {"02_nvt": {"final_pdb": str(tmp_path / "02_nvt.pdb")}})
    found = stage_files(tmp_path, "02_nvt")
    assert found.trajectory is not None
    assert found.trajectory.endswith(extension)


def test_a_pdb_is_never_taken_for_a_trajectory(tmp_path: Path) -> None:
    """A stage's final snapshot goes to <stem>.pdb, and so would a pdb-format
    trajectory, so that path is not reliably either one."""
    (tmp_path / "02_nvt.pdb").write_text("")
    write_manifest(tmp_path, {"02_nvt": {"final_pdb": str(tmp_path / "02_nvt.pdb")}})
    found = stage_files(tmp_path, "02_nvt")
    assert found.trajectory is None
    assert found.topology is not None


def test_a_trajectory_with_no_topology_beside_it_is_refused(
    tmp_path: Path,
) -> None:
    """Neither XTC nor DCD carries a topology, so one has to be found."""
    (tmp_path / "02_nvt.xtc").write_bytes(b"")
    write_manifest(tmp_path, {"02_nvt": {"final_pdb": str(tmp_path / "02_nvt.pdb")}})
    with pytest.raises(AnalysisError, match="no topology beside it"):
        stage_files(tmp_path, "02_nvt")


def test_a_trajectory_falls_back_to_the_final_snapshot_for_its_topology(
    tmp_path: Path,
) -> None:
    """A run copied without its _topology.pdb is still readable."""
    (tmp_path / "02_nvt.xtc").write_bytes(b"")
    (tmp_path / "02_nvt.pdb").write_text("")
    write_manifest(tmp_path, {"02_nvt": {"final_pdb": str(tmp_path / "02_nvt.pdb")}})
    found = stage_files(tmp_path, "02_nvt")
    assert found.topology is not None
    assert found.topology.endswith("02_nvt.pdb")


def test_a_stage_that_wrote_nothing_readable_says_what_it_looked_for(
    tmp_path: Path,
) -> None:
    """Every stage writes a final PDB, so its absence means a crash mid-stage."""
    write_manifest(tmp_path, {"02_nvt": {}})
    with pytest.raises(AnalysisError, match="wrote no structure"):
        open_run(tmp_path, "02_nvt")


def test_a_real_trajectory_reads_back_with_the_frames_that_were_asked_for(
    dimer_run_directory: Path,
) -> None:
    """The whole seam, against a trajectory this package wrote itself."""
    ensemble = open_run(dimer_run_directory, "02_nvt")
    assert ensemble.n_frames == 10
    assert ensemble.n_chains == 32
    assert ensemble.atoms_per_chain == 2
    assert not ensemble.is_snapshot
    assert ensemble.interval_ps == pytest.approx(0.2, abs=1e-6)


def test_coordinates_come_back_in_nanometres_not_angstrom(
    dimer_run_directory: Path,
) -> None:
    """MDAnalysis works in Angstrom and this package in nanometres, and a
    silent factor of ten would make every length plausible and wrong."""
    ensemble = open_run(dimer_run_directory, "02_nvt")
    frame = next(ensemble.frames())
    assert frame.box_nm == pytest.approx([2.4, 2.4, 2.4], abs=1e-4)
    assert float(np.abs(frame.positions_nm).max()) < 2.5


def test_the_topology_pdb_holds_the_lattice_the_run_started_from(
    dimer_trajectory: Any,
) -> None:
    """Written at stage start, so it is the untouched lattice - which makes the
    dimers exactly 0.6 nm apart and their radius of gyration exactly 0.3 nm."""
    from openmmpolymer.trajectory import StageFiles, _open_stage

    files = StageFiles(
        stage="02_nvt",
        prefix="02_nvt",
        csv=None,
        final_pdb=None,
        final_state=None,
        trajectory=None,
        topology="02_nvt_topology.pdb",
        n_molecules=32,
        atoms_per_chain=2,
    )
    ensemble = _open_stage(files)
    chains = ensemble.per_chain(next(ensemble.frames()).positions_nm)
    separations = np.linalg.norm(chains[:, 1] - chains[:, 0], axis=1)
    assert separations == pytest.approx(0.6, abs=1e-4)


def test_masses_come_from_the_topology_rather_than_being_guessed(
    dimer_run_directory: Path,
) -> None:
    """MDAnalysis would guess them, and guess wrong under hydrogen-mass
    repartitioning and for virtual sites."""
    ensemble = open_run(dimer_run_directory, "02_nvt")
    assert ensemble.masses_amu == pytest.approx([39.948, 39.948], abs=1e-3)
    assert not ensemble.is_hydrogen.any()


def test_a_stride_skips_frames_and_keeps_their_times(
    dimer_run_directory: Path,
) -> None:
    """Long runs are read at a stride, and the time axis has to follow."""
    ensemble = open_run(dimer_run_directory, "02_nvt")
    positions, times = chain_positions(ensemble, stride=2)
    assert positions.shape == (5, 32, 2, 3)
    assert times == pytest.approx([0.2, 0.6, 1.0, 1.4, 1.8], abs=1e-5)


def test_the_last_frame_is_the_one_the_stage_ended_on(
    dimer_run_directory: Path,
) -> None:
    """It is the frame a final measurement should be taken from."""
    ensemble = open_run(dimer_run_directory, "02_nvt")
    assert ensemble.last_frame().time_ps == pytest.approx(2.0, abs=1e-5)


def test_every_frame_reports_its_own_box(dimer_run_directory: Path) -> None:
    """Under a barostat the cell changes, and a density or a g(r) cut-off that
    used the first frame's box would drift out of date."""
    assert boxes_nm(open_run(dimer_run_directory, "02_nvt")).shape == (10, 3)


def test_a_stage_with_no_trajectory_reads_back_as_a_single_snapshot(
    dimer_trajectory: Any, tmp_path: Path
) -> None:
    """The ordinary case: no shipped protocol writes a trajectory at all, and
    every stage still leaves a final PDB."""
    write_manifest(
        tmp_path,
        {"00_minimise": {"final_pdb": str(Path("00_minimise.pdb").resolve())}},
    )
    ensemble = open_run(tmp_path, "00_minimise")
    assert ensemble.is_snapshot
    assert ensemble.n_frames == 1
    assert ensemble.interval_ps == 0.0
    assert ensemble.trajectory_path is None


def test_a_snapshot_is_refused_where_a_trajectory_is_needed() -> None:
    """A displacement or a relaxation time needs more than one frame, and
    saying so beats returning a number from one."""
    snapshot = synthetic_ensemble(np.zeros((4, 3)), n_chains=2)
    with pytest.raises(AnalysisError, match="needs a trajectory"):
        require_trajectory(snapshot, "A measurement")


def test_an_explicit_block_size_overrides_what_the_residues_say(
    dimer_run_directory: Path,
) -> None:
    """The argon cell's residues are one atom each, and the dimers it stands in
    for are defined by index arithmetic, not by its topology."""
    ensemble = open_run(dimer_run_directory, "02_nvt", atoms_per_chain=4)
    assert ensemble.n_chains == 16
    assert ensemble.atoms_per_chain == 4


def test_a_block_size_that_does_not_divide_the_cell_is_refused(
    dimer_run_directory: Path,
) -> None:
    """Silently truncating would average over atoms from the wrong chains."""
    with pytest.raises(AnalysisError, match="does not divide"):
        open_run(dimer_run_directory, "02_nvt", atoms_per_chain=7)


def test_positions_for_the_wrong_cell_are_refused(
    dimer_run_directory: Path,
) -> None:
    """Reshaping the wrong array would give per-chain blocks of other chains."""
    ensemble = open_run(dimer_run_directory, "02_nvt")
    with pytest.raises(AnalysisError, match="Positions hold"):
        ensemble.per_chain(np.zeros((10, 3)))


def test_a_backbone_of_one_atom_has_no_end_to_end_vector() -> None:
    """Two atoms is the minimum that spans anything."""
    with pytest.raises(AnalysisError, match="no end-to-end vector"):
        backbone_indices([0], 4)


def test_a_backbone_indexing_past_the_chain_is_refused() -> None:
    """The indices are within one chain, and using whole-cell indices instead
    is the easy mistake to make."""
    with pytest.raises(AnalysisError, match="outside a chain"):
        backbone_indices([0, 64], 2)


def test_a_load_too_large_for_memory_is_refused_with_the_arithmetic() -> None:
    """Better than being killed by the OOM reaper halfway through."""
    from openmmpolymer.trajectory import _check_size

    with pytest.raises(AnalysisError, match="GiB of positions"):
        _check_size(50_000, 10_000)


# When the XDR reader fails to open a frameless trajectory it leaves a
# half-built reader behind, whose __del__ then raises into nothing. pytest turns
# that unraisable exception into a warning, and this suite makes warnings
# errors, so the failure would be reported against MDAnalysis's deallocator
# rather than against the assertion below. Scoped to this one test, and a
# builtin marker, so --strict-markers is satisfied without registering one.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_an_empty_trajectory_says_why_it_is_empty(
    dimer_argon_run: Any, tmp_path: Path
) -> None:
    """A stage shorter than its own frame interval writes no frames at all, and
    the fix is a smaller interval_ps rather than anything about the analysis."""
    from openmmpolymer.reporters import TrajectoryOptions
    from openmmpolymer.simulate import run_nvt

    result = run_nvt(
        dimer_argon_run,
        "09_empty",
        temperature_k=120.0,
        duration_ps=2.0,
        friction_ps=20.0,
        trajectory=TrajectoryOptions("xtc", interval_ps=10.0),
    )
    write_manifest(
        tmp_path,
        {"09_empty": {"final_pdb": str(Path(result.final_pdb or "").resolve())}},
        box={"n_molecules": 32, "atoms_per_chain": 2},
    )
    (tmp_path / "09_empty.xtc").write_bytes(Path("09_empty.xtc").read_bytes())
    (tmp_path / "09_empty_topology.pdb").write_bytes(
        Path("09_empty_topology.pdb").read_bytes()
    )
    with pytest.raises(AnalysisError, match="could not be read"):
        open_run(tmp_path, "09_empty")


def test_a_trajectory_the_universe_cannot_match_to_its_topology_is_refused(
    dimer_run_directory: Path,
) -> None:
    """A topology from a different system would put every atom's coordinates on
    the wrong atom, which is not something to discover from the answers."""
    from openmmpolymer.trajectory import AnalysisError as Error

    with pytest.raises(Error, match="does not divide"):
        open_run(dimer_run_directory, "02_nvt", atoms_per_chain=5)


def test_a_topology_with_uneven_residues_is_refused(tmp_path: Path) -> None:
    """Every molecule should be a copy of the same chain, and an uneven cell
    would make every per-chain average silently wrong."""
    from openmm import app, unit

    from openmmpolymer.trajectory import _blocks

    topology = app.Topology()
    chain = topology.addChain()
    carbon = app.Element.getBySymbol("C")
    for count in (2, 3):
        residue = topology.addResidue("POL", chain)
        for index in range(count):
            topology.addAtom(f"C{index}", carbon, residue)
    topology.setPeriodicBoxVectors([[2, 0, 0], [0, 2, 0], [0, 0, 2]] * unit.nanometer)
    with pytest.raises(AnalysisError, match=r"do not divide evenly|atoms but residue"):
        _blocks(topology, None)


def test_an_empty_topology_is_refused() -> None:
    """Nothing to measure, and the reshape would divide by zero."""
    from openmm import app

    from openmmpolymer.trajectory import _blocks

    with pytest.raises(AnalysisError, match="no atoms"):
        _blocks(app.Topology(), None)


def test_a_trajectory_of_a_different_system_is_refused(
    dimer_trajectory: Any, tmp_path: Path
) -> None:
    """Coordinates read against the wrong topology put every atom's position on
    some other atom, and the answers would look perfectly reasonable.

    MDAnalysis catches this itself and names both atom counts; what matters
    here is that it reaches the caller as this package's own error type.
    """
    from openmmpolymer.trajectory import StageFiles, _open_stage

    del dimer_trajectory  # written for its 02_nvt.xtc in the working directory
    other = tmp_path / "other_topology.pdb"
    from openmm import app, unit

    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("AR", app.Element.getBySymbol("Ar"), residue)
    topology.setPeriodicBoxVectors(
        [[2.4, 0, 0], [0, 2.4, 0], [0, 0, 2.4]] * unit.nanometer
    )
    with other.open("w") as handle:
        app.PDBFile.writeFile(topology, [[0.0, 0.0, 0.0]] * unit.nanometer, handle)
    files = StageFiles(
        stage="02_nvt",
        prefix="02_nvt",
        csv=None,
        final_pdb=None,
        final_state=None,
        trajectory="02_nvt.xtc",
        topology=str(other),
    )
    with pytest.raises(AnalysisError, match="could not be read"):
        _open_stage(files)


def test_an_atom_without_an_element_has_no_mass() -> None:
    """A PDB written without its element column would otherwise give a cell of
    massless chains and a centre of mass at the origin."""
    from openmm import app

    from openmmpolymer.trajectory import _chain_atoms

    topology = app.Topology()
    residue = topology.addResidue("POL", topology.addChain())
    topology.addAtom("X", None, residue)
    with pytest.raises(AnalysisError, match="no element"):
        _chain_atoms(topology, 1)


def test_a_wrapped_trajectory_is_reported_rather_than_measured_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stages write unwrapped coordinates on purpose. A run made the other way
    is still readable, and every conformational number off it is nonsense."""
    import logging

    from openmmpolymer.trajectory import _check_unwrapped

    frames = np.zeros((2, 2, 3), dtype=np.float64)
    frames[1, 0, 0] = 2.0  # further than half a 2.4 nm cell
    wrapped = synthetic_ensemble(frames, n_chains=1, box_nm=2.4, interval_ps=1.0)
    with caplog.at_level(logging.WARNING, logger="openmmpolymer.trajectory"):
        _check_unwrapped(wrapped)
    assert "looks wrapped" in caplog.text


def test_a_one_frame_ensemble_is_not_checked_for_wrapping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """There is no displacement between frames to measure, so the check has
    nothing to say rather than indexing off the end of the trajectory."""
    import logging

    from openmmpolymer.trajectory import _check_unwrapped

    snapshot = synthetic_ensemble(np.zeros((2, 3)), n_chains=1, box_nm=2.4)
    with caplog.at_level(logging.WARNING, logger="openmmpolymer.trajectory"):
        _check_unwrapped(snapshot)
    assert caplog.text == ""


def test_a_manifest_recording_nonsense_for_its_cell_is_treated_as_silent() -> None:
    """A manifest is a file and a file can be edited, so a block size that is
    not a usable count falls back to the topology rather than being trusted."""
    from openmmpolymer.trajectory import _recorded_int

    assert _recorded_int({}, "atoms_per_chain") is None
    assert _recorded_int({"atoms_per_chain": 0}, "atoms_per_chain") is None
    assert _recorded_int({"atoms_per_chain": True}, "atoms_per_chain") is None
    assert _recorded_int({"atoms_per_chain": "12"}, "atoms_per_chain") is None
    assert _recorded_int({"atoms_per_chain": 12}, "atoms_per_chain") == 12


def test_a_pdb_trajectory_is_not_mistaken_for_the_closing_snapshot(
    dimer_argon_run: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A PDB trajectory records no frame times, so it is read as the closing
    snapshot rather than analysed with fabricated lags - and the run is told
    which format to use if it wanted the frames measured."""
    import logging

    from openmmpolymer.reporters import TrajectoryOptions
    from openmmpolymer.simulate import run_nvt

    result = run_nvt(
        dimer_argon_run,
        "02_nvt",
        temperature_k=120.0,
        duration_ps=2.0,
        friction_ps=20.0,
        trajectory=TrajectoryOptions("pdb", interval_ps=0.5),
    )
    for name in ("02_nvt.pdb", "02_nvt_trajectory.pdb"):
        (tmp_path / name).write_bytes(Path(name).read_bytes())
    write_manifest(
        tmp_path,
        {"02_nvt": {"final_pdb": str(tmp_path / "02_nvt.pdb")}},
        box={"n_molecules": 32, "atoms_per_chain": 2},
    )
    assert result.final_pdb == "02_nvt.pdb"

    with caplog.at_level(logging.INFO, logger="openmmpolymer.trajectory"):
        found = stage_files(tmp_path, "02_nvt")
    assert found.trajectory is None
    assert found.topology is not None
    assert found.topology.endswith("02_nvt.pdb")
    assert "trajectory='xtc'" in caplog.text

    ensemble = open_run(tmp_path, "02_nvt")
    assert ensemble.is_snapshot
    assert ensemble.n_frames == 1
