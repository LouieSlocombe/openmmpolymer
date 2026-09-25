"""Tests for what a stage writes while it runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import openmm as mm
import pytest
from openmm.app.internal.xtc_utils import get_xtc_nframes

from openmmpolymer.reporters import (
    CSV_COLUMNS,
    AtomicStateReporter,
    TrajectoryOptions,
    _path_for,
    reporting,
    rotate_existing,
    steps_for,
)
from openmmpolymer.simulate import run_nvt

from .helpers import bare_simulation


def _simulation(run: Any) -> Any:
    """A bare Simulation over a run context's cell, on the CPU."""
    return bare_simulation(
        mm.XmlSerializer.deserialize(run.system_xml),
        run.box.topology,
        run.box.positions_nm,
        platform="CPU",
    )


@pytest.mark.parametrize(
    ("duration_ps", "timestep_fs", "expected"),
    [(1.0, 1.0, 1000), (0.5, 2.0, 250), (10.0, 0.5, 20_000), (0.0, 2.0, 1)],
)
def test_steps_for_converts_time_to_steps(
    duration_ps: float, timestep_fs: float, expected: int
) -> None:
    """A stage is specified in time and run in steps, and never zero of them."""
    assert steps_for(duration_ps, timestep_fs) == expected


def test_rotate_existing_moves_a_file_aside_rather_than_deleting_it(
    tmp_path: Path,
) -> None:
    """A partial trajectory from a crashed attempt is still evidence."""
    target = tmp_path / "traj.xtc"
    target.write_text("frames")
    moved = rotate_existing(target)
    assert moved is not None
    assert Path(moved).read_text() == "frames"
    assert not target.exists()


def test_rotate_existing_numbers_successive_attempts(tmp_path: Path) -> None:
    """A second crash must not overwrite the first one's evidence."""
    target = tmp_path / "traj.xtc"
    for _ in range(2):
        target.write_text("frames")
        rotate_existing(target)
    assert {path.name for path in tmp_path.iterdir()} == {
        "traj.attempt1.xtc",
        "traj.attempt2.xtc",
    }


def test_rotate_existing_ignores_a_file_that_is_not_there(tmp_path: Path) -> None:
    """The ordinary case, on a first run."""
    assert rotate_existing(tmp_path / "absent.xtc") is None


def test_rotate_existing_ignores_an_empty_file(tmp_path: Path) -> None:
    """XTCReporter only refuses a file with something in it."""
    target = tmp_path / "traj.xtc"
    target.touch()
    assert rotate_existing(target) is None


def test_the_state_reporter_alternates_and_points_at_the_last_complete_file(
    argon_run: Any,
) -> None:
    """A crash during a write must not destroy the only checkpoint there was.

    The pointer moves only once a file is complete, and a pointer to a file
    that is not there reads as no state at all: half a restart is worse than
    none.
    """
    reporter = AtomicStateReporter("stage", 10)
    assert reporter.current_state() is None
    simulation = _simulation(argon_run)

    reporter.report(simulation, None)
    assert reporter.current_state() == reporter.state_path("a")
    reporter.report(simulation, None)
    assert reporter.current_state() == reporter.state_path("b")
    assert reporter.state_path("a").is_file()
    mm.XmlSerializer.deserialize(reporter.state_path("b").read_text())

    reporter.state_path("b").unlink()
    assert reporter.current_state() is None


def test_the_reporter_asks_openmm_for_the_right_interval(tmp_path: Path) -> None:
    """The modern reporter protocol is a dict, not a tuple."""

    class FakeSimulation:
        currentStep = 7

    described = AtomicStateReporter(tmp_path / "stage", 10).describeNextReport(
        FakeSimulation()
    )
    assert described == {"steps": 3, "periodic": None, "include": []}


def test_trajectory_options_default_to_unwrapped_xtc() -> None:
    """Wrapping splits a chain across a face, and a split chain has no Rg."""
    options = TrajectoryOptions()
    assert options.format == "xtc"
    assert options.enforce_periodic_box is False


def test_a_trajectory_format_nothing_writes_is_refused(argon_run: Any) -> None:
    """``"xyz"`` used to write a PDB into ``<stem>.xyz`` without a word.

    Refused when the options are built, and a bare format string is refused
    before the stage writes anything.
    """
    with pytest.raises(ValueError, match="format='xyz'"):
        TrajectoryOptions("xyz")
    with (
        pytest.raises(ValueError, match="format='xyz'"),
        reporting(
            _simulation(argon_run),
            "stage",
            total_steps=20,
            report_interval=10,
            trajectory="xyz",
        ),
    ):
        pass
    assert list(Path().iterdir()) == []


@pytest.mark.parametrize("trajectory_format", ["xtc", "dcd"])
def test_reporting_writes_a_topology_beside_a_binary_trajectory(
    argon_run: Any, trajectory_format: str
) -> None:
    """Written up front, because a crashed run is the one that needs it.

    Both binary formats, because neither carries a topology of its own and the
    reporters are built by separate branches.
    """
    simulation = _simulation(argon_run)
    with reporting(
        simulation,
        "stage",
        total_steps=100,
        report_interval=50,
        trajectory=trajectory_format,
    ) as paths:
        simulation.step(100)
    assert Path(paths.topology or "").is_file()
    assert Path(paths.trajectory or "").is_file()
    assert (paths.trajectory or "").endswith(trajectory_format)


def test_reporting_can_write_no_trajectory_at_all(argon_run: Any) -> None:
    """Equilibration stages usually should not."""
    simulation = _simulation(argon_run)
    with reporting(
        simulation, "stage", total_steps=50, report_interval=25, trajectory="none"
    ) as paths:
        simulation.step(50)
    assert paths.trajectory is None
    assert paths.topology is None


def test_reporting_takes_its_reporters_away_afterwards(argon_run: Any) -> None:
    """A stage that left its reporters attached would report into the next one."""
    simulation = _simulation(argon_run)
    with reporting(simulation, "stage", total_steps=20, report_interval=10):
        assert simulation.reporters
    assert simulation.reporters == []


def test_reporting_clears_its_reporters_even_when_the_stage_fails(
    argon_run: Any,
) -> None:
    """Especially then."""
    simulation = _simulation(argon_run)
    with (
        pytest.raises(RuntimeError, match="stage blew up"),
        reporting(simulation, "stage", total_steps=20, report_interval=10),
    ):
        raise RuntimeError("stage blew up")
    assert simulation.reporters == []


def test_the_csv_columns_are_the_documented_ones(argon_run: Any) -> None:
    """They are what the convergence numbers are read from."""
    simulation = _simulation(argon_run)
    with reporting(
        simulation, "stage", total_steps=20, report_interval=10, trajectory="none"
    ) as paths:
        simulation.step(20)
    header = Path(paths.csv).read_text().splitlines()[0]
    assert len(header.split(",")) == len(CSV_COLUMNS)
    assert "Progress" not in header


def test_a_requested_frame_interval_is_what_the_trajectory_gets(
    dimer_argon_run: Any,
) -> None:
    """TrajectoryOptions.interval_ps is documented as the time between frames,
    and was read nowhere: the stride came from the state-data interval instead,
    so a caller asking for a frame every picosecond got something unrelated.
    """
    run_nvt(
        dimer_argon_run,
        "02_nvt",
        temperature_k=120.0,
        duration_ps=2.0,
        friction_ps=20.0,
        trajectory=TrajectoryOptions("xtc", interval_ps=0.5),
    )
    # 2 ps at 2 fs is 1000 steps; a frame every 0.5 ps is every 250 of them.
    assert get_xtc_nframes(b"02_nvt.xtc") == 4


def test_naming_a_format_as_a_string_keeps_the_interval_it_always_had(
    dimer_argon_run: Any,
) -> None:
    """Making interval_ps real should not quietly change what every existing
    stage writes, and a bare format string is what the stages pass."""
    run_nvt(
        dimer_argon_run,
        "02_nvt",
        temperature_k=120.0,
        duration_ps=2.0,
        friction_ps=20.0,
        trajectory="xtc",
        report_interval_ps=0.02,
    )
    # Unchanged behaviour: ten times the state-data interval, so every 100 steps.
    assert get_xtc_nframes(b"02_nvt.xtc") == 10


def test_a_stage_too_short_to_reach_a_frame_says_so(
    dimer_argon_run: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """It writes an unreadable empty trajectory, and the default interval of
    ten picoseconds does it to any stage shorter than that."""
    with caplog.at_level(logging.WARNING, logger="openmmpolymer.simulate"):
        run_nvt(
            dimer_argon_run,
            "02_nvt",
            temperature_k=120.0,
            duration_ps=2.0,
            friction_ps=20.0,
            trajectory=TrajectoryOptions("xtc", interval_ps=10.0),
        )
    assert "no frames in it" in caplog.text


def test_a_pdb_trajectory_is_kept_apart_from_the_closing_structure(
    dimer_argon_run: Any,
) -> None:
    """It used to go to <stem>.pdb, which is also where the stage writes its
    closing structure, so the snapshot overwrote every frame on the way out.

    The snapshot is one frame and the trajectory is many, and every caller of
    StageResult.final_pdb expects the former. A trajectory from an earlier
    attempt is still evidence, and is moved aside from the trajectory's own
    path rather than the snapshot's.
    """
    Path("02_nvt_trajectory.pdb").write_text("REMARK an earlier attempt\n")
    result = run_nvt(
        dimer_argon_run,
        "02_nvt",
        temperature_k=120.0,
        duration_ps=2.0,
        friction_ps=20.0,
        trajectory=TrajectoryOptions("pdb", interval_ps=0.5),
    )
    frames = Path("02_nvt_trajectory.pdb").read_text()
    assert frames.count("\nMODEL ") == 4
    assert frames.count("ENDMDL") == 4
    assert result.final_pdb == "02_nvt.pdb"
    snapshot = Path("02_nvt.pdb").read_text()
    assert "\nMODEL " not in snapshot
    assert snapshot.count("CRYST1") == 1
    assert "earlier attempt" in Path("02_nvt_trajectory.attempt1.pdb").read_text()


def test_the_reported_trajectory_path_is_the_trajectory(
    dimer_argon_run: Any,
) -> None:
    """ReporterPaths.trajectory is what a caller opens, so it has to name the
    file with the frames in it rather than the snapshot beside it."""
    simulation = _simulation(dimer_argon_run)
    with reporting(
        simulation,
        "02_nvt",
        total_steps=100,
        report_interval=10,
        trajectory=TrajectoryOptions("pdb"),
        trajectory_interval=25,
    ) as paths:
        simulation.step(100)
        assert paths.trajectory == "02_nvt_trajectory.pdb"
        # A PDB carries its own topology, so none is written beside it.
        assert paths.topology is None


@pytest.mark.parametrize(
    ("trajectory_format", "expected"),
    [
        ("xtc", "05_npt.xtc"),
        ("dcd", "05_npt.dcd"),
        ("pdb", "05_npt_trajectory.pdb"),
    ],
)
def test_each_format_has_its_own_path(trajectory_format: str, expected: str) -> None:
    """Only pdb is special, and the binary formats must keep the names every
    run already on disk used."""
    assert _path_for("05_npt", trajectory_format).name == expected
