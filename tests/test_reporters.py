"""Tests for what a stage writes while it runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openmmpolymer.reporters import (
    CSV_COLUMNS,
    AtomicStateReporter,
    TrajectoryOptions,
    reporting,
    rotate_existing,
    steps_for,
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


def test_the_state_reporter_alternates_between_two_files(tmp_path: Path) -> None:
    """A crash during a write must not destroy the only checkpoint there was."""
    reporter = AtomicStateReporter(tmp_path / "stage", 10)
    assert reporter.state_path("a") != reporter.state_path("b")
    assert reporter.current_state() is None


def test_the_state_pointer_names_the_file_that_was_written(tmp_path: Path) -> None:
    """The pointer moves only once the file is complete."""
    reporter = AtomicStateReporter(tmp_path / "stage", 10)
    reporter.state_path("a").write_text("<State/>")
    reporter.pointer_path.write_text("a")
    assert reporter.current_state() == reporter.state_path("a")


def test_a_pointer_to_a_missing_file_reads_as_no_state(tmp_path: Path) -> None:
    """Half a restart is worse than none."""
    reporter = AtomicStateReporter(tmp_path / "stage", 10)
    reporter.pointer_path.write_text("b")
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


def _simulation(run: Any) -> Any:
    """Build a bare Simulation over a run context's cell."""
    import openmm as mm
    from openmm import app, unit

    system = mm.XmlSerializer.deserialize(run.system_xml)
    integrator = mm.LangevinMiddleIntegrator(
        100.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtoseconds
    )
    simulation = app.Simulation(
        run.box.topology, system, integrator, mm.Platform.getPlatformByName("CPU")
    )
    simulation.context.setPositions(run.box.positions)
    return simulation


def test_reporting_writes_a_topology_beside_a_binary_trajectory(
    argon_run: Any,
) -> None:
    """Written up front, because a crashed run is the one that needs it."""
    simulation = _simulation(argon_run)
    with reporting(
        simulation, "stage", total_steps=100, report_interval=50, trajectory="xtc"
    ) as paths:
        simulation.step(100)
    assert Path(paths.topology or "").is_file()
    assert Path(paths.trajectory or "").is_file()


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
    from openmm.app.internal.xtc_utils import get_xtc_nframes

    from openmmpolymer.reporters import TrajectoryOptions
    from openmmpolymer.simulate import run_nvt

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
    from openmm.app.internal.xtc_utils import get_xtc_nframes

    from openmmpolymer.simulate import run_nvt

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
    import logging

    from openmmpolymer.reporters import TrajectoryOptions
    from openmmpolymer.simulate import run_nvt

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
