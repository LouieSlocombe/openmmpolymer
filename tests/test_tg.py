"""Tests for the two-pass glass-transition scan and what it reports.

Everything that decides something - which waypoint to carry on from, how wide
the window is, when to refuse - is tested against manifests written by hand,
because those decisions are arithmetic over recorded numbers and running
dynamics to reach them would only hide what is being checked. The plumbing
that has to survive a real Context is tested once, on an argon cell, which
runs the whole thing in a couple of seconds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer.tg import (
    COARSE_STEM,
    FINE_STEM,
    PRECOOL_STEM,
    TgError,
    TgSpec,
    analyse_tg,
    coarse_schedule,
    cooling_rate_series,
    fine_schedule,
    fine_window,
    nominal_fine_schedule,
    pick_waypoint,
    run_tg_scan,
    tg_fine_scan,
    write_tg_report,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    QUICK_EQUILIBRATION,
    argon_context,
    two_line_curve,
    write_quench,
    write_quenches,
)

#: Settings that put the whole two-pass scan inside a couple of seconds on the
#: argon cell. The cell has no glass transition, so the window is named rather
#: than fitted - which is also the escape hatch a real caller would use.
QUICK = TgSpec(
    melt_temperature_k=150.0,
    t_floor_k=90.0,
    coarse_step_k=10.0,
    coarse_hold_ps=0.4,
    window_k=20.0,
    fine_step_k=5.0,
    fine_hold_ps=0.4,
    stage_ps=2.0,
    samples_per_segment=4,
    min_points_per_branch=2,
)


# --------------------------------------------------------------------------
# The window and the waypoint
# --------------------------------------------------------------------------


def test_the_fine_window_is_centred_on_the_coarse_transition() -> None:
    """Sixty kelvin either side, which is the coarse fit's own uncertainty."""
    assert fine_window(420.0, TgSpec()) == (480.0, 360.0)


def test_a_window_running_off_the_top_of_the_ladder_is_clamped() -> None:
    """There is no data above the melt temperature to put a branch through."""
    top, bottom = fine_window(620.0, TgSpec())
    assert top == pytest.approx(650.0)
    assert bottom == pytest.approx(560.0)


def test_a_window_running_off_the_bottom_is_clamped_too() -> None:
    """And the clamp is said out loud, not applied quietly."""
    top, bottom = fine_window(170.0, TgSpec())
    assert top == pytest.approx(230.0)
    assert bottom == pytest.approx(150.0)


def test_a_window_that_clamps_to_nothing_is_refused() -> None:
    """The transition sits outside the range that was actually scanned."""
    with pytest.raises(TgError, match="empty"):
        fine_window(900.0, TgSpec(melt_temperature_k=650.0, t_floor_k=600.0))


def test_the_fine_pass_carries_on_from_the_lowest_waypoint_above_the_window() -> None:
    """As late on the coarse trajectory as it can, still entering from above.

    Restarting higher would re-cool ground the coarse pass already covered;
    restarting lower would enter the window from inside it.
    """
    candidates = [(650.0, "hot"), (500.0, "just above"), (475.0, "inside")]
    assert pick_waypoint(candidates, 480.0) == (500.0, "just above")


def test_with_nothing_above_the_window_the_hottest_waypoint_is_taken() -> None:
    """Better than dropping in from the melt, and it says so."""
    assert pick_waypoint([(300.0, "a"), (280.0, "b")], 480.0) == (300.0, "a")


def test_no_waypoints_at_all_is_not_a_choice() -> None:
    """The caller then has to fall back on something else."""
    assert pick_waypoint([], 480.0) is None


# --------------------------------------------------------------------------
# Schedules and chunking
# --------------------------------------------------------------------------


def test_the_coarse_ladder_covers_the_range_at_the_step_asked_for() -> None:
    """Twenty-one temperatures from 650 K to 150 K in 25 K drops."""
    schedule = coarse_schedule(TgSpec())

    assert schedule.n_temperatures == 21
    assert schedule.temperatures_k[0] == pytest.approx(650.0)
    assert schedule.temperatures_k[-1] == pytest.approx(150.0)
    assert schedule.cooling_rate_k_per_ns == pytest.approx(25.0)
    assert schedule.total_ps == pytest.approx(21_000.0)


def test_the_size_of_the_fine_ladder_is_known_before_the_coarse_pass_runs() -> None:
    """The window is the same width wherever it lands, so the cost is too."""
    assert nominal_fine_schedule(TgSpec()).n_temperatures == 25


def test_the_chunks_visit_every_temperature_once_and_in_order() -> None:
    """A split for resume is bookkeeping; the ladder has to survive it."""
    spec = TgSpec(stage_ps=3000.0)
    schedule = fine_schedule(480.0, 360.0, spec)
    stages = tg_fine_scan(schedule, spec, timestep_fs=1.0).stages
    chunks = [stage.options["temperatures_k"] for stage in stages]
    walked = [temperature for chunk in chunks for temperature in chunk]

    assert len(chunks) > 1
    assert walked == list(schedule.temperatures_k)
    assert walked == sorted(walked, reverse=True)
    assert len(walked) == len(set(walked))


def test_no_chunk_is_left_holding_a_single_temperature() -> None:
    """A stage with no temperature step is one nothing can classify."""
    spec = TgSpec(stage_ps=2.0, fine_hold_ps=0.4)
    schedule = fine_schedule(140.0, 100.0, spec)
    stages = tg_fine_scan(schedule, spec, timestep_fs=1.0).stages

    assert min(len(stage.options["temperatures_k"]) for stage in stages) > 1


def test_a_fine_window_that_does_not_cool_is_refused() -> None:
    """A ladder needs somewhere to go."""
    with pytest.raises(TgError, match="does not cool"):
        fine_schedule(400.0, 400.0, TgSpec())


# --------------------------------------------------------------------------
# What a scan refuses, and when
# --------------------------------------------------------------------------


def test_an_unresolved_coarse_fit_refuses_and_names_the_ways_out(
    argon_scan_run: Any,
) -> None:
    """A fine pass is tens of nanoseconds.

    Spending them on a window derived from a fit that already reported it
    found a corner in noise produces a curve with nothing in it, and no way
    to tell that apart from a polymer with no transition in range. And a dead
    end with no exits is worse than the run it prevented.
    """
    with pytest.raises(TgError, match="did not resolve") as refusal:
        run_tg_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    assert "tg_approx_k" in str(refusal.value)


def test_a_scan_over_its_budget_stops_before_it_creates_anything(
    argon_run: Any,
) -> None:
    """Decided before it starts, not discovered three days in."""
    with pytest.raises(TgError, match="max_total_ns"):
        run_tg_scan(
            argon_run,
            "run",
            spec=replace(QUICK, max_total_ns=1.0e-6),
            tg_approx_k=120.0,
            **QUICK_EQUILIBRATION,
        )
    assert not Path("run").exists()


@pytest.mark.parametrize("rate", [-5.0, 0.0])
def test_every_cooling_rate_is_checked_before_any_dynamics(
    argon_run: Any, rate: float
) -> None:
    """A bad last rate cannot wait until the passes before it have run."""
    with pytest.raises(ValueError, match="rates_k_per_ns"):
        cooling_rate_series(
            argon_run,
            "run",
            rates_k_per_ns=(12500.0, rate),
            spec=QUICK,
            tg_approx_k=120.0,
            **QUICK_EQUILIBRATION,
        )
    assert not Path("run").exists()


def test_a_resume_that_names_another_window_is_refused_before_any_dynamics(
    argon_run: Any,
) -> None:
    """The window is part of what was asked for, not only the spec."""
    Path("run").mkdir()
    Path("run/tg_workflow.json").write_text(
        json.dumps({"request": {"spec": asdict(QUICK), "tg_approx_k": 100.0}})
    )
    with pytest.raises(TgError, match=r"\(tg_approx_k\)"):
        run_tg_scan(
            argon_run, "run", spec=QUICK, tg_approx_k=120.0, **QUICK_EQUILIBRATION
        )
    assert not Path("run/manifest.json").exists()


def test_two_passes_at_the_same_rate_are_refused(argon_run: Any) -> None:
    """They would be one measurement recorded under two names."""
    with pytest.raises(TgError, match="not distinct"):
        cooling_rate_series(argon_run, "run", rates_k_per_ns=(10.0, 10.0), spec=QUICK)
    assert not Path("run").exists()


# --------------------------------------------------------------------------
# The whole scan, on argon
# --------------------------------------------------------------------------


@pytest.mark.parametrize("initial_resume", [True, False])
def test_the_whole_scan_runs_records_what_it_derived_and_resumes(
    argon_scan_run: Any,
    initial_resume: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The load-bearing one: both passes, the waypoint between them, resume.

    The cost goes to the log before anything runs, because in a queue the
    outstanding cost is the only number anyone can act on. What the scan
    derived goes beside the manifest rather than in it. And a resume that asks
    for something else is refused: it would keep numbers measured under the
    settings it replaced.
    """
    with caplog.at_level(logging.INFO, logger="openmmpolymer.tg"):
        result = run_tg_scan(
            argon_scan_run,
            "run",
            spec=QUICK,
            tg_approx_k=120.0,
            resume=initial_resume,
            **QUICK_EQUILIBRATION,
        )
    assert "still to run" in caplog.text

    assert result.restart == "waypoint"
    assert "waypoint" in Path(result.start_state).name
    assert Path(result.start_state).is_file()
    # The window's top is 140 K, so the fine pass carries on from the lowest
    # coarse temperature at or above it.
    assert result.fine_schedule.temperatures_k[0] == pytest.approx(140.0)
    assert result.fine_schedule.temperatures_k[-1] == pytest.approx(100.0)
    assert result.coarse_curve.n_points == 7
    assert result.fine_curve is not None
    assert result.fine_curve.n_points == 9

    workflow = json.loads(Path("run/tg_workflow.json").read_text())
    assert workflow["tg_used_k"] == pytest.approx(120.0)
    assert workflow["window_top_k"] == pytest.approx(140.0)
    assert workflow["window_bottom_k"] == pytest.approx(100.0)
    assert workflow["restart"] == "waypoint"
    assert workflow["request"]["spec"]["window_k"] == pytest.approx(20.0)
    assert workflow["coarse_stages"]

    # One timestep across every fine chunk: a measurement integrated three
    # ways is a confound nobody would choose.
    manifest = json.loads(Path("run/manifest.json").read_text())
    fine = [name for name in manifest["stages"] if name.startswith(FINE_STEM)]
    assert len(fine) > 1
    assert {
        manifest["provenance"]["stages"][name]["request"]["options"]["timestep_fs"]
        for name in fine
    } == {workflow["timestep_fs"]}

    again = run_tg_scan(
        argon_scan_run, "run", spec=QUICK, tg_approx_k=120.0, **QUICK_EQUILIBRATION
    )
    assert again.coarse_summary.results == ()
    assert again.fine_summary.results == ()
    assert again.fine_summary.skipped

    with pytest.raises(TgError, match="fine_hold_ps"):
        run_tg_scan(
            argon_scan_run,
            "run",
            spec=replace(QUICK, fine_hold_ps=0.8),
            tg_approx_k=120.0,
            **QUICK_EQUILIBRATION,
        )


def test_a_deleted_waypoint_falls_back_to_a_pre_cool_from_the_melt(
    argon_scan_run: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A different thermal history, so it warns rather than doing it quietly."""
    run_tg_scan(
        argon_scan_run, "run", spec=QUICK, tg_approx_k=120.0, **QUICK_EQUILIBRATION
    )
    for path in Path("run").glob(f"{COARSE_STEM}*waypoint*"):
        path.unlink()
    for path in Path("run").glob(f"{FINE_STEM}*"):
        path.unlink()

    with caplog.at_level(logging.WARNING, logger="openmmpolymer.tg"):
        again = run_tg_scan(
            argon_scan_run,
            "run",
            spec=QUICK,
            tg_approx_k=120.0,
            **QUICK_EQUILIBRATION,
        )
    assert again.restart == "precool"
    assert "different thermal history" in caplog.text
    assert PRECOOL_STEM in {result.name for result in again.fine_summary.results}


def test_several_cooling_rates_start_from_one_configuration() -> None:
    """Independent histories from a common melt, which is what makes them
    comparable at all.

    Run on dimers with a backbone, because every pass measures the chains at
    its end: a fine pass that did not would clear the dimensions and the
    backbone the coarse pass recorded, and the melt check and the structure
    report both read them from the manifest.
    """
    fits = cooling_rate_series(
        argon_context(216, 2.8, atoms_per_molecule=2),
        "run",
        rates_k_per_ns=(12500.0, 6250.0),
        spec=replace(QUICK, stage_ps=1.0e6),
        tg_approx_k=120.0,
        chain_backbone=(0, 1),
        atoms_per_chain=2,
        **QUICK_EQUILIBRATION,
    )
    assert [fit.cooling_rate_k_per_ns for fit in fits] == [
        pytest.approx(12500.0),
        pytest.approx(6250.0),
    ]
    chains = json.loads(Path("run/manifest.json").read_text())["chains"]
    assert chains["backbone"] == [0, 1]
    assert chains["mean_radius_of_gyration_nm"] > 0.0


# --------------------------------------------------------------------------
# Reading a finished directory
# --------------------------------------------------------------------------


def two_pass_directory(directory: Path) -> Path:
    """A manifest shaped like a finished two-pass scan, with no dynamics.

    The coarse pass steps 40 K and the fine one 20 K, both around the same
    340 K knot, so which is which is a question the data answers.
    """
    coarse_t, coarse_d = two_line_curve(transition_k=340.0, n_points=11)
    fine_t, fine_d = two_line_curve(transition_k=340.0, n_points=21)
    return write_quenches(
        directory,
        {
            "06_coarse_quench_00": {
                "temperature_k": list(coarse_t[::-1]),
                "density_g_cm3": list(coarse_d[::-1]),
                "segment_duration_ps": [1000.0] * 11,
            },
            "08_fine_quench_00": {
                "temperature_k": list(fine_t[::-1][:11]),
                "density_g_cm3": list(fine_d[::-1][:11]),
                "segment_duration_ps": [4000.0] * 11,
            },
            "08_fine_quench_01": {
                "temperature_k": list(fine_t[::-1][11:]),
                "density_g_cm3": list(fine_d[::-1][11:]),
                "segment_duration_ps": [4000.0] * 10,
            },
        },
    )


def test_the_chunks_of_one_pass_come_back_as_one_curve(tmp_path: Path) -> None:
    """The split is for resume; the pieces are one cooling history."""
    report = analyse_tg(two_pass_directory(tmp_path), melt_stage=None)

    assert len(report.curves) == 2
    assert [curve.n_points for curve in report.curves] == [11, 21]
    assert report.stages[1] == "08_fine_quench_00, 08_fine_quench_01"


def test_the_fine_pass_is_the_answer_and_the_coarse_one_is_the_evidence(
    tmp_path: Path,
) -> None:
    """Told apart by their temperature step, not by what they were called."""
    report = analyse_tg(two_pass_directory(tmp_path), melt_stage=None)

    assert report.coarse is not None
    assert report.fine is not None
    assert report.coarse.cooling_rate_k_per_ns == pytest.approx(40.0)
    assert report.fine.cooling_rate_k_per_ns == pytest.approx(5.0)
    assert report.temperature_k == pytest.approx(340.0)
    assert report.cooling_rate_k_per_ns == pytest.approx(5.0)
    assert report.resolved


def test_passes_are_found_by_shape_rather_than_by_the_names_they_were_given(
    tmp_path: Path,
) -> None:
    """Nothing has to agree in advance on what a quench stage is called."""
    coarse_t, coarse_d = two_line_curve(transition_k=340.0, n_points=11)
    fine_t, fine_d = two_line_curve(transition_k=340.0, n_points=21)
    write_quenches(
        tmp_path,
        {
            "screen": {
                "temperature_k": list(coarse_t[::-1]),
                "density_g_cm3": list(coarse_d[::-1]),
                "segment_duration_ps": [1000.0] * 11,
            },
            "resolve": {
                "temperature_k": list(fine_t[::-1]),
                "density_g_cm3": list(fine_d[::-1]),
                "segment_duration_ps": [4000.0] * 21,
            },
        },
    )
    report = analyse_tg(tmp_path, melt_stage=None)

    assert report.stages == ("screen", "resolve")
    assert report.coarse is not None
    assert report.fine is not None
    assert report.fine.cooling_rate_k_per_ns == pytest.approx(5.0)


def test_a_run_with_no_resolved_transition_reports_no_temperature(
    tmp_path: Path,
) -> None:
    """A straight line has no break, and the fit says so rather than guessing."""
    temperature, _ = two_line_curve()
    straight = 1.0 / (1.0 + 5.0e-4 * temperature)
    write_quench(tmp_path, temperature[::-1], straight[::-1])
    report = analyse_tg(tmp_path, melt_stage=None)

    assert report.temperature_k is None
    assert not report.resolved
    assert any("corner in noise" in note for note in report.notes)


def test_a_directory_with_no_quench_in_it_is_refused(
    dimer_run_directory: Path,
) -> None:
    """There is nothing here to read a transition off."""
    with pytest.raises(AnalysisError, match="stepped down a ladder"):
        analyse_tg(dimer_run_directory)


def test_several_rates_pooled_from_several_directories_give_one_fit(
    tmp_path: Path,
) -> None:
    """A rate series is likelier to be separate runs than separate stages."""
    directories = []
    for index, (transition_k, hold_ps) in enumerate(
        ((340.0, 20_000.0), (360.0, 2_000.0), (380.0, 200.0))
    ):
        directory = tmp_path / f"run{index}"
        directory.mkdir()
        temperature, density = two_line_curve(transition_k=transition_k)
        write_quench(
            directory,
            temperature[::-1],
            density[::-1],
            stage="08_fine_quench_00",
            segment_duration_ps=[hold_ps] * 21,
        )
        directories.append(directory)

    report = analyse_tg(
        directories[0],
        extra_run_dirs=directories[1:],
        melt_stage=None,
        target_rate_k_per_ns=0.01,
    )

    assert report.log_linear is not None
    assert report.log_linear.n_rates == 3
    assert report.log_linear.sensitivity_k_per_decade == pytest.approx(20.0)
    assert report.log_linear.temperature_k == pytest.approx(300.0)
    assert report.vft is not None


def test_only_the_finest_scans_are_weighed_against_each_other(
    tmp_path: Path,
) -> None:
    """A 40 K screening scan is not the same quality of number as a 20 K one.

    Two fine passes here and one coarse, so the rate fit has exactly two
    points rather than three.
    """
    coarse_t, coarse_d = two_line_curve(transition_k=340.0, n_points=11)
    stages: dict[str, Any] = {
        "06_coarse": {
            "temperature_k": list(coarse_t[::-1]),
            "density_g_cm3": list(coarse_d[::-1]),
            "segment_duration_ps": [1000.0] * 11,
        }
    }
    for transition_k, hold_ps in ((340.0, 4000.0), (360.0, 8000.0)):
        temperature, density = two_line_curve(transition_k=transition_k)
        stages[f"08_fine_{hold_ps:.0f}"] = {
            "temperature_k": list(temperature[::-1]),
            "density_g_cm3": list(density[::-1]),
            "segment_duration_ps": [hold_ps] * 21,
        }
    write_quenches(tmp_path, stages)

    report = analyse_tg(tmp_path, melt_stage=None)

    assert report.log_linear is not None
    assert report.log_linear.n_rates == 2
    assert report.vft is None
    assert any("vft" in note for note in report.notes)


def test_one_quench_alone_has_no_rate_dependence_to_fit(tmp_path: Path) -> None:
    """And that is a silence, not an error."""
    temperature, density = two_line_curve(transition_k=340.0)
    write_quench(tmp_path, temperature[::-1], density[::-1])
    report = analyse_tg(tmp_path, melt_stage=None)

    assert report.coarse is None
    assert report.log_linear is None
    assert report.temperature_k == pytest.approx(340.0)


def test_a_curve_too_short_to_fit_drops_out_without_shifting_the_others(
    tmp_path: Path,
) -> None:
    """Curves and fits stay parallel, or the report pairs the wrong two.

    The short pass sits in the middle here on purpose: zipping the two lists
    back together after one has lost an entry silently pairs every later
    curve with the fit belonging to the one before it.
    """
    long_t, long_d = two_line_curve(transition_k=340.0)
    short_t, short_d = two_line_curve(transition_k=340.0, n_points=4)
    fine_t, fine_d = two_line_curve(transition_k=340.0, n_points=41)
    write_quenches(
        tmp_path,
        {
            "06_coarse": {
                "temperature_k": list(long_t[::-1]),
                "density_g_cm3": list(long_d[::-1]),
                "segment_duration_ps": [1000.0] * 21,
            },
            "07_stub": {
                "temperature_k": list(short_t[::-1]),
                "density_g_cm3": list(short_d[::-1]),
                "segment_duration_ps": [1000.0] * 4,
            },
            "08_fine": {
                "temperature_k": list(fine_t[::-1]),
                "density_g_cm3": list(fine_d[::-1]),
                "segment_duration_ps": [4000.0] * 41,
            },
        },
    )
    report = analyse_tg(tmp_path, melt_stage=None)

    assert report.stages == ("06_coarse", "08_fine")
    assert len(report.curves) == len(report.transitions) == 2
    assert report.curves[1].temperature_step_k == pytest.approx(10.0)
    assert report.fine is report.transitions[1]
    assert any("07_stub" in note for note in report.notes)


# --------------------------------------------------------------------------
# Writing the report
# --------------------------------------------------------------------------


def test_nothing_is_written_until_the_report_is(tmp_path: Path) -> None:
    """The discipline plots.py keeps, one layer up: reading writes nothing."""
    directory = two_pass_directory(tmp_path)
    before = sorted(path.name for path in directory.iterdir())

    analyse_tg(directory, melt_stage=None)

    assert sorted(path.name for path in directory.iterdir()) == before


def test_the_report_writes_one_record_and_a_figure_per_pass(
    tmp_path: Path,
) -> None:
    """Two quench figures and a rate figure, beside the machine-readable one.

    And the manifest - the resume ledger of a three-day run - is left alone.
    """
    directory = two_pass_directory(tmp_path)
    manifest = (directory / "manifest.json").read_bytes()
    files = write_tg_report(analyse_tg(directory, melt_stage=None))

    assert Path(files.json).name == "tg.json"
    assert Path(files.json).parent.name == "analysis"
    assert len(files.figures) == 2
    assert all(Path(path).is_file() for path in files.figures)
    assert (directory / "manifest.json").read_bytes() == manifest


def test_the_record_carries_the_expansivities_and_what_wrote_it(
    tmp_path: Path,
) -> None:
    """They are properties, which a record built with asdict would drop."""
    report = analyse_tg(two_pass_directory(tmp_path), melt_stage=None)
    record = json.loads(Path(write_tg_report(report, figures=False).json).read_text())

    assert record["fine"]["melt_expansivity_per_k"] == pytest.approx(8.0e-4)
    assert record["fine"]["glass_expansivity_per_k"] == pytest.approx(2.0e-4)
    assert record["fine"]["expansivity_ordered"] is True
    assert record["openmmpolymer"]
    assert record["stages"] == list(report.stages)


def test_the_equilibration_figure_is_drawn_when_the_melt_was_checked(
    tmp_path: Path,
) -> None:
    """The volume series the verdict rested on, with its settling point on it."""
    temperature, density = two_line_curve(transition_k=340.0)
    write_quenches(
        tmp_path,
        {
            "05_npt": {
                "temperature_k": [650.0],
                "density_g_cm3": [0.85],
                "total_ps": 2000.0,
            },
            "06_quench": {
                "temperature_k": list(temperature[::-1]),
                "density_g_cm3": list(density[::-1]),
                "segment_duration_ps": [1000.0] * 21,
            },
        },
    )
    report = analyse_tg(tmp_path, melt_stage="05_npt")

    assert report.melt is not None
    assert report.melt.volume is not None
    assert report.melt.displacement is None
    assert not report.melt.equilibrated

    files = write_tg_report(report)
    assert any(Path(path).name.startswith("equilibration") for path in files.figures)
