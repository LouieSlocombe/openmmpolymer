"""Tests for what a run's own numbers say about whether it settled."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from openmmpolymer.reporters import CSV_COLUMNS
from openmmpolymer.timeseries import (
    CSV_FIELDS,
    DSC_COOLING_RATE_K_PER_NS,
    MAX_EXTRAPOLATION_DECADES,
    GlassTransition,
    cooling_rate_extrapolation,
    equilibration,
    glass_transition,
    quench_curve,
    quench_stages,
    read_state_data,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    state_data_csv,
    transition_at,
    two_line_curve,
    write_quench,
    write_quenches,
)


def test_the_column_map_covers_every_column_a_stage_writes() -> None:
    """The reporter and the reader have to agree, and they are written apart."""
    assert len(CSV_FIELDS) == len(CSV_COLUMNS)


def test_a_stage_csv_reads_back_column_by_column(tmp_path: Path) -> None:
    """The field names genfromtxt derives from OpenMM's header are not obvious -
    ``Density (g/mL)`` becomes ``Density_gmL`` - so they are pinned here."""
    path = tmp_path / "02_nvt.csv"
    path.write_text(
        state_data_csv(
            [
                [0, 0.0, -100.0, 10.0, -90.0, 300.0, 13.8, 0.85],
                [10, 0.02, -101.0, 11.0, -90.0, 301.0, 13.7, 0.86],
            ]
        )
    )
    data = read_state_data(path, stage="02_nvt")
    assert data.stage == "02_nvt"
    assert data.n_rows == 2
    assert data.time_ps == pytest.approx([0.0, 0.02])
    assert data.temperature_k == pytest.approx([300.0, 301.0])
    assert data.density_g_cm3 == pytest.approx([0.85, 0.86])
    assert data.volume_nm3 == pytest.approx([13.8, 13.7])
    assert data.duration_ps == pytest.approx(0.02)


def test_a_one_row_csv_still_reads_back_as_arrays(tmp_path: Path) -> None:
    """genfromtxt hands back a zero-dimensional record for a single row, whose
    fields are scalars rather than arrays, and everything downstream indexes."""
    path = tmp_path / "02_nvt.csv"
    path.write_text(state_data_csv([[0, 0.0, -1.0, 1.0, 0.0, 300.0, 13.8, 0.85]]))
    data = read_state_data(path)
    assert data.n_rows == 1
    assert data.temperature_k.shape == (1,)
    assert data.duration_ps == 0.0


def test_a_missing_csv_is_named_rather_than_crashing(tmp_path: Path) -> None:
    """A stage that died before its first report leaves no CSV at all."""
    with pytest.raises(AnalysisError, match="No state-data CSV"):
        read_state_data(tmp_path / "nothing.csv")


def test_the_human_readable_log_is_refused_as_a_substitute(tmp_path: Path) -> None:
    """A stage writes two files and only one is numbers; the other renders
    progress as ``20.0%``, so reading it would give silent NaNs."""
    path = tmp_path / "02_nvt.log"
    path.write_text('#"Progress (%)","Step","Time (ps)"\n20.0%,100,0.2\n')
    with pytest.raises(AnalysisError, match="missing the column"):
        read_state_data(path)


def test_a_header_with_no_rows_says_the_stage_wrote_nothing(tmp_path: Path) -> None:
    """The reporter writes its header immediately and its first row later."""
    path = tmp_path / "02_nvt.csv"
    path.write_text(state_data_csv([]))
    with pytest.raises(AnalysisError, match="no rows"):
        read_state_data(path)


def test_an_exponential_approach_settles_near_its_plateau() -> None:
    """The case every equilibration is: a quantity relaxing to a value, where
    the answer is the value and the question is how much to throw away."""
    rng = np.random.default_rng(7)
    times = np.arange(0.0, 400.0, 0.5)
    values = 1.0 - 0.5 * np.exp(-times / 20.0) + rng.normal(0.0, 0.002, times.size)
    settled = equilibration(times, values)
    assert settled.equilibrated
    assert settled.start_ps > 60.0
    assert float(settled.window(values).mean()) == pytest.approx(1.0, abs=5.0e-3)


def test_a_series_still_drifting_is_not_reported_as_settled() -> None:
    """The failure this is for: a density heading somewhere, read as a mean."""
    times = np.arange(0.0, 400.0, 0.5)
    values = 1.0 + 0.5 * times / times[-1]
    settled = equilibration(times, values)
    assert not settled.equilibrated
    assert settled.relative_drift > 0.1


def test_pure_noise_settles_immediately_and_keeps_everything() -> None:
    """Nothing to discard, so discarding any of it would be throwing away
    evidence for no reason."""
    rng = np.random.default_rng(3)
    times = np.arange(0.0, 200.0, 0.5)
    settled = equilibration(times, rng.normal(0.85, 0.01, times.size))
    assert settled.equilibrated
    assert settled.start_index == 0
    assert settled.n_samples == times.size


def test_a_series_that_never_moves_is_handled_rather_than_dividing_by_zero() -> None:
    """A constant has no variance to make a drift relative to, and a
    perfectly-constrained volume in NVT really is constant."""
    times = np.arange(0.0, 100.0, 0.5)
    settled = equilibration(times, np.full(times.size, 0.9))
    assert settled.equilibrated
    assert settled.correlation_time_ps == 0.0
    assert settled.relative_drift == pytest.approx(0.0, abs=1.0e-9)


def test_a_correlated_series_is_worth_fewer_samples_than_it_has_rows() -> None:
    """Reporting often does not buy independent evidence, and quoting a
    standard error over the row count would understate it."""
    rng = np.random.default_rng(5)
    values = np.zeros(2000)
    for index in range(1, values.size):
        values[index] = 0.95 * values[index - 1] + rng.normal(0.0, 0.1)
    settled = equilibration(np.arange(values.size, dtype=np.float64), values + 10.0)
    assert settled.n_independent_samples < settled.n_samples / 5
    assert settled.correlation_time_ps > 1.0


def test_two_series_of_different_lengths_are_refused() -> None:
    """They have to be the same measurement, and a mismatch is a mistake."""
    with pytest.raises(AnalysisError, match="the same series"):
        equilibration(np.arange(5.0), np.arange(4.0))


def test_too_few_rows_to_say_anything_says_so() -> None:
    """Two points fit a line exactly and say nothing about settling."""
    with pytest.raises(AnalysisError, match="too few"):
        equilibration(np.array([0.0, 1.0]), np.array([1.0, 2.0]))


def test_a_quench_curve_comes_back_ordered_by_temperature(tmp_path: Path) -> None:
    """A quench records its steps going down, and a curve is read going up."""
    temperature, density = two_line_curve()
    write_quench(tmp_path, temperature[::-1], density[::-1])
    curve = quench_curve(tmp_path)
    assert curve.n_points == 21
    assert curve.temperature_k[0] < curve.temperature_k[-1]
    assert curve.specific_volume_cm3_g == pytest.approx(1.0 / curve.density_g_cm3)


def test_the_cooling_rate_is_recovered_from_the_stage_csv(tmp_path: Path) -> None:
    """Neither StageResult nor SystemSpec records a duration, so the CSV's last
    time is the only thing that knows how long each temperature was held."""
    temperature, density = two_line_curve()
    write_quench(tmp_path, temperature[::-1], density[::-1])
    curve = quench_curve(tmp_path)
    assert curve.hold_ps == pytest.approx(4200.0 / 21)
    assert curve.cooling_rate_k_per_ns == pytest.approx(100.0, rel=1e-6)


def test_without_a_csv_the_cooling_rate_is_none_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """An unknown rate reported as None is usable; an invented one is not."""
    temperature, density = two_line_curve()
    write_quench(tmp_path, temperature[::-1], density[::-1], with_csv=False)
    curve = quench_curve(tmp_path)
    assert curve.hold_ps is None
    assert curve.cooling_rate_k_per_ns is None


def test_a_stage_that_was_not_a_quench_is_refused(tmp_path: Path) -> None:
    """Only a quench records a density per temperature."""
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "equilibrate",
                "seed": 1,
                "versions": {},
                "system": {},
                "stages": {"05_npt": {"name": "05_npt", "samples": {}}},
                "chains": None,
            }
        )
    )
    with pytest.raises(AnalysisError, match="not a quench"):
        quench_curve(tmp_path, "05_npt")


def test_a_missing_manifest_and_a_missing_stage_are_both_named(
    tmp_path: Path,
) -> None:
    """Two different mistakes with two different fixes."""
    with pytest.raises(AnalysisError, match="No manifest"):
        quench_curve(tmp_path)
    temperature, density = two_line_curve()
    write_quench(tmp_path, temperature, density)
    with pytest.raises(AnalysisError, match="has no stage"):
        quench_curve(tmp_path, "99_nope")


def test_a_recorded_density_of_zero_is_refused(tmp_path: Path) -> None:
    """Its specific volume is not defined, and dividing would give an inf that
    then fits a perfectly respectable-looking straight line."""
    temperature, density = two_line_curve()
    density = density.copy()
    density[0] = 0.0
    write_quench(tmp_path, temperature, density)
    with pytest.raises(AnalysisError, match="not defined"):
        quench_curve(tmp_path)


def test_two_exact_lines_give_back_the_transition_they_were_built_from(
    tmp_path: Path,
) -> None:
    """The measurement this is for, on data where the answer is known exactly."""
    temperature, density = two_line_curve(transition_k=350.0)
    write_quench(tmp_path, temperature[::-1], density[::-1])
    fitted = glass_transition(quench_curve(tmp_path))
    assert fitted.resolved
    assert fitted.temperature_k == pytest.approx(350.0, abs=0.5)
    assert fitted.melt_expansion_per_k == pytest.approx(8.0e-4, rel=1e-3)
    assert fitted.glass_expansion_per_k == pytest.approx(2.0e-4, rel=1e-3)
    assert fitted.residual_cm3_g < 1.0e-9
    assert fitted.n_points_glass + fitted.n_points_melt == 21


def test_the_cooling_rate_travels_with_the_transition(tmp_path: Path) -> None:
    """The temperature is not comparable with an experiment without it, so it
    is carried onto the result rather than left behind on the curve."""
    temperature, density = two_line_curve()
    write_quench(tmp_path, temperature[::-1], density[::-1])
    curve = quench_curve(tmp_path)
    fitted = glass_transition(curve)
    assert fitted.cooling_rate_k_per_ns == curve.cooling_rate_k_per_ns


def test_the_crossing_volume_puts_both_branches_through_the_transition(
    tmp_path: Path,
) -> None:
    """It is what lets a plot draw the two fitted lines, and it should be the
    volume the curve actually has there."""
    temperature, density = two_line_curve(transition_k=350.0)
    write_quench(tmp_path, temperature[::-1], density[::-1])
    fitted = glass_transition(quench_curve(tmp_path))
    assert fitted.specific_volume_cm3_g == pytest.approx(1.0, abs=1.0e-3)


def test_a_straight_line_with_no_break_is_not_reported_as_a_transition(
    tmp_path: Path,
) -> None:
    """A melt that never vitrified has no corner, and the best two-line fit of
    a straight line is still a straight line."""
    temperature = np.linspace(200.0, 600.0, 21)
    write_quench(tmp_path, temperature, 1.0 / (1.0 + 5.0e-4 * temperature))
    fitted = glass_transition(quench_curve(tmp_path))
    assert not fitted.resolved


def test_a_curve_too_short_for_two_branches_is_refused(tmp_path: Path) -> None:
    """Four points a side is already few; two is a line through two points."""
    temperature, density = two_line_curve(n_points=6)
    write_quench(tmp_path, temperature, density)
    with pytest.raises(AnalysisError, match="two branches"):
        glass_transition(quench_curve(tmp_path), min_points_per_branch=4)


def test_a_transition_found_at_the_very_end_is_not_resolved(tmp_path: Path) -> None:
    """A break in the last few points is a corner in noise, not a transition."""
    temperature, density = two_line_curve(transition_k=215.0, n_points=21)
    write_quench(tmp_path, temperature, density)
    fitted = glass_transition(quench_curve(tmp_path), min_points_per_branch=4)
    assert not fitted.resolved


def test_a_curve_whose_branches_are_the_wrong_way_round_is_not_resolved(
    tmp_path: Path,
) -> None:
    """A glass expands less than its melt, so the flatter branch has to be the
    cold one; the other way round is a fit that found something else."""
    temperature, density = two_line_curve(
        transition_k=350.0, glass_slope=8.0e-4, melt_slope=2.0e-4
    )
    write_quench(tmp_path, temperature, density)
    assert not glass_transition(quench_curve(tmp_path)).resolved


def test_the_correlation_time_is_reported_in_picoseconds(tmp_path: Path) -> None:
    """Every other time in this package is, and a correlation time in rows
    would be a number that changed when the reporting interval did."""
    del tmp_path
    rng = np.random.default_rng(9)
    values = np.zeros(4000)
    for index in range(1, values.size):
        values[index] = 0.98 * values[index - 1] + rng.normal(0.0, 0.1)
    spacing = 0.5
    times = np.arange(values.size, dtype=np.float64) * spacing
    coarse = equilibration(times, values + 5.0)
    fine = equilibration(times * 2.0, values + 5.0)
    assert fine.correlation_time_ps == pytest.approx(
        2.0 * coarse.correlation_time_ps, rel=1.0e-6
    )
    assert math.isfinite(coarse.correlation_time_ps)


def test_a_series_whose_candidate_windows_run_out_still_settles() -> None:
    """The coarse grid of candidate starts can propose a window with too few
    rows left in it, which is skipped rather than fitted."""
    times = np.arange(6.0)
    settled = equilibration(times, np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert settled.n_samples >= 3


def test_two_identical_branches_are_not_a_transition(tmp_path: Path) -> None:
    """Their intersection is undefined - the lines are parallel - so there is
    no temperature to report and the fit says so."""
    temperature = np.linspace(200.0, 600.0, 21)
    write_quench(tmp_path, temperature, np.full(21, 1.0))
    fitted = glass_transition(quench_curve(tmp_path))
    assert not fitted.resolved


def test_a_series_that_straddles_zero_is_scaled_by_its_spread(tmp_path: Path) -> None:
    """A total energy averaging zero has no meaningful relative drift against
    its mean, and dividing by it would give an enormous number or a NaN."""
    del tmp_path
    from openmmpolymer.timeseries import _scale_of

    assert _scale_of(np.array([-1.0, 1.0, -1.0, 1.0])) == pytest.approx(1.0)
    assert _scale_of(np.zeros(4)) > 0.0


def test_a_single_row_has_no_spacing_and_no_drift() -> None:
    """Both are differences between rows, and there is only one."""
    from openmmpolymer.timeseries import _relative_drift, _spacing_ps

    assert _spacing_ps(np.array([1.0])) == 0.0
    assert _relative_drift(np.array([1.0]), np.array([2.0]), 1.0) == 0.0


def test_a_window_spanning_no_time_has_no_drift() -> None:
    """Every row at the same time cannot drift, whatever the values do."""
    from openmmpolymer.timeseries import _relative_drift

    assert _relative_drift(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), 1.0) == 0.0


def test_a_line_through_two_points_is_fitted_without_residuals() -> None:
    """numpy.linalg.lstsq returns an empty residual array for an exactly
    determined fit, so the sum has to be worked out rather than read off."""
    from openmmpolymer.timeseries import _fit_line

    (slope, intercept), total = _fit_line(np.array([0.0, 1.0]), np.array([1.0, 3.0]))
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert total == pytest.approx(0.0, abs=1e-20)


def test_a_cooling_rate_is_none_when_the_recorded_csv_has_gone(
    tmp_path: Path,
) -> None:
    """A run directory moved without its CSVs still has its manifest, and an
    invented rate would be worse than none."""
    temperature, density = two_line_curve()
    write_quench(tmp_path, temperature, density)
    (tmp_path / "06_quench.csv").unlink()
    assert quench_curve(tmp_path).cooling_rate_k_per_ns is None


def test_a_cooling_rate_is_none_when_the_recorded_csv_is_unreadable(
    tmp_path: Path,
) -> None:
    """A truncated CSV from a killed run is not a reason to refuse the curve,
    which comes from the manifest and is fine."""
    temperature, density = two_line_curve()
    write_quench(tmp_path, temperature, density)
    (tmp_path / "06_quench.csv").write_text("not a csv at all\n")
    curve = quench_curve(tmp_path)
    assert curve.cooling_rate_k_per_ns is None
    assert curve.n_points == 21


def test_a_single_temperature_has_no_cooling_rate() -> None:
    """A rate is a difference between steps, and there is only one."""
    from openmmpolymer.timeseries import _cooling_rate

    assert _cooling_rate(np.array([300.0]), 200.0) is None
    assert _cooling_rate(np.array([300.0, 280.0]), 0.0) is None


# --------------------------------------------------------------------------
# Thermal expansivity
# --------------------------------------------------------------------------


def test_the_expansivities_are_the_slopes_over_the_crossing_volume(
    tmp_path: Path,
) -> None:
    """The slopes are cm3/g/K and depend on the density; aV is per kelvin.

    The knot sits on a grid point, so the fit is exact and the crossing volume
    is exactly the 1.0 the curve was built around - which makes the two
    coefficients exactly the two slopes.
    """
    temperature, density = two_line_curve(transition_k=340.0)
    write_quench(tmp_path, temperature, density)
    fit = glass_transition(quench_curve(tmp_path))

    assert fit.specific_volume_cm3_g == pytest.approx(1.0)
    assert fit.melt_expansivity_per_k == pytest.approx(8.0e-4)
    assert fit.glass_expansivity_per_k == pytest.approx(2.0e-4)


def test_a_resolved_transition_always_expands_faster_as_a_melt(
    tmp_path: Path,
) -> None:
    """Which is why it is reported rather than checked a second time.

    Both coefficients divide the same crossing volume, so their ordering is
    the ordering of the slopes, and that is already what resolved requires.
    """
    temperature, density = two_line_curve(transition_k=340.0)
    write_quench(tmp_path, temperature, density)
    fit = glass_transition(quench_curve(tmp_path))

    assert fit.resolved
    assert fit.melt_expansivity_per_k > fit.glass_expansivity_per_k


def test_an_expansivity_with_no_volume_to_divide_by_is_not_a_number() -> None:
    """Not zero, which would read as a material that does not expand."""
    fit = transition_at(10.0, 350.0, specific_volume_cm3_g=0.0)

    assert math.isnan(fit.melt_expansivity_per_k)
    assert math.isnan(fit.glass_expansivity_per_k)


# --------------------------------------------------------------------------
# Finding and pooling quenches
# --------------------------------------------------------------------------


def test_quench_stages_finds_the_ladders_and_skips_everything_else(
    tmp_path: Path,
) -> None:
    """Every stage records a temperature per segment; only a quench descends."""
    temperature, density = two_line_curve(n_points=21)
    write_quenches(
        tmp_path,
        {
            "03_compress": {
                "temperature_k": [600.0] * 7,
                "density_g_cm3": [0.9] * 7,
            },
            "04_anneal": {
                "temperature_k": [300.0, 600.0, 300.0, 600.0],
                "density_g_cm3": [1.0, 0.9, 1.0, 0.9],
            },
            "05_npt": {"temperature_k": [450.0], "density_g_cm3": [0.95]},
            "06_quench": {
                "temperature_k": list(temperature[::-1]),
                "density_g_cm3": list(density[::-1]),
            },
        },
    )
    assert quench_stages(tmp_path) == ("06_quench",)


def test_a_run_with_no_quench_in_it_says_which_stages_there_are(
    tmp_path: Path,
) -> None:
    """Naming what is there is the difference between a hint and a dead end."""
    write_quenches(
        tmp_path,
        {"05_npt": {"temperature_k": [450.0], "density_g_cm3": [0.95]}},
    )
    with pytest.raises(AnalysisError, match="05_npt"):
        quench_stages(tmp_path)


def test_a_curve_can_be_assembled_from_several_stages(tmp_path: Path) -> None:
    """A long ladder is split into stages for resume; the curve is still one."""
    temperature, density = two_line_curve(transition_k=340.0)
    descending_t, descending_d = list(temperature[::-1]), list(density[::-1])
    write_quenches(
        tmp_path,
        {
            "06_quench_00": {
                "temperature_k": descending_t[:11],
                "density_g_cm3": descending_d[:11],
                "segment_duration_ps": [200.0] * 11,
            },
            "06_quench_01": {
                "temperature_k": descending_t[11:],
                "density_g_cm3": descending_d[11:],
                "segment_duration_ps": [200.0] * 10,
            },
        },
    )
    curve = quench_curve(tmp_path, ("06_quench_00", "06_quench_01"))

    assert curve.n_points == 21
    assert curve.temperature_k[0] < curve.temperature_k[-1]
    assert curve.hold_ps == pytest.approx(200.0)
    assert glass_transition(curve).temperature_k == pytest.approx(340.0)


def test_a_recorded_hold_is_preferred_to_dividing_the_csv(tmp_path: Path) -> None:
    """The CSV division assumes one uniform hold over one whole stage.

    A ladder split across stages, or resumed part-way through, breaks that
    assumption quietly - the rate comes out wrong rather than unknown - so a
    stage that recorded its own durations is believed instead.
    """
    temperature, density = two_line_curve()
    write_quench(
        tmp_path,
        temperature,
        density,
        total_ps=4200.0,
        segment_duration_ps=[500.0] * 21,
    )
    curve = quench_curve(tmp_path)

    assert curve.hold_ps == pytest.approx(500.0)
    assert curve.cooling_rate_k_per_ns == pytest.approx(40.0)


def test_a_multi_stage_curve_without_recorded_holds_has_no_rate(
    tmp_path: Path,
) -> None:
    """Across two stages the CSV division is a different wrong number each."""
    temperature, density = two_line_curve()
    write_quenches(
        tmp_path,
        {
            "a": {
                "temperature_k": list(temperature[:11][::-1]),
                "density_g_cm3": list(density[:11][::-1]),
            },
            "b": {
                "temperature_k": list(temperature[11:][::-1]),
                "density_g_cm3": list(density[11:][::-1]),
            },
        },
    )
    assert quench_curve(tmp_path, ("a", "b")).cooling_rate_k_per_ns is None


def test_a_stage_that_held_one_temperature_has_no_cooling_rate(
    tmp_path: Path,
) -> None:
    """Zero would read as cooling infinitely slowly, the opposite of the truth."""
    write_quench(tmp_path, [450.0], [0.95], segment_duration_ps=[200.0])
    assert quench_curve(tmp_path).cooling_rate_k_per_ns is None


def test_an_empty_state_data_file_is_refused_rather_than_crashing(
    tmp_path: Path,
) -> None:
    """A stage killed before its first report leaves the file and nothing in it."""
    empty = tmp_path / "05_npt.csv"
    empty.write_text("")
    with pytest.raises(AnalysisError, match="empty"):
        read_state_data(empty)


# --------------------------------------------------------------------------
# Cooling-rate extrapolation
# --------------------------------------------------------------------------


def log_linear_rates(tmp_path: Path) -> tuple[GlassTransition, ...]:
    """Three quenches whose transitions sit exactly on a line in log rate.

    Knots on grid points, so each fit is exact; total_ps chosen so the
    recovered rates are 1, 10 and 100 K/ns. That makes Tg = 340 + 20 log10(R),
    and every number the fit reports is a round one.
    """
    stages = {}
    for transition_k, total_ps in (
        (340.0, 420000.0),
        (360.0, 42000.0),
        (380.0, 4200.0),
    ):
        temperature, density = two_line_curve(transition_k=transition_k)
        stages[f"q{transition_k:.0f}"] = {
            "temperature_k": list(temperature[::-1]),
            "density_g_cm3": list(density[::-1]),
            "total_ps": total_ps,
        }
    write_quenches(tmp_path, stages)
    return tuple(
        glass_transition(quench_curve(tmp_path, name)) for name in sorted(stages)
    )


def test_a_quench_ladder_recovers_the_rate_it_was_built_to_have(
    tmp_path: Path,
) -> None:
    """The fixture's whole job, pinned so the fit tests rest on something."""
    fits = log_linear_rates(tmp_path)
    assert [fit.cooling_rate_k_per_ns for fit in fits] == [
        pytest.approx(1.0),
        pytest.approx(10.0),
        pytest.approx(100.0),
    ]
    assert [fit.temperature_k for fit in fits] == [
        pytest.approx(340.0),
        pytest.approx(360.0),
        pytest.approx(380.0),
    ]


def test_the_transition_extrapolates_exactly_along_a_line_in_log_rate(
    tmp_path: Path,
) -> None:
    """Two decades below the data, 340 + 20 * (-2) and nothing else."""
    fit = cooling_rate_extrapolation(
        log_linear_rates(tmp_path), target_rate_k_per_ns=0.01
    )

    assert fit.temperature_k == pytest.approx(300.0)
    assert fit.sensitivity_k_per_decade == pytest.approx(20.0)
    assert fit.extrapolation_decades == pytest.approx(2.0)
    assert fit.residual_k == pytest.approx(0.0, abs=1.0e-9)
    assert fit.resolved


def test_a_target_between_the_measured_rates_is_not_an_extrapolation(
    tmp_path: Path,
) -> None:
    """Zero decades, and the only case where resolved means much."""
    fit = cooling_rate_extrapolation(
        log_linear_rates(tmp_path), target_rate_k_per_ns=30.0
    )

    assert fit.extrapolation_decades == 0.0
    assert fit.resolved


def test_an_extrapolation_to_an_experimental_rate_is_never_resolved(
    tmp_path: Path,
) -> None:
    """Ten decades. The number is still reported; the claim is not made."""
    fit = cooling_rate_extrapolation(log_linear_rates(tmp_path))

    assert fit.target_rate_k_per_ns == pytest.approx(DSC_COOLING_RATE_K_PER_NS)
    assert fit.extrapolation_decades > MAX_EXTRAPOLATION_DECADES
    assert math.isfinite(fit.temperature_k)
    assert not fit.resolved


def test_a_vft_fit_recovers_the_parameters_it_was_built_from() -> None:
    """Three rates exactly determine it, and the search has to find them."""
    t0, b_k, r0 = 300.0, 400.0, 1.0e4
    rates = (2.0, 5.0, 10.0)
    fits = [transition_at(r, t0 + b_k / math.log(r0 / r)) for r in rates]

    fit = cooling_rate_extrapolation(fits, form="vft")

    assert fit.parameters["t0_k"] == pytest.approx(t0, rel=1.0e-4)
    assert fit.parameters["b_k"] == pytest.approx(b_k, rel=1.0e-4)
    assert math.exp(fit.parameters["ln_r0"]) == pytest.approx(r0, rel=1.0e-3)
    expected = t0 + b_k / math.log(r0 / DSC_COOLING_RATE_K_PER_NS)
    assert fit.temperature_k == pytest.approx(expected, rel=1.0e-4)


def test_the_two_forms_disagree_by_a_hundred_kelvin_over_ten_decades() -> None:
    """Which is the whole argument for offering both.

    A quench overestimates an experimental transition by 20 to 50 K, so on
    this melt the honest answer is near 313 K. The straight line in log rate
    runs away to 190 K; VFT, which has a finite limit, does not.
    """
    t0, b_k, r0 = 300.0, 400.0, 1.0e4
    fits = [transition_at(r, t0 + b_k / math.log(r0 / r)) for r in (2.0, 5.0, 10.0)]

    straight = cooling_rate_extrapolation(fits, form="log_linear")
    curved = cooling_rate_extrapolation(fits, form="vft")

    assert straight.temperature_k == pytest.approx(190.0, abs=2.0)
    assert curved.temperature_k == pytest.approx(313.0, abs=2.0)
    assert curved.temperature_k - straight.temperature_k > 100.0


def test_the_wlf_constants_are_the_vft_ones_rewritten() -> None:
    """WLF and VFT are one relation, so one fit answers for both."""
    t0, b_k, r0 = 300.0, 400.0, 1.0e4
    fits = [transition_at(r, t0 + b_k / math.log(r0 / r)) for r in (2.0, 5.0, 10.0)]
    fit = cooling_rate_extrapolation(fits, form="vft")

    reference = 350.0
    c1, c2 = fit.wlf_constants(reference)

    assert c2 == pytest.approx(reference - t0, rel=1.0e-3)
    assert c1 == pytest.approx(b_k / (math.log(10.0) * c2), rel=1.0e-3)


def test_wlf_constants_need_a_reference_above_the_fitted_floor() -> None:
    """Below T0 they diverge, and a divergent constant is not a constant."""
    t0, b_k, r0 = 300.0, 400.0, 1.0e4
    fits = [transition_at(r, t0 + b_k / math.log(r0 / r)) for r in (2.0, 5.0, 10.0)]
    fit = cooling_rate_extrapolation(fits, form="vft")

    with pytest.raises(AnalysisError, match="diverge"):
        fit.wlf_constants(250.0)


def test_wlf_constants_come_only_from_a_vft_fit() -> None:
    """A straight line in log rate has no T0 to reference them to."""
    fits = [transition_at(r, 340.0 + 20.0 * math.log10(r)) for r in (1.0, 10.0, 100.0)]
    fit = cooling_rate_extrapolation(fits, form="log_linear")

    with pytest.raises(AnalysisError, match="VFT"):
        fit.wlf_constants(350.0)


def test_a_vft_search_that_degenerates_to_a_straight_line_says_so() -> None:
    """Exactly log-linear data sends R0 to infinity; that is not a fit."""
    fits = [transition_at(r, 340.0 + 20.0 * math.log10(r)) for r in (1.0, 10.0, 100.0)]

    assert not cooling_rate_extrapolation(fits, form="vft").resolved


def test_a_transition_that_rises_as_cooling_slows_is_not_resolved() -> None:
    """Cooling more slowly gives the melt longer to keep up, never less."""
    fits = [transition_at(r, 400.0 - 20.0 * math.log10(r)) for r in (1.0, 10.0, 100.0)]

    assert not cooling_rate_extrapolation(fits).resolved


def test_a_fit_over_transitions_that_did_not_resolve_does_not_resolve() -> None:
    """Fitting a line through three corners found in noise finds a fourth."""
    fits = [
        transition_at(r, 340.0 + 20.0 * math.log10(r), resolved=r != 10.0)
        for r in (1.0, 10.0, 100.0)
    ]

    assert not cooling_rate_extrapolation(fits, target_rate_k_per_ns=30.0).resolved


def test_exactly_as_many_rates_as_parameters_does_not_resolve() -> None:
    """The residual is then zero by construction and says nothing."""
    fits = [transition_at(r, 340.0 + 20.0 * math.log10(r)) for r in (1.0, 100.0)]
    fit = cooling_rate_extrapolation(fits, target_rate_k_per_ns=30.0)

    assert fit.n_rates == fit.n_parameters
    assert fit.residual_k == pytest.approx(0.0, abs=1.0e-9)
    assert not fit.resolved


def test_vft_needs_more_rates_than_it_has_parameters() -> None:
    """Two points cannot determine three numbers, and it says which."""
    fits = [transition_at(r, 340.0 + 20.0 * math.log10(r)) for r in (1.0, 100.0)]

    with pytest.raises(AnalysisError, match="3 parameters"):
        cooling_rate_extrapolation(fits, form="vft")


def test_one_rate_cannot_show_a_rate_dependence() -> None:
    """It takes two measurements to see something move."""
    with pytest.raises(AnalysisError, match="rate dependence"):
        cooling_rate_extrapolation([transition_at(10.0, 350.0)])


def test_two_quenches_at_the_same_rate_are_refused() -> None:
    """A degenerate design matrix, and nothing to learn from it either."""
    fits = [transition_at(10.0, 350.0), transition_at(10.0, 352.0)]

    with pytest.raises(AnalysisError, match="same cooling rate"):
        cooling_rate_extrapolation(fits)


def test_a_transition_with_no_recorded_rate_is_refused() -> None:
    """There is nothing to plot it against, and it says which one."""
    fits = [transition_at(10.0, 350.0), transition_at(None, 340.0)]

    with pytest.raises(AnalysisError, match="index 1"):
        cooling_rate_extrapolation(fits)


def test_an_unknown_extrapolation_form_is_refused_at_the_call_site() -> None:
    """An argument, so a ValueError, like every other choice in the package."""
    fits = [transition_at(r, 340.0 + 20.0 * math.log10(r)) for r in (1.0, 10.0)]

    with pytest.raises(ValueError, match="form="):
        cooling_rate_extrapolation(fits, form="arrhenius")
