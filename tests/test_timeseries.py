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
    equilibration,
    glass_transition,
    quench_curve,
    read_state_data,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import state_data_csv


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


def write_quench(
    directory: Path,
    temperature_k: np.ndarray,
    density_g_cm3: np.ndarray,
    *,
    with_csv: bool = True,
    stage: str = "06_quench",
) -> Path:
    """Write a manifest holding a quench's samples, and optionally its CSV."""
    csv = None
    if with_csv:
        csv = str(directory / f"{stage}.csv")
        rows = [
            [index * 1000, index * 42.0, -1.0, 1.0, 0.0, 300.0, 13.8, 0.9]
            for index in range(1, 101)
        ]
        Path(csv).write_text(state_data_csv(rows))
    payload = {
        "protocol": "melt-quench",
        "seed": 1,
        "versions": {},
        "system": {},
        "stages": {
            stage: {
                "name": stage,
                "csv": csv,
                "samples": {
                    "segment_temperature_k": list(temperature_k),
                    "segment_density_g_cm3": list(density_g_cm3),
                },
            }
        },
        "chains": None,
        "box": None,
    }
    (directory / "manifest.json").write_text(json.dumps(payload))
    return directory


def two_line_curve(
    transition_k: float = 350.0,
    n_points: int = 21,
    glass_slope: float = 2.0e-4,
    melt_slope: float = 8.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """A specific-volume curve made of two exact straight lines."""
    temperature = np.linspace(200.0, 600.0, n_points)
    volume = np.where(
        temperature <= transition_k,
        1.0 + glass_slope * (temperature - transition_k),
        1.0 + melt_slope * (temperature - transition_k),
    )
    return temperature, 1.0 / volume


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
