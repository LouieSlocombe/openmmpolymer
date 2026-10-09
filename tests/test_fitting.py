"""Tests for the numerics the analyses share."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openmmpolymer._fitting import (
    extrapolation_decades,
    finite_or_none,
    fit_line,
    group_nearby_rates,
    median_spacing,
    relative_span,
    rms,
    separable_fit,
    slope_error,
    standard_error,
    standard_error_from_moments,
    statistical_inefficiency,
)

from .helpers import ar1


def test_a_line_through_two_points_is_fitted_without_residuals() -> None:
    """numpy.linalg.lstsq returns an empty residual array for an exactly
    determined fit, so the sum has to be worked out rather than read off."""
    (slope, intercept), total = fit_line(np.array([0.0, 1.0]), np.array([1.0, 3.0]))
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert total == pytest.approx(0.0, abs=1e-20)


@pytest.mark.parametrize(
    ("offset", "span"),
    [(1e5, 100.0), (1e6, 0.2), (1e7, 1.0), (1e5, 1e-6), (0.0, 1e-20)],
)
def test_a_line_keeps_its_slope_for_narrow_or_offset_coordinates(
    offset: float, span: float
) -> None:
    x = offset + np.linspace(0.0, span, 12)
    y = 3.0 * ((x - offset) / span) + 5.0
    (slope, intercept), total = fit_line(x, y)
    assert slope == pytest.approx(3.0 / span, rel=1e-12)
    assert intercept == pytest.approx(5.0 - 3.0 * offset / span, rel=1e-12)
    assert total == pytest.approx(0.0, abs=1e-20)


def test_line_residuals_and_degenerate_minimum_norm_solution_are_preserved() -> None:
    (slope, intercept), total = fit_line(np.arange(4.0), np.array([1.0, 4.0, 4.0, 8.0]))
    assert (slope, intercept, total) == pytest.approx((2.1, 1.1, 2.7))
    for constant in (2.0, 2.1):
        (slope, intercept), total = fit_line(
            np.full(3, constant), np.array([1.0, 2.0, 3.0])
        )
        intercept_expected = 2.0 / (constant**2 + 1.0)
        assert (slope, intercept, total) == pytest.approx(
            (constant * intercept_expected, intercept_expected, 2.0)
        )


def test_moments_use_population_variance_and_preserve_unknown_counts() -> None:
    samples = np.array([1.0, 2.0, 4.0, 7.0])
    expected = float(np.std(samples, ddof=0)) / math.sqrt(samples.size)
    mean, mean_sq = float(samples.mean()), float((samples * samples).mean())
    assert standard_error_from_moments(mean, mean_sq, 4.0) == expected
    errors = standard_error_from_moments(
        np.array([mean, mean, mean, 2.0]),
        np.array([mean_sq, mean_sq, mean_sq, 4.0 - 1e-15]),
        np.array([4.0, 1.0, 0.0, 3.0]),
    )
    assert errors[0] == expected
    assert np.isnan(errors[1:3]).all()
    assert errors[3] == 0.0
    assert standard_error_from_moments(2.0, 4.0 - 1e-15, 3.0) == 0.0
    assert math.isnan(standard_error_from_moments(mean, mean_sq, 1.0))
    assert math.isnan(standard_error_from_moments(mean, mean_sq, 0.0))


def test_rms_does_not_square_large_or_small_dimensional_values() -> None:
    for scale in (1.0, 1e200, 1e-200):
        assert rms(np.array([3.0, 4.0]) * scale) == pytest.approx(
            math.sqrt(12.5) * scale, rel=1e-15, abs=0.0
        )
    assert rms(np.zeros(3)) == 0.0
    assert rms(np.array([math.inf])) == math.inf


def test_nearby_rate_groups_compare_to_the_first_member_not_each_neighbour() -> None:
    rates = [2.0, 1.0 + 1.5e-8, 1.0 + 0.75e-8, 1.0, 2.0]
    assert group_nearby_rates(rates, rate=float) == [
        [1.0, 1.0 + 0.75e-8],
        [1.0 + 1.5e-8],
        [2.0, 2.0],
    ]
    assert group_nearby_rates([], rate=float) == []


def test_extrapolation_distance_counts_only_decades_outside_the_measured_range() -> (
    None
):
    rates = np.array([1.0, 100.0])
    assert [extrapolation_decades(rates, target) for target in (0.01, 10.0, 1e5)] == [
        2.0,
        0.0,
        3.0,
    ]
    assert extrapolation_decades(np.array([1e200, 1e201]), 1e-200) == 400.0


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
def test_nonfinite_or_missing_measurements_remain_unknown(value: float | None) -> None:
    assert finite_or_none(value) is None
    assert finite_or_none(0.0) == 0.0
    assert finite_or_none(-2.0) == -2.0


def test_relative_span_and_spacing_keep_zero_missing_and_direction_distinct() -> None:
    assert relative_span(np.array([1.0, 3.0]), 2.0) == 1.0
    assert relative_span(np.zeros(2), 0.0) == 0.0
    assert relative_span(np.ones(2), 0.0) == math.inf
    assert relative_span(np.array([]), 1.0) == math.inf
    assert relative_span(np.array([math.nan]), 1.0) == math.inf
    assert median_spacing(np.array([3.0, 1.0, -1.0])) == -2.0
    assert median_spacing(np.array([3.0, 1.0, -1.0]), absolute=True) == 2.0
    assert median_spacing(np.array([3.0])) == 0.0


def test_a_slope_through_two_points_has_no_finite_error() -> None:
    """Zero would make the least supported fit look like the best one."""
    assert slope_error(np.array([0.0, 1.0]), 0.0) == math.inf
    assert slope_error(np.full(4, 2.0), 1.0) == math.inf
    assert slope_error(np.arange(4.0), 2.0) == pytest.approx(math.sqrt(2.0 / 2 / 5))


@pytest.mark.parametrize(("planted", "edge"), [(0.4, ""), (1.0, "upper")])
def test_a_separable_fit_recovers_the_nonlinear_parameter(
    planted: float, edge: str
) -> None:
    """Linear in two parameters for any fixed third, so the search is exact."""
    x = np.geomspace(0.1, 100.0, 40)
    fitted, slope, intercept, total, at = separable_fit(
        lambda power: x**power, 3.0 * x**planted + 2.0, (0.1, 1.0)
    )
    assert fitted == pytest.approx(planted, abs=1e-6)
    assert (slope, intercept) == pytest.approx((3.0, 2.0), rel=1e-5)
    assert total == pytest.approx(0.0, abs=1e-12)
    assert at == edge


def test_a_correlated_series_is_worth_the_same_samples_in_any_unit() -> None:
    """The bug an absolute variance floor causes: in small enough units every
    row reads as independent, because the series looks constant."""
    values = ar1(2000, 0.95) + 10.0
    inefficiency = statistical_inefficiency(values)
    assert inefficiency > 5.0
    for scale in (1.0e-20, 1.0e-160, 1.0e20):
        assert statistical_inefficiency(values * scale) == pytest.approx(
            inefficiency, rel=1e-9
        )


def test_a_constant_series_counts_each_row_once() -> None:
    """There are no correlations to measure in something that never moved."""
    assert statistical_inefficiency(np.full(100, 0.9)) == 1.0
    assert statistical_inefficiency(np.zeros(10)) == 1.0


def test_the_standard_error_uses_the_sample_deviation_over_independent_samples() -> (
    None
):
    """Bessel-corrected, and over the independent count rather than the rows."""
    values = np.array([1.0, 2.0, 4.0, 7.0])
    expected = float(np.std(values, ddof=1)) / math.sqrt(2.0)
    assert standard_error(values, 2.0) == pytest.approx(expected, rel=1e-12)
    assert standard_error(values * 1.0e-200, 2.0) == pytest.approx(
        expected * 1.0e-200, rel=1e-12
    )
