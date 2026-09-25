"""Tests for the numerics the analyses share."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openmmpolymer._fitting import (
    fit_line,
    separable_fit,
    slope_error,
    standard_error,
    statistical_inefficiency,
)


def ar1(n_samples: int, memory: float, *, seed: int = 5) -> np.ndarray:
    """A correlated series: each sample keeps *memory* of the last one."""
    generator = np.random.default_rng(seed)
    values = np.zeros(n_samples)
    for index in range(1, n_samples):
        values[index] = memory * values[index - 1] + generator.normal(0.0, 0.1)
    return values


def test_a_line_through_two_points_is_fitted_without_residuals() -> None:
    """numpy.linalg.lstsq returns an empty residual array for an exactly
    determined fit, so the sum has to be worked out rather than read off."""
    (slope, intercept), total = fit_line(np.array([0.0, 1.0]), np.array([1.0, 3.0]))
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert total == pytest.approx(0.0, abs=1e-20)


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
