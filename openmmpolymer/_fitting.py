"""The fitting and sampling numerics the analyses share, in numpy alone.

The package does not depend on scipy, so the few primitives the analyses need
are written out here once: a straight line and the standard error of its
slope, a search for the three-parameter relations that are linear in all but
one of them, and the statistics of a correlated series.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

#: Guards a ratio whose denominator is zero - a series that never moved, a
#: chain of zero length - without judging what counts as small.
TINY = 1.0e-30

#: What counts as zero in the units the mechanical fits work in: a modulus in
#: MPa, a strain, a pressure in bar. Rounding leaves residue near 1e-16 of
#: these and anything physical is orders of magnitude larger.
NEGLIGIBLE = 1.0e-12

#: A spread this small against a series' largest magnitude is rounding error,
#: a few dozen ulps, rather than anything that was sampled.
ROUNDING = 64.0 * float(np.finfo(np.float64).eps)

#: The grid :func:`separable_fit` sweeps, and the zooms it then makes a grid
#: step either side of the best point so far.
_SWEEP_POINTS = 181
_ZOOM_POINTS = 41
_ZOOMS = 3


def fit_line(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[tuple[float, float], float]:
    """Least-squares line through *x*, *y*, with its sum of squared residuals.

    ``numpy.linalg.lstsq`` returns an empty residual array for an exactly
    determined fit, so the sum is worked out when it is not handed back.
    """
    design = np.vstack([x, np.ones_like(x)]).T
    solution, residuals, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(solution[0]), float(solution[1])
    if residuals.size:
        total = float(residuals[0])
    else:
        predicted = design @ solution
        total = float(((y - predicted) ** 2).sum())
    return (slope, intercept), total


def slope_error(x: npt.NDArray[np.float64], residual_sum: float) -> float:
    """Standard error of a least-squares slope.

    ``sqrt( sum(residual^2) / (n - 2) / sum((x - xbar)^2) )``, which is
    infinite when there are only two points: a line through two points has no
    residual and no error, and reporting zero would make the least supported
    fit look like the best one.
    """
    n = x.size
    spread = float(((x - x.mean()) ** 2).sum())
    if n <= 2 or spread < NEGLIGIBLE:
        return math.inf
    return math.sqrt(max(residual_sum, 0.0) / (n - 2) / spread)


def separable_fit(
    column: Callable[[float], npt.NDArray[np.float64]],
    y: npt.NDArray[np.float64],
    bracket: tuple[float, float],
) -> tuple[float, float, float, float, str]:
    """Fit ``y = a column(p) + b`` for a *p* somewhere in *bracket*.

    Three parameters and no scipy, but the problem separates: for any fixed
    *p* the relation is linear in *a* and *b*, so the nonlinear fit collapses
    to a one-dimensional search with an exact least-squares solve inside it.
    A 181-point sweep of the bracket and three 41-point zooms around the best
    so far is some three hundred solves of a two-by-two system:
    deterministic, derivative-free, and with no way to fail to converge.

    Returns *p*, *a*, *b*, the sum of squared residuals, and which end of the
    bracket - ``"lower"``, ``"upper"`` or ``""`` - the optimum ran to.
    """

    def solve(parameter: float) -> tuple[float, float, float]:
        design = np.vstack([column(parameter), np.ones_like(y)]).T
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        # Worked out rather than read from lstsq, which has none for two points.
        residual = y - design @ solution
        return float(solution[0]), float(solution[1]), float(residual @ residual)

    lower, upper = bracket
    grid = np.linspace(lower, upper, _SWEEP_POINTS)
    best_parameter, best = lower, solve(lower)
    for _ in range(1 + _ZOOMS):
        for candidate in grid:
            trial = solve(float(candidate))
            if trial[2] < best[2]:
                best_parameter, best = float(candidate), trial
        step = float(grid[1] - grid[0])
        grid = np.linspace(
            max(lower, best_parameter - step),
            min(upper, best_parameter + step),
            _ZOOM_POINTS,
        )

    edge = ""
    if best_parameter <= lower + 1.0e-9:
        edge = "lower"
    elif best_parameter >= upper - 1.0e-9:
        edge = "upper"
    return best_parameter, *best, edge


def statistical_inefficiency(values: npt.NDArray[np.float64]) -> float:
    """Return ``1 + 2 sum C(t)``, how many rows one independent sample is worth.

    The series is standardised first - divided by its largest magnitude,
    centred, divided by its spread - so the answer cannot depend on the unit
    it was recorded in: an absolute floor on the variance would read a series
    in small enough units as constant, and count every row as independent. A
    series that is constant to rounding has no correlations to measure, and
    each of its rows counts once.
    """
    normal = values / (float(np.max(np.abs(values))) or 1.0)
    if float(np.ptp(normal)) <= ROUNDING:
        return 1.0
    centred = normal - np.mean(normal)
    return _correlation_sum(centred / float(np.std(centred)))


def _correlation_sum(values: npt.NDArray[np.float64]) -> float:
    """``1 + 2 sum C(t)`` over a series that varies.

    The autocorrelation comes from an FFT, and the sum is truncated at its
    first non-positive term: past that point the estimator is noise, and
    summing the noise is how a correlation time ends up longer than the run.
    """
    n = values.size
    centred = values - values.mean()
    variance = float(centred @ centred) / n
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n].real
    correlation /= n * variance

    total = 0.0
    for lag in range(1, n):
        term = float(correlation[lag])
        if term <= 0.0:
            break
        total += term * (1.0 - lag / n)
    return max(1.0, 1.0 + 2.0 * total)


def standard_error(values: npt.NDArray[np.float64], n_independent: float) -> float:
    """The standard error of the mean of *values*, worth *n_independent* samples.

    The sample standard deviation - with Bessel's correction, ``ddof=1`` -
    over the square root of the independent count rather than of the rows,
    worked in units of the largest value so that no unit is small enough to
    underflow.
    """
    scale = float(np.max(np.abs(values))) or 1.0
    return float(np.std(values / scale, ddof=1)) * scale / math.sqrt(n_independent)
