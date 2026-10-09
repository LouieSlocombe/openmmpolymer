"""The fitting and sampling numerics the analyses share, in numpy alone.

The package does not depend on scipy, so the few primitives the analyses need
are written out here once: a straight line and the standard error of its
slope, a search for the three-parameter relations that are linear in all but
one of them, and the statistics of a correlated series.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import overload

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

#: Unit conversion for dynamics durations and rates.
PS_PER_NS = 1000.0

#: Largest relative standard error for a resolved fitted modulus.
MAX_RELATIVE_STANDARD_ERROR = 0.25

#: Largest relative scatter among replicas for a resolved measurement.
MAX_REPLICA_SPREAD = 0.3

#: A finite-rate fit remains resolved at most two decades outside its data.
MAX_EXTRAPOLATION_DECADES = 2.0

#: The grid :func:`separable_fit` sweeps, and the zooms it then makes a grid
#: step either side of the best point so far.
_SWEEP_POINTS = 181
_ZOOM_POINTS = 41
_ZOOMS = 3


def fit_line(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[tuple[float, float], float]:
    """Least-squares line through *x*, *y*, with its sum of squared residuals.

    Centre and scale x so that a narrow window at a large offset remains
    full rank. Degenerate x keeps the original minimum-norm solution.
    ``numpy.linalg.lstsq`` returns an empty residual array for an exactly
    determined fit, so the sum is worked out when it is not handed back.
    """
    origin = float(x.mean()) if x.size else 0.0
    centred = x - origin
    scale = float(np.max(np.abs(centred))) if x.size and not np.all(x == x[0]) else 0.0
    design_x = centred / scale if scale else x
    design = np.vstack([design_x, np.ones_like(x)]).T
    solution, residuals, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(solution[0]), float(solution[1])
    if scale:
        slope /= scale
        intercept -= slope * origin
    if residuals.size:
        total = float(residuals[0])
    else:
        predicted = design @ solution
        total = float(((y - predicted) ** 2).sum())
    return (slope, intercept), total


def rms(values: npt.NDArray[np.float64]) -> float:
    """Compute RMS without squaring the original dimensional values."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0 or not math.isfinite(scale):
        return scale
    return float(np.sqrt(np.mean((values / scale) ** 2))) * scale


def finite_or_none(value: float | None) -> float | None:
    """A measured value, or None when it is missing or nonfinite."""
    return None if value is None or not math.isfinite(value) else float(value)


def group_nearby_rates[T](
    items: Iterable[T], *, rate: Callable[[T], float]
) -> list[list[T]]:
    """Group sorted rates within 1e-8 of each group's first rate.

    Compare against the representative, not the preceding member: a chain of
    close neighbours need not represent one rate.
    """
    groups: list[list[T]] = []
    for item in sorted(items, key=rate):
        if groups and math.isclose(
            rate(item), rate(groups[-1][0]), rel_tol=1e-8, abs_tol=0.0
        ):
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def extrapolation_decades(
    rates: npt.NDArray[np.float64], target: float, *, log_difference: bool = False
) -> float:
    """Distance beyond positive measured rates, preserving boundary arithmetic.

    Thermal fits historically take the log of ratios, whereas property-rate
    fits subtract logs. These can differ by an ulp at their acceptance limit;
    each caller retains its original order. Ratios use log differences only
    when the intermediate ratio overflows or underflows.
    """
    if log_difference:
        logs = np.log10(rates)
        value = math.log10(target)
        return max(float(logs.min()) - value, value - float(logs.max()), 0.0)

    def log_ratio(numerator: float, denominator: float) -> float:
        ratio = numerator / denominator
        if 0.0 < ratio < math.inf:
            return math.log10(ratio)
        return math.log10(numerator) - math.log10(denominator)

    return max(
        0.0,
        log_ratio(float(rates.min()), target),
        log_ratio(target, float(rates.max())),
    )


def relative_span(values: npt.NDArray[np.float64], scale: float) -> float:
    """How far values spread over a scale; infinite for missing/undefined data."""
    if not values.size or np.any(~np.isfinite(values)):
        return math.inf
    if scale == 0.0:
        return 0.0 if np.all(values == 0.0) else math.inf
    return float(np.ptp(values / scale))


def median_spacing(values: npt.NDArray[np.float64], *, absolute: bool = False) -> float:
    """Typical adjacent gap, optionally independent of direction."""
    if values.size < 2:
        return 0.0
    steps = np.diff(values)
    return float(np.median(np.abs(steps) if absolute else steps))


@overload
def standard_error_from_moments(mean: float, mean_sq: float, count: float) -> float: ...


@overload
def standard_error_from_moments(
    mean: npt.NDArray[np.float64],
    mean_sq: npt.NDArray[np.float64],
    count: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...


def standard_error_from_moments(
    mean: float | npt.NDArray[np.float64],
    mean_sq: float | npt.NDArray[np.float64],
    count: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Population-variance SE, clamping roundoff; NaN with at most one sample.

    This assumes independent readings. It is distinct from standard_error's
    Bessel-corrected sample variance and effective independent sample count.
    Keep scalar and array arithmetic in their original operation order.
    """
    if not isinstance(mean, np.ndarray):
        n = float(count)
        return (
            math.sqrt(max(0.0, float(mean_sq) - mean**2) / n) if n > 1.0 else math.nan
        )
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.sqrt(variance / count)
    return np.where(np.asarray(count) > 1.0, error, np.nan)


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
