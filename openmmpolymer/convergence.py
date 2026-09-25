"""Observation-window stability without mistaking short runs for equilibrium.

A quantity that stops changing as the observed time grows is stable over the
windows observed, and that is all it is: it says nothing about slower degrees
of freedom the run never sampled. So every analysis here compares prefixes of
growing length - which overlap, so their differences are sensitivities rather
than standard errors - and resolves only when the last three of them end at
three distinct sampled times and agree within a relative tolerance.

Stationary observables use autocorrelation-adjusted errors, expanding windows
and disjoint tail blocks (:func:`time_window_convergence`). Relaxation curves
are instead refitted as decays (:func:`relaxation_window_convergence`): their
time dependence is physical and is not a stationary sampling trace. The
structural measurements follow the same rules in
:mod:`openmmpolymer.structural_convergence`, and
:mod:`openmmpolymer.convergence_report` reads all three off a saved run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from ._fitting import ROUNDING, standard_error, statistical_inefficiency
from ._validation import require_positive
from .relaxation import (
    SIGNAL_TO_NOISE_FLOOR,
    KWWFit,
    PronyFit,
    RelaxationCurve,
    _signal_window,
    fit_kww,
    fit_prony,
)
from .strain_rate import MAX_RATE_RESIDUAL_TO_ERROR, MAX_RELATIVE_RATE_RESIDUAL, _rms
from .trajectory import AnalysisError

#: Prefix lengths compared, as fractions of the observed time.
DEFAULT_WINDOW_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

#: How far the compared windows may disagree, relative to their scale, and
#: still count as stable.
DEFAULT_RELATIVE_TOLERANCE = 0.1

#: Fewest samples a compared window may rest on: independent samples for a
#: stationary trace, sampled frames for a structural measurement.
DEFAULT_MIN_SAMPLES = 20

#: One MPa ps in Pa s, the unit a viscosity integrated from a relaxation
#: modulus is quoted in.
PA_S_PER_MPA_PS = 1.0e-6

#: What every window analysis says of a single frame.
SNAPSHOT_REFUSAL = "A single snapshot cannot establish observation-window convergence."

#: What every window analysis says when its last three prefixes end on fewer
#: than three sampled times - fractions closer together than the sampling
#: interval, which make one window read three times.
ENDPOINTS_REFUSAL = (
    "The last three prefixes do not contain three distinct sampled endpoints."
)

#: The relaxation parameters refitted in every window, and their units.
_RELAXATION_METRICS = {
    "equilibrium_modulus_mpa": "MPa",
    "kww_mean_tau_ps": "ps",
    "kww_viscosity_pa_s": "Pa s",
    "prony_viscosity_pa_s": "Pa s",
}


@dataclass(frozen=True)
class WindowEstimate:
    """The mean within one prefix, after its initial fraction is discarded.

    Args:
        fraction: How much of the observed time the prefix covers.
        duration_ps: Time from the first sample to the prefix's last.
        n_samples: Samples kept after the discard.
        n_effective: How many of them are independent, from the statistical
            inefficiency. Zero for a constant window, which samples nothing.
        mean: Their mean.
        standard_error: Its standard error over the independent samples, or
            NaN when there are too few or they are constant.
        resolved: Whether the window has enough independent samples and an
            error within the tolerance.
        notes: Why it did not resolve, if it did not.
    """

    fraction: float
    duration_ps: float
    n_samples: int
    n_effective: float
    mean: float
    standard_error: float
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class WindowConvergence:
    """Stability evidence for one observable, not proof of equilibration.

    Prefix windows overlap and are not independent replicas. Independent
    disjoint blocks of the retained trajectory's latter half provide an
    additional check. Effective sample counts account for autocorrelation
    within each window and each block, not just their raw frame counts.

    Args:
        property_name: What was measured.
        value_unit: Its unit.
        windows: One estimate per prefix fraction.
        block_means: The mean of each of three disjoint blocks spanning the
            second half of the retained samples.
        block_standard_errors: Their standard errors.
        block_effective_samples: Their independent sample counts.
        relative_change: Spread of the last three prefix means over the
            series' scale.
        relative_block_spread: Spread of the block means over the same scale.
        relative_tolerance: The most either spread may be.
        min_effective_samples: Fewest independent samples a window or block
            may rest on.
        discard_fraction: The share of each prefix discarded from its start.
        resolved: Whether nothing refused it.
        notes: The standing caveat, then every refusal.
    """

    property_name: str
    value_unit: str
    windows: tuple[WindowEstimate, ...]
    block_means: npt.NDArray[np.float64]
    block_standard_errors: npt.NDArray[np.float64]
    block_effective_samples: npt.NDArray[np.float64]
    relative_change: float
    relative_block_spread: float
    relative_tolerance: float
    min_effective_samples: float
    discard_fraction: float
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ParameterConvergence:
    """Stability across overlapping model refits, with no fabricated SE.

    Args:
        property_name: The fitted parameter.
        value_unit: Its unit.
        values: Its value in each window, NaN where the fit gave none.
        relative_change: Spread of the last three values over their scale.
        resolved: Whether the last three windows were valid and agree.
        notes: The standing caveat, then every refusal.
    """

    property_name: str
    value_unit: str
    values: npt.NDArray[np.float64]
    relative_change: float
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RelaxationWindowEstimate:
    """Both relaxation fits over one prefix of the decay.

    Args:
        fraction: How much of the observed decay the prefix covers.
        duration_ps: Time from the step strain to the prefix's last bin.
        kww: The stretched exponential fitted to the prefix.
        prony: The Prony series fitted to it.
        tail_decayed: Whether the prefix saw its decay reach a plateau the
            Prony series describes.
        notes: Why the Prony fit could not be trusted, if it could not.
    """

    fraction: float
    duration_ps: float
    kww: KWWFit
    prony: PronyFit
    tail_decayed: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelaxationWindowConvergence:
    """Refits of successively longer G(t), retaining censored decay tails.

    Args:
        windows: One pair of fits per prefix fraction.
        metrics: The stability of each refitted parameter, by name.
        relative_tolerance: The most the last three values may spread.
        resolved: Whether every parameter resolved.
        notes: The standing caveats.
    """

    windows: tuple[RelaxationWindowEstimate, ...]
    metrics: dict[str, ParameterConvergence]
    relative_tolerance: float
    resolved: bool
    notes: tuple[str, ...]


def require_fractions(values: Sequence[float]) -> tuple[float, ...]:
    """The prefix fractions to compare, checked.

    Three or more, because stability is judged on the last three; increasing
    and in (0, 1]; and ending at one, because the whole observation is the
    window everything else is compared with.

    Raises:
        ValueError: They are not.
    """
    fractions = tuple(float(value) for value in values)
    if (
        len(fractions) < 3
        or fractions[-1] != 1.0
        or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0 for value in fractions
        )
        or any(a >= b for a, b in pairwise(fractions))
    ):
        raise ValueError(
            "window_fractions must contain at least three increasing fractions "
            "in (0, 1], ending at 1."
        )
    return fractions


def require_window_options(
    window_fractions: Sequence[float],
    relative_tolerance: float,
    min_effective_samples: float,
    discard_fraction: float,
) -> tuple[float, ...]:
    """Check everything a stationary window analysis is asked, and return the
    fractions.

    Raises:
        ValueError: One of them is out of range.
    """
    fractions = require_fractions(window_fractions)
    require_positive(relative_tolerance, None, name="relative_tolerance")
    if not math.isfinite(min_effective_samples) or min_effective_samples < 1.0:
        raise ValueError("min_effective_samples must be finite and at least one.")
    if not math.isfinite(discard_fraction) or not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must be finite and in [0, 1).")
    return fractions


def distinct_endpoints(durations_ps: Sequence[float]) -> bool:
    """Whether the last three windows end at three different sampled times."""
    return len(set(durations_ps[-3:])) == 3


def _prefix_stop(times: npt.NDArray[np.float64], fraction: float, since: float) -> int:
    """How many samples lie within *fraction* of the time elapsed *since*.

    At least one, so every prefix has something to report.
    """
    reach = since + fraction * (times[-1] - since)
    return max(1, int(np.searchsorted(times, reach, side="right")))


def _series(
    time_ps: npt.ArrayLike, values: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A trace as two arrays, refusing one that cannot be windowed honestly.

    Raises:
        AnalysisError: The arrays are empty, unequal, not one-dimensional,
            not finite, or the times do not increase.
    """
    times, data = (
        np.asarray(time_ps, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )
    if times.ndim != 1 or data.ndim != 1 or times.size != data.size or not times.size:
        raise AnalysisError(
            "Times and values must be nonempty one-dimensional arrays of equal length."
        )
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(data)):
        raise AnalysisError(
            "Times and values must be finite; missing data cannot be silently dropped."
        )
    if np.any(np.diff(times) <= 0.0):
        raise AnalysisError("Times must be strictly increasing.")
    return times, data


def _sample_estimate(data: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """The mean of one window, its standard error, and its independent samples.

    No error and no independent samples for fewer than three samples, or for
    a window constant to rounding, which is evidence of nothing sampled.
    """
    if not data.size:
        return math.nan, math.nan, 0.0
    scale = float(np.max(np.abs(data))) or 1.0
    normal = data / scale
    mean = float(np.mean(normal)) * scale
    if data.size < 3 or float(np.ptp(normal)) <= ROUNDING:
        return mean, math.nan, 0.0
    effective = data.size / statistical_inefficiency(data)
    return mean, standard_error(data, effective), effective


def _relative_span(values: npt.NDArray[np.float64], scale: float) -> float:
    """How far *values* spread, over *scale*; infinite when that means nothing."""
    if not values.size or np.any(~np.isfinite(values)):
        return math.inf
    if scale == 0.0:
        return 0.0 if np.all(values == 0.0) else math.inf
    return float(np.ptp(values / scale))


def time_window_convergence(
    time_ps: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    property_name: str,
    value_unit: str,
    window_fractions: Sequence[float] = DEFAULT_WINDOW_FRACTIONS,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    min_effective_samples: float = float(DEFAULT_MIN_SAMPLES),
    discard_fraction: float = 0.1,
) -> WindowConvergence:
    """Compare expanding prefix means and three disjoint late-time blocks.

    The last three prefix estimates and every tail block must have enough
    effective samples and errors below the requested relative tolerance.
    Their mean spreads must independently satisfy that tolerance. Disjoint
    block means must also agree within three combined standard errors, so an
    arbitrary additive energy zero cannot conceal statistically visible drift.
    At least three distinct sampled prefix endpoints are required. A constant
    trace supplies no evidence of thermal sampling and remains unresolved.
    Uneven sampling is flagged because the autocorrelation estimate assumes
    equal time steps. Resolution describes this observable on sampled windows,
    never equilibrium of unmeasured or slower degrees of freedom.

    Raises:
        ValueError: An option is out of range.
        AnalysisError: The trace cannot be windowed; see :func:`_series`.
    """
    fractions = require_window_options(
        window_fractions, relative_tolerance, min_effective_samples, discard_fraction
    )
    times, data = _series(time_ps, values)
    scale = max(abs(float(np.mean(data))), float(np.std(data)))
    windows: list[WindowEstimate] = []
    for fraction in fractions:
        stop = _prefix_stop(times, fraction, float(times[0]))
        start = int(stop * discard_fraction)
        mean, error, effective = _sample_estimate(data[start:stop])
        notes: list[str] = []
        if stop - start < 3:
            notes.append("Too few samples in this window.")
        if not math.isfinite(error):
            notes.append(
                "Uncertainty unavailable; constant or insufficient data give no "
                "thermal sampling evidence."
            )
        if effective < min_effective_samples:
            notes.append("Too few effective independent samples.")
        if math.isfinite(error) and (
            scale == 0.0 or error > relative_tolerance * scale
        ):
            notes.append("Mean uncertainty exceeds the requested relative tolerance.")
        windows.append(
            WindowEstimate(
                fraction=fraction,
                duration_ps=float(times[stop - 1] - times[0]),
                n_samples=stop - start,
                n_effective=effective,
                mean=mean,
                standard_error=error,
                resolved=not notes,
                notes=tuple(notes),
            )
        )
    retained = data[int(data.size * discard_fraction) :]
    blocks = np.array_split(retained[retained.size // 2 :], 3)
    estimates = [_sample_estimate(block) for block in blocks]
    block_means = np.asarray([item[0] for item in estimates])
    block_errors = np.asarray([item[1] for item in estimates])
    block_effective = np.asarray([item[2] for item in estimates])
    relative_change = _relative_span(
        np.asarray([item.mean for item in windows[-3:]]), scale
    )
    block_spread = _relative_span(block_means, scale)
    refusals: list[str] = []
    if data.size < 3:
        refusals.append(
            "A snapshot or fewer than three time samples cannot establish convergence."
        )
    if times.size > 2 and not np.allclose(
        np.diff(times), np.median(np.diff(times)), rtol=1e-5, atol=1e-9
    ):
        refusals.append(
            "Uneven time spacing prevents a reliable sample-autocorrelation estimate."
        )
    if any(not window.resolved for window in windows[-3:]):
        refusals.append(
            "At least one of the last three windows has insufficient sampling or "
            "uncertainty."
        )
    if not distinct_endpoints([window.duration_ps for window in windows]):
        refusals.append(ENDPOINTS_REFUSAL)
    if np.any(block_effective < min_effective_samples) or np.any(
        ~np.isfinite(block_errors)
    ):
        refusals.append(
            "Disjoint tail blocks have insufficient effective samples or unknown "
            "uncertainty."
        )
    elif scale == 0.0 or np.any(block_errors > relative_tolerance * scale):
        refusals.append(
            "Disjoint tail-block mean uncertainty exceeds the relative tolerance."
        )
    if np.all(np.isfinite(block_errors)) and any(
        abs(block_means[left] - block_means[right])
        > 3.0 * math.hypot(float(block_errors[left]), float(block_errors[right]))
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        refusals.append(
            "Disjoint tail-block means disagree beyond 3 combined standard errors; "
            "this drift guard is independent of the observable's additive zero."
        )
    if relative_change > relative_tolerance:
        refusals.append(
            "The last three prefix means have not stabilised within the relative "
            "tolerance."
        )
    if block_spread > relative_tolerance:
        refusals.append(
            "Disjoint tail-block means have not stabilised within the relative "
            "tolerance."
        )
    if float(np.ptp(data)) == 0.0:
        refusals.append(
            "Constant data provide no thermal sampling evidence; uncertainty is "
            "unknown."
        )
    return WindowConvergence(
        property_name=property_name,
        value_unit=value_unit,
        windows=tuple(windows),
        block_means=block_means,
        block_standard_errors=block_errors,
        block_effective_samples=block_effective,
        relative_change=relative_change,
        relative_block_spread=block_spread,
        relative_tolerance=relative_tolerance,
        min_effective_samples=min_effective_samples,
        discard_fraction=discard_fraction,
        resolved=not refusals,
        notes=(
            "Observable window stability only; overlapping prefixes are not "
            "independent replicas or proof of equilibrium.",
            *refusals,
        ),
    )


def relaxation_window_convergence(
    curve: RelaxationCurve,
    *,
    window_fractions: Sequence[float] = DEFAULT_WINDOW_FRACTIONS,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> RelaxationWindowConvergence:
    """Refit each elapsed-time prefix and test parameter stability and decay.

    Relaxation is not a stationary mean. The last three prefixes must each
    resolve the relevant model and observe its decay/plateau. Model refits
    overlap, so their spread is sensitivity to observation time, not a
    standard error. Prony fits additionally must satisfy the shared response
    residual guards, and use four spectral terms per decade to reduce grid
    sensitivity. Liquid viscosities integrate the fitted relaxation and are
    refused when an independently resolved nonzero equilibrium modulus remains.

    Raises:
        ValueError: An option is out of range.
        AnalysisError: The curve's arrays cannot be refitted honestly.
    """
    fractions = require_fractions(window_fractions)
    require_positive(relative_tolerance, None, name="relative_tolerance")
    times, values = _series(curve.time_ps, curve.modulus_mpa)
    if np.any(times <= 0.0):
        raise AnalysisError("Relaxation bins must have positive times.")
    if any(
        array.shape != times.shape
        for array in (curve.bin_index, curve.standard_error_mpa, curve.n_samples)
    ):
        raise AnalysisError("Relaxation arrays must have matching shapes.")
    if np.any(~np.isfinite(curve.standard_error_mpa)) or np.any(
        curve.standard_error_mpa < 0.0
    ):
        raise AnalysisError(
            "Relaxation standard errors must be finite and nonnegative."
        )
    windows: list[RelaxationWindowEstimate] = []
    records: dict[str, list[float]] = {name: [] for name in _RELAXATION_METRICS}
    validity: dict[str, list[bool]] = {name: [] for name in _RELAXATION_METRICS}
    for fraction in fractions:
        stop = _prefix_stop(times, fraction, 0.0)
        prefix = replace(
            curve,
            time_ps=times[:stop],
            modulus_mpa=values[:stop],
            bin_index=curve.bin_index[:stop],
            standard_error_mpa=curve.standard_error_mpa[:stop],
            n_samples=curve.n_samples[:stop],
        )
        # A coarse one-per-decade spectrum changes its discretisation error
        # noticeably when a prefix moves its upper endpoint. Four terms per
        # decade keep that numerical effect below the default stability guard.
        kww, prony = fit_kww(prefix), fit_prony(prefix, per_decade=4)
        fit_window = _signal_window(
            prefix.modulus_mpa, prefix.standard_error_mpa, SIGNAL_TO_NOISE_FLOOR
        )
        fitted_values = prefix.modulus_mpa[fit_window]
        fitted_errors = prefix.standard_error_mpa[fit_window]
        residual_ok = False
        window_notes: list[str] = []
        if fitted_values.size and math.isfinite(prony.residual_mpa):
            response_scale = float(np.mean(np.abs(fitted_values)))
            reported_error = _rms(fitted_errors)
            rounding = ROUNDING * float(np.max(np.abs(fitted_values)))
            residual_ok = bool(
                prony.residual_mpa <= MAX_RELATIVE_RATE_RESIDUAL * response_scale
                and (
                    reported_error == 0.0
                    or prony.residual_mpa
                    <= max(MAX_RATE_RESIDUAL_TO_ERROR * reported_error, rounding)
                )
            )
        if not residual_ok:
            window_notes.append(
                f"Prony residual exceeds the {MAX_RELATIVE_RATE_RESIDUAL:.0%} "
                f"response-scale or {MAX_RATE_RESIDUAL_TO_ERROR:g}-times-reported-"
                "error guard, or is unavailable."
            )
        initial = abs(float(values[0]))
        tail = float(np.median(values[max(0, stop - 3) : stop]))
        amplitude = initial - prony.equilibrium_mpa
        plateau = bool(
            prony.resolved
            and residual_ok
            and prony.plateau_reached
            and amplitude > 0.0
            and abs(tail - prony.equilibrium_mpa) <= 0.1 * amplitude
        )
        zero_tail = bool(initial > 0.0 and abs(tail) <= 0.1 * initial)
        # A plateau smaller than the known baseline floor is unresolved from
        # zero; otherwise even a small positive plateau makes liquid viscosity
        # mathematically divergent. Numerical NNLS residue gets a tiny allowance.
        zero_floor = max(
            curve.noise_floor_mpa if math.isfinite(curve.noise_floor_mpa) else 0.0,
            initial * 1e-8,
        )
        liquid = bool(plateau and prony.equilibrium_mpa <= zero_floor and zero_tail)
        plateau_conflict = bool(plateau and prony.equilibrium_mpa > zero_floor)
        kww_ok = bool(kww.resolved and zero_tail and not plateau_conflict)
        windows.append(
            RelaxationWindowEstimate(
                fraction=fraction,
                duration_ps=float(times[stop - 1]),
                kww=kww,
                prony=prony,
                tail_decayed=plateau,
                notes=tuple(window_notes),
            )
        )
        records["equilibrium_modulus_mpa"].append(prony.equilibrium_mpa)
        validity["equilibrium_modulus_mpa"].append(plateau)
        records["kww_mean_tau_ps"].append(kww.mean_tau_ps)
        validity["kww_mean_tau_ps"].append(kww_ok)
        records["kww_viscosity_pa_s"].append(
            kww.modulus_mpa * kww.mean_tau_ps * PA_S_PER_MPA_PS
        )
        validity["kww_viscosity_pa_s"].append(kww_ok)
        records["prony_viscosity_pa_s"].append(
            float(prony.weights_mpa @ prony.tau_ps) * PA_S_PER_MPA_PS
        )
        validity["prony_viscosity_pa_s"].append(liquid)
    metrics: dict[str, ParameterConvergence] = {}
    distinct = distinct_endpoints([window.duration_ps for window in windows])
    for name, measured in records.items():
        array = np.asarray(measured, dtype=np.float64)
        final = array[-3:]
        # Equilibrium G can be zero; scale its absolute stability to measured
        # initial G rather than dividing by a physically correct zero plateau.
        scale = (
            abs(float(values[0]))
            if name == "equilibrium_modulus_mpa"
            else abs(float(final[-1]))
        )
        change = _relative_span(final, scale)
        refusals: list[str] = []
        if not distinct:
            refusals.append(ENDPOINTS_REFUSAL)
        if not (
            distinct and all(validity[name][-3:]) and bool(np.all(np.isfinite(final)))
        ):
            refusals.append(
                "At least one of the last three windows has an unresolved fit, "
                "unobserved decay/plateau, or non-liquid tail."
            )
        if not change <= relative_tolerance:
            refusals.append(
                "The last three parameter estimates have not stabilised within the "
                "relative tolerance."
            )
        metrics[name] = ParameterConvergence(
            property_name=name,
            value_unit=_RELAXATION_METRICS[name],
            values=array,
            relative_change=change,
            resolved=not refusals,
            notes=(
                "Overlapping refits measure window sensitivity, not statistical "
                "uncertainty.",
                *refusals,
            ),
        )
    notes = [
        "Finite observation-window stability does not establish an equilibrium "
        "limit or an unsampled slow relaxation mode."
    ]
    if curve.n_replicas < 2:
        notes.append(
            "One relaxation replica: within-run bin errors are not uncertainty "
            "across independent preparations."
        )
    return RelaxationWindowConvergence(
        windows=tuple(windows),
        metrics=metrics,
        relative_tolerance=relative_tolerance,
        resolved=all(item.resolved for item in metrics.values()),
        notes=tuple(notes),
    )
