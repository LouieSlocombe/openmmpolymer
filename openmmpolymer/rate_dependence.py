"""Empirical finite-rate corrections with shared uncertainty and refusal guards.

Rates must use the units recorded in :class:`RateProperty`. These fits compare
like measurements at different rates; they do not establish equilibrium or a
zero-rate limit. Missing events and unknown errors remain explicit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from ._fitting import ROUNDING
from ._validation import require_positive
from .elasticity import MAX_RELATIVE_STANDARD_ERROR
from .trajectory import AnalysisError

#: The two empirical relations every rate series is fitted with.
RATE_FORMS = ("log_linear", "power_law")

#: Conservative reporting guards, not statistical goodness-of-fit thresholds:
#: the RMS residual may be at most this many times the RMS reported error in
#: the fitted response scale, and at most this fraction of the mean absolute
#: measured value.
MAX_RATE_RESIDUAL_TO_ERROR = 3.0
MAX_RELATIVE_RATE_RESIDUAL = 0.10

#: Mean temperatures recorded by independent trajectories need not be identical.
_TEMPERATURE_TOLERANCE_K = 1.0


@dataclass(frozen=True)
class RateProperty:
    """Meaning, units, expected trend, and strict physical bounds of a measure.

    ``trend`` is ``increasing``, ``decreasing``, or ``any``; flat responses
    satisfy every trend. Bounds are exclusive and may be omitted for signed
    responses. A trend is a reporting guard, never a constraint on the fit.
    """

    name: str
    label: str
    value_unit: str
    rate_unit: str
    trend: str = "any"
    lower_bound: float | None = 0.0
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.label.strip():
            raise ValueError("Property name and label must be nonempty.")
        if self.trend not in ("increasing", "decreasing", "any"):
            raise ValueError("trend must be increasing, decreasing, or any.")
        for bound in (self.lower_bound, self.upper_bound):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("Property bounds must be finite or None.")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            raise ValueError("lower_bound must be smaller than upper_bound.")


@dataclass(frozen=True)
class RateObservation:
    """One measured value, retaining unresolved events and missing uncertainty.

    ``conditions`` must describe all relevant shared settings, such as strain
    window, deformation direction, fit method, and thermal branch. Conditions
    are compared before fitting. Material identity and preparation remain the
    caller's responsibility. ``None`` is an unknown error, not a zero error.
    """

    rate: float
    value: float | None
    standard_error: float | None
    resolved: bool
    temperature_k: float | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RateExtrapolation:
    """A finite-rate prediction, with uncertainty in the fitted mean.

    Input arrays contain one entry per distinct rate. Replicas are pooled by
    their mean, and their error is the larger of propagated mean error and
    between-replica sample standard deviation. The target error includes input
    errors and excess residual scatter, but excludes model choice, correlated
    runs and systematic simulation errors. Unknown input errors yield an
    unknown target error unless independent replicas provide measurable
    between-replica spread.

    ``reference_rate`` is the geometric mean of the rates, and
    ``sensitivity_per_decade`` the derivative with respect to log10(rate)
    there. ``residual_to_error_ratio`` compares RMS residual with RMS reported
    error in the fitted response scale (the value, or its logarithm); it is
    None when no error is known and nonzero. ``relative_residual`` uses the
    mean absolute measured value, so signed responses cannot conceal
    disagreement by cancelling their mean.
    """

    property: RateProperty
    form: str
    value: float
    target_rate: float
    rates: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    standard_errors: npt.NDArray[np.float64]
    parameters: dict[str, float]
    residual: float
    relative_residual: float
    residual_to_error_ratio: float | None
    standard_error: float
    sensitivity_per_decade: float
    reference_rate: float
    n_rates: int
    extrapolation_decades: float
    resolved: bool
    notes: tuple[str, ...]

    @overload
    def predict(self, rate: float) -> float: ...

    @overload
    def predict(self, rate: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    def predict(
        self, rate: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """Evaluate at finite positive rates; this does not imply resolution.

        A logarithmic fit may predict values outside the property's bounds
        sufficiently far away.
        """
        rates = np.asarray(rate, dtype=np.float64)
        if np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
            raise ValueError("rate must contain finite positive rates.")
        log_ratio = np.log10(rates) - math.log10(self.reference_rate)
        reference = self.parameters["reference_value"]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if self.form == "log_linear":
                prediction = (
                    reference + self.parameters["sensitivity_per_decade"] * log_ratio
                )
            else:
                prediction = np.exp(
                    math.log(reference)
                    + self.parameters["exponent"] * math.log(10.0) * log_ratio
                )
        return float(prediction) if rates.ndim == 0 else np.asarray(prediction)


@dataclass(frozen=True)
class RateReport:
    """All observations and both attempted models, including unavailable fits."""

    property: RateProperty
    observations: tuple[RateObservation, ...]
    log_linear: RateExtrapolation | None
    power_law: RateExtrapolation | None
    notes: tuple[str, ...]
    run_dirs: tuple[str, ...] = ()
    target_rate: float | None = None
    max_extrapolation_decades: float = 2.0


def validate_rate_request(
    target_rate: float, max_extrapolation_decades: float
) -> tuple[float, float]:
    """Check the target rate and extrapolation allowance, before any work.

    Every rate workflow asks for these two, and refuses them before it reads
    or runs anything, rather than after the dynamics they would qualify.
    """
    target = require_positive(target_rate, None, name="target_rate")
    maximum = float(max_extrapolation_decades)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("max_extrapolation_decades must be finite and nonnegative.")
    return target, maximum


def _rms(values: npt.NDArray[np.float64]) -> float:
    """Compute RMS without squaring the original dimensional values."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0 or not math.isfinite(scale):
        return scale
    return float(np.sqrt(np.mean((values / scale) ** 2))) * scale


def _regression(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
    target_x: float,
) -> tuple[float, float, npt.NDArray[np.float64], float, float]:
    """Centered OLS with known-error covariance and excess residual scatter.

    Equal weight per rate avoids arbitrarily infinite weight when an input
    fit reports zero error. With hat matrix H and input variance D, the
    expected noise contribution to RSS is trace((I-H)D). Subtracting this
    before estimating additional scatter avoids counting it twice.
    """
    scale_x = float(np.max(np.abs(x)))
    z = x / scale_x
    target_z = target_x / scale_x
    # Scaling the response keeps squared residuals well behaved for large values.
    scale_y = max(float(np.max(np.abs(y))), 1.0)
    values = y / scale_y
    sigma = errors / scale_y
    # Anchoring preserves a constant response exactly. Summing n identical
    # values first can round the mean away from that value, leaving a tiny
    # artificial slope whose sign would incorrectly refuse a flat response.
    mean = float(values[0] + np.mean(values - values[0]))
    squared = float(z @ z)
    slope = float(z @ (values - mean)) / squared
    fitted = mean + slope * z
    residual = values - fitted
    target_weights = 1.0 / x.size + target_z * z / squared
    leverage = 1.0 / x.size + z * z / squared
    known_rss = float(np.sum(np.maximum(1.0 - leverage, 0.0) * sigma**2))
    extra_variance = max((float(residual @ residual) - known_rss) / (x.size - 2), 0.0)
    variance = float(np.sum((target_weights * sigma) ** 2))
    variance += extra_variance * (1.0 / x.size + target_z**2 / squared)
    return (
        mean * scale_y,
        slope * scale_y / scale_x,
        np.asarray(fitted * scale_y, dtype=np.float64),
        (mean + slope * target_z) * scale_y,
        math.sqrt(variance) * scale_y,
    )


def _within_bounds(value: float, property: RateProperty) -> bool:
    return (
        math.isfinite(value)
        and (property.lower_bound is None or value > property.lower_bound)
        and (property.upper_bound is None or value < property.upper_bound)
    )


def _same_conditions(left: Any, right: Any) -> bool:
    """Compare nested metadata without ambiguous array truth values."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_conditions(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list, np.ndarray)) and isinstance(
        right, (tuple, list, np.ndarray)
    ):
        return len(left) == len(right) and all(
            _same_conditions(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return math.isclose(float(left), float(right), rel_tol=1.0e-8, abs_tol=1.0e-9)
    return bool(left == right)


def _pool_observations(
    observations: Sequence[RateObservation], property: RateProperty
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    temperatures: list[float] = []
    grouped: dict[float, list[RateObservation]] = {}
    for index, observation in enumerate(observations):
        if not math.isfinite(observation.rate) or observation.rate <= 0.0:
            raise AnalysisError(f"Observation {index} needs a finite positive rate.")
        if observation.value is None:
            raise AnalysisError(
                f"Observation {index} is missing or censored; rate fitting cannot "
                "discard an unobserved event."
            )
        if not _within_bounds(observation.value, property):
            raise AnalysisError(
                f"Observation {index} is nonfinite or outside strict physical bounds."
            )
        error = observation.standard_error
        if error is not None and (not math.isfinite(error) or error < 0.0):
            raise AnalysisError(
                f"Observation {index} standard error must be finite and nonnegative or None."
            )
        temperature = observation.temperature_k
        if temperature is not None:
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise AnalysisError(
                    f"Observation {index} temperature must be finite and positive or None."
                )
            temperatures.append(temperature)
        if not _same_conditions(observation.conditions, observations[0].conditions):
            raise AnalysisError(
                "Fit observations with the same measurement conditions."
            )
        grouped.setdefault(observation.rate, []).append(observation)
    if (
        temperatures
        and max(temperatures) - min(temperatures) > _TEMPERATURE_TOLERANCE_K
    ):
        raise AnalysisError("Fit observations at the same temperature (within 1 K).")
    # Rates inferred from resumed chunks can differ by rounding. They are
    # replicas of one rate, not extra independent logarithmic abscissae.
    merged: dict[float, list[RateObservation]] = {}
    representative: float | None = None
    for rate in sorted(grouped):
        if representative is None or not math.isclose(
            rate, representative, rel_tol=1e-8, abs_tol=0.0
        ):
            representative = rate
            merged[representative] = []
        merged[representative].extend(grouped[rate])
    grouped = merged
    if len(grouped) < 3:
        raise AnalysisError("Measure the property at three or more distinct rates.")
    rates = np.asarray(sorted(grouped), dtype=np.float64)
    values: list[float] = []
    errors: list[float] = []
    for rate in rates:
        replicas = grouped[float(rate)]
        data = np.asarray([item.value for item in replicas], dtype=np.float64)
        # Scale before averaging to avoid overflow for otherwise finite inputs.
        scale = max(float(np.max(np.abs(data))), 1.0)
        mean = float(np.mean(data / scale)) * scale
        values.append(mean)
        unknown = any(item.standard_error is None for item in replicas)
        input_errors = np.asarray(
            [item.standard_error or 0.0 for item in replicas], dtype=np.float64
        )
        mean_error = _rms(input_errors) / math.sqrt(len(replicas))
        replica_error = (
            _rms(data / scale - mean / scale)
            * scale
            * math.sqrt(len(replicas) / (len(replicas) - 1))
            if len(replicas) > 1
            else 0.0
        )
        errors.append(
            math.nan
            if unknown and replica_error == 0.0
            else max(mean_error, replica_error)
        )
    return rates, np.asarray(values), np.asarray(errors)


def rate_extrapolation(
    observations: Sequence[RateObservation],
    *,
    property: RateProperty,
    target_rate: float,
    form: str = "log_linear",
    max_extrapolation_decades: float = 2.0,
) -> RateExtrapolation:
    """Fit a logarithmic line or positive-valued power law at a finite rate.

    ``log_linear`` fits ``y = y_ref + b log10(rate / reference_rate)``;
    ``power_law`` fits ``ln(y)`` against log10(rate), equivalently
    ``y = y_ref (rate / reference_rate)**exponent``, propagating input errors
    to first order into the logarithm and back. Both give each distinct rate
    equal weight, and propagate input errors plus excess residual scatter
    into the target's fit-mean uncertainty.

    Three distinct rates are required; rates within a relative 1e-8 of each
    other are one rate. Replicas are pooled without counting them as
    independent rates. Missing/censored measurements, incompatible conditions,
    or temperatures spanning more than 1 K raise ``AnalysisError``. Invalid
    requests raise ``ValueError``. Unresolved finite observations remain in
    the fit and prevent it from being reported as resolved.

    A small target error does not establish that the model fits the data, so
    two residual guards apply independently: the RMS residual must be at most
    three times the RMS known error in the fitted response scale, allowing
    floating-point rounding, and 10% of the mean absolute measured value. The
    fit-mean error must be at most 25% of the absolute target, and the target
    within the caller's extrapolation distance of the measured rates. These
    are reporting guards, not a statistical test of empirical model validity.
    The target is required because neither form defines a zero-rate value.
    """
    if form not in RATE_FORMS:
        raise ValueError(f"form must be one of {RATE_FORMS}, got {form!r}.")
    target, maximum = validate_rate_request(target_rate, max_extrapolation_decades)
    rates, measured, errors = _pool_observations(observations, property)
    if form == "power_law" and np.any(measured <= 0.0):
        raise AnalysisError("A power-law fit requires strictly positive observations.")
    log_rate = np.log10(rates)
    if np.unique(log_rate).size < 3:
        raise AnalysisError("The rates are too close to resolve in log space.")
    # Work in log space instead of taking products or ratios of extreme rates.
    reference_log = float(np.mean(log_rate))
    reference_rate = 10.0**reference_log
    x = log_rate - reference_log
    # Correct the last rounding-sized offset so the OLS intercept is centred.
    offset = float(np.mean(x))
    x -= offset
    target_x = math.log10(target) - reference_log - offset
    notes = ["Empirical finite-rate estimate; not a zero-rate or equilibrium property."]
    refusals: list[str] = []
    unknown_errors = bool(np.any(~np.isfinite(errors)))
    if rates.size < len(observations):
        notes.append(
            "Repeated rates were pooled; uncertainties retain the larger of "
            "propagated mean error and between-replica sample standard deviation."
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        values = measured if form == "log_linear" else np.log(measured)
        input_errors = errors if form == "log_linear" else errors / measured
        intercept, slope, fitted, prediction, error = _regression(
            x, values, np.nan_to_num(input_errors, nan=0.0), target_x
        )
        response_residual = _rms(values - fitted)
        response_error = _rms(input_errors)
        residual_to_error = (
            response_residual / response_error
            if math.isfinite(response_error) and response_error > 0.0
            else None
        )
        rounding_tolerance = ROUNDING * float(np.max(np.abs(values)))
        # The tiny centring adjustment belongs to the regression, not y_ref.
        intercept -= slope * offset
        if form == "log_linear":
            reference_value = intercept
            sensitivity = slope
            parameters = {
                "reference_value": reference_value,
                "sensitivity_per_decade": sensitivity,
            }
        else:
            reference_value = float(np.exp(intercept))
            prediction = float(np.exp(prediction))
            error *= prediction
            fitted = np.exp(fitted)
            sensitivity = reference_value * slope
            parameters = {
                "reference_value": reference_value,
                "exponent": slope / math.log(10.0),
            }
        residual = _rms(measured - fitted)
        scale = float(np.max(np.abs(measured)))
        mean_absolute = (
            float(np.mean(np.abs(measured) / scale)) * scale if scale else 0.0
        )
        relative_residual = (
            residual / mean_absolute
            if mean_absolute
            else (0.0 if residual == 0.0 else math.inf)
        )
    decades = max(
        float(log_rate[0]) - math.log10(target),
        math.log10(target) - float(log_rate[-1]),
        0.0,
    )
    if not all(item.resolved for item in observations):
        refusals.append("At least one input observation is unresolved.")
    temperatures = [item.temperature_k for item in observations]
    if any(item is None for item in temperatures) and any(
        item is not None for item in temperatures
    ):
        refusals.append(
            "Some input temperatures are unknown; comparability is unverified."
        )
    if unknown_errors:
        error = math.nan
        refusals.append(
            "Input uncertainty is unknown; target standard error is unavailable."
        )
    if not math.isfinite(sensitivity):
        refusals.append("Rate sensitivity is nonfinite.")
    elif (property.trend == "increasing" and sensitivity < 0.0) or (
        property.trend == "decreasing" and sensitivity > 0.0
    ):
        refusals.append(
            f"Rate sensitivity contradicts the expected {property.trend} trend."
        )
    if not _within_bounds(prediction, property):
        refusals.append(
            "The target value is nonfinite or outside strict physical bounds."
        )
    if not unknown_errors and (
        not math.isfinite(error)
        or error > MAX_RELATIVE_STANDARD_ERROR * abs(prediction)
    ):
        refusals.append(
            "Target standard error exceeds 25% of the absolute fitted value."
        )
    if not math.isfinite(residual):
        refusals.append("The fitted residual is nonfinite.")
    if (
        residual_to_error is not None
        and residual_to_error > MAX_RATE_RESIDUAL_TO_ERROR
        and response_residual > rounding_tolerance
    ):
        refusals.append(
            "The model RMS residual exceeds 3 times the RMS reported errors "
            "in the fitted response scale."
        )
    if (
        not math.isfinite(relative_residual)
        or relative_residual > MAX_RELATIVE_RATE_RESIDUAL
    ):
        refusals.append(
            "The model RMS residual exceeds 10% of the mean absolute measured value."
        )
    if decades > maximum:
        refusals.append(
            f"Target is {decades:.3g} decades outside measured rates, "
            f"exceeding the {maximum:.3g}-decade limit."
        )
    notes.extend(refusals)
    return RateExtrapolation(
        property=property,
        form=form,
        value=float(prediction),
        target_rate=target,
        rates=rates,
        values=measured,
        standard_errors=errors,
        parameters=parameters,
        residual=residual,
        relative_residual=float(relative_residual),
        residual_to_error_ratio=residual_to_error,
        standard_error=float(error),
        sensitivity_per_decade=float(sensitivity),
        reference_rate=reference_rate,
        n_rates=int(rates.size),
        extrapolation_decades=decades,
        resolved=not refusals,
        notes=tuple(notes),
    )


def analyse_rate_observations(
    observations: Sequence[RateObservation],
    *,
    property: RateProperty,
    target_rate: float,
    max_extrapolation_decades: float = 2.0,
    run_dirs: Sequence[str] = (),
) -> RateReport:
    """Attempt both models without dropping unavailable or censored data.

    Analysis refusals are recorded with unavailable fits, while invalid target
    rates or extrapolation limits still raise ``ValueError``. Report notes
    retain per-observation qualifications and distinguish model disagreement
    from a confidence interval.
    """
    target, maximum = validate_rate_request(target_rate, max_extrapolation_decades)
    inputs = tuple(observations)
    notes = [
        "Model disagreement is sensitivity to model choice, not a confidence interval."
    ]
    for index, observation in enumerate(inputs):
        notes.extend(f"Observation {index}: {note}" for note in observation.notes)
    fits: dict[str, RateExtrapolation | None] = {}
    for form in RATE_FORMS:
        try:
            fits[form] = rate_extrapolation(
                inputs,
                property=property,
                target_rate=target,
                form=form,
                max_extrapolation_decades=maximum,
            )
        except AnalysisError as exc:
            fits[form] = None
            notes.append(f"{form} unavailable: {exc}")
    return RateReport(
        property=property,
        observations=inputs,
        log_linear=fits["log_linear"],
        power_law=fits["power_law"],
        notes=tuple(notes),
        run_dirs=tuple(str(item) for item in run_dirs),
        target_rate=target,
        max_extrapolation_decades=maximum,
    )
