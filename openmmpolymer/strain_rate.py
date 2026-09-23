"""Compare Young's moduli measured at several finite strain rates.

These are empirical rate corrections, not measurements of an equilibrium
modulus. A logarithmic line and a power law can agree over the simulated
window and disagree outside it; neither establishes a zero-rate plateau.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
import numpy.typing as npt

from .elasticity import MAX_RELATIVE_STANDARD_ERROR, ElasticModulus
from .trajectory import AnalysisError

STRAIN_RATE_FORMS = ("log_linear", "power_law")

# Conservative reporting guards, not statistical goodness-of-fit thresholds.
MAX_RATE_RESIDUAL_TO_ERROR = 3.0
MAX_RELATIVE_RATE_RESIDUAL = 0.10

# Mean temperatures recorded by independent trajectories need not be identical.
_TEMPERATURE_TOLERANCE_K = 1.0


@dataclass(frozen=True)
class StrainRateExtrapolation:
    """An empirical Young's modulus at a specified, nonzero strain rate.

    The measurement arrays are sorted by rate. ``reference_rate_per_ns`` is
    their geometric mean, and ``sensitivity_mpa_per_decade`` is the derivative
    with respect to log10(rate) there. ``temperature_k`` is the mean of the
    input temperatures, whose full spread must be at most 1 K.

    ``standard_error_mpa`` estimates uncertainty in the fitted mean at the
    target, propagating the input modulus errors and any excess residual
    scatter. It excludes uncertainty in the choice of empirical model,
    correlated errors between runs, and systematic simulation errors.
    ``residual_mpa`` is the RMS residual on the original modulus scale.
    ``relative_residual`` divides that RMS by the mean measured modulus.
    ``residual_to_error_ratio`` compares RMS residual with RMS reported input
    error in the fitted response scale (E or ln(E)); it is None when all
    reported errors are zero. Residuals exceeding three times the reported
    errors, apart from floating-point rounding, or 10% of the mean modulus
    keep a result unresolved regardless of the target's fit-mean error.
    These conservative reporting guards do not establish model validity.

    A resolved result requires resolved inputs, a nonnegative rate
    sensitivity, a finite positive target modulus, relative target error at
    most :data:`~openmmpolymer.elasticity.MAX_RELATIVE_STANDARD_ERROR`, and
    extrapolation within the caller's permitted distance. ``notes`` explain
    refusals. A resolved finite-rate estimate is still not an equilibrium
    modulus.
    """

    form: str
    modulus_mpa: float
    target_rate_per_ns: float
    temperature_k: float
    strain_limit: float
    strain_rate_per_ns: npt.NDArray[np.float64]
    moduli_mpa: npt.NDArray[np.float64]
    standard_errors_mpa: npt.NDArray[np.float64]
    parameters: dict[str, float]
    residual_mpa: float
    relative_residual: float
    residual_to_error_ratio: float | None
    standard_error_mpa: float
    sensitivity_mpa_per_decade: float
    reference_rate_per_ns: float
    n_rates: int
    extrapolation_decades: float
    resolved: bool
    notes: tuple[str, ...]

    @property
    def modulus_gpa(self) -> float:
        """The target estimate in GPa."""
        return self.modulus_mpa / 1000.0

    @overload
    def predict(self, rate_per_ns: float) -> float: ...

    @overload
    def predict(
        self, rate_per_ns: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def predict(
        self, rate_per_ns: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """Evaluate the fit at positive, finite rates in inverse nanoseconds.

        Evaluation does not make an out-of-range prediction resolved. A
        logarithmic fit may predict negative moduli sufficiently far away.
        """
        rate = np.asarray(rate_per_ns, dtype=np.float64)
        if np.any(~np.isfinite(rate)) or np.any(rate <= 0.0):
            raise ValueError("rate_per_ns must contain finite positive rates.")
        reference = self.parameters["reference_modulus_mpa"]
        log_ratio = np.log10(rate) - math.log10(self.reference_rate_per_ns)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if self.form == "log_linear":
                prediction = (
                    reference
                    + self.parameters["sensitivity_mpa_per_decade"] * log_ratio
                )
            else:
                prediction = np.exp(
                    math.log(reference)
                    + self.parameters["exponent"] * math.log(10.0) * log_ratio
                )
        if rate.ndim == 0:
            return float(prediction)
        return np.asarray(prediction, dtype=np.float64)


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
    if scale_x == 0.0:
        raise AnalysisError("The strain rates are too close to resolve in log space.")
    z = x / scale_x
    target_z = target_x / scale_x
    # Scaling the response keeps squared residuals well behaved for large E.
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


def strain_rate_extrapolation(
    fits: Sequence[ElasticModulus],
    *,
    target_rate_per_ns: float,
    form: str = "log_linear",
    max_extrapolation_decades: float = 2.0,
) -> StrainRateExtrapolation:
    """Fit a finite-rate modulus using at least three distinct strain rates.

    ``log_linear`` fits ``E = Eref + b log10(rate / reference_rate)``.
    ``power_law`` fits ``ln(E)`` against log10(rate), equivalently
    ``E = Eref * (rate / reference_rate)**exponent``. The latter uses a
    first-order propagation of input errors into log modulus and back to
    MPa. Both use equal weight per distinct rate and propagate input errors
    plus excess residual scatter into target fit-mean uncertainty.

    A small target error does not establish that the model fits the data.
    Independently, the RMS residual must be no larger than three times the
    RMS reported error in the fitted response scale, allowing floating-point
    rounding, and no larger than 10% of the mean measured modulus in MPa.
    If every input error is zero, only the relative residual guard applies.
    These are reporting guards, not a statistical goodness-of-fit test.

    Use the same material, preparation, temperature, deformation direction,
    and strain window at each rate. The caller must ensure these conditions;
    recorded temperatures and strain limits are checked here. Temperatures
    may differ by at most 1 K to accommodate fluctuations in recorded means.
    A rate has units ns^-1; multiply a rate in s^-1 by 1e-9 before passing it.

    The target is required because neither empirical form defines a physical
    zero-rate modulus. Lower rates are not forced to yield lower moduli:
    negative fitted sensitivity is retained and marked unresolved. The
    default two-decade extrapolation limit is a reporting guard, not evidence
    that either model is valid over that distance.

    Raises:
        ValueError: The form, target rate, or extrapolation limit is invalid.
        AnalysisError: Fewer than three distinct rates, repeated rates,
            missing or invalid input values, or incompatible metadata.
            Pool independent replicas at each rate before calling this
            function, retaining their uncertainty in the pooled modulus.
    """
    if form not in STRAIN_RATE_FORMS:
        raise ValueError(f"form must be one of {STRAIN_RATE_FORMS}, got {form!r}.")
    target = float(target_rate_per_ns)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_rate_per_ns must be finite and positive.")
    maximum = float(max_extrapolation_decades)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("max_extrapolation_decades must be finite and nonnegative.")
    if len(fits) < 3:
        raise AnalysisError("Measure Young's modulus at three or more distinct rates.")

    for index, fit in enumerate(fits):
        measured = fit.strain_rate_per_ns
        if measured is None or not math.isfinite(measured) or measured <= 0.0:
            raise AnalysisError(
                f"Modulus at index {index} needs a recorded finite positive strain rate."
            )
        if not math.isfinite(fit.modulus_mpa) or fit.modulus_mpa <= 0.0:
            raise AnalysisError(
                f"Modulus at index {index} must be finite and positive."
            )
        if not math.isfinite(fit.standard_error_mpa) or fit.standard_error_mpa < 0.0:
            raise AnalysisError(
                f"Standard error at index {index} must be finite and nonnegative."
            )
        if not math.isfinite(fit.temperature_k) or fit.temperature_k <= 0.0:
            raise AnalysisError(
                f"Temperature at index {index} must be finite and positive."
            )
        if not math.isfinite(fit.strain_limit) or fit.strain_limit <= 0.0:
            raise AnalysisError(
                f"Strain limit at index {index} must be finite and positive."
            )

    rate = np.asarray([fit.strain_rate_per_ns for fit in fits], dtype=np.float64)
    if np.unique(rate).size != rate.size:
        raise AnalysisError(
            "Repeated strain rates are not independent rate measurements. "
            "Pool replicas at each rate first, retaining their uncertainty."
        )
    temperature = np.asarray([fit.temperature_k for fit in fits], dtype=np.float64)
    if float(np.max(temperature) - np.min(temperature)) > _TEMPERATURE_TOLERANCE_K:
        raise AnalysisError("Fit moduli at the same temperature (within 1 K).")
    limits = np.asarray([fit.strain_limit for fit in fits], dtype=np.float64)
    if not np.allclose(limits, limits[0], rtol=0.0, atol=1.0e-12):
        raise AnalysisError("Fit every modulus over the same strain limit.")

    order = np.argsort(rate)
    rate = rate[order]
    moduli = np.asarray([fit.modulus_mpa for fit in fits], dtype=np.float64)[order]
    errors = np.asarray([fit.standard_error_mpa for fit in fits], dtype=np.float64)[
        order
    ]
    log_rate = np.log10(rate)
    if np.unique(log_rate).size < 3:
        raise AnalysisError("The strain rates are too close to resolve in log space.")
    # Work in log space instead of taking products or ratios of extreme rates.
    reference_log = float(np.mean(log_rate))
    reference_rate = 10.0**reference_log
    x = log_rate - reference_log
    # Correct the last rounding-sized offset so the OLS intercept is centered.
    offset = float(np.mean(x))
    x = x - offset
    target_x = math.log10(target) - reference_log - offset
    notes = ["Empirical finite-rate estimate; not a zero-rate or equilibrium modulus."]
    refusals: list[str] = []

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        values = moduli if form == "log_linear" else np.log(moduli)
        input_errors = errors if form == "log_linear" else errors / moduli
        intercept, slope, fitted, prediction, error = _regression(
            x, values, input_errors, target_x
        )
        response_residual = _rms(values - fitted)
        response_error = _rms(input_errors)
        residual_to_error = (
            response_residual / response_error if response_error > 0.0 else None
        )
        rounding_tolerance = (
            64.0 * np.finfo(np.float64).eps * float(np.max(np.abs(values)))
        )
        # The tiny centering adjustment belongs to the regression, not Eref.
        intercept -= slope * offset
        if form == "log_linear":
            reference_modulus = intercept
            sensitivity = slope
            parameters = {
                "reference_modulus_mpa": reference_modulus,
                "sensitivity_mpa_per_decade": sensitivity,
            }
        else:
            reference_modulus = float(np.exp(intercept))
            prediction = float(np.exp(prediction))
            error = prediction * error
            fitted = np.exp(fitted)
            sensitivity = reference_modulus * slope
            parameters = {
                "reference_modulus_mpa": reference_modulus,
                "exponent": slope / math.log(10.0),
            }
        residual = _rms(moduli - fitted)
        modulus_scale = float(np.max(moduli))
        mean_modulus = float(np.mean(moduli / modulus_scale)) * modulus_scale
        relative_residual = residual / mean_modulus

    decades = max(
        float(log_rate[0]) - math.log10(target),
        math.log10(target) - float(log_rate[-1]),
        0.0,
    )
    if not all(fit.resolved for fit in fits):
        refusals.append("At least one input Young's modulus is unresolved.")
    if not math.isfinite(sensitivity) or sensitivity < 0.0:
        refusals.append("Rate sensitivity is negative or nonfinite.")
    if not math.isfinite(prediction) or prediction <= 0.0:
        refusals.append("The target modulus is nonpositive or nonfinite.")
    if not math.isfinite(error) or error > MAX_RELATIVE_STANDARD_ERROR * abs(
        prediction
    ):
        refusals.append("Target standard error exceeds 25% of the fitted modulus.")
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
    if relative_residual > MAX_RELATIVE_RATE_RESIDUAL:
        refusals.append(
            "The model RMS residual exceeds 10% of the mean measured modulus."
        )
    if decades > maximum:
        refusals.append(
            f"Target is {decades:.3g} decades outside measured rates, "
            f"exceeding the {maximum:.3g}-decade limit."
        )
    notes.extend(refusals)
    return StrainRateExtrapolation(
        form=form,
        modulus_mpa=float(prediction),
        target_rate_per_ns=target,
        temperature_k=float(np.mean(temperature)),
        strain_limit=float(limits[0]),
        strain_rate_per_ns=rate,
        moduli_mpa=moduli,
        standard_errors_mpa=errors,
        parameters=parameters,
        residual_mpa=residual,
        relative_residual=float(relative_residual),
        residual_to_error_ratio=residual_to_error,
        standard_error_mpa=float(error),
        sensitivity_mpa_per_decade=float(sensitivity),
        reference_rate_per_ns=reference_rate,
        n_rates=int(rate.size),
        extrapolation_decades=decades,
        resolved=not refusals,
        notes=tuple(notes),
    )
