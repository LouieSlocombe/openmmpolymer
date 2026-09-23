"""Apparent tensile properties from recorded strain-controlled deformation.

Yield uses the intersection with an offset elastic line; breaking strength
requires a peak followed by sustained stress loss. These are criteria on
measured curves, not direct observations of permanent strain or covalent
fracture. The fixed-topology force fields used here cannot break polymer bonds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from ._validation import require_integer
from .elasticity import StressStrain, youngs_modulus


@dataclass(frozen=True)
class BreakingStrength:
    """A nominal-stress peak and whether a terminal stress loss supports it.

    ``peak_stress_mpa`` and ``strain_at_peak`` describe the largest observed
    nominal stress even when the curve is unresolved. ``strength_mpa`` is
    that peak only when the stress-loss criterion is met. The three failure
    fields then describe the first sample of the final sustained low-stress
    interval, with ``failure_bracket`` bounding the threshold crossing between
    two sampled strains. They are None for an unresolved curve.

    ``resolved`` refers only to the apparent tensile-strength criterion. It
    does not establish bond scission or predict an experimental breaking
    strength. Temperature and strain rate travel with the result because the
    apparent strength depends on both.
    """

    peak_stress_mpa: float
    strain_at_peak: float
    strength_mpa: float | None
    failure_strain: float | None
    failure_stress_mpa: float | None
    failure_bracket: tuple[float, float] | None
    resolved: bool
    temperature_k: float
    strain_rate_per_ns: float | None
    nominal_stress_mpa: npt.NDArray[np.float64]
    notes: tuple[str, ...]


def _finite_array(
    value: npt.NDArray[np.float64], *, name: str
) -> npt.NDArray[np.float64]:
    """Reject non-real and nonfinite values before deriving another array."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real, finite numbers.")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real, finite numbers.") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _nominal_tensile_stress(
    curve: StressStrain,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate a tensile curve and convert differential to nominal stress."""
    if curve.controlled != "strain":
        raise ValueError("Strength analysis requires a strain-controlled curve.")

    strain = _finite_array(curve.strain, name="strain")
    if strain.ndim != 1 or strain.size == 0:
        raise ValueError("strain must be a nonempty one-dimensional array.")
    if np.any(strain < 0.0) or np.any(np.diff(strain) <= 0.0):
        raise ValueError("strain must be nonnegative and strictly increasing.")

    axial = _finite_array(curve.stress_mpa, name="stress_mpa")
    lateral_stress = _finite_array(curve.lateral_stress_mpa, name="lateral_stress_mpa")
    lateral_strain = _finite_array(curve.lateral_strain, name="lateral_strain")
    if axial.shape != strain.shape:
        raise ValueError(
            "stress_mpa must have the same one-dimensional shape as strain."
        )
    expected = (strain.size, 2)
    if lateral_stress.shape != expected or lateral_strain.shape != expected:
        raise ValueError(
            "lateral_stress_mpa and lateral_strain must have shape (n, 2)."
        )
    if np.any(lateral_strain <= -1.0):
        raise ValueError(
            "lateral_strain must give strictly positive lateral stretches."
        )
    if not math.isfinite(curve.temperature_k) or curve.temperature_k <= 0.0:
        raise ValueError("temperature_k must be finite and greater than zero.")
    if curve.strain_rate_per_ns is not None and (
        not math.isfinite(curve.strain_rate_per_ns) or curve.strain_rate_per_ns < 0.0
    ):
        raise ValueError("strain_rate_per_ns must be finite and nonnegative, or None.")

    # Finite inputs can still overflow while forming area or stress. Reject
    # that explicitly rather than finding a peak in an infinite curve.
    with np.errstate(over="ignore", invalid="ignore"):
        area_ratio = np.prod(1.0 + lateral_strain, axis=1)
        nominal = (axial - lateral_stress.mean(axis=1)) * area_ratio
    if not np.all(np.isfinite(nominal)) or np.any(area_ratio <= 0.0):
        raise ValueError(
            "The nominal stress must remain finite and the lateral area "
            "must remain finite and positive."
        )

    return strain, nominal


def breaking_strength(
    curve: StressStrain,
    *,
    failure_fraction: float = 0.5,
    confirmation_steps: int = 3,
) -> BreakingStrength:
    """Read an apparent ultimate tensile strength from a deformation curve.

    Nominal stress is the differential Cauchy stress (axial stress minus
    mean lateral stress) multiplied by the current lateral area divided by
    the reference area: ``prod(1 + lateral_strain)``. This correction matters
    at large deformation, where the peak of true stress need not be the peak
    of nominal stress.

    A positive interior peak must be followed by at least
    ``confirmation_steps`` consecutive terminal samples at or below
    ``failure_fraction * peak``. A temporary stress drop followed by recovery
    is therefore insufficient. The first sample of this final low-stress
    interval is the reported failure strain; the preceding sample brackets
    the threshold crossing. No interpolation or uncertainty estimate is
    implied by that bracket.

    Args:
        curve: Finite, strain-controlled tensile samples with strictly
            increasing, nonnegative engineering strain.
        failure_fraction: Fraction of the peak defining stress loss, strictly
            between zero and one.
        confirmation_steps: At least two terminal samples below the threshold.

    Raises:
        ValueError: The curve is empty, malformed, nonfinite, not tensile or
            strain-controlled, or a numeric argument is outside its range.
        TypeError: ``confirmation_steps`` is not an integer.
    """
    if not math.isfinite(failure_fraction) or not 0.0 < failure_fraction < 1.0:
        raise ValueError("failure_fraction must be finite and between zero and one.")
    confirmation_steps = require_integer(
        confirmation_steps, name="confirmation_steps", minimum=2
    )
    strain, nominal = _nominal_tensile_stress(curve)

    peak_index = int(np.argmax(nominal))
    peak = float(nominal[peak_index])
    notes = [
        "This is an apparent stress-loss criterion, not evidence of covalent fracture.",
        "Fixed-topology force fields cannot predict bond scission.",
    ]
    failure_index: int | None = None
    if peak <= 0.0:
        notes.append("No positive tensile-stress peak was observed.")
    elif peak_index == 0 or peak_index == strain.size - 1:
        notes.append(
            "The observed peak is at a boundary; an interior peak is required."
        )
    else:
        # A positive peak is always above its fractional threshold, so this
        # array is nonempty and the start always has a preceding sample.
        last_above = int(np.flatnonzero(nominal > failure_fraction * peak)[-1])
        terminal_start = last_above + 1
        if strain.size - terminal_start >= confirmation_steps:
            failure_index = terminal_start
            notes.append(
                "A terminal interval of at least "
                f"{confirmation_steps} samples remained at or below "
                f"{failure_fraction:g} of the nominal-stress peak."
            )
        else:
            notes.append(
                "The curve has insufficient terminal samples at or below "
                f"{failure_fraction:g} of its peak; extend the deformation "
                "to distinguish sustained stress loss from a plateau or recovery."
            )

    resolved = failure_index is not None
    return BreakingStrength(
        peak_stress_mpa=peak,
        strain_at_peak=float(strain[peak_index]),
        strength_mpa=peak if resolved else None,
        failure_strain=float(strain[failure_index])
        if failure_index is not None
        else None,
        failure_stress_mpa=(
            float(nominal[failure_index]) if failure_index is not None else None
        ),
        failure_bracket=(
            (float(strain[failure_index - 1]), float(strain[failure_index]))
            if failure_index is not None
            else None
        ),
        resolved=resolved,
        temperature_k=curve.temperature_k,
        strain_rate_per_ns=curve.strain_rate_per_ns,
        nominal_stress_mpa=nominal,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class ElongationAtBreak:
    """Engineering elongation at the onset of confirmed terminal stress loss.

    ``strain_at_break`` is the dimensionless engineering strain
    ``(L - L0) / L0``; ``elongation_percent`` is 100 times that strain.
    ``break_stress_mpa`` is nominal stress at the same sampled point, and
    ``break_bracket`` bounds the threshold crossing between consecutive
    sampled engineering strains. These four values are None when unresolved.

    The peak stress, its strain, and the nominal-stress curve remain available
    for diagnostics. The peak strain is generally earlier than the reported
    break strain. Neither this operational stress-loss criterion nor the
    sampling bracket establishes a molecular rupture event or its uncertainty.
    """

    strain_at_break: float | None
    elongation_percent: float | None
    break_stress_mpa: float | None
    break_bracket: tuple[float, float] | None
    peak_stress_mpa: float
    strain_at_peak: float
    resolved: bool
    temperature_k: float
    strain_rate_per_ns: float | None
    nominal_stress_mpa: npt.NDArray[np.float64]
    notes: tuple[str, ...]


def elongation_at_break(
    curve: StressStrain,
    *,
    failure_fraction: float = 0.5,
    confirmation_steps: int = 3,
) -> ElongationAtBreak:
    """Calculate apparent elongation at break from a tensile deformation curve.

    Use the same nominal-stress conversion and sustained terminal stress-loss
    criterion as :func:`breaking_strength`. After a positive interior peak,
    at least ``confirmation_steps`` final samples must remain at or below
    ``failure_fraction`` of that peak. Report the engineering strain at the
    first sample of this final interval, together with its percentage and
    the preceding strain as a sampling bracket. No crossing is interpolated.

    The peak strain and the last recorded strain are not substitutes for an
    unresolved break strain. A plateau, recovery, boundary peak, or too few
    confirmation samples leaves the break values None while retaining the
    observed curve and peak. This is an apparent stress-loss criterion:
    fixed-topology force fields cannot model covalent bond scission.

    Args:
        curve: Finite, strain-controlled tensile samples with strictly
            increasing, nonnegative engineering strain.
        failure_fraction: Fraction of peak nominal stress defining the loss,
            strictly between zero and one.
        confirmation_steps: At least two terminal samples below the threshold.

    Raises:
        ValueError: The curve or a numeric argument is invalid; see
            :func:`breaking_strength` for the shared validation requirements.
        TypeError: ``confirmation_steps`` is not an integer.
    """
    strength = breaking_strength(
        curve,
        failure_fraction=failure_fraction,
        confirmation_steps=confirmation_steps,
    )
    strain_at_break = strength.failure_strain
    elongation_percent = (
        100.0 * strain_at_break if strain_at_break is not None else None
    )
    if elongation_percent is not None and not math.isfinite(elongation_percent):
        raise ValueError("elongation_percent must remain finite.")
    return ElongationAtBreak(
        strain_at_break=strain_at_break,
        elongation_percent=elongation_percent,
        break_stress_mpa=strength.failure_stress_mpa,
        break_bracket=strength.failure_bracket,
        peak_stress_mpa=strength.peak_stress_mpa,
        strain_at_peak=strength.strain_at_peak,
        resolved=strength.resolved,
        temperature_k=strength.temperature_k,
        strain_rate_per_ns=strength.strain_rate_per_ns,
        nominal_stress_mpa=strength.nominal_stress_mpa,
        notes=strength.notes,
    )


@dataclass(frozen=True)
class YieldStrength:
    """Offset proof stress from nominal tensile stress and engineering strain.

    ``strength_mpa`` and ``yield_strain`` interpolate the first crossing after
    the elastic fitting window. ``yield_bracket`` gives the two measured
    strains surrounding it, not a confidence interval. All three are None
    when the fit or crossing is unresolved. Finite fit diagnostics remain
    available even for unresolved curves; unavailable values are None.

    The offset line includes the fitted intercept, correcting for initial
    stress. This operational criterion does not establish permanent strain
    through an unloading experiment.
    """

    strength_mpa: float | None
    yield_strain: float | None
    yield_bracket: tuple[float, float] | None
    resolved: bool
    modulus_mpa: float | None
    intercept_mpa: float | None
    offset_strain: float
    fit_min_strain: float
    fit_max_strain: float
    fit_points: int
    fit_resolved: bool
    standard_error_mpa: float | None
    half_disagreement: float | None
    temperature_k: float
    strain_rate_per_ns: float | None
    nominal_stress_mpa: npt.NDArray[np.float64]
    notes: tuple[str, ...]


def yield_strength(
    curve: StressStrain,
    *,
    offset_strain: float = 0.002,
    fit_min_strain: float = 0.0,
    fit_max_strain: float = 0.02,
) -> YieldStrength:
    """Calculate a configurable offset yield strength (0.2% by default).

    Fit ``stress = E * strain + intercept`` to nominal tensile stress in
    the inclusive elastic window. The offset line is
    ``E * (strain - offset_strain) + intercept``. Interpolate its first
    crossing from below the measured curve to above it after that window.
    No crossing is extrapolated beyond the recorded data.

    The elastic fit must contain at least five points and pass the positive
    slope, relative standard error and half-window slope checks used by
    :func:`~openmmpolymer.elasticity.youngs_modulus`. A missing crossing, an
    unreliable fit or a crossing within the fit window is unresolved.
    A resolved result is an apparent proof stress at the recorded temperature
    and strain rate, not a direct measurement of irreversible deformation.

    Args:
        curve: Finite tensile samples with strictly increasing, nonnegative
            engineering strain and strain control.
        offset_strain: Positive engineering strain offset (0.002 means 0.2%).
        fit_min_strain: Nonnegative lower bound of the elastic fitting window.
        fit_max_strain: Upper bound, strictly greater than ``fit_min_strain``.

    Raises:
        ValueError: Invalid controls or malformed/nonfinite tensile data.
    """
    if not math.isfinite(offset_strain) or offset_strain <= 0.0:
        raise ValueError("offset_strain must be finite and greater than zero.")
    if not math.isfinite(fit_min_strain) or fit_min_strain < 0.0:
        raise ValueError("fit_min_strain must be finite and nonnegative.")
    if not math.isfinite(fit_max_strain) or fit_max_strain <= fit_min_strain:
        raise ValueError("fit_max_strain must be finite and above fit_min_strain.")
    strain, nominal = _nominal_tensile_stress(curve)
    inside = (strain >= fit_min_strain) & (strain <= fit_max_strain)
    count = int(np.count_nonzero(inside))
    fit_curve = replace(
        curve,
        strain=strain[inside],
        stress_mpa=nominal[inside],
        lateral_strain=np.zeros((count, 2), dtype=np.float64),
        lateral_stress_mpa=np.zeros((count, 2), dtype=np.float64),
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fit = youngs_modulus(fit_curve, strain_limit=fit_max_strain)
    notes = [
        f"Apparent {100.0 * offset_strain:g}% offset nominal yield strength; "
        "the offset is an operational definition, not an unloading measurement.",
        "Strength depends on temperature and strain rate.",
    ]
    strength: float | None = None
    crossing: float | None = None
    bracket: tuple[float, float] | None = None
    if not fit.resolved:
        notes.append(
            "Elastic fit unresolved: require at least five points, a positive "
            "slope, relative standard error below 25%, and half-window slope "
            "disagreement no greater than 50%. Adjust sampling or the fit window."
        )
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            offset_line = fit.modulus_mpa * (strain - offset_strain) + fit.intercept_mpa
            difference = nominal - offset_line
        if not np.all(np.isfinite(difference)):
            raise ValueError("The offset stress difference must remain finite.")
        # Least-squares roundoff can place an exact sampled intersection a
        # few ulps above the line, including at the final sample. Treat only
        # that numerical-scale residual as equality, not a physical tolerance.
        tolerance = (
            64 * np.finfo(np.float64).eps * np.maximum(abs(nominal), abs(offset_line))
        )
        difference[abs(difference) <= tolerance] = 0.0
        # Start at the final elastic sample to retain a crossing bracket that
        # straddles the requested window boundary. A curve already below the
        # offset line there cannot support a later crossing as its first yield.
        start = int(np.flatnonzero(inside)[-1])
        if difference[start] <= 0.0:
            notes.append(
                "The curve is already at or below the offset line at the end "
                "of the elastic fit; choose an earlier elastic window."
            )
        else:
            below = np.flatnonzero(difference[start + 1 :] <= 0.0)
            if below.size:
                right = start + 1 + int(below[0])
                left = right - 1
                # Scale first so two finite, opposite-signed residuals cannot
                # overflow their denominator during interpolation.
                scale = max(abs(difference[left]), abs(difference[right]))
                weight = float(
                    (difference[left] / scale)
                    / (difference[left] / scale - difference[right] / scale)
                )
                candidate = float(
                    (1.0 - weight) * strain[left] + weight * strain[right]
                )
                candidate_stress = float(
                    (1.0 - weight) * nominal[left] + weight * nominal[right]
                )
                if candidate <= fit_max_strain:
                    notes.append(
                        "The offset crossing lies within the elastic fit window; "
                        "choose an earlier elastic window."
                    )
                elif candidate_stress <= 0.0:
                    notes.append(
                        "The offset crossing is not at positive tensile stress."
                    )
                else:
                    crossing, strength = candidate, candidate_stress
                    bracket = (float(strain[left]), float(strain[right]))
                    notes.append(
                        "The yield point is linearly interpolated between measured "
                        "holds; its strain bracket is not a confidence interval."
                    )
            else:
                notes.append(
                    "No offset crossing was observed after the elastic window; "
                    "extend the deformation to resolve yield."
                )
    return YieldStrength(
        strength_mpa=strength,
        yield_strain=crossing,
        yield_bracket=bracket,
        resolved=strength is not None,
        modulus_mpa=fit.modulus_mpa if math.isfinite(fit.modulus_mpa) else None,
        intercept_mpa=fit.intercept_mpa if math.isfinite(fit.intercept_mpa) else None,
        offset_strain=offset_strain,
        fit_min_strain=fit_min_strain,
        fit_max_strain=fit_max_strain,
        fit_points=count,
        fit_resolved=fit.resolved,
        standard_error_mpa=fit.standard_error_mpa
        if math.isfinite(fit.standard_error_mpa)
        else None,
        half_disagreement=fit.half_disagreement
        if math.isfinite(fit.half_disagreement)
        else None,
        temperature_k=curve.temperature_k,
        strain_rate_per_ns=curve.strain_rate_per_ns,
        nominal_stress_mpa=nominal,
        notes=tuple(notes),
    )
