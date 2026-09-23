"""Apparent tensile strength from a recorded strain-controlled deformation.

The peak of nominal stress is only reported as a strength when a sustained
loss of stress follows it. This is a criterion on a measured curve, not a
model of covalent fracture: the fixed-topology force fields used here cannot
break polymer bonds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._validation import require_integer
from .elasticity import StressStrain


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
    if curve.controlled != "strain":
        raise ValueError("Breaking strength requires a strain-controlled curve.")

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
