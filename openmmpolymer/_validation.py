"""Small, shared runtime validators for public numeric arguments.

Nothing here is public. Every value that crosses into this package as a number
goes through one of these first, so a typo or a negative temperature fails at
the call site rather than a hundred picoseconds into a run.

Public signatures in this package take plain floats with the unit in the name
(``temperature_k``, ``duration_ps``). These helpers additionally accept an
``openmm.unit.Quantity``, because callers coming from ``openmmnqe`` will reach
for one, and convert it to the float the rest of the package works in.
"""

from __future__ import annotations

import math
from typing import Any, cast


def _as_float(value: object, expected_unit: Any, *, name: str) -> float:
    """Convert a quantity or a bare number to a scalar in *expected_unit*.

    Args:
        value: An ``openmm.unit.Quantity`` or anything ``float()`` accepts.
        expected_unit: The unit a quantity is converted into. Ignored for a
            bare number, which is taken to already be in that unit.
        name: Parameter name, used in the error message.

    Returns:
        The value as a float in *expected_unit*.

    Raises:
        TypeError: The value is neither a compatible quantity nor a number.
    """
    # Imported lazily so this module stays usable in the pure-arithmetic tests
    # that do not want an OpenMM import.
    from openmm import unit

    try:
        if unit.is_quantity(value):
            return float(cast(Any, value).value_in_unit(expected_unit))
        return float(cast(Any, value))
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            f"{name}={value!r} is not a number or a quantity in "
            f"{expected_unit}. Pass a plain float in {expected_unit}, or an "
            "openmm.unit.Quantity that converts to it."
        ) from error


def require_finite(value: object, expected_unit: Any, *, name: str) -> float:
    """Return *value* as a float, rejecting NaN and infinity.

    Args:
        value: An ``openmm.unit.Quantity`` or a bare number.
        expected_unit: The unit the result is expressed in.
        name: Parameter name, used in the error message.

    Returns:
        The finite value as a float.

    Raises:
        ValueError: The value is NaN or infinite.
    """
    number = _as_float(value, expected_unit, name=name)
    if not math.isfinite(number):
        raise ValueError(f"{name}={value!r} must be finite.")
    return number


def require_positive(value: object, expected_unit: Any, *, name: str) -> float:
    """Return *value* as a strictly positive float.

    Args:
        value: An ``openmm.unit.Quantity`` or a bare number.
        expected_unit: The unit the result is expressed in.
        name: Parameter name, used in the error message.

    Returns:
        The positive value as a float.

    Raises:
        ValueError: The value is not strictly positive, NaN or infinite.
    """
    number = require_finite(value, expected_unit, name=name)
    if number <= 0.0:
        raise ValueError(f"{name}={value!r} must be greater than zero.")
    return number


def require_integer(value: object, *, name: str, minimum: int = 1) -> int:
    """Return *value* as an integer no smaller than *minimum*.

    Args:
        value: Anything ``int()`` accepts without losing information.
        name: Parameter name, used in the error message.
        minimum: Smallest value accepted.

    Returns:
        The value as an int.

    Raises:
        TypeError: The value is not an integer.
        ValueError: The value is below *minimum*.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name}={value!r} must be an int.")
    if value < minimum:
        raise ValueError(f"{name}={value!r} must be at least {minimum}.")
    return value


def require_choice(value: str, valid: tuple[str, ...], *, name: str) -> str:
    """Return *value* if it is one of *valid*, else raise naming the options.

    Args:
        value: The candidate.
        valid: Every accepted value.
        name: Parameter name, used in the error message.

    Returns:
        The validated value.

    Raises:
        ValueError: The value is not in *valid*.
    """
    if value not in valid:
        raise ValueError(f"{name}={value!r} is not one of {', '.join(sorted(valid))}.")
    return value
