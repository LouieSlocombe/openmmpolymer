"""Shared validation and durable records for imposed-rate workflows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from ._validation import require_positive
from .protocols import _write_atomically


def validate_hold_times(
    values: Sequence[float], *, name: str = "hold_times_ps"
) -> tuple[float, ...]:
    """Require three distinct positive holds, preserving their requested order."""
    holds = tuple(require_positive(value, None, name=name) for value in values)
    if len(holds) < 3 or any(
        math.isclose(first, second, rel_tol=1e-8)
        for first, second in pairwise(sorted(holds))
    ):
        raise ValueError(f"{name} needs at least three distinct positive holds.")
    return holds


def validate_extrapolation_limit(value: float) -> None:
    """Reject an undefined or negative extrapolation allowance."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("max_extrapolation_decades must be finite and nonnegative.")


def state_digest(path: str | Path) -> str:
    """Fingerprint a saved state used as the common source of rate branches."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_workflow(path: Path, record: dict[str, Any]) -> None:
    """Keep strict JSON settings durable before any resumable stage starts."""
    _write_atomically(path, json.dumps(record, indent=2, allow_nan=False) + "\n")
