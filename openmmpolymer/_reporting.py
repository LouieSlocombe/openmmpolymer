"""Shared conversion of analysis diagnostics to strict JSON values."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    """Use JSON null for an undefined diagnostic, retaining the resolved flag."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
