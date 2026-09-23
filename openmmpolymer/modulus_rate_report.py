"""Write measured and extrapolated Young's moduli with their qualifications."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .modulus_rates import ModulusRateReport
from .protocols import _write_atomically
from .tg import ReportFiles


def _json_value(value: Any) -> Any:
    """Use JSON null for an undefined diagnostic, retaining the resolved flag."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_modulus_rate_report(
    report: ModulusRateReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``modulus_rates.json`` and a plot for each empirical rate model.

    Defaults to the first measured run's ``analysis`` directory. The JSON
    retains every per-rate fit, both model predictions and diagnostics, and
    their difference at the target. This difference is model sensitivity,
    not a confidence interval. Nonfinite diagnostics are written as null.
    """
    from .plots import plot_strain_rate

    directory = (
        Path(report.run_dirs[0]) / "analysis"
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = asdict(report)
    record["model_difference_mpa"] = abs(
        report.log_linear.modulus_mpa - report.power_law.modulus_mpa
    )
    record["uncertainty_description"] = (
        "standard_error_mpa describes uncertainty in the fitted mean, including "
        "input errors and residual scatter; it excludes uncertainty from choosing "
        "the rate model or changing material relaxation mechanisms."
    )
    json_path = directory / "modulus_rates.json"
    _write_atomically(
        json_path,
        json.dumps(_json_value(record), indent=2, allow_nan=False) + "\n",
    )
    written: list[str] = []
    if figures:
        for fit in (report.log_linear, report.power_law):
            path = directory / f"modulus_rate_{fit.form}.{figure_format}"
            figure = plot_strain_rate(fit)
            figure.savefig(path, bbox_inches="tight")
            written.append(str(path))
    return ReportFiles(json=str(json_path), figures=tuple(written))
