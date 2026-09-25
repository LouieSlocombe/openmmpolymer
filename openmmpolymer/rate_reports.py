"""Strict JSON and figures for a finite-rate measurement series."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from ._files import ReportFiles, json_value, write_json
from .plots import plot_rate_dependence
from .rate_dependence import RateReport


def write_rate_report(
    report: RateReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``<property>_rates.json`` and a figure for each available model.

    The JSON retains missing/censored observations, source paths, model
    refusals and uncertainty qualifications, and the difference between the
    two models at the target - sensitivity to the choice of model, not a
    confidence interval. Nonfinite diagnostics become JSON null. The default
    directory is the first run's ``analysis`` folder; reports without run
    paths require an explicit output directory.
    """
    if output_dir is None and not report.run_dirs:
        raise ValueError("output_dir is required when the report has no run_dirs.")
    if not re.fullmatch(r"[A-Za-z0-9]+", figure_format):
        raise ValueError(
            "figure_format must be a filename extension such as png or svg."
        )
    directory = (
        Path(report.run_dirs[0]) / "analysis"
        if output_dir is None
        else Path(output_dir)
    )
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", report.property.name).strip("_")
    if not name:
        raise ValueError("The property name must contain a filename-safe character.")
    directory.mkdir(parents=True, exist_ok=True)
    record = asdict(report)
    record["model_difference"] = (
        abs(report.log_linear.value - report.power_law.value)
        if report.log_linear is not None and report.power_law is not None
        else None
    )
    record["uncertainty_description"] = (
        "standard_error describes uncertainty in the fitted mean, including input "
        "errors and excess residual scatter; pooling retains between-replica sample "
        "standard deviation as a conservative floor. Unknown input uncertainty "
        "without measurable replica spread yields null target uncertainty. "
        "Errors exclude model choice, correlated runs and systematic simulation errors."
    )
    path = directory / f"{name}_rates.json"
    write_json(path, json_value(record))
    written: list[str] = []
    if figures:
        for fit in (report.log_linear, report.power_law):
            if fit is not None:
                figure_path = directory / f"{name}_rate_{fit.form}.{figure_format}"
                plot_rate_dependence(fit).savefig(figure_path, bbox_inches="tight")
                written.append(str(figure_path))
    return ReportFiles(json=str(path), figures=tuple(written))
