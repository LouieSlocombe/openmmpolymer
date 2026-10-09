"""Strict JSON and figures for a finite-rate measurement series."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from ._files import ReportFiles, analysis_directory, figure_stem, write_report
from .plots import plot_rate_dependence
from .rate_dependence import RateReport

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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
    directory = analysis_directory(
        report.run_dirs[0] if report.run_dirs else "", output_dir
    )
    name = figure_stem(report.property.name)
    if not name:
        raise ValueError("The property name must contain a filename-safe character.")
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
    return write_report(
        directory,
        f"{name}_rates.json",
        record,
        _figures(report, name) if figures else (),
        figure_format,
    )


def _figures(report: RateReport, name: str) -> Iterator[tuple[str, Figure]]:
    """A figure for each available rate model."""
    for fit in (report.log_linear, report.power_law):
        if fit is not None:
            yield f"{name}_rate_{fit.form}", plot_rate_dependence(fit)
