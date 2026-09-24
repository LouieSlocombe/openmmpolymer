"""Unit-aware figures and strict JSON for general finite-rate measurements."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ._files import ReportFiles, json_value, write_json
from .rate_dependence import RateExtrapolation, RateReport


def plot_rate_dependence(fit: RateExtrapolation) -> Figure:
    """Plot pooled measurements, target errors, and unsupported rate intervals.

    Error bars show one propagated standard error, with conservative replica
    spread retained when pooling. Unknown errors are labelled explicitly.
    Figures are created directly without registering pyplot global state.
    """
    figure = Figure(figsize=(7.0, 4.5), dpi=150)
    axis = figure.subplots()
    minimum, maximum = float(fit.rates.min()), float(fit.rates.max())
    target = fit.target_rate
    if target < minimum or target > maximum:
        boundary = minimum if target < minimum else maximum
        axis.axvspan(
            min(target, boundary),
            max(target, boundary),
            color="#b03a2e",
            alpha=0.15,
            label="extrapolated interval",
        )
    span = np.geomspace(min(target, minimum), max(target, maximum), 200)
    predicted = fit.predict(span)
    axis.plot(
        span,
        np.where(np.isfinite(predicted), predicted, np.nan),
        linewidth=0.9,
        linestyle="--",
        color="#7f8c8d",
        label=f"{fit.form} fit",
    )
    known_errors = np.isfinite(fit.standard_errors)
    for mask, label in (
        (known_errors, "measured (1 SE; replica spread retained)"),
        (~known_errors, "measured (SE unavailable)"),
    ):
        if np.any(mask):
            axis.errorbar(
                fit.rates[mask],
                fit.values[mask],
                yerr=fit.standard_errors[mask] if np.all(known_errors[mask]) else None,
                marker="o",
                markersize=4.0,
                linestyle="none",
                elinewidth=0.9,
                capsize=2,
                color="#1f4e79",
                label=label,
            )
    if math.isfinite(fit.value):
        has_error = math.isfinite(fit.standard_error)
        error_label = "1 SE" if has_error else "SE unavailable"
        axis.errorbar(
            [target],
            [fit.value],
            yerr=[fit.standard_error] if has_error else None,
            marker="*",
            markersize=9.0,
            linestyle="none",
            elinewidth=0.9,
            capsize=2,
            color="#b03a2e",
            label=(
                f"target = {fit.value:.4g} {fit.property.value_unit} "
                f"at {target:.3g} {fit.property.rate_unit} ({error_label})"
            ),
        )
    else:
        axis.axvline(
            target,
            color="#b03a2e",
            linewidth=0.9,
            label=f"target estimate not finite at {target:.3g} {fit.property.rate_unit}",
        )
    axis.set_xscale("log")
    axis.set_xlabel(f"Rate ({fit.property.rate_unit})")
    unit = f" ({fit.property.value_unit})" if fit.property.value_unit else ""
    axis.set_ylabel(f"{fit.property.label}{unit}")
    verdict = "resolved" if fit.resolved else "not resolved"
    axis.set_title(
        f"{fit.property.label}: {fit.form}\n"
        f"{fit.extrapolation_decades:.1f} decades extrapolated, {verdict}",
        fontsize=9,
    )
    axis.legend(fontsize=7, frameon=False)
    return figure


def write_rate_report(
    report: RateReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``<property>_rates.json`` and figures for each available model.

    The JSON retains missing/censored observations, source paths, model
    refusals and uncertainty qualifications. Nonfinite diagnostics become
    JSON null. The default directory is the first run's ``analysis`` folder;
    reports without run paths require an explicit output directory.
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
