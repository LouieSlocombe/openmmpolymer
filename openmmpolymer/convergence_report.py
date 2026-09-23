"""JSON and figures retaining observation-window convergence qualifications."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ._reporting import json_value
from .convergence import (
    ConvergenceReport,
    RelaxationWindowConvergence,
    WindowConvergence,
)
from .protocols import _write_atomically
from .structural_convergence import StructuralWindowConvergence
from .tg import ReportFiles


def plot_window_convergence(result: WindowConvergence) -> Figure:
    """Show prefix means/errors and disjoint tail-block estimates side by side."""
    figure = Figure(figsize=(9.0, 4.0), dpi=150)
    prefix, blocks = figure.subplots(1, 2)
    for axis, positions, values, errors, label in (
        (
            prefix,
            [item.duration_ps for item in result.windows],
            [item.mean for item in result.windows],
            [item.standard_error for item in result.windows],
            "overlapping prefixes",
        ),
        (
            blocks,
            [1.0, 2.0, 3.0],
            result.block_means,
            result.block_standard_errors,
            "disjoint tail blocks",
        ),
    ):
        x, y, error = np.asarray(positions), np.asarray(values), np.asarray(errors)
        finite = np.isfinite(y)
        known = finite & np.isfinite(error)
        if np.any(known):
            axis.errorbar(
                x[known],
                y[known],
                yerr=error[known],
                marker="o",
                linestyle="none",
                capsize=3,
                color="#1f4e79",
                label="mean (1 SE)",
            )
        unknown = finite & ~known
        if np.any(unknown):
            axis.plot(
                x[unknown], y[unknown], "x", color="#b03a2e", label="SE unavailable"
            )
        axis.set_title(label, fontsize=9)
        axis.set_ylabel(f"{result.property_name} ({result.value_unit})")
        if np.any(finite):
            axis.legend(fontsize=7, frameon=False)
    prefix.set_xlabel("Observation duration (ps)")
    blocks.set_xlabel("Late-time block")
    blocks.set_xticks([1, 2, 3])
    verdict = "resolved" if result.resolved else "not resolved"
    figure.suptitle(
        f"Window stability: {verdict}; tolerance {result.relative_tolerance:.0%}",
        fontsize=10,
    )
    figure.tight_layout()
    return figure


def plot_relaxation_convergence(result: RelaxationWindowConvergence) -> Figure:
    """Show model parameters as the observed decay window increases."""
    figure = Figure(figsize=(9.0, 7.0), dpi=150)
    axes = figure.subplots(2, 2).reshape(-1)
    durations = np.asarray([item.duration_ps for item in result.windows])
    for axis, metric in zip(axes, result.metrics.values(), strict=True):
        finite = np.isfinite(metric.values)
        axis.plot(durations[finite], metric.values[finite], "o-", color="#1f4e79")
        if np.any(finite):
            scale = float(np.max(np.abs(metric.values[finite])))
            if scale > 0.0 and float(np.ptp(metric.values[finite])) < 1e-8 * scale:
                centre = float(np.mean(metric.values[finite]))
                axis.set_ylim(centre - 0.05 * scale, centre + 0.05 * scale)
        axis.ticklabel_format(axis="y", useOffset=False)
        axis.set_xlabel("Observation duration (ps)")
        axis.set_ylabel(f"{metric.property_name} ({metric.value_unit})")
        verdict = "resolved" if metric.resolved else "not resolved"
        difference = (
            f"{metric.relative_change:.1%}"
            if math.isfinite(metric.relative_change)
            else "unavailable"
        )
        axis.set_title(f"{verdict}; change {difference}", fontsize=9)
    figure.suptitle(
        "Relaxation window sensitivity; overlapping refits are not error bars",
        fontsize=10,
    )
    figure.tight_layout()
    return figure


def plot_structural_convergence(result: StructuralWindowConvergence) -> Figure:
    """Show structural prefix estimates without treating their spread as SE."""
    rows = max(1, math.ceil(len(result.parameters) / 2))
    figure = Figure(figsize=(10.0, 2.8 * rows), dpi=150)
    axes = figure.subplots(rows, 2, squeeze=False).reshape(-1)
    durations = np.asarray([item.duration_ps for item in result.windows])
    for axis, metric in zip(axes, result.parameters.values(), strict=False):
        values = np.asarray(metric.values, dtype=np.float64)
        finite = np.isfinite(values)
        valid = finite & np.asarray(metric.valid, dtype=bool)
        axis.plot(
            durations[finite], values[finite], "--", color="#7f8c8d", linewidth=0.7
        )
        axis.plot(
            durations[valid], values[valid], "o", color="#1f4e79", label="valid window"
        )
        if np.any(finite & ~valid):
            axis.plot(
                durations[finite & ~valid],
                values[finite & ~valid],
                "x",
                color="#b03a2e",
                label="unresolved window",
            )
        axis.set_xlabel("Observation duration (ps)")
        axis.set_ylabel(f"{metric.property_name} ({metric.value_unit})", fontsize=8)
        verdict = "resolved" if metric.resolved else "not resolved"
        axis.set_title(verdict, fontsize=9)
        axis.legend(fontsize=7, frameon=False)
    for axis in axes[len(result.parameters) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Structural window sensitivity; curves and tail blocks retained in JSON",
        fontsize=10,
    )
    figure.tight_layout()
    return figure


def write_convergence_report(
    report: ConvergenceReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write strict ``convergence.json`` and figures for available diagnostics."""
    if not re.fullmatch(r"[A-Za-z0-9]+", figure_format):
        raise ValueError(
            "figure_format must be a filename extension such as png or svg."
        )
    directory = (
        Path(report.run_dir) / "analysis" if output_dir is None else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = asdict(report)
    record["interpretation"] = (
        "Window stability is conditional on recorded observables and sampled times. "
        "Overlapping prefixes are not independent replicas; stationary traces also "
        "require disjoint-block stability and autocorrelation-adjusted sample counts. "
        "Relaxation parameters require observed decay/plateau and resolved underlying "
        "models. This does not establish equilibrium or eliminate systematic bias."
    )
    path = directory / "convergence.json"
    _write_atomically(
        path, json.dumps(json_value(record), indent=2, allow_nan=False) + "\n"
    )
    written: list[str] = []
    if figures:
        for name, result in report.results.items():
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
            figure_path = directory / f"convergence_{safe}.{figure_format}"
            plot_window_convergence(result).savefig(figure_path, bbox_inches="tight")
            written.append(str(figure_path))
        if report.relaxation is not None:
            figure_path = directory / f"convergence_relaxation.{figure_format}"
            plot_relaxation_convergence(report.relaxation).savefig(
                figure_path, bbox_inches="tight"
            )
            written.append(str(figure_path))
        if report.structural is not None:
            figure_path = directory / f"convergence_structural.{figure_format}"
            plot_structural_convergence(report.structural).savefig(
                figure_path, bbox_inches="tight"
            )
            written.append(str(figure_path))
    return ReportFiles(json=str(path), figures=tuple(written))
