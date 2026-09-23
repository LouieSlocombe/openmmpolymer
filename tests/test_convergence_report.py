"""Window reports preserve refusal flags, nonfinite errors and plotted means."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.convergence import (
    ConvergenceReport,
    relaxation_window_convergence,
    time_window_convergence,
)
from openmmpolymer.convergence_report import (
    plot_relaxation_convergence,
    plot_structural_convergence,
    plot_window_convergence,
    write_convergence_report,
)
from openmmpolymer.structural_convergence import structural_window_convergence

from .test_convergence import stationary
from .test_relaxation import planted
from .test_structural_convergence import OPTIONS, _rods


def test_prefix_and_disjoint_block_figure_retains_errors_and_verdict() -> None:
    data = stationary()
    result = time_window_convergence(
        np.arange(data.size), data, property_name="density", value_unit="g/cm^3"
    )
    figure = plot_window_convergence(result)
    prefix: Any = figure.axes[0]
    blocks: Any = figure.axes[1]
    np.testing.assert_allclose(
        prefix.containers[0].lines[0].get_xdata(),
        [item.duration_ps for item in result.windows],
    )
    np.testing.assert_allclose(
        prefix.containers[0].lines[0].get_ydata(),
        [item.mean for item in result.windows],
    )
    np.testing.assert_allclose(
        blocks.containers[0].lines[0].get_ydata(), result.block_means
    )
    segments = prefix.containers[0].lines[2][0].get_segments()
    for segment, window in zip(segments, result.windows, strict=True):
        np.testing.assert_allclose(
            segment[:, 1],
            [window.mean - window.standard_error, window.mean + window.standard_error],
        )
    assert "overlapping" in prefix.get_title()
    assert "disjoint" in blocks.get_title()
    output = io.BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_constant_snapshot_plot_labels_unknown_uncertainty() -> None:
    result = time_window_convergence(
        [0.0], [10.0], property_name="radius", value_unit="nm"
    )
    figure = plot_window_convergence(result)
    assert "SE unavailable" in figure.axes[0].get_legend_handles_labels()[1]
    assert any("not resolved" in item.get_text() for item in figure.texts)


def test_relaxation_parameter_plot_keeps_unobserved_tails_unresolved() -> None:
    times = np.geomspace(0.1, 10000.0, 140)
    result = relaxation_window_convergence(
        planted(times, 1000.0 * np.exp(-times / 10000.0))
    )
    figure = plot_relaxation_convergence(result)
    assert len(figure.axes) == 4
    assert all("not resolved" in axis.get_title() for axis in figure.axes)
    output = io.BytesIO()
    figure.savefig(output, format="svg")
    assert b"<svg" in output.getvalue()


def test_strict_json_retains_unknown_error_and_qualifications(tmp_path: Path) -> None:
    result = time_window_convergence(
        [0.0], [10.0], property_name="radius", value_unit="nm"
    )
    report = ConvergenceReport(
        str(tmp_path), "hold", {"radius": result}, None, ("Only a snapshot.",)
    )
    files = write_convergence_report(report, figures=False)
    assert Path(files.json) == tmp_path / "analysis" / "convergence.json"
    raw = Path(files.json).read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    data = json.loads(raw)
    assert data["results"]["radius"]["windows"][-1]["standard_error"] is None
    assert not data["results"]["radius"]["resolved"]
    assert data["relaxation"] is None
    assert data["notes"] == ["Only a snapshot."]
    assert "not independent replicas" in data["interpretation"]
    assert files.figures == ()


def test_report_writes_stationary_and_relaxation_figures(tmp_path: Path) -> None:
    data = stationary()
    result = time_window_convergence(
        np.arange(data.size), data, property_name="density", value_unit="g/cm^3"
    )
    times = np.geomspace(0.1, 10000.0, 140)
    relaxation = relaxation_window_convergence(
        planted(times, 1000.0 * np.exp(-times / 50.0))
    )
    report = ConvergenceReport(
        str(tmp_path), "hold", {"density": result}, relaxation, ()
    )
    files = write_convergence_report(report, tmp_path / "report", figure_format="svg")
    assert len(files.figures) == 2
    assert all("<svg" in Path(path).read_text() for path in files.figures)
    payload = json.loads(Path(files.json).read_text())
    assert (
        payload["relaxation"]["metrics"]["kww_viscosity_pa_s"]["value_unit"] == "Pa s"
    )


def test_unsafe_format_rejected_before_writing(tmp_path: Path) -> None:
    report = ConvergenceReport(str(tmp_path), "hold", {}, None, ())
    with pytest.raises(ValueError, match="figure_format"):
        write_convergence_report(report, figure_format="../svg")
    assert not (tmp_path / "analysis").exists()


def test_structural_report_preserves_missing_metrics_curves_and_unresolved_figures(
    tmp_path: Path,
) -> None:
    structural = structural_window_convergence(_rods(), range(5), **OPTIONS)
    figure = plot_structural_convergence(structural)
    assert all(
        "not resolved" in axis.get_title() for axis in figure.axes if axis.get_visible()
    )
    report = ConvergenceReport(str(tmp_path), "hold", {}, None, (), structural)
    files = write_convergence_report(report, tmp_path)
    assert len(files.figures) == 1
    assert Path(files.figures[0]).read_bytes().startswith(b"\x89PNG")
    payload = json.loads(Path(files.json).read_text())
    assert (
        payload["structural"]["parameters"]["diffusion_coefficient_cm2_s"]["values"]
        == [None] * 4
    )
    assert "radial_distribution" in payload["structural"]["windows"][0]["curves"]
