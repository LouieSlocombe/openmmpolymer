"""Generic reports expose units, errors, extrapolation and missing data."""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.rate_dependence import (
    RateObservation,
    RateProperty,
    RateReport,
    analyse_rate_observations,
)
from openmmpolymer.rate_reports import plot_rate_dependence, write_rate_report


def report(*, target: float = 0.01, unknown: bool = False) -> RateReport:
    property = RateProperty("yield_strength", "Yield strength", "MPa", "strain/ns")
    source = [
        RateObservation(
            rate,
            value,
            None if unknown else 10.0,
            True,
            source=f"run-{index}",
            notes=("Finite rate.",),
        )
        for index, (rate, value) in enumerate([(0.1, 900), (1.0, 1000), (10.0, 1100)])
    ]
    return analyse_rate_observations(source, property=property, target_rate=target)


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_plot_preserves_numeric_fit_target_errorbars_and_units(form: str) -> None:
    fit = report().log_linear if form == "log_linear" else report().power_law
    assert fit is not None
    figure = plot_rate_dependence(fit)
    axis: Any = figure.axes[0]
    line = next(line for line in axis.get_lines() if line.get_label() == f"{form} fit")
    np.testing.assert_allclose(line.get_ydata(), fit.predict(line.get_xdata()))
    measured, target = axis.containers
    np.testing.assert_allclose(measured.lines[0].get_ydata(), fit.values)
    np.testing.assert_allclose(target.lines[0].get_ydata(orig=False), [fit.value])
    segments = target.lines[2][0].get_segments()
    np.testing.assert_allclose(
        segments[0][:, 1],
        [fit.value - fit.standard_error, fit.value + fit.standard_error],
    )
    assert axis.get_xlabel() == "Rate (strain/ns)"
    assert axis.get_ylabel() == "Yield strength (MPa)"
    assert "resolved" in axis.get_title()
    data = io.BytesIO()
    figure.savefig(data, format="png")
    assert data.getvalue().startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "target, expected", [(0.01, (0.01, 0.1)), (100.0, (10.0, 100.0))]
)
def test_plot_shades_unsupported_interval_on_either_side(
    target: float, expected: tuple[float, float]
) -> None:
    fit = report(target=target).log_linear
    assert fit is not None
    axis: Any = plot_rate_dependence(fit).axes[0]
    patch = axis.patches[0]
    vertices = patch.get_path().transformed(patch.get_transform() - axis.transData)
    np.testing.assert_allclose(
        [vertices.vertices[:, 0].min(), vertices.vertices[:, 0].max()], expected
    )


def test_interpolation_has_no_unsupported_region() -> None:
    fit = report(target=0.5).log_linear
    assert fit is not None
    assert not plot_rate_dependence(fit).axes[0].patches


def test_unknown_error_and_distant_target_are_visibly_unresolved() -> None:
    fit = report(target=1e-8, unknown=True).log_linear
    assert fit is not None
    axis: Any = plot_rate_dependence(fit).axes[0]
    assert "not resolved" in axis.get_title()
    assert "7.0 decades" in axis.get_title()
    labels = axis.get_legend_handles_labels()[1]
    assert sum("SE unavailable" in label for label in labels) == 2


def test_plot_with_nonfinite_target_keeps_target_position() -> None:
    fit = report().log_linear
    assert fit is not None
    axis: Any = plot_rate_dependence(replace(fit, value=math.inf, resolved=False)).axes[
        0
    ]
    assert any(
        "target estimate not finite" in label
        for label in axis.get_legend_handles_labels()[1]
    )


def test_strict_json_preserves_measurements_model_difference_and_units(
    tmp_path: Path,
) -> None:
    source = report()
    files = write_rate_report(source, tmp_path)
    assert Path(files.json).name == "yield_strength_rates.json"
    raw = Path(files.json).read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    data = json.loads(raw)
    assert data["property"]["value_unit"] == "MPa"
    assert data["property"]["rate_unit"] == "strain/ns"
    assert data["observations"][0]["source"] == "run-0"
    assert len(data["observations"]) == 3
    assert data["model_difference"] == pytest.approx(
        abs(data["log_linear"]["value"] - data["power_law"]["value"])
    )
    assert data["target_rate"] == 0.01
    assert len(files.figures) == 2
    assert all(Path(path).read_bytes().startswith(b"\x89PNG") for path in files.figures)


def test_json_unknown_errors_are_null_and_models_remain_unresolved(
    tmp_path: Path,
) -> None:
    files = write_rate_report(report(unknown=True), tmp_path, figures=False)
    data = json.loads(Path(files.json).read_text())
    assert data["log_linear"]["standard_error"] is None
    assert data["log_linear"]["standard_errors"] == [None] * 3
    assert not data["log_linear"]["resolved"]
    assert files.figures == ()


def test_censored_report_has_no_fake_figures_or_prediction(tmp_path: Path) -> None:
    source = report()
    censored = analyse_rate_observations(
        [
            *source.observations,
            RateObservation(100.0, None, None, False, notes=("No failure.",)),
        ],
        property=source.property,
        target_rate=0.01,
    )
    files = write_rate_report(censored, tmp_path)
    data = json.loads(Path(files.json).read_text())
    assert data["observations"][-1]["value"] is None
    assert data["observations"][-1]["notes"] == ["No failure."]
    assert data["log_linear"] is None and data["power_law"] is None
    assert data["model_difference"] is None
    assert data["target_rate"] == 0.01
    assert files.figures == ()


def test_default_directory_and_svg_format(tmp_path: Path) -> None:
    source = replace(report(), run_dirs=(str(tmp_path),))
    files = write_rate_report(source, figure_format="svg")
    assert Path(files.json).parent == tmp_path / "analysis"
    assert len(files.figures) == 2
    assert all("<svg" in Path(path).read_text() for path in files.figures)


def test_missing_output_directory_and_unsafe_format_fail_before_writing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="output_dir"):
        write_rate_report(report())
    with pytest.raises(ValueError, match="figure_format"):
        write_rate_report(report(), tmp_path, figure_format="../png")
    assert list(tmp_path.iterdir()) == []
