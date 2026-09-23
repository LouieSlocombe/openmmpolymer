"""The rate figure distinguishes measured stiffness from extrapolated values."""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest

from openmmpolymer.elasticity import ElasticModulus
from openmmpolymer.plots import plot_strain_rate
from openmmpolymer.strain_rate import strain_rate_extrapolation


def _moduli(*, form: str = "log_linear") -> list[ElasticModulus]:
    rates = [0.01, 0.1, 1.0]
    values = (
        [800.0, 900.0, 1000.0]
        if form == "log_linear"
        else [800.0 * (rate / rates[0]) ** 0.1 for rate in rates]
    )
    return [
        ElasticModulus(
            modulus_mpa=value,
            intercept_mpa=0.0,
            strain_limit=0.015,
            n_points=8,
            residual_mpa=0.1,
            standard_error_mpa=10.0,
            half_disagreement=0.0,
            temperature_k=298.15,
            strain_rate_per_ns=rate,
            resolved=True,
        )
        for rate, value in zip(rates, values, strict=True)
    ]


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_rate_figure_preserves_data_target_and_fit(form: str) -> None:
    """Both supported relations are plotted using the analysis prediction."""
    result = strain_rate_extrapolation(
        _moduli(form=form), target_rate_per_ns=0.001, form=form
    )
    figure = plot_strain_rate(result)
    axis = figure.axes[0]
    fit = next(line for line in axis.get_lines() if line.get_label() == f"{form} fit")
    np.testing.assert_allclose(fit.get_ydata(), result.predict(fit.get_xdata()))
    assert fit.get_xdata()[0] == pytest.approx(0.001)
    assert fit.get_xdata()[-1] == pytest.approx(1.0)
    measured, target = axis.containers
    np.testing.assert_allclose(measured.lines[0].get_xdata(), result.strain_rate_per_ns)
    np.testing.assert_allclose(measured.lines[0].get_ydata(), result.moduli_mpa)
    np.testing.assert_allclose(target.lines[0].get_xdata(orig=False), [0.001])
    np.testing.assert_allclose(
        target.lines[0].get_ydata(orig=False), [result.modulus_mpa]
    )
    for container, values, errors in (
        (measured, result.moduli_mpa, result.standard_errors_mpa),
        (target, [result.modulus_mpa], [result.standard_error_mpa]),
    ):
        segments = container.lines[2][0].get_segments()
        for segment, value, error in zip(segments, values, errors, strict=True):
            np.testing.assert_allclose(segment[:, 1], [value - error, value + error])
    assert axis.get_xscale() == "log"
    assert axis.get_xlabel() == "Strain rate (strain/ns)"
    assert axis.get_ylabel() == "Young's modulus (MPa)"
    assert "298 K" in axis.get_title()
    assert "0.015" in axis.get_title()
    assert "1.0 decades extrapolated, resolved" in axis.get_title()
    output = io.BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


@pytest.mark.parametrize(
    ("target", "bounds"),
    [(0.001, (0.001, 0.01)), (10.0, (1.0, 10.0))],
)
def test_rate_figure_shades_only_unsampled_interval(
    target: float, bounds: tuple[float, float]
) -> None:
    """A target on either side of the sampled range needs the same caveat."""
    result = strain_rate_extrapolation(_moduli(), target_rate_per_ns=target)
    axis = plot_strain_rate(result).axes[0]
    patch = next(
        patch for patch in axis.patches if patch.get_label() == "extrapolated interval"
    )
    vertices = patch.get_path().transformed(patch.get_transform() - axis.transData)
    np.testing.assert_allclose(
        [vertices.vertices[:, 0].min(), vertices.vertices[:, 0].max()], bounds
    )


def test_interpolated_rate_has_no_extrapolation_shading() -> None:
    result = strain_rate_extrapolation(_moduli(), target_rate_per_ns=0.05)
    axis = plot_strain_rate(result).axes[0]
    assert not axis.patches
    assert "0.0 decades extrapolated" in axis.get_title()


def test_distant_target_is_visibly_unresolved() -> None:
    """A good fit cannot hide an unsupported extrapolation distance."""
    result = strain_rate_extrapolation(_moduli(), target_rate_per_ns=1e-8)
    assert not result.resolved
    axis = plot_strain_rate(result).axes[0]
    assert "6.0 decades extrapolated, not resolved" in axis.get_title()


def test_nonfinite_target_is_identified_without_drawing_a_value() -> None:
    """Numerically unsupported results retain a target rate but no marker."""
    result = replace(
        strain_rate_extrapolation(_moduli(), target_rate_per_ns=0.001),
        modulus_mpa=float("inf"),
        standard_error_mpa=float("inf"),
        resolved=False,
    )
    figure = plot_strain_rate(result)
    axis = figure.axes[0]
    assert len(axis.containers) == 1
    assert any(
        "target estimate not finite" in label
        for label in axis.get_legend_handles_labels()[1]
    )
    figure.savefig(io.BytesIO(), format="png")


def test_nonfinite_uncertainty_is_marked_unavailable() -> None:
    result = replace(
        strain_rate_extrapolation(_moduli(), target_rate_per_ns=0.001),
        standard_error_mpa=float("inf"),
        resolved=False,
    )
    axis = plot_strain_rate(result).axes[0]
    assert "SE unavailable" in axis.containers[1].get_label()
    assert not axis.containers[1].has_yerr
