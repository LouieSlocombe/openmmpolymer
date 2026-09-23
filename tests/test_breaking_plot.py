"""The strength figure displays nominal stress and the limits of its verdict."""

from __future__ import annotations

import io

import numpy as np

from openmmpolymer.elasticity import StressStrain
from openmmpolymer.plots import plot_breaking_strength
from openmmpolymer.strength import breaking_strength


def _strength_curve(*, failed: bool = True, rate: float | None = 0.2) -> StressStrain:
    """A tensile response with known nominal stresses and lateral contraction."""
    strain = np.arange(8, dtype=np.float64) * 0.1
    nominal = (
        np.asarray([0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0])
        if failed
        else np.linspace(0.0, 100.0, strain.size)
    )
    lateral_strain = np.column_stack([-0.2 * strain, -0.2 * strain])
    area_ratio = np.prod(1.0 + lateral_strain, axis=1)
    return StressStrain(
        stage="06_breaking_r0_00",
        axis=2,
        strain=strain,
        stress_mpa=nominal / area_ratio + 0.7,
        lateral_strain=lateral_strain,
        lateral_stress_mpa=np.full((strain.size, 2), 0.7),
        temperature_k=298.15,
        strain_rate_per_ns=rate,
    )


def test_strength_figure_plots_nominal_stress_and_marks_the_sampled_peak() -> None:
    """At finite strain the nominal and Cauchy maxima need not agree."""
    curve = _strength_curve()
    result = breaking_strength(curve)
    figure = plot_breaking_strength(curve, result)
    axis = figure.axes[0]
    lines = {line.get_label(): line for line in axis.get_lines()}
    measured = lines["nominal tensile stress"]
    np.testing.assert_allclose(measured.get_xdata(), curve.strain)
    np.testing.assert_allclose(
        measured.get_ydata(), [0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0]
    )
    assert not np.allclose(measured.get_ydata(), curve.tensile_stress_mpa)
    peak = next(
        line for label, line in lines.items() if label.startswith("sampled peak")
    )
    np.testing.assert_allclose(peak.get_xdata(), [0.3])
    np.testing.assert_allclose(peak.get_ydata(), [100.0])
    assert axis.get_ylabel() == "Nominal tensile stress (MPa)"
    assert axis.get_xlabel() == "Engineering strain along z"
    assert "0.2 strain/ns" in axis.get_title()
    assert "298 K" in axis.get_title()
    assert "Apparent tensile strength = 100.0 MPa" in axis.get_title()

    # A headless report can save the returned figure without importing pyplot.
    output = io.BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_strength_figure_shades_the_failure_interval_and_marks_the_drop() -> None:
    """The reported failure is bracketed by samples, not a precise crossing."""
    curve = _strength_curve()
    result = breaking_strength(curve)
    assert result.resolved
    assert result.failure_bracket is not None
    assert result.failure_strain is not None
    assert result.failure_stress_mpa is not None
    axis = plot_breaking_strength(curve, result=result).axes[0]
    assert len(axis.patches) == 1
    bracket = axis.patches[0]
    assert bracket.get_label() == "failure strain bracket"
    vertices = bracket.get_path().transformed(bracket.get_transform() - axis.transData)
    np.testing.assert_allclose(
        [vertices.vertices[:, 0].min(), vertices.vertices[:, 0].max()],
        result.failure_bracket,
    )
    drop = next(
        line for line in axis.get_lines() if line.get_label() == "sustained stress drop"
    )
    np.testing.assert_allclose(drop.get_xdata(), [result.failure_strain])
    np.testing.assert_allclose(drop.get_ydata(), [result.failure_stress_mpa])


def test_a_rising_curve_reports_no_strength_or_failure_bracket() -> None:
    """An endpoint maximum remains a sample, even with a missing strain rate."""
    curve = _strength_curve(failed=False, rate=None)
    result = breaking_strength(curve)
    assert not result.resolved
    axis = plot_breaking_strength(curve, result).axes[0]
    assert "Apparent tensile strength not resolved" in axis.get_title()
    assert "rate not recorded" in axis.get_title()
    assert not axis.patches
    labels = axis.get_legend_handles_labels()[1]
    assert any(label.startswith("sampled peak") for label in labels)
    assert "sustained stress drop" not in labels
