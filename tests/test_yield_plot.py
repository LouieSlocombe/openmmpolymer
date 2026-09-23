"""The yield figure makes the offset construction and sampling interval visible."""

from __future__ import annotations

import io

import numpy as np

from openmmpolymer.elasticity import StressStrain
from openmmpolymer.plots import plot_yield_strength
from openmmpolymer.strength import yield_strength


def _yield_curve(*, yielded: bool = True, rate: float | None = 0.2) -> StressStrain:
    """An elastic line with a nonzero intercept followed by a plateau."""
    strain = np.asarray([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
    nominal = (
        np.asarray([2.0, 7.0, 12.0, 17.0, 22.0, 24.0, 25.0, 26.0, 26.0])
        if yielded
        else 1000.0 * strain + 2.0
    )
    lateral_strain = np.column_stack([-0.2 * strain, -0.2 * strain])
    area_ratio = np.prod(1.0 + lateral_strain, axis=1)
    return StressStrain(
        stage="06_yield_r0_00",
        axis=2,
        strain=strain,
        stress_mpa=nominal / area_ratio + 0.7,
        lateral_strain=lateral_strain,
        lateral_stress_mpa=np.full((strain.size, 2), 0.7),
        temperature_k=298.15,
        strain_rate_per_ns=rate,
    )


def test_yield_figure_plots_nominal_response_and_the_offset_construction() -> None:
    """The displayed lines use nominal stress and preserve the fitted intercept."""
    curve = _yield_curve()
    result = yield_strength(curve)
    figure = plot_yield_strength(curve, result)
    axis = figure.axes[0]
    lines = {line.get_label(): line for line in axis.get_lines()}
    measured = lines["nominal tensile stress"]
    np.testing.assert_allclose(measured.get_xdata(), curve.strain)
    np.testing.assert_allclose(
        measured.get_ydata(), [2.0, 7.0, 12.0, 17.0, 22.0, 24.0, 25.0, 26.0, 26.0]
    )
    assert not np.allclose(measured.get_ydata(), curve.tensile_stress_mpa)
    fit = next(line for label, line in lines.items() if label.startswith("elastic fit"))
    np.testing.assert_allclose(fit.get_xdata(), [0.0, 0.02])
    np.testing.assert_allclose(fit.get_ydata(), [2.0, 22.0])
    offset = lines["0.2% offset line"]
    np.testing.assert_allclose(
        offset.get_ydata(), 1000.0 * offset.get_xdata(), atol=1e-12
    )
    assert axis.get_ylim()[1] < 30.0
    assert axis.get_ylabel() == "Nominal tensile stress (MPa)"
    assert axis.get_xlabel() == "Engineering strain along z"
    assert "0.2 strain/ns" in axis.get_title()
    assert "298 K" in axis.get_title()
    assert "0.2% offset yield strength = 23.3 MPa" in axis.get_title()

    # A headless report can render the directly constructed Figure.
    output = io.BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_yield_figure_distinguishes_interpolation_from_the_sampling_bracket() -> None:
    """A yield point lies inside the sampled interval that establishes it."""
    curve = _yield_curve()
    result = yield_strength(curve)
    assert result.resolved
    axis = plot_yield_strength(curve, result=result).axes[0]
    patches = {patch.get_label(): patch for patch in axis.patches}
    for label, limits in (
        ("elastic fit window", (0.0, 0.02)),
        ("yield strain bracket", (0.02, 0.025)),
    ):
        patch = patches[label]
        vertices = patch.get_path().transformed(patch.get_transform() - axis.transData)
        np.testing.assert_allclose(
            [vertices.vertices[:, 0].min(), vertices.vertices[:, 0].max()],
            limits,
            atol=1e-15,
        )
    point = next(
        line
        for line in axis.get_lines()
        if line.get_label() == "interpolated offset intersection"
    )
    np.testing.assert_allclose(point.get_xdata(), [0.0233333333333333])
    np.testing.assert_allclose(point.get_ydata(), [23.3333333333333])


def test_an_elastic_curve_reports_no_yield_point_or_bracket() -> None:
    """No intersection is shown for an unresolved, purely elastic response."""
    curve = _yield_curve(yielded=False, rate=None)
    result = yield_strength(curve)
    assert not result.resolved
    axis = plot_yield_strength(curve, result).axes[0]
    assert "0.2% offset yield strength not resolved" in axis.get_title()
    assert "rate not recorded" in axis.get_title()
    labels = axis.get_legend_handles_labels()[1]
    assert "elastic fit window" in labels
    assert "0.2% offset line" in labels
    assert "yield strain bracket" not in labels
    assert "interpolated offset intersection" not in labels


def test_custom_offset_and_multiple_chunks_are_identified() -> None:
    """The figure records the criterion and condenses a long chunk list."""
    from dataclasses import replace

    curve = replace(_yield_curve(), stage="06_yield_r0_00, 06_yield_r0_01")
    result = yield_strength(curve, offset_strain=0.01)
    assert result.resolved
    axis = plot_yield_strength(curve, result).axes[0]
    assert "1% offset yield strength" in axis.get_title()
    assert "06_yield_r0_00 (+1 chunks)" in axis.get_title()
    assert "1% offset line" in axis.get_legend_handles_labels()[1]


def test_an_unavailable_elastic_fit_is_drawn_without_invalid_lines() -> None:
    """A window with only one observation still produces a useful figure."""
    curve = _yield_curve()
    result = yield_strength(curve, fit_max_strain=0.001)
    assert result.modulus_mpa is None
    axis = plot_yield_strength(curve, result).axes[0]
    assert "yield strength not resolved" in axis.get_title()
    labels = axis.get_legend_handles_labels()[1]
    assert "elastic fit window" in labels
    assert not any(label.startswith("elastic fit:") for label in labels)
    assert "0.2% offset line" not in labels


def test_an_under_sampled_elastic_fit_is_labelled_unresolved() -> None:
    """A calculable slope is not presented as a reliable elastic modulus."""
    curve = _yield_curve()
    result = yield_strength(curve, fit_max_strain=0.011)
    assert result.modulus_mpa is not None
    assert not result.fit_resolved
    axis = plot_yield_strength(curve, result).axes[0]
    labels = axis.get_legend_handles_labels()[1]
    fit_label = next(label for label in labels if label.startswith("elastic fit:"))
    assert "unresolved" in fit_label
    assert "yield strength not resolved" in axis.get_title()
    assert "interpolated offset intersection" not in labels
