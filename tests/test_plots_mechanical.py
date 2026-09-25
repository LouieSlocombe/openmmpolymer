"""Tests for the mechanical figures: stress and strain, the moduli, strength.

The tensile curves are built with a known nominal stress and a lateral
contraction, so a figure that plotted the recorded stress instead of the
nominal one - or put a marker anywhere the analysis did not - is caught by an
equality rather than by eye.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.elasticity import StressStrain, poisson_ratio, youngs_modulus
from openmmpolymer.mechanical import analyse_mechanics
from openmmpolymer.plots import (
    plot_breaking_strength,
    plot_elongation_at_break,
    plot_moduli,
    plot_stress_strain,
    plot_yield_strength,
)
from openmmpolymer.strength import (
    breaking_strength,
    elongation_at_break,
    yield_strength,
)

from .helpers import (
    nominal_curve,
    write_bulk,
    write_deformation,
    write_shear,
)

#: A response that peaks at 30% strain and then loses its stress for good.
FAILING = [0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0]

#: An elastic line with a nonzero intercept, then a plateau.
YIELDING = [2.0, 7.0, 12.0, 17.0, 22.0, 24.0, 25.0, 26.0, 26.0]
YIELD_STRAINS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]


def failure_curve(*, failed: bool = True, rate: float | None = 0.2) -> StressStrain:
    """Eight holds 10% apart, failing after the peak or rising throughout."""
    strain = np.arange(8, dtype=np.float64) * 0.1
    nominal = FAILING if failed else np.linspace(0.0, 100.0, strain.size)
    return nominal_curve(strain, nominal, stage="06_breaking_r0_00", rate_per_ns=rate)


def yield_curve(
    *, yielded: bool = True, rate: float | None = 0.2, stage: str = "06_yield_r0_00"
) -> StressStrain:
    """A yielding response, or a purely elastic one at the same strains."""
    strain = np.asarray(YIELD_STRAINS)
    nominal = YIELDING if yielded else 1000.0 * strain + 2.0
    return nominal_curve(strain, nominal, stage=stage, rate_per_ns=rate)


def linear_curve(
    *, modulus_mpa: float = 2000.0, rate: float | None = 0.04
) -> StressStrain:
    """A stiff linear response with a lateral contraction of 0.35."""
    strain = np.linspace(0.002, 0.02, 10)
    return nominal_curve(
        strain,
        modulus_mpa * strain,
        stage="06_deform_r0_00",
        rate_per_ns=rate,
        poisson=0.35,
        lateral_stress_mpa=0.0,
    )


def span_extent(axis: Any, patch: Any) -> list[float]:
    """The strain a shaded span covers, in the panel's data coordinates."""
    vertices = patch.get_path().transformed(patch.get_transform() - axis.transData)
    return [vertices.vertices[:, 0].min(), vertices.vertices[:, 0].max()]


# --------------------------------------------------------------------------
# Stress and strain, and the moduli
# --------------------------------------------------------------------------


def test_a_stress_strain_figure_draws_the_fit_and_its_window() -> None:
    """Two panels: the curve with its fit, and the lateral response."""
    curve = linear_curve()
    fit = youngs_modulus(curve, strain_limit=0.015)
    figure = plot_stress_strain(curve, fit=fit, poisson=poisson_ratio(curve))
    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylabel() == "Tensile stress (MPa)"
    assert "strain/ns" in figure.axes[0].get_title()
    assert span_extent(figure.axes[0], figure.axes[0].patches[0]) == pytest.approx(
        [0.0, 0.015]
    )
    assert figure.axes[1].get_xlabel() == "Engineering strain along z"


def test_a_stress_strain_figure_says_when_a_rate_was_not_recorded() -> None:
    """Rather than leaving the caveat off the figure entirely."""
    figure = plot_stress_strain(linear_curve(rate=None))
    assert len(figure.axes) == 2
    assert "rate not recorded" in figure.axes[0].get_title()


def test_an_unresolved_fit_is_labelled_as_one() -> None:
    """A figure of a number the noise does not support should say so."""
    curve = linear_curve(modulus_mpa=-500.0)
    fit = youngs_modulus(curve, strain_limit=0.02)
    assert not fit.resolved
    figure = plot_stress_strain(curve, fit=fit)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any("unresolved" in label for label in labels)


@pytest.mark.parametrize("complete", [False, True], ids=["youngs-only", "all-moduli"])
def test_moduli_figure_draws_available_measurements(
    tmp_path: Path, complete: bool
) -> None:
    write_deformation(tmp_path, modulus_mpa=2000.0, poisson=0.35)
    if complete:
        write_bulk(tmp_path, stage="08_bulk", modulus_mpa=2222.0)
        write_shear(tmp_path, modulus_mpa=741.0)
    axis = plot_moduli(analyse_mechanics(tmp_path, strain_limit=0.05)).axes[0]
    expected = ["E", "K", "G"] if complete else ["E"]
    assert [text.get_text() for text in axis.get_xticklabels()] == expected
    assert axis.get_ylabel() == "Modulus (MPa)"
    if complete:
        assert "consistent" in axis.get_title()


# --------------------------------------------------------------------------
# Breaking strength and elongation at break
# --------------------------------------------------------------------------


def test_strength_figure_plots_nominal_stress_and_marks_the_sampled_peak() -> None:
    """At finite strain the nominal and Cauchy maxima need not agree."""
    curve = failure_curve()
    result = breaking_strength(curve)
    assert result.resolved
    axis = plot_breaking_strength(curve, result).axes[0]
    lines = {line.get_label(): line for line in axis.get_lines()}
    measured = lines["nominal tensile stress"]
    np.testing.assert_allclose(measured.get_xdata(), curve.strain)
    np.testing.assert_allclose(measured.get_ydata(), FAILING)
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

    assert result.failure_bracket is not None
    assert result.failure_strain is not None
    assert result.failure_stress_mpa is not None
    assert [patch.get_label() for patch in axis.patches] == ["failure strain bracket"]
    assert span_extent(axis, axis.patches[0]) == pytest.approx(result.failure_bracket)
    drop = next(
        line for line in axis.get_lines() if line.get_label() == "sustained stress drop"
    )
    np.testing.assert_allclose(drop.get_xdata(), [result.failure_strain])
    np.testing.assert_allclose(drop.get_ydata(), [result.failure_stress_mpa])


def test_elongation_figure_uses_percent_and_marks_break_separately_from_peak() -> None:
    """The plotted 50% onset differs from the 30% peak and 70% final hold,
    and the bracket shares the curve's percentage units."""
    curve = failure_curve()
    axis = plot_elongation_at_break(curve, elongation_at_break(curve)).axes[0]
    lines = {line.get_label(): line for line in axis.get_lines()}
    np.testing.assert_allclose(
        lines["nominal tensile stress"].get_xdata(), 100.0 * curve.strain
    )
    np.testing.assert_allclose(lines["nominal tensile stress"].get_ydata(), FAILING)
    peak = next(
        line for label, line in lines.items() if label.startswith("sampled peak")
    )
    np.testing.assert_allclose(peak.get_xdata(), [30.0])
    np.testing.assert_allclose(peak.get_ydata(), [100.0])
    onset = lines["onset of sustained stress drop"]
    np.testing.assert_allclose(onset.get_xdata(), [50.0])
    np.testing.assert_allclose(onset.get_ydata(), [35.0])
    assert [patch.get_label() for patch in axis.patches] == ["break elongation bracket"]
    assert span_extent(axis, axis.patches[0]) == pytest.approx([40.0, 50.0])
    assert axis.get_xlabel() == "Engineering elongation along z (%)"
    assert axis.get_ylabel() == "Nominal tensile stress (MPa)"
    assert "0.2 strain/ns" in axis.get_title()
    assert "Apparent elongation at break = 50.0%" in axis.get_title()


@pytest.mark.parametrize("elongation", [False, True], ids=["strength", "elongation"])
def test_rising_curve_has_no_failure_marker_or_bracket(elongation: bool) -> None:
    curve = failure_curve(failed=False, rate=None)
    if elongation:
        result = elongation_at_break(curve)
        assert not result.resolved
        axis = plot_elongation_at_break(curve, result).axes[0]
        title = "Apparent elongation at break not resolved"
        marker = "onset of sustained stress drop"
    else:
        strength = breaking_strength(curve)
        assert not strength.resolved
        axis = plot_breaking_strength(curve, strength).axes[0]
        title = "Apparent tensile strength not resolved"
        marker = "sustained stress drop"
    assert title in axis.get_title()
    assert "rate not recorded" in axis.get_title()
    assert not axis.patches
    labels = axis.get_legend_handles_labels()[1]
    assert any(label.startswith("sampled peak") for label in labels)
    assert marker not in labels


# --------------------------------------------------------------------------
# Yield strength
# --------------------------------------------------------------------------


def test_yield_figure_plots_nominal_response_and_the_offset_construction() -> None:
    """The displayed lines use nominal stress and preserve the fitted intercept."""
    curve = yield_curve()
    result = yield_strength(curve)
    assert result.resolved
    axis = plot_yield_strength(curve, result).axes[0]
    lines = {line.get_label(): line for line in axis.get_lines()}
    measured = lines["nominal tensile stress"]
    np.testing.assert_allclose(measured.get_xdata(), curve.strain)
    np.testing.assert_allclose(measured.get_ydata(), YIELDING)
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

    patches = {patch.get_label(): patch for patch in axis.patches}
    assert span_extent(axis, patches["elastic fit window"]) == pytest.approx(
        [0.0, 0.02], abs=1e-15
    )
    assert span_extent(axis, patches["yield strain bracket"]) == pytest.approx(
        [0.02, 0.025], abs=1e-15
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
    curve = yield_curve(yielded=False, rate=None)
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
    curve = yield_curve(stage="06_yield_r0_00, 06_yield_r0_01")
    result = yield_strength(curve, offset_strain=0.01)
    assert result.resolved
    axis = plot_yield_strength(curve, result).axes[0]
    assert "1% offset yield strength" in axis.get_title()
    assert "06_yield_r0_00 (+1 chunks)" in axis.get_title()
    assert "1% offset line" in axis.get_legend_handles_labels()[1]


def test_an_unavailable_elastic_fit_is_drawn_without_invalid_lines() -> None:
    """A window with only one observation still produces a useful figure."""
    curve = yield_curve()
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
    curve = yield_curve()
    result = yield_strength(curve, fit_max_strain=0.011)
    assert result.modulus_mpa is not None
    assert not result.fit_resolved
    axis = plot_yield_strength(curve, result).axes[0]
    labels = axis.get_legend_handles_labels()[1]
    fit_label = next(label for label in labels if label.startswith("elastic fit:"))
    assert "unresolved" in fit_label
    assert "yield strength not resolved" in axis.get_title()
    assert "interpolated offset intersection" not in labels
