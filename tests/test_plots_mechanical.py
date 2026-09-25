"""Tests for the mechanical figures: stress and strain, the moduli, strength.

The tensile curves are built with a known nominal stress and a lateral
contraction, so a figure that plotted the recorded stress instead of the
nominal one - or put a marker anywhere the analysis did not - is caught by an
equality rather than by eye.
"""

from __future__ import annotations

from dataclasses import replace
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
    plot_strain_rate,
    plot_stress_strain,
    plot_yield_strength,
)
from openmmpolymer.strain_rate import strain_rate_extrapolation
from openmmpolymer.strength import (
    breaking_strength,
    elongation_at_break,
    yield_strength,
)

from .helpers import (
    nominal_curve,
    rate_moduli,
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
    assert "rate not recorded" in figure.axes[0].get_title()


def test_an_unresolved_fit_is_labelled_as_one() -> None:
    """A figure of a number the noise does not support should say so."""
    curve = linear_curve(modulus_mpa=-500.0)
    fit = youngs_modulus(curve, strain_limit=0.02)
    assert not fit.resolved
    figure = plot_stress_strain(curve, fit=fit)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any("unresolved" in label for label in labels)


def test_a_stress_strain_figure_needs_no_fit() -> None:
    """The curve alone is a figure; the fit is an overlay."""
    assert len(plot_stress_strain(linear_curve()).axes) == 2


def test_a_moduli_figure_draws_the_measured_against_the_implied(
    tmp_path: Path,
) -> None:
    """The gap between the bars and the markers is the whole point of it."""
    write_deformation(tmp_path, modulus_mpa=2000.0, poisson=0.35)
    write_bulk(tmp_path, stage="08_bulk", modulus_mpa=2222.0)
    write_shear(tmp_path, modulus_mpa=741.0)
    report = analyse_mechanics(tmp_path, strain_limit=0.05)
    figure = plot_moduli(report)
    axis = figure.axes[0]
    assert [text.get_text() for text in axis.get_xticklabels()] == ["E", "K", "G"]
    assert axis.get_ylabel() == "Modulus (MPa)"
    assert "consistent" in axis.get_title()


def test_a_moduli_figure_survives_a_report_with_only_a_modulus_in_it(
    tmp_path: Path,
) -> None:
    """A run that skipped the other passes still gets a figure."""
    write_deformation(tmp_path)
    report = analyse_mechanics(tmp_path, strain_limit=0.05)
    axis = plot_moduli(report).axes[0]
    assert [text.get_text() for text in axis.get_xticklabels()] == ["E"]


# --------------------------------------------------------------------------
# Modulus against strain rate
# --------------------------------------------------------------------------


def moduli_for(form: str) -> list[Any]:
    """Three moduli on the relation *form* fits, a decade of rate apart."""
    if form == "log_linear":
        return rate_moduli([800.0, 900.0, 1000.0])
    return rate_moduli([800.0 * (rate / 0.01) ** 0.1 for rate in (0.01, 0.1, 1.0)])


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_rate_figure_preserves_data_target_and_fit(form: str) -> None:
    """Both supported relations are plotted using the analysis prediction."""
    result = strain_rate_extrapolation(
        moduli_for(form), target_rate_per_ns=0.001, form=form
    )
    axis = plot_strain_rate(result).axes[0]
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


@pytest.mark.parametrize(
    ("target", "bounds"),
    [(0.001, (0.001, 0.01)), (10.0, (1.0, 10.0))],
)
def test_rate_figure_shades_only_unsampled_interval(
    target: float, bounds: tuple[float, float]
) -> None:
    """A target on either side of the sampled range needs the same caveat."""
    result = strain_rate_extrapolation(
        moduli_for("log_linear"), target_rate_per_ns=target
    )
    axis = plot_strain_rate(result).axes[0]
    patch = next(
        patch for patch in axis.patches if patch.get_label() == "extrapolated interval"
    )
    assert span_extent(axis, patch) == pytest.approx(bounds)


def test_interpolated_rate_has_no_extrapolation_shading() -> None:
    result = strain_rate_extrapolation(
        moduli_for("log_linear"), target_rate_per_ns=0.05
    )
    axis = plot_strain_rate(result).axes[0]
    assert not axis.patches
    assert "0.0 decades extrapolated" in axis.get_title()


def test_distant_target_is_visibly_unresolved() -> None:
    """A good fit cannot hide an unsupported extrapolation distance."""
    result = strain_rate_extrapolation(
        moduli_for("log_linear"), target_rate_per_ns=1e-8
    )
    assert not result.resolved
    axis = plot_strain_rate(result).axes[0]
    assert "6.0 decades extrapolated, not resolved" in axis.get_title()


def test_nonfinite_target_is_identified_without_drawing_a_value() -> None:
    """Numerically unsupported results retain a target rate but no marker."""
    result = replace(
        strain_rate_extrapolation(moduli_for("log_linear"), target_rate_per_ns=0.001),
        modulus_mpa=float("inf"),
        standard_error_mpa=float("inf"),
        resolved=False,
    )
    axis = plot_strain_rate(result).axes[0]
    assert len(axis.containers) == 1
    assert any(
        "target estimate not finite" in label
        for label in axis.get_legend_handles_labels()[1]
    )


def test_nonfinite_uncertainty_is_marked_unavailable() -> None:
    result = replace(
        strain_rate_extrapolation(moduli_for("log_linear"), target_rate_per_ns=0.001),
        standard_error_mpa=float("inf"),
        resolved=False,
    )
    axis = plot_strain_rate(result).axes[0]
    assert "SE unavailable" in axis.containers[1].get_label()
    assert not axis.containers[1].has_yerr


# --------------------------------------------------------------------------
# Breaking strength and elongation at break
# --------------------------------------------------------------------------


def test_strength_figure_plots_nominal_stress_and_marks_the_sampled_peak() -> None:
    """At finite strain the nominal and Cauchy maxima need not agree."""
    curve = failure_curve()
    axis = plot_breaking_strength(curve, breaking_strength(curve)).axes[0]
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


def test_strength_figure_shades_the_failure_interval_and_marks_the_drop() -> None:
    """The reported failure is bracketed by samples, not a precise crossing."""
    curve = failure_curve()
    result = breaking_strength(curve)
    assert result.resolved
    assert result.failure_bracket is not None
    assert result.failure_strain is not None
    assert result.failure_stress_mpa is not None
    axis = plot_breaking_strength(curve, result).axes[0]
    assert [patch.get_label() for patch in axis.patches] == ["failure strain bracket"]
    assert span_extent(axis, axis.patches[0]) == pytest.approx(result.failure_bracket)
    drop = next(
        line for line in axis.get_lines() if line.get_label() == "sustained stress drop"
    )
    np.testing.assert_allclose(drop.get_xdata(), [result.failure_strain])
    np.testing.assert_allclose(drop.get_ydata(), [result.failure_stress_mpa])


def test_a_rising_curve_reports_no_strength_or_failure_bracket() -> None:
    """An endpoint maximum remains a sample, even with a missing strain rate."""
    curve = failure_curve(failed=False, rate=None)
    result = breaking_strength(curve)
    assert not result.resolved
    axis = plot_breaking_strength(curve, result).axes[0]
    assert "Apparent tensile strength not resolved" in axis.get_title()
    assert "rate not recorded" in axis.get_title()
    assert not axis.patches
    labels = axis.get_legend_handles_labels()[1]
    assert any(label.startswith("sampled peak") for label in labels)
    assert "sustained stress drop" not in labels


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


def test_unresolved_elongation_has_no_break_marker_or_bracket() -> None:
    """A rising curve is rendered without assigning break to its final hold."""
    curve = failure_curve(failed=False, rate=None)
    axis = plot_elongation_at_break(curve, elongation_at_break(curve)).axes[0]
    assert "Apparent elongation at break not resolved" in axis.get_title()
    assert "rate not recorded" in axis.get_title()
    assert not axis.patches
    labels = axis.get_legend_handles_labels()[1]
    assert any(label.startswith("sampled peak") for label in labels)
    assert "onset of sustained stress drop" not in labels


# --------------------------------------------------------------------------
# Yield strength
# --------------------------------------------------------------------------


def test_yield_figure_plots_nominal_response_and_the_offset_construction() -> None:
    """The displayed lines use nominal stress and preserve the fitted intercept."""
    curve = yield_curve()
    axis = plot_yield_strength(curve, yield_strength(curve)).axes[0]
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


def test_yield_figure_distinguishes_interpolation_from_the_sampling_bracket() -> None:
    """A yield point lies inside the sampled interval that establishes it."""
    curve = yield_curve()
    result = yield_strength(curve)
    assert result.resolved
    axis = plot_yield_strength(curve, result).axes[0]
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
