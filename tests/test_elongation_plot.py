"""The elongation figure distinguishes sampled failure strain from peak stress."""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np

from openmmpolymer.elasticity import StressStrain
from openmmpolymer.plots import plot_elongation_at_break
from openmmpolymer.strength import elongation_at_break


def _elongation_curve(*, failed: bool = True, rate: float | None = 0.2) -> StressStrain:
    """A known nominal tensile response with distinct peak and break strains."""
    strain = np.arange(8, dtype=np.float64) * 0.1
    nominal = (
        np.asarray([0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0])
        if failed
        else np.linspace(0.0, 100.0, strain.size)
    )
    lateral_strain = np.column_stack([-0.2 * strain, -0.2 * strain])
    area_ratio = np.prod(1.0 + lateral_strain, axis=1)
    return StressStrain(
        stage="06_elongation_r0_00",
        axis=2,
        strain=strain,
        stress_mpa=nominal / area_ratio + 0.7,
        lateral_strain=lateral_strain,
        lateral_stress_mpa=np.full((strain.size, 2), 0.7),
        temperature_k=298.15,
        strain_rate_per_ns=rate,
    )


def test_elongation_figure_uses_percent_and_marks_break_separately_from_peak() -> None:
    """The plotted 50% onset differs from the 30% peak and 70% final hold."""
    curve = _elongation_curve()
    figure = plot_elongation_at_break(curve, elongation_at_break(curve))
    axis = figure.axes[0]
    lines = {line.get_label(): line for line in axis.get_lines()}
    measured = lines["nominal tensile stress"]
    np.testing.assert_allclose(measured.get_xdata(), 100.0 * curve.strain)
    np.testing.assert_allclose(
        measured.get_ydata(), [0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0]
    )
    assert not np.allclose(measured.get_ydata(), curve.tensile_stress_mpa)
    peak = next(
        line for label, line in lines.items() if label.startswith("sampled peak")
    )
    np.testing.assert_allclose(peak.get_xdata(), [30.0])
    np.testing.assert_allclose(peak.get_ydata(), [100.0])
    onset = lines["onset of sustained stress drop"]
    np.testing.assert_allclose(onset.get_xdata(), [50.0])
    np.testing.assert_allclose(onset.get_ydata(), [35.0])
    assert axis.get_xlabel() == "Engineering elongation along z (%)"
    assert axis.get_ylabel() == "Nominal tensile stress (MPa)"
    assert "0.2 strain/ns" in axis.get_title()
    assert "298 K" in axis.get_title()
    assert "Apparent elongation at break = 50.0%" in axis.get_title()

    output = io.BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_elongation_figure_shades_the_sampling_bracket_in_percent() -> None:
    """The bracket and marker share the same percentage units as the curve."""
    axis = plot_elongation_at_break(_elongation_curve()).axes[0]
    assert len(axis.patches) == 1
    bracket = axis.patches[0]
    assert bracket.get_label() == "break elongation bracket"
    vertices = bracket.get_path().transformed(bracket.get_transform() - axis.transData)
    np.testing.assert_allclose(
        [vertices.vertices[:, 0].min(), vertices.vertices[:, 0].max()],
        [40.0, 50.0],
    )


def test_unresolved_elongation_has_no_break_marker_or_bracket() -> None:
    """A rising curve is rendered without assigning break to its final hold."""
    curve = _elongation_curve(failed=False, rate=None)
    axis = plot_elongation_at_break(curve).axes[0]
    assert "Apparent elongation at break not resolved" in axis.get_title()
    assert "rate not recorded" in axis.get_title()
    assert not axis.patches
    labels = axis.get_legend_handles_labels()[1]
    assert any(label.startswith("sampled peak") for label in labels)
    assert "onset of sustained stress drop" not in labels


def test_automatic_analysis_uses_custom_criterion_and_condenses_chunks() -> None:
    """A stricter threshold needs enough final low-stress holds to resolve."""
    curve = replace(
        _elongation_curve(), stage="06_elongation_r0_00, 06_elongation_r0_01"
    )
    axis = plot_elongation_at_break(
        curve, failure_fraction=0.33, confirmation_steps=2
    ).axes[0]
    assert "06_elongation_r0_00 (+1 chunks)" in axis.get_title()
    assert "Apparent elongation at break = 60.0%" in axis.get_title()
    unconfirmed = plot_elongation_at_break(
        curve, failure_fraction=0.33, confirmation_steps=3
    ).axes[0]
    assert "Apparent elongation at break not resolved" in unconfirmed.get_title()
    assert not unconfirmed.patches
