"""Elongation uses confirmed failure strain, not peak or final sampled strain."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from openmmpolymer.elasticity import StressStrain
from openmmpolymer.strength import elongation_at_break


def _curve(stress_mpa: list[float]) -> StressStrain:
    count = len(stress_mpa)
    return StressStrain(
        stage="tension",
        axis=2,
        strain=np.arange(count, dtype=np.float64) / 10.0,
        stress_mpa=np.asarray(stress_mpa, dtype=np.float64),
        lateral_strain=np.zeros((count, 2), dtype=np.float64),
        lateral_stress_mpa=np.zeros((count, 2), dtype=np.float64),
        temperature_k=298.15,
        strain_rate_per_ns=0.2,
    )


def test_elongation_reports_the_onset_of_confirmed_loss_as_engineering_percent() -> (
    None
):
    curve = replace(
        _curve([0.0, 40.0, 100.0, 60.0, 40.0, 35.0, 30.0]),
        strain=np.asarray([0.0, 0.1, 0.25, 0.5, 1.2, 1.5, 1.7]),
    )
    result = elongation_at_break(curve)

    assert result.resolved
    assert result.strain_at_break == 1.2
    assert result.elongation_percent == 120.0
    assert result.break_stress_mpa == 40.0
    assert result.break_bracket == (0.5, 1.2)
    assert result.peak_stress_mpa == 100.0
    assert result.strain_at_peak == 0.25
    assert result.temperature_k == 298.15
    assert result.strain_rate_per_ns == 0.2
    assert result.nominal_stress_mpa == pytest.approx(curve.stress_mpa)
    assert any("covalent fracture" in note for note in result.notes)
    assert any("cannot predict bond scission" in note for note in result.notes)


def test_area_correction_and_lateral_pressure_determine_the_break_strain() -> None:
    # Raw axial stress never falls below half of its peak. Differential nominal
    # stress instead has three final samples at or below half of its 80 MPa peak.
    curve = replace(
        _curve([30.0, 110.0, 150.0, 90.0, 80.0, 80.0]),
        lateral_stress_mpa=np.tile([28.0, 32.0], (6, 1)),
        lateral_strain=np.column_stack(
            [np.asarray([0.0, 0.0, -0.5, -0.5, -0.5, -0.5]), np.zeros(6)]
        ),
    )
    result = elongation_at_break(curve)

    assert result.resolved
    assert result.nominal_stress_mpa == pytest.approx([0, 80, 60, 30, 25, 25])
    assert result.strain_at_peak == 0.1
    assert result.strain_at_break == 0.3
    assert result.elongation_percent == 30.0
    assert result.break_stress_mpa == 30.0
    assert result.break_bracket == (0.2, 0.3)


@pytest.mark.parametrize(
    "stress",
    [
        [0.0],
        [0.0, 10.0, 20.0, 30.0],  # Continued hardening.
        [0.0, 20.0, 20.0, 20.0, 20.0],  # Plateau.
        [20.0, 5.0, 4.0, 3.0],  # Initial-boundary peak.
        [0.0, 20.0, 9.0, 8.0],  # Insufficient confirmation.
        [0.0, 20.0, 8.0, 7.0, 6.0, 18.0],  # Recovery after a long dip.
        [0.0, 20.0, 8.0, 7.0, 6.0, 18.0, 8.0, 7.0],  # Short final dip.
        [-10.0, -2.0, -3.0, -4.0, -5.0],  # No tensile peak.
    ],
)
def test_unresolved_break_preserves_diagnostics_without_an_elongation(
    stress: list[float],
) -> None:
    curve = _curve(stress)
    result = elongation_at_break(curve)

    assert not result.resolved
    assert result.strain_at_break is None
    assert result.elongation_percent is None
    assert result.break_stress_mpa is None
    assert result.break_bracket is None
    assert result.peak_stress_mpa == max(stress)
    assert result.strain_at_peak == curve.strain[np.argmax(stress)]
    assert result.nominal_stress_mpa == pytest.approx(stress)
    assert result.notes


def test_recovery_moves_elongation_to_the_final_confirmed_threshold_crossing() -> None:
    curve = _curve([0.0, 20.0, 8.0, 7.0, 6.0, 18.0, 9.0, 8.0, 7.0])
    result = elongation_at_break(curve)

    assert result.resolved
    assert result.strain_at_break == 0.6
    assert result.elongation_percent == 60.0
    assert result.break_bracket == (0.5, 0.6)


def test_configured_fraction_and_confirmation_control_resolution() -> None:
    curve = _curve([0.0, 100.0, 25.0, 25.0])
    assert not elongation_at_break(curve, failure_fraction=0.25).resolved
    result = elongation_at_break(curve, failure_fraction=0.25, confirmation_steps=2)
    assert result.resolved
    assert result.strain_at_break == 0.2
    assert result.elongation_percent == 20.0
    assert result.break_stress_mpa == 25.0
    assert not elongation_at_break(
        curve, failure_fraction=0.249, confirmation_steps=2
    ).resolved


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.0, 1.1, np.nan, np.inf])
def test_invalid_failure_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="failure_fraction"):
        elongation_at_break(_curve([0.0, 1.0]), failure_fraction=fraction)


@pytest.mark.parametrize("steps", [True, 2.0, "3"])
def test_confirmation_must_be_an_integer(steps: Any) -> None:
    with pytest.raises(TypeError, match="confirmation_steps"):
        elongation_at_break(_curve([0.0, 1.0]), confirmation_steps=steps)


@pytest.mark.parametrize("steps", [0, 1, -1])
def test_confirmation_needs_at_least_two_samples(steps: int) -> None:
    with pytest.raises(ValueError, match="confirmation_steps"):
        elongation_at_break(_curve([0.0, 1.0]), confirmation_steps=steps)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strain", np.asarray([0.0, np.nan]), "finite"),
        ("strain", np.asarray([0.1, 0.0]), "strictly increasing"),
        ("stress_mpa", np.asarray([0.0, np.inf]), "finite"),
        ("lateral_strain", np.full((2, 2), -1.0), "positive lateral"),
        ("controlled", "stress", "strain-controlled"),
    ],
)
def test_invalid_curves_are_rejected(field: str, value: Any, message: str) -> None:
    curve = replace(_curve([0.0, 1.0]), **{field: value})
    with pytest.raises(ValueError, match=message):
        elongation_at_break(curve)


def test_unknown_strain_rate_is_preserved() -> None:
    curve = replace(_curve([0.0, 20.0, 8.0, 7.0, 6.0]), strain_rate_per_ns=None)
    assert elongation_at_break(curve).strain_rate_per_ns is None


def test_finite_strains_that_overflow_percentage_conversion_are_rejected() -> None:
    curve = replace(
        _curve([0.0, 20.0, 8.0, 7.0, 6.0]),
        strain=np.asarray([0.0, 1e306, 2e306, 3e306, 4e306]),
    )
    with pytest.raises(ValueError, match="elongation_percent must remain finite"):
        elongation_at_break(curve)
