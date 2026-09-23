"""Analytical stress curves exercise the strength criterion and its refusals."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from openmmpolymer.elasticity import StressStrain
from openmmpolymer.strength import breaking_strength


def _curve(stress_mpa: list[float]) -> StressStrain:
    """A curve with unit lateral area, sampled every 0.1 engineering strain."""
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


def test_a_sustained_stress_loss_reports_peak_and_crossing_separately() -> None:
    curve = _curve([0.0, 40.0, 100.0, 60.0, 40.0, 35.0, 30.0])
    result = breaking_strength(curve)
    assert result.resolved
    assert result.peak_stress_mpa == result.strength_mpa == 100.0
    assert result.strain_at_peak == 0.2
    assert result.failure_strain == 0.4
    assert result.failure_stress_mpa == 40.0
    assert result.failure_bracket == (0.3, 0.4)
    assert result.temperature_k == curve.temperature_k
    assert result.strain_rate_per_ns == curve.strain_rate_per_ns
    assert result.nominal_stress_mpa == pytest.approx(curve.stress_mpa)
    assert any("covalent fracture" in note for note in result.notes)
    assert any("cannot predict bond scission" in note for note in result.notes)


def test_the_area_correction_moves_the_peak_and_removes_lateral_stress() -> None:
    # Differential true stress peaks at 120 MPa. Nominal stress instead peaks
    # at the previous sample's 80 MPa because the area has halved by then.
    curve = _curve([7.0, 87.0, 127.0, 57.0, 47.0, 37.0])
    area = np.asarray([1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
    curve = replace(
        curve,
        lateral_stress_mpa=np.tile([5.0, 9.0], (6, 1)),
        lateral_strain=np.column_stack([area - 1.0, np.zeros(6)]),
    )
    result = breaking_strength(curve)
    assert result.resolved
    assert result.nominal_stress_mpa == pytest.approx([0, 80, 60, 25, 20, 15])
    assert result.strength_mpa == 80.0
    assert result.strain_at_peak == 0.1
    assert result.failure_strain == 0.3


@pytest.mark.parametrize(
    "stress",
    [
        [0.0],
        [0.0, 10.0, 20.0, 30.0],  # Still hardening.
        [0.0, 20.0, 20.0, 20.0, 20.0],  # A plateau is not stress loss.
        [20.0, 5.0, 4.0, 3.0],  # No observation before the peak.
        [0.0, 20.0, 9.0, 8.0],  # Too few low samples.
        [0.0, 20.0, 8.0, 7.0, 6.0, 18.0],  # Recovery after a long dip.
        [0.0, 20.0, 8.0, 7.0, 6.0, 18.0, 8.0, 7.0],  # Short final dip.
        [0.0, -1.0, -2.0, -3.0, -4.0],
        [-10.0, -2.0, -3.0, -4.0, -5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_unresolved_curves_keep_the_observed_peak_without_a_strength(
    stress: list[float],
) -> None:
    result = breaking_strength(_curve(stress))
    assert not result.resolved
    assert result.peak_stress_mpa == max(stress)
    assert result.strength_mpa is None
    assert result.failure_strain is None
    assert result.failure_stress_mpa is None
    assert result.failure_bracket is None


def test_only_the_last_threshold_crossing_is_used_after_recovery() -> None:
    curve = _curve([0.0, 20.0, 8.0, 7.0, 6.0, 18.0, 9.0, 8.0, 7.0])
    result = breaking_strength(curve)
    assert result.resolved
    assert result.failure_strain == 0.6
    assert result.failure_bracket == (0.5, 0.6)


def test_threshold_equality_counts_and_the_confirmation_length_is_configurable() -> (
    None
):
    curve = _curve([0.0, 100.0, 25.0, 25.0])
    assert not breaking_strength(curve, failure_fraction=0.25).resolved
    result = breaking_strength(curve, failure_fraction=0.25, confirmation_steps=2)
    assert result.resolved
    assert result.failure_stress_mpa == 25.0
    assert not breaking_strength(
        curve, failure_fraction=0.249, confirmation_steps=2
    ).resolved


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.0, 1.1, np.nan, np.inf])
def test_failure_fraction_must_be_finite_and_strictly_between_zero_and_one(
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="failure_fraction"):
        breaking_strength(_curve([0.0, 1.0]), failure_fraction=fraction)


@pytest.mark.parametrize("steps", [0, 1, -1])
def test_confirmation_needs_at_least_two_samples(steps: int) -> None:
    with pytest.raises(ValueError, match="confirmation_steps"):
        breaking_strength(_curve([0.0, 1.0]), confirmation_steps=steps)


@pytest.mark.parametrize("steps", [True, 2.0, "3"])
def test_confirmation_must_be_an_integer(steps: Any) -> None:
    with pytest.raises(TypeError, match="confirmation_steps"):
        breaking_strength(_curve([0.0, 1.0]), confirmation_steps=steps)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strain", np.zeros((2, 1)), "one-dimensional"),
        ("strain", np.asarray([-0.1, 0.1]), "nonnegative"),
        ("strain", np.asarray([0.0, 0.0]), "strictly increasing"),
        ("strain", np.asarray([0.1, 0.0]), "strictly increasing"),
        ("strain", np.asarray([0.0, np.nan]), "finite"),
        ("strain", np.asarray([0.0, np.inf]), "finite"),
        ("stress_mpa", np.zeros((2, 1)), "same one-dimensional shape"),
        ("stress_mpa", np.zeros(3), "same one-dimensional shape"),
        ("stress_mpa", np.asarray([0.0, np.inf]), "finite"),
        ("stress_mpa", np.asarray([0.0, 1.0j]), "real"),
        ("stress_mpa", np.asarray(["bad", "data"]), "real"),
        ("lateral_strain", np.zeros((2, 3)), "shape"),
        ("lateral_strain", np.full((2, 2), -1.0), "positive lateral"),
        ("lateral_strain", np.full((2, 2), -1.1), "positive lateral"),
        ("lateral_strain", np.full((2, 2), np.nan), "finite"),
        ("lateral_stress_mpa", np.zeros(2), "shape"),
        ("lateral_stress_mpa", np.full((2, 2), np.inf), "finite"),
        ("controlled", "stress", "strain-controlled"),
        ("temperature_k", 0.0, "temperature_k"),
        ("temperature_k", np.nan, "temperature_k"),
        ("strain_rate_per_ns", np.inf, "strain_rate_per_ns"),
        ("strain_rate_per_ns", -0.1, "strain_rate_per_ns"),
    ],
)
def test_malformed_curves_are_rejected(field: str, value: Any, message: str) -> None:
    curve = replace(_curve([0.0, 1.0]), **{field: value})
    with pytest.raises(ValueError, match=message):
        breaking_strength(curve)


def test_an_empty_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        breaking_strength(_curve([]))


def test_finite_inputs_that_overflow_in_the_area_correction_are_rejected() -> None:
    curve = replace(_curve([0.0, 1.0]), lateral_strain=np.full((2, 2), 1e200))
    with pytest.raises(ValueError, match="remain finite"):
        breaking_strength(curve)


def test_missing_strain_rate_is_preserved() -> None:
    curve = replace(_curve([0.0, 1.0]), strain_rate_per_ns=None)
    assert breaking_strength(curve).strain_rate_per_ns is None
