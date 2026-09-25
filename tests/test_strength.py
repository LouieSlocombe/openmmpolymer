"""Analytical curves exercise the tensile criteria and their refusals."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from openmmpolymer.elasticity import StressStrain
from openmmpolymer.strength import (
    breaking_strength,
    elongation_at_break,
    yield_strength,
)


def _curve(
    stress_mpa: Any, strain: npt.NDArray[np.float64] | None = None
) -> StressStrain:
    """A curve with unit lateral area, sampled every 0.1 strain unless given."""
    stress = np.asarray(stress_mpa, dtype=np.float64)
    return StressStrain(
        stage="tension",
        axis=2,
        strain=np.arange(stress.size, dtype=np.float64) / 10.0
        if strain is None
        else strain,
        stress_mpa=stress,
        lateral_strain=np.zeros((stress.size, 2), dtype=np.float64),
        lateral_stress_mpa=np.zeros((stress.size, 2), dtype=np.float64),
        temperature_k=298.15,
        strain_rate_per_ns=0.2,
    )


def _bilinear() -> StressStrain:
    """Elastic at 1000 MPa up to 30 MPa at strain 0.03, then hardening at 100."""
    strain = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06])
    return _curve(np.minimum(1000 * strain, 30 + 100 * (strain - 0.03)), strain)


#: Curves whose peak no sustained terminal loss of stress confirms.
UNRESOLVED = [
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
]


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


@pytest.mark.parametrize("stress", UNRESOLVED)
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


@pytest.mark.parametrize(
    ("stress", "criterion"),
    [
        ([0.0, 40.0, 100.0, 60.0, 40.0, 35.0, 30.0], {}),
        ([0.0, 20.0, 8.0, 7.0, 6.0, 18.0, 9.0, 8.0, 7.0], {}),
        ([0.0, 100.0, 25.0, 25.0], {"failure_fraction": 0.25}),
        ([0.0, 100.0, 25.0, 25.0], {"failure_fraction": 0.25, "confirmation_steps": 2}),
        *(
            pytest.param(stress, {}, id=f"unresolved{i}")
            for i, stress in enumerate(UNRESOLVED)
        ),
    ],
)
def test_elongation_at_break_is_the_breaking_endpoint_in_percent(
    stress: list[float], criterion: dict[str, Any]
) -> None:
    curve = _curve(stress)
    strength = breaking_strength(curve, **criterion)
    result = elongation_at_break(curve, **criterion)
    assert result.strain_at_break == strength.failure_strain
    assert result.elongation_percent == (
        None if strength.failure_strain is None else 100.0 * strength.failure_strain
    )
    assert result.break_stress_mpa == strength.failure_stress_mpa
    assert result.break_bracket == strength.failure_bracket
    for field in (
        "peak_stress_mpa",
        "strain_at_peak",
        "resolved",
        "temperature_k",
        "strain_rate_per_ns",
        "notes",
    ):
        assert getattr(result, field) == getattr(strength, field)
    assert np.array_equal(result.nominal_stress_mpa, strength.nominal_stress_mpa)


@pytest.mark.parametrize(
    "criterion",
    [{"failure_fraction": 1.0}, {"confirmation_steps": 1}, {"confirmation_steps": 2.0}],
)
def test_elongation_at_break_refuses_what_breaking_strength_refuses(
    criterion: dict[str, Any],
) -> None:
    curve = _curve([0.0, 1.0])
    with pytest.raises((TypeError, ValueError)) as refused:
        breaking_strength(curve, **criterion)
    with pytest.raises(refused.type, match=re.escape(str(refused.value))):
        elongation_at_break(curve, **criterion)


def test_elongation_is_the_engineering_strain_of_the_onset_not_of_the_peak() -> None:
    curve = _curve(
        [0.0, 40.0, 100.0, 60.0, 40.0, 35.0, 30.0],
        np.asarray([0.0, 0.1, 0.25, 0.5, 1.2, 1.5, 1.7]),
    )
    result = elongation_at_break(curve)
    assert result.resolved
    assert result.strain_at_break == 1.2
    assert result.elongation_percent == 120.0
    assert result.break_stress_mpa == 40.0
    assert result.break_bracket == (0.5, 1.2)
    assert result.strain_at_peak == 0.25


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
    assert result.break_bracket == (0.2, 0.3)


def test_finite_strains_that_overflow_percentage_conversion_are_rejected() -> None:
    curve = _curve(
        [0.0, 20.0, 8.0, 7.0, 6.0], np.asarray([0.0, 1e306, 2e306, 3e306, 4e306])
    )
    with pytest.raises(ValueError, match="elongation_percent must remain finite"):
        elongation_at_break(curve)


def test_bilinear_hardening_has_an_analytical_offset_intersection() -> None:
    curve = _bilinear()
    result = yield_strength(curve)
    # 1000*(strain-.002) = 30 + 100*(strain-.03).
    expected_strain = 29 / 900
    assert result.resolved and result.fit_resolved
    assert result.modulus_mpa == pytest.approx(1000)
    assert result.intercept_mpa == pytest.approx(0, abs=1e-12)
    assert result.yield_strain == pytest.approx(expected_strain)
    assert result.strength_mpa == pytest.approx(1000 * (expected_strain - 0.002))
    assert result.yield_bracket == (0.03, 0.04)
    assert result.fit_points == 5
    assert result.temperature_k == curve.temperature_k
    assert result.strain_rate_per_ns == curve.strain_rate_per_ns
    assert result.strength_mpa < max(curve.stress_mpa)


def test_initial_stress_is_retained_in_the_shifted_elastic_line() -> None:
    curve = _bilinear()
    base = yield_strength(curve)
    result = yield_strength(replace(curve, stress_mpa=curve.stress_mpa + 7))
    assert result.resolved
    assert result.yield_strain == pytest.approx(base.yield_strain)
    assert result.intercept_mpa == pytest.approx(7)
    assert base.strength_mpa is not None
    assert result.strength_mpa == pytest.approx(base.strength_mpa + 7)


def test_nominal_area_conversion_and_lateral_stress_subtraction() -> None:
    curve = _bilinear()
    # An isochoric uniaxial extension has A/A0 = 1/(1+strain).
    area = 1 / (1 + curve.strain)
    measured = replace(
        curve,
        stress_mpa=curve.stress_mpa / area + 6,
        lateral_stress_mpa=np.tile([4, 8], (curve.n_points, 1)),
        lateral_strain=np.column_stack([np.sqrt(area) - 1] * 2),
    )
    result = yield_strength(measured)
    assert result.nominal_stress_mpa == pytest.approx(curve.stress_mpa)
    assert result.strength_mpa == pytest.approx(yield_strength(curve).strength_mpa)


def test_offset_and_fit_window_are_configurable() -> None:
    curve = _bilinear()
    result = yield_strength(curve, offset_strain=0.01)
    assert result.resolved
    assert result.yield_strain == pytest.approx(37 / 900)
    # Exclude a preloading toe while retaining five points in the linear fit.
    curve = replace(curve, stress_mpa=1000 * curve.strain + 7)
    curve.stress_mpa[0] = 50
    curve.stress_mpa[-1] = 40
    result = yield_strength(curve, fit_min_strain=0.005, fit_max_strain=0.03)
    assert result.resolved
    assert result.fit_points == 5
    assert result.modulus_mpa == pytest.approx(1000)


@pytest.mark.parametrize("slope", [0.0, -1000.0, 1000.0])
def test_linear_curves_never_invent_a_yield(slope: float) -> None:
    curve = _bilinear()
    result = yield_strength(replace(curve, stress_mpa=slope * curve.strain))
    assert not result.resolved
    assert result.strength_mpa is None
    assert result.yield_strain is None
    assert result.yield_bracket is None
    assert result.fit_resolved == (slope > 0)


@pytest.mark.parametrize("max_strain", [0.001, 0.006, 0.016])
def test_too_few_elastic_points_preserve_diagnostics(max_strain: float) -> None:
    result = yield_strength(_bilinear(), fit_max_strain=max_strain)
    assert not result.resolved
    assert not result.fit_resolved
    if result.fit_points == 1:
        assert result.modulus_mpa is None
        assert result.intercept_mpa is None
        assert result.standard_error_mpa is None
    else:
        assert result.modulus_mpa == pytest.approx(1000)


def test_a_noisy_elastic_window_is_unresolved() -> None:
    curve = _bilinear()
    curve.stress_mpa[:5] = [0, 80, -50, 70, 20]
    result = yield_strength(curve)
    assert not result.fit_resolved and not result.resolved
    assert any("Elastic fit unresolved" in note for note in result.notes)


def test_crossing_within_elastic_window_is_not_misreported_as_later_yield() -> None:
    curve = _bilinear()
    # The fitted window ends at .029, between the last elastic sample (.02)
    # and a much lower next measurement. Their crossing precedes .029.
    curve.stress_mpa[5:] = 20
    result = yield_strength(curve, fit_max_strain=0.029)
    assert result.fit_resolved
    assert not result.resolved
    assert any("within the elastic fit window" in note for note in result.notes)


def test_a_nonpositive_crossing_is_not_a_tensile_yield() -> None:
    curve = _bilinear()
    result = yield_strength(replace(curve, stress_mpa=curve.stress_mpa - 40))
    assert result.fit_resolved and not result.resolved
    assert any("positive tensile stress" in note for note in result.notes)


def test_a_curve_already_below_the_offset_line_has_no_later_yield() -> None:
    curve = _bilinear()
    # A last elastic sample 0.3 MPa low sits under a 0.01% offset line.
    curve.stress_mpa[4] = 19.7
    result = yield_strength(curve, offset_strain=0.0001)
    assert result.fit_resolved and not result.resolved
    assert any("already at or below the offset line" in note for note in result.notes)


def test_an_offset_line_that_overflows_is_refused() -> None:
    curve = _bilinear()
    curve.strain[-1] = 1e306
    with pytest.raises(ValueError, match="offset stress difference must remain"):
        yield_strength(curve)


def test_first_crossing_is_retained_after_recovery_and_later_peaks() -> None:
    curve = _bilinear()
    curve.stress_mpa[-1] = 100
    assert yield_strength(curve).yield_strain == pytest.approx(29 / 900)


@pytest.mark.parametrize("slope", [100.0, 1000.0, 1234.567])
def test_an_exact_crossing_at_the_last_sample_survives_fit_roundoff(
    slope: float,
) -> None:
    curve = _bilinear()
    stress = slope * curve.strain
    stress[-1] = slope * (curve.strain[-1] - 0.002)
    result = yield_strength(replace(curve, stress_mpa=stress))
    assert result.resolved
    assert result.yield_strain == curve.strain[-1]
    assert result.strength_mpa == stress[-1]
    assert result.yield_bracket == (0.04, 0.06)


@pytest.mark.parametrize(
    "options",
    [
        {"offset_strain": 0.0},
        {"offset_strain": -0.1},
        {"offset_strain": np.nan},
        {"offset_strain": np.inf},
        {"fit_min_strain": -0.01},
        {"fit_min_strain": np.nan},
        {"fit_max_strain": np.inf},
        {"fit_max_strain": np.nan},
        {"fit_min_strain": 0.02, "fit_max_strain": 0.02},
        {"fit_min_strain": 0.02, "fit_max_strain": 0.01},
    ],
)
def test_invalid_yield_controls_are_rejected(options: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        yield_strength(_bilinear(), **options)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("controlled", "stress"),
        ("strain", np.arange(8)[::-1]),
        ("stress_mpa", np.full(8, np.nan)),
        ("lateral_strain", np.full((8, 2), -1.0)),
        ("temperature_k", 0.0),
        ("strain_rate_per_ns", np.inf),
    ],
)
def test_yield_uses_the_shared_tensile_validation(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        yield_strength(replace(_bilinear(), **{field: value}))
