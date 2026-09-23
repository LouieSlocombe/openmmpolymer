"""Analytical curves separate offset proof stress from a stress maximum."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from openmmpolymer.elasticity import StressStrain
from openmmpolymer.strength import yield_strength


def _curve() -> StressStrain:
    strain = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06])
    return StressStrain(
        stage="tensile",
        axis=2,
        strain=strain,
        stress_mpa=np.minimum(1000 * strain, 30 + 100 * (strain - 0.03)),
        lateral_strain=np.zeros((strain.size, 2)),
        lateral_stress_mpa=np.zeros((strain.size, 2)),
        temperature_k=298.15,
        strain_rate_per_ns=0.1,
    )


def test_bilinear_hardening_has_an_analytical_offset_intersection() -> None:
    curve = _curve()
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
    curve = _curve()
    base = yield_strength(curve)
    result = yield_strength(replace(curve, stress_mpa=curve.stress_mpa + 7))
    assert result.resolved
    assert result.yield_strain == pytest.approx(base.yield_strain)
    assert result.intercept_mpa == pytest.approx(7)
    assert base.strength_mpa is not None
    assert result.strength_mpa == pytest.approx(base.strength_mpa + 7)


def test_nominal_area_conversion_and_lateral_stress_subtraction() -> None:
    curve = _curve()
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
    curve = _curve()
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
    curve = _curve()
    result = yield_strength(replace(curve, stress_mpa=slope * curve.strain))
    assert not result.resolved
    assert result.strength_mpa is None
    assert result.yield_strain is None
    assert result.yield_bracket is None
    assert result.fit_resolved == (slope > 0)


@pytest.mark.parametrize("max_strain", [0.001, 0.006, 0.016])
def test_too_few_elastic_points_preserve_diagnostics(max_strain: float) -> None:
    result = yield_strength(_curve(), fit_max_strain=max_strain)
    assert not result.resolved
    assert not result.fit_resolved
    if result.fit_points == 1:
        assert result.modulus_mpa is None
        assert result.intercept_mpa is None
        assert result.standard_error_mpa is None
    else:
        assert result.modulus_mpa == pytest.approx(1000)


def test_a_noisy_elastic_window_is_unresolved() -> None:
    curve = _curve()
    curve.stress_mpa[:5] = [0, 80, -50, 70, 20]
    result = yield_strength(curve)
    assert not result.fit_resolved and not result.resolved
    assert any("Elastic fit unresolved" in note for note in result.notes)


def test_crossing_within_elastic_window_is_not_misreported_as_later_yield() -> None:
    curve = _curve()
    # The fitted window ends at .029, between the last elastic sample (.02)
    # and a much lower next measurement. Their crossing precedes .029.
    curve.stress_mpa[5:] = 20
    result = yield_strength(curve, fit_max_strain=0.029)
    assert result.fit_resolved
    assert not result.resolved
    assert any("within the elastic fit window" in note for note in result.notes)


def test_a_nonpositive_crossing_is_not_a_tensile_yield() -> None:
    curve = _curve()
    result = yield_strength(replace(curve, stress_mpa=curve.stress_mpa - 40))
    assert result.fit_resolved and not result.resolved
    assert any("positive tensile stress" in note for note in result.notes)


def test_first_crossing_is_retained_after_recovery_and_later_peaks() -> None:
    curve = _curve()
    curve.stress_mpa[-1] = 100
    assert yield_strength(curve).yield_strain == pytest.approx(29 / 900)


@pytest.mark.parametrize("slope", [100.0, 1000.0, 1234.567])
def test_an_exact_crossing_at_the_last_sample_survives_fit_roundoff(
    slope: float,
) -> None:
    curve = _curve()
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
def test_invalid_controls_are_rejected(options: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        yield_strength(_curve(), **options)


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
def test_invalid_curves_use_shared_tensile_validation(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        yield_strength(replace(_curve(), **{field: value}))
