"""Finite-rate modulus estimates and the limits of what those fits resolve."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from openmmpolymer.elasticity import ElasticModulus
from openmmpolymer.strain_rate import strain_rate_extrapolation
from openmmpolymer.trajectory import AnalysisError


def modulus_at(
    rate: float | None, modulus: float, *, error: float = 10.0
) -> ElasticModulus:
    return ElasticModulus(
        modulus_mpa=modulus,
        intercept_mpa=0.0,
        strain_limit=0.015,
        n_points=10,
        residual_mpa=0.1,
        standard_error_mpa=error,
        half_disagreement=0.0,
        temperature_k=298.15,
        strain_rate_per_ns=rate,
        resolved=True,
    )


def logarithmic_fits() -> list[ElasticModulus]:
    return [
        modulus_at(rate, value) for rate, value in [(0.1, 900), (1, 1000), (10, 1100)]
    ]


def test_logarithmic_correction_recovers_known_modulus_and_input_uncertainty() -> None:
    result = strain_rate_extrapolation(logarithmic_fits(), target_rate_per_ns=0.01)
    assert result.modulus_mpa == pytest.approx(800.0)
    assert result.modulus_gpa == pytest.approx(0.8)
    assert result.reference_rate_per_ns == pytest.approx(1.0)
    assert result.parameters["reference_modulus_mpa"] == pytest.approx(1000.0)
    assert result.sensitivity_mpa_per_decade == pytest.approx(100.0)
    assert result.residual_mpa == pytest.approx(0.0, abs=1e-10)
    # Input SE remains even though the rate dependence is an exact line.
    assert result.standard_error_mpa == pytest.approx(10 * math.sqrt(7 / 3))
    assert result.extrapolation_decades == pytest.approx(1.0)
    assert result.n_rates == 3
    assert result.temperature_k == pytest.approx(298.15)
    assert result.strain_limit == pytest.approx(0.015)
    assert result.resolved
    with pytest.raises(FrozenInstanceError):
        result.modulus_mpa = 123.0  # type: ignore[misc]


def test_arrays_sort_measurements_and_uncertainties_together() -> None:
    fits = logarithmic_fits()
    fits[2] = replace(fits[2], standard_error_mpa=30.0)
    result = strain_rate_extrapolation(fits[::-1], target_rate_per_ns=1.0)
    np.testing.assert_array_equal(result.strain_rate_per_ns, [0.1, 1.0, 10.0])
    np.testing.assert_array_equal(result.moduli_mpa, [900.0, 1000.0, 1100.0])
    np.testing.assert_array_equal(result.standard_errors_mpa, [10.0, 10.0, 30.0])
    assert result.standard_error_mpa == pytest.approx(math.sqrt(1100.0) / 3)


def test_power_law_recovers_exponent_and_propagates_relative_uncertainty() -> None:
    rates = np.asarray([0.1, 1.0, 10.0])
    values = 1000.0 * rates**0.2
    fits = [
        modulus_at(float(r), float(e), error=float(e) * 0.01)
        for r, e in zip(rates, values, strict=True)
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=0.01, form="power_law")
    expected = 1000.0 * 0.01**0.2
    assert result.modulus_mpa == pytest.approx(expected)
    assert result.parameters["exponent"] == pytest.approx(0.2)
    assert result.parameters["reference_modulus_mpa"] == pytest.approx(1000.0)
    assert result.sensitivity_mpa_per_decade == pytest.approx(
        1000.0 * 0.2 * math.log(10)
    )
    assert result.standard_error_mpa == pytest.approx(
        expected * 0.01 * math.sqrt(7 / 3)
    )
    assert result.residual_mpa == pytest.approx(0.0, abs=1e-9)
    assert result.resolved


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_predict_evaluates_scalar_and_array_consistently(form: str) -> None:
    result = strain_rate_extrapolation(
        logarithmic_fits(), target_rate_per_ns=0.03, form=form
    )
    assert result.predict(0.03) == pytest.approx(result.modulus_mpa)
    predictions = result.predict(np.asarray([0.03, 1.0]))
    assert predictions[0] == pytest.approx(result.modulus_mpa)
    assert predictions[1] == pytest.approx(result.parameters["reference_modulus_mpa"])


@pytest.mark.parametrize("rate", [0.0, -1.0, math.nan, math.inf])
def test_predict_rejects_invalid_rates(rate: float) -> None:
    result = strain_rate_extrapolation(logarithmic_fits(), target_rate_per_ns=1.0)
    with pytest.raises(ValueError, match="finite positive"):
        result.predict(rate)
    with pytest.raises(ValueError, match="finite positive"):
        result.predict(np.asarray([1.0, rate]))


def test_residual_scatter_contributes_even_when_input_errors_are_zero() -> None:
    fits = [
        modulus_at(rate, value, error=0.0)
        for rate, value in [(0.01, 820), (0.1, 880), (1.0, 980), (10, 1120)]
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=math.sqrt(0.1))
    # Deviations [20, -20, -20, 20] are orthogonal to the line.
    assert result.modulus_mpa == pytest.approx(950.0)
    assert result.residual_mpa == pytest.approx(20.0)
    assert result.standard_error_mpa == pytest.approx(math.sqrt(200.0))
    assert result.resolved


def test_known_error_is_not_counted_again_as_residual_scatter() -> None:
    fits = [
        modulus_at(rate, value, error=100.0)
        for rate, value in [(0.01, 820), (0.1, 880), (1.0, 980), (10, 1120)]
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=math.sqrt(0.1))
    assert result.standard_error_mpa == pytest.approx(50.0)


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_more_rates_cannot_make_a_discontinuous_response_resolve(form: str) -> None:
    rates = np.geomspace(0.01, 100.0, 21)
    fits = [
        modulus_at(float(rate), 1000.0 if rate < 1.0 else 2000.0, error=10.0)
        for rate in rates
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=0.005, form=form)
    # Mean uncertainty alone would accept this wrong functional form.
    assert result.standard_error_mpa < 0.25 * result.modulus_mpa
    assert result.residual_to_error_ratio is not None
    assert result.residual_to_error_ratio > 20.0
    assert result.relative_residual > 0.10
    assert not result.resolved
    assert any("3 times" in note for note in result.notes)


@pytest.mark.parametrize("error", [0.0, 1000.0])
@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_large_relative_residual_refuses_zero_or_loose_input_errors(
    error: float, form: str
) -> None:
    rates = np.geomspace(0.01, 100.0, 101)
    fits = [
        modulus_at(float(rate), 1000.0 if rate < 1.0 else 2000.0, error=error)
        for rate in rates
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=1.0, form=form)
    assert result.standard_error_mpa < 0.25 * result.modulus_mpa
    if error == 0.0:
        assert result.residual_to_error_ratio is None
    else:
        assert result.residual_to_error_ratio is not None
        assert result.residual_to_error_ratio < 3.0
    assert result.relative_residual > 0.10
    assert not result.resolved
    assert any("10%" in note for note in result.notes)


def test_small_relative_residual_still_must_match_the_reported_precision() -> None:
    fits = [
        modulus_at(rate, value, error=1.0)
        for rate, value in [(0.01, 820), (0.1, 880), (1.0, 980), (10, 1120)]
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=math.sqrt(0.1))
    assert result.relative_residual == pytest.approx(20.0 / 950.0)
    assert result.residual_to_error_ratio == pytest.approx(20.0)
    assert not result.resolved


def test_floating_point_residual_does_not_refuse_a_numerically_exact_relation() -> None:
    rates = np.geomspace(0.1, 11.13, 13)
    fits = [
        modulus_at(float(rate), 1000.0 + 100.0 * math.log10(rate), error=1e-30)
        for rate in rates
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=0.05)
    assert result.relative_residual < 1e-14
    assert result.resolved


def test_negative_sensitivity_is_reported_without_forcing_a_downward_correction() -> (
    None
):
    fits = [
        modulus_at(rate, value) for rate, value in [(0.1, 1100), (1, 1000), (10, 900)]
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=0.01)
    assert result.modulus_mpa == pytest.approx(1200.0)
    assert result.sensitivity_mpa_per_decade == pytest.approx(-100.0)
    assert not result.resolved
    assert any("negative" in note for note in result.notes)


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_rate_independent_modulus_is_valid(form: str) -> None:
    fits = [modulus_at(rate, 1000.0) for rate in [0.1, 1.0, 10.0]]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=0.01, form=form)
    assert result.modulus_mpa == pytest.approx(1000.0)
    assert result.sensitivity_mpa_per_decade == pytest.approx(0.0)
    assert result.resolved


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
@pytest.mark.parametrize("modulus", [0.15, 1.6322159939602894])
def test_uneven_rates_do_not_give_a_flat_response_a_rounding_slope(
    form: str, modulus: float
) -> None:
    rates = np.geomspace(0.1, 11.13, 10)
    rates[1] *= 1.13
    fits = [modulus_at(float(rate), modulus, error=modulus * 0.01) for rate in rates]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=1.0, form=form)
    assert result.modulus_mpa == pytest.approx(modulus)
    assert result.sensitivity_mpa_per_decade == 0.0
    assert result.resolved


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_nearly_identical_rates_cannot_support_a_precise_extrapolation(
    form: str,
) -> None:
    rates = [1.0 - 1e-12, 1.0, 1.0 + 1e-12]
    fits = [modulus_at(rate, 1000.0, error=10.0) for rate in rates]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=0.01, form=form)
    assert result.modulus_mpa == pytest.approx(1000.0)
    assert math.isfinite(result.standard_error_mpa)
    assert result.standard_error_mpa > 1e13
    assert not result.resolved
    assert any("25%" in note for note in result.notes)


def test_nonpositive_prediction_is_retained_but_unresolved() -> None:
    fits = [
        modulus_at(rate, value) for rate, value in [(0.1, 100), (1, 200), (10, 300)]
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=0.001)
    assert result.modulus_mpa == pytest.approx(-100.0)
    assert not result.resolved
    assert any("nonpositive" in note for note in result.notes)


def test_large_target_uncertainty_is_unresolved() -> None:
    fits = [replace(fit, standard_error_mpa=300.0) for fit in logarithmic_fits()]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=0.01)
    assert result.standard_error_mpa > 0.25 * result.modulus_mpa
    assert not result.resolved
    assert any("25%" in note for note in result.notes)


def test_extrapolation_limit_is_measured_from_nearest_observed_rate() -> None:
    fits = logarithmic_fits()
    boundary = strain_rate_extrapolation(fits, target_rate_per_ns=0.001)
    too_far = strain_rate_extrapolation(fits, target_rate_per_ns=0.0001)
    permitted = strain_rate_extrapolation(
        fits, target_rate_per_ns=0.0001, max_extrapolation_decades=3.0
    )
    interpolation = strain_rate_extrapolation(
        fits, target_rate_per_ns=0.3, max_extrapolation_decades=0.0
    )
    above = strain_rate_extrapolation(fits, target_rate_per_ns=10000.0)
    assert boundary.extrapolation_decades == pytest.approx(2.0)
    assert boundary.resolved
    assert too_far.extrapolation_decades == pytest.approx(3.0)
    assert not too_far.resolved
    assert permitted.resolved
    assert interpolation.extrapolation_decades == 0.0
    assert interpolation.resolved
    assert above.extrapolation_decades == pytest.approx(3.0)
    assert not above.resolved


def test_unresolved_input_propagates_to_the_rate_estimate() -> None:
    fits = logarithmic_fits()
    fits[0] = replace(fits[0], resolved=False)
    result = strain_rate_extrapolation(fits, target_rate_per_ns=1.0)
    assert not result.resolved
    assert any("input" in note for note in result.notes)


def test_small_temperature_fluctuations_are_accepted_and_averaged() -> None:
    fits = logarithmic_fits()
    fits[0] = replace(fits[0], temperature_k=297.8)
    fits[2] = replace(fits[2], temperature_k=298.7)
    result = strain_rate_extrapolation(fits, target_rate_per_ns=1.0)
    assert result.temperature_k == pytest.approx((297.8 + 298.15 + 298.7) / 3.0)
    assert result.resolved


@pytest.mark.parametrize("count", [0, 1, 2])
def test_three_rates_are_required(count: int) -> None:
    with pytest.raises(AnalysisError, match="three or more distinct"):
        strain_rate_extrapolation(logarithmic_fits()[:count], target_rate_per_ns=1.0)


def test_replicas_cannot_masquerade_as_additional_rates() -> None:
    fits = logarithmic_fits()
    with pytest.raises(AnalysisError, match="Pool replicas"):
        strain_rate_extrapolation([*fits, fits[0]], target_rate_per_ns=1.0)


@pytest.mark.parametrize("value", [None, 0.0, -1.0, math.nan, math.inf])
def test_input_rate_must_be_present_finite_and_positive(value: float | None) -> None:
    fits = logarithmic_fits()
    fits[0] = replace(fits[0], strain_rate_per_ns=value)
    with pytest.raises(AnalysisError, match="strain rate"):
        strain_rate_extrapolation(fits, target_rate_per_ns=1.0)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("field", ["modulus_mpa", "temperature_k", "strain_limit"])
def test_input_values_must_be_finite_and_positive(field: str, value: float) -> None:
    fits = logarithmic_fits()
    fits[0] = replace(
        fits[0],
        modulus_mpa=value if field == "modulus_mpa" else fits[0].modulus_mpa,
        temperature_k=value if field == "temperature_k" else fits[0].temperature_k,
        strain_limit=value if field == "strain_limit" else fits[0].strain_limit,
    )
    with pytest.raises(AnalysisError, match="finite and positive"):
        strain_rate_extrapolation(fits, target_rate_per_ns=1.0)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_input_error_must_be_finite_and_nonnegative(value: float) -> None:
    fits = logarithmic_fits()
    fits[0] = replace(fits[0], standard_error_mpa=value)
    with pytest.raises(AnalysisError, match="Standard error"):
        strain_rate_extrapolation(fits, target_rate_per_ns=1.0)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("temperature_k", 305.0, "same temperature"),
        ("strain_limit", 0.03, "same strain limit"),
    ],
)
def test_incompatible_metadata_cannot_be_mixed(
    field: str, value: float, message: str
) -> None:
    fits = logarithmic_fits()
    fits[0] = replace(
        fits[0],
        temperature_k=value if field == "temperature_k" else fits[0].temperature_k,
        strain_limit=value if field == "strain_limit" else fits[0].strain_limit,
    )
    with pytest.raises(AnalysisError, match=message):
        strain_rate_extrapolation(fits, target_rate_per_ns=1.0)


@pytest.mark.parametrize("target", [0.0, -1.0, math.nan, math.inf])
def test_target_must_be_finite_and_positive(target: float) -> None:
    with pytest.raises(ValueError, match="target_rate_per_ns"):
        strain_rate_extrapolation(logarithmic_fits(), target_rate_per_ns=target)


@pytest.mark.parametrize("limit", [-1.0, math.nan, math.inf])
def test_extrapolation_limit_must_be_finite_and_nonnegative(limit: float) -> None:
    with pytest.raises(ValueError, match="max_extrapolation_decades"):
        strain_rate_extrapolation(
            logarithmic_fits(), target_rate_per_ns=1.0, max_extrapolation_decades=limit
        )


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="form must"):
        strain_rate_extrapolation(
            logarithmic_fits(), target_rate_per_ns=1.0, form="equilibrium"
        )


def test_centering_handles_extreme_rate_units_without_overflowing_ratios() -> None:
    fits = [
        modulus_at(rate, value)
        for rate, value in [(1e-300, 900), (1e-200, 1000), (1e-100, 1100)]
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=1e-250)
    assert result.modulus_mpa == pytest.approx(950.0)
    assert result.reference_rate_per_ns == pytest.approx(1e-200)
    assert result.predict(1e-250) == pytest.approx(950.0)
    assert result.resolved


def test_overflowing_power_law_prediction_is_unresolved_without_runtime_warning() -> (
    None
):
    fits = [
        modulus_at(rate, value, error=0.0)
        for rate, value in [(0.1, 1.0), (1, 1000.0), (10, 1000000.0)]
    ]
    result = strain_rate_extrapolation(fits, target_rate_per_ns=1e300, form="power_law")
    assert math.isinf(result.modulus_mpa)
    assert not result.resolved
    assert math.isinf(result.predict(1e300))
