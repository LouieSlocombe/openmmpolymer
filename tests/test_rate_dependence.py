"""General rate fits retain units, censored events, replicas and uncertainty."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from openmmpolymer.elasticity import ElasticModulus
from openmmpolymer.rate_dependence import (
    RateObservation,
    RateProperty,
    analyse_rate_observations,
    rate_extrapolation,
)
from openmmpolymer.strain_rate import strain_rate_extrapolation
from openmmpolymer.trajectory import AnalysisError

PROPERTY = RateProperty(
    "yield_strength", "Yield strength", "MPa", "strain/ns", "increasing"
)


def observations() -> list[RateObservation]:
    return [
        RateObservation(rate, value, 10.0, True, 298.15, {"axis": "x"})
        for rate, value in [(0.1, 900.0), (1.0, 1000.0), (10.0, 1100.0)]
    ]


def _value(item: RateObservation) -> float:
    assert item.value is not None
    return item.value


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_matches_young_modulus_regression_and_guards(form: str) -> None:
    source = observations()
    young = [
        ElasticModulus(
            modulus_mpa=_value(item),
            intercept_mpa=0.0,
            strain_limit=0.015,
            n_points=10,
            residual_mpa=0.1,
            standard_error_mpa=10.0,
            half_disagreement=0.0,
            temperature_k=298.15,
            strain_rate_per_ns=item.rate,
            resolved=True,
        )
        for item in source
    ]
    expected = strain_rate_extrapolation(young, target_rate_per_ns=0.01, form=form)
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.01, form=form)
    assert result.value == pytest.approx(expected.modulus_mpa)
    assert result.standard_error == pytest.approx(expected.standard_error_mpa)
    assert result.sensitivity_per_decade == pytest.approx(
        expected.sensitivity_mpa_per_decade
    )
    assert result.residual == pytest.approx(expected.residual_mpa)
    assert result.relative_residual == pytest.approx(expected.relative_residual)
    assert result.resolved == expected.resolved
    assert result.reference_rate == pytest.approx(expected.reference_rate_per_ns)
    assert result.extrapolation_decades == expected.extrapolation_decades
    assert result.predict(0.01) == pytest.approx(result.value)
    np.testing.assert_allclose(
        result.predict(result.rates), expected.predict(result.rates)
    )


def test_logarithmic_exact_line_preserves_input_uncertainty_and_sorting() -> None:
    result = rate_extrapolation(
        observations()[::-1], property=PROPERTY, target_rate=0.01
    )
    assert result.value == pytest.approx(800.0)
    assert result.standard_error == pytest.approx(10.0 * math.sqrt(7 / 3))
    assert result.sensitivity_per_decade == pytest.approx(100.0)
    np.testing.assert_array_equal(result.rates, [0.1, 1.0, 10.0])
    assert result.resolved
    with pytest.raises(FrozenInstanceError):
        result.value = 123.0  # type: ignore[misc]


def test_power_law_recovers_exponent_with_relative_error_propagation() -> None:
    source = [
        RateObservation(rate, 1000.0 * rate**0.2, 10.0 * rate**0.2, True)
        for rate in (0.1, 1.0, 10.0)
    ]
    result = rate_extrapolation(
        source, property=PROPERTY, target_rate=0.01, form="power_law"
    )
    expected = 1000.0 * 0.01**0.2
    assert result.value == pytest.approx(expected)
    assert result.parameters["exponent"] == pytest.approx(0.2)
    assert result.standard_error == pytest.approx(expected * 0.01 * math.sqrt(7 / 3))
    assert result.sensitivity_per_decade == pytest.approx(200.0 * math.log(10.0))


def test_unit_changes_transform_values_errors_and_slopes() -> None:
    source = observations()
    original = rate_extrapolation(source, property=PROPERTY, target_rate=0.01)
    in_gpa = [
        replace(
            item,
            rate=item.rate * 1e9,
            value=_value(item) / 1000.0,
            standard_error=0.01,
        )
        for item in source
    ]
    converted = rate_extrapolation(
        in_gpa,
        property=replace(PROPERTY, value_unit="GPa", rate_unit="s^-1"),
        target_rate=1e7,
    )
    assert converted.value == pytest.approx(original.value / 1000.0)
    assert converted.standard_error == pytest.approx(original.standard_error / 1000.0)
    assert converted.sensitivity_per_decade == pytest.approx(
        original.sensitivity_per_decade / 1000.0
    )
    assert converted.resolved


@pytest.mark.parametrize(
    "trend, resolved", [("any", True), ("decreasing", True), ("increasing", False)]
)
def test_trend_is_configurable_without_constraining_fit(
    trend: str, resolved: bool
) -> None:
    source = [replace(item, value=2000.0 - _value(item)) for item in observations()]
    result = rate_extrapolation(
        source, property=replace(PROPERTY, trend=trend), target_rate=0.01
    )
    assert result.value == pytest.approx(1200.0)
    assert result.sensitivity_per_decade == pytest.approx(-100.0)
    assert result.resolved is resolved


@pytest.mark.parametrize("trend", ["any", "increasing", "decreasing"])
def test_constant_response_satisfies_every_trend(trend: str) -> None:
    source = [replace(item, value=1000.0) for item in observations()]
    result = rate_extrapolation(
        source, property=replace(PROPERTY, trend=trend), target_rate=0.01
    )
    assert result.sensitivity_per_decade == 0.0
    assert result.resolved


def test_signed_poisson_ratio_allows_negative_log_prediction_but_not_power_law() -> (
    None
):
    property = RateProperty(
        "poisson_ratio",
        "Poisson ratio",
        "",
        "strain/ns",
        lower_bound=-1.0,
        upper_bound=0.5,
    )
    source = [
        RateObservation(rate, value, 0.001, True)
        for rate, value in [(0.1, -0.2), (1.0, -0.1), (10.0, 0.0)]
    ]
    report = analyse_rate_observations(source, property=property, target_rate=0.01)
    assert report.log_linear is not None
    assert report.log_linear.value == pytest.approx(-0.3)
    assert report.log_linear.resolved
    assert report.power_law is None
    assert any("strictly positive" in note for note in report.notes)


@pytest.mark.parametrize(
    "bound,target", [("lower_bound", 0.01), ("upper_bound", 100.0)]
)
def test_target_must_be_strictly_inside_physical_bounds(
    bound: str, target: float
) -> None:
    property = (
        replace(PROPERTY, lower_bound=800.0)
        if bound == "lower_bound"
        else replace(PROPERTY, upper_bound=1200.0)
    )
    result = rate_extrapolation(observations(), property=property, target_rate=target)
    assert not result.resolved
    assert any("strict physical bounds" in note for note in result.notes)


def test_distinct_rate_count_cannot_be_inflated_with_replicas() -> None:
    source = observations()[:2] * 10
    with pytest.raises(AnalysisError, match="three or more distinct"):
        rate_extrapolation(source, property=PROPERTY, target_rate=0.01)


def test_near_equal_rates_are_replicas_not_independent_rate_information() -> None:
    source = [
        RateObservation(rate, 1000.0, 1.0, True) for rate in (1.0, 1.0 + 1e-10, 10.0)
    ]
    with pytest.raises(AnalysisError, match="three or more distinct"):
        rate_extrapolation(source, property=PROPERTY, target_rate=0.01)


def test_pooling_preserves_replica_spread_without_counting_extra_rates() -> None:
    source = [
        replace(item, value=_value(item) + offset, standard_error=0.0)
        for item in observations()
        for offset in (-10.0, 10.0)
    ]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.01)
    assert result.n_rates == 3
    np.testing.assert_allclose(result.values, [900.0, 1000.0, 1100.0])
    np.testing.assert_allclose(result.standard_errors, [math.sqrt(200.0)] * 3)
    assert result.standard_error == pytest.approx(math.sqrt(200.0 * 7 / 3))
    assert result.resolved


def test_pooling_unknown_within_replica_errors_can_use_nonzero_sample_spread() -> None:
    source = [
        replace(item, value=_value(item) + offset, standard_error=None)
        for item in observations()
        for offset in (-10.0, 10.0)
    ]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.01)
    np.testing.assert_allclose(result.standard_errors, [math.sqrt(200.0)] * 3)
    assert result.resolved


@pytest.mark.parametrize("repeat", [1, 2])
def test_unknown_error_never_becomes_zero_confidence(repeat: int) -> None:
    source = [replace(item, standard_error=None) for item in observations()] * repeat
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.01)
    assert result.value == pytest.approx(800.0)
    assert math.isnan(result.standard_error)
    assert np.all(np.isnan(result.standard_errors))
    assert result.residual_to_error_ratio is None
    assert not result.resolved
    assert any("uncertainty is unknown" in note for note in result.notes)


def test_unresolved_replica_cannot_be_hidden_by_pooling() -> None:
    source = observations() + [replace(item, resolved=False) for item in observations()]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.01)
    assert not result.resolved
    assert any("input observation is unresolved" in note for note in result.notes)


def test_censored_value_is_retained_and_blocks_both_models() -> None:
    source = [
        *observations(),
        RateObservation(
            100.0, None, None, False, notes=("No failure before trajectory ended.",)
        ),
    ]
    report = analyse_rate_observations(source, property=PROPERTY, target_rate=0.01)
    assert report.observations == tuple(source)
    assert report.log_linear is None and report.power_law is None
    assert report.target_rate == 0.01
    assert any("No failure" in note for note in report.notes)
    assert any("censored" in note for note in report.notes)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"temperature_k": 299.151}, "within 1 K"),
        ({"temperature_k": -1.0}, "temperature"),
        ({"conditions": {"axis": "y"}}, "same measurement conditions"),
        ({"conditions": {}}, "same measurement conditions"),
        ({"rate": 0.0}, "positive rate"),
        ({"rate": math.inf}, "positive rate"),
        ({"value": math.nan}, "physical bounds"),
        ({"value": 0.0}, "physical bounds"),
        ({"standard_error": -1.0}, "standard error"),
        ({"standard_error": math.inf}, "standard error"),
    ],
)
def test_incompatible_or_invalid_observations_cannot_fit(
    mutation: dict[str, Any], message: str
) -> None:
    source = observations()
    source[-1] = replace(source[-1], **mutation)
    with pytest.raises(AnalysisError, match=message):
        rate_extrapolation(source, property=PROPERTY, target_rate=0.01)


def test_nested_conditions_and_temperature_tolerance() -> None:
    source = [
        replace(
            item,
            temperature_k=298.0 + index / 2,
            conditions={"windows": [0.0, 0.015], "axis": {"name": "x"}},
        )
        for index, item in enumerate(observations())
    ]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.01)
    assert result.resolved


def test_partially_missing_temperatures_are_unresolved() -> None:
    source = observations()
    source[-1] = replace(source[-1], temperature_k=None)
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.01)
    assert not result.resolved
    assert any("temperatures are unknown" in note for note in result.notes)


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
@pytest.mark.parametrize("error", [0.0, 1.0, 1000.0])
def test_dense_discontinuous_response_is_unresolved_despite_small_mean_error(
    form: str, error: float
) -> None:
    source = [
        RateObservation(float(rate), 1000.0 if rate < 1.0 else 2000.0, error, True)
        for rate in np.geomspace(0.01, 100.0, 101)
    ]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=1.0, form=form)
    assert result.standard_error < 0.25 * result.value
    assert result.relative_residual > 0.1
    assert not result.resolved
    assert any("10%" in note for note in result.notes)


def test_small_relative_residual_must_match_reported_precision() -> None:
    source = [
        RateObservation(rate, value, 1.0, True)
        for rate, value in [(0.01, 820.0), (0.1, 880.0), (1.0, 980.0), (10.0, 1120.0)]
    ]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.1)
    assert result.relative_residual == pytest.approx(20 / 950)
    assert result.residual_to_error_ratio == pytest.approx(20)
    assert not result.resolved
    assert any("3 times" in note for note in result.notes)


def test_extrapolation_distance_and_large_uncertainty_remain_unresolved() -> None:
    distant = rate_extrapolation(observations(), property=PROPERTY, target_rate=1e-8)
    assert distant.extrapolation_decades == 7.0
    assert not distant.resolved
    broad = rate_extrapolation(
        [replace(item, standard_error=1000.0) for item in observations()],
        property=PROPERTY,
        target_rate=0.01,
    )
    assert not broad.resolved
    assert any("25%" in note for note in broad.notes)


@pytest.mark.parametrize("target", [0.0, -1.0, math.inf, math.nan])
def test_invalid_request_is_not_swallowed_by_report(target: float) -> None:
    with pytest.raises(ValueError, match="target_rate"):
        analyse_rate_observations([], property=PROPERTY, target_rate=target)
    fit = rate_extrapolation(observations(), property=PROPERTY, target_rate=0.01)
    with pytest.raises(ValueError, match="finite positive"):
        fit.predict(target)


@pytest.mark.parametrize("maximum", [-1.0, math.inf, math.nan])
def test_invalid_distance_is_not_swallowed(maximum: float) -> None:
    with pytest.raises(ValueError, match="max_extrapolation_decades"):
        analyse_rate_observations(
            [], property=PROPERTY, target_rate=0.01, max_extrapolation_decades=maximum
        )


def test_bad_form_and_property_contracts_are_invalid_requests() -> None:
    with pytest.raises(ValueError, match="form"):
        rate_extrapolation(
            observations(), property=PROPERTY, target_rate=0.01, form="cubic"
        )
    with pytest.raises(ValueError, match="trend"):
        replace(PROPERTY, trend="positive")
    with pytest.raises(ValueError, match="bounds"):
        replace(PROPERTY, lower_bound=math.nan)
    with pytest.raises(ValueError, match="smaller"):
        replace(PROPERTY, upper_bound=0.0)


def test_empty_report_keeps_request_and_unavailable_models() -> None:
    report = analyse_rate_observations(
        [], property=PROPERTY, target_rate=0.01, run_dirs=["run-a"]
    )
    assert report.log_linear is None and report.power_law is None
    assert report.target_rate == 0.01
    assert report.run_dirs == ("run-a",)
