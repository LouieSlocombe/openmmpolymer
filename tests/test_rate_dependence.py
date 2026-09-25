"""General rate fits retain units, censored events, replicas and uncertainty."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from openmmpolymer.rate_dependence import (
    RateObservation,
    RateProperty,
    analyse_rate_observations,
    rate_extrapolation,
)
from openmmpolymer.trajectory import AnalysisError

PROPERTY = RateProperty(
    "yield_strength", "Yield strength", "MPa", "strain/ns", "increasing"
)


def observations() -> list[RateObservation]:
    return [
        RateObservation(rate, value, 10.0, True, 298.15, {"axis": "x"})
        for rate, value in [(0.1, 900.0), (1.0, 1000.0), (10.0, 1100.0)]
    ]


def measured(
    pairs: list[tuple[float, float]], *, error: float = 10.0
) -> list[RateObservation]:
    return [RateObservation(rate, value, error, True) for rate, value in pairs]


def _value(item: RateObservation) -> float:
    assert item.value is not None
    return item.value


#: Deviations of +20, -20, -20, +20 MPa about a line through (0.01 ... 10) /ns,
#: orthogonal to it, so the fitted line is exact and the residual is 20 MPa.
SCATTERED = [(0.01, 820.0), (0.1, 880.0), (1.0, 980.0), (10.0, 1120.0)]


def test_logarithmic_exact_line_preserves_input_uncertainty_and_sorting() -> None:
    result = rate_extrapolation(
        observations()[::-1], property=PROPERTY, target_rate=0.01
    )
    assert result.value == pytest.approx(800.0)
    assert result.standard_error == pytest.approx(10.0 * math.sqrt(7 / 3))
    assert result.sensitivity_per_decade == pytest.approx(100.0)
    assert result.parameters["reference_value"] == pytest.approx(1000.0)
    assert result.reference_rate == pytest.approx(1.0)
    assert result.residual == pytest.approx(0.0, abs=1e-10)
    assert result.extrapolation_decades == pytest.approx(1.0)
    np.testing.assert_array_equal(result.rates, [0.1, 1.0, 10.0])
    assert result.resolved
    with pytest.raises(FrozenInstanceError):
        result.value = 123.0  # type: ignore[misc]


def test_arrays_sort_values_and_uncertainties_together() -> None:
    source = observations()
    source[2] = replace(source[2], standard_error=30.0)
    result = rate_extrapolation(source[::-1], property=PROPERTY, target_rate=1.0)
    np.testing.assert_array_equal(result.values, [900.0, 1000.0, 1100.0])
    np.testing.assert_array_equal(result.standard_errors, [10.0, 10.0, 30.0])
    assert result.standard_error == pytest.approx(math.sqrt(1100.0) / 3)


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
    assert result.parameters["reference_value"] == pytest.approx(1000.0)
    assert result.standard_error == pytest.approx(expected * 0.01 * math.sqrt(7 / 3))
    assert result.sensitivity_per_decade == pytest.approx(200.0 * math.log(10.0))
    assert result.residual == pytest.approx(0.0, abs=1e-9)
    assert result.resolved


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_predict_evaluates_scalar_and_array_consistently(form: str) -> None:
    result = rate_extrapolation(
        observations(), property=PROPERTY, target_rate=0.03, form=form
    )
    assert result.predict(0.03) == pytest.approx(result.value)
    predictions = result.predict(np.asarray([0.03, 1.0]))
    assert predictions[0] == pytest.approx(result.value)
    assert predictions[1] == pytest.approx(result.parameters["reference_value"])


@pytest.mark.parametrize("rate", [0.0, -1.0, math.nan, math.inf])
def test_predict_rejects_invalid_rates(rate: float) -> None:
    result = rate_extrapolation(observations(), property=PROPERTY, target_rate=1.0)
    with pytest.raises(ValueError, match="finite positive"):
        result.predict(rate)
    with pytest.raises(ValueError, match="finite positive"):
        result.predict(np.asarray([1.0, rate]))


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


def test_centring_handles_extreme_rate_units_without_overflowing_ratios() -> None:
    source = measured([(1e-300, 900.0), (1e-200, 1000.0), (1e-100, 1100.0)])
    result = rate_extrapolation(source, property=PROPERTY, target_rate=1e-250)
    assert result.value == pytest.approx(950.0)
    assert result.reference_rate == pytest.approx(1e-200)
    assert result.predict(1e-250) == pytest.approx(950.0)
    assert result.resolved


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
    if not resolved:
        assert any("contradicts" in note for note in result.notes)


@pytest.mark.parametrize("trend", ["any", "increasing", "decreasing"])
@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_constant_response_satisfies_every_trend(trend: str, form: str) -> None:
    source = [replace(item, value=1000.0) for item in observations()]
    result = rate_extrapolation(
        source, property=replace(PROPERTY, trend=trend), target_rate=0.01, form=form
    )
    assert result.value == pytest.approx(1000.0)
    assert result.sensitivity_per_decade == 0.0
    assert result.resolved


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
@pytest.mark.parametrize("value", [0.15, 1.6322159939602894])
def test_uneven_rates_do_not_give_a_flat_response_a_rounding_slope(
    form: str, value: float
) -> None:
    rates = np.geomspace(0.1, 11.13, 10)
    rates[1] *= 1.13
    source = [RateObservation(float(rate), value, value * 0.01, True) for rate in rates]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=1.0, form=form)
    assert result.value == pytest.approx(value)
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


def test_overflowing_power_law_prediction_is_unresolved_without_runtime_warning() -> (
    None
):
    source = measured([(0.1, 1.0), (1.0, 1000.0), (10.0, 1000000.0)], error=0.0)
    result = rate_extrapolation(
        source, property=PROPERTY, target_rate=1e300, form="power_law"
    )
    assert math.isinf(result.value)
    assert not result.resolved
    assert math.isinf(result.predict(1e300))


@pytest.mark.parametrize("count", [0, 1, 2])
def test_three_rates_are_required(count: int) -> None:
    with pytest.raises(AnalysisError, match="three or more distinct"):
        rate_extrapolation(observations()[:count], property=PROPERTY, target_rate=1.0)


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
    assert any("Repeated rates were pooled" in note for note in result.notes)


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
        ({"temperature_k": math.nan}, "temperature"),
        ({"conditions": {"axis": "y"}}, "same measurement conditions"),
        ({"conditions": {}}, "same measurement conditions"),
        ({"rate": 0.0}, "positive rate"),
        ({"rate": -1.0}, "positive rate"),
        ({"rate": math.nan}, "positive rate"),
        ({"rate": math.inf}, "positive rate"),
        ({"value": math.nan}, "physical bounds"),
        ({"value": math.inf}, "physical bounds"),
        ({"value": 0.0}, "physical bounds"),
        ({"standard_error": -1.0}, "standard error"),
        ({"standard_error": math.nan}, "standard error"),
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


def test_residual_scatter_contributes_even_when_input_errors_are_zero() -> None:
    result = rate_extrapolation(
        measured(SCATTERED, error=0.0), property=PROPERTY, target_rate=math.sqrt(0.1)
    )
    assert result.value == pytest.approx(950.0)
    assert result.residual == pytest.approx(20.0)
    assert result.standard_error == pytest.approx(math.sqrt(200.0))
    assert result.residual_to_error_ratio is None
    assert result.resolved


def test_known_error_is_not_counted_again_as_residual_scatter() -> None:
    result = rate_extrapolation(
        measured(SCATTERED, error=100.0), property=PROPERTY, target_rate=math.sqrt(0.1)
    )
    assert result.standard_error == pytest.approx(50.0)


def test_small_relative_residual_must_match_reported_precision() -> None:
    result = rate_extrapolation(
        measured(SCATTERED, error=1.0), property=PROPERTY, target_rate=0.1
    )
    assert result.relative_residual == pytest.approx(20 / 950)
    assert result.residual_to_error_ratio == pytest.approx(20)
    assert not result.resolved
    assert any("3 times" in note for note in result.notes)


def test_floating_point_residual_does_not_refuse_a_numerically_exact_relation() -> None:
    source = [
        RateObservation(float(rate), 1000.0 + 100.0 * math.log10(rate), 1e-30, True)
        for rate in np.geomspace(0.1, 11.13, 13)
    ]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.05)
    assert result.relative_residual < 1e-14
    assert result.resolved


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_more_rates_cannot_make_a_discontinuous_response_resolve(form: str) -> None:
    source = [
        RateObservation(float(rate), 1000.0 if rate < 1.0 else 2000.0, 10.0, True)
        for rate in np.geomspace(0.01, 100.0, 21)
    ]
    result = rate_extrapolation(source, property=PROPERTY, target_rate=0.005, form=form)
    # Mean uncertainty alone would accept this wrong functional form.
    assert result.standard_error < 0.25 * result.value
    assert result.residual_to_error_ratio is not None
    assert result.residual_to_error_ratio > 20.0
    assert not result.resolved
    assert any("3 times" in note for note in result.notes)


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
    if error == 0.0:
        assert result.residual_to_error_ratio is None
    elif error == 1000.0:
        assert result.residual_to_error_ratio is not None
        assert result.residual_to_error_ratio < 3.0
    assert result.relative_residual > 0.1
    assert not result.resolved
    assert any("10%" in note for note in result.notes)


def test_extrapolation_limit_is_measured_from_nearest_observed_rate() -> None:
    def fit(target: float, maximum: float = 2.0) -> Any:
        return rate_extrapolation(
            observations(),
            property=PROPERTY,
            target_rate=target,
            max_extrapolation_decades=maximum,
        )

    boundary, too_far = fit(0.001), fit(0.0001)
    assert boundary.extrapolation_decades == pytest.approx(2.0)
    assert boundary.resolved
    assert too_far.extrapolation_decades == pytest.approx(3.0)
    assert not too_far.resolved
    assert any("2-decade limit" in note for note in too_far.notes)
    assert fit(0.0001, 3.0).resolved
    interpolated = fit(0.3, 0.0)
    assert interpolated.extrapolation_decades == 0.0
    assert interpolated.resolved
    above = fit(10000.0)
    assert above.extrapolation_decades == pytest.approx(3.0)
    assert not above.resolved


def test_large_target_uncertainty_is_unresolved() -> None:
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
    with pytest.raises(ValueError, match="nonempty"):
        replace(PROPERTY, label=" ")
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
    assert any("three or more distinct" in note for note in report.notes)
