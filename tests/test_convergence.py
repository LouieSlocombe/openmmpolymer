"""Windows require independent sampling and observed relaxation tails."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from openmmpolymer.convergence import (
    relaxation_window_convergence,
    time_window_convergence,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import planted_relaxation as planted
from .helpers import stationary_trace as stationary


def test_stationary_series_resolves_with_independent_blocks_and_autocorrelation() -> (
    None
):
    data = stationary()
    result = time_window_convergence(
        np.arange(data.size), data, property_name="radius", value_unit="nm"
    )
    assert result.resolved
    assert len(result.windows) == 4
    assert result.windows[-1].mean == pytest.approx(np.mean(data[600:]))
    assert result.windows[-1].n_samples == 5400
    assert 0.0 < result.windows[-1].n_effective <= 5400
    assert result.windows[-1].standard_error > 0.0
    assert np.all(result.block_effective_samples >= 20.0)
    assert result.relative_change < 0.1 and result.relative_block_spread < 0.1
    np.testing.assert_allclose(
        result.block_means, [np.mean(chunk) for chunk in np.array_split(data[3300:], 3)]
    )


def test_errors_and_effective_samples_are_invariant_under_unit_changes() -> None:
    data = stationary()
    a = time_window_convergence(
        np.arange(data.size), data, property_name="x", value_unit="u"
    )
    b = time_window_convergence(
        np.arange(data.size), data * 1e-20, property_name="x", value_unit="v"
    )
    assert b.resolved
    assert b.relative_change == pytest.approx(a.relative_change)
    for first, second in zip(a.windows, b.windows, strict=True):
        assert first.n_effective == pytest.approx(second.n_effective)
        assert first.standard_error * 1e-20 == pytest.approx(
            second.standard_error, rel=1e-8, abs=0.0
        )


def test_slow_correlations_do_not_count_every_frame_as_independent() -> None:
    # Dense time sampling of only one smooth oscillation is very little
    # independent information, despite six thousand observations.
    data = 10.0 + 0.1 * np.sin(np.linspace(0.0, 2 * np.pi, 6000))
    result = time_window_convergence(
        np.arange(data.size), data, property_name="x", value_unit="u"
    )
    assert not result.resolved
    assert result.windows[-1].n_effective < 20.0
    assert any("sampling" in note for note in result.notes)


def test_drift_is_not_hidden_by_overlapping_prefix_means() -> None:
    data = stationary() + np.linspace(0.0, 20.0, 6000)
    result = time_window_convergence(
        np.arange(data.size), data, property_name="x", value_unit="u"
    )
    assert not result.resolved
    assert result.relative_block_spread > 0.1
    assert any("tail-block means" in note for note in result.notes)


def test_small_prefix_change_still_requires_disjoint_tail_stability() -> None:
    data = stationary()
    data[3300:4200] -= 2.0
    data[5100:] += 2.0
    result = time_window_convergence(
        np.arange(data.size), data, property_name="x", value_unit="u"
    )
    assert result.relative_change < 0.1
    assert result.relative_block_spread > 0.1
    assert not result.resolved


def test_energy_zero_cannot_hide_statistically_significant_block_drift() -> None:
    data = np.random.default_rng(7).normal(0.0, 1.0, 6000)
    data[3300:4200] -= 0.5
    data[5100:] += 0.5
    for offset in (0.0, 1000.0):
        result = time_window_convergence(
            np.arange(data.size),
            data + offset,
            property_name="energy",
            value_unit="kJ/mol",
        )
        assert not result.resolved
        assert any("3 combined standard errors" in note for note in result.notes)


def test_nearly_identical_prefixes_are_not_three_measured_windows() -> None:
    data = stationary()
    result = time_window_convergence(
        np.arange(data.size),
        data,
        property_name="x",
        value_unit="u",
        window_fractions=(0.999999, 0.9999999, 1.0),
    )
    assert not result.resolved
    assert any("distinct sampled endpoints" in note for note in result.notes)


@pytest.mark.parametrize("size", [1, 2, 6000])
def test_snapshot_and_constant_trace_never_fabricate_sampling_evidence(
    size: int,
) -> None:
    result = time_window_convergence(
        np.arange(size), np.full(size, 10.0), property_name="x", value_unit="u"
    )
    assert not result.resolved
    assert all(math.isnan(window.standard_error) for window in result.windows)
    assert all(window.n_effective == 0.0 for window in result.windows)
    assert any("no thermal sampling evidence" in note for note in result.notes)


def test_irregular_sampling_is_explicitly_unresolved() -> None:
    data = stationary()
    times = np.arange(data.size, dtype=np.float64)
    times[3000:] += 1.0
    result = time_window_convergence(times, data, property_name="x", value_unit="u")
    assert not result.resolved
    assert any("Uneven time spacing" in note for note in result.notes)


@pytest.mark.parametrize(
    "fractions",
    [
        (),
        (0.5, 1.0),
        (0.25, 0.75, 0.5, 1.0),
        (0.0, 0.5, 1.0),
        (0.25, 0.5, 0.9),
        (0.25, math.nan, 1.0),
    ],
)
def test_invalid_windows_are_rejected(fractions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="window_fractions"):
        time_window_convergence(
            [0, 1],
            [1, 2],
            property_name="x",
            value_unit="u",
            window_fractions=fractions,
        )
    with pytest.raises(ValueError, match="window_fractions"):
        relaxation_window_convergence(
            planted(np.asarray([1.0, 2.0]), np.asarray([10.0, 5.0])),
            window_fractions=fractions,
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("relative_tolerance", 0.0),
        ("relative_tolerance", math.nan),
        ("min_effective_samples", 0.0),
        ("discard_fraction", 1.0),
        ("discard_fraction", -0.1),
    ],
)
def test_invalid_numeric_options_raise(name: str, value: float) -> None:
    options: dict[str, Any] = {name: value}
    with pytest.raises(ValueError, match=name):
        time_window_convergence(
            [0, 1], [1, 2], property_name="x", value_unit="u", **options
        )


@pytest.mark.parametrize(
    "times,values", [([], []), ([0, 1], [1]), ([0, 0], [1, 2]), ([0, 1], [1, math.nan])]
)
def test_invalid_series_cannot_be_filtered_to_appear_converged(
    times: list[float], values: list[float]
) -> None:
    with pytest.raises(AnalysisError):
        time_window_convergence(times, values, property_name="x", value_unit="u")


def test_observed_exponential_decay_stabilises_time_and_viscosity() -> None:
    time = np.geomspace(0.1, 10000.0, 140)
    result = relaxation_window_convergence(planted(time, 1000.0 * np.exp(-time / 50.0)))
    assert result.resolved
    assert result.metrics["kww_mean_tau_ps"].values[-1] == pytest.approx(50.0)
    assert result.metrics["kww_viscosity_pa_s"].values[-1] == pytest.approx(0.05)
    assert result.metrics["prony_viscosity_pa_s"].values[-1] == pytest.approx(
        0.05, rel=0.04
    )
    assert all(window.tail_decayed for window in result.windows)
    assert any("One relaxation replica" in note for note in result.notes)


def test_stable_extrapolated_tau_is_unresolved_without_observed_decay() -> None:
    time = np.geomspace(0.1, 10000.0, 140)
    result = relaxation_window_convergence(
        planted(time, 1000.0 * np.exp(-time / 10000.0))
    )
    tau = result.metrics["kww_mean_tau_ps"]
    assert tau.relative_change < 0.001
    assert not tau.resolved
    assert not result.resolved
    assert all(not window.tail_decayed for window in result.windows)


def test_solid_plateau_cannot_report_finite_liquid_viscosity() -> None:
    time = np.geomspace(0.1, 10000.0, 140)
    result = relaxation_window_convergence(
        planted(time, 50.0 + 1000.0 * np.exp(-time / 50.0))
    )
    assert result.metrics["equilibrium_modulus_mpa"].resolved
    assert result.metrics["equilibrium_modulus_mpa"].values[-1] == pytest.approx(
        50.0, rel=0.05
    )
    assert not result.metrics["kww_viscosity_pa_s"].resolved
    assert not result.metrics["prony_viscosity_pa_s"].resolved


def test_relaxation_duplicate_prefix_endpoints_cannot_resolve() -> None:
    time = np.geomspace(0.1, 10000.0, 140)
    result = relaxation_window_convergence(
        planted(time, 1000.0 * np.exp(-time / 50.0)),
        window_fractions=(0.999999, 0.9999999, 1.0),
    )
    assert not result.resolved
    assert all(not metric.resolved for metric in result.metrics.values())
    assert any(
        "distinct sampled endpoints" in note
        for note in result.metrics["kww_mean_tau_ps"].notes
    )


def test_prony_parameter_stability_cannot_hide_a_poor_response_fit() -> None:
    time = np.geomspace(0.1, 10000.0, 140)
    values = 1000.0 * np.exp(-time / 50.0) * (1.0 + 0.8 * np.sin(8.0 * np.log(time)))
    result = relaxation_window_convergence(planted(time, values))
    assert not result.metrics["prony_viscosity_pa_s"].resolved
    assert not result.metrics["equilibrium_modulus_mpa"].resolved
    assert any(
        "Prony residual" in note for window in result.windows for note in window.notes
    )


def test_insufficient_relaxation_bins_retains_unresolved_refits() -> None:
    curve = planted(np.asarray([1.0, 2.0]), np.asarray([10.0, 5.0]))
    result = relaxation_window_convergence(curve)
    assert len(result.windows) == 4
    assert not result.resolved
    assert all(not parameter.resolved for parameter in result.metrics.values())


def test_precise_kww_decay_does_not_inherit_discrete_prony_fit_failure() -> None:
    time = np.geomspace(0.1, 10000.0, 140)
    result = relaxation_window_convergence(
        planted(
            time, 1000.0 * np.exp(-time / 50.0), error_mpa=np.full(time.size, 0.001)
        )
    )
    assert result.metrics["kww_mean_tau_ps"].resolved
    assert result.metrics["kww_viscosity_pa_s"].resolved
    assert not result.metrics["prony_viscosity_pa_s"].resolved


def test_invalid_relaxation_error_cannot_be_ignored() -> None:
    curve = planted(np.asarray([1.0, 2.0]), np.asarray([10.0, 5.0]))
    with pytest.raises(AnalysisError, match="standard errors"):
        relaxation_window_convergence(
            replace(curve, standard_error_mpa=np.asarray([1.0, math.nan]))
        )
