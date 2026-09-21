"""Tests for the relaxation modulus and the two fits read off it.

Split the way the mechanical tests are split. Everything that decides
something - how chunks merge, what the baseline does, which points a fit rests
on, when a fit refuses - is tested against manifests written by hand, because
those are arithmetic over recorded numbers and running dynamics to reach them
would hide what is being checked. The planted answers are exact, so the
assertions are equalities.

The two fits get the same treatment and one thing more: both are hand-rolled
numerics standing in for a library this package does not depend on, so they are
checked against planted parameters they must recover to machine precision, and
against the degenerate inputs - no signal, no decay, a decay longer than the
run - where the honest answer is to refuse.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from openmmpolymer.elasticity import MPA_PER_BAR
from openmmpolymer.relaxation import (
    KWW_BETA_BRACKET,
    MAX_EDGE_WEIGHT,
    RelaxationCurve,
    _merge_bins,
    _prony_design,
    _signal_window,
    _standard_error,
    fit_kww,
    fit_prony,
    mean_curve,
    nnls,
    prony_times_ps,
    relax_stages,
    relaxation_curve,
)
from openmmpolymer.stress import deviatoric_strain
from openmmpolymer.trajectory import AnalysisError

from .helpers import write_relaxation


def planted(
    time_ps: np.ndarray,
    modulus_mpa: np.ndarray,
    *,
    error_mpa: np.ndarray | None = None,
    floor: float = 0.0,
    mode: str = "shear",
    poisson: float = 0.5,
) -> RelaxationCurve:
    """A curve built straight from arrays, for testing a fit on its own."""
    return RelaxationCurve(
        stage="planted",
        mode=mode,
        bin_index=np.arange(time_ps.size),
        time_ps=time_ps,
        modulus_mpa=modulus_mpa,
        standard_error_mpa=(np.zeros(time_ps.size) if error_mpa is None else error_mpa),
        n_samples=np.full(time_ps.size, 100.0),
        step_strain=0.03,
        strain_measure=0.03,
        temperature_k=298.15,
        poisson=poisson,
        baseline_mpa=0.0,
        noise_floor_mpa=floor,
    )


# --------------------------------------------------------------------------
# Reading a run directory
# --------------------------------------------------------------------------


def test_a_relaxation_is_found_by_what_it_recorded(tmp_path: Path) -> None:
    """Not by its name, so a hand-made stage is read like any other."""
    write_relaxation(tmp_path, stem="something_else")
    assert relax_stages(tmp_path) == ("something_else_00",)


def test_a_directory_with_no_relaxation_says_so(tmp_path: Path) -> None:
    """The refusal names what it did find, which is what a caller needs."""
    (tmp_path / "manifest.json").write_text(
        '{"protocol": "x", "seed": 1, "stages": {"05_npt": {"samples": {}}}}'
    )
    with pytest.raises(AnalysisError, match="05_npt"):
        relax_stages(tmp_path)


def test_the_planted_decay_comes_back_exactly(tmp_path: Path) -> None:
    """The reader's scaling is the inverse of the writer's, to the last bit."""
    write_relaxation(tmp_path, modulus_mpa=800.0, tau_ps=50.0, beta=0.6)
    curve = relaxation_curve(tmp_path)
    expected = 800.0 * np.exp(-((curve.time_ps / 50.0) ** 0.6))
    assert np.allclose(curve.modulus_mpa, expected, rtol=0.0, atol=1e-9)


def test_chunks_of_one_relaxation_merge_into_the_curve_one_run_would_give(
    tmp_path: Path,
) -> None:
    """The reason the bin edges come from the settings rather than the data.

    A relaxation split for resume must read back as one curve, not as several
    short ones and not as one with every shared bin in it twice.
    """
    whole = relaxation_curve(write_relaxation(tmp_path / "a").parent)
    split = relaxation_curve(write_relaxation(tmp_path / "b", chunks=4).parent)
    assert split.n_points == whole.n_points
    assert np.array_equal(split.bin_index, whole.bin_index)
    assert np.allclose(split.modulus_mpa, whole.modulus_mpa, rtol=0.0, atol=1e-9)


def test_a_bin_straddling_a_chunk_boundary_merges_exactly() -> None:
    """The claim the whole recording format rests on.

    A chunk boundary falls wherever ``stage_ps`` puts it, which is generally
    inside a bin rather than between two - so that bin is written twice, by
    two stages, with different counts. The count and the mean square recorded
    beside each mean are what make adding them give back exactly the numbers
    one unbroken run would have written; a stored standard error could not,
    and an unweighted average of the two means would be wrong whenever the
    counts differ.
    """
    early = np.asarray([10.0, 12.0, 14.0])
    late = np.asarray([20.0, 22.0, 24.0, 26.0, 28.0])
    early_t = np.asarray([1.0, 1.1, 1.2])
    late_t = np.asarray([1.3, 1.4, 1.5, 1.6, 1.7])
    merged = _merge_bins(
        {
            "segment_bin": [7.0, 7.0],
            "segment_samples": [float(early.size), float(late.size)],
            "segment_relax_time_ps": [float(early_t.mean()), float(late_t.mean())],
            "segment_stress_bar": [float(early.mean()), float(late.mean())],
            "segment_stress_sq_bar2": [
                float((early**2).mean()),
                float((late**2).mean()),
            ],
        }
    )
    whole = np.concatenate([early, late])
    assert merged["n"][0] == whole.size
    assert merged["mean"][0] == pytest.approx(whole.mean(), abs=1e-12)
    assert merged["mean_sq"][0] == pytest.approx((whole**2).mean(), abs=1e-12)
    assert merged["time_ps"][0] == pytest.approx(
        np.concatenate([early_t, late_t]).mean(), abs=1e-12
    )
    # And the error that falls out of them is the one the whole sample gives.
    error = _standard_error(merged["mean"], merged["mean_sq"], merged["n"])
    assert error[0] == pytest.approx(whole.std(ddof=0) / np.sqrt(whole.size))
    # An unweighted average of the two means would have given 19.0, not 19.5.
    assert merged["mean"][0] != pytest.approx(
        0.5 * (early.mean() + late.mean()), abs=1e-6
    )


def test_the_baseline_is_subtracted_once_and_only_once(tmp_path: Path) -> None:
    """Recorded by the first chunk alone, so a merge must not apply it twice."""
    write_relaxation(tmp_path, modulus_mpa=500.0, baseline_bar=250.0, chunks=3)
    curve = relaxation_curve(tmp_path)
    expected = 500.0 * np.exp(-((curve.time_ps / 100.0) ** 0.5))
    assert np.allclose(curve.modulus_mpa, expected, rtol=0.0, atol=1e-9)
    measure = 2.0 * deviatoric_strain(0.03, 0.5)
    assert curve.baseline_mpa == pytest.approx(250.0 * MPA_PER_BAR / measure)


def test_two_different_strains_are_not_read_as_one_curve(tmp_path: Path) -> None:
    """They measure different things, and averaging them would hide that."""
    first = write_relaxation(tmp_path, step_strain=0.02, stem="06_relax_r0")
    write_relaxation(tmp_path, step_strain=0.05, stem="06_relax_r0", chunks=1, merge={})
    del first
    stages = relax_stages(tmp_path)
    write_relaxation(tmp_path, step_strain=0.02, stem="07_other")
    with pytest.raises(AnalysisError, match="strained by different amounts"):
        relaxation_curve(tmp_path, (*stages, "07_other_00"))


def test_a_shear_step_is_told_from_a_tensile_one_by_what_it_recorded(
    tmp_path: Path,
) -> None:
    """A shear step imposes no lateral contraction, so it records no ratio."""
    write_relaxation(tmp_path / "s", mode="shear")
    write_relaxation(tmp_path / "t", mode="tensile")
    assert relaxation_curve(tmp_path / "s").mode == "shear"
    assert relaxation_curve(tmp_path / "t").mode == "tensile"


def test_youngs_modulus_is_derived_and_shear_is_measured(tmp_path: Path) -> None:
    """E = 2(1+nu)G, with the ratio passed in rather than assumed."""
    write_relaxation(tmp_path, mode="tensile", poisson=0.5)
    curve = relaxation_curve(tmp_path)
    assert np.allclose(curve.youngs_modulus_mpa(), 3.0 * curve.modulus_mpa)
    assert np.allclose(curve.youngs_modulus_mpa(0.35), 2.7 * curve.modulus_mpa)


def test_a_shear_curve_refuses_to_invent_a_poisson_ratio(tmp_path: Path) -> None:
    """It measured G directly and has nothing to convert with."""
    write_relaxation(tmp_path, mode="shear")
    with pytest.raises(AnalysisError, match="Poisson"):
        relaxation_curve(tmp_path).youngs_modulus_mpa()


def test_replicas_average_and_carry_their_own_spread(tmp_path: Path) -> None:
    """The error bar changes meaning once there is more than one run."""
    curves = [
        relaxation_curve(write_relaxation(tmp_path / f"r{i}", modulus_mpa=m).parent)
        for i, m in enumerate((900.0, 1000.0, 1100.0))
    ]
    mean = mean_curve(curves)
    assert mean.n_replicas == 3
    assert mean.modulus_mpa[0] == pytest.approx(curves[1].modulus_mpa[0])
    assert np.all(mean.standard_error_mpa > 0.0)


def test_replicas_on_different_grids_are_refused(tmp_path: Path) -> None:
    """Averaging them bin for bin would line up times that are not the same."""
    first = relaxation_curve(write_relaxation(tmp_path / "a").parent)
    second = relaxation_curve(
        write_relaxation(tmp_path / "b", first_ps=50.0, total_ps=80.0).parent
    )
    object.__setattr__(second, "bin_index", second.bin_index + 10_000)
    with pytest.raises(AnalysisError, match="share no bin"):
        mean_curve([first, second])


# --------------------------------------------------------------------------
# The fitting window
# --------------------------------------------------------------------------


def test_the_window_is_truncated_at_the_end_not_filtered_point_by_point() -> None:
    """Dropping the negative points would bias the tail up.

    They are the downward half of the noise, and the upward half would stay -
    so the fit would describe the asymmetry of the cut rather than the decay.
    Cutting the window instead keeps whatever is inside it unbiased.
    """
    modulus = np.asarray([10.0, 8.0, 6.0, -0.5, 4.0, -0.2, 0.1])
    error = np.full(modulus.size, 1.0)
    window = _signal_window(modulus, error, 3.0)
    # The last bin standing 3 sigma clear is index 4, so 5 and 6 go whatever
    # their sign, and the negative at 3 goes because a log cannot take it.
    assert window.tolist() == [True, True, True, False, True, False, False]


def test_a_curve_with_no_signal_leaves_an_empty_window() -> None:
    """And every fit then refuses rather than describing the noise."""
    modulus = np.asarray([0.5, -0.3, 0.2, -0.1])
    window = _signal_window(modulus, np.full(4, 10.0), 3.0)
    assert not window.any()


# --------------------------------------------------------------------------
# The stretched exponential
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("modulus_mpa", "tau_ps", "beta"),
    [(900.0, 250.0, 0.42), (1500.0, 30.0, 0.75), (500.0, 1000.0, 1.0)],
)
def test_the_kww_recovers_planted_parameters(
    modulus_mpa: float, tau_ps: float, beta: float
) -> None:
    """To machine precision, which is what makes the separation worth doing.

    For a fixed exponent the relation is linear in ``t**beta``, so this is a
    one-dimensional search with an exact solve inside it rather than a
    nonlinear fit that might not converge.
    """
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    fit = fit_kww(planted(time_ps, modulus_mpa * np.exp(-((time_ps / tau_ps) ** beta))))
    assert fit.beta == pytest.approx(beta, abs=1e-5)
    assert fit.tau_ps == pytest.approx(tau_ps, rel=1e-5)
    assert fit.modulus_mpa == pytest.approx(modulus_mpa, rel=1e-5)
    assert fit.resolved


def test_a_single_exponential_is_a_legitimate_answer() -> None:
    """beta = 1 sits on the bracket, and unlike the floor it is not a failure."""
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    fit = fit_kww(planted(time_ps, 500.0 * np.exp(-time_ps / 300.0)))
    assert fit.at_bound == "upper"
    assert fit.beta == pytest.approx(1.0)
    assert fit.resolved


def test_the_mean_relaxation_time_is_the_integral_of_the_decay() -> None:
    """(tau/beta) Gamma(1/beta), which for a single exponential is tau."""
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    single = fit_kww(planted(time_ps, 500.0 * np.exp(-time_ps / 300.0)))
    assert single.mean_tau_ps == pytest.approx(300.0, rel=1e-4)
    stretched = fit_kww(planted(time_ps, 900.0 * np.exp(-((time_ps / 250.0) ** 0.42))))
    assert stretched.mean_tau_ps > stretched.tau_ps


def test_a_curve_that_is_not_decaying_does_not_resolve() -> None:
    """There is no time constant to report, and none is reported."""
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    fit = fit_kww(planted(time_ps, 100.0 + 0.01 * time_ps))
    assert not fit.resolved
    assert math.isnan(fit.tau_ps)


def test_too_few_points_refuses_rather_than_fitting_them() -> None:
    """Three bins is not a curve however well a line goes through them."""
    time_ps = np.geomspace(1.0, 100.0, 3)
    fit = fit_kww(planted(time_ps, 100.0 * np.exp(-time_ps / 20.0)))
    assert not fit.resolved
    assert fit.n_points == 3


def test_the_exponent_stays_inside_its_bracket() -> None:
    """Below the floor a stretched exponential is a power law, and gamma blows up."""
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    fit = fit_kww(planted(time_ps, 900.0 * np.exp(-((time_ps / 250.0) ** 0.02))))
    assert KWW_BETA_BRACKET[0] <= fit.beta <= KWW_BETA_BRACKET[1]
    assert fit.at_bound == "lower"
    assert not fit.resolved


# --------------------------------------------------------------------------
# Non-negative least squares
# --------------------------------------------------------------------------


def test_nnls_recovers_an_exact_non_negative_solution() -> None:
    """The planted spectrum comes back, zeros and all."""
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    tau_ps = prony_times_ps(0.1, 1.0e4)
    design = _prony_design(time_ps, tau_ps)
    planted_weights = np.zeros(tau_ps.size + 1)
    planted_weights[[0, 2, 4]] = [300.0, 150.0, 60.0]
    planted_weights[-1] = 40.0
    found = nnls(design, design @ planted_weights)
    assert np.allclose(found, planted_weights, rtol=0.0, atol=1e-8)


def test_nnls_never_returns_a_negative_weight() -> None:
    """Which plain least squares does most of the time on realistic data.

    A Prony series with a negative weight is a relaxation spectrum with
    negative weight somewhere, which no material has. Measured over two
    hundred draws at each of four noise levels, plain least squares produced
    one in about four cases out of five - so "fit it unconstrained and refuse
    if any came back negative" would throw away four runs in five, and not
    the four that deserved it. The bound below is loose because the point is
    the order of magnitude, not the exact rate.
    """
    rng = np.random.default_rng(1)
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    tau_ps = prony_times_ps(0.1, 1.0e4)
    design = _prony_design(time_ps, tau_ps)
    planted_weights = np.zeros(tau_ps.size + 1)
    planted_weights[[0, 2, 4]] = [300.0, 150.0, 60.0]
    planted_weights[-1] = 40.0
    unconstrained_negatives = 0
    for _ in range(20):
        target = design @ planted_weights + rng.normal(0.0, 30.0, time_ps.size)
        found = nnls(design, target)
        assert bool((found >= 0.0).all())
        plain, *_ = np.linalg.lstsq(design, target, rcond=None)
        unconstrained_negatives += int((plain < 0.0).any())
    assert unconstrained_negatives >= 12


def test_nnls_satisfies_its_own_optimality_conditions() -> None:
    """Zero weights want to go down, non-zero ones are stationary."""
    rng = np.random.default_rng(4)
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    design = _prony_design(time_ps, prony_times_ps(0.1, 1.0e4))
    target = 500.0 * np.exp(-time_ps / 200.0) + rng.normal(0.0, 5.0, time_ps.size)
    found = nnls(design, target)
    gradient = design.T @ (target - design @ found)
    assert bool((gradient[found <= 1e-10] <= 1e-6).all())
    assert bool((np.abs(gradient[found > 1e-10]) <= 1e-6).all())


def test_nnls_returns_zeros_when_there_is_nothing_to_fit() -> None:
    """It terminates and says so, rather than failing to converge."""
    time_ps = np.geomspace(0.1, 1.0e4, 60)
    design = _prony_design(time_ps, prony_times_ps(0.1, 1.0e4))
    assert bool((nnls(design, -np.abs(np.ones(time_ps.size))) == 0.0).all())


# --------------------------------------------------------------------------
# The Prony series
# --------------------------------------------------------------------------


def test_the_prony_fit_recovers_a_planted_spectrum() -> None:
    """Equilibrium modulus included, and the unused terms stay at zero."""
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    tau_ps = prony_times_ps(float(time_ps[0]), float(time_ps[-1]))
    weights = np.zeros(tau_ps.size + 1)
    weights[[0, 2, 4]] = [300.0, 150.0, 60.0]
    weights[-1] = 40.0
    curve = planted(time_ps, _prony_design(time_ps, tau_ps) @ weights)
    fit = fit_prony(curve)
    assert fit.equilibrium_mpa == pytest.approx(40.0, abs=1e-6)
    assert np.allclose(fit.weights_mpa, weights[:-1], rtol=0.0, atol=1e-6)
    assert fit.n_active == 3
    assert fit.resolved


def test_the_grid_stops_short_of_the_run_so_g_inf_stays_separable() -> None:
    """An exponential as slow as the run is nearly a constant, and the split
    between the two would be arbitrary."""
    tau_ps = prony_times_ps(0.1, 1.0e4)
    assert tau_ps[-1] == pytest.approx(1.0e4 / 3.0)
    assert tau_ps[0] == pytest.approx(0.1)


def test_a_decay_slower_than_the_run_does_not_claim_a_plateau() -> None:
    """The slowest term takes the weight, which is the fit saying so."""
    time_ps = np.geomspace(0.1, 1.0e4, 120)
    fit = fit_prony(planted(time_ps, 5000.0 * np.exp(-time_ps / 3.0e5) + 10.0))
    assert fit.edge_weight > MAX_EDGE_WEIGHT
    assert not fit.plateau_reached
    assert not fit.resolved
