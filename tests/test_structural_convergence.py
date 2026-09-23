"""Structural stability never replaces unobserved decay with a fitted number."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from openmmpolymer import structural_convergence as convergence
from openmmpolymer.conformation import centre_of_mass_msd
from openmmpolymer.structural_convergence import structural_window_convergence
from openmmpolymer.trajectory import AnalysisError

from .helpers import freely_rotating_chain, rod_positions, synthetic_ensemble

OPTIONS: dict[str, Any] = {
    "q_max_per_nm": 8,
    "q_bins": 8,
    "rdf_bins": 20,
    "min_frames": 3,
    "min_vectors_per_bin": 1,
    "max_distribution_frames": None,
    "max_structure_factor_frames": None,
}


def _rods(n_frames: int = 72) -> Any:
    positions = np.concatenate(
        [rod_positions(5, 0.15) + np.array([0, index * 0.8, 0]) for index in range(4)]
    )
    frames = np.repeat(positions[None, :, :], n_frames, axis=0)
    return synthetic_ensemble(frames, n_chains=4, box_nm=4)


def test_frozen_rods_keep_persistence_diffusion_and_relaxation_censored() -> None:
    report = structural_window_convergence(_rods(), range(5), **OPTIONS)
    assert not report.resolved
    for name in (
        "persistence_length_nm",
        "diffusion_coefficient_cm2_s",
        "end_to_end_relaxation_time_ps",
    ):
        parameter = report.parameters[name]
        assert parameter.values == (None,) * 4
        assert not parameter.resolved
    assert report.parameters["characteristic_ratio"].values == pytest.approx([4] * 4)
    assert not report.parameters["ratio_of_squares"].resolved
    assert not any(parameter.resolved for parameter in report.parameters.values())
    assert any(
        "frozen" in note for note in report.parameters["characteristic_ratio"].notes
    )
    assert all("bond_correlation" in window.curves for window in report.windows)
    assert all("centre_of_mass_msd" in window.curves for window in report.windows)


def test_disjoint_blocks_read_tail_frames_and_have_no_overlap() -> None:
    ensemble = _rods()
    for index in range(72):
        ensemble.universe.positions_angstrom[index, :, 2] += index
    block = convergence._block(ensemble, 24, 36)
    frames = list(block.frames())
    assert len(frames) == 12
    assert [frame.index for frame in frames] == list(range(24, 36))
    assert frames[0].positions_nm[0, 2] == pytest.approx(2.4)
    assert frames[-1].positions_nm[0, 2] == pytest.approx(3.5)
    report = structural_window_convergence(ensemble, range(5), **OPTIONS)
    assert [(block.first_frame, block.n_frames) for block in report.blocks] == [
        (36, 12),
        (48, 12),
        (60, 12),
    ]


def test_one_snapshot_cannot_resolve_any_window_stability() -> None:
    report = structural_window_convergence(_rods(1), range(5), **OPTIONS)
    assert not any(parameter.resolved for parameter in report.parameters.values())
    assert any("snapshot" in note for note in report.notes)
    assert all(window.n_frames == 1 for window in report.windows)


def test_absent_backbone_keeps_chain_measures_missing_but_retains_pair_curves() -> None:
    report = structural_window_convergence(_rods(), **OPTIONS)
    assert report.parameters["characteristic_ratio"].values == (None,) * 4
    assert all("radial_distribution" in window.curves for window in report.windows)
    assert all("structure_factor" in window.curves for window in report.windows)


def test_shrinking_box_uses_one_legal_radius_and_identical_grids() -> None:
    ensemble = _rods()
    ensemble.universe.dimensions_angstrom[:, :3] = np.linspace(50, 40, 72)[:, None]
    report = structural_window_convergence(ensemble, range(5), **OPTIONS)
    assert report.settings["r_max_nm"] == 2
    radius = report.windows[0].curves["radial_distribution"]["r_nm"]
    q = report.windows[0].curves["structure_factor"]["q_per_nm"]
    for window in (*report.windows, *report.blocks):
        assert window.curves["radial_distribution"]["r_nm"] == radius
        assert window.curves["structure_factor"]["q_per_nm"] == q
    with pytest.raises(AnalysisError, match="smallest"):
        structural_window_convergence(ensemble, range(5), r_max_nm=2.1, **OPTIONS)


def test_distribution_caps_remain_fixed_across_prefixes_and_prevent_false_resolution() -> (
    None
):
    report = structural_window_convergence(
        _rods(144),
        range(5),
        **{
            **OPTIONS,
            "min_frames": 20,
            "max_distribution_frames": 12,
            "max_structure_factor_frames": 6,
        },
    )
    assert report.settings["pair_stride"] == 12
    assert report.settings["factor_stride"] == 24
    assert [window.sample_counts["rdf_first_peak_nm"] for window in report.windows] == [
        3,
        6,
        9,
        12,
    ]
    assert not report.parameters["rdf_first_peak_nm"].resolved
    assert not report.parameters["structure_factor_peak_per_nm"].resolved
    assert any(
        "sampled frames" in note
        for note in report.parameters["rdf_first_peak_nm"].notes
    )


def test_tail_blocks_reject_a_change_hidden_by_long_prefix_averages() -> None:
    ensemble = _rods(120)
    # Only the final one-sixth bends. Its small share of prefix means hides
    # a larger final-block difference in characteristic ratio.
    for frame in range(100, 120):
        for chain in range(4):
            ensemble.universe.positions_angstrom[frame, chain * 5 + 4, 0] -= 1
            ensemble.universe.positions_angstrom[frame, chain * 5 + 4, 2] += 1
    report = structural_window_convergence(
        ensemble, range(5), relative_tolerance=0.1, **OPTIONS
    )
    parameter = report.parameters["characteristic_ratio"]
    assert parameter.relative_change is not None and parameter.relative_change < 0.1
    assert (
        parameter.block_relative_change is not None
        and parameter.block_relative_change > 0.1
    )
    assert not parameter.resolved


def test_ballistic_motion_is_not_a_diffusion_coefficient() -> None:
    ensemble = _rods()
    velocities = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])
    for frame in range(72):
        ensemble.universe.positions_angstrom[frame] += (
            np.repeat(velocities, 5, axis=0) * frame * 0.01
        )
    report = structural_window_convergence(ensemble, range(5), **OPTIONS)
    assert all(
        value is None
        for value in report.parameters["diffusion_coefficient_cm2_s"].values
    )
    assert all(
        any("does not establish diffusion" in note for note in window.notes)
        for window in report.windows
    )


def test_random_rotations_have_an_observed_relaxation_time() -> None:
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(96, 20, 3))
    directions /= np.linalg.norm(directions, axis=-1)[..., None]
    origins = rng.uniform(0.5, 2.5, size=(20, 3))
    frames = np.empty((96, 40, 3))
    frames[:, 0::2] = origins
    frames[:, 1::2] = origins + 0.15 * directions
    report = structural_window_convergence(
        synthetic_ensemble(frames, n_chains=20, box_nm=4), (0, 1), **OPTIONS
    )
    parameter = report.parameters["end_to_end_relaxation_time_ps"]
    assert all(value is not None and 0 < value < 1 for value in parameter.values)
    assert parameter.resolved
    assert parameter.block_values == ()
    assert all("not standard errors" in note for note in parameter.notes)


def test_persistence_value_requires_observed_backbone_decay() -> None:
    positions = np.array(
        [
            freely_rotating_chain(20, 0.153, 0.5, n_chains=30, seed=index)
            for index in range(72)
        ]
    )
    ensemble = synthetic_ensemble(positions, n_chains=30, box_nm=4)
    report = structural_window_convergence(
        ensemble,
        range(21),
        **{**OPTIONS, "max_distribution_frames": 4, "max_structure_factor_frames": 4},
    )
    parameter = report.parameters["persistence_length_nm"]
    assert all(value is not None and 0.3 < value < 0.7 for value in parameter.values)
    assert parameter.resolved


def test_prefix_msd_matches_refitting_that_exact_recorded_prefix() -> None:
    rng = np.random.default_rng(91)
    n_frames, n_chains = 96, 100
    frames = np.cumsum(rng.normal(scale=0.02, size=(n_frames, n_chains, 3)), axis=0)
    ensemble = synthetic_ensemble(frames, n_chains=n_chains, box_nm=4)
    report = structural_window_convergence(
        ensemble,
        **{**OPTIONS, "max_distribution_frames": 4, "max_structure_factor_frames": 4},
    )
    direct = centre_of_mass_msd(replace(ensemble, n_frames=48))
    second = report.windows[1]
    assert second.curves["centre_of_mass_msd"]["msd_nm2"] == pytest.approx(
        direct.msd_nm2
    )
    assert (
        second.values["diffusion_coefficient_cm2_s"]
        == direct.diffusion_coefficient_cm2_s
    )


@pytest.mark.parametrize(
    "fractions",
    [
        (0.5, 1),
        (0, 0.5, 1),
        (0.5, 0.25, 1),
        (0.25, 0.5, 0.9),
        (0.25, 0.5, float("nan")),
    ],
)
def test_invalid_prefix_requests_are_rejected(fractions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="window_fractions"):
        structural_window_convergence(
            _rods(), range(5), window_fractions=fractions, **OPTIONS
        )


def test_undersampled_reciprocal_bins_remain_unavailable_in_all_prefixes() -> None:
    report = structural_window_convergence(
        _rods(), range(5), **{**OPTIONS, "min_vectors_per_bin": 100000}
    )
    assert report.parameters["structure_factor_peak_per_nm"].values == (None,) * 4
    assert report.parameters["structure_factor_peak_height"].values == (None,) * 4
    assert all(not selected for selected in report.settings["common_q_bins"])


def test_uniform_translation_of_frozen_coordinates_is_not_configurational_sampling() -> (
    None
):
    ensemble = _rods()
    ensemble.universe.positions_angstrom += np.arange(72)[:, None, None]
    report = structural_window_convergence(ensemble, range(5), **OPTIONS)
    assert not any(window.configuration_varied for window in report.windows)
    assert not any(parameter.resolved for parameter in report.parameters.values())


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_recorded_frame_interval_is_rejected(interval: float) -> None:
    with pytest.raises(AnalysisError, match="interval"):
        structural_window_convergence(
            replace(_rods(), interval_ps=interval), range(5), **OPTIONS
        )
