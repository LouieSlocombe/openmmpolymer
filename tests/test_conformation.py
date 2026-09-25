"""Tests for chain dimensions, and for whether the chains were still moving."""

from __future__ import annotations

import math
import weakref
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from openmmpolymer import conformation, protocols
from openmmpolymer.conformation import (
    DIFFUSIVE_SLOPE_RANGE,
    NM2_PS_TO_CM2_S,
    centre_of_mass_msd,
    chain_conformation,
    end_to_end_relaxation,
    persistence_length,
)
from openmmpolymer.protocols import chain_dimensions
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    dimer_cell,
    freely_rotating_chain,
    random_walk_frames,
    rod_positions,
    rotating_dimer,
    synthetic_ensemble,
)


def test_a_rigid_rod_gives_back_its_closed_form_radius_of_gyration() -> None:
    """``Rg = l sqrt((n^2-1)/12)`` for equal masses on a line, so the answer is
    known without simulating anything."""
    n_beads, spacing = 10, 0.15
    ensemble = synthetic_ensemble(rod_positions(n_beads, spacing), n_chains=1)
    measured = chain_conformation(ensemble, range(n_beads)).mean
    exact = spacing * math.sqrt((n_beads**2 - 1) / 12.0)
    assert measured.mean_radius_of_gyration_nm == pytest.approx(exact, rel=1e-12)
    assert measured.mean_squared_end_to_end_nm2 == pytest.approx(
        ((n_beads - 1) * spacing) ** 2, rel=1e-12
    )


def test_a_dimer_cell_gives_its_exact_dimensions() -> None:
    """Two equal masses at separation l: Rg is l/2, R^2 is l^2, the
    characteristic ratio is one and the ratio of squares is four."""
    ensemble = synthetic_ensemble(dimer_cell(), n_chains=32)
    measured = chain_conformation(ensemble, (0, 1)).mean
    assert measured.mean_radius_of_gyration_nm == pytest.approx(0.3, rel=1e-12)
    assert measured.mean_squared_end_to_end_nm2 == pytest.approx(0.36, rel=1e-12)
    assert measured.characteristic_ratio == pytest.approx(1.0, rel=1e-12)
    assert measured.ratio_of_squares == pytest.approx(4.0, rel=1e-12)


def test_a_single_frame_agrees_with_the_existing_snapshot_measurement() -> None:
    """chain_conformation generalises chain_dimensions over frames, so on one
    frame the two have to be the same number."""
    positions = freely_rotating_chain(20, 0.153, 0.8, n_chains=5, seed=1)
    ensemble = synthetic_ensemble(positions, n_chains=5)
    over_frames = chain_conformation(
        ensemble, range(21), expected_characteristic_ratio=7.0
    ).mean
    directly = chain_dimensions(
        positions, list(range(21)), 21, 5, expected_characteristic_ratio=7.0
    )
    assert over_frames == directly


def test_trajectory_ratios_use_pooled_means_even_when_one_frame_is_closed() -> None:
    """A closed backbone still contributes its radius and bond lengths to
    the pooled denominators, despite both of its own ratios being zero."""
    frames = np.zeros((2, 3, 3), dtype=np.float64)
    frames[0, :, 0] = (0.0, 1.0, 0.0)
    frames[1, :, 0] = (0.0, 2.0, 4.0)
    series = chain_conformation(
        synthetic_ensemble(frames, n_chains=1),
        (0, 1, 2),
        expected_characteristic_ratio=16.0 / 9.0,
    )

    # R² is 0 and 16, Rg² is 2/9 and 8/3, and mean bond length is 1 and 2.
    assert series.mean_squared_end_to_end_nm2 == pytest.approx([0.0, 16.0])
    assert series.mean_radius_of_gyration_nm == pytest.approx(
        [math.sqrt(2.0) / 3.0, math.sqrt(8.0 / 3.0)]
    )
    assert series.mean.mean_squared_end_to_end_nm2 == pytest.approx(8.0)
    assert series.mean.mean_radius_of_gyration_nm == pytest.approx(
        (math.sqrt(2.0) / 3.0 + math.sqrt(8.0 / 3.0)) / 2.0
    )
    assert series.mean.ratio_of_squares == pytest.approx(72.0 / 13.0)
    assert series.mean.characteristic_ratio == pytest.approx(16.0 / 9.0)
    assert series.mean.expected_characteristic_ratio == 16.0 / 9.0
    assert series.mean.consistent


@pytest.mark.parametrize(
    ("n_frames", "n_chains", "stride"),
    [(1, 3, 1), (2, 1, 1), (7, 3, 2), (5, 2, 8)],
)
def test_pooled_dimensions_match_selected_frames_with_masses_and_a_backbone_subset(
    n_frames: int, n_chains: int, stride: int
) -> None:
    """Pooling must retain the weights of every chain and selected frame,
    while using all atoms for Rg and only the backbone for bonds and R²."""
    atoms_per_chain = 5
    generator = np.random.default_rng(31)
    positions = generator.normal(size=(n_frames, n_chains, atoms_per_chain, 3))
    positions *= np.arange(1, n_frames + 1)[:, None, None, None]
    positions *= np.arange(1, n_chains + 1)[None, :, None, None]
    positions[0, :, 4, :] = positions[0, :, 0, :]
    frames = positions.reshape(n_frames, n_chains * atoms_per_chain, 3)
    masses = np.array([12.0, 1.0, 16.0, 2.0, 14.0])
    backbone = (0, 2, 4)
    expected_ratio = 1.75
    interval = 2.5
    ensemble = synthetic_ensemble(
        frames,
        n_chains=n_chains,
        masses_amu=masses,
        interval_ps=interval,
        stage="weighted_chains",
    )
    selected = list(ensemble.frames(stride=stride))
    snapshots = [
        chain_dimensions(
            frame.positions_nm,
            backbone,
            atoms_per_chain,
            n_chains,
            masses=masses,
            expected_characteristic_ratio=expected_ratio,
        )
        for frame in selected
    ]
    pooled = chain_dimensions(
        np.concatenate([frame.positions_nm for frame in selected]),
        backbone,
        atoms_per_chain,
        n_chains * len(selected),
        masses=masses,
        expected_characteristic_ratio=expected_ratio,
    )

    series = chain_conformation(
        ensemble,
        backbone,
        stride=stride,
        expected_characteristic_ratio=expected_ratio,
    )
    assert series.stage == "weighted_chains"
    assert series.n_frames == len(selected)
    assert series.n_chains == n_chains
    assert series.time_ps == pytest.approx([frame.time_ps for frame in selected])
    assert series.mean_squared_end_to_end_nm2 == pytest.approx(
        [frame.mean_squared_end_to_end_nm2 for frame in snapshots]
    )
    assert series.mean_radius_of_gyration_nm == pytest.approx(
        [frame.mean_radius_of_gyration_nm for frame in snapshots]
    )
    assert series.mean.mean_squared_end_to_end_nm2 == pytest.approx(
        pooled.mean_squared_end_to_end_nm2
    )
    assert series.mean.mean_radius_of_gyration_nm == pytest.approx(
        pooled.mean_radius_of_gyration_nm
    )
    assert series.mean.ratio_of_squares == pytest.approx(pooled.ratio_of_squares)
    assert series.mean.characteristic_ratio == pytest.approx(
        pooled.characteristic_ratio
    )
    assert series.mean.expected_characteristic_ratio == expected_ratio
    assert series.mean.consistent == pooled.consistent
    assert (series.settled is None) == (len(selected) < 3)


def test_each_selected_frame_is_measured_once_without_retaining_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trajectory should only keep scalar statistics from prior frames,
    and should not revisit their coordinates to compute the pooled mean."""
    frames = np.repeat(dimer_cell(2)[None, :, :], 7, axis=0)
    ensemble = synthetic_ensemble(frames, n_chains=2)
    measure = protocols._chain_dimension_sums
    measured_shapes: list[tuple[int, ...]] = []
    coordinates: list[weakref.ReferenceType[npt.NDArray[np.float64]]] = []

    def measure_once(
        positions_nm: npt.NDArray[np.float64], *args: Any, **kwargs: Any
    ) -> protocols._ChainDimensionSums:
        assert all(reference() is None for reference in coordinates)
        measured_shapes.append(positions_nm.shape)
        coordinates.append(weakref.ref(positions_nm))
        return measure(positions_nm, *args, **kwargs)

    monkeypatch.setattr(protocols, "_chain_dimension_sums", measure_once)
    monkeypatch.setattr(conformation, "_chain_dimension_sums", measure_once)
    series = chain_conformation(ensemble, (0, 1), stride=3)

    assert measured_shapes == [(4, 3)] * 3
    assert series.n_frames == 3
    assert all(reference() is None for reference in coordinates)


def test_a_snapshot_has_no_settling_to_report() -> None:
    """One frame is not a series, and reporting a verdict from it would be the
    thing this whole module exists to avoid."""
    series = chain_conformation(synthetic_ensemble(dimer_cell(), n_chains=32), (0, 1))
    assert series.settled is None
    assert series.n_frames == 1


def test_a_trajectory_reports_whether_its_dimensions_had_settled() -> None:
    """The dimensions of a melt mean something only if they had stopped moving
    while they were being measured."""
    frames = np.repeat(dimer_cell()[None, :, :], 20, axis=0)
    series = chain_conformation(
        synthetic_ensemble(frames, n_chains=32, interval_ps=1.0), (0, 1)
    )
    assert series.settled is not None
    assert series.settled.equilibrated
    assert series.n_frames == 20
    assert series.time_ps.size == 20
    assert series.mean_radius_of_gyration_nm == pytest.approx(0.3, rel=1e-9)


def test_a_stride_measures_fewer_frames() -> None:
    """A long trajectory does not need every frame to give a mean."""
    frames = np.repeat(dimer_cell()[None, :, :], 20, axis=0)
    ensemble = synthetic_ensemble(frames, n_chains=32, interval_ps=1.0)
    assert chain_conformation(ensemble, (0, 1), stride=4).n_frames == 5


def test_a_backbone_that_does_not_fit_the_chains_is_refused() -> None:
    """The indices are within one chain, and whole-cell indices are the easy
    mistake - they would silently measure across two molecules."""
    ensemble = synthetic_ensemble(dimer_cell(), n_chains=32)
    with pytest.raises(AnalysisError, match="outside a chain"):
        chain_conformation(ensemble, (0, 40))


@pytest.mark.parametrize("expected_nm", [0.5, 1.0, 2.0])
def test_a_chain_of_known_stiffness_gives_back_its_persistence_length(
    expected_nm: float,
) -> None:
    """A freely-rotating chain's bond correlation is exponential by
    construction, so the length that comes out is checkable."""
    positions = freely_rotating_chain(60, 0.153, expected_nm, n_chains=300, seed=11)
    measured = persistence_length(
        synthetic_ensemble(positions, n_chains=300), range(61)
    )
    assert measured.bond_length_nm == pytest.approx(0.153, rel=1e-9)
    assert measured.n_bonds == 60
    assert measured.contour_length_nm == pytest.approx(60 * 0.153, rel=1e-9)
    assert measured.decayed
    assert measured.persistence_length_nm == pytest.approx(expected_nm, rel=0.06)


def test_the_first_correlation_is_the_bond_angle_cosine() -> None:
    """The curve's first point is a direct average with no fitting in it, so it
    pins the arithmetic independently of the fit."""
    expected_nm, bond_nm = 0.9, 0.153
    positions = freely_rotating_chain(50, bond_nm, expected_nm, n_chains=200, seed=5)
    measured = persistence_length(
        synthetic_ensemble(positions, n_chains=200), range(51)
    )
    assert measured.correlation[0] == pytest.approx(1.0, rel=1e-9)
    assert measured.correlation[1] == pytest.approx(
        math.exp(-bond_nm / expected_nm), rel=1e-6
    )


def test_a_rod_never_decorrelates_along_its_own_length() -> None:
    """Its persistence length is longer than it is, so the number is an
    extrapolation and ``decayed`` says so."""
    measured = persistence_length(
        synthetic_ensemble(rod_positions(20, 0.153), n_chains=1), range(20)
    )
    assert measured.correlation[-1] == pytest.approx(1.0, rel=1e-9)
    assert not measured.decayed


def test_a_backbone_with_too_few_bonds_has_no_curve_to_fit() -> None:
    """Three bonds is the least that gives a decay rather than two points."""
    ensemble = synthetic_ensemble(dimer_cell(), n_chains=32)
    with pytest.raises(AnalysisError, match="too few for a correlation curve"):
        persistence_length(ensemble, (0, 1))


def test_a_vector_rotating_at_a_known_rate_decorrelates_on_schedule() -> None:
    """``cos(theta) = 1/e`` at a known angle, so at a fixed rotation rate the
    relaxation time is a known number of picoseconds."""
    rate, interval = 0.02, 0.5
    measured = end_to_end_relaxation(
        rotating_dimer(300, rate, interval_ps=interval), (0, 1)
    )
    assert measured.decorrelated
    assert measured.relaxation_time_ps == pytest.approx(
        math.acos(1.0 / math.e) / rate * interval, rel=1e-3
    )
    assert measured.correlation[0] == pytest.approx(1.0, rel=1e-9)


def test_a_vector_that_never_turns_reports_no_relaxation_time() -> None:
    """The honest answer for a run shorter than the chains' own relaxation,
    which is every protocol this package ships."""
    measured = end_to_end_relaxation(rotating_dimer(50, 0.0, interval_ps=1.0), (0, 1))
    assert not measured.decorrelated
    assert measured.relaxation_time_ps is None
    assert measured.trajectory_ps == pytest.approx(49.0)


def test_chains_of_zero_length_have_nothing_to_correlate() -> None:
    """A backbone path naming the same atom twice would divide by zero."""
    frames = np.zeros((10, 4, 3), dtype=np.float64)
    ensemble = synthetic_ensemble(frames, n_chains=2, interval_ps=1.0)
    with pytest.raises(AnalysisError, match="zero length"):
        end_to_end_relaxation(ensemble, (0, 1))


def test_uniform_drift_is_ballistic_and_earns_no_diffusion_coefficient() -> None:
    """``MSD = |v|^2 tau^2`` exactly, so the log-log slope is two - and a
    diffusion coefficient fitted to that would be meaningless."""
    n_chains, n_frames, interval = 40, 60, 0.5
    generator = np.random.default_rng(4)
    velocities = generator.normal(0.0, 0.05, size=(n_chains, 3))
    times = np.arange(n_frames, dtype=np.float64) * interval
    frames = np.zeros((n_frames, n_chains * 2, 3), dtype=np.float64)
    for index, time in enumerate(times):
        frames[index, 0::2, :] = velocities * time
        frames[index, 1::2, :] = velocities * time + np.array([0.0, 0.0, 0.6])
    measured = centre_of_mass_msd(
        synthetic_ensemble(frames, n_chains=n_chains, interval_ps=interval),
        remove_box_scaling=False,
    )
    assert measured.log_slope == pytest.approx(2.0, rel=1e-6)
    assert not measured.diffusive
    assert measured.diffusion_coefficient_cm2_s is None


def test_a_random_walk_gives_back_the_diffusion_coefficient_it_was_built_with() -> None:
    """The unit conversion is where this breaks: one nm^2/ps is 1e-2 cm^2/s,
    and ``MSD = 6 D tau`` puts a factor of six in front of it."""
    expected = 0.5
    frames = random_walk_frames(1200, 400, expected, 0.5, seed=17)
    measured = centre_of_mass_msd(
        synthetic_ensemble(frames, n_chains=400, interval_ps=0.5, box_nm=100.0),
        remove_box_scaling=False,
    )
    assert measured.diffusive
    assert DIFFUSIVE_SLOPE_RANGE[0] <= measured.log_slope <= DIFFUSIVE_SLOPE_RANGE[1]
    assert measured.diffusion_coefficient_cm2_s == pytest.approx(
        expected * NM2_PS_TO_CM2_S, rel=0.1
    )


def test_a_cell_drifting_as_a_whole_does_not_read_as_diffusion() -> None:
    """Subtracting the cell's own centre of mass is what stops residual drift
    of the box being reported as the chains moving."""
    n_frames, n_chains = 40, 20
    frames = np.zeros((n_frames, n_chains * 2, 3), dtype=np.float64)
    base = dimer_cell(n_chains)
    for index in range(n_frames):
        frames[index] = base + np.array([0.1 * index, 0.0, 0.0])
    measured = centre_of_mass_msd(
        synthetic_ensemble(frames, n_chains=n_chains, interval_ps=1.0),
        remove_box_scaling=False,
    )
    assert float(measured.msd_nm2.max()) < 1e-20


def shrinking_cell(n_frames: int = 40, n_chains: int = 27) -> Any:
    """A cell compressed affinely, with nothing moving relative to the cell.

    Every molecule is translated exactly as the barostat would translate it -
    which is what ``scale_molecules_as_rigid`` makes true - so all the apparent
    displacement is the cell shrinking and none of it is diffusion.
    """
    base = dimer_cell(n_chains)
    edges = np.linspace(4.0, 3.6, n_frames)
    frames = np.asarray([base * (edge / edges[0]) for edge in edges], dtype=np.float64)
    boxes = np.stack([np.full(3, edge) for edge in edges])
    return synthetic_ensemble(frames, n_chains=n_chains, interval_ps=1.0, box_nm=boxes)


def test_removing_the_barostats_scaling_reports_how_much_it_removed() -> None:
    """A claim that a correction was applied is worth less than the size of it."""
    edges = np.linspace(4.0, 3.6, 40)
    measured = centre_of_mass_msd(shrinking_cell(), remove_box_scaling=True)
    assert measured.box_drift_fraction == pytest.approx(0.4 / edges.mean(), rel=1e-6)
    assert float(measured.msd_nm2.max()) < 1e-18


def test_leaving_the_scaling_in_makes_affine_shrinkage_look_like_motion() -> None:
    """Which is why removing it is the default: the chains have not moved
    relative to the cell, and without the correction they appear to have."""
    uncorrected = centre_of_mass_msd(shrinking_cell(), remove_box_scaling=False)
    assert float(uncorrected.msd_nm2.max()) > 1e-3


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_an_impossible_lag_fraction_is_refused(fraction: float) -> None:
    """It is a fraction of the trajectory, so zero and more than one are both
    meaningless rather than merely unhelpful."""
    frames = random_walk_frames(20, 4, 0.5, 1.0, seed=2)
    ensemble = synthetic_ensemble(frames, n_chains=4, interval_ps=1.0)
    with pytest.raises(AnalysisError, match="between zero and one"):
        centre_of_mass_msd(ensemble, max_lag_fraction=fraction)


def test_a_trajectory_too_short_for_two_lags_is_refused() -> None:
    """One lag is a single point, which no decay can be read off."""
    frames = random_walk_frames(3, 4, 0.5, 1.0, seed=2)
    ensemble = synthetic_ensemble(frames, n_chains=4, interval_ps=1.0)
    with pytest.raises(AnalysisError, match="too few to measure a decay"):
        centre_of_mass_msd(ensemble, max_lag_fraction=0.5)


def test_massless_chains_have_no_centre_of_mass() -> None:
    """A topology whose elements went missing would otherwise divide by zero."""
    frames = random_walk_frames(20, 4, 0.5, 1.0, seed=2)
    ensemble = synthetic_ensemble(
        frames, n_chains=4, interval_ps=1.0, masses_amu=np.zeros(2)
    )
    with pytest.raises(AnalysisError, match="no centre of mass"):
        centre_of_mass_msd(ensemble)


def test_the_measurements_that_need_a_trajectory_refuse_a_snapshot() -> None:
    """A displacement and a relaxation time are about change over time, and a
    number from one frame would be a fabrication."""
    snapshot = synthetic_ensemble(dimer_cell(), n_chains=32)
    with pytest.raises(AnalysisError, match="needs a trajectory"):
        centre_of_mass_msd(snapshot)
    with pytest.raises(AnalysisError, match="needs a trajectory"):
        end_to_end_relaxation(snapshot, (0, 1))


def test_a_correlation_already_below_the_threshold_crosses_at_the_first_lag() -> None:
    """A chain that has forgotten its orientation within one frame relaxed
    faster than the trajectory can resolve, and the first lag is the only
    answer available."""
    from openmmpolymer.conformation import _crossing

    lags = np.array([0.0, 1.0, 2.0])
    assert _crossing(lags, np.array([0.1, 0.05, 0.0]), 0.5) == pytest.approx(0.0)


def test_a_correlation_that_never_crosses_has_no_crossing() -> None:
    """Reported as None, not as the last lag measured."""
    from openmmpolymer.conformation import _crossing

    assert _crossing(np.arange(3.0), np.ones(3), 0.5) is None


@pytest.mark.parametrize(
    "correlation",
    [
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0]),
    ],
)
def test_a_correlation_with_nothing_to_fit_gives_no_decay_length(
    correlation: np.ndarray,
) -> None:
    """Fewer than two positive points is not a curve, and returning zero says
    so without pretending to a fit."""
    from openmmpolymer.conformation import _decay_length

    distance = np.arange(correlation.size, dtype=np.float64)
    assert _decay_length(distance, correlation) == 0.0


def test_a_decay_over_no_distance_gives_no_decay_length() -> None:
    """A bond length of zero would make every separation the same distance."""
    from openmmpolymer.conformation import _decay_length

    assert _decay_length(np.zeros(4), np.array([1.0, 0.8, 0.6, 0.4])) == 0.0


def test_a_correlation_that_grows_has_an_infinite_decay_length() -> None:
    """Not a negative one, which is what the reciprocal of a positive slope
    would give and would read as a physical length."""
    from openmmpolymer.conformation import _decay_length

    length = _decay_length(
        np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.2, 0.3, 0.4, 0.5])
    )
    assert length == math.inf


def test_a_displacement_with_too_few_usable_points_has_no_slope() -> None:
    """A log-log slope needs two positive points, and a stationary cell gives
    none at all."""
    from openmmpolymer.conformation import _log_slope

    assert _log_slope(np.array([0.0, 1.0]), np.array([0.0, 0.0])) == 0.0


def test_a_slope_over_fewer_than_a_decade_uses_every_lag_it_has() -> None:
    """A short trajectory does not span a decade of lag times, and refusing to
    report a slope at all would be worse than widening the window."""
    from openmmpolymer.conformation import _log_slope

    lags = np.array([1.0, 1.2, 1.4])
    assert _log_slope(lags, lags) == pytest.approx(1.0, rel=1e-9)


def test_a_diffusion_fit_needs_more_than_one_lag() -> None:
    """Two points define the line; one defines nothing."""
    from openmmpolymer.conformation import _diffusion_cm2_s

    assert _diffusion_cm2_s(np.array([0.0, 1.0]), np.array([0.0, 1.0])) is None


def test_a_displacement_that_shrinks_gives_no_diffusion_coefficient() -> None:
    """A negative slope is not a small diffusion coefficient."""
    from openmmpolymer.conformation import _diffusion_cm2_s

    lags = np.arange(10, dtype=np.float64)
    assert _diffusion_cm2_s(lags, -lags) is None


def test_a_cell_with_no_size_reports_no_box_drift() -> None:
    """Dividing by a mean edge of zero would give a NaN that then travels."""
    from openmmpolymer.conformation import _box_drift

    assert _box_drift(np.zeros((3, 3))) == 0.0
