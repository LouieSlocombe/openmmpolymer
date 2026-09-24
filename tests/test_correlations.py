"""Tests for the pair distribution and the structure factor."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from openmmpolymer.correlations import (
    radial_distribution,
    structure_factor,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import _lattice, synthetic_ensemble

#: The simple cubic lattice the argon fixtures use: 64 sites, 0.6 nm apart, in
#: a 2.4 nm cell. Its neighbour shells are exactly 6, 12 and 8 at 0.6,
#: 0.6*sqrt(2) and 0.6*sqrt(3) nm, all inside half the cell - so the
#: coordination numbers are integers that can be asserted outright.
LATTICE_SPACING_NM = 0.6
LATTICE_EDGE_NM = 2.4


def lattice_ensemble(n_chains: int = 64) -> Any:
    """The 64-site cubic lattice, split into *n_chains* equal molecules."""
    return synthetic_ensemble(
        _lattice(64, LATTICE_EDGE_NM), n_chains=n_chains, box_nm=LATTICE_EDGE_NM
    )


def coordination_at(distribution: Any, radius_nm: float) -> float:
    """The running neighbour count at the bin covering *radius_nm*."""
    index = int(np.searchsorted(distribution.r_nm, radius_nm))
    return float(distribution.coordination_number[index])


@pytest.mark.parametrize(
    ("radius_nm", "neighbours"),
    [(0.7, 6.0), (0.95, 18.0), (1.1, 26.0)],
)
def test_a_cubic_lattice_gives_its_exact_coordination_numbers(
    radius_nm: float, neighbours: float
) -> None:
    """Six, then twelve more, then eight more. Integers, so nothing about the
    periodic boundaries or the shell volumes can be slightly wrong."""
    measured = radial_distribution(
        lattice_ensemble(), n_bins=240, heavy_atoms_only=False
    )
    assert coordination_at(measured, radius_nm) == pytest.approx(neighbours, abs=1e-9)


def test_the_shells_sit_where_the_lattice_puts_them() -> None:
    """Which is the check that the minimum image convention is being applied:
    without it the shells past the cell's half-width go missing."""
    measured = radial_distribution(
        lattice_ensemble(), n_bins=240, heavy_atoms_only=False
    )
    occupied = measured.r_nm[measured.g_r > 0.0]
    for expected in (
        LATTICE_SPACING_NM,
        LATTICE_SPACING_NM * math.sqrt(2),
        LATTICE_SPACING_NM * math.sqrt(3),
    ):
        assert np.min(np.abs(occupied - expected)) < 0.01


def test_an_ideal_gas_normalises_to_one() -> None:
    """The whole point of the normalisation: a structureless fluid has to come
    out flat at one, or every peak height read off a melt is wrong."""
    generator = np.random.default_rng(5)
    edge, n_atoms = 6.0, 4000
    gas = generator.uniform(0.0, edge, size=(n_atoms, 3))
    measured = radial_distribution(
        synthetic_ensemble(gas, n_chains=n_atoms, box_nm=edge),
        n_bins=60,
        heavy_atoms_only=False,
    )
    tail = measured.g_r[measured.r_nm > 1.0]
    assert tail == pytest.approx(1.0, abs=0.05)


def test_the_number_density_is_the_cell_it_was_measured_in() -> None:
    """It is what the normalisation divides by, so it is worth reporting."""
    measured = radial_distribution(lattice_ensemble(), heavy_atoms_only=False)
    assert measured.number_density_nm3 == pytest.approx(64 / LATTICE_EDGE_NM**3)


def test_pairs_inside_a_molecule_are_left_out() -> None:
    """The intramolecular distribution is a property of the force field. Read as
    the same lattice in dimers, each site loses exactly its bonded partner."""
    measured = radial_distribution(
        lattice_ensemble(n_chains=32), n_bins=240, heavy_atoms_only=False
    )
    assert coordination_at(measured, 0.7) == pytest.approx(5.0, abs=1e-9)


def test_the_default_radius_is_as_far_as_the_cell_allows() -> None:
    """Half the smallest edge, which is where the minimum image stops holding."""
    measured = radial_distribution(lattice_ensemble(), heavy_atoms_only=False)
    assert measured.r_max_nm == pytest.approx(LATTICE_EDGE_NM / 2)


def test_the_default_radius_survives_the_kernels_float32_box() -> None:
    """An NPT cell whose float32 half-edge rounds below the float64 one: the
    distance kernel must not refuse the radius it was just handed. A hundred
    atoms or more, because below that MDAnalysis never uses its grid search."""
    edge = 2.7858951568603514
    assert float(np.float32(edge)) / 2.0 < edge / 2.0
    measured = radial_distribution(
        synthetic_ensemble(_lattice(216, edge), n_chains=216, box_nm=edge),
        heavy_atoms_only=False,
    )
    assert measured.r_max_nm == pytest.approx(edge / 2)


def test_a_radius_past_half_the_cell_is_refused() -> None:
    """Past it the convention counts one neighbour as two, and the distance
    kernel applies it anyway without complaining."""
    with pytest.raises(AnalysisError, match="more than half the smallest"):
        radial_distribution(lattice_ensemble(), r_max_nm=2.0, heavy_atoms_only=False)


def test_a_cell_of_one_molecule_has_no_intermolecular_pairs() -> None:
    """Saying so beats returning an empty distribution that looks like data."""
    with pytest.raises(AnalysisError, match="no intermolecular pairs"):
        radial_distribution(lattice_ensemble(n_chains=1), heavy_atoms_only=False)


def test_dropping_hydrogens_from_an_all_hydrogen_cell_is_refused() -> None:
    """It leaves nothing to measure, and an empty array is not an answer."""
    ensemble = synthetic_ensemble(
        _lattice(64, LATTICE_EDGE_NM),
        n_chains=32,
        box_nm=LATTICE_EDGE_NM,
        is_hydrogen=np.ones(2, dtype=bool),
    )
    with pytest.raises(AnalysisError, match="left no atoms"):
        radial_distribution(ensemble)


def test_hydrogens_can_be_dropped() -> None:
    """A melt's hydrogens trace the same structure as the carbons they hang
    off, at four times the pair count."""
    ensemble = synthetic_ensemble(
        _lattice(64, LATTICE_EDGE_NM),
        n_chains=32,
        box_nm=LATTICE_EDGE_NM,
        is_hydrogen=np.array([False, True]),
    )
    heavy = radial_distribution(ensemble, n_bins=60, heavy_atoms_only=True)
    everything = radial_distribution(ensemble, n_bins=60, heavy_atoms_only=False)
    assert heavy.heavy_atoms_only
    assert heavy.n_pairs < everything.n_pairs


def test_the_bins_span_zero_to_the_measured_radius() -> None:
    """A peak position is read off these, so their placement matters."""
    measured = radial_distribution(
        lattice_ensemble(), n_bins=10, heavy_atoms_only=False
    )
    assert measured.r_nm.size == 10
    width = measured.r_max_nm / 10
    assert measured.r_nm[0] == pytest.approx(width / 2)
    assert measured.r_nm[-1] == pytest.approx(measured.r_max_nm - width / 2)


def test_the_frame_count_is_reported_alongside_the_answer() -> None:
    """A g(r) from one frame and one from five hundred have the same shape and
    very different standing."""
    frames = np.repeat(_lattice(64, LATTICE_EDGE_NM)[None, :, :], 6, axis=0)
    measured = radial_distribution(
        synthetic_ensemble(
            frames, n_chains=64, box_nm=LATTICE_EDGE_NM, interval_ps=1.0
        ),
        n_bins=60,
        heavy_atoms_only=False,
        stride=2,
    )
    assert measured.n_frames == 3


def test_the_structure_factor_peaks_on_the_lattices_reciprocal_vector() -> None:
    """A perfect lattice puts everything into Bragg peaks at 2 pi / a, and the
    peak of a coherent sum over N scatterers is N."""
    measured = structure_factor(
        lattice_ensemble(),
        q_max_per_nm=30.0,
        n_bins=120,
        heavy_atoms_only=False,
        # A Bragg peak is sharp, so the bin holding it has only the few
        # wavevectors of that exact magnitude the cell allows. The default
        # floor is set for an amorphous halo, which is the opposite case.
        min_vectors_per_bin=1,
    )
    assert measured.first_peak_per_nm == pytest.approx(
        2 * math.pi / LATTICE_SPACING_NM, rel=0.02
    )
    assert float(measured.s_q.max()) == pytest.approx(64.0, rel=1e-6)


def test_the_resolution_floor_is_the_cell_it_was_measured_in() -> None:
    """Nothing below 2 pi / L is measurable, whatever the curve does, because
    the cell cannot hold a longer wave."""
    measured = structure_factor(
        lattice_ensemble(), q_max_per_nm=30.0, heavy_atoms_only=False
    )
    assert measured.q_min_per_nm == pytest.approx(2 * math.pi / LATTICE_EDGE_NM)


def test_every_bin_reports_how_many_wavevectors_fed_it() -> None:
    """A bin averaged over three wavevectors is not worth what one averaged
    over three hundred is."""
    measured = structure_factor(
        lattice_ensemble(), q_max_per_nm=20.0, n_bins=40, heavy_atoms_only=False
    )
    assert measured.n_vectors.sum() > 0
    assert measured.s_q[measured.n_vectors == 0] == pytest.approx(0.0)


def test_asking_for_more_wavevectors_than_will_be_summed_is_refused() -> None:
    """The answer to a cell too big for the q range is to ask for less, not to
    wait for a sum that will not finish."""
    with pytest.raises(AnalysisError, match="Lower q_max_per_nm"):
        structure_factor(lattice_ensemble(), q_max_per_nm=1.0e6, heavy_atoms_only=False)


def test_the_structure_factor_also_refuses_a_cell_with_no_heavy_atoms() -> None:
    """Same reason as the pair distribution: nothing left to sum over."""
    ensemble = synthetic_ensemble(
        _lattice(64, LATTICE_EDGE_NM),
        n_chains=32,
        box_nm=LATTICE_EDGE_NM,
        is_hydrogen=np.ones(2, dtype=bool),
    )
    with pytest.raises(AnalysisError, match="left no atoms"):
        structure_factor(ensemble)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_bin_count_that_is_not_a_count_is_refused(bad: int) -> None:
    """Zero bins is not a coarse histogram, it is no histogram."""
    with pytest.raises(ValueError, match="at least 1"):
        radial_distribution(lattice_ensemble(), n_bins=bad, heavy_atoms_only=False)


def test_a_radius_inside_the_cell_is_used_as_given() -> None:
    """Asking for less than the cell allows is a legitimate way to keep the
    pair count down on a big cell."""
    measured = radial_distribution(
        lattice_ensemble(), r_max_nm=0.8, n_bins=40, heavy_atoms_only=False
    )
    assert measured.r_max_nm == pytest.approx(0.8)
    assert coordination_at(measured, 0.7) == pytest.approx(6.0, abs=1e-9)


def test_a_cell_too_sparse_to_hold_a_pair_gives_an_empty_distribution() -> None:
    """Two molecules further apart than the radius asked for is not an error,
    it is a g(r) of zero - and the pair count says why."""
    far = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)
    measured = radial_distribution(
        synthetic_ensemble(far, n_chains=2, box_nm=20.0),
        r_max_nm=0.5,
        n_bins=10,
        heavy_atoms_only=False,
    )
    assert measured.n_pairs == 0
    assert measured.g_r == pytest.approx(np.zeros(10))


def test_the_peak_is_not_reported_below_the_resolution_floor() -> None:
    """The bins under 2 pi / L are fed by a handful of the longest wavevectors
    the cell holds, where the sum follows the cell's own periodicity rather
    than the structure. A peak there contradicts the floor reported beside it.
    """
    measured = structure_factor(
        lattice_ensemble(), q_max_per_nm=25.0, n_bins=100, heavy_atoms_only=False
    )
    assert measured.first_peak_per_nm >= measured.q_min_per_nm


def test_a_structure_factor_with_nothing_resolvable_reports_no_peak() -> None:
    """Asking only for wavevectors the cell cannot hold leaves no peak to name,
    and zero says that more honestly than the first bin's centre would."""
    from openmmpolymer.correlations import _peak_above

    q = np.array([0.5, 1.0, 1.5])
    counted = np.full(3, 50, dtype=np.int64)
    assert _peak_above(q, np.array([5.0, 1.0, 1.0]), counted, 10.0, 20) == 0.0


def test_a_thinly_populated_bin_cannot_be_the_peak() -> None:
    """The longest waves a cell holds come in threes, and their sum follows the
    cell's periodicity rather than the structure inside it. On a real melt they
    overtake the amorphous halo tenfold, and the halo is the answer wanted."""
    from openmmpolymer.correlations import _peak_above

    q = np.array([2.0, 4.0, 12.0])
    s_q = np.array([26.0, 4.0, 3.0])
    vectors = np.array([3, 6, 196], dtype=np.int64)
    assert _peak_above(q, s_q, vectors, 1.9, 20) == pytest.approx(12.0)
