"""What the chains look like, and whether they were moving while measured.

The static and the dynamic measurements live together because one is evidence
for the other. ``<R^2>`` and the radius of gyration describe a melt only if the
chains have explored their own configuration space, and the only thing that can
say whether they have is how fast the end-to-end vector decorrelates. A cell
packed by packmol and equilibrated for a few hundred picoseconds has perfectly
respectable chain dimensions and has not relaxed at all;
:class:`~openmmpolymer.protocols.ChainDimensions` says as much about a single
frame, and :func:`end_to_end_relaxation` is what turns that caution into a
number.

So :func:`chain_conformation` reports the dimensions *and* the equilibration of
the series it averaged, and :func:`centre_of_mass_msd` refuses to divide a
mean-squared displacement by six until its log-log slope says the chains are
actually diffusing. A sub-linear slope is caged motion, and fitting a diffusion
coefficient to it produces a number that looks like one and is not.

The per-frame arithmetic is shared with
:func:`~openmmpolymer.protocols.chain_dimensions`. Counts and sums accumulated
while measuring each frame give the trajectory's pooled means and their ratios
without retaining or measuring the coordinates again.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._fitting import TINY
from ._validation import require_integer
from .protocols import ChainDimensions, _chain_dimension_sums, _ChainDimensionSums
from .timeseries import Equilibration, equilibration
from .trajectory import (
    AnalysisError,
    Ensemble,
    _load_chain_frames,
    backbone_indices,
    chain_positions,
    require_trajectory,
)

log = logging.getLogger(__name__)

#: A correlation function is taken to have decayed once it drops below this.
#: One over e is the conventional mark, and what a relaxation time is read at.
DECORRELATION_THRESHOLD = 1.0 / math.e

#: Log-log slopes of a mean-squared displacement that count as diffusive.
#: Ballistic motion gives two and caged motion much less than one; only inside
#: this window does ``MSD = 6 D t`` mean anything.
DIFFUSIVE_SLOPE_RANGE = (0.9, 1.1)

#: One nm^2/ps in cm^2/s. A nanometre is 1e-7 cm and a picosecond 1e-12 s, so
#: the ratio is 1e-14 / 1e-12.
NM2_PS_TO_CM2_S = 1.0e-2

#: Decades of lag time the diffusive slope is measured over, at the long-time
#: end where the ballistic and caged regimes have been left behind.
_SLOPE_DECADES = 1.0


@dataclass(frozen=True)
class ConformationSeries:
    """Chain dimensions frame by frame, and their average.

    Args:
        stage: Which stage this came from.
        time_ps: Time at each frame measured.
        mean_squared_end_to_end_nm2: ``<R^2>`` at each frame, averaged over
            chains.
        mean_radius_of_gyration_nm: Mass-weighted radius of gyration at each
            frame, averaged over chains.
        mean: The dimensions over every chain in every frame measured.
        settled: The equilibration of the ``<R^2>`` series.
            :attr:`~openmmpolymer.protocols.ChainDimensions.consistent` on
            :attr:`mean` says the dimensions look like a melt's; this says
            whether they had stopped moving while being measured. For a single
            snapshot there is no series, so this is None.
        n_chains: How many molecules were averaged over.
        n_frames: How many frames were measured.
    """

    stage: str
    time_ps: npt.NDArray[np.float64]
    mean_squared_end_to_end_nm2: npt.NDArray[np.float64]
    mean_radius_of_gyration_nm: npt.NDArray[np.float64]
    mean: ChainDimensions
    settled: Equilibration | None
    n_chains: int
    n_frames: int


@dataclass(frozen=True)
class PersistenceLength:
    """How far along a backbone the direction is remembered.

    Args:
        separation: Backbone bond separations, in bonds.
        correlation: ``<cos theta(s)>`` at each separation.
        bond_length_nm: Mean backbone bond length.
        persistence_length_nm: From a straight-line fit of the log
            correlation against contour distance, over the first decay only.
        n_bonds: Backbone bonds in one chain.
        contour_length_nm: Bonds times bond length - how long the chain
            actually is.
        decayed: Whether the correlation reached
            :data:`DECORRELATION_THRESHOLD` within the chain. When False the
            persistence length is an extrapolation past the end of the
            molecule, and says more about the fit than about the polymer.
    """

    separation: npt.NDArray[np.float64]
    correlation: npt.NDArray[np.float64]
    bond_length_nm: float
    persistence_length_nm: float
    n_bonds: int
    contour_length_nm: float
    decayed: bool


@dataclass(frozen=True)
class EndToEndRelaxation:
    """How fast the end-to-end vector forgets where it was pointing.

    Args:
        lag_ps: Lag times measured.
        correlation: ``<u(t+tau).u(t)> / <u.u>`` at each lag, over every chain
            and every time origin.
        relaxation_time_ps: Where the correlation crosses
            :data:`DECORRELATION_THRESHOLD`, or None if it never does.
        trajectory_ps: How long the trajectory was. When
            :attr:`decorrelated` is False this is the only honest statement
            available: the relaxation time is longer than this.
        n_chains: Chains averaged over.
        n_origins: Time origins averaged over at the shortest lag.
        decorrelated: Whether the correlation decayed within the lags
            measured. This is the measurement that says whether the chains
            relaxed at their own scale. No protocol shipped with this package
            runs for a Rouse time, so False is the expected answer and not a
            failure.
    """

    lag_ps: npt.NDArray[np.float64]
    correlation: npt.NDArray[np.float64]
    relaxation_time_ps: float | None
    trajectory_ps: float
    n_chains: int
    n_origins: int
    decorrelated: bool


@dataclass(frozen=True)
class MeanSquaredDisplacement:
    """How far the chains' centres of mass wandered.

    Args:
        lag_ps: Lag times measured.
        msd_nm2: Mean-squared displacement of a chain's centre of mass at each
            lag, over every chain and time origin.
        log_slope: Slope of log MSD against log lag over the last decade. One
            is diffusion, two is ballistic, well below one is caged.
        diffusion_coefficient_cm2_s: ``MSD / 6 tau`` from a fit over the
            diffusive part, or None when :attr:`diffusive` is False.
        box_drift_fraction: How much the cell edge changed over the frames
            used, as a fraction of its mean. What ``remove_box_scaling``
            had to take out.
        n_chains: Chains averaged over.
        n_origins: Time origins averaged over at the shortest lag.
        diffusive: Whether :attr:`log_slope` sits within
            :data:`DIFFUSIVE_SLOPE_RANGE`. When False the diffusion
            coefficient is None rather than misleading: a melt run short of
            its entanglement time is sub-diffusive, and dividing that MSD by
            six gives a number with the units of a diffusion coefficient and
            none of the meaning.
    """

    lag_ps: npt.NDArray[np.float64]
    msd_nm2: npt.NDArray[np.float64]
    log_slope: float
    diffusion_coefficient_cm2_s: float | None
    box_drift_fraction: float
    n_chains: int
    n_origins: int
    diffusive: bool


def chain_conformation(
    ensemble: Ensemble,
    backbone: Sequence[int],
    *,
    expected_characteristic_ratio: float = 7.0,
    stride: int = 1,
) -> ConformationSeries:
    """Measure chain dimensions over a stage.

    Args:
        ensemble: A snapshot or a trajectory.
        backbone: Backbone atom indices within one chain, in order, as
            :attr:`~openmmpolymer.chain.ChainResult.backbone` gives them.
        expected_characteristic_ratio: The polymer's C-infinity.
        stride: Measure every *stride*-th frame.

    Returns:
        The dimensions frame by frame and on average.

    Raises:
        AnalysisError: The backbone does not fit the chains.
        ValueError: *stride* is not a positive integer.
    """
    path = backbone_indices(backbone, ensemble.atoms_per_chain)
    require_integer(stride, name="stride")
    indices = [int(index) for index in path]

    squares: list[float] = []
    radii: list[float] = []
    times: list[float] = []
    pooled = _ChainDimensionSums()
    for frame in ensemble.frames(stride=stride):
        sums = _chain_dimension_sums(
            frame.positions_nm,
            indices,
            ensemble.atoms_per_chain,
            ensemble.n_chains,
            masses=ensemble.masses_amu,
        )
        measured = sums.dimensions(
            len(indices) - 1,
            expected_characteristic_ratio=expected_characteristic_ratio,
        )
        squares.append(measured.mean_squared_end_to_end_nm2)
        radii.append(measured.mean_radius_of_gyration_nm)
        times.append(frame.time_ps)
        pooled.add(sums)

    # Both ratios use pooled means, not averages of each frame's ratios.
    overall = pooled.dimensions(
        len(indices) - 1,
        expected_characteristic_ratio=expected_characteristic_ratio,
    )
    time_ps = np.asarray(times, dtype=np.float64)
    squared = np.asarray(squares, dtype=np.float64)
    settled = equilibration(time_ps, squared) if time_ps.size >= 3 else None
    return ConformationSeries(
        stage=ensemble.stage,
        time_ps=time_ps,
        mean_squared_end_to_end_nm2=squared,
        mean_radius_of_gyration_nm=np.asarray(radii, dtype=np.float64),
        mean=overall,
        settled=settled,
        n_chains=ensemble.n_chains,
        n_frames=len(times),
    )


def persistence_length(
    ensemble: Ensemble, backbone: Sequence[int], *, stride: int = 1
) -> PersistenceLength:
    """Measure how far correlation survives along the backbone.

    Args:
        ensemble: A snapshot or a trajectory.
        backbone: Backbone atom indices within one chain, in order.
        stride: Measure every *stride*-th frame.

    Returns:
        The correlation curve and the length fitted to it.

    Raises:
        AnalysisError: The backbone has fewer than three bonds, so there is no
            curve to fit.
    """
    path = backbone_indices(backbone, ensemble.atoms_per_chain)
    require_integer(stride, name="stride")
    if path.size < 4:
        raise AnalysisError(
            f"A backbone of {path.size} atoms gives {path.size - 1} bonds, too "
            "few for a correlation curve. Three or more are needed."
        )

    positions, _ = chain_positions(ensemble, stride=stride)
    backbone_atoms = positions[:, :, path, :]
    bonds = backbone_atoms[:, :, 1:, :] - backbone_atoms[:, :, :-1, :]
    lengths = np.linalg.norm(bonds, axis=-1)
    bond_length = float(lengths.mean())
    units = bonds / np.clip(lengths, TINY, None)[..., None]

    n_bonds = units.shape[2]
    separations = np.arange(n_bonds, dtype=np.float64)
    correlation = np.asarray(
        [
            float(
                (units[:, :, : n_bonds - gap, :] * units[:, :, gap:, :]).sum(-1).mean()
            )
            for gap in range(n_bonds)
        ],
        dtype=np.float64,
    )

    decayed = bool(np.any(correlation < DECORRELATION_THRESHOLD))
    fitted = _decay_length(separations * bond_length, correlation)
    return PersistenceLength(
        separation=separations,
        correlation=correlation,
        bond_length_nm=bond_length,
        persistence_length_nm=fitted,
        n_bonds=int(n_bonds),
        contour_length_nm=n_bonds * bond_length,
        decayed=decayed,
    )


def end_to_end_relaxation(
    ensemble: Ensemble,
    backbone: Sequence[int],
    *,
    max_lag_fraction: float = 0.5,
) -> EndToEndRelaxation:
    """Measure how fast the end-to-end vector decorrelates.

    Args:
        ensemble: A trajectory. A snapshot cannot answer this.
        backbone: Backbone atom indices within one chain, in order.
        max_lag_fraction: Longest lag to measure, as a fraction of the
            trajectory. Beyond about half there are too few time origins left
            for the average to mean anything.

    Returns:
        The correlation function and, if it decayed, the time it decayed in.

    Raises:
        AnalysisError: *ensemble* is a single snapshot, or *max_lag_fraction*
            leaves no lags to measure.
    """
    require_trajectory(ensemble, "An end-to-end relaxation time")
    path = backbone_indices(backbone, ensemble.atoms_per_chain)

    positions, times = chain_positions(ensemble)
    vectors = positions[:, :, path[-1], :] - positions[:, :, path[0], :]
    n_lags = _lag_count(vectors.shape[0], max_lag_fraction)

    reference = float((vectors * vectors).sum(-1).mean())
    if reference <= TINY:
        raise AnalysisError(
            "Every chain's end-to-end vector has zero length, so there is "
            "nothing to correlate. Is the backbone path right?"
        )
    correlation = np.asarray(
        [
            float((vectors[: vectors.shape[0] - lag] * vectors[lag:]).sum(-1).mean())
            / reference
            for lag in range(n_lags)
        ],
        dtype=np.float64,
    )
    lag_ps = np.arange(n_lags, dtype=np.float64) * ensemble.interval_ps
    crossing = _crossing(lag_ps, correlation, DECORRELATION_THRESHOLD)
    return EndToEndRelaxation(
        lag_ps=lag_ps,
        correlation=correlation,
        relaxation_time_ps=crossing,
        trajectory_ps=float(times[-1] - times[0]),
        n_chains=ensemble.n_chains,
        n_origins=int(vectors.shape[0]),
        decorrelated=crossing is not None,
    )


def centre_of_mass_msd(
    ensemble: Ensemble,
    *,
    max_lag_fraction: float = 0.5,
    stride: int = 1,
    remove_box_scaling: bool = True,
) -> MeanSquaredDisplacement:
    """Measure how far the chains' centres of mass moved.

    The chain centre of mass, not the atoms: a per-atom displacement in a
    polymer melt is dominated by internal modes, which is a different
    observable rather than a noisier version of this one. The whole cell's
    centre of mass is subtracted so that any residual drift of the box does
    not read as diffusion.

    Args:
        ensemble: A trajectory. A snapshot cannot answer this.
        max_lag_fraction: Longest lag to measure, as a fraction of the
            trajectory.
        stride: Use every *stride*-th frame.
        remove_box_scaling: Undo the barostat's affine scaling of the cell.
            Exact rather than approximate, because
            :attr:`~openmmpolymer.mdsystem.SystemSpec.scale_molecules_as_rigid`
            is on by default, so a volume move translates each molecule
            rigidly and is a pure affine map on the centres of mass.

    Returns:
        The displacement curve, its slope, and a diffusion coefficient when
        the slope earns one.

    Raises:
        AnalysisError: *ensemble* is a single snapshot, or *max_lag_fraction*
            leaves no lags to measure.
    """
    require_trajectory(ensemble, "A mean-squared displacement")
    require_integer(stride, name="stride")

    positions, _, boxes = _load_chain_frames(ensemble, stride=stride)
    weights = ensemble.masses_amu
    total = float(weights.sum())
    if total <= TINY:
        raise AnalysisError("The chains have no mass, so they have no centre of mass.")
    centres = np.einsum("a,fcad->fcd", weights, positions) / total

    drift = _box_drift(boxes)
    if remove_box_scaling:
        volumes = boxes.prod(axis=1)
        centres = centres * (volumes[0] / volumes)[:, None, None] ** (1.0 / 3.0)
    centres = centres - centres.mean(axis=1, keepdims=True)

    n_lags = _lag_count(centres.shape[0], max_lag_fraction)
    msd = np.asarray(
        [
            float(
                ((centres[lag:] - centres[: centres.shape[0] - lag]) ** 2)
                .sum(-1)
                .mean()
            )
            for lag in range(n_lags)
        ],
        dtype=np.float64,
    )
    lag_ps = np.arange(n_lags, dtype=np.float64) * (ensemble.interval_ps * stride)
    slope = _log_slope(lag_ps, msd)
    diffusive = DIFFUSIVE_SLOPE_RANGE[0] <= slope <= DIFFUSIVE_SLOPE_RANGE[1]
    coefficient: float | None = None
    if diffusive:
        coefficient = _diffusion_cm2_s(lag_ps, msd)
    else:
        log.info(
            "%s: the mean-squared displacement goes as lag^%.2f, outside "
            "%.1f-%.1f, so no diffusion coefficient is reported. A melt run "
            "shorter than its entanglement time is sub-diffusive.",
            ensemble.stage,
            slope,
            *DIFFUSIVE_SLOPE_RANGE,
        )
    return MeanSquaredDisplacement(
        lag_ps=lag_ps,
        msd_nm2=msd,
        log_slope=slope,
        diffusion_coefficient_cm2_s=coefficient,
        box_drift_fraction=drift,
        n_chains=ensemble.n_chains,
        n_origins=int(centres.shape[0]),
        diffusive=bool(diffusive),
    )


def _lag_count(n_frames: int, max_lag_fraction: float) -> int:
    """How many lags to measure, including zero.

    Raises:
        AnalysisError: The fraction is not usable, or leaves under two lags.
    """
    if not 0.0 < max_lag_fraction <= 1.0:
        raise AnalysisError(
            f"max_lag_fraction={max_lag_fraction!r} must be between zero and one."
        )
    lags = int(n_frames * max_lag_fraction)
    if lags < 2:
        raise AnalysisError(
            f"{n_frames} frames at max_lag_fraction={max_lag_fraction} leaves "
            f"{lags} lag(s), too few to measure a decay. Write more frames, or "
            "raise max_lag_fraction."
        )
    return lags


def _crossing(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], level: float
) -> float | None:
    """Where *y* first drops below *level*, linearly interpolated."""
    below = np.flatnonzero(y < level)
    if below.size == 0:
        return None
    index = int(below[0])
    if index == 0:
        return float(x[0])
    # The previous point is at or above the level and this one is below it,
    # so the difference between them is positive and safe to divide by.
    fraction = (y[index - 1] - level) / (y[index - 1] - y[index])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def _decay_length(
    distance: npt.NDArray[np.float64], correlation: npt.NDArray[np.float64]
) -> float:
    """Fit ``log(correlation) = -distance / length`` and return the length.

    Fitted over the first decay only - down to
    :data:`DECORRELATION_THRESHOLD` and one point past it - rather than over
    every positive value. That is where the length is defined, and it is also
    where the estimator behaves: on a freely-rotating chain of known
    stiffness, fitting the whole positive tail comes out 13 to 36 per cent low
    because the tail is noise around zero, while stopping at one over e
    recovers the right answer to within about three per cent. A chain whose
    correlation never falls that far is fitted over everything positive, and
    reported with ``decayed`` False.
    """
    positive = correlation > 0.0
    usable = int(np.argmin(positive)) if not positive.all() else positive.size
    if usable < 2:
        return 0.0
    below = np.flatnonzero(correlation[:usable] < DECORRELATION_THRESHOLD)
    cut = int(below[0]) + 1 if below.size else usable
    cut = max(2, min(cut, usable))
    x = distance[:cut]
    y = np.log(correlation[:cut])
    if float(x[-1] - x[0]) < TINY:
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    if slope >= -TINY:
        return math.inf
    return -1.0 / slope


def _log_slope(lag_ps: npt.NDArray[np.float64], msd: npt.NDArray[np.float64]) -> float:
    """Slope of log MSD against log lag, over the last decade of lags."""
    usable = (lag_ps > 0.0) & (msd > 0.0)
    if usable.sum() < 2:
        return 0.0
    x = np.log10(lag_ps[usable])
    y = np.log10(msd[usable])
    window = x >= x[-1] - _SLOPE_DECADES
    if window.sum() < 2:
        window = np.ones_like(x, dtype=bool)
    return float(np.polyfit(x[window], y[window], 1)[0])


def _diffusion_cm2_s(
    lag_ps: npt.NDArray[np.float64], msd: npt.NDArray[np.float64]
) -> float | None:
    """Fit ``MSD = 6 D tau`` over the second half of the lags.

    The early lags carry the ballistic and caged parts, so the fit starts
    halfway along where the motion has settled into its long-time behaviour.
    """
    start = max(1, lag_ps.size // 2)
    x = lag_ps[start:]
    y = msd[start:]
    if x.size < 2:
        return None
    slope = float(np.polyfit(x, y, 1)[0])
    if slope <= 0.0:
        return None
    return slope / 6.0 * NM2_PS_TO_CM2_S


def _box_drift(boxes_nm_: npt.NDArray[np.float64]) -> float:
    """Spread of the cell edge over the frames used, over its mean."""
    edges = boxes_nm_.mean(axis=1)
    mean = float(edges.mean())
    if mean <= TINY:
        return 0.0
    return float(edges.max() - edges.min()) / mean
