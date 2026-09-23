"""Observation-window stability for structural and dynamical measurements.

Every prefix uses the same coordinate sampling policy and histogram grids.
Stationary quantities are also measured in disjoint tail blocks. Prefixes
overlap, so their differences are stability diagnostics, never standard errors.
Unobserved diffusion, orientational relaxation and backbone decay remain
censored; extending a fit beyond the recorded trajectory cannot resolve them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields, replace
from itertools import pairwise
from typing import Any

import numpy as np

from ._validation import require_integer, require_positive
from .conformation import (
    centre_of_mass_msd,
    chain_conformation,
    end_to_end_relaxation,
    persistence_length,
)
from .correlations import (
    MIN_WAVEVECTORS_PER_BIN,
    RadialDistribution,
    StructureFactor,
    radial_distribution,
    structure_factor,
)
from .structure import MAX_DISTRIBUTION_FRAMES, MAX_STRUCTURE_FACTOR_FRAMES
from .trajectory import AnalysisError, Ensemble, Frame, backbone_indices, boxes_nm


@dataclass(frozen=True)
class StructuralParameterConvergence:
    """Parameter stability, with unknown values and no manufactured error bar."""

    property_name: str
    value_unit: str
    values: tuple[float | None, ...]
    valid: tuple[bool, ...]
    block_values: tuple[float | None, ...]
    relative_change: float | None
    block_relative_change: float | None
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class StructuralWindow:
    """The measured scalars and full curves from one trajectory interval."""

    fraction: float
    first_frame: int
    n_frames: int
    duration_ps: float
    values: dict[str, float | None]
    valid: dict[str, bool]
    sample_counts: dict[str, int]
    curves: dict[str, dict[str, tuple[float, ...]]]
    configuration_varied: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class StructuralWindowConvergence:
    """All prefixes and disjoint blocks, including censored or sparse results.

    ``resolved`` requires every reported parameter to resolve. Read individual
    entries in ``parameters`` when only a subset is applicable to the system.
    This is observation-window stability, not proof of melt equilibration.
    """

    stage: str
    windows: tuple[StructuralWindow, ...]
    blocks: tuple[StructuralWindow, ...]
    parameters: dict[str, StructuralParameterConvergence]
    relative_tolerance: float
    min_frames: int
    settings: dict[str, Any]
    resolved: bool
    notes: tuple[str, ...]


_PARAMETERS = {
    "persistence_length_nm": ("nm", True),
    "characteristic_ratio": ("dimensionless", True),
    "ratio_of_squares": ("dimensionless", True),
    "diffusion_coefficient_cm2_s": ("cm^2/s", False),
    "end_to_end_relaxation_time_ps": ("ps", False),
    "rdf_first_peak_nm": ("nm", True),
    "rdf_first_peak_height": ("dimensionless", True),
    "structure_factor_peak_per_nm": ("1/nm", True),
    "structure_factor_peak_height": ("dimensionless", True),
}


@dataclass(frozen=True)
class _BlockEnsemble(Ensemble):
    """An offset view that shares the reader without copying its coordinates."""

    first_frame: int = 0

    def frames(
        self, *, start: int = 0, stop: int | None = None, stride: int = 1
    ) -> Iterator[Frame]:
        last = self.n_frames if stop is None else min(stop, self.n_frames)
        whole = replace(self, n_frames=self.first_frame + self.n_frames, first_frame=0)
        yield from Ensemble.frames(
            whole,
            start=self.first_frame + start,
            stop=self.first_frame + last,
            stride=stride,
        )


def _block(ensemble: Ensemble, start: int, stop: int) -> Ensemble:
    return _BlockEnsemble(
        **{
            field.name: getattr(ensemble, field.name)
            for field in fields(Ensemble)
            if field.name != "n_frames"
        },
        n_frames=stop - start,
        first_frame=start,
    )


def _fractions(values: Sequence[float]) -> tuple[float, ...]:
    fractions = tuple(float(value) for value in values)
    if (
        len(fractions) < 3
        or any(not math.isfinite(v) or not 0 < v <= 1 for v in fractions)
        or tuple(sorted(set(fractions))) != fractions
        or fractions[-1] != 1
    ):
        raise ValueError(
            "window_fractions needs at least three increasing fractions in (0, 1], ending at 1."
        )
    return fractions


def _stride(n_frames: int, stride: int, cap: int | None) -> int:
    if cap is None:
        return stride
    require_integer(cap, name="frame cap")
    return max(stride, math.ceil(n_frames / cap))


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _curve(**columns: Any) -> dict[str, tuple[float, ...]]:
    return {
        name: tuple(float(value) for value in values)
        for name, values in columns.items()
    }


def _configuration_varied(ensemble: Ensemble) -> bool:
    """Identical coordinates or translation alone supply no sampling evidence."""
    reference = None
    varied = False
    for frame in ensemble.frames():
        if not np.all(np.isfinite(frame.positions_nm)):
            raise AnalysisError(
                "Structural convergence needs finite recorded coordinates."
            )
        centred = frame.positions_nm - np.mean(frame.positions_nm, axis=0)
        if reference is None:
            reference = centred
        elif not np.allclose(centred, reference, rtol=1e-10, atol=1e-10):
            varied = True
    return varied


def _measure(
    ensemble: Ensemble,
    backbone: Sequence[int] | None,
    *,
    fraction: float,
    first_frame: int,
    stride: int,
    pair_stride: int,
    factor_stride: int,
    r_max_nm: float,
    rdf_bins: int,
    q_max_per_nm: float,
    q_bins: int,
    heavy_atoms_only: bool,
    max_lag_fraction: float,
) -> tuple[StructuralWindow, StructureFactor | None]:
    values: dict[str, float | None] = {name: None for name in _PARAMETERS}
    valid = {name: False for name in _PARAMETERS}
    counts = {name: 0 for name in _PARAMETERS}
    curves: dict[str, dict[str, tuple[float, ...]]] = {}
    notes: list[str] = []

    def attempt(label: str, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except AnalysisError as error:
            notes.append(f"{label}: {error}")
            return None

    if backbone is not None:
        conformation = attempt(
            "chain dimensions",
            lambda: chain_conformation(ensemble, backbone, stride=stride),
        )
        if conformation is not None:
            for name in ("characteristic_ratio", "ratio_of_squares"):
                value = _finite(getattr(conformation.mean, name))
                values[name], valid[name] = value, value is not None and value > 0
                counts[name] = conformation.n_frames
        persistence = attempt(
            "persistence length",
            lambda: persistence_length(ensemble, backbone, stride=stride),
        )
        if persistence is not None:
            name = "persistence_length_nm"
            value = (
                _finite(persistence.persistence_length_nm)
                if persistence.decayed
                else None
            )
            values[name], valid[name] = value, value is not None and value > 0
            counts[name] = len(range(0, ensemble.n_frames, stride))
            curves["bond_correlation"] = _curve(
                separation_bonds=persistence.separation,
                correlation=persistence.correlation,
            )
            if not persistence.decayed:
                notes.append(
                    "Backbone correlation did not decay within the chain; persistence length remains censored."
                )
        relaxation = attempt(
            "end-to-end relaxation",
            lambda: end_to_end_relaxation(
                ensemble, backbone, max_lag_fraction=max_lag_fraction
            ),
        )
        if relaxation is not None:
            name = "end_to_end_relaxation_time_ps"
            value = (
                _finite(relaxation.relaxation_time_ps)
                if relaxation.decorrelated
                else None
            )
            values[name], valid[name] = value, value is not None and value > 0
            counts[name] = relaxation.n_origins
            curves["end_to_end_correlation"] = _curve(
                lag_ps=relaxation.lag_ps, correlation=relaxation.correlation
            )
            if not relaxation.decorrelated:
                notes.append(
                    "End-to-end correlation did not cross 1/e within observed lags; relaxation time remains censored."
                )
    else:
        notes.append(
            "No backbone supplied; chain dimensions, persistence and orientational relaxation are unavailable."
        )
    displacement = attempt(
        "COM diffusion",
        lambda: centre_of_mass_msd(
            ensemble, stride=stride, max_lag_fraction=max_lag_fraction
        ),
    )
    if displacement is not None:
        name = "diffusion_coefficient_cm2_s"
        value = (
            _finite(displacement.diffusion_coefficient_cm2_s)
            if displacement.diffusive
            else None
        )
        values[name], valid[name] = value, value is not None and value > 0
        counts[name] = displacement.n_origins
        curves["centre_of_mass_msd"] = _curve(
            lag_ps=displacement.lag_ps, msd_nm2=displacement.msd_nm2
        )
        if not displacement.diffusive:
            notes.append(
                f"COM MSD slope {displacement.log_slope:.3g} does not establish diffusion; coefficient remains censored."
            )
    distribution: RadialDistribution | None = attempt(
        "RDF",
        lambda: radial_distribution(
            ensemble,
            r_max_nm=r_max_nm,
            n_bins=rdf_bins,
            heavy_atoms_only=heavy_atoms_only,
            stride=pair_stride,
        ),
    )
    if distribution is not None:
        curves["radial_distribution"] = _curve(
            r_nm=distribution.r_nm,
            g_r=distribution.g_r,
            coordination_number=distribution.coordination_number,
        )
        for name, value in (
            ("rdf_first_peak_nm", distribution.first_peak_nm),
            ("rdf_first_peak_height", distribution.first_peak_height),
        ):
            finite = _finite(value)
            values[name], valid[name] = (
                finite,
                finite is not None and finite > 0 and distribution.n_pairs > 0,
            )
            counts[name] = distribution.n_frames
    factor: StructureFactor | None = attempt(
        "structure factor",
        lambda: structure_factor(
            ensemble,
            q_max_per_nm=q_max_per_nm,
            n_bins=q_bins,
            heavy_atoms_only=heavy_atoms_only,
            stride=factor_stride,
        ),
    )
    if factor is not None:
        curves["structure_factor"] = _curve(
            q_per_nm=factor.q_per_nm, s_q=factor.s_q, n_vectors=factor.n_vectors
        )
        for name in ("structure_factor_peak_per_nm", "structure_factor_peak_height"):
            counts[name] = factor.n_frames
    return StructuralWindow(
        fraction,
        first_frame,
        ensemble.n_frames,
        max(0, ensemble.n_frames - 1) * ensemble.interval_ps,
        values,
        valid,
        counts,
        curves,
        _configuration_varied(ensemble),
        tuple(notes),
    ), factor


def _relative_change(values: Sequence[float | None]) -> float | None:
    if any(value is None for value in values) or not values:
        return None
    array = np.asarray(values, dtype=float)
    scale = max(abs(float(array[-1])), float(np.mean(np.abs(array))))
    return float(np.ptp(array) / scale) if scale > 0 else 0.0


def _parameter(
    name: str,
    unit: str,
    stationary: bool,
    windows: Sequence[StructuralWindow],
    blocks: Sequence[StructuralWindow],
    tolerance: float,
    minimum: int,
) -> StructuralParameterConvergence:
    values = tuple(window.values[name] for window in windows)
    valid = tuple(window.valid[name] for window in windows)
    block_values = tuple(block.values[name] for block in blocks) if stationary else ()
    # Compare the final three prefixes, but retain earlier censored windows.
    selected = windows[-3:]
    change = _relative_change([window.values[name] for window in selected])
    block_change = _relative_change(block_values) if stationary else None
    notes = [
        "Overlapping prefix differences are stability estimates, not standard errors."
    ]
    if len({window.n_frames for window in selected}) < 3:
        notes.append("Fewer than three distinct observed prefix lengths.")
    if not all(window.valid[name] for window in selected):
        notes.append(
            "At least one of the last three windows is missing, censored or invalid."
        )
    if any(not window.configuration_varied for window in selected):
        notes.append(
            "A compared prefix shows no configurational variation; repeated frozen coordinates cannot establish temporal convergence."
        )
    if any(window.sample_counts[name] < minimum for window in selected):
        notes.append(
            f"Fewer than {minimum} sampled frames in a compared prefix; increase trajectory sampling or lift the frame cap."
        )
    if change is None or change > tolerance:
        notes.append(
            "The last three prefix estimates do not agree within relative_tolerance."
        )
    if stationary:
        if len(blocks) < 3 or any(not block.valid[name] for block in blocks):
            notes.append(
                "Three valid disjoint tail blocks are required for a stationary observable."
            )
        if any(block.sample_counts[name] < minimum for block in blocks):
            notes.append(
                f"Disjoint tail blocks need at least {minimum} sampled frames each."
            )
        if any(not block.configuration_varied for block in blocks):
            notes.append("A disjoint tail block shows no configurational variation.")
        if block_change is None or block_change > tolerance:
            notes.append(
                "Disjoint tail estimates do not agree within relative_tolerance."
            )
    return StructuralParameterConvergence(
        name,
        unit,
        values,
        valid,
        block_values,
        change,
        block_change,
        len(notes) == 1,
        tuple(notes),
    )


def structural_window_convergence(
    ensemble: Ensemble,
    backbone: Sequence[int] | None = None,
    *,
    window_fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    relative_tolerance: float = 0.1,
    min_frames: int = 20,
    stride: int = 1,
    max_distribution_frames: int | None = MAX_DISTRIBUTION_FRAMES,
    max_structure_factor_frames: int | None = MAX_STRUCTURE_FACTOR_FRAMES,
    r_max_nm: float | None = None,
    rdf_bins: int = 150,
    q_max_per_nm: float = 40.0,
    q_bins: int = 100,
    heavy_atoms_only: bool = True,
    max_lag_fraction: float = 0.5,
    min_vectors_per_bin: int = MIN_WAVEVECTORS_PER_BIN,
) -> StructuralWindowConvergence:
    """Refit trajectory prefixes using fixed bins, lags policy and sampling.

    Pair/S(q) strides are chosen once from the entire trajectory. Their
    default caps match ordinary structural reports (50/8 frames), which may
    be too sparse to resolve convergence. Set caps to ``None`` to use every
    ``stride``-th frame. The same RDF radius is legal for every observed box.
    S(q) peak comparisons use a common bin mask with adequate wavevectors per
    frame in every prefix and block. Missing modes stay missing.

    Stationary parameters also need agreement among three disjoint blocks
    spanning the final half of the trajectory. Dynamical parameters require
    their existing diffusion/decorrelation tests in each of the final three
    prefixes. These diagnostics do not measure statistical uncertainty or
    certify independently sampled configurations.
    """
    fractions = _fractions(window_fractions)
    relative_tolerance = require_positive(
        relative_tolerance, None, name="relative_tolerance"
    )
    min_frames = require_integer(min_frames, minimum=3, name="min_frames")
    stride = require_integer(stride, name="stride")
    require_integer(rdf_bins, name="rdf_bins")
    require_integer(q_bins, name="q_bins")
    require_integer(min_vectors_per_bin, name="min_vectors_per_bin")
    require_positive(q_max_per_nm, None, name="q_max_per_nm")
    if not math.isfinite(max_lag_fraction) or not 0 < max_lag_fraction <= 0.5:
        raise ValueError("max_lag_fraction must be in (0, 0.5].")
    if ensemble.n_frames < 1:
        raise AnalysisError("The ensemble contains no frames.")
    if not ensemble.is_snapshot and (
        not math.isfinite(ensemble.interval_ps) or ensemble.interval_ps <= 0
    ):
        raise AnalysisError(
            "A trajectory needs a finite positive recorded frame interval."
        )
    path = (
        None
        if backbone is None
        else tuple(
            int(index) for index in backbone_indices(backbone, ensemble.atoms_per_chain)
        )
    )
    boxes = boxes_nm(ensemble)
    if not np.all(np.isfinite(boxes)) or np.any(boxes <= 0):
        raise AnalysisError("Structural convergence needs finite positive box edges.")
    half = float(np.min(boxes)) / 2
    radius = (
        half if r_max_nm is None else require_positive(r_max_nm, None, name="r_max_nm")
    )
    if radius > half:
        raise AnalysisError("r_max_nm exceeds half the smallest recorded box edge.")
    pair_stride = _stride(ensemble.n_frames, stride, max_distribution_frames)
    factor_stride = _stride(ensemble.n_frames, stride, max_structure_factor_frames)
    options: dict[str, Any] = dict(
        stride=stride,
        pair_stride=pair_stride,
        factor_stride=factor_stride,
        r_max_nm=radius,
        rdf_bins=rdf_bins,
        q_max_per_nm=q_max_per_nm,
        q_bins=q_bins,
        heavy_atoms_only=heavy_atoms_only,
        max_lag_fraction=max_lag_fraction,
    )
    measured = [
        _measure(
            replace(ensemble, n_frames=max(1, int(ensemble.n_frames * fraction))),
            path,
            fraction=fraction,
            first_frame=0,
            **options,
        )
        for fraction in fractions
    ]
    boundaries = np.linspace(ensemble.n_frames // 2, ensemble.n_frames, 4, dtype=int)
    blocks = [
        _measure(
            _block(ensemble, int(start), int(stop)),
            path,
            fraction=float((stop - start) / ensemble.n_frames),
            first_frame=int(start),
            **options,
        )
        for start, stop in pairwise(boundaries)
        if stop > start
    ]
    # A fixed, conservative q floor and eligibility mask avoid changing which
    # reciprocal bins can win the peak simply because a prefix is longer.
    all_measured = measured + blocks
    factors = [factor for _, factor in all_measured]
    mask = np.ones(q_bins, dtype=bool)
    floor = 2 * math.pi / float(np.min(boxes))
    for factor in factors:
        if factor is None:
            mask[:] = False
        else:
            mask &= (factor.q_per_nm >= floor) & (
                factor.n_vectors / factor.n_frames >= min_vectors_per_bin
            )
    updated = []
    for window, factor in all_measured:
        if factor is not None and mask.any():
            index = int(np.argmax(np.where(mask, factor.s_q, -np.inf)))
            values, valid = dict(window.values), dict(window.valid)
            for name, value in (
                ("structure_factor_peak_per_nm", factor.q_per_nm[index]),
                ("structure_factor_peak_height", factor.s_q[index]),
            ):
                values[name] = _finite(float(value))
                valid[name] = values[name] is not None and float(value) > 0
            window = replace(window, values=values, valid=valid)
        else:
            window = replace(
                window,
                notes=(
                    *window.notes,
                    "No common adequately populated S(q) peak bins across all windows.",
                ),
            )
        updated.append(window)
    prefix_windows = tuple(updated[: len(measured)])
    block_windows = tuple(updated[len(measured) :])
    parameters = {
        name: _parameter(
            name,
            unit,
            stationary,
            prefix_windows,
            block_windows,
            relative_tolerance,
            min_frames,
        )
        for name, (unit, stationary) in _PARAMETERS.items()
    }
    notes = [
        "Window stability is not a zero-rate correction, independent-replica uncertainty or proof of equilibration.",
        "Prefixes overlap; reported differences are not standard errors. All curves retain the same bin grids and global strides.",
        "End-to-end relaxation uses every recorded frame; other structural and COM analyses use the recorded stride settings.",
    ]
    if ensemble.is_snapshot:
        notes.append(
            "A single snapshot cannot establish observation-window convergence."
        )
        parameters = {
            name: replace(
                parameter,
                resolved=False,
                notes=(*parameter.notes, "Only a snapshot is available."),
            )
            for name, parameter in parameters.items()
        }
    return StructuralWindowConvergence(
        ensemble.stage,
        prefix_windows,
        block_windows,
        parameters,
        relative_tolerance,
        min_frames,
        {
            **options,
            "min_vectors_per_bin_per_frame": min_vectors_per_bin,
            "common_q_floor_per_nm": floor,
            "common_q_bins": tuple(bool(value) for value in mask),
        },
        all(item.resolved for item in parameters.values()),
        tuple(notes),
    )
