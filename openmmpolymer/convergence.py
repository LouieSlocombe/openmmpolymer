"""Observation-window stability without mistaking short runs for equilibrium.

Stationary observables use autocorrelation-adjusted errors, expanding windows,
and disjoint tail blocks. Relaxation curves are instead refitted as decays:
their time dependence is physical and is not a stationary sampling trace.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .conformation import chain_conformation
from .protocols import RunManifest
from .relaxation import (
    SIGNAL_TO_NOISE_FLOOR,
    KWWFit,
    PronyFit,
    RelaxationCurve,
    _signal_window,
    fit_kww,
    fit_prony,
    relaxation_curve,
)
from .strain_rate import MAX_RATE_RESIDUAL_TO_ERROR, MAX_RELATIVE_RATE_RESIDUAL, _rms
from .structure import _resolve_backbone, _select_stage
from .timeseries import _statistical_inefficiency, read_state_data
from .trajectory import AnalysisError, open_run, stage_files

if TYPE_CHECKING:
    from .structural_convergence import StructuralWindowConvergence

DEFAULT_WINDOW_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class WindowEstimate:
    """Mean within one prefix after its stated initial fraction is discarded."""

    fraction: float
    duration_ps: float
    n_samples: int
    n_effective: float
    mean: float
    standard_error: float
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class WindowConvergence:
    """Stability evidence for one observable, not proof of equilibration.

    Prefix windows overlap and are not independent replicas. Independent
    disjoint blocks of the retained trajectory's latter half provide an
    additional check. Effective sample counts account for autocorrelation
    within each window and each block, not just their raw frame counts.
    """

    property_name: str
    value_unit: str
    windows: tuple[WindowEstimate, ...]
    block_means: npt.NDArray[np.float64]
    block_standard_errors: npt.NDArray[np.float64]
    block_effective_samples: npt.NDArray[np.float64]
    relative_change: float
    relative_block_spread: float
    relative_tolerance: float
    min_effective_samples: float
    discard_fraction: float
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ParameterConvergence:
    """Stability across overlapping model refits, with no fabricated SE."""

    property_name: str
    value_unit: str
    values: npt.NDArray[np.float64]
    relative_change: float
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RelaxationWindowEstimate:
    fraction: float
    duration_ps: float
    kww: KWWFit
    prony: PronyFit
    tail_decayed: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelaxationWindowConvergence:
    """Refits of successively longer G(t), retaining censored decay tails."""

    windows: tuple[RelaxationWindowEstimate, ...]
    metrics: dict[str, ParameterConvergence]
    relative_tolerance: float
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ConvergenceReport:
    run_dir: str
    stage: str
    results: dict[str, WindowConvergence]
    relaxation: RelaxationWindowConvergence | None
    notes: tuple[str, ...]
    structural: StructuralWindowConvergence | None = None


def _window_options(
    fractions: Sequence[float], tolerance: float, minimum: float, discard: float
) -> tuple[float, ...]:
    windows = tuple(float(value) for value in fractions)
    if (
        len(windows) < 3
        or windows[-1] != 1.0
        or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in windows)
        or any(a >= b for a, b in pairwise(windows))
    ):
        raise ValueError(
            "window_fractions must contain at least three increasing fractions in (0, 1], ending at 1."
        )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    if not math.isfinite(minimum) or minimum < 1.0:
        raise ValueError("min_effective_samples must be finite and at least one.")
    if not math.isfinite(discard) or not 0.0 <= discard < 1.0:
        raise ValueError("discard_fraction must be finite and in [0, 1).")
    return windows


def _series(
    time_ps: npt.ArrayLike, values: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    times, data = (
        np.asarray(time_ps, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )
    if times.ndim != 1 or data.ndim != 1 or times.size != data.size or not times.size:
        raise AnalysisError(
            "Times and values must be nonempty one-dimensional arrays of equal length."
        )
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(data)):
        raise AnalysisError(
            "Times and values must be finite; missing data cannot be silently dropped."
        )
    if np.any(np.diff(times) <= 0.0):
        raise AnalysisError("Times must be strictly increasing.")
    return times, data


def _sample_estimate(data: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    if not data.size:
        return math.nan, math.nan, 0.0
    scale = float(np.max(np.abs(data))) or 1.0
    normal = data / scale
    mean = float(np.mean(normal)) * scale
    if data.size < 3 or float(np.ptp(normal)) <= 64.0 * np.finfo(float).eps:
        return mean, math.nan, 0.0
    # Normalise by spread so _statistical_inefficiency's absolute tiny-value
    # threshold cannot make unit conversion remove genuine correlations.
    centred = normal - np.mean(normal)
    deviation = float(np.std(centred))
    effective = float(data.size) / _statistical_inefficiency(centred / deviation)
    error = float(np.std(normal, ddof=1)) * scale / math.sqrt(effective)
    return mean, error, effective


def _relative_span(values: npt.NDArray[np.float64], scale: float) -> float:
    if not values.size or np.any(~np.isfinite(values)):
        return math.inf
    if scale == 0.0:
        return 0.0 if np.all(values == 0.0) else math.inf
    return float(np.ptp(values / scale))


def time_window_convergence(
    time_ps: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    property_name: str,
    value_unit: str,
    window_fractions: Sequence[float] = DEFAULT_WINDOW_FRACTIONS,
    relative_tolerance: float = 0.1,
    min_effective_samples: float = 20.0,
    discard_fraction: float = 0.1,
) -> WindowConvergence:
    """Compare expanding prefix means and three disjoint late-time blocks.

    The last three prefix estimates and every tail block must have enough
    effective samples and errors below the requested relative tolerance.
    Their mean spreads must independently satisfy that tolerance. Disjoint
    block means must also agree within three combined standard errors, so an
    arbitrary additive energy zero cannot conceal statistically visible drift.
    At least three distinct sampled prefix endpoints are required. A constant
    trace supplies no evidence of thermal sampling and remains unresolved.
    Uneven sampling is flagged because the autocorrelation estimate assumes
    equal time steps. Resolution describes this observable on sampled windows,
    never equilibrium of unmeasured or slower degrees of freedom.
    """
    fractions = _window_options(
        window_fractions, relative_tolerance, min_effective_samples, discard_fraction
    )
    times, data = _series(time_ps, values)
    scale = max(abs(float(np.mean(data))), float(np.std(data)))
    span = float(times[-1] - times[0])
    windows: list[WindowEstimate] = []
    for fraction in fractions:
        stop = int(np.searchsorted(times, times[0] + fraction * span, side="right"))
        start = int(stop * discard_fraction)
        mean, error, effective = _sample_estimate(data[start:stop])
        notes: list[str] = []
        if stop - start < 3:
            notes.append("Too few samples in this window.")
        if not math.isfinite(error):
            notes.append(
                "Uncertainty unavailable; constant or insufficient data give no thermal sampling evidence."
            )
        if effective < min_effective_samples:
            notes.append("Too few effective independent samples.")
        if math.isfinite(error) and (
            scale == 0.0 or error > relative_tolerance * scale
        ):
            notes.append("Mean uncertainty exceeds the requested relative tolerance.")
        windows.append(
            WindowEstimate(
                fraction,
                float(times[stop - 1] - times[0]),
                stop - start,
                effective,
                mean,
                error,
                not notes,
                tuple(notes),
            )
        )
    retained = data[int(data.size * discard_fraction) :]
    blocks = np.array_split(retained[retained.size // 2 :], 3)
    estimates = [_sample_estimate(block) for block in blocks]
    block_means = np.asarray([item[0] for item in estimates])
    block_errors = np.asarray([item[1] for item in estimates])
    block_effective = np.asarray([item[2] for item in estimates])
    relative_change = _relative_span(
        np.asarray([item.mean for item in windows[-3:]]), scale
    )
    block_spread = _relative_span(block_means, scale)
    notes = [
        "Observable window stability only; overlapping prefixes are not independent replicas or proof of equilibrium."
    ]
    refusals: list[str] = []
    if data.size < 3:
        refusals.append(
            "A snapshot or fewer than three time samples cannot establish convergence."
        )
    if times.size > 2 and not np.allclose(
        np.diff(times), np.median(np.diff(times)), rtol=1e-5, atol=1e-9
    ):
        refusals.append(
            "Uneven time spacing prevents a reliable sample-autocorrelation estimate."
        )
    if any(not window.resolved for window in windows[-3:]):
        refusals.append(
            "At least one of the last three windows has insufficient sampling or uncertainty."
        )
    if len({window.duration_ps for window in windows[-3:]}) < 3:
        refusals.append(
            "The last three prefixes do not contain three distinct sampled endpoints."
        )
    if np.any(block_effective < min_effective_samples) or np.any(
        ~np.isfinite(block_errors)
    ):
        refusals.append(
            "Disjoint tail blocks have insufficient effective samples or unknown uncertainty."
        )
    elif scale == 0.0 or np.any(block_errors > relative_tolerance * scale):
        refusals.append(
            "Disjoint tail-block mean uncertainty exceeds the relative tolerance."
        )
    if np.all(np.isfinite(block_errors)) and any(
        abs(block_means[left] - block_means[right])
        > 3.0 * math.hypot(float(block_errors[left]), float(block_errors[right]))
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        refusals.append(
            "Disjoint tail-block means disagree beyond 3 combined standard errors; "
            "this drift guard is independent of the observable's additive zero."
        )
    if relative_change > relative_tolerance:
        refusals.append(
            "The last three prefix means have not stabilised within the relative tolerance."
        )
    if block_spread > relative_tolerance:
        refusals.append(
            "Disjoint tail-block means have not stabilised within the relative tolerance."
        )
    if float(np.ptp(data)) == 0.0:
        refusals.append(
            "Constant data provide no thermal sampling evidence; uncertainty is unknown."
        )
    return WindowConvergence(
        property_name,
        value_unit,
        tuple(windows),
        block_means,
        block_errors,
        block_effective,
        relative_change,
        block_spread,
        relative_tolerance,
        min_effective_samples,
        discard_fraction,
        not refusals,
        tuple([*notes, *refusals]),
    )


def relaxation_window_convergence(
    curve: RelaxationCurve,
    *,
    window_fractions: Sequence[float] = DEFAULT_WINDOW_FRACTIONS,
    relative_tolerance: float = 0.1,
) -> RelaxationWindowConvergence:
    """Refit each elapsed-time prefix and test parameter stability and decay.

    Relaxation is not a stationary mean. The last three prefixes must each
    resolve the relevant model and observe its decay/plateau. Model refits
    overlap, so their spread is sensitivity to observation time, not a
    standard error. Prony fits additionally must satisfy the shared response
    residual guards, and use four spectral terms per decade to reduce grid
    sensitivity. Liquid viscosities integrate the fitted relaxation and are
    refused when an independently resolved nonzero equilibrium modulus remains.
    """
    fractions = _window_options(window_fractions, relative_tolerance, 1.0, 0.0)
    times, values = _series(curve.time_ps, curve.modulus_mpa)
    if np.any(times <= 0.0):
        raise AnalysisError("Relaxation bins must have positive times.")
    if any(
        array.shape != times.shape
        for array in (curve.bin_index, curve.standard_error_mpa, curve.n_samples)
    ):
        raise AnalysisError("Relaxation arrays must have matching shapes.")
    if np.any(~np.isfinite(curve.standard_error_mpa)) or np.any(
        curve.standard_error_mpa < 0.0
    ):
        raise AnalysisError(
            "Relaxation standard errors must be finite and nonnegative."
        )
    windows: list[RelaxationWindowEstimate] = []
    records: dict[str, list[float]] = {
        name: []
        for name in (
            "equilibrium_modulus_mpa",
            "kww_mean_tau_ps",
            "kww_viscosity_pa_s",
            "prony_viscosity_pa_s",
        )
    }
    validity: dict[str, list[bool]] = {name: [] for name in records}
    for fraction in fractions:
        stop = max(1, int(np.searchsorted(times, fraction * times[-1], side="right")))
        prefix = replace(
            curve,
            time_ps=times[:stop],
            modulus_mpa=values[:stop],
            bin_index=curve.bin_index[:stop],
            standard_error_mpa=curve.standard_error_mpa[:stop],
            n_samples=curve.n_samples[:stop],
        )
        # A coarse one-per-decade spectrum changes its discretisation error
        # noticeably when a prefix moves its upper endpoint. Four terms per
        # decade keep that numerical effect below the default stability guard.
        kww, prony = fit_kww(prefix), fit_prony(prefix, per_decade=4)
        fit_window = _signal_window(
            prefix.modulus_mpa, prefix.standard_error_mpa, SIGNAL_TO_NOISE_FLOOR
        )
        fitted_values = prefix.modulus_mpa[fit_window]
        fitted_errors = prefix.standard_error_mpa[fit_window]
        residual_ok = False
        window_notes: list[str] = []
        if fitted_values.size and math.isfinite(prony.residual_mpa):
            response_scale = float(np.mean(np.abs(fitted_values)))
            reported_error = _rms(fitted_errors)
            rounding = 64 * np.finfo(float).eps * float(np.max(np.abs(fitted_values)))
            residual_ok = bool(
                prony.residual_mpa <= MAX_RELATIVE_RATE_RESIDUAL * response_scale
                and (
                    reported_error == 0.0
                    or prony.residual_mpa
                    <= max(MAX_RATE_RESIDUAL_TO_ERROR * reported_error, rounding)
                )
            )
        if not residual_ok:
            window_notes.append(
                "Prony residual exceeds the 10% response-scale or 3-times-reported-error guard, or is unavailable."
            )
        initial = abs(float(values[0]))
        tail = float(np.median(values[max(0, stop - 3) : stop]))
        amplitude = initial - prony.equilibrium_mpa
        plateau = bool(
            prony.resolved
            and residual_ok
            and prony.plateau_reached
            and amplitude > 0.0
            and abs(tail - prony.equilibrium_mpa) <= 0.1 * amplitude
        )
        zero_tail = bool(initial > 0.0 and abs(tail) <= 0.1 * initial)
        # A plateau smaller than the known baseline floor is unresolved from
        # zero; otherwise even a small positive plateau makes liquid viscosity
        # mathematically divergent. Numerical NNLS residue gets a tiny allowance.
        zero_floor = max(
            curve.noise_floor_mpa if math.isfinite(curve.noise_floor_mpa) else 0.0,
            initial * 1e-8,
        )
        liquid = bool(plateau and prony.equilibrium_mpa <= zero_floor and zero_tail)
        plateau_conflict = bool(plateau and prony.equilibrium_mpa > zero_floor)
        kww_ok = bool(kww.resolved and zero_tail and not plateau_conflict)
        windows.append(
            RelaxationWindowEstimate(
                fraction,
                float(times[stop - 1]),
                kww,
                prony,
                plateau,
                tuple(window_notes),
            )
        )
        records["equilibrium_modulus_mpa"].append(prony.equilibrium_mpa)
        validity["equilibrium_modulus_mpa"].append(plateau)
        records["kww_mean_tau_ps"].append(kww.mean_tau_ps)
        validity["kww_mean_tau_ps"].append(kww_ok)
        records["kww_viscosity_pa_s"].append(kww.modulus_mpa * kww.mean_tau_ps * 1e-6)
        validity["kww_viscosity_pa_s"].append(kww_ok)
        records["prony_viscosity_pa_s"].append(
            float(prony.weights_mpa @ prony.tau_ps) * 1e-6
        )
        validity["prony_viscosity_pa_s"].append(liquid)
    metrics: dict[str, ParameterConvergence] = {}
    distinct_windows = len({window.duration_ps for window in windows[-3:]}) >= 3
    for name, measured in records.items():
        array = np.asarray(measured, dtype=np.float64)
        final = array[-3:]
        # Equilibrium G can be zero; scale its absolute stability to measured
        # initial G rather than dividing by a physically correct zero plateau.
        scale = (
            abs(float(values[0]))
            if name == "equilibrium_modulus_mpa"
            else abs(float(final[-1]))
        )
        change = _relative_span(final, scale)
        valid = (
            distinct_windows
            and all(validity[name][-3:])
            and bool(np.all(np.isfinite(final)))
        )
        notes = [
            "Overlapping refits measure window sensitivity, not statistical uncertainty."
        ]
        if not distinct_windows:
            notes.append(
                "The last three prefixes do not contain three distinct sampled endpoints."
            )
        if not valid:
            notes.append(
                "At least one of the last three windows has an unresolved fit, unobserved decay/plateau, or non-liquid tail."
            )
        if change > relative_tolerance:
            notes.append(
                "The last three parameter estimates have not stabilised within the relative tolerance."
            )
        unit = (
            "MPa"
            if name == "equilibrium_modulus_mpa"
            else ("ps" if name == "kww_mean_tau_ps" else "Pa s")
        )
        metrics[name] = ParameterConvergence(
            name,
            unit,
            array,
            change,
            valid and change <= relative_tolerance,
            tuple(notes),
        )
    notes = [
        "Finite observation-window stability does not establish an equilibrium limit or an unsampled slow relaxation mode."
    ]
    if curve.n_replicas < 2:
        notes.append(
            "One relaxation replica: within-run bin errors are not uncertainty across independent preparations."
        )
    return RelaxationWindowConvergence(
        tuple(windows),
        metrics,
        relative_tolerance,
        all(item.resolved for item in metrics.values()),
        tuple(notes),
    )


def analyse_convergence(
    run_dir: str | Path,
    *,
    stage: str | None = None,
    backbone: Sequence[int] | None = None,
    stride: int = 1,
    window_fractions: Sequence[float] = DEFAULT_WINDOW_FRACTIONS,
    relative_tolerance: float = 0.1,
    min_effective_samples: float = 20.0,
    discard_fraction: float = 0.1,
) -> ConvergenceReport:
    """Read time-window diagnostics from a saved run without running dynamics.

    State traces provide density, temperature and potential energy; a saved
    trajectory and known/inferred backbone add radius of gyration and squared
    end-to-end distance. Relaxation logs are analysed as decays separately.
    Missing data and snapshots stay explicit in report notes.
    """
    fractions = _window_options(
        window_fractions, relative_tolerance, min_effective_samples, discard_fraction
    )
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    directory = Path(run_dir)
    try:
        files, _ = _select_stage(directory, stage)
    except AnalysisError:
        # CSV-only and relaxation-only runs can be useful without coordinates.
        files = stage_files(directory, stage)
    notes: list[str] = []
    results: dict[str, WindowConvergence] = {}

    def measure(
        times: npt.ArrayLike, values: npt.ArrayLike, name: str, unit: str
    ) -> None:
        try:
            results[name] = time_window_convergence(
                times,
                values,
                property_name=name,
                value_unit=unit,
                window_fractions=fractions,
                relative_tolerance=relative_tolerance,
                min_effective_samples=min_effective_samples,
                discard_fraction=discard_fraction,
            )
        except AnalysisError as exc:
            notes.append(f"{name} unavailable: {exc}")

    if files.csv:
        try:
            state = read_state_data(files.csv, stage=files.stage)
        except AnalysisError as exc:
            notes.append(f"State-data convergence unavailable: {exc}")
        else:
            for name, unit in (
                ("density_g_cm3", "g/cm^3"),
                ("temperature_k", "K"),
                ("potential_energy_kj_mol", "kJ/mol"),
            ):
                measure(state.time_ps, getattr(state, name), name, unit)
    else:
        notes.append("No state-data CSV is available for the selected stage.")
    structural = None
    try:
        ensemble = open_run(directory, files.stage)
        path, _, _ = _resolve_backbone(
            directory, RunManifest.load(directory), ensemble, backbone, True, notes
        )
        if ensemble.is_snapshot:
            notes.append(
                "Only a snapshot is available; structural observation-window convergence is unresolved."
            )
        if path is not None:
            series = chain_conformation(ensemble, path, stride=stride)
            measure(
                series.time_ps,
                series.mean_radius_of_gyration_nm,
                "mean_radius_of_gyration_nm",
                "nm",
            )
            measure(
                series.time_ps,
                series.mean_squared_end_to_end_nm2,
                "mean_squared_end_to_end_nm2",
                "nm^2",
            )
        from .structural_convergence import structural_window_convergence

        structural = structural_window_convergence(
            ensemble,
            backbone=path,
            window_fractions=fractions,
            relative_tolerance=relative_tolerance,
            stride=stride,
        )
    except AnalysisError as exc:
        notes.append(f"Structural convergence unavailable: {exc}")
    relaxation = None
    try:
        source_refusals: list[str] = []
        if stage is None:
            from .viscoelastic import analyse_relaxation

            source = analyse_relaxation(directory)
            notes.extend(source.notes)
            if source.mean is None:
                raise AnalysisError("No independent relaxation ensemble could be read.")
            curve = source.mean
            if source.linearity is not None and not source.linearity.linear:
                source_refusals.append(
                    "The measured relaxation failed its strain-linearity check."
                )
            if any(note.startswith("Skipped ") for note in source.notes):
                source_refusals.append(
                    "At least one recorded relaxation replica could not be analysed."
                )
        else:
            curve = relaxation_curve(directory, stage=stage)
        relaxation = relaxation_window_convergence(
            curve, window_fractions=fractions, relative_tolerance=relative_tolerance
        )
        if source_refusals:
            relaxation = replace(
                relaxation,
                metrics={
                    name: replace(
                        metric, resolved=False, notes=(*metric.notes, *source_refusals)
                    )
                    for name, metric in relaxation.metrics.items()
                },
                resolved=False,
                notes=(*relaxation.notes, *source_refusals),
            )
    except AnalysisError as exc:
        notes.append(f"Relaxation convergence unavailable: {exc}")
    return ConvergenceReport(
        str(directory), files.stage, results, relaxation, tuple(notes), structural
    )
