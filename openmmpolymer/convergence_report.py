"""Observation-window convergence read off a saved run, and reported.

:func:`analyse_convergence` runs every window analysis a run directory has the
data for - the stationary traces, the structural measurements and the
relaxation refits - without running any dynamics, and keeps what it could not
measure as notes. :func:`write_convergence_report` writes the lot as strict
JSON, with figures that retain every refusal and unknown error.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy.typing as npt

from ._files import ReportFiles, json_value, write_json
from ._validation import require_integer
from .conformation import chain_conformation
from .convergence import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_RELATIVE_TOLERANCE,
    DEFAULT_WINDOW_FRACTIONS,
    SNAPSHOT_REFUSAL,
    RelaxationWindowConvergence,
    WindowConvergence,
    relaxation_window_convergence,
    require_window_options,
    time_window_convergence,
)
from .plots import (
    plot_relaxation_convergence,
    plot_structural_convergence,
    plot_window_convergence,
)
from .relaxation import relaxation_curve
from .structural_convergence import (
    StructuralWindowConvergence,
    structural_window_convergence,
)
from .structure import resolve_backbone, select_stage
from .timeseries import read_state_data
from .trajectory import AnalysisError, load_manifest, open_run, stage_files
from .viscoelastic import analyse_relaxation


@dataclass(frozen=True)
class ConvergenceReport:
    """Everything a saved run says about the stability of its windows.

    Args:
        run_dir: The directory read.
        stage: The stage whose state data and coordinates were measured.
        results: One stationary-window analysis per observable, by name.
        relaxation: The relaxation refits, or None when there were none.
        notes: What could not be measured, and why.
        structural: The structural windows, or None when there were no
            coordinates to measure.
    """

    run_dir: str
    stage: str
    results: dict[str, WindowConvergence]
    relaxation: RelaxationWindowConvergence | None
    notes: tuple[str, ...]
    structural: StructuralWindowConvergence | None = None


def analyse_convergence(
    run_dir: str | Path,
    *,
    stage: str | None = None,
    backbone: Sequence[int] | None = None,
    stride: int = 1,
    window_fractions: Sequence[float] = DEFAULT_WINDOW_FRACTIONS,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    min_effective_samples: float = float(DEFAULT_MIN_SAMPLES),
    discard_fraction: float = 0.1,
) -> ConvergenceReport:
    """Read time-window diagnostics from a saved run without running dynamics.

    State traces provide density, temperature and potential energy; a saved
    trajectory and known/inferred backbone add radius of gyration and squared
    end-to-end distance. Relaxation logs are analysed as decays separately.
    Missing data and snapshots stay explicit in report notes.

    Raises:
        ValueError: An option is out of range.
        TypeError: *stride* is not an integer.
        AnalysisError: There is no manifest, or no such stage in it.
    """
    fractions = require_window_options(
        window_fractions, relative_tolerance, min_effective_samples, discard_fraction
    )
    require_integer(stride, name="stride")
    directory = Path(run_dir)
    try:
        files, _ = select_stage(directory, stage)
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
        path, _, _ = resolve_backbone(
            directory, load_manifest(directory), ensemble, backbone, True, notes
        )
        if ensemble.is_snapshot:
            notes.append(SNAPSHOT_REFUSAL)
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
        run_dir=str(directory),
        stage=files.stage,
        results=results,
        relaxation=relaxation,
        notes=tuple(notes),
        structural=structural,
    )


def write_convergence_report(
    report: ConvergenceReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write strict ``convergence.json`` and figures for available diagnostics.

    Args:
        report: What :func:`analyse_convergence` found.
        output_dir: Where to write, defaulting to ``<run_dir>/analysis``.
        figures: Write figures as well as the record.
        figure_format: What matplotlib should save them as.

    Returns:
        Where everything went.

    Raises:
        ValueError: *figure_format* is not a plain filename extension.
    """
    if not re.fullmatch(r"[A-Za-z0-9]+", figure_format):
        raise ValueError(
            "figure_format must be a filename extension such as png or svg."
        )
    directory = (
        Path(report.run_dir) / "analysis" if output_dir is None else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = asdict(report)
    record["interpretation"] = (
        "Window stability is conditional on recorded observables and sampled times. "
        "Overlapping prefixes are not independent replicas; stationary traces also "
        "require disjoint-block stability and autocorrelation-adjusted sample counts. "
        "Relaxation parameters require observed decay/plateau and resolved underlying "
        "models. This does not establish equilibrium or eliminate systematic bias."
    )
    path = directory / "convergence.json"
    write_json(path, json_value(record))
    written: list[str] = []
    if figures:
        drawn: dict[str, Any] = {}
        for name, result in report.results.items():
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
            drawn[f"convergence_{safe}"] = plot_window_convergence(result)
        if report.relaxation is not None:
            drawn["convergence_relaxation"] = plot_relaxation_convergence(
                report.relaxation
            )
        if report.structural is not None:
            drawn["convergence_structural"] = plot_structural_convergence(
                report.structural
            )
        for stem, figure in drawn.items():
            figure_path = directory / f"{stem}.{figure_format}"
            figure.savefig(figure_path, bbox_inches="tight")
            written.append(str(figure_path))
    return ReportFiles(json=str(path), figures=tuple(written))
