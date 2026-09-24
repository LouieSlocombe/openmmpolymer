"""Compare Young's modulus at several strain rates from one relaxed cell.

Only the hold after each strain increment changes between rates. The fits
remain finite-rate estimates; neither extrapolation claims a static modulus.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._files import write_json
from ._validation import require_positive
from ._workflow import (
    equilibrated_box_nm,
    sample_spread,
    settled_state,
    validate_extrapolation_limit,
    validate_hold_times,
)
from .elasticity import ElasticModulus, StressStrain, youngs_modulus
from .mechanical import (
    DEFAULT_SPEC,
    MAX_REPLICA_SPREAD,
    MechanicalError,
    ModulusSchedule,
    ModulusSpec,
    _pool,
    analyse_mechanics,
    deform_protocol,
    deform_schedule,
    equilibration_protocol,
)
from .protocols import Protocol, run_protocol, validate_run_inputs
from .simulate import RunContext, safe_timestep_fs
from .strain_rate import StrainRateExtrapolation, strain_rate_extrapolation
from .trajectory import AnalysisError

log = logging.getLogger(__name__)

WORKFLOW_NAME = "modulus_rate_workflow.json"


@dataclass(frozen=True)
class ModulusRateReport:
    """One pooled modulus per rate, followed by two empirical rate models."""

    run_dirs: tuple[str, ...]
    fits: tuple[ElasticModulus, ...]
    log_linear: StrainRateExtrapolation
    power_law: StrainRateExtrapolation
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ModulusRatePlan:
    """The common equilibration and every extension included in the budget."""

    equilibration: Protocol
    schedules: tuple[ModulusSchedule, ...]
    n_replicas: int

    @property
    def total_ns(self) -> float:
        """Total dynamics, counting every replica at every rate."""
        return (
            self.equilibration.total_duration_ps
            + self.n_replicas * sum(item.total_ps for item in self.schedules)
        ) / 1000.0


def _analysis_options(
    target_rate_per_ns: float, max_extrapolation_decades: float
) -> None:
    require_positive(target_rate_per_ns, None, name="target_rate_per_ns")
    validate_extrapolation_limit(max_extrapolation_decades)


def validate_modulus_rate_scan(
    spec: ModulusSpec,
    relax_ps: Sequence[float],
    *,
    target_rate_per_ns: float,
    max_extrapolation_decades: float = 2.0,
    **equilibration: Any,
) -> ModulusRatePlan:
    """Validate a rate series and budget without creating a run directory.

    At least three distinct positive holds are needed. Only extension passes
    are planned; load, bulk and shear settings in ``spec`` are not run.
    """
    _analysis_options(target_rate_per_ns, max_extrapolation_decades)
    holds = validate_hold_times(relax_ps, name="relax_ps")
    schedules = tuple(deform_schedule(replace(spec, relax_ps=value)) for value in holds)
    plan = ModulusRatePlan(
        equilibration=equilibration_protocol(spec, **equilibration),
        schedules=schedules,
        n_replicas=spec.n_replicas,
    )
    if spec.max_total_ns is not None and plan.total_ns > spec.max_total_ns:
        raise MechanicalError(
            f"The modulus rate scan is {plan.total_ns:.3g} ns across all rates and "
            f"replicas, over the {spec.max_total_ns:.3g} ns budget. "
            "Shorten holds, reduce replicas, or raise max_total_ns."
        )
    return plan


def run_modulus_rate_scan(
    run: RunContext,
    output_dir: str | Path = "run",
    *,
    relax_ps: Sequence[float],
    target_rate_per_ns: float,
    spec: ModulusSpec = DEFAULT_SPEC,
    max_extrapolation_decades: float = 2.0,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> ModulusRateReport:
    """Equilibrate once and branch every rate and replica from that state.

    ``relax_ps`` overrides the single hold in ``spec``. Temperature, axis,
    strain increments and fitting window are shared. Each replica redraws
    velocities; later chunks continue that replica. The scan measures only
    extension, skipping the unrelated load, bulk and shear passes.

    A workflow record is written before dynamics, including the complete
    equilibration request. Resuming with changed settings is refused even
    when the preceding attempt stopped during equilibration.
    """
    plan = validate_modulus_rate_scan(
        spec,
        relax_ps,
        target_rate_per_ns=target_rate_per_ns,
        max_extrapolation_decades=max_extrapolation_decades,
        **equilibration,
    )
    directory = Path(output_dir).resolve()
    relative_dirs = [f"rate_{index:02d}" for index in range(len(plan.schedules))]
    request = json.loads(
        json.dumps(
            {
                "spec": asdict(spec),
                "relax_ps": [item.relax_ps for item in plan.schedules],
                "equilibration": {
                    "protocol": asdict(plan.equilibration),
                },
                "target_rate_per_ns": target_rate_per_ns,
                "max_extrapolation_decades": max_extrapolation_decades,
                "system": asdict(run.spec),
                "seed": run.seed,
            },
            allow_nan=False,
            default=str,
        )
    )
    workflow = directory / WORKFLOW_NAME
    if workflow.is_file():
        previous = json.loads(workflow.read_text())
        if previous.get("request") != request:
            raise MechanicalError(
                f"{workflow} records different settings or rate holds. "
                "Use a fresh directory or restore the original settings."
            )
    elif (directory / "equilibration" / "manifest.json").exists() or any(
        (directory / name / "manifest.json").exists() for name in relative_dirs
    ):
        raise MechanicalError(
            f"{directory} already contains runs without a rate workflow record; "
            "their settings cannot be verified. Use a fresh directory."
        )
    if resume:
        validate_run_inputs(run, directory / "equilibration")
    directory.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"request": request, "run_dirs": relative_dirs}
    write_json(workflow, record)
    log.info(
        "Modulus rate scan: %d rates, %d replicas per rate, %.3g ns in total.",
        len(plan.schedules),
        spec.n_replicas,
        plan.total_ns,
    )
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    settled_dir = directory / "equilibration"
    settled = run_protocol(
        plan.equilibration, run, settled_dir, resume=resume, **chains
    )
    start_state = settled_state(
        settled, settled_dir, error=MechanicalError, verb="deform"
    )
    origin = equilibrated_box_nm(start_state)
    timestep_fs = safe_timestep_fs(spec.temperature_k, run.spec)
    record.update(
        start_state=str(start_state),
        reference_box_nm=origin,
        timestep_fs=timestep_fs,
    )
    write_json(workflow, record)
    for rate_index, (name, schedule) in enumerate(
        zip(relative_dirs, plan.schedules, strict=True)
    ):
        rate_spec = replace(spec, relax_ps=schedule.relax_ps)
        for replica in range(spec.n_replicas):
            run_protocol(
                deform_protocol(
                    rate_spec,
                    timestep_fs=timestep_fs,
                    replica=rate_index * spec.n_replicas + replica,
                    reference_box_nm=origin,
                ),
                run,
                directory / name,
                # A forced rerun resets the rate's manifest once. Subsequent
                # replicas must retain the newly completed earlier replicas.
                resume=resume or replica > 0,
                state_in=start_state,
                **chains,
            )
    return analyse_modulus_rates(
        [directory],
        target_rate_per_ns=target_rate_per_ns,
        strain_limit=spec.elastic_strain_limit,
        max_extrapolation_decades=max_extrapolation_decades,
    )


def _expand_run_dirs(run_dirs: Sequence[str | Path]) -> tuple[Path, ...]:
    directories: list[Path] = []
    for value in run_dirs:
        directory = Path(value).resolve()
        workflow = directory / WORKFLOW_NAME
        if workflow.is_file():
            record = json.loads(workflow.read_text())
            names = record.get("run_dirs")
            if not isinstance(names, list) or not names:
                raise AnalysisError(f"{workflow} records no rate run directories.")
            candidates = [directory / str(name) for name in names]
        else:
            candidates = [directory]
        for candidate in candidates:
            candidate = candidate.resolve()
            if not (candidate / "manifest.json").is_file():
                raise AnalysisError(f"No completed rate manifest in {candidate}.")
            if candidate in directories:
                raise AnalysisError(f"{candidate} was supplied more than once.")
            directories.append(candidate)
        if workflow.is_file():
            _check_recorded_scan(workflow, record, candidates)
    if not directories:
        raise AnalysisError(
            "Supply run directories containing strain-rate measurements."
        )
    return tuple(directories)


def _check_recorded_scan(
    workflow: Path, record: dict[str, Any], directories: Sequence[Path]
) -> None:
    """A partial rate series cannot become a complete analysis by accident."""
    request = record.get("request", {})
    if "spec" not in request:
        return
    spec = ModulusSpec(**request["spec"])
    holds = request.get("relax_ps", [])
    if len(holds) != len(directories):
        raise AnalysisError(
            f"{workflow} has inconsistent rate directory and hold counts."
        )
    for rate_index, (directory, hold) in enumerate(
        zip(directories, holds, strict=True)
    ):
        manifest = json.loads((directory / "manifest.json").read_text())
        entries = manifest.get("stages", {})
        rate_spec = replace(spec, relax_ps=float(hold))
        for replica in range(spec.n_replicas):
            protocol = deform_protocol(
                rate_spec,
                timestep_fs=2.0,
                replica=rate_index * spec.n_replicas + replica,
            )
            for stage in protocol.stages:
                entry = entries.get(stage.name)
                if entry is None:
                    raise AnalysisError(
                        f"{directory} is incomplete: expected replica stage {stage.name}."
                    )
                samples = entry.get("samples", {})
                recorded = np.asarray(samples.get("segment_strain", []), dtype=float)
                start = float(stage.options["strain_start"])
                count = int(stage.options["n_steps"])
                expected = np.asarray(
                    [
                        (1 + start) * (1 + spec.strain_increment) ** (index + 1) - 1
                        for index in range(count)
                    ]
                )
                if recorded.shape != expected.shape or not np.allclose(
                    recorded, expected, rtol=1.0e-6, atol=1.0e-9
                ):
                    raise AnalysisError(
                        f"{directory}/{stage.name} is incomplete or has a different strain ladder."
                    )
                if samples.get("deform_axis", [None])[0] != spec.axis:
                    raise AnalysisError(
                        f"{directory}/{stage.name} has a different deformation axis."
                    )
                durations = np.asarray(
                    samples.get("segment_duration_ps", []), dtype=float
                )
                if durations.shape != expected.shape or not np.allclose(
                    durations, hold, rtol=1.0e-8, atol=0.0
                ):
                    raise AnalysisError(
                        f"{directory}/{stage.name} has missing or different rate holds."
                    )
                target_temperatures = samples.get("segment_temperature_k", [])
                if target_temperatures and not np.allclose(
                    target_temperatures, spec.temperature_k, rtol=0.0, atol=1.0e-8
                ):
                    raise AnalysisError(
                        f"{directory}/{stage.name} has a different requested temperature."
                    )


def _check_comparable(curves: Sequence[StressStrain]) -> None:
    """Do not let pooling disguise a different axis, grid or temperature."""
    reference = curves[0]
    for curve in curves:
        if curve.axis != reference.axis or curve.controlled != "strain":
            raise AnalysisError(
                "All rate measurements must use the same deformation axis."
            )
        if curve.strain.shape != reference.strain.shape or not np.allclose(
            curve.strain, reference.strain, rtol=1.0e-6, atol=1.0e-9
        ):
            raise AnalysisError(
                "All rate measurements must use the same strain increments and range."
            )
        if (
            not math.isfinite(curve.temperature_k)
            or abs(curve.temperature_k - reference.temperature_k) > 1.0
        ):
            raise AnalysisError("All rate measurements must use the same temperature.")
        rate = curve.strain_rate_per_ns
        if rate is None or not math.isfinite(rate) or rate <= 0:
            raise AnalysisError(
                "Every deformation must record a finite positive strain rate."
            )
    if (
        max(item.temperature_k for item in curves)
        - min(item.temperature_k for item in curves)
        > 1.0
    ):
        raise AnalysisError("All rate measurements must use the same temperature.")


def analyse_modulus_rates(
    run_dirs: Sequence[str | Path],
    *,
    target_rate_per_ns: float,
    strain_limit: float = 0.015,
    max_extrapolation_decades: float = 2.0,
) -> ModulusRateReport:
    """Pool saved replicas by rate and compare logarithmic and power-law fits.

    A rate-scan root expands to the directories its workflow records. Every
    recorded directory must exist. Curves must share a strain ladder, axis
    and temperature, with at least three distinct rates overall. Repeated
    rates contribute replicas rather than additional rate-fit observations.

    Per-rate uncertainty is at least the replica standard deviation, and
    unresolved replicas or excessive replica spread keep that rate unresolved.
    """
    _analysis_options(target_rate_per_ns, max_extrapolation_decades)
    require_positive(strain_limit, None, name="strain_limit")
    directories = _expand_run_dirs(run_dirs)
    curves: list[StressStrain] = []
    for directory in directories:
        report = analyse_mechanics(directory, strain_limit=strain_limit)
        if not report.curves:
            raise AnalysisError(
                f"{directory} has no strain-controlled extension to fit."
            )
        curves.extend(report.curves)
    _check_comparable(curves)
    groups: list[list[StressStrain]] = []
    for curve in sorted(curves, key=lambda item: float(item.strain_rate_per_ns or 0)):
        if groups and math.isclose(
            float(curve.strain_rate_per_ns or 0),
            float(groups[-1][0].strain_rate_per_ns or 0),
            rel_tol=1.0e-8,
        ):
            groups[-1].append(curve)
        else:
            groups.append([curve])
    if len(groups) < 3:
        raise AnalysisError(
            "A modulus rate fit needs at least three distinct strain rates."
        )
    fits: list[ElasticModulus] = []
    notes: list[str] = []
    for group in groups:
        pooled = _pool(group)
        assert pooled is not None
        fit = youngs_modulus(pooled, strain_limit=strain_limit)
        replicas = [youngs_modulus(curve, strain_limit=strain_limit) for curve in group]
        spread = sample_spread([replica.modulus_mpa for replica in replicas])
        consistent = spread is None or spread <= MAX_REPLICA_SPREAD * abs(
            fit.modulus_mpa
        )
        resolved = (
            fit.resolved
            and all(replica.resolved for replica in replicas)
            and consistent
        )
        fit = replace(
            fit,
            standard_error_mpa=max(fit.standard_error_mpa, spread or 0.0),
            resolved=resolved,
        )
        fits.append(fit)
        if spread is None:
            notes.append(
                f"Rate {fit.strain_rate_per_ns:.6g} /ns has one replica; no replica spread is available."
            )
        if not resolved:
            notes.append(
                f"Rate {fit.strain_rate_per_ns:.6g} /ns is unresolved, including replica quality and spread."
            )
    options = {
        "target_rate_per_ns": target_rate_per_ns,
        "max_extrapolation_decades": max_extrapolation_decades,
    }
    log_linear = strain_rate_extrapolation(fits, form="log_linear", **options)
    power_law = strain_rate_extrapolation(fits, form="power_law", **options)
    notes.extend(log_linear.notes)
    notes.extend(power_law.notes)
    notes.append(
        "Rate extrapolations are empirical finite-rate estimates, not a measured quasi-static modulus."
    )
    return ModulusRateReport(
        run_dirs=tuple(str(directory) for directory in directories),
        fits=tuple(fits),
        log_linear=log_linear,
        power_law=power_law,
        notes=tuple(dict.fromkeys(notes)),
    )
