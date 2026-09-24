"""Resumable tensile scans for apparent engineering elongation at break.

The endpoint is the first sampled strain of a confirmed terminal stress loss,
expressed as a percentage of the equilibrated reference length. Fixed-topology
force fields do not model covalent fracture. The tensile ladder, validation and
stress-loss criterion are shared with the breaking-strength workflow.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._files import ReportFiles, write_json
from ._workflow import equilibrated_box_nm, run_fingerprint
from .breaking import (
    BreakingSchedule,
    BreakingSpec,
    _read_replica,
    breaking_protocol,
    breaking_schedule,
)
from .elasticity import StressStrain
from .protocols import (
    Protocol,
    RunManifest,
    run_protocol,
    standard_melt_equilibration,
    validate_run_inputs,
)
from .simulate import RunContext, safe_timestep_fs
from .strength import ElongationAtBreak, elongation_at_break
from .trajectory import AnalysisError

log = logging.getLogger(__name__)

PROTOCOL_NAME = "elongation"
DEFORM_STEM = "06_elongation"
WORKFLOW_NAME = "elongation_workflow.json"
_STAGE_PATTERN = re.compile(r"06_elongation_r(\d+)_(\d+)$")


class ElongationError(RuntimeError):
    """An elongation-at-break scan cannot safely start or resume."""


@dataclass(frozen=True)
class ElongationSpec(BreakingSpec):
    """Tensile ladder settings and the apparent break criterion.

    Controls and validation match :class:`BreakingSpec`. Strains are fractions
    of the reference length: ``max_strain=1.0`` requests at least 100% extension.
    A break requires ``confirmation_steps`` terminal nominal-stress samples at
    or below ``failure_fraction`` of the positive interior peak. The endpoint
    is the first sample in that interval, rather than the strain at the peak.
    """


DEFAULT_SPEC = ElongationSpec()


@dataclass(frozen=True)
class ElongationSchedule(BreakingSchedule):
    """One replica's compounded strain ladder, duration and average rate."""


@dataclass(frozen=True)
class ElongationReport:
    """Apparent elongation at break for independent velocity replicas.

    ``elongation_percent`` is the mean of individual replica percentages,
    supplied only when every requested replica is complete and resolved.
    ``replica_spread_percent`` is their sample standard deviation in percentage
    points when at least two replicas resolve. It measures trajectory
    variability, not uncertainty across independent morphologies.
    ``replica_indices`` retains the original IDs if replicas are missing.
    """

    run_dir: str
    manifest_path: str
    curves: tuple[StressStrain, ...]
    replicas: tuple[ElongationAtBreak, ...]
    replica_indices: tuple[int, ...]
    elongation_percent: float | None
    replica_spread_percent: float | None
    resolved: bool
    failure_fraction: float
    confirmation_steps: int
    notes: tuple[str, ...]


def elongation_schedule(spec: ElongationSpec = DEFAULT_SPEC) -> ElongationSchedule:
    """Price the compounded increments applied by the tensile engine."""
    schedule = breaking_schedule(spec)
    return ElongationSchedule(schedule.n_steps, schedule.increment, schedule.relax_ps)


def elongation_protocol(
    spec: ElongationSpec = DEFAULT_SPEC,
    *,
    timestep_fs: float = 2.0,
    replica: int = 0,
    reference_box_nm: Sequence[float] | None = None,
) -> Protocol:
    """One tensile replica in resumable chunks sharing a strain reference.

    Execute with :func:`run_elongation_scan` so every replica branches from
    the equilibrated cell and uses the same reference length.
    """
    ladder = breaking_protocol(
        spec,
        timestep_fs=timestep_fs,
        replica=replica,
        reference_box_nm=reference_box_nm,
    )
    return Protocol(
        PROTOCOL_NAME,
        tuple(
            replace(stage, name=stage.name.replace("06_breaking", DEFORM_STEM, 1))
            for stage in ladder.stages
        ),
    )


def _equilibration(spec: ElongationSpec, options: dict[str, Any]) -> Protocol:
    base = standard_melt_equilibration(
        target_temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        **options,
    )
    return Protocol(PROTOCOL_NAME, base.stages)


def elongation_scan(
    spec: ElongationSpec = DEFAULT_SPEC, **equilibration: Any
) -> Protocol:
    """The full schedule for inspection and dry-run cost estimation.

    Use :func:`run_elongation_scan` for execution: replicas must branch rather
    than follow each other as this flat description of the cost lists them.
    """
    return Protocol(
        PROTOCOL_NAME,
        (
            *_equilibration(spec, equilibration).stages,
            *(
                stage
                for replica in range(spec.n_replicas)
                for stage in elongation_protocol(spec, replica=replica).stages
            ),
        ),
    )


def _workflow(directory: Path) -> dict[str, Any]:
    path = directory / WORKFLOW_NAME
    if not path.is_file():
        return {}
    try:
        record = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise ElongationError(f"Cannot read {path}: {error}") from error
    if not isinstance(record, dict):
        raise ElongationError(f"{path} must contain a workflow record.")
    return record


def _check_resume_stages(
    manifest: RunManifest, settle: Protocol, ladders: Sequence[Protocol]
) -> None:
    """Refuse gaps that would mix new dynamics with completed descendants."""
    completed = set(manifest.stages)
    expected = {stage.name for plan in (settle, *ladders) for stage in plan.stages}
    if completed - expected:
        raise ElongationError("Cannot resume: the manifest contains unexpected stages.")
    for plan in (settle, *ladders):
        names = [stage.name for stage in plan.stages]
        present = [name for name in names if name in completed]
        if present != names[: len(present)]:
            raise ElongationError(
                "Cannot resume: completed stages have missing predecessors. "
                "Restore them or rerun with resume=False."
            )
    settled = {stage.name for stage in settle.stages}
    if completed - settled and not settled <= completed:
        raise ElongationError(
            "Cannot resume: tensile stages exist before equilibration is complete. "
            "Restore missing stages or rerun with resume=False."
        )


def run_elongation_scan(
    run: RunContext,
    run_dir: str | Path = "elongation",
    *,
    spec: ElongationSpec = DEFAULT_SPEC,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> ElongationReport:
    """Equilibrate, stretch independent velocity replicas, and read elongation at break.

    Settings are saved atomically before the first stage so an interrupted
    scan can resume only with the same request. ``resume=False`` reruns the
    complete scan. A foreign manifest requires a fresh directory; a changed
    request requires a fresh directory or an explicit complete rerun.
    Engine failures propagate and are never interpreted as material failure.
    """
    directory = Path(run_dir)
    settle = _equilibration(spec, equilibration)
    timestep = safe_timestep_fs(spec.temperature_k, run.spec)
    ladders = tuple(
        elongation_protocol(spec, timestep_fs=timestep, replica=replica)
        for replica in range(spec.n_replicas)
    )
    total_ns = (
        settle.total_duration_ps + sum(ladder.total_duration_ps for ladder in ladders)
    ) / 1000.0
    if spec.max_total_ns is not None and total_ns > spec.max_total_ns:
        raise ElongationError(
            f"The elongation scan costs {total_ns:.3g} ns, above max_total_ns="
            f"{spec.max_total_ns:g}; shorten the ladder or raise the budget."
        )
    settings = asdict(spec)
    settings.pop("max_total_ns")  # A budget change does not change the dynamics.
    request = json.loads(
        json.dumps(
            {
                "spec": settings,
                "equilibration": [asdict(stage) for stage in settle.stages],
                **run_fingerprint(run),
            }
        )
    )
    previous = _workflow(directory) if resume else {}
    manifest = RunManifest.load(directory) if resume else None
    if manifest is not None and manifest.protocol != PROTOCOL_NAME:
        raise ElongationError(
            f"{directory} contains a different protocol; use a fresh directory."
        )
    if previous and previous.get("request") != request:
        raise ElongationError(
            "Cannot resume elongation scan with different settings. Restore the "
            "original request, use a fresh directory, or rerun with resume=False."
        )
    if manifest is not None and manifest.stages and not previous:
        raise ElongationError(
            f"{directory} already contains stages without a matching elongation "
            "workflow record. Use a fresh directory or rerun with resume=False."
        )
    if manifest is not None:
        _check_resume_stages(manifest, settle, ladders)
        missing_states = [
            name
            for name, stage in manifest.stages.items()
            if not Path(stage.get("final_state", "")).is_file()
        ]
        if missing_states:
            raise ElongationError(
                "Cannot resume: completed stages have missing state files "
                f"({', '.join(missing_states)}). Restore them or rerun with "
                "resume=False so old descendants are not mixed with new dynamics."
            )
    record = {
        "request": request,
        "replica_stages": [
            [stage.name for stage in ladder.stages] for ladder in ladders
        ],
        "steps_per_replica": elongation_schedule(spec).n_steps,
        "timestep_fs": timestep,
    }
    if resume:
        validate_run_inputs(run, directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / WORKFLOW_NAME, record)
    log.info(
        "Elongation scan: %d replicas, %.3g ns total, %.3g average strain/ns; "
        "apparent stress-loss endpoint with fixed bonds.",
        spec.n_replicas,
        total_ns,
        elongation_schedule(spec).strain_rate_per_ns,
    )
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    settled = run_protocol(settle, run, directory, resume=resume, **chains)
    start_state = settled.final_state
    if not start_state or not Path(start_state).is_file():
        raise ElongationError("The equilibration did not leave a readable final state.")
    origin = equilibrated_box_nm(start_state)
    record.update({"reference_box_nm": origin, "start_state": start_state})
    write_json(directory / WORKFLOW_NAME, record)
    for replica in range(spec.n_replicas):
        run_protocol(
            elongation_protocol(
                spec,
                timestep_fs=timestep,
                replica=replica,
                reference_box_nm=origin,
            ),
            run,
            directory,
            # The first call alone resets a manifest for an explicit rerun.
            # Subsequent calls preserve the equilibration and other replicas.
            resume=True,
            state_in=start_state,
            **chains,
        )
    return analyse_elongation(directory)


def elongation_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Find dedicated elongation stages, excluding other tensile workflows."""
    manifest = RunManifest.load(run_dir)
    names = tuple(
        name
        for name in (manifest.stages if manifest is not None else ())
        if _STAGE_PATTERN.fullmatch(name)
    )
    if not names:
        raise AnalysisError(f"No elongation-at-break stages in {run_dir}.")
    return names


def _recorded_spec(record: dict[str, Any]) -> ElongationSpec:
    """Require a saved criterion and a schedule consistent with its settings."""
    if not record:
        # Unrecorded curves retain diagnostics, but cannot resolve a headline.
        return DEFAULT_SPEC
    try:
        settings = record["request"]["spec"]
        if not isinstance(settings, dict):
            raise ValueError("spec must contain the saved settings")
        required = set(asdict(DEFAULT_SPEC)) - {"max_total_ns"}
        missing = required - settings.keys()
        if missing:
            raise ValueError(f"missing settings: {', '.join(sorted(missing))}")
        spec = ElongationSpec(**settings)
        expected = [
            [stage.name for stage in elongation_protocol(spec, replica=index).stages]
            for index in range(spec.n_replicas)
        ]
        if (
            record.get("replica_stages") != expected
            or record.get("steps_per_replica") != elongation_schedule(spec).n_steps
        ):
            raise ValueError("the recorded ladder does not match the saved settings")
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisError(f"Malformed elongation workflow record: {error}") from error
    return spec


def analyse_elongation(run_dir: str | Path) -> ElongationReport:
    """Read elongation without running dynamics or modifying recorded results.

    Replicas are fitted separately: pooling their points would mix different
    failure strains. An incomplete requested ladder or missing replica leaves
    the headline unresolved even if the available replicas show a stress drop.
    The saved criterion is used on every reanalysis.
    """
    directory = Path(run_dir)
    names = elongation_stages(directory)
    record = _workflow(directory)
    spec = _recorded_spec(record)
    groups: dict[int, list[tuple[int, str]]] = {}
    for name in names:
        match = _STAGE_PATTERN.fullmatch(name)
        assert match is not None
        replica, chunk = map(int, match.groups())
        groups.setdefault(replica, []).append((chunk, name))
    curves: list[StressStrain] = []
    fits: list[ElongationAtBreak] = []
    complete = True
    notes: list[str] = [
        "Apparent engineering elongation at break from a terminal stress-loss "
        "criterion; this is not proof of fracture or covalent bond scission.",
        "Elongation depends on strain rate, temperature, cell size, morphology and "
        "force field. Velocity replicas do not sample independent morphologies.",
        "Nominal stress uses each hold's mean Cauchy stress and final transverse "
        "area; it approximates the mean force when the lateral area fluctuates.",
    ]
    for replica in sorted(groups):
        chunks = sorted(groups[replica])
        replica_complete = [chunk for chunk, _ in chunks] == list(range(len(chunks)))
        try:
            curve = _read_replica(directory, [name for _, name in chunks])
            fit = elongation_at_break(
                curve,
                failure_fraction=spec.failure_fraction,
                confirmation_steps=spec.confirmation_steps,
            )
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise AnalysisError(
                f"Malformed elongation replica {replica}: {error}"
            ) from error
        expected = record.get("replica_stages", [])
        replica_complete &= (
            replica < len(expected)
            and [name for _, name in chunks] == expected[replica]
            and curve.n_points == record.get("steps_per_replica")
        )
        complete &= replica_complete
        if not replica_complete:
            fit = replace(
                fit,
                resolved=False,
                elongation_percent=None,
                strain_at_break=None,
                break_stress_mpa=None,
                break_bracket=None,
                notes=(
                    *fit.notes,
                    "Incomplete replica: a stress-loss endpoint is provisional.",
                ),
            )
        curves.append(curve)
        fits.append(fit)
    if record:
        expected = record["replica_stages"]
        complete &= set(names) == {name for group in expected for name in group}
        complete &= len(curves) == spec.n_replicas
        complete &= all(
            curve.n_points == record["steps_per_replica"] for curve in curves
        )
    else:
        complete = False
        notes.append(
            "No workflow record: the requested scan extent cannot be verified."
        )
    if not complete:
        notes.append(
            "The requested scan is incomplete; elongation at break is unresolved."
        )
    resolved = complete and all(fit.resolved for fit in fits)
    elongations = [
        fit.elongation_percent for fit in fits if fit.elongation_percent is not None
    ]
    return ElongationReport(
        run_dir=str(directory),
        manifest_path=str(directory / "manifest.json"),
        curves=tuple(curves),
        replicas=tuple(fits),
        replica_indices=tuple(sorted(groups)),
        elongation_percent=float(np.mean(elongations)) if resolved else None,
        replica_spread_percent=float(np.std(elongations, ddof=1))
        if resolved and len(elongations) > 1
        else None,
        resolved=resolved,
        failure_fraction=spec.failure_fraction,
        confirmation_steps=spec.confirmation_steps,
        notes=tuple(notes),
    )


def write_elongation_report(
    report: ElongationReport,
    output_dir: str | Path | None = None,
    *,
    formats: Sequence[str] = ("png",),
) -> ReportFiles:
    """Write ``analysis/elongation.json`` and a nominal stress plot per replica.

    Pass ``formats=()`` for JSON alone. Both the differential Cauchy stress
    and its nominal-area conversion are included so the stress-loss endpoint
    is auditable.
    """
    from .plots import plot_elongation_at_break

    directory = (
        Path(report.run_dir) / "analysis" if output_dir is None else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = asdict(report)
    for curve, fit in zip(record["curves"], record["replicas"], strict=True):
        for key in ("strain", "stress_mpa", "lateral_strain", "lateral_stress_mpa"):
            curve[key] = curve[key].tolist()
        fit["nominal_stress_mpa"] = fit["nominal_stress_mpa"].tolist()
    json_path = directory / "elongation.json"
    write_json(json_path, record)
    figures: list[str] = []
    for index, curve, fit in zip(
        report.replica_indices, report.curves, report.replicas, strict=True
    ):
        if formats:
            figure = plot_elongation_at_break(curve, fit)
            for extension in formats:
                path = directory / f"elongation_r{index}.{extension}"
                figure.savefig(path, bbox_inches="tight")
                figures.append(str(path))
    return ReportFiles(json=str(json_path), figures=tuple(figures))
