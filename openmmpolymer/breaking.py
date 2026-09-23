"""A resumable tensile-strength scan and its recorded stress-loss criterion.

Each replica starts from the same equilibrated cell with fresh velocities.
The complete prescribed ladder is sampled: stopping at the first stress dip
would confuse yield or a fluctuation with the ultimate tensile strength.
Fixed-topology force fields do not model covalent bond scission, so this is
an apparent strength of the simulated cell, not a chemical fracture test.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._validation import require_integer, require_positive
from .elasticity import StressStrain, stress_strain
from .mechanical import _equilibrated_box_nm
from .protocols import (
    Protocol,
    RunManifest,
    Stage,
    _write_atomically,
    run_protocol,
    standard_melt_equilibration,
    validate_run_inputs,
)
from .reporters import TrajectoryOptions
from .simulate import RunContext, safe_timestep_fs
from .strength import BreakingStrength, breaking_strength
from .tg import ReportFiles
from .trajectory import AnalysisError

log = logging.getLogger(__name__)

PROTOCOL_NAME = "breaking"
DEFORM_STEM = "06_breaking"
WORKFLOW_NAME = "breaking_workflow.json"
_STAGE_PATTERN = re.compile(r"06_breaking_r(\d+)_(\d+)$")


class BreakingError(RuntimeError):
    """A tensile-strength scan cannot safely start or resume."""


@dataclass(frozen=True)
class BreakingSpec:
    """Settings for a tensile ladder and its apparent failure criterion.

    Increments compound; ``max_strain`` is the target engineering strain,
    reached or slightly exceeded by the last increment. ``relax_ps`` is the
    duration of each hold, whose second half supplies the mean stress.
    ``failure_fraction`` is the fraction of the peak nominal stress below
    which at least ``confirmation_steps`` terminal points must remain.
    This criterion records stress loss, not covalent fracture. A ladder that
    ends while still strengthening remains unresolved.

    ``stage_ps`` bounds the duration of each resumable chunk. Replicas share
    a configuration but draw fresh velocities. Their sample standard
    deviation measures trajectory variability, not morphology uncertainty.
    ``trajectory_ps`` optionally saves XTC frames for structural inspection.
    ``max_total_ns`` includes equilibration and every replica.
    """

    temperature_k: float = 298.15
    pressure_bar: float = 1.0
    axis: int = 2
    strain_increment: float = 0.01
    max_strain: float = 1.0
    relax_ps: float = 50.0
    n_replicas: int = 3
    samples_per_step: int = 250
    stage_ps: float = 1000.0
    failure_fraction: float = 0.5
    confirmation_steps: int = 3
    trajectory_ps: float | None = None
    max_total_ns: float | None = None

    def __post_init__(self) -> None:
        """Reject unusable settings before any dynamics or output."""
        for name in (
            "temperature_k",
            "pressure_bar",
            "strain_increment",
            "max_strain",
            "relax_ps",
            "stage_ps",
            "failure_fraction",
        ):
            require_positive(getattr(self, name), None, name=name)
        require_integer(self.axis, minimum=0, name="axis")
        require_integer(self.n_replicas, name="n_replicas")
        require_integer(self.samples_per_step, minimum=2, name="samples_per_step")
        require_integer(self.confirmation_steps, minimum=2, name="confirmation_steps")
        if self.axis > 2:
            raise ValueError("axis must be 0, 1 or 2.")
        if self.strain_increment >= self.max_strain:
            raise ValueError("strain_increment must be below max_strain.")
        if self.stage_ps < self.relax_ps:
            raise ValueError("stage_ps must hold at least one relax_ps increment.")
        if self.failure_fraction >= 1.0:
            raise ValueError("failure_fraction must be strictly between zero and one.")
        for name in ("trajectory_ps", "max_total_ns"):
            value = getattr(self, name)
            if value is not None:
                require_positive(value, None, name=name)


DEFAULT_SPEC = BreakingSpec()


@dataclass(frozen=True)
class BreakingSchedule:
    """One replica's compounded strain ladder and simulated duration."""

    n_steps: int
    increment: float
    relax_ps: float

    @property
    def max_strain(self) -> float:
        """Actual final engineering strain, including the last full increment."""
        return float((1.0 + self.increment) ** self.n_steps - 1.0)

    @property
    def total_ps(self) -> float:
        """Total duration of the holds."""
        return self.n_steps * self.relax_ps

    @property
    def strain_rate_per_ns(self) -> float:
        """Average engineering strain rate; the logarithmic rate is constant."""
        return self.max_strain / self.total_ps * 1000.0


@dataclass(frozen=True)
class BreakingReport:
    """Separate replica peaks, criterion status, and all measured curves.

    ``strength_mpa`` is the mean of resolved replica peaks, supplied only
    when every requested replica completed and met the stress-loss criterion.
    Unresolved scans still report every observed peak through ``replicas``.
    ``replica_spread_mpa`` is the sample standard deviation of those peaks
    when the report resolves and there are at least two replicas.
    """

    run_dir: str
    manifest_path: str
    curves: tuple[StressStrain, ...]
    replicas: tuple[BreakingStrength, ...]
    strength_mpa: float | None
    replica_spread_mpa: float | None
    resolved: bool
    failure_fraction: float
    confirmation_steps: int
    notes: tuple[str, ...]


def breaking_schedule(spec: BreakingSpec = DEFAULT_SPEC) -> BreakingSchedule:
    """Price one tensile ladder using the increments the engine applies."""
    steps = math.ceil(math.log1p(spec.max_strain) / math.log1p(spec.strain_increment))
    return BreakingSchedule(max(1, steps), spec.strain_increment, spec.relax_ps)


def breaking_protocol(
    spec: BreakingSpec = DEFAULT_SPEC,
    *,
    timestep_fs: float = 2.0,
    replica: int = 0,
    reference_box_nm: Sequence[float] | None = None,
) -> Protocol:
    """One replica, split into resumable chunks sharing a strain reference.

    Run through :func:`run_breaking_scan` to supply the equilibrated cell to
    every chunk and to branch replicas from the same starting configuration.
    """
    require_integer(replica, minimum=0, name="replica")
    require_positive(timestep_fs, None, name="timestep_fs")
    schedule = breaking_schedule(spec)
    per_chunk = int(spec.stage_ps // spec.relax_ps)
    stages: list[Stage] = []
    for chunk, done in enumerate(range(0, schedule.n_steps, per_chunk)):
        options: dict[str, Any] = {
            "temperature_k": spec.temperature_k,
            "pressure_bar": spec.pressure_bar,
            "axis": spec.axis,
            "strain_increment": spec.strain_increment,
            "n_steps": min(per_chunk, schedule.n_steps - done),
            "relax_ps": spec.relax_ps,
            "strain_start": (1.0 + spec.strain_increment) ** done - 1.0,
            "samples_per_step": spec.samples_per_step,
            "timestep_fs": timestep_fs,
            "new_velocities": chunk == 0,
        }
        if reference_box_nm is not None:
            options["reference_box_nm"] = list(reference_box_nm)
        if spec.trajectory_ps is not None:
            options["trajectory"] = TrajectoryOptions("xtc", spec.trajectory_ps)
        stages.append(Stage(f"{DEFORM_STEM}_r{replica}_{chunk:03d}", "deform", options))
    return Protocol(PROTOCOL_NAME, tuple(stages))


def _equilibration(spec: BreakingSpec, options: dict[str, Any]) -> Protocol:
    base = standard_melt_equilibration(
        target_temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        **options,
    )
    return Protocol(PROTOCOL_NAME, base.stages)


def breaking_scan(spec: BreakingSpec = DEFAULT_SPEC, **equilibration: Any) -> Protocol:
    """The full schedule for inspection and dry-run cost estimation.

    Use :func:`run_breaking_scan` for execution: replicas must branch rather
    than follow each other as this flat description of the cost lists them.
    """
    return Protocol(
        PROTOCOL_NAME,
        (
            *_equilibration(spec, equilibration).stages,
            *(
                stage
                for replica in range(spec.n_replicas)
                for stage in breaking_protocol(spec, replica=replica).stages
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
        raise BreakingError(f"Cannot read {path}: {error}") from error
    if not isinstance(record, dict):
        raise BreakingError(f"{path} must contain a workflow record.")
    return record


def run_breaking_scan(
    run: RunContext,
    run_dir: str | Path = "breaking",
    *,
    spec: BreakingSpec = DEFAULT_SPEC,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> BreakingReport:
    """Equilibrate, stretch independent velocity replicas, and read strength.

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
        breaking_protocol(spec, timestep_fs=timestep, replica=replica)
        for replica in range(spec.n_replicas)
    )
    total_ns = (
        settle.total_duration_ps + sum(ladder.total_duration_ps for ladder in ladders)
    ) / 1000.0
    if spec.max_total_ns is not None and total_ns > spec.max_total_ns:
        raise BreakingError(
            f"The breaking scan costs {total_ns:.3g} ns, above max_total_ns="
            f"{spec.max_total_ns:g}; shorten the ladder or raise the budget."
        )
    settings = asdict(spec)
    settings.pop("max_total_ns")  # A budget change does not change the dynamics.
    request = json.loads(
        json.dumps(
            {
                "spec": settings,
                "equilibration": [asdict(stage) for stage in settle.stages],
                "system": asdict(run.spec),
                "seed": run.seed,
                "system_sha256": hashlib.sha256(run.system_xml.encode()).hexdigest(),
                "coordinates_sha256": hashlib.sha256(
                    np.asarray(run.box.positions_nm, dtype=np.float64).tobytes()
                ).hexdigest(),
                "box_nm": list(run.box.box_nm),
            }
        )
    )
    previous = _workflow(directory) if resume else {}
    manifest = RunManifest.load(directory) if resume else None
    if manifest is not None and manifest.protocol != PROTOCOL_NAME:
        raise BreakingError(
            f"{directory} contains a different protocol; use a fresh directory."
        )
    if previous and previous.get("request") != request:
        raise BreakingError(
            "Cannot resume breaking scan with different settings. Restore the "
            "original request, use a fresh directory, or rerun with resume=False."
        )
    if manifest is not None and manifest.stages and not previous:
        raise BreakingError(
            f"{directory} already contains stages without a matching breaking "
            "workflow record. Use a fresh directory or rerun with resume=False."
        )
    if manifest is not None:
        missing_states = [
            name
            for name, stage in manifest.stages.items()
            if not Path(stage.get("final_state", "")).is_file()
        ]
        if missing_states:
            raise BreakingError(
                "Cannot resume: completed stages have missing state files "
                f"({', '.join(missing_states)}). Restore them or rerun with "
                "resume=False so old descendants are not mixed with new dynamics."
            )
    record = {
        "request": request,
        "replica_stages": [
            [stage.name for stage in ladder.stages] for ladder in ladders
        ],
        "steps_per_replica": breaking_schedule(spec).n_steps,
        "timestep_fs": timestep,
    }
    if resume:
        validate_run_inputs(run, directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_atomically(directory / WORKFLOW_NAME, json.dumps(record, indent=2) + "\n")
    log.info(
        "Breaking scan: %d replicas, %.3g ns total, %.3g average strain/ns; "
        "fixed-topology apparent tensile strength, not bond scission.",
        spec.n_replicas,
        total_ns,
        breaking_schedule(spec).strain_rate_per_ns,
    )
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    settled = run_protocol(settle, run, directory, resume=resume, **chains)
    start_state = settled.final_state
    if not start_state or not Path(start_state).is_file():
        raise BreakingError("The equilibration did not leave a readable final state.")
    origin = _equilibrated_box_nm(start_state)
    record.update({"reference_box_nm": origin, "start_state": start_state})
    _write_atomically(directory / WORKFLOW_NAME, json.dumps(record, indent=2) + "\n")
    for replica in range(spec.n_replicas):
        run_protocol(
            breaking_protocol(
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
    return analyse_breaking(directory)


def breaking_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Find only the dedicated strength stages, excluding modulus extensions."""
    manifest = RunManifest.load(run_dir)
    names = tuple(
        name
        for name in (manifest.stages if manifest is not None else ())
        if _STAGE_PATTERN.fullmatch(name)
    )
    if not names:
        raise AnalysisError(f"No breaking-strength stages in {run_dir}.")
    return names


def _read_replica(directory: Path, names: Sequence[str]) -> StressStrain:
    """Check chunk continuity before the common reader sorts the samples."""
    manifest = RunManifest.load(directory)
    assert manifest is not None  # breaking_stages already found recorded stages
    origin: list[float] | None = None
    axis: int | None = None
    last_strain = -1.0
    for name in names:
        samples = manifest.stages[name]["samples"]
        reference = np.asarray(samples["reference_box_nm"], dtype=np.float64)
        recorded_axis = samples["deform_axis"]
        if (
            reference.shape != (3,)
            or not np.all(np.isfinite(reference))
            or np.any(reference <= 0.0)
        ):
            raise ValueError(f"{name} has an invalid reference box.")
        if len(recorded_axis) != 1 or recorded_axis[0] not in (0, 1, 2):
            raise ValueError(f"{name} has an invalid deformation axis.")
        if origin is None:
            origin, axis = reference.tolist(), int(recorded_axis[0])
        elif reference.tolist() != origin or int(recorded_axis[0]) != axis:
            raise ValueError(f"{name} changes the reference box or deformation axis.")
        strain = np.asarray(samples["segment_strain"], dtype=np.float64)
        if (
            strain.ndim != 1
            or strain.size == 0
            or not np.all(np.isfinite(strain))
            or strain[0] <= last_strain
            or np.any(np.diff(strain) <= 0.0)
        ):
            raise ValueError(f"{name} has non-increasing or invalid strains.")
        for key in (
            "segment_duration_ps",
            "segment_box_x_nm",
            "segment_box_y_nm",
            "segment_box_z_nm",
            "segment_stress_xx_bar",
            "segment_stress_yy_bar",
            "segment_stress_zz_bar",
        ):
            values = np.asarray(samples[key], dtype=np.float64)
            if values.shape != strain.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} has invalid {key} samples.")
            if (key.endswith("_nm") or key == "segment_duration_ps") and np.any(
                values <= 0.0
            ):
                raise ValueError(f"{name} has nonpositive {key} samples.")
        last_strain = float(strain[-1])
    return stress_strain(directory, names)


def analyse_breaking(run_dir: str | Path) -> BreakingReport:
    """Read strength without running dynamics or modifying recorded results.

    Replicas are fitted separately: pooling their points would mix different
    failure strains. An incomplete requested ladder or missing replica leaves
    the headline unresolved even if the available replicas show a stress drop.
    The saved criterion is used on every reanalysis.
    """
    directory = Path(run_dir)
    names = breaking_stages(directory)
    record = _workflow(directory)
    spec = BreakingSpec(**record.get("request", {}).get("spec", {}))
    groups: dict[int, list[tuple[int, str]]] = {}
    for name in names:
        match = _STAGE_PATTERN.fullmatch(name)
        assert match is not None
        replica, chunk = map(int, match.groups())
        groups.setdefault(replica, []).append((chunk, name))
    curves: list[StressStrain] = []
    fits: list[BreakingStrength] = []
    complete = True
    notes: list[str] = [
        "Apparent ultimate nominal tensile strength from a terminal stress-loss "
        "criterion; this is not proof of fracture or covalent bond scission.",
        "Strength depends on strain rate, temperature, cell size, morphology and "
        "force field. Velocity replicas do not sample independent morphologies.",
        "Nominal stress uses each hold's mean Cauchy stress and final transverse "
        "area; it approximates the mean force when the lateral area fluctuates.",
    ]
    for replica in sorted(groups):
        chunks = sorted(groups[replica])
        replica_complete = [chunk for chunk, _ in chunks] == list(range(len(chunks)))
        try:
            curve = _read_replica(directory, [name for _, name in chunks])
            fit = breaking_strength(
                curve,
                failure_fraction=spec.failure_fraction,
                confirmation_steps=spec.confirmation_steps,
            )
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise AnalysisError(
                f"Malformed breaking replica {replica}: {error}"
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
                strength_mpa=None,
                failure_strain=None,
                failure_stress_mpa=None,
                failure_bracket=None,
                notes=(
                    *fit.notes,
                    "Incomplete replica: the observed peak is provisional.",
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
            "The requested scan is incomplete; observed peaks are provisional."
        )
    resolved = complete and all(fit.resolved for fit in fits)
    peaks = [fit.peak_stress_mpa for fit in fits]
    return BreakingReport(
        run_dir=str(directory),
        manifest_path=str(directory / "manifest.json"),
        curves=tuple(curves),
        replicas=tuple(fits),
        strength_mpa=float(np.mean(peaks)) if resolved else None,
        replica_spread_mpa=float(np.std(peaks, ddof=1))
        if resolved and len(peaks) > 1
        else None,
        resolved=resolved,
        failure_fraction=spec.failure_fraction,
        confirmation_steps=spec.confirmation_steps,
        notes=tuple(notes),
    )


def write_breaking_report(
    report: BreakingReport,
    output_dir: str | Path | None = None,
    *,
    formats: Sequence[str] = ("png",),
) -> ReportFiles:
    """Write ``analysis/breaking.json`` and a nominal stress plot per replica.

    Pass ``formats=()`` for JSON alone. Both the differential Cauchy stress
    and its nominal-area conversion are included so the peak is auditable.
    """
    from .plots import plot_breaking_strength

    directory = (
        Path(report.run_dir) / "analysis" if output_dir is None else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = asdict(report)
    for curve, fit in zip(record["curves"], record["replicas"], strict=True):
        for key in ("strain", "stress_mpa", "lateral_strain", "lateral_stress_mpa"):
            curve[key] = curve[key].tolist()
        fit["nominal_stress_mpa"] = fit["nominal_stress_mpa"].tolist()
    json_path = directory / "breaking.json"
    _write_atomically(json_path, json.dumps(record, indent=2, allow_nan=False) + "\n")
    figures: list[str] = []
    for index, (curve, fit) in enumerate(
        zip(report.curves, report.replicas, strict=True)
    ):
        if formats:
            figure = plot_breaking_strength(curve, fit)
            for extension in formats:
                path = directory / f"breaking_r{index}.{extension}"
                figure.savefig(path, bbox_inches="tight")
                figures.append(str(path))
    return ReportFiles(json=str(json_path), figures=tuple(figures))
