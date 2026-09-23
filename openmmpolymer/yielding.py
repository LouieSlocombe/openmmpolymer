"""A resumable tensile scan for offset proof stress (apparent yield strength).

Each replica starts from the same equilibrated cell with fresh velocities.
The saved elastic-fit window and strain offset define the reported strength.
An offset-line crossing describes the simulated loading curve; without an
unloading measurement it does not establish permanent plastic deformation.
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

from ._validation import require_finite, require_integer, require_positive
from .breaking import _read_replica
from .elasticity import StressStrain
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
from .strength import YieldStrength, yield_strength
from .tg import ReportFiles
from .trajectory import AnalysisError

log = logging.getLogger(__name__)

PROTOCOL_NAME = "yield"
DEFORM_STEM = "06_yield"
WORKFLOW_NAME = "yield_workflow.json"
_STAGE_PATTERN = re.compile(r"06_yield_r(\d+)_(\d+)$")


class YieldError(RuntimeError):
    """A yield-strength scan cannot safely start or resume."""


@dataclass(frozen=True)
class YieldSpec:
    """Settings for a tensile ladder and its offset proof-stress criterion.

    Increments compound; ``max_strain`` is the target engineering strain,
    reached or slightly exceeded by the last increment. ``relax_ps`` is the
    duration of each hold, whose second half supplies the mean stress.
    ``offset_strain`` shifts the initial elastic fit to define the proof
    stress. The default 0.002 is a 0.2% engineering-strain offset. The fit uses
    nominal stress between ``fit_min_strain`` and ``fit_max_strain``. A ladder
    with an unreliable elastic fit or no later crossing remains unresolved.

    ``stage_ps`` bounds the duration of each resumable chunk. Replicas share
    a configuration but draw fresh velocities. Their sample standard
    deviation measures trajectory variability, not morphology uncertainty.
    ``trajectory_ps`` optionally saves XTC frames for structural inspection.
    ``max_total_ns`` includes equilibration and every replica.
    """

    temperature_k: float = 298.15
    pressure_bar: float = 1.0
    axis: int = 2
    strain_increment: float = 0.002
    max_strain: float = 0.3
    relax_ps: float = 50.0
    n_replicas: int = 3
    samples_per_step: int = 250
    stage_ps: float = 1000.0
    offset_strain: float = 0.002
    fit_min_strain: float = 0.0
    fit_max_strain: float = 0.02
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
            "offset_strain",
            "fit_max_strain",
        ):
            require_positive(getattr(self, name), None, name=name)
        require_integer(self.axis, minimum=0, name="axis")
        require_integer(self.n_replicas, name="n_replicas")
        require_integer(self.samples_per_step, minimum=2, name="samples_per_step")
        if self.axis > 2:
            raise ValueError("axis must be 0, 1 or 2.")
        if self.strain_increment >= self.max_strain:
            raise ValueError("strain_increment must be below max_strain.")
        if self.stage_ps < self.relax_ps:
            raise ValueError("stage_ps must hold at least one relax_ps increment.")
        require_finite(self.fit_min_strain, None, name="fit_min_strain")
        if not 0.0 <= self.fit_min_strain < self.fit_max_strain:
            raise ValueError(
                "fit_min_strain must be nonnegative and below fit_max_strain."
            )
        if self.fit_max_strain >= self.max_strain:
            raise ValueError("fit_max_strain must be below max_strain.")
        if self.offset_strain >= self.max_strain:
            raise ValueError("offset_strain must be below max_strain.")
        for name in ("trajectory_ps", "max_total_ns"):
            value = getattr(self, name)
            if value is not None:
                require_positive(value, None, name=name)


DEFAULT_SPEC = YieldSpec()


@dataclass(frozen=True)
class YieldSchedule:
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
class YieldReport:
    """Separate replica proof stresses and all measured curves.

    ``strength_mpa`` is the mean of resolved replica proof stresses, supplied
    only when every requested replica completed and resolved its offset-line
    crossing. Unresolved scans retain the measured curves and fit diagnostics.
    ``replica_spread_mpa`` is the sample standard deviation of those strengths
    when the report resolves and there are at least two replicas.
    ``replica_indices`` preserves the recorded IDs when some replicas are absent.
    """

    run_dir: str
    manifest_path: str
    curves: tuple[StressStrain, ...]
    replicas: tuple[YieldStrength, ...]
    replica_indices: tuple[int, ...]
    strength_mpa: float | None
    replica_spread_mpa: float | None
    resolved: bool
    offset_strain: float
    fit_min_strain: float
    fit_max_strain: float
    notes: tuple[str, ...]


def yield_schedule(spec: YieldSpec = DEFAULT_SPEC) -> YieldSchedule:
    """Price one tensile ladder using the increments the engine applies."""
    steps = math.ceil(math.log1p(spec.max_strain) / math.log1p(spec.strain_increment))
    return YieldSchedule(max(1, steps), spec.strain_increment, spec.relax_ps)


def yield_protocol(
    spec: YieldSpec = DEFAULT_SPEC,
    *,
    timestep_fs: float = 2.0,
    replica: int = 0,
    reference_box_nm: Sequence[float] | None = None,
) -> Protocol:
    """One replica, split into resumable chunks sharing a strain reference.

    Run through :func:`run_yield_scan` to supply the equilibrated cell to
    every chunk and to branch replicas from the same starting configuration.
    """
    require_integer(replica, minimum=0, name="replica")
    require_positive(timestep_fs, None, name="timestep_fs")
    schedule = yield_schedule(spec)
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


def _equilibration(spec: YieldSpec, options: dict[str, Any]) -> Protocol:
    base = standard_melt_equilibration(
        target_temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        **options,
    )
    return Protocol(PROTOCOL_NAME, base.stages)


def yield_scan(spec: YieldSpec = DEFAULT_SPEC, **equilibration: Any) -> Protocol:
    """The full schedule for inspection and dry-run cost estimation.

    Use :func:`run_yield_scan` for execution: replicas must branch rather
    than follow each other as this flat description of the cost lists them.
    """
    return Protocol(
        PROTOCOL_NAME,
        (
            *_equilibration(spec, equilibration).stages,
            *(
                stage
                for replica in range(spec.n_replicas)
                for stage in yield_protocol(spec, replica=replica).stages
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
        raise YieldError(f"Cannot read {path}: {error}") from error
    if not isinstance(record, dict):
        raise YieldError(f"{path} must contain a workflow record.")
    return record


def run_yield_scan(
    run: RunContext,
    run_dir: str | Path = "yield",
    *,
    spec: YieldSpec = DEFAULT_SPEC,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> YieldReport:
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
        yield_protocol(spec, timestep_fs=timestep, replica=replica)
        for replica in range(spec.n_replicas)
    )
    total_ns = (
        settle.total_duration_ps + sum(ladder.total_duration_ps for ladder in ladders)
    ) / 1000.0
    if spec.max_total_ns is not None and total_ns > spec.max_total_ns:
        raise YieldError(
            f"The yield scan costs {total_ns:.3g} ns, above max_total_ns="
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
        raise YieldError(
            f"{directory} contains a different protocol; use a fresh directory."
        )
    if previous and previous.get("request") != request:
        raise YieldError(
            "Cannot resume yield scan with different settings. Restore the "
            "original request, use a fresh directory, or rerun with resume=False."
        )
    if manifest is not None and manifest.stages and not previous:
        raise YieldError(
            f"{directory} already contains stages without a matching yield "
            "workflow record. Use a fresh directory or rerun with resume=False."
        )
    if manifest is not None:
        missing_states = [
            name
            for name, stage in manifest.stages.items()
            if not Path(stage.get("final_state", "")).is_file()
        ]
        if missing_states:
            raise YieldError(
                "Cannot resume: completed stages have missing state files "
                f"({', '.join(missing_states)}). Restore them or rerun with "
                "resume=False so old descendants are not mixed with new dynamics."
            )
    record = {
        "request": request,
        "replica_stages": [
            [stage.name for stage in ladder.stages] for ladder in ladders
        ],
        "steps_per_replica": yield_schedule(spec).n_steps,
        "timestep_fs": timestep,
    }
    if resume:
        validate_run_inputs(run, directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_atomically(directory / WORKFLOW_NAME, json.dumps(record, indent=2) + "\n")
    log.info(
        "Yield scan: %d replicas, %.3g ns total, %.3g average strain/ns; "
        "offset proof stress at %.3g engineering strain.",
        spec.n_replicas,
        total_ns,
        yield_schedule(spec).strain_rate_per_ns,
        spec.offset_strain,
    )
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    settled = run_protocol(settle, run, directory, resume=resume, **chains)
    start_state = settled.final_state
    if not start_state or not Path(start_state).is_file():
        raise YieldError("The equilibration did not leave a readable final state.")
    origin = _equilibrated_box_nm(start_state)
    record.update({"reference_box_nm": origin, "start_state": start_state})
    _write_atomically(directory / WORKFLOW_NAME, json.dumps(record, indent=2) + "\n")
    for replica in range(spec.n_replicas):
        run_protocol(
            yield_protocol(
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
    return analyse_yield(directory)


def yield_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Find dedicated yield stages, excluding modulus and breaking scans."""
    manifest = RunManifest.load(run_dir)
    names = tuple(
        name
        for name in (manifest.stages if manifest is not None else ())
        if _STAGE_PATTERN.fullmatch(name)
    )
    if not names:
        raise AnalysisError(f"No yield-strength stages in {run_dir}.")
    return names


def analyse_yield(run_dir: str | Path) -> YieldReport:
    """Read strength without running dynamics or modifying recorded results.

    Replicas are fitted separately: pooling their points would mix different
    yield strains. An incomplete requested ladder or missing replica leaves
    the headline unresolved even if the available curve crosses its offset
    line. The saved offset and elastic-fit window are used on every reanalysis.
    """
    directory = Path(run_dir)
    names = yield_stages(directory)
    record = _workflow(directory)
    spec = YieldSpec(**record.get("request", {}).get("spec", {}))
    groups: dict[int, list[tuple[int, str]]] = {}
    for name in names:
        match = _STAGE_PATTERN.fullmatch(name)
        assert match is not None
        replica, chunk = map(int, match.groups())
        groups.setdefault(replica, []).append((chunk, name))
    curves: list[StressStrain] = []
    fits: list[YieldStrength] = []
    complete = True
    notes: list[str] = [
        "Apparent nominal yield strength from an offset proof-stress criterion. "
        "An unloading measurement is needed to establish permanent strain.",
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
            fit = yield_strength(
                curve,
                offset_strain=spec.offset_strain,
                fit_min_strain=spec.fit_min_strain,
                fit_max_strain=spec.fit_max_strain,
            )
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise AnalysisError(
                f"Malformed yield replica {replica}: {error}"
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
                yield_strain=None,
                yield_bracket=None,
                notes=(
                    *fit.notes,
                    "Incomplete replica: an offset-line crossing is provisional.",
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
            "The requested scan is incomplete; yield strengths are provisional."
        )
    resolved = complete and all(fit.resolved for fit in fits)
    strengths = [fit.strength_mpa for fit in fits if fit.strength_mpa is not None]
    return YieldReport(
        run_dir=str(directory),
        manifest_path=str(directory / "manifest.json"),
        curves=tuple(curves),
        replicas=tuple(fits),
        replica_indices=tuple(sorted(groups)),
        strength_mpa=float(np.mean(strengths)) if resolved else None,
        replica_spread_mpa=float(np.std(strengths, ddof=1))
        if resolved and len(strengths) > 1
        else None,
        resolved=resolved,
        offset_strain=spec.offset_strain,
        fit_min_strain=spec.fit_min_strain,
        fit_max_strain=spec.fit_max_strain,
        notes=tuple(notes),
    )


def write_yield_report(
    report: YieldReport,
    output_dir: str | Path | None = None,
    *,
    formats: Sequence[str] = ("png",),
) -> ReportFiles:
    """Write ``analysis/yield.json`` and a nominal stress plot per replica.

    Pass ``formats=()`` for JSON alone. Both the differential Cauchy stress
    and its nominal-area conversion are included so the crossing is auditable.
    """
    from .plots import plot_yield_strength

    directory = (
        Path(report.run_dir) / "analysis" if output_dir is None else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = asdict(report)
    for curve, fit in zip(record["curves"], record["replicas"], strict=True):
        for key in ("strain", "stress_mpa", "lateral_strain", "lateral_stress_mpa"):
            curve[key] = curve[key].tolist()
        fit["nominal_stress_mpa"] = fit["nominal_stress_mpa"].tolist()
    json_path = directory / "yield.json"
    _write_atomically(json_path, json.dumps(record, indent=2, allow_nan=False) + "\n")
    figures: list[str] = []
    for index, curve, fit in zip(
        report.replica_indices, report.curves, report.replicas, strict=True
    ):
        if formats:
            figure = plot_yield_strength(curve, fit)
            for extension in formats:
                path = directory / f"yield_r{index}.{extension}"
                figure.savefig(path, bbox_inches="tight")
                figures.append(str(path))
    return ReportFiles(json=str(json_path), figures=tuple(figures))
