"""Hold-time sensitivity of elastic measurements from comparable loading paths.

Shear and tensile paths use strain/ns; pressure and stress controlled paths use
bar/ns. These are nominal path speeds, including every hold and both branches
of a reversing ladder. Finite-rate extrapolation does not establish equilibrium.

Young's modulus runs the same extension branches as Poisson's ratio but keeps
two things from the rate scan it began as. Its scans record their request in
``modulus_rate_workflow.json``, key for key as that scan did, so they resume
and read as before. And each rate is one observation, fitted through every
replica's points at once as a single modulus scan fits them, with the
replicas' spread as the floor of its error, so the rate series reads the same
modulus a single scan would.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._files import write_json
from ._validation import require_positive
from ._workflow import (
    chain_options,
    equilibrate,
    group_by_stem,
    record_scan_request,
    require_distinct,
    resumable_record,
    run_fingerprint,
    sample_spread,
    start_fingerprint,
    strain_ladder,
    validate_hold_times,
)
from .elasticity import (
    StressStrain,
    bulk_modulus,
    deform_stages,
    load_curve,
    poisson_ratio,
    shear_modulus,
    stress_strain,
    youngs_modulus,
)
from .mechanical import (
    BULK_STEM,
    DEFAULT_SPEC,
    MAX_REPLICA_SPREAD,
    MechanicalError,
    ModulusSpec,
    _pool,
    deform_protocol,
    equilibration_protocol,
    extra_stages,
)
from .protocols import Protocol, Stage, run_protocol
from .rate_dependence import (
    _TEMPERATURE_TOLERANCE_K,
    RateObservation,
    RateProperty,
    RateReport,
    analyse_rate_observations,
    validate_rate_request,
)
from .simulate import RunContext, safe_timestep_fs
from .trajectory import AnalysisError

WORKFLOW_NAME = "elastic_rate_workflow.json"
YOUNGS_WORKFLOW_NAME = "modulus_rate_workflow.json"
ELASTIC_RATE_PROPERTIES = {
    "youngs_modulus": RateProperty(
        "youngs_modulus", "Young's modulus", "MPa", "strain/ns", trend="increasing"
    ),
    "poisson_ratio": RateProperty(
        "poisson_ratio",
        "Poisson's ratio",
        "dimensionless",
        "strain/ns",
        trend="any",
        lower_bound=0.0,
        upper_bound=0.5,
    ),
    "shear_modulus": RateProperty(
        "shear_modulus", "Shear modulus", "MPa", "strain/ns", trend="increasing"
    ),
    "bulk_modulus": RateProperty(
        "bulk_modulus", "Bulk modulus", "MPa", "bar/ns", trend="any"
    ),
    "load_modulus": RateProperty(
        "load_modulus",
        "Constant-stress Young's modulus",
        "MPa",
        "bar/ns",
        trend="increasing",
    ),
}
_PATH_KEYS = {
    "poisson_ratio": "segment_strain",
    "shear_modulus": "segment_shear_strain",
    "bulk_modulus": "segment_pressure_bar",
    "load_modulus": "segment_applied_stress_bar",
}
_KIND = {
    "youngs_modulus": "deform",
    "poisson_ratio": "deform",
    "shear_modulus": "shear",
    "bulk_modulus": "compress",
    "load_modulus": "load",
}
_HOLD = {
    "youngs_modulus": "relax_ps",
    "poisson_ratio": "relax_ps",
    "shear_modulus": "shear_ps_each",
    "bulk_modulus": "bulk_ps_each",
    "load_modulus": "load_ps_each",
}


@dataclass(frozen=True)
class ElasticRatePlan:
    """One common preparation and every replica's entire loading path."""

    equilibration: Protocol
    protocols: tuple[tuple[Protocol, ...], ...]
    hold_times_ps: tuple[float, ...]
    property_name: str

    @property
    def total_ns(self) -> float:
        """Dynamics cost including preparation, every rate and every replica."""
        return (
            self.equilibration.total_duration_ps
            + sum(p.total_duration_ps for group in self.protocols for p in group)
        ) / 1000.0


def _property(name: str) -> RateProperty:
    try:
        return ELASTIC_RATE_PROPERTIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown elastic property {name!r}; choose {', '.join(ELASTIC_RATE_PROPERTIES)}."
        ) from None


def _rate_spec(spec: ModulusSpec, name: str, hold: float) -> ModulusSpec:
    options: dict[str, Any] = {_HOLD[name]: hold}
    return replace(spec, **options)


def _protocol(
    spec: ModulusSpec,
    name: str,
    replica: int,
    *,
    timestep_fs: float = 2.0,
    reference_box_nm: Sequence[float] | None = None,
) -> Protocol:
    if _KIND[name] == "deform":
        return deform_protocol(
            spec,
            timestep_fs=timestep_fs,
            replica=replica,
            reference_box_nm=reference_box_nm,
        )
    selected = [
        s for s in extra_stages(spec, timestep_fs=timestep_fs) if s.kind == _KIND[name]
    ]
    if not selected:
        raise ValueError(f"{name} needs its loading ladder enabled in ModulusSpec.")
    stage = selected[0]
    options = {**stage.options, "new_velocities": True}
    if stage.kind == "shear":
        options["plane"] = (0, 2)
    return Protocol(
        name=f"{name}_rate_scan",
        stages=(
            Stage(
                f"{stage.name}_r{replica}",
                stage.kind,
                options,
            ),
        ),
    )


def _expected_path(stage: Stage) -> tuple[str, list[float], float]:
    options = stage.options
    if stage.kind == "deform":
        path = strain_ladder(
            options["strain_start"], options["strain_increment"], options["n_steps"]
        )
        return "segment_strain", path.tolist(), float(options["relax_ps"])
    name, key = {
        "shear": ("strains", "segment_shear_strain"),
        "compress": ("pressures_bar", "segment_pressure_bar"),
        "load": ("stresses_bar", "segment_applied_stress_bar"),
    }[stage.kind]
    return key, [float(v) for v in options[name]], float(options["duration_ps_each"])


def validate_elastic_rate_scan(
    spec: ModulusSpec,
    hold_times_ps: Sequence[float],
    *,
    property_name: str,
    target_rate: float,
    max_extrapolation_decades: float = 2.0,
    **equilibration: Any,
) -> ElasticRatePlan:
    """Validate a selected property and price all rates before any output exists."""
    _property(property_name)
    validate_rate_request(target_rate, max_extrapolation_decades)
    require_positive(spec.elastic_strain_limit, None, name="strain_limit")
    holds = validate_hold_times(hold_times_ps)
    groups = tuple(
        tuple(
            _protocol(
                _rate_spec(spec, property_name, hold),
                property_name,
                i * spec.n_replicas + replica,
            )
            for replica in range(spec.n_replicas)
        )
        for i, hold in enumerate(holds)
    )
    for group in groups:
        path = [v for stage in group[0].stages for v in _expected_path(stage)[1]]
        if not all(math.isfinite(v) for v in path) or len(set(path)) < 2:
            raise ValueError(
                "The loading ladder needs at least two distinct finite values."
            )
        if property_name == "bulk_modulus" and min(path) <= 0:
            raise ValueError("Bulk pressures must be positive.")
        if property_name == "bulk_modulus" and not math.isclose(
            path[0], spec.pressure_bar
        ):
            raise ValueError("The bulk ladder must start at the preparation pressure.")
        if property_name == "load_modulus" and path[0] != 0.0:
            raise ValueError("The load ladder must start at zero applied stress.")
        if property_name == "shear_modulus" and any(abs(v) >= 0.5 for v in path):
            raise ValueError("Shear strain magnitudes must be below 0.5.")
    plan = ElasticRatePlan(
        equilibration_protocol(spec, **equilibration), groups, holds, property_name
    )
    if spec.max_total_ns is not None and plan.total_ns > spec.max_total_ns:
        raise MechanicalError(
            f"The elastic rate scan is {plan.total_ns:.3g} ns across every rate and replica, "
            f"over the {spec.max_total_ns:.3g} ns budget."
        )
    return plan


def _request(
    run: RunContext,
    spec: ModulusSpec,
    plan: ElasticRatePlan,
    target_rate: float,
    max_extrapolation_decades: float,
) -> dict[str, Any]:
    """What a scan records and a resume must repeat, as JSON gives it back.

    Young's modulus keeps the request its original scan recorded, key for
    key, so that scans begun before it joined this family still resume.
    """
    fields: dict[str, Any] = (
        {
            "spec": asdict(spec),
            "relax_ps": list(plan.hold_times_ps),
            "equilibration": {"protocol": asdict(plan.equilibration)},
            "target_rate_per_ns": target_rate,
            "max_extrapolation_decades": max_extrapolation_decades,
            "system": asdict(run.spec),
            "seed": run.seed,
        }
        if plan.property_name == "youngs_modulus"
        else {
            "property_name": plan.property_name,
            "spec": asdict(spec),
            "hold_times_ps": plan.hold_times_ps,
            "equilibration": asdict(plan.equilibration),
            "target_rate": target_rate,
            "max_extrapolation_decades": max_extrapolation_decades,
            **run_fingerprint(run),
        }
    )
    request: dict[str, Any] = json.loads(
        json.dumps(fields, allow_nan=False, default=str)
    )
    return request


def run_elastic_rate_scan(
    run: RunContext,
    output_dir: str | Path = "run",
    *,
    property_name: str,
    hold_times_ps: Sequence[float],
    target_rate: float,
    spec: ModulusSpec = DEFAULT_SPEC,
    max_extrapolation_decades: float = 2.0,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> RateReport:
    """Vary only the holds; start each replica from one common relaxed cell.

    All stages get distinct deterministic random streams, with fresh velocities
    for the first stage of each replica. Settings are durable before preparation
    begins, and a changed request cannot silently reuse completed stages.
    Young's modulus keeps the record its original scan kept, and one written
    before scans fingerprinted their common state still resumes.
    """
    plan = validate_elastic_rate_scan(
        spec,
        hold_times_ps,
        property_name=property_name,
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
        **equilibration,
    )
    youngs = property_name == "youngs_modulus"
    directory = Path(output_dir).resolve()
    workflow = directory / (YOUNGS_WORKFLOW_NAME if youngs else WORKFLOW_NAME)
    names = [f"rate_{i:02d}" for i in range(len(plan.protocols))]
    request = _request(run, spec, plan, target_rate, max_extrapolation_decades)
    record = resumable_record(
        run,
        workflow,
        request,
        names,
        resume=resume,
        error=MechanicalError,
        fingerprinted=not youngs,
    )
    record_scan_request(
        workflow,
        record,
        request,
        [directory / "equilibration", *(directory / name for name in names)],
        resume=resume,
        run_dirs=names,
    )
    chains = chain_options(
        chain_backbone, atoms_per_chain, expected_characteristic_ratio
    )
    start, origin = equilibrate(
        plan.equilibration,
        run,
        directory / "equilibration",
        resume=resume,
        error=MechanicalError,
        verb="deform",
        **chains,
    )
    timestep = safe_timestep_fs(spec.temperature_k, run.spec)
    record.update(
        start_state=start,
        start_state_sha256=start_fingerprint(start, record, error=MechanicalError),
        reference_box_nm=origin,
        timestep_fs=timestep,
    )
    write_json(workflow, record)
    for i, (name, hold) in enumerate(zip(names, plan.hold_times_ps, strict=True)):
        rate_spec = _rate_spec(spec, property_name, hold)
        for replica in range(spec.n_replicas):
            protocol = _protocol(
                rate_spec,
                property_name,
                i * spec.n_replicas + replica,
                timestep_fs=timestep,
                reference_box_nm=origin,
            )
            run_protocol(
                protocol,
                run,
                directory / name,
                state_in=start,
                # The first replica creates each rate's fresh manifest;
                # later replicas keep the ones completed before them.
                resume=resume or replica > 0,
                **chains,
            )
    return analyse_elastic_rates(
        [directory],
        property_name=property_name,
        target_rate=target_rate,
        strain_limit=spec.elastic_strain_limit,
        max_extrapolation_decades=max_extrapolation_decades,
    )


def _read_stages(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise AnalysisError(f"No completed rate manifest in {directory}.")
    return dict(json.loads(path.read_text()).get("stages", {}))


def _check_recorded_scan(
    directory: Path, record: dict[str, Any], property_name: str
) -> None:
    request = record["request"]
    spec = ModulusSpec(**request["spec"])
    holds = request.get("hold_times_ps", request.get("relax_ps", []))
    names = record.get("run_dirs", [])
    if len(holds) != len(names):
        raise AnalysisError("Recorded hold and rate directory counts differ.")
    for i, (name, hold) in enumerate(zip(names, holds, strict=True)):
        stages = _read_stages(directory / name)
        rate_spec = _rate_spec(spec, property_name, hold)
        for replica in range(spec.n_replicas):
            protocol = _protocol(
                rate_spec, property_name, i * spec.n_replicas + replica
            )
            for stage in protocol.stages:
                entry = stages.get(stage.name)
                if entry is None:
                    raise AnalysisError(
                        f"{directory / name} is incomplete: missing {stage.name}."
                    )
                key, path, expected_hold = _expected_path(stage)
                samples = entry.get("samples", {})
                for field, expected in (
                    (key, path),
                    ("segment_duration_ps", [expected_hold] * len(path)),
                    ("segment_temperature_k", [spec.temperature_k] * len(path)),
                ):
                    recorded = np.asarray(samples.get(field, []), dtype=float)
                    if recorded.shape != (len(path),) or not np.allclose(
                        recorded, expected, rtol=1e-8, atol=1e-9
                    ):
                        raise AnalysisError(
                            f"{directory / name}/{stage.name} is incomplete or has different {field}."
                        )
                if stage.kind == "shear" and samples.get("shear_plane", [0, 2]) != [
                    0,
                    2,
                ]:
                    raise AnalysisError(
                        "Recorded shear plane differs from the workflow."
                    )
                pressure = samples.get("lateral_pressure_bar")
                if pressure is not None and pressure != [spec.pressure_bar]:
                    raise AnalysisError(
                        "Recorded lateral pressure differs from the workflow."
                    )
                axis_key = {"deform": "deform_axis", "load": "load_axis"}.get(
                    stage.kind
                )
                if axis_key and samples.get(axis_key) != [spec.axis]:
                    raise AnalysisError(
                        "Recorded loading axis differs from the workflow."
                    )
                if stage.kind == "deform" and "reference_box_nm" in record:
                    reference = np.asarray(
                        samples.get("reference_box_nm", []), dtype=float
                    )
                    if reference.shape != (3,) or not np.allclose(
                        reference, record["reference_box_nm"], rtol=1e-8, atol=1e-9
                    ):
                        raise AnalysisError(
                            "Recorded reference box differs from the workflow."
                        )


def _recorded_spec(record: dict[str, Any]) -> dict[str, Any] | None:
    """Carry preparation provenance alongside the measurement's settings."""
    request = record.get("request", {})
    spec = request.get("spec")
    if spec is None:
        return None
    preparation = request.get("equilibration")
    # Young's modulus workflows nest the same protocol one level deeper.
    if isinstance(preparation, dict) and "protocol" in preparation:
        preparation = preparation["protocol"]
    return {
        **spec,
        "_provenance": {
            "preparation": preparation,
            "system_sha256": request.get("system_sha256"),
            "preparation_state_sha256": record.get("start_state_sha256"),
        },
    }


def _expand(
    run_dirs: Sequence[str | Path], property_name: str
) -> list[tuple[Path, dict[str, Any] | None]]:
    result: list[tuple[Path, dict[str, Any] | None]] = []
    for value in run_dirs:
        directory = Path(value).resolve()
        workflow = directory / WORKFLOW_NAME
        if not workflow.is_file() and property_name == "poisson_ratio":
            workflow = directory / YOUNGS_WORKFLOW_NAME
        if workflow.is_file():
            record = json.loads(workflow.read_text())
            request = record.get("request", {})
            if request.get("property_name", "poisson_ratio") != property_name:
                raise AnalysisError(f"{workflow} measures another property.")
            names = record.get("run_dirs", [])
            if not isinstance(names, list) or not names:
                raise AnalysisError(f"{workflow} records no rate run directories.")
            if "spec" in request:
                _check_recorded_scan(directory, record, property_name)
            candidates = [(directory / name, _recorded_spec(record)) for name in names]
        else:
            # A measurement child remains independently analysable with its
            # parent's provenance, including its requested lateral pressure.
            parent = directory.parent / WORKFLOW_NAME
            if not parent.is_file() and property_name == "poisson_ratio":
                parent = directory.parent / YOUNGS_WORKFLOW_NAME
            spec_data = None
            if parent.is_file():
                record = json.loads(parent.read_text())
                if directory.name in record.get("run_dirs", []):
                    if (
                        record["request"].get("property_name", "poisson_ratio")
                        != property_name
                    ):
                        raise AnalysisError(f"{directory} measures another property.")
                    _check_recorded_scan(directory.parent, record, property_name)
                    spec_data = _recorded_spec(record)
            candidates = [(directory, spec_data)]
        for candidate, spec_data in candidates:
            candidate = candidate.resolve()
            _read_stages(candidate)
            result.append((candidate, spec_data))
    require_distinct([path for path, _ in result], what="elastic measurements")
    return result


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _observations(
    directory: Path, name: str, spec: dict[str, Any] | None, strain_limit: float
) -> list[RateObservation]:
    stages = _read_stages(directory)
    key = _PATH_KEYS[name]
    if name == "poisson_ratio":
        groups = group_by_stem(deform_stages(directory))
    else:
        groups = [
            (stage,)
            for stage, entry in stages.items()
            if entry.get("samples", {}).get(key)
            and (
                name != "bulk_modulus"
                or stage == BULK_STEM
                or stage.startswith(BULK_STEM + "_")
            )
        ]
    if not groups:
        raise AnalysisError(f"{directory} has no {name} measurement.")
    observations = []
    for group in groups:
        samples: dict[str, list[float]] = {}
        reference_box: list[float] | None = None
        for stage in group:
            if name == "poisson_ratio":
                reference = stages[stage].get("samples", {}).get("reference_box_nm")
                if reference_box is not None and reference != reference_box:
                    raise AnalysisError(
                        "Loading chunks must share the same reference box."
                    )
                reference_box = reference
            for field, values in stages[stage].get("samples", {}).items():
                samples.setdefault(field, []).extend(values)
        path = np.asarray(samples[key], dtype=float)
        duration = np.asarray(samples.get("segment_duration_ps", []), dtype=float)
        if (
            duration.shape != path.shape
            or not np.all(np.isfinite(duration))
            or np.any(duration <= 0)
        ):
            raise AnalysisError(
                f"{directory}/{group[0]} needs recorded positive durations for every rung."
            )
        if not np.all(np.isfinite(path)) or path.size < 2:
            raise AnalysisError("Every loading path needs at least two finite points.")
        # Deformation and shear start at zero. Pressure starts at the first
        # recorded pressure: its arrival from preparation is not a measured leg.
        origin = path[0] if name == "bulk_modulus" else 0.0
        distance = float(np.abs(np.diff(np.r_[origin, path])).sum())
        rate = distance / float(duration.sum()) * 1000.0
        if not math.isfinite(rate) or rate <= 0:
            raise AnalysisError(
                "Every loading path must record a positive nominal rate."
            )
        notes: list[str] = []
        has_baseline = name != "load_modulus" or path[0] == 0.0
        if not has_baseline:
            notes.append("The load ladder has no initial zero-stress baseline.")
        temperatures = samples.get("segment_temperature_k", [])
        if temperatures:
            if (
                len(temperatures) != path.size
                or not all(math.isfinite(t) for t in temperatures)
                or max(temperatures) - min(temperatures) > 1e-8
            ):
                raise AnalysisError(
                    "Each loading path must hold the same requested temperature."
                )
            temperature = float(temperatures[0])
        else:
            requested = [stages[s].get("temperature_k") for s in group]
            if any(v is None or not math.isfinite(v) for v in requested):
                temperature = None
                notes.append(
                    "Requested temperature is missing; preparation comparability is unverified."
                )
            elif max(requested) - min(requested) > 1e-8:
                raise AnalysisError(
                    "Loading chunks use different requested temperatures."
                )
            else:
                temperature = float(requested[0])
        pressure = spec.get("pressure_bar") if spec else None
        if name == "bulk_modulus":
            pressure = float(path[0])
        elif pressure is None:
            recorded_pressure = samples.get("lateral_pressure_bar", [])
            if recorded_pressure and len(set(recorded_pressure)) == 1:
                pressure = float(recorded_pressure[0])
            else:
                notes.append(
                    "Lateral pressure is missing; preparation comparability is unverified."
                )
        plane: list[float] | None = None
        if name == "shear_modulus":
            plane = samples.get("shear_plane")
            if plane is None and spec is not None:
                plane = [0.0, 2.0]
            if plane is None:
                notes.append("Shear plane is missing; loading direction is unverified.")
            elif (
                len(plane) != 2
                or len(set(plane)) != 2
                or any(v not in (0, 1, 2) for v in plane)
            ):
                raise AnalysisError(
                    "Every shear measurement must record a valid shear plane."
                )
        axis = None
        if name in ("poisson_ratio", "load_modulus"):
            axis_key = "deform_axis" if name == "poisson_ratio" else "load_axis"
            axes = samples.get(axis_key, [])
            if not axes or len(set(axes)) != 1 or axes[0] not in (0, 1, 2):
                raise AnalysisError(
                    "Every tensile measurement must record one valid loading axis."
                )
            axis = int(axes[0])
        try:
            if name == "poisson_ratio":
                fit = poisson_ratio(
                    stress_strain(directory, group), strain_limit=strain_limit
                )
                value, error, resolved = fit.ratio, fit.standard_error, fit.resolved
            elif name == "shear_modulus":
                shear = shear_modulus(directory, group)
                value, error, resolved = (
                    shear.modulus_mpa,
                    shear.standard_error_mpa,
                    shear.resolved,
                )
            elif name == "bulk_modulus":
                bulk = bulk_modulus(directory, group)
                value, error, resolved = (
                    bulk.modulus_mpa,
                    bulk.standard_error_mpa,
                    bulk.resolved,
                )
                notes.append(
                    "Bulk standard error propagates the log-volume slope fit; "
                    "independent replicas are needed to assess preparation variability."
                )
            else:
                load = youngs_modulus(
                    load_curve(directory, group),
                    strain_limit=strain_limit,
                    min_points=2,
                )
                value, error, resolved = (
                    load.modulus_mpa,
                    load.standard_error_mpa,
                    load.resolved,
                )
        except (AnalysisError, KeyError, ValueError) as exc:
            value, error, resolved = math.nan, None, False
            notes.append(f"Measurement cannot be fitted: {exc}")
        conditions: dict[str, Any] = {
            **(
                (spec or {}).get("_provenance")
                or {
                    "preparation": None,
                    "system_sha256": None,
                    "preparation_state_sha256": None,
                }
            ),
            "loading_path": path.tolist(),
            "axis": axis,
            "shear_plane": plane,
            "pressure_bar": pressure,
            "relative_holds": (duration / duration.sum()).tolist(),
            "strain_limit": strain_limit
            if name in ("poisson_ratio", "load_modulus")
            else None,
            "rate_definition": "total absolute loading path / total hold time",
        }
        observations.append(
            RateObservation(
                rate=rate,
                value=_finite(value),
                standard_error=None if error is None else _finite(error),
                resolved=bool(
                    resolved
                    and temperature is not None
                    and pressure is not None
                    and (name != "shear_modulus" or plane is not None)
                    and has_baseline
                ),
                temperature_k=temperature,
                conditions=conditions,
                source=f"{directory}: {', '.join(group)}",
                notes=tuple(notes),
            )
        )
    return observations


# --------------------------------------------------------------------------
# Young's modulus: one pooled fit per rate, from its own scan record
# --------------------------------------------------------------------------


def _youngs_directories(run_dirs: Sequence[str | Path]) -> list[Path]:
    """Each scan root's recorded rate directories, or the directory itself."""
    directories: list[Path] = []
    for value in run_dirs:
        directory = Path(value).resolve()
        workflow = directory / YOUNGS_WORKFLOW_NAME
        if not workflow.is_file():
            _read_stages(directory)
            directories.append(directory)
            continue
        record = json.loads(workflow.read_text())
        names = record.get("run_dirs")
        if not isinstance(names, list) or not names:
            raise AnalysisError(f"{workflow} records no rate run directories.")
        rate_dirs = [(directory / str(name)).resolve() for name in names]
        for rate_dir in rate_dirs:
            _read_stages(rate_dir)
        _check_youngs_scan(workflow, record, rate_dirs)
        directories.extend(rate_dirs)
    require_distinct(directories, what="strain-rate measurements")
    return directories


def _check_youngs_scan(
    workflow: Path, record: dict[str, Any], directories: Sequence[Path]
) -> None:
    """A partial rate series cannot become a complete analysis by accident.

    Looser than the other properties' check, which these scans predate: a
    requested temperature is compared only where one was recorded.
    """
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
        entries = _read_stages(directory)
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
                expected = strain_ladder(
                    float(stage.options["strain_start"]),
                    spec.strain_increment,
                    int(stage.options["n_steps"]),
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
    """Do not let pooling disguise a different axis, ladder or temperature."""
    reference = curves[0]
    for curve in curves:
        if curve.axis != reference.axis:
            raise AnalysisError(
                "All rate measurements must use the same deformation axis."
            )
        if curve.strain.shape != reference.strain.shape or not np.allclose(
            curve.strain, reference.strain, rtol=1.0e-6, atol=1.0e-9
        ):
            raise AnalysisError(
                "All rate measurements must use the same strain increments and range."
            )
        rate = curve.strain_rate_per_ns
        if rate is None or not math.isfinite(rate) or rate <= 0:
            raise AnalysisError(
                "Every deformation must record a finite positive strain rate."
            )
    temperatures = [curve.temperature_k for curve in curves]
    if (
        not all(math.isfinite(value) for value in temperatures)
        or max(temperatures) - min(temperatures) > _TEMPERATURE_TOLERANCE_K
    ):
        raise AnalysisError("All rate measurements must use the same temperature.")


def _youngs_observations(
    directories: Sequence[Path], strain_limit: float
) -> list[RateObservation]:
    """One observation per distinct rate, fitted through all its replicas' points.

    The pooled fit is the one a single modulus scan reports. Its own error
    sees only how straight the pooled points lie, so the replicas' sample
    spread is its floor, and a rate is unresolved if any replica is, or if
    they spread by more than
    :data:`~openmmpolymer.mechanical.MAX_REPLICA_SPREAD` of the pooled value.
    """
    curves: list[tuple[str, StressStrain]] = []
    for directory in directories:
        try:
            groups = group_by_stem(deform_stages(directory))
        except AnalysisError:
            groups = []
        if not groups:
            raise AnalysisError(
                f"{directory} has no strain-controlled extension to fit."
            )
        for group in groups:
            curve = stress_strain(directory, group)
            curves.append((f"{directory}: {curve.stage}", curve))
    _check_comparable([curve for _, curve in curves])
    by_rate: list[list[tuple[str, StressStrain]]] = []
    for item in sorted(curves, key=lambda item: float(item[1].strain_rate_per_ns or 0)):
        if by_rate and math.isclose(
            float(item[1].strain_rate_per_ns or 0),
            float(by_rate[-1][0][1].strain_rate_per_ns or 0),
            rel_tol=1.0e-8,
        ):
            by_rate[-1].append(item)
        else:
            by_rate.append([item])
    observations: list[RateObservation] = []
    for members in by_rate:
        replicas = [curve for _, curve in members]
        pooled = _pool(replicas)
        assert pooled is not None and pooled.strain_rate_per_ns is not None
        fit = youngs_modulus(pooled, strain_limit=strain_limit)
        fits = [youngs_modulus(curve, strain_limit=strain_limit) for curve in replicas]
        spread = sample_spread([replica.modulus_mpa for replica in fits])
        resolved = (
            fit.resolved
            and all(replica.resolved for replica in fits)
            and (spread is None or spread <= MAX_REPLICA_SPREAD * abs(fit.modulus_mpa))
        )
        rate = pooled.strain_rate_per_ns
        notes: list[str] = []
        if spread is None:
            notes.append(
                f"Rate {rate:.6g} /ns has one replica; no replica spread is available."
            )
        if not resolved:
            notes.append(
                f"Rate {rate:.6g} /ns is unresolved, including replica quality and spread."
            )
        observations.append(
            RateObservation(
                rate=rate,
                value=fit.modulus_mpa,
                standard_error=max(fit.standard_error_mpa, spread or 0.0),
                resolved=resolved,
                temperature_k=fit.temperature_k,
                conditions={"strain_limit": fit.strain_limit},
                source="; ".join(source for source, _ in members),
                notes=tuple(notes),
            )
        )
    return observations


def analyse_elastic_rates(
    run_dirs: Sequence[str | Path],
    *,
    property_name: str,
    target_rate: float,
    strain_limit: float = 0.015,
    max_extrapolation_decades: float = 2.0,
) -> RateReport:
    """Fit logarithmic and power-law responses to matching saved loading paths.

    Repeated rates are replicas, not extra rate-fit points. Missing uncertainty
    stays unknown, and unresolved primitive measurements remain unresolved.
    Poisson ratios may also be read from a Young's modulus rate scan. Young's
    moduli come from their own scans, or from modulus runs given directly,
    one pooled observation per rate.
    """
    prop = _property(property_name)
    validate_rate_request(target_rate, max_extrapolation_decades)
    require_positive(strain_limit, None, name="strain_limit")
    if property_name == "youngs_modulus":
        directories = _youngs_directories(run_dirs)
        observations = _youngs_observations(directories, strain_limit)
    else:
        expanded = _expand(run_dirs, property_name)
        directories = [directory for directory, _ in expanded]
        observations = [
            item
            for directory, spec in expanded
            for item in _observations(directory, property_name, spec, strain_limit)
        ]
    return analyse_rate_observations(
        observations,
        property=prop,
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
        run_dirs=[str(directory) for directory in directories],
    )
