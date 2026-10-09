"""Rate sensitivity of offset yield, ultimate strength and elongation at break.

Replicas retain their individual event locations: pooling their stress-strain
points would erase the variation this analysis needs to measure.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._files import write_json
from ._fitting import MAX_EXTRAPOLATION_DECADES
from ._seeds import derive_seed
from ._validation import require_choice
from ._workflow import (
    chain_options,
    enforce_budget,
    equilibrate,
    rate_request,
    record_scan_request,
    require_distinct,
    resumable_record,
    run_branches,
    run_fingerprint,
    start_fingerprint,
    strain_ladder,
    validate_hold_times,
)
from .chain import DEFAULT_CHARACTERISTIC_RATIO
from .protocols import (
    Protocol,
    RunManifest,
    standard_melt_equilibration,
)
from .rate_dependence import (
    RateObservation,
    RateProperty,
    RateReport,
    analyse_rate_observations,
    validate_rate_request,
)
from .simulate import RunContext, safe_timestep_fs
from .tensile import (
    BREAKING,
    ELONGATION,
    YIELD,
    BreakingSpec,
    ElongationSpec,
    TensileMeasurement,
    TensileSpec,
    YieldSpec,
    analyse_breaking,
    analyse_elongation,
    analyse_yield,
    tensile_protocol,
    tensile_schedule,
    workflow_record,
)
from .trajectory import AnalysisError

WORKFLOW_NAME = "tensile_rate_workflow.json"
TENSILE_RATE_PROPERTIES = {
    "yield_strength": RateProperty(
        "yield_strength",
        "Offset yield strength",
        "MPa",
        "strain/ns",
        trend="increasing",
    ),
    "yield_strain": RateProperty(
        "yield_strain", "Offset yield strain", "strain", "strain/ns"
    ),
    "breaking_strength": RateProperty(
        "breaking_strength", "Apparent ultimate tensile strength", "MPa", "strain/ns"
    ),
    "elongation_at_break": RateProperty(
        "elongation_at_break", "Apparent elongation at break", "%", "strain/ns"
    ),
}

#: For each property, the tensile measurement whose scans record it, how a
#: finished run of that measurement is read, and which replica fit field
#: holds the property.
_EVENTS: dict[str, tuple[TensileMeasurement[Any], Callable[..., Any], str]] = {
    "yield_strength": (YIELD, analyse_yield, "strength_mpa"),
    "yield_strain": (YIELD, analyse_yield, "yield_strain"),
    "breaking_strength": (BREAKING, analyse_breaking, "strength_mpa"),
    "elongation_at_break": (ELONGATION, analyse_elongation, "elongation_percent"),
}


def _event(
    property_name: str,
) -> tuple[TensileMeasurement[Any], Callable[..., Any], str]:
    require_choice(
        property_name,
        tuple(_EVENTS),
        name="property_name",
        message=lambda: f"Unknown tensile rate property {property_name!r}.",
    )
    return _EVENTS[property_name]


def tensile_rate_spec_type(
    property_name: str,
) -> type[BreakingSpec | ElongationSpec | YieldSpec]:
    """Settings type supplied by the same descriptor used to run and read events."""
    return cast(
        type[BreakingSpec | ElongationSpec | YieldSpec], _event(property_name)[0].spec
    )


@dataclass(frozen=True)
class TensileRatePlan:
    """One equilibration plus each rate's complete replica ladders."""

    property_name: str
    equilibration: Protocol
    specs: tuple[TensileSpec, ...]
    total_ns: float


def validate_tensile_rate_scan(
    spec: TensileSpec,
    hold_times_ps: Sequence[float],
    *,
    property_name: str,
    target_rate: float,
    max_extrapolation_decades: float = MAX_EXTRAPOLATION_DECADES,
    **equilibration: Any,
) -> TensileRatePlan:
    """Validate all holds and the total budget before writing or building MD."""
    measurement = _event(property_name)[0]
    if type(spec) is not measurement.spec:
        raise ValueError(f"{property_name} requires {measurement.spec.__name__}.")
    validate_rate_request(target_rate, max_extrapolation_decades)
    holds = validate_hold_times(hold_times_ps)
    # A spec refuses a stage shorter than one hold, and the budget below
    # covers the whole scan rather than any one rate.
    specs = tuple(
        replace(
            spec, relax_ps=hold, stage_ps=max(spec.stage_ps, hold), max_total_ns=None
        )
        for hold in holds
    )
    settle = standard_melt_equilibration(
        target_temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        **equilibration,
    )
    cost = (
        settle.total_duration_ps
        + sum(item.n_replicas * tensile_schedule(item).total_ps for item in specs)
    ) / 1000
    enforce_budget(
        cost,
        spec.max_total_ns,
        error=ValueError,
        message=lambda: (
            f"Tensile rate scan needs {cost:.3g} ns across all rates and replicas, above max_total_ns={spec.max_total_ns:g}."
        ),
    )
    return TensileRatePlan(property_name, settle, specs, cost)


def run_tensile_rate_scan(
    run: RunContext,
    output_dir: str | Path = "run",
    *,
    property_name: str,
    hold_times_ps: Sequence[float],
    target_rate: float,
    spec: TensileSpec | None = None,
    max_extrapolation_decades: float = MAX_EXTRAPOLATION_DECADES,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = DEFAULT_CHARACTERISTIC_RATIO,
    **equilibration: Any,
) -> RateReport:
    """Vary hold time from one equilibrated cell, retaining all failure criteria.

    Every rate has an independently derived seed. Every replica starts from
    the same coordinates with fresh velocities. Unobserved yield or terminal
    stress loss remains censored; the scan never fabricates an event by
    extrapolating its stress-strain curve. Engine failures propagate. A forced
    rerun (``resume=False``) replaces the saved request instead of comparing
    it.
    """
    measurement = _event(property_name)[0]
    selected: TensileSpec = measurement.spec() if spec is None else spec
    plan = validate_tensile_rate_scan(
        selected,
        hold_times_ps,
        property_name=property_name,
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
        **equilibration,
    )
    directory = Path(output_dir).resolve()
    workflow = directory / WORKFLOW_NAME
    names = [f"rate_{index:02d}" for index in range(len(plan.specs))]
    request = rate_request(
        {
            "property_name": property_name,
            "specs": [asdict(item) for item in plan.specs],
            "equilibration": asdict(plan.equilibration),
            **run_fingerprint(run),
        }
    )
    record = resumable_record(
        run, workflow, request, names, resume=resume, error=ValueError
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
        error=ValueError,
        verb="strain",
        **chains,
    )
    fingerprint = start_fingerprint(start, record, error=ValueError)
    record.update(
        start_state=start, start_state_sha256=fingerprint, reference_box_nm=origin
    )
    write_json(workflow, record)
    timestep = safe_timestep_fs(selected.temperature_k, run.spec)
    for index, (name, rate_spec) in enumerate(zip(names, plan.specs, strict=True)):
        rate_dir = directory / name
        rate_dir.mkdir(parents=True, exist_ok=True)
        ladders = tuple(
            tensile_protocol(
                rate_spec,
                timestep_fs=timestep,
                replica=replica,
                reference_box_nm=origin,
            )
            for replica in range(rate_spec.n_replicas)
        )
        # Each rate directory is an ordinary scan of its measurement, which
        # that measurement's own analysis reads under the criterion it ran.
        write_json(
            rate_dir / measurement.workflow_name,
            workflow_record(
                {
                    "spec": asdict(rate_spec),
                    "system_sha256": request["system_sha256"],
                    "system": request["system"],
                    "equilibration": request["equilibration"]["stages"],
                    "preparation_state_sha256": fingerprint,
                },
                rate_spec,
                ladders,
                timestep,
                state={"start_state": start, "reference_box_nm": origin},
            ),
        )
        rate_run = replace(run, seed=derive_seed(run.seed, "tensile_rates", str(index)))
        run_branches(ladders, rate_run, rate_dir, start, resume_first=resume, **chains)
    return analyse_tensile_rates(
        [directory],
        property_name=property_name,
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
    )


def _directories(
    run_dirs: Sequence[str | Path],
    property_name: str,
    measurement: TensileMeasurement[Any],
) -> list[Path]:
    result: list[Path] = []
    for raw in run_dirs:
        root = Path(raw).resolve()
        path = root / WORKFLOW_NAME
        if not path.is_file():
            result.append(root)
            continue
        record = json.loads(path.read_text())
        recorded_property = record.get("request", {}).get("property_name")
        # A yield scan measures its strength and its strain alike.
        if recorded_property != property_name and {
            recorded_property,
            property_name,
        } != {"yield_strength", "yield_strain"}:
            raise AnalysisError(
                f"{path} records {recorded_property}, not {property_name}."
            )
        names = record.get("run_dirs", [])
        if not names:
            raise AnalysisError(f"{path} records no rate directories.")
        candidates = [root / name for name in names]
        specs = record.get("request", {}).get("specs", [])
        if len(specs) != len(candidates):
            raise AnalysisError(f"{path} has inconsistent requested rate counts.")
        for candidate, spec in zip(candidates, specs, strict=True):
            child = candidate / measurement.workflow_name
            if (
                not child.is_file()
                or json.loads(child.read_text()).get("request", {}).get("spec") != spec
            ):
                raise AnalysisError(
                    f"{candidate} has missing or different rate settings."
                )
        result.extend(candidate.resolve() for candidate in candidates)
    for candidate in result:
        if not (candidate / "manifest.json").is_file():
            raise AnalysisError(f"Missing rate manifest in {candidate}.")
    require_distinct(result, what="tensile measurements")
    return result


def analyse_tensile_rates(
    run_dirs: Sequence[str | Path],
    *,
    property_name: str,
    target_rate: float,
    max_extrapolation_decades: float = MAX_EXTRAPOLATION_DECADES,
) -> RateReport:
    """Compare saved scans with matching ladders and event criteria.

    Single-replica event uncertainty is unknown. Independent repeated rates
    supply the between-replica spread; event brackets remain diagnostics and
    are never relabelled as standard errors. All censored replicas survive
    in the report and prevent a confidently extrapolated event, and a scan
    missing replicas or chunks is still read but never resolves.
    """
    measurement, analyse, field = _event(property_name)
    validate_rate_request(target_rate, max_extrapolation_decades)
    observations: list[RateObservation] = []
    directories = _directories(run_dirs, property_name, measurement)
    for directory in directories:
        manifest = RunManifest.load(directory)
        assert manifest is not None
        report = analyse(directory, manifest=manifest)
        workflow = directory / measurement.workflow_name
        request = (
            json.loads(workflow.read_text()).get("request", {})
            if workflow.is_file()
            else {}
        )
        settings = request.get("spec", {})
        if settings:
            _validate_recorded_ladder(directory, measurement.spec(**settings), manifest)
        conditions = {
            key: value
            for key, value in settings.items()
            if key
            not in {
                "relax_ps",
                "stage_ps",
                "n_replicas",
                "samples_per_step",
                "trajectory_ps",
                "max_total_ns",
                "temperature_k",
            }
        }
        if "system_sha256" in request:
            conditions["system_sha256"] = request["system_sha256"]
        if request.get("equilibration") is not None:
            conditions["preparation"] = request["equilibration"]
        if request.get("preparation_state_sha256") is not None:
            conditions["preparation_state_sha256"] = request["preparation_state_sha256"]
        if request.get("system") or manifest.system:
            conditions["system"] = request.get("system", manifest.system)
        if manifest.box is not None:
            conditions["composition"] = {
                key: manifest.box[key]
                for key in ("n_molecules", "atoms_per_chain")
                if key in manifest.box
            }
        conditions.update({key: getattr(report, key) for key in measurement.criterion})
        for index, curve, fit in zip(
            report.replica_indices, report.curves, report.replicas, strict=True
        ):
            rate = curve.strain_rate_per_ns
            if rate is None or not math.isfinite(rate) or rate <= 0:
                raise AnalysisError(
                    f"{directory} has a missing or invalid strain rate."
                )
            notes = list(fit.notes)
            notes.extend(report.notes)
            for key in ("preparation", "system_sha256", "composition"):
                if key not in conditions:
                    notes.append(
                        f"Recorded {key} is unavailable; cross-rate comparability is not fully verified."
                    )
            bracket = next(
                (
                    getattr(fit, name)
                    for name in ("break_bracket", "failure_bracket", "yield_bracket")
                    if getattr(fit, name, None) is not None
                ),
                None,
            )
            if bracket is not None:
                notes.append(
                    f"Sampled event bracket {bracket} is not a standard error."
                )
            observations.append(
                RateObservation(
                    rate=rate,
                    value=getattr(fit, field),
                    standard_error=None,
                    resolved=bool(report.resolved and fit.resolved),
                    temperature_k=curve.temperature_k,
                    conditions={
                        **conditions,
                        "axis": curve.axis,
                        "strain_grid": np.round(curve.strain, 10).tolist(),
                    },
                    source=f"{directory}:replica_{index}",
                    notes=tuple(dict.fromkeys(notes)),
                )
            )
    return analyse_rate_observations(
        observations,
        property=TENSILE_RATE_PROPERTIES[property_name],
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
        run_dirs=tuple(str(item) for item in directories),
    )


def _validate_recorded_ladder(
    directory: Path, spec: TensileSpec, manifest: RunManifest
) -> None:
    """Check physical samples against the saved request, not just point counts."""
    for replica in range(spec.n_replicas):
        for stage in tensile_protocol(spec, replica=replica).stages:
            if stage.name not in manifest.stages:
                # The ordinary reader retains missing chunks as unresolved.
                continue
            samples = manifest.stages[stage.name].get("samples", {})
            strains = np.asarray(samples.get("segment_strain", []), dtype=float)
            count = stage.options["n_steps"]
            expected = strain_ladder(
                stage.options["strain_start"], spec.strain_increment, count
            )
            if (
                strains.ndim != 1
                or strains.size > count
                or not np.allclose(
                    strains, expected[: strains.size], rtol=1e-8, atol=1e-10
                )
            ):
                raise AnalysisError(
                    f"{directory}/{stage.name} has a different recorded strain ladder."
                )
            holds = np.asarray(samples.get("segment_duration_ps", []), dtype=float)
            if holds.shape != strains.shape or not np.allclose(
                holds, spec.relax_ps, rtol=1e-8, atol=0
            ):
                raise AnalysisError(
                    f"{directory}/{stage.name} has different recorded hold times."
                )
            if samples.get("deform_axis") != [spec.axis]:
                raise AnalysisError(
                    f"{directory}/{stage.name} has a different recorded axis."
                )
            for key, requested in (
                ("segment_temperature_k", spec.temperature_k),
                ("lateral_pressure_bar", spec.pressure_bar),
            ):
                if key in samples and not np.allclose(
                    samples[key], requested, rtol=0, atol=1e-8
                ):
                    raise AnalysisError(
                        f"{directory}/{stage.name} has different recorded {key}."
                    )
