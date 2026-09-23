"""Comparable cooling/heating histories and finite thermal-rate estimates.

Rates are kelvin per nanosecond, never strain rates. Glass transitions use
one fixed cooling ladder; melting uses a supplied crystal and one fixed
heating ladder. Both empirical extrapolations remain finite-rate estimates.
In particular, a heating extrapolation does not establish equilibrium melting.
The existing two-pass Tg scan and its log-linear/VFT analysis are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ._validation import require_integer, require_positive
from .protocols import (
    Protocol,
    RunManifest,
    Stage,
    _write_atomically,
    run_protocol,
)
from .rate_dependence import (
    RateObservation,
    RateProperty,
    RateReport,
    analyse_rate_observations,
)
from .reporters import TrajectoryOptions
from .simulate import RunContext, safe_timestep_fs
from .tg import TgSpec, _group_passes, coarse_schedule, tg_coarse_scan
from .timeseries import glass_transition, quench_curve, quench_stages
from .tm import TmSpec, heating_curve, melting_scan, melting_temperature
from .trajectory import AnalysisError

WORKFLOW_NAME = "thermal_rate_workflow.json"
PROTOCOL_NAME = "thermal_rate_scan"

THERMAL_RATE_PROPERTIES = {
    "glass_transition": RateProperty(
        "glass_transition",
        "Glass transition temperature",
        "K",
        "K/ns",
        trend="increasing",
    ),
    "melting_temperature": RateProperty(
        "melting_temperature",
        "Apparent melting temperature",
        "K",
        "K/ns",
        trend="increasing",
    ),
}


class ThermalRateError(RuntimeError):
    """A thermal series is unsafe, over budget, or conflicts with saved inputs."""


@dataclass(frozen=True)
class ThermalRatePlan:
    """A shared preparation and all histories charged to the time budget."""

    property_name: str
    equilibration: Protocol
    protocols: tuple[Protocol, ...]
    temperatures_k: tuple[float, ...]
    hold_times_ps: tuple[float, ...]
    n_replicas: int

    @property
    def total_ns(self) -> float:
        """All dynamics, including every hold in every replica."""
        return (
            self.equilibration.total_duration_ps
            + self.n_replicas * sum(item.total_duration_ps for item in self.protocols)
        ) / 1000.0


def _property(property_name: str) -> RateProperty:
    try:
        return THERMAL_RATE_PROPERTIES[property_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown thermal property {property_name!r}; choose "
            f"{', '.join(THERMAL_RATE_PROPERTIES)}."
        ) from error


def _analysis_options(target_rate: float, max_extrapolation_decades: float) -> None:
    require_positive(target_rate, None, name="target_rate")
    if not math.isfinite(max_extrapolation_decades) or max_extrapolation_decades < 0:
        raise ValueError("max_extrapolation_decades must be finite and nonnegative.")


def _thermal_protocol(
    temperatures: tuple[float, ...], hold: float, spec: TgSpec | TmSpec
) -> Protocol:
    size = max(1, int(spec.stage_ps // hold))
    chunks = [
        temperatures[index : index + size]
        for index in range(0, len(temperatures), size)
    ]
    if isinstance(spec, TgSpec) and len(chunks) > 1 and len(chunks[-1]) == 1:
        tail = chunks.pop()
        chunks[-1] += tail
    stages = []
    for index, chunk in enumerate(chunks):
        options: dict[str, Any] = {
            "temperatures_k": chunk,
            "hold_ps": hold,
            "pressure_bar": spec.pressure_bar,
            "samples_per_segment": spec.samples_per_segment,
            "report_interval_ps": min(10.0, hold / spec.samples_per_segment),
            "new_velocities": index == 0,
        }
        if isinstance(spec, TmSpec):
            options["barostat"] = spec.barostat
            if spec.trajectory_ps is not None:
                options["trajectory"] = TrajectoryOptions(
                    "xtc", interval_ps=spec.trajectory_ps
                )
        stages.append(
            Stage(
                f"thermal_{index:03d}",
                "quench" if isinstance(spec, TgSpec) else "heat",
                options,
            )
        )
    return Protocol(PROTOCOL_NAME, tuple(stages))


def validate_thermal_rate_scan(
    spec: TgSpec | TmSpec,
    hold_times_ps: Sequence[float],
    *,
    property_name: str,
    target_rate: float,
    n_replicas: int = 3,
    max_extrapolation_decades: float = 2.0,
    **equilibration: Any,
) -> ThermalRatePlan:
    """Validate three or more rates and the full budget without filesystem writes.

    Tg uses ``melt_temperature_k``, ``t_floor_k`` and ``coarse_step_k`` from
    its spec as a fixed ladder. Its adaptive fine-window settings are unused.
    Tm uses its ordinary heating ladder. Preparation runs only once; the
    initial temperature is nevertheless held in every measurement history.
    """
    _property(property_name)
    _analysis_options(target_rate, max_extrapolation_decades)
    n_replicas = require_integer(n_replicas, name="n_replicas")
    holds = tuple(
        require_positive(value, None, name="hold_times_ps") for value in hold_times_ps
    )
    if len(holds) < 3 or any(
        math.isclose(a, b, rel_tol=1e-8) for a, b in pairwise(sorted(holds))
    ):
        raise ValueError("hold_times_ps needs at least three distinct positive holds.")
    if property_name == "glass_transition":
        if not isinstance(spec, TgSpec):
            raise ValueError("glass_transition requires a TgSpec.")
        temperatures = coarse_schedule(spec).temperatures_k
        if len(temperatures) < 2 * spec.min_points_per_branch:
            raise ValueError(
                "The cooling ladder needs twice min_points_per_branch temperatures."
            )
        preparation = tg_coarse_scan(spec, **equilibration)
        preparation = replace(
            preparation,
            stages=tuple(
                stage for stage in preparation.stages if stage.kind != "quench"
            ),
        )
    else:
        if not isinstance(spec, TmSpec):
            raise ValueError("melting_temperature requires a TmSpec.")
        if equilibration:
            raise ValueError("Tm preparation is specified by TmSpec.equilibration_ps.")
        temperatures = spec.temperatures_k
        preparation = melting_scan(replace(spec, max_total_ns=None))
        preparation = replace(preparation, stages=preparation.stages[:2])
    plan = ThermalRatePlan(
        property_name,
        preparation,
        tuple(_thermal_protocol(temperatures, hold, spec) for hold in holds),
        temperatures,
        holds,
        n_replicas,
    )
    if spec.max_total_ns is not None and plan.total_ns > spec.max_total_ns:
        raise ThermalRateError(
            f"Thermal rate scan requires {plan.total_ns:.3g} ns across all rates and "
            f"replicas, over the {spec.max_total_ns:.3g} ns budget."
        )
    return plan


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(directory: Path, record: dict[str, Any]) -> None:
    _write_atomically(
        directory / WORKFLOW_NAME, json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def _check_system(run: RunContext) -> None:
    import openmm as mm

    system = mm.XmlSerializer.deserialize(run.system_xml)
    if any(
        "Barostat" in type(force).__name__ or isinstance(force, mm.AndersenThermostat)
        for force in system.getForces()
    ):
        raise ThermalRateError(
            "The supplied System must contain no barostat or Andersen thermostat; "
            "the thermal stages provide their own temperature and pressure control."
        )


def _replica_protocol(
    protocol: Protocol, rate: int, replica: int, timestep: float
) -> Protocol:
    return replace(
        protocol,
        stages=tuple(
            replace(
                stage,
                name=f"r{rate:02d}_rep{replica:02d}_{stage.name}",
                options={**stage.options, "timestep_fs": timestep},
            )
            for stage in protocol.stages
        ),
    )


def run_thermal_rate_scan(
    run: RunContext,
    output_dir: str | Path = "run",
    *,
    property_name: str,
    hold_times_ps: Sequence[float],
    target_rate: float,
    spec: TgSpec | TmSpec | None = None,
    n_replicas: int = 3,
    max_extrapolation_decades: float = 2.0,
    state_in: str | Path | None = None,
    crystalline: bool = False,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> RateReport:
    """Prepare once and branch every rate/replica from the same saved state.

    ``crystalline=True`` is required for Tm and asserts supplied crystalline
    coordinates, including ``state_in`` when given. This is not an automated
    crystallinity test. Tg prepares a melt; Tm minimises and settles the crystal
    cold without a melt preparation. Each history redraws velocities and uses
    distinct random-stream labels. Settings and input fingerprints are saved
    atomically before any dynamics, and resume rejects changed inputs.

    A common preparation does not itself prove equilibration or independent
    morphology. Replicas share coordinates and probe dynamical variability.
    """
    _property(property_name)
    if property_name == "melting_temperature" and not crystalline:
        raise ThermalRateError(
            "Tm requires supplied crystalline coordinates; pass crystalline=True only for a prepared crystal."
        )
    if spec is None:
        spec = TgSpec() if property_name == "glass_transition" else TmSpec()
    plan = validate_thermal_rate_scan(
        spec,
        hold_times_ps,
        property_name=property_name,
        target_rate=target_rate,
        n_replicas=n_replicas,
        max_extrapolation_decades=max_extrapolation_decades,
        **equilibration,
    )
    n_replicas = plan.n_replicas
    _check_system(run)
    timestep = safe_timestep_fs(max(plan.temperatures_k), run.spec)
    entries = []
    for rate_index, protocol in enumerate(plan.protocols):
        for replica in range(n_replicas):
            actual = _replica_protocol(protocol, rate_index, replica, timestep)
            entries.append(
                {
                    "directory": f"rate_{rate_index:02d}/replica_{replica:02d}",
                    "hold_ps": plan.hold_times_ps[rate_index],
                    "stages": [stage.name for stage in actual.stages],
                    "protocol": asdict(actual),
                }
            )
    request = json.loads(
        json.dumps(
            {
                "property_name": property_name,
                "spec": asdict(spec),
                "hold_times_ps": plan.hold_times_ps,
                "n_replicas": n_replicas,
                "equilibration": asdict(plan.equilibration),
                "target_rate": target_rate,
                "max_extrapolation_decades": max_extrapolation_decades,
                "system_spec": asdict(run.spec),
                "system_sha256": hashlib.sha256(run.system_xml.encode()).hexdigest(),
                "coordinates_sha256": hashlib.sha256(
                    np.asarray(run.box.positions_nm, dtype=np.float64).tobytes()
                ).hexdigest(),
                "box_nm": list(run.box.box_nm),
                "seed": run.seed,
                "state_sha256": None if state_in is None else _digest(state_in),
                "crystalline_supplied": crystalline
                if property_name == "melting_temperature"
                else None,
                "chain_backbone": chain_backbone,
                "atoms_per_chain": atoms_per_chain,
                "expected_characteristic_ratio": expected_characteristic_ratio,
            },
            allow_nan=False,
            default=str,
        )
    )
    directory = Path(output_dir).resolve()
    workflow = directory / WORKFLOW_NAME
    previous: dict[str, Any] = {}
    if workflow.is_file():
        previous = json.loads(workflow.read_text())
        if previous.get("request") != request:
            raise ThermalRateError(
                "Thermal rate settings or starting inputs changed; use a fresh directory."
            )
    elif directory.exists() and any(directory.rglob("manifest.json")):
        raise ThermalRateError(
            "Existing runs lack the thermal rate workflow record; use a fresh directory."
        )
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "request": request,
        "entries": entries,
        "temperatures_k": plan.temperatures_k,
        "total_ns": plan.total_ns,
    }
    if resume and "start_state_sha256" in previous:
        record.update(
            {name: previous[name] for name in ("start_state", "start_state_sha256")}
        )
    _save(directory, record)
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    settled = run_protocol(
        plan.equilibration,
        run,
        directory / "equilibration",
        resume=resume,
        state_in=state_in,
        **chains,
    )
    start = Path(settled.final_state)
    if not start.is_file():
        raise ThermalRateError("The common preparation did not save a final state.")
    if resume and record.get("start_state_sha256", _digest(start)) != _digest(start):
        raise ThermalRateError(
            "The common preparation state changed; use a fresh directory."
        )
    record.update(start_state=str(start), start_state_sha256=_digest(start))
    _save(directory, record)
    for rate_index, protocol in enumerate(plan.protocols):
        for replica in range(n_replicas):
            run_protocol(
                _replica_protocol(protocol, rate_index, replica, timestep),
                run,
                directory / f"rate_{rate_index:02d}/replica_{replica:02d}",
                resume=resume,
                state_in=start,
                **chains,
            )
    return analyse_thermal_rates(
        [directory],
        property_name=property_name,
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
    )


def _check_completed(directory: Path, entry: dict[str, Any]) -> None:
    manifest = RunManifest.load(directory)
    if manifest is None:
        raise AnalysisError(f"Missing thermal rate manifest in {directory}.")
    stages = entry["protocol"]["stages"]
    if set(manifest.stages) != {stage["name"] for stage in stages}:
        raise AnalysisError(f"Incomplete or unexpected thermal stages in {directory}.")
    for stage in stages:
        samples = manifest.stages[stage["name"]].get("samples", {})
        temperatures = np.asarray(stage["options"]["temperatures_k"], dtype=float)
        required = ["segment_density_g_cm3"]
        if stage["kind"] == "heat":
            required.extend(("segment_enthalpy_kj_mol", "segment_pressure_bar"))
        for name in required:
            values = np.asarray(samples.get(name, []), dtype=float)
            if (
                values.shape != temperatures.shape
                or not np.all(np.isfinite(values))
                or (name != "segment_enthalpy_kj_mol" and np.any(values <= 0))
            ):
                raise AnalysisError(
                    f"{directory}/{stage['name']} has incomplete or invalid {name}."
                )
        for name, expected in (
            ("segment_temperature_k", temperatures),
            ("segment_duration_ps", np.full(len(temperatures), entry["hold_ps"])),
        ):
            values = np.asarray(samples.get(name, []), dtype=float)
            if values.shape != expected.shape or not np.allclose(
                values, expected, rtol=1e-8, atol=0
            ):
                raise AnalysisError(
                    f"{directory}/{stage['name']} has incomplete or changed thermal ladder/holds."
                )
        if "segment_pressure_bar" in samples:
            pressures = np.asarray(samples["segment_pressure_bar"], dtype=float)
            if pressures.shape != temperatures.shape or not np.allclose(
                pressures, stage["options"]["pressure_bar"], rtol=1e-8
            ):
                raise AnalysisError(
                    f"{directory}/{stage['name']} has a changed pressure."
                )


def _directories(
    run_dirs: Sequence[str | Path], property_name: str
) -> list[tuple[Path, dict[str, Any] | None, dict[str, Any] | None]]:
    result: list[tuple[Path, dict[str, Any] | None, dict[str, Any] | None]] = []
    seen: set[Path] = set()
    for value in run_dirs:
        directory = Path(value).resolve()
        workflow = directory / WORKFLOW_NAME
        if workflow.is_file():
            record = json.loads(workflow.read_text())
            if record.get("request", {}).get("property_name") != property_name:
                raise AnalysisError(
                    f"{workflow} measures a different thermal property."
                )
            entries = record.get("entries", [])
            expected = len(record["request"].get("hold_times_ps", [])) * record[
                "request"
            ].get("n_replicas", 0)
            if not entries or len(entries) != expected:
                raise AnalysisError(f"{workflow} records an incomplete thermal series.")
            for entry in entries:
                candidate = (directory / entry["directory"]).resolve()
                if not candidate.is_relative_to(directory):
                    raise AnalysisError(f"{workflow} has a directory outside its scan.")
                _check_completed(candidate, entry)
                result.append((candidate, record, entry))
        else:
            result.append((directory, None, None))
    for directory, _, _ in result:
        if directory in seen:
            raise AnalysisError(f"{directory} was supplied more than once.")
        seen.add(directory)
    if not result:
        raise AnalysisError("Supply saved thermal rate run directories.")
    return result


def _metadata(
    directory: Path, record: dict[str, Any] | None, manifest: RunManifest
) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "system": manifest.system,
        "composition": manifest.box,
    }
    if record is not None:
        request = record["request"]
        conditions.update(
            pressure_bar=request["spec"]["pressure_bar"],
            barostat=request["spec"].get("barostat", "isotropic"),
            preparation=request["equilibration"],
            system_sha256=request["system_sha256"],
            preparation_state_sha256=record.get("start_state_sha256"),
            crystalline_supplied=request.get("crystalline_supplied"),
        )
    else:
        for name in ("tg_workflow.json", "tm_workflow.json"):
            path = directory / name
            if path.is_file():
                request = json.loads(path.read_text()).get("request", {})
                conditions["pressure_bar"] = request.get("spec", {}).get("pressure_bar")
                conditions["barostat"] = request.get("spec", {}).get(
                    "barostat", "isotropic"
                )
                if request.get("system_sha256"):
                    conditions["system_sha256"] = request["system_sha256"]
    return conditions


def analyse_thermal_rates(
    run_dirs: Sequence[str | Path],
    *,
    property_name: str,
    target_rate: float,
    max_extrapolation_decades: float = 2.0,
) -> RateReport:
    """Fit comparable saved thermal histories without rerunning dynamics.

    A rate-scan root requires every recorded replica and every ladder hold.
    Ordinary Tg/Tm directories are also accepted. Tg takes the finest ladder
    family from each directory, excluding its coarse screening pass. Unknown
    preparation/pressure metadata is reported. Known conflicting conditions,
    or different temperature ladders, are refused. Unresolved transitions are
    retained and cannot turn into resolved extrapolations through omission.

    Neither transition estimator supplies a temperature standard error for a
    single history. Repeated-rate histories permit replica-based uncertainty;
    a melting bracket remains a finite-grid bracket, never a standard error.
    """
    property_ = _property(property_name)
    _analysis_options(target_rate, max_extrapolation_decades)
    directories = _directories(run_dirs, property_name)
    observations: list[RateObservation] = []
    notes: list[str] = [
        "Thermal rate extrapolations are finite-rate empirical estimates; neither equilibrium nor zero-rate transition temperatures are established.",
        "Replica uncertainty describes trajectories from the supplied preparation; common coordinates do not establish independent morphology or melt equilibration.",
    ]
    if property_name == "melting_temperature":
        notes.append(
            "Heating transitions can depend on superheating, finite size and crystal morphology. "
            "Density and enthalpy do not prove loss of crystalline order; inspect saved structures or trajectories."
        )
    for directory, record, entry in directories:
        manifest = RunManifest.load(directory)
        if manifest is None:
            raise AnalysisError(f"No thermal manifest in {directory}.")
        metadata = _metadata(directory, record, manifest)
        if property_name == "glass_transition":
            groups = (
                [tuple(entry["stages"])]
                if entry is not None
                else _group_passes(directory, quench_stages(directory))
            )
            curves = [quench_curve(directory, group) for group in groups]
            finest = min(curve.temperature_step_k for curve in curves)
            for curve in curves:
                if not math.isclose(curve.temperature_step_k, finest, rel_tol=1e-8):
                    continue
                if curve.cooling_rate_k_per_ns is None:
                    raise AnalysisError(
                        f"Unknown or irregular cooling rate in {directory}/{curve.stage}."
                    )
                minimum = (
                    4
                    if record is None
                    else record["request"]["spec"]["min_points_per_branch"]
                )
                detail: tuple[str, ...] = (
                    "Single-history Tg temperature uncertainty is unknown.",
                )
                try:
                    fit = glass_transition(curve, min_points_per_branch=minimum)
                    temperature: float | None = fit.temperature_k
                    resolved = fit.resolved
                except AnalysisError as error:
                    temperature = None
                    resolved = False
                    detail += (str(error),)
                observations.append(
                    RateObservation(
                        rate=curve.cooling_rate_k_per_ns,
                        value=temperature,
                        standard_error=None,
                        resolved=resolved,
                        conditions={
                            **metadata,
                            "temperatures_k": tuple(
                                float(t) for t in curve.temperature_k
                            ),
                            "direction": "cooling",
                            "min_points_per_branch": minimum,
                        },
                        source=f"{directory}:{curve.stage}",
                        notes=detail,
                    )
                )
        else:
            heating = heating_curve(
                directory, None if entry is None else entry["stages"]
            )
            rate = heating.heating_rate_k_per_ns
            if rate is None:
                raise AnalysisError(
                    f"Unknown or irregular heating rate in {directory}."
                )
            minimum = (
                3
                if record is None
                else record["request"]["spec"]["min_points_per_branch"]
            )
            transition = melting_temperature(heating, min_points_per_branch=minimum)
            bracket = transition.bracket_k
            detail = (
                ("No resolved melting bracket.",)
                if bracket is None
                else (
                    f"Finite-grid melting bracket: [{bracket[0]:g}, {bracket[1]:g}] K; this is not a confidence interval or standard error.",
                )
            )
            observations.append(
                RateObservation(
                    rate=rate,
                    value=transition.temperature_k,
                    standard_error=None,
                    resolved=transition.resolved,
                    conditions={
                        **metadata,
                        "temperatures_k": heating.temperature_k,
                        "direction": "heating",
                        "pressure_bar": heating.pressure_bar[0],
                        "min_points_per_branch": minimum,
                    },
                    source=str(directory),
                    notes=transition.notes + detail,
                )
            )
    # Missing provenance cannot be recovered. Compare all recorded values and
    # retain common known values without inventing metadata for legacy runs.
    keys = set().union(*(item.conditions for item in observations))
    common: dict[str, Any] = {}
    for key in sorted(keys):
        present = [
            item.conditions[key]
            for item in observations
            if item.conditions.get(key) is not None
        ]
        if present and any(value != present[0] for value in present[1:]):
            raise AnalysisError(
                f"Thermal histories have different {key}; rate fits need comparable conditions."
            )
        if len(present) != len(observations):
            notes.append(
                f"Some thermal histories lack recorded {key}; comparability is not fully verified."
            )
        elif present:
            common[key] = present[0]
    for key in ("pressure_bar", "preparation_state_sha256", "system_sha256"):
        if key not in common:
            notes.append(
                f"Recorded {key} is unavailable for all histories; comparability is not fully verified."
            )
    observations = [replace(item, conditions=common) for item in observations]
    report = analyse_rate_observations(
        observations,
        property=property_,
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
    )
    return replace(
        report,
        notes=tuple(notes) + report.notes,
        run_dirs=tuple(str(directory) for directory, _, _ in directories),
    )
