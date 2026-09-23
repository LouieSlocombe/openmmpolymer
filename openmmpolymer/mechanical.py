"""Measuring a polymer's elastic constants, and reporting what was measured.

Four numbers, from three deformations and a pressure ladder. A uniaxial
extension gives Young's modulus and Poisson's ratio; a shear ladder gives the
shear modulus; a gentle compression gives the bulk modulus. The fourth is the
point of the first three: for an isotropic solid ``E`` and ``nu`` fix ``K``
and ``G``, so measuring all four over-determines the pair and the gap between
measured and implied checks every one of them at once.

Every pass branches from the *same* equilibrated cell rather than running one
after another. A cell that has just been stretched to five per cent is not the
cell the next measurement wants, and chaining them would measure the shear
modulus of something with a deformation history. So this module calls
:func:`~openmmpolymer.protocols.run_protocol` once per pass with an explicit
starting state, which is also what makes the replicas replicas.

Two things it will not do. It will not call three runs from one configuration
an error bar unless they were given different velocities - inheriting the
equilibrated state's velocities as well as its positions gives the same
trajectory three times, and a spread over those is zero dressed up as
uncertainty. And it will not promote a slope to a modulus: OpenMM's own
warning about the instantaneous pressure is that its fluctuations are
enormous, so every fit here can come back unresolved, and on a small cell at
a short hold it often should.

The strain rate travels with every number, the way the cooling rate travels
with a glass transition. An all-atom extension runs some ten orders of
magnitude faster than a tensile test, and a modulus measured at 10^7 per
second is not the one a datasheet quotes.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._validation import require_integer, require_positive
from .elasticity import (
    BulkModulus,
    ElasticConsistency,
    ElasticModulus,
    PoissonRatio,
    ShearModulus,
    StressStrain,
    bulk_modulus,
    deform_stages,
    elastic_consistency,
    load_curve,
    poisson_ratio,
    shear_modulus,
    stress_strain,
    youngs_modulus,
)
from .protocols import (
    Protocol,
    RunManifest,
    RunSummary,
    Stage,
    run_protocol,
    standard_melt_equilibration,
)
from .simulate import RunContext, safe_timestep_fs
from .tg import ReportFiles
from .trajectory import AnalysisError

log = logging.getLogger(__name__)

#: What every pass calls itself in the manifest, so an interrupted run still
#: says which workflow it belongs to.
PROTOCOL_NAME = "mechanical"

#: Stage-name stems. Numbered so the run directory sorts into run order, and
#: free of dots because a stage's name becomes a file stem and
#: ``Path.with_suffix`` would read a dot as an extension.
DEFORM_STEM = "06_deform"
LOAD_STEM = "07_load"
BULK_STEM = "08_bulk"
SHEAR_STEM = "09_shear"

#: The equilibrated stage every pass branches from.
EQUILIBRATION_STAGE = "05_npt"

#: Where the workflow records what it derived, beside the manifest but not in
#: it: the manifest is the record of what ran, and an analysis-shaped field in
#: it would be stale after a later resume.
WORKFLOW_NAME = "mechanical_workflow.json"

#: How far the replicas may disagree, relative to their mean, before the
#: pooled modulus stops claiming to be resolved. Generous, because three
#: replicas of a forty-chain cell is a small sample of a noisy quantity - and
#: still worth having, because the alternative is quoting one run's number
#: with no spread at all.
MAX_REPLICA_SPREAD = 0.3


class MechanicalError(RuntimeError):
    """A mechanical measurement could not be run, or was asked for twice over."""


@dataclass(frozen=True)
class ModulusSpec:
    """Everything the passes need to know, in one place.

    Args:
        temperature_k: The temperature everything is measured at. Above the
            polymer's glass transition this measures a rubber, which is a
            real number about a different state of the material; nothing here
            knows which side of it you are on, so find the transition first.
        pressure_bar: The pressure held on every axis that is not driven.
        axis: The axis to stretch, 0, 1 or 2.
        strain_increment: Engineering strain added per step.
        max_strain: The strain the ladder stops at.
        relax_ps: Time to relax after each increment. The mean is over the
            second half, so this is twice the averaging window.
        elastic_strain_limit: The strain the modulus is fitted up to.
        n_replicas: How many times to repeat the extension from the same
            configuration with fresh velocities. The spread across them is
            the error bar; one replica has none.
        samples_per_step: Stress readings per increment. High on purpose.
            The instantaneous pressure decorrelates in well under a
            picosecond, so a handful of readings spread over a 50 ps window
            throws away almost all of the statistics the window already paid
            for. Each reading costs OpenMM about six energy evaluations,
            which against a relaxation window of tens of thousands of steps
            is single-figure per cent - a cheap way to take the standard
            error on the modulus down by a factor of several.
        stage_ps: Most dynamics one stage may hold before the ladder is split
            into another. This is the resume granularity.
        load_stresses_bar: Applied stresses for the constant-stress
            cross-check, or None to skip it. It shares no machinery with the
            extension - the stress is imposed and only the box is measured -
            so the two agreeing is real evidence.
        load_ps_each: Time at each applied stress. Long, because the cell's
            length is its slowest coordinate.
        bulk_pressures_bar: A gentle pressure ladder for the bulk modulus, up
            and back down so the hysteresis is measurable, or None to skip
            it. Gentle on purpose: the package's default compression ladder
            reaches a kilobar, far outside linear response.
        bulk_ps_each: Time at each pressure.
        shear_strains: Shear strains for the shear modulus, or None to skip.
        shear_ps_each: Time at each shear strain.
        max_total_ns: Refuse to start if the passes would exceed this.
    """

    temperature_k: float = 298.15
    pressure_bar: float = 1.0
    axis: int = 2
    strain_increment: float = 0.002
    max_strain: float = 0.05
    relax_ps: float = 50.0
    elastic_strain_limit: float = 0.015
    n_replicas: int = 3
    samples_per_step: int = 250
    stage_ps: float = 10_000.0
    load_stresses_bar: tuple[float, ...] | None = (0.0, 100.0, 200.0, 300.0)
    load_ps_each: float = 1000.0
    bulk_pressures_bar: tuple[float, ...] | None = (
        1.0,
        100.0,
        200.0,
        300.0,
        200.0,
        100.0,
        1.0,
    )
    bulk_ps_each: float = 500.0
    shear_strains: tuple[float, ...] | None = (0.005, 0.010, 0.015, 0.020)
    shear_ps_each: float = 200.0
    max_total_ns: float | None = None

    def __post_init__(self) -> None:
        """Reject a spec that cannot describe a deformation, at the call site."""
        for name in (
            "temperature_k",
            "pressure_bar",
            "strain_increment",
            "max_strain",
            "relax_ps",
            "elastic_strain_limit",
            "stage_ps",
        ):
            require_positive(getattr(self, name), None, name=name)
        require_integer(self.n_replicas, name="n_replicas")
        require_integer(self.samples_per_step, name="samples_per_step")
        if self.axis not in (0, 1, 2):
            raise ValueError(f"axis={self.axis!r} must be 0, 1 or 2.")
        if self.strain_increment >= self.max_strain:
            raise ValueError(
                f"strain_increment={self.strain_increment} is not below "
                f"max_strain={self.max_strain}: a ladder needs more than one "
                "rung."
            )
        if self.elastic_strain_limit > self.max_strain:
            raise ValueError(
                f"elastic_strain_limit={self.elastic_strain_limit} is above "
                f"max_strain={self.max_strain}, so the fit would run past the "
                "end of the curve."
            )
        for name in ("load_ps_each", "bulk_ps_each", "shear_ps_each"):
            require_positive(getattr(self, name), None, name=name)
        if self.max_total_ns is not None:
            require_positive(self.max_total_ns, None, name="max_total_ns")
        if self.shear_strains is not None and not self.shear_strains:
            raise ValueError("shear_strains is empty; pass None to skip the pass.")
        if self.load_stresses_bar is not None and not self.load_stresses_bar:
            raise ValueError("load_stresses_bar is empty; pass None to skip the pass.")
        if self.bulk_pressures_bar is not None and len(self.bulk_pressures_bar) < 2:
            raise ValueError(
                "bulk_pressures_bar needs at least two rungs; pass None to skip."
            )


#: The default settings, as a shared frozen singleton so it can be a default
#: argument without being rebuilt on every call.
DEFAULT_SPEC = ModulusSpec()


@dataclass(frozen=True)
class ModulusSchedule:
    """One extension's ladder, and what it costs.

    Args:
        n_steps: How many increments it applies.
        increment: Engineering strain added per step.
        relax_ps: Time held at each.
    """

    n_steps: int
    increment: float
    relax_ps: float

    @property
    def max_strain(self) -> float:
        """The strain the ladder reaches, compounding each increment."""
        return float((1.0 + self.increment) ** self.n_steps - 1.0)

    @property
    def total_ps(self) -> float:
        """How much dynamics the whole ladder is."""
        return self.relax_ps * self.n_steps

    @property
    def strain_rate_per_ns(self) -> float:
        """The rate the ladder amounts to, in strain per nanosecond."""
        return self.max_strain / self.total_ps * 1000.0


@dataclass(frozen=True)
class ModulusResult:
    """What one mechanical scan measured.

    Args:
        run_dir: Where it ran.
        manifest_path: The manifest it wrote.
        youngs: Young's modulus, fitted to every replica's points together,
            or None when no extension ran.
        poisson: Poisson's ratio, likewise pooled.
        bulk: The measured bulk modulus, or None if that pass was skipped.
        shear: The measured shear modulus, or None likewise.
        consistency: The measured constants against the ones ``E`` and ``nu``
            imply, or None when there was nothing to compare.
        replicas: One fit per replica, in the order they ran.
        replica_spread_mpa: The spread across them - the honest error bar,
            and the reason to run more than one.
        load_modulus: The constant-stress cross-check, or None if skipped.
            It shares no machinery with :attr:`youngs`, so the two agreeing
            is evidence that both are right.
        schedule: The ladder each replica walked.
        curves: Each replica's stress-strain curve.
        resolved: There is a Young's modulus whose fit resolved and whose
            replicas agree.
    """

    run_dir: str
    manifest_path: str
    youngs: ElasticModulus | None
    poisson: PoissonRatio | None
    bulk: BulkModulus | None
    shear: ShearModulus | None
    consistency: ElasticConsistency | None
    replicas: tuple[ElasticModulus, ...]
    replica_spread_mpa: float | None
    load_modulus: ElasticModulus | None
    schedule: ModulusSchedule
    curves: tuple[StressStrain, ...]
    resolved: bool

    @property
    def modulus_mpa(self) -> float | None:
        """The headline number, or None when nothing resolved."""
        if self.youngs is None or not self.resolved:
            return None
        return self.youngs.modulus_mpa


@dataclass(frozen=True)
class ModulusReport:
    """What a finished run directory says about its mechanics.

    Args:
        run_dir: The directory read.
        curves: Every stress-strain curve found, one per deformation pass.
        replicas: The fit to each.
        youngs: The pooled fit, or None.
        poisson: The pooled Poisson's ratio, or None.
        bulk: The measured bulk modulus, or None.
        shear: The measured shear modulus, or None.
        load: The constant-stress curve, or None.
        load_modulus: The fit to it, or None.
        consistency: The four constants against each other, or None.
        replica_spread_mpa: Spread across the replicas, or None.
        method_gap: Relative gap between the strain-controlled and
            stress-controlled moduli, or NaN when only one was measured. The
            two share no machinery, so this is the strongest single check in
            the report.
        notes: Anything that could not be read, in plain English.
    """

    run_dir: str
    curves: tuple[StressStrain, ...]
    replicas: tuple[ElasticModulus, ...]
    youngs: ElasticModulus | None
    poisson: PoissonRatio | None
    bulk: BulkModulus | None
    shear: ShearModulus | None
    load: StressStrain | None
    load_modulus: ElasticModulus | None
    consistency: ElasticConsistency | None
    replica_spread_mpa: float | None
    method_gap: float
    notes: tuple[str, ...]


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


def deform_schedule(spec: ModulusSpec) -> ModulusSchedule:
    """The ladder one extension walks.

    The step count is what it takes to reach ``max_strain`` by compounding
    increments, because that is what the deformation actually does: each
    increment scales a cell that the last one already scaled.

    Args:
        spec: What to run.

    Returns:
        The schedule.
    """
    steps = math.ceil(math.log1p(spec.max_strain) / math.log1p(spec.strain_increment))
    return ModulusSchedule(
        n_steps=max(1, int(steps)),
        increment=spec.strain_increment,
        relax_ps=spec.relax_ps,
    )


def _deform_stages(
    stem: str, schedule: ModulusSchedule, spec: ModulusSpec, *, timestep_fs: float
) -> tuple[Stage, ...]:
    """Turn one ladder into the stages that walk it.

    Split into chunks no longer than ``stage_ps``, which is bookkeeping and
    not physics: each chunk starts from the state the one before it left, so
    the deformation is continuous. What it buys is resume granularity, a
    stage being the unit a run picks itself back up at.

    Each chunk is told the strain it starts at and the cell the strain is
    measured against, because neither survives in the state file: a resumed
    chunk opens an already-stretched cell and would otherwise call that
    stretch the origin.
    """
    per_chunk = max(1, int(spec.stage_ps // schedule.relax_ps))
    stages: list[Stage] = []
    done = 0
    index = 0
    while done < schedule.n_steps:
        count = min(per_chunk, schedule.n_steps - done)
        stages.append(
            Stage(
                f"{stem}_{index:02d}",
                "deform",
                {
                    "temperature_k": spec.temperature_k,
                    "pressure_bar": spec.pressure_bar,
                    "axis": spec.axis,
                    "strain_increment": spec.strain_increment,
                    "n_steps": count,
                    "relax_ps": spec.relax_ps,
                    "strain_start": (1.0 + spec.strain_increment) ** done - 1.0,
                    "samples_per_step": spec.samples_per_step,
                    "timestep_fs": timestep_fs,
                    # Only the first chunk may take the current cell as the
                    # origin; the rest are handed it by the driver, which
                    # knows what the equilibrated cell was.
                    "new_velocities": index == 0,
                },
            )
        )
        done += count
        index += 1
    return tuple(stages)


def deform_protocol(
    spec: ModulusSpec,
    *,
    timestep_fs: float,
    replica: int = 0,
    reference_box_nm: Sequence[float] | None = None,
) -> Protocol:
    """One replica's extension, and nothing else.

    One timestep is pinned across every chunk rather than let each derate to
    its own temperature, because this is a measurement and chunks integrated
    differently are a confound nobody would choose.

    Args:
        spec: What to run.
        timestep_fs: The timestep every chunk uses.
        replica: Which repeat this is. It only changes the stage names, which
            is enough: every random stream is derived from the stage label,
            so a differently-named replica is a differently-seeded one.
        reference_box_nm: The unstrained cell, passed to every chunk so that
            a resumed one measures strain against the same origin as the
            first.

    Returns:
        The protocol.
    """
    stem = f"{DEFORM_STEM}_r{replica}"
    stages = _deform_stages(stem, deform_schedule(spec), spec, timestep_fs=timestep_fs)
    if reference_box_nm is not None:
        origin = [float(value) for value in reference_box_nm]
        stages = tuple(
            Stage(stage.name, stage.kind, {**stage.options, "reference_box_nm": origin})
            for stage in stages
        )
    return Protocol(name=PROTOCOL_NAME, stages=stages)


def equilibration_protocol(
    spec: ModulusSpec = DEFAULT_SPEC, **equilibration: Any
) -> Protocol:
    """Settle the melt at the temperature the mechanics will be measured at.

    Args:
        spec: What to run.
        **equilibration: Passed to
            :func:`~openmmpolymer.protocols.standard_melt_equilibration`.

    Returns:
        The protocol.
    """
    base = standard_melt_equilibration(
        target_temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        **equilibration,
    )
    return Protocol(name=PROTOCOL_NAME, stages=base.stages)


def mechanical_scan(spec: ModulusSpec = DEFAULT_SPEC, **equilibration: Any) -> Protocol:
    """Equilibrate, then walk one extension - what ``--dry-run`` prices.

    The whole scan is several protocols, because each pass branches from the
    same equilibrated cell rather than following the one before it. This is
    the first two of them, which is what a cost estimate can be built from
    without running anything.

    Args:
        spec: What to run.
        **equilibration: Passed to
            :func:`~openmmpolymer.protocols.standard_melt_equilibration`.

    Returns:
        The protocol.
    """
    base = equilibration_protocol(spec, **equilibration)
    timestep_fs = 2.0
    return Protocol(
        name=PROTOCOL_NAME,
        stages=(
            *base.stages,
            *_deform_stages(
                f"{DEFORM_STEM}_r0",
                deform_schedule(spec),
                spec,
                timestep_fs=timestep_fs,
            ),
        ),
    )


def extra_stages(spec: ModulusSpec, *, timestep_fs: float) -> tuple[Stage, ...]:
    """The load, bulk and shear passes, in that order.

    Stages rather than a protocol, because each starts from the equilibrated
    cell rather than from the one before it: the driver runs each as its own
    one-stage protocol. Returned as a tuple for the same reason - skipping
    all three is a legitimate request, and an empty ``Protocol`` is not a
    thing this package allows.

    Args:
        spec: What to run.
        timestep_fs: The timestep each uses.

    Returns:
        The stages, which may be none.
    """
    stages: list[Stage] = []
    if spec.load_stresses_bar is not None:
        stages.append(
            Stage(
                LOAD_STEM,
                "load",
                {
                    "temperature_k": spec.temperature_k,
                    "pressure_bar": spec.pressure_bar,
                    "axis": spec.axis,
                    "stresses_bar": list(spec.load_stresses_bar),
                    "duration_ps_each": spec.load_ps_each,
                    "samples_per_step": spec.samples_per_step,
                    "timestep_fs": timestep_fs,
                },
            )
        )
    if spec.bulk_pressures_bar is not None:
        stages.append(
            Stage(
                BULK_STEM,
                "compress",
                {
                    "temperature_k": spec.temperature_k,
                    "pressures_bar": list(spec.bulk_pressures_bar),
                    "duration_ps_each": spec.bulk_ps_each,
                    "samples_per_segment": spec.samples_per_step,
                    "timestep_fs": timestep_fs,
                },
            )
        )
    if spec.shear_strains is not None:
        stages.append(
            Stage(
                SHEAR_STEM,
                "shear",
                {
                    "temperature_k": spec.temperature_k,
                    "strains": list(spec.shear_strains),
                    "duration_ps_each": spec.shear_ps_each,
                    "samples_per_step": spec.samples_per_step,
                    "timestep_fs": timestep_fs,
                },
            )
        )
    return tuple(stages)


# --------------------------------------------------------------------------
# The workflow record and the cost
# --------------------------------------------------------------------------


def _request(spec: ModulusSpec) -> dict[str, Any]:
    """What the caller asked for, as the thing a resume is compared against.

    Round-tripped through JSON before it is compared with anything, because
    that is the form it is stored in. Several of these fields are tuples,
    and a tuple comes back from a file as a list: comparing the two directly
    makes every second run look like a change of settings and refuse to
    resume.
    """
    stored = json.loads(json.dumps(asdict(spec), default=str))
    return {"spec": cast("dict[str, Any]", stored)}


def _check_request(run_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Refuse a resume that quietly asks for something else.

    A stage's options are recorded nowhere, so a protocol rerun with
    different settings resumes and keeps the old result without a word. That
    is survivable for an equilibration and not for a measurement, where the
    modulus would then belong to a ladder nobody walked.
    """
    path = run_dir / WORKFLOW_NAME
    if not path.is_file():
        return {}
    record: dict[str, Any] = json.loads(path.read_text())
    previous = record.get("request")
    if previous is not None and previous != request:
        changed = sorted(
            key
            for key in set(previous.get("spec", {})) | set(request["spec"])
            if previous.get("spec", {}).get(key) != request["spec"].get(key)
        )
        raise MechanicalError(
            f"{path} records a scan run with different settings "
            f"({', '.join(changed) or 'unknown'}), and resuming would keep "
            "results measured under the old ones. Run into a fresh "
            "directory, or put the settings back."
        )
    return record


def _save_workflow(run_dir: Path, record: dict[str, Any]) -> str:
    """Write the workflow record. Recomputable, so not written atomically."""
    path = run_dir / WORKFLOW_NAME
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return str(path)


def _report_cost(
    equilibration: Protocol,
    schedule: ModulusSchedule,
    spec: ModulusSpec,
    extras: Sequence[Stage],
    manifest: RunManifest | None,
) -> None:
    """Say what the whole thing costs before any of it runs.

    Raises:
        MechanicalError: The total is over ``max_total_ns``.
    """
    from .protocols import _stage_duration_ps

    settle_ps = equilibration.total_duration_ps
    deform_ps = schedule.total_ps * spec.n_replicas
    extra_ps = sum(_stage_duration_ps(stage) for stage in extras)
    total_ps = settle_ps + deform_ps + extra_ps

    done = set(manifest.stages) if manifest is not None else set()
    remaining = total_ps - sum(
        _stage_duration_ps(stage)
        for stage in (*equilibration.stages, *extras)
        if stage.name in done
    )
    for replica in range(spec.n_replicas):
        stem = f"{DEFORM_STEM}_r{replica}"
        remaining -= sum(
            schedule.relax_ps * int(stage.options["n_steps"])
            for stage in _deform_stages(stem, schedule, spec, timestep_fs=2.0)
            if stage.name in done
        )

    log.info(
        "Mechanical scan: %.1f ns equilibration, %.1f ns of extension (%d "
        "replicas of %d steps to %.1f%% strain at %.3g /ns), %.1f ns of "
        "load, bulk and shear - %.1f ns in total, %.1f ns of it still to run.",
        settle_ps / 1000.0,
        deform_ps / 1000.0,
        spec.n_replicas,
        schedule.n_steps,
        100.0 * schedule.max_strain,
        schedule.strain_rate_per_ns,
        extra_ps / 1000.0,
        total_ps / 1000.0,
        max(0.0, remaining) / 1000.0,
    )
    if spec.max_total_ns is not None and total_ps / 1000.0 > spec.max_total_ns:
        raise MechanicalError(
            f"The scan is {total_ps / 1000.0:.1f} ns, over the "
            f"{spec.max_total_ns:.1f} ns budget. Shorten relax_ps, drop a "
            "replica, skip a pass, or raise max_total_ns."
        )


def _equilibrated_box_nm(state_path: str | Path) -> list[float]:
    """The cell edges a saved state carries, in nanometres.

    Read from the state rather than from ``run.box``, which is the *packed*
    cell: everything since has compressed it, and strain measured against
    the packed edges would be measured against a cell that stopped existing
    at the first barostat move.
    """
    import openmm as mm
    from openmm import unit

    state = mm.XmlSerializer.deserialize(Path(state_path).read_text())
    vectors = state.getPeriodicBoxVectors()
    return [
        float(vectors[axis][axis].value_in_unit(unit.nanometer)) for axis in range(3)
    ]


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def run_modulus_scan(
    run: RunContext,
    run_dir: str | Path = "run",
    *,
    spec: ModulusSpec = DEFAULT_SPEC,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> ModulusResult:
    """Equilibrate a cell and measure its elastic constants.

    Every pass branches from the equilibrated cell, not from the pass before
    it, so each is run as its own protocol with that state named explicitly.
    A cell that has just been stretched is not the cell the next measurement
    wants.

    Args:
        run: The run context.
        run_dir: Where to work. An interrupted scan resumes from here.
        spec: What to measure and how.
        resume: Whether to pick up from what the manifest records.
        chain_backbone: Backbone atom indices, for the chain measurements.
        atoms_per_chain: Likewise.
        expected_characteristic_ratio: Likewise.
        **equilibration: Passed to
            :func:`~openmmpolymer.protocols.standard_melt_equilibration`.

    Returns:
        What was measured.

    Raises:
        MechanicalError: The scan is over budget, or this directory holds one
            run with different settings.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request = _request(spec)
    record = _check_request(directory, request) if resume else {}

    settle = equilibration_protocol(spec, **equilibration)
    schedule = deform_schedule(spec)
    timestep_fs = safe_timestep_fs(spec.temperature_k, run.spec)
    extras = extra_stages(spec, timestep_fs=timestep_fs)
    _report_cost(
        settle,
        schedule,
        spec,
        extras,
        RunManifest.load(directory) if resume else None,
    )

    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    settled = run_protocol(settle, run, directory, resume=resume, **chains)
    start_state = _last_state(settled, directory)
    origin = _equilibrated_box_nm(start_state)
    log.info(
        "Equilibrated cell is %s nm; every pass starts from %s.",
        [round(value, 4) for value in origin],
        Path(start_state).name,
    )

    for replica in range(spec.n_replicas):
        run_protocol(
            deform_protocol(
                spec,
                timestep_fs=timestep_fs,
                replica=replica,
                reference_box_nm=origin,
            ),
            run,
            directory,
            # Preparation already reset a forced rerun's manifest. Keep all
            # newly completed preparation and replica stages from here on.
            resume=True,
            state_in=start_state,
            **chains,
        )

    for stage in extras:
        run_protocol(
            Protocol(name=PROTOCOL_NAME, stages=(stage,)),
            run,
            directory,
            resume=True,
            state_in=start_state,
            **chains,
        )

    report = analyse_mechanics(directory, strain_limit=spec.elastic_strain_limit)
    record.update(
        {
            "request": request,
            "reference_box_nm": origin,
            "start_state": str(start_state),
            "timestep_fs": timestep_fs,
            "n_replicas": spec.n_replicas,
        }
    )
    _save_workflow(directory, record)

    resolved = bool(
        report.youngs is not None
        and report.youngs.resolved
        and (
            report.replica_spread_mpa is None
            or report.youngs.modulus_mpa <= 0.0
            or report.replica_spread_mpa
            <= MAX_REPLICA_SPREAD * report.youngs.modulus_mpa
        )
    )
    _log_result(report, schedule, resolved)
    return ModulusResult(
        run_dir=str(directory),
        manifest_path=str(directory / "manifest.json"),
        youngs=report.youngs,
        poisson=report.poisson,
        bulk=report.bulk,
        shear=report.shear,
        consistency=report.consistency,
        replicas=report.replicas,
        replica_spread_mpa=report.replica_spread_mpa,
        load_modulus=report.load_modulus,
        schedule=schedule,
        curves=report.curves,
        resolved=resolved,
    )


def _last_state(summary: RunSummary, directory: Path) -> str:
    """The state the equilibration finished at, whether it ran or resumed.

    Taken from the summary, which threads the state through skipped stages as
    well as run ones, and not by scanning the manifest for the last thing with
    a state file. The manifest is in run order, so on a resume the last entry
    is whatever the previous attempt got furthest through - a deformation, a
    load or a shear - and every pass branches from the *equilibrated* cell,
    not from one that has already been pulled. Worse than the wrong starting
    configuration: ``run_modulus_scan`` reads the strain origin off this
    state, so the strain every remaining chunk reports would be measured
    against a cell that was already at five per cent.
    """
    state = summary.final_state
    if state and state != "None" and Path(state).is_file():
        return str(state)
    raise MechanicalError(
        f"{directory} has no finished equilibration stage to deform from. "
        "Run the equilibration first, or delete the manifest and start over."
    )


def _log_result(
    report: ModulusReport, schedule: ModulusSchedule, resolved: bool
) -> None:
    """One line per measured constant, each with what qualifies it."""
    if report.youngs is None:
        log.info("Mechanical scan: no extension to fit.")
        return
    log.info(
        "Mechanical scan: E = %.0f MPa%s at %.3g strain/ns, %.0f K%s.",
        report.youngs.modulus_mpa,
        ""
        if report.replica_spread_mpa is None
        else f" +/- {report.replica_spread_mpa:.0f} over "
        f"{len(report.replicas)} replicas",
        schedule.strain_rate_per_ns,
        report.youngs.temperature_k,
        "" if resolved else " (not resolved)",
    )
    if report.poisson is not None:
        log.info(
            "  nu = %.3f%s.",
            report.poisson.ratio,
            "" if report.poisson.resolved else " (not resolved)",
        )
    for label, fit in (("K", report.bulk), ("G", report.shear)):
        if fit is not None:
            log.info(
                "  %s = %.0f MPa%s.",
                label,
                fit.modulus_mpa,
                "" if fit.resolved else " (not resolved)",
            )
    if report.consistency is not None:
        log.info(
            "  E and nu imply K = %.0f, G = %.0f MPa; gaps %.0f%% and %.0f%%%s.",
            report.consistency.bulk_implied_mpa,
            report.consistency.shear_implied_mpa,
            100.0 * report.consistency.bulk_gap,
            100.0 * report.consistency.shear_gap,
            "" if report.consistency.consistent else " - not consistent",
        )
    if math.isfinite(report.method_gap):
        log.info(
            "  the constant-stress cross-check differs by %.0f%%.",
            100.0 * report.method_gap,
        )


# --------------------------------------------------------------------------
# Reading a finished run
# --------------------------------------------------------------------------


def _replica_groups(run_dir: str | Path) -> list[tuple[str, ...]]:
    """Group the deformation stages into one list of chunks per replica.

    Grouped on the stem the chunk suffix hangs off, so a ladder split for
    resume comes back as one curve and two replicas do not come back as one.
    A stage named by something else entirely - a single ``deform`` run made
    by hand - becomes its own group, which is what it is.
    """
    groups: dict[str, list[str]] = {}
    for name in deform_stages(run_dir):
        stem, _, tail = name.rpartition("_")
        key = stem if stem and tail.isdigit() else name
        groups.setdefault(key, []).append(name)
    return [tuple(names) for names in groups.values()]


def analyse_mechanics(
    run_dir: str | Path,
    *,
    strain_limit: float = 0.015,
    min_points: int = 5,
) -> ModulusReport:
    """Read everything a finished run has to say about its mechanics.

    Reads and returns; writes nothing. Deformations are found by what they
    recorded rather than by what they were called, replicas are told apart by
    their stage stems, and a pass that is not there simply does not appear -
    a run that skipped the shear ladder is not a broken run.

    The headline fit is to every replica's points at once rather than to the
    mean of their separate fits, so a replica with more usable points inside
    the window carries more of the answer. Their spread is reported beside
    it, and is the only honest error bar here.

    Args:
        run_dir: A directory a run wrote to.
        strain_limit: The strain to fit the modulus up to.
        min_points: Fewest points a fit may rest on.

    Returns:
        The report.

    Raises:
        AnalysisError: There is no manifest, or nothing in it was a
            deformation of any kind.
    """
    directory = Path(run_dir)
    notes: list[str] = []
    curves: list[StressStrain] = []
    replicas: list[ElasticModulus] = []

    try:
        groups = _replica_groups(directory)
    except AnalysisError as error:
        groups = []
        notes.append(f"No extension to fit: {error}")

    for group in groups:
        curve = stress_strain(directory, group)
        fit = youngs_modulus(curve, strain_limit=strain_limit, min_points=min_points)
        curves.append(curve)
        replicas.append(fit)

    pooled = _pool(curves)
    youngs = (
        youngs_modulus(pooled, strain_limit=strain_limit, min_points=min_points)
        if pooled is not None
        else None
    )
    poisson = (
        poisson_ratio(pooled, strain_limit=strain_limit, min_points=min_points)
        if pooled is not None
        else None
    )
    spread = _spread([fit.modulus_mpa for fit in replicas])

    # Named rather than found by shape, and this is the one place that rule
    # is inverted. A bulk ladder is run by the shared `compress` runner, so
    # it records exactly what the equilibration's compression ladder records
    # - and that one climbs to a kilobar at the melt temperature, which
    # fitted as a bulk modulus is a confident number about nothing. Two
    # different measurements that record the same shape cannot be told apart
    # by their shape, so this asks for the stage this workflow wrote.
    manifest = RunManifest.load(directory)
    stage_names = list(manifest.stages) if manifest is not None else []
    if BULK_STEM in stage_names:
        bulk = _optional(
            lambda: bulk_modulus(directory, BULK_STEM),
            notes,
            "No bulk-modulus pass to fit",
        )
    else:
        bulk = None
        notes.append(
            f"No bulk-modulus pass: {directory} has no {BULK_STEM} stage. A "
            "compression ladder run as part of equilibration is not one - it "
            "is far outside linear response and at the wrong temperature."
        )
    shear = _optional(lambda: shear_modulus(directory), notes, "No shear ladder to fit")
    load = _optional(lambda: load_curve(directory), notes, "No constant-stress pass")
    load_fit = (
        youngs_modulus(load, strain_limit=strain_limit, min_points=2)
        if load is not None
        else None
    )

    consistency = (
        elastic_consistency(youngs, poisson, bulk=bulk, shear=shear)
        if youngs is not None and poisson is not None
        else None
    )
    gap = math.nan
    if (
        youngs is not None
        and load_fit is not None
        and abs(youngs.modulus_mpa) > 1.0e-12
    ):
        gap = abs(load_fit.modulus_mpa - youngs.modulus_mpa) / abs(youngs.modulus_mpa)

    if not curves and bulk is None and shear is None and load is None:
        raise AnalysisError(
            f"Nothing in {directory} was a mechanical measurement. It "
            f"records: {', '.join(stage_names) or 'nothing'}."
        )

    return ModulusReport(
        run_dir=str(directory),
        curves=tuple(curves),
        replicas=tuple(replicas),
        youngs=youngs,
        poisson=poisson,
        bulk=bulk,
        shear=shear,
        load=load,
        load_modulus=load_fit,
        consistency=consistency,
        replica_spread_mpa=spread,
        method_gap=gap,
        notes=tuple(notes),
    )


def _optional[T](read: Callable[[], T], notes: list[str], what: str) -> T | None:
    """Run a reader, turning "there is nothing there" into a note.

    A skipped pass and a broken one look identical from the outside, so the
    distinction is drawn here once rather than at each of three call sites.
    """
    try:
        return read()
    except AnalysisError as error:
        notes.append(f"{what}: {error}")
        return None


def _pool(curves: Sequence[StressStrain]) -> StressStrain | None:
    """Every replica's points as one curve, sorted by strain."""
    if not curves:
        return None
    if len(curves) == 1:
        return curves[0]
    strain = np.concatenate([curve.strain for curve in curves])
    order = np.argsort(strain)
    rates = [
        curve.strain_rate_per_ns
        for curve in curves
        if curve.strain_rate_per_ns is not None
    ]
    return StressStrain(
        stage=" | ".join(curve.stage for curve in curves),
        axis=curves[0].axis,
        strain=strain[order],
        stress_mpa=np.concatenate([curve.stress_mpa for curve in curves])[order],
        lateral_strain=np.concatenate([curve.lateral_strain for curve in curves])[
            order
        ],
        lateral_stress_mpa=np.concatenate(
            [curve.lateral_stress_mpa for curve in curves]
        )[order],
        temperature_k=float(np.mean([curve.temperature_k for curve in curves])),
        strain_rate_per_ns=float(np.mean(rates)) if rates else None,
        controlled=curves[0].controlled,
    )


def _spread(values: Sequence[float]) -> float | None:
    """The sample standard deviation of the replicas, or None below two.

    None rather than zero: one replica has no spread to report, and a zero
    would read as several runs that agreed perfectly.
    """
    usable = [value for value in values if math.isfinite(value)]
    if len(usable) < 2:
        return None
    return float(np.std(usable, ddof=1))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _modulus_record(fit: ElasticModulus) -> dict[str, Any]:
    """One modulus fit as plain JSON types.

    Written out field by field rather than with ``asdict``, which would drop
    the properties and render nothing helpfully. Spelling it out also pins
    what is on disk independently of how the dataclasses happen to be laid
    out.
    """
    return {
        "modulus_mpa": fit.modulus_mpa,
        "intercept_mpa": fit.intercept_mpa,
        "strain_limit": fit.strain_limit,
        "n_points": fit.n_points,
        "residual_mpa": fit.residual_mpa,
        "standard_error_mpa": fit.standard_error_mpa,
        "relative_standard_error": fit.relative_standard_error,
        "half_disagreement": fit.half_disagreement,
        "temperature_k": fit.temperature_k,
        "strain_rate_per_ns": fit.strain_rate_per_ns,
        "resolved": fit.resolved,
    }


def _poisson_record(fit: PoissonRatio) -> dict[str, Any]:
    """Poisson's ratio as plain JSON types."""
    return {
        "ratio": fit.ratio,
        "standard_error": fit.standard_error,
        "n_points": fit.n_points,
        "strain_limit": fit.strain_limit,
        "resolved": fit.resolved,
    }


def _bulk_record(fit: BulkModulus) -> dict[str, Any]:
    """The bulk modulus as plain JSON types."""
    return {
        "stage": fit.stage,
        "modulus_mpa": fit.modulus_mpa,
        "standard_error_mpa": fit.standard_error_mpa,
        "relative_standard_error": fit.relative_standard_error,
        "residual_log_volume": fit.residual_log_volume,
        "half_disagreement": fit.half_disagreement,
        "compression_mpa": fit.compression_mpa,
        "decompression_mpa": fit.decompression_mpa,
        "hysteresis": fit.hysteresis,
        "n_points": fit.n_points,
        "temperature_k": fit.temperature_k,
        "resolved": fit.resolved,
    }


def _shear_record(fit: ShearModulus) -> dict[str, Any]:
    """The shear modulus as plain JSON types."""
    return {
        "stage": fit.stage,
        "modulus_mpa": fit.modulus_mpa,
        "standard_error_mpa": fit.standard_error_mpa,
        "n_points": fit.n_points,
        "temperature_k": fit.temperature_k,
        "resolved": fit.resolved,
    }


def _consistency_record(check: ElasticConsistency) -> dict[str, Any]:
    """The over-determination check as plain JSON types."""
    return {
        "bulk_implied_mpa": check.bulk_implied_mpa,
        "shear_implied_mpa": check.shear_implied_mpa,
        "bulk_measured_mpa": check.bulk_measured_mpa,
        "shear_measured_mpa": check.shear_measured_mpa,
        "bulk_gap": check.bulk_gap,
        "shear_gap": check.shear_gap,
        "consistent": check.consistent,
    }


def write_mechanical_report(
    report: ModulusReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write a mechanical report out, as JSON and as figures.

    Args:
        report: What :func:`analyse_mechanics` found.
        output_dir: Where to write, defaulting to ``<run_dir>/analysis``.
            Give one when the run directory should not be touched.
        figures: Write figures as well as the record.
        figure_format: What matplotlib should save them as.

    Returns:
        Where everything went.
    """
    from importlib.metadata import PackageNotFoundError, version

    from .plots import plot_moduli, plot_stress_strain

    directory = (
        Path(report.run_dir) / "analysis" if output_dir is None else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)

    try:
        own = version("openmmpolymer")
    except PackageNotFoundError:  # pragma: no cover - uninstalled checkout
        own = "0.0.0+unknown"
    manifest = RunManifest.load(report.run_dir)
    record: dict[str, Any] = {
        "openmmpolymer": own,
        "run_dir": report.run_dir,
        "versions": {} if manifest is None else manifest.versions,
        "stages": [curve.stage for curve in report.curves],
        "youngs": None if report.youngs is None else _modulus_record(report.youngs),
        "poisson": None if report.poisson is None else _poisson_record(report.poisson),
        "bulk": None if report.bulk is None else _bulk_record(report.bulk),
        "shear": None if report.shear is None else _shear_record(report.shear),
        "load_modulus": (
            None
            if report.load_modulus is None
            else _modulus_record(report.load_modulus)
        ),
        "replicas": [_modulus_record(fit) for fit in report.replicas],
        "replica_spread_mpa": report.replica_spread_mpa,
        "method_gap": report.method_gap,
        "consistency": (
            None
            if report.consistency is None
            else _consistency_record(report.consistency)
        ),
        "notes": list(report.notes),
    }
    json_path = directory / "mechanics.json"
    json_path.write_text(json.dumps(record, indent=2, default=str) + "\n")

    written: list[str] = []
    if figures:
        for curve, fit in zip(report.curves, report.replicas, strict=False):
            stem = curve.stage.replace(", ", "_").replace(" ", "_")
            written.append(
                _save(
                    plot_stress_strain(curve, fit=fit, poisson=report.poisson),
                    directory / f"stress_strain_{stem}.{figure_format}",
                )
            )
        if report.load is not None:
            written.append(
                _save(
                    plot_stress_strain(report.load, fit=report.load_modulus),
                    directory / f"load_curve.{figure_format}",
                )
            )
        if report.youngs is not None:
            written.append(
                _save(plot_moduli(report), directory / f"moduli.{figure_format}")
            )
    return ReportFiles(json=str(json_path), figures=tuple(written))


def _save(figure: Any, path: Path) -> str:
    """Save a figure and close it, returning where it went."""
    figure.savefig(path, bbox_inches="tight")
    return str(path)
