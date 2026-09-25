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

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._files import ReportFiles, write_json
from ._validation import require_axis, require_integer
from ._workflow import (
    StrainSchedule,
    check_request,
    deformation_stages,
    equilibrate,
    equilibration_at,
    group_by_stem,
    optional,
    remaining_ps,
    require_positive_fields,
    run_branches,
    sample_spread,
    scan_listing,
    spec_request,
    with_reference_box,
    write_report_files,
)
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
from .plots import plot_moduli, plot_stress_strain
from .protocols import Protocol, RunManifest, Stage
from .simulate import RunContext, safe_timestep_fs
from .trajectory import AnalysisError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)

PROTOCOL_NAME = "mechanical"
DEFORM_STEM = "06_deform"
LOAD_STEM = "07_load"
BULK_STEM = "08_bulk"
SHEAR_STEM = "09_shear"
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
        samples_per_step: Stress readings per increment. High on purpose, for
            the reason :func:`~openmmpolymer.simulate.run_deform` gives: the
            readings are cheap and the pressure decorrelates fast.
        stage_ps: Most dynamics one stage may hold before the ladder is split
            into another.
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
        require_positive_fields(
            self,
            (
                "temperature_k",
                "pressure_bar",
                "strain_increment",
                "max_strain",
                "relax_ps",
                "elastic_strain_limit",
                "stage_ps",
                "load_ps_each",
                "bulk_ps_each",
                "shear_ps_each",
            ),
            optional=("max_total_ns",),
        )
        require_integer(self.n_replicas, name="n_replicas")
        require_integer(self.samples_per_step, name="samples_per_step")
        require_axis(self.axis)
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
        if self.shear_strains is not None and not self.shear_strains:
            raise ValueError("shear_strains is empty; pass None to skip the pass.")
        if self.load_stresses_bar is not None and not self.load_stresses_bar:
            raise ValueError("load_stresses_bar is empty; pass None to skip the pass.")
        if self.bulk_pressures_bar is not None and len(self.bulk_pressures_bar) < 2:
            raise ValueError(
                "bulk_pressures_bar needs at least two rungs; pass None to skip."
            )


DEFAULT_SPEC = ModulusSpec()


@dataclass(frozen=True)
class ModulusSchedule(StrainSchedule):
    """One extension's ladder, and what it costs.

    Args:
        n_steps: How many increments it applies.
        increment: Engineering strain added per step.
        relax_ps: Time held at each.
    """


@dataclass(frozen=True)
class ModulusReport:
    """What a scan, or a finished run directory, says about its mechanics.

    Args:
        run_dir: The directory read.
        curves: Every stress-strain curve found, one per replica.
        replicas: The fit to each.
        youngs: Young's modulus fitted to every replica's points together, or
            None when nothing was extended.
        poisson: Poisson's ratio, pooled likewise, or None.
        bulk: The measured bulk modulus, or None.
        shear: The measured shear modulus, or None.
        load: The constant-stress curve, or None.
        load_modulus: The fit to it, or None.
        consistency: The four constants against each other, or None.
        replica_spread_mpa: Spread across the replicas - the honest error
            bar - or None below two.
        method_gap: Relative gap between the strain-controlled and
            stress-controlled moduli, or NaN when only one was measured. The
            two share no machinery, so this is the strongest single check in
            the report.
        resolved: There is a Young's modulus whose fit resolved and whose
            replicas agree to within :data:`MAX_REPLICA_SPREAD`.
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
    resolved: bool
    notes: tuple[str, ...]


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


def deform_schedule(spec: ModulusSpec) -> ModulusSchedule:
    """The ladder one extension walks.

    The step count is what it takes to reach ``max_strain`` by compounding
    increments, because that is what the deformation actually does: each
    increment scales a cell that the last one already scaled.
    """
    return ModulusSchedule.reaching(
        spec.max_strain, spec.strain_increment, spec.relax_ps
    )


def deform_protocol(
    spec: ModulusSpec,
    *,
    timestep_fs: float,
    replica: int = 0,
    reference_box_nm: Sequence[float] | None = None,
) -> Protocol:
    """One replica's extension, and nothing else.

    Each chunk is told the strain it starts at, because a state file does not
    carry one; only the first draws fresh velocities, and the rest continue
    its trajectory.

    Args:
        spec: What to run.
        timestep_fs: The timestep every chunk uses.
        replica: Which repeat this is. It only changes the stage names, which
            is enough: every random stream is derived from the stage label,
            so a differently-named replica is a differently-seeded one.
        reference_box_nm: The unstrained cell, passed to every chunk.

    Returns:
        The protocol.
    """
    stages = deformation_stages(
        deform_schedule(spec),
        stem=f"{DEFORM_STEM}_r{replica}",
        chunk_digits=2,
        stage_ps=spec.stage_ps,
        temperature_k=spec.temperature_k,
        pressure_bar=spec.pressure_bar,
        axis=spec.axis,
        samples_per_step=spec.samples_per_step,
        timestep_fs=timestep_fs,
    )
    return Protocol(PROTOCOL_NAME, with_reference_box(stages, reference_box_nm))


def equilibration_protocol(
    spec: ModulusSpec = DEFAULT_SPEC, **equilibration: Any
) -> Protocol:
    """Settle the melt at the temperature the mechanics will be measured at.

    *equilibration* is passed to
    :func:`~openmmpolymer.protocols.standard_melt_equilibration`.
    """
    return equilibration_at(
        PROTOCOL_NAME, spec.temperature_k, spec.pressure_bar, **equilibration
    )


def extra_stages(spec: ModulusSpec, *, timestep_fs: float) -> tuple[Stage, ...]:
    """The load, bulk and shear passes, in that order, and possibly none.

    Stages rather than a protocol, because each starts from the equilibrated
    cell rather than from the one before it, and because skipping all three
    is a legitimate request that an empty ``Protocol`` could not express.
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


def _branches(
    spec: ModulusSpec,
    *,
    timestep_fs: float,
    reference_box_nm: Sequence[float] | None = None,
) -> list[Protocol]:
    """Every pass that starts from the equilibrated cell, in the order run."""
    return [
        *(
            deform_protocol(
                spec,
                timestep_fs=timestep_fs,
                replica=replica,
                reference_box_nm=reference_box_nm,
            )
            for replica in range(spec.n_replicas)
        ),
        *(
            Protocol(PROTOCOL_NAME, (stage,))
            for stage in extra_stages(spec, timestep_fs=timestep_fs)
        ),
    ]


def mechanical_scan(spec: ModulusSpec = DEFAULT_SPEC, **equilibration: Any) -> Protocol:
    """Every stage the scan runs, for inspection and dry-run cost estimation.

    The equilibration, each replica's extension and the load, bulk and shear
    passes, at a 2 fs timestep that changes no duration.
    :func:`run_modulus_scan` runs the passes as separate protocols, each
    branching from the equilibrated cell, rather than one after another as
    they are listed here.

    Args:
        spec: What to run.
        **equilibration: Passed to
            :func:`~openmmpolymer.protocols.standard_melt_equilibration`.

    Returns:
        The listing.
    """
    return scan_listing(
        PROTOCOL_NAME,
        [
            equilibration_protocol(spec, **equilibration),
            *_branches(spec, timestep_fs=2.0),
        ],
    )


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def _report_cost(
    spec: ModulusSpec, settle: Protocol, manifest: RunManifest | None
) -> None:
    """Say what the whole scan costs before any of it runs.

    Raises:
        MechanicalError: The total is over ``max_total_ns``.
    """
    schedule = deform_schedule(spec)
    listing = scan_listing(PROTOCOL_NAME, [settle, *_branches(spec, timestep_fs=2.0)])
    total_ps = listing.total_duration_ps
    log.info(
        "Mechanical scan: %.1f ns equilibration, %.1f ns of extension (%d "
        "replicas of %d steps to %.1f%% strain at %.3g /ns), %.1f ns of "
        "load, bulk and shear - %.1f ns in total, %.1f ns of it still to run.",
        settle.total_duration_ps / 1000.0,
        schedule.total_ps * spec.n_replicas / 1000.0,
        spec.n_replicas,
        schedule.n_steps,
        100.0 * schedule.max_strain,
        schedule.strain_rate_per_ns,
        sum(stage.duration_ps for stage in extra_stages(spec, timestep_fs=2.0))
        / 1000.0,
        total_ps / 1000.0,
        remaining_ps(listing.stages, manifest) / 1000.0,
    )
    if spec.max_total_ns is not None and total_ps / 1000.0 > spec.max_total_ns:
        raise MechanicalError(
            f"The scan is {total_ps / 1000.0:.1f} ns, over the "
            f"{spec.max_total_ns:.1f} ns budget. Shorten relax_ps, drop a "
            "replica, skip a pass, or raise max_total_ns."
        )


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
) -> ModulusReport:
    """Equilibrate a cell, measure its elastic constants, and report them.

    Nothing is written before the budget and a resumed directory's settings
    have been checked. The workflow record is written once the scan is done.

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
        What :func:`analyse_mechanics` reads back from the finished run.

    Raises:
        MechanicalError: The scan is over budget, or this directory holds one
            run with different settings.
    """
    directory = Path(run_dir)
    request = spec_request(spec)
    record = (
        check_request(directory / WORKFLOW_NAME, request, error=MechanicalError)
        if resume
        else {}
    )
    settle = equilibration_protocol(spec, **equilibration)
    _report_cost(spec, settle, RunManifest.load(directory) if resume else None)

    timestep_fs = safe_timestep_fs(spec.temperature_k, run.spec)
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    start_state, origin = equilibrate(
        settle,
        run,
        directory,
        resume=resume,
        error=MechanicalError,
        verb="deform",
        **chains,
    )
    run_branches(
        _branches(spec, timestep_fs=timestep_fs, reference_box_nm=origin),
        run,
        directory,
        start_state,
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
    write_json(directory / WORKFLOW_NAME, record, strict=False)
    _log_result(report, deform_schedule(spec))
    return report


def _log_result(report: ModulusReport, schedule: ModulusSchedule) -> None:
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
        "" if report.resolved else " (not resolved)",
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
        groups = group_by_stem(deform_stages(directory))
    except AnalysisError as error:
        groups = []
        notes.append(f"No extension to fit: {error}")

    for group in groups:
        curve = stress_strain(directory, group)
        curves.append(curve)
        replicas.append(
            youngs_modulus(curve, strain_limit=strain_limit, min_points=min_points)
        )

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
    spread = sample_spread([fit.modulus_mpa for fit in replicas])

    # Named rather than found by shape, and this is the one place that rule
    # is inverted. A bulk ladder is run by the shared `compress` runner, so it
    # records exactly what the equilibration's compression ladder records -
    # and that one climbs to a kilobar at the melt temperature, which fitted
    # as a bulk modulus is a confident number about nothing.
    manifest = RunManifest.load(directory)
    stage_names = list(manifest.stages) if manifest is not None else []
    if BULK_STEM in stage_names:
        bulk = optional(
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
    shear = optional(lambda: shear_modulus(directory), notes, "No shear ladder to fit")
    load = optional(lambda: load_curve(directory), notes, "No constant-stress pass")
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
        resolved=bool(
            youngs is not None
            and youngs.resolved
            and (
                spread is None
                or youngs.modulus_mpa <= 0.0
                or spread <= MAX_REPLICA_SPREAD * youngs.modulus_mpa
            )
        ),
        notes=tuple(notes),
    )


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


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _modulus_record(fit: ElasticModulus) -> dict[str, Any]:
    """One modulus fit as plain JSON types."""
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
    """Write ``mechanics.json`` and its figures into ``<run_dir>/analysis``.

    Or into *output_dir*, when the run directory should not be touched.
    """
    fields: dict[str, Any] = {
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
        "resolved": report.resolved,
        "notes": list(report.notes),
    }
    return write_report_files(
        report.run_dir,
        output_dir,
        "mechanics.json",
        fields,
        _figures(report) if figures else (),
        figure_format,
    )


def _figures(report: ModulusReport) -> Iterator[tuple[str, Figure]]:
    """Each replica's stress-strain curve, the constant-stress one, the moduli."""
    for curve, fit in zip(report.curves, report.replicas, strict=False):
        stem = curve.stage.replace(", ", "_").replace(" ", "_")
        yield (
            f"stress_strain_{stem}",
            plot_stress_strain(curve, fit=fit, poisson=report.poisson),
        )
    if report.load is not None:
        yield "load_curve", plot_stress_strain(report.load, fit=report.load_modulus)
    if report.youngs is not None:
        yield "moduli", plot_moduli(report)
