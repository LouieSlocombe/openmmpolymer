"""Measuring how a polymer's stress decays, and reporting what was measured.

One deformation, applied once, and then nothing but waiting.
:mod:`openmmpolymer.mechanical` asks how hard a cell pushes back; this asks how
long it keeps pushing, which is the question a polymer has a more interesting
answer to. What comes back is ``G(t)`` - and through it ``E(t)`` - fitted as a
stretched exponential and as a discrete relaxation spectrum.

Every replica branches from the *same* equilibrated cell with fresh velocities,
the way every mechanical pass does, and for the same reason: a cell that has
just been held at three per cent strain for ten nanoseconds is not the cell the
next measurement wants. Replicas matter more here than anywhere else in this
package, and not only for the error bar. The stress is binned logarithmically
in time, so the earliest bins hold one reading each; a single run resolves
about the decade around the largest stress and loses the rest in noise, and
adding independent runs is the only thing that pushes the usable window out at
both ends. Budget them first and the sampling cadence second.

Two things it will not do. It will not call a decay measured when the run
stopped before the decay did - a spectrum whose slowest term carries the weight
is saying exactly that, and ``plateau_reached`` reports it. And it will not let
a stretched exponential stand unqualified over a curve that has a rubbery
plateau under it: a KWW decays to zero by construction, so when the Prony fit
finds an equilibrium modulus and the KWW claims to describe the same curve, the
two functional forms disagree and the report says so rather than quoting the
prettier of them.

The strain travels with every number, the way the strain rate travels with a
modulus and the cooling rate with a glass transition. A relaxation modulus is
only a material property inside the linear viscoelastic region, and the only
way to find out whether a strain was inside it is to repeat the measurement at
another one - which ``linearity_strains`` does, and which is off by default
because it doubles the cost of the whole scan.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._files import ReportFiles, write_json
from ._validation import (
    require_axis,
    require_choice,
    require_integer,
    require_plane,
    require_positive,
)
from ._workflow import (
    equilibrated_box_nm,
    group_by_stem,
    sample_spread,
    settled_state,
)
from .protocols import (
    Protocol,
    RunManifest,
    Stage,
    run_protocol,
    standard_melt_equilibration,
)
from .relaxation import (
    MIN_RELAXATION_POINTS,
    KWWFit,
    PronyFit,
    RelaxationCurve,
    fit_kww,
    fit_prony,
    mean_curve,
    relax_stages,
    relaxation_curve,
)
from .simulate import RELAX_MODES, RunContext, relax_bin_edges_ps, safe_timestep_fs
from .trajectory import AnalysisError

log = logging.getLogger(__name__)

#: What every pass calls itself in the manifest, so an interrupted run still
#: says which workflow it belongs to.
PROTOCOL_NAME = "viscoelastic"

#: Stage-name stems. Numbered so the run directory sorts into run order, and
#: free of dots because a stage's name becomes a file stem.
RELAX_STEM = "06_relax"
LINEARITY_STEM = "07_linearity"

#: The equilibrated stage every replica branches from.
EQUILIBRATION_STAGE = "05_npt"

#: Where the workflow records what it derived, beside the manifest but not in
#: it: the manifest is the record of what ran, and an analysis-shaped field in
#: it would be stale after a later resume.
WORKFLOW_NAME = "viscoelastic_workflow.json"

#: How far the replicas may disagree about the initial modulus, relative to
#: their mean, before the curve stops claiming to be resolved.
MAX_REPLICA_SPREAD = 0.3

#: How large the pre-strain deviatoric stress may be, as a fraction of the
#: initial response, before the cell was carrying too much to measure from.
MAX_BASELINE_FRACTION = 0.25

#: How far two strains' moduli may differ before they are not both inside the
#: linear viscoelastic region.
MAX_LINEARITY_GAP = 0.2

#: How large an equilibrium modulus has to be, relative to the initial one,
#: before a stretched exponential - which decays to zero - is the wrong shape
#: for the curve.
MIN_PLATEAU_FRACTION = 0.05

#: Smallest denominator worth dividing by.
_TINY = 1.0e-12


class ViscoelasticError(RuntimeError):
    """A relaxation could not be run, or was asked for twice over."""


@dataclass(frozen=True)
class RelaxationSpec:
    """Everything a relaxation scan needs to know, in one place.

    Args:
        temperature_k: The temperature everything is measured at. Which side
            of the glass transition it falls on decides what is being
            measured, and nothing here knows which - below it the decay is
            local and over in nanoseconds, above it the chains themselves have
            to move and this sees the start of it at best. Find the transition
            first with :func:`~openmmpolymer.tg.run_tg_scan`.
        pressure_bar: The pressure the cell is equilibrated at, before the box
            is locked.
        mode: ``"tensile"`` or ``"shear"``. Both measure ``G(t)``; a shear
            step does it without imposing a lateral contraction at all, which
            above the glass transition is the cleaner deformation.
        axis: The axis to stretch, for a tensile step.
        plane: ``(driven, gradient)`` axes, for a shear step.
        step_strain: The strain applied, all at once. It has to be small
            enough to be inside the linear viscoelastic region and large
            enough to be seen over the virial noise, and those pull opposite
            ways - which is what *linearity_strains* is for.
        poisson: The lateral contraction a tensile step imposes. The default
            preserves the volume. It is a property of the deformation and not
            of the polymer, and nothing measured here depends on the two
            agreeing.
        ramp_ps: Apply the strain over this long rather than instantaneously.
            Zero by default, because an instantaneous step is what a
            relaxation modulus is defined against.
        baseline_ps: Time at the locked box before straining, to measure what
            the cell was already carrying. Its scatter is the floor the decay
            is read against, so this sets how far down the curve is legible.
        relax_ps: The whole relaxation, per replica.
        n_replicas: Independent runs from the same configuration with fresh
            velocities. The first knob to turn: their spread is the only
            honest error bar, and averaging them is what makes the early bins
            - which hold one reading each - mean anything.
        sample_every_ps: Time between stress readings, early on.
        late_sample_every_ps: Time between them after *late_after_ps*.
        late_after_ps: When to change down. Each reading costs about six
            energy evaluations, so a single dense cadence over a long run is
            a tenth of the run's cost and this pair is a thousandth.
        bins_per_decade: Logarithmic time bins per decade.
        stage_ps: Most relaxation one stage may hold before it is split into
            another. This is the resume granularity and nothing else: the
            chunks are continuous, and each is told where it sits on the
            clock.
        linearity_strains: Other strains to repeat the whole measurement at,
            or None to skip. Inside the linear region the moduli coincide;
            outside it they do not, and there is no other way to find out from
            one strain alone. Off by default because each strain costs another
            full set of replicas.
        write_raw: Write every stress reading beside the binned curve.
        max_total_ns: Refuse to start if the scan would exceed this.
    """

    temperature_k: float = 298.15
    pressure_bar: float = 1.0
    mode: str = "tensile"
    axis: int = 2
    plane: tuple[int, int] = (0, 2)
    step_strain: float = 0.03
    poisson: float = 0.5
    ramp_ps: float = 0.0
    baseline_ps: float = 1000.0
    relax_ps: float = 10_000.0
    n_replicas: int = 4
    sample_every_ps: float = 0.05
    late_sample_every_ps: float = 5.0
    late_after_ps: float = 200.0
    bins_per_decade: int = 20
    stage_ps: float = 20_000.0
    linearity_strains: tuple[float, ...] | None = None
    write_raw: bool = True
    max_total_ns: float | None = None

    def __post_init__(self) -> None:
        """Reject a spec that cannot describe a relaxation, at the call site."""
        for name in (
            "temperature_k",
            "pressure_bar",
            "baseline_ps",
            "relax_ps",
            "sample_every_ps",
            "late_sample_every_ps",
            "late_after_ps",
            "stage_ps",
        ):
            require_positive(getattr(self, name), None, name=name)
        require_positive(abs(self.step_strain), None, name="step_strain")
        require_integer(self.n_replicas, minimum=1, name="n_replicas")
        require_integer(self.bins_per_decade, minimum=1, name="bins_per_decade")
        require_choice(self.mode, RELAX_MODES, name="mode")
        require_axis(self.axis)
        require_plane(self.plane)
        if self.ramp_ps < 0.0:
            raise ValueError(f"ramp_ps={self.ramp_ps} cannot be negative.")
        if self.sample_every_ps >= self.relax_ps:
            raise ValueError(
                f"sample_every_ps={self.sample_every_ps} is not below "
                f"relax_ps={self.relax_ps}, so there is no time axis to bin."
            )
        if self.linearity_strains is not None:
            if not self.linearity_strains:
                raise ValueError(
                    "linearity_strains is empty; pass None to skip the pass."
                )
            if any(abs(value) <= 0.0 for value in self.linearity_strains):
                raise ValueError(
                    f"linearity_strains={self.linearity_strains} must all be "
                    "non-zero strains."
                )
        if self.max_total_ns is not None:
            require_positive(self.max_total_ns, None, name="max_total_ns")


#: The default settings, as a shared frozen singleton so it can be a default
#: argument without being rebuilt on every call.
DEFAULT_SPEC = RelaxationSpec()


@dataclass(frozen=True)
class RelaxationSchedule:
    """One replica's relaxation, and what it costs.

    Args:
        n_chunks: How many stages it is split into for resume.
        chunk_ps: The longest of them.
        relax_ps: The whole relaxation.
        baseline_ps: The pre-strain window in front of it.
        sample_every_ps: The earliest a reading can land, which is where the
            logarithmic grid starts.
        n_bins: Logarithmic bins the decay is pooled into.
    """

    n_chunks: int
    chunk_ps: float
    relax_ps: float
    baseline_ps: float
    sample_every_ps: float
    n_bins: int

    @property
    def total_ps(self) -> float:
        """How much dynamics one replica is."""
        return self.baseline_ps + self.relax_ps

    @property
    def decades(self) -> float:
        """How many decades of time the relaxation can span."""
        return float(math.log10(self.relax_ps / self.sample_every_ps))


@dataclass(frozen=True)
class LinearityCheck:
    """Whether two or more strains gave the same modulus.

    The only test of the linear viscoelastic region there is from inside a
    simulation. A relaxation modulus is a material property only where it does
    not depend on the strain that produced it; below that it does, and the
    number is about the deformation instead.

    Args:
        strains: The strains compared.
        initial_mpa: The initial modulus measured at each.
        gap: The largest relative difference between them.
        linear: Whether that gap is below :data:`MAX_LINEARITY_GAP`. False
            when nothing could be compared: a check that did not run is not a
            check that passed.
    """

    strains: tuple[float, ...]
    initial_mpa: tuple[float, ...]
    gap: float
    linear: bool


@dataclass(frozen=True)
class RelaxationReport:
    """What a finished run directory says about its stress relaxation.

    Args:
        run_dir: The directory read.
        curves: Every replica's curve, in the order they were found.
        mean: Their ensemble average, or None when there was nothing to read.
        kww: The stretched exponential fitted to that average, or None.
        prony: The relaxation spectrum fitted to it, or None.
        replica_spread_mpa: How far the replicas disagreed about the initial
            modulus - the honest error bar, and None below two replicas.
        linearity: The comparison across strains, or None if only one was run.
        plateau_conflict: The Prony fit found an equilibrium modulus worth
            having and the KWW - which decays to zero - claimed to describe
            the same curve anyway. Two functional forms disagreeing about
            whether the material relaxes completely, which is worth more than
            either of them agreeing with itself.
        baseline_fraction: The pre-strain deviatoric stress over the initial
            response. Large means the cell was not isotropic to begin with.
        notes: Anything that could not be read, in plain English.
    """

    run_dir: str
    curves: tuple[RelaxationCurve, ...]
    mean: RelaxationCurve | None
    kww: KWWFit | None
    prony: PronyFit | None
    replica_spread_mpa: float | None
    linearity: LinearityCheck | None
    plateau_conflict: bool
    baseline_fraction: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RelaxationResult:
    """What one relaxation scan measured.

    Args:
        run_dir: Where it ran.
        manifest_path: The manifest it wrote.
        mean: The ensemble-averaged ``G(t)``, or None when nothing ran.
        kww: The stretched exponential, or None.
        prony: The relaxation spectrum, or None.
        curves: One curve per replica.
        replica_spread_mpa: Their spread about the initial modulus.
        linearity: The strain-dependence check, or None if it was skipped.
        schedule: What each replica walked.
        resolved: There is a decay whose fits resolved, whose replicas agree,
            and which was measured from a cell that was not already stressed.
    """

    run_dir: str
    manifest_path: str
    mean: RelaxationCurve | None
    kww: KWWFit | None
    prony: PronyFit | None
    curves: tuple[RelaxationCurve, ...]
    replica_spread_mpa: float | None
    linearity: LinearityCheck | None
    schedule: RelaxationSchedule
    resolved: bool

    @property
    def mean_tau_ps(self) -> float | None:
        """The headline number, or None when nothing resolved."""
        if self.kww is None or not self.resolved:
            return None
        return self.kww.mean_tau_ps


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


def relax_schedule(spec: RelaxationSpec) -> RelaxationSchedule:
    """What one replica walks, and how it is split for resume.

    Args:
        spec: What to run.

    Returns:
        The schedule.
    """
    chunk_ps = min(spec.stage_ps, spec.relax_ps)
    edges = relax_bin_edges_ps(
        spec.sample_every_ps, spec.relax_ps, spec.bins_per_decade
    )
    return RelaxationSchedule(
        n_chunks=max(1, math.ceil(spec.relax_ps / chunk_ps)),
        chunk_ps=chunk_ps,
        relax_ps=spec.relax_ps,
        baseline_ps=spec.baseline_ps,
        sample_every_ps=spec.sample_every_ps,
        n_bins=int(edges.size - 1),
    )


def _relax_stages(
    stem: str, spec: RelaxationSpec, *, timestep_fs: float, strain: float
) -> tuple[Stage, ...]:
    """Turn one relaxation into the stages that run it.

    Split into chunks no longer than ``stage_ps``, which is bookkeeping and
    not physics: each chunk starts from the state the one before it left, so
    the hold is continuous and the cell never knows. What it buys is resume
    granularity, a stage being the unit a run picks itself back up at.

    Each chunk is told where it sits on the relaxation clock and that the
    strain is already applied, because neither survives in a state file. A
    resumed chunk that called its own start time zero would fold the slow end
    of the decay back on top of the fast end, and one that strained again
    would measure the response to six per cent while reporting three.

    Only the first chunk measures a baseline, ramps, or draws fresh
    velocities - the rest are a continuation of it, not a repeat.
    """
    schedule = relax_schedule(spec)
    stages: list[Stage] = []
    done = 0.0
    for index in range(schedule.n_chunks):
        length = min(schedule.chunk_ps, spec.relax_ps - done)
        first = index == 0
        options: dict[str, Any] = {
            "temperature_k": spec.temperature_k,
            "mode": spec.mode,
            "step_strain": strain,
            "poisson": spec.poisson,
            "duration_ps": length,
            # The grid spans the whole relaxation and not this chunk of it,
            # which is what puts every chunk and every replica on one set of
            # bins and makes merging them the same addition.
            "total_ps": spec.relax_ps,
            "time_offset_ps": done,
            "strain_applied": not first,
            "baseline_ps": spec.baseline_ps if first else 0.0,
            "ramp_ps": spec.ramp_ps if first else 0.0,
            "sample_every_ps": spec.sample_every_ps,
            "late_sample_every_ps": spec.late_sample_every_ps,
            "late_after_ps": spec.late_after_ps,
            "bins_per_decade": spec.bins_per_decade,
            "new_velocities": first,
            "timestep_fs": timestep_fs,
            "write_raw": spec.write_raw,
        }
        if spec.mode == "shear":
            options["plane"] = tuple(spec.plane)
        else:
            options["axis"] = spec.axis
        stages.append(Stage(f"{stem}_{index:02d}", "relax", options))
        done += length
    return tuple(stages)


def relax_protocol(
    spec: RelaxationSpec,
    *,
    timestep_fs: float,
    replica: int = 0,
    strain: float | None = None,
    stem: str = RELAX_STEM,
    reference_box_nm: Sequence[float] | None = None,
) -> Protocol:
    """One replica's relaxation, and nothing else.

    One timestep is pinned across every chunk rather than let each derate to
    its own temperature, because this is a measurement and chunks integrated
    differently are a confound nobody would choose.

    Args:
        spec: What to run.
        timestep_fs: The timestep every chunk uses.
        replica: Which repeat this is. It only changes the stage names, which
            is enough: every random stream derives from the stage label, so a
            differently-named replica is a differently-seeded one.
        strain: The strain to apply, or None for the spec's own. The linearity
            pass gives a different one.
        stem: What to call the stages.
        reference_box_nm: The unstrained cell, recorded by every chunk so a
            resumed one reports the same origin as the first.

    Returns:
        The protocol.
    """
    applied = spec.step_strain if strain is None else float(strain)
    stages = _relax_stages(
        f"{stem}_r{replica}", spec, timestep_fs=timestep_fs, strain=applied
    )
    if reference_box_nm is not None:
        origin = [float(value) for value in reference_box_nm]
        stages = tuple(
            Stage(stage.name, stage.kind, {**stage.options, "reference_box_nm": origin})
            for stage in stages
        )
    return Protocol(name=PROTOCOL_NAME, stages=stages)


def equilibration_protocol(
    spec: RelaxationSpec = DEFAULT_SPEC, **equilibration: Any
) -> Protocol:
    """Settle the melt at the temperature the relaxation will be measured at.

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


def relaxation_scan(
    spec: RelaxationSpec = DEFAULT_SPEC, **equilibration: Any
) -> Protocol:
    """Equilibrate, then run one relaxation - what ``--dry-run`` prices.

    The whole scan is several protocols, because every replica branches from
    the same equilibrated cell rather than following the one before it. This
    is the first two of them, which is what a cost estimate can be built from
    without running anything.

    Args:
        spec: What to run.
        **equilibration: Passed to
            :func:`~openmmpolymer.protocols.standard_melt_equilibration`.

    Returns:
        The protocol.
    """
    base = equilibration_protocol(spec, **equilibration)
    return Protocol(
        name=PROTOCOL_NAME,
        stages=(
            *base.stages,
            *_relax_stages(
                f"{RELAX_STEM}_r0", spec, timestep_fs=2.0, strain=spec.step_strain
            ),
        ),
    )


# --------------------------------------------------------------------------
# The workflow record and the cost
# --------------------------------------------------------------------------


def _request(spec: RelaxationSpec) -> dict[str, Any]:
    """What the caller asked for, as the thing a resume is compared against.

    Round-tripped through JSON before it is compared with anything, because
    that is the form it is stored in. Several of these fields are tuples, and
    a tuple comes back from a file as a list: comparing the two directly makes
    every second run look like a change of settings and refuse to resume.
    """
    stored = json.loads(json.dumps(asdict(spec), default=str))
    return {"spec": cast("dict[str, Any]", stored)}


def _check_request(run_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Refuse a resume that quietly asks for something else.

    A stage's options are recorded nowhere, so a protocol rerun with different
    settings resumes and keeps the old result without a word. That is
    survivable for an equilibration and not for a measurement - and here it is
    worse than elsewhere, because the bin edges come from the settings, so a
    changed duration or bin count would merge two incompatible grids into one
    curve.
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
        raise ViscoelasticError(
            f"{path} records a scan run with different settings "
            f"({', '.join(changed) or 'unknown'}), and resuming would keep "
            "results measured under the old ones. Run into a fresh directory, "
            "or put the settings back."
        )
    return record


def _save_workflow(run_dir: Path, record: dict[str, Any]) -> str:
    """Write the workflow record."""
    return write_json(run_dir / WORKFLOW_NAME, record, strict=False)


def _strains(spec: RelaxationSpec) -> tuple[tuple[str, float], ...]:
    """Every strain the scan runs, with the stem each is named under."""
    passes = [(RELAX_STEM, spec.step_strain)]
    for index, value in enumerate(spec.linearity_strains or ()):
        passes.append((f"{LINEARITY_STEM}_e{index}", float(value)))
    return tuple(passes)


def _report_cost(
    equilibration: Protocol,
    schedule: RelaxationSchedule,
    spec: RelaxationSpec,
    manifest: RunManifest | None,
) -> None:
    """Say what the whole thing costs before any of it runs.

    Raises:
        ViscoelasticError: The total is over ``max_total_ns``.
    """
    settle_ps = equilibration.total_duration_ps
    passes = len(_strains(spec))
    relax_ps = schedule.total_ps * spec.n_replicas * passes
    total_ps = settle_ps + relax_ps

    done = set(manifest.stages) if manifest is not None else set()
    remaining = total_ps - sum(
        stage.duration_ps for stage in equilibration.stages if stage.name in done
    )
    for stem, strain in _strains(spec):
        for replica in range(spec.n_replicas):
            remaining -= sum(
                stage.duration_ps
                for stage in _relax_stages(
                    f"{stem}_r{replica}", spec, timestep_fs=2.0, strain=strain
                )
                if stage.name in done
            )

    log.info(
        "Relaxation scan: %.1f ns equilibration, %.1f ns of relaxation (%d "
        "replicas of %.1f ns at %+.3f strain%s, %d chunks each, %d bins over "
        "%.1f decades) - %.1f ns in total, %.1f ns of it still to run.",
        settle_ps / 1000.0,
        relax_ps / 1000.0,
        spec.n_replicas,
        schedule.total_ps / 1000.0,
        spec.step_strain,
        "" if passes == 1 else f" and {passes - 1} more for linearity",
        schedule.n_chunks,
        schedule.n_bins,
        schedule.decades,
        total_ps / 1000.0,
        max(0.0, remaining) / 1000.0,
    )
    if spec.max_total_ns is not None and total_ps / 1000.0 > spec.max_total_ns:
        raise ViscoelasticError(
            f"The scan is {total_ps / 1000.0:.1f} ns, over the "
            f"{spec.max_total_ns:.1f} ns budget. Shorten relax_ps, drop a "
            "replica, skip the linearity pass, or raise max_total_ns."
        )


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def run_relaxation_scan(
    run: RunContext,
    run_dir: str | Path = "run",
    *,
    spec: RelaxationSpec = DEFAULT_SPEC,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> RelaxationResult:
    """Equilibrate a cell, strain it once, and watch the stress decay.

    Every replica branches from the equilibrated cell, not from the replica
    before it, so each is run as its own protocol with that state named
    explicitly and fresh velocities drawn. A cell that has spent ten
    nanoseconds held at three per cent strain is not the cell the next
    measurement wants, and inheriting its velocities as well as its positions
    would give the same trajectory every time - a spread of zero dressed up as
    an error bar.

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
        ViscoelasticError: The scan is over budget, or this directory holds one
            run with different settings.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request = _request(spec)
    record = _check_request(directory, request) if resume else {}

    settle = equilibration_protocol(spec, **equilibration)
    schedule = relax_schedule(spec)
    timestep_fs = safe_timestep_fs(spec.temperature_k, run.spec)
    _report_cost(
        settle, schedule, spec, RunManifest.load(directory) if resume else None
    )

    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    settled = run_protocol(settle, run, directory, resume=resume, **chains)
    start_state = settled_state(
        settled, directory, error=ViscoelasticError, verb="strain"
    )
    origin = equilibrated_box_nm(start_state)
    log.info(
        "Equilibrated cell is %s nm; every replica starts from %s.",
        [round(value, 4) for value in origin],
        Path(start_state).name,
    )

    for stem, strain in _strains(spec):
        for replica in range(spec.n_replicas):
            run_protocol(
                relax_protocol(
                    spec,
                    timestep_fs=timestep_fs,
                    replica=replica,
                    strain=strain,
                    stem=stem,
                    reference_box_nm=origin,
                ),
                run,
                directory,
                # Only preparation resets the manifest for a forced rerun.
                resume=True,
                state_in=start_state,
                **chains,
            )

    report = analyse_relaxation(directory)
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

    initial = report.mean.initial_modulus_mpa if report.mean is not None else math.nan
    resolved = bool(
        report.kww is not None
        and report.kww.resolved
        and not report.plateau_conflict
        and report.baseline_fraction <= MAX_BASELINE_FRACTION
        and (
            report.replica_spread_mpa is None
            or not math.isfinite(initial)
            or abs(initial) <= _TINY
            or report.replica_spread_mpa <= MAX_REPLICA_SPREAD * abs(initial)
        )
    )
    _log_result(report, schedule, resolved)
    return RelaxationResult(
        run_dir=str(directory),
        manifest_path=str(directory / "manifest.json"),
        mean=report.mean,
        kww=report.kww,
        prony=report.prony,
        curves=report.curves,
        replica_spread_mpa=report.replica_spread_mpa,
        linearity=report.linearity,
        schedule=schedule,
        resolved=resolved,
    )


def _log_result(
    report: RelaxationReport, schedule: RelaxationSchedule, resolved: bool
) -> None:
    """One line per measured quantity, each with what qualifies it."""
    if report.mean is None:
        log.info("Relaxation scan: nothing was strained.")
        return
    log.info(
        "Relaxation scan: G(0) = %.1f MPa over %.1f decades from %d replicas, "
        "at %+.3f strain, %.0f K%s.",
        report.mean.initial_modulus_mpa,
        report.mean.decades,
        report.mean.n_replicas,
        report.mean.step_strain,
        report.mean.temperature_k,
        "" if resolved else " (not resolved)",
    )
    if report.kww is not None:
        log.info(
            "  KWW: beta = %.3f, tau = %.4g ps, <tau> = %.4g ps%s.",
            report.kww.beta,
            report.kww.tau_ps,
            report.kww.mean_tau_ps,
            "" if report.kww.resolved else " (not resolved)",
        )
    if report.prony is not None:
        log.info(
            "  Prony: G_inf = %.2f MPa over %d of %d terms%s.",
            report.prony.equilibrium_mpa,
            report.prony.n_active,
            report.prony.n_terms,
            "" if report.prony.plateau_reached else " - the decay had not finished",
        )
    if report.plateau_conflict:
        log.warning(
            "  The spectrum finds an equilibrium modulus but the stretched "
            "exponential, which decays to zero, claims the same curve. One of "
            "the two is the wrong shape for this material; take the spectrum."
        )
    if report.linearity is not None and not report.linearity.linear:
        log.warning(
            "  Strains %s gave moduli differing by %.0f%%, so at least one of "
            "them is outside the linear viscoelastic region.",
            [round(value, 4) for value in report.linearity.strains],
            100.0 * report.linearity.gap,
        )
    if math.isfinite(report.baseline_fraction):
        log.info(
            "  the cell started at %.0f%% of the initial response.",
            100.0 * report.baseline_fraction,
        )
    del schedule


# --------------------------------------------------------------------------
# Reading a finished run
# --------------------------------------------------------------------------


def _primary_strain(by_strain: dict[float, list[RelaxationCurve]]) -> float:
    """Which strain's ensemble is the measurement, and which are the check.

    Named rather than found by shape, and this is the one place that rule is
    inverted - the same inversion
    :func:`~openmmpolymer.mechanical.analyse_mechanics` makes for its bulk
    pass, and for the same reason. A linearity pass records *exactly* what the
    measurement records, because it is the same measurement at another strain;
    nothing in the samples distinguishes them, and the workflow runs the same
    number of replicas of each, so counting replicas cannot either. Left to
    tie-break on the strain itself, a linearity pass at a larger strain would
    quietly become the headline result: the reported modulus and both fits
    would belong to the pass that only existed to check the other one.

    So this asks for the stages this workflow writes the measurement under. A
    directory assembled some other way has no such stages, and then the
    ensemble with the most replicas is the best guess available.
    """
    for strain, curves in by_strain.items():
        if all(curve.stage.startswith(RELAX_STEM) for curve in curves):
            return strain
    return max(by_strain, key=lambda value: (len(by_strain[value]), abs(value)))


def _linearity(by_strain: dict[float, RelaxationCurve]) -> LinearityCheck | None:
    """Compare the initial modulus measured at each strain.

    Inside the linear viscoelastic region the curves coincide, because a
    relaxation modulus is a property of the material there and not of the
    deformation. Outside it they separate, and no amount of looking at one
    strain would have said so.
    """
    if len(by_strain) < 2:
        return None
    strains = tuple(sorted(by_strain))
    initial = tuple(by_strain[value].initial_modulus_mpa for value in strains)
    usable = [value for value in initial if math.isfinite(value)]
    if len(usable) < 2 or abs(float(np.mean(usable))) < _TINY:
        return LinearityCheck(strains, initial, math.nan, False)
    gap = (max(usable) - min(usable)) / abs(float(np.mean(usable)))
    return LinearityCheck(strains, initial, gap, gap <= MAX_LINEARITY_GAP)


def analyse_relaxation(
    run_dir: str | Path,
    *,
    min_points: int = MIN_RELAXATION_POINTS,
) -> RelaxationReport:
    """Read everything a finished run has to say about its stress relaxation.

    Reads and returns; writes nothing. Relaxations are found by what they
    recorded rather than by what they were called, replicas are told apart by
    their stage stems, and several strains - a linearity pass - come back as
    several ensembles rather than one confused average.

    The headline fits go to the ensemble average rather than to each replica
    separately. That is not the choice
    :func:`~openmmpolymer.mechanical.analyse_mechanics` makes about a modulus,
    and the difference is real: pooling points there gives a longer curve to
    fit, while averaging here gives *the same* curve with less noise on it,
    which is the only way the early bins - a single reading each - say
    anything at all.

    Args:
        run_dir: A directory a run wrote to.
        min_points: Fewest bins a fit may rest on.

    Returns:
        The report.

    Raises:
        AnalysisError: There is no manifest, or nothing in it was a relaxation.
    """
    directory = Path(run_dir)
    notes: list[str] = []
    curves: list[RelaxationCurve] = []

    for group in group_by_stem(relax_stages(directory)):
        try:
            curves.append(relaxation_curve(directory, group))
        except AnalysisError as error:
            notes.append(f"Skipped {', '.join(group)}: {error}")

    if not curves:
        raise AnalysisError(
            f"Nothing in {directory} was a stress relaxation that could be "
            f"read.{' ' + ' '.join(notes) if notes else ''}"
        )

    # Grouped by the strain each was run at, so a linearity pass is several
    # ensembles rather than one average over curves that measure different
    # things.
    by_strain: dict[float, list[RelaxationCurve]] = {}
    for curve in curves:
        by_strain.setdefault(round(curve.step_strain, 9), []).append(curve)
    primary = _primary_strain(by_strain)
    ensemble = by_strain[primary]

    mean = mean_curve(ensemble)
    kww = fit_kww(mean, min_points=min_points)
    prony = fit_prony(mean, min_points=min_points)
    spread = sample_spread([curve.initial_modulus_mpa for curve in ensemble])
    linearity = _linearity(
        {value: mean_curve(group) for value, group in by_strain.items()}
    )

    initial = mean.initial_modulus_mpa
    fraction = (
        abs(mean.baseline_mpa / initial)
        if math.isfinite(initial) and abs(initial) > _TINY
        else math.inf
    )
    if fraction > MAX_BASELINE_FRACTION:
        notes.append(
            f"The cell was already carrying {100.0 * fraction:.0f}% of the "
            "initial response as deviatoric stress before it was strained, so "
            "it was not the isotropic cell this measurement assumes."
        )

    # A stretched exponential decays to zero by construction, so a curve with
    # a rubbery plateau under it is a curve a KWW cannot describe. The two
    # fits share no machinery, which is what makes the spectrum finding one
    # worth acting on: usually the KWW simply refuses, and then its refusal is
    # the message; occasionally it resolves anyway, and then the two forms
    # flatly contradict each other about whether the material ever relaxes.
    plateau = bool(
        prony.resolved
        and math.isfinite(initial)
        and abs(initial) > _TINY
        and prony.equilibrium_mpa / abs(initial) > MIN_PLATEAU_FRACTION
    )
    conflict = bool(plateau and kww.resolved)
    if plateau:
        notes.append(
            f"The spectrum finds an equilibrium modulus of "
            f"{prony.equilibrium_mpa:.2f} MPa, which is "
            f"{100.0 * prony.equilibrium_mpa / abs(initial):.0f}% of the "
            "initial response. A stretched exponential decays to zero and "
            "cannot represent that, so quote the spectrum."
            + (
                " The stretched exponential nonetheless reports itself "
                "resolved over the same curve, so the two disagree outright."
                if conflict
                else " The stretched exponential does not resolve here, which "
                "is it saying the same thing."
            )
        )
    if not prony.plateau_reached:
        notes.append(
            "The slowest term in the spectrum carries most of the weight, so "
            "the decay was still going when the run stopped. Whatever "
            "equilibrium modulus came out is a plateau nobody watched it "
            "reach; run longer before quoting one."
        )

    return RelaxationReport(
        run_dir=str(directory),
        curves=tuple(curves),
        mean=mean,
        kww=kww,
        prony=prony,
        replica_spread_mpa=spread,
        linearity=linearity,
        plateau_conflict=conflict,
        baseline_fraction=fraction,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _curve_record(curve: RelaxationCurve) -> dict[str, Any]:
    """One curve as plain JSON types.

    Written out field by field rather than with ``asdict``, which would drop
    the properties and render the arrays unhelpfully. Spelling it out also
    pins what is on disk independently of how the dataclasses are laid out.
    """
    return {
        "stage": curve.stage,
        "mode": curve.mode,
        "time_ps": curve.time_ps.tolist(),
        "modulus_mpa": curve.modulus_mpa.tolist(),
        "standard_error_mpa": curve.standard_error_mpa.tolist(),
        "n_samples": curve.n_samples.tolist(),
        "step_strain": curve.step_strain,
        "strain_measure": curve.strain_measure,
        "temperature_k": curve.temperature_k,
        "poisson": curve.poisson,
        "baseline_mpa": curve.baseline_mpa,
        "noise_floor_mpa": curve.noise_floor_mpa,
        "instant_mpa": curve.instant_mpa,
        "initial_modulus_mpa": curve.initial_modulus_mpa,
        "n_replicas": curve.n_replicas,
        "n_points": curve.n_points,
        "decades": curve.decades,
    }


def _kww_record(fit: KWWFit) -> dict[str, Any]:
    """A stretched exponential as plain JSON types."""
    return {
        "modulus_mpa": fit.modulus_mpa,
        "tau_ps": fit.tau_ps,
        "beta": fit.beta,
        "mean_tau_ps": fit.mean_tau_ps,
        "residual": fit.residual,
        "half_disagreement": fit.half_disagreement,
        "n_points": fit.n_points,
        "window_ps": list(fit.window_ps),
        "extrapolation_decades": fit.extrapolation_decades,
        "at_bound": fit.at_bound,
        "temperature_k": fit.temperature_k,
        "step_strain": fit.step_strain,
        "resolved": fit.resolved,
    }


def _prony_record(fit: PronyFit) -> dict[str, Any]:
    """A relaxation spectrum as plain JSON types."""
    return {
        "tau_ps": fit.tau_ps.tolist(),
        "weights_mpa": fit.weights_mpa.tolist(),
        "equilibrium_mpa": fit.equilibrium_mpa,
        "unrelaxed_mpa": fit.unrelaxed_mpa,
        "residual_mpa": fit.residual_mpa,
        "n_points": fit.n_points,
        "n_terms": fit.n_terms,
        "n_active": fit.n_active,
        "edge_weight": fit.edge_weight,
        "window_ps": list(fit.window_ps),
        "plateau_reached": fit.plateau_reached,
        "temperature_k": fit.temperature_k,
        "step_strain": fit.step_strain,
        "resolved": fit.resolved,
    }


def write_relaxation_report(
    report: RelaxationReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write a relaxation report out, as JSON and as figures.

    Args:
        report: What :func:`analyse_relaxation` found.
        output_dir: Where to write, defaulting to ``<run_dir>/analysis``. Give
            one when the run directory should not be touched.
        figures: Write figures as well as the record.
        figure_format: What matplotlib should save them as.

    Returns:
        Where everything went.
    """
    from importlib.metadata import PackageNotFoundError, version

    from .plots import plot_relaxation, plot_relaxation_spectrum

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
        "mean": None if report.mean is None else _curve_record(report.mean),
        "kww": None if report.kww is None else _kww_record(report.kww),
        "prony": None if report.prony is None else _prony_record(report.prony),
        "replicas": [_curve_record(curve) for curve in report.curves],
        "replica_spread_mpa": report.replica_spread_mpa,
        "linearity": (
            None
            if report.linearity is None
            else {
                "strains": list(report.linearity.strains),
                "initial_mpa": list(report.linearity.initial_mpa),
                "gap": report.linearity.gap,
                "linear": report.linearity.linear,
            }
        ),
        "plateau_conflict": report.plateau_conflict,
        "baseline_fraction": report.baseline_fraction,
        "notes": list(report.notes),
    }
    json_path = directory / "relaxation.json"
    write_json(json_path, record, strict=False)

    written: list[str] = []
    if figures and report.mean is not None:
        written.append(
            _save(
                plot_relaxation(
                    report.mean,
                    kww=report.kww,
                    prony=report.prony,
                    replicas=report.curves,
                ),
                directory / f"relaxation.{figure_format}",
            )
        )
        if report.prony is not None and report.prony.n_terms:
            written.append(
                _save(
                    plot_relaxation_spectrum(report.prony),
                    directory / f"relaxation_spectrum.{figure_format}",
                )
            )
    return ReportFiles(json=str(json_path), figures=tuple(written))


def _save(figure: Any, path: Path) -> str:
    """Save a figure and close it, returning where it went."""
    figure.savefig(path, bbox_inches="tight")
    return str(path)
