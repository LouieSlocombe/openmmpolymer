"""Locating a glass transition, in two passes, and reporting what was found.

A quench records a density at every temperature on the way down, and the break
in that curve is where the melt stopped keeping up with the cooling. Resolving
the break needs small temperature steps and long holds, and paying for that
resolution from the melt temperature all the way to the floor is most of the
cost for none of the answer. So this module runs the ladder twice: a coarse
scan to find roughly where the break is, then a fine one across a window
centred on it.

That makes the second pass depend on the first pass' result, which a
:class:`~openmmpolymer.protocols.Protocol` cannot express - it is a fixed list
of stages, run blind. The adaptivity lives here instead, in what builds the
protocols: run one, read the manifest back, fit it, and build the next. Nothing
in the runner changes, and resume keeps working because the fit is recomputed
from the manifest rather than carried in memory.

The fine pass starts from a state the coarse pass saved as it went past, not
from the equilibrated melt and not from the bottom of the coarse ladder. A
glass remembers how it was cooled, so a fine window entered by reheating a
solid is measuring a different thermal history from the one that located it.
Continuing from the waypoint costs nothing and keeps one history.

Two things this module will not do. It will not guess a window when the coarse
fit found a corner in noise: a fine pass is tens of nanoseconds, and spending
them on a window derived from a fit that already said it found nothing is worse
than stopping. And the headline number it reports is the transition at the rate
it was actually cooled at, never the rate-extrapolated one - that extrapolation
spans about ten decades, and promoting it to the answer would be exactly the
overclaiming the rest of this package is built to avoid.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ._validation import require_integer, require_positive
from .conformation import MeanSquaredDisplacement, centre_of_mass_msd
from .protocols import (
    Protocol,
    RunManifest,
    RunSummary,
    Stage,
    run_protocol,
    standard_melt_equilibration,
    validate_run_inputs,
)
from .reporters import TrajectoryOptions
from .simulate import RunContext, quench_temperatures, safe_timestep_fs
from .timeseries import (
    DSC_COOLING_RATE_K_PER_NS,
    CoolingRateExtrapolation,
    Equilibration,
    GlassTransition,
    QuenchCurve,
    cooling_rate_extrapolation,
    equilibration,
    glass_transition,
    quench_curve,
    quench_stages,
    read_state_data,
)
from .trajectory import AnalysisError, open_run

log = logging.getLogger(__name__)

#: What both passes call themselves in the manifest, so an interrupted run
#: still says which workflow it belongs to.
PROTOCOL_NAME = "tg_two_pass"

#: Stage-name stems. Numbered so the run directory sorts into run order, and
#: free of dots because a stage's name becomes a file stem and
#: ``Path.with_suffix`` would read a dot as an extension.
COARSE_STEM = "06_coarse_quench"
PRECOOL_STEM = "07_precool"
FINE_STEM = "08_fine_quench"

#: Where the workflow records what it derived, beside the manifest but not in
#: it. The manifest is the resume ledger and is read by every version of this
#: package; an analysis-shaped field in it would be stale after a later resume
#: and would break an older install's ``RunManifest.load``, which passes every
#: key it finds to the constructor.
WORKFLOW_NAME = "tg_workflow.json"

#: How far below the melt temperature each annealing cycle dips. The anneal
#: needs somewhere to cycle to once the run settles at the temperature it will
#: start cooling from. Clamped to the ladder's floor, because a melt that is
#: cold to begin with would otherwise be asked to cycle through zero kelvin,
#: where a thermostat has nothing to hold and a barostat divides by it.
ANNEAL_DEPTH_K = 150.0

#: How far a chain's centre of mass has to travel, in units of its own squared
#: radius of gyration, before a melt has plausibly forgotten how it was packed.
MSD_RG_MULTIPLE = 2.0


class TgError(RuntimeError):
    """A glass-transition scan could not be run, or was asked for twice over."""


@dataclass(frozen=True)
class TgSpec:
    """Everything the two passes need to know, in one place.

    Args:
        melt_temperature_k: Where the melt is equilibrated and cooling starts.
        t_floor_k: The bottom of the coarse ladder.
        coarse_step_k: Temperature drop per coarse step.
        coarse_hold_ps: Time held at each coarse temperature.
        window_k: Half-width of the fine window around the coarse transition.
            Wide on purpose: it has to cover the coarse fit's own uncertainty
            and still leave both branches enough points to fit.
        fine_step_k: Temperature drop per fine step.
        fine_hold_ps: Time held at each fine temperature. The first half of
            each hold is discarded, so this is twice the averaging window.
        pressure_bar: The pressure held throughout.
        stage_ps: Most dynamics one stage may hold before the ladder is split
            into another. This is the resume granularity: an interrupted run
            repeats at most this much.
        samples_per_segment: Density readings taken per temperature. The mean
            is over the second half, so this is twice the number of readings
            behind each point on the curve.
        min_points_per_branch: Passed to
            :func:`~openmmpolymer.timeseries.glass_transition`.
        npt_trajectory_ps: Frame interval for the equilibration stage, or None
            for no trajectory. Needed for :func:`melt_equilibration` to say
            anything about the chains, and off by default because it is frames
            of the whole cell.
        max_total_ns: Refuse to start if the two passes would exceed this.
    """

    melt_temperature_k: float = 650.0
    t_floor_k: float = 150.0
    coarse_step_k: float = 25.0
    coarse_hold_ps: float = 1000.0
    window_k: float = 60.0
    fine_step_k: float = 5.0
    fine_hold_ps: float = 3000.0
    pressure_bar: float = 1.0
    stage_ps: float = 10_000.0
    samples_per_segment: int = 30
    min_points_per_branch: int = 4
    npt_trajectory_ps: float | None = None
    max_total_ns: float | None = None

    def __post_init__(self) -> None:
        """Reject a spec that cannot describe a quench, at the call site."""
        for name in (
            "melt_temperature_k",
            "t_floor_k",
            "coarse_step_k",
            "coarse_hold_ps",
            "window_k",
            "fine_step_k",
            "fine_hold_ps",
            "pressure_bar",
            "stage_ps",
        ):
            require_positive(getattr(self, name), None, name=name)
        require_integer(self.samples_per_segment, name="samples_per_segment")
        require_integer(self.min_points_per_branch, name="min_points_per_branch")
        if self.npt_trajectory_ps is not None:
            require_positive(self.npt_trajectory_ps, None, name="npt_trajectory_ps")
        if self.max_total_ns is not None:
            require_positive(self.max_total_ns, None, name="max_total_ns")
        if self.t_floor_k >= self.melt_temperature_k:
            raise ValueError(
                f"t_floor_k={self.t_floor_k} is not below "
                f"melt_temperature_k={self.melt_temperature_k}: a quench cools."
            )


#: The default settings, as a shared frozen singleton so it can be a default
#: argument without being rebuilt on every call.
DEFAULT_SPEC = TgSpec()


@dataclass(frozen=True)
class TgSchedule:
    """One pass' ladder, and what it costs.

    Built from :func:`~openmmpolymer.simulate.quench_temperatures`, so the
    number quoted before a run starts is the number of temperatures that run.

    Args:
        temperatures_k: What it visits, descending.
        hold_ps: Time held at each.
        step_k: The nominal drop per step.
    """

    temperatures_k: tuple[float, ...]
    hold_ps: float
    step_k: float

    @property
    def n_temperatures(self) -> int:
        """How many temperatures are held."""
        return len(self.temperatures_k)

    @property
    def total_ps(self) -> float:
        """How much dynamics the whole ladder is."""
        return self.hold_ps * self.n_temperatures

    @property
    def cooling_rate_k_per_ns(self) -> float:
        """The rate the ladder amounts to, in kelvin per nanosecond."""
        return self.step_k / self.hold_ps * 1000.0


@dataclass(frozen=True)
class MeltEquilibration:
    """Whether a melt had settled before it was cooled.

    Two conditions, and both have to be shown rather than assumed. The cell
    volume has to have stopped drifting faster than its own noise, and the
    chains' centres of mass have to have travelled further than the chains are
    big, diffusively. The first says the density is a density; only the second
    says anything about the chains, and a melt whose density settled in two
    hundred picoseconds can still be exactly as packmol left it.

    Args:
        stage: Which stage was checked.
        volume: What the cell volume did, or None if it could not be read.
        displacement: What the chains did, or None for the same reason.
        radius_of_gyration_nm: The chain size the displacement is measured
            against.
        displacement_target_nm2: :data:`MSD_RG_MULTIPLE` times its square.
        displacement_nm2: How far the chains actually went, at the longest lag.
        displacement_lag_ps: The lag that was read at. The longest lag has the
            fewest time origins behind it, so it is worth knowing.
        volume_settled: Whether the volume stopped drifting.
        chains_moved: Whether the chains went further than
            *displacement_target_nm2*, diffusively.
        equilibrated: Both of the above. False whenever either could not be
            measured - a verdict with half its evidence missing is not a pass,
            and :attr:`unchecked` is where the difference between "this melt is
            not equilibrated" and "the data that would say was never written"
            is recorded.
        unchecked: One sentence per thing that could not be measured, each
            naming what to do about it.
    """

    stage: str
    volume: Equilibration | None
    displacement: MeanSquaredDisplacement | None
    radius_of_gyration_nm: float | None
    displacement_target_nm2: float | None
    displacement_nm2: float | None
    displacement_lag_ps: float | None
    volume_settled: bool
    chains_moved: bool
    equilibrated: bool
    unchecked: tuple[str, ...]


@dataclass(frozen=True)
class TgResult:
    """What a two-pass scan did and found.

    Args:
        run_dir: Where it wrote.
        manifest_path: The manifest.
        temperature_k: The answer - the fine pass' transition when it
            resolved, else the coarse one when that did, else None.
        approximate: The coarse fit, reported whether or not the window came
            from it. None when the coarse curve was too short to fit at all,
            which only gets this far if a window was named outright.
        transition: The fine fit.
        coarse_curve: The coarse specific-volume curve.
        fine_curve: The fine one.
        coarse_schedule: The coarse ladder.
        fine_schedule: The fine ladder.
        restart: ``"waypoint"`` when the fine pass continued the coarse
            cooling, ``"precool"`` when it had to start again from the melt.
        start_state: The state the fine pass started from.
        coarse_summary: What the first pass ran.
        fine_summary: What the second pass ran.
        resolved: Whether :attr:`temperature_k` came from a fit that resolved.
    """

    run_dir: str
    manifest_path: str
    temperature_k: float | None
    approximate: GlassTransition | None
    transition: GlassTransition | None
    coarse_curve: QuenchCurve
    fine_curve: QuenchCurve | None
    coarse_schedule: TgSchedule
    fine_schedule: TgSchedule
    restart: str
    start_state: str
    coarse_summary: RunSummary
    fine_summary: RunSummary
    resolved: bool


@dataclass(frozen=True)
class TgReport:
    """Everything a finished run directory has to say about its transition.

    Args:
        run_dir: The directory read.
        stages: The quench stages found, in the order they were read.
        curves: One curve per quench stage.
        transitions: One fit per curve.
        coarse: The fit from the widest-stepped curve, when there is more than
            one curve.
        fine: The fit from the finest-stepped curve, slowest-cooled first.
        log_linear: Tg against log rate, when there were several rates.
        vft: The same data under the Vogel-Fulcher-Tammann relation, when
            there were at least three.
        melt: Whether the melt had settled before cooling, if it was checked.
        temperature_k: The headline - the transition at the rate it was
            actually cooled at, or None when nothing resolved. Deliberately
            not the rate-extrapolated number.
        cooling_rate_k_per_ns: The rate that headline was measured at.
        resolved: Whether there is a headline.
        notes: What could not be done, in plain sentences.
    """

    run_dir: str
    stages: tuple[str, ...]
    curves: tuple[QuenchCurve, ...]
    transitions: tuple[GlassTransition, ...]
    coarse: GlassTransition | None
    fine: GlassTransition | None
    log_linear: CoolingRateExtrapolation | None
    vft: CoolingRateExtrapolation | None
    melt: MeltEquilibration | None
    temperature_k: float | None
    cooling_rate_k_per_ns: float | None
    resolved: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ReportFiles:
    """Where :func:`write_report` put things.

    Args:
        json: The machine-readable record.
        figures: Every figure written, in the order they were made.
    """

    json: str
    figures: tuple[str, ...]


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


def coarse_schedule(spec: TgSpec) -> TgSchedule:
    """The ladder the coarse pass walks."""
    return TgSchedule(
        temperatures_k=tuple(
            quench_temperatures(
                spec.melt_temperature_k, spec.t_floor_k, spec.coarse_step_k
            )
        ),
        hold_ps=spec.coarse_hold_ps,
        step_k=spec.coarse_step_k,
    )


def fine_schedule(
    t_start_k: float, t_end_k: float, spec: TgSpec, *, hold_ps: float | None = None
) -> TgSchedule:
    """The ladder the fine pass walks, between two temperatures."""
    if t_end_k >= t_start_k:
        raise TgError(
            f"The fine window runs from {t_start_k:.0f} K to {t_end_k:.0f} K, "
            "which does not cool. Widen window_k, or lower t_floor_k so the "
            "window has somewhere to go."
        )
    return TgSchedule(
        temperatures_k=tuple(quench_temperatures(t_start_k, t_end_k, spec.fine_step_k)),
        hold_ps=spec.fine_hold_ps if hold_ps is None else hold_ps,
        step_k=spec.fine_step_k,
    )


def _chunks(schedule: TgSchedule, stage_ps: float) -> list[tuple[float, ...]]:
    """Split a ladder into pieces no longer than *stage_ps* of dynamics.

    The split is bookkeeping, not physics: each piece starts from the state the
    one before it left, so the cooling history is continuous. What it buys is
    resume granularity, since a stage is the unit a run picks itself back up
    at, and a hundred nanoseconds in one stage is a hundred nanoseconds to
    repeat.
    """
    per_chunk = max(1, int(stage_ps // schedule.hold_ps))
    ladder = schedule.temperatures_k
    chunks = [
        ladder[start : start + per_chunk] for start in range(0, len(ladder), per_chunk)
    ]
    # A trailing chunk of one temperature is a stage with no temperature step,
    # which nothing downstream can tell apart from a pass with a different
    # step. Fold it into the one before it rather than leave it stranded.
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()
    return chunks


def _quench_stages(
    stem: str,
    schedule: TgSchedule,
    spec: TgSpec,
    *,
    waypoints: bool = False,
    timestep_fs: float | None = None,
) -> tuple[Stage, ...]:
    """Turn a ladder into the stages that walk it."""
    stages: list[Stage] = []
    for index, chunk in enumerate(_chunks(schedule, spec.stage_ps)):
        options: dict[str, Any] = {
            "temperatures_k": list(chunk),
            "hold_ps": schedule.hold_ps,
            "pressure_bar": spec.pressure_bar,
            "samples_per_segment": spec.samples_per_segment,
        }
        if waypoints:
            options["waypoints"] = True
        if timestep_fs is not None:
            options["timestep_fs"] = timestep_fs
        stages.append(Stage(f"{stem}_{index:02d}", "quench", options))
    return tuple(stages)


def tg_coarse_scan(spec: TgSpec = DEFAULT_SPEC, **equilibration: Any) -> Protocol:
    """Equilibrate the melt, then screen for the transition in coarse steps.

    The melt is settled *at* the temperature cooling starts from rather than
    at some lower target, so the hottest point on the curve - the one that
    anchors the melt branch - is measured on a cell that is already there
    rather than one still catching up. The anneal dips
    :data:`ANNEAL_DEPTH_K` below it so that it still has somewhere to cycle.

    Args:
        spec: What to run.
        **equilibration: Passed to
            :func:`~openmmpolymer.protocols.standard_melt_equilibration`.

    Returns:
        The protocol.
    """
    base = standard_melt_equilibration(
        target_temperature_k=spec.melt_temperature_k,
        melt_temperature_k=spec.melt_temperature_k,
        anneal_t_low_k=max(spec.melt_temperature_k - ANNEAL_DEPTH_K, spec.t_floor_k),
        pressure_bar=spec.pressure_bar,
        npt_trajectory=(
            "none"
            if spec.npt_trajectory_ps is None
            else TrajectoryOptions("xtc", interval_ps=spec.npt_trajectory_ps)
        ),
        **equilibration,
    )
    return Protocol(
        name=PROTOCOL_NAME,
        stages=(
            *base.stages,
            *_quench_stages(COARSE_STEM, coarse_schedule(spec), spec, waypoints=True),
        ),
    )


def tg_fine_scan(
    schedule: TgSchedule,
    spec: TgSpec,
    *,
    timestep_fs: float,
    stem: str = FINE_STEM,
) -> Protocol:
    """The second pass on its own: the fine ladder and nothing else.

    One timestep is pinned across every chunk rather than let each derate to
    its own hottest temperature, because this pass is a measurement and three
    chunks integrated three different ways is a confound nobody would choose.

    Args:
        schedule: The ladder to walk.
        spec: The rest of the settings.
        timestep_fs: The timestep every chunk uses.
        stem: Stage-name stem, which a multi-rate series varies.

    Returns:
        The protocol.
    """
    return Protocol(
        name=PROTOCOL_NAME,
        stages=_quench_stages(stem, schedule, spec, timestep_fs=timestep_fs),
    )


# --------------------------------------------------------------------------
# The window, the waypoint, and the cost
# --------------------------------------------------------------------------


def fine_window(transition_k: float, spec: TgSpec) -> tuple[float, float]:
    """The temperatures the fine pass should cover, clamped to the ladder.

    Args:
        transition_k: Where the coarse pass put the transition.
        spec: The settings, for the window width and the two limits.

    Returns:
        ``(top, bottom)``.

    Raises:
        TgError: Clamping left nothing between them.
    """
    top = min(transition_k + spec.window_k, spec.melt_temperature_k)
    bottom = max(transition_k - spec.window_k, spec.t_floor_k)
    if top < transition_k + spec.window_k:
        log.warning(
            "The fine window's top was clamped from %.0f K to the melt "
            "temperature, %.0f K.",
            transition_k + spec.window_k,
            top,
        )
    if bottom > transition_k - spec.window_k:
        log.warning(
            "The fine window's bottom was clamped from %.0f K to the floor, %.0f K.",
            transition_k - spec.window_k,
            bottom,
        )
    if top <= bottom:
        raise TgError(
            f"A window of +/-{spec.window_k:.0f} K around {transition_k:.0f} K "
            f"clamps to {top:.0f}-{bottom:.0f} K, which is empty. The coarse "
            "transition sits outside the range that was scanned; widen it with "
            "melt_temperature_k and t_floor_k."
        )
    return top, bottom


def _waypoints(manifest: RunManifest, stages: Sequence[str]) -> list[tuple[float, str]]:
    """Every waypoint the named stages recorded and still has on disk.

    Read from the manifest rather than found by globbing the directory. The
    manifest is written when a stage completes, so it can only describe a run
    that finished; files on disk can be left over from a crashed attempt at a
    longer ladder, and nothing about their names says so.
    """
    found: list[tuple[float, str]] = []
    for name in stages:
        recorded = manifest.stages.get(name) or {}
        temperatures = (recorded.get("samples") or {}).get("segment_temperature_k")
        for temperature, path in zip(
            temperatures or (), recorded.get("waypoints") or (), strict=False
        ):
            if path and Path(str(path)).is_file():
                found.append((float(temperature), str(path)))
    return found


def pick_waypoint(
    candidates: Sequence[tuple[float, str]], window_top_k: float
) -> tuple[float, str] | None:
    """The coarse state the fine pass should carry on from.

    The *lowest* temperature still at or above the window's top: that restarts
    as late on the coarse trajectory as it can while still entering the window
    from above, so the fine pass inherits a cell that has been cooling all
    along rather than one dropped in from the melt.

    Args:
        candidates: ``(temperature, path)`` pairs.
        window_top_k: The top of the fine window.

    Returns:
        The chosen pair, or None when there are no candidates at all.
    """
    if not candidates:
        return None
    above = [pair for pair in candidates if pair[0] >= window_top_k - 1.0e-9]
    if above:
        return min(above, key=lambda pair: pair[0])
    hottest = max(candidates, key=lambda pair: pair[0])
    log.warning(
        "No waypoint sits at or above the window's top of %.0f K; carrying on "
        "from the hottest there is, %.0f K.",
        window_top_k,
        hottest[0],
    )
    return hottest


def _remaining_ps(protocol: Protocol, manifest: RunManifest | None) -> float:
    """How much of a protocol is not already recorded as done."""
    from .protocols import _stage_duration_ps

    done = set() if manifest is None else set(manifest.stages)
    return sum(
        _stage_duration_ps(stage) for stage in protocol.stages if stage.name not in done
    )


def _report_cost(
    coarse: Protocol,
    coarse_ladder: TgSchedule,
    fine_ladders: Sequence[TgSchedule],
    spec: TgSpec,
    manifest: RunManifest | None,
) -> None:
    """Say what the whole thing costs before any of it is spent.

    The fine pass' point count is known before the coarse pass runs, because
    it is the window divided by the step wherever the window lands. Two
    numbers are reported: the total, and how much of it is still outstanding
    given what the manifest already records - in a queue the second is the
    only one anyone can act on.
    """
    fine_ps = sum(ladder.total_ps for ladder in fine_ladders)
    equilibration_ps = coarse.total_duration_ps - coarse_ladder.total_ps
    remaining_ps = _remaining_ps(coarse, manifest) + fine_ps
    total_ps = coarse.total_duration_ps + fine_ps
    log.info(
        "Tg scan: %.1f ns equilibration, %.1f ns coarse (%d points at %.1f "
        "K/ns), %.1f ns fine over %d pass(es) at %s K/ns, %d points each - "
        "%.1f ns in total, %.1f ns of it still to run.",
        equilibration_ps / 1000.0,
        coarse_ladder.total_ps / 1000.0,
        coarse_ladder.n_temperatures,
        coarse_ladder.cooling_rate_k_per_ns,
        fine_ps / 1000.0,
        len(fine_ladders),
        ", ".join(f"{ladder.cooling_rate_k_per_ns:.2f}" for ladder in fine_ladders),
        fine_ladders[0].n_temperatures if fine_ladders else 0,
        total_ps / 1000.0,
        remaining_ps / 1000.0,
    )
    if spec.max_total_ns is not None and total_ps / 1000.0 > spec.max_total_ns:
        raise TgError(
            f"This scan is {total_ps / 1000.0:.1f} ns against a max_total_ns "
            f"of {spec.max_total_ns:.1f}. Shorten the holds, widen the steps, "
            "or raise the limit - but decide before it starts, not after."
        )


# --------------------------------------------------------------------------
# The workflow record
# --------------------------------------------------------------------------


def _request(spec: TgSpec, tg_approx_k: float | None) -> dict[str, Any]:
    """What the caller asked for, as the thing a resume is compared against."""
    return {"spec": asdict(spec), "tg_approx_k": tg_approx_k}


def _check_request(run_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Refuse a resume that quietly asks for something else.

    A stage's options are recorded nowhere, so a protocol rerun with different
    settings resumes and keeps the old result without a word. That is survivable
    for an equilibration and not for a measurement, where the number would then
    belong to a schedule nobody ran.
    """
    path = run_dir / WORKFLOW_NAME
    if not path.is_file():
        return {}
    record: dict[str, Any] = json.loads(path.read_text())
    previous = record.get("request")
    if previous is not None and previous != request:
        changed = [
            key
            for key in set(previous.get("spec", {})) | set(request["spec"])
            if previous.get("spec", {}).get(key) != request["spec"].get(key)
        ]
        if previous.get("tg_approx_k") != request["tg_approx_k"]:
            changed.append("tg_approx_k")
        raise TgError(
            f"{path} records a scan run with different settings "
            f"({', '.join(sorted(changed)) or 'unknown'}), and resuming would "
            "keep results measured under the old ones. Run into a fresh "
            "directory, or put the settings back."
        )
    return record


def _save_workflow(run_dir: Path, record: dict[str, Any]) -> str:
    """Write the workflow record. Recomputable, so not written atomically."""
    path = run_dir / WORKFLOW_NAME
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return str(path)


# --------------------------------------------------------------------------
# The melt check
# --------------------------------------------------------------------------


def _recorded_float(record: dict[str, Any] | None, key: str) -> float | None:
    """Read one float out of the manifest, if it is there and usable."""
    if not record:
        return None
    value = record.get(key)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _melt_verdict(
    stage: str,
    volume: Equilibration | None,
    displacement: MeanSquaredDisplacement | None,
    radius_of_gyration_nm: float | None,
    unchecked: tuple[str, ...],
) -> MeltEquilibration:
    """Combine the two halves of the check into one verdict."""
    settled = volume is not None and volume.equilibrated
    target = (
        None
        if radius_of_gyration_nm is None
        else MSD_RG_MULTIPLE * radius_of_gyration_nm**2
    )
    travelled: float | None = None
    lag: float | None = None
    if displacement is not None and displacement.msd_nm2.size:
        travelled = float(displacement.msd_nm2[-1])
        lag = float(displacement.lag_ps[-1])
    moved = (
        displacement is not None
        and displacement.diffusive
        and travelled is not None
        and target is not None
        and travelled > target
    )
    return MeltEquilibration(
        stage=stage,
        volume=volume,
        displacement=displacement,
        radius_of_gyration_nm=radius_of_gyration_nm,
        displacement_target_nm2=target,
        displacement_nm2=travelled,
        displacement_lag_ps=lag,
        volume_settled=settled,
        chains_moved=moved,
        equilibrated=settled and moved,
        unchecked=unchecked,
    )


def melt_equilibration(
    run_dir: str | Path,
    stage: str = "05_npt",
    *,
    radius_of_gyration_nm: float | None = None,
    max_lag_fraction: float = 0.5,
    stride: int = 1,
) -> MeltEquilibration:
    """Check whether a melt had settled before it was cooled.

    Reading a trajectory leaves MDAnalysis' offset and lock files beside it, so
    this writes into the run directory even though it is an analysis, and it
    cannot be pointed at a read-only archive.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        stage: The equilibration stage to check.
        radius_of_gyration_nm: Override the chain size recorded in the
            manifest.
        max_lag_fraction: Passed to
            :func:`~openmmpolymer.conformation.centre_of_mass_msd`.
        stride: Frames to skip when reading the trajectory.

    Returns:
        The verdict, and what could not be checked.

    Raises:
        AnalysisError: There is no manifest to read.
    """
    directory = Path(run_dir)
    manifest = RunManifest.load(directory)
    if manifest is None:
        raise AnalysisError(f"No manifest in {directory}.")

    unchecked: list[str] = []
    volume = _volume_settling(stage, manifest, unchecked)
    displacement = _chain_displacement(
        directory, stage, max_lag_fraction, stride, unchecked
    )
    radius = radius_of_gyration_nm
    if radius is None:
        radius = _recorded_float(manifest.chains, "mean_radius_of_gyration_nm")
        if radius is None:
            unchecked.append(
                "chain displacement: the manifest records no radius of "
                "gyration, so there is nothing to measure the displacement "
                "against. Give run_protocol a chain_backbone, or pass "
                "radius_of_gyration_nm."
            )
    verdict = _melt_verdict(stage, volume, displacement, radius, tuple(unchecked))
    for reason in verdict.unchecked:
        log.info("%s", reason)
    return verdict


def _volume_settling(
    stage: str,
    manifest: RunManifest,
    unchecked: list[str],
) -> Equilibration | None:
    """What the cell volume did over the stage, or None with a reason why not."""
    recorded = manifest.stages.get(stage)
    if recorded is None:
        unchecked.append(
            f"box volume: the manifest has no stage {stage!r}. It records: "
            f"{', '.join(manifest.stages) or 'nothing'}."
        )
        return None
    csv = recorded.get("csv")
    if not csv or not Path(str(csv)).is_file():
        unchecked.append(
            f"box volume: stage {stage!r} left no state-data CSV to read, so "
            "there is no volume series."
        )
        return None
    try:
        series = read_state_data(csv, stage=stage)
        return equilibration(series.time_ps, series.volume_nm3)
    except AnalysisError as error:
        unchecked.append(f"box volume: {error}")
        return None


def _chain_displacement(
    directory: Path,
    stage: str,
    max_lag_fraction: float,
    stride: int,
    unchecked: list[str],
) -> MeanSquaredDisplacement | None:
    """How far the chains went, or None with a reason why it is not known."""
    try:
        ensemble = open_run(directory, stage)
    except AnalysisError as error:
        unchecked.append(f"chain displacement: {error}")
        return None
    if ensemble.is_snapshot:
        unchecked.append(
            f"chain displacement: stage {stage!r} wrote no trajectory, so it "
            "could not be measured. Give the equilibration stage a trajectory "
            "- standard_melt_equilibration takes npt_trajectory, and TgSpec "
            "takes npt_trajectory_ps."
        )
        return None
    try:
        return centre_of_mass_msd(
            ensemble, max_lag_fraction=max_lag_fraction, stride=stride
        )
    except AnalysisError as error:
        unchecked.append(f"chain displacement: {error}")
        return None


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def nominal_fine_schedule(spec: TgSpec, *, hold_ps: float | None = None) -> TgSchedule:
    """How big the fine ladder will be, before it is known where it lands.

    The window is the same width wherever it ends up, so the point count - and
    therefore the cost - is knowable before the coarse pass has run a step.
    The temperatures here are offsets from the window's top rather than real
    ones, because only how many there are is being asked.
    """
    return TgSchedule(
        temperatures_k=tuple(
            quench_temperatures(2.0 * spec.window_k, 0.0, spec.fine_step_k)
        ),
        hold_ps=spec.fine_hold_ps if hold_ps is None else hold_ps,
        step_k=spec.fine_step_k,
    )


def _rate_label(rate_k_per_ns: float) -> str:
    """A cooling rate as a filename-safe label.

    No dots: a stage's name becomes a file stem, and ``Path.with_suffix``
    would read the first dot as the start of an extension and write
    ``08_fine_quench_0.state.xml`` for every rate below one.
    """
    return f"{rate_k_per_ns:g}".replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class _Approach:
    """Everything the coarse pass settled about how the fine pass should run."""

    manifest: RunManifest
    coarse_stages: tuple[str, ...]
    coarse_curve: QuenchCurve
    coarse_schedule: TgSchedule
    approximate: GlassTransition | None
    transition_k: float
    window: tuple[float, float]
    restart: str
    start_state: str
    start_temperature_k: float
    precool: Stage | None
    timestep_fs: float
    summary: RunSummary
    record: dict[str, Any]


def _equilibration_state(protocol: Protocol, manifest: RunManifest) -> str | None:
    """The state the last non-quench stage left, for a pre-cool to start from."""
    for stage in reversed(protocol.stages):
        if stage.kind != "quench":
            recorded = manifest.stages.get(stage.name) or {}
            state = recorded.get("final_state")
            return None if state is None else str(state)
    return None


def _approach(
    run: RunContext,
    run_dir: str | Path,
    spec: TgSpec,
    tg_approx_k: float | None,
    *,
    rates_k_per_ns: Sequence[float] | None,
    resume: bool,
    chain_backbone: Sequence[int] | None,
    atoms_per_chain: int | None,
    expected_characteristic_ratio: float,
    equilibration: dict[str, Any],
) -> _Approach:
    """Run the coarse pass, fit it, and decide how the fine one begins."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    coarse = tg_coarse_scan(spec, **equilibration)
    ladder = coarse_schedule(spec)
    holds = (
        [spec.fine_hold_ps]
        if rates_k_per_ns is None
        else [spec.fine_step_k / float(rate) * 1000.0 for rate in rates_k_per_ns]
    )
    _report_cost(
        coarse,
        ladder,
        [nominal_fine_schedule(spec, hold_ps=hold) for hold in holds],
        spec,
        RunManifest.load(directory) if resume else None,
    )

    request = _request(spec, tg_approx_k)
    record = _check_request(directory, request) if resume else {}
    if resume:
        validate_run_inputs(run, directory)
    record["request"] = request
    _save_workflow(directory, record)

    summary = run_protocol(
        coarse,
        run,
        directory,
        resume=resume,
        chain_backbone=chain_backbone,
        atoms_per_chain=atoms_per_chain,
        expected_characteristic_ratio=expected_characteristic_ratio,
    )
    manifest = RunManifest.load(directory)
    if manifest is None:  # pragma: no cover - run_protocol always writes one
        raise TgError(f"The coarse pass left no manifest in {directory}.")

    stages = tuple(stage.name for stage in coarse.stages if stage.kind == "quench")
    curve = quench_curve(directory, stages)
    approximate, reason = _fit_coarse(curve, spec)
    transition_k = _chosen_transition(approximate, reason, tg_approx_k, spec)
    window = fine_window(transition_k, spec)

    restart, start_state, start_temperature_k, precool = _restart_from(
        coarse, manifest, spec, window, record
    )
    timestep_fs = safe_timestep_fs(start_temperature_k, run.spec)
    record.update(
        {
            "tg_approx_k": (None if approximate is None else approximate.temperature_k),
            "tg_used_k": transition_k,
            "window_top_k": window[0],
            "window_bottom_k": window[1],
            "restart": restart,
            "start_state": start_state,
            "start_temperature_k": start_temperature_k,
            "timestep_fs": timestep_fs,
            "coarse_stages": list(stages),
        }
    )
    _save_workflow(directory, record)

    return _Approach(
        manifest=manifest,
        coarse_stages=stages,
        coarse_curve=curve,
        coarse_schedule=ladder,
        approximate=approximate,
        transition_k=transition_k,
        window=window,
        restart=restart,
        start_state=start_state,
        start_temperature_k=start_temperature_k,
        precool=precool,
        timestep_fs=timestep_fs,
        summary=summary,
        record=record,
    )


def _fit_coarse(curve: QuenchCurve, spec: TgSpec) -> tuple[GlassTransition | None, str]:
    """Fit the coarse curve, or say why it could not be fitted.

    A curve too short for two branches is not an error here. A caller who
    named the window outright does not need the fit, and refusing at this
    point would stop a scan that has everything it needs.
    """
    try:
        transition = glass_transition(
            curve, min_points_per_branch=spec.min_points_per_branch
        )
    except AnalysisError as error:
        return None, str(error)
    return transition, ""


def _chosen_transition(
    approximate: GlassTransition | None,
    reason: str,
    tg_approx_k: float | None,
    spec: TgSpec,
) -> float:
    """Where to centre the fine window, and refuse to guess if nothing says.

    A fine pass is tens of nanoseconds. Spending them on a window derived from
    a fit that already reported it found a corner in noise rather than a
    transition is worse than stopping, because it produces a curve with
    nothing in it and no way to tell that apart from a polymer that has no
    transition in range.
    """
    if tg_approx_k is not None:
        if (
            approximate is not None
            and approximate.resolved
            and abs(tg_approx_k - approximate.temperature_k) > spec.window_k / 2.0
        ):
            log.warning(
                "tg_approx_k=%.0f K is more than half a window from the coarse "
                "fit's %.0f K. The window follows what was asked for, but one "
                "of the two is wrong.",
                tg_approx_k,
                approximate.temperature_k,
            )
        return float(tg_approx_k)
    if approximate is None:
        raise TgError(
            f"The coarse pass cannot be fitted: {reason} Until it can there is "
            "nothing to centre a fine window on. Extend the coarse ladder, "
            "quench in smaller steps, lower min_points_per_branch, or pass "
            "tg_approx_k to name the window yourself."
        )
    if not approximate.resolved:
        raise TgError(
            f"The coarse fit put a break at {approximate.temperature_k:.0f} K "
            f"with slopes {approximate.melt_expansion_per_k:.3g} (melt) and "
            f"{approximate.glass_expansion_per_k:.3g} (glass) cm^3/g/K, and "
            "did not resolve: either the glassy branch is not flat enough "
            "relative to the melt, or the two lines cross outside the data. "
            "Fitting two lines to a straight one always finds a corner, so "
            "this is not a transition to centre a window on. Extend the "
            "coarse ladder past the transition, quench in smaller steps, "
            "lower min_points_per_branch, or pass tg_approx_k to name the "
            "window yourself."
        )
    return float(approximate.temperature_k)


def _restart_from(
    coarse: Protocol,
    manifest: RunManifest,
    spec: TgSpec,
    window: tuple[float, float],
    record: dict[str, Any],
) -> tuple[str, str, float, Stage | None]:
    """Where the fine pass starts, and any pre-cool needed to get it there.

    Decided once. A fine pass already under way keeps the state it started
    from, because a waypoint deleted since then would otherwise insert a
    pre-cool into a ladder that is already running past it. "Under way" means
    the same thing it means to the runner: a stage recorded whose final state
    is still on disk, because that is the one the runner will skip.
    """
    started = any(
        Path(str(recorded.get("final_state", ""))).is_file()
        for name, recorded in manifest.stages.items()
        if name.startswith(FINE_STEM)
    )
    if started and record.get("start_state"):
        return (
            str(record.get("restart", "waypoint")),
            str(record["start_state"]),
            float(record["start_temperature_k"]),
            None,
        )

    top = window[0]
    chosen = pick_waypoint(
        _waypoints(manifest, [s.name for s in coarse.stages if s.kind == "quench"]),
        top,
    )
    if chosen is not None:
        return "waypoint", chosen[1], chosen[0], None

    state = _equilibration_state(coarse, manifest)
    if state is None:
        raise TgError(
            "The coarse pass recorded no waypoints and left no equilibrated "
            "state to fall back on, so the fine pass has nowhere to start. "
            "Rerun the coarse pass - it saves a waypoint at every temperature "
            "when tg_coarse_scan builds it."
        )
    log.warning(
        "No coarse waypoint survives on disk, so the fine pass is pre-cooled "
        "from the equilibrated melt instead of carrying on from the coarse "
        "run. The fine window then has a different thermal history from the "
        "scan that located it, which is a real difference and not bookkeeping."
    )
    if top >= spec.melt_temperature_k - 1.0e-9:
        return "melt", state, spec.melt_temperature_k, None
    precool = Stage(
        PRECOOL_STEM,
        "quench",
        {
            "t_start": spec.melt_temperature_k,
            "t_end": top,
            "step_k": spec.coarse_step_k,
            "hold_ps": spec.coarse_hold_ps,
            "pressure_bar": spec.pressure_bar,
            "samples_per_segment": spec.samples_per_segment,
        },
    )
    return "precool", state, top, precool


def _fine_protocol(
    schedule: TgSchedule,
    spec: TgSpec,
    approach: _Approach,
    *,
    stem: str = FINE_STEM,
) -> Protocol:
    """The fine pass, with the pre-cool in front of it when there is one."""
    fine = tg_fine_scan(schedule, spec, timestep_fs=approach.timestep_fs, stem=stem)
    if approach.precool is None:
        return fine
    return Protocol(name=PROTOCOL_NAME, stages=(approach.precool, *fine.stages))


def _run_fine(
    run: RunContext,
    run_dir: Path,
    approach: _Approach,
    spec: TgSpec,
    *,
    hold_ps: float | None = None,
    stem: str = FINE_STEM,
    resume: bool = True,
) -> tuple[RunSummary, QuenchCurve, GlassTransition, TgSchedule]:
    """Walk one fine ladder and fit what it recorded."""
    schedule = fine_schedule(
        approach.start_temperature_k, approach.window[1], spec, hold_ps=hold_ps
    )
    protocol = _fine_protocol(schedule, spec, approach, stem=stem)
    log.info(
        "Fine pass %s: %d temperatures from %.0f K to %.0f K at %.2f K/ns "
        "(%.1f ns), starting from the %s state at %.0f K.",
        stem,
        schedule.n_temperatures,
        schedule.temperatures_k[0],
        schedule.temperatures_k[-1],
        schedule.cooling_rate_k_per_ns,
        schedule.total_ps / 1000.0,
        approach.restart,
        approach.start_temperature_k,
    )
    summary = run_protocol(
        protocol, run, run_dir, resume=resume, state_in=approach.start_state
    )
    names = tuple(stage.name for stage in protocol.stages if stage.name != PRECOOL_STEM)
    curve = quench_curve(run_dir, names)
    return (
        summary,
        curve,
        glass_transition(curve, min_points_per_branch=spec.min_points_per_branch),
        schedule,
    )


def run_tg_scan(
    run: RunContext,
    run_dir: str | Path = "run",
    *,
    spec: TgSpec = DEFAULT_SPEC,
    tg_approx_k: float | None = None,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> TgResult:
    """Equilibrate, screen coarsely for the transition, then resolve it.

    Idempotent: called again on a directory it already wrote to, it re-reads
    the manifest, re-fits the coarse pass - deterministic arithmetic over
    recorded numbers, and a matter of milliseconds - derives the same window,
    and runs only the stages that are not already recorded. That is what makes
    an interrupted run resumable across the boundary between the two passes,
    without a separate piece of workflow state to keep in step.

    Args:
        run: The run context.
        run_dir: Where everything is written.
        spec: What to run.
        tg_approx_k: Centre the fine window here instead of on the coarse fit.
            The coarse fit is still run and still reported.
        resume: Skip stages already recorded as complete.
        chain_backbone: Passed to
            :func:`~openmmpolymer.protocols.run_protocol`.
        atoms_per_chain: Likewise.
        expected_characteristic_ratio: Likewise.
        **equilibration: Passed to
            :func:`~openmmpolymer.protocols.standard_melt_equilibration`.

    Returns:
        What both passes did and what they found.

    Raises:
        TgError: The coarse fit did not resolve and no window was named, the
            scan exceeds ``spec.max_total_ns``, or the directory records a
            scan run with different settings.
    """
    directory = Path(run_dir)
    approach = _approach(
        run,
        directory,
        spec,
        tg_approx_k,
        rates_k_per_ns=None,
        resume=resume,
        chain_backbone=chain_backbone,
        atoms_per_chain=atoms_per_chain,
        expected_characteristic_ratio=expected_characteristic_ratio,
        equilibration=equilibration,
    )
    # The coarse pass already reset a forced rerun's manifest. The fine pass
    # must preserve it, including the waypoint it starts from.
    summary, curve, transition, schedule = _run_fine(
        run, directory, approach, spec, resume=True
    )
    coarse = approach.approximate
    temperature = (
        transition.temperature_k
        if transition.resolved
        else coarse.temperature_k
        if coarse is not None and coarse.resolved
        else None
    )
    log.info(
        "Tg scan: %s at %.2f K/ns (coarse said %s at %.1f K/ns).",
        f"{temperature:.0f} K" if temperature is not None else "no clear transition",
        schedule.cooling_rate_k_per_ns,
        "nothing" if coarse is None else f"{coarse.temperature_k:.0f} K",
        approach.coarse_schedule.cooling_rate_k_per_ns,
    )
    return TgResult(
        run_dir=str(directory),
        manifest_path=str(directory / "manifest.json"),
        temperature_k=temperature,
        approximate=approach.approximate,
        transition=transition,
        coarse_curve=approach.coarse_curve,
        fine_curve=curve,
        coarse_schedule=approach.coarse_schedule,
        fine_schedule=schedule,
        restart=approach.restart,
        start_state=approach.start_state,
        coarse_summary=approach.summary,
        fine_summary=summary,
        resolved=temperature is not None,
    )


def cooling_rate_series(
    run: RunContext,
    run_dir: str | Path = "run",
    *,
    rates_k_per_ns: Sequence[float] = (10.0, 5.0, 2.0),
    spec: TgSpec = DEFAULT_SPEC,
    tg_approx_k: float | None = None,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
    expected_characteristic_ratio: float = 7.0,
    **equilibration: Any,
) -> tuple[GlassTransition, ...]:
    """Walk the fine window several times, each at a different cooling rate.

    Every pass starts from the same state - one equilibrated melt, cooled once
    to the top of the window - and goes down from there at its own rate. So
    they are independent cooling histories from a common configuration, which
    is what makes the transitions comparable, and what
    :func:`~openmmpolymer.timeseries.cooling_rate_extrapolation` then needs.

    Args:
        run: The run context.
        run_dir: Where everything is written.
        rates_k_per_ns: The rates to measure at. Each sets its own hold,
            ``fine_step_k / rate``.
        spec: The rest of the settings; ``fine_hold_ps`` is overridden per rate.
        tg_approx_k: Centre the window here instead of on the coarse fit.
        resume: Skip stages already recorded as complete.
        chain_backbone: Passed to
            :func:`~openmmpolymer.protocols.run_protocol`.
        atoms_per_chain: Likewise.
        expected_characteristic_ratio: Likewise.
        **equilibration: Passed to
            :func:`~openmmpolymer.protocols.standard_melt_equilibration`.

    Returns:
        One fit per rate, in the order the rates were given.

    Raises:
        TgError: As :func:`run_tg_scan`, or the rates are not distinct.
    """
    if len(set(rates_k_per_ns)) != len(rates_k_per_ns):
        raise TgError(
            f"The rates {tuple(rates_k_per_ns)} are not distinct, so two "
            "passes would be the same measurement under two names."
        )
    directory = Path(run_dir)
    approach = _approach(
        run,
        directory,
        spec,
        tg_approx_k,
        rates_k_per_ns=rates_k_per_ns,
        resume=resume,
        chain_backbone=chain_backbone,
        atoms_per_chain=atoms_per_chain,
        expected_characteristic_ratio=expected_characteristic_ratio,
        equilibration=equilibration,
    )
    transitions: list[GlassTransition] = []
    for rate in rates_k_per_ns:
        hold_ps = require_positive(
            spec.fine_step_k / float(rate) * 1000.0, None, name="hold_ps"
        )
        _, _, transition, _ = _run_fine(
            run,
            directory,
            approach,
            spec,
            hold_ps=hold_ps,
            stem=f"{FINE_STEM}_{_rate_label(float(rate))}",
            resume=True,
        )
        transitions.append(transition)
    return tuple(transitions)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def analyse_run(
    run_dir: str | Path,
    *,
    extra_run_dirs: Sequence[str | Path] = (),
    melt_stage: str | None = "05_npt",
    min_points_per_branch: int = 4,
    target_rate_k_per_ns: float = DSC_COOLING_RATE_K_PER_NS,
    radius_of_gyration_nm: float | None = None,
) -> TgReport:
    """Read everything a finished run has to say about its glass transition.

    Reads and returns; writes nothing. That is the same discipline
    :mod:`openmmpolymer.plots` keeps one layer down, for the same reason - it
    makes the whole analysis testable without a filesystem, and it keeps every
    write in :func:`write_report`.

    Quench stages are found by what they recorded rather than by what they
    were called, and the coarse pass is told from the fine one by its
    temperature step. The rate fit uses only the finest-stepped family, so a
    25 K screening scan is not weighed against 5 K measurements as though they
    were the same quality of number.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        extra_run_dirs: More directories to pool quenches from, for a rate
            series run separately.
        melt_stage: The equilibration stage to check, or None to skip it.
        min_points_per_branch: Passed to
            :func:`~openmmpolymer.timeseries.glass_transition`.
        target_rate_k_per_ns: The rate to extrapolate to.
        radius_of_gyration_nm: Override the chain size in the manifest.

    Returns:
        The report.

    Raises:
        AnalysisError: There is no manifest, or nothing in it was a quench.
    """
    directory = Path(run_dir)
    notes: list[str] = []
    stages: list[str] = []
    curves: list[QuenchCurve] = []
    for index, candidate in enumerate([directory, *map(Path, extra_run_dirs)]):
        for group in _group_passes(candidate, quench_stages(candidate)):
            joined = ", ".join(group)
            stages.append(joined if index == 0 else f"{candidate}:{joined}")
            curves.append(quench_curve(candidate, group))

    # Paired as they are fitted, and a curve too short to fit drops out of
    # both lists together. Zipping them back up afterwards would pair the
    # wrong curve with the wrong fit the moment one in the middle failed.
    pairs: list[tuple[QuenchCurve, GlassTransition]] = []
    for curve in curves:
        try:
            pairs.append(
                (
                    curve,
                    glass_transition(
                        curve, min_points_per_branch=min_points_per_branch
                    ),
                )
            )
        except AnalysisError as error:
            notes.append(f"{curve.stage}: {error}")
    if not pairs:
        raise AnalysisError(
            f"No quench in {directory} gave a curve long enough to fit. "
            f"{' '.join(notes)}"
        )
    fitted = {id(curve) for curve, _ in pairs}
    stages = [
        name for name, curve in zip(stages, curves, strict=True) if id(curve) in fitted
    ]
    curves = [curve for curve, _ in pairs]
    transitions = [transition for _, transition in pairs]

    # Widest step first, and within one step size the fastest first, so the
    # last entry is the finest and slowest scan there is - the best
    # measurement in the directory - and the first is the screening pass.
    paired = sorted(
        pairs,
        key=lambda pair: (
            -pair[0].temperature_step_k,
            -(pair[0].cooling_rate_k_per_ns or 0.0),
        ),
    )
    fine = paired[-1][1]
    coarse = paired[0][1] if paired[0][0] is not paired[-1][0] else None

    log_linear, vft = _rate_fits(paired, target_rate_k_per_ns, notes)
    melt = _melt_report(directory, melt_stage, radius_of_gyration_nm, notes)

    headline = (
        fine
        if fine.resolved
        else coarse
        if coarse is not None and coarse.resolved
        else None
    )
    if headline is None:
        notes.append(
            "No quench resolved a transition, so there is no temperature to "
            "report. The two-line fit found a corner in noise, which is what "
            "it always finds."
        )
    return TgReport(
        run_dir=str(directory),
        stages=tuple(stages),
        curves=tuple(curves),
        transitions=tuple(transitions),
        coarse=coarse,
        fine=fine,
        log_linear=log_linear,
        vft=vft,
        melt=melt,
        temperature_k=None if headline is None else headline.temperature_k,
        cooling_rate_k_per_ns=(
            None if headline is None else headline.cooling_rate_k_per_ns
        ),
        resolved=headline is not None,
        notes=tuple(notes),
    )


def _group_passes(run_dir: str | Path, names: Sequence[str]) -> list[tuple[str, ...]]:
    """Group the stages that walked one ladder between them.

    A long ladder is split into stages so that an interrupted run resumes at
    the stage it stopped in rather than at the top of the ramp, and that split
    is bookkeeping - the pieces are one cooling history and belong on one
    curve. Two stages are taken to be pieces of the same pass when they
    stepped the same way and held for the same time, which is a property of
    what they recorded rather than of what they were named.
    """
    grouped: dict[tuple[float, float], list[str]] = {}
    order: list[tuple[float, float]] = []
    for name in names:
        curve = quench_curve(run_dir, name)
        key = (
            round(curve.temperature_step_k, 6),
            round(-1.0 if curve.hold_ps is None else curve.hold_ps, 6),
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(name)
    return [tuple(grouped[key]) for key in order]


def _rate_fits(
    paired: Sequence[tuple[QuenchCurve, GlassTransition]],
    target_rate_k_per_ns: float,
    notes: list[str],
) -> tuple[CoolingRateExtrapolation | None, CoolingRateExtrapolation | None]:
    """Fit the transition against cooling rate, over the finest family only."""
    finest = min(curve.temperature_step_k for curve, _ in paired)
    family = [
        transition
        for curve, transition in paired
        if math.isclose(curve.temperature_step_k, finest, rel_tol=1.0e-9)
    ]
    if len(family) < 2:
        return None, None
    fits: list[CoolingRateExtrapolation | None] = []
    for form in ("log_linear", "vft"):
        try:
            fits.append(
                cooling_rate_extrapolation(
                    family, target_rate_k_per_ns=target_rate_k_per_ns, form=form
                )
            )
        except AnalysisError as error:
            fits.append(None)
            notes.append(f"{form} rate fit: {error}")
    return fits[0], fits[1]


def _melt_report(
    directory: Path,
    melt_stage: str | None,
    radius_of_gyration_nm: float | None,
    notes: list[str],
) -> MeltEquilibration | None:
    """Check the melt, turning a failure to check into a note."""
    if melt_stage is None:
        return None
    try:
        return melt_equilibration(
            directory, melt_stage, radius_of_gyration_nm=radius_of_gyration_nm
        )
    except AnalysisError as error:
        notes.append(f"melt check: {error}")
        return None


def _transition_record(transition: GlassTransition) -> dict[str, Any]:
    """One fit as plain JSON types.

    Written out field by field rather than with ``asdict``, which drops the
    expansivities because they are properties, and which would render a
    curve's numpy arrays as strings. Spelling the record out also pins what is
    on disk independently of how the dataclasses happen to be laid out.
    """
    return {
        "temperature_k": transition.temperature_k,
        "specific_volume_cm3_g": transition.specific_volume_cm3_g,
        "melt_expansion_per_k": transition.melt_expansion_per_k,
        "glass_expansion_per_k": transition.glass_expansion_per_k,
        "melt_expansivity_per_k": transition.melt_expansivity_per_k,
        "glass_expansivity_per_k": transition.glass_expansivity_per_k,
        "expansivity_ordered": (
            transition.melt_expansivity_per_k > transition.glass_expansivity_per_k
        ),
        "residual_cm3_g": transition.residual_cm3_g,
        "n_points_melt": transition.n_points_melt,
        "n_points_glass": transition.n_points_glass,
        "cooling_rate_k_per_ns": transition.cooling_rate_k_per_ns,
        "resolved": transition.resolved,
    }


def _extrapolation_record(fit: CoolingRateExtrapolation) -> dict[str, Any]:
    """One rate extrapolation as plain JSON types."""
    return {
        "form": fit.form,
        "temperature_k": fit.temperature_k,
        "target_rate_k_per_ns": fit.target_rate_k_per_ns,
        "cooling_rate_k_per_ns": [float(x) for x in fit.cooling_rate_k_per_ns],
        "transition_k": [float(x) for x in fit.transition_k],
        "sensitivity_k_per_decade": fit.sensitivity_k_per_decade,
        "parameters": fit.parameters,
        "residual_k": fit.residual_k,
        "n_rates": fit.n_rates,
        "n_parameters": fit.n_parameters,
        "extrapolation_decades": fit.extrapolation_decades,
        "resolved": fit.resolved,
    }


def _melt_record(melt: MeltEquilibration) -> dict[str, Any]:
    """The melt verdict as plain JSON types."""
    return {
        "stage": melt.stage,
        "volume_settled": melt.volume_settled,
        "chains_moved": melt.chains_moved,
        "equilibrated": melt.equilibrated,
        "radius_of_gyration_nm": melt.radius_of_gyration_nm,
        "displacement_nm2": melt.displacement_nm2,
        "displacement_target_nm2": melt.displacement_target_nm2,
        "displacement_lag_ps": melt.displacement_lag_ps,
        "unchecked": list(melt.unchecked),
    }


def write_report(
    report: TgReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write a report out, as JSON and as figures.

    The only thing in the analysis half of this package that writes anything.
    Unlike the manifest, which is written atomically because losing it costs a
    three-day run, this record is a second of arithmetic away from being
    rebuilt, so it is written plainly.

    Args:
        report: What :func:`analyse_run` found.
        output_dir: Where to write, defaulting to ``<run_dir>/analysis``. Give
            one when the run directory should not be touched.
        figures: Write figures as well as the record.
        figure_format: What matplotlib should save them as.

    Returns:
        Where everything went.
    """
    from importlib.metadata import PackageNotFoundError, version

    from .plots import plot_cooling_rate, plot_quench_curve, plot_state_data

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
        "stages": list(report.stages),
        "temperature_k": report.temperature_k,
        "cooling_rate_k_per_ns": report.cooling_rate_k_per_ns,
        "resolved": report.resolved,
        "transitions": [_transition_record(fit) for fit in report.transitions],
        "coarse": None if report.coarse is None else _transition_record(report.coarse),
        "fine": None if report.fine is None else _transition_record(report.fine),
        "log_linear": (
            None
            if report.log_linear is None
            else _extrapolation_record(report.log_linear)
        ),
        "vft": None if report.vft is None else _extrapolation_record(report.vft),
        "melt": None if report.melt is None else _melt_record(report.melt),
        "notes": list(report.notes),
    }
    json_path = directory / "tg.json"
    json_path.write_text(json.dumps(record, indent=2, default=str) + "\n")

    written: list[str] = []
    if figures:
        for curve, transition in zip(report.curves, report.transitions, strict=False):
            stem = curve.stage.replace(", ", "_").replace(" ", "_")
            written.append(
                _save(
                    plot_quench_curve(curve, transition=transition),
                    directory / f"quench_{stem}.{figure_format}",
                )
            )
        for fit in (report.log_linear, report.vft):
            if fit is not None:
                written.append(
                    _save(
                        plot_cooling_rate(fit),
                        directory / f"cooling_rate_{fit.form}.{figure_format}",
                    )
                )
        written.extend(
            _melt_figure(report, manifest, directory, figure_format, plot_state_data)
        )
    return ReportFiles(json=str(json_path), figures=tuple(written))


def _melt_figure(
    report: TgReport,
    manifest: RunManifest | None,
    directory: Path,
    figure_format: str,
    plot_state_data: Any,
) -> list[str]:
    """The equilibration figure, when its series is still there to plot."""
    if report.melt is None or report.melt.volume is None or manifest is None:
        return []
    csv = (manifest.stages.get(report.melt.stage) or {}).get("csv")
    if not csv:
        return []
    series = read_state_data(csv, stage=report.melt.stage)
    return [
        _save(
            plot_state_data(series, settled=report.melt.volume),
            directory / f"equilibration.{figure_format}",
        )
    ]


def _save(figure: Any, path: Path) -> str:
    """Save a figure and close it, returning where it went."""
    figure.savefig(path, bbox_inches="tight")
    return str(path)
