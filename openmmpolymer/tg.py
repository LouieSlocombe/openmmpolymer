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

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._files import ReportFiles, write_json
from ._validation import require_integer, require_positive
from ._workflow import (
    check_request,
    optional,
    remaining_ps,
    require_positive_fields,
    resume_chunks,
    spec_request,
    write_report_files,
)
from .melt_check import MeltEquilibration, melt_equilibration
from .plots import plot_cooling_rate, plot_quench_curve, plot_state_data
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
    GlassTransition,
    QuenchCurve,
    cooling_rate_extrapolation,
    glass_transition,
    quench_curve,
    quench_stages,
    read_state_data,
)
from .trajectory import AnalysisError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)

PROTOCOL_NAME = "tg_two_pass"
COARSE_STEM = "06_coarse_quench"
PRECOOL_STEM = "07_precool"
FINE_STEM = "08_fine_quench"
WORKFLOW_NAME = "tg_workflow.json"

#: How far below the melt temperature each annealing cycle dips. The anneal
#: needs somewhere to cycle to once the run settles at the temperature it will
#: start cooling from. Clamped to the ladder's floor, because a melt that is
#: cold to begin with would otherwise be asked to cycle through zero kelvin,
#: where a thermostat has nothing to hold and a barostat divides by it.
ANNEAL_DEPTH_K = 150.0


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
            into another; an interrupted run repeats at most this much.
        samples_per_segment: Density readings taken per temperature. The mean
            is over the second half, so this is twice the number of readings
            behind each point on the curve.
        min_points_per_branch: Passed to
            :func:`~openmmpolymer.timeseries.glass_transition`.
        npt_trajectory_ps: Frame interval for the equilibration stage, or None
            for no trajectory. Needed for
            :func:`~openmmpolymer.melt_check.melt_equilibration` to say
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
        require_positive_fields(
            self,
            (
                "melt_temperature_k",
                "t_floor_k",
                "coarse_step_k",
                "coarse_hold_ps",
                "window_k",
                "fine_step_k",
                "fine_hold_ps",
                "pressure_bar",
                "stage_ps",
            ),
            optional=("npt_trajectory_ps", "max_total_ns"),
        )
        require_integer(self.samples_per_segment, name="samples_per_segment")
        require_integer(self.min_points_per_branch, name="min_points_per_branch")
        if self.t_floor_k >= self.melt_temperature_k:
            raise ValueError(
                f"t_floor_k={self.t_floor_k} is not below "
                f"melt_temperature_k={self.melt_temperature_k}: a quench cools."
            )


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
            cooling, ``"precool"`` when it had to start again from the melt,
            and ``"melt"`` when its window starts at the melt temperature.
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
    """A ladder's temperatures split for resume, with none left on its own.

    A trailing chunk of one temperature would be a stage with no temperature
    step, which nothing downstream can tell apart from a pass with a
    different step, so it is folded into the one before it.
    """
    ladder = schedule.temperatures_k
    chunks = [
        ladder[chunk.start : chunk.stop]
        for chunk in resume_chunks(len(ladder), schedule.hold_ps, stage_ps)
    ]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        last = chunks.pop()
        chunks[-1] += last
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


def _report_cost(
    coarse: Protocol,
    coarse_ladder: TgSchedule,
    fine_ladders: Sequence[TgSchedule],
    spec: TgSpec,
    manifest: RunManifest | None,
) -> None:
    """Say what the whole thing costs before any of it is spent.

    Two numbers: the total, and how much of it the manifest does not already
    record - in a queue the second is the only one anyone can act on.

    Raises:
        TgError: The total is over ``max_total_ns``.
    """
    fine_ps = sum(ladder.total_ps for ladder in fine_ladders)
    total_ps = coarse.total_duration_ps + fine_ps
    log.info(
        "Tg scan: %.1f ns equilibration, %.1f ns coarse (%d points at %.1f "
        "K/ns), %.1f ns fine over %d pass(es) at %s K/ns, %d points each - "
        "%.1f ns in total, %.1f ns of it still to run.",
        (coarse.total_duration_ps - coarse_ladder.total_ps) / 1000.0,
        coarse_ladder.total_ps / 1000.0,
        coarse_ladder.n_temperatures,
        coarse_ladder.cooling_rate_k_per_ns,
        fine_ps / 1000.0,
        len(fine_ladders),
        ", ".join(f"{ladder.cooling_rate_k_per_ns:.2f}" for ladder in fine_ladders),
        fine_ladders[0].n_temperatures if fine_ladders else 0,
        total_ps / 1000.0,
        (remaining_ps(coarse.stages, manifest) + fine_ps) / 1000.0,
    )
    if spec.max_total_ns is not None and total_ps / 1000.0 > spec.max_total_ns:
        raise TgError(
            f"This scan is {total_ps / 1000.0:.1f} ns against a max_total_ns "
            f"of {spec.max_total_ns:.1f}. Shorten the holds, widen the steps, "
            "or raise the limit - but decide before it starts, not after."
        )


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def _rate_label(rate_k_per_ns: float) -> str:
    """A cooling rate as a stage-name label, which carries no dots."""
    return f"{rate_k_per_ns:g}".replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class _Approach:
    """Everything the coarse pass settled about how the fine pass should run."""

    coarse_curve: QuenchCurve
    coarse_schedule: TgSchedule
    approximate: GlassTransition | None
    window: tuple[float, float]
    restart: str
    start_state: str
    start_temperature_k: float
    precool: Stage | None
    timestep_fs: float
    summary: RunSummary


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
    directory: Path,
    spec: TgSpec,
    tg_approx_k: float | None,
    *,
    fine_holds: Sequence[float],
    resume: bool,
    chains: dict[str, Any],
    equilibration: dict[str, Any],
) -> _Approach:
    """Run the coarse pass, fit it, and decide how the fine one begins.

    The request is recorded before the coarse pass runs, and what it derived
    once it has, so a resume can check both. Nothing is written before the
    budget and a resumed directory's settings have been checked.
    """
    coarse = tg_coarse_scan(spec, **equilibration)
    ladder = coarse_schedule(spec)
    _report_cost(
        coarse,
        ladder,
        [nominal_fine_schedule(spec, hold_ps=hold) for hold in fine_holds],
        spec,
        RunManifest.load(directory) if resume else None,
    )

    path = directory / WORKFLOW_NAME
    request = spec_request(spec, tg_approx_k=tg_approx_k)
    record = check_request(path, request, error=TgError) if resume else {}
    if resume:
        validate_run_inputs(run, directory)
    record["request"] = request
    directory.mkdir(parents=True, exist_ok=True)
    write_json(path, record, strict=False)

    summary = run_protocol(coarse, run, directory, resume=resume, **chains)
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
    write_json(path, record, strict=False)

    return _Approach(
        coarse_curve=curve,
        coarse_schedule=ladder,
        approximate=approximate,
        window=window,
        restart=restart,
        start_state=start_state,
        start_temperature_k=start_temperature_k,
        precool=precool,
        timestep_fs=timestep_fs,
        summary=summary,
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
    """Where to centre the fine window, refusing to guess if nothing says.

    A window derived from a fit that already reported a corner in noise
    rather than a transition produces a curve with nothing in it, and no way
    to tell that apart from a polymer that has no transition in range.
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
            str(record["restart"]),
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
    stem: str,
) -> Protocol:
    """The fine pass, with the pre-cool in front of it when there is one."""
    fine = tg_fine_scan(schedule, spec, timestep_fs=approach.timestep_fs, stem=stem)
    if approach.precool is None:
        return fine
    return Protocol(name=PROTOCOL_NAME, stages=(approach.precool, *fine.stages))


def _run_fine(
    run: RunContext,
    directory: Path,
    approach: _Approach,
    spec: TgSpec,
    *,
    chains: dict[str, Any],
    hold_ps: float | None = None,
    stem: str = FINE_STEM,
) -> tuple[RunSummary, QuenchCurve, GlassTransition, TgSchedule]:
    """Walk one fine ladder and fit what it recorded.

    Always resuming: the coarse pass already reset a forced rerun's manifest,
    and the fine pass has to keep what it recorded, the waypoint it starts
    from included.
    """
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
        protocol,
        run,
        directory,
        resume=True,
        state_in=approach.start_state,
        **chains,
    )
    names = tuple(stage.name for stage in protocol.stages if stage.name != PRECOOL_STEM)
    curve = quench_curve(directory, names)
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
    an interrupted run resumable across the boundary between the two passes.

    Args:
        run: The run context.
        run_dir: Where everything is written.
        spec: What to run.
        tg_approx_k: Centre the fine window here instead of on the coarse fit.
            The coarse fit is still run and still reported.
        resume: Skip stages already recorded as complete.
        chain_backbone: Passed to
            :func:`~openmmpolymer.protocols.run_protocol`, for both passes.
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
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    approach = _approach(
        run,
        directory,
        spec,
        tg_approx_k,
        fine_holds=(spec.fine_hold_ps,),
        resume=resume,
        chains=chains,
        equilibration=equilibration,
    )
    summary, curve, transition, schedule = _run_fine(
        run, directory, approach, spec, chains=chains
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
            ``fine_step_k / rate``, which overrides ``spec.fine_hold_ps``.
        spec: The rest of the settings. It, and every argument after it, is
            as for :func:`run_tg_scan`.
        tg_approx_k: Likewise.
        resume: Likewise.
        chain_backbone: Likewise.
        atoms_per_chain: Likewise.
        expected_characteristic_ratio: Likewise.
        **equilibration: Likewise.

    Returns:
        One fit per rate, in the order the rates were given.

    Raises:
        TgError: As :func:`run_tg_scan`, or the rates are not distinct.
        ValueError: A rate is not a positive number. Every rate is checked
            before anything runs.
    """
    holds = _fine_holds(rates_k_per_ns, spec)
    directory = Path(run_dir)
    chains: dict[str, Any] = {
        "chain_backbone": chain_backbone,
        "atoms_per_chain": atoms_per_chain,
        "expected_characteristic_ratio": expected_characteristic_ratio,
    }
    approach = _approach(
        run,
        directory,
        spec,
        tg_approx_k,
        fine_holds=holds,
        resume=resume,
        chains=chains,
        equilibration=equilibration,
    )
    return tuple(
        _run_fine(
            run,
            directory,
            approach,
            spec,
            chains=chains,
            hold_ps=hold_ps,
            stem=f"{FINE_STEM}_{_rate_label(float(rate))}",
        )[2]
        for rate, hold_ps in zip(rates_k_per_ns, holds, strict=True)
    )


def _fine_holds(rates_k_per_ns: Sequence[float], spec: TgSpec) -> tuple[float, ...]:
    """The hold each cooling rate asks for, every rate checked up front."""
    if len(set(rates_k_per_ns)) != len(rates_k_per_ns):
        raise TgError(
            f"The rates {tuple(rates_k_per_ns)} are not distinct, so two "
            "passes would be the same measurement under two names."
        )
    holds: list[float] = []
    for rate in rates_k_per_ns:
        rate_k_per_ns = require_positive(rate, None, name="rates_k_per_ns")
        holds.append(
            require_positive(
                spec.fine_step_k / rate_k_per_ns * 1000.0, None, name="hold_ps"
            )
        )
    return tuple(holds)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def analyse_tg(
    run_dir: str | Path,
    *,
    extra_run_dirs: Sequence[str | Path] = (),
    melt_stage: str | None = "05_npt",
    min_points_per_branch: int = 4,
    target_rate_k_per_ns: float = DSC_COOLING_RATE_K_PER_NS,
    radius_of_gyration_nm: float | None = None,
) -> TgReport:
    """Read everything a finished run has to say about its glass transition.

    Reads and returns; writes nothing of its own. Quench stages are found by
    what they recorded rather than by what they were called, and the coarse
    pass is told from the fine one by its temperature step. The rate fit uses
    only the finest-stepped family, so a 25 K screening scan is not weighed
    against 5 K measurements as though they were the same quality of number.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        extra_run_dirs: More directories to pool quenches from, for a rate
            series run separately.
        melt_stage: The equilibration stage to check with
            :func:`~openmmpolymer.melt_check.melt_equilibration`, or None to
            skip it.
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
    melt = (
        None
        if melt_stage is None
        else optional(
            lambda: melt_equilibration(
                directory, melt_stage, radius_of_gyration_nm=radius_of_gyration_nm
            ),
            notes,
            "melt check",
        )
    )

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

    The pieces a ladder was split into for resume are one cooling history and
    belong on one curve. Two stages are taken to be pieces of the same pass
    when they stepped the same way and held for the same time, which is a
    property of what they recorded rather than of what they were named.
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


def _transition_record(transition: GlassTransition) -> dict[str, Any]:
    """One fit as plain JSON types, its expansivities included."""
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


def write_tg_report(
    report: TgReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``tg.json`` and its figures into ``<run_dir>/analysis``.

    Or into *output_dir*, when the run directory should not be touched.
    """
    fields: dict[str, Any] = {
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
    return write_report_files(
        report.run_dir,
        output_dir,
        "tg.json",
        fields,
        _figures(report) if figures else (),
        figure_format,
    )


def _figures(report: TgReport) -> Iterator[tuple[str, Figure]]:
    """A figure per quench and per rate fit, and the melt's volume series."""
    for curve, transition in zip(report.curves, report.transitions, strict=False):
        stem = curve.stage.replace(", ", "_").replace(" ", "_")
        yield f"quench_{stem}", plot_quench_curve(curve, transition=transition)
    for fit in (report.log_linear, report.vft):
        if fit is not None:
            yield f"cooling_rate_{fit.form}", plot_cooling_rate(fit)
    melt = report.melt
    if melt is None or melt.volume is None:
        return
    manifest = RunManifest.load(report.run_dir)
    csv = (
        None if manifest is None else (manifest.stages.get(melt.stage) or {}).get("csv")
    )
    if csv:
        yield (
            "equilibration",
            plot_state_data(
                read_state_data(csv, stage=melt.stage), settled=melt.volume
            ),
        )
