"""What a run's own numbers say about whether it settled.

Nothing here opens a trajectory. A stage writes two state-data files and only
one of them is meant to be read back: ``<stem>.csv`` is all numbers, and
``<stem>.log`` renders progress as ``20.0%`` and an unknown remaining time as
``--`` for a person to read. So the CSV is the input, and the field names
``numpy.genfromtxt`` derives from OpenMM's commented header are pinned in one
place here rather than spelled out at each call site.

The rest of the module answers the question the README declines to answer for
you. Equilibration is reported, not claimed: :func:`equilibration` says where a
series stopped drifting faster than its own noise and how many genuinely
independent samples sit after that point, which is evidence about one
observable and not a verdict on the melt. A density that settles in 200 ps says
nothing about chains whose Rouse time is tens of nanoseconds.

The same care applies to the quench. :func:`quench_curve` reads back the
specific-volume curve a quench stage leaves in the manifest, and
:func:`glass_transition` fits the two straight lines a glass transition is read
off. The break between them is a real feature of the data; the temperature it
sits at is not comparable with an experiment, because the cooling rate here is
some ten orders of magnitude faster. That rate is reported alongside the
number, so the caveat travels with it.

:func:`cooling_rate_extrapolation` is what closes that last gap, and it is the
most easily misread thing here. Quench the same melt at several rates and the
transition moves; fit how it moves and the fit can be evaluated at a rate no
simulation could run. But the distance is about ten decades, so the answer is
an extrapolation wearing a measurement's clothes. The result therefore carries
how far it was extrapolated, and ``resolved`` is False past two decades - which
means that an extrapolation to a real calorimeter is *always* unresolved. That
is the design working rather than failing.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from ._fitting import (
    TINY,
    fit_line,
    separable_fit,
    standard_error,
    statistical_inefficiency,
)
from ._validation import require_choice, require_integer, require_positive
from .trajectory import (
    AnalysisError,
    load_manifest,
    stage_names,
    stage_record,
    stages_holding,
)

log = logging.getLogger(__name__)

#: How ``numpy.genfromtxt(..., names=True)`` renders OpenMM's commented CSV
#: header, mapped to the field names used here. OpenMM writes
#: ``#"Step","Time (ps)",...``; genfromtxt strips the punctuation and runs the
#: words together, so ``Density (g/mL)`` becomes ``Density_gmL``.
CSV_FIELDS = {
    "step": "Step",
    "time_ps": "Time_ps",
    "potential_energy_kj_mol": "Potential_Energy_kJmole",
    "kinetic_energy_kj_mol": "Kinetic_Energy_kJmole",
    "total_energy_kj_mol": "Total_Energy_kJmole",
    "temperature_k": "Temperature_K",
    "volume_nm3": "Box_Volume_nm3",
    "density_g_cm3": "Density_gmL",
}

#: Independent samples a retained window needs before its mean is worth
#: quoting. Ten is few, and it is a floor rather than a target.
MIN_INDEPENDENT_SAMPLES = 10.0

#: Fractional drift across the retained window that still counts as settled,
#: unless twice the window's own standard error is larger.
MAX_RELATIVE_DRIFT = 0.01

#: How far into a series the settling point may be looked for. Past halfway
#: there is more discarded than kept, and the run was too short whatever the
#: numbers say.
MAX_START_FRACTION = 0.5

#: Candidate settling points tried. The optimum is broad, so a coarse grid
#: finds it as well as testing every index and is far cheaper.
_START_CANDIDATES = 50

#: How flat the glassy branch has to be relative to the melt before a break
#: counts as a transition. A real glass expands perhaps a third to a half as
#: fast as its melt; two branches of nearly equal slope are a straight line
#: with a kink fitted to it, which is what any two-line fit of a straight line
#: returns.
MAX_GLASS_MELT_SLOPE_RATIO = 0.8

#: A standard calorimeter scan, 10 K/min, in the K/ns this package works in.
#: The rate an experimental glass transition is measured at, and so the rate
#: :func:`cooling_rate_extrapolation` aims at by default.
DSC_COOLING_RATE_K_PER_NS = 10.0 / 60.0 * 1.0e-9

#: How the transition may be taken to depend on cooling rate.
EXTRAPOLATION_FORMS = ("log_linear", "vft")

#: How far past the measured rates an extrapolation may reach and still be
#: called resolved. Two decades is generous; the gap between a quench and a
#: calorimeter is about ten, so this is the threshold that keeps the honest
#: answer honest.
MAX_EXTRAPOLATION_DECADES = 2.0

#: Free parameters in each form, which is what decides whether a residual
#: means anything.
_FORM_PARAMETERS = {"log_linear": 2, "vft": 3}

#: Bracket for the VFT inner search, as natural-log gaps above the fastest
#: measured rate. Wide enough that the optimum sitting on an end means the fit
#: has degenerated rather than that the bracket was too narrow.
_VFT_GAP_BRACKET = (math.log(1.0e-3), math.log(1.0e3))


@dataclass(frozen=True)
class StateData:
    """One stage's state-data CSV, one array per column.

    Args:
        stage: Which stage this came from, if it was named.
        path: Where it was read from.
        step: Step number at each row.
        time_ps: Time within the stage.
        potential_energy_kj_mol: Potential energy.
        kinetic_energy_kj_mol: Kinetic energy.
        total_energy_kj_mol: Total energy.
        temperature_k: Instantaneous temperature.
        volume_nm3: Cell volume.
        density_g_cm3: Mass density. OpenMM writes g/mL, which is the same
            number.
    """

    stage: str
    path: str
    step: npt.NDArray[np.float64]
    time_ps: npt.NDArray[np.float64]
    potential_energy_kj_mol: npt.NDArray[np.float64]
    kinetic_energy_kj_mol: npt.NDArray[np.float64]
    total_energy_kj_mol: npt.NDArray[np.float64]
    temperature_k: npt.NDArray[np.float64]
    volume_nm3: npt.NDArray[np.float64]
    density_g_cm3: npt.NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        """How many rows were read."""
        return int(self.time_ps.size)

    @property
    def duration_ps(self) -> float:
        """Time from the first row to the last."""
        if self.n_rows < 2:
            return 0.0
        return float(self.time_ps[-1] - self.time_ps[0])


@dataclass(frozen=True)
class Equilibration:
    """Where a series settled, and how much evidence sits after that.

    Args:
        start_index: First row of the retained window.
        start_ps: Time that row sits at.
        n_samples: Rows in the retained window.
        n_independent_samples: Rows divided by the statistical inefficiency -
            how many genuinely uncorrelated samples the window is worth,
            whatever unit the series is in.
        correlation_time_ps: The series' own correlation time.
        relative_standard_error: Standard error of the window's mean over
            those independent samples, as a fraction of its scale.
        relative_drift: Change across the window from a straight-line fit, as
            a fraction of its scale.
        equilibrated: Whether the window holds at least
            :data:`MIN_INDEPENDENT_SAMPLES` independent samples and drifts by
            no more than :data:`MAX_RELATIVE_DRIFT` or twice its own standard
            error, whichever is larger. This is one observable. It says that
            *this* quantity stopped drifting faster than its own noise, not
            that the system is equilibrated - a melt's density settles in a
            few hundred picoseconds while its chains need tens of
            nanoseconds, and no amount of converged density says anything
            about them.
    """

    start_index: int
    start_ps: float
    n_samples: int
    n_independent_samples: float
    correlation_time_ps: float
    relative_standard_error: float
    relative_drift: float
    equilibrated: bool

    def window(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the retained part of a series measured alongside this one.

        Args:
            values: A series with the same rows as the one analysed.

        Returns:
            Everything from :attr:`start_index` onwards.
        """
        return np.asarray(values, dtype=np.float64)[self.start_index :]


@dataclass(frozen=True)
class QuenchCurve:
    """The specific-volume curve a quench leaves behind.

    Args:
        stage: Which stage produced it.
        temperature_k: The temperature held at each step, ascending.
        density_g_cm3: The mean density measured at each step.
        specific_volume_cm3_g: One over the density, which is what a glass
            transition is read off.
        hold_ps: How long each temperature was held, or None when the stage's
            CSV was not available to work it out from.
        cooling_rate_k_per_ns: The rate implied by the step and the hold, or
            None for the same reason. Some ten orders of magnitude faster than
            any experiment, which is why it is reported.
    """

    stage: str
    temperature_k: npt.NDArray[np.float64]
    density_g_cm3: npt.NDArray[np.float64]
    specific_volume_cm3_g: npt.NDArray[np.float64]
    hold_ps: float | None
    cooling_rate_k_per_ns: float | None

    @property
    def n_points(self) -> int:
        """How many temperatures were held."""
        return int(self.temperature_k.size)

    @property
    def temperature_step_k(self) -> float:
        """The typical gap between the temperatures held.

        What tells a coarse screening scan from a fine one, without having to
        know what either stage was called.
        """
        return _temperature_step(self.temperature_k)


@dataclass(frozen=True)
class GlassTransition:
    """A two-line fit to a specific-volume curve.

    Args:
        temperature_k: Where the two branches meet.
        specific_volume_cm3_g: Specific volume at the crossing. With the two
            slopes it fixes both fitted lines, so a plot needs nothing else.
        melt_expansion_per_k: Slope of the high-temperature branch, in
            cm^3/g/K. See :attr:`melt_expansivity_per_k` for the same thing as
            a thermal expansion coefficient.
        glass_expansion_per_k: Slope of the low-temperature branch.
        residual_cm3_g: Root-mean-square residual of the joint fit.
        n_points_melt: Points on the high-temperature branch.
        n_points_glass: Points on the low-temperature branch.
        cooling_rate_k_per_ns: Carried through from the curve, because the
            number above is meaningless without it.
        resolved: Whether the glassy branch is flatter than the melt by at
            least :data:`MAX_GLASS_MELT_SLOPE_RATIO`, and the two lines cross
            where the data actually changes slope rather than somewhere off
            past the end of a branch. When False the fit found a corner in
            noise rather than a transition - which is what fitting two lines
            to a straight line always returns. Even when True this is not a
            measured Tg: the shape is informative, the temperature is not
            comparable with a dilatometry experiment cooled a billion times
            slower.
    """

    temperature_k: float
    specific_volume_cm3_g: float
    melt_expansion_per_k: float
    glass_expansion_per_k: float
    residual_cm3_g: float
    n_points_melt: int
    n_points_glass: int
    cooling_rate_k_per_ns: float | None
    resolved: bool

    @property
    def melt_expansivity_per_k(self) -> float:
        """Volumetric thermal expansion coefficient of the melt, in 1/K.

        ``(1/v)(dv/dT)`` at the crossing - the branch slope over the specific
        volume the two lines meet at. The slopes above are in cm^3/g/K and
        depend on the polymer's density; this is the dimensionless-per-kelvin
        quantity an experiment reports.
        """
        return self._expansivity(self.melt_expansion_per_k)

    @property
    def glass_expansivity_per_k(self) -> float:
        """Volumetric thermal expansion coefficient of the glass, in 1/K.

        Always below :attr:`melt_expansivity_per_k` when :attr:`resolved` is
        True, and not as a separate check: both coefficients divide the same
        crossing volume, so their ordering is the ordering of the two slopes,
        which is what :func:`_is_transition` already requires - and requires
        more strictly, at a ratio of
        :data:`MAX_GLASS_MELT_SLOPE_RATIO` rather than merely being smaller. A
        glass that expanded as fast as its melt would not be a glass.
        """
        return self._expansivity(self.glass_expansion_per_k)

    def _expansivity(self, slope_cm3_g_k: float) -> float:
        """A branch slope as a coefficient, or NaN when there is no volume.

        Not zero, which would read as a material that does not expand. The
        only way here is a fit whose crossing was extrapolated somewhere the
        data never went, and that deserves to propagate rather than look like
        a measurement.
        """
        if abs(self.specific_volume_cm3_g) < TINY:
            return math.nan
        return slope_cm3_g_k / self.specific_volume_cm3_g


@dataclass(frozen=True)
class CoolingRateExtrapolation:
    """How the transition moves with cooling rate, and where that points.

    Args:
        form: Which relation was fitted, from :data:`EXTRAPOLATION_FORMS`.
        cooling_rate_k_per_ns: The rates measured, ascending.
        transition_k: The transition found at each.
        target_rate_k_per_ns: The rate the fit was evaluated at.
        temperature_k: The transition the fit predicts there. Always reported,
            because the caller named a rate and asked; whether it means
            anything is :attr:`resolved`.
        sensitivity_k_per_decade: How far the transition moves per decade of
            cooling rate, at the target. The robustly measured part of all
            this, and the number worth quoting when the extrapolated one is
            not.
        parameters: The fitted parameters, named.
        residual_k: Root-mean-square residual in kelvin. Zero by construction
            when there are exactly as many rates as parameters, which says
            nothing about the fit - hence the sample-count condition on
            :attr:`resolved`.
        n_rates: Rates fitted.
        n_parameters: Free parameters in this form.
        extrapolation_decades: How far past the measured rates the target
            sits, in decades. Zero when it sits between them.
        resolved: Whether every input transition resolved, there are more
            rates than parameters, the fit is physical, and
            :attr:`extrapolation_decades` is within
            :data:`MAX_EXTRAPOLATION_DECADES`. A quench runs at some K/ns and a
            calorimeter at 1e-10 K/ns, so an extrapolation to an experimental
            rate is ten decades and never resolves. The number is still worth
            reporting and is still not a measurement.
    """

    form: str
    cooling_rate_k_per_ns: npt.NDArray[np.float64]
    transition_k: npt.NDArray[np.float64]
    target_rate_k_per_ns: float
    temperature_k: float
    sensitivity_k_per_decade: float
    parameters: dict[str, float]
    residual_k: float
    n_rates: int
    n_parameters: int
    extrapolation_decades: float
    resolved: bool

    @overload
    def predict(self, rate_k_per_ns: float) -> float: ...

    @overload
    def predict(
        self, rate_k_per_ns: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def predict(
        self, rate_k_per_ns: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """The transition the fitted relation puts at a cooling rate, or several.

        Evaluating it anywhere resolves nothing: how far a rate sits from the
        measured ones is what :attr:`extrapolation_decades` is for.
        """
        transition = _transition_at(
            self.form, self.parameters, np.asarray(rate_k_per_ns, dtype=np.float64)
        )
        return float(transition) if np.ndim(transition) == 0 else transition

    def wlf_constants(self, reference_k: float) -> tuple[float, float]:
        """The WLF constants this VFT fit is the same relation as.

        WLF and VFT are one relation in two parameterisations: with
        ``C2 = Tr - T0`` and ``C1 = B / (ln(10) C2)`` they predict the same
        shift factors. So this package fits VFT and converts, rather than
        shipping a second fit of the same arithmetic under a second name.

        Args:
            reference_k: The reference temperature, usually the transition.

        Returns:
            ``(C1, C2)``.

        Raises:
            AnalysisError: This is not a VFT fit, or the reference temperature
                is at or below the fitted ``T0``, where the constants diverge.
        """
        if self.form != "vft":
            raise AnalysisError(
                f"WLF constants come from a VFT fit; this one is {self.form!r}. "
                "Refit with form='vft', which needs at least three rates."
            )
        c2 = reference_k - self.parameters["t0_k"]
        if c2 <= 0.0:
            raise AnalysisError(
                f"A reference of {reference_k} K is at or below the fitted "
                f"T0 of {self.parameters['t0_k']:.1f} K, where the WLF "
                "constants diverge. Use a reference above it."
            )
        return self.parameters["b_k"] / (math.log(10.0) * c2), c2


def read_state_data(csv_path: str | Path, *, stage: str = "") -> StateData:
    """Read a stage's numeric state-data CSV.

    Args:
        csv_path: A ``<stem>.csv`` written by a stage.
        stage: Which stage it came from, recorded on the result.

    Returns:
        One array per column.

    Raises:
        AnalysisError: The file is missing, empty, or does not carry the
            columns a state-data CSV has.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise AnalysisError(f"No state-data CSV at {path}.")
    if path.stat().st_size == 0:
        # A stage that stopped before its first report leaves the file the
        # reporter opened and nothing in it, not even the header. numpy reads
        # that as a malformed table rather than an empty one.
        raise AnalysisError(
            f"{path} is empty, so the stage stopped before it wrote any state "
            "data. Lower report_interval_ps, or run the stage for longer."
        )
    try:
        table = np.genfromtxt(path, delimiter=",", names=True)
    except (IndexError, ValueError) as error:
        raise AnalysisError(
            f"{path} could not be read as a state-data CSV: {error}"
        ) from error
    names = table.dtype.names or ()
    missing = [column for column in CSV_FIELDS.values() if column not in names]
    if missing:
        raise AnalysisError(
            f"{path} is missing the column(s) {', '.join(missing)}. A stage's "
            f"numeric CSV carries {', '.join(CSV_FIELDS.values())}; the "
            f"human-readable <stem>.log is not a substitute for it."
        )
    columns = {
        field: np.atleast_1d(np.asarray(table[column], dtype=np.float64))
        for field, column in CSV_FIELDS.items()
    }
    if columns["time_ps"].size == 0:
        raise AnalysisError(
            f"{path} has a header but no rows, so the stage wrote no state "
            "data before it stopped."
        )
    return StateData(stage=stage, path=str(path), **columns)


def equilibration(
    time_ps: npt.NDArray[np.float64], values: npt.NDArray[np.float64]
) -> Equilibration:
    """Find where a series settled, and how much of it is independent.

    Every candidate settling point is scored by how many uncorrelated samples
    remain after it - the series' length divided by its statistical
    inefficiency - and the best-scoring one wins. Discarding more of a
    correlated series can leave more evidence than keeping all of it, which is
    why the maximum is not simply at zero.

    Args:
        time_ps: Time at each row.
        values: The observable, with the same rows.

    Returns:
        Where it settled, and what the retained window is worth.

    Raises:
        AnalysisError: The two arrays differ in length, or there are fewer
            than three rows to work with.
    """
    times = np.atleast_1d(np.asarray(time_ps, dtype=np.float64))
    series = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if times.size != series.size:
        raise AnalysisError(
            f"{times.size} times against {series.size} values: they have to be "
            "the same series."
        )
    if series.size < 3:
        raise AnalysisError(
            f"{series.size} rows is too few to say anything about settling. "
            "Lower report_interval_ps, or run the stage for longer."
        )

    limit = max(1, int(series.size * MAX_START_FRACTION))
    candidates = np.unique(
        np.linspace(0, limit, min(limit + 1, _START_CANDIDATES)).astype(int)
    )
    best_start = 0
    best_effective = -np.inf
    best_inefficiency = 1.0
    for start in candidates:
        window = series[start:]
        if window.size < 3:
            continue
        inefficiency = statistical_inefficiency(window)
        effective = window.size / inefficiency
        if effective > best_effective:
            best_start = int(start)
            best_effective = effective
            best_inefficiency = inefficiency

    window = series[best_start:]
    scale = _scale_of(window)
    drift = _relative_drift(times[best_start:], window, scale)
    error = standard_error(window, best_effective) / scale
    return Equilibration(
        start_index=best_start,
        start_ps=float(times[best_start]),
        n_samples=int(window.size),
        n_independent_samples=float(best_effective),
        correlation_time_ps=(
            max(0.0, (best_inefficiency - 1.0) / 2.0) * _spacing_ps(times)
        ),
        relative_standard_error=error,
        relative_drift=drift,
        equilibrated=(
            best_effective >= MIN_INDEPENDENT_SAMPLES
            and drift <= max(MAX_RELATIVE_DRIFT, 2.0 * error)
        ),
    )


def _recorded_ladder(samples: dict[str, Any]) -> tuple[list[float], list[float]]:
    """A stage's temperatures and densities, or two empty lists."""
    return (
        list(samples.get("segment_temperature_k") or ()),
        list(samples.get("segment_density_g_cm3") or ()),
    )


def _is_quench(samples: dict[str, Any]) -> bool:
    """Whether a stage cooled, from the temperatures it recorded.

    Every stage records a temperature and a density per segment, so holding a
    ladder is not on its own distinctive - a compression holds seven segments
    at one temperature and an anneal goes up as often as down. What makes a
    quench a quench is that it visited more than one temperature and every one
    was colder than the last.
    """
    temperatures, densities = _recorded_ladder(samples)
    if len(temperatures) < 2 or not densities:
        return False
    return all(cooler < hotter for hotter, cooler in itertools.pairwise(temperatures))


def quench_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name every stage in a run that cooled the cell down a ladder.

    A run that quenched twice - a coarse scan to locate the transition and a
    fine one to resolve it - gives both, and which is the coarse one is then
    a question for the data: :attr:`QuenchCurve.temperature_step_k` tells
    them apart.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.

    Returns:
        The stage names, in the order the manifest records them.

    Raises:
        AnalysisError: There is no manifest, or nothing in it held a ladder.
    """
    return stages_holding(
        run_dir,
        _is_quench,
        "that it stepped down a ladder of temperatures, so nothing there was a quench",
    )


def quench_curve(
    run_dir: str | Path, stage: str | Sequence[str] = "06_quench"
) -> QuenchCurve:
    """Read back the specific-volume curve a quench recorded.

    The densities are already in the manifest: a quench holds each temperature
    for a while and records the mean of the second half of each hold. Several
    stages may be named, and their points are pooled into one curve - a long
    ladder is split into stages so that an interrupted run resumes at the
    stage it stopped in, and that split is bookkeeping rather than physics.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        stage: Which stage cooled the cell, or several to pool.

    Returns:
        The curve, ordered by ascending temperature.

    Raises:
        AnalysisError: There is no manifest, a named stage is not in it, or
            one recorded no per-temperature densities.
    """
    directory = Path(run_dir)
    manifest = load_manifest(directory)
    names = stage_names(stage)

    temperatures: list[float] = []
    densities: list[float] = []
    holds: list[float] = []
    csv_paths: list[Any] = []
    for name in names:
        recorded = stage_record(manifest, name, directory)
        samples = recorded.get("samples") or {}
        step_temperatures, step_densities = _recorded_ladder(samples)
        if not step_temperatures or not step_densities:
            raise AnalysisError(
                f"Stage {name!r} recorded no per-temperature densities, so it "
                "was not a quench. A quench holds a ladder of temperatures and "
                "records the density at each."
            )
        temperatures.extend(step_temperatures)
        densities.extend(step_densities)
        holds.extend(samples.get("segment_duration_ps") or ())
        csv_paths.append(recorded.get("csv"))

    temperature = np.asarray(temperatures, dtype=np.float64)
    density = np.asarray(densities, dtype=np.float64)
    order = np.argsort(temperature)
    temperature, density = temperature[order], density[order]
    if np.any(density <= 0.0):
        raise AnalysisError(
            f"Stage {', '.join(names)} recorded a density of zero or less, so "
            "its specific volume is not defined."
        )

    hold_ps = _hold_of(holds, csv_paths, names, len(temperatures))
    return QuenchCurve(
        stage=", ".join(names),
        temperature_k=temperature,
        density_g_cm3=density,
        specific_volume_cm3_g=1.0 / density,
        hold_ps=hold_ps,
        cooling_rate_k_per_ns=_cooling_rate(temperature, hold_ps),
    )


def glass_transition(
    curve: QuenchCurve, *, min_points_per_branch: int = 4
) -> GlassTransition:
    """Fit two straight lines to a specific-volume curve.

    Every break that leaves enough points on both sides is tried, and the one
    with the smallest total residual wins. The transition temperature is where
    the two fitted lines cross.

    Args:
        curve: A curve from :func:`quench_curve`.
        min_points_per_branch: Points each branch must keep. Below about four
            a branch is fitting noise.

    Returns:
        The fit, and whether it found a transition or a corner in noise.

    Raises:
        AnalysisError: The curve is too short to give both branches
            *min_points_per_branch* points.
    """
    per_branch = require_integer(min_points_per_branch, name="min_points_per_branch")
    temperature = curve.temperature_k
    volume = curve.specific_volume_cm3_g
    if temperature.size < 2 * per_branch:
        raise AnalysisError(
            f"{temperature.size} temperatures cannot give two branches of "
            f"{per_branch} points. Quench in smaller steps, or lower "
            "min_points_per_branch."
        )

    best: tuple[float, int, tuple[float, float], tuple[float, float]] | None = None
    for break_index in range(per_branch, temperature.size - per_branch + 1):
        glass, glass_residual = fit_line(
            temperature[:break_index], volume[:break_index]
        )
        melt, melt_residual = fit_line(temperature[break_index:], volume[break_index:])
        total = glass_residual + melt_residual
        if best is None or total < best[0]:
            best = (total, break_index, glass, melt)

    assert best is not None  # the loop runs at least once
    total, break_index, glass, melt = best
    glass_slope, glass_intercept = glass
    melt_slope, melt_intercept = melt

    separation = melt_slope - glass_slope
    if abs(separation) < TINY:
        crossing = float(temperature[break_index - 1])
        resolved = False
    else:
        crossing = float((glass_intercept - melt_intercept) / separation)
        resolved = _is_transition(temperature, break_index, crossing, glass, melt)

    return GlassTransition(
        temperature_k=crossing,
        specific_volume_cm3_g=glass_intercept + glass_slope * crossing,
        melt_expansion_per_k=melt_slope,
        glass_expansion_per_k=glass_slope,
        residual_cm3_g=float(np.sqrt(total / temperature.size)),
        n_points_melt=int(temperature.size - break_index),
        n_points_glass=int(break_index),
        cooling_rate_k_per_ns=curve.cooling_rate_k_per_ns,
        resolved=resolved,
    )


def cooling_rate_extrapolation(
    transitions: Sequence[GlassTransition],
    *,
    target_rate_k_per_ns: float = DSC_COOLING_RATE_K_PER_NS,
    form: str = "log_linear",
) -> CoolingRateExtrapolation:
    """Fit how the transition moves with cooling rate, and evaluate the fit.

    Quench the same equilibrated melt at several rates and the transition
    shifts: slower cooling gives the cell longer to keep finding a denser
    packing, so it stays liquid to a lower temperature. Fitting that shift is
    the only route from a quench to a number an experiment could recognise,
    and it is a long way - a calorimeter scans about ten decades slower than
    the slowest quench anyone runs.

    Two relations are offered, and the difference between them over that gap
    is not small. On a melt with T0 = 300 K, B = 400 K and R0 = 1e4 K/ns,
    measured at 2, 5 and 10 K/ns:

    =============  ==================
    form           Tg at 10 K/min
    =============  ==================
    ``log_linear``  189 K
    ``vft``         313 K
    =============  ==================

    A quench overestimates an experimental transition by 20 to 50 K, which
    puts the honest answer near 313 K. So ``log_linear`` is the default
    because it is always determined and because its slope - how far the
    transition moves per decade - is a genuinely measured quantity worth
    quoting on its own; and ``vft`` is the form to reach for when the target
    is an experimental rate and there are three or more measurements to fit.

    Args:
        transitions: Fits from :func:`glass_transition`, each carrying the
            rate it was measured at. Two or more, and three or more for
            ``vft``.
        target_rate_k_per_ns: The rate to evaluate the fit at. Defaults to
            :data:`DSC_COOLING_RATE_K_PER_NS`.
        form: One of :data:`EXTRAPOLATION_FORMS`.

    Returns:
        The fit, what it predicts, and how far past the data that prediction
        sits.

    Raises:
        ValueError: *form* is not one of :data:`EXTRAPOLATION_FORMS`, or the
            target rate is not positive.
        AnalysisError: There are too few transitions for the form, one has no
            recorded cooling rate, or two were measured at the same rate.
    """
    require_choice(form, EXTRAPOLATION_FORMS, name="form")
    target = require_positive(target_rate_k_per_ns, None, name="target_rate_k_per_ns")
    n_parameters = _FORM_PARAMETERS[form]

    if len(transitions) < 2:
        raise AnalysisError(
            f"{len(transitions)} transition(s) cannot show a rate dependence. "
            "Quench the same equilibrated melt at two or more cooling rates."
        )
    rates: list[float] = []
    for index, fit in enumerate(transitions):
        measured = fit.cooling_rate_k_per_ns
        if measured is None:
            raise AnalysisError(
                f"The transition at index {index} (break at "
                f"{fit.temperature_k:.0f} K) has no cooling rate, so there is "
                "nothing to plot it against. Its stage recorded neither its "
                "segment durations nor a readable CSV."
            )
        if measured <= 0.0:
            raise AnalysisError(
                f"The transition at index {index} records a cooling rate of "
                f"{measured}, which is not a rate anything was cooled at."
            )
        rates.append(float(measured))

    rate = np.asarray(rates, dtype=np.float64)
    if np.unique(rate).size != rate.size:
        raise AnalysisError(
            "Two transitions were measured at the same cooling rate, so the "
            "fit has no rate dependence to see. Vary the hold or the "
            "temperature step between the quenches."
        )
    if rate.size < n_parameters:
        raise AnalysisError(
            f"A {form!r} fit has {n_parameters} parameters and there are "
            f"{rate.size} rates, so it is not determined. Measure at least "
            f"{n_parameters} rates, or use form='log_linear'."
        )

    transition = np.asarray(
        [fit.temperature_k for fit in transitions], dtype=np.float64
    )
    order = np.argsort(rate)
    rate, transition = rate[order], transition[order]

    if form == "log_linear":
        parameters, total = _fit_log_linear(rate, transition)
        sensitivity = parameters["b_k_per_decade"]
        physical = sensitivity > 0.0
    else:
        parameters, total, degenerate = _fit_vft(rate, transition)
        gap = parameters["ln_r0"] - math.log(target)
        sensitivity = math.log(10.0) * parameters["b_k"] / gap**2
        physical = parameters["b_k"] > 0.0 and not degenerate

    decades = _extrapolation_decades(rate, target)
    return CoolingRateExtrapolation(
        form=form,
        cooling_rate_k_per_ns=rate,
        transition_k=transition,
        target_rate_k_per_ns=target,
        temperature_k=float(_transition_at(form, parameters, np.asarray(target))),
        sensitivity_k_per_decade=float(sensitivity),
        parameters=parameters,
        residual_k=float(np.sqrt(total / rate.size)),
        n_rates=int(rate.size),
        n_parameters=n_parameters,
        extrapolation_decades=decades,
        resolved=(
            all(fit.resolved for fit in transitions)
            and rate.size > n_parameters
            and physical
            and decades <= MAX_EXTRAPOLATION_DECADES
        ),
    )


def _fit_log_linear(
    rate_k_per_ns: npt.NDArray[np.float64], transition_k: npt.NDArray[np.float64]
) -> tuple[dict[str, float], float]:
    """Fit ``Tg = a + b log10(R)``, a straight line in log rate."""
    (slope, intercept), total = fit_line(np.log10(rate_k_per_ns), transition_k)
    return {"a_k": intercept, "b_k_per_decade": slope}, total


def _fit_vft(
    rate_k_per_ns: npt.NDArray[np.float64], transition_k: npt.NDArray[np.float64]
) -> tuple[dict[str, float], float, bool]:
    """Fit ``Tg = T0 + B / (ln R0 - ln R)``, which is linear in T0 and B.

    ``ln R0`` is searched as a log gap above the fastest measured rate, which
    keeps ``ln(R0/R) > 0`` for every point without a constraint.

    Returns the parameters, the sum of squared residuals, and whether the
    optimum ran to an end of the bracket - which means the fit has degenerated
    toward a straight line rather than found a curve.
    """
    log_rate = np.log(rate_k_per_ns)
    fastest = float(log_rate.max())
    gap, b_k, t0_k, total, edge = separable_fit(
        lambda gap: 1.0 / (fastest + math.exp(gap) - log_rate),
        transition_k,
        _VFT_GAP_BRACKET,
    )
    if edge:
        log.info(
            "The VFT search ran to the edge of its bracket, so these data are "
            "a straight line in log rate and R0 is unbounded. Report the "
            "log-linear fit instead."
        )
    parameters = {"t0_k": t0_k, "b_k": b_k, "ln_r0": fastest + math.exp(gap)}
    return parameters, total, bool(edge)


def _transition_at(
    form: str, parameters: dict[str, float], rate_k_per_ns: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The transition a fitted relation puts at each cooling rate."""
    if form == "log_linear":
        return parameters["a_k"] + parameters["b_k_per_decade"] * np.log10(
            rate_k_per_ns
        )
    return parameters["t0_k"] + parameters["b_k"] / (
        parameters["ln_r0"] - np.log(rate_k_per_ns)
    )


def _extrapolation_decades(
    rate_k_per_ns: npt.NDArray[np.float64], target_k_per_ns: float
) -> float:
    """How far past the measured rates the target sits, in decades."""
    slowest, fastest = float(rate_k_per_ns.min()), float(rate_k_per_ns.max())
    return max(
        0.0,
        math.log10(slowest / target_k_per_ns),
        math.log10(target_k_per_ns / fastest),
    )


def _is_transition(
    temperature_k: npt.NDArray[np.float64],
    break_index: int,
    crossing_k: float,
    glass: tuple[float, float],
    melt: tuple[float, float],
) -> bool:
    """Whether a two-line fit found a transition or a corner in noise.

    Two things have to hold. The glassy branch must be meaningfully flatter
    than the melt, because fitting two lines to a straight line succeeds
    perfectly and reports two nearly equal slopes. And the lines must cross
    where the data changes slope - inside the gap the break sits in, allowing
    one temperature step either side for noise - because a crossing
    extrapolated far outside that gap is the two branches disagreeing about a
    region neither of them fitted.
    """
    glass_slope, melt_slope = glass[0], melt[0]
    if melt_slope <= 0.0 or glass_slope > MAX_GLASS_MELT_SLOPE_RATIO * melt_slope:
        return False
    spacing = _temperature_step(temperature_k)
    lower = float(temperature_k[break_index - 1]) - spacing
    upper = float(temperature_k[min(break_index, temperature_k.size - 1)]) + spacing
    return lower <= crossing_k <= upper


def _scale_of(values: npt.NDArray[np.float64]) -> float:
    """A scale to make a drift or an error relative to.

    The mean, unless the series straddles zero - a total energy can - in which
    case its spread is the only meaningful scale.
    """
    return max(abs(float(values.mean())), float(np.std(values)), TINY)


def _spacing_ps(times: npt.NDArray[np.float64]) -> float:
    """Time between rows, from the series rather than assumed."""
    if times.size < 2:
        return 0.0
    return float(np.median(np.diff(times)))


def _relative_drift(
    times: npt.NDArray[np.float64], values: npt.NDArray[np.float64], scale: float
) -> float:
    """Change across the window from a straight-line fit, over *scale*."""
    if values.size < 2:
        return 0.0
    span = float(times[-1] - times[0])
    if abs(span) < TINY:
        return 0.0
    slope = float(np.polyfit(times, values, 1)[0])
    return abs(slope * span) / scale


def _hold_of(
    recorded_holds: Sequence[float],
    csv_paths: Sequence[Any],
    names: Sequence[str],
    n_segments: int,
) -> float | None:
    """How long each temperature was held.

    A stage that recorded its segment durations says so outright, which is the
    only answer that survives a ladder split across stages or a stage resumed
    part-way through. Failing that, fall back to dividing one stage's CSV by
    its segment count - all a manifest written before those durations were
    recorded can offer - and only for a single stage, because across several
    that division is a different wrong number for each of them, and a wrong
    cooling rate is worse than none.
    """
    if n_segments > 0 and len(recorded_holds) == n_segments:
        return float(np.median(np.asarray(recorded_holds, dtype=np.float64)))
    if len(names) != 1:
        log.info(
            "Stages %s did not record their segment durations, so the cooling "
            "rate across them cannot be recovered.",
            ", ".join(names),
        )
        return None
    return _hold_ps(csv_paths[0], n_segments)


def _hold_ps(csv_path: Any, n_segments: int) -> float | None:
    """How long each temperature was held, from the stage's CSV.

    A :class:`~openmmpolymer.simulate.StageResult` records how many steps ran
    but not the timestep, and ``SystemSpec`` records neither duration, so the
    CSV's last time is the only thing that knows.
    """
    if not csv_path or n_segments <= 0:
        return None
    path = Path(str(csv_path))
    if not path.is_file():
        log.info("No CSV at %s, so the cooling rate cannot be recovered.", path)
        return None
    try:
        series = read_state_data(path)
    except AnalysisError:
        return None
    return float(series.time_ps[-1]) / n_segments


def _temperature_step(temperature_k: npt.NDArray[np.float64]) -> float:
    """The typical gap between the temperatures on a ladder."""
    if temperature_k.size < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(temperature_k))))


def _cooling_rate(
    temperature_k: npt.NDArray[np.float64], hold_ps: float | None
) -> float | None:
    """Kelvin per nanosecond, from the temperature step and the hold.

    None rather than zero when the ladder never stepped: a stage that held one
    temperature was not cooling, and a rate of zero would read as cooling
    infinitely slowly - which is the opposite of the truth.
    """
    if hold_ps is None or hold_ps <= 0.0 or temperature_k.size < 2:
        return None
    step = _temperature_step(temperature_k)
    return None if step <= 0.0 else step / hold_ps * 1000.0
