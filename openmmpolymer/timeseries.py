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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .trajectory import AnalysisError

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
#: unless the window's own standard error is larger.
MAX_RELATIVE_DRIFT = 0.01

#: How far into a series the settling point may sit. Past halfway there is
#: more discarded than kept, and the run was too short whatever the numbers
#: say.
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

#: Guards a relative quantity whose scale is zero, for a series that never
#: moves - a constant density, or an energy held at exactly zero.
_TINY = 1.0e-30


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
            how many genuinely uncorrelated samples the window is worth.
        correlation_time_ps: The series' own correlation time.
        relative_standard_error: Standard error of the window's mean, as a
            fraction of its scale.
        relative_drift: Change across the window from a straight-line fit, as
            a fraction of its scale.
        equilibrated: Whether the window settled early enough, holds at least
            :data:`MIN_INDEPENDENT_SAMPLES` independent samples, and drifts by
            no more than :data:`MAX_RELATIVE_DRIFT` or its own standard error,
            whichever is larger. This is one observable. It says that *this*
            quantity stopped drifting faster than its own noise, not that the
            system is equilibrated - a melt's density settles in a few hundred
            picoseconds while its chains need tens of nanoseconds, and no
            amount of converged density says anything about them.
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


@dataclass(frozen=True)
class GlassTransition:
    """A two-line fit to a specific-volume curve.

    Args:
        temperature_k: Where the two branches meet.
        specific_volume_cm3_g: Specific volume at the crossing. With the two
            slopes it fixes both fitted lines, so a plot needs nothing else.
        melt_expansion_per_k: Slope of the high-temperature branch, in
            cm^3/g/K.
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
    table = np.genfromtxt(path, delimiter=",", names=True)
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
        inefficiency = _statistical_inefficiency(window)
        effective = window.size / inefficiency
        if effective > best_effective:
            best_start = int(start)
            best_effective = effective
            best_inefficiency = inefficiency

    window = series[best_start:]
    scale = _scale_of(window)
    spacing = _spacing_ps(times)
    drift = _relative_drift(times[best_start:], window, scale)
    standard_error = float(np.std(window) / np.sqrt(best_effective)) / scale
    return Equilibration(
        start_index=best_start,
        start_ps=float(times[best_start]),
        n_samples=int(window.size),
        n_independent_samples=float(best_effective),
        correlation_time_ps=max(0.0, (best_inefficiency - 1.0) / 2.0) * spacing,
        relative_standard_error=standard_error,
        relative_drift=drift,
        equilibrated=(
            best_start <= series.size * MAX_START_FRACTION
            and best_effective >= MIN_INDEPENDENT_SAMPLES
            and drift <= max(MAX_RELATIVE_DRIFT, 2.0 * standard_error)
        ),
    )


def quench_curve(run_dir: str | Path, stage: str = "06_quench") -> QuenchCurve:
    """Read back the specific-volume curve a quench stage recorded.

    The densities are already in the manifest: a quench holds each temperature
    for a while and records the mean of the second half of each hold. What is
    not in the manifest is how long the holds were, so the stage's CSV is read
    too, purely to recover the cooling rate.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        stage: Which stage cooled the cell.

    Returns:
        The curve, ordered by ascending temperature.

    Raises:
        AnalysisError: There is no manifest, the stage is not in it, or it
            recorded no per-temperature densities.
    """
    from .protocols import RunManifest

    directory = Path(run_dir)
    manifest = RunManifest.load(directory)
    if manifest is None:
        raise AnalysisError(f"No manifest in {directory}.")
    recorded = manifest.stages.get(stage)
    if recorded is None:
        raise AnalysisError(
            f"The manifest in {directory} has no stage {stage!r}. It records: "
            f"{', '.join(manifest.stages) or 'nothing'}."
        )
    samples = recorded.get("samples") or {}
    temperatures = samples.get("segment_temperature_k")
    densities = samples.get("segment_density_g_cm3")
    if not temperatures or not densities:
        raise AnalysisError(
            f"Stage {stage!r} recorded no per-temperature densities, so it was "
            "not a quench. A quench holds a ladder of temperatures and records "
            "the density at each."
        )

    temperature = np.asarray(temperatures, dtype=np.float64)
    density = np.asarray(densities, dtype=np.float64)
    order = np.argsort(temperature)
    temperature, density = temperature[order], density[order]
    if np.any(density <= 0.0):
        raise AnalysisError(
            f"Stage {stage!r} recorded a density of zero or less, so its "
            "specific volume is not defined."
        )

    hold_ps = _hold_ps(recorded.get("csv"), len(temperatures))
    return QuenchCurve(
        stage=stage,
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
    from ._validation import require_integer

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
        glass, glass_residual = _fit_line(
            temperature[:break_index], volume[:break_index]
        )
        melt, melt_residual = _fit_line(temperature[break_index:], volume[break_index:])
        total = glass_residual + melt_residual
        if best is None or total < best[0]:
            best = (total, break_index, glass, melt)

    assert best is not None  # the loop runs at least once
    total, break_index, glass, melt = best
    glass_slope, glass_intercept = glass
    melt_slope, melt_intercept = melt

    separation = melt_slope - glass_slope
    if abs(separation) < _TINY:
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
    spacing = (
        float(np.median(np.abs(np.diff(temperature_k))))
        if temperature_k.size > 1
        else 0.0
    )
    lower = float(temperature_k[break_index - 1]) - spacing
    upper = float(temperature_k[min(break_index, temperature_k.size - 1)]) + spacing
    return lower <= crossing_k <= upper


def _statistical_inefficiency(values: npt.NDArray[np.float64]) -> float:
    """Return ``1 + 2 * sum(C(t))``, the samples one sample is worth.

    The autocorrelation comes from an FFT, and the sum is truncated at its
    first non-positive term: past that point the estimator is noise, and
    summing the noise is how a correlation time ends up longer than the run.
    """
    n = values.size
    centred = values - values.mean()
    variance = float(centred @ centred) / n
    if variance <= _TINY:
        return 1.0
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n].real
    correlation /= n * variance

    total = 0.0
    for lag in range(1, n):
        term = float(correlation[lag])
        if term <= 0.0:
            break
        total += term * (1.0 - lag / n)
    return max(1.0, 1.0 + 2.0 * total)


def _scale_of(values: npt.NDArray[np.float64]) -> float:
    """A scale to make a drift or an error relative to.

    The mean, unless the series straddles zero - a total energy can - in which
    case its spread is the only meaningful scale.
    """
    return max(abs(float(values.mean())), float(np.std(values)), _TINY)


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
    if abs(span) < _TINY:
        return 0.0
    slope = float(np.polyfit(times, values, 1)[0])
    return abs(slope * span) / scale


def _fit_line(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[tuple[float, float], float]:
    """Least-squares line through *x*, *y*, with its sum of squared residuals."""
    design = np.vstack([x, np.ones_like(x)]).T
    solution, residuals, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(solution[0]), float(solution[1])
    if residuals.size:
        total = float(residuals[0])
    else:
        predicted = design @ solution
        total = float(((y - predicted) ** 2).sum())
    return (slope, intercept), total


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


def _cooling_rate(
    temperature_k: npt.NDArray[np.float64], hold_ps: float | None
) -> float | None:
    """Kelvin per nanosecond, from the temperature step and the hold."""
    if hold_ps is None or hold_ps <= 0.0 or temperature_k.size < 2:
        return None
    step = float(np.median(np.abs(np.diff(temperature_k))))
    return step / hold_ps * 1000.0
