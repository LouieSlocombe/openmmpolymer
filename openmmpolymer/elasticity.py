"""Elastic constants, read off what a deformation recorded.

Arithmetic over numbers a run already wrote down. Nothing here opens a
Context, nothing here writes a file: the same discipline
:mod:`openmmpolymer.timeseries` keeps, and for the same reason - it makes the
whole analysis testable without a filesystem and keeps every write in one
place.

Four constants, and the fourth is the point. ``E`` and ``nu`` come from one
uniaxial extension; ``K`` and ``G`` are measured separately, from a pressure
ladder and a shear ladder. For an isotropic solid the four are two, related
by ``K = E / 3(1 - 2nu)`` and ``G = E / 2(1 + nu)``, so measuring all four
over-determines the pair and the gap between measured and implied is a check
on all of them at once. It is the only number here that is not a straight
line fitted to points someone chose the ends of.

Every fit carries a ``resolved`` flag and every flag can be False, which is
the same refusal :func:`~openmmpolymer.timeseries.glass_transition` makes. A
modulus is a slope, and a slope through a short stretch of a very noisy curve
is always *something*. The reasons it can be False are: too few points; a
standard error too large to support the number; a negative modulus; and - the
one that catches the real mistake - the two halves of the fitting window
disagreeing about the slope, which is what fitting a line across the yield
point, or across nothing but noise, looks like from the inside.

A Poisson's ratio outside ``0 < nu < 0.5`` is refused too. An isotropic solid
cannot have one, so it is not a surprising measurement, it is a cell whose
lateral axes have not relaxed.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .timeseries import _fit_line
from .trajectory import AnalysisError

log = logging.getLogger(__name__)

#: 1 bar in MPa. Stress is recorded in bar, to match the pressures every other
#: stage is set in, and reported in MPa, which is what a modulus is quoted in.
MPA_PER_BAR = 0.1

#: Largest relative standard error a fitted modulus may carry and still call
#: itself resolved.
MAX_RELATIVE_STANDARD_ERROR = 0.25

#: How far the two halves of a fitting window may disagree about the slope,
#: relative to the whole-window slope, before the fit is reporting a curve
#: rather than a line.
MAX_HALF_SLOPE_DISAGREEMENT = 0.5

#: How far the compression and decompression branches of a pressure ladder may
#: disagree before the cell is being driven rather than probed.
MAX_BULK_HYSTERESIS = 0.25

#: How far a measured K or G may sit from the one E and nu imply before the
#: four constants stop describing one isotropic solid.
MAX_CONSISTENCY_GAP = 0.3

#: Smallest denominator worth dividing by.
_TINY = 1.0e-12


# --------------------------------------------------------------------------
# What a deformation leaves behind
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StressStrain:
    """A stress-strain curve, as a deformation stage recorded it.

    Args:
        stage: Which stage or stages produced it.
        axis: The axis that was driven, 0, 1 or 2.
        strain: Engineering strain along that axis, ascending.
        stress_mpa: The stress along it, tension positive.
        lateral_strain: ``(n, 2)`` - engineering strain on the two other axes,
            in ascending axis order.
        lateral_stress_mpa: ``(n, 2)`` - the stress on them, which a barostat
            was holding at the target pressure and which says how well it
            managed.
        temperature_k: The temperature it was deformed at.
        strain_rate_per_ns: Strain added per nanosecond. Reported with every
            number derived from this curve, because an all-atom deformation
            runs some ten orders of magnitude faster than a tensile test and
            a modulus measured here is not the quasi-static one.
        controlled: ``"strain"`` when the strain was imposed and the stress
            measured, ``"stress"`` when it was the other way round.
    """

    stage: str
    axis: int
    strain: npt.NDArray[np.float64]
    stress_mpa: npt.NDArray[np.float64]
    lateral_strain: npt.NDArray[np.float64]
    lateral_stress_mpa: npt.NDArray[np.float64]
    temperature_k: float
    strain_rate_per_ns: float | None
    controlled: str = "strain"

    @property
    def n_points(self) -> int:
        """How many strains were held."""
        return int(self.strain.size)

    @property
    def tensile_stress_mpa(self) -> npt.NDArray[np.float64]:
        """The axial stress less the mean of the two lateral ones.

        The lateral axes are only at their set pressure on average and only
        as fast as the cell relaxes. Subtracting what they actually did takes
        a drift off the driven axis that would otherwise be read as stiffness.
        """
        return self.stress_mpa - self.lateral_stress_mpa.mean(axis=1)

    @property
    def transverse_strain(self) -> npt.NDArray[np.float64]:
        """The mean of the two lateral strains, which Poisson's ratio uses."""
        return np.asarray(self.lateral_strain.mean(axis=1), dtype=np.float64)


@dataclass(frozen=True)
class ElasticModulus:
    """A straight line through the elastic part of a stress-strain curve.

    Args:
        modulus_mpa: The slope, which is Young's modulus.
        intercept_mpa: Where that line crosses zero strain. A large one is a
            cell that was already under stress when the deformation started.
        strain_limit: The strain the fit stopped at.
        n_points: Points inside the window.
        residual_mpa: Root-mean-square residual of the fit.
        standard_error_mpa: Standard error of the slope.
        half_disagreement: How differently the first and second halves of the
            window slope, relative to the whole. Small for a line; large for
            a curve that a line has been drawn through anyway.
        temperature_k: Carried through from the curve.
        strain_rate_per_ns: Likewise, and for the same reason the cooling rate
            travels with a glass transition: the number above is meaningless
            without it.
        resolved: Whether this is a modulus rather than a slope. False when
            there are too few points, when the standard error is more than
            :data:`MAX_RELATIVE_STANDARD_ERROR` of the modulus, when the two
            halves of the window disagree by more than
            :data:`MAX_HALF_SLOPE_DISAGREEMENT`, or when the modulus is not
            positive. Even when True this is a modulus at this strain rate
            and this temperature, not a quasi-static one.
    """

    modulus_mpa: float
    intercept_mpa: float
    strain_limit: float
    n_points: int
    residual_mpa: float
    standard_error_mpa: float
    half_disagreement: float
    temperature_k: float
    strain_rate_per_ns: float | None
    resolved: bool

    @property
    def modulus_gpa(self) -> float:
        """The same number in GPa, which is how a stiff polymer is quoted."""
        return self.modulus_mpa / 1000.0

    @property
    def relative_standard_error(self) -> float:
        """The standard error as a fraction of the modulus, or infinity."""
        if abs(self.modulus_mpa) < _TINY:
            return math.inf
        return abs(self.standard_error_mpa / self.modulus_mpa)


@dataclass(frozen=True)
class PoissonRatio:
    """How much the cell narrowed for how much it was stretched.

    Args:
        ratio: Minus the slope of mean transverse strain against axial
            strain. Fitted over the window rather than taken as a ratio at
            one point, which divides two small noisy numbers.
        standard_error: Standard error of that slope.
        n_points: Points inside the window.
        strain_limit: The strain the fit stopped at.
        resolved: Whether the fit is usable *and* the answer is physical.
            An isotropic solid has ``0 < nu < 0.5``; outside that the lateral
            axes have not relaxed, whatever the fit quality says.
    """

    ratio: float
    standard_error: float
    n_points: int
    strain_limit: float
    resolved: bool


@dataclass(frozen=True)
class BulkModulus:
    """How hard the cell is to compress, from a pressure ladder.

    Args:
        stage: Which stage produced it.
        modulus_mpa: ``-V (dP/dV)``, fitted as ``-1 / (d ln V / dP)`` over the
            whole ladder.
        pressure_bar: The pressures held.
        volume_nm3: The mean volume at each.
        compression_mpa: The same fit over the rising half of the ladder only.
        decompression_mpa: And over the falling half. A ladder that goes up
            and comes back down gives these for free.
        hysteresis: The relative gap between those two. Large means the cell
            did not come back, so the ladder was a deformation rather than a
            measurement.
        n_points: Points in the whole fit.
        temperature_k: The temperature it was held at.
        resolved: A positive modulus, enough points, and a hysteresis below
            :data:`MAX_BULK_HYSTERESIS`.
    """

    stage: str
    modulus_mpa: float
    pressure_bar: npt.NDArray[np.float64]
    volume_nm3: npt.NDArray[np.float64]
    compression_mpa: float | None
    decompression_mpa: float | None
    hysteresis: float
    n_points: int
    temperature_k: float
    resolved: bool


@dataclass(frozen=True)
class ShearModulus:
    """The slope of shear stress against shear strain.

    Args:
        stage: Which stage produced it.
        modulus_mpa: The slope, which is ``G``.
        strain: The shear strains held.
        stress_mpa: The shear stress measured at each.
        standard_error_mpa: Standard error of the slope.
        n_points: Points in the fit.
        temperature_k: The temperature it was held at.
        resolved: A positive modulus whose standard error supports it. A
            liquid has ``G = 0`` and will not resolve, which is the correct
            answer rather than a failure - a melt above its glass transition
            has no static shear modulus to measure.
    """

    stage: str
    modulus_mpa: float
    strain: npt.NDArray[np.float64]
    stress_mpa: npt.NDArray[np.float64]
    standard_error_mpa: float
    n_points: int
    temperature_k: float
    resolved: bool


@dataclass(frozen=True)
class ElasticConsistency:
    """Whether the four measured constants describe one isotropic solid.

    ``K = E / 3(1 - 2nu)`` and ``G = E / 2(1 + nu)`` hold for an isotropic
    linear elastic material. Measuring ``K`` and ``G`` as well as ``E`` and
    ``nu`` over-determines the pair, so the gap between what was measured and
    what is implied tests all four at once - and unlike each of them, it is
    not a straight line fitted through a window someone picked.

    Args:
        bulk_implied_mpa: The ``K`` that ``E`` and ``nu`` imply.
        shear_implied_mpa: The ``G`` they imply.
        bulk_measured_mpa: The measured ``K``, or None if it was not measured.
        shear_measured_mpa: Likewise for ``G``.
        bulk_gap: Relative gap between the two ``K``s, or NaN.
        shear_gap: Likewise for ``G``.
        consistent: Every gap that could be computed is below
            :data:`MAX_CONSISTENCY_GAP`. False when nothing could be
            computed: a check that did not run is not a check that passed.
    """

    bulk_implied_mpa: float
    shear_implied_mpa: float
    bulk_measured_mpa: float | None
    shear_measured_mpa: float | None
    bulk_gap: float
    shear_gap: float
    consistent: bool


# --------------------------------------------------------------------------
# Reading a run directory
# --------------------------------------------------------------------------


def _manifest_stages(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Every stage a manifest records, or a message saying there is none."""
    from .protocols import RunManifest

    directory = Path(run_dir)
    manifest = RunManifest.load(directory)
    if manifest is None:
        raise AnalysisError(f"No manifest in {directory}.")
    return manifest.stages


def _is_kind(recorded: dict[str, Any], key: str) -> bool:
    """Whether a recorded stage holds a ladder of *key*."""
    samples = recorded.get("samples") or {}
    values = samples.get(key)
    return isinstance(values, list) and len(values) >= 1


def _stages_holding(run_dir: str | Path, key: str, what: str) -> tuple[str, ...]:
    """Name every stage whose samples carry *key*, in manifest order."""
    stages = _manifest_stages(run_dir)
    found = tuple(name for name, recorded in stages.items() if _is_kind(recorded, key))
    if not found:
        raise AnalysisError(
            f"No stage in {Path(run_dir)} recorded {what}. It records: "
            f"{', '.join(stages) or 'nothing'}."
        )
    return found


def deform_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name every stage in a run that stretched the cell along an axis.

    Found by what a stage recorded rather than by what it was called, the
    same way :func:`~openmmpolymer.timeseries.quench_stages` finds a quench,
    so a ladder split into chunks for resume - or repeated as several
    replicas - is read without anything having to agree on names in advance.

    Args:
        run_dir: A directory a run wrote to.

    Returns:
        The stage names, in the order the manifest records them.

    Raises:
        AnalysisError: There is no manifest, or nothing in it was a
            deformation.
    """
    return _stages_holding(run_dir, "segment_strain", "a ladder of strains")


def load_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name every stage that pulled at a known stress.

    Raises:
        AnalysisError: There is no manifest, or nothing in it was a load.
    """
    return _stages_holding(
        run_dir, "segment_applied_stress_bar", "a ladder of applied stresses"
    )


def shear_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name every stage that sheared the cell.

    Raises:
        AnalysisError: There is no manifest, or nothing in it was a shear.
    """
    return _stages_holding(run_dir, "segment_shear_strain", "a ladder of shear strains")


def bulk_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name every stage that stepped through a ladder of pressures.

    Raises:
        AnalysisError: There is no manifest, or nothing in it was a ladder.
    """
    return _stages_holding(run_dir, "segment_pressure_bar", "a ladder of pressures")


def _gather(
    run_dir: str | Path, names: Sequence[str]
) -> tuple[dict[str, list[float]], float]:
    """Concatenate the samples of several stages, in the order given.

    A strain ladder split across stages for resume is one curve; reading each
    chunk as its own would give several short ones and fit a modulus to each.
    """
    stages = _manifest_stages(run_dir)
    merged: dict[str, list[float]] = {}
    temperatures: list[float] = []
    for name in names:
        recorded = stages.get(name)
        if recorded is None:
            raise AnalysisError(
                f"{Path(run_dir)} has no stage called {name!r}. It records: "
                f"{', '.join(stages) or 'nothing'}."
            )
        samples = recorded.get("samples") or {}
        for key, values in samples.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
        mean = recorded.get("mean_temperature_k")
        if mean is not None:
            temperatures.append(float(mean))
    return merged, float(np.mean(temperatures)) if temperatures else math.nan


def _strain_rate_per_ns(
    strains: Sequence[float], durations: Sequence[float]
) -> float | None:
    """Strain added per nanosecond, or None when the holds were not recorded.

    Total strain over total time. The holds are read back rather than
    inferred, for the reason ``segment_duration_ps`` is recorded at all: a
    ladder split across stages or resumed part-way through knows its own
    total time only if each piece wrote down its own.
    """
    if not durations or len(durations) != len(strains) or min(durations) <= 0.0:
        return None
    total_ps = float(sum(durations))
    if total_ps <= 0.0:
        return None
    return abs(float(strains[-1])) / total_ps * 1000.0


def stress_strain(
    run_dir: str | Path, stage: str | Sequence[str] | None = None
) -> StressStrain:
    """Read the stress-strain curve a deformation stage left behind.

    Args:
        run_dir: A directory a run wrote to.
        stage: The stage or stages to read, or None for every deformation
            stage in the manifest, in order. Several chunks of one ladder
            read as one curve.

    Returns:
        The curve.

    Raises:
        AnalysisError: There is nothing there to read, or what is there is
            not a deformation.
    """
    names = (
        deform_stages(run_dir)
        if stage is None
        else ((stage,) if isinstance(stage, str) else tuple(stage))
    )
    samples, temperature = _gather(run_dir, names)
    if "segment_strain" not in samples:
        raise AnalysisError(
            f"{', '.join(names)} recorded no strains, so nothing there was a "
            "deformation."
        )

    axis = int(samples.get("deform_axis", [2.0])[0])
    lateral = [index for index in range(3) if index != axis]
    reference = samples.get("reference_box_nm")
    if reference is None or len(reference) < 3:
        raise AnalysisError(
            f"{', '.join(names)} recorded no unstrained cell to measure "
            "strain against. A deformation stage writes reference_box_nm; "
            "this manifest predates that or was not written by one."
        )
    origin = np.asarray(reference[:3], dtype=np.float64)

    strain = np.asarray(samples["segment_strain"], dtype=np.float64)
    order = np.argsort(strain)
    names_xyz = "xyz"
    boxes = np.column_stack(
        [
            np.asarray(samples[f"segment_box_{n}_nm"], dtype=np.float64)
            for n in names_xyz
        ]
    )
    stresses = (
        np.column_stack(
            [
                np.asarray(samples[f"segment_stress_{n}{n}_bar"], dtype=np.float64)
                for n in names_xyz
            ]
        )
        * MPA_PER_BAR
    )

    return StressStrain(
        stage=", ".join(names),
        axis=axis,
        strain=strain[order],
        stress_mpa=stresses[order, axis],
        lateral_strain=(boxes[:, lateral] / origin[lateral] - 1.0)[order],
        lateral_stress_mpa=stresses[order][:, lateral],
        temperature_k=temperature,
        strain_rate_per_ns=_strain_rate_per_ns(
            samples["segment_strain"], samples.get("segment_duration_ps", [])
        ),
        controlled="strain",
    )


def load_curve(
    run_dir: str | Path, stage: str | Sequence[str] | None = None
) -> StressStrain:
    """Read the curve a constant-stress load stage left behind.

    The same shape as :func:`stress_strain` and deliberately so - the two are
    fitted by the same function and compared directly - but here the stress
    is the imposed quantity and the strain is what was measured. No virial
    was involved in producing it, which is the whole reason for running one.

    Args:
        run_dir: A directory a run wrote to.
        stage: The stage or stages to read, or None for every load stage.

    Returns:
        The curve, with ``controlled="stress"``.

    Raises:
        AnalysisError: There is nothing there to read.
    """
    names = (
        load_stages(run_dir)
        if stage is None
        else ((stage,) if isinstance(stage, str) else tuple(stage))
    )
    samples, temperature = _gather(run_dir, names)
    axis = int(samples.get("load_axis", [2.0])[0])
    lateral = [index for index in range(3) if index != axis]
    applied = np.asarray(samples["segment_applied_stress_bar"], dtype=np.float64)
    boxes = np.column_stack(
        [np.asarray(samples[f"segment_box_{n}_nm"], dtype=np.float64) for n in "xyz"]
    )
    # The zero-stress rung is the unstrained cell, measured in the same
    # ensemble as the rest rather than inherited from a different stage.
    origin = boxes[int(np.argmin(np.abs(applied)))]
    order = np.argsort(applied)
    strain = (boxes[:, axis] / origin[axis] - 1.0)[order]
    return StressStrain(
        stage=", ".join(names),
        axis=axis,
        strain=strain,
        stress_mpa=applied[order] * MPA_PER_BAR,
        lateral_strain=(boxes[:, lateral] / origin[lateral] - 1.0)[order],
        # Zero rather than NaN, and measured rather than missing: the
        # applied stress was set as `lateral_pressure - sigma`, so it is
        # already expressed relative to what the lateral axes are held at.
        # Subtracting them again in `tensile_stress_mpa` would double-count.
        lateral_stress_mpa=np.zeros((applied.size, 2), dtype=np.float64),
        temperature_k=temperature,
        strain_rate_per_ns=None,
        controlled="stress",
    )


# --------------------------------------------------------------------------
# The fits
# --------------------------------------------------------------------------


def _slope_error(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], residual_sum: float
) -> float:
    """Standard error of a least-squares slope.

    ``sqrt( sum(residual^2) / (n - 2) / sum((x - xbar)^2) )``, which is
    infinite when there are only two points: a line through two points has no
    residual and no error, and reporting zero would make the least supported
    fit look like the best one.
    """
    n = x.size
    spread = float(((x - x.mean()) ** 2).sum())
    if n <= 2 or spread < _TINY:
        return math.inf
    return math.sqrt(max(residual_sum, 0.0) / (n - 2) / spread)


def _half_disagreement(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], slope: float
) -> float:
    """How differently the two halves of a window slope, relative to the whole.

    The check that catches the real mistake. A fit reports a residual, and a
    gently curving stress-strain curve fitted across its knee has a small
    one - the line sits happily above the data at both ends and below it in
    the middle. Splitting the window and comparing the two slopes does not:
    a curve slopes differently at its two ends by construction, and a line
    does not.
    """
    if x.size < 4 or abs(slope) < _TINY:
        return 0.0 if x.size < 4 else math.inf
    middle = x.size // 2
    slopes = []
    for part in (slice(None, middle), slice(middle, None)):
        if x[part].size < 2:
            return math.inf
        (piece, _), _ = _fit_line(x[part], y[part])
        slopes.append(piece)
    return abs(slopes[0] - slopes[1]) / abs(slope)


def _window(
    strain: npt.NDArray[np.float64], strain_limit: float
) -> npt.NDArray[np.bool_]:
    """Which points fall inside the elastic window."""
    return np.asarray(np.abs(strain) <= strain_limit + _TINY, dtype=np.bool_)


def youngs_modulus(
    curve: StressStrain,
    *,
    strain_limit: float = 0.015,
    min_points: int = 5,
) -> ElasticModulus:
    """Fit Young's modulus to the linear part of a stress-strain curve.

    Args:
        curve: The curve, from :func:`stress_strain` or :func:`load_curve`.
        strain_limit: The strain to fit up to. Past the linear elastic range
            the slope is not a modulus, and where that range ends is a
            property of the polymer rather than a constant - which is why
            :attr:`ElasticModulus.half_disagreement` is reported rather than
            this being assumed correct.
        min_points: Fewest points a fit may rest on.

    Returns:
        The fit, ``resolved`` False when it is a slope rather than a modulus.

    Raises:
        ValueError: The window is not a strain.
    """
    if strain_limit <= 0.0:
        raise ValueError(f"strain_limit={strain_limit} must be above zero.")
    inside = _window(curve.strain, strain_limit)
    x = curve.strain[inside]
    y = curve.tensile_stress_mpa[inside]
    if x.size < 2:
        return ElasticModulus(
            modulus_mpa=math.nan,
            intercept_mpa=math.nan,
            strain_limit=strain_limit,
            n_points=int(x.size),
            residual_mpa=math.nan,
            standard_error_mpa=math.inf,
            half_disagreement=math.inf,
            temperature_k=curve.temperature_k,
            strain_rate_per_ns=curve.strain_rate_per_ns,
            resolved=False,
        )

    (slope, intercept), residual_sum = _fit_line(x, y)
    error = _slope_error(x, y, residual_sum)
    disagreement = _half_disagreement(x, y, slope)
    resolved = bool(
        x.size >= min_points
        and slope > 0.0
        and error < MAX_RELATIVE_STANDARD_ERROR * abs(slope)
        and disagreement <= MAX_HALF_SLOPE_DISAGREEMENT
    )
    return ElasticModulus(
        modulus_mpa=float(slope),
        intercept_mpa=float(intercept),
        strain_limit=strain_limit,
        n_points=int(x.size),
        residual_mpa=float(math.sqrt(max(residual_sum, 0.0) / x.size)),
        standard_error_mpa=float(error),
        half_disagreement=float(disagreement),
        temperature_k=curve.temperature_k,
        strain_rate_per_ns=curve.strain_rate_per_ns,
        resolved=resolved,
    )


def poisson_ratio(
    curve: StressStrain,
    *,
    strain_limit: float = 0.015,
    min_points: int = 5,
) -> PoissonRatio:
    """Fit Poisson's ratio to the transverse response of a curve.

    Minus the slope of mean transverse strain against axial strain, fitted
    over the window rather than taken as a ratio at the last point - which
    divides one small noisy number by another.

    Args:
        curve: The curve.
        strain_limit: The strain to fit up to.
        min_points: Fewest points a fit may rest on.

    Returns:
        The ratio, ``resolved`` False when the fit is poor or the answer is
        not one an isotropic solid can have.
    """
    inside = _window(curve.strain, strain_limit)
    x = curve.strain[inside]
    y = curve.transverse_strain[inside]
    if x.size < 2:
        return PoissonRatio(
            ratio=math.nan,
            standard_error=math.inf,
            n_points=int(x.size),
            strain_limit=strain_limit,
            resolved=False,
        )
    (slope, _), residual_sum = _fit_line(x, y)
    error = _slope_error(x, y, residual_sum)
    ratio = -float(slope)
    resolved = bool(
        x.size >= min_points
        and 0.0 < ratio < 0.5
        and error < MAX_RELATIVE_STANDARD_ERROR * abs(slope)
    )
    return PoissonRatio(
        ratio=ratio,
        standard_error=float(error),
        n_points=int(x.size),
        strain_limit=strain_limit,
        resolved=resolved,
    )


def bulk_modulus(
    run_dir: str | Path,
    stage: str | Sequence[str] | None = None,
    *,
    min_points: int = 4,
) -> BulkModulus:
    """Fit the bulk modulus to a pressure ladder.

    ``K = -V (dP/dV)``, fitted as ``-1 / (d ln V / dP)`` so that the whole
    ladder contributes one slope rather than a finite difference between
    neighbouring rungs.

    A compression stage that climbs and comes back down gives the two
    branches separately for nothing, and their gap is the check that the
    ladder probed the cell rather than driving it. That is why a ladder for
    this should be gentle: the package's default compression reaches a
    kilobar, which is a long way outside linear response for a polymer.

    The volume comes from the recorded density, which is the cell's mass over
    its volume, so the mass cancels and no extra bookkeeping is needed.

    Args:
        run_dir: A directory a run wrote to.
        stage: The stage or stages to read, or None for every pressure
            ladder in the manifest.
        min_points: Fewest rungs a fit may rest on.

    Returns:
        The fit.

    Raises:
        AnalysisError: There is nothing there to read.
    """
    names = (
        bulk_stages(run_dir)
        if stage is None
        else ((stage,) if isinstance(stage, str) else tuple(stage))
    )
    samples, temperature = _gather(run_dir, names)
    pressure = np.asarray(samples["segment_pressure_bar"], dtype=np.float64)
    density = np.asarray(samples["segment_density_g_cm3"], dtype=np.float64)
    if density.size != pressure.size or np.any(density <= 0.0):
        raise AnalysisError(
            f"{', '.join(names)} recorded {pressure.size} pressures and "
            f"{density.size} usable densities, which cannot be paired."
        )
    # Volume in arbitrary units: only d(ln V)/dP is used, so the constant
    # of proportionality between 1/density and volume drops out.
    log_volume = -np.log(density)

    def fit(mask: npt.NDArray[np.bool_]) -> float | None:
        if int(mask.sum()) < 2:
            return None
        (slope, _), _ = _fit_line(pressure[mask], log_volume[mask])
        if abs(slope) < _TINY:
            return None
        return float(-1.0 / slope * MPA_PER_BAR)

    everything = np.ones(pressure.size, dtype=np.bool_)
    peak = int(np.argmax(pressure))
    rising = np.zeros(pressure.size, dtype=np.bool_)
    rising[: peak + 1] = True
    falling = np.zeros(pressure.size, dtype=np.bool_)
    falling[peak:] = True

    modulus = fit(everything)
    up = fit(rising) if peak >= 1 else None
    down = fit(falling) if peak <= pressure.size - 2 else None
    hysteresis = math.nan
    if up is not None and down is not None and abs(up) > _TINY:
        hysteresis = abs(down - up) / abs(up)

    resolved = bool(
        modulus is not None
        and modulus > 0.0
        and pressure.size >= min_points
        and (math.isnan(hysteresis) or hysteresis <= MAX_BULK_HYSTERESIS)
    )
    return BulkModulus(
        stage=", ".join(names),
        modulus_mpa=math.nan if modulus is None else modulus,
        pressure_bar=pressure,
        volume_nm3=1.0 / density,
        compression_mpa=up,
        decompression_mpa=down,
        hysteresis=hysteresis,
        n_points=int(pressure.size),
        temperature_k=temperature,
        resolved=resolved,
    )


def shear_modulus(
    run_dir: str | Path,
    stage: str | Sequence[str] | None = None,
    *,
    min_points: int = 3,
) -> ShearModulus:
    """Fit the shear modulus to a ladder of shear strains.

    Args:
        run_dir: A directory a run wrote to.
        stage: The stage or stages to read, or None for every shear stage.
        min_points: Fewest strains a fit may rest on.

    Returns:
        The fit. ``resolved`` False for a liquid, which has no static shear
        modulus - the right answer for a melt above its glass transition
        rather than a failure to measure one.

    Raises:
        AnalysisError: There is nothing there to read.
    """
    names = (
        shear_stages(run_dir)
        if stage is None
        else ((stage,) if isinstance(stage, str) else tuple(stage))
    )
    samples, temperature = _gather(run_dir, names)
    strain = np.asarray(samples["segment_shear_strain"], dtype=np.float64)
    stress = (
        np.asarray(samples["segment_shear_stress_bar"], dtype=np.float64) * MPA_PER_BAR
    )
    order = np.argsort(strain)
    strain, stress = strain[order], stress[order]
    if strain.size < 2:
        return ShearModulus(
            stage=", ".join(names),
            modulus_mpa=math.nan,
            strain=strain,
            stress_mpa=stress,
            standard_error_mpa=math.inf,
            n_points=int(strain.size),
            temperature_k=temperature,
            resolved=False,
        )
    (slope, _), residual_sum = _fit_line(strain, stress)
    error = _slope_error(strain, stress, residual_sum)
    resolved = bool(
        strain.size >= min_points
        and slope > 0.0
        and error < MAX_RELATIVE_STANDARD_ERROR * abs(slope)
    )
    return ShearModulus(
        stage=", ".join(names),
        modulus_mpa=float(slope),
        strain=strain,
        stress_mpa=stress,
        standard_error_mpa=float(error),
        n_points=int(strain.size),
        temperature_k=temperature,
        resolved=resolved,
    )


def elastic_consistency(
    youngs: ElasticModulus,
    poisson: PoissonRatio,
    *,
    bulk: BulkModulus | None = None,
    shear: ShearModulus | None = None,
) -> ElasticConsistency:
    """Check the measured constants against the ones E and nu imply.

    Args:
        youngs: The Young's modulus fit.
        poisson: The Poisson's ratio fit.
        bulk: The measured bulk modulus, or None if it was not measured.
        shear: Likewise for the shear modulus.

    Returns:
        The comparison. ``consistent`` is False when neither could be
        compared, because a check that did not run is not a check that
        passed.
    """
    nu, modulus = poisson.ratio, youngs.modulus_mpa
    denominator = 3.0 * (1.0 - 2.0 * nu)
    implied_bulk = modulus / denominator if abs(denominator) > _TINY else math.nan
    implied_shear = modulus / (2.0 * (1.0 + nu)) if abs(1.0 + nu) > _TINY else math.nan

    def gap(measured: float | None, implied: float) -> float:
        if measured is None or not math.isfinite(implied) or abs(implied) < _TINY:
            return math.nan
        return abs(measured - implied) / abs(implied)

    bulk_measured = bulk.modulus_mpa if bulk is not None else None
    shear_measured = shear.modulus_mpa if shear is not None else None
    bulk_gap = gap(bulk_measured, implied_bulk)
    shear_gap = gap(shear_measured, implied_shear)
    compared = [value for value in (bulk_gap, shear_gap) if math.isfinite(value)]
    return ElasticConsistency(
        bulk_implied_mpa=implied_bulk,
        shear_implied_mpa=implied_shear,
        bulk_measured_mpa=bulk_measured,
        shear_measured_mpa=shear_measured,
        bulk_gap=bulk_gap,
        shear_gap=shear_gap,
        consistent=bool(compared) and max(compared) <= MAX_CONSISTENCY_GAP,
    )
