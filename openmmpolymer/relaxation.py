"""The relaxation modulus, read off what a step strain recorded.

Arithmetic over numbers a run already wrote down. Nothing here opens a
Context, nothing here writes a file - the discipline
:mod:`openmmpolymer.elasticity` and :mod:`openmmpolymer.timeseries` keep, and
for the same reason.

:mod:`openmmpolymer.elasticity` measures how hard a cell pushes back. This
measures how long it keeps pushing. A single affine step strain is applied, the
box is locked, and the stress decays: ``E(t)`` for a tensile step, ``G(t)`` for
a shear one. It is the function a Prony series and a stretched exponential are
fitted to, and the thing a viscoelastic material model is actually made of.

What is measured is the *shear* relaxation modulus, whichever step was
applied, and that is not a convention. For an isotropic solid under an imposed
diagonal strain, ``sigma = 2 G e + lambda tr(e) I``, so the differential stress
``sigma_zz - (sigma_xx + sigma_yy) / 2`` is exactly ``2 G (e_axial -
e_lateral)`` and the Lame constant cancels. A tensile step therefore measures
``G(t)`` with no assumption at all about Poisson's ratio, the bulk modulus, or
whether the deformation preserved the volume; ``E(t) = 2 (1 + nu) G(t)`` is the
derived number, and it is derived with the material's own ratio rather than
the one the box was scaled by. Writing ``E(t) = sigma_diff / e0`` instead - as
the textbook form does - quietly assumes both are one half, and is about one
per cent out at a four per cent strain even when they are.

Four things shape everything below.

The stress is recorded in logarithmic time bins, with the count and the mean
square beside each mean. That is what makes a relaxation split across stages
for resume read back as one curve: two chunks sharing a bin merge to exactly
the numbers one unbroken run would have written, which a stored standard error
could not do.

The decay has a floor. The pre-strain window measures what the cell was already
carrying, and its scatter is the level below which a relaxed cell and a
relaxing one are the same measurement. Every curve carries that floor and every
fit refuses to rest on points underneath it, because a stretched exponential
fitted to noise is an extremely convincing description of nothing.

What makes the early part of the curve mean anything is replicas, not
sampling. The first bins hold a reading or two, so a single run resolves the
decade or so where the stress is largest and nothing else; several independent
runs averaged bin for bin push the usable window out at both ends. That is the
reason the bin edges come from the settings rather than from the data: it is
what lets the chunks of one pass merge, and independent replicas line up, bin
for bin.

And there is no scipy here, which is a constraint rather than a preference -
the package does not depend on it. Both fits are numpy. The KWW separates: for
a fixed exponent the relation is linear in ``t**beta``, so a three-parameter
nonlinear fit is a one-dimensional search with an exact least-squares solve
inside it, the same trick :func:`~openmmpolymer.timeseries.cooling_rate_extrapolation`
uses for VFT. The Prony series separates differently: fix the time constants on
a grid and the weights are linear, subject to being non-negative - which is
what the Lawson-Hanson solve below is for, because a Prony series with a
negative weight is not a Prony series.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ._fitting import NEGLIGIBLE, separable_fit
from .elasticity import MPA_PER_BAR, _gather, _ladder, _require_stress_estimator
from .trajectory import AnalysisError, stage_names, stages_holding

log = logging.getLogger(__name__)

#: Fewest points a fit may rest on. A decay spanning three decades with twenty
#: bins to the decade has sixty; anything near this floor is a run that barely
#: sampled its own relaxation.
MIN_RELAXATION_POINTS = 8

#: The exponent bracket the KWW search runs over. One is a single exponential;
#: below about 0.05 the relation is so stretched that the fit is describing the
#: first point and the last one and nothing between them.
KWW_BETA_BRACKET = (0.10, 1.0)

#: Time constants per decade in the Prony grid. One, because the residual is
#: noise-dominated and a finer grid buys nothing: measured on planted data,
#: two per decade left the residual unchanged and made the equilibrium
#: modulus slightly worse, because a weight the data cannot place gets split
#: between neighbouring columns rather than pinned.
PRONY_PER_DECADE = 1

#: The Prony grid stops at this fraction of the observed window. An
#: exponential whose time constant is near the run length is barely
#: distinguishable from a constant - the cosine between that column and the
#: column of ones is 0.99 - so the split between it and the equilibrium
#: modulus becomes arbitrary: against a planted value of 400, a grid reaching
#: the whole window recovered 567 and one reaching a third of it 404.
PRONY_SLOWEST_FRACTION = 1.0 / 3.0

#: How many standard errors a point must stand above zero to be inside the
#: fitting window. Not a nicety: the logarithm a stretched exponential is
#: fitted through biases low by about half the squared relative error, which
#: is five per cent at a signal-to-noise of three and thirteen at two.
SIGNAL_TO_NOISE_FLOOR = 3.0

#: How far the two halves of a fitted window may disagree about the
#: stretching exponent before the fit has stopped describing the curve.
#:
#: Loose on purpose. On clean data the disagreement tells a genuine stretched
#: exponential (0.00) from one with a five per cent plateau under it (0.26),
#: but with five per cent noise a genuine one gives a median of 0.25 and draws
#: past 0.7, so any threshold tight enough to catch a small plateau throws
#: away half the honest fits too. It is set to fire on gross misfit only. A
#: plateau is caught by something that can tell: :func:`fit_prony` fits an
#: equilibrium modulus explicitly, and a KWW - which decays to zero by
#: construction - claiming to describe a curve that has one is a disagreement
#: between two independent functional forms, which
#: :func:`~openmmpolymer.viscoelastic.analyse_relaxation` checks.
MAX_KWW_HALF_DISAGREEMENT = 0.5

#: Root-mean-square residual, in log modulus, a KWW may carry and still claim
#: to describe the curve.
MAX_KWW_RESIDUAL = 0.15

#: How far past the measured window the mean relaxation time may sit, in
#: decades. It is an integral to infinity, so quoting one longer than the run
#: is extrapolation - the same refusal
#: :data:`~openmmpolymer.timeseries.MAX_EXTRAPOLATION_DECADES` makes.
MAX_KWW_EXTRAPOLATION_DECADES = 1.0

#: Decades of time a fit must span before it is a curve rather than a corner.
MIN_FITTED_DECADES = 1.0

#: How much of the relaxing weight may sit on the slowest time constant before
#: the run is judged not to have seen the end of its own relaxation.
MAX_EDGE_WEIGHT = 0.5


# --------------------------------------------------------------------------
# What a step strain leaves behind
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RelaxationCurve:
    """A shear relaxation modulus, as a step-strain stage recorded it.

    ``G(t)`` whichever step produced it - see the module docstring for why a
    tensile step measures the shear modulus without assuming anything. Young's
    modulus is a method rather than a field for the same reason: it needs the
    material's Poisson ratio, which this measurement does not contain.

    Args:
        stage: Which stage or stages produced it.
        mode: ``"tensile"`` or ``"shear"`` - which step was applied.
        bin_index: Which bin of the shared logarithmic grid each point came
            from. The merge key: replicas and chunks line up by this and not
            by their times, which differ slightly because each bin reports the
            mean time of the readings that landed in it.
        time_ps: Mean time of the readings in each bin, ascending.
        modulus_mpa: ``G(t)``, with the pre-strain baseline already taken off.
        standard_error_mpa: Its standard error. Within a bin for one replica,
            and across replicas once several have been averaged - the second
            is the honest one, and the first understates the truth because
            readings a tenth of a picosecond apart are not independent.
        n_samples: Stress readings behind each bin.
        step_strain: The strain that was applied.
        strain_measure: What the stress was divided by to get a modulus:
            ``2 (e_axial - e_lateral)`` for a tensile step, the shear strain
            itself for a shear one.
        temperature_k: The temperature it relaxed at.
        poisson: The lateral contraction the tensile step imposed, which is a
            property of the deformation and not of the polymer. NaN for a
            shear step, which needs none.
        baseline_mpa: The deviatoric stress the cell was already carrying, in
            modulus units, subtracted from every point.
        noise_floor_mpa: The uncertainty in that zero level. A modulus below
            this has not been told apart from a fully relaxed cell.
        instant_mpa: The modulus read immediately after the step, before any
            dynamics - the unrelaxed, affine response. NaN when the stage did
            not record one.
        n_replicas: Independent runs averaged. One means
            :attr:`standard_error_mpa` is a within-run scatter and not an
            error bar over anything.
    """

    stage: str
    mode: str
    bin_index: npt.NDArray[np.int64]
    time_ps: npt.NDArray[np.float64]
    modulus_mpa: npt.NDArray[np.float64]
    standard_error_mpa: npt.NDArray[np.float64]
    n_samples: npt.NDArray[np.float64]
    step_strain: float
    strain_measure: float
    temperature_k: float
    poisson: float
    baseline_mpa: float
    noise_floor_mpa: float
    instant_mpa: float = math.nan
    n_replicas: int = 1

    @property
    def n_points(self) -> int:
        """How many bins carry a modulus."""
        return int(self.time_ps.size)

    @property
    def decades(self) -> float:
        """How many decades of time the curve spans."""
        if self.n_points < 2 or self.time_ps[0] <= 0.0:
            return 0.0
        return float(math.log10(self.time_ps[-1] / self.time_ps[0]))

    @property
    def initial_modulus_mpa(self) -> float:
        """The earliest binned modulus, which is the nearest thing to G(0).

        The nearest thing and not the same thing: the first bin sits one
        sampling interval after the strain, and the decay has already begun.
        :attr:`instant_mpa` is the reading taken before any dynamics at all.
        """
        return float(self.modulus_mpa[0]) if self.n_points else math.nan

    def youngs_modulus_mpa(
        self, poisson: float | None = None
    ) -> npt.NDArray[np.float64]:
        """``E(t) = 2 (1 + nu) G(t)``, derived rather than measured.

        The one place a Poisson ratio enters, and it is the *material's*, not
        the one the box was scaled by. Pass a measured one where there is one -
        :func:`~openmmpolymer.elasticity.poisson_ratio` fits it from an
        extension - rather than letting the default stand in for it.

        Args:
            poisson: The material's Poisson ratio. None takes the one the
                deformation imposed, which for the usual incompressible step
                is 0.5 and makes this ``3 G(t)``.

        Returns:
            Young's relaxation modulus at each time.
        """
        ratio = self.poisson if poisson is None else float(poisson)
        if not math.isfinite(ratio):
            raise AnalysisError(
                f"{self.stage} was a shear step, which imposes no lateral "
                "contraction, so there is no Poisson ratio to fall back on. "
                "Pass the material's own."
            )
        return np.asarray(2.0 * (1.0 + ratio) * self.modulus_mpa, dtype=np.float64)


@dataclass(frozen=True)
class KWWFit:
    """A stretched exponential through a relaxation modulus.

    ``G(t) = G0 exp[-(t / tau)^beta]``. One time constant and one exponent
    standing in for a whole spectrum: ``beta = 1`` is a single exponential,
    and the further below one it sits the broader the distribution underneath.

    Args:
        modulus_mpa: ``G0``, the fit extrapolated back to zero time. An
            extrapolation from the first fitted bin, not a measurement -
            :attr:`RelaxationCurve.instant_mpa` is the measurement.
        tau_ps: The KWW time constant. Poorly determined on its own, and
            strongly anti-correlated with *beta*.
        beta: The stretching exponent.
        mean_tau_ps: ``(tau / beta) Gamma(1 / beta)``, the integral of the
            decay. **This is the number to quote.** It is the combination the
            data actually determine: across noisy realisations of one truth,
            ``tau`` scatters by a factor of three and ``beta`` by fifteen per
            cent while this moves by twenty.
        residual: Root-mean-square residual, in log modulus.
        half_disagreement: How differently the first and second halves of the
            window want *beta*. Small for a stretched exponential; large for a
            curve with a plateau that one has been drawn through anyway.
        n_points: Points the fit rested on.
        window_ps: First and last time in that window.
        extrapolation_decades: How far :attr:`mean_tau_ps` runs past the end
            of the data.
        at_bound: ``"lower"``, ``"upper"`` or ``""``. The two ends are not
            alike: ``beta = 1`` is a single exponential and a legitimate
            answer, while the floor is the fit degenerating toward a power law.
        temperature_k: Carried through from the curve.
        step_strain: Likewise - a modulus outside the linear region belongs to
            that strain and to nothing more general.
        resolved: Whether this describes a decay.
    """

    modulus_mpa: float
    tau_ps: float
    beta: float
    mean_tau_ps: float
    residual: float
    half_disagreement: float
    n_points: int
    window_ps: tuple[float, float]
    extrapolation_decades: float
    at_bound: str
    temperature_k: float
    step_strain: float
    resolved: bool


@dataclass(frozen=True)
class PronyFit:
    """A discrete relaxation spectrum, on times fixed in advance.

    ``G(t) = G_inf + sum_i G_i exp(-t / tau_i)``. The times are a logarithmic
    grid rather than free parameters, which is what makes the weights a linear
    problem and therefore one numpy can solve exactly; the weights are
    constrained non-negative, which is what keeps it a relaxation spectrum
    rather than an arbitrary sum of exponentials passing through the points.

    Args:
        tau_ps: The relaxation times.
        weights_mpa: The weight on each, all non-negative and mostly zero -
            the constraint sparsifies, and what is left is the answer saying
            which decades the data constrain.
        equilibrium_mpa: ``G_inf``. Zero for a liquid, which is the right
            answer rather than a failure; zero is also what a run too short to
            see its own plateau reports, which *plateau_reached* separates.
        unrelaxed_mpa: ``G_inf + sum G_i``, the fit at zero time. An
            extrapolation off the end of the grid and not a measurement: a
            one-per-decade grid cannot follow the curvature a real decay has
            at short times, and this overshoots.
        residual_mpa: Root-mean-square residual.
        n_points: Points the fit rested on.
        n_terms: Time constants in the grid.
        n_active: How many carry a non-zero weight.
        edge_weight: Fraction of the relaxing weight on the slowest time
            constant.
        window_ps: First and last time fitted.
        plateau_reached: Whether the run outlasted its own relaxation. False
            when the slowest term carries more than :data:`MAX_EDGE_WEIGHT`,
            which means the decay was still going when the data stopped and
            ``G_inf`` is a plateau nobody saw.
        temperature_k: Carried through from the curve.
        step_strain: Likewise.
        resolved: Enough points over enough decades, some weight somewhere,
            and a plateau that was reached.
    """

    tau_ps: npt.NDArray[np.float64]
    weights_mpa: npt.NDArray[np.float64]
    equilibrium_mpa: float
    unrelaxed_mpa: float
    residual_mpa: float
    n_points: int
    n_terms: int
    n_active: int
    edge_weight: float
    window_ps: tuple[float, float]
    plateau_reached: bool
    temperature_k: float
    step_strain: float
    resolved: bool


# --------------------------------------------------------------------------
# Reading a run directory
# --------------------------------------------------------------------------


def relax_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Name every stage in a run that applied a step strain and held it.

    Found by what each stage recorded rather than by its name, as
    :func:`~openmmpolymer.trajectory.stages_holding` explains.

    Args:
        run_dir: A directory a run wrote to.

    Returns:
        The stage names, in the order the manifest records them.

    Raises:
        AnalysisError: There is no manifest, or nothing in it was a relaxation.
    """
    return stages_holding(run_dir, _ladder("segment_bin"), "a binned stress relaxation")


def _merge_bins(samples: dict[str, list[float]]) -> dict[str, npt.NDArray[np.float64]]:
    """Add several chunks' bins together, bin for bin.

    :func:`~openmmpolymer.elasticity._gather` concatenates, which is right for
    a strain ladder and would be silently wrong here: two chunks of one
    relaxation would give a curve with every shared bin in it twice, and
    nothing downstream would notice. Grouping the concatenation by bin index
    fixes that, and the count and the mean square recorded beside each mean
    make it exact rather than an average of averages over unequal samples.
    """
    index = np.asarray(samples["segment_bin"], dtype=np.int64)
    counts = np.asarray(samples["segment_samples"], dtype=np.float64)
    unique, inverse = np.unique(index, return_inverse=True)
    size = unique.size

    def pooled(key: str) -> npt.NDArray[np.float64]:
        values = np.asarray(samples[key], dtype=np.float64)
        return np.bincount(inverse, weights=counts * values, minlength=size)

    total = np.bincount(inverse, weights=counts, minlength=size)
    return {
        "bin": unique.astype(np.float64),
        "n": total,
        "time_ps": pooled("segment_relax_time_ps") / total,
        "mean": pooled("segment_stress_bar") / total,
        "mean_sq": pooled("segment_stress_sq_bar2") / total,
    }


def _standard_error(
    mean: npt.NDArray[np.float64],
    mean_sq: npt.NDArray[np.float64],
    count: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The standard error of each bin's mean, or NaN for a bin of one.

    Clamped at zero before the square root: the variance is a difference of
    two large similar numbers and can come out a hair negative.

    It is also an underestimate, and knowingly so. It assumes the readings in
    a bin are independent, and stress readings a tenth of a picosecond apart
    in a melt are not. The error bar worth believing is the one across
    replicas, which :func:`mean_curve` computes.
    """
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.sqrt(variance / count)
    return np.where(count > 1.0, error, np.nan)


def relaxation_curve(
    run_dir: str | Path, stage: str | Sequence[str] | None = None
) -> RelaxationCurve:
    """Read the relaxation modulus one step-strain pass left behind.

    Args:
        run_dir: A directory a run wrote to.
        stage: The stage or stages to read, or None for every relaxation stage
            in the manifest. Several chunks of one pass read as one curve.

    Returns:
        ``G(t)``, with the pre-strain baseline subtracted.

    Raises:
        AnalysisError: There is nothing there to read, or what is there did
            not record a relaxation.
    """
    names = relax_stages(run_dir) if stage is None else stage_names(stage)
    samples, temperature = _gather(run_dir, names)
    if "relax_plane" in samples:
        _require_stress_estimator(samples, names)
    if "segment_bin" not in samples:
        raise AnalysisError(
            f"{', '.join(names)} recorded no relaxation bins, so nothing "
            "there was a step strain that was then held."
        )

    measures = samples.get("relax_strain_measure") or []
    if not measures or abs(measures[0]) < NEGLIGIBLE:
        raise AnalysisError(
            f"{', '.join(names)} recorded no strain to divide by, so its "
            "stress cannot be turned into a modulus."
        )
    # Chunks of one pass all record the same one; a caller who has grouped two
    # different passes together has asked for a curve that does not exist.
    if max(measures) - min(measures) > NEGLIGIBLE:
        raise AnalysisError(
            f"{', '.join(names)} were strained by different amounts "
            f"({sorted(set(measures))}), so they are not chunks of one "
            "relaxation and must not be read as one curve."
        )
    measure = float(measures[0])

    merged = _merge_bins(samples)
    order = np.argsort(merged["time_ps"])
    scale = MPA_PER_BAR / measure
    baseline = _first(samples, "baseline_stress_bar", 0.0)
    zero_level_error = _standard_error(
        np.asarray([baseline]),
        np.asarray([_first(samples, "baseline_stress_sq_bar2", 0.0)]),
        np.asarray([_first(samples, "baseline_samples", 0.0)]),
    )

    instant = samples.get("instant_stress_bar") or []
    mode = "shear" if "relax_plane" in samples else "tensile"
    return RelaxationCurve(
        stage=", ".join(names),
        mode=mode,
        bin_index=merged["bin"][order].astype(np.int64),
        time_ps=merged["time_ps"][order],
        modulus_mpa=(merged["mean"][order] - baseline) * scale,
        standard_error_mpa=_standard_error(
            merged["mean"][order], merged["mean_sq"][order], merged["n"][order]
        )
        * abs(scale),
        n_samples=merged["n"][order],
        step_strain=_first(samples, "step_strain", math.nan),
        strain_measure=measure,
        temperature_k=temperature,
        poisson=_first(samples, "relax_poisson", math.nan),
        baseline_mpa=baseline * scale,
        noise_floor_mpa=float(zero_level_error[0]) * abs(scale),
        instant_mpa=(instant[0] - baseline) * scale if instant else math.nan,
    )


def _first(samples: dict[str, list[float]], key: str, missing: float) -> float:
    """The first value the stages recorded under *key*, or *missing*."""
    return float((samples.get(key) or [missing])[0])


def mean_curve(curves: Sequence[RelaxationCurve]) -> RelaxationCurve:
    """Average several replicas of one relaxation, bin for bin.

    The error bar changes meaning here, and for the better. One replica can
    only report the scatter of correlated readings inside a bin; several
    independent runs report how far they disagree with each other, which is
    the thing worth quoting - the same reason
    :data:`~openmmpolymer.mechanical.MAX_REPLICA_SPREAD` exists.

    Args:
        curves: The replicas, which must share a grid.

    Returns:
        The ensemble average, over the bins every replica populated.

    Raises:
        AnalysisError: There are no curves, or no bin they all share.
    """
    if not curves:
        raise AnalysisError("There are no relaxation curves to average.")
    if len(curves) == 1:
        return curves[0]

    common = curves[0].bin_index
    for curve in curves[1:]:
        common = np.intersect1d(common, curve.bin_index)
    if common.size == 0:
        raise AnalysisError(
            "These replicas share no bin, so they were not run on one grid "
            "and cannot be averaged. Check that they were given the same "
            "total duration and bins_per_decade."
        )

    def stack(
        pick: Callable[[RelaxationCurve], npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        return np.stack(
            [pick(curve)[np.isin(curve.bin_index, common)] for curve in curves]
        )

    moduli = stack(lambda curve: curve.modulus_mpa)
    counts = stack(lambda curve: curve.n_samples)
    first = curves[0]
    instants = [
        curve.instant_mpa for curve in curves if math.isfinite(curve.instant_mpa)
    ]
    floors = [
        curve.noise_floor_mpa
        for curve in curves
        if math.isfinite(curve.noise_floor_mpa)
    ]
    return RelaxationCurve(
        stage=" | ".join(curve.stage for curve in curves),
        mode=first.mode,
        bin_index=common,
        time_ps=stack(lambda curve: curve.time_ps).mean(axis=0),
        modulus_mpa=moduli.mean(axis=0),
        standard_error_mpa=moduli.std(axis=0, ddof=1) / math.sqrt(len(curves)),
        n_samples=counts.sum(axis=0),
        step_strain=first.step_strain,
        strain_measure=first.strain_measure,
        temperature_k=float(np.mean([curve.temperature_k for curve in curves])),
        poisson=first.poisson,
        baseline_mpa=float(np.mean([curve.baseline_mpa for curve in curves])),
        # The floors add in quadrature and then shrink with the average, which
        # is the whole reason replicas buy back the tail of the decay.
        noise_floor_mpa=(
            float(np.sqrt(np.sum(np.square(floors))) / len(curves))
            if floors
            else math.nan
        ),
        instant_mpa=float(np.mean(instants)) if instants else math.nan,
        n_replicas=len(curves),
    )


# --------------------------------------------------------------------------
# Fitting the decay
# --------------------------------------------------------------------------


def _signal_window(
    modulus_mpa: npt.NDArray[np.float64],
    error_mpa: npt.NDArray[np.float64],
    floor: float,
) -> npt.NDArray[np.bool_]:
    """The bins a fit may rest on: from the start, up to where signal is lost.

    Truncated at the end rather than filtered point by point, and that
    distinction is the whole reason this function exists. Simply dropping the
    bins where the modulus came out negative would drop the downward half of
    the noise and keep the upward half, so the tail would be biased up, the
    stretching exponent down and the time constant out - the fit would be
    describing the asymmetry of the cut rather than the polymer.

    The test runs from the end because the earliest bins are noisy for the
    opposite reason - one reading each - and the usable stretch is in the
    middle. How far above its error a bin must stand is
    :data:`SIGNAL_TO_NOISE_FLOOR`, and why.
    """
    scale = np.where(np.isfinite(error_mpa), error_mpa, 0.0)
    usable = modulus_mpa > floor * scale
    window = np.zeros(modulus_mpa.size, dtype=np.bool_)
    if not usable.any():
        return window
    last = int(np.flatnonzero(usable)[-1])
    window[: last + 1] = modulus_mpa[: last + 1] > 0.0
    return window


def _fitted(
    curve: RelaxationCurve, signal_to_noise: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[float, float]]:
    """The times and moduli a fit rests on, and the first and last time."""
    window = _signal_window(
        curve.modulus_mpa, curve.standard_error_mpa, signal_to_noise
    )
    time_ps = curve.time_ps[window]
    span = (
        (float(time_ps[0]), float(time_ps[-1]))
        if time_ps.size
        else (math.nan, math.nan)
    )
    return time_ps, curve.modulus_mpa[window], span


def _kww_search(
    time_ps: npt.NDArray[np.float64], modulus_mpa: npt.NDArray[np.float64]
) -> tuple[dict[str, float], float, str]:
    """Fit ``G = G0 exp[-(t/tau)^beta]``, which is linear in ``t^beta``.

    For any fixed ``beta``, ``ln G = ln G0 - t^beta / tau^beta``, so the fit
    is a separable search over the exponent.

    Unweighted, and that is a choice rather than an omission. Bins equally
    spaced in log time already weight the fit equally per decade, which is the
    right weighting for a relaxation spectrum. Weighting by the measured
    signal-to-noise would give a bin that happened to fluctuate low less say
    in the answer, which is the same upward bias
    :func:`_signal_window` exists to avoid, just spread out.

    Returns the parameters, the sum of squared residuals in log modulus, and
    which end of the bracket - if either - the optimum ran to.
    """
    beta, slope, intercept, total, edge = separable_fit(
        lambda beta: time_ps**beta, np.log(modulus_mpa), KWW_BETA_BRACKET
    )
    if slope >= 0.0:
        # Not decaying, so there is no time constant to report.
        return {"beta": beta}, total, edge
    return (
        {
            "beta": beta,
            "modulus_mpa": math.exp(intercept),
            "tau_ps": (-1.0 / slope) ** (1.0 / beta),
        },
        total,
        edge,
    )


def _mean_tau_ps(tau_ps: float, beta: float) -> float:
    """``(tau / beta) Gamma(1 / beta)``, the integral of the decay.

    ``math.gamma`` overflows once ``1 / beta`` reaches about 172, which the
    floor of :data:`KWW_BETA_BRACKET` keeps well out of reach.
    """
    if not math.isfinite(tau_ps):
        return math.nan
    return float(tau_ps / beta * math.gamma(1.0 / beta))


def fit_kww(
    curve: RelaxationCurve,
    *,
    min_points: int = MIN_RELAXATION_POINTS,
    signal_to_noise: float = SIGNAL_TO_NOISE_FLOOR,
) -> KWWFit:
    """Fit a stretched exponential to a relaxation modulus.

    Args:
        curve: What :func:`relaxation_curve` or :func:`mean_curve` returned.
        min_points: Fewest bins the fit may rest on.
        signal_to_noise: How far a bin must stand above its own error to be
            inside the window.

    Returns:
        The fit, whose ``resolved`` says whether to believe it.
    """
    time_ps, modulus, span = _fitted(curve, signal_to_noise)
    if time_ps.size < max(2, min_points):
        return KWWFit(
            modulus_mpa=math.nan,
            tau_ps=math.nan,
            beta=math.nan,
            mean_tau_ps=math.nan,
            residual=math.nan,
            half_disagreement=math.nan,
            n_points=int(time_ps.size),
            window_ps=span,
            extrapolation_decades=math.nan,
            at_bound="",
            temperature_k=curve.temperature_k,
            step_strain=curve.step_strain,
            resolved=False,
        )

    parameters, total, edge = _kww_search(time_ps, modulus)
    beta = parameters["beta"]
    tau = parameters.get("tau_ps", math.nan)
    modulus_0 = parameters.get("modulus_mpa", math.nan)
    mean_tau = _mean_tau_ps(tau, beta)
    residual = math.sqrt(total / time_ps.size)

    # The check elasticity.ElasticModulus makes of a slope: the two halves of a
    # genuine KWW want one exponent, and a curve with something else in it -
    # most often a plateau - wants two.
    half = time_ps.size // 2
    disagreement = math.nan
    if half >= 2:
        early, _, _ = _kww_search(time_ps[:half], modulus[:half])
        late, _, _ = _kww_search(time_ps[half:], modulus[half:])
        disagreement = abs(early["beta"] - late["beta"]) / beta

    decades = (
        max(0.0, math.log10(mean_tau / span[1]))
        if math.isfinite(mean_tau) and mean_tau > 0.0 and span[1] > 0.0
        else math.inf
    )
    resolved = bool(
        time_ps.size >= min_points
        and edge != "lower"
        and math.isfinite(tau)
        and tau > 0.0
        and math.isfinite(modulus_0)
        and modulus_0 > 0.0
        and math.log10(span[1] / span[0]) >= MIN_FITTED_DECADES
        and residual <= MAX_KWW_RESIDUAL
        and (
            not math.isfinite(disagreement) or disagreement <= MAX_KWW_HALF_DISAGREEMENT
        )
        and decades <= MAX_KWW_EXTRAPOLATION_DECADES
    )
    return KWWFit(
        modulus_mpa=modulus_0,
        tau_ps=tau,
        beta=beta,
        mean_tau_ps=mean_tau,
        residual=residual,
        half_disagreement=disagreement,
        n_points=int(time_ps.size),
        window_ps=span,
        extrapolation_decades=decades,
        at_bound=edge,
        temperature_k=curve.temperature_k,
        step_strain=curve.step_strain,
        resolved=resolved,
    )


def nnls(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    tolerance: float = 1.0e-10,
    max_iterations: int | None = None,
) -> npt.NDArray[np.float64]:
    """Solve ``min ||A x - b||`` subject to ``x >= 0``, in numpy alone.

    The Lawson-Hanson active-set algorithm. It is here because a Prony series
    with a negative coefficient is not a Prony series - it is a relaxation
    spectrum with negative weight somewhere, which no material has - and
    because plain least squares produces one most of the time: on planted data
    with realistic noise, about four draws in five came back with a negative
    coefficient, at every noise level tried. So "fit it unconstrained and
    refuse if any weight is negative" is not a fallback - it would throw away
    four runs out of five, and not the four that deserved it.

    Deterministic and terminating. Each outer round moves one index into the
    free set and no set repeats, so it stops on its own; the iteration caps
    are against floating point, not the reason it ends. Fed pure noise it
    returns zeros, which is the right shape for a caller that has to be able
    to say it found nothing.

    Args:
        design: The ``(rows, columns)`` matrix.
        target: The ``(rows,)`` right-hand side.
        tolerance: How far from zero counts as zero.
        max_iterations: Outer rounds, defaulting to three per column.

    Returns:
        The non-negative solution, ``(columns,)``.
    """
    columns = design.shape[1]
    limit = 3 * columns if max_iterations is None else max_iterations
    solution = np.zeros(columns, dtype=np.float64)
    free = np.zeros(columns, dtype=np.bool_)

    for _ in range(limit):
        gradient = design.T @ (target - design @ solution)
        pinned = ~free
        if not pinned.any():
            break
        # The pinned coefficient that most wants to move off zero.
        candidate = int(np.flatnonzero(pinned)[np.argmax(gradient[pinned])])
        if gradient[candidate] <= tolerance:
            break
        free[candidate] = True

        for _ in range(columns + 1):
            trial = np.zeros(columns, dtype=np.float64)
            unconstrained, *_ = np.linalg.lstsq(design[:, free], target, rcond=None)
            trial[free] = unconstrained
            if float(trial[free].min()) > tolerance:
                solution = trial
                break
            # Step as far towards the unconstrained answer as the first
            # coefficient it would take negative allows, then pin that one.
            blocking = free & (trial <= tolerance)
            step = float(
                np.min(
                    solution[blocking]
                    / np.maximum(solution[blocking] - trial[blocking], NEGLIGIBLE)
                )
            )
            solution = solution + step * (trial - solution)
            free &= solution > tolerance
            solution[~free] = 0.0
            if not free.any():
                break
    return solution


def _prony_design(
    time_ps: npt.NDArray[np.float64], tau_ps: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The linear problem a fixed set of relaxation times turns G(t) into.

    ``G_inf`` is the last column, a column of ones. That is not a trick to
    make it fit: it is the ``tau -> infinity`` limit of the same family, and
    putting it in the same matrix is what makes the non-negativity constraint
    apply to it too - which it should, since a relaxed modulus cannot be
    negative.
    """
    return np.column_stack(
        [np.exp(-time_ps[:, None] / tau_ps), np.ones(time_ps.size, dtype=np.float64)]
    )


def prony_times_ps(
    first_ps: float, last_ps: float, per_decade: int = PRONY_PER_DECADE
) -> npt.NDArray[np.float64]:
    """The relaxation times a Prony fit uses, fixed rather than fitted.

    Fixing them is what makes the weights linear and so exactly solvable. The
    top of the grid is the choice that matters, and it stops at
    :data:`PRONY_SLOWEST_FRACTION` of the window so that ``G_inf`` stays
    separable from the slowest mode.

    Args:
        first_ps: The earliest time fitted.
        last_ps: The latest.
        per_decade: Times per decade.

    Returns:
        The grid, ascending.
    """
    top = max(first_ps, last_ps * PRONY_SLOWEST_FRACTION)
    if top <= first_ps:
        return np.asarray([first_ps], dtype=np.float64)
    count = max(2, round(per_decade * math.log10(top / first_ps)) + 1)
    return np.geomspace(first_ps, top, count)


def fit_prony(
    curve: RelaxationCurve,
    *,
    min_points: int = MIN_RELAXATION_POINTS,
    per_decade: int = PRONY_PER_DECADE,
    signal_to_noise: float = SIGNAL_TO_NOISE_FLOOR,
) -> PronyFit:
    """Fit a generalised Maxwell model to a relaxation modulus.

    Args:
        curve: What :func:`relaxation_curve` or :func:`mean_curve` returned.
        min_points: Fewest bins the fit may rest on.
        per_decade: Relaxation times per decade in the grid.
        signal_to_noise: How far a bin must stand above its own error to be
            inside the window.

    Returns:
        The spectrum, whose ``resolved`` says whether to believe it.
    """
    time_ps, modulus, span = _fitted(curve, signal_to_noise)
    if time_ps.size < max(2, min_points):
        return PronyFit(
            tau_ps=np.zeros(0, dtype=np.float64),
            weights_mpa=np.zeros(0, dtype=np.float64),
            equilibrium_mpa=math.nan,
            unrelaxed_mpa=math.nan,
            residual_mpa=math.nan,
            n_points=int(time_ps.size),
            n_terms=0,
            n_active=0,
            edge_weight=math.nan,
            window_ps=span,
            plateau_reached=False,
            temperature_k=curve.temperature_k,
            step_strain=curve.step_strain,
            resolved=False,
        )

    tau_ps = prony_times_ps(span[0], span[1], per_decade)
    design = _prony_design(time_ps, tau_ps)
    coefficients = nnls(design, modulus)
    weights, equilibrium = coefficients[:-1], float(coefficients[-1])
    residual = modulus - design @ coefficients

    relaxing = float(weights.sum())
    edge = float(weights[-1] / relaxing) if relaxing > NEGLIGIBLE else math.nan
    # A spectrum whose slowest term carries the weight is saying the decay was
    # still going when the data stopped, so whatever G_inf came out is a
    # plateau nobody watched the curve reach.
    plateau = bool(math.isfinite(edge) and edge <= MAX_EDGE_WEIGHT)
    active = int(np.count_nonzero(weights > NEGLIGIBLE))

    resolved = bool(
        time_ps.size >= min_points
        and active >= 1
        and math.log10(span[1] / span[0]) >= MIN_FITTED_DECADES
        and plateau
    )
    return PronyFit(
        tau_ps=tau_ps,
        weights_mpa=weights,
        equilibrium_mpa=equilibrium,
        unrelaxed_mpa=equilibrium + relaxing,
        residual_mpa=float(math.sqrt(float(residual @ residual) / time_ps.size)),
        n_points=int(time_ps.size),
        n_terms=int(tau_ps.size),
        n_active=active,
        edge_weight=edge,
        window_ps=span,
        plateau_reached=plateau,
        temperature_k=curve.temperature_k,
        step_strain=curve.step_strain,
        resolved=resolved,
    )
