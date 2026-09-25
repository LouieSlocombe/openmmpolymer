"""Figures for the analysis results, built without pyplot.

Every function here constructs a ``matplotlib.figure.Figure`` directly and
returns it. That is not a stylistic preference. Importing ``pyplot`` selects a
backend at import time, which in a headless job means guessing; it registers
every figure in a process-global table that leaks unless the caller remembers
to close them; and it puts global state into a library that has none anywhere
else. A bare ``Figure`` needs no backend at all until something asks it to
render, and is garbage-collected with the variable holding it.

So these return a figure and write nothing. Saving it, showing it or embedding
it is the caller's business - ``figure.savefig("density.png")`` - and the code
in this package that writes files stays the code that runs simulations and the
driver that reports on them. For the same reason no function takes an ``ax``
to draw into: each one owns a multi-panel layout, and passing axes in would
break that while inviting ``pyplot`` back.

Where a result carries a caveat, the caveat is drawn. The rate a number was
measured at goes in the title, because a transition or a modulus read off a
simulation is not comparable with an experiment run orders of magnitude
slower. What the data cannot reach - below a structure factor's resolution
floor, past the end of a chain or a run, across the decades an extrapolation
spans, under a relaxation's own noise floor - is shaded. And a fit the
analysis did not resolve says so in its label.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._fitting import TINY
from .conformation import (
    DECORRELATION_THRESHOLD,
    ConformationSeries,
    EndToEndRelaxation,
    MeanSquaredDisplacement,
    PersistenceLength,
)
from .correlations import RadialDistribution, StructureFactor
from .elasticity import ElasticModulus, PoissonRatio, StressStrain
from .relaxation import KWWFit, PronyFit, RelaxationCurve
from .timeseries import (
    CoolingRateExtrapolation,
    Equilibration,
    GlassTransition,
    QuenchCurve,
    StateData,
)

if TYPE_CHECKING:
    from .strain_rate import StrainRateExtrapolation
    from .strength import BreakingStrength, ElongationAtBreak, YieldStrength

#: Figure size in inches. Wide enough for a four-panel column to stay legible
#: at a report's width.
FIGURE_SIZE_IN = (7.0, 4.5)

#: Dots per inch. Enough for a screen and for print, without the file size of
#: a vector-free 300.
FIGURE_DPI = 150

#: Colour for the thing being measured, for what the analysis drew on top of
#: it, for the reference lines behind it, and for a second, independent
#: estimate beside the first. Named rather than repeated so a figure stays
#: one figure.
_DATA_COLOUR = "#1f4e79"
_GUIDE_COLOUR = "#b03a2e"
_REFERENCE_COLOUR = "#7f8c8d"
_ALTERNATIVE_COLOUR = "#6c3483"


# --------------------------------------------------------------------------
# Thermal: the state data and the quench
# --------------------------------------------------------------------------


def plot_state_data(data: StateData, *, settled: Equilibration | None = None) -> Any:
    """Plot a stage's temperature, density, energy and volume against time.

    Args:
        data: A stage's state data, from
            :func:`~openmmpolymer.timeseries.read_state_data`.
        settled: Where the series settled, from
            :func:`~openmmpolymer.timeseries.equilibration`. Drawn as the
            point discarded up to, with the retained mean across the rest.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    panels = (
        ("Temperature (K)", data.temperature_k),
        ("Density (g/cm3)", data.density_g_cm3),
        ("Potential energy (kJ/mol)", data.potential_energy_kj_mol),
        ("Box volume (nm3)", data.volume_nm3),
    )
    figure, axes = _figure(len(panels), 1, height_per_row=1.3)
    for axis, (label, values) in zip(axes, panels, strict=True):
        _measured(axis, data.time_ps, values, markersize=None)
        axis.set_ylabel(label, fontsize=8)
        if settled is not None:
            _settling(axis, settled)
            _level(axis, float(settled.window(values).mean()), linestyle=":")
    axes[-1].set_xlabel("Time (ps)")
    _title(axes[0], _state_title(data, settled))
    return figure


def plot_quench_curve(
    curve: QuenchCurve, *, transition: GlassTransition | None = None
) -> Any:
    """Plot specific volume against temperature, with the two fitted branches.

    Args:
        curve: A curve from :func:`~openmmpolymer.timeseries.quench_curve`.
        transition: A fit from
            :func:`~openmmpolymer.timeseries.glass_transition`. Its branches
            are drawn and its temperature marked.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axis = _panel()
    _measured(axis, curve.temperature_k, curve.specific_volume_cm3_g, label="measured")
    if transition is not None:
        span = np.asarray(
            [curve.temperature_k.min(), curve.temperature_k.max()], dtype=np.float64
        )
        for slope, name in (
            (transition.glass_expansion_per_k, "glass"),
            (transition.melt_expansion_per_k, "melt"),
        ):
            offset = _branch_offset(transition, slope)
            _guide(axis, span, offset + slope * span, label=f"{name} fit")
        axis.axvline(
            transition.temperature_k,
            color=_GUIDE_COLOUR,
            linewidth=1.0,
            label=f"Tg = {transition.temperature_k:.0f} K",
        )
        axis.set_ylim(
            float(curve.specific_volume_cm3_g.min()) * 0.995,
            float(curve.specific_volume_cm3_g.max()) * 1.005,
        )
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel("Specific volume (cm3/g)")
    _title(axis, _quench_title(curve, transition))
    _legend(axis)
    return figure


def plot_cooling_rate(extrapolation: CoolingRateExtrapolation) -> Any:
    """Plot the transition against cooling rate, and where the fit points.

    The fitted curve is drawn all the way from the target rate to the fastest
    measurement, and the region with no data in it is shaded, so the gap the
    number was carried across is something the eye can see rather than
    something the caption claims.

    Args:
        extrapolation: A fit from
            :func:`~openmmpolymer.timeseries.cooling_rate_extrapolation`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axis = _panel()
    rates = extrapolation.cooling_rate_k_per_ns
    target = extrapolation.target_rate_k_per_ns
    lowest = min(float(rates.min()), target)

    _span(axis, lowest, float(rates.min()), _GUIDE_COLOUR, alpha=0.15)
    span = np.geomspace(lowest, float(rates.max()), 200)
    _guide(
        axis,
        span,
        extrapolation.predict(span),
        linewidth=0.9,
        label=f"{extrapolation.form} fit",
    )
    _measured(
        axis,
        rates,
        extrapolation.transition_k,
        markersize=4.0,
        linestyle="none",
        label="measured",
    )
    _point(
        axis,
        target,
        extrapolation.temperature_k,
        "*",
        9.0,
        label=f"{extrapolation.temperature_k:.0f} K at {target:.3g} K/ns",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Cooling rate (K/ns)")
    axis.set_ylabel("Transition temperature (K)")
    _title(axis, _cooling_rate_title(extrapolation))
    _legend(axis)
    return figure


def _state_title(data: StateData, settled: Equilibration | None) -> str:
    """A title naming the stage and what settling was found."""
    stage = data.stage or "state data"
    if settled is None:
        return f"{stage}: {data.n_rows} rows over {data.duration_ps:.0f} ps"
    verdict = "settled" if settled.equilibrated else "still drifting"
    return (
        f"{stage}: {verdict} from {settled.start_ps:.0f} ps, "
        f"{settled.n_independent_samples:.0f} independent samples"
    )


def _quench_title(curve: QuenchCurve, transition: GlassTransition | None) -> str:
    """A title carrying the cooling rate, so the caveat travels with the plot.

    The expansion coefficients go on a second line rather than into the
    legend, which names the two fitted branches and nothing else.
    """
    rate = (
        "cooling rate unknown"
        if curve.cooling_rate_k_per_ns is None
        else f"cooled at {curve.cooling_rate_k_per_ns:.0f} K/ns"
    )
    if transition is None:
        return f"{curve.stage}: {curve.n_points} temperatures, {rate}"
    if not transition.resolved:
        return f"{curve.stage}: no clear transition, {rate}"
    return (
        f"{curve.stage}: break at {transition.temperature_k:.0f} K, {rate}\n"
        f"aV {transition.melt_expansivity_per_k:.2e} (melt) against "
        f"{transition.glass_expansivity_per_k:.2e} (glass) per K"
    )


def _cooling_rate_title(extrapolation: CoolingRateExtrapolation) -> str:
    """A title saying how far the number was carried past the measurements."""
    verdict = "" if extrapolation.resolved else ", not resolved"
    return (
        f"{extrapolation.form}: Tg = {extrapolation.temperature_k:.0f} K at "
        f"{extrapolation.target_rate_k_per_ns:.3g} K/ns, extrapolated "
        f"{extrapolation.extrapolation_decades:.1f} decades{verdict}\n"
        f"{extrapolation.sensitivity_k_per_decade:.1f} K per decade over "
        f"{extrapolation.n_rates} measured rates"
    )


def _branch_offset(transition: GlassTransition, slope: float) -> float:
    """The intercept of a fitted branch, from its slope and the crossing.

    Both branches pass through the transition by construction, so the volume
    there plus a slope fixes each line.
    """
    return transition.specific_volume_cm3_g - slope * transition.temperature_k


# --------------------------------------------------------------------------
# Structure and dynamics
# --------------------------------------------------------------------------


def plot_conformation(series: ConformationSeries) -> Any:
    """Plot chain dimensions against time.

    Args:
        series: A series from
            :func:`~openmmpolymer.conformation.chain_conformation`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axes = _figure(2, 1, height_per_row=1.8)
    _measured(
        axes[0],
        series.time_ps,
        series.mean_squared_end_to_end_nm2,
        markersize=None,
        label="measured",
    )
    expected = _expected_square(series)
    if expected is not None:
        _level(axes[0], expected, label="expected from C-infinity")
    axes[0].set_ylabel("<R^2> (nm2)", fontsize=8)
    _legend(axes[0])

    _measured(
        axes[1], series.time_ps, series.mean_radius_of_gyration_nm, markersize=None
    )
    axes[1].set_ylabel("Rg (nm)", fontsize=8)
    axes[1].set_xlabel("Time (ps)")
    if series.settled is not None:
        for axis in axes:
            _settling(axis, series.settled)
    _title(axes[0], _conformation_title(series))
    return figure


def plot_correlations(
    distribution: RadialDistribution, *, structure: StructureFactor | None = None
) -> Any:
    """Plot the pair distribution, and the structure factor beside it.

    Args:
        distribution: A distribution from
            :func:`~openmmpolymer.correlations.radial_distribution`.
        structure: A structure factor from
            :func:`~openmmpolymer.correlations.structure_factor`, drawn on a
            second panel when given.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axes = _figure(1, 1 if structure is None else 2)
    axis = axes[0]
    _measured(axis, distribution.r_nm, distribution.g_r, markersize=None, linewidth=1.0)
    _level(axis, 1.0)
    axis.set_xlabel("r (nm)")
    axis.set_ylabel("g(r), intermolecular")
    _title(
        axis, f"{distribution.n_pairs:,} pairs over {distribution.n_frames} frame(s)"
    )

    if structure is not None:
        second = axes[1]
        _measured(
            second, structure.q_per_nm, structure.s_q, markersize=None, linewidth=1.0
        )
        _span(second, 0.0, structure.q_min_per_nm, _GUIDE_COLOUR, alpha=0.15)
        second.set_xlabel("q (1/nm)")
        second.set_ylabel("S(q)")
        _title(
            second, f"below {structure.q_min_per_nm:.1f} /nm the cell cannot resolve"
        )
    return figure


def plot_dynamics(
    msd: MeanSquaredDisplacement, *, relaxation: EndToEndRelaxation | None = None
) -> Any:
    """Plot the mean-squared displacement, and the end-to-end decay beside it.

    A slope-one guide is drawn through the displacement, so a sub-diffusive
    curve looks sub-diffusive.

    Args:
        msd: A displacement curve from
            :func:`~openmmpolymer.conformation.centre_of_mass_msd`.
        relaxation: A correlation function from
            :func:`~openmmpolymer.conformation.end_to_end_relaxation`, drawn
            on a second panel when given.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axes = _figure(1, 1 if relaxation is None else 2)
    axis = axes[0]
    usable = (msd.lag_ps > 0.0) & (msd.msd_nm2 > 0.0)
    _measured(
        axis,
        msd.lag_ps[usable],
        msd.msd_nm2[usable],
        markersize=None,
        linewidth=1.0,
        label=f"measured (slope {msd.log_slope:.2f})",
    )
    if usable.any():
        lags = msd.lag_ps[usable]
        reference = msd.msd_nm2[usable][-1] * (lags / lags[-1])
        _guide(axis, lags, reference, colour=_GUIDE_COLOUR, label="slope 1 (diffusive)")
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlabel("Lag (ps)")
    axis.set_ylabel("Centre-of-mass MSD (nm2)")
    _title(axis, _msd_title(msd))
    _legend(axis)

    if relaxation is not None:
        second = axes[1]
        _measured(
            second,
            relaxation.lag_ps,
            relaxation.correlation,
            markersize=None,
            linewidth=1.0,
        )
        _level(second, DECORRELATION_THRESHOLD, _GUIDE_COLOUR)
        second.set_xlabel("Lag (ps)")
        second.set_ylabel("End-to-end correlation")
        _title(second, _relaxation_title(relaxation))
    return figure


def plot_persistence(length: PersistenceLength) -> Any:
    """Plot the bond-direction correlation along the backbone, and its decay.

    The 1/e line is where the persistence length is read off, so it is drawn.
    A curve that never gets down to it within the chain has been fitted and
    then extrapolated past the end of the molecule, and that region is shaded:
    a persistence length longer than the chain it was measured on should look
    like one.

    Args:
        length: A measurement from
            :func:`~openmmpolymer.conformation.persistence_length`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axis = _panel()
    separation = length.separation
    last = float(separation[-1]) if separation.size else float(length.n_bonds)
    fitted = (
        math.isfinite(length.persistence_length_nm)
        and length.persistence_length_nm > 0.0
        and length.bond_length_nm > 0.0
    )
    x_max = last
    if not length.decayed:
        x_max = 1.5 * last
        if fitted:
            reach = 1.2 * length.persistence_length_nm / length.bond_length_nm
            x_max = max(x_max, min(reach, 10.0 * last))

    _measured(axis, separation, length.correlation, markersize=3.0, label="measured")
    _level(axis, DECORRELATION_THRESHOLD, _GUIDE_COLOUR, label="1/e")
    if fitted:
        span = np.linspace(0.0, x_max, 200)
        _guide(
            axis,
            span,
            np.exp(-span * length.bond_length_nm / length.persistence_length_nm),
            label=f"exp(-s l_b / l_p), l_p = {length.persistence_length_nm:.2f} nm",
        )
    if not length.decayed:
        _span(axis, last, x_max, _REFERENCE_COLOUR, label="past the end of the chain")
        axis.set_xlim(0.0, x_max)
    axis.set_xlabel("Separation (bonds)")
    axis.set_ylabel("<cos theta(s)>")
    _title(axis, _persistence_title(length))
    _legend(axis)
    return figure


def _conformation_title(series: ConformationSeries) -> str:
    """A title naming what was averaged and whether it had stopped moving."""
    scope = f"{series.n_chains} chains over {series.n_frames} frame(s)"
    if series.settled is None:
        return f"{series.stage}: {scope}, single snapshot"
    moving = "settled" if series.settled.equilibrated else "still moving"
    return f"{series.stage}: {scope}, <R^2> {moving}"


def _msd_title(msd: MeanSquaredDisplacement) -> str:
    """A title that says whether a diffusion coefficient was earned."""
    if msd.diffusion_coefficient_cm2_s is None:
        return f"not diffusive (slope {msd.log_slope:.2f}), so no diffusion coefficient"
    return f"D = {msd.diffusion_coefficient_cm2_s:.2e} cm2/s"


def _relaxation_title(relaxation: EndToEndRelaxation) -> str:
    """A title that says whether the chains relaxed, or only that they did not."""
    if relaxation.relaxation_time_ps is None:
        return f"not decorrelated in {relaxation.trajectory_ps:.0f} ps"
    return f"relaxes in {relaxation.relaxation_time_ps:.0f} ps"


def _persistence_title(length: PersistenceLength) -> str:
    """A title that says whether the length was measured or extrapolated."""
    contour = f"{length.contour_length_nm:.2f} nm contour"
    fitted = length.persistence_length_nm
    if math.isinf(fitted):
        return f"no decay along a {contour} ({length.n_bonds} bonds): rod-like"
    if not fitted > 0.0:
        return f"no persistence length could be fitted over a {contour}"
    if not length.decayed:
        return (
            f"l_p = {fitted:.2f} nm, extrapolated: not decayed to 1/e within "
            f"the {contour}"
        )
    return (
        f"l_p = {fitted:.2f} nm from {length.n_bonds} bonds of "
        f"{length.bond_length_nm:.3f} nm ({contour})"
    )


def _expected_square(series: ConformationSeries) -> float | None:
    """``<R^2>`` implied by the expected characteristic ratio, for a guide line.

    ``C = <R^2> / (n l^2)``, and the measurement reports both the measured C
    and the expected one against the same bond length, so the ratio of the
    two scales the measured mean square.
    """
    measured = series.mean.characteristic_ratio
    if abs(measured) < TINY:
        return None
    scale = series.mean.expected_characteristic_ratio / measured
    return series.mean.mean_squared_end_to_end_nm2 * scale


# --------------------------------------------------------------------------
# Mechanical: stress and strain, the moduli, and strength
# --------------------------------------------------------------------------


def plot_stress_strain(
    curve: StressStrain,
    *,
    fit: ElasticModulus | None = None,
    poisson: PoissonRatio | None = None,
) -> Any:
    """Plot stress and transverse strain against axial strain.

    Two panels. The upper one is the stress-strain curve with the fitted
    modulus drawn through it and its window shaded, so a fit that ran past
    the linear range is visible as one. The lower is the lateral response
    Poisson's ratio comes from, which is where a cell that never relaxed
    sideways shows up.

    Args:
        curve: A curve from :func:`~openmmpolymer.elasticity.stress_strain`
            or :func:`~openmmpolymer.elasticity.load_curve`.
        fit: A fit from :func:`~openmmpolymer.elasticity.youngs_modulus`.
            Its line is drawn and its window shaded.
        poisson: A fit from :func:`~openmmpolymer.elasticity.poisson_ratio`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, (stress, lateral) = _figure(2, 1, height_per_row=2.6)
    span = np.asarray([0.0, float(curve.strain.max())], dtype=np.float64)

    _measured(stress, curve.strain, curve.tensile_stress_mpa, label="measured")
    if fit is not None and np.isfinite(fit.modulus_mpa):
        _guide(
            stress,
            span,
            fit.intercept_mpa + fit.modulus_mpa * span,
            colour=_GUIDE_COLOUR,
            label=f"E = {fit.modulus_mpa:.0f} MPa{_unresolved(fit.resolved)}",
        )
        _span(stress, 0.0, fit.strain_limit, _REFERENCE_COLOUR)
    _zero_line(stress)
    stress.set_ylabel("Tensile stress (MPa)")
    _title(stress, _stress_title(curve, fit))
    _legend(stress)

    _measured(lateral, curve.strain, curve.transverse_strain, label="measured")
    if poisson is not None and np.isfinite(poisson.ratio):
        _guide(
            lateral,
            span,
            -poisson.ratio * span,
            colour=_GUIDE_COLOUR,
            label=f"nu = {poisson.ratio:.3f}{_unresolved(poisson.resolved)}",
        )
    _zero_line(lateral)
    lateral.set_xlabel(_strain_axis(curve))
    lateral.set_ylabel("Mean transverse strain")
    _legend(lateral)
    return figure


def plot_moduli(report: Any) -> Any:
    """Plot the measured elastic constants against the ones E and nu imply.

    The bars are what was measured; the markers are what Young's modulus and
    Poisson's ratio say ``K`` and ``G`` have to be for an isotropic solid.
    The figure exists for the gap between them, which is the one check here
    that is not a straight line through points someone chose the ends of. An
    unresolved constant is drawn hollow, so a report full of numbers the
    noise does not support looks like one.

    Args:
        report: A report from
            :func:`~openmmpolymer.mechanical.analyse_mechanics`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axis = _panel()

    labels: list[str] = []
    values: list[float] = []
    resolved: list[bool] = []
    errors: list[float] = []
    for label, fit, error in (
        ("E", report.youngs, report.replica_spread_mpa),
        ("K", report.bulk, None),
        ("G", report.shear, None),
    ):
        if fit is None:
            continue
        labels.append(label)
        values.append(float(fit.modulus_mpa))
        resolved.append(bool(fit.resolved))
        if label == "K" and math.isfinite(fit.standard_error_mpa):
            error = fit.standard_error_mpa
        errors.append(0.0 if error is None else float(error))

    positions = np.arange(len(labels), dtype=np.float64)
    axis.bar(
        positions,
        values,
        yerr=errors if any(errors) else None,
        capsize=3,
        width=0.55,
        color=[_DATA_COLOUR if ok else "none" for ok in resolved],
        edgecolor=_DATA_COLOUR,
        linewidth=1.0,
        label="measured",
    )
    check = report.consistency
    if check is not None:
        for label, implied in (
            ("K", check.bulk_implied_mpa),
            ("G", check.shear_implied_mpa),
        ):
            if label in labels and np.isfinite(implied):
                _point(
                    axis,
                    labels.index(label),
                    implied,
                    "_",
                    22,
                    markeredgewidth=2.0,
                    label="implied by E and nu"
                    if label == labels[min(len(labels) - 1, 1)]
                    else None,
                )
    if report.load_modulus is not None and "E" in labels:
        _point(
            axis,
            labels.index("E"),
            report.load_modulus.modulus_mpa,
            "x",
            8,
            colour=_ALTERNATIVE_COLOUR,
            label="constant-stress cross-check",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Modulus (MPa)")
    _title(axis, _moduli_title(report))
    handles, names = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, names, fontsize=7, frameon=False)
    return figure


def plot_strain_rate(extrapolation: StrainRateExtrapolation) -> Any:
    """Plot measured Young's moduli and a finite-rate empirical estimate.

    Error bars show one standard error. The unsampled interval between the
    measurements and target is shaded, including when the target lies above
    the sampled rates. The title retains the temperature, elastic fit window
    and resolution verdict so that a long extrapolation stays visible.

    Args:
        extrapolation: A fit from
            :func:`~openmmpolymer.strain_rate.strain_rate_extrapolation`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axis = _panel()
    rates = extrapolation.strain_rate_per_ns
    target = extrapolation.target_rate_per_ns
    minimum = float(rates.min())
    maximum = float(rates.max())
    if target < minimum or target > maximum:
        boundary = minimum if target < minimum else maximum
        _span(
            axis,
            min(target, boundary),
            max(target, boundary),
            _GUIDE_COLOUR,
            alpha=0.15,
            label="extrapolated interval",
        )
    span = np.geomspace(min(target, minimum), max(target, maximum), 200)
    predicted = extrapolation.predict(span)
    _guide(
        axis,
        span,
        np.where(np.isfinite(predicted), predicted, np.nan),
        linewidth=0.9,
        label=f"{extrapolation.form} fit",
    )
    axis.errorbar(
        rates,
        extrapolation.moduli_mpa,
        yerr=extrapolation.standard_errors_mpa,
        marker="o",
        markersize=4.0,
        linestyle="none",
        elinewidth=0.9,
        capsize=2,
        color=_DATA_COLOUR,
        label="measured (1 SE)",
    )
    if math.isfinite(extrapolation.modulus_mpa):
        target_error = extrapolation.standard_error_mpa
        has_error = math.isfinite(target_error)
        error_label = "1 SE" if has_error else "SE unavailable"
        axis.errorbar(
            [target],
            [extrapolation.modulus_mpa],
            yerr=[target_error] if has_error else None,
            marker="*",
            markersize=9.0,
            linestyle="none",
            elinewidth=0.9,
            capsize=2,
            color=_GUIDE_COLOUR,
            label=(
                f"target = {extrapolation.modulus_mpa:.1f} MPa "
                f"at {target:.3g} strain/ns ({error_label})"
            ),
        )
    else:
        axis.axvline(
            target,
            color=_GUIDE_COLOUR,
            linewidth=0.9,
            label=f"target estimate not finite at {target:.3g} strain/ns",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Strain rate (strain/ns)")
    axis.set_ylabel("Young's modulus (MPa)")
    verdict = "resolved" if extrapolation.resolved else "not resolved"
    _title(
        axis,
        f"{extrapolation.temperature_k:.0f} K, "
        f"elastic strain <= {extrapolation.strain_limit:g}\n"
        f"{extrapolation.form}: {extrapolation.extrapolation_decades:.1f} "
        f"decades extrapolated, {verdict}",
    )
    _legend(axis)
    return figure


def plot_breaking_strength(curve: StressStrain, result: BreakingStrength) -> Any:
    """Plot the nominal tensile response, its peak and a resolved stress drop.

    The nominal stress includes the measured change in lateral area. A
    sampled maximum is always labelled as a sampled maximum; it is reported
    as an apparent tensile strength only when the analysis resolved the
    subsequent sustained loss of stress. The shaded failure bracket shows
    the sampling interval, rather than suggesting a precise rupture strain.

    Args:
        curve: The stress-strain curve analysed for strength.
        result: Its result from :func:`~openmmpolymer.strength.breaking_strength`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    return _failure_figure(
        curve,
        result,
        percent=False,
        peak_colour=_GUIDE_COLOUR,
        bracket=result.failure_bracket,
        bracket_label="failure strain bracket",
        failure=(result.failure_strain, result.failure_stress_mpa),
        failure_label="sustained stress drop",
        verdict=(
            f"Apparent tensile strength = {result.strength_mpa:.1f} MPa"
            if result.resolved and result.strength_mpa is not None
            else "Apparent tensile strength not resolved"
        ),
    )


def plot_elongation_at_break(curve: StressStrain, result: ElongationAtBreak) -> Any:
    """Plot apparent elongation at break on the nominal tensile response.

    The horizontal axis and the break bracket use percent engineering strain.
    The marker identifies the first sampled hold in a confirmed terminal loss
    of stress, rather than the strain at peak stress, which is drawn as
    context rather than as the answer. Its shaded bracket shows the sampling
    interval, not a confidence interval or an interpolated crack.

    Args:
        curve: The tensile curve analysed for elongation at break.
        result: Its result from
            :func:`~openmmpolymer.strength.elongation_at_break`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    return _failure_figure(
        curve,
        result,
        percent=True,
        peak_colour=_REFERENCE_COLOUR,
        bracket=result.break_bracket,
        bracket_label="break elongation bracket",
        failure=(result.elongation_percent, result.break_stress_mpa),
        failure_label="onset of sustained stress drop",
        verdict=(
            f"Apparent elongation at break = {result.elongation_percent:.1f}%"
            if result.resolved and result.elongation_percent is not None
            else "Apparent elongation at break not resolved"
        ),
    )


def plot_yield_strength(curve: StressStrain, result: YieldStrength) -> Any:
    """Plot an offset yield construction on the nominal tensile response.

    The fit window and the offset line show how the criterion was chosen.
    A resolved intersection is interpolated between samples; its shaded
    bracket shows the sampling interval, not a confidence interval. An
    unresolved result is labelled without drawing a yield-strength marker.

    Args:
        curve: The stress-strain curve analysed for yield strength.
        result: Its result from :func:`~openmmpolymer.strength.yield_strength`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axis = _nominal_stress_figure(curve.strain, result.nominal_stress_mpa)
    fit_span = np.asarray(
        [result.fit_min_strain, result.fit_max_strain], dtype=np.float64
    )
    _span(
        axis,
        result.fit_min_strain,
        result.fit_max_strain,
        _REFERENCE_COLOUR,
        label="elastic fit window",
    )
    offset_label = f"{100.0 * result.offset_strain:g}% offset"
    if (
        result.modulus_mpa is not None
        and result.intercept_mpa is not None
        and math.isfinite(result.modulus_mpa)
        and math.isfinite(result.intercept_mpa)
    ):
        _guide(
            axis,
            fit_span,
            result.modulus_mpa * fit_span + result.intercept_mpa,
            linewidth=1.0,
            label=(
                f"elastic fit: E = {result.modulus_mpa:.1f} MPa"
                f"{_unresolved(result.fit_resolved)}"
            ),
        )
        span = np.asarray(
            [float(curve.strain.min()), float(curve.strain.max())], dtype=np.float64
        )
        _guide(
            axis,
            span,
            result.modulus_mpa * (span - result.offset_strain) + result.intercept_mpa,
            colour=_GUIDE_COLOUR,
            linewidth=1.0,
            label=f"{offset_label} line",
        )
    if result.resolved:
        if result.yield_bracket is not None:
            _span(
                axis, *result.yield_bracket, _GUIDE_COLOUR, label="yield strain bracket"
            )
        if result.yield_strain is not None and result.strength_mpa is not None:
            _point(
                axis,
                result.yield_strain,
                result.strength_mpa,
                "*",
                10,
                label="interpolated offset intersection",
            )

    # Extending the elastic line to the final strain can give stresses far
    # above the measured response. Keep that extrapolation from setting the
    # scale and hiding the curve whose intersection is being reported.
    low = min(0.0, float(result.nominal_stress_mpa.min()))
    high = max(0.0, float(result.nominal_stress_mpa.max()))
    margin = max(high - low, 1.0) * 0.05
    axis.set_ylim(low - margin, high + margin)
    _zero_line(axis)
    axis.set_xlabel(_strain_axis(curve))
    axis.set_ylabel("Nominal tensile stress (MPa)")
    verdict = (
        f"{offset_label} yield strength = {result.strength_mpa:.1f} MPa"
        if result.resolved and result.strength_mpa is not None
        else f"{offset_label} yield strength not resolved"
    )
    _title(axis, _strength_title(curve, result, verdict))
    _legend(axis)
    return figure


def _failure_figure(
    curve: StressStrain,
    result: BreakingStrength | ElongationAtBreak,
    *,
    percent: bool,
    peak_colour: str,
    bracket: tuple[float, float] | None,
    bracket_label: str,
    failure: tuple[float | None, float | None],
    failure_label: str,
    verdict: str,
) -> Any:
    """The nominal response, its sampled peak, and a resolved loss of stress.

    What a breaking strength and an elongation at break are both read off:
    the peak is marked as a sample, and the sustained drop after it only when
    the analysis resolved one, with the holds either side of it shaded.
    *failure* is already in the units the axis is drawn in.
    """
    scale = 100.0 if percent else 1.0
    figure, axis = _nominal_stress_figure(
        scale * curve.strain, result.nominal_stress_mpa
    )
    _point(
        axis,
        scale * result.strain_at_peak,
        result.peak_stress_mpa,
        "*",
        10,
        colour=peak_colour,
        label=f"sampled peak = {result.peak_stress_mpa:.1f} MPa",
    )
    if result.resolved:
        if bracket is not None:
            _span(
                axis,
                scale * bracket[0],
                scale * bracket[1],
                _GUIDE_COLOUR,
                label=bracket_label,
            )
        strain, stress = failure
        if strain is not None and stress is not None:
            _point(axis, strain, stress, "D", 5, label=failure_label)
    _zero_line(axis)
    axis.set_xlabel(
        _strain_axis(curve, "elongation", " (%)") if percent else _strain_axis(curve)
    )
    axis.set_ylabel("Nominal tensile stress (MPa)")
    _title(axis, _strength_title(curve, result, verdict))
    _legend(axis)
    return figure


def _nominal_stress_figure(
    strain: npt.NDArray[np.float64], stress_mpa: npt.NDArray[np.float64]
) -> tuple[Any, Any]:
    """Start a tensile-strength figure with its measured nominal response."""
    figure, axis = _panel()
    _measured(axis, strain, stress_mpa, label="nominal tensile stress")
    return figure, axis


def _strength_title(
    curve: StressStrain,
    result: BreakingStrength | ElongationAtBreak | YieldStrength,
    verdict: str,
) -> str:
    """Keep the source, loading conditions and verdict with each strength plot."""
    chunks = curve.stage.split(", ")
    stage = chunks[0]
    if len(chunks) > 1:
        stage += f" (+{len(chunks) - 1} chunks)"
    return (
        f"{stage}: {result.temperature_k:.0f} K, "
        f"{_strain_rate_label(result.strain_rate_per_ns)}\n{verdict}"
    )


def _strain_rate_label(rate_per_ns: float | None) -> str:
    """Describe the measured rate, including when it was not recorded."""
    return (
        "rate not recorded" if rate_per_ns is None else f"{rate_per_ns:.3g} strain/ns"
    )


def _stress_title(curve: StressStrain, fit: ElasticModulus | None) -> str:
    """A title carrying the strain rate, because the modulus depends on it."""
    control = (
        "strain-controlled" if curve.controlled == "strain" else "stress-controlled"
    )
    rate = _strain_rate_label(curve.strain_rate_per_ns)
    head = f"{curve.stage} - {control}, {rate}, {curve.temperature_k:.0f} K"
    if fit is None:
        return head
    return (
        f"{head}\nE = {fit.modulus_mpa:.0f} MPa over {fit.n_points} points"
        f"{'' if fit.resolved else ', not resolved'}"
    )


def _moduli_title(report: Any) -> str:
    """A title saying whether the constants describe one isotropic solid."""
    check = report.consistency
    if check is None:
        return "Elastic constants"
    gaps = [
        f"{name} {100.0 * gap:.0f}%"
        for name, gap in (("K", check.bulk_gap), ("G", check.shear_gap))
        if np.isfinite(gap)
    ]
    if not gaps:
        return "Elastic constants - nothing to check them against"
    verdict = "consistent" if check.consistent else "not consistent"
    return f"Elastic constants - {verdict} ({', '.join(gaps)} from E and nu)"


# --------------------------------------------------------------------------
# Relaxation
# --------------------------------------------------------------------------


def plot_relaxation(
    curve: RelaxationCurve,
    *,
    kww: KWWFit | None = None,
    prony: PronyFit | None = None,
    replicas: Sequence[RelaxationCurve] = (),
) -> Any:
    """Plot a relaxation modulus against time, with whatever was fitted to it.

    Two panels, and the lower one is the point. The upper is ``G(t)`` on log
    axes with the fits drawn through it and the baseline noise floor shaded,
    so a decay that has run into the floor is visible as one rather than read
    as a plateau. The lower is how many stress readings stand behind each
    point: it falls to one a bin at the fast end, which is exactly where the
    curve looks smoothest and is least certain, and climbs into the thousands
    at the slow end where the modulus is smallest.

    Args:
        curve: A curve from
            :func:`~openmmpolymer.relaxation.relaxation_curve` or
            :func:`~openmmpolymer.relaxation.mean_curve`.
        kww: A fit from :func:`~openmmpolymer.relaxation.fit_kww`.
        prony: A fit from :func:`~openmmpolymer.relaxation.fit_prony`.
        replicas: The individual runs behind an ensemble average, drawn faint
            underneath it.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, (decay, counts) = _figure(2, 1, height_per_row=2.6)

    for replica in replicas:
        if replica.n_points:
            decay.plot(
                replica.time_ps,
                np.abs(replica.modulus_mpa),
                linewidth=0.6,
                color=_REFERENCE_COLOUR,
                alpha=0.35,
            )
    _measured(
        decay,
        curve.time_ps,
        np.abs(curve.modulus_mpa),
        markersize=2.5,
        label=f"measured ({curve.n_replicas} replica(s))",
    )
    if curve.n_points and np.any(np.isfinite(curve.standard_error_mpa)):
        decay.fill_between(
            curve.time_ps,
            np.abs(curve.modulus_mpa) - curve.standard_error_mpa,
            np.abs(curve.modulus_mpa) + curve.standard_error_mpa,
            color=_DATA_COLOUR,
            alpha=0.18,
            linewidth=0,
        )
    if kww is not None and math.isfinite(kww.tau_ps) and curve.n_points:
        _guide(
            decay,
            curve.time_ps,
            kww.modulus_mpa * np.exp(-((curve.time_ps / kww.tau_ps) ** kww.beta)),
            colour=_GUIDE_COLOUR,
            linewidth=0.9,
            label=(
                f"KWW: beta = {kww.beta:.2f}, <tau> = {kww.mean_tau_ps:.3g} ps"
                f"{_unresolved(kww.resolved)}"
            ),
        )
    if prony is not None and prony.n_terms and curve.n_points:
        fitted = prony.equilibrium_mpa + (
            np.exp(-curve.time_ps[:, None] / prony.tau_ps) @ prony.weights_mpa
        )
        _guide(
            decay,
            curve.time_ps,
            fitted,
            colour=_ALTERNATIVE_COLOUR,
            linewidth=0.9,
            linestyle=":",
            label=(
                f"Prony: {prony.n_active} terms, G_inf = "
                f"{prony.equilibrium_mpa:.3g} MPa"
                f"{'' if prony.plateau_reached else ' (still decaying)'}"
            ),
        )
    if math.isfinite(curve.noise_floor_mpa) and curve.noise_floor_mpa > 0.0:
        decay.axhspan(
            0.0,
            curve.noise_floor_mpa,
            color=_REFERENCE_COLOUR,
            alpha=0.18,
            linewidth=0,
            label="below the baseline noise",
        )
    decay.set_xscale("log")
    decay.set_yscale("log")
    decay.set_ylabel("|G(t)| (MPa)")
    _title(decay, _relaxation_modulus_title(curve))
    _legend(decay)

    counts.plot(
        curve.time_ps,
        curve.n_samples,
        drawstyle="steps-mid",
        linewidth=0.9,
        color=_REFERENCE_COLOUR,
    )
    _level(counts, 1.0, _GUIDE_COLOUR, linewidth=0.6)
    counts.set_xscale("log")
    counts.set_yscale("log")
    counts.set_xlabel("Time since the step strain (ps)")
    counts.set_ylabel("Readings per bin")
    return figure


def plot_relaxation_spectrum(prony: PronyFit) -> Any:
    """Plot the discrete relaxation spectrum a Prony fit found.

    The weights against their time constants, with the equilibrium modulus
    drawn beside them. Most of the weights are zero - the non-negativity
    constraint sparsifies, so what is left is the fit saying which decades the
    data actually constrain, and a spectrum with everything piled on the
    slowest term is one saying the decay outlasted the run.

    Args:
        prony: A fit from :func:`~openmmpolymer.relaxation.fit_prony`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, spectrum = _panel()
    spectrum.stem(
        prony.tau_ps,
        prony.weights_mpa,
        basefmt=" ",
        linefmt="-",
        markerfmt="o",
        label="relaxation weights",
    )
    if math.isfinite(prony.equilibrium_mpa):
        _level(
            spectrum,
            prony.equilibrium_mpa,
            _GUIDE_COLOUR,
            label=f"G_inf = {prony.equilibrium_mpa:.3g} MPa",
        )
    if prony.n_terms and math.isfinite(prony.window_ps[1]):
        _span(
            spectrum,
            prony.window_ps[1],
            max(prony.window_ps[1] * 1.001, float(prony.tau_ps[-1]) * 10.0),
            _REFERENCE_COLOUR,
            label="past the end of the run",
        )
    spectrum.set_xscale("log")
    spectrum.set_xlabel("Relaxation time (ps)")
    spectrum.set_ylabel("Weight (MPa)")
    _title(
        spectrum,
        f"{prony.n_active} of {prony.n_terms} terms carry weight"
        f"{_unresolved(prony.resolved)}",
    )
    _legend(spectrum)
    return figure


def _relaxation_modulus_title(curve: RelaxationCurve) -> str:
    """A title carrying the strain and the temperature the decay belongs to."""
    return (
        f"{curve.mode} step of {curve.step_strain:+.3f} at "
        f"{curve.temperature_k:.0f} K: {curve.n_points} bins over "
        f"{curve.decades:.1f} decades"
    )


# --------------------------------------------------------------------------
# The scaffolding every figure is drawn with
# --------------------------------------------------------------------------


def _figure(
    n_rows: int = 1, n_columns: int = 1, *, height_per_row: float | None = None
) -> tuple[Any, list[Any]]:
    """Build a figure and return it with its axes flattened.

    Constructed rather than obtained from ``pyplot``, so no backend is chosen
    and no global figure registry is touched, and imported only when a figure
    is wanted, so reading a run never pays for matplotlib.
    """
    from matplotlib.figure import Figure

    height = (
        FIGURE_SIZE_IN[1]
        if height_per_row is None
        else max(FIGURE_SIZE_IN[1], height_per_row * n_rows)
    )
    figure = Figure(
        figsize=(FIGURE_SIZE_IN[0], height), dpi=FIGURE_DPI, layout="constrained"
    )
    grid = figure.subplots(n_rows, n_columns, squeeze=False)
    return figure, [axis for row in grid for axis in row]


def _panel() -> tuple[Any, Any]:
    """A figure of one panel, and that panel."""
    figure, axes = _figure()
    return figure, axes[0]


def _title(axis: Any, text: str) -> None:
    """Title a panel, in the size every panel is titled in."""
    axis.set_title(text, fontsize=9)


def _legend(axis: Any) -> None:
    """Add a panel's legend, small and unframed."""
    axis.legend(fontsize=7, frameon=False)


def _measured(
    axis: Any, x: Any, y: Any, *, markersize: float | None = 3.5, **style: Any
) -> None:
    """Draw what was measured, with a marker at each point unless told not to."""
    marker = {} if markersize is None else {"marker": "o", "markersize": markersize}
    axis.plot(x, y, **{"color": _DATA_COLOUR, "linewidth": 0.9, **marker, **style})


def _guide(
    axis: Any, x: Any, y: Any, *, colour: str = _REFERENCE_COLOUR, **style: Any
) -> None:
    """Draw a fitted or reference line through the data, dashed."""
    axis.plot(x, y, **{"color": colour, "linewidth": 0.8, "linestyle": "--", **style})


def _level(
    axis: Any, value: float, colour: str = _REFERENCE_COLOUR, **style: Any
) -> None:
    """Draw a horizontal reference level across the panel, dashed."""
    axis.axhline(
        value, **{"color": colour, "linewidth": 0.8, "linestyle": "--", **style}
    )


def _zero_line(axis: Any) -> None:
    """Draw the zero a stress or a strain is read against."""
    _level(axis, 0.0, linewidth=0.6, linestyle="-")


def _settling(axis: Any, settled: Equilibration) -> None:
    """Mark where a series settled, which is what it is discarded up to."""
    axis.axvline(settled.start_ps, color=_GUIDE_COLOUR, linewidth=0.9, linestyle="--")


def _point(
    axis: Any,
    x: float,
    y: float,
    marker: str,
    markersize: float,
    *,
    colour: str = _GUIDE_COLOUR,
    **style: Any,
) -> None:
    """Mark one point the analysis picked out."""
    axis.plot(
        [x],
        [y],
        **{
            "marker": marker,
            "markersize": markersize,
            "linestyle": "none",
            "color": colour,
            **style,
        },
    )


def _span(axis: Any, low: float, high: float, colour: str, **style: Any) -> None:
    """Shade the stretch of the horizontal axis between *low* and *high*."""
    axis.axvspan(low, high, **{"color": colour, "alpha": 0.12, "linewidth": 0, **style})


def _unresolved(resolved: bool) -> str:
    """What a legend label adds for a fit the analysis did not resolve."""
    return "" if resolved else " (unresolved)"


def _strain_axis(curve: StressStrain, quantity: str = "strain", unit: str = "") -> str:
    """Label an axis with the engineering strain along the driven direction."""
    return f"Engineering {quantity} along {'xyz'[curve.axis]}{unit}"
