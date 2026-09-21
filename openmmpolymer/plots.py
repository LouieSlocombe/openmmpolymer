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
driver that reports on them.
For the same reason no function takes an ``ax`` to draw into: each one owns a
multi-panel layout, and passing axes in would break that while inviting
``pyplot`` back.

Where a result carries a caveat, the caveat is drawn. A quench curve is titled
with its cooling rate, because the transition temperature read off it is not
comparable with an experiment cooled ten orders of magnitude slower. A
structure factor shades the region below ``2 pi / L``, where the cell cannot
hold a wave. A mean-squared displacement gets a slope-one guide line, so a
sub-diffusive curve looks sub-diffusive. A rate extrapolation shades the
decades it reached across, because that is the whole story of the figure. And
a stress-strain curve is titled with its strain rate, for the same reason a
quench curve is titled with its cooling rate: the number read off it is not
the one an experiment measures.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .conformation import (
    DECORRELATION_THRESHOLD,
    ConformationSeries,
    EndToEndRelaxation,
    MeanSquaredDisplacement,
)
from .correlations import RadialDistribution, StructureFactor
from .elasticity import ElasticModulus, PoissonRatio, StressStrain
from .timeseries import (
    CoolingRateExtrapolation,
    Equilibration,
    GlassTransition,
    QuenchCurve,
    StateData,
)

log = logging.getLogger(__name__)

#: Figure size in inches. Wide enough for a four-panel column to stay legible
#: at a report's width.
FIGURE_SIZE_IN = (7.0, 4.5)

#: Dots per inch. Enough for a screen and for print, without the file size of
#: a vector-free 300.
FIGURE_DPI = 150

#: Colour for the thing being measured, and for the reference lines drawn
#: behind it. Named rather than repeated so a figure stays one figure.
_DATA_COLOUR = "#1f4e79"
_GUIDE_COLOUR = "#b03a2e"
_REFERENCE_COLOUR = "#7f8c8d"


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
        axis.plot(data.time_ps, values, color=_DATA_COLOUR, linewidth=0.9)
        axis.set_ylabel(label, fontsize=8)
        if settled is not None:
            axis.axvline(
                settled.start_ps, color=_GUIDE_COLOUR, linewidth=0.9, linestyle="--"
            )
            axis.axhline(
                float(settled.window(values).mean()),
                color=_REFERENCE_COLOUR,
                linewidth=0.8,
                linestyle=":",
            )
    axes[-1].set_xlabel("Time (ps)")
    axes[0].set_title(_state_title(data, settled), fontsize=9)
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
    figure, axes = _figure(1, 1)
    axis = axes[0]
    axis.plot(
        curve.temperature_k,
        curve.specific_volume_cm3_g,
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        color=_DATA_COLOUR,
        label="measured",
    )
    if transition is not None:
        span = np.asarray(
            [curve.temperature_k.min(), curve.temperature_k.max()], dtype=np.float64
        )
        for slope, name in (
            (transition.glass_expansion_per_k, "glass"),
            (transition.melt_expansion_per_k, "melt"),
        ):
            offset = _branch_offset(transition, slope)
            axis.plot(
                span,
                offset + slope * span,
                linewidth=0.8,
                linestyle="--",
                color=_REFERENCE_COLOUR,
                label=f"{name} fit",
            )
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
    axis.set_title(_quench_title(curve, transition), fontsize=9)
    axis.legend(fontsize=7, frameon=False)
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
    figure, axes = _figure(1, 1)
    axis = axes[0]
    rates = extrapolation.cooling_rate_k_per_ns
    target = extrapolation.target_rate_k_per_ns
    lowest = min(float(rates.min()), target)

    axis.axvspan(lowest, float(rates.min()), color=_GUIDE_COLOUR, alpha=0.15)
    span = np.geomspace(lowest, float(rates.max()), 200)
    axis.plot(
        span,
        _predicted(extrapolation, span),
        linewidth=0.9,
        linestyle="--",
        color=_REFERENCE_COLOUR,
        label=f"{extrapolation.form} fit",
    )
    axis.plot(
        rates,
        extrapolation.transition_k,
        marker="o",
        markersize=4.0,
        linestyle="none",
        color=_DATA_COLOUR,
        label="measured",
    )
    axis.plot(
        [target],
        [extrapolation.temperature_k],
        marker="*",
        markersize=9.0,
        linestyle="none",
        color=_GUIDE_COLOUR,
        label=f"{extrapolation.temperature_k:.0f} K at {target:.3g} K/ns",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Cooling rate (K/ns)")
    axis.set_ylabel("Transition temperature (K)")
    axis.set_title(_cooling_rate_title(extrapolation), fontsize=9)
    axis.legend(fontsize=7, frameon=False)
    return figure


def _predicted(extrapolation: CoolingRateExtrapolation, rate_k_per_ns: Any) -> Any:
    """The fitted relation evaluated over a range of rates."""
    parameters = extrapolation.parameters
    if extrapolation.form == "log_linear":
        return parameters["a_k"] + parameters["b_k_per_decade"] * np.log10(
            rate_k_per_ns
        )
    return parameters["t0_k"] + parameters["b_k"] / (
        parameters["ln_r0"] - np.log(rate_k_per_ns)
    )


def plot_conformation(series: ConformationSeries) -> Any:
    """Plot chain dimensions against time.

    Args:
        series: A series from
            :func:`~openmmpolymer.conformation.chain_conformation`.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    figure, axes = _figure(2, 1, height_per_row=1.8)
    axes[0].plot(
        series.time_ps,
        series.mean_squared_end_to_end_nm2,
        color=_DATA_COLOUR,
        linewidth=0.9,
        label="measured",
    )
    expected = _expected_square(series)
    if expected is not None:
        axes[0].axhline(
            expected,
            color=_REFERENCE_COLOUR,
            linewidth=0.8,
            linestyle="--",
            label="expected from C-infinity",
        )
    axes[0].set_ylabel("<R^2> (nm2)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)

    axes[1].plot(
        series.time_ps,
        series.mean_radius_of_gyration_nm,
        color=_DATA_COLOUR,
        linewidth=0.9,
    )
    axes[1].set_ylabel("Rg (nm)", fontsize=8)
    axes[1].set_xlabel("Time (ps)")
    if series.settled is not None:
        for axis in axes:
            axis.axvline(
                series.settled.start_ps,
                color=_GUIDE_COLOUR,
                linewidth=0.9,
                linestyle="--",
            )
    axes[0].set_title(_conformation_title(series), fontsize=9)
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
    axis.plot(distribution.r_nm, distribution.g_r, color=_DATA_COLOUR, linewidth=1.0)
    axis.axhline(1.0, color=_REFERENCE_COLOUR, linewidth=0.8, linestyle="--")
    axis.set_xlabel("r (nm)")
    axis.set_ylabel("g(r), intermolecular")
    axis.set_title(
        f"{distribution.n_pairs:,} pairs over {distribution.n_frames} frame(s)",
        fontsize=9,
    )

    if structure is not None:
        second = axes[1]
        second.plot(
            structure.q_per_nm, structure.s_q, color=_DATA_COLOUR, linewidth=1.0
        )
        second.axvspan(0.0, structure.q_min_per_nm, color=_GUIDE_COLOUR, alpha=0.15)
        second.set_xlabel("q (1/nm)")
        second.set_ylabel("S(q)")
        second.set_title(
            f"below {structure.q_min_per_nm:.1f} /nm the cell cannot resolve",
            fontsize=9,
        )
    return figure


def plot_dynamics(
    msd: MeanSquaredDisplacement, *, relaxation: EndToEndRelaxation | None = None
) -> Any:
    """Plot the mean-squared displacement, and the end-to-end decay beside it.

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
    axis.plot(
        msd.lag_ps[usable],
        msd.msd_nm2[usable],
        color=_DATA_COLOUR,
        linewidth=1.0,
        label=f"measured (slope {msd.log_slope:.2f})",
    )
    if usable.any():
        lags = msd.lag_ps[usable]
        reference = msd.msd_nm2[usable][-1] * (lags / lags[-1])
        axis.plot(
            lags,
            reference,
            color=_GUIDE_COLOUR,
            linewidth=0.8,
            linestyle="--",
            label="slope 1 (diffusive)",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlabel("Lag (ps)")
    axis.set_ylabel("Centre-of-mass MSD (nm2)")
    axis.set_title(_msd_title(msd), fontsize=9)
    axis.legend(fontsize=7, frameon=False)

    if relaxation is not None:
        second = axes[1]
        second.plot(
            relaxation.lag_ps,
            relaxation.correlation,
            color=_DATA_COLOUR,
            linewidth=1.0,
        )
        second.axhline(
            DECORRELATION_THRESHOLD,
            color=_GUIDE_COLOUR,
            linewidth=0.8,
            linestyle="--",
        )
        second.set_xlabel("Lag (ps)")
        second.set_ylabel("End-to-end correlation")
        second.set_title(_relaxation_title(relaxation), fontsize=9)
    return figure


def _figure(
    n_rows: int, n_columns: int, *, height_per_row: float | None = None
) -> tuple[Any, list[Any]]:
    """Build a figure and return it with its axes flattened.

    Constructed rather than obtained from ``pyplot``, so no backend is chosen
    and no global figure registry is touched.
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


def _branch_offset(transition: GlassTransition, slope: float) -> float:
    """The intercept of a fitted branch, from its slope and the crossing.

    Both branches pass through the transition by construction, so the volume
    there plus a slope fixes each line.
    """
    return transition.specific_volume_cm3_g - slope * transition.temperature_k


def _expected_square(series: ConformationSeries) -> float | None:
    """``<R^2>`` implied by the expected characteristic ratio, for a guide line.

    ``C = <R^2> / (n l^2)``, and the measurement reports both the measured C
    and the expected one against the same bond length, so the ratio of the
    two scales the measured mean square.
    """
    measured = series.mean.characteristic_ratio
    if abs(measured) < 1.0e-30:
        return None
    scale = series.mean.expected_characteristic_ratio / measured
    return series.mean.mean_squared_end_to_end_nm2 * scale


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
    figure, axes = _figure(2, 1, height_per_row=2.6)
    stress, lateral = axes

    stress.plot(
        curve.strain,
        curve.tensile_stress_mpa,
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        color=_DATA_COLOUR,
        label="measured",
    )
    if fit is not None and np.isfinite(fit.modulus_mpa):
        span = np.asarray([0.0, float(curve.strain.max())], dtype=np.float64)
        stress.plot(
            span,
            fit.intercept_mpa + fit.modulus_mpa * span,
            linewidth=0.8,
            linestyle="--",
            color=_GUIDE_COLOUR,
            label=(
                f"E = {fit.modulus_mpa:.0f} MPa"
                f"{'' if fit.resolved else ' (unresolved)'}"
            ),
        )
        stress.axvspan(0.0, fit.strain_limit, color=_REFERENCE_COLOUR, alpha=0.12, lw=0)
    stress.axhline(0.0, color=_REFERENCE_COLOUR, linewidth=0.6)
    stress.set_ylabel("Tensile stress (MPa)")
    stress.set_title(_stress_title(curve, fit), fontsize=9)
    stress.legend(fontsize=7, frameon=False)

    lateral.plot(
        curve.strain,
        curve.transverse_strain,
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        color=_DATA_COLOUR,
        label="measured",
    )
    if poisson is not None and np.isfinite(poisson.ratio):
        span = np.asarray([0.0, float(curve.strain.max())], dtype=np.float64)
        lateral.plot(
            span,
            -poisson.ratio * span,
            linewidth=0.8,
            linestyle="--",
            color=_GUIDE_COLOUR,
            label=(
                f"nu = {poisson.ratio:.3f}{'' if poisson.resolved else ' (unresolved)'}"
            ),
        )
    lateral.axhline(0.0, color=_REFERENCE_COLOUR, linewidth=0.6)
    lateral.set_xlabel(f"Engineering strain along {'xyz'[curve.axis]}")
    lateral.set_ylabel("Mean transverse strain")
    lateral.legend(fontsize=7, frameon=False)
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
    figure, axes = _figure(1, 1)
    axis = axes[0]

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
    check = getattr(report, "consistency", None)
    if check is not None:
        for label, implied in (
            ("K", check.bulk_implied_mpa),
            ("G", check.shear_implied_mpa),
        ):
            if label in labels and np.isfinite(implied):
                axis.plot(
                    [labels.index(label)],
                    [implied],
                    marker="_",
                    markersize=22,
                    markeredgewidth=2.0,
                    linestyle="none",
                    color=_GUIDE_COLOUR,
                    label="implied by E and nu"
                    if label == labels[min(len(labels) - 1, 1)]
                    else None,
                )
    if report.load_modulus is not None and "E" in labels:
        axis.plot(
            [labels.index("E")],
            [report.load_modulus.modulus_mpa],
            marker="x",
            markersize=8,
            linestyle="none",
            color="#7d3c98",
            label="constant-stress cross-check",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Modulus (MPa)")
    axis.set_title(_moduli_title(report), fontsize=9)
    handles, names = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, names, fontsize=7, frameon=False)
    return figure


def _stress_title(curve: StressStrain, fit: ElasticModulus | None) -> str:
    """A title carrying the strain rate, because the modulus depends on it."""
    control = (
        "strain-controlled" if curve.controlled == "strain" else "stress-controlled"
    )
    rate = (
        "rate not recorded"
        if curve.strain_rate_per_ns is None
        else f"{curve.strain_rate_per_ns:.3g} strain/ns"
    )
    head = f"{curve.stage} - {control}, {rate}, {curve.temperature_k:.0f} K"
    if fit is None:
        return head
    return (
        f"{head}\nE = {fit.modulus_mpa:.0f} MPa over {fit.n_points} points"
        f"{'' if fit.resolved else ', not resolved'}"
    )


def _moduli_title(report: Any) -> str:
    """A title saying whether the constants describe one isotropic solid."""
    check = getattr(report, "consistency", None)
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
