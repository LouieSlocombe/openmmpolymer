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
the one an experiment measures. A relaxation modulus gets two of these at
once: the level its own baseline scatter could not see past is shaded, and
the readings behind each point are drawn underneath, because the early part
of that curve rests on one reading a bin and looks far more certain than it
is. A persistence length shades the separations past the end of the chain,
because a fit that had to reach past the molecule to find 1/e is an
extrapolation whatever number it came back with.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

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
    from .strength import BreakingStrength, ElongationAtBreak, YieldStrength

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
    figure, axes = _figure(1, 1)
    axis = axes[0]
    axis.plot(
        curve.strain,
        result.nominal_stress_mpa,
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        color=_DATA_COLOUR,
        label="nominal tensile stress",
    )
    axis.plot(
        [result.strain_at_peak],
        [result.peak_stress_mpa],
        marker="*",
        markersize=10,
        linestyle="none",
        color=_GUIDE_COLOUR,
        label=f"sampled peak = {result.peak_stress_mpa:.1f} MPa",
    )
    if result.resolved:
        if result.failure_bracket is not None:
            axis.axvspan(
                *result.failure_bracket,
                color=_GUIDE_COLOUR,
                alpha=0.12,
                linewidth=0,
                label="failure strain bracket",
            )
        if result.failure_strain is not None and result.failure_stress_mpa is not None:
            axis.plot(
                [result.failure_strain],
                [result.failure_stress_mpa],
                marker="D",
                markersize=5,
                linestyle="none",
                color=_GUIDE_COLOUR,
                label="sustained stress drop",
            )
    axis.axhline(0.0, color=_REFERENCE_COLOUR, linewidth=0.6)
    axis.set_xlabel(f"Engineering strain along {'xyz'[curve.axis]}")
    axis.set_ylabel("Nominal tensile stress (MPa)")
    rate = (
        "rate not recorded"
        if result.strain_rate_per_ns is None
        else f"{result.strain_rate_per_ns:.3g} strain/ns"
    )
    verdict = (
        f"Apparent tensile strength = {result.strength_mpa:.1f} MPa"
        if result.resolved and result.strength_mpa is not None
        else "Apparent tensile strength not resolved"
    )
    chunks = curve.stage.split(", ")
    label = (
        chunks[0] if len(chunks) == 1 else f"{chunks[0]} (+{len(chunks) - 1} chunks)"
    )
    axis.set_title(
        f"{label}: {result.temperature_k:.0f} K, {rate}\n{verdict}",
        fontsize=9,
    )
    axis.legend(fontsize=7, frameon=False)
    return figure


def plot_elongation_at_break(
    curve: StressStrain,
    result: ElongationAtBreak | None = None,
    *,
    failure_fraction: float = 0.5,
    confirmation_steps: int = 3,
) -> Any:
    """Plot apparent elongation at break on the nominal tensile response.

    The horizontal axis and the break bracket use percent engineering strain.
    The marker identifies the first sampled hold in a confirmed terminal loss
    of stress, rather than the strain at peak stress. Its shaded bracket shows
    the sampling interval, not a confidence interval or an interpolated crack.

    Args:
        curve: The tensile curve analysed for elongation at break.
        result: Its result from
            :func:`~openmmpolymer.strength.elongation_at_break`. If omitted,
            analyse the curve with the supplied failure criterion.
        failure_fraction: Fraction of peak stress defining a drop when
            ``result`` is omitted.
        confirmation_steps: Required terminal holds at or below that threshold when
            ``result`` is omitted.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    if result is None:
        from .strength import elongation_at_break

        result = elongation_at_break(
            curve,
            failure_fraction=failure_fraction,
            confirmation_steps=confirmation_steps,
        )
    figure, axes = _figure(1, 1)
    axis = axes[0]
    axis.plot(
        100.0 * curve.strain,
        result.nominal_stress_mpa,
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        color=_DATA_COLOUR,
        label="nominal tensile stress",
    )
    axis.plot(
        [100.0 * result.strain_at_peak],
        [result.peak_stress_mpa],
        marker="*",
        markersize=10,
        linestyle="none",
        color=_REFERENCE_COLOUR,
        label=f"sampled peak = {result.peak_stress_mpa:.1f} MPa",
    )
    if result.resolved:
        if result.break_bracket is not None:
            axis.axvspan(
                *(100.0 * strain for strain in result.break_bracket),
                color=_GUIDE_COLOUR,
                alpha=0.12,
                linewidth=0,
                label="break elongation bracket",
            )
        if (
            result.elongation_percent is not None
            and result.break_stress_mpa is not None
        ):
            axis.plot(
                [result.elongation_percent],
                [result.break_stress_mpa],
                marker="D",
                markersize=5,
                linestyle="none",
                color=_GUIDE_COLOUR,
                label="onset of sustained stress drop",
            )
    axis.axhline(0.0, color=_REFERENCE_COLOUR, linewidth=0.6)
    axis.set_xlabel(f"Engineering elongation along {'xyz'[curve.axis]} (%)")
    axis.set_ylabel("Nominal tensile stress (MPa)")
    rate = (
        "rate not recorded"
        if result.strain_rate_per_ns is None
        else f"{result.strain_rate_per_ns:.3g} strain/ns"
    )
    verdict = (
        f"Apparent elongation at break = {result.elongation_percent:.1f}%"
        if result.resolved and result.elongation_percent is not None
        else "Apparent elongation at break not resolved"
    )
    chunks = curve.stage.split(", ")
    label = (
        chunks[0] if len(chunks) == 1 else f"{chunks[0]} (+{len(chunks) - 1} chunks)"
    )
    axis.set_title(
        f"{label}: {result.temperature_k:.0f} K, {rate}\n{verdict}",
        fontsize=9,
    )
    axis.legend(fontsize=7, frameon=False)
    return figure


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
    figure, axes = _figure(1, 1)
    axis = axes[0]
    axis.plot(
        curve.strain,
        result.nominal_stress_mpa,
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        color=_DATA_COLOUR,
        label="nominal tensile stress",
    )
    fit_span = np.asarray(
        [result.fit_min_strain, result.fit_max_strain], dtype=np.float64
    )
    axis.axvspan(
        *fit_span,
        color=_REFERENCE_COLOUR,
        alpha=0.12,
        linewidth=0,
        label="elastic fit window",
    )
    offset_label = f"{100.0 * result.offset_strain:g}% offset"
    if (
        result.modulus_mpa is not None
        and result.intercept_mpa is not None
        and math.isfinite(result.modulus_mpa)
        and math.isfinite(result.intercept_mpa)
    ):
        axis.plot(
            fit_span,
            result.modulus_mpa * fit_span + result.intercept_mpa,
            linewidth=1.0,
            linestyle="--",
            color=_REFERENCE_COLOUR,
            label=(
                f"elastic fit: E = {result.modulus_mpa:.1f} MPa"
                f"{'' if result.fit_resolved else ' (unresolved)'}"
            ),
        )
        span = np.asarray(
            [float(curve.strain.min()), float(curve.strain.max())], dtype=np.float64
        )
        axis.plot(
            span,
            result.modulus_mpa * (span - result.offset_strain) + result.intercept_mpa,
            linewidth=1.0,
            linestyle="--",
            color=_GUIDE_COLOUR,
            label=f"{offset_label} line",
        )
    if result.resolved:
        if result.yield_bracket is not None:
            axis.axvspan(
                *result.yield_bracket,
                color=_GUIDE_COLOUR,
                alpha=0.12,
                linewidth=0,
                label="yield strain bracket",
            )
        if result.yield_strain is not None and result.strength_mpa is not None:
            axis.plot(
                [result.yield_strain],
                [result.strength_mpa],
                marker="*",
                markersize=10,
                linestyle="none",
                color=_GUIDE_COLOUR,
                label="interpolated offset intersection",
            )

    # Extending the elastic line to the final strain can give stresses far
    # above the measured response. Keep that extrapolation from setting the
    # scale and hiding the curve whose intersection is being reported.
    low = min(0.0, float(result.nominal_stress_mpa.min()))
    high = max(0.0, float(result.nominal_stress_mpa.max()))
    margin = max(high - low, 1.0) * 0.05
    axis.set_ylim(low - margin, high + margin)
    axis.axhline(0.0, color=_REFERENCE_COLOUR, linewidth=0.6)
    axis.set_xlabel(f"Engineering strain along {'xyz'[curve.axis]}")
    axis.set_ylabel("Nominal tensile stress (MPa)")
    rate = (
        "rate not recorded"
        if result.strain_rate_per_ns is None
        else f"{result.strain_rate_per_ns:.3g} strain/ns"
    )
    verdict = (
        f"{offset_label} yield strength = {result.strength_mpa:.1f} MPa"
        if result.resolved and result.strength_mpa is not None
        else f"{offset_label} yield strength not resolved"
    )
    chunks = curve.stage.split(", ")
    label = (
        chunks[0] if len(chunks) == 1 else f"{chunks[0]} (+{len(chunks) - 1} chunks)"
    )
    axis.set_title(
        f"{label}: {result.temperature_k:.0f} K, {rate}\n{verdict}",
        fontsize=9,
    )
    axis.legend(fontsize=7, frameon=False)
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
    figure, axes = _figure(2, 1, height_per_row=2.6)
    decay, counts = axes

    for replica in replicas:
        if replica.n_points:
            decay.plot(
                replica.time_ps,
                np.abs(replica.modulus_mpa),
                linewidth=0.6,
                color=_REFERENCE_COLOUR,
                alpha=0.35,
            )
    decay.plot(
        curve.time_ps,
        np.abs(curve.modulus_mpa),
        marker="o",
        markersize=2.5,
        linewidth=0.9,
        color=_DATA_COLOUR,
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
        decay.plot(
            curve.time_ps,
            kww.modulus_mpa * np.exp(-((curve.time_ps / kww.tau_ps) ** kww.beta)),
            linewidth=0.9,
            linestyle="--",
            color=_GUIDE_COLOUR,
            label=(
                f"KWW: beta = {kww.beta:.2f}, <tau> = {kww.mean_tau_ps:.3g} ps"
                f"{'' if kww.resolved else ' (unresolved)'}"
            ),
        )
    if prony is not None and prony.n_terms and curve.n_points:
        fitted = prony.equilibrium_mpa + (
            np.exp(-curve.time_ps[:, None] / prony.tau_ps) @ prony.weights_mpa
        )
        decay.plot(
            curve.time_ps,
            fitted,
            linewidth=0.9,
            linestyle=":",
            color="#6c3483",
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
            lw=0,
            label="below the baseline noise",
        )
    decay.set_xscale("log")
    decay.set_yscale("log")
    decay.set_ylabel("|G(t)| (MPa)")
    decay.set_title(_relaxation_modulus_title(curve), fontsize=9)
    decay.legend(fontsize=7, frameon=False)

    counts.plot(
        curve.time_ps,
        curve.n_samples,
        drawstyle="steps-mid",
        linewidth=0.9,
        color=_REFERENCE_COLOUR,
    )
    counts.axhline(1.0, color=_GUIDE_COLOUR, linewidth=0.6, linestyle="--")
    counts.set_xscale("log")
    counts.set_yscale("log")
    counts.set_xlabel("Time since the step strain (ps)")
    counts.set_ylabel("Readings per bin")
    return figure


def _relaxation_modulus_title(curve: RelaxationCurve) -> str:
    """A title carrying the strain and the temperature the decay belongs to."""
    return (
        f"{curve.mode} step of {curve.step_strain:+.3f} at "
        f"{curve.temperature_k:.0f} K: {curve.n_points} bins over "
        f"{curve.decades:.1f} decades"
    )


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
    figure, axes = _figure(1, 1)
    spectrum = axes[0]

    spectrum.stem(
        prony.tau_ps,
        prony.weights_mpa,
        basefmt=" ",
        linefmt="-",
        markerfmt="o",
        label="relaxation weights",
    )
    if math.isfinite(prony.equilibrium_mpa):
        spectrum.axhline(
            prony.equilibrium_mpa,
            color=_GUIDE_COLOUR,
            linewidth=0.8,
            linestyle="--",
            label=f"G_inf = {prony.equilibrium_mpa:.3g} MPa",
        )
    if prony.n_terms and math.isfinite(prony.window_ps[1]):
        spectrum.axvspan(
            prony.window_ps[1],
            max(prony.window_ps[1] * 1.001, float(prony.tau_ps[-1]) * 10.0),
            color=_REFERENCE_COLOUR,
            alpha=0.12,
            lw=0,
            label="past the end of the run",
        )
    spectrum.set_xscale("log")
    spectrum.set_xlabel("Relaxation time (ps)")
    spectrum.set_ylabel("Weight (MPa)")
    spectrum.set_title(
        f"{prony.n_active} of {prony.n_terms} terms carry weight"
        f"{'' if prony.resolved else ' (unresolved)'}",
        fontsize=9,
    )
    spectrum.legend(fontsize=7, frameon=False)
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
    figure, axes = _figure(1, 1)
    axis = axes[0]
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

    axis.plot(
        separation,
        length.correlation,
        marker="o",
        markersize=3.0,
        linewidth=0.9,
        color=_DATA_COLOUR,
        label="measured",
    )
    axis.axhline(
        DECORRELATION_THRESHOLD,
        color=_GUIDE_COLOUR,
        linewidth=0.8,
        linestyle="--",
        label="1/e",
    )
    if fitted:
        span = np.linspace(0.0, x_max, 200)
        axis.plot(
            span,
            np.exp(-span * length.bond_length_nm / length.persistence_length_nm),
            linewidth=0.8,
            linestyle="--",
            color=_REFERENCE_COLOUR,
            label=f"exp(-s l_b / l_p), l_p = {length.persistence_length_nm:.2f} nm",
        )
    if not length.decayed:
        axis.axvspan(
            last,
            x_max,
            color=_REFERENCE_COLOUR,
            alpha=0.12,
            lw=0,
            label="past the end of the chain",
        )
        axis.set_xlim(0.0, x_max)
    axis.set_xlabel("Separation (bonds)")
    axis.set_ylabel("<cos theta(s)>")
    axis.set_title(_persistence_title(length), fontsize=9)
    axis.legend(fontsize=7, frameon=False)
    return figure


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
