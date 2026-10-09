"""Console summaries shared by the CLI's live runs and saved analyses.

Formatting stays separate from argument parsing and workflow dispatch so that
both paths present the same units, uncertainty and resolution caveats.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sized
from typing import Any


def _say(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def _unresolved(resolved: bool) -> str:
    """The caveat a number carries when what measured it did not resolve."""
    return "" if resolved else " (not resolved)"


def _print_chains(chains: Any) -> None:
    """Report the final chain dimensions, when they were measured."""
    if chains is not None:
        print(
            f"chains: Rg {chains.mean_radius_of_gyration_nm:.3f} nm, "
            f"C {chains.characteristic_ratio:.2f} "
            f"({'consistent' if chains.consistent else 'not relaxed'})"
        )


def _convergence_lines(report: Any) -> Iterator[str]:
    """Whether each estimate settled as the observation window grew."""
    yield f"observation-window convergence: stage {report.stage}"
    groups = [
        report.results,
        {} if report.relaxation is None else report.relaxation.metrics,
        {} if report.structural is None else report.structural.parameters,
    ]
    for results in groups:
        for name, result in results.items():
            yield f"{name}: {'resolved' if result.resolved else 'unresolved'}"
            yield from (f"  {note}" for note in result.notes)


def _rate_lines(report: Any) -> Iterator[str]:
    """A unit-aware headline for each model, including unavailable predictions."""
    quantity = report.property
    for form in ("log_linear", "power_law"):
        fit = getattr(report, form)
        if fit is None:
            yield f"{quantity.label}, {form}: unavailable (not resolved)"
            continue
        uncertainty = (
            f"{fit.standard_error:.3g}"
            if math.isfinite(fit.standard_error)
            else "unknown"
        )
        yield (
            f"{quantity.label}, {form}: {fit.value:.5g} +/- {uncertainty} "
            f"{quantity.value_unit} (fit SE) at {fit.target_rate:.4g} "
            f"{quantity.rate_unit}; {fit.n_rates} rates, extrapolated "
            f"{fit.extrapolation_decades:.2f} decades{_unresolved(fit.resolved)}"
        )
        yield from (f"note ({form}): {note}" for note in fit.notes)
    if report.log_linear is not None and report.power_law is not None:
        difference = abs(report.log_linear.value - report.power_law.value)
        yield f"model difference at target: {difference:.4g} {quantity.value_unit}"


def _cooling_rate(rate_k_per_ns: float | None) -> str:
    return "rate unknown" if rate_k_per_ns is None else f"{rate_k_per_ns:.2f} K/ns"


def _tg_lines(report: Any) -> Iterator[str]:
    """The quenches read, the melt they started from, and what they found."""
    yield "quenches: " + ", ".join(
        f"{curve.stage} ({curve.temperature_step_k:.0f} K steps, "
        f"{_cooling_rate(curve.cooling_rate_k_per_ns)})"
        for curve in report.curves
    )
    melt = report.melt
    if melt is not None:
        settled = "volume settled" if melt.volume_settled else "volume still drifting"
        moved = "chains moved" if melt.chains_moved else "chains have not"
        yield f"melt {melt.stage}: {settled}; {moved}"
        yield from (f"  unchecked: {reason}" for reason in melt.unchecked)
    for label, transition in (("coarse", report.coarse), ("fine", report.fine)):
        if transition is not None:
            yield _transition_line(label, transition)
    for extrapolation in (report.log_linear, report.vft):
        if extrapolation is not None:
            yield _extrapolation_line(extrapolation)


def _transition_line(label: str, fit: Any) -> str:
    """One line for a fitted transition, with both expansivities."""
    rate = _cooling_rate(fit.cooling_rate_k_per_ns)
    if not fit.resolved:
        return f"{label}: no clear transition at {rate}"
    return (
        f"{label}: Tg = {fit.temperature_k:.0f} K at {rate}, aV "
        f"{fit.melt_expansivity_per_k:.2e} / {fit.glass_expansivity_per_k:.2e} per K"
    )


def _extrapolation_line(extrapolation: Any) -> str:
    """One line for a rate extrapolation, caveat included."""
    return (
        f"{extrapolation.form}: {extrapolation.temperature_k:.0f} K at "
        f"{extrapolation.target_rate_k_per_ns:.3g} K/ns, "
        f"{extrapolation.sensitivity_k_per_decade:.1f} K per decade over "
        f"{extrapolation.n_rates} rates - extrapolated "
        f"{extrapolation.extrapolation_decades:.1f} decades"
        f"{'' if extrapolation.resolved else ', not resolved'}"
    )


def _melting_lines(report: Any) -> Iterator[str]:
    """The finite heating bracket, without implying an equilibrium Tm."""
    transition = report.transition
    if not transition.resolved or transition.temperature_k is None:
        yield "tm: no clear melting transition (not resolved)"
    else:
        low, high = transition.bracket_k
        yield (
            f"tm: apparent Tm = {transition.temperature_k:g} K "
            f"(heating bracket {low:g}-{high:g} K)"
        )


def _modulus_lines(report: Any) -> list[str]:
    """One line per elastic constant, each carrying what qualifies it."""
    lines = []
    youngs = report.youngs
    if youngs is not None:
        spread = _replica_spread(report.replica_spread_mpa, report.replicas, ".0f")
        rate = youngs.strain_rate_per_ns
        speed = "rate unknown" if rate is None else f"{rate:.3g} strain/ns"
        lines.append(
            f"E = {youngs.modulus_mpa:.0f} MPa{spread} at {speed}, "
            f"{youngs.temperature_k:.0f} K{_unresolved(report.resolved)}"
        )
    if report.poisson is not None:
        lines.append(
            f"nu = {report.poisson.ratio:.3f}{_unresolved(report.poisson.resolved)}"
        )
    for label, fit in (("K", report.bulk), ("G", report.shear)):
        if fit is not None:
            error = (
                f"{fit.standard_error_mpa:.2g}"
                if math.isfinite(fit.standard_error_mpa)
                else "unknown"
            )
            lines.append(
                f"{label} = {fit.modulus_mpa:.0f} +/- "
                f"{error} MPa (fit SE){_unresolved(fit.resolved)}"
            )
    if report.load_modulus is not None:
        lines.append(
            "constant-stress cross-check: "
            f"E = {report.load_modulus.modulus_mpa:.0f} MPa"
        )
    if report.consistency is not None:
        lines.append(_consistency_line(report.consistency))
    return lines or ["modulus: nothing was deformed"]


def _consistency_line(check: Any) -> str:
    """One line for the over-determination check."""
    implied = (
        f"E and nu imply K = {check.bulk_implied_mpa:.0f}, "
        f"G = {check.shear_implied_mpa:.0f} MPa"
    )
    gaps = ", ".join(f"{name} {100.0 * gap:.0f}%" for name, gap in check.gaps)
    if not gaps:
        return f"{implied} - nothing measured to check them against"
    return (
        f"{implied}; measured differ by {gaps}"
        f"{'' if check.consistent else ' - not consistent'}"
    )


def _strain_rate(rate_per_ns: float | None) -> str:
    if rate_per_ns is None:
        return "unknown strain rate"
    return f"{rate_per_ns:.3g} strain/ns"


def _replica_spread(
    spread: float | None, replicas: Sized, digits: str = ".3g", unit: str = ""
) -> str:
    if spread is None:
        return ""
    return f" +/- {spread:{digits}}{unit} over {len(replicas)} replicas"


def _breaking_lines(report: Any) -> Iterator[str]:
    """Keep an unconfirmed peak distinct from an apparent tensile strength."""
    if report.strength_mpa is None or not report.resolved:
        yield "breaking: apparent tensile strength not resolved"
    else:
        yield (
            "breaking: apparent ultimate nominal tensile strength = "
            f"{report.strength_mpa:.4g} MPa"
            f"{_replica_spread(report.replica_spread_mpa, report.replicas)}"
        )
    for index, result in zip(report.replica_indices, report.replicas, strict=True):
        yield (
            f"  replica {index}: peak {result.peak_stress_mpa:.4g} MPa at "
            f"strain {result.strain_at_peak:.4g}, {result.temperature_k:.0f} K, "
            f"{_strain_rate(result.strain_rate_per_ns)}{_unresolved(result.resolved)}"
        )
        if result.failure_strain is not None:
            yield (
                f"    stress drop at strain {result.failure_strain:.4g}, "
                f"stress {result.failure_stress_mpa:.4g} MPa"
            )


def _elongation_lines(report: Any) -> Iterator[str]:
    """Report the confirmed break strain separately from the stress maximum."""
    if report.elongation_percent is None or not report.resolved:
        yield "elongation: apparent elongation at break not resolved"
    else:
        spread = _replica_spread(
            report.replica_spread_percent, report.replicas, unit=" percentage points"
        )
        yield (
            "elongation: apparent elongation at break = "
            f"{report.elongation_percent:.4g}%{spread}"
        )
    for index, result in zip(report.replica_indices, report.replicas, strict=True):
        elongation = (
            f"{result.elongation_percent:.4g}% at engineering strain "
            f"{result.strain_at_break:.4g}"
            if result.resolved
            and result.elongation_percent is not None
            and result.strain_at_break is not None
            else "not resolved"
        )
        yield (
            f"  replica {index}: elongation at break {elongation}, "
            f"{result.temperature_k:.0f} K, {_strain_rate(result.strain_rate_per_ns)}"
        )
        yield (
            f"    peak {result.peak_stress_mpa:.4g} MPa at "
            f"strain {result.strain_at_peak:.4g}"
        )
        if result.resolved and result.break_stress_mpa is not None:
            yield f"    stress at break {result.break_stress_mpa:.4g} MPa"


def _yield_lines(report: Any) -> Iterator[str]:
    """Print the proof stress together with its offset, temperature and rate."""
    if report.strength_mpa is None or not report.resolved:
        yield "yield: apparent offset yield strength not resolved"
    else:
        yield (
            "yield: apparent offset yield strength = "
            f"{report.strength_mpa:.4g} MPa"
            f"{_replica_spread(report.replica_spread_mpa, report.replicas)}"
        )
    for index, result in zip(report.replica_indices, report.replicas, strict=True):
        strength = (
            f"{result.strength_mpa:.4g} MPa at strain {result.yield_strain:.4g}"
            if result.resolved
            and result.strength_mpa is not None
            and result.yield_strain is not None
            else "not resolved"
        )
        yield (
            f"  replica {index}: {100.0 * result.offset_strain:g}% offset "
            f"proof stress {strength}, {result.temperature_k:.0f} K, "
            f"{_strain_rate(result.strain_rate_per_ns)}"
        )
        if result.modulus_mpa is not None:
            yield (
                f"    initial elastic slope {result.modulus_mpa:.4g} MPa "
                f"over strain {result.fit_min_strain:g} to {result.fit_max_strain:g}"
            )


def _relaxation_lines(report: Any) -> list[str]:
    """One line per fitted quantity, each carrying what qualifies it.

    Takes the :class:`~openmmpolymer.viscoelastic.RelaxationReport` a scan
    returns or ``--analyse`` reads back; both carry the overall verdict.
    """
    mean = report.mean
    if mean is None:
        return ["relax: nothing was strained"]
    spread = _replica_spread(report.replica_spread_mpa, report.curves)
    lines = [
        f"G(0) = {mean.initial_modulus_mpa:.4g} MPa{spread} at "
        f"{mean.step_strain:+.3f} strain, {mean.temperature_k:.0f} K, over "
        f"{mean.decades:.1f} decades{_unresolved(report.resolved)}"
    ]
    kww = report.kww
    if kww is not None:
        lines.append(
            f"KWW: beta = {kww.beta:.3f}, tau = {kww.tau_ps:.4g} ps, "
            f"<tau> = {kww.mean_tau_ps:.4g} ps{_unresolved(kww.resolved)}"
        )
    prony = report.prony
    if prony is not None:
        lines.append(
            f"Prony: G_inf = {prony.equilibrium_mpa:.4g} MPa over "
            f"{prony.n_active} of {prony.n_terms} terms"
            f"{'' if prony.plateau_reached else ' - still decaying'}"
        )
    linearity = report.linearity
    if linearity is not None:
        lines.append(
            f"linearity: strains {[round(v, 4) for v in linearity.strains]} "
            f"differ by {100.0 * linearity.gap:.0f}%"
            f"{'' if linearity.linear else ' - outside the linear region'}"
        )
    return lines


def _structure_lines(report: Any) -> Iterator[str]:
    """One line per measurement, each carrying its own caveat."""
    frames = (
        "single snapshot"
        if report.is_snapshot
        else f"{report.n_frames} frames at {report.interval_ps:g} ps"
    )
    yield (
        f"structure: stage {report.stage} ({frames}), {report.n_chains} chains "
        f"of {report.atoms_per_chain} atoms"
    )
    if report.backbone is None:
        yield "backbone: unknown, so no chain measurements"
    else:
        origin = report.backbone_source + (
            "" if report.backbone_file is None else f" from {report.backbone_file}"
        )
        yield f"backbone: {len(report.backbone)} atoms, {origin}"

    distribution = report.distribution
    if distribution is not None:
        yield (
            f"g(r): first peak {distribution.first_peak_height:.2f} at "
            f"{distribution.first_peak_nm:.3f} nm, {distribution.n_pairs:,} "
            f"intermolecular pairs over {distribution.n_frames} frame(s)"
        )
    structure = report.structure
    if structure is not None:
        if structure.first_peak_per_nm > 0.0:
            yield (
                f"S(q): peak at {structure.first_peak_per_nm:.1f} /nm; nothing "
                f"below {structure.q_min_per_nm:.1f} /nm is resolvable in this cell"
            )
        else:
            yield f"S(q): no resolvable peak above {structure.q_min_per_nm:.1f} /nm"

    conformation = report.conformation
    if conformation is not None:
        mean = conformation.mean
        line = (
            f"chains: <R^2> = {mean.mean_squared_end_to_end_nm2:.3f} nm2, "
            f"Rg = {mean.mean_radius_of_gyration_nm:.3f} nm, "
            f"C = {mean.characteristic_ratio:.2f} against an expected "
            f"{mean.expected_characteristic_ratio:.2f}"
        )
        if not mean.consistent:
            line += " - not consistent with a relaxed melt"
        if conformation.settled is not None:
            line += (
                ", <R^2> settled"
                if conformation.settled.equilibrated
                else ", <R^2> still moving"
            )
        yield line

    if report.persistence is not None:
        yield _persistence_line(report.persistence)

    displacement = report.displacement
    if displacement is not None:
        if displacement.diffusion_coefficient_cm2_s is None:
            yield (
                f"MSD: slope {displacement.log_slope:.2f}, not diffusive, so no "
                "diffusion coefficient"
            )
        else:
            yield (
                f"MSD: slope {displacement.log_slope:.2f}, "
                f"D = {displacement.diffusion_coefficient_cm2_s:.3e} cm2/s"
            )

    relaxation = report.relaxation
    if relaxation is not None:
        if relaxation.relaxation_time_ps is None:
            yield (
                f"end-to-end: not decorrelated in {relaxation.trajectory_ps:.0f} "
                "ps; the relaxation time is longer than the run"
            )
        else:
            yield f"end-to-end: relaxes in {relaxation.relaxation_time_ps:.0f} ps"

    recorded = report.recorded_chains
    if recorded is not None:
        yield (
            "manifest recorded at the end of the run: "
            f"<R^2> = {recorded.mean_squared_end_to_end_nm2:.3f} nm2, "
            f"Rg = {recorded.mean_radius_of_gyration_nm:.3f} nm"
        )


def _persistence_line(persistence: Any) -> str:
    """One line for a persistence length, with the extrapolation caveat."""
    if persistence.regime == "rod_like":
        return (
            "persistence length: no decay along the chain "
            f"({persistence.contour_length_nm:.2f} nm contour), rod-like"
        )
    if persistence.regime == "unfitted":
        return (
            "persistence length: no persistence length could be fitted over a "
            f"{persistence.contour_length_nm:.2f} nm contour"
        )
    line = (
        f"persistence length: {persistence.persistence_length_nm:.3f} nm over "
        f"{persistence.n_bonds} bonds ({persistence.contour_length_nm:.2f} nm "
        "contour)"
    )
    if persistence.regime == "extrapolated":
        line += " - never decayed to 1/e within the chain, so this is an extrapolation"
    return line
