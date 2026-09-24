"""A common entry point for measured properties with an imposed loading rate."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .elastic_rates import (
    ELASTIC_RATE_PROPERTIES,
    analyse_elastic_rates,
    run_elastic_rate_scan,
    validate_elastic_rate_scan,
)
from .mechanical import ModulusSpec
from .modulus_rates import (
    ModulusRateReport,
    analyse_modulus_rates,
    run_modulus_rate_scan,
    validate_modulus_rate_scan,
)
from .rate_dependence import (
    RateObservation,
    RateProperty,
    RateReport,
    analyse_rate_observations,
)
from .simulate import RunContext
from .tensile import BreakingSpec, ElongationSpec, YieldSpec
from .tensile_rates import (
    TENSILE_RATE_PROPERTIES,
    analyse_tensile_rates,
    run_tensile_rate_scan,
    validate_tensile_rate_scan,
)
from .tg import TgSpec
from .thermal_rates import (
    THERMAL_RATE_PROPERTIES,
    analyse_thermal_rates,
    run_thermal_rate_scan,
    validate_thermal_rate_scan,
)
from .tm import TmSpec

RATE_PROPERTIES = {
    "youngs_modulus": RateProperty(
        "youngs_modulus", "Young's modulus", "MPa", "strain/ns", trend="increasing"
    ),
    **ELASTIC_RATE_PROPERTIES,
    **TENSILE_RATE_PROPERTIES,
    **THERMAL_RATE_PROPERTIES,
}
RateScanSpec = ModulusSpec | BreakingSpec | ElongationSpec | YieldSpec | TgSpec | TmSpec


def default_rate_spec(property_name: str) -> RateScanSpec:
    """Return the ordinary measurement settings for a registered property."""
    if property_name == "youngs_modulus" or property_name in ELASTIC_RATE_PROPERTIES:
        return ModulusSpec()
    if property_name in ("yield_strength", "yield_strain"):
        return YieldSpec()
    if property_name == "breaking_strength":
        return BreakingSpec()
    if property_name == "elongation_at_break":
        return ElongationSpec()
    if property_name == "glass_transition":
        return TgSpec()
    if property_name == "melting_temperature":
        return TmSpec()
    raise ValueError(
        f"Unknown rate property {property_name!r}; choose {tuple(RATE_PROPERTIES)}."
    )


def _young_report(
    report: ModulusRateReport, target_rate: float, maximum: float
) -> RateReport:
    observations = [
        RateObservation(
            rate=float(fit.strain_rate_per_ns),
            value=fit.modulus_mpa,
            standard_error=fit.standard_error_mpa,
            resolved=fit.resolved,
            temperature_k=fit.temperature_k,
            conditions={"strain_limit": fit.strain_limit},
            notes=report.notes,
        )
        for fit in report.fits
        if fit.strain_rate_per_ns is not None
    ]
    return analyse_rate_observations(
        observations,
        property=RATE_PROPERTIES["youngs_modulus"],
        target_rate=target_rate,
        max_extrapolation_decades=maximum,
        run_dirs=report.run_dirs,
    )


def analyse_property_rates(
    run_dirs: Sequence[str | Path],
    *,
    property_name: str,
    target_rate: float,
    strain_limit: float = 0.015,
    max_extrapolation_decades: float = 2.0,
) -> RateReport:
    """Read a rate series in its property's units, preserving event censoring."""
    default_rate_spec(property_name)
    options: dict[str, Any] = {
        "property_name": property_name,
        "target_rate": target_rate,
        "max_extrapolation_decades": max_extrapolation_decades,
    }
    if property_name == "youngs_modulus":
        return _young_report(
            analyse_modulus_rates(
                run_dirs,
                target_rate_per_ns=target_rate,
                strain_limit=strain_limit,
                max_extrapolation_decades=max_extrapolation_decades,
            ),
            target_rate,
            max_extrapolation_decades,
        )
    if property_name in ELASTIC_RATE_PROPERTIES:
        return analyse_elastic_rates(run_dirs, strain_limit=strain_limit, **options)
    if property_name in TENSILE_RATE_PROPERTIES:
        return analyse_tensile_rates(run_dirs, **options)
    return analyse_thermal_rates(run_dirs, **options)


def validate_property_rate_scan(
    spec: RateScanSpec,
    hold_times_ps: Sequence[float],
    *,
    property_name: str,
    target_rate: float,
    n_replicas: int = 3,
    max_extrapolation_decades: float = 2.0,
    **equilibration: Any,
) -> Any:
    """Price every rate and replica, using the appropriate preparation protocol.

    ``n_replicas`` configures thermal scans; mechanical scans use their spec's
    ``n_replicas`` so their existing controls retain the same meaning.
    """
    expected = type(default_rate_spec(property_name))
    if type(spec) is not expected:
        raise ValueError(f"{property_name} requires {expected.__name__}.")
    options: dict[str, Any] = {
        "property_name": property_name,
        "target_rate": target_rate,
        "max_extrapolation_decades": max_extrapolation_decades,
        **equilibration,
    }
    if property_name == "youngs_modulus":
        assert isinstance(spec, ModulusSpec)
        return validate_modulus_rate_scan(
            spec,
            hold_times_ps,
            target_rate_per_ns=target_rate,
            max_extrapolation_decades=max_extrapolation_decades,
            **equilibration,
        )
    if property_name in ELASTIC_RATE_PROPERTIES:
        assert isinstance(spec, ModulusSpec)
        return validate_elastic_rate_scan(spec, hold_times_ps, **options)
    if property_name in TENSILE_RATE_PROPERTIES:
        assert isinstance(spec, (YieldSpec, BreakingSpec))
        return validate_tensile_rate_scan(spec, hold_times_ps, **options)
    assert isinstance(spec, (TgSpec, TmSpec))
    return validate_thermal_rate_scan(
        spec, hold_times_ps, n_replicas=n_replicas, **options
    )


def run_property_rate_scan(
    run: RunContext,
    output_dir: str | Path = "run",
    *,
    property_name: str,
    hold_times_ps: Sequence[float],
    target_rate: float,
    spec: RateScanSpec | None = None,
    n_replicas: int = 3,
    max_extrapolation_decades: float = 2.0,
    **options: Any,
) -> RateReport:
    """Run independent rate branches from shared preparation, then compare models.

    Tm retains its requirement for prepared crystalline coordinates and
    ``crystalline=True``. Remaining keywords go to the property's workflow.
    """
    selected = default_rate_spec(property_name) if spec is None else spec
    expected = type(default_rate_spec(property_name))
    if type(selected) is not expected:
        raise ValueError(f"{property_name} requires {expected.__name__}.")
    common: dict[str, Any] = {
        "property_name": property_name,
        "hold_times_ps": hold_times_ps,
        "target_rate": target_rate,
        "spec": selected,
        "max_extrapolation_decades": max_extrapolation_decades,
        **options,
    }
    if property_name == "youngs_modulus":
        assert isinstance(selected, ModulusSpec)
        return _young_report(
            run_modulus_rate_scan(
                run,
                output_dir,
                relax_ps=hold_times_ps,
                target_rate_per_ns=target_rate,
                spec=selected,
                max_extrapolation_decades=max_extrapolation_decades,
                **options,
            ),
            target_rate,
            max_extrapolation_decades,
        )
    if property_name in ELASTIC_RATE_PROPERTIES:
        return run_elastic_rate_scan(run, output_dir, **common)
    if property_name in TENSILE_RATE_PROPERTIES:
        return run_tensile_rate_scan(run, output_dir, **common)
    return run_thermal_rate_scan(run, output_dir, n_replicas=n_replicas, **common)
