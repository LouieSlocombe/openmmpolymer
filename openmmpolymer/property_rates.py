"""A common entry point for measured properties with an imposed loading rate.

Each property belongs to a family - elastic, tensile or thermal - whose module
knows its loading ladder, its scan record and how to read it back. This one
only looks up the family and the settings type a property takes, and hands the
request over.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from .elastic_rates import (
    ELASTIC_RATE_PROPERTIES,
    ElasticRatePlan,
    analyse_elastic_rates,
    run_elastic_rate_scan,
    validate_elastic_rate_scan,
)
from .mechanical import ModulusSpec
from .rate_dependence import RateReport
from .simulate import RunContext
from .tensile import BreakingSpec, ElongationSpec, YieldSpec
from .tensile_rates import (
    TENSILE_RATE_PROPERTIES,
    TensileRatePlan,
    analyse_tensile_rates,
    run_tensile_rate_scan,
    validate_tensile_rate_scan,
)
from .tg import TgSpec
from .thermal_rates import (
    THERMAL_RATE_PROPERTIES,
    ThermalRatePlan,
    analyse_thermal_rates,
    run_thermal_rate_scan,
    validate_thermal_rate_scan,
)
from .tm import TmSpec

RATE_PROPERTIES = {
    **ELASTIC_RATE_PROPERTIES,
    **TENSILE_RATE_PROPERTIES,
    **THERMAL_RATE_PROPERTIES,
}
RateScanSpec = ModulusSpec | BreakingSpec | ElongationSpec | YieldSpec | TgSpec | TmSpec
RatePlan = ElasticRatePlan | TensileRatePlan | ThermalRatePlan


class _Family(NamedTuple):
    """How one module's properties are priced, run and read.

    ``shared`` names the generic functions' settings that this family takes
    as its own: the strain window elastic fits use, and the replica count of
    a thermal scan - mechanical scans keep theirs in the spec.
    """

    validate: Callable[..., RatePlan]
    run: Callable[..., RateReport]
    analyse: Callable[..., RateReport]
    shared: tuple[str, ...] = ()


_ELASTIC = _Family(
    validate_elastic_rate_scan,
    run_elastic_rate_scan,
    analyse_elastic_rates,
    ("strain_limit",),
)
_TENSILE = _Family(
    validate_tensile_rate_scan, run_tensile_rate_scan, analyse_tensile_rates
)
_THERMAL = _Family(
    validate_thermal_rate_scan,
    run_thermal_rate_scan,
    analyse_thermal_rates,
    ("n_replicas",),
)
_FAMILIES = {
    **dict.fromkeys(ELASTIC_RATE_PROPERTIES, _ELASTIC),
    **dict.fromkeys(TENSILE_RATE_PROPERTIES, _TENSILE),
    **dict.fromkeys(THERMAL_RATE_PROPERTIES, _THERMAL),
}
_SPECS: dict[str, type[RateScanSpec]] = {
    **dict.fromkeys(ELASTIC_RATE_PROPERTIES, ModulusSpec),
    "yield_strength": YieldSpec,
    "yield_strain": YieldSpec,
    "breaking_strength": BreakingSpec,
    "elongation_at_break": ElongationSpec,
    "glass_transition": TgSpec,
    "melting_temperature": TmSpec,
}


def default_rate_spec(property_name: str) -> RateScanSpec:
    """Return the ordinary measurement settings for a registered property."""
    try:
        return _SPECS[property_name]()
    except KeyError:
        raise ValueError(
            f"Unknown rate property {property_name!r}; choose {tuple(RATE_PROPERTIES)}."
        ) from None


def _family(property_name: str, spec: RateScanSpec | None = None) -> _Family:
    """The family measuring *property_name*, refusing settings of another type."""
    expected = type(default_rate_spec(property_name))
    if spec is not None and type(spec) is not expected:
        raise ValueError(f"{property_name} requires {expected.__name__}.")
    return _FAMILIES[property_name]


def _shared(family: _Family, **settings: Any) -> dict[str, Any]:
    return {name: value for name, value in settings.items() if name in family.shared}


def analyse_property_rates(
    run_dirs: Sequence[str | Path],
    *,
    property_name: str,
    target_rate: float,
    strain_limit: float = 0.015,
    max_extrapolation_decades: float = 2.0,
) -> RateReport:
    """Read a rate series in its property's units, preserving event censoring.

    ``strain_limit`` is the window elastic moduli and ratios are fitted over;
    the other properties carry their own criteria in their saved scans.
    """
    family = _family(property_name)
    return family.analyse(
        run_dirs,
        property_name=property_name,
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
        **_shared(family, strain_limit=strain_limit),
    )


def validate_property_rate_scan(
    spec: RateScanSpec,
    hold_times_ps: Sequence[float],
    *,
    property_name: str,
    target_rate: float,
    n_replicas: int = 3,
    max_extrapolation_decades: float = 2.0,
    **equilibration: Any,
) -> RatePlan:
    """Price every rate and replica, using the appropriate preparation protocol.

    ``n_replicas`` configures thermal scans; mechanical scans use their spec's
    ``n_replicas`` so their existing controls retain the same meaning.
    """
    family = _family(property_name, spec)
    return family.validate(
        spec,
        hold_times_ps,
        property_name=property_name,
        target_rate=target_rate,
        max_extrapolation_decades=max_extrapolation_decades,
        **_shared(family, n_replicas=n_replicas),
        **equilibration,
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
    family = _family(property_name, spec)
    return family.run(
        run,
        output_dir,
        property_name=property_name,
        hold_times_ps=hold_times_ps,
        target_rate=target_rate,
        spec=default_rate_spec(property_name) if spec is None else spec,
        max_extrapolation_decades=max_extrapolation_decades,
        **_shared(family, n_replicas=n_replicas),
        **options,
    )
