"""One entry point dispatches every rate property to its family, unchanged."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmmpolymer import elastic_rates, rate_dependence
from openmmpolymer.elastic_rates import YOUNGS_WORKFLOW_NAME
from openmmpolymer.mdsystem import SystemSpec
from openmmpolymer.mechanical import ModulusSpec
from openmmpolymer.property_rates import (
    RATE_PROPERTIES,
    analyse_property_rates,
    default_rate_spec,
    run_property_rate_scan,
    validate_property_rate_scan,
)
from openmmpolymer.tensile import BreakingSpec, ElongationSpec, YieldSpec
from openmmpolymer.tg import TgSpec
from openmmpolymer.thermal_rates import ThermalRatePlan
from openmmpolymer.tm import TmSpec

from .helpers import (
    fake_scan_dynamics,
    planted_extension_runner,
    write_modulus_rate_series,
)

SPECS = {
    "youngs_modulus": ModulusSpec,
    "poisson_ratio": ModulusSpec,
    "shear_modulus": ModulusSpec,
    "bulk_modulus": ModulusSpec,
    "load_modulus": ModulusSpec,
    "yield_strength": YieldSpec,
    "yield_strain": YieldSpec,
    "breaking_strength": BreakingSpec,
    "elongation_at_break": ElongationSpec,
    "glass_transition": TgSpec,
    "melting_temperature": TmSpec,
}


def test_every_registered_property_has_its_settings_and_rate_unit() -> None:
    assert list(RATE_PROPERTIES) == list(SPECS)
    for name, spec_type in SPECS.items():
        assert type(default_rate_spec(name)) is spec_type
        assert RATE_PROPERTIES[name].name == name
        assert RATE_PROPERTIES[name].rate_unit in {"strain/ns", "bar/ns", "K/ns"}


@pytest.mark.parametrize("property_name", list(SPECS))
def test_every_property_is_priced_by_its_own_family(property_name: str) -> None:
    plan = validate_property_rate_scan(
        default_rate_spec(property_name),
        (10.0, 20.0, 30.0),
        property_name=property_name,
        target_rate=0.001,
        n_replicas=2,
    )
    assert plan.total_ns > 0
    assert plan.property_name == property_name
    # A thermal scan takes its replica count here; the others keep theirs in the spec.
    if isinstance(plan, ThermalRatePlan):
        assert plan.n_replicas == 2


@pytest.mark.parametrize(
    "holds",
    [
        (1.0, 2.0),
        (1.0, 1.0, 2.0),
        (1.0, 1.0 + 1e-9, 2.0),
        (0.0, 1.0, 2.0),
        (-1.0, 1.0, 2.0),
        (1.0, math.nan, 2.0),
        (1.0, 2.0, math.inf),
    ],
)
def test_hold_times_must_be_three_distinct_positive_values(
    holds: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        validate_property_rate_scan(
            ModulusSpec(), holds, property_name="youngs_modulus", target_rate=0.001
        )


def test_unknown_property_or_wrong_spec_cannot_dispatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown rate property"):
        default_rate_spec("density")
    with pytest.raises(ValueError, match="Unknown rate property"):
        analyse_property_rates([tmp_path], property_name="density", target_rate=0.1)
    with pytest.raises(ValueError, match="requires YieldSpec"):
        validate_property_rate_scan(
            TgSpec(), (1, 2, 3), property_name="yield_strength", target_rate=0.1
        )
    with pytest.raises(ValueError, match="requires ModulusSpec"):
        run_property_rate_scan(
            object(),  # type: ignore[arg-type]
            tmp_path / "unstarted",
            property_name="youngs_modulus",
            hold_times_ps=(1, 2, 3),
            target_rate=0.1,
            spec=YieldSpec(),
        )
    assert not (tmp_path / "unstarted").exists()


def test_saved_youngs_moduli_are_read_through_the_common_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted: list[str] = []
    fit = rate_dependence.rate_extrapolation

    def counting(*args: Any, **kwargs: Any) -> Any:
        fitted.append(kwargs["form"])
        return fit(*args, **kwargs)

    monkeypatch.setattr(rate_dependence, "rate_extrapolation", counting)
    report = analyse_property_rates(
        write_modulus_rate_series(tmp_path),
        property_name="youngs_modulus",
        target_rate=0.001,
        strain_limit=0.02,
    )
    # One pooled observation per rate, and each model fitted once.
    assert fitted == ["log_linear", "power_law"]
    assert report.property is RATE_PROPERTIES["youngs_modulus"]
    assert len(report.observations) == 3
    assert all(
        item.conditions == {"strain_limit": 0.02} for item in report.observations
    )
    assert report.log_linear is not None
    assert report.log_linear.value == pytest.approx(1700.0)
    assert report.log_linear.resolved


def test_a_youngs_scan_runs_and_reads_back_through_the_common_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    fake_scan_dynamics(monkeypatch, elastic_rates, planted_extension_runner(calls))
    run: Any = SimpleNamespace(
        spec=SystemSpec(),
        seed=11,
        system_xml="system",
        box=SimpleNamespace(positions_nm=np.zeros((2, 3)), box_nm=(5.0, 5.0, 5.0)),
    )
    spec = ModulusSpec(n_replicas=2, max_strain=0.02)
    report = run_property_rate_scan(
        run,
        tmp_path,
        property_name="youngs_modulus",
        hold_times_ps=(50.0, 150.0, 500.0),
        target_rate=0.001,
        spec=spec,
    )
    assert len(calls) == 1 + 3 * spec.n_replicas
    record = json.loads((tmp_path / YOUNGS_WORKFLOW_NAME).read_text())
    assert record["request"]["relax_ps"] == [50.0, 150.0, 500.0]
    assert record["request"]["target_rate_per_ns"] == 0.001
    assert report.run_dirs == tuple(
        str(tmp_path.resolve() / name) for name in record["run_dirs"]
    )
    assert report.log_linear is not None
    assert report.log_linear.value == pytest.approx(1700.0)
    assert all(not item.notes for item in report.observations)
    again = analyse_property_rates(
        [tmp_path], property_name="youngs_modulus", target_rate=0.001
    )
    assert again.log_linear is not None
    assert again.log_linear.value == report.log_linear.value
