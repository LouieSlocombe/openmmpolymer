"""Recorded rate requests remain byte-compatible through orchestration changes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openmmpolymer import elastic_rates, tensile_rates, thermal_rates
from openmmpolymer._workflow import rate_request
from openmmpolymer.simulate import RunContext

# Captured before the issue #40 rate-runner consolidation. Keep these independent
# of the normalization helper and include insertion order in the serialized oracle.
REQUEST_DIGESTS = {
    "youngs_modulus": "0287c78485b8ed1b196f5fa6595d51e785cfd80051acfaedf03e69f400d5ffae",
    "poisson_ratio": "8331835fa991d16133368f1f23f9da482543d6b8efd95decaf66ff81059b9894",
    "shear_modulus": "1600272888307808e11785d71d23512fa6e99b31d96b645668878ed73ea0e745",
    "bulk_modulus": "d8348c55314ffc4db918a73e9bab18e4f846fb25436eb0ba5bae54172ffdabcb",
    "load_modulus": "26e9e647c25aec3991ff73f01c52ce7fb09131fc81a4ccbd1babf6e279db93c3",
    "yield_strength": "b702c560c43247fcee471f9dcea6433d8bf313b4e81c4c95bd059e099ee5ee89",
    "yield_strain": "331f9d1b88706a3da580aa44dba034e178efa894c049319b18a278a18350464a",
    "breaking_strength": "3008c86dbbe770b62254c4d94e3c6837fda74dabd7df707e0298698bfd5e5730",
    "elongation_at_break": "d95bd9492584e237485a4e5c193255bd9bd21eccd3063c149923690bb04784f6",
    "glass_transition": "b23178c1ddd9682e2f69fbb2e13f5f99bdb7cd752b2fa8847dc68edb6dcaa7b7",
    "melting_temperature": "36c20c639762dbd92058ce6c558c8c2b2329a17ddd8007fa2568912eb864c826",
}


class RequestCaptured(Exception):
    """Stop a public runner before it writes or starts dynamics."""


@pytest.mark.parametrize(
    ("module", "runner", "property_name"),
    [
        (module, runner, name)
        for module, runner, names in (
            (
                elastic_rates,
                elastic_rates.run_elastic_rate_scan,
                (
                    "youngs_modulus",
                    "poisson_ratio",
                    "shear_modulus",
                    "bulk_modulus",
                    "load_modulus",
                ),
            ),
            (
                tensile_rates,
                tensile_rates.run_tensile_rate_scan,
                (
                    "yield_strength",
                    "yield_strain",
                    "breaking_strength",
                    "elongation_at_break",
                ),
            ),
            (
                thermal_rates,
                thermal_rates.run_thermal_rate_scan,
                ("glass_transition", "melting_temperature"),
            ),
        )
        for name in names
    ],
)
def test_rate_request_bytes_preserve_the_recorded_baseline(
    module: ModuleType,
    runner: Callable[..., Any],
    property_name: str,
    argon_run: RunContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def capture(
        run: RunContext,
        workflow: Path,
        request: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        serialized = json.dumps(request, indent=2, allow_nan=False).encode()
        assert hashlib.sha256(serialized).hexdigest() == REQUEST_DIGESTS[property_name]
        raise RequestCaptured

    monkeypatch.setattr(module, "resumable_record", capture)
    options = {"crystalline": True, "n_replicas": 2} if module is thermal_rates else {}
    with pytest.raises(RequestCaptured):
        runner(
            argon_run,
            tmp_path,
            property_name=property_name,
            hold_times_ps=(50.0, 100.0, 200.0),
            target_rate=0.01,
            **options,
        )
    assert not tuple(tmp_path.iterdir())


def test_rate_request_preserves_key_order_and_stringifies_other_values() -> None:
    fields = {
        "path": Path("recorded-state.xml"),
        "ladder": (1.0, 2.0),
        "optional": None,
    }
    assert json.dumps(rate_request(fields)) == (
        '{"path": "recorded-state.xml", "ladder": [1.0, 2.0], "optional": null}'
    )
    assert fields["ladder"] == (1.0, 2.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rate_request_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        rate_request({"nested": [value]})
