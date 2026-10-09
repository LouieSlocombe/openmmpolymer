"""Pin the full stage records and workflow requests before consolidation (#40)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from openmmpolymer import (
    _workflow,
    elastic_rates,
    mechanical,
    tensile,
    tg,
    thermal_rates,
    tm,
    viscoelastic,
)
from openmmpolymer.protocols import (
    Protocol,
    _canonical,
    _stage_options,
    melt_quench,
    standard_melt_equilibration,
)

from .helpers import argon_context

# Recorded twice at the pre-consolidation baseline (audit Appendix A).
# Stage names seed the RNG and the full options/requests are compared on resume:
# changing one refuses the resume of every affected run already on disk.
RECORDED_BUILDERS_SHA256 = {
    "elastic_rate_plan bulk_modulus": "9308b0c4cf7dc357",
    "elastic_rate_plan load_modulus": "691cd44028f5ff9d",
    "elastic_rate_plan poisson_ratio": "5f68e777ee009d1d",
    "elastic_rate_plan shear_modulus": "7096f07711d8ef63",
    "elastic_rate_plan youngs_modulus": "5f68e777ee009d1d",
    "mechanical.deform_protocol r1": "ba5e1a8d6d39ab97",
    "mechanical_scan": "9308a799398744e8",
    "melt_quench": "71c740e121869fa9",
    "relaxation_scan": "7950cb1ec7ef20fb",
    "relaxation_scan +linearity": "73d0c3391e3f75e4",
    "request mechanical": "b23d209ba672543b",
    "request tg": "4e87700207f313a4",
    "request tm": "8d129746f33b9eaa",
    "request viscoelastic": "b791323344b0b56b",
    "standard_melt_equilibration": "0f0aa1230f5b8744",
    "tensile_protocol BreakingSpec r2": "4a90b874340b79d1",
    "tensile_protocol ElongationSpec r2": "68f095a751e8e871",
    "tensile_protocol YieldSpec r2": "cfffa477b4eb1b54",
    "tensile_scan BreakingSpec": "41bb7e24e5b7d663",
    "tensile_scan ElongationSpec": "b1648611c011e43d",
    "tensile_scan YieldSpec": "9058ea3bae5fcf35",
    "tg_coarse_scan": "ab806f52805ad1bf",
    "tg_fine_scan": "eac763867bc77d7e",
    "thermal_rate_plan glass_transition": "95dd619e68775ac6",
    "thermal_rate_plan melting_temperature": "cdcd436d5a4de410",
    "tm.melting_scan": "fedb89b842b91c28",
}


def _digest(value: Any) -> str:
    text = json.dumps(_canonical(value), sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _stages(protocol: Protocol) -> list[tuple[str, str, dict[str, Any]]]:
    """Include runner defaults, not only options explicitly set by the builder."""
    return [
        (stage.name, stage.kind, _stage_options(stage)) for stage in protocol.stages
    ]


def _request_digest(request: dict[str, Any]) -> str:
    # Only these machine/OpenMM-dependent fingerprint fields are omitted.
    fingerprint = {
        "system",
        "system_spec",
        "seed",
        "system_sha256",
        "coordinates_sha256",
        "box_nm",
    }
    return _digest(
        {key: value for key, value in request.items() if key not in fingerprint}
    )


def test_every_builder_and_workflow_request_keeps_its_recorded_digest() -> None:
    modulus = mechanical.ModulusSpec()
    relaxation = viscoelastic.RelaxationSpec()
    glass = tg.TgSpec()
    recorded = {
        "standard_melt_equilibration": _digest(_stages(standard_melt_equilibration())),
        "melt_quench": _digest(_stages(melt_quench())),
        "mechanical_scan": _digest(_stages(mechanical.mechanical_scan(modulus))),
        "mechanical.deform_protocol r1": _digest(
            _stages(
                mechanical.deform_protocol(
                    modulus,
                    timestep_fs=2.0,
                    replica=1,
                    reference_box_nm=(3.0, 3.1, 3.2),
                )
            )
        ),
        "relaxation_scan": _digest(_stages(viscoelastic.relaxation_scan(relaxation))),
        "relaxation_scan +linearity": _digest(
            _stages(
                viscoelastic.relaxation_scan(
                    replace(relaxation, linearity_strains=(0.01, 0.06))
                )
            )
        ),
        "tg_coarse_scan": _digest(_stages(tg.tg_coarse_scan(glass))),
        "tg_fine_scan": _digest(
            _stages(
                tg.tg_fine_scan(tg.nominal_fine_schedule(glass), glass, timestep_fs=2.0)
            )
        ),
        "tm.melting_scan": _digest(_stages(tm.melting_scan(tm.TmSpec()))),
    }
    for tensile_spec in (
        tensile.BreakingSpec(),
        tensile.ElongationSpec(),
        tensile.YieldSpec(),
    ):
        name = type(tensile_spec).__name__
        recorded[f"tensile_scan {name}"] = _digest(
            _stages(tensile.tensile_scan(tensile_spec))
        )
        recorded[f"tensile_protocol {name} r2"] = _digest(
            _stages(
                tensile.tensile_protocol(
                    tensile_spec,
                    timestep_fs=2.0,
                    replica=2,
                    reference_box_nm=(3.0, 3.0, 3.0),
                )
            )
        )

    holds = (50.0, 100.0, 200.0)
    for property_name in (
        "youngs_modulus",
        "poisson_ratio",
        "shear_modulus",
        "bulk_modulus",
        "load_modulus",
    ):
        elastic_plan = elastic_rates.validate_elastic_rate_scan(
            modulus, holds, property_name=property_name, target_rate=0.01
        )
        recorded[f"elastic_rate_plan {property_name}"] = _digest(
            [_stages(elastic_plan.equilibration)]
            + [
                [_stages(protocol) for protocol in group]
                for group in elastic_plan.protocols
            ]
        )
    for property_name, thermal_spec in (
        ("glass_transition", glass),
        ("melting_temperature", tm.TmSpec()),
    ):
        thermal_plan = thermal_rates.validate_thermal_rate_scan(
            thermal_spec,
            holds,
            property_name=property_name,
            target_rate=0.01,
            n_replicas=2,
        )
        recorded[f"thermal_rate_plan {property_name}"] = _digest(
            [_stages(thermal_plan.equilibration)]
            + [_stages(protocol) for protocol in thermal_plan.protocols]
        )

    run = argon_context(64, 2.4)
    chains = _workflow.chain_options(None, None, 7.0)
    recorded["request mechanical"] = _request_digest(
        _workflow.scan_request(
            run, modulus, mechanical.equilibration_protocol(modulus), **chains
        )
    )
    recorded["request viscoelastic"] = _request_digest(
        _workflow.scan_request(
            run, relaxation, viscoelastic.equilibration_protocol(relaxation), **chains
        )
    )
    recorded["request tg"] = _digest(_workflow.spec_request(glass, tg_approx_k=None))
    recorded["request tm"] = _request_digest(
        _workflow.spec_request(
            tm.TmSpec(),
            drop=("max_total_ns",),
            **_workflow.run_fingerprint(run, spec_key="system_spec"),
            state_sha256=None,
        )
    )
    assert recorded == RECORDED_BUILDERS_SHA256
