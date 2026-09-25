"""Reference-benchmark acceptance rules and an opt-in real polymer smoke run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from benchmarks import pe_melt
from benchmarks.pe_melt import REFERENCE_PATH, _protocol, comparison, run_benchmark
from openmmpolymer.chain import ChainSpec, assemble_chain
from openmmpolymer.protocols import ProtocolError

from .helpers import snapshot_files


@pytest.fixture
def reference() -> dict[str, Any]:
    case = json.loads(REFERENCE_PATH.read_text())
    return {
        **case["reference"],
        "temperature_k": 400.0,
        "density_g_cm3": 1.0 / case["reference"]["specific_volume_cm3_g"]["400"],
    }


@pytest.fixture
def measurements(reference: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "temperature_k": 400.0,
            "density_g_cm3": reference["density_g_cm3"] * scale,
            "density_equilibration": {"equilibrated": True},
            "chains_decorrelated": True,
        }
        for seed, scale in zip((11, 29, 47), (0.99, 1.0, 1.01), strict=True)
    ]


def test_benchmark_reference_matches_the_chain_formula() -> None:
    from rdkit.Chem import rdMolDescriptors

    case = json.loads(REFERENCE_PATH.read_text())
    chain = assemble_chain(
        ChainSpec(case["monomer_smiles"], case["degree_of_polymerization"])
    )
    assert rdMolDescriptors.CalcMolFormula(chain) == case["reference"]["molecule"]


def test_reference_protocol_uses_the_configured_production_length() -> None:
    protocol = _protocol(400.0, 1.01325, False, production_ps=12345.0)
    measurement = protocol.stages[-1]
    assert measurement.options["duration_ps"] == 12345.0
    assert measurement.options["temperature_k"] == 400.0
    assert measurement.options["pressure_bar"] == 1.01325


def test_smoke_protocol_is_short_but_records_multiple_real_frames() -> None:
    protocol = _protocol(400.0, 1.01325, True, production_ps=10000.0)
    measurement = protocol.stages[-1]
    assert measurement.options["duration_ps"] == 0.5
    assert measurement.options["trajectory"].interval_ps == 0.05


def test_reference_agreement_requires_supported_sampling(
    measurements: list[dict[str, Any]], reference: dict[str, Any]
) -> None:
    result = comparison(measurements, reference, smoke=False)
    assert result["status"] == "passed"
    assert result["density_g_cm3"] == pytest.approx(reference["density_g_cm3"])
    assert result["replica_standard_error_g_cm3"] == pytest.approx(
        reference["density_g_cm3"] * 0.01 / math.sqrt(3)
    )


def test_smoke_never_claims_a_scientific_pass(
    measurements: list[dict[str, Any]], reference: dict[str, Any]
) -> None:
    assert comparison(measurements, reference, smoke=True)["status"] == "smoke_only"


@pytest.mark.parametrize("reason", ("few_replicas", "same_seed", "density", "chains"))
def test_density_agreement_alone_does_not_pass(
    measurements: list[dict[str, Any]], reference: dict[str, Any], reason: str
) -> None:
    if reason == "few_replicas":
        measurements.pop()
    elif reason == "same_seed":
        measurements[1]["seed"] = measurements[0]["seed"]
    elif reason == "density":
        measurements[1]["density_equilibration"]["equilibrated"] = False
    else:
        measurements[1]["chains_decorrelated"] = False
    result = comparison(measurements, reference, smoke=False)
    assert result["all_replicas_within_range"]
    assert result["status"] == "unresolved"


@pytest.mark.parametrize("quantity", ("density_g_cm3", "temperature_k"))
def test_well_sampled_disagreement_is_a_failure(
    measurements: list[dict[str, Any]], reference: dict[str, Any], quantity: str
) -> None:
    measurements[0][quantity] *= 1.3
    result = comparison(measurements, reference, smoke=False)
    assert result["sampling_sufficient"]
    assert result["status"] == "failed"


@pytest.mark.parametrize("value", (math.nan, math.inf, 0.0, -1.0))
def test_invalid_density_cannot_produce_an_acceptance_result(
    measurements: list[dict[str, Any]], reference: dict[str, Any], value: float
) -> None:
    measurements[0]["density_g_cm3"] = value
    with pytest.raises(ValueError, match="positive and finite"):
        comparison(measurements, reference, smoke=False)


def test_empty_measurements_are_not_vacuously_within_the_reference(
    reference: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="At least one measured replica"):
        comparison([], reference, smoke=False)


def test_only_a_temperature_the_reference_covers_can_be_run(tmp_path: Path) -> None:
    """The allowed temperatures are the reference's own, read from its JSON."""
    with pytest.raises(ValueError, match="only at 350 K, 400 K"):
        run_benchmark(tmp_path / "benchmark", temperature_k=375)


@pytest.mark.parametrize("changed_system", (False, True))
def test_a_rerun_rebuilds_in_scratch_without_touching_the_replica(
    staged_melt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed_system: bool,
) -> None:
    """Rebuilding in scratch must preserve the evidence even when the check fails.

    The force-field cache the replicas share is part of that evidence: the
    rebuild works from a copy of it.
    """

    class ReachedProtocol(Exception):
        pass

    def protocol(protocol: Any, run: Any, directory: Path, **kwargs: Any) -> None:
        raise ReachedProtocol(directory)

    monkeypatch.setattr(pe_melt, "run_protocol", protocol)
    output = tmp_path / "benchmark"
    with pytest.raises(ReachedProtocol):
        run_benchmark(output, smoke=True, seeds=(11,), platform="CPU")
    original = snapshot_files(output)
    staged_melt["system_suffix"] = "\n" if changed_system else ""
    with pytest.raises(ProtocolError if changed_system else ReachedProtocol) as raised:
        run_benchmark(output, smoke=True, seeds=(11,), platform="CPU")
    assert snapshot_files(output) == original
    first, second = staged_melt["builds"]
    assert first == output / "seed_11" / "build"
    assert second != first
    assert not second.exists()
    if not changed_system:
        assert raised.value.args == (output / "seed_11",)


@pytest.mark.slow
@pytest.mark.forcefield
@pytest.mark.packmol
def test_real_polyethylene_benchmark_smoke(tmp_path: Path) -> None:
    """Exercise a real C44H90 force field, packing, dynamics, and analysis."""
    report = run_benchmark(
        tmp_path / "benchmark", smoke=True, seeds=(11,), platform="CPU"
    )
    assert report["comparison"]["status"] == "smoke_only"
    assert not report["comparison"]["sampling_sufficient"]
    assert report["replicas"][0]["n_atoms"] == 4020
    assert report["replicas"][0]["frames"] == 10
    assert math.isfinite(report["replicas"][0]["density_g_cm3"])
    assert json.loads((tmp_path / "benchmark" / "benchmark.json").read_text())
    build = tmp_path / "benchmark" / "seed_11" / "build"
    original = snapshot_files(build)
    repeated = run_benchmark(
        tmp_path / "benchmark", smoke=True, seeds=(11,), platform="CPU"
    )
    assert repeated["comparison"] == report["comparison"]
    assert snapshot_files(build) == original
