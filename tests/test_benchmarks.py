"""Reference-benchmark acceptance rules and an opt-in real polymer smoke run."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks import pe_melt
from benchmarks.pe_melt import REFERENCE_PATH, _protocol, comparison, run_benchmark
from openmmpolymer.chain import ChainSpec, assemble_chain
from openmmpolymer.protocols import (
    ProtocolError,
    RunManifest,
    _run_identity,
    record_build_request,
)


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
    assert (
        case["reference"]["kind"] == "published united-atom simulation, not experiment"
    )
    assert "not a published uncertainty" in case["reference"]["tolerance_basis"]


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


@pytest.mark.parametrize("changed_system", (False, True))
def test_benchmark_replay_validates_identity_without_overwriting_build_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argon_run: Any,
    changed_system: bool,
) -> None:
    """Rebuilding in scratch must preserve evidence even when validation fails."""
    from openmm import XmlSerializer, unit

    output = tmp_path / "benchmark"
    record_build_request(
        output,
        {
            "case": json.loads(REFERENCE_PATH.read_text()),
            "temperature_k": 400,
            "smoke": True,
            "seeds": (11,),
        },
    )
    replica = output / "seed_11"
    (replica / "build").mkdir(parents=True)
    (replica / "build" / "chain.sdf").write_text("original molecule")
    (replica / "build" / "polymer_ff.xml").write_text("original parameters")
    (replica / "build" / "packed.pdb").write_text("original packing")
    RunManifest(
        protocol="pe_density_smoke",
        seed=11,
        provenance={"version": 1, "run": _run_identity(argon_run), "stages": {}},
    ).save(replica)
    original = {
        path.relative_to(replica): path.read_bytes()
        for path in replica.rglob("*")
        if path.is_file()
    }
    prepared = argon_run
    if changed_system:
        system = XmlSerializer.deserialize(argon_run.system_xml)
        system.setParticleMass(0, system.getParticleMass(0) + 1.0 * unit.dalton)
        prepared = replace(argon_run, system_xml=XmlSerializer.serialize(system))
    scratch_builds = []

    def build(spec: Any, *, output_dir: Path, **kwargs: Any) -> Any:
        scratch_builds.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        source = output_dir / "chain.sdf"
        source.write_text("rebuilt molecule")
        return SimpleNamespace(
            sdf_paths=(str(source),),
            pdb_paths=(str(source),),
            molar_mass_g_mol=100.0,
            backbone=(0, 1),
            n_atoms=2,
        )

    def forcefield(source: str, destination: Path, **kwargs: Any) -> Any:
        destination.write_text("rebuilt parameters")
        return prepared.forcefield

    def packing(components: Any, edge: float, destination: Path, **kwargs: Any) -> Any:
        destination.write_text("rebuilt packing")
        return SimpleNamespace(packed_pdb=str(destination), box_nm=(edge,) * 3)

    class ReachedProtocol(Exception):
        pass

    def protocol(*args: Any, **kwargs: Any) -> None:
        assert args[2] == replica
        raise ReachedProtocol("identity accepted")

    monkeypatch.setattr(pe_melt, "build_chain", build)
    monkeypatch.setattr(pe_melt, "assign_charges", lambda *args: None)
    monkeypatch.setattr(pe_melt, "build_polymer_forcefield", forcefield)
    monkeypatch.setattr(pe_melt, "check_target_density", lambda *args: None)
    monkeypatch.setattr(pe_melt, "pack_box", packing)
    monkeypatch.setattr(pe_melt, "assemble_box", lambda *args: prepared.box)
    monkeypatch.setattr(pe_melt, "check_packing", lambda *args: None)
    monkeypatch.setattr(pe_melt, "prepare_box", lambda box, forcefield: box)
    monkeypatch.setattr(pe_melt, "prepare_run", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(pe_melt, "run_protocol", protocol)
    with pytest.raises(
        ProtocolError if changed_system else ReachedProtocol,
        match="starting inputs changed" if changed_system else "identity accepted",
    ):
        run_benchmark(output, smoke=True, seeds=(11,), platform="CPU")
    assert {
        path.relative_to(replica): path.read_bytes()
        for path in replica.rglob("*")
        if path.is_file()
    } == original
    assert scratch_builds and scratch_builds[0] != replica / "build"
    assert not scratch_builds[0].exists()


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
    original_build = {path.name: path.read_bytes() for path in build.iterdir()}
    repeated = run_benchmark(
        tmp_path / "benchmark", smoke=True, seeds=(11,), platform="CPU"
    )
    assert repeated["comparison"] == report["comparison"]
    assert {path.name: path.read_bytes() for path in build.iterdir()} == original_build
