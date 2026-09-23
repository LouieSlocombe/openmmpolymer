"""Rate workflows preserve preparation, distinct rates and replica quality."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmmpolymer import modulus_rates
from openmmpolymer.mdsystem import SystemSpec
from openmmpolymer.mechanical import MechanicalError, ModulusSpec, deform_schedule
from openmmpolymer.modulus_rates import (
    WORKFLOW_NAME,
    analyse_modulus_rates,
    run_modulus_rate_scan,
    validate_modulus_rate_scan,
)
from openmmpolymer.protocols import Protocol
from openmmpolymer.trajectory import AnalysisError

from .helpers import write_deformation

HOLDS = (50.0, 150.0, 500.0)
SPEC = ModulusSpec(n_replicas=2, max_strain=0.02)


def _series(root: Path) -> list[Path]:
    directories: list[Path] = []
    for index, hold in enumerate(HOLDS):
        rate = ((1.002) ** 10 - 1.0) / (10 * hold) * 1000
        directory = root / f"rate_{index:02d}"
        write_deformation(
            directory,
            modulus_mpa=2000.0 + 300.0 * math.log10(rate / 0.01),
            relax_ps=hold,
        )
        directories.append(directory)
    return directories


def test_saved_rates_recover_a_planted_logarithmic_law(tmp_path: Path) -> None:
    directories = _series(tmp_path)
    report = analyse_modulus_rates(directories, target_rate_per_ns=0.001)
    assert report.log_linear.modulus_mpa == pytest.approx(1700.0)
    assert report.log_linear.sensitivity_mpa_per_decade == pytest.approx(300.0)
    assert report.log_linear.resolved
    assert len(report.fits) == 3
    assert all(fit.resolved for fit in report.fits)
    assert any("one replica" in note for note in report.notes)
    rates = [float(fit.strain_rate_per_ns or 0) for fit in report.fits]
    assert rates == sorted(rates)


def test_same_rate_directories_are_replicas_not_extra_rate_observations(
    tmp_path: Path,
) -> None:
    directories = _series(tmp_path)
    duplicate = tmp_path / "replica"
    write_deformation(duplicate, relax_ps=HOLDS[-1], modulus_mpa=2000.0)
    report = analyse_modulus_rates([*directories, duplicate], target_rate_per_ns=0.001)
    assert len(report.fits) == 3
    assert report.log_linear.n_rates == 3
    slower_original = 2000 + 300 * math.log10(((1.002) ** 10 - 1) / 5 / 0.01)
    assert report.fits[0].modulus_mpa == pytest.approx((slower_original + 2000) / 2)
    assert (
        report.fits[0].standard_error_mpa
        >= np.std([slower_original, 2000], ddof=1) - 1.0e-8
    )


def test_excessive_replica_spread_remains_unresolved(tmp_path: Path) -> None:
    directories = _series(tmp_path)
    write_deformation(
        directories[0],
        stage="06_deform_r1_00",
        modulus_mpa=6000.0,
        relax_ps=HOLDS[0],
    )
    report = analyse_modulus_rates(directories, target_rate_per_ns=0.001)
    assert not report.fits[-1].resolved
    assert report.fits[-1].standard_error_mpa > 2000.0
    assert not report.log_linear.resolved
    assert not report.power_law.resolved
    assert any("replica quality and spread" in note for note in report.notes)


@pytest.mark.parametrize(
    "changed",
    [{"axis": 0}, {"increment": 0.001}, {"temperature_k": 310.0}],
)
def test_saved_rates_must_measure_comparable_deformations(
    tmp_path: Path, changed: dict[str, Any]
) -> None:
    directories = _series(tmp_path)
    write_deformation(directories[-1], relax_ps=HOLDS[-1], **changed)
    with pytest.raises(AnalysisError, match="same"):
        analyse_modulus_rates(directories, target_rate_per_ns=0.001)


def test_missing_recorded_rate_and_repeated_directories_are_refused(
    tmp_path: Path,
) -> None:
    directories = _series(tmp_path)
    path = directories[0] / "manifest.json"
    record = json.loads(path.read_text())
    del record["stages"]["06_deform_r0_00"]["samples"]["segment_duration_ps"]
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="positive strain rate"):
        analyse_modulus_rates(directories, target_rate_per_ns=0.001)
    with pytest.raises(AnalysisError, match="more than once"):
        analyse_modulus_rates(
            [directories[1], directories[1]], target_rate_per_ns=0.001
        )


def test_scan_root_expands_only_its_recorded_directories(tmp_path: Path) -> None:
    directories = _series(tmp_path)
    (tmp_path / WORKFLOW_NAME).write_text(
        json.dumps({"run_dirs": [directory.name for directory in directories]})
    )
    write_deformation(tmp_path / "unrelated", relax_ps=123.0)
    report = analyse_modulus_rates([tmp_path], target_rate_per_ns=0.001)
    assert len(report.run_dirs) == 3
    (directories[1] / "manifest.json").unlink()
    with pytest.raises(AnalysisError, match="No completed rate manifest"):
        analyse_modulus_rates([tmp_path], target_rate_per_ns=0.001)


@pytest.mark.parametrize(
    "holds", [(50.0, 100.0), (50.0, 50.0, 100.0), (0.0, 1.0, 2.0), (1.0, 2.0, math.inf)]
)
def test_rate_validation_precedes_dynamics(holds: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        validate_modulus_rate_scan(SPEC, holds, target_rate_per_ns=0.001)


def test_budget_counts_all_rates_and_replicas() -> None:
    plan = validate_modulus_rate_scan(SPEC, HOLDS, target_rate_per_ns=0.001)
    expected = plan.equilibration.total_duration_ps + SPEC.n_replicas * sum(
        deform_schedule(replace(SPEC, relax_ps=hold)).total_ps for hold in HOLDS
    )
    assert plan.total_ns == pytest.approx(expected / 1000)
    with pytest.raises(MechanicalError, match="all rates and replicas"):
        validate_modulus_rate_scan(
            replace(SPEC, max_total_ns=plan.total_ns - 0.001),
            HOLDS,
            target_rate_per_ns=0.001,
        )


@pytest.fixture
def fake_dynamics(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def run_protocol(
        protocol: Protocol, run: Any, directory: Path, **kwargs: Any
    ) -> Any:
        calls.append({"protocol": protocol, "directory": directory, **kwargs})
        directory.mkdir(parents=True, exist_ok=True)
        final_state = directory / "state.xml"
        final_state.write_text("state")
        if protocol.stages[0].kind == "deform":
            assert len(protocol.stages) == 1
            stage = protocol.stages[0]
            options = stage.options
            hold = options["relax_ps"]
            count = options["n_steps"]
            increment = options["strain_increment"]
            rate = ((1 + increment) ** count - 1) / (count * hold) * 1000
            write_deformation(
                directory,
                modulus_mpa=2000.0 + 300 * math.log10(rate / 0.01),
                n_steps=count,
                increment=increment,
                relax_ps=hold,
                stage=stage.name,
            )
        return SimpleNamespace(final_state=str(final_state))

    monkeypatch.setattr(modulus_rates, "run_protocol", run_protocol)
    monkeypatch.setattr(modulus_rates, "_equilibrated_box_nm", lambda path: [5.0] * 3)
    return calls


def test_every_rate_and_replica_branches_from_one_equilibrated_state(
    tmp_path: Path, fake_dynamics: list[dict[str, Any]]
) -> None:
    run: Any = SimpleNamespace(spec=SystemSpec(), seed=11)
    report = run_modulus_rate_scan(
        run, tmp_path, spec=SPEC, relax_ps=HOLDS, target_rate_per_ns=0.001
    )
    assert len(fake_dynamics) == 1 + 3 * SPEC.n_replicas
    branches = fake_dynamics[1:]
    first_stage_names = [call["protocol"].stages[0].name for call in branches]
    assert len(set(first_stage_names)) == len(first_stage_names)
    assert {call["state_in"] for call in branches} == {
        str(tmp_path / "equilibration" / "state.xml")
    }
    assert [call["protocol"].stages[0].options["relax_ps"] for call in branches] == [
        value for value in HOLDS for _ in range(SPEC.n_replicas)
    ]
    assert all(
        call["protocol"].stages[0].options["reference_box_nm"] == [5.0] * 3
        for call in branches
    )
    assert all(
        len(fit.strain_rate_per_ns) == 3
        for fit in (report.log_linear, report.power_law)
    )
    assert len(report.fits) == 3


def test_changed_hamiltonian_is_rejected_before_workflow_metadata_is_overwritten(
    tmp_path: Path, fake_dynamics: list[dict[str, Any]], argon_run: Any
) -> None:
    from openmmpolymer.protocols import ProtocolError, Stage
    from openmmpolymer.protocols import run_protocol as real_run_protocol

    run_modulus_rate_scan(
        argon_run, tmp_path, spec=SPEC, relax_ps=HOLDS, target_rate_per_ns=0.001
    )
    # Record real input provenance without running the expensive rate scan.
    real_run_protocol(
        Protocol("initial", (Stage("00_minimise", "minimise"),)),
        argon_run,
        tmp_path / "equilibration",
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    changed = replace(argon_run, system_xml=argon_run.system_xml + "\n")
    with pytest.raises(ProtocolError, match="starting inputs changed"):
        run_modulus_rate_scan(
            changed, tmp_path, spec=SPEC, relax_ps=HOLDS, target_rate_per_ns=0.001
        )
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "changed", [{"relax_ps": (50.0, 200.0, 500.0)}, {"npt_ps": 50.0}]
)
def test_changed_rates_or_equilibration_cannot_resume_an_interrupted_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: dict[str, Any]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(modulus_rates, "run_protocol", fail)
    run: Any = SimpleNamespace(spec=SystemSpec(), seed=11)
    options: dict[str, Any] = {
        "spec": SPEC,
        "relax_ps": HOLDS,
        "target_rate_per_ns": 0.001,
    }
    with pytest.raises(RuntimeError, match="interrupted"):
        run_modulus_rate_scan(run, tmp_path, **options)
    assert (tmp_path / WORKFLOW_NAME).is_file()
    with pytest.raises(MechanicalError, match="different settings"):
        run_modulus_rate_scan(run, tmp_path, **{**options, **changed})


def test_forced_rerun_keeps_earlier_replicas_in_each_rate_manifest(
    tmp_path: Path, fake_dynamics: list[dict[str, Any]]
) -> None:
    run: Any = SimpleNamespace(spec=SystemSpec(), seed=11)
    run_modulus_rate_scan(
        run, tmp_path, spec=SPEC, relax_ps=HOLDS, target_rate_per_ns=0.001, resume=False
    )
    assert [call["resume"] for call in fake_dynamics] == [
        False,
        False,
        True,
        False,
        True,
        False,
        True,
    ]


def test_budget_failure_creates_no_directory(tmp_path: Path) -> None:
    target = tmp_path / "unstarted"
    run: Any = SimpleNamespace(spec=SystemSpec(), seed=11)
    with pytest.raises(MechanicalError, match="budget"):
        run_modulus_rate_scan(
            run,
            target,
            spec=replace(SPEC, max_total_ns=0.001),
            relax_ps=HOLDS,
            target_rate_per_ns=0.001,
        )
    assert not target.exists()


def test_recorded_scan_refuses_an_incomplete_replica(
    tmp_path: Path, fake_dynamics: list[dict[str, Any]]
) -> None:
    run: Any = SimpleNamespace(spec=SystemSpec(), seed=11)
    run_modulus_rate_scan(
        run, tmp_path, spec=SPEC, relax_ps=HOLDS, target_rate_per_ns=0.001
    )
    manifest = tmp_path / "rate_00" / "manifest.json"
    record = json.loads(manifest.read_text())
    del record["stages"]["06_deform_r1_00"]
    manifest.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="incomplete"):
        analyse_modulus_rates([tmp_path], target_rate_per_ns=0.001)


@pytest.mark.slow
def test_real_rate_scan_retains_replicas_common_reference_and_resume(
    tmp_path: Path, argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Argon is not an elastic solid. Inspect real output rather than asking
    # its noisy extension slopes to support a positive modulus extrapolation.
    monkeypatch.setattr(
        modulus_rates, "analyse_modulus_rates", lambda *args, **kwargs: None
    )
    spec = ModulusSpec(
        temperature_k=120.0,
        strain_increment=0.004,
        max_strain=0.012,
        elastic_strain_limit=0.012,
        n_replicas=2,
        samples_per_step=4,
        stage_ps=0.3,
    )
    holds = (0.1, 0.2, 0.3)
    options: dict[str, Any] = {
        "spec": spec,
        "relax_ps": holds,
        "target_rate_per_ns": 0.1,
        "nvt_ps": 0.2,
        "compress_ps_each": 0.2,
        "npt_ps": 0.3,
        "anneal_cycles": 1,
        "anneal_window_ps": 0.1,
        "anneal_hold_ps": 0.1,
        "compress_pressures_bar": (1.0, 20.0, 1.0),
    }
    run_modulus_rate_scan(argon_run, tmp_path, **options)
    workflow = json.loads((tmp_path / WORKFLOW_NAME).read_text())
    reference = workflow["reference_box_nm"]
    manifests = [tmp_path / name / "manifest.json" for name in workflow["run_dirs"]]
    before = [json.loads(path.read_text())["stages"] for path in manifests]
    for rate_index, (stages, hold) in enumerate(zip(before, holds, strict=True)):
        for replica in range(spec.n_replicas):
            replica_index = rate_index * spec.n_replicas + replica
            entries = [
                entry for name, entry in stages.items() if f"_r{replica_index}_" in name
            ]
            assert entries
            assert (
                sum(len(entry["samples"]["segment_strain"]) for entry in entries) == 3
            )
            for entry in entries:
                samples = entry["samples"]
                assert samples["reference_box_nm"] == pytest.approx(reference)
                assert samples["segment_duration_ps"] == pytest.approx(
                    [hold] * len(samples["segment_strain"])
                )
                assert Path(entry["final_state"]).is_file()
    run_modulus_rate_scan(argon_run, tmp_path, **options)
    after = [json.loads(path.read_text())["stages"] for path in manifests]
    assert after == before
