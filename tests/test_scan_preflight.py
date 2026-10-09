"""Pin the scan resume checks before consolidating their pre-flight code."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer import _workflow, mechanical, tensile, tg, tm, viscoelastic
from openmmpolymer.protocols import Protocol, ProtocolError, RunManifest, run_protocol
from openmmpolymer.simulate import RunContext

from .helpers import QUICK_EQUILIBRATION, snapshot_files


@dataclass(frozen=True)
class Scan:
    run: Callable[..., Any]
    record_name: str
    error: type[Exception]
    options: dict[str, Any]
    refuses_missing_state: bool = False


SCANS = {
    "mechanical": Scan(
        mechanical.run_modulus_scan,
        mechanical.WORKFLOW_NAME,
        mechanical.MechanicalError,
        QUICK_EQUILIBRATION,
    ),
    "viscoelastic": Scan(
        viscoelastic.run_relaxation_scan,
        viscoelastic.WORKFLOW_NAME,
        viscoelastic.ViscoelasticError,
        QUICK_EQUILIBRATION,
    ),
    "tg": Scan(tg.run_tg_scan, tg.WORKFLOW_NAME, tg.TgError, QUICK_EQUILIBRATION),
    "tm": Scan(tm.run_tm_scan, tm.WORKFLOW_NAME, tm.TmError, {"crystalline": True}),
    **{
        name: Scan(
            run, measurement.workflow_name, measurement.error, QUICK_EQUILIBRATION, True
        )
        for name, run, measurement in (
            ("breaking", tensile.run_breaking_scan, tensile.BREAKING),
            ("elongation", tensile.run_elongation_scan, tensile.ELONGATION),
            ("yield", tensile.run_yield_scan, tensile.YIELD),
        )
    },
}


class DynamicsReached(RuntimeError):
    """Stop a scan at the dynamics boundary, retaining its real saved inputs."""


def _patch_dynamics(
    monkeypatch: pytest.MonkeyPatch, runner: Callable[..., Any]
) -> None:
    # Patch the shared runner and the thermal entry points so every public
    # scan stops at the same dynamics boundary.
    for module in (_workflow, tensile, tg, tm):
        if hasattr(module, "run_protocol"):
            monkeypatch.setattr(module, "run_protocol", runner)


def _begin_scan(
    scan: Scan, run: RunContext, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RunManifest]:
    """Save one real stage, including provenance, then interrupt the scan.

    Every scan starts with minimisation. One such stage is enough to exercise
    the pre-flight without repeating seven full scans for each damage case.
    """
    directory = Path("run")

    def first_stage(protocol: Protocol, *args: Any, **options: Any) -> None:
        run_protocol(Protocol(protocol.name, protocol.stages[:1]), *args, **options)
        raise DynamicsReached

    _patch_dynamics(monkeypatch, first_stage)
    with pytest.raises(DynamicsReached):
        scan.run(run, directory, **scan.options)
    manifest = RunManifest.load(directory)
    assert manifest is not None
    assert len(manifest.stages) == 1
    assert (directory / scan.record_name).is_file()
    return directory, manifest


@pytest.mark.parametrize(
    ("name", "damage"),
    [
        pytest.param(
            name,
            damage,
            id=f"{name}-{damage}",
        )
        for name in SCANS
        for damage in (
            "foreign-protocol",
            "changed-request",
            "missing-record",
            "changed-inputs",
        )
    ],
)
def test_scan_refuses_unverifiable_resume_before_writing_or_dynamics(
    name: str, damage: str, argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = SCANS[name]
    directory, manifest = _begin_scan(scan, argon_run, monkeypatch)
    workflow = directory / scan.record_name
    error = scan.error
    if damage == "foreign-protocol":
        manifest.protocol = "foreign-protocol"
        manifest.save(directory)
        message = "different protocol"
    elif damage == "changed-request":
        record = json.loads(workflow.read_text())
        record["request"]["spec"]["pressure_bar"] += 1.0
        workflow.write_text(json.dumps(record))
        message = "different settings|settings or starting inputs changed"
    elif damage == "missing-record":
        workflow.unlink()
        message = "without a request|workflow record"
    else:
        assert manifest.provenance is not None
        manifest.provenance["run"]["seed"] += 1
        manifest.save(directory)
        error = ProtocolError
        message = "starting inputs changed"
    before = snapshot_files(directory, mtimes=True)

    def unexpected_dynamics(*args: Any, **options: Any) -> None:
        raise DynamicsReached("An unverifiable scan reached dynamics")

    _patch_dynamics(monkeypatch, unexpected_dynamics)
    with pytest.raises(error, match=message):
        scan.run(argon_run, directory, **scan.options)
    assert snapshot_files(directory, mtimes=True) == before


@pytest.mark.parametrize("name", SCANS)
def test_missing_state_preserves_each_scans_repair_or_refusal_policy(
    name: str, argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = SCANS[name]
    directory, manifest = _begin_scan(scan, argon_run, monkeypatch)
    stage_name, stage = next(iter(manifest.stages.items()))
    state = Path(stage["final_state"])
    state.unlink()
    before = snapshot_files(directory, mtimes=True)
    repaired: list[str] = []

    def repair_first_stage(protocol: Protocol, *args: Any, **options: Any) -> None:
        summary = run_protocol(
            Protocol(protocol.name, protocol.stages[:1]), *args, **options
        )
        assert not summary.skipped
        assert state.is_file()
        repaired.append(stage_name)
        raise DynamicsReached

    _patch_dynamics(monkeypatch, repair_first_stage)
    if scan.refuses_missing_state:
        with pytest.raises(scan.error, match="missing state files"):
            scan.run(argon_run, directory, **scan.options)
        assert not repaired
        assert snapshot_files(directory, mtimes=True) == before
    else:
        with pytest.raises(DynamicsReached):
            scan.run(argon_run, directory, **scan.options)
        assert repaired == [stage_name]


@pytest.mark.parametrize("name", SCANS)
def test_empty_manifest_without_workflow_preserves_the_tensile_exception(
    name: str, argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = SCANS[name]
    directory, manifest = _begin_scan(scan, argon_run, monkeypatch)
    manifest.stages.clear()
    assert manifest.provenance is not None
    manifest.provenance["stages"].clear()
    manifest.save(directory)
    (directory / scan.record_name).unlink()
    before = snapshot_files(directory, mtimes=True)

    def interrupted(*args: Any, **kwargs: Any) -> None:
        raise DynamicsReached

    _patch_dynamics(monkeypatch, interrupted)
    if name in ("breaking", "elongation", "yield"):
        with pytest.raises(DynamicsReached):
            scan.run(argon_run, directory, **scan.options)
        assert (directory / scan.record_name).is_file()
    else:
        with pytest.raises(scan.error, match="without a request"):
            scan.run(argon_run, directory, **scan.options)
        assert snapshot_files(directory, mtimes=True) == before


@pytest.mark.parametrize("name", ("breaking", "elongation", "yield"))
@pytest.mark.parametrize(
    "record", [{"request": []}, {"request": {"spec": []}}, {"request": None}]
)
def test_tensile_refuses_malformed_nested_request_without_writes(
    name: str, record: dict[str, Any], argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = SCANS[name]
    directory = Path("run")
    directory.mkdir()
    (directory / scan.record_name).write_text(json.dumps(record))
    before = snapshot_files(directory, mtimes=True)

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise DynamicsReached

    _patch_dynamics(monkeypatch, unexpected)
    with pytest.raises(scan.error, match="different settings"):
        scan.run(argon_run, directory, **scan.options)
    assert snapshot_files(directory, mtimes=True) == before
