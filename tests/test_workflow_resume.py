"""Fresh scans replace old requests; resumed scans verify their saved inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer import _workflow
from openmmpolymer._files import file_sha256
from openmmpolymer._workflow import record_scan_request, resumable_record
from openmmpolymer.protocols import ProtocolError, RunManifest


@pytest.mark.parametrize(
    ("damage", "message", "error"),
    [
        ("request", "different settings", ValueError),
        ("orphan", "settings cannot be verified", ValueError),
        ("stage_state", "completed stages with missing states", ValueError),
        ("start_state", "recorded fingerprint", ValueError),
        ("fingerprint", "no preparation-state fingerprint", ValueError),
        ("run_inputs", "starting inputs changed", ProtocolError),
    ],
)
def test_forced_rerun_ignores_records_that_cannot_be_resumed(
    tmp_path: Path,
    argon_run: Any,
    damage: str,
    message: str,
    error: type[Exception],
) -> None:
    workflow = tmp_path / "rate_workflow.json"
    request = {"spec": {"n_replicas": 1}}
    start = tmp_path / "start.xml"
    start.write_text("original equilibrated state")
    record: dict[str, Any] = {
        "request": request,
        "start_state": str(start),
        "start_state_sha256": file_sha256(start),
    }
    branch = tmp_path / "rate_00"
    branch.mkdir()
    manifest = RunManifest(protocol="scan", seed=argon_run.seed)
    if damage == "request":
        record["request"] = {"spec": {"n_replicas": 2}}
    elif damage == "stage_state":
        manifest.stages = {"finished": {"final_state": str(tmp_path / "gone.xml")}}
    elif damage == "start_state":
        start.unlink()
    elif damage == "fingerprint":
        del record["start_state_sha256"]
    elif damage == "run_inputs":
        preparation = tmp_path / "equilibration"
        preparation.mkdir()
        RunManifest(
            protocol="scan",
            seed=argon_run.seed,
            provenance={"version": 1, "run": {}},
        ).save(preparation)
    manifest.save(branch)
    if damage != "orphan":
        workflow.write_text(json.dumps(record))
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(error, match=message):
        resumable_record(
            argon_run, workflow, request, ["rate_00"], resume=True, error=ValueError
        )
    assert (
        resumable_record(
            argon_run, workflow, request, ["rate_00"], resume=False, error=ValueError
        )
        == {}
    )
    assert {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize("resume", [False, True])
def test_recording_request_resets_only_participating_manifests_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    workflow = tmp_path / "rate_workflow.json"
    workflow.write_text("previous request")
    runs = [tmp_path / "equilibration", tmp_path / "rate_00", tmp_path / "rate_01"]
    unrelated = tmp_path / "unrelated"
    for directory in [*runs, unrelated]:
        directory.mkdir()
        (directory / "manifest.json").write_text("previous manifest")
        (directory / "state.xml").write_text("saved state")
    request = {"spec": {"n_replicas": 1}}
    record: dict[str, Any] = {}

    def interrupted_write(path: Path, contents: Any, *, strict: bool) -> str:
        assert path == workflow
        assert contents == {"request": request, "run_dirs": ["rate_00", "rate_01"]}
        assert strict is False
        assert all(
            (directory / "manifest.json").exists() == resume for directory in runs
        )
        assert (unrelated / "manifest.json").read_text() == "previous manifest"
        assert all(
            (directory / "state.xml").read_text() == "saved state" for directory in runs
        )
        raise RuntimeError("interrupted request write")

    monkeypatch.setattr(_workflow, "write_json", interrupted_write)
    with pytest.raises(RuntimeError, match="interrupted request write"):
        record_scan_request(
            workflow,
            record,
            request,
            runs,
            resume=resume,
            run_dirs=["rate_00", "rate_01"],
        )
    assert workflow.read_text() == "previous request"
