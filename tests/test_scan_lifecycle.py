"""Interrupted scans keep the request that belongs to their saved replicas."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer import _workflow, mechanical, viscoelastic
from openmmpolymer.protocols import Protocol, RunManifest, RunSummary, run_protocol
from openmmpolymer.simulate import RunContext

from .helpers import QUICK_EQUILIBRATION


@dataclass(frozen=True)
class Scan:
    run: Callable[..., Any]
    spec: mechanical.ModulusSpec | viscoelastic.RelaxationSpec
    record_name: str
    measurement_stem: str
    error: type[Exception]


@pytest.fixture(params=("mechanical", "relaxation"))
def scan(request: pytest.FixtureRequest) -> Scan:
    if request.param == "mechanical":
        return Scan(
            mechanical.run_modulus_scan,
            mechanical.ModulusSpec(
                temperature_k=120.0,
                strain_increment=0.004,
                max_strain=0.012,
                relax_ps=0.3,
                elastic_strain_limit=0.012,
                n_replicas=2,
                samples_per_step=4,
                stage_ps=1.0,
                load_stresses_bar=None,
                bulk_pressures_bar=None,
                shear_strains=None,
            ),
            mechanical.WORKFLOW_NAME,
            mechanical.DEFORM_STEM,
            mechanical.MechanicalError,
        )
    return Scan(
        viscoelastic.run_relaxation_scan,
        viscoelastic.RelaxationSpec(
            temperature_k=120.0,
            step_strain=0.04,
            baseline_ps=0.5,
            relax_ps=3.0,
            stage_ps=3.0,
            n_replicas=2,
            sample_every_ps=0.05,
            late_sample_every_ps=0.2,
            late_after_ps=1.0,
            bins_per_decade=8,
        ),
        viscoelastic.WORKFLOW_NAME,
        viscoelastic.RELAX_STEM,
        viscoelastic.ViscoelasticError,
    )


class Interrupted(RuntimeError):
    """Stop at a known point in a scan without losing its saved artifacts."""


def _files(directory: Path) -> dict[Path, tuple[bytes, int]]:
    """Refused requests must not rewrite even an otherwise identical file."""
    return {
        path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_request_is_saved_before_the_first_dynamics(
    scan: Scan, argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        saved = json.loads((Path("run") / scan.record_name).read_text())
        assert saved["request"]["spec"]["n_replicas"] == 2
        assert saved["request"]["seed"] == argon_run.seed
        assert saved["request"]["equilibration"]
        raise Interrupted("before equilibration")

    monkeypatch.setattr(_workflow, "run_protocol", interrupt)
    with pytest.raises(Interrupted, match="before equilibration"):
        scan.run(argon_run, "run", spec=scan.spec, **QUICK_EQUILIBRATION)
    assert calls == 1


@pytest.mark.parametrize(
    "change",
    (
        "replicas",
        "equilibration",
        "seed",
        "system",
        "coordinates",
        "settings",
        "chains",
    ),
)
def test_interrupted_request_refuses_changed_inputs_without_writing(
    scan: Scan, argon_run: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise Interrupted("before equilibration")

    monkeypatch.setattr(_workflow, "run_protocol", interrupt)
    with pytest.raises(Interrupted):
        scan.run(argon_run, "run", spec=scan.spec, **QUICK_EQUILIBRATION)
    before = _files(Path("run"))

    run, spec = argon_run, scan.spec
    options = dict(QUICK_EQUILIBRATION)
    if change == "replicas":
        spec = replace(spec, n_replicas=1)
    elif change == "equilibration":
        options["nvt_ps"] *= 2
    elif change == "seed":
        run = replace(run, seed=run.seed + 1)
    elif change == "system":
        run = replace(run, system_xml=run.system_xml + "\n")
    elif change == "coordinates":
        positions = run.box.positions_nm.copy()
        positions[0, 0] += 0.01
        run = replace(run, box=replace(run.box, positions_nm=positions))
    elif change == "settings":
        run = replace(run, spec=replace(run.spec, hydrogen_mass_amu=3.0))
    else:
        options["expected_characteristic_ratio"] = 8.0

    with pytest.raises(scan.error, match="different settings"):
        scan.run(run, "run", spec=spec, **options)
    assert calls == 1
    assert _files(Path("run")) == before


@pytest.mark.parametrize(
    ("existing", "message"),
    (
        ("orphan-manifest", "without a request"),
        ("missing-request", "without a request"),
        ("foreign-protocol", "different protocol"),
        ("legacy-request", "different settings"),
    ),
)
def test_unverifiable_runs_require_an_explicit_fresh_start(
    scan: Scan,
    argon_run: Any,
    monkeypatch: pytest.MonkeyPatch,
    existing: str,
    message: str,
) -> None:
    directory = Path("run")
    workflow = directory / scan.record_name
    protocols: list[str] = []

    def interrupt(protocol: Protocol, *args: Any, **kwargs: Any) -> None:
        protocols.append(protocol.name)
        assert not (directory / "manifest.json").exists()
        saved = json.loads(workflow.read_text())
        assert saved["request"]["spec"]["n_replicas"] == 2
        assert saved["request"]["seed"] == argon_run.seed
        assert saved["request"]["equilibration"]
        raise Interrupted("before equilibration")

    monkeypatch.setattr(_workflow, "run_protocol", interrupt)
    with pytest.raises(Interrupted):
        scan.run(argon_run, directory, spec=scan.spec, **QUICK_EQUILIBRATION)

    record = json.loads(workflow.read_text())
    if existing == "orphan-manifest":
        workflow.unlink()
    elif existing == "missing-request":
        del record["request"]
        workflow.write_text(json.dumps(record))
    elif existing == "legacy-request":
        record["request"] = {"spec": record["request"]["spec"]}
        workflow.write_text(json.dumps(record))
    RunManifest(
        protocol="other_scan" if existing == "foreign-protocol" else protocols[0],
        seed=argon_run.seed,
        stages={"unverified_measurement": {"samples": {}}},
    ).save(directory)
    before = _files(directory)

    with pytest.raises(scan.error, match=message):
        scan.run(argon_run, directory, spec=scan.spec, **QUICK_EQUILIBRATION)
    assert len(protocols) == 1
    assert _files(directory) == before

    with pytest.raises(Interrupted):
        scan.run(
            argon_run,
            directory,
            spec=scan.spec,
            resume=False,
            **QUICK_EQUILIBRATION,
        )
    assert len(protocols) == 2


def test_interrupted_replicas_resume_consistently_and_force_rerun_replaces_them(
    scan: Scan, argon_scan_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the two-to-one replica bug against real CPU state files.

    The interruption comes after both replicas wrote their measurements but
    before the scan can analyse them or finish its metadata. An unchanged
    resume must retain two; only an explicit rerun may replace them with one.
    """
    interrupted = False

    def interrupt_after_second_replica(
        protocol: Protocol, run: RunContext, directory: Path, **options: Any
    ) -> RunSummary:
        nonlocal interrupted
        if protocol.stages[0].name.startswith(scan.measurement_stem):
            saved = json.loads((directory / scan.record_name).read_text())
            assert saved["start_state"] == options["state_in"]
            assert saved["reference_box_nm"] == pytest.approx(
                protocol.stages[0].options["reference_box_nm"]
            )
        summary = run_protocol(protocol, run, directory, **options)
        if not interrupted and any(
            stage.name.startswith(f"{scan.measurement_stem}_r1_")
            for stage in protocol.stages
        ):
            interrupted = True
            raise Interrupted("after two replicas")
        return summary

    monkeypatch.setattr(_workflow, "run_protocol", interrupt_after_second_replica)
    with pytest.raises(Interrupted, match="after two replicas"):
        scan.run(argon_scan_run, "run", spec=scan.spec, **QUICK_EQUILIBRATION)

    directory = Path("run")
    before = _files(directory)
    one_replica = replace(scan.spec, n_replicas=1)
    with pytest.raises(scan.error, match="different settings"):
        scan.run(argon_scan_run, directory, spec=one_replica, **QUICK_EQUILIBRATION)
    assert _files(directory) == before

    manifest_before = (directory / "manifest.json").read_bytes()
    resumed = scan.run(argon_scan_run, directory, spec=scan.spec, **QUICK_EQUILIBRATION)
    assert len(resumed.curves) == 2
    assert (directory / "manifest.json").read_bytes() == manifest_before

    def interrupt_before_replacement(*args: Any, **kwargs: Any) -> None:
        saved = json.loads((directory / scan.record_name).read_text())
        assert saved["request"]["spec"]["n_replicas"] == 1
        if (directory / "manifest.json").is_file():
            manifest = json.loads((directory / "manifest.json").read_text())
            assert not any(
                name.startswith(scan.measurement_stem) for name in manifest["stages"]
            )
        raise Interrupted("before replacement equilibration")

    monkeypatch.setattr(_workflow, "run_protocol", interrupt_before_replacement)
    with pytest.raises(Interrupted, match="before replacement equilibration"):
        scan.run(
            argon_scan_run,
            directory,
            spec=one_replica,
            resume=False,
            **QUICK_EQUILIBRATION,
        )

    monkeypatch.setattr(_workflow, "run_protocol", run_protocol)
    replaced = scan.run(
        argon_scan_run, directory, spec=one_replica, **QUICK_EQUILIBRATION
    )
    assert len(replaced.curves) == 1
    saved = json.loads((directory / scan.record_name).read_text())
    assert saved["n_replicas"] == saved["request"]["spec"]["n_replicas"] == 1
    manifest = json.loads((directory / "manifest.json").read_text())
    assert not any(
        name.startswith(f"{scan.measurement_stem}_r1_") for name in manifest["stages"]
    )
