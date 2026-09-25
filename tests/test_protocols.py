"""Tests for protocol assembly, the manifest, and resuming a run."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import openmm as mm
import pytest

from openmmpolymer import protocols, simulate
from openmmpolymer._files import file_sha256
from openmmpolymer.mdsystem import SystemSpec
from openmmpolymer.protocols import (
    MANIFEST_NAME,
    STAGE_RUNNERS,
    Protocol,
    ProtocolError,
    RunManifest,
    RunSummary,
    Stage,
    _canonical,
    _stage_options,
    chain_dimensions,
    check_build_request,
    melt_quench,
    record_build_request,
    run_protocol,
    standard_melt_equilibration,
)
from openmmpolymer.reporters import TrajectoryOptions
from openmmpolymer.simulate import (
    DEFAULT_COMPRESSION_BAR,
    Segment,
    StageResult,
    heating_temperatures,
    quench_temperatures,
    run_heat,
)

from .helpers import argon_context

#: A protocol short enough to run in a test but shaped like a real one.
QUICK = Protocol(
    name="quick",
    stages=(
        Stage("00_minimise", "minimise"),
        # Heavily damped and long enough to reach the thermostat's
        # temperature, so the manifest's "what it actually ran at" is
        # assertable rather than a number still on its way there.
        Stage(
            "01_nvt",
            "nvt",
            {"temperature_k": 100.0, "duration_ps": 2.0, "friction_ps": 20.0},
        ),
        Stage(
            "02_npt",
            "npt",
            {
                "temperature_k": 100.0,
                "duration_ps": 2.0,
                "friction_ps": 20.0,
                "barostat_frequency": 5,
            },
        ),
    ),
)

#: SHA-256 of each stage kind's default options, canonical and key-sorted, as
#: a manifest records them. Every stage's request carries its runner's
#: defaults - and, for a runner taking ``**kwargs``, those of ``run_segments``
#: - into the provenance a resume is checked against, so renaming or
#: re-defaulting any keyword makes every run already on disk refuse to resume.
#: Change one of these only knowing that.
RECORDED_DEFAULTS_SHA256 = {
    "anneal": "2f5b428eabd6389c766c5785cc31e307139e980b14f3ff9bdcd96d9ade4b53a3",
    "compress": "19e60ee4d5da965f44884d1d2d0bef087d5f1fc5fa10c73eb1a2cb78332bb923",
    "deform": "1dcd8e2203fff90cc82b0b27446b4a3109a02d93db3fb8cb40e497a4d66103bf",
    "heat": "4a10602459e5224074cff70d062b0852cfb689c39273f417bdd47ba52aba70b8",
    "load": "502035a7efd1c39108ed009a6518606fedead4dfdb40b4d177608ced87cb4f55",
    "minimise": "01264583f592f0a58d3307525cc8eebe5dfd47c425a2a421ad3f398f567ba27b",
    "npt": "735f2f94e9a6176584cc86524292f5db7876aba5c284879e35f1cb497801ce72",
    "nvt": "bed635ba5c28f35dae4b3ab668507ca3fe52ad643dc9baea76c2373c91c8cd00",
    "production": "9b3061d4fa5e2e8788cf145a3e3ccafac24c9e1f809da2239d867f8e34f3cd41",
    "pushoff": "914e78cc5b99af6fd335b17c41da3b6d615d012c791f65ae0b8f6d81a38d44d3",
    "quench": "16189e0bee6346454029fe678ff407a692c5cf5f7b072542b3c3608b0b0f35c0",
    "relax": "fbed5cc22943873546088616bb6ce8fa666c7fdd1aa83a636559beafc90e33ba",
    "shear": "6b8e4e218eb7fbf3cd0e154218020056b4e9a3b05c0bc4d05eb6e684c874f981",
}


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory: pytest.TempPathFactory) -> RunSummary:
    """:data:`QUICK`, run once for the tests that only read what it wrote."""
    return run_protocol(
        QUICK, argon_context(64, 2.4), tmp_path_factory.mktemp("quick") / "run"
    )


def test_the_defaults_every_manifest_records_are_pinned() -> None:
    """Changing one refuses the resume of every run already on disk."""
    recorded = {}
    for kind in STAGE_RUNNERS:
        options = _stage_options(Stage("x", kind, {}))
        options.pop("state_in")
        options.pop("output_prefix")
        text = json.dumps(_canonical(options), sort_keys=True, allow_nan=False)
        recorded[kind] = hashlib.sha256(text.encode()).hexdigest()
    assert recorded == RECORDED_DEFAULTS_SHA256


def test_a_stage_kind_must_be_one_that_can_run() -> None:
    """A typo in a protocol should not surface hours in."""
    with pytest.raises(ValueError, match="it must be one of"):
        Stage("00", "equilibriate")


def test_a_protocol_needs_stages() -> None:
    """An empty protocol is a mistake."""
    with pytest.raises(ValueError, match="no stages"):
        Protocol(name="empty", stages=())


def test_stage_names_must_differ() -> None:
    """Each one names the files it writes."""
    with pytest.raises(ValueError, match="repeats a stage name"):
        Protocol(
            name="clash",
            stages=(Stage("00", "minimise"), Stage("00", "minimise")),
        )


def test_the_standard_protocol_runs_the_documented_order() -> None:
    """Minimise, relieve the packing, mobilise, compress, anneal, settle."""
    protocol = standard_melt_equilibration()
    assert [stage.kind for stage in protocol.stages] == [
        "minimise",
        "pushoff",
        "nvt",
        "compress",
        "anneal",
        "npt",
    ]


def test_the_standard_protocol_ends_at_the_target_temperature() -> None:
    """Whatever it was melted at, it finishes where it was asked to."""
    protocol = standard_melt_equilibration(target_temperature_k=380.0)
    assert protocol.stages[-1].options["temperature_k"] == 380.0


def test_melt_quench_is_the_standard_protocol_plus_a_quench() -> None:
    """The cooling curve is the only thing it adds."""
    base = standard_melt_equilibration()
    quenched = melt_quench()
    assert len(quenched.stages) == len(base.stages) + 1
    assert quenched.stages[-1].kind == "quench"


def test_total_duration_adds_up_the_dynamics_asked_for() -> None:
    """Enough to tell nanoseconds from microseconds before starting."""
    assert standard_melt_equilibration(nvt_ps=100.0, npt_ps=200.0).total_duration_ps > 0


def test_running_a_protocol_writes_a_manifest(quick_run: RunSummary) -> None:
    """The record of what happened, and the basis of picking it up again."""
    manifest = json.loads(Path(quick_run.manifest_path).read_text())

    assert manifest["protocol"] == "quick"
    assert manifest["seed"] == 11
    assert set(manifest["stages"]) == {"00_minimise", "01_nvt", "02_npt"}
    assert manifest["versions"]["openmm"]
    assert manifest["system"]["nonbonded_cutoff_nm"] == SystemSpec().nonbonded_cutoff_nm


def test_the_manifest_records_what_each_stage_actually_did(
    quick_run: RunSummary,
) -> None:
    """The temperature asked for and the one reached are different numbers."""
    entry = json.loads(Path(quick_run.manifest_path).read_text())["stages"]["01_nvt"]
    assert entry["steps"] > 0
    assert entry["mean_temperature_k"] == pytest.approx(100.0, abs=40.0)
    assert Path(entry["final_state"]).is_file()


def test_stages_write_into_the_run_directory(quick_run: RunSummary) -> None:
    """One directory holds the whole run, in the order it ran."""
    written = sorted(path.name for path in Path(quick_run.run_dir).glob("*.state.xml"))
    assert written == ["00_minimise.state.xml", "01_nvt.state.xml", "02_npt.state.xml"]


def test_the_manifest_records_what_was_in_the_cell(quick_run: RunSummary) -> None:
    """SystemSpec records the settings a run was given but not the thing it was
    given them for, so analysis of a finished run had to infer the block
    structure every measurement indexes by."""
    manifest = RunManifest.load(Path(quick_run.run_dir))
    assert manifest is not None
    assert manifest.box == {
        "n_molecules": 64,
        "atoms_per_chain": 1,
        "box_nm": [2.4, 2.4, 2.4],
    }


def test_a_resumed_run_skips_what_is_already_done(argon_run: Any) -> None:
    """The point of the manifest."""
    first = run_protocol(QUICK, argon_run, "run")
    assert first.skipped == ()

    second = run_protocol(QUICK, argon_run, "run")
    assert second.skipped == ("00_minimise", "01_nvt", "02_npt")
    assert second.results == ()


def test_a_resumed_run_carries_on_from_where_it_stopped(argon_run: Any) -> None:
    """Half a protocol, then the rest of it."""
    half = Protocol(name="quick", stages=QUICK.stages[:2])
    run_protocol(half, argon_run, "run")

    resumed = run_protocol(QUICK, argon_run, "run")
    assert resumed.skipped == ("00_minimise", "01_nvt")
    assert [result.name for result in resumed.results] == ["02_npt"]


def test_explicit_forwarded_defaults_match_the_original_request(argon_run: Any) -> None:
    run_protocol(QUICK, argon_run, "run")
    explicit = replace(
        QUICK.stages[1],
        options={**QUICK.stages[1].options, "barostat_frequency": 25},
    )
    resumed = run_protocol(
        replace(QUICK, stages=(QUICK.stages[0], explicit, QUICK.stages[2])),
        argon_run,
        "run",
    )
    assert resumed.results == ()


def test_resume_can_be_turned_off(argon_run: Any) -> None:
    """Forcing a rerun has to be possible, or a changed setting is stuck."""
    run_protocol(QUICK, argon_run, "run")
    again = run_protocol(QUICK, argon_run, "run", resume=False)
    assert again.skipped == ()
    assert len(again.results) == 3


def test_a_stage_whose_state_has_gone_is_run_again(argon_run: Any) -> None:
    """A manifest entry is not enough; the state it points at has to be there."""
    run_protocol(QUICK, argon_run, "run")
    Path("run/01_nvt.state.xml").unlink()
    resumed = run_protocol(QUICK, argon_run, "run")
    assert resumed.skipped == ("00_minimise",)
    assert [result.name for result in resumed.results] == ["01_nvt", "02_npt"]


def test_a_resumed_run_reads_each_state_once(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent's final state is also its child's input, and each stage's output
    the next one's; hashing each afresh for every role read it several times.
    """
    run_protocol(QUICK, argon_run, "run")
    Path("run/02_npt.state.xml").unlink()
    read: list[Path] = []

    def counted(path: str | Path) -> str:
        read.append(Path(path).resolve())
        return file_sha256(path)

    monkeypatch.setattr(protocols, "file_sha256", counted)
    resumed = run_protocol(QUICK, argon_run, "run")
    assert [result.name for result in resumed.results] == ["02_npt"]
    assert sorted(path.name for path in read) == [
        "00_minimise.state.xml",
        "01_nvt.state.xml",
        "02_npt.state.xml",
    ]


def _saved_artifacts(directory: Path) -> dict[str, bytes]:
    return {
        str(path): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(
    "option,value", [("temperature_k", 120.0), ("duration_ps", 3.0)]
)
def test_changed_stage_settings_are_rejected_without_overwriting_artifacts(
    argon_run: Any, option: str, value: float
) -> None:
    run_protocol(QUICK, argon_run, "run")
    before = _saved_artifacts(Path("run"))
    changed = replace(
        QUICK.stages[1], options={**QUICK.stages[1].options, option: value}
    )
    protocol = replace(QUICK, stages=(QUICK.stages[0], changed, QUICK.stages[2]))
    with pytest.raises(ProtocolError, match="settings or starting state changed"):
        run_protocol(protocol, argon_run, "run")
    assert _saved_artifacts(Path("run")) == before


@pytest.mark.parametrize(
    "change", ["seed", "spec", "system", "positions", "topology", "box"]
)
def test_changed_initial_inputs_are_rejected_without_overwriting_the_manifest(
    argon_run: Any, change: str
) -> None:
    protocol = Protocol("initial", (QUICK.stages[0],))
    run_protocol(protocol, argon_run, "run")
    before = _saved_artifacts(Path("run"))
    if change == "seed":
        argon_run = replace(argon_run, seed=99)
    elif change == "spec":
        argon_run = replace(
            argon_run, spec=replace(argon_run.spec, nonbonded_cutoff_nm=0.8)
        )
    elif change == "system":
        system = mm.XmlSerializer.deserialize(argon_run.system_xml)
        system.setParticleMass(0, 41.0)
        argon_run = replace(argon_run, system_xml=mm.XmlSerializer.serialize(system))
    elif change == "positions":
        positions = argon_run.box.positions_nm.copy()
        positions[0, 0] += 0.01
        argon_run = replace(
            argon_run, box=replace(argon_run.box, positions_nm=positions)
        )
    elif change == "topology":
        next(argon_run.box.topology.atoms()).name = "different"
    else:
        argon_run = replace(
            argon_run, box=replace(argon_run.box, box_nm=(2.5, 2.4, 2.4))
        )
    with pytest.raises(ProtocolError, match="starting inputs changed"):
        run_protocol(protocol, argon_run, "run")
    assert _saved_artifacts(Path("run")) == before


def test_changing_protocol_name_or_stage_order_cannot_reuse_old_results(
    argon_run: Any,
) -> None:
    run_protocol(QUICK, argon_run, "run")
    before = _saved_artifacts(Path("run"))
    for changed in (
        replace(QUICK, name="other"),
        replace(QUICK, stages=tuple(reversed(QUICK.stages))),
    ):
        with pytest.raises(ProtocolError, match="Cannot resume"):
            run_protocol(changed, argon_run, "run")
        assert _saved_artifacts(Path("run")) == before


def test_external_starting_state_is_checked_by_content(argon_run: Any) -> None:
    initial = run_protocol(
        Protocol("prepare", (QUICK.stages[0],)), argon_run, "prepare"
    )
    protocol = Protocol("branch", (QUICK.stages[1],))
    run_protocol(protocol, argon_run, "branch", state_in=initial.final_state)
    before = _saved_artifacts(Path("branch"))
    path = Path(initial.final_state)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ProtocolError, match="starting state changed"):
        run_protocol(protocol, argon_run, "branch", state_in=initial.final_state)
    assert _saved_artifacts(Path("branch")) == before


def test_altered_completed_state_reruns_its_descendants(argon_run: Any) -> None:
    run_protocol(QUICK, argon_run, "run")
    path = Path("run/01_nvt.state.xml")
    path.write_text(path.read_text() + "\n")
    resumed = run_protocol(QUICK, argon_run, "run")
    assert resumed.skipped == ("00_minimise",)
    assert [result.name for result in resumed.results] == ["01_nvt", "02_npt"]


def test_separate_branches_resume_and_upstream_reruns_invalidate_all_branches(
    argon_run: Any,
) -> None:
    prepare = Protocol("quick", QUICK.stages[:2])
    initial = run_protocol(prepare, argon_run, "run")
    branches = [
        Protocol("quick", (replace(QUICK.stages[2], name=name),))
        for name in ("branch_a", "branch_b")
    ]
    for protocol in branches:
        run_protocol(protocol, argon_run, "run", state_in=initial.final_state)
    for protocol in branches:
        resumed = run_protocol(protocol, argon_run, "run", state_in=initial.final_state)
        assert resumed.skipped == (protocol.stages[0].name,)
    Path(initial.final_state).unlink()
    repaired = run_protocol(prepare, argon_run, "run")
    manifest = RunManifest.load("run")
    assert manifest is not None
    assert set(manifest.stages) == {"00_minimise", "01_nvt"}
    for protocol in branches:
        resumed = run_protocol(
            protocol, argon_run, "run", state_in=repaired.final_state
        )
        assert resumed.skipped == ()


def test_branch_only_resume_cannot_start_from_an_invalidated_parent(
    argon_run: Any,
) -> None:
    prepare = Protocol("quick", QUICK.stages[:2])
    initial = run_protocol(prepare, argon_run, "run")
    branch = Protocol("quick", QUICK.stages[2:])
    run_protocol(branch, argon_run, "run", state_in=initial.final_state)
    path = Path(initial.final_state)
    path.write_text(path.read_text() + "\n")
    before = _saved_artifacts(Path("run"))
    with pytest.raises(ProtocolError, match="upstream preparation"):
        run_protocol(branch, argon_run, "run", state_in=initial.final_state)
    assert _saved_artifacts(Path("run")) == before


def test_invalidated_descendants_stay_invalid_when_upstream_rerun_fails(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_protocol(QUICK, argon_run, "run")
    Path("run/01_nvt.state.xml").unlink()

    @wraps(STAGE_RUNNERS["nvt"])
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("interrupted")

    monkeypatch.setitem(STAGE_RUNNERS, "nvt", fail)
    with pytest.raises(ProtocolError, match="interrupted"):
        run_protocol(QUICK, argon_run, "run")
    manifest = RunManifest.load("run")
    assert manifest is not None
    assert set(manifest.stages) == {"00_minimise"}


def test_legacy_manifest_is_readable_but_requires_explicit_rerun(
    argon_run: Any,
) -> None:
    run_protocol(QUICK, argon_run, "run")
    path = Path("run/manifest.json")
    data = json.loads(path.read_text())
    data.pop("provenance")
    path.write_text(json.dumps(data))
    before = _saved_artifacts(Path("run"))
    assert RunManifest.load("run") is not None
    with pytest.raises(ProtocolError, match="legacy manifest"):
        run_protocol(QUICK, argon_run, "run")
    assert _saved_artifacts(Path("run")) == before
    assert run_protocol(QUICK, argon_run, "run", resume=False).skipped == ()


def test_build_request_guards_assets_before_cli_rebuilds(tmp_path: Path) -> None:
    request = {"monomer": "[*]CC[*]", "caps": ("H", "H"), "seed": 7}
    record_build_request(tmp_path, request)
    (tmp_path / "build").mkdir()
    asset = tmp_path / "build/chain.pdb"
    asset.write_text("original build")
    check_build_request(tmp_path, {**request, "caps": ["H", "H"]})
    before = _saved_artifacts(tmp_path)
    with pytest.raises(ProtocolError, match="inputs changed"):
        record_build_request(tmp_path, {**request, "seed": 8})
    assert _saved_artifacts(tmp_path) == before


def test_build_request_rejects_unverified_existing_artifacts(tmp_path: Path) -> None:
    (tmp_path / "build").mkdir()
    with pytest.raises(ProtocolError, match="lack input provenance"):
        check_build_request(tmp_path, {"seed": 7})


def test_a_failing_stage_leaves_the_manifest_behind(argon_run: Any) -> None:
    """Whatever finished stays recorded, so the rerun does not repeat it."""
    broken = Protocol(
        name="broken",
        stages=(
            Stage("00_minimise", "minimise"),
            Stage("01_quench", "quench", {"t_start": 100.0, "t_end": 200.0}),
        ),
    )
    with pytest.raises(ProtocolError, match="01_quench"):
        run_protocol(broken, argon_run, "run")

    manifest = json.loads((Path("run") / MANIFEST_NAME).read_text())
    assert "00_minimise" in manifest["stages"]
    assert "01_quench" not in manifest["stages"]


def test_the_manifest_round_trips(tmp_path: Path) -> None:
    """Loading gives back what was saved.

    It is written to a temporary file and moved into place, never in place,
    so a write that never finishes cannot leave half a manifest.
    """
    original = RunManifest(
        protocol="p", seed=7, versions={"openmm": "8.6.1"}, stages={"00": {"steps": 5}}
    )
    original.save(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))
    loaded = RunManifest.load(tmp_path)
    assert loaded is not None
    assert loaded.seed == 7
    assert loaded.stages["00"]["steps"] == 5


def test_loading_a_manifest_that_is_not_there_gives_none(tmp_path: Path) -> None:
    """A first run has no manifest, and that is not an error."""
    assert RunManifest.load(tmp_path) is None


def _ideal_chain(n_bonds: int, bond_nm: float, ratio: float) -> np.ndarray:
    """A chain laid out so that its characteristic ratio is exactly *ratio*."""
    end_to_end = math.sqrt(ratio * n_bonds) * bond_nm
    points = [np.array([0.0, 0.0, 0.0])]
    # A flat zig-zag whose bonds are all bond_nm and whose span is end_to_end.
    step = end_to_end / n_bonds
    height = math.sqrt(max(bond_nm**2 - step**2, 0.0))
    for index in range(1, n_bonds + 1):
        points.append(np.array([index * step, height * (index % 2), 0.0]))
    return np.asarray(points)


def test_chain_dimensions_recovers_the_ratio_it_was_built_with() -> None:
    """The measurement has to be right before its verdict means anything."""
    chain = _ideal_chain(40, 0.153, 7.0)
    positions = np.vstack([chain, chain + 5.0])
    measured = chain_dimensions(
        positions,
        range(41),
        41,
        2,
        expected_characteristic_ratio=7.0,
    )
    assert measured.characteristic_ratio == pytest.approx(7.0, rel=0.05)
    assert measured.consistent


def test_chain_dimensions_reports_a_collapsed_coil_as_inconsistent() -> None:
    """A packed cell of ETKDG globules should not read as an equilibrated melt."""
    chain = _ideal_chain(40, 0.153, 2.0)
    measured = chain_dimensions(
        chain, range(41), 41, 1, expected_characteristic_ratio=7.0
    )
    assert not measured.consistent
    assert measured.characteristic_ratio < 7.0


def test_chain_dimensions_are_recorded_in_the_manifest(
    dimer_argon_run: Any,
) -> None:
    """Reported, not claimed: it is a snapshot and the manifest says what it is."""
    summary = run_protocol(
        QUICK,
        dimer_argon_run,
        "run",
        chain_backbone=(0, 1),
        atoms_per_chain=2,
        expected_characteristic_ratio=7.0,
    )
    assert summary.chains is not None
    manifest = json.loads(Path(summary.manifest_path).read_text())
    assert manifest["chains"]["expected_characteristic_ratio"] == 7.0
    assert manifest["chains"]["backbone"] == [0, 1]


def test_a_manifest_written_before_the_cell_was_recorded_still_loads(
    tmp_path: Path,
) -> None:
    """New provenance fields must not prevent analysis of old manifests."""
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "old",
                "seed": 1,
                "versions": {},
                "system": {},
                "stages": {},
                "chains": None,
            }
        )
    )
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    assert manifest.box is None


def test_the_total_duration_counts_the_quench_it_used_to_omit() -> None:
    """A budget that leaves out the most expensive stage gets believed."""
    quench = melt_quench(t_end=200.0, step_k=20.0, hold_ps=200.0)
    equilibration = standard_melt_equilibration()

    ladder_ps = 200.0 * len(quench_temperatures(600.0, 200.0, 20.0))
    assert quench.total_duration_ps == pytest.approx(
        equilibration.total_duration_ps + ladder_ps
    )


def test_heat_is_registered_and_its_endpoint_ladder_duration_is_counted() -> None:
    """Both explicitly chunked and generated ladders count every hold."""
    assert STAGE_RUNNERS["heat"] is run_heat
    default = Protocol("heat", (Stage("01_heat", "heat"),))
    assert default.total_duration_ps == pytest.approx(
        200.0 * len(heating_temperatures(200.0, 600.0, 20.0))
    )
    custom = Stage(
        "01_heat",
        "heat",
        {"t_start": 100.0, "t_end": 175.0, "step_k": 30.0, "hold_ps": 10.0},
    )
    assert custom.duration_ps == pytest.approx(40.0)
    chunk = Stage("02_heat", "heat", {"temperatures_k": [200.0], "hold_ps": 10.0})
    assert chunk.duration_ps == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("kind", "options", "temperatures", "durations"),
    [
        ("quench", {}, list(range(600, 199, -20)), [200.0] * 21),
        ("heat", {}, list(range(200, 601, 20)), [200.0] * 21),
        (
            "quench",
            {"t_start": 150.0, "t_end": 90.0, "step_k": 40.0, "hold_ps": 0.7},
            [150.0, 110.0, 90.0],
            [0.7] * 3,
        ),
        (
            "heat",
            {"t_start": 100.0, "t_end": 175.0, "step_k": 30.0, "hold_ps": 0.7},
            [100.0, 130.0, 160.0, 175.0],
            [0.7] * 4,
        ),
        (
            "quench",
            {"temperatures_k": np.array([120.0]), "t_start": -1.0, "hold_ps": 0.7},
            [120.0],
            [0.7],
        ),
        (
            "heat",
            {"temperatures_k": [120.0], "t_end": -1.0, "hold_ps": 0.7},
            [120.0],
            [0.7],
        ),
        ("compress", {}, [600.0] * 7, [100.0] * 7),
        (
            "compress",
            {"pressures_bar": [1.0, 5.0, 1.0], "duration_ps_each": 0.7},
            [600.0] * 3,
            [0.7] * 3,
        ),
        (
            "anneal",
            {"n_cycles": 2, "ramp_windows": 2, "window_ps": 0.3, "hold_ps": 0.7},
            [450.0, 600.0, 600.0, 450.0, 300.0, 300.0] * 2,
            [0.3, 0.3, 0.7, 0.3, 0.3, 0.7] * 2,
        ),
        (
            "anneal",
            {"n_cycles": 2, "ramp_windows": 0, "hold_ps": 0.7},
            [600.0, 300.0] * 2,
            [0.7] * 4,
        ),
    ],
)
def test_duration_counts_the_segments_handed_to_execution(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    options: dict[str, Any],
    temperatures: list[float],
    durations: list[float],
) -> None:
    """Endpoint clipping, explicit chunks and annealing holds keep their order."""
    executed: list[Segment] = []

    @wraps(simulate.run_segments)
    def record_segments(
        run: Any,
        name: str,
        segments: Sequence[Segment],
        output_prefix: str | Path,
        **kwargs: Any,
    ) -> StageResult:
        executed.extend(segments)
        return StageResult(name, 0, 0.0, "unused.state.xml")

    monkeypatch.setattr(simulate, "run_segments", record_segments)
    STAGE_RUNNERS[kind](None, **options)
    assert [segment.temperature_k for segment in executed] == temperatures
    assert [segment.duration_ps for segment in executed] == durations
    assert Stage("test", kind, options).duration_ps == pytest.approx(sum(durations))
    if kind == "compress":
        assert [segment.pressure_bar for segment in executed] == list(
            options.get("pressures_bar", DEFAULT_COMPRESSION_BAR)
        )


@pytest.mark.parametrize("kind", ["heat", "quench"])
@pytest.mark.parametrize(
    "temperatures",
    [[], [100.0, 100.0], [float("nan")], [float("inf")], [0.0]],
)
def test_invalid_explicit_ladders_fail_in_budgets_and_execution(
    kind: str, temperatures: list[float]
) -> None:
    """An empty explicit ladder must never fall back to the default ramp."""
    options = {"temperatures_k": temperatures}
    with pytest.raises(ValueError) as estimate_error:
        _ = Stage("test", kind, options).duration_ps
    with pytest.raises(ValueError) as execution_error:
        STAGE_RUNNERS[kind](None, **options)
    assert str(estimate_error.value) == str(execution_error.value)


@pytest.mark.parametrize(
    ("kind", "options"),
    [("compress", {"pressures_bar": []}), ("anneal", {"n_cycles": 0})],
)
def test_empty_schedules_fail_in_budgets_and_execution(
    kind: str, options: dict[str, Any]
) -> None:
    """No dynamics to execute is an invalid stage, rather than a zero cost."""
    with pytest.raises(ValueError, match="no segments") as estimate_error:
        _ = Stage("test", kind, options).duration_ps
    with pytest.raises(ValueError, match="no segments") as execution_error:
        STAGE_RUNNERS[kind](None, **options)
    assert str(estimate_error.value) == str(execution_error.value)


def test_heating_samples_are_retained_in_a_resumable_manifest(
    argon_run: Any, tmp_path: Path
) -> None:
    """Protocol execution persists the signals needed by melting analysis."""
    protocol = Protocol(
        "heat",
        (
            Stage("00_minimise", "minimise"),
            Stage(
                "01_heat",
                "heat",
                {"temperatures_k": [100.0, 120.0], "hold_ps": 0.2},
            ),
        ),
    )
    run_protocol(protocol, argon_run, tmp_path)
    manifest = RunManifest.load(tmp_path)
    assert manifest is not None
    samples = manifest.stages["01_heat"]["samples"]
    assert samples["segment_temperature_k"] == [100.0, 120.0]
    assert len(samples["segment_enthalpy_kj_mol"]) == 2
    assert samples["segment_pressure_bar"] == [1.0, 1.0]
    resumed = run_protocol(protocol, argon_run, tmp_path, resume=True)
    assert resumed.skipped == ("00_minimise", "01_heat")
    reloaded = RunManifest.load(tmp_path)
    assert reloaded is not None
    assert reloaded.stages["01_heat"]["samples"] == samples


def test_the_total_duration_counts_an_anneal_and_a_compression_too() -> None:
    """Both state their time as a ladder, and both used to count as zero."""
    only_anneal = Protocol(
        name="anneal",
        stages=(
            Stage(
                "00_anneal",
                "anneal",
                {"n_cycles": 2, "ramp_windows": 4, "window_ps": 10.0, "hold_ps": 5.0},
            ),
        ),
    )
    assert only_anneal.total_duration_ps == pytest.approx(2 * 2 * (4 * 10.0 + 5.0))

    only_compress = Protocol(
        name="compress",
        stages=(Stage("00_compress", "compress", {"duration_ps_each": 50.0}),),
    )
    assert only_compress.total_duration_ps == pytest.approx(
        50.0 * len(DEFAULT_COMPRESSION_BAR)
    )


def test_the_anneal_cycles_to_the_target_unless_told_otherwise() -> None:
    """The default has to stay exactly what every run has always had."""
    assert standard_melt_equilibration(target_temperature_k=430.0).stages[4].options[
        "t_low"
    ] == pytest.approx(430.0)


def test_a_run_that_settles_at_the_melt_can_still_anneal() -> None:
    """A scan settles where it will start cooling, so the two must separate.

    Without this the anneal would cycle between one temperature and itself,
    and the hottest point on the quench curve - the anchor of the melt branch
    - would be measured on a cell still catching up from a jump.
    """
    protocol = standard_melt_equilibration(
        target_temperature_k=650.0, melt_temperature_k=650.0, anneal_t_low_k=500.0
    )
    anneal = protocol.stages[4].options
    assert anneal["t_low"] == pytest.approx(500.0)
    assert anneal["t_high"] == pytest.approx(650.0)


def test_the_equilibration_stage_writes_no_trajectory_unless_asked() -> None:
    """Frames of the whole cell are tens of megabytes a run."""
    assert standard_melt_equilibration().stages[5].options["trajectory"] == "none"
    asked = standard_melt_equilibration(
        npt_trajectory=TrajectoryOptions("xtc", interval_ps=5.0)
    )
    assert asked.stages[5].options["trajectory"].interval_ps == pytest.approx(5.0)


def test_the_compression_ladder_can_be_replaced() -> None:
    """A kilobar squeezes a sparse cell past twice the nonbonded cutoff."""
    gentle = standard_melt_equilibration(compress_pressures_bar=(1.0, 20.0, 1.0))
    assert gentle.stages[3].options["pressures_bar"] == (1.0, 20.0, 1.0)


def test_a_quench_can_start_somewhere_other_than_the_melt_temperature() -> None:
    """Welded together, a run cannot settle at one temperature and cool from
    another - which is how the hottest point came to be measured on a cell
    that was still reheating."""
    assert melt_quench().stages[-1].options["t_start"] == pytest.approx(600.0)
    assert melt_quench(t_start=520.0).stages[-1].options["t_start"] == pytest.approx(
        520.0
    )


# --------------------------------------------------------------------------
# The mechanical stage kinds
# --------------------------------------------------------------------------


def test_every_mechanical_stage_prices_itself() -> None:
    """A stage missing from the cost estimate is a stage that gets believed.

    All three are long, so a --dry-run that left them out would under-report
    the most expensive part of the run.
    """
    cases = (
        (Stage("d", "deform", {"n_steps": 25, "relax_ps": 50.0}), 1250.0),
        (
            Stage(
                "l",
                "load",
                {"stresses_bar": (0.0, 100.0, 200.0), "duration_ps_each": 1000.0},
            ),
            3000.0,
        ),
        (
            Stage("s", "shear", {"strains": (0.005, 0.01), "duration_ps_each": 200.0}),
            400.0,
        ),
    )
    for stage, expected in cases:
        assert stage.duration_ps == pytest.approx(expected)


def test_a_mechanical_stage_takes_its_runners_defaults() -> None:
    """Priced from the runner's own signature, so the two cannot drift."""
    duration = Stage("d", "deform", {}).duration_ps
    assert duration > 0.0
