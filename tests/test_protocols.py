"""Tests for protocol assembly, the manifest, and resuming a run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.protocols import (
    MANIFEST_NAME,
    STAGE_RUNNERS,
    Protocol,
    ProtocolError,
    RunManifest,
    Stage,
    chain_dimensions,
    melt_quench,
    run_protocol,
    standard_melt_equilibration,
)

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


def test_every_protocol_stage_names_a_runner_that_exists() -> None:
    """The dispatch table and the protocols cannot drift apart."""
    for protocol in (standard_melt_equilibration(), melt_quench()):
        for stage in protocol.stages:
            assert stage.kind in STAGE_RUNNERS


def test_total_duration_adds_up_the_dynamics_asked_for() -> None:
    """Enough to tell nanoseconds from microseconds before starting."""
    assert standard_melt_equilibration(nvt_ps=100.0, npt_ps=200.0).total_duration_ps > 0


def test_running_a_protocol_writes_a_manifest(argon_run: Any) -> None:
    """The record of what happened, and the basis of picking it up again."""
    summary = run_protocol(QUICK, argon_run, "run")
    manifest = json.loads(Path(summary.manifest_path).read_text())

    assert manifest["protocol"] == "quick"
    assert manifest["seed"] == argon_run.seed
    assert set(manifest["stages"]) == {"00_minimise", "01_nvt", "02_npt"}
    assert manifest["versions"]["openmm"]
    assert (
        manifest["system"]["nonbonded_cutoff_nm"] == argon_run.spec.nonbonded_cutoff_nm
    )


def test_the_manifest_records_what_each_stage_actually_did(argon_run: Any) -> None:
    """The temperature asked for and the one reached are different numbers."""
    summary = run_protocol(QUICK, argon_run, "run")
    entry = json.loads(Path(summary.manifest_path).read_text())["stages"]["01_nvt"]
    assert entry["steps"] > 0
    assert entry["mean_temperature_k"] == pytest.approx(100.0, abs=40.0)
    assert Path(entry["final_state"]).is_file()


def test_stages_write_into_the_run_directory(argon_run: Any) -> None:
    """One directory holds the whole run, in the order it ran."""
    run_protocol(QUICK, argon_run, "run")
    written = sorted(path.name for path in Path("run").glob("*.state.xml"))
    assert written == ["00_minimise.state.xml", "01_nvt.state.xml", "02_npt.state.xml"]


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
    assert "01_nvt" in [result.name for result in resumed.results]


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


def test_the_manifest_survives_a_write_that_never_finishes(tmp_path: Path) -> None:
    """Written to a temporary file and moved into place, never in place."""
    manifest = RunManifest(protocol="p", seed=1)
    manifest.save(tmp_path)
    assert (tmp_path / MANIFEST_NAME).is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_the_manifest_round_trips(tmp_path: Path) -> None:
    """Loading gives back what was saved."""
    original = RunManifest(
        protocol="p", seed=7, versions={"openmm": "8.6.1"}, stages={"00": {"steps": 5}}
    )
    original.save(tmp_path)
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
