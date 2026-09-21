"""Tests for the mechanical scan and what it reports.

Split the way the glass-transition tests are split. Everything that decides
something - how long the ladder is, which pass is skipped, what a resume
refuses, how replicas are grouped - is tested against manifests written by
hand, because those are arithmetic over recorded numbers and running
dynamics to reach them would hide what is being checked. The plumbing that
has to survive a real Context is tested once, on an argon cell, which runs
the whole workflow in a couple of seconds.

Argon is a liquid at these settings, so it has no shear modulus and its
Young's modulus is meaningless. That is deliberate: the argon tests assert
that the machinery did what it was told - the box moved by exactly the
strain, the strain recorded is the strain applied, a resume repeats nothing
- and leave every physical number to the planted manifests, where the answer
is known exactly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.elasticity import MAX_CONSISTENCY_GAP
from openmmpolymer.forcefield import PolymerForceField
from openmmpolymer.mdsystem import PackedBox
from openmmpolymer.mechanical import (
    BULK_STEM,
    DEFORM_STEM,
    LOAD_STEM,
    SHEAR_STEM,
    WORKFLOW_NAME,
    MechanicalError,
    ModulusSpec,
    analyse_mechanics,
    deform_protocol,
    deform_schedule,
    extra_stages,
    mechanical_scan,
    run_modulus_scan,
    write_mechanical_report,
)
from openmmpolymer.simulate import prepare_run
from openmmpolymer.trajectory import AnalysisError

from .helpers import argon_system, write_bulk, write_deformation, write_shear

#: Settings that put the whole scan inside a couple of seconds on argon. The
#: cell has no elastic constants worth the name, so nothing here asserts one.
QUICK = ModulusSpec(
    temperature_k=120.0,
    strain_increment=0.004,
    max_strain=0.012,
    relax_ps=0.3,
    elastic_strain_limit=0.012,
    n_replicas=2,
    samples_per_step=4,
    stage_ps=1.0,
    load_stresses_bar=(0.0, 200.0),
    load_ps_each=0.3,
    bulk_pressures_bar=(1.0, 50.0, 100.0, 50.0, 1.0),
    bulk_ps_each=0.3,
    shear_strains=(0.005, 0.010, 0.015),
    shear_ps_each=0.3,
)

#: The equilibration, shortened to match, and gentle for the same reason
#: the Tg tests are: a kilobar squeezes a small argon cell past its cutoff.
QUICK_EQUILIBRATION: dict[str, Any] = {
    "nvt_ps": 0.2,
    "compress_ps_each": 0.2,
    "npt_ps": 0.3,
    "anneal_cycles": 1,
    "anneal_window_ps": 0.1,
    "anneal_hold_ps": 0.1,
    "compress_pressures_bar": (1.0, 20.0, 1.0),
}


@pytest.fixture
def argon_scan_run() -> Any:
    """An argon cell big enough to survive an NPT equilibration and a strain."""
    system, topology, positions = argon_system(216, 2.8)
    box = PackedBox(
        topology=topology,
        positions_nm=positions,
        box_nm=(2.8, 2.8, 2.8),
        n_molecules=216,
    )
    return prepare_run(
        box,
        PolymerForceField("unused.xml", (), "AR", "smirnoff"),
        platform="CPU",
        seed=11,
        system=system,
    )


# --------------------------------------------------------------------------
# Schedules and what a scan costs
# --------------------------------------------------------------------------


def test_the_ladder_reaches_the_strain_it_was_asked_for() -> None:
    """Increments compound, so the step count is a logarithm, not a ratio."""
    schedule = deform_schedule(ModulusSpec(strain_increment=0.002, max_strain=0.05))
    assert schedule.n_steps == 25
    assert schedule.max_strain >= 0.05
    assert (1.0 + 0.002) ** 24 - 1.0 < 0.05


def test_the_strain_rate_is_the_whole_ladder_over_the_whole_time() -> None:
    """The number that has to travel with every modulus."""
    schedule = deform_schedule(
        ModulusSpec(strain_increment=0.002, max_strain=0.05, relax_ps=50.0)
    )
    assert schedule.total_ps == pytest.approx(25 * 50.0)
    assert schedule.strain_rate_per_ns == pytest.approx(
        schedule.max_strain / schedule.total_ps * 1000.0
    )


def test_a_long_ladder_is_split_into_resumable_chunks() -> None:
    """A stage is the unit a run picks itself up at, so it has a ceiling."""
    spec = ModulusSpec(
        strain_increment=0.002, max_strain=0.05, relax_ps=50.0, stage_ps=500.0
    )
    stages = deform_protocol(spec, timestep_fs=2.0).stages
    assert len(stages) == 3
    assert sum(int(stage.options["n_steps"]) for stage in stages) == 25
    # Each chunk knows the strain it starts at, because the state file does
    # not carry one and a resumed chunk would otherwise restart the count.
    assert [round(float(stage.options["strain_start"]), 6) for stage in stages] == [
        0.0,
        round((1.002) ** 10 - 1.0, 6),
        round((1.002) ** 20 - 1.0, 6),
    ]


def test_only_the_first_chunk_of_a_replica_redraws_velocities() -> None:
    """A replica is one trajectory; its later chunks continue it."""
    spec = replace(QUICK, stage_ps=0.3, max_strain=0.012)
    stages = deform_protocol(spec, timestep_fs=2.0).stages
    assert len(stages) > 1
    assert [bool(stage.options["new_velocities"]) for stage in stages] == [
        True,
        *([False] * (len(stages) - 1)),
    ]


def test_replicas_differ_only_in_their_stage_names() -> None:
    """Which is enough: every random stream is derived from the label."""
    first = deform_protocol(QUICK, timestep_fs=2.0, replica=0).stages
    second = deform_protocol(QUICK, timestep_fs=2.0, replica=1).stages
    assert [stage.name for stage in first] != [stage.name for stage in second]
    assert all(name.startswith(f"{DEFORM_STEM}_r0") for name in (s.name for s in first))
    assert [stage.options for stage in first] == [stage.options for stage in second]


def test_skipping_every_extra_pass_is_a_legitimate_request() -> None:
    """E and nu only, which is a configuration rather than another code path."""
    spec = replace(
        QUICK, load_stresses_bar=None, bulk_pressures_bar=None, shear_strains=None
    )
    assert extra_stages(spec, timestep_fs=2.0) == ()
    assert mechanical_scan(spec, **QUICK_EQUILIBRATION).stages


def test_each_pass_can_be_skipped_on_its_own() -> None:
    """And the ones left keep their order."""
    cases = (
        (LOAD_STEM, replace(QUICK, load_stresses_bar=None)),
        (BULK_STEM, replace(QUICK, bulk_pressures_bar=None)),
        (SHEAR_STEM, replace(QUICK, shear_strains=None)),
    )
    for stem, spec in cases:
        stages = extra_stages(spec, timestep_fs=2.0)
        assert stem not in [stage.name for stage in stages]
        assert len(stages) == 2


def test_a_scan_over_its_budget_stops_before_it_writes_anything(
    argon_scan_run: Any,
) -> None:
    """The refusal has to come before the first stage, not after."""
    with pytest.raises(MechanicalError, match="max_total_ns"):
        run_modulus_scan(
            argon_scan_run,
            "run",
            spec=replace(QUICK, max_total_ns=1.0e-6),
            **QUICK_EQUILIBRATION,
        )
    assert not Path("run/manifest.json").exists()


def test_the_cost_is_reported_before_anything_runs(
    argon_scan_run: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """So a scan that is too expensive is visible rather than discovered."""
    with caplog.at_level(logging.INFO, logger="openmmpolymer.mechanical"):
        run_modulus_scan(
            argon_scan_run,
            "run",
            spec=replace(
                QUICK,
                n_replicas=1,
                load_stresses_bar=None,
                bulk_pressures_bar=None,
                shear_strains=None,
            ),
            **QUICK_EQUILIBRATION,
        )
    assert "still to run" in caplog.text


# --------------------------------------------------------------------------
# The whole scan, on argon
# --------------------------------------------------------------------------


def test_the_whole_scan_runs_and_then_resumes_without_repeating_itself(
    argon_scan_run: Any,
) -> None:
    """The load-bearing one: every pass, the branching, and resume."""
    first = run_modulus_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    manifest = json.loads(Path("run/manifest.json").read_text())
    for stem in (LOAD_STEM, BULK_STEM, SHEAR_STEM):
        assert stem in manifest["stages"]
    assert len(first.replicas) == 2
    assert first.replica_spread_mpa is not None

    before = Path("run/manifest.json").read_bytes()
    second = run_modulus_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    assert second.youngs is not None and first.youngs is not None
    assert second.youngs.modulus_mpa == pytest.approx(first.youngs.modulus_mpa)
    assert Path("run/manifest.json").read_bytes() == before


def test_the_box_moves_by_exactly_the_strain_that_was_recorded(
    argon_scan_run: Any,
) -> None:
    """Strain is bookkeeping, and bookkeeping should be exact."""
    run_modulus_scan(
        argon_scan_run,
        "run",
        spec=replace(
            QUICK,
            n_replicas=1,
            load_stresses_bar=None,
            bulk_pressures_bar=None,
            shear_strains=None,
        ),
        **QUICK_EQUILIBRATION,
    )
    manifest = json.loads(Path("run/manifest.json").read_text())
    samples = next(
        recorded["samples"]
        for name, recorded in manifest["stages"].items()
        if name.startswith(DEFORM_STEM)
    )
    reference = samples["reference_box_nm"][2]
    for length, strain in zip(
        samples["segment_box_z_nm"], samples["segment_strain"], strict=True
    ):
        assert length / reference - 1.0 == pytest.approx(strain, abs=1e-12)


def test_the_lateral_axes_move_while_the_driven_one_is_held(
    argon_scan_run: Any,
) -> None:
    """The uniaxial-strain ensemble: scaleZ False, the other two at pressure."""
    run_modulus_scan(
        argon_scan_run,
        "run",
        spec=replace(
            QUICK,
            n_replicas=1,
            load_stresses_bar=None,
            bulk_pressures_bar=None,
            shear_strains=None,
        ),
        **QUICK_EQUILIBRATION,
    )
    manifest = json.loads(Path("run/manifest.json").read_text())
    samples = next(
        recorded["samples"]
        for name, recorded in manifest["stages"].items()
        if name.startswith(DEFORM_STEM)
    )
    lateral = np.asarray(samples["segment_box_x_nm"], dtype=np.float64)
    assert lateral.std() > 0.0


def test_replicas_are_given_different_velocities(argon_scan_run: Any) -> None:
    """Or three runs from one configuration are one run, and the spread is a lie."""
    result = run_modulus_scan(
        argon_scan_run,
        "run",
        spec=replace(
            QUICK,
            load_stresses_bar=None,
            bulk_pressures_bar=None,
            shear_strains=None,
        ),
        **QUICK_EQUILIBRATION,
    )
    assert len(result.replicas) == 2
    assert result.replicas[0].modulus_mpa != result.replicas[1].modulus_mpa
    assert result.replica_spread_mpa is not None
    assert result.replica_spread_mpa > 0.0


def test_every_pass_starts_from_the_same_equilibrated_cell(
    argon_scan_run: Any,
) -> None:
    """A cell that has just been stretched is not the next measurement's cell."""
    run_modulus_scan(argon_scan_run, "run", spec=QUICK, **QUICK_EQUILIBRATION)
    record = json.loads((Path("run") / WORKFLOW_NAME).read_text())
    assert Path(record["start_state"]).name.startswith("05_npt")
    assert len(record["reference_box_nm"]) == 3


def test_a_resumed_scan_still_starts_from_the_equilibrated_cell(
    argon_scan_run: Any,
) -> None:
    """The case the test above cannot see, and the one that matters.

    On a fresh run the equilibration's stages are the only ones that have
    run, so anything that looks for "the last state" finds the right one. On a
    resume they are all skipped and the manifest - which is in run order -
    already holds every deformation after them, so looking backwards through
    it lands on the end of a strained pass instead. Both halves of the scan
    then go wrong: passes that had not finished would branch from a cell that
    was already at the top of the ladder, and the strain origin read off that
    state is a stretched cell, so every strain still to be recorded would be
    measured against the wrong length.

    Asserted on the box rather than only on the file name, because the cubic
    shape is what says it is the equilibrated cell and not a deformed one -
    a shear pass leaves the volume alone and would pass a name-only check by
    luck.
    """
    spec = replace(QUICK, load_stresses_bar=None, bulk_pressures_bar=None)
    run_modulus_scan(argon_scan_run, "run", spec=spec, **QUICK_EQUILIBRATION)
    fresh = json.loads((Path("run") / WORKFLOW_NAME).read_text())

    run_modulus_scan(argon_scan_run, "run", spec=spec, **QUICK_EQUILIBRATION)
    resumed = json.loads((Path("run") / WORKFLOW_NAME).read_text())

    assert Path(resumed["start_state"]).name.startswith("05_npt")
    assert resumed["reference_box_nm"] == pytest.approx(fresh["reference_box_nm"])
    origin = resumed["reference_box_nm"]
    assert origin == pytest.approx([origin[0]] * 3)


def test_a_resume_that_asks_for_something_else_is_refused(
    argon_scan_run: Any,
) -> None:
    """Or the modulus belongs to a ladder nobody walked."""
    spec = replace(
        QUICK,
        n_replicas=1,
        load_stresses_bar=None,
        bulk_pressures_bar=None,
        shear_strains=None,
    )
    run_modulus_scan(argon_scan_run, "run", spec=spec, **QUICK_EQUILIBRATION)
    with pytest.raises(MechanicalError, match="different settings"):
        run_modulus_scan(
            argon_scan_run,
            "run",
            spec=replace(spec, relax_ps=0.5),
            **QUICK_EQUILIBRATION,
        )


# --------------------------------------------------------------------------
# Reading a finished directory
# --------------------------------------------------------------------------


def test_replicas_are_grouped_by_their_stems(tmp_path: Path) -> None:
    """Two replicas of two chunks each is two curves, not one and not four."""
    for replica in range(2):
        for chunk in range(2):
            write_deformation(
                tmp_path,
                n_steps=5,
                stage=f"{DEFORM_STEM}_r{replica}_{chunk:02d}",
            )
    report = analyse_mechanics(tmp_path, strain_limit=0.05)
    assert len(report.curves) == 2
    assert len(report.replicas) == 2
    assert report.youngs is not None


def test_a_skipped_bulk_pass_is_not_found_in_the_equilibration(
    tmp_path: Path,
) -> None:
    """The one place the find-by-shape rule is inverted, and why.

    A bulk ladder runs on the shared compress runner, so it records exactly
    what the equilibration's compression ladder records - and that one
    climbs far outside linear response at the melt temperature. Fitting it
    as a bulk modulus gives a confident number about nothing.
    """
    write_deformation(tmp_path)
    write_bulk(tmp_path, stage="03_compress", temperature_k=600.0)
    report = analyse_mechanics(tmp_path, strain_limit=0.05)
    assert report.bulk is None
    assert any("bulk" in note for note in report.notes)


def test_a_bulk_pass_that_did_run_is_read(tmp_path: Path) -> None:
    """And it is the one this workflow wrote, not the equilibration's."""
    write_deformation(tmp_path)
    write_bulk(tmp_path, stage="03_compress", modulus_mpa=99.0, temperature_k=600.0)
    write_bulk(tmp_path, stage=BULK_STEM, modulus_mpa=1500.0)
    report = analyse_mechanics(tmp_path, strain_limit=0.05)
    assert report.bulk is not None
    assert report.bulk.modulus_mpa == pytest.approx(1500.0)


def test_a_directory_with_no_mechanics_in_it_says_so(tmp_path: Path) -> None:
    """Rather than reporting a modulus of nothing."""
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "t",
                "seed": 1,
                "versions": {},
                "system": {},
                "stages": {},
                "chains": None,
                "box": None,
            }
        )
    )
    with pytest.raises(AnalysisError, match="mechanical measurement"):
        analyse_mechanics(tmp_path)


def test_the_two_methods_are_compared_when_both_ran(tmp_path: Path) -> None:
    """They share no machinery, so the gap between them is the real check."""
    write_deformation(tmp_path, modulus_mpa=2000.0)
    manifest = tmp_path / "manifest.json"
    record = json.loads(manifest.read_text())
    record["stages"][LOAD_STEM] = {
        "samples": {
            "segment_applied_stress_bar": [0.0, 100.0, 200.0],
            "segment_box_x_nm": [5.0, 5.0, 5.0],
            "segment_box_y_nm": [5.0, 5.0, 5.0],
            "segment_box_z_nm": [5.0, 5.025, 5.05],
            "load_axis": [2.0],
        },
        "mean_temperature_k": 298.15,
    }
    manifest.write_text(json.dumps(record))
    report = analyse_mechanics(tmp_path, strain_limit=0.05, min_points=3)
    assert report.load_modulus is not None
    assert report.load_modulus.modulus_mpa == pytest.approx(2000.0)
    assert report.method_gap == pytest.approx(0.0, abs=1e-9)


def test_one_replica_has_no_spread_rather_than_a_spread_of_zero(
    tmp_path: Path,
) -> None:
    """None, not 0.0 - a zero would read as runs that agreed perfectly."""
    write_deformation(tmp_path)
    assert analyse_mechanics(tmp_path, strain_limit=0.05).replica_spread_mpa is None


def test_analysing_writes_nothing(tmp_path: Path) -> None:
    """The discipline the whole analysis layer keeps."""
    write_deformation(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    analyse_mechanics(tmp_path, strain_limit=0.05)
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_the_report_writes_a_record_and_its_figures(tmp_path: Path) -> None:
    """One JSON, pinned field by field, and a figure per curve."""
    write_deformation(tmp_path, modulus_mpa=2000.0, poisson=0.35)
    write_bulk(tmp_path, stage=BULK_STEM, modulus_mpa=2222.0)
    write_shear(tmp_path, modulus_mpa=741.0)
    report = analyse_mechanics(tmp_path, strain_limit=0.05)
    files = write_mechanical_report(report, tmp_path / "analysis")

    record = json.loads(Path(files.json).read_text())
    assert record["youngs"]["modulus_mpa"] == pytest.approx(2000.0)
    assert record["poisson"]["ratio"] == pytest.approx(0.35)
    assert record["bulk"]["modulus_mpa"] == pytest.approx(2222.0)
    assert record["shear"]["modulus_mpa"] == pytest.approx(741.0)
    assert record["consistency"]["bulk_gap"] < MAX_CONSISTENCY_GAP
    assert record["consistency"]["consistent"]
    assert files.figures
    assert all(Path(path).is_file() for path in files.figures)


def test_the_record_carries_the_strain_rate_with_the_modulus(
    tmp_path: Path,
) -> None:
    """The caveat has to be in the file, not only in the log."""
    write_deformation(tmp_path)
    files = write_mechanical_report(
        analyse_mechanics(tmp_path, strain_limit=0.05),
        tmp_path / "analysis",
        figures=False,
    )
    record = json.loads(Path(files.json).read_text())
    assert record["youngs"]["strain_rate_per_ns"] is not None
    assert record["youngs"]["strain_rate_per_ns"] > 0.0
