"""Tests for the relaxation scan and what it reports.

Split the way the mechanical tests are split. Everything that decides
something - how the hold is chunked, which chunk measures a baseline, what a
resume refuses, how replicas are grouped, when a fit is contradicted - is
tested against manifests written by hand, because those are arithmetic over
recorded numbers. The plumbing that has to survive a real Context is tested
once, on argon, which runs the whole stage in a couple of seconds.

Argon is a liquid at these settings, so it has no relaxation modulus worth
quoting and every fit over it should refuse. That is deliberate: the argon
tests assert that the machinery did what it was told - the box moved by
exactly the strain and then did not move again, a resumed chunk carries on the
clock rather than restarting it - and leave every physical number to the
planted manifests, where the answer is known exactly.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.protocols import run_protocol
from openmmpolymer.relaxation import relaxation_curve
from openmmpolymer.simulate import relax_bin_edges_ps, run_minimise, run_relax
from openmmpolymer.stress import deviatoric_strain
from openmmpolymer.trajectory import AnalysisError
from openmmpolymer.viscoelastic import (
    LINEARITY_STEM,
    MAX_LINEARITY_GAP,
    RELAX_STEM,
    WORKFLOW_NAME,
    RelaxationSpec,
    ViscoelasticError,
    _relax_stages,
    _strains,
    analyse_relaxation,
    relax_protocol,
    relax_schedule,
    relaxation_scan,
    write_relaxation_report,
)

from .helpers import write_relaxation

# --------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"step_strain": 0.0}, "step_strain"),
        ({"relax_ps": -1.0}, "relax_ps"),
        ({"n_replicas": 0}, "n_replicas"),
        ({"mode": "twist"}, "mode"),
        ({"axis": 3}, "axis"),
        ({"plane": (1, 1)}, "plane"),
        ({"ramp_ps": -1.0}, "ramp_ps"),
        ({"sample_every_ps": 1.0e6}, "no time axis"),
        ({"linearity_strains": ()}, "pass None"),
        ({"linearity_strains": (0.0,)}, "non-zero"),
    ],
)
def test_a_spec_that_cannot_describe_a_relaxation_is_refused(
    change: dict[str, Any], message: str
) -> None:
    """At the call site, rather than several hours into the run."""
    with pytest.raises(ValueError, match=message):
        RelaxationSpec(**change)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_the_hold_is_split_into_chunks_no_longer_than_stage_ps() -> None:
    """Resume granularity, and nothing else: the chunks are continuous."""
    spec = RelaxationSpec(relax_ps=1000.0, stage_ps=300.0)
    schedule = relax_schedule(spec)
    assert schedule.n_chunks == 4
    stages = _relax_stages("06_relax_r0", spec, timestep_fs=2.0, strain=0.03)
    assert [stage.name for stage in stages] == [
        "06_relax_r0_00",
        "06_relax_r0_01",
        "06_relax_r0_02",
        "06_relax_r0_03",
    ]
    assert sum(stage.options["duration_ps"] for stage in stages) == pytest.approx(
        1000.0
    )


def test_only_the_first_chunk_strains_baselines_and_redraws_velocities() -> None:
    """The rest are a continuation of it, not a repeat of it.

    A chunk that strained again would measure the response to twice the
    strain while reporting it as one, and one that redrew its velocities
    would restart the trajectory in the middle of the decay.
    """
    spec = RelaxationSpec(relax_ps=1000.0, stage_ps=300.0, baseline_ps=500.0)
    stages = _relax_stages("06_relax_r0", spec, timestep_fs=2.0, strain=0.03)
    first, *rest = stages
    assert first.options["baseline_ps"] == 500.0
    assert first.options["new_velocities"] is True
    assert first.options["strain_applied"] is False
    for stage in rest:
        assert stage.options["baseline_ps"] == 0.0
        assert stage.options["ramp_ps"] == 0.0
        assert stage.options["new_velocities"] is False
        assert stage.options["strain_applied"] is True


def test_every_chunk_is_told_where_it_sits_on_the_relaxation_clock() -> None:
    """A state file does not carry it, and a chunk that assumed zero would
    fold the slow end of the decay onto the fast end."""
    spec = RelaxationSpec(relax_ps=900.0, stage_ps=300.0)
    stages = _relax_stages("06_relax_r0", spec, timestep_fs=2.0, strain=0.03)
    assert [stage.options["time_offset_ps"] for stage in stages] == [0.0, 300.0, 600.0]


def test_every_chunk_spans_the_whole_relaxation_in_its_bins() -> None:
    """Which is what puts chunks and replicas on one grid, so they merge."""
    spec = RelaxationSpec(relax_ps=900.0, stage_ps=300.0)
    stages = _relax_stages("06_relax_r0", spec, timestep_fs=2.0, strain=0.03)
    assert {stage.options["total_ps"] for stage in stages} == {900.0}


def test_replicas_differ_only_in_their_names_which_is_enough() -> None:
    """Every random stream derives from the stage label, so a differently
    named replica is a differently seeded one."""
    spec = RelaxationSpec(relax_ps=100.0, stage_ps=100.0)
    first = relax_protocol(spec, timestep_fs=2.0, replica=0)
    second = relax_protocol(spec, timestep_fs=2.0, replica=1)
    assert first.stages[0].name != second.stages[0].name
    assert first.stages[0].options == second.stages[0].options


def test_the_linearity_pass_runs_the_same_measurement_at_other_strains() -> None:
    """And is off unless asked for, because it doubles the scan."""
    assert _strains(RelaxationSpec()) == ((RELAX_STEM, 0.03),)
    spec = RelaxationSpec(step_strain=0.03, linearity_strains=(0.01, 0.06))
    assert _strains(spec) == (
        (RELAX_STEM, 0.03),
        (f"{LINEARITY_STEM}_e0", 0.01),
        (f"{LINEARITY_STEM}_e1", 0.06),
    )


def test_a_scan_over_budget_is_refused_before_it_starts(tmp_path: Path) -> None:
    """The point of a budget is that it is checked first."""
    from openmmpolymer.viscoelastic import _report_cost, equilibration_protocol

    spec = RelaxationSpec(relax_ps=100_000.0, n_replicas=8, max_total_ns=1.0)
    with pytest.raises(ViscoelasticError, match="over the"):
        _report_cost(equilibration_protocol(spec), relax_schedule(spec), spec, None)


def test_the_dry_run_protocol_prices_the_relaxation(tmp_path: Path) -> None:
    """A stage kind the estimator does not know prices at zero, which would
    quietly omit the most expensive thing in the protocol."""
    spec = RelaxationSpec(relax_ps=5000.0, baseline_ps=1000.0, stage_ps=5000.0)
    protocol = relaxation_scan(spec, nvt_ps=10.0, npt_ps=10.0)
    relax_only = sum(
        stage.options["duration_ps"] + stage.options["baseline_ps"]
        for stage in protocol.stages
        if stage.kind == "relax"
    )
    assert relax_only == pytest.approx(6000.0)
    assert protocol.total_duration_ps > 6000.0


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def test_a_resume_with_changed_settings_is_refused(tmp_path: Path) -> None:
    """Resuming would keep results measured under the old ones - and here the
    bin edges come from the settings, so it would merge two grids into one."""
    from openmmpolymer.viscoelastic import _check_request, _request

    spec = RelaxationSpec()
    (tmp_path / WORKFLOW_NAME).write_text(json.dumps({"request": _request(spec)}))
    _check_request(tmp_path, _request(spec))
    with pytest.raises(ViscoelasticError, match="step_strain"):
        _check_request(tmp_path, _request(replace(spec, step_strain=0.09)))


def test_a_spec_round_trips_through_json_before_it_is_compared(
    tmp_path: Path,
) -> None:
    """Its tuples come back as lists, which would make every second run look
    like a change of settings."""
    from openmmpolymer.viscoelastic import _check_request, _request

    spec = RelaxationSpec(linearity_strains=(0.01, 0.06), plane=(0, 2))
    (tmp_path / WORKFLOW_NAME).write_text(json.dumps({"request": _request(spec)}))
    assert _check_request(tmp_path, _request(spec)) is not None


# --------------------------------------------------------------------------
# Reading a finished run
# --------------------------------------------------------------------------


def test_replicas_are_grouped_by_their_stems_and_averaged(tmp_path: Path) -> None:
    """A hold split for resume is one curve; two replicas are not one."""
    for replica, modulus in enumerate((900.0, 1000.0, 1100.0)):
        write_relaxation(
            tmp_path,
            stem=f"{RELAX_STEM}_r{replica}",
            chunks=2,
            modulus_mpa=modulus,
            merge=json.loads((tmp_path / "manifest.json").read_text())["stages"]
            if (tmp_path / "manifest.json").is_file()
            else None,
        )
    report = analyse_relaxation(tmp_path)
    assert len(report.curves) == 3
    assert report.mean is not None
    assert report.mean.n_replicas == 3
    assert report.replica_spread_mpa is not None


def test_a_linearity_pass_is_reported_rather_than_averaged_in(
    tmp_path: Path,
) -> None:
    """Two strains measure different things, and a mean over them would hide
    exactly the difference the pass exists to show."""
    write_relaxation(tmp_path, stem=f"{RELAX_STEM}_r0", step_strain=0.03)
    stages = json.loads((tmp_path / "manifest.json").read_text())["stages"]
    write_relaxation(tmp_path, stem=f"{RELAX_STEM}_r1", step_strain=0.03, merge=stages)
    stages = json.loads((tmp_path / "manifest.json").read_text())["stages"]
    # A modulus twice as large at the other strain: firmly non-linear.
    write_relaxation(
        tmp_path,
        stem=f"{LINEARITY_STEM}_e0_r0",
        step_strain=0.06,
        modulus_mpa=2000.0,
        merge=stages,
    )
    report = analyse_relaxation(tmp_path)
    assert report.mean is not None
    assert report.mean.n_replicas == 2
    assert report.linearity is not None
    assert report.linearity.strains == (0.03, 0.06)
    assert report.linearity.gap > MAX_LINEARITY_GAP
    assert not report.linearity.linear


def test_a_plateau_the_stretched_exponential_cannot_hold_is_reported(
    tmp_path: Path,
) -> None:
    """Two functional forms disagreeing about whether the material relaxes
    completely, which is worth more than either agreeing with itself."""
    write_relaxation(
        tmp_path, modulus_mpa=1000.0, equilibrium_mpa=300.0, tau_ps=100.0, beta=0.6
    )
    report = analyse_relaxation(tmp_path)
    assert report.prony is not None
    assert report.prony.equilibrium_mpa == pytest.approx(300.0, rel=0.05)
    assert any("quote the spectrum" in note for note in report.notes)
    # The stretched exponential does not merely fit it badly - it refuses,
    # which is the same statement made by the fit that cannot hold a plateau.
    assert report.kww is not None and not report.kww.resolved
    assert not report.plateau_conflict


def test_a_curve_that_relaxes_to_zero_raises_no_conflict(tmp_path: Path) -> None:
    """A liquid has no equilibrium modulus, and that is the right answer."""
    write_relaxation(
        tmp_path, modulus_mpa=1000.0, equilibrium_mpa=0.0, tau_ps=100.0, beta=0.6
    )
    report = analyse_relaxation(tmp_path)
    assert not report.plateau_conflict
    assert not any("quote the spectrum" in note for note in report.notes)
    assert report.kww is not None and report.kww.resolved


def test_a_cell_that_was_already_stressed_says_so(tmp_path: Path) -> None:
    """The differential stress cancels an isotropic background, not a
    deviatoric one, so this is a caveat and not a correction."""
    write_relaxation(tmp_path, modulus_mpa=10.0, baseline_bar=500.0)
    report = analyse_relaxation(tmp_path)
    assert report.baseline_fraction > 0.25
    assert any("not the isotropic cell" in note for note in report.notes)


def test_a_directory_with_nothing_to_read_refuses(tmp_path: Path) -> None:
    """Rather than returning an empty report that reads like a measurement."""
    (tmp_path / "manifest.json").write_text(
        '{"protocol": "x", "seed": 1, "stages": {}}'
    )
    with pytest.raises(AnalysisError):
        analyse_relaxation(tmp_path)


def test_the_written_record_spells_out_what_is_on_disk(tmp_path: Path) -> None:
    """Field by field rather than by asdict, so the file's shape is pinned
    independently of how the dataclasses happen to be laid out."""
    write_relaxation(tmp_path)
    files = write_relaxation_report(analyse_relaxation(tmp_path), figures=False)
    record = json.loads(Path(files.json).read_text())
    assert set(record) == {
        "openmmpolymer",
        "run_dir",
        "versions",
        "stages",
        "mean",
        "kww",
        "prony",
        "replicas",
        "replica_spread_mpa",
        "linearity",
        "plateau_conflict",
        "baseline_fraction",
        "notes",
    }
    assert record["kww"]["resolved"] is True
    assert isinstance(record["prony"]["tau_ps"], list)


# --------------------------------------------------------------------------
# The plumbing, on argon
# --------------------------------------------------------------------------


@pytest.fixture
def relaxed_argon(argon_run: Any) -> Any:
    """A short tensile relaxation of the argon cell, minimised first."""
    minimised = run_minimise(argon_run, "00_minimise", temperature_k=120.0)
    return run_relax(
        argon_run,
        "06_relax_r0_00",
        temperature_k=120.0,
        step_strain=0.04,
        baseline_ps=1.0,
        duration_ps=5.0,
        sample_every_ps=0.05,
        late_sample_every_ps=0.2,
        late_after_ps=1.0,
        bins_per_decade=10,
        timestep_fs=2.0,
        state_in=minimised.final_state,
    )


def _box_nm(state_path: str) -> np.ndarray:
    """The three cell edges a serialised state carries."""
    import openmm as mm
    from openmm import unit

    state = mm.XmlSerializer.deserialize(Path(state_path).read_text())
    vectors = state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    return np.asarray([vectors[axis][axis] for axis in range(3)])


def test_the_cell_is_strained_by_exactly_what_was_asked_for(
    argon_run: Any, relaxed_argon: Any
) -> None:
    """Axially by the strain, laterally by (1 + strain) ** -poisson, and at
    the default ratio the volume is preserved exactly."""
    before = _box_nm(run_minimise(argon_run, "check", temperature_k=120.0).final_state)
    after = _box_nm(relaxed_argon.final_state)
    assert after[2] / before[2] == pytest.approx(1.04, abs=1e-12)
    assert after[0] / before[0] == pytest.approx(1.04**-0.5, abs=1e-12)
    assert np.prod(after) / np.prod(before) == pytest.approx(1.0, abs=1e-12)


def test_the_cell_does_not_move_again_for_the_whole_hold(
    relaxed_argon: Any,
) -> None:
    """The one thing this stage must guarantee. The applied strain is the
    measurement, so a barostat that moves the box is measuring something else
    - and the runner raises rather than reporting it."""
    recorded = relaxed_argon.samples["relax_volume_ratio"][0]
    assert recorded == pytest.approx(1.0, abs=1e-9)


def test_the_probe_barostat_reports_a_tensor_without_moving_anything(
    relaxed_argon: Any,
) -> None:
    """A barostat at frequency=0 exists only to be asked: OpenMM reports a
    pressure through one and refuses it for a force not in the Context."""
    samples = relaxed_argon.samples
    for name in ("xx", "yy", "zz"):
        assert np.all(np.isfinite(samples[f"segment_stress_{name}_bar"]))


def test_the_recorded_strain_measure_is_what_a_modulus_divides_by(
    relaxed_argon: Any,
) -> None:
    """2 (e_axial - e_lateral) for a tensile step, computed from the factors
    actually applied rather than linearised."""
    assert relaxed_argon.samples["relax_strain_measure"][0] == pytest.approx(
        2.0 * deviatoric_strain(0.04, 0.5)
    )


def test_a_shear_step_records_its_plane_and_a_tensile_one_its_axis(
    argon_run: Any,
) -> None:
    """Which is how the reader tells them apart, without either being named."""
    minimised = run_minimise(argon_run, "00_minimise", temperature_k=120.0)
    common: dict[str, Any] = {
        "temperature_k": 120.0,
        "baseline_ps": 0.5,
        "duration_ps": 2.0,
        "sample_every_ps": 0.05,
        "late_sample_every_ps": 0.2,
        "late_after_ps": 1.0,
        "bins_per_decade": 8,
        "timestep_fs": 2.0,
        "state_in": minimised.final_state,
    }
    shear = run_relax(argon_run, "s", mode="shear", step_strain=0.02, **common)
    tensile = run_relax(argon_run, "t", mode="tensile", step_strain=0.02, **common)
    assert "relax_plane" in shear.samples and "relax_axis" not in shear.samples
    assert "relax_axis" in tensile.samples and "relax_plane" not in tensile.samples
    assert shear.samples["stress_estimator_version"] == [1.0]
    # A shear step measures G directly, so it divides by the strain itself.
    assert shear.samples["relax_strain_measure"][0] == pytest.approx(0.02)


def test_every_reading_is_written_beside_the_binned_curve(
    relaxed_argon: Any,
) -> None:
    """So the decay can be re-binned or analysed another way without running
    the whole thing again. All-numeric, so genfromtxt reads it."""
    raw = np.genfromtxt("06_relax_r0_00_stress.csv", delimiter=",", names=True)
    assert raw.dtype.names == (
        "time_ps",
        "sigma_xx_bar",
        "sigma_yy_bar",
        "sigma_zz_bar",
        "sigma_bar",
    )
    assert raw.shape[0] == int(sum(relaxed_argon.samples["segment_samples"]))


def test_a_resumed_chunk_carries_on_the_clock_rather_than_restarting_it(
    argon_run: Any,
) -> None:
    """A state file does not carry the time since the strain, so the chunk is
    told - and a chunk that assumed zero would fold the slow end of the decay
    back onto the fast end."""
    minimised = run_minimise(argon_run, "00_minimise", temperature_k=120.0)
    common: dict[str, Any] = {
        "temperature_k": 120.0,
        "step_strain": 0.04,
        "total_ps": 10.0,
        "sample_every_ps": 0.05,
        "late_sample_every_ps": 0.2,
        "late_after_ps": 1.0,
        "bins_per_decade": 10,
        "timestep_fs": 2.0,
    }
    first = run_relax(
        argon_run,
        "06_relax_r0_00",
        baseline_ps=1.0,
        duration_ps=5.0,
        time_offset_ps=0.0,
        state_in=minimised.final_state,
        **common,
    )
    second = run_relax(
        argon_run,
        "06_relax_r0_01",
        baseline_ps=0.0,
        duration_ps=5.0,
        time_offset_ps=5.0,
        strain_applied=True,
        state_in=first.final_state,
        **common,
    )
    assert min(second.samples["segment_relax_time_ps"]) > max(
        first.samples["segment_relax_time_ps"]
    )
    assert min(second.samples["segment_bin"]) > max(first.samples["segment_bin"])
    # The continuation neither strains again nor measures a second baseline.
    assert "baseline_stress_bar" not in second.samples
    assert _box_nm(second.final_state) == pytest.approx(_box_nm(first.final_state))


def test_every_replica_and_chunk_lands_on_one_grid(argon_run: Any) -> None:
    """Derived from the settings and never from the data, which is what makes
    merging chunks and averaging replicas the same addition."""
    del argon_run
    grid = relax_bin_edges_ps(0.05, 10.0, 10)
    # Bit-identical, not merely close: the bins are added together by index,
    # so a grid that drifted would silently line up different times.
    assert np.array_equal(relax_bin_edges_ps(0.05, 10.0, 10), grid)
    # And it is the *total* that sets the grid, not one chunk's duration -
    # which is why every chunk is handed the total.
    assert not np.array_equal(relax_bin_edges_ps(0.05, 5.0, 10), grid)


def test_a_relaxation_run_through_the_protocol_reads_back(argon_run: Any) -> None:
    """End to end on the smallest thing that exercises it: two chunks, run by
    the protocol runner, read back as one curve."""
    minimised = run_minimise(argon_run, "00_minimise", temperature_k=120.0)
    spec = RelaxationSpec(
        temperature_k=120.0,
        step_strain=0.04,
        baseline_ps=1.0,
        relax_ps=4.0,
        stage_ps=2.0,
        sample_every_ps=0.05,
        late_sample_every_ps=0.2,
        late_after_ps=1.0,
        bins_per_decade=10,
    )
    summary = run_protocol(
        relax_protocol(spec, timestep_fs=2.0, replica=0),
        argon_run,
        "run",
        state_in=minimised.final_state,
    )
    assert len(summary.results) == 2
    curve = relaxation_curve("run")
    assert curve.n_points > 5
    assert curve.mode == "tensile"
    assert curve.step_strain == pytest.approx(0.04)
    # Argon is a liquid, so it has no static relaxation modulus and nothing
    # here should claim one. Asserting the refusal rather than a number is the
    # whole point of testing the plumbing on argon.
    report = analyse_relaxation("run")
    assert report.mean is not None
    assert report.kww is not None and not report.kww.resolved


@pytest.fixture
def scanned_argon(argon_run: Any, request: pytest.FixtureRequest) -> Any:
    """A whole relaxation scan over argon, equilibration and all.

    Deliberately the smallest thing that exercises the driver end to end: two
    replicas so there is a spread, one linearity strain so there is a
    comparison, and an equilibration cut to the bone because what is being
    tested is the wiring and not the melt.
    """
    from openmmpolymer.viscoelastic import run_relaxation_scan

    spec = RelaxationSpec(
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
        linearity_strains=(0.02,),
    )
    return spec, run_relaxation_scan(
        argon_run,
        "run",
        spec=spec,
        resume=getattr(request, "param", True),
        melt_temperature_k=150.0,
        nvt_ps=0.5,
        compress_ps_each=0.2,
        npt_ps=0.5,
        anneal_cycles=1,
        anneal_window_ps=0.1,
        anneal_hold_ps=0.1,
        compress_pressures_bar=(1.0, 50.0),
    )


@pytest.mark.parametrize("scanned_argon", [True, False], indirect=True)
def test_a_whole_scan_runs_and_reports(scanned_argon: Any) -> None:
    """Every replica branches from the equilibrated cell, and the record says
    which one."""
    spec, result = scanned_argon
    assert len(result.curves) == spec.n_replicas * len(_strains(spec))
    assert result.mean is not None and result.mean.n_replicas == spec.n_replicas
    assert result.replica_spread_mpa is not None
    assert result.linearity is not None
    record = json.loads((Path("run") / WORKFLOW_NAME).read_text())
    assert record["n_replicas"] == spec.n_replicas
    assert Path(record["start_state"]).name == "05_npt.state.xml"
    # Argon is a liquid; nothing about it should come back resolved.
    assert not result.resolved


def test_every_replica_starts_from_the_equilibrated_cell_not_a_strained_one(
    scanned_argon: Any,
) -> None:
    """The manifest is in run order, so the last entry of a finished scan is
    the end of somebody's strained hold. Branching from that would measure a
    cell that had already been pulled, and the reference length would be its
    stretched one rather than the equilibrated one."""
    _, result = scanned_argon
    record = json.loads((Path("run") / WORKFLOW_NAME).read_text())
    origin = np.asarray(record["reference_box_nm"])
    assert origin == pytest.approx(origin[0])  # equilibrated, so still cubic
    del result


def test_a_scan_resumes_without_repeating_anything(
    argon_run: Any, scanned_argon: Any
) -> None:
    """The curve has to come back identical, not merely similar: a resumed
    chunk that redid any work would change it."""
    from openmmpolymer.viscoelastic import run_relaxation_scan

    spec, first = scanned_argon
    again = run_relaxation_scan(
        argon_run,
        "run",
        spec=spec,
        melt_temperature_k=150.0,
        nvt_ps=0.5,
        compress_ps_each=0.2,
        npt_ps=0.5,
        anneal_cycles=1,
        anneal_window_ps=0.1,
        anneal_hold_ps=0.1,
        compress_pressures_bar=(1.0, 50.0),
    )
    assert first.mean is not None and again.mean is not None
    assert np.array_equal(again.mean.modulus_mpa, first.mean.modulus_mpa)


def test_a_resumed_scan_with_different_settings_is_refused(
    argon_run: Any, scanned_argon: Any
) -> None:
    """Otherwise it would keep results measured under the old ones - and the
    bin edges come from the settings, so it would merge two grids."""
    from openmmpolymer.viscoelastic import run_relaxation_scan

    spec, _ = scanned_argon
    with pytest.raises(ViscoelasticError, match="different settings"):
        run_relaxation_scan(argon_run, "run", spec=replace(spec, step_strain=0.09))


def test_the_report_writes_its_figures_beside_the_record(
    scanned_argon: Any,
) -> None:
    """Saving is the driver's job; the plotting helpers only build figures."""
    del scanned_argon
    files = write_relaxation_report(analyse_relaxation("run"))
    assert Path(files.json).name == "relaxation.json"
    assert files.figures
    assert all(Path(path).is_file() for path in files.figures)


def test_the_linearity_pass_does_not_become_the_headline_result(
    tmp_path: Path,
) -> None:
    """The check must not displace the thing it was checking.

    A linearity pass records exactly what the measurement records - it is the
    same measurement at another strain - and the workflow runs the same number
    of replicas of each. So nothing in the samples and nothing in the counts
    tells them apart, and tie-breaking on the strain itself would quietly make
    a larger linearity strain the reported answer: the headline modulus and
    both fits would belong to the pass that only existed to check the other.
    """
    stages: dict[str, Any] | None = None
    for replica in range(2):
        write_relaxation(
            tmp_path,
            stem=f"{RELAX_STEM}_r{replica}",
            step_strain=0.03,
            modulus_mpa=1000.0,
            merge=stages,
        )
        stages = json.loads((tmp_path / "manifest.json").read_text())["stages"]
    for replica in range(2):
        write_relaxation(
            tmp_path,
            stem=f"{LINEARITY_STEM}_e0_r{replica}",
            step_strain=0.06,
            modulus_mpa=4000.0,
            merge=stages,
        )
        stages = json.loads((tmp_path / "manifest.json").read_text())["stages"]

    report = analyse_relaxation(tmp_path)
    assert report.mean is not None
    assert report.mean.step_strain == pytest.approx(0.03)
    assert report.mean.initial_modulus_mpa < 1100.0
    assert RELAX_STEM in report.mean.stage
    assert LINEARITY_STEM not in report.mean.stage
    # Both still appear in the comparison, which is the pass's actual job.
    assert report.linearity is not None
    assert report.linearity.strains == (0.03, 0.06)
