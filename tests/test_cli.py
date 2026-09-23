"""Tests for the command-line driver."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmmpolymer.__main__ import (
    _DESTS,
    PROTOCOLS,
    _modulus_spec,
    _protocol_options,
    _tg_spec,
    _tm_spec,
    build_parser,
    main,
)
from openmmpolymer.protocols import melt_quench

from .helpers import (
    transition_at,
    two_line_curve,
    write_deformation,
    write_polymer_snapshot,
    write_quench,
)


def test_the_parser_says_what_the_command_does() -> None:
    """It is the first thing anyone reads."""
    parser = build_parser()
    assert parser.prog == "openmmpolymer"
    assert "polymer melt" in (parser.description or "")


def test_the_monomer_is_the_one_required_argument() -> None:
    """Everything else has a default that works."""
    arguments = build_parser().parse_args(["[*]CC[*]"])
    assert arguments.monomer == "[*]CC[*]"
    assert arguments.degree_of_polymerization == 20
    assert arguments.chains == 30


def test_cli_passes_polyester_caps_and_polymer_dimensions_to_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported polyester must be constructible through the CLI as well."""
    from openmmpolymer import __main__ as cli
    from openmmpolymer.chain import ChainSpec, assemble_chain

    class BuildReached(Exception):
        pass

    def build(spec: ChainSpec, *args: Any, **kwargs: Any) -> Any:
        assert spec.head_cap == "[*][H]"
        assert spec.tail_cap == "[*]O"
        assert spec.characteristic_ratio == 5.5
        assert assemble_chain(spec).GetNumAtoms() > 0
        raise BuildReached

    monkeypatch.setattr(cli, "build_chain", build)
    with pytest.raises(BuildReached):
        main(
            [
                "[*]OC(C)C(=O)[*]",
                "-n",
                "3",
                "--head-cap",
                "[*][H]",
                "--tail-cap",
                "[*]O",
                "--characteristic-ratio",
                "5.5",
                "--dry-run",
            ]
        )


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_characteristic_ratio_is_rejected_by_the_parser(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["[*]CC[*]", "--characteristic-ratio", value])


def test_default_build_ratio_and_request_identity_remain_seven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmmpolymer import __main__ as cli
    from openmmpolymer.chain import ChainSpec

    class BuildReached(Exception):
        pass

    def build(spec: ChainSpec, *args: Any, **kwargs: Any) -> Any:
        assert spec.characteristic_ratio == 7.0
        raise BuildReached

    def request(output: Path, options: dict[str, Any]) -> None:
        assert options["characteristic_ratio"] == 7.0

    monkeypatch.setattr(cli, "build_chain", build)
    monkeypatch.setattr(cli, "record_build_request", request)
    with pytest.raises(BuildReached):
        main(["[*]CC[*]", "--dry-run"])


@pytest.mark.parametrize(
    "changed",
    [["--seed", "7"], ["--temperature", "500"], ["--tail-cap", "[*]O"]],
)
def test_changed_cli_request_is_refused_before_overwriting_build_assets(
    changed: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openmmpolymer import __main__ as cli

    class BuildReached(Exception):
        pass

    def build(*args: Any, **kwargs: Any) -> Any:
        raise BuildReached

    monkeypatch.setattr(cli, "build_chain", build)
    arguments = ["[*]CC[*]", "-o", str(tmp_path / "run")]
    with pytest.raises(BuildReached):
        main(arguments)
    artifact = tmp_path / "run" / "build" / "polymer_ff.xml"
    artifact.parent.mkdir()
    artifact.write_text("existing parameters")
    with pytest.raises(SystemExit, match="2"):
        main([*arguments, *changed])
    assert artifact.read_text() == "existing parameters"
    # Presentation, device selection and switching from build-only to dynamics
    # do not change the prepared physical system.
    with pytest.raises(BuildReached):
        main([*arguments, "--platform", "CPU", "--dry-run", "-v"])


@pytest.fixture
def staged_cli_build(monkeypatch: pytest.MonkeyPatch, argon_run: Any) -> dict[str, Any]:
    """Real input fingerprints with inexpensive stand-ins for chemistry tools."""
    from openmmpolymer import __main__ as cli
    from openmmpolymer.chain import ChainResult

    control: dict[str, Any] = {"system_suffix": "", "fail": False, "paths": []}

    def build(arguments: Any, build_dir: Path, cache_dir: Path) -> Any:
        del arguments, cache_dir
        control["paths"].append(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        for name in ("chain_0.sdf", "chain_0.pdb", "polymer_ff.xml", "packed.pdb"):
            (build_dir / name).write_text(f"prepared artifact {len(control['paths'])}")
        if control["fail"]:
            raise RuntimeError("preparation failed")
        chain = ChainResult(
            sdf_paths=(str(build_dir / "chain_0.sdf"),),
            pdb_paths=(str(build_dir / "chain_0.pdb"),),
            smiles="[Ar]",
            n_atoms=1,
            molar_mass_g_mol=39.948,
        )
        run = replace(
            argon_run,
            system_xml=argon_run.system_xml + control["system_suffix"],
            forcefield=replace(
                argon_run.forcefield, forcefield_xml=str(build_dir / "polymer_ff.xml")
            ),
        )
        return chain, run

    monkeypatch.setattr(cli, "_build_cli_melt", build)
    return control


def _cli_artifacts() -> dict[str, bytes]:
    return {
        str(path): path.read_bytes()
        for path in Path("run").rglob("*")
        if path.is_file()
    }


def test_repeated_cli_preparation_preserves_original_build_files(
    staged_cli_build: dict[str, Any],
) -> None:
    assert main(["[*]CC[*]", "--dry-run"]) == 0
    before = _cli_artifacts()
    assert main(["[*]CC[*]", "--dry-run", "--platform", "CPU"]) == 0
    assert _cli_artifacts() == before
    assert staged_cli_build["paths"][0] == Path("run/build")
    assert staged_cli_build["paths"][1] != Path("run/build")
    assert not staged_cli_build["paths"][1].exists()


@pytest.mark.parametrize("failure", ["different_system", "build_error"])
def test_failed_cli_repreparation_cannot_overwrite_existing_assets(
    staged_cli_build: dict[str, Any], failure: str
) -> None:
    assert main(["[*]CC[*]", "--dry-run"]) == 0
    before = _cli_artifacts()
    if failure == "different_system":
        staged_cli_build["system_suffix"] = "\n"
        expected: type[BaseException] = SystemExit
    else:
        staged_cli_build["fail"] = True
        expected = RuntimeError
    with pytest.raises(expected):
        main(["[*]CC[*]", "--dry-run"])
    assert _cli_artifacts() == before
    assert not list(Path("run").glob(".build-check-*"))


def test_cli_dry_run_can_continue_to_dynamics_using_verified_build_files(
    staged_cli_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from openmmpolymer import __main__ as cli

    class DynamicsReached(Exception):
        pass

    def dynamics(protocol: Any, run: Any, *args: Any, **kwargs: Any) -> Any:
        assert Path(run.forcefield.forcefield_xml) == Path("run/build/polymer_ff.xml")
        assert Path(run.forcefield.forcefield_xml).is_file()
        raise DynamicsReached

    assert main(["[*]CC[*]", "--dry-run"]) == 0
    before = _cli_artifacts()
    monkeypatch.setattr(cli, "run_protocol", dynamics)
    with pytest.raises(DynamicsReached):
        main(["[*]CC[*]"])
    assert _cli_artifacts() == before


@pytest.mark.parametrize("manifest_dir", ["run", "run/equilibration"])
def test_cli_checks_actual_inputs_against_existing_workflow_manifests(
    staged_cli_build: dict[str, Any], argon_run: Any, manifest_dir: str
) -> None:
    from openmmpolymer.protocols import Protocol, Stage, run_protocol

    assert main(["[*]CC[*]", "--dry-run"]) == 0
    run_protocol(
        Protocol("other", (Stage("00_minimise", "minimise"),)),
        replace(argon_run, seed=99),
        manifest_dir,
    )
    before = _cli_artifacts()
    with pytest.raises(SystemExit):
        main(["[*]CC[*]", "--dry-run"])
    assert _cli_artifacts() == before


def test_cli_refuses_a_modified_original_forcefield_before_rebuilding(
    staged_cli_build: dict[str, Any],
) -> None:
    assert main(["[*]CC[*]", "--dry-run"]) == 0
    Path("run/build/polymer_ff.xml").write_text("modified original")
    before = _cli_artifacts()
    with pytest.raises(SystemExit):
        main(["[*]CC[*]", "--dry-run"])
    assert len(staged_cli_build["paths"]) == 1
    assert _cli_artifacts() == before


def test_the_parser_rejects_a_protocol_it_cannot_run() -> None:
    """argparse catches it before any chemistry starts."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["[*]CC[*]", "--protocol", "anneal-forever"])


def test_every_offered_protocol_is_buildable() -> None:
    """The choices and the factories cannot drift apart."""
    for entry in PROTOCOLS.values():
        assert entry.factory().stages


def test_the_parser_rejects_an_unknown_charge_method() -> None:
    """Same reason: fail at the front door."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["[*]CC[*]", "--charge-method", "am1bbc"])


def test_the_cell_size_guard_reaches_the_command_line() -> None:
    """A cell too small for the cutoff is refused before anything expensive."""
    from openmmpolymer.mdsystem import SystemAssemblyError

    with pytest.raises(SystemAssemblyError, match="times as many chains"):
        main(["[*]CC[*]", "-n", "4", "-c", "2", "--dry-run", "-o", "out"])


@pytest.mark.packmol
@pytest.mark.slow
def test_a_dry_run_builds_packs_and_stops(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole build half of the pipeline, without dynamics."""
    import shutil

    if shutil.which("packmol") is None:
        pytest.skip("packmol is not on PATH")
    exit_code = main(
        [
            "[*]CC[*]",
            "-n",
            "6",
            "-c",
            "40",
            "-r",
            "PE",
            "--charge-method",
            "gasteiger",
            # Checked against the density this cell will actually reach, not
            # a melt's: the point here is the wiring, not the physics.
            "--target-density",
            "0.3",
            "--dry-run",
            "-o",
            "out",
        ]
    )
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "dry run" in captured
    assert Path("out/build/packed.pdb").is_file()
    assert Path("out/build/polymer_ff.xml").is_file()


def test_every_protocol_option_is_a_real_parameter_of_its_factory() -> None:
    """The table and the factories cannot drift apart without a failure here.

    A flag that reaches no factory is parsed, ignored, and never arrives -
    which is exactly how t_end, step_k and hold_ps came to be unreachable
    from the command line while looking perfectly present in --help.
    """
    for name, entry in PROTOCOLS.items():
        parameters = inspect.signature(entry.factory).parameters
        takes_anything = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        for option in entry.options:
            assert takes_anything or option in parameters, (name, option)


def test_the_flat_cooling_flags_reach_the_spec_a_scan_takes() -> None:
    """The tg factory takes **kwargs, so the check above cannot see it."""
    parameters = inspect.signature(_tg_spec).parameters
    for option in PROTOCOLS["tg"].options:
        assert option in parameters, option


def test_the_quench_controls_reach_the_protocol(tmp_path: Path) -> None:
    """They were unreachable: main passed three keywords and no more."""
    arguments = build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "melt-quench",
            "--t-start",
            "640",
            "--t-end",
            "160",
            "--step-k",
            "25",
            "--hold-ps",
            "1000",
        ]
    )
    entry = PROTOCOLS["melt-quench"]
    options = {
        name: getattr(arguments, _DESTS.get(name, name)) for name in entry.options
    }
    quench = entry.factory(**options).stages[-1].options

    assert quench["t_start"] == pytest.approx(640.0)
    assert quench["t_end"] == pytest.approx(160.0)
    assert quench["step_k"] == pytest.approx(25.0)
    assert quench["hold_ps"] == pytest.approx(1000.0)


def test_a_start_temperature_defaults_to_the_melt_temperature() -> None:
    """What every existing invocation has always got."""
    arguments = build_parser().parse_args(["[*]CC[*]"])
    assert arguments.t_start is None
    assert melt_quench(t_start=None).stages[-1].options["t_start"] == pytest.approx(
        600.0
    )


def test_a_malformed_cooling_rate_list_is_refused_at_the_front_door() -> None:
    """argparse catches it before any chemistry starts."""
    for bad in ("10,fast,2", "10", "10,10", "0,10"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["[*]CC[*]", "--cooling-rates", bad])


def test_a_well_formed_cooling_rate_list_is_accepted() -> None:
    """Three rates, in the order they were written."""
    arguments = build_parser().parse_args(["[*]CC[*]", "--cooling-rates", "10,5,2"])
    assert arguments.cooling_rates == (10.0, 5.0, 2.0)


def test_the_monomer_is_not_needed_to_read_a_finished_directory() -> None:
    """Reading a run back is a different verb over a different input."""
    arguments = build_parser().parse_args(["--analyse", "run", "other"])

    assert arguments.monomer is None
    assert arguments.analyse == ["run", "other"]


def test_building_a_melt_still_needs_a_monomer() -> None:
    """The one required argument, unless the other mode was asked for."""
    with pytest.raises(SystemExit):
        main([])


def test_melting_and_cooling_keep_their_own_defaults() -> None:
    """Adding heating must not reverse any existing cooling command."""
    parser = build_parser()
    cooling = parser.parse_args(["[*]CC[*]", "--protocol", "melt-quench"])
    heating = parser.parse_args(["--protocol", "tm"])
    assert (cooling.t_start, cooling.t_end, cooling.step_k, cooling.hold_ps) == (
        None,
        200.0,
        20.0,
        200.0,
    )
    assert (heating.t_start, heating.t_end, heating.step_k, heating.hold_ps) == (
        250.0,
        650.0,
        10.0,
        1000.0,
    )
    for option in PROTOCOLS["tm"].options:
        assert option in inspect.signature(_tm_spec).parameters
    explicit = parser.parse_args(
        [
            "--t-start",
            "100",
            "--t-end",
            "200",
            "--step-k",
            "5",
            "--hold-ps",
            "10",
            "--protocol",
            "tm",
        ]
    )
    spec = _tm_spec(**_protocol_options(explicit, PROTOCOLS["tm"]))
    assert (spec.t_start_k, spec.t_end_k, spec.step_k, spec.hold_ps) == (
        100,
        200,
        5,
        10,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--protocol", "tm"],
        ["--protocol", "tm", "--crystal-pdb", "crystal.pdb"],
        [
            "[*]CC[*]",
            "--protocol",
            "tm",
            "--crystal-pdb",
            "crystal.pdb",
            "--system-xml",
            "system.xml",
        ],
        ["[*]CC[*]", "--crystal-pdb", "crystal.pdb"],
    ],
)
def test_melting_requires_a_prepared_crystal_before_creating_files(
    argv: list[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main([*argv, "-o", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "controls",
    [
        ["--t-start", "500", "--t-end", "300"],
        ["--step-k", "0"],
        ["--hold-ps", "-1"],
        ["--max-total-ns", "0.001"],
    ],
)
def test_melting_rejects_invalid_schedules_before_reading_the_crystal(
    controls: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import openmmpolymer.__main__ as cli

    def unexpected_read(arguments: Any) -> Any:
        raise AssertionError("The schedule should fail before the crystal is read")

    monkeypatch.setattr(cli, "_prepared_crystal", unexpected_read)
    with pytest.raises(SystemExit):
        main(
            [
                "--protocol",
                "tm",
                "--crystal-pdb",
                "crystal.pdb",
                "--system-xml",
                "system.xml",
                "-o",
                str(tmp_path / "output"),
                *controls,
            ]
        )
    assert not (tmp_path / "output").exists()


def _crystal_files(argon_box: tuple[Any, Any], directory: Path) -> list[str]:
    """Write a prepared periodic cell and exactly its serialized System."""
    import openmm as mm
    from openmm import app

    box, system = argon_box
    pdb = directory / "crystal.pdb"
    xml = directory / "system.xml"
    with pdb.open("w") as stream:
        app.PDBFile.writeFile(box.topology, box.positions, stream)
    xml.write_text(mm.XmlSerializer.serialize(system))
    return ["--crystal-pdb", str(pdb), "--system-xml", str(xml)]


def test_melting_dry_run_validates_prepared_inputs_without_building(
    argon_box: tuple[Any, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _crystal_files(argon_box, tmp_path)
    output = tmp_path / "output"
    assert main(["--protocol", "tm", *args, "--dry-run", "-o", str(output)]) == 0
    assert "crystal and heating schedule validated" in capsys.readouterr().out
    assert not output.exists()


def test_melting_dry_run_accepts_a_matching_crystalline_state(
    argon_box: tuple[Any, Any], tmp_path: Path
) -> None:
    import openmm as mm

    box, system = argon_box
    args = _crystal_files(argon_box, tmp_path)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(box.positions)
    context.setVelocitiesToTemperature(250.0, 1)
    state = tmp_path / "crystal-state.xml"
    state.write_text(
        mm.XmlSerializer.serialize(
            context.getState(getPositions=True, getVelocities=True)
        )
    )
    assert main(["--protocol", "tm", *args, "--state-in", str(state), "--dry-run"]) == 0


@pytest.mark.parametrize(
    "invalid", ["atom_count", "barostat", "thermostat", "box", "periodicity", "state"]
)
def test_prepared_crystal_inputs_are_checked_before_the_scan(
    invalid: str, argon_box: tuple[Any, Any], tmp_path: Path
) -> None:
    import openmm as mm

    box, system = argon_box
    if invalid == "atom_count":
        system.addParticle(1.0)
    elif invalid == "barostat":
        system.addForce(mm.MonteCarloBarostat(1.0, 300.0))
    elif invalid == "thermostat":
        system.addForce(mm.AndersenThermostat(300.0, 1.0))
    elif invalid == "periodicity":
        system.getForce(0).setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    args = _crystal_files((box, system), tmp_path)
    if invalid == "box":
        pdb = tmp_path / "crystal.pdb"
        pdb.write_text(
            "\n".join(
                line
                for line in pdb.read_text().splitlines()
                if not line.startswith("CRYST1")
            )
        )
    elif invalid == "state":
        args.extend(["--state-in", str(tmp_path / "missing-state.xml")])
    with pytest.raises(SystemExit):
        main(["--protocol", "tm", *args, "--dry-run", "-o", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("interleaved", [False, True])
def test_crystal_molecule_layout_must_match_the_reporters_assumptions(
    interleaved: bool,
    argon_box: tuple[Any, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    box, _ = argon_box
    atoms = list(box.topology.atoms())
    if interleaved:
        pairs = [
            (0, 2),
            (1, 3),
            *[(index, index + 1) for index in range(4, len(atoms), 2)],
        ]
    else:
        pairs = [(0, 1)]
    for left, right in pairs:
        box.topology.addBond(atoms[left], atoms[right])
    args = _crystal_files(argon_box, tmp_path)
    with pytest.raises(SystemExit):
        main(["--protocol", "tm", *args, "--dry-run"])
    message = "contiguous atom blocks" if interleaved else "equal atom counts"
    assert message in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_melting_cli_dispatches_the_prepared_cell_and_explicit_schedule(
    argon_box: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import openmmpolymer.__main__ as cli

    seen: dict[str, Any] = {}
    report = SimpleNamespace(notes=("Finite heating rate; inspect crystalline order.",))

    def scan(run: Any, output: Path, **kwargs: Any) -> Any:
        seen.update(kwargs, run=run, output=output)
        return SimpleNamespace(
            temperature_k=415.0, bracket_k=(410.0, 420.0), resolved=True, report=report
        )

    def write(actual: Any, **kwargs: Any) -> Any:
        assert actual is report
        assert kwargs["figures"] is False
        return SimpleNamespace(json=tmp_path / "output/analysis/tm.json", figures=())

    monkeypatch.setattr(cli, "run_tm_scan", scan)
    monkeypatch.setattr(cli, "write_melting_report", write)
    args = _crystal_files(argon_box, tmp_path)
    assert (
        main(
            [
                "--protocol",
                "tm",
                *args,
                "--t-start",
                "280",
                "--t-end",
                "500",
                "--step-k",
                "20",
                "--hold-ps",
                "50",
                "--tm-equilibration-ps",
                "10",
                "--tm-stage-ps",
                "100",
                "--no-figures",
                "-o",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    assert seen["crystalline"] is True
    assert seen["state_in"] is None
    assert seen["spec"].t_start_k == 280
    assert seen["spec"].t_end_k == 500
    assert seen["spec"].hold_ps == 50
    assert seen["run"].box.topology.getNumAtoms() == argon_box[0].topology.getNumAtoms()
    assert seen["run"].box.n_molecules == 64
    assert seen["run"].spec.constraints == "none"
    assert "apparent Tm = 415 K (heating bracket 410-420 K)" in capsys.readouterr().out


@pytest.mark.parametrize("resolved", [True, False])
def test_analyse_dispatches_melting_and_reports_unresolved_results(
    resolved: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import openmmpolymer.__main__ as cli
    from openmmpolymer.tm import heating_stages

    transition = SimpleNamespace(
        temperature_k=415.0 if resolved else None,
        bracket_k=(410.0, 420.0) if resolved else None,
        resolved=resolved,
    )
    report = SimpleNamespace(transition=transition, notes=("Finite heating scan.",))
    monkeypatch.setattr(cli, "_has_stages", lambda path, find: find is heating_stages)

    def analyse(run_dir: Path, **kwargs: Any) -> Any:
        assert run_dir == tmp_path
        assert kwargs["min_points_per_branch"] == 4
        return report

    def write(actual: Any, output: Any, **kwargs: Any) -> Any:
        assert actual is report
        assert output == str(tmp_path / "reports")
        assert kwargs["figures"] is False
        return SimpleNamespace(json=tmp_path / "reports/tm.json", figures=())

    monkeypatch.setattr(cli, "analyse_melting", analyse)
    monkeypatch.setattr(cli, "write_melting_report", write)
    assert (
        main(
            [
                "--analyse",
                str(tmp_path),
                "--no-figures",
                "--min-points-per-branch",
                "4",
                "-o",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert (
        "apparent Tm = 415 K" if resolved else "no clear melting transition"
    ) in printed


def test_the_analysis_mode_reports_a_transition_and_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No monomer, no OpenMM, no chemistry: a directory in, a number out."""
    temperature, density = two_line_curve(transition_k=340.0)
    write_quench(
        tmp_path,
        temperature[::-1],
        density[::-1],
        stage="06_quench",
        segment_duration_ps=[1000.0] * 21,
    )
    exit_code = main(["--analyse", str(tmp_path), "--no-melt-check", "--no-figures"])
    printed = capsys.readouterr().out

    assert exit_code == 0
    assert "quenches: 06_quench" in printed
    assert "Tg = 340 K" in printed
    assert "aV" in printed
    assert (tmp_path / "analysis" / "tg.json").is_file()


def test_an_unresolved_analysis_is_a_result_rather_than_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A curve with no break in it is something the science can say."""
    temperature = np.linspace(200.0, 600.0, 21)
    straight = 1.0 / (1.0 + 5.0e-4 * temperature)
    write_quench(
        tmp_path,
        temperature[::-1],
        straight[::-1],
        segment_duration_ps=[1000.0] * 21,
    )
    exit_code = main(["--analyse", str(tmp_path), "--no-melt-check", "--no-figures"])

    assert exit_code == 0
    assert "no clear transition" in capsys.readouterr().out


class _FakeChain:
    """Just the two fields the driver passes through to the run."""

    backbone = (0, 1)
    n_atoms = 2


def test_a_tg_run_hands_the_flat_flags_to_the_scan_as_a_spec(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The translation from seventeen flags to one spec, checked end to end."""
    import openmmpolymer.__main__ as cli

    seen: dict[str, Any] = {}

    def fake_scan(run: Any, run_dir: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(
            temperature_k=418.0,
            resolved=True,
            restart="waypoint",
            approximate=SimpleNamespace(temperature_k=425.0),
            fine_schedule=SimpleNamespace(cooling_rate_k_per_ns=1.67),
            fine_summary=SimpleNamespace(chains=None),
        )

    monkeypatch.setattr(cli, "run_tg_scan", fake_scan)
    arguments = build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "tg",
            "--t-end",
            "180",
            "--step-k",
            "20",
            "--fine-window-k",
            "50",
            "--check-melt",
            "4",
            "--characteristic-ratio",
            "5.5",
        ]
    )
    entry = PROTOCOLS["tg"]
    options = {
        name: getattr(arguments, _DESTS.get(name, name)) for name in entry.options
    }
    exit_code = cli._run_tg_scan(arguments, None, Path("run"), _FakeChain(), options)

    spec = seen["spec"]
    assert exit_code == 0
    assert spec.t_floor_k == pytest.approx(180.0)
    assert spec.coarse_step_k == pytest.approx(20.0)
    assert spec.window_k == pytest.approx(50.0)
    assert spec.npt_trajectory_ps == pytest.approx(4.0)
    assert seen["expected_characteristic_ratio"] == 5.5
    assert "Tg = 418 K" in capsys.readouterr().out


def test_a_tg_run_at_several_rates_reports_each_and_then_the_fit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every measurement, then the extrapolation with its span attached."""
    import openmmpolymer.__main__ as cli

    def fake_series(run: Any, run_dir: Any, **kwargs: Any) -> Any:
        return tuple(
            transition_at(rate, 340.0 + 20.0 * np.log10(rate))
            for rate in kwargs["rates_k_per_ns"]
        )

    monkeypatch.setattr(cli, "cooling_rate_series", fake_series)
    arguments = build_parser().parse_args(
        ["[*]CC[*]", "--protocol", "tg", "--cooling-rates", "100,10,1"]
    )
    entry = PROTOCOLS["tg"]
    options = {
        name: getattr(arguments, _DESTS.get(name, name)) for name in entry.options
    }
    exit_code = cli._run_tg_scan(arguments, None, Path("run"), _FakeChain(), options)
    printed = capsys.readouterr().out

    assert exit_code == 0
    assert printed.count("tg: ") == 3
    assert "log_linear" in printed
    assert "not resolved" in printed
    assert "per decade" in printed


# --------------------------------------------------------------------------
# The modulus protocol, and what --analyse dispatches on
# --------------------------------------------------------------------------


def test_the_flat_mechanics_flags_reach_the_spec_a_scan_takes() -> None:
    """The modulus factory takes **kwargs, so the general check cannot see it."""
    parameters = inspect.signature(_modulus_spec).parameters
    for option in PROTOCOLS["modulus"].options:
        assert option in parameters, option


def test_the_mechanics_flags_arrive_where_they_were_aimed() -> None:
    """Parsed, mapped through the destination table, and into the spec."""
    arguments = build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "modulus",
            "-t",
            "310",
            "--strain-increment",
            "0.001",
            "--max-strain",
            "0.03",
            "--replicas",
            "2",
            "--deform-axis",
            "0",
            "--load-stresses",
            "0,150,300",
        ]
    )
    spec = _modulus_spec(**_protocol_options(arguments, PROTOCOLS["modulus"]))
    assert spec.temperature_k == pytest.approx(310.0)
    assert spec.strain_increment == pytest.approx(0.001)
    assert spec.max_strain == pytest.approx(0.03)
    assert spec.n_replicas == 2
    assert spec.axis == 0
    assert spec.load_stresses_bar == (0.0, 150.0, 300.0)


def test_a_pass_named_in_skip_is_dropped() -> None:
    """Clearer than passing an empty list to the flag that configures it."""
    arguments = build_parser().parse_args(
        ["[*]CC[*]", "--protocol", "modulus", "--skip", "bulk", "shear"]
    )
    spec = _modulus_spec(**_protocol_options(arguments, PROTOCOLS["modulus"]))
    assert spec.bulk_pressures_bar is None
    assert spec.shear_strains is None
    assert spec.load_stresses_bar is not None


def test_a_malformed_number_list_is_refused_at_the_front_door() -> None:
    """With a message about the flag, not a traceback from inside a run."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["[*]CC[*]", "--load-stresses", "not,numbers"])


def test_analyse_reports_mechanics_when_the_directory_holds_a_deformation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dispatched on what the run recorded, not on a second flag."""
    write_deformation(Path("run"), modulus_mpa=2000.0, poisson=0.35)
    exit_code = main(["--analyse", "run", "--no-figures"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "E = 2000 MPa" in captured
    assert "nu = 0.350" in captured
    assert Path("run/analysis/mechanics.json").is_file()


def test_analyse_reports_both_when_the_directory_holds_both(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One verb, and it does not have to be told which kind of run this was."""
    Path("run").mkdir()
    temperature, density = two_line_curve(transition_k=340.0)
    write_quench(
        Path("run"),
        temperature[::-1],
        density[::-1],
        segment_duration_ps=[1000.0] * 21,
    )
    write_deformation(Path("run"), modulus_mpa=1800.0)
    exit_code = main(["--analyse", "run", "--no-melt-check", "--no-figures"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "E = 1800 MPa" in captured
    assert "quenches:" in captured
    assert Path("run/analysis/tg.json").is_file()
    assert Path("run/analysis/mechanics.json").is_file()


def test_analyse_says_so_when_a_directory_holds_neither(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Rather than raising from inside a reader that was asked the wrong thing."""
    Path("run").mkdir()
    Path("run/manifest.json").write_text(
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
    exit_code = main(["--analyse", "run"])
    assert exit_code == 1
    assert "nothing to report" in capsys.readouterr().out


def test_the_strain_rate_is_printed_with_the_modulus(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The caveat travels with the number, on the command line too."""
    write_deformation(Path("run"))
    main(["--analyse", "run", "--no-figures"])
    assert "strain/ns" in capsys.readouterr().out


def test_the_flat_relaxation_flags_reach_the_spec_a_scan_takes() -> None:
    """The relax factory takes **kwargs, so the check above cannot see it."""
    from openmmpolymer.__main__ import _RELAXATION, _relaxation_spec

    parameters = inspect.signature(_relaxation_spec).parameters
    for option in _RELAXATION:
        assert option in parameters, option


def test_the_relaxation_controls_reach_the_protocol() -> None:
    """A flag that is parsed, ignored and never arrives is the failure this
    whole table exists to prevent."""
    from openmmpolymer.__main__ import _protocol_options, _relaxation_spec

    arguments = build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "relax",
            "--step-strain",
            "0.05",
            "--relaxation-ps",
            "500",
            "--baseline-ps",
            "50",
            "--relax-replicas",
            "2",
            "--relax-mode",
            "shear",
            "--bins-per-decade",
            "12",
            "--linearity-strains",
            "0.01,0.09",
        ]
    )
    spec = _relaxation_spec(**_protocol_options(arguments, PROTOCOLS["relax"]))
    assert spec.step_strain == 0.05
    assert spec.relax_ps == 500.0
    assert spec.baseline_ps == 50.0
    assert spec.n_replicas == 2
    assert spec.mode == "shear"
    assert spec.bins_per_decade == 12
    assert spec.linearity_strains == (0.01, 0.09)


def test_analysing_a_relaxation_directory_reports_and_writes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--analyse dispatches on what the directory recorded, not on a flag.

    It also crosses the one seam where the scan and the reader differ: a scan
    carries an overall verdict and a directory read back does not, and the
    shared printer has to cope with both rather than assuming the richer one.
    """
    from .helpers import write_relaxation

    stages: dict[str, Any] | None = None
    for replica in range(3):
        write_relaxation(
            tmp_path,
            stem=f"06_relax_r{replica}",
            modulus_mpa=900.0,
            tau_ps=150.0,
            beta=0.45,
            chunks=2,
            merge=stages,
        )
        stages = json.loads((tmp_path / "manifest.json").read_text())["stages"]

    assert main(["--analyse", str(tmp_path)]) == 0
    captured = capsys.readouterr().out
    assert "G(0)" in captured
    # The planted parameters come back exactly, which is what makes this a
    # test of the wiring rather than of the fit.
    assert "beta = 0.450" in captured
    assert "tau = 150 ps" in captured
    assert (tmp_path / "analysis" / "relaxation.json").is_file()
    assert (tmp_path / "analysis" / "relaxation.png").is_file()


def test_a_directory_that_is_none_of_the_reported_kinds_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal names all report kinds, including heating scans."""
    (tmp_path / "manifest.json").write_text(
        '{"protocol": "x", "seed": 1, "stages": {"05_npt": {"samples": {}}}}'
    )
    assert main(["--analyse", str(tmp_path)]) == 1
    captured = capsys.readouterr().out
    assert "quench, a heating scan, a deformation or a relaxation" in captured
    assert "coordinates" in captured


# --------------------------------------------------------------------------
# The structure report, and what --analyse dispatches on
# --------------------------------------------------------------------------


def test_analyse_reports_structure_when_a_stage_left_coordinates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A plain run with a closing structure is enough for a report."""
    write_polymer_snapshot(Path("run"))
    exit_code = main(["--analyse", "run", "--no-figures", "--backbone", "0,1,2,3,4"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "g(r):" in captured
    assert "backbone: 5 atoms, argument" in captured
    assert "chains: <R^2>" in captured
    assert "rod-like" in captured
    assert Path("run/analysis/structure.json").is_file()


@pytest.mark.parametrize("override", [None, "7.0"])
def test_structure_analysis_preserves_saved_ratio_and_accepts_explicit_override(
    override: str | None,
) -> None:
    write_polymer_snapshot(Path("run"))
    manifest_path = Path("run/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["chains"] = {"expected_characteristic_ratio": 5.5}
    manifest_path.write_text(json.dumps(manifest))
    arguments = ["--analyse", "run", "--no-figures"]
    if override is not None:
        arguments.extend(["--characteristic-ratio", override])
    assert main(arguments) == 0
    record = json.loads(Path("run/analysis/structure.json").read_text())
    assert record["conformation"]["mean"]["expected_characteristic_ratio"] == (
        5.5 if override is None else float(override)
    )


def test_the_structure_flags_are_parsed() -> None:
    arguments = build_parser().parse_args(
        [
            "--analyse",
            "run",
            "--structure-stage",
            "03_npt",
            "--backbone",
            "0,1,4",
            "--stride",
            "4",
            "--no-structure",
        ]
    )
    assert arguments.structure_stage == "03_npt"
    assert arguments.backbone == (0, 1, 4)
    assert arguments.stride == 4
    assert arguments.no_structure


@pytest.mark.parametrize(
    "flags",
    [
        ["--backbone", "0,x"],
        ["--backbone", "3"],
        ["--stride", "0"],
        ["--stride", "two"],
    ],
)
def test_a_malformed_structure_flag_is_refused_at_the_front_door(
    flags: list[str],
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--analyse", "run", *flags])


def test_no_structure_leaves_a_structure_only_directory_with_nothing_to_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_polymer_snapshot(Path("run"))
    assert main(["--analyse", "run", "--no-structure"]) == 1
    assert "nothing to report" in capsys.readouterr().out
