"""Tests for the command-line driver.

The build is stood in for twice over: ``no_build`` stops a run where the melt
would start building, and ``staged_melt`` keeps the build's staging real but
fakes its chemistry. The scans are stood in for by name, as the driver looks
each one up when it runs it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmmpolymer import __main__ as cli
from openmmpolymer.__main__ import PROTOCOLS, build_parser, main
from openmmpolymer.mechanical import ModulusSpec, mechanical_scan
from openmmpolymer.protocols import (
    Protocol,
    _canonical,
    melt_quench,
    standard_melt_equilibration,
)
from openmmpolymer.reporters import TrajectoryOptions
from openmmpolymer.tensile import BreakingSpec, ElongationSpec, YieldSpec, tensile_scan
from openmmpolymer.tg import TgSpec, nominal_fine_schedule, tg_coarse_scan
from openmmpolymer.tm import TmSpec, melting_scan
from openmmpolymer.viscoelastic import RelaxationSpec, relaxation_scan

from .helpers import (
    BuildReached,
    snapshot_files,
    transition_at,
    two_line_curve,
    write_crystal,
    write_deformation,
    write_polymer_snapshot,
    write_quench,
    write_quenches,
    write_relaxation,
)

#: SHA-256 of what every command-line run's ``build_request.json`` is made
#: of: each destination with its flags, default and const, and the namespace
#: each protocol's shortest command line parses to once that protocol's own
#: defaults are filled in. The request is ``vars(arguments)``, compared whole
#: when a run resumes, so renaming, removing or re-defaulting a flag - or
#: changing what a protocol fills in - makes every run already on disk refuse
#: to resume. Change one of these only knowing that.
RECORDED_REQUEST_SHA256 = {
    "defaults": "7dee2dafb06a7a3f98caac0787aae9d6fc956ab72f71c646bf11eab9405cdce2",
    "analyse": "de0816302b550e30bbd6bab0569f077527562e7294cbecfac7bfb72a469930e9",
    "breaking": "1c0733748e93a35670c4f69ff0579045da4a0f640cb4386adfdf9bb4826d1a6f",
    "elongation": "2be7461048f580a378bef14b4a703814d77baa0768e0ff8b425f4aa1ca8a86de",
    "equilibrate": "9498f1b1fe4db82ed20c7d380737e222dee09e51c41843c27116360e444a5b29",
    "melt-quench": "f9b959049e64e256f001fd6a20d47a7e0ff4db0e85de330b327c035c9a115397",
    "modulus": "0088a39eb842c91b36d816105068d7e3cdbb0e8f1b442b79d1123710400d4fda",
    "relax": "3cb7cf85ba7c1d63a0b1d1d8984aa6270d43cc507ea47e8f3d29fe115ef1d80b",
    "tg": "5935f5f501003989b6265f9e06eed094a93e41a14d1a46b4db3a8827d06251f3",
    "tm": "f66ed12a1460e2f2e5546b1bdf5075004755b20b86baa02b9cdc3b29910fc23e",
    "yield": "012ff7978ea93edc33b04a21caffd630ca654b2d45a5786e0e9ad0a6e071f44a",
}

#: What the build request leaves out: where things go, which device runs
#: them, how much is said, and whether - or under what budget - dynamics
#: start. None of them changes the physical system.
UNRECORDED = {
    "output_dir",
    "platform",
    "dry_run",
    "verbose",
    "no_figures",
    "figure_format",
    "max_total_ns",
}

#: What each protocol's shortest command line asks for. Seven of the command
#: line's defaults differ from the library's, all on purpose.
DEFAULT_SETTINGS: dict[str, Any] = {
    "equilibrate": standard_melt_equilibration(),
    "melt-quench": melt_quench(),
    "tg": TgSpec(
        melt_temperature_k=600.0,
        t_floor_k=200.0,
        coarse_step_k=20.0,
        coarse_hold_ps=200.0,
    ),
    "tm": TmSpec(min_points_per_branch=4),
    "modulus": ModulusSpec(temperature_k=450.0),
    "breaking": BreakingSpec(),
    "elongation": ElongationSpec(),
    "yield": YieldSpec(),
    "relax": RelaxationSpec(temperature_k=450.0),
}


def _tensile_flags(name: str, criterion: str) -> str:
    """Every flag of one tensile measurement: the shared ladder, its own name."""
    return (
        f"-t 310 --pressure 2 --deform-axis 0 --max-total-ns 900 "
        f"--{name}-strain-increment 0.001 --{name}-max-strain 0.2 "
        f"--{name}-relax-ps 20 --{name}-replicas 2 --{name}-samples-per-step 40 "
        f"--{name}-stage-ps 200 --{name}-trajectory-ps 5 {criterion}"
    )


#: The ladder those flags set, in the spec's own terms.
TENSILE_LADDER: dict[str, Any] = {
    "temperature_k": 310.0,
    "pressure_bar": 2.0,
    "axis": 0,
    "strain_increment": 0.001,
    "max_strain": 0.2,
    "relax_ps": 20.0,
    "n_replicas": 2,
    "samples_per_step": 40,
    "stage_ps": 200.0,
    "trajectory_ps": 5.0,
    "max_total_ns": 900.0,
}

#: For each protocol, a command line setting every flag it takes, and the
#: settings those have to make.
EVERY_FLAG: dict[str, tuple[str, Any]] = {
    "equilibrate": (
        "-t 320 --melt-temperature 620 --pressure 2 --check-melt 5",
        standard_melt_equilibration(
            target_temperature_k=320.0,
            melt_temperature_k=620.0,
            pressure_bar=2.0,
            npt_trajectory=TrajectoryOptions("xtc", interval_ps=5.0),
        ),
    ),
    "melt-quench": (
        "-t 320 --melt-temperature 620 --pressure 2 --check-melt "
        "--t-start 640 --t-end 160 --step-k 25 --hold-ps 1000",
        melt_quench(
            target_temperature_k=320.0,
            melt_temperature_k=620.0,
            pressure_bar=2.0,
            npt_trajectory=TrajectoryOptions("xtc", interval_ps=10.0),
            t_start=640.0,
            t_end=160.0,
            step_k=25.0,
            hold_ps=1000.0,
        ),
    ),
    "tg": (
        "--melt-temperature 610 --pressure 2 --t-end 180 --step-k 15 --hold-ps 300 "
        "--fine-step-k 4 --fine-hold-ps 2500 --fine-window-k 50 --check-melt 4 "
        "--min-points-per-branch 5 --max-total-ns 900",
        TgSpec(
            melt_temperature_k=610.0,
            pressure_bar=2.0,
            t_floor_k=180.0,
            coarse_step_k=15.0,
            coarse_hold_ps=300.0,
            fine_step_k=4.0,
            fine_hold_ps=2500.0,
            window_k=50.0,
            npt_trajectory_ps=4.0,
            min_points_per_branch=5,
            max_total_ns=900.0,
        ),
    ),
    "tm": (
        "--t-start 280 --t-end 500 --step-k 20 --hold-ps 50 --pressure 2 "
        "--tm-equilibration-ps 10 --tm-stage-ps 100 --tm-trajectory-ps 5 "
        "--tm-barostat isotropic --min-points-per-branch 5 --max-total-ns 900",
        TmSpec(
            t_start_k=280.0,
            t_end_k=500.0,
            step_k=20.0,
            hold_ps=50.0,
            pressure_bar=2.0,
            equilibration_ps=10.0,
            stage_ps=100.0,
            trajectory_ps=5.0,
            barostat="isotropic",
            min_points_per_branch=5,
            max_total_ns=900.0,
        ),
    ),
    "modulus": (
        "-t 310 --pressure 2 --deform-axis 0 --max-total-ns 900 "
        "--strain-increment 0.001 --max-strain 0.03 --relax-ps 20 "
        "--elastic-strain-limit 0.01 --replicas 2 --load-stresses 0,150,300 "
        "--bulk-pressures 1,50,1 --shear-strains 0.01",
        ModulusSpec(
            temperature_k=310.0,
            pressure_bar=2.0,
            axis=0,
            strain_increment=0.001,
            max_strain=0.03,
            relax_ps=20.0,
            elastic_strain_limit=0.01,
            n_replicas=2,
            load_stresses_bar=(0.0, 150.0, 300.0),
            bulk_pressures_bar=(1.0, 50.0, 1.0),
            shear_strains=(0.01,),
            max_total_ns=900.0,
        ),
    ),
    "breaking": (
        _tensile_flags("breaking", "--failure-fraction 0.4 --confirmation-steps 4"),
        BreakingSpec(**TENSILE_LADDER, failure_fraction=0.4, confirmation_steps=4),
    ),
    "elongation": (
        _tensile_flags("elongation", "--failure-fraction 0.4 --confirmation-steps 4"),
        ElongationSpec(**TENSILE_LADDER, failure_fraction=0.4, confirmation_steps=4),
    ),
    "yield": (
        _tensile_flags(
            "yield",
            "--yield-offset-strain 0.005 --yield-fit-min-strain 0.001 "
            "--yield-fit-max-strain 0.015",
        ),
        YieldSpec(
            **TENSILE_LADDER,
            offset_strain=0.005,
            fit_min_strain=0.001,
            fit_max_strain=0.015,
        ),
    ),
    "relax": (
        "-t 310 --pressure 2 --deform-axis 0 --max-total-ns 900 --relax-mode shear "
        "--step-strain 0.05 --baseline-ps 50 --relaxation-ps 500 --relax-replicas 2 "
        "--sample-every-ps 0.1 --bins-per-decade 12 --relax-stage-ps 250 "
        "--linearity-strains 0.01,0.09",
        RelaxationSpec(
            temperature_k=310.0,
            pressure_bar=2.0,
            axis=0,
            mode="shear",
            step_strain=0.05,
            baseline_ps=50.0,
            relax_ps=500.0,
            n_replicas=2,
            sample_every_ps=0.1,
            bins_per_decade=12,
            stage_ps=250.0,
            linearity_strains=(0.01, 0.09),
            max_total_ns=900.0,
        ),
    ),
}


def _sha256(value: Any) -> str:
    text = json.dumps(_canonical(value), sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _argv(protocol: str, *flags: str) -> list[str]:
    """A command line for *protocol*: tm starts from a crystal, not a monomer."""
    monomer = [] if protocol == "tm" else ["[*]CC[*]"]
    return [*monomer, "--protocol", protocol, *flags]


# --------------------------------------------------------------------------
# What a rerun has to repeat
# --------------------------------------------------------------------------


def test_every_setting_a_build_request_records_is_pinned() -> None:
    """Changing one refuses the resume of every command-line run on disk."""
    parser = build_parser()
    recorded = {
        "defaults": _sha256(
            {
                action.dest: [action.option_strings, action.default, action.const]
                for action in parser._actions
            }
        ),
        "analyse": _sha256(vars(parser.parse_args(["--analyse", "run"]))),
    }
    for protocol in (
        "breaking",
        "elongation",
        "equilibrate",
        "melt-quench",
        "modulus",
        "relax",
        "tg",
        "tm",
        "yield",
    ):
        recorded[protocol] = _sha256(vars(parser.parse_args(_argv(protocol))))
    assert recorded == RECORDED_REQUEST_SHA256


def test_the_build_request_is_every_setting_but_where_and_how_it_runs(
    no_build: list[Any],
) -> None:
    argv = ["[*]CC[*]", "--conformers", "50", "--platform", "CPU", "--no-figures"]
    argv += ["--max-total-ns", "1e6", "-o", "elsewhere", "-v"]
    with pytest.raises(BuildReached):
        main(argv)
    recorded = json.loads(Path("elsewhere/build_request.json").read_text())
    versions = recorded.pop("runtime_versions")
    assert set(versions) == {"openmm", "rdkit", "forcefill", "openff-toolkit", "numpy"}
    namespace = vars(build_parser().parse_args(argv))
    assert recorded == _canonical(
        {
            **{key: value for key, value in namespace.items() if key not in UNRECORDED},
            "conformers": 30,
            "characteristic_ratio": 7.0,
        }
    )
    assert no_build[0].characteristic_ratio == 7.0


@pytest.mark.parametrize(
    "changed",
    [["--seed", "7"], ["--temperature", "500"], ["--tail-cap", "[*]O"]],
)
def test_a_changed_request_is_refused_before_it_can_overwrite_the_build(
    changed: list[str], no_build: list[Any]
) -> None:
    arguments = ["[*]CC[*]", "-o", "run"]
    with pytest.raises(BuildReached):
        main(arguments)
    artifact = Path("run/build/polymer_ff.xml")
    artifact.parent.mkdir()
    artifact.write_text("existing parameters")
    with pytest.raises(SystemExit, match="2"):
        main([*arguments, *changed])
    assert artifact.read_text() == "existing parameters"
    # Presentation, device selection and switching from build-only to dynamics
    # do not change the prepared physical system.
    with pytest.raises(BuildReached):
        main([*arguments, "--platform", "CPU", "--dry-run", "-v"])


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    "flags",
    [
        *(
            ["--characteristic-ratio", value]
            for value in ("0", "-1", "nan", "inf", "bad")
        ),
        ["--protocol", "anneal-forever"],
        ["--charge-method", "am1bbc"],
        *(["--cooling-rates", rates] for rates in ("10,fast,2", "10", "10,10", "0,10")),
        ["--load-stresses", "not,numbers"],
        ["--check-melt", "0"],
        ["--backbone", "0,x"],
        ["--backbone", "3"],
        ["--stride", "0"],
        ["--stride", "two"],
    ],
)
def test_a_malformed_flag_is_refused_at_the_front_door(flags: list[str]) -> None:
    """With a message about the flag, before any chemistry starts."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["[*]CC[*]", *flags])


def test_list_flags_parse_in_the_order_they_were_written() -> None:
    arguments = build_parser().parse_args(
        [
            "--analyse",
            "run",
            "other",
            "--cooling-rates",
            "10,5,2",
            "--backbone",
            "0,1,4",
            "--stride",
            "4",
            "--structure-stage",
            "03_npt",
            "--no-structure",
        ]
    )
    assert arguments.monomer is None
    assert arguments.analyse == ["run", "other"]
    assert arguments.cooling_rates == (10.0, 5.0, 2.0)
    assert arguments.backbone == (0, 1, 4)
    assert arguments.stride == 4
    assert arguments.structure_stage == "03_npt"
    assert arguments.no_structure


def test_building_a_melt_still_needs_a_monomer() -> None:
    """The one required argument, unless the other mode was asked for."""
    with pytest.raises(SystemExit):
        main([])


# --------------------------------------------------------------------------
# The protocol table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", sorted(EVERY_FLAG))
def test_every_flag_a_protocol_takes_reaches_its_settings(protocol: str) -> None:
    """A flag the table sends nowhere is parsed, ignored and never arrives.

    That is how t_end, step_k and hold_ps were once unreachable while looking
    perfectly present in --help, and how --min-points-per-branch never reached
    a tg scan.
    """
    flags, expected = EVERY_FLAG[protocol]
    arguments = build_parser().parse_args(_argv(protocol, *flags.split()))
    assert PROTOCOLS[protocol].settings(arguments) == expected


@pytest.mark.parametrize("protocol", sorted(DEFAULT_SETTINGS))
def test_the_command_lines_own_defaults_are_the_ones_it_means(protocol: str) -> None:
    """Including the seven that differ from the library's."""
    arguments = build_parser().parse_args(_argv(protocol))
    assert PROTOCOLS[protocol].settings(arguments) == DEFAULT_SETTINGS[protocol]


def test_a_pass_named_in_skip_is_dropped() -> None:
    """Clearer than passing an empty list to the flag that configures it."""
    arguments = build_parser().parse_args(_argv("modulus", "--skip", "bulk", "shear"))
    spec = PROTOCOLS["modulus"].settings(arguments)
    assert spec.bulk_pressures_bar is None
    assert spec.shear_strains is None
    assert spec.load_stresses_bar == ModulusSpec().load_stresses_bar


# --------------------------------------------------------------------------
# Checked and priced before anything is built
# --------------------------------------------------------------------------


def _listed_ps(protocol: str, rates: tuple[float, ...] = ()) -> float:
    """Every stage of *protocol*'s default run, as the library lists it."""
    settings = DEFAULT_SETTINGS[protocol]
    if protocol in ("equilibrate", "melt-quench"):
        return float(settings.total_duration_ps)
    if protocol == "tg":
        holds = [settings.fine_step_k / rate * 1000.0 for rate in rates]
        fine = sum(
            nominal_fine_schedule(settings, hold_ps=hold).total_ps
            for hold in holds or [None]
        )
        return float(tg_coarse_scan(settings).total_duration_ps + fine)
    listings: dict[str, Callable[[Any], Protocol]] = {
        "tm": melting_scan,
        "modulus": mechanical_scan,
        "relax": relaxation_scan,
        **dict.fromkeys(("breaking", "elongation", "yield"), tensile_scan),
    }
    return listings[protocol](settings).total_duration_ps


@pytest.mark.parametrize(
    ("protocol", "rates"),
    [(protocol, ()) for protocol in sorted(DEFAULT_SETTINGS)] + [("tg", (10.0, 5.0))],
)
def test_every_run_is_priced_in_full_before_anything_is_built(
    protocol: str,
    rates: tuple[float, ...],
    no_build: list[Any],
    argon_box: tuple[Any, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every replica, pass and fine window, as a dry run reports it."""
    flags = ["--dry-run"]
    if rates:
        flags += ["--cooling-rates", ",".join(f"{rate:g}" for rate in rates)]
    if protocol == "tm":
        pdb, xml = write_crystal(*argon_box, Path("."))
        flags += ["--crystal-pdb", str(pdb), "--system-xml", str(xml)]
        assert main(_argv("tm", *flags)) == 0
    else:
        with pytest.raises(BuildReached):
            main(_argv(protocol, *flags))
    total_ns = _listed_ps(protocol, rates) / 1000.0
    assert f"{protocol} run: {total_ns:.3g} ns of dynamics in total" in (
        capsys.readouterr().out
    )


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------


def test_the_chain_flags_reach_the_builder(no_build: list[Any]) -> None:
    """A capped polyester is built as asked, with its own C-infinity."""
    from openmmpolymer.chain import assemble_chain

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
                "--tacticity",
                "isotactic",
                "-r",
                "PLA",
                "--seed",
                "7",
                "--dry-run",
            ]
        )
    (spec,) = no_build
    assert (spec.head_cap, spec.tail_cap) == ("[*][H]", "[*]O")
    assert (spec.degree_of_polymerization, spec.characteristic_ratio) == (3, 5.5)
    assert (spec.tacticity, spec.residue_name, spec.seed) == ("isotactic", "PLA", 7)
    assert assemble_chain(spec).GetNumAtoms() > 0


@pytest.mark.parametrize("failure", ["refused", "broken"])
def test_a_rebuild_that_is_refused_or_breaks_leaves_the_build_as_it_was(
    staged_melt: dict[str, Any], failure: str
) -> None:
    """A refusal is a usage error; a build that breaks says so itself."""
    assert main(["[*]CC[*]", "--dry-run"]) == 0
    before = snapshot_files(Path("run"))
    if failure == "refused":
        staged_melt["system_suffix"] = "\n"
        expected: type[BaseException] = SystemExit
    else:
        staged_melt["fail"] = True
        expected = RuntimeError
    with pytest.raises(expected):
        main(["[*]CC[*]", "--dry-run"])
    assert snapshot_files(Path("run")) == before


def test_a_dry_run_can_go_on_to_dynamics_on_the_verified_build(
    staged_melt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ran: list[Any] = []

    def dynamics(protocol: Any, run: Any, output: Path, **options: Any) -> Any:
        ran.append((protocol, output, options))
        assert Path(run.forcefield.forcefield_xml) == Path("run/build/polymer_ff.xml")
        chains = SimpleNamespace(
            mean_radius_of_gyration_nm=1.25, characteristic_ratio=6.5, consistent=True
        )
        return SimpleNamespace(
            protocol=protocol.name,
            results=(1, 2),
            wall_seconds=90.0,
            manifest_path="run/manifest.json",
            chains=chains,
        )

    assert main(["[*]CC[*]", "--dry-run"]) == 0
    before = snapshot_files(Path("run"))
    monkeypatch.setattr(cli, "run_protocol", dynamics)
    assert main(["[*]CC[*]"]) == 0
    assert snapshot_files(Path("run")) == before
    (protocol, output, options) = ran[0]
    assert protocol == DEFAULT_SETTINGS["equilibrate"]
    assert output == Path("run")
    assert options == {
        "chain_backbone": (),
        "atoms_per_chain": 1,
        "expected_characteristic_ratio": 7.0,
    }
    printed = capsys.readouterr().out
    assert "standard_melt_equilibration: 2 stages in 1.5 min" in printed
    assert "chains: Rg 1.250 nm, C 6.50 (consistent)" in printed


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
    assert "chain: " in captured and "packed: 40 chains" in captured
    assert "dry run" in captured
    assert Path("out/build/packed.pdb").is_file()
    assert Path("out/build/polymer_ff.xml").is_file()


# --------------------------------------------------------------------------
# The runs
# --------------------------------------------------------------------------


def test_a_tg_run_hands_the_flat_flags_to_the_scan_as_a_spec(
    staged_melt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The translation from seventeen flags to one spec, checked end to end."""
    seen: dict[str, Any] = {}

    def scan(run: Any, run_dir: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(
            temperature_k=418.0,
            resolved=True,
            restart="waypoint",
            approximate=SimpleNamespace(temperature_k=425.0),
            fine_schedule=SimpleNamespace(cooling_rate_k_per_ns=1.67),
            fine_summary=SimpleNamespace(chains=None),
        )

    monkeypatch.setattr(cli, "run_tg_scan", scan)
    flags = ["--t-end", "180", "--fine-window-k", "50", "--check-melt", "4"]
    flags += ["--characteristic-ratio", "5.5", "--tg-approx", "420"]
    assert main(_argv("tg", *flags)) == 0

    spec = seen["spec"]
    assert (spec.t_floor_k, spec.window_k, spec.npt_trajectory_ps) == (180, 50, 4)
    assert seen["tg_approx_k"] == 420.0
    assert seen["expected_characteristic_ratio"] == 5.5
    # A tg scan settles its melt from its spec, not from separate keywords.
    assert "melt_temperature_k" not in seen
    assert "tg: Tg = 418 K at 1.67 K/ns (coarse said 425 K)" in capsys.readouterr().out


def test_a_tg_run_at_several_rates_reports_each_and_then_the_fit(
    staged_melt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every measurement, then the extrapolation with its span attached."""

    def series(run: Any, run_dir: Any, **kwargs: Any) -> Any:
        return tuple(
            transition_at(rate, 340.0 + 20.0 * np.log10(rate))
            for rate in kwargs["rates_k_per_ns"]
        )

    monkeypatch.setattr(cli, "cooling_rate_series", series)
    assert main(_argv("tg", "--cooling-rates", "100,10,1", "--rate-form", "vft")) == 0
    printed = capsys.readouterr().out
    assert printed.count("tg: ") == 3
    assert "vft: " in printed
    assert "not resolved" in printed
    assert "per decade" in printed


@pytest.mark.parametrize(
    ("protocol", "scan", "report", "line"),
    [
        (
            "modulus",
            "run_modulus_scan",
            {"youngs": None, "poisson": None, "bulk": None, "shear": None}
            | {"load_modulus": None, "consistency": None},
            "modulus: nothing was deformed",
        ),
        ("relax", "run_relaxation_scan", {"mean": None}, "relax: nothing was strained"),
    ],
)
def test_a_scan_settles_its_melt_as_the_flags_ask_and_says_what_it_found(
    protocol: str,
    scan: str,
    report: dict[str, Any],
    line: str,
    staged_melt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--melt-temperature and --check-melt reach every melt a scan settles."""
    seen: dict[str, Any] = {}

    def run_scan(run: Any, output: Path, **kwargs: Any) -> Any:
        seen.update(kwargs, output=output)
        return SimpleNamespace(**report)

    monkeypatch.setattr(cli, scan, run_scan)
    flags = ["--melt-temperature", "620", "--check-melt", "5", "-o", "measured"]
    assert main(_argv(protocol, *flags)) == 0
    assert seen["output"] == Path("measured")
    assert seen["spec"] == DEFAULT_SETTINGS[protocol]
    assert seen["melt_temperature_k"] == 620.0
    assert seen["npt_trajectory"] == TrajectoryOptions("xtc", interval_ps=5.0)
    assert seen["expected_characteristic_ratio"] == 7.0
    assert line in capsys.readouterr().out


# --------------------------------------------------------------------------
# Melting a supplied crystal
# --------------------------------------------------------------------------


def test_melting_dry_run_validates_prepared_inputs_without_building(
    argon_box: tuple[Any, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    pdb, xml = write_crystal(*argon_box, Path("."))
    crystal = ["--crystal-pdb", str(pdb), "--system-xml", str(xml)]
    assert main(["--protocol", "tm", *crystal, "--dry-run", "-o", "output"]) == 0
    assert "crystal and heating schedule validated" in capsys.readouterr().out
    assert not Path("output").exists()


@pytest.mark.parametrize("invalid", ["barostat", "state"])
def test_a_crystal_the_scan_could_not_heat_is_refused_before_any_file(
    invalid: str, argon_box: tuple[Any, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    import openmm as mm

    box, system = argon_box
    if invalid == "barostat":
        system.addForce(mm.MonteCarloBarostat(1.0, 300.0))
    pdb, xml = write_crystal(box, system, Path("."))
    crystal = ["--crystal-pdb", str(pdb), "--system-xml", str(xml)]
    if invalid == "state":
        crystal += ["--state-in", "missing-state.xml"]
    with pytest.raises(SystemExit, match="2"):
        main(["--protocol", "tm", *crystal, "--dry-run", "-o", "output"])
    message = "no barostat" if invalid == "barostat" else "starting State"
    assert message in capsys.readouterr().err
    assert not Path("output").exists()


def test_melting_cli_dispatches_the_prepared_cell_and_explicit_schedule(
    argon_box: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}
    transition = SimpleNamespace(
        temperature_k=415.0, bracket_k=(410.0, 420.0), resolved=True
    )
    report = SimpleNamespace(
        transition=transition,
        notes=("Finite heating rate; inspect crystalline order.",),
    )

    def scan(run: Any, output: Path, **kwargs: Any) -> Any:
        seen.update(kwargs, run=run, output=output)
        return SimpleNamespace(report=report)

    def write(actual: Any, output_dir: Any, **kwargs: Any) -> Any:
        assert actual is report
        assert output_dir is None
        assert kwargs["figures"] is False
        return SimpleNamespace(json="output/analysis/tm.json", figures=())

    monkeypatch.setattr(cli, "run_tm_scan", scan)
    monkeypatch.setattr(cli, "write_melting_report", write)
    pdb, xml = write_crystal(*argon_box, Path("."))
    flags = ["--crystal-pdb", str(pdb), "--system-xml", str(xml), "--no-figures"]
    flags += ["--t-start", "280", "--t-end", "500", "--step-k", "20", "--hold-ps"]
    flags += ["50", "--tm-equilibration-ps", "10", "--tm-stage-ps", "100"]
    assert main(_argv("tm", *flags, "-o", "output")) == 0
    assert seen["crystalline"] is True
    assert seen["state_in"] is None
    assert seen["output"] == Path("output")
    assert (seen["spec"].t_start_k, seen["spec"].t_end_k) == (280, 500)
    assert seen["spec"].hold_ps == 50
    assert seen["run"].box.n_molecules == 64
    assert seen["run"].spec.constraints == "none"
    printed = capsys.readouterr().out
    assert "apparent Tm = 415 K (heating bracket 410-420 K)" in printed
    assert "note: Finite heating rate" in printed
    assert "wrote output/analysis/tm.json and 0 figure(s)" in printed


# --------------------------------------------------------------------------
# What --analyse reports
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resolved", [True, False])
def test_analyse_dispatches_melting_and_reports_unresolved_results(
    resolved: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
        assert kwargs["min_points_per_branch"] == 5
        return report

    def write(actual: Any, output: Any, **kwargs: Any) -> Any:
        assert actual is report
        assert output == str(tmp_path / "reports")
        assert kwargs["figures"] is False
        return SimpleNamespace(json=tmp_path / "reports/tm.json", figures=())

    monkeypatch.setattr(cli, "analyse_melting", analyse)
    monkeypatch.setattr(cli, "write_melting_report", write)
    argv = ["--analyse", str(tmp_path), "--no-figures", "--min-points-per-branch"]
    assert main([*argv, "5", "-o", str(tmp_path / "reports")]) == 0
    printed = capsys.readouterr().out
    assert (
        "apparent Tm = 415 K" if resolved else "no clear melting transition"
    ) in printed


@pytest.mark.parametrize("resolved", [True, False])
def test_saved_quench_analysis_reports_the_transition_verdict(
    resolved: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    temperature, density = two_line_curve(transition_k=340.0)
    if not resolved:
        density = 1.0 / (1.0 + 5.0e-4 * temperature)
    write_quench(
        tmp_path,
        temperature[::-1],
        density[::-1],
        segment_duration_ps=[1000.0] * 21,
    )
    assert main(["--analyse", str(tmp_path), "--no-melt-check", "--no-figures"]) == 0
    printed = capsys.readouterr().out
    assert "quenches: 06_quench (20 K steps, 20.00 K/ns)" in printed
    if resolved:
        assert "Tg = 340 K" in printed
        assert "aV" in printed
    else:
        assert "no clear transition" in printed
    assert (tmp_path / "analysis/tg.json").is_file()


@pytest.mark.parametrize("form", ["log_linear", "vft"])
def test_the_rate_form_leaves_every_extrapolation_in_an_analysis(
    form: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--rate-form picks the one a --cooling-rates run headlines, nothing more."""
    coarse_t, coarse_d = two_line_curve(transition_k=340.0, n_points=11)
    stages: dict[str, Any] = {
        "06_coarse": {
            "temperature_k": list(coarse_t[::-1]),
            "density_g_cm3": list(coarse_d[::-1]),
            "segment_duration_ps": [1000.0] * 11,
        }
    }
    for transition_k, hold_ps in ((360.0, 1000.0), (340.0, 4000.0), (320.0, 16000.0)):
        temperature, density = two_line_curve(transition_k=transition_k)
        stages[f"08_fine_{hold_ps:.0f}"] = {
            "temperature_k": list(temperature[::-1]),
            "density_g_cm3": list(density[::-1]),
            "segment_duration_ps": [hold_ps] * 21,
        }
    write_quenches(tmp_path, stages)
    argv = ["--analyse", str(tmp_path), "--no-melt-check", "--no-figures"]
    assert main([*argv, "--rate-form", form]) == 0
    printed = capsys.readouterr().out
    assert "\nlog_linear: " in printed
    assert "\nvft: " in printed


def _fake(**fields: Any) -> Any:
    return SimpleNamespace(**fields)


#: For each printer, a report with every caveat it can carry, and its lines.
PRINTED: dict[str, tuple[Any, list[str]]] = {
    "_modulus_lines": (
        _fake(
            youngs=_fake(
                modulus_mpa=2012.4, strain_rate_per_ns=0.025, temperature_k=298.15
            ),
            replica_spread_mpa=41.6,
            replicas=(1, 2, 3),
            resolved=False,
            poisson=_fake(ratio=0.3512, resolved=True),
            bulk=_fake(modulus_mpa=3301.0, standard_error_mpa=120.0, resolved=True),
            shear=_fake(modulus_mpa=741.0, standard_error_mpa=15.3, resolved=False),
            load_modulus=_fake(modulus_mpa=1950.2),
            consistency=_fake(
                bulk_implied_mpa=2220.0,
                shear_implied_mpa=744.8,
                bulk_gap=0.33,
                shear_gap=float("nan"),
                consistent=False,
            ),
        ),
        [
            "E = 2012 MPa +/- 42 over 3 replicas at 0.025 strain/ns, 298 K "
            "(not resolved)",
            "nu = 0.351",
            "K = 3301 +/- 1.2e+02 MPa (fit SE)",
            "G = 741 +/- 15 MPa (fit SE) (not resolved)",
            "constant-stress cross-check: E = 1950 MPa",
            "E and nu imply K = 2220, G = 745 MPa; measured differ by K 33% - not "
            "consistent",
        ],
    ),
    "_tg_lines": (
        _fake(
            curves=(
                _fake(
                    stage="06_quench",
                    temperature_step_k=20.0,
                    cooling_rate_k_per_ns=None,
                ),
            ),
            melt=_fake(
                stage="05_npt",
                volume_settled=False,
                chains_moved=True,
                unchecked=("no trajectory to follow the chains",),
            ),
            coarse=None,
            fine=_fake(cooling_rate_k_per_ns=5.0, resolved=False),
            log_linear=None,
            vft=None,
        ),
        [
            "quenches: 06_quench (20 K steps, rate unknown)",
            "melt 05_npt: volume still drifting; chains moved",
            "  unchecked: no trajectory to follow the chains",
            "fine: no clear transition at 5.00 K/ns",
        ],
    ),
    "_relaxation_lines": (
        _fake(
            mean=_fake(
                initial_modulus_mpa=912.3,
                step_strain=0.03,
                temperature_k=450.0,
                decades=3.2,
            ),
            replica_spread_mpa=12.34,
            curves=(1, 2, 3),
            resolved=True,
            kww=None,
            prony=_fake(
                equilibrium_mpa=1.234, n_active=3, n_terms=8, plateau_reached=False
            ),
            linearity=_fake(strains=(0.01, 0.03), gap=0.052, linear=True),
        ),
        [
            "G(0) = 912.3 MPa +/- 12.3 over 3 replicas at +0.030 strain, 450 K, "
            "over 3.2 decades",
            "Prony: G_inf = 1.234 MPa over 3 of 8 terms - still decaying",
            "linearity: strains [0.01, 0.03] differ by 5%",
        ],
    ),
    "_structure_lines": (
        _fake(
            is_snapshot=False,
            n_frames=10,
            interval_ps=2.0,
            stage="05_npt",
            n_chains=32,
            atoms_per_chain=2,
            backbone=None,
            distribution=None,
            structure=_fake(first_peak_per_nm=0.0, q_min_per_nm=2.6),
            conformation=_fake(
                mean=_fake(
                    mean_squared_end_to_end_nm2=1.5,
                    mean_radius_of_gyration_nm=0.5,
                    characteristic_ratio=6.1,
                    expected_characteristic_ratio=7.0,
                    consistent=True,
                ),
                settled=_fake(equilibrated=False),
            ),
            persistence=_fake(
                persistence_length_nm=0.45,
                n_bonds=20,
                contour_length_nm=2.5,
                decayed=False,
            ),
            displacement=_fake(log_slope=0.62, diffusion_coefficient_cm2_s=None),
            relaxation=_fake(relaxation_time_ps=None, trajectory_ps=20.0),
            recorded_chains=_fake(
                mean_squared_end_to_end_nm2=1.4, mean_radius_of_gyration_nm=0.49
            ),
        ),
        [
            "structure: stage 05_npt (10 frames at 2 ps), 32 chains of 2 atoms",
            "backbone: unknown, so no chain measurements",
            "S(q): no resolvable peak above 2.6 /nm",
            "chains: <R^2> = 1.500 nm2, Rg = 0.500 nm, C = 6.10 against an expected "
            "7.00, <R^2> still moving",
            "persistence length: 0.450 nm over 20 bonds (2.50 nm contour) - never "
            "decayed to 1/e within the chain, so this is an extrapolation",
            "MSD: slope 0.62, not diffusive, so no diffusion coefficient",
            "end-to-end: not decorrelated in 20 ps; the relaxation time is longer "
            "than the run",
            "manifest recorded at the end of the run: <R^2> = 1.400 nm2, Rg = 0.490 nm",
        ],
    ),
}


@pytest.mark.parametrize("printer", sorted(PRINTED))
def test_each_number_is_printed_with_what_qualifies_it(printer: str) -> None:
    """A run and its --analyse print a report the same way, caveats and all."""
    report, lines = PRINTED[printer]
    assert list(getattr(cli, printer)(report)) == lines


@pytest.mark.parametrize("with_quench,modulus", [(False, 2000.0), (True, 1800.0)])
def test_saved_analysis_detects_mechanics_and_any_accompanying_quench(
    with_quench: bool, modulus: float, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = Path("run")
    directory.mkdir()
    flags = ["--no-figures"]
    if with_quench:
        temperature, density = two_line_curve(transition_k=340.0)
        write_quench(
            directory,
            temperature[::-1],
            density[::-1],
            segment_duration_ps=[1000.0] * 21,
        )
        flags.append("--no-melt-check")
    write_deformation(directory, modulus_mpa=modulus, poisson=0.35)
    assert main(["--analyse", str(directory), *flags]) == 0
    printed = capsys.readouterr().out
    assert f"E = {modulus:g} MPa" in printed
    assert "strain/ns" in printed
    assert "nu = 0.350" in printed
    assert (directory / "analysis/mechanics.json").is_file()
    if with_quench:
        assert "quenches:" in printed
        assert (directory / "analysis/tg.json").is_file()


@pytest.mark.parametrize("kind", ["empty", "unmeasured", "structure_skipped"])
def test_a_directory_with_nothing_to_report_says_so(
    kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rather than raising from inside a reader asked the wrong thing."""
    flags: list[str] = []
    if kind == "structure_skipped":
        write_polymer_snapshot(Path("run"))
        flags = ["--no-structure"]
    else:
        Path("run").mkdir()
        stages: dict[str, Any] = {"05_npt": {"samples": {}}}
        Path("run/manifest.json").write_text(
            json.dumps(
                {
                    "protocol": "x",
                    "seed": 1,
                    "stages": {} if kind == "empty" else stages,
                }
            )
        )
    assert main(["--analyse", "run", *flags]) == 1
    captured = capsys.readouterr().out
    assert "quench, a heating scan, a deformation or a relaxation" in captured
    assert "nothing to report" in captured


def test_analysing_a_relaxation_directory_reports_and_writes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--analyse dispatches on what the directory recorded, not on a flag.

    It also crosses the one seam where the scan and the reader differ: a scan
    carries an overall verdict and a directory read back does not, and the
    shared printer has to cope with both rather than assuming the richer one.
    """
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
