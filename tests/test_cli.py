"""Tests for the command-line driver."""

from __future__ import annotations

import inspect
import json
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
    build_parser,
    main,
)
from openmmpolymer.protocols import melt_quench

from .helpers import (
    transition_at,
    two_line_curve,
    write_deformation,
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
