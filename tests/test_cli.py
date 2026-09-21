"""Tests for the command-line driver."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmmpolymer.__main__ import PROTOCOLS, build_parser, main


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
    for factory in PROTOCOLS.values():
        assert factory(target_temperature_k=400.0).stages


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
