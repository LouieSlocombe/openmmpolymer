"""Rate analysis reaches saved measurements and preserves its qualifications."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmmpolymer import __main__ as cli
from openmmpolymer.modulus_rate_report import write_modulus_rate_report
from openmmpolymer.modulus_rates import ModulusRateReport, analyse_modulus_rates

from .helpers import write_deformation


def _runs(tmp_path: Path) -> list[str]:
    """Three rates with E increasing by 100 MPa per decade."""
    directories = []
    for index, hold in enumerate((50.0, 500.0, 5000.0)):
        directory = tmp_path / f"rate_{index}"
        write_deformation(directory, relax_ps=hold, modulus_mpa=2200.0 - 100 * index)
        directories.append(str(directory))
    return directories


def test_saved_rate_analysis_reports_both_models_and_writes_strict_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories = _runs(tmp_path)
    output = tmp_path / "analysis"
    assert (
        cli.main(
            [
                "--analyse",
                *directories,
                "--target-strain-rate",
                "0.0001",
                "--elastic-strain-limit",
                "0.02",
                "--no-figures",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "log_linear: E =" in text
    assert "power_law: E =" in text
    assert "0.0001 strain/ns" in text
    assert "100 MPa per decade" in text
    assert "3 measured rates" in text
    assert "model difference at target" in text
    record = json.loads((output / "modulus_rates.json").read_text())
    assert len(record["fits"]) == 3
    assert record["log_linear"]["strain_limit"] == 0.02
    assert record["log_linear"]["modulus_mpa"] < 2000.0
    assert record["power_law"]["target_rate_per_ns"] == 0.0001
    assert isinstance(record["log_linear"]["strain_rate_per_ns"], list)
    assert list(output.glob("*.png")) == []


def test_distant_extrapolation_is_printed_and_saved_as_unresolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories = _runs(tmp_path)
    assert (
        cli.main(
            ["--analyse", *directories, "--target-strain-rate", "1e-12", "--no-figures"]
        )
        == 0
    )
    assert capsys.readouterr().out.count("(not resolved)") == 2
    record = json.loads(
        (Path(directories[0]) / "analysis/modulus_rates.json").read_text()
    )
    for form in ("log_linear", "power_law"):
        assert record[form]["resolved"] is False
        assert record[form]["extrapolation_decades"] > 8


@pytest.mark.parametrize(
    "controls",
    [
        ["--modulus-relax-times", "10,50,100"],
        ["--target-strain-rate", "0.01"],
        ["--modulus-relax-times", "10,50", "--target-strain-rate", "0.01"],
        ["--modulus-relax-times", "10,10,50", "--target-strain-rate", "0.01"],
        ["--modulus-relax-times", "10,nan,50", "--target-strain-rate", "0.01"],
        ["--modulus-relax-times", "10,-50,100", "--target-strain-rate", "0.01"],
        ["--modulus-relax-times", "10,50,100", "--target-strain-rate", "0"],
        ["--modulus-relax-times", "10,50,100", "--target-strain-rate", "nan"],
        ["--modulus-relax-times", "10,50,100", "--target-strain-rate", "inf"],
        [
            "--modulus-relax-times",
            "10,50,100",
            "--target-strain-rate",
            "0.01",
            "--max-total-ns",
            "0.001",
            "--dry-run",
        ],
    ],
)
def test_invalid_rate_scan_is_rejected_before_building_or_writing(
    controls: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid rate controls reached the monomer build")

    monkeypatch.setattr(cli, "build_chain", unexpected)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["[*]CC[*]", "--protocol", "modulus", "-o", "output", *controls])
    assert not (tmp_path / "output").exists()


def test_rate_scan_flags_cannot_be_ignored_by_another_protocol() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "[*]CC[*]",
                "--protocol",
                "yield",
                "--modulus-relax-times",
                "10,50,100",
                "--target-strain-rate",
                "0.01",
            ]
        )


def test_saved_rate_analysis_requires_enough_rates_and_a_target(tmp_path: Path) -> None:
    directories = _runs(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--analyse", *directories, "--protocol", "modulus"])
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--analyse", directories[0], "--target-strain-rate", "0.0001"])


def test_scan_dispatch_passes_holds_and_target_without_changing_single_rate_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = analyse_modulus_rates(_runs(tmp_path), target_rate_per_ns=0.0001)
    received: dict[str, Any] = {}

    def scan(run: Any, output_dir: Path, **kwargs: Any) -> ModulusRateReport:
        received.update(kwargs)
        return report

    monkeypatch.setattr(cli, "run_modulus_rate_scan", scan)
    arguments = cli.build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "modulus",
            "--modulus-relax-times",
            "50,500,5000",
            "--target-strain-rate",
            "0.0001",
            "--elastic-strain-limit",
            "0.02",
            "--no-figures",
        ]
    )
    assert (
        cli._run_modulus_scan(
            arguments,
            object(),
            tmp_path / "output",
            SimpleNamespace(backbone=(0, 1), n_atoms=2),
            cli._protocol_options(arguments, cli.PROTOCOLS["modulus"]),
        )
        == 0
    )
    assert received["relax_ps"] == (50.0, 500.0, 5000.0)
    assert received["target_rate_per_ns"] == 0.0001
    assert received["spec"].elastic_strain_limit == 0.02
    assert received["chain_backbone"] == (0, 1)
    assert (tmp_path / "output/analysis/modulus_rates.json").is_file()


def test_rate_report_writes_both_figures_and_null_for_undefined_diagnostics(
    tmp_path: Path,
) -> None:
    report = analyse_modulus_rates(_runs(tmp_path), target_rate_per_ns=0.0001)
    report = replace(
        report,
        log_linear=replace(
            report.log_linear, standard_error_mpa=math.inf, resolved=False
        ),
    )
    files = write_modulus_rate_report(report)
    record = json.loads(Path(files.json).read_text())
    assert record["log_linear"]["standard_error_mpa"] is None
    assert not record["log_linear"]["resolved"]
    assert "model_difference_mpa" in record
    assert len(files.figures) == 2
    assert all(Path(path).stat().st_size > 1000 for path in files.figures)
