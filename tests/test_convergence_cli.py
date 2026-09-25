"""Saved time-window diagnostics are accessible without starting dynamics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer import __main__ as cli

from .helpers import state_data_csv, write_manifest


def _state_run(directory: Path) -> None:
    directory.mkdir()
    noise = np.random.default_rng(7).normal(0.0, 0.5, 6000)
    rows = [
        [float(i), float(i), -1000 + x, 500.0, -500 + x, 300 + x, 100.0, 1 + x / 1000]
        for i, x in enumerate(noise)
    ]
    csv = directory / "hold.csv"
    csv.write_text(state_data_csv(rows))
    write_manifest(
        directory, {"hold": {"name": "hold", "csv": str(csv), "samples": {}}}
    )


def test_saved_convergence_cli_writes_strict_json_and_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "saved"
    _state_run(directory)
    output = tmp_path / "report"
    assert (
        cli.main(
            [
                "--analyse",
                str(directory),
                "--convergence",
                "--convergence-stage",
                "hold",
                "--window-fractions",
                ".2,.4,.7,1",
                "--convergence-tolerance",
                ".2",
                "--min-effective-samples",
                "15",
                "--convergence-discard-fraction",
                ".2",
                "--no-figures",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    record = json.loads((output / "convergence.json").read_text())
    density = record["results"]["density_g_cm3"]
    assert density["resolved"]
    assert density["relative_tolerance"] == 0.2
    assert density["min_effective_samples"] == 15
    assert density["discard_fraction"] == 0.2
    assert [window["fraction"] for window in density["windows"]] == [0.2, 0.4, 0.7, 1]
    assert not list(output.glob("*.png"))
    assert "density_g_cm3: resolved" in capsys.readouterr().out


@pytest.mark.parametrize(
    "controls",
    [
        [],
        ["--analyse", "one", "two"],
        ["--analyse", "one", "--target-strain-rate", "0.1"],
        [
            "--analyse",
            "one",
            "--rate-property",
            "yield_strength",
            "--target-property-rate",
            "0.1",
        ],
        ["--analyse", "one", "--window-fractions", ".5,1"],
        ["--analyse", "one", "--convergence-tolerance", "nan"],
        ["--analyse", "one", "--min-effective-samples", "0"],
        ["--analyse", "one", "--convergence-discard-fraction", "1"],
    ],
)
def test_invalid_convergence_controls_cannot_reach_dynamics_or_write_output(
    controls: list[str], no_build: list[Any]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--convergence", "-o", "new", *controls])
    assert not no_build
    assert not Path("new").exists()


def test_convergence_defaults_output_to_saved_analysis(tmp_path: Path) -> None:
    directory = tmp_path / "saved"
    _state_run(directory)
    assert cli.main(["--analyse", str(directory), "--convergence", "--no-figures"]) == 0
    assert (directory / "analysis/convergence.json").is_file()
