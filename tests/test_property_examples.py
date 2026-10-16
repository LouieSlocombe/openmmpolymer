"""Exercise the property walkthrough commands without starting long MD scans."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import create_autospec

import pytest

import openmmpolymer

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Example:
    filename: str
    runner: str
    analyser: str
    writer: str
    output: str


EXAMPLES = (
    Example("glass_transition", "run_tg_scan", "analyse_tg", "write_tg_report", "tg"),
    Example(
        "melting_temperature",
        "run_tm_scan",
        "analyse_melting",
        "write_melting_report",
        "tm",
    ),
    Example(
        "youngs_modulus",
        "run_modulus_scan",
        "analyse_mechanics",
        "write_mechanical_report",
        "modulus",
    ),
    Example(
        "yield_strength",
        "run_yield_scan",
        "analyse_yield",
        "write_yield_report",
        "yield",
    ),
    Example(
        "breaking_strength",
        "run_breaking_scan",
        "analyse_breaking",
        "write_breaking_report",
        "breaking",
    ),
    Example(
        "elongation_at_break",
        "run_elongation_scan",
        "analyse_elongation",
        "write_elongation_report",
        "elongation",
    ),
)

RESOLVED_HEADLINES = {
    "glass_transition": "5. Tg = 26.85 °C",
    "melting_temperature": "5. Apparent Tm = 26.85 °C",
    "youngs_modulus": "E = 1250 MPa",
    "yield_strength": "Apparent offset yield strength: 125 MPa",
    "breaking_strength": "Apparent breaking strength: 125 MPa",
    "elongation_at_break": "Apparent elongation at break: 50%",
}


def _load(example: Example) -> ModuleType:
    path = ROOT / "examples" / f"{example.filename}.py"
    spec = importlib.util.spec_from_file_location(f"example_{example.filename}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(*, resolved: bool = True, has_value: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        resolved=resolved,
        temperature_k=300.0 if has_value else None,
        cooling_rate_k_per_ns=10.0,
        curve=SimpleNamespace(heating_rate_k_per_ns=10.0),
        transition=SimpleNamespace(bracket_k=(290.0, 310.0)),
        youngs=(
            SimpleNamespace(resolved=True, modulus_mpa=1250.0, strain_rate_per_ns=0.1)
            if has_value
            else None
        ),
        strength_mpa=125.0 if has_value else None,
        elongation_percent=50.0 if has_value else None,
        replica_spread_mpa=5.0,
        method_gap=math.nan,
        replica_indices=(0,),
        curves=(SimpleNamespace(strain_rate_per_ns=0.1),),
        notes=("Check the sampling window.",),
    )


def _workflow(
    monkeypatch: pytest.MonkeyPatch, example: Example, report: SimpleNamespace
) -> tuple[ModuleType, dict[str, Any], object]:
    """Patch public functions before loading either eager or lazy example imports."""
    chain = SimpleNamespace(backbone=(0, 1), n_atoms=8)
    run = object()
    mocks = {}
    results = {
        "build_melt": (chain, run),
        "load_crystal": run,
        example.runner: None,
        example.analyser: report,
        example.writer: SimpleNamespace(json="analysis/result.json"),
    }
    for name, result in results.items():
        mock = create_autospec(getattr(openmmpolymer, name), return_value=result)
        monkeypatch.setattr(openmmpolymer, name, mock)
        mocks[name] = mock
    return _load(example), mocks, run


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.filename)
def test_property_example_help_needs_no_simulation_inputs(example: Example) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / f"{example.filename}.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--analyse-only" in result.stdout
    assert "--no-figures" in result.stdout


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.filename)
@pytest.mark.parametrize("analyse_only", (False, True))
def test_property_example_runs_or_only_reads_and_always_writes_the_report(
    example: Example,
    analyse_only: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    report = _report()
    module, mocks, run = _workflow(monkeypatch, example, report)
    directory = tmp_path / "measurements"
    arguments = ["--output", str(directory)]
    if analyse_only:
        arguments.extend(("--analyse-only", "--no-figures"))
    elif example.filename == "melting_temperature":
        arguments.extend(("--crystal-pdb", "crystal.pdb", "--system-xml", "system.xml"))
    assert module.build_parser().parse_args([]).output == Path(
        f"{example.output}-example"
    )

    module.main(arguments)

    if analyse_only:
        mocks["build_melt"].assert_not_called()
        mocks["load_crystal"].assert_not_called()
        mocks[example.runner].assert_not_called()
    else:
        mocks[example.runner].assert_called_once()
        assert mocks[example.runner].call_args.args[:2] == (run, directory)
        if example.filename == "melting_temperature":
            mocks["load_crystal"].assert_called_once_with(
                Path("crystal.pdb"), Path("system.xml")
            )
            mocks["build_melt"].assert_not_called()
            assert mocks[example.runner].call_args.kwargs["crystalline"] is True
        else:
            mocks["build_melt"].assert_called_once()
            assert mocks["build_melt"].call_args.kwargs["directory"] == directory
            assert mocks[example.runner].call_args.kwargs["chain_backbone"] == (0, 1)
            assert mocks[example.runner].call_args.kwargs["atoms_per_chain"] == 8
    mocks[example.analyser].assert_called_once()
    assert mocks[example.analyser].call_args.args == (directory,)
    mocks[example.writer].assert_called_once_with(report, figures=not analyse_only)
    output = capsys.readouterr().out
    assert "analysis/result.json" in output
    assert "Check the sampling window." in output
    assert RESOLVED_HEADLINES[example.filename] in output
    if example.filename == "melting_temperature":
        assert "Sampled bracket = 16.85 to 36.85 °C" in output


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.filename)
@pytest.mark.parametrize(("resolved", "has_value"), ((False, True), (True, False)))
def test_unsupported_or_missing_estimates_are_saved_but_not_quoted(
    example: Example,
    resolved: bool,
    has_value: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(resolved=resolved, has_value=has_value)
    module, mocks, _ = _workflow(monkeypatch, example, report)
    module.main(["--analyse-only"])

    output = capsys.readouterr().out
    assert "unresolved" in output.lower()
    assert RESOLVED_HEADLINES[example.filename] not in output
    if example.filename in ("glass_transition", "melting_temperature"):
        assert "°C" not in output
    mocks[example.writer].assert_called_once_with(report, figures=True)
    mocks["build_melt"].assert_not_called()
    mocks["load_crystal"].assert_not_called()
    mocks[example.runner].assert_not_called()


@pytest.mark.parametrize(
    "arguments", ([], ["--crystal-pdb", "crystal.pdb"], ["--system-xml", "system.xml"])
)
def test_a_new_melting_scan_requires_both_prepared_crystal_files(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = next(item for item in EXAMPLES if item.filename == "melting_temperature")
    module, mocks, _ = _workflow(monkeypatch, example, _report())
    with pytest.raises(SystemExit) as raised:
        module.main(arguments)
    assert raised.value.code == 2
    assert "--crystal-pdb and --system-xml are required" in capsys.readouterr().err
    for mock in mocks.values():
        mock.assert_not_called()
