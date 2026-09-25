"""The three tensile measurements on the command line: breaking strength,
elongation at break and offset yield strength.

They share a ladder, a scan and a report path, so most tests here run once for
each; what each prints is its own, because each says what its number is not.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmmpolymer import __main__ as cli
from openmmpolymer.tensile import BreakingSpec, ElongationSpec, YieldSpec

SPECS = {"breaking": BreakingSpec, "elongation": ElongationSpec, "yield": YieldSpec}


def _report(name: str, *, resolved: bool = True) -> Any:
    """A report with one replica: a known peak, then the event each reads."""
    replica = SimpleNamespace(
        peak_stress_mpa=120.0,
        strain_at_peak=0.25,
        temperature_k=298.15,
        strain_rate_per_ns=0.2,
        resolved=resolved,
    )
    report = SimpleNamespace(
        replica_spread_mpa=None,
        replica_indices=(0,),
        replicas=(replica,),
        resolved=resolved,
        notes=("Fixed bonds cannot describe chemical bond scission.",),
    )
    if name == "breaking":
        replica.failure_stress_mpa = 40.0 if resolved else None
        replica.failure_strain = 0.6 if resolved else None
        report.strength_mpa = 120.0 if resolved else None
    elif name == "elongation":
        replica.elongation_percent = 60.0 if resolved else None
        replica.strain_at_break = 0.6 if resolved else None
        replica.break_stress_mpa = 40.0 if resolved else None
        report.elongation_percent = 60.0 if resolved else None
        report.replica_spread_percent = None
    else:
        replica.strength_mpa = 40.0 if resolved else None
        replica.yield_strain = 0.042 if resolved else None
        replica.modulus_mpa = 1000.0
        replica.fit_min_strain = 0.0
        replica.fit_max_strain = 0.02
        replica.offset_strain = 0.002
        report.strength_mpa = 40.0 if resolved else None
    return report


def _lines(name: str, report: Any) -> str:
    return "\n".join(getattr(cli, f"_{name}_lines")(report))


def test_strength_output_keeps_the_peak_separate_from_stress_at_drop() -> None:
    report = _report("breaking")
    report.replica_spread_mpa = 2.0
    lines = _lines("breaking", report)
    assert "ultimate nominal tensile strength = 120 MPa +/- 2 over 1 replicas" in lines
    assert "peak 120 MPa at strain 0.25, 298 K, 0.2 strain/ns" in lines
    assert "stress drop at strain 0.6, stress 40 MPa" in lines


def test_break_output_uses_the_drop_strain_and_reports_the_peak_separately() -> None:
    report = _report("elongation")
    report.replica_spread_percent = 2.0
    lines = _lines("elongation", report)
    assert "elongation at break = 60% +/- 2 percentage points" in lines
    assert "elongation at break 60% at engineering strain 0.6" in lines
    assert "peak 120 MPa at strain 0.25" in lines
    assert "stress at break 40 MPa" in lines
    assert "298 K, 0.2 strain/ns" in lines


def test_yield_output_identifies_the_proof_stress_criterion_and_conditions() -> None:
    report = _report("yield")
    report.replica_spread_mpa = 2.0
    lines = _lines("yield", report)
    assert "offset yield strength = 40 MPa +/- 2" in lines
    assert "0.2% offset proof stress 40 MPa at strain 0.042" in lines
    assert "initial elastic slope 1000 MPa over strain 0 to 0.02" in lines
    assert "298 K, 0.2 strain/ns" in lines


@pytest.mark.parametrize(
    ("name", "unresolved", "never"),
    [
        ("breaking", "strength not resolved", ["strength =", "stress drop"]),
        (
            "elongation",
            "elongation at break not resolved",
            ["elongation at break =", "stress at break"],
        ),
        ("yield", "strength not resolved", ["strength =", "initial elastic slope"]),
    ],
)
def test_an_unconfirmed_event_is_never_printed_as_the_measurement(
    name: str, unresolved: str, never: list[str]
) -> None:
    report = _report(name, resolved=False)
    report.replicas[0].strain_rate_per_ns = None
    if name == "yield":
        report.replicas[0].modulus_mpa = None
    lines = _lines(name, report)
    assert unresolved in lines
    assert "unknown strain rate" in lines
    for text in never:
        assert text not in lines


@pytest.mark.parametrize("name", sorted(SPECS))
def test_missing_replicas_do_not_renumber_the_remaining_results(name: str) -> None:
    report = _report(name, resolved=False)
    report.replica_indices = (2,)
    lines = _lines(name, report)
    assert "replica 2: " in lines
    assert "replica 0:" not in lines


@pytest.mark.parametrize(
    ("name", "controls"),
    [
        *((name, ["--failure-fraction", "1.5"]) for name in ("breaking", "elongation")),
        *((name, [f"--{name}-strain-increment", "0"]) for name in sorted(SPECS)),
        ("elongation", ["--confirmation-steps", "0"]),
        ("elongation", ["--elongation-replicas", "0"]),
        ("yield", ["--yield-offset-strain", "0"]),
        ("yield", ["--yield-fit-min-strain", "0.03", "--yield-fit-max-strain", "0.02"]),
        ("yield", ["--yield-fit-max-strain", "0.5"]),
    ],
)
def test_invalid_tensile_settings_are_rejected_before_building(
    name: str, controls: list[str], no_build: list[Any]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["[*]CC[*]", "--protocol", name, "-o", "output", *controls])
    assert not no_build
    assert not Path("output").exists()


@pytest.mark.parametrize("name", sorted(SPECS))
def test_an_explicit_temperature_wins_over_the_tensile_default(name: str) -> None:
    """Room temperature is only a default; -t 450 means 450 K here too."""
    arguments = cli.build_parser().parse_args(
        ["[*]CC[*]", "-t", "450", "--protocol", name]
    )
    assert cli.PROTOCOLS[name].settings(arguments).temperature_k == 450.0


@pytest.mark.parametrize("name", sorted(SPECS))
def test_analyse_detects_a_tensile_scan_and_writes_its_report(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tensile ladder deforms the cell, but it is not a modulus measurement."""
    report = _report(name)
    read: list[Path] = []
    written: list[tuple[Any, Any, bool, str]] = []

    def analyse(run_dir: Path) -> Any:
        read.append(run_dir)
        return report

    def write(
        result: Any, output_dir: Any, *, figures: bool, figure_format: str
    ) -> Any:
        written.append((result, output_dir, figures, figure_format))
        return SimpleNamespace(json=f"reports/{name}.json", figures=())

    def unexpected_mechanics(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("a finite tensile ladder was dispatched to the modulus report")

    for kind in (*SPECS, "quench", "heating", "load", "shear", "relax"):
        monkeypatch.setattr(cli, f"{kind}_stages", lambda path: ())
    ladder = (f"06_{name}_r0_000",)
    monkeypatch.setattr(cli, f"{name}_stages", lambda path: ladder)
    monkeypatch.setattr(cli, "deform_stages", lambda path: ladder)
    monkeypatch.setattr(cli, f"analyse_{name}", analyse)
    monkeypatch.setattr(cli, f"write_{name}_report", write)
    monkeypatch.setattr(cli, "analyse_mechanics", unexpected_mechanics)
    argv = ["--analyse", "tensile", "--no-figures", "--no-structure", "-o", "reports"]
    assert cli.main(argv) == 0
    assert read == [Path("tensile")]
    assert written == [(report, "reports", False, "png")]
    output = capsys.readouterr().out
    assert "chemical bond scission" in output
    assert f"wrote reports/{name}.json" in output


@pytest.mark.parametrize("name", sorted(SPECS))
def test_the_scan_settles_the_melt_and_writes_into_its_analysis_directory(
    name: str,
    staged_melt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(name)
    calls: list[Any] = []

    def run_scan(run: Any, directory: Path, **kwargs: Any) -> Any:
        calls.append((directory, kwargs))
        return report

    def write(result: Any, directory: Any, *, figures: bool, figure_format: str) -> Any:
        calls.append((result, directory, figures, figure_format))
        return SimpleNamespace(json=f"tensile/analysis/{name}.json", figures=("a.svg",))

    monkeypatch.setattr(cli, f"run_{name}_scan", run_scan)
    monkeypatch.setattr(cli, f"write_{name}_report", write)
    argv = ["[*]CC[*]", "--protocol", name, "-o", "tensile", "--figure-format", "svg"]
    assert cli.main([*argv, "--characteristic-ratio", "5.5"]) == 0
    assert calls[0] == (
        Path("tensile"),
        {
            "spec": SPECS[name](),
            "chain_backbone": (),
            "atoms_per_chain": 1,
            "expected_characteristic_ratio": 5.5,
            "melt_temperature_k": 600.0,
        },
    )
    assert calls[1] == (report, None, True, "svg")
    assert f"wrote tensile/analysis/{name}.json and 1 figure(s)" in (
        capsys.readouterr().out
    )
