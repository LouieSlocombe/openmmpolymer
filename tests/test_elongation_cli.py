"""Elongation at break has its own tensile controls and percentage reporting."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmmpolymer import __main__ as cli
from openmmpolymer.tensile import ElongationSpec


def _report(*, resolved: bool = True) -> Any:
    """A confirmed drop after a separate, earlier stress maximum."""
    replica = SimpleNamespace(
        peak_stress_mpa=120.0,
        strain_at_peak=0.25,
        elongation_percent=60.0 if resolved else None,
        strain_at_break=0.6 if resolved else None,
        break_stress_mpa=40.0 if resolved else None,
        break_bracket=(0.5, 0.6) if resolved else None,
        temperature_k=298.15,
        strain_rate_per_ns=0.2,
        resolved=resolved,
    )
    return SimpleNamespace(
        elongation_percent=60.0 if resolved else None,
        replica_spread_percent=None,
        replica_indices=(0,),
        replicas=(replica,),
        resolved=resolved,
        notes=("Fixed bonds cannot describe chemical bond scission.",),
    )


def test_every_elongation_option_reaches_its_spec() -> None:
    parameters = inspect.signature(cli._elongation_spec).parameters
    assert set(cli.PROTOCOLS["elongation"].options) <= parameters.keys()
    arguments = cli.build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "elongation",
            "-t",
            "310",
            "--pressure",
            "2",
            "--deform-axis",
            "0",
            "--elongation-strain-increment",
            "0.02",
            "--elongation-max-strain",
            "0.8",
            "--elongation-relax-ps",
            "20",
            "--elongation-replicas",
            "2",
            "--elongation-samples-per-step",
            "40",
            "--elongation-stage-ps",
            "200",
            "--elongation-trajectory-ps",
            "5",
            "--failure-fraction",
            "0.4",
            "--confirmation-steps",
            "4",
            "--max-total-ns",
            "100",
        ]
    )
    spec = cli._elongation_spec(
        **cli._protocol_options(arguments, cli.PROTOCOLS["elongation"])
    )
    assert spec == ElongationSpec(
        temperature_k=310.0,
        pressure_bar=2.0,
        axis=0,
        strain_increment=0.02,
        max_strain=0.8,
        relax_ps=20.0,
        n_replicas=2,
        samples_per_step=40,
        stage_ps=200.0,
        trajectory_ps=5.0,
        failure_fraction=0.4,
        confirmation_steps=4,
        max_total_ns=100.0,
    )


def test_elongation_and_modulus_defaults_stay_independent() -> None:
    parser = cli.build_parser()
    arguments = parser.parse_args(["[*]CC[*]", "--protocol", "elongation"])
    spec = cli._elongation_spec(
        **cli._protocol_options(arguments, cli.PROTOCOLS["elongation"])
    )
    assert spec == ElongationSpec()
    assert arguments.strain_increment == 0.002
    assert arguments.max_strain == 0.05
    assert arguments.breaking_max_strain == 1.0
    assert arguments.yield_max_strain == 0.3
    assert parser.parse_args(["[*]CC[*]"]).temperature == 450.0
    explicit = parser.parse_args(["[*]CC[*]", "-t", "450", "--protocol", "elongation"])
    assert explicit.temperature == 450.0


@pytest.mark.parametrize(
    "controls",
    [
        ["--failure-fraction", "1.5"],
        ["--elongation-strain-increment", "0"],
        ["--confirmation-steps", "0"],
        ["--elongation-replicas", "0"],
        ["--max-total-ns", "0.001", "--dry-run"],
    ],
)
def test_invalid_elongation_settings_are_rejected_before_building(
    controls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_build(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid tensile settings reached the monomer build")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_chain", unexpected_build)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["[*]CC[*]", "--protocol", "elongation", "-o", "output", *controls])
    assert not (tmp_path / "output").exists()


def test_break_output_uses_the_drop_strain_and_reports_the_peak_separately() -> None:
    report = _report()
    report.replica_spread_percent = 2.0
    lines = "\n".join(cli._elongation_lines(report))
    assert "elongation at break = 60% +/- 2 percentage points" in lines
    assert "elongation at break 60% at engineering strain 0.6" in lines
    assert "peak 120 MPa at strain 0.25" in lines
    assert "stress at break 40 MPa" in lines
    assert "0.2 strain/ns" in lines
    assert "298 K" in lines


def test_an_unconfirmed_drop_is_never_printed_as_elongation_at_break() -> None:
    report = _report(resolved=False)
    report.replicas[0].strain_rate_per_ns = None
    lines = "\n".join(cli._elongation_lines(report))
    assert "elongation at break not resolved" in lines
    assert "peak 120 MPa" in lines
    assert "elongation at break =" not in lines
    assert "stress at break" not in lines
    assert "unknown strain rate" in lines


def test_missing_replicas_do_not_renumber_the_remaining_results() -> None:
    report = _report(resolved=False)
    report.replica_indices = (2,)
    lines = "\n".join(cli._elongation_lines(report))
    assert "replica 2: elongation at break not resolved" in lines
    assert "replica 0:" not in lines


def test_analyse_detects_elongation_and_writes_the_requested_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report()
    read: list[Path] = []
    written: list[tuple[Any, Any, bool, str]] = []

    def analyse(run_dir: Path) -> Any:
        read.append(run_dir)
        return report

    def write(
        result: Any, output_dir: Any, *, figures: bool, figure_format: str
    ) -> Any:
        written.append((result, output_dir, figures, figure_format))
        return SimpleNamespace(json="reports/elongation.json", figures=())

    for name in (
        "quench_stages",
        "heating_stages",
        "breaking_stages",
        "yield_stages",
        "load_stages",
        "shear_stages",
        "relax_stages",
    ):
        monkeypatch.setattr(cli, name, lambda path: ())
    monkeypatch.setattr(cli, "elongation_stages", lambda path: ("06_elongation_r0_00",))
    monkeypatch.setattr(cli, "deform_stages", lambda path: ("06_elongation_r0_00",))
    monkeypatch.setattr(cli, "analyse_elongation", analyse)
    monkeypatch.setattr(cli, "write_elongation_report", write)

    def unexpected_mechanics(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("a finite tensile ladder was dispatched to the modulus report")

    monkeypatch.setattr(cli, "analyse_mechanics", unexpected_mechanics)
    assert (
        cli.main(
            ["--analyse", "tensile", "--no-figures", "--no-structure", "-o", "reports"]
        )
        == 0
    )
    assert read == [Path("tensile")]
    assert written == [(report, "reports", False, "png")]
    output = capsys.readouterr().out
    assert "elongation at break = 60%" in output
    assert "chemical bond scission" in output
    assert "wrote reports/elongation.json" in output


def test_the_scan_passes_chain_metadata_and_writes_into_its_analysis_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = cli.build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "elongation",
            "-o",
            "tensile",
            "--figure-format",
            "svg",
        ]
    )
    arguments.characteristic_ratio = 5.5
    report = _report()
    calls: list[Any] = []

    def run_scan(run: Any, directory: Path, **kwargs: Any) -> Any:
        calls.append((run, directory, kwargs))
        return report

    def write(result: Any, directory: Any, *, figures: bool, figure_format: str) -> Any:
        calls.append((result, directory, figures, figure_format))
        return SimpleNamespace(
            json="tensile/analysis/elongation.json", figures=("curve.svg",)
        )

    monkeypatch.setattr(cli, "run_elongation_scan", run_scan)
    monkeypatch.setattr(cli, "write_elongation_report", write)
    chain = SimpleNamespace(backbone=(0, 1), n_atoms=20)
    assert (
        cli._run_elongation_scan(arguments, "context", Path("tensile"), chain, {}) == 0
    )
    assert calls[0] == (
        "context",
        Path("tensile"),
        {
            "spec": ElongationSpec(),
            "chain_backbone": (0, 1),
            "atoms_per_chain": 20,
            "expected_characteristic_ratio": 5.5,
        },
    )
    assert calls[1] == (report, None, True, "svg")
