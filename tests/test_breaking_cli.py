"""The finite tensile workflow is reachable without reusing modulus defaults."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmmpolymer import __main__ as cli
from openmmpolymer.tensile import BreakingSpec


def _report(*, resolved: bool = True) -> Any:
    """A known peak and a distinct stress drop for CLI formatting checks."""
    replica = SimpleNamespace(
        peak_stress_mpa=120.0,
        strain_at_peak=0.25,
        failure_stress_mpa=40.0 if resolved else None,
        failure_strain=0.6 if resolved else None,
        temperature_k=298.15,
        strain_rate_per_ns=0.2,
        resolved=resolved,
    )
    return SimpleNamespace(
        strength_mpa=120.0 if resolved else None,
        replica_spread_mpa=None,
        replica_indices=(0,),
        replicas=(replica,),
        resolved=resolved,
        notes=("Fixed bonds cannot describe chemical bond scission.",),
    )


def test_every_breaking_option_reaches_its_spec() -> None:
    parameters = inspect.signature(cli._breaking_spec).parameters
    assert set(cli.PROTOCOLS["breaking"].options) <= parameters.keys()
    arguments = cli.build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "breaking",
            "-t",
            "310",
            "--pressure",
            "2",
            "--deform-axis",
            "0",
            "--breaking-strain-increment",
            "0.02",
            "--breaking-max-strain",
            "0.8",
            "--breaking-relax-ps",
            "20",
            "--breaking-replicas",
            "2",
            "--breaking-samples-per-step",
            "40",
            "--breaking-stage-ps",
            "200",
            "--breaking-trajectory-ps",
            "5",
            "--failure-fraction",
            "0.4",
            "--confirmation-steps",
            "4",
            "--max-total-ns",
            "100",
        ]
    )
    spec = cli._breaking_spec(
        **cli._protocol_options(arguments, cli.PROTOCOLS["breaking"])
    )
    assert spec == BreakingSpec(
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


def test_breaking_and_modulus_defaults_stay_independent() -> None:
    parser = cli.build_parser()
    arguments = parser.parse_args(["[*]CC[*]", "--protocol", "breaking"])
    spec = cli._breaking_spec(
        **cli._protocol_options(arguments, cli.PROTOCOLS["breaking"])
    )
    assert spec == BreakingSpec()
    assert arguments.strain_increment == 0.002
    assert arguments.max_strain == 0.05
    assert parser.parse_args(["[*]CC[*]"]).temperature == 450.0
    explicit = parser.parse_args(["[*]CC[*]", "-t", "450", "--protocol", "breaking"])
    assert explicit.temperature == 450.0


@pytest.mark.parametrize(
    "controls",
    [
        ["--failure-fraction", "1.5"],
        ["--breaking-strain-increment", "0"],
        ["--max-total-ns", "0.001", "--dry-run"],
    ],
)
def test_invalid_breaking_settings_are_rejected_before_building(
    controls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_build(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid tensile settings reached the monomer build")

    monkeypatch.setattr(cli, "build_chain", unexpected_build)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["[*]CC[*]", "--protocol", "breaking", "-o", "output", *controls])
    assert not (tmp_path / "output").exists()


def test_strength_output_keeps_the_peak_separate_from_stress_at_drop() -> None:
    lines = "\n".join(cli._breaking_lines(_report()))
    assert "ultimate nominal tensile strength = 120 MPa" in lines
    assert "peak 120 MPa at strain 0.25" in lines
    assert "stress drop at strain 0.6, stress 40 MPa" in lines
    assert "0.2 strain/ns" in lines
    assert "298 K" in lines


def test_an_unconfirmed_peak_is_never_printed_as_strength() -> None:
    report = _report(resolved=False)
    report.replicas[0].strain_rate_per_ns = None
    lines = "\n".join(cli._breaking_lines(report))
    assert "strength not resolved" in lines
    assert "peak 120 MPa" in lines
    assert "strength =" not in lines
    assert "stress drop" not in lines
    assert "unknown strain rate" in lines


def test_missing_replicas_do_not_renumber_the_remaining_results() -> None:
    report = _report(resolved=False)
    report.replica_indices = (1,)
    lines = "\n".join(cli._breaking_lines(report))
    assert "replica 1: peak 120 MPa" in lines
    assert "replica 0:" not in lines


def test_analyse_detects_breaking_and_writes_the_requested_report(
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
        return SimpleNamespace(json="reports/breaking.json", figures=())

    for name in (
        "quench_stages",
        "heating_stages",
        "load_stages",
        "shear_stages",
        "relax_stages",
    ):
        monkeypatch.setattr(cli, name, lambda path: ())
    monkeypatch.setattr(cli, "breaking_stages", lambda path: ("06_breaking_r0_00",))
    monkeypatch.setattr(cli, "deform_stages", lambda path: ("06_breaking_r0_00",))
    monkeypatch.setattr(cli, "analyse_breaking", analyse)
    monkeypatch.setattr(cli, "write_breaking_report", write)

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
    assert "ultimate nominal tensile strength" in output
    assert "chemical bond scission" in output
    assert "wrote reports/breaking.json" in output


def test_the_scan_passes_chain_metadata_and_writes_into_its_analysis_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = cli.build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "breaking",
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
            json="tensile/analysis/breaking.json", figures=("curve.svg",)
        )

    monkeypatch.setattr(cli, "run_breaking_scan", run_scan)
    monkeypatch.setattr(cli, "write_breaking_report", write)
    chain = SimpleNamespace(backbone=(0, 1), n_atoms=20)
    assert cli._run_breaking_scan(arguments, "context", Path("tensile"), chain, {}) == 0
    assert calls[0] == (
        "context",
        Path("tensile"),
        {
            "spec": BreakingSpec(),
            "chain_backbone": (0, 1),
            "atoms_per_chain": 20,
            "expected_characteristic_ratio": 5.5,
        },
    )
    assert calls[1] == (report, None, True, "svg")
