"""Offset yield scans reach the CLI with their own controls and report path."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmmpolymer import __main__ as cli
from openmmpolymer.yielding import YieldSpec


def _report(*, resolved: bool = True) -> Any:
    """A proof stress distinct from the initial elastic slope."""
    replica = SimpleNamespace(
        strength_mpa=40.0 if resolved else None,
        yield_strain=0.042 if resolved else None,
        modulus_mpa=1000.0,
        fit_min_strain=0.0,
        fit_max_strain=0.02,
        offset_strain=0.002,
        temperature_k=298.15,
        strain_rate_per_ns=0.04,
        resolved=resolved,
    )
    return SimpleNamespace(
        strength_mpa=40.0 if resolved else None,
        replica_spread_mpa=None,
        replica_indices=(0,),
        replicas=(replica,),
        resolved=resolved,
        notes=("Offset proof stress does not establish irreversible deformation.",),
    )


def test_every_yield_option_reaches_its_spec() -> None:
    parameters = inspect.signature(cli._yield_spec).parameters
    assert set(cli.PROTOCOLS["yield"].options) <= parameters.keys()
    arguments = cli.build_parser().parse_args(
        [
            "[*]CC[*]",
            "--protocol",
            "yield",
            "-t",
            "310",
            "--pressure",
            "2",
            "--deform-axis",
            "0",
            "--yield-strain-increment",
            "0.001",
            "--yield-max-strain",
            "0.2",
            "--yield-relax-ps",
            "20",
            "--yield-replicas",
            "2",
            "--yield-samples-per-step",
            "40",
            "--yield-stage-ps",
            "200",
            "--yield-trajectory-ps",
            "5",
            "--yield-offset-strain",
            "0.005",
            "--yield-fit-min-strain",
            "0.001",
            "--yield-fit-max-strain",
            "0.015",
            "--max-total-ns",
            "100",
        ]
    )
    spec = cli._yield_spec(**cli._protocol_options(arguments, cli.PROTOCOLS["yield"]))
    assert spec == YieldSpec(
        temperature_k=310.0,
        pressure_bar=2.0,
        axis=0,
        strain_increment=0.001,
        max_strain=0.2,
        relax_ps=20.0,
        n_replicas=2,
        samples_per_step=40,
        stage_ps=200.0,
        trajectory_ps=5.0,
        offset_strain=0.005,
        fit_min_strain=0.001,
        fit_max_strain=0.015,
        max_total_ns=100.0,
    )


def test_yield_defaults_stay_independent_of_other_tensile_workflows() -> None:
    parser = cli.build_parser()
    arguments = parser.parse_args(["[*]CC[*]", "--protocol", "yield"])
    spec = cli._yield_spec(**cli._protocol_options(arguments, cli.PROTOCOLS["yield"]))
    assert spec == YieldSpec()
    assert arguments.breaking_max_strain == 1.0
    assert arguments.max_strain == 0.05
    assert parser.parse_args(["[*]CC[*]"]).temperature == 450.0
    explicit = parser.parse_args(["[*]CC[*]", "-t", "450", "--protocol", "yield"])
    assert explicit.temperature == 450.0


@pytest.mark.parametrize(
    "controls",
    [
        ["--yield-offset-strain", "0"],
        ["--yield-strain-increment", "0"],
        ["--yield-fit-min-strain", "0.03", "--yield-fit-max-strain", "0.02"],
        ["--yield-fit-max-strain", "0.5"],
        ["--max-total-ns", "0.001", "--dry-run"],
    ],
)
def test_invalid_yield_settings_are_rejected_before_building(
    controls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_build(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid yield settings reached the monomer build")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_chain", unexpected_build)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["[*]CC[*]", "--protocol", "yield", "-o", "output", *controls])
    assert not (tmp_path / "output").exists()


def test_yield_output_identifies_the_proof_stress_criterion_and_conditions() -> None:
    report = _report()
    report.replica_spread_mpa = 2.0
    lines = "\n".join(cli._yield_lines(report))
    assert "offset yield strength = 40 MPa +/- 2" in lines
    assert "0.2% offset proof stress 40 MPa at strain 0.042" in lines
    assert "initial elastic slope 1000 MPa over strain 0 to 0.02" in lines
    assert "0.04 strain/ns" in lines
    assert "298 K" in lines


def test_an_unresolved_crossing_is_never_printed_as_strength() -> None:
    report = _report(resolved=False)
    report.replicas[0].strain_rate_per_ns = None
    report.replicas[0].modulus_mpa = None
    lines = "\n".join(cli._yield_lines(report))
    assert "strength not resolved" in lines
    assert "strength =" not in lines
    assert "initial elastic slope" not in lines
    assert "unknown strain rate" in lines


def test_missing_replicas_do_not_renumber_the_remaining_results() -> None:
    report = _report(resolved=False)
    report.replica_indices = (1,)
    lines = "\n".join(cli._yield_lines(report))
    assert "replica 1: 0.2% offset proof stress" in lines
    assert "replica 0:" not in lines


def test_analyse_detects_yield_and_writes_the_requested_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report()
    read: list[Path] = []
    written: list[tuple[Any, Any, Any]] = []

    def analyse(run_dir: Path) -> Any:
        read.append(run_dir)
        return report

    def write(result: Any, output_dir: Any, *, formats: Any) -> Any:
        written.append((result, output_dir, formats))
        return SimpleNamespace(json="reports/yield.json", figures=())

    for name in (
        "quench_stages",
        "heating_stages",
        "breaking_stages",
        "load_stages",
        "shear_stages",
        "relax_stages",
    ):
        monkeypatch.setattr(cli, name, lambda path: ())
    monkeypatch.setattr(cli, "yield_stages", lambda path: ("06_yield_r0_00",))
    monkeypatch.setattr(cli, "deform_stages", lambda path: ("06_yield_r0_00",))
    monkeypatch.setattr(cli, "analyse_yield", analyse)
    monkeypatch.setattr(cli, "write_yield_report", write)

    def unexpected_mechanics(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("a yield ladder was dispatched to the modulus report")

    monkeypatch.setattr(cli, "analyse_mechanics", unexpected_mechanics)
    assert (
        cli.main(
            ["--analyse", "tensile", "--no-figures", "--no-structure", "-o", "reports"]
        )
        == 0
    )
    assert read == [Path("tensile")]
    assert written == [(report, "reports", ())]
    output = capsys.readouterr().out
    assert "offset yield strength" in output
    assert "irreversible deformation" in output
    assert "wrote reports/yield.json" in output


def test_the_scan_passes_chain_metadata_and_writes_into_its_analysis_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = cli.build_parser().parse_args(
        ["[*]CC[*]", "--protocol", "yield", "-o", "tensile", "--figure-format", "svg"]
    )
    arguments.characteristic_ratio = 5.5
    report = _report()
    calls: list[Any] = []

    def run_scan(run: Any, directory: Path, **kwargs: Any) -> Any:
        calls.append((run, directory, kwargs))
        return report

    def write(result: Any, directory: Any, *, formats: Any) -> Any:
        calls.append((result, directory, formats))
        return SimpleNamespace(
            json="tensile/analysis/yield.json", figures=("curve.svg",)
        )

    monkeypatch.setattr(cli, "run_yield_scan", run_scan)
    monkeypatch.setattr(cli, "write_yield_report", write)
    chain = SimpleNamespace(backbone=(0, 1), n_atoms=20)
    assert cli._run_yield_scan(arguments, "context", Path("tensile"), chain, {}) == 0
    assert calls[0] == (
        "context",
        Path("tensile"),
        {
            "spec": YieldSpec(),
            "chain_backbone": (0, 1),
            "atoms_per_chain": 20,
            "expected_characteristic_ratio": 5.5,
        },
    )
    assert calls[1] == (report, None, ("svg",))
