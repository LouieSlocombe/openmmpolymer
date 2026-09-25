"""Rate controls on the command line keep their units, old flags and reports.

``--modulus-relax-times`` and ``--target-strain-rate`` predate the other rate
properties; they are the Young's modulus spellings of ``--rate-hold-times``
and ``--target-property-rate``, and report through the same writer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer import __main__ as cli
from openmmpolymer.property_rates import (
    RATE_PROPERTIES,
    default_rate_spec,
    validate_property_rate_scan,
)
from openmmpolymer.rate_dependence import RateReport

from .helpers import (
    deformation_rate_per_ns,
    write_modulus_rate_series,
    write_tensile_rate_series,
)


def _modulus_runs(tmp_path: Path) -> list[str]:
    """Three rates a decade apart, Young's modulus rising 100 MPa per decade."""
    fastest = deformation_rate_per_ns(50.0)
    directories = write_modulus_rate_series(
        tmp_path,
        (50.0, 500.0, 5000.0),
        modulus=lambda rate: 2200.0 + 100.0 * math.log10(rate / fastest),
    )
    return [str(directory) for directory in directories]


def _refused_before_building(argv: list[str], no_build: list[Any]) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(argv)
    assert not no_build
    assert not Path("output").exists()


@pytest.mark.parametrize("property_name", list(RATE_PROPERTIES))
def test_every_property_has_a_compatible_cli_spec_and_rate_units(
    property_name: str,
) -> None:
    protocol = cli._rate_protocol(property_name)
    arguments = cli.build_parser().parse_args(
        [
            "--protocol",
            protocol,
            "--rate-property",
            property_name,
            "--rate-hold-times",
            "10,20,30",
            "--target-property-rate",
            "0.001",
        ]
    )
    request = cli._property_rate_request(arguments)
    assert request.property_name == property_name
    assert request.hold_times_ps == (10.0, 20.0, 30.0)
    spec = cli.PROTOCOLS[protocol].settings(arguments)
    assert type(spec) is type(default_rate_spec(property_name))
    plan = validate_property_rate_scan(
        spec,
        request.hold_times_ps,
        property_name=property_name,
        target_rate=request.target_rate,
    )
    assert plan.total_ns > 0


def test_the_youngs_modulus_flags_are_the_generic_ones() -> None:
    old = cli.build_parser().parse_args(
        [
            "--protocol",
            "modulus",
            "--modulus-relax-times",
            "50,150,500",
            "--target-strain-rate",
            "0.001",
        ]
    )
    new = cli.build_parser().parse_args(
        [
            "--protocol",
            "modulus",
            "--rate-property",
            "youngs_modulus",
            "--rate-hold-times",
            "50,150,500",
            "--target-property-rate",
            "0.001",
        ]
    )
    both = cli.build_parser().parse_args(
        [
            "--protocol",
            "modulus",
            "--rate-property",
            "youngs_modulus",
            "--rate-hold-times",
            "50,150,500",
            "--modulus-relax-times",
            "50,150,500",
            "--target-property-rate",
            "0.001",
            "--target-strain-rate",
            "0.001",
        ]
    )
    request = cli._property_rate_request(new)
    assert request.property_name == "youngs_modulus"
    assert cli._property_rate_request(old) == request
    assert cli._property_rate_request(both) == request


def test_the_strain_rate_target_alone_analyses_youngs_modulus(tmp_path: Path) -> None:
    arguments = cli.build_parser().parse_args(
        ["--analyse", str(tmp_path), "--target-strain-rate", "0.001"]
    )
    request = cli._property_rate_request(arguments)
    assert request.property_name == "youngs_modulus"
    assert request.target_rate == 0.001
    assert request.hold_times_ps == ()


def test_analysis_does_not_apply_unrelated_cli_temperature_defaults(
    tmp_path: Path,
) -> None:
    arguments = cli.build_parser().parse_args(
        [
            "--analyse",
            str(tmp_path),
            "--rate-property",
            "melting_temperature",
            "--target-property-rate",
            "0.001",
        ]
    )
    request = cli._property_rate_request(arguments)
    assert request == cli._RateRequest("melting_temperature", 0.001, ())


@pytest.mark.parametrize(
    "controls",
    [
        ["--rate-property", "yield_strength", "--rate-hold-times", "1,2,3"],
        [
            "--rate-property",
            "bulk_modulus",
            "--rate-hold-times",
            "1,2,3",
            "--target-strain-rate",
            "0.1",
        ],
        [
            "--rate-property",
            "yield_strength",
            "--rate-hold-times",
            "1,1,2",
            "--target-property-rate",
            "0.1",
        ],
        [
            "--rate-property",
            "yield_strength",
            "--rate-hold-times",
            "1,2,3",
            "--target-property-rate",
            "nan",
        ],
        [
            "--rate-property",
            "yield_strength",
            "--rate-hold-times",
            "1,2,3",
            "--target-property-rate",
            "-1",
        ],
        [
            "--rate-property",
            "yield_strength",
            "--rate-hold-times",
            "1,2,3",
            "--target-property-rate",
            "0.1",
            "--max-total-ns",
            ".001",
        ],
        [
            "--rate-property",
            "yield_strength",
            "--rate-hold-times",
            "1,2,3",
            "--target-property-rate",
            "0.1",
            "--max-rate-extrapolation-decades",
            "-1",
        ],
        [
            "--rate-property",
            "yield_strength",
            "--rate-hold-times",
            "1,2,3",
            "--target-property-rate",
            "0.1",
            "--target-strain-rate",
            "0.2",
        ],
        [
            "--rate-property",
            "yield_strength",
            "--rate-hold-times",
            "1,2,3",
            "--modulus-relax-times",
            "1,2,4",
            "--target-property-rate",
            "0.1",
        ],
        [
            "--rate-property",
            "yield_strength",
            "--modulus-relax-times",
            "1,2,3",
            "--target-property-rate",
            "0.1",
        ],
    ],
)
def test_invalid_rate_controls_stop_before_building(
    controls: list[str], no_build: list[Any]
) -> None:
    _refused_before_building(
        ["[*]CC[*]", "--protocol", "yield", "--dry-run", "-o", "output", *controls],
        no_build,
    )


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
def test_invalid_youngs_rate_scan_is_rejected_before_building_or_writing(
    controls: list[str], no_build: list[Any]
) -> None:
    _refused_before_building(
        ["[*]CC[*]", "--protocol", "modulus", "-o", "output", *controls], no_build
    )


def test_youngs_rate_flags_cannot_be_ignored_by_another_protocol_or_analysis(
    tmp_path: Path, no_build: list[Any]
) -> None:
    rate = ["--modulus-relax-times", "10,50,100", "--target-strain-rate", "0.01"]
    _refused_before_building(["[*]CC[*]", "--protocol", "yield", *rate], no_build)
    _refused_before_building(["--analyse", str(tmp_path), *rate], no_build)


def test_saved_youngs_rates_report_through_the_common_writer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "analysis"
    argv = [
        "--analyse",
        *_modulus_runs(tmp_path),
        "--target-strain-rate",
        "0.0001",
        "--elastic-strain-limit",
        "0.02",
        "--no-figures",
        "-o",
        str(output),
    ]
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert "Young's modulus, log_linear: " in text
    assert "Young's modulus, power_law: " in text
    assert "MPa (fit SE) at 0.0001 strain/ns; 3 rates" in text
    assert "model difference at target" in text
    assert not (output / "modulus_rates.json").exists()
    record = json.loads((output / "youngs_modulus_rates.json").read_text())
    assert record["property"]["name"] == "youngs_modulus"
    assert len(record["observations"]) == 3
    assert all(
        item["conditions"] == {"strain_limit": 0.02} for item in record["observations"]
    )
    assert record["log_linear"]["value"] < 2000.0
    assert record["log_linear"]["sensitivity_per_decade"] == pytest.approx(100.0)
    assert record["power_law"]["target_rate"] == 0.0001
    assert isinstance(record["log_linear"]["rates"], list)
    assert list(output.glob("*.png")) == []


def test_distant_extrapolation_is_printed_and_saved_as_unresolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories = _modulus_runs(tmp_path)
    argv = ["--analyse", *directories, "--target-strain-rate", "1e-12", "--no-figures"]
    assert cli.main(argv) == 0
    assert capsys.readouterr().out.count("(not resolved)") == 2
    record = json.loads(
        (Path(directories[0]) / "analysis/youngs_modulus_rates.json").read_text()
    )
    for form in ("log_linear", "power_law"):
        assert record[form]["resolved"] is False
        assert record[form]["extrapolation_decades"] > 8


def test_saved_rate_analysis_needs_a_target_and_reports_too_few_rates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories = _modulus_runs(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--analyse", *directories, "--protocol", "modulus"])
    capsys.readouterr()
    argv = ["--analyse", directories[0], "--target-strain-rate", "0.0001"]
    assert cli.main([*argv, "--no-figures"]) == 0
    text = capsys.readouterr().out
    assert text.count("unavailable (not resolved)") == 2
    assert "three or more distinct rates" in text


def test_saved_yield_rates_use_common_reporter_and_print_correct_units(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories = write_tensile_rate_series(tmp_path, "yield_strength")
    output = tmp_path / "reports"
    argv = [
        "--analyse",
        *(str(item) for item in directories),
        "--rate-property",
        "yield_strength",
        "--target-property-rate",
        "0.1",
        "--no-figures",
        "-o",
        str(output),
    ]
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert "Offset yield strength, log_linear" in text
    assert "strain/ns" in text and "MPa" in text
    record = json.loads((output / "yield_strength_rates.json").read_text())
    assert record["property"]["name"] == "yield_strength"
    assert len(record["observations"]) == 6
    assert record["target_rate"] == 0.1
    assert not list(output.glob("*.png"))


def test_strain_rate_alias_selects_yield_instead_of_youngs(tmp_path: Path) -> None:
    directories = write_tensile_rate_series(tmp_path, "yield_strength")
    argv = [
        "--analyse",
        *(str(item) for item in directories),
        "--protocol",
        "yield",
        "--target-strain-rate",
        "0.1",
        "--no-figures",
    ]
    assert cli.main(argv) == 0
    assert (directories[0] / "analysis/yield_strength_rates.json").is_file()
    assert not (directories[0] / "analysis/youngs_modulus_rates.json").exists()


@pytest.mark.parametrize(
    "property_name", ["bulk_modulus", "yield_strength", "glass_transition"]
)
def test_new_cli_scan_reaches_rate_workflow_with_chain_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staged_melt: dict[str, Any],
    property_name: str,
) -> None:
    received: dict[str, Any] = {}

    def scan(run: Any, output_dir: Path, **kwargs: Any) -> RateReport:
        # The verified build, not the scratch rebuild it was checked against.
        assert (
            Path(run.forcefield.forcefield_xml) == output_dir / "build/polymer_ff.xml"
        )
        assert output_dir == tmp_path / "output"
        received.update(kwargs)
        return RateReport(RATE_PROPERTIES[property_name], (), None, None, ())

    monkeypatch.setattr(cli, "run_property_rate_scan", scan)
    argv = [
        "[*]CC[*]",
        "--charge-method",
        "none",
        "--characteristic-ratio",
        "5.5",
        "--protocol",
        cli._rate_protocol(property_name),
        "--rate-property",
        property_name,
        "--rate-hold-times",
        "50,150,500",
        "--target-property-rate",
        ".1",
        "--no-figures",
        "-o",
        str(tmp_path / "output"),
    ]
    assert cli.main(argv) == 0
    assert received["chain_backbone"] == ()
    assert received["atoms_per_chain"] == 1
    assert received["expected_characteristic_ratio"] == 5.5
    # A mechanical scan settles its melt as the flags ask; tg, from its spec.
    assert ("melt_temperature_k" in received) == (property_name != "glass_transition")
    assert received["hold_times_ps"] == (50, 150, 500)
    assert received["target_rate"] == 0.1
    assert (tmp_path / "output/analysis" / f"{property_name}_rates.json").is_file()


def test_a_youngs_scan_from_the_original_flags_runs_the_common_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staged_melt: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, Any] = {}

    def scan(run: Any, output_dir: Path, **kwargs: Any) -> RateReport:
        received.update(kwargs)
        return RateReport(RATE_PROPERTIES["youngs_modulus"], (), None, None, ())

    monkeypatch.setattr(cli, "run_property_rate_scan", scan)
    argv = [
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
        "-o",
        str(tmp_path / "output"),
    ]
    assert cli.main(argv) == 0
    assert received["property_name"] == "youngs_modulus"
    assert received["hold_times_ps"] == (50.0, 500.0, 5000.0)
    assert received["target_rate"] == 0.0001
    assert received["spec"].elastic_strain_limit == 0.02
    assert received["chain_backbone"] == ()
    assert "youngs_modulus rate scan: " in capsys.readouterr().out
    assert (tmp_path / "output/analysis/youngs_modulus_rates.json").is_file()
    assert not (tmp_path / "output/analysis/modulus_rates.json").exists()


def test_new_tm_rate_scan_preserves_crystalline_input_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = object()
    monkeypatch.setattr(cli, "load_crystal", lambda *args, **kwargs: prepared)

    def scan(run: Any, output_dir: Path, **kwargs: Any) -> RateReport:
        assert run is prepared
        assert kwargs["crystalline"] is True
        assert kwargs["state_in"] == "crystal.xml"
        assert kwargs["n_replicas"] == 2
        assert kwargs["hold_times_ps"] == (50, 150, 500)
        return RateReport(RATE_PROPERTIES["melting_temperature"], (), None, None, ())

    monkeypatch.setattr(cli, "run_property_rate_scan", scan)
    argv = [
        "--protocol",
        "tm",
        "--crystal-pdb",
        "crystal.pdb",
        "--system-xml",
        "system.xml",
        "--state-in",
        "crystal.xml",
        "--rate-property",
        "melting_temperature",
        "--rate-hold-times",
        "50,150,500",
        "--target-property-rate",
        "1",
        "--thermal-rate-replicas",
        "2",
        "--no-figures",
        "-o",
        str(tmp_path / "output"),
    ]
    assert cli.main(argv) == 0
    assert (tmp_path / "output/analysis/melting_temperature_rates.json").is_file()
