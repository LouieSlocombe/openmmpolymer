"""Common property-rate controls preserve units, old flags and saved results."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmmpolymer import __main__ as cli
from openmmpolymer.property_rates import (
    RATE_PROPERTIES,
    default_rate_spec,
    validate_property_rate_scan,
)
from openmmpolymer.rate_dependence import RateReport

from .test_tensile_rates import _series


@pytest.mark.parametrize("property_name", list(RATE_PROPERTIES))
def test_every_property_has_a_compatible_cli_spec_and_rate_units(
    property_name: str,
) -> None:
    protocol = cli._RATE_PROTOCOLS[property_name]
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
    selected, target, spec = cli._property_rate_request(arguments)
    assert selected == property_name
    assert type(spec) is type(default_rate_spec(property_name))
    plan = validate_property_rate_scan(
        spec, arguments.rate_hold_times, property_name=selected, target_rate=target
    )
    assert plan.total_ns > 0
    assert RATE_PROPERTIES[selected].rate_unit in {"strain/ns", "bar/ns", "K/ns"}


def test_analysis_does_not_apply_unrelated_cli_temperature_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    name, target, spec = cli._property_rate_request(arguments)
    assert name == "melting_temperature" and target == 0.001
    assert type(spec) is type(default_rate_spec(name))


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
    controls: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid property rates reached monomer build")

    monkeypatch.setattr(cli, "build_chain", unexpected)
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            ["[*]CC[*]", "--protocol", "yield", "--dry-run", "-o", "new", *controls]
        )
    assert not (tmp_path / "new").exists()


def test_saved_yield_rates_use_common_reporter_and_print_correct_units(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories = _series(tmp_path, "yield_strength")
    output = tmp_path / "reports"
    assert (
        cli.main(
            [
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
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "Offset yield strength, log_linear" in text
    assert "strain/ns" in text and "MPa" in text
    record = json.loads((output / "yield_strength_rates.json").read_text())
    assert record["property"]["name"] == "yield_strength"
    assert len(record["observations"]) == 6
    assert record["target_rate"] == 0.1
    assert not list(output.glob("*.png"))


def test_strain_rate_alias_selects_yield_instead_of_youngs(tmp_path: Path) -> None:
    directories = _series(tmp_path, "yield_strength")
    assert (
        cli.main(
            [
                "--analyse",
                *(str(item) for item in directories),
                "--protocol",
                "yield",
                "--target-strain-rate",
                "0.1",
                "--no-figures",
            ]
        )
        == 0
    )
    assert (directories[0] / "analysis/yield_strength_rates.json").is_file()
    assert not (directories[0] / "analysis/modulus_rates.json").exists()


def test_unknown_property_or_wrong_spec_cannot_dispatch() -> None:
    with pytest.raises(ValueError, match="Unknown rate property"):
        default_rate_spec("density")
    with pytest.raises(ValueError, match="requires"):
        validate_property_rate_scan(
            default_rate_spec("glass_transition"),
            (1, 2, 3),
            property_name="yield_strength",
            target_rate=0.1,
        )


@pytest.mark.parametrize(
    "property_name", ["bulk_modulus", "yield_strength", "glass_transition"]
)
def test_new_cli_scan_reaches_rate_workflow_with_chain_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, property_name: str
) -> None:
    chain = SimpleNamespace(
        n_atoms=4,
        molar_mass_g_mol=50.0,
        embedder="test",
        sdf_paths=["chain.sdf"],
        pdb_paths=["chain.pdb"],
        backbone=(0, 1, 2),
    )
    box = SimpleNamespace(
        n_molecules=30,
        topology=SimpleNamespace(getNumAtoms=lambda: 120),
        positions_nm=[],
    )
    prepared = object()
    substitutes = {
        "build_chain": chain,
        "check_target_density": 5.0,
        "build_polymer_forcefield": SimpleNamespace(forcefield_xml="field.xml"),
        "distribute_conformers": [],
        "pack_box": SimpleNamespace(packed_pdb="packed.pdb", box_nm=(5, 5, 5)),
        "assemble_box": box,
        "check_packing": None,
        "prepare_box": box,
        "prepare_run": prepared,
    }
    for name, value in substitutes.items():
        monkeypatch.setattr(cli, name, lambda *args, _value=value, **kwargs: _value)
    received: dict[str, Any] = {}

    def scan(run: Any, output_dir: Path, **kwargs: Any) -> RateReport:
        assert run is prepared
        assert output_dir == tmp_path / "output"
        received.update(kwargs)
        return RateReport(RATE_PROPERTIES[property_name], (), None, None, ())

    monkeypatch.setattr(cli, "run_property_rate_scan", scan)
    assert (
        cli.main(
            [
                "[*]CC[*]",
                "--charge-method",
                "none",
                "--protocol",
                cli._RATE_PROTOCOLS[property_name],
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
        )
        == 0
    )
    assert received["chain_backbone"] == (0, 1, 2)
    assert received["atoms_per_chain"] == 4
    assert received["hold_times_ps"] == (50, 150, 500)
    assert received["target_rate"] == 0.1
    assert (tmp_path / "output/analysis" / f"{property_name}_rates.json").is_file()


def test_new_tm_rate_scan_preserves_crystalline_input_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = object()
    monkeypatch.setattr(cli, "_prepared_crystal", lambda arguments: prepared)

    def scan(run: Any, output_dir: Path, **kwargs: Any) -> RateReport:
        assert run is prepared
        assert kwargs["crystalline"] is True
        assert kwargs["state_in"] == "crystal.xml"
        assert kwargs["n_replicas"] == 2
        assert kwargs["hold_times_ps"] == (50, 150, 500)
        return RateReport(RATE_PROPERTIES["melting_temperature"], (), None, None, ())

    monkeypatch.setattr(cli, "run_property_rate_scan", scan)
    assert (
        cli.main(
            [
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
        )
        == 0
    )
    assert (tmp_path / "output/analysis/melting_temperature_rates.json").is_file()
