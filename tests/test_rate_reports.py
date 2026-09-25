"""Generic reports expose units, errors, extrapolation and missing data."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import openmmpolymer
from openmmpolymer.rate_dependence import RateObservation, analyse_rate_observations
from openmmpolymer.rate_reports import write_rate_report

from .helpers import planted_rate_report


def test_strict_json_preserves_measurements_model_difference_and_units(
    tmp_path: Path,
) -> None:
    source = planted_rate_report()
    files = write_rate_report(source, tmp_path)
    assert Path(files.json).name == "yield_strength_rates.json"
    raw = Path(files.json).read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    data = json.loads(raw)
    assert set(data) == {
        "property",
        "observations",
        "log_linear",
        "power_law",
        "notes",
        "run_dirs",
        "target_rate",
        "max_extrapolation_decades",
        "model_difference",
        "uncertainty_description",
    }
    assert data["property"]["value_unit"] == "MPa"
    assert data["property"]["rate_unit"] == "strain/ns"
    assert data["observations"][0]["source"] == "run-0"
    assert len(data["observations"]) == 3
    assert data["model_difference"] == pytest.approx(
        abs(data["log_linear"]["value"] - data["power_law"]["value"])
    )
    assert data["target_rate"] == 0.01
    assert [Path(path).name for path in files.figures] == [
        "yield_strength_rate_log_linear.png",
        "yield_strength_rate_power_law.png",
    ]
    assert all(Path(path).read_bytes().startswith(b"\x89PNG") for path in files.figures)


def test_json_writes_unknown_and_nonfinite_errors_as_null(tmp_path: Path) -> None:
    unknown = write_rate_report(
        planted_rate_report(unknown=True), tmp_path / "unknown", figures=False
    )
    data = json.loads(Path(unknown.json).read_text())
    assert data["log_linear"]["standard_error"] is None
    assert data["log_linear"]["standard_errors"] == [None] * 3
    assert not data["log_linear"]["resolved"]
    assert unknown.figures == ()
    source = planted_rate_report()
    assert source.log_linear is not None
    infinite = replace(
        source,
        log_linear=replace(source.log_linear, standard_error=math.inf, resolved=False),
    )
    files = write_rate_report(infinite, tmp_path / "infinite", figures=False)
    assert (
        json.loads(Path(files.json).read_text())["log_linear"]["standard_error"] is None
    )


def test_json_only_report_does_not_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_plot(*args: object, **kwargs: object) -> None:
        pytest.fail("A JSON-only report must not construct figures.")

    monkeypatch.setattr(
        "openmmpolymer.rate_reports.plot_rate_dependence", unexpected_plot
    )
    files = write_rate_report(planted_rate_report(), tmp_path, figures=False)
    assert Path(files.json).is_file()
    assert files.figures == ()


def test_censored_report_has_no_fake_figures_or_prediction(tmp_path: Path) -> None:
    source = planted_rate_report()
    censored = analyse_rate_observations(
        [
            *source.observations,
            RateObservation(100.0, None, None, False, notes=("No failure.",)),
        ],
        property=source.property,
        target_rate=0.01,
    )
    files = write_rate_report(censored, tmp_path)
    data = json.loads(Path(files.json).read_text())
    assert data["observations"][-1]["value"] is None
    assert data["observations"][-1]["notes"] == ["No failure."]
    assert data["log_linear"] is None and data["power_law"] is None
    assert data["model_difference"] is None
    assert data["target_rate"] == 0.01
    assert files.figures == ()


def test_default_directory_and_svg_format(tmp_path: Path) -> None:
    source = replace(planted_rate_report(), run_dirs=(str(tmp_path),))
    files = write_rate_report(source, figure_format="svg")
    assert Path(files.json).parent == tmp_path / "analysis"
    assert len(files.figures) == 2
    assert all("<svg" in Path(path).read_text() for path in files.figures)


def test_property_names_become_safe_file_names(tmp_path: Path) -> None:
    source = planted_rate_report()
    unsafe = replace(source, property=replace(source.property, name="E / rate"))
    assert Path(write_rate_report(unsafe, tmp_path, figures=False).json).name == (
        "E_rate_rates.json"
    )
    nameless = replace(source, property=replace(source.property, name="///"))
    with pytest.raises(ValueError, match="filename-safe"):
        write_rate_report(nameless, tmp_path, figures=False)


@pytest.mark.parametrize("figures", [True, False])
def test_missing_output_directory_and_unsafe_format_fail_before_writing(
    tmp_path: Path,
    figures: bool,
) -> None:
    with pytest.raises(ValueError, match="output_dir"):
        write_rate_report(planted_rate_report(), figures=figures)
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="figure_format"):
        write_rate_report(
            planted_rate_report(), output, figures=figures, figure_format="../png"
        )
    assert not output.exists()


def test_importing_the_writer_leaves_matplotlib_unloaded() -> None:
    """Reading and reporting a run should not pay for a plotting library."""
    probe = (
        "import sys, openmmpolymer.rate_reports; sys.exit('matplotlib' in sys.modules)"
    )
    root = str(Path(openmmpolymer.__file__).parents[1])
    environment = {**os.environ, "PYTHONPATH": root}
    subprocess.run([sys.executable, "-c", probe], check=True, env=environment)
