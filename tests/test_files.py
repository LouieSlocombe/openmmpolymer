"""Direct contracts for the small, shared result-writing helpers."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from openmmpolymer._files import (
    analysis_directory,
    figure_stem,
    json_value,
    write_atomically,
    write_json,
    write_report,
)


def test_analysis_directory_defaults_to_run_without_creating_files(
    tmp_path: Path,
) -> None:
    assert analysis_directory(tmp_path, None) == tmp_path / "analysis"
    assert analysis_directory(tmp_path, "elsewhere") == Path("elsewhere")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("06_coarse_00, 06_coarse_01", "06_coarse_00_06_coarse_01"),
        ("../../outside/a\\b", "outside_a_b"),
        ("E / rate", "E_rate"),
        ("_known-stage_", "known-stage"),
        ("../", ""),
    ],
)
def test_figure_stem_is_one_safe_filename_component(name: str, expected: str) -> None:
    assert figure_stem(name) == expected
    assert figure_stem(name, fallback="observable") == (expected or "observable")


def test_json_value_converts_nested_numpy_and_nonfinite_values() -> None:
    """Strict JSON keeps shape and finite precision, with null for missing data."""
    record = {
        "array": np.array([[1.25, np.nan], [np.inf, -np.inf]]),
        "nested": (np.int64(7), {"flag": np.bool_(True), "value": np.float32(0.5)}),
        "missing": None,
        "label": "sample",
    }
    result = json_value(record)
    assert result == {
        "array": [[1.25, None], [None, None]],
        "nested": [7, {"flag": True, "value": 0.5}],
        "missing": None,
        "label": "sample",
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert isinstance(record["array"], np.ndarray)


def test_write_atomically_replaces_only_after_the_text_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.json"
    target.write_text("previous complete record")
    replace = os.replace

    def checked_replace(source: Path, destination: Path) -> None:
        assert destination == target
        assert target.read_text() == "previous complete record"
        assert source.read_text() == "next complete record\n"
        replace(source, destination)

    monkeypatch.setattr("openmmpolymer._files.os.replace", checked_replace)
    write_atomically(str(target), "next complete record\n")
    assert target.read_text() == "next complete record\n"
    assert list(tmp_path.iterdir()) == [target]


def test_a_failed_atomic_replace_preserves_the_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.json"
    target.write_text("previous complete record")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("openmmpolymer._files.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_atomically(target, "next complete record")
    assert target.read_text() == "previous complete record"


def test_write_json_returns_the_path_and_ends_an_indented_record_with_a_newline(
    tmp_path: Path,
) -> None:
    target = tmp_path / "record.json"
    assert write_json(target, {"value": [2.0, None]}) == str(target)
    assert target.read_text() == '{\n  "value": [\n    2.0,\n    null\n  ]\n}\n'


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, Path("unsupported")])
def test_strict_json_refuses_unserializable_values_before_touching_disk(
    tmp_path: Path, value: object
) -> None:
    target = tmp_path / "record.json"
    target.write_text("previous complete record")
    with pytest.raises((TypeError, ValueError)):
        write_json(target, {"value": value})
    assert target.read_text() == "previous complete record"
    assert list(tmp_path.iterdir()) == [target]


def test_lenient_json_keeps_nonfinite_numbers_and_stringifies_other_objects(
    tmp_path: Path,
) -> None:
    target = tmp_path / "record.json"
    write_json(
        target,
        {"values": [math.nan, math.inf, -math.inf], "path": Path("run")},
        strict=False,
    )
    record = json.loads(target.read_text())
    assert math.isnan(record["values"][0])
    assert record["values"][1:] == [math.inf, -math.inf]
    assert record["path"] == "run"


def test_write_report_writes_strict_json_before_iterating_figures_in_order(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "nested" / "report"

    def figures() -> Iterator[tuple[str, Figure]]:
        assert json.loads((directory / "result.json").read_text()) == {"error": None}
        for name in ("second", "first"):
            yield name, Figure(figsize=(1, 1))

    files = write_report(
        directory, "result.json", {"error": math.nan}, figures(), "png"
    )
    assert files.json == str(directory / "result.json")
    assert files.figures == tuple(
        str(directory / f"{name}.png") for name in ("second", "first")
    )
    assert all(Path(path).read_bytes().startswith(b"\x89PNG") for path in files.figures)


@pytest.mark.parametrize(
    "figure_format", ["", ".png", "../png", "not-a-format", "png/svg"]
)
def test_bad_report_format_refuses_before_creating_a_directory_or_iterating_figures(
    tmp_path: Path, figure_format: str
) -> None:
    directory = tmp_path / "report"

    def figures() -> Iterator[tuple[str, Figure]]:
        yield "unused", pytest.fail("Invalid formats must be rejected before plotting.")

    with pytest.raises(ValueError, match="figure_format"):
        write_report(directory, "result.json", {}, figures(), figure_format)
    assert not directory.exists()


def test_a_json_only_report_adds_no_fields(tmp_path: Path) -> None:
    files = write_report(tmp_path, "result.json", {"sample": 7}, (), "svg")
    assert json.loads(Path(files.json).read_text()) == {"sample": 7}
    assert files.figures == ()
