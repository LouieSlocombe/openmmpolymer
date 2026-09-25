"""Writing results to disk: atomically, as JSON, and fingerprinted."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class ReportFiles:
    """Where a report writer put things.

    Args:
        json: The machine-readable record.
        figures: Every figure written, in the order they were made.
    """

    json: str
    figures: tuple[str, ...]


def write_report(
    directory: str | Path,
    name: str,
    record: dict[str, Any],
    figures: Iterable[tuple[str, Figure]],
    figure_format: str,
) -> ReportFiles:
    """Write a prepared record and its figures, without adding report fields.

    Create *directory*, convert undefined diagnostics to JSON null, and write
    the strict JSON record atomically. Then save each ``(stem, figure)`` pair
    as ``<stem>.<figure_format>`` in iteration order. The caller owns the
    schema, filenames and figure selection; an iterable can defer plotting
    until after the JSON has been written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = write_json(directory / name, json_value(record))
    saved: list[str] = []
    for stem, figure in figures:
        path = directory / f"{stem}.{figure_format}"
        figure.savefig(path, bbox_inches="tight")
        saved.append(str(path))
    return ReportFiles(json=json_path, figures=tuple(saved))


def write_atomically(path: str | Path, text: str) -> None:
    """Write *text* to *path* without ever leaving it half-written."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, target)


def write_json(path: str | Path, record: Any, *, strict: bool = True) -> str:
    """Write *record* as indented JSON, atomically, and return the path.

    A strict record refuses NaN and infinity, so every number in it is one any
    JSON reader accepts; pass it through :func:`json_value` first to turn an
    undefined diagnostic into null. A lenient one - a manifest or a workflow
    record - writes nonfinite numbers as JavaScript spells them and falls back
    to ``str`` for anything else.
    """
    text = (
        json.dumps(record, indent=2, allow_nan=False)
        if strict
        else json.dumps(record, indent=2, default=str)
    )
    write_atomically(path, text + "\n")
    return str(path)


def json_value(value: Any) -> Any:
    """Make *value* strict JSON: plain Python numbers, with None for nonfinite ones."""
    if isinstance(value, (np.ndarray, np.generic)):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def file_sha256(path: str | Path) -> str:
    """The SHA-256 of a file's bytes, in hex."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
