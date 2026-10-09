"""The documentation's examples use names, keywords and flags that exist."""

from __future__ import annotations

import ast
import inspect
import re
import shlex
from pathlib import Path

import pytest

import openmmpolymer
from openmmpolymer.__main__ import build_parser

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTATION = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def _blocks(language: str) -> list[str]:
    return [
        block
        for path in DOCUMENTATION
        for block in re.findall(
            rf"```{language}\n(.*?)```", path.read_text(), flags=re.DOTALL
        )
    ]


PYTHON = _blocks("python")
COMMANDS = [
    shlex.split(command)[1:]
    for block in _blocks("bash")
    for command in block.replace("\\\n", " ").splitlines()
    if command.startswith("openmmpolymer ")
]


@pytest.mark.parametrize("source", PYTHON)
def test_python_examples_call_the_api_as_it_is(source: str) -> None:
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "openmmpolymer"
        for alias in node.names
    }
    assert imported <= set(openmmpolymer.__all__)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in imported:
            continue
        parameters = inspect.signature(getattr(openmmpolymer, node.func.id)).parameters
        if any(
            item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        ):
            continue
        for keyword in node.keywords:
            assert keyword.arg in parameters, f"{node.func.id}({keyword.arg}=...)"


@pytest.mark.parametrize("argv", COMMANDS, ids=" ".join)
def test_command_line_examples_parse(argv: list[str]) -> None:
    build_parser().parse_args(argv)


def test_the_examples_were_found() -> None:
    """A change to the fences must not quietly leave nothing to check."""
    assert PYTHON and COMMANDS


def test_guide_rate_table_matches_registered_properties_and_units() -> None:
    from openmmpolymer.property_rates import RATE_PROPERTIES

    text = (ROOT / "docs" / "guide.md").read_text()
    section = text.split("| `property_name` |", 1)[1].split("```", 1)[0]
    rows = [
        line.split("|")[1:-1] for line in section.splitlines() if line.startswith("| `")
    ]
    documented = {
        row[0].strip().strip("`"): (row[2].strip(), row[3].strip()) for row in rows
    }
    assert len(rows) == len(documented)
    assert documented == {
        name: (property.value_unit, property.rate_unit)
        for name, property in RATE_PROPERTIES.items()
    }
