"""The README's examples use names, keywords and flags that exist."""

from __future__ import annotations

import ast
import inspect
import re
import shlex
from pathlib import Path

import pytest

import openmmpolymer
from openmmpolymer.__main__ import build_parser

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()


def _blocks(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", README, flags=re.DOTALL)


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
