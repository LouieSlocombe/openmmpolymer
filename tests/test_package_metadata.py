"""Tests that pin the public surface and keep the install routes in step."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

import openmmpolymer

ROOT = Path(__file__).resolve().parent.parent

#: pyproject names whose conda-forge package is spelled differently.
CONDA_NAMES = {"MDAnalysis": "mdanalysis", "matplotlib": "matplotlib-base"}


@pytest.fixture(scope="module")
def dependencies() -> list[str]:
    """The runtime requirements pyproject declares."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return list(tomllib.load(handle)["project"]["dependencies"])


def test_the_public_api_is_explicit_and_complete() -> None:
    """Every name in __all__ exists, and nothing is listed twice."""
    assert len(openmmpolymer.__all__) == len(set(openmmpolymer.__all__))
    assert all(hasattr(openmmpolymer, name) for name in openmmpolymer.__all__)


def test_every_stage_is_exported() -> None:
    """A protocol names them, so a caller has to be able to reach them."""
    from openmmpolymer.protocols import STAGE_RUNNERS

    names = {runner.__name__ for runner in STAGE_RUNNERS.values()}
    assert names <= set(openmmpolymer.__all__)


def test_the_conda_environment_declares_every_dependency_and_floor(
    dependencies: list[str],
) -> None:
    """conda-forge is the only install route that resolves, so it must agree."""
    environment = (ROOT / "build_tools" / "environment.yml").read_text()
    for entry in dependencies:
        name, _, floor = entry.partition(">=")
        wanted = CONDA_NAMES.get(name, name) + (f">={floor}" if floor else "")
        assert re.search(rf"- {re.escape(wanted)}\s", environment), wanted
    # packmol comes from ambertools; asking for it separately makes the
    # environment unsolvable, because conda-forge's own builds pin numpy < 2.
    assert "\n  - packmol" not in environment


def test_the_minimum_ci_job_tests_the_declared_openmm_floor(
    dependencies: list[str],
) -> None:
    """The floor is only a promise if something runs against it."""
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow.is_file():
        pytest.skip("CI configuration is not part of a source distribution")
    floor = next(entry for entry in dependencies if entry.startswith("openmm>="))
    assert f"openmm=={floor.removeprefix('openmm>=')}" in workflow.read_text()


def test_the_source_distribution_carries_the_reference_workflow(
    tmp_path: Path,
) -> None:
    """Inspect a real archive: a manifest that looks right can still omit inputs."""
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "openmmpolymer",
        "tests",
        "benchmarks",
        "build_tools",
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
    ):
        original = ROOT / name
        if original.is_dir():
            ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
            shutil.copytree(original, source / name, ignore=ignore)
        else:
            shutil.copy2(original, source / name)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_sdist; build_sdist('dist')",
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    with tarfile.open(next((source / "dist").glob("*.tar.gz"))) as archive:
        members = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    expected = {
        "benchmarks/__init__.py",
        "benchmarks/pe_melt.py",
        "benchmarks/polyethylene.json",
        "benchmarks/README.md",
        "build_tools/environment.yml",
        "tests/__init__.py",
        "tests/helpers.py",
        "tests/conftest.py",
        "openmmpolymer/py.typed",
    }
    assert expected <= members
