"""Tests that pin the public surface and the declared metadata."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

import openmmpolymer

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def project() -> dict[str, object]:
    """The parsed pyproject, for the metadata tests."""
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def test_the_public_api_is_explicit_and_complete() -> None:
    """Every name in __all__ exists, and nothing is listed twice."""
    assert len(openmmpolymer.__all__) == len(set(openmmpolymer.__all__))
    assert all(hasattr(openmmpolymer, name) for name in openmmpolymer.__all__)


def test_the_four_layers_are_all_reachable_from_the_top() -> None:
    """Chain, charges, force field, packing, system, stages, protocol."""
    expected = {
        "build_chain",
        "assign_charges",
        "build_polymer_forcefield",
        "pack_box",
        "assemble_box",
        "prepare_run",
        "run_protocol",
        "standard_melt_equilibration",
    }
    assert expected <= set(openmmpolymer.__all__)


def test_every_stage_is_exported() -> None:
    """A protocol names them, so a caller has to be able to reach them."""
    from openmmpolymer.protocols import STAGE_RUNNERS

    for runner in STAGE_RUNNERS.values():
        assert runner.__name__ in openmmpolymer.__all__


def test_the_runtime_version_matches_the_installed_metadata() -> None:
    """A wheel that reports a different version from its metadata is broken."""
    if openmmpolymer.__version__ == "0.0.0+unknown":
        pytest.skip("running from a checkout that is not installed")
    assert openmmpolymer.__version__ == version("openmmpolymer")


def test_the_declared_version_matches_the_installed_one(
    project: dict[str, object],
) -> None:
    """pyproject is the single source of truth."""
    if openmmpolymer.__version__ == "0.0.0+unknown":
        pytest.skip("running from a checkout that is not installed")
    declared = project["project"]["version"]  # type: ignore[index]
    assert openmmpolymer.__version__.startswith(str(declared))


def test_the_hard_dependencies_are_declared(project: dict[str, object]) -> None:
    """Everything imported at run time, and nothing imported lazily instead."""
    declared = {
        entry.split(">")[0].split("=")[0].strip()
        for entry in project["project"]["dependencies"]  # type: ignore[index]
    }
    assert {
        "numpy",
        "openmm",
        "rdkit",
        "forcefill",
        "openff-toolkit",
        "MDAnalysis",
        "matplotlib",
    } <= declared


def test_no_dependency_is_a_direct_url(project: dict[str, object]) -> None:
    """A direct URL cannot be published to an index."""
    for entry in project["project"]["dependencies"]:  # type: ignore[index]
        assert "@" not in entry


def test_the_python_floor_matches_forcefills(project: dict[str, object]) -> None:
    """forcefill declares 3.12, and this cannot install where that will not."""
    assert project["project"]["requires-python"] == ">=3.12"  # type: ignore[index]


def test_every_pytest_marker_is_registered(project: dict[str, object]) -> None:
    """--strict-markers is on, so an unregistered marker is a collection error."""
    configured = project["tool"]["pytest"]["ini_options"]["markers"]  # type: ignore[index]
    names = {entry.split(":")[0] for entry in configured}
    assert names == {"forcefield", "packmol", "slow", "cuda"}


def test_warnings_are_errors_by_default(project: dict[str, object]) -> None:
    """Every ignore below it is one that was observed, and says which."""
    filters = project["tool"]["pytest"]["ini_options"]["filterwarnings"]  # type: ignore[index]
    assert filters[0] == "error"
    assert all(entry.startswith("ignore:") for entry in filters[1:])


def test_the_coverage_floor_is_recorded(project: dict[str, object]) -> None:
    """Below the achieved figure, so coverage-neutral changes pass."""
    assert project["tool"]["coverage"]["report"]["fail_under"] == 90  # type: ignore[index]


def test_the_package_ships_its_typing_marker() -> None:
    """py.typed is what makes the annotations usable downstream."""
    from importlib.resources import files

    assert files("openmmpolymer").joinpath("py.typed").is_file()


def test_the_conda_environment_covers_the_hard_dependencies() -> None:
    """conda-forge is the only install route that resolves, so it must be right."""
    text = (PYPROJECT.parent / "build_tools" / "environment.yml").read_text()
    for package in ("openmm", "rdkit", "ambertools", "openff-toolkit", "mdanalysis"):
        assert f"- {package}" in text
    assert "forcefill" in text
    # Spelled out rather than left to `- matplotlib` matching it by substring,
    # because which of the two is asked for is the point: the plotting helpers
    # never touch pyplot, so the GUI toolkit would be dead weight.
    assert "- matplotlib-base" in text
    # packmol comes from ambertools; asking for it separately makes the
    # environment unsolvable, because conda-forge's builds of it pin numpy < 2.
    assert "\n  - packmol" not in text
