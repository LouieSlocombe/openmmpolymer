"""Fixtures shared by the test suite.

Two ideas carry most of the weight here. The stages write relative paths, so
every test gets its own working directory. And a cell of argon atoms with a
real ``NonbondedForce`` exercises every stage - minimisation, the barostat, the
temperature ramps, the reporters, resume - in milliseconds and without a
force-field file, which is what lets the heavy legs be reserved for the things
only they can test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer.forcefield import PolymerForceField
from openmmpolymer.mdsystem import PackedBox

from .helpers import DIMER_FFXML, argon_system


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contain every test's generated files in its own temporary directory."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def dimer_forcefield(tmp_path: Path) -> PolymerForceField:
    """A real force-field XML on disk, covering one two-atom residue."""
    path = tmp_path / "dimer.xml"
    path.write_text(DIMER_FFXML)
    return PolymerForceField(
        forcefield_xml=str(path),
        base_forcefield=(),
        residue_name="DIM",
        backend="smirnoff",
    )


@pytest.fixture
def argon_box() -> tuple[PackedBox, Any]:
    """A 64-atom argon cell and its System, ready for any stage."""
    system, topology, positions = argon_system(64, 2.4)
    box = PackedBox(
        topology=topology,
        positions_nm=positions,
        box_nm=(2.4, 2.4, 2.4),
        n_molecules=64,
    )
    return box, system


@pytest.fixture
def argon_run(argon_box: tuple[PackedBox, Any]) -> Any:
    """A run context over the argon cell, on the deterministic CPU platform."""
    from openmmpolymer.simulate import prepare_run

    box, system = argon_box
    return prepare_run(
        box,
        PolymerForceField("unused.xml", (), "AR", "smirnoff"),
        platform="CPU",
        seed=11,
        system=system,
    )


@pytest.fixture
def fake_packmol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Put a stub packmol on PATH that copies a prepared answer into place.

    The stub reads the real generated input from stdin, so a test can assert
    what was asked for as well as what came back.
    """
    directory = tmp_path / "bin"
    directory.mkdir()
    script = directory / "packmol"
    script.write_text(
        "#!/bin/sh\n"
        "cat > packmol_input_seen.inp\n"
        'out=$(grep "^output " packmol_input_seen.inp | cut -d" " -f2)\n'
        'if [ -f "$PACKMOL_FAKE_OUTPUT" ]; then cp "$PACKMOL_FAKE_OUTPUT" "$out"; fi\n'
        "echo '  Success! '\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}{':'}{Path('/usr/bin')}")
    monkeypatch.delenv("PACKMOL", raising=False)
    yield script
