"""Fixtures shared by the test suite.

Two ideas carry most of the weight here. The stages write relative paths, so
every test gets its own working directory. And a cell of argon atoms with a
real ``NonbondedForce`` exercises every stage - minimisation, the barostat, the
temperature ramps, the reporters, resume - in milliseconds and without a
force-field file, which is what lets the heavy legs be reserved for the things
only they can test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# The test cells are tens of atoms: one CPU thread runs them three times faster
# than a thread per core spinning on barriers.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

from openmmpolymer.forcefield import PolymerForceField
from openmmpolymer.mdsystem import PackedBox

from .helpers import DIMER_FFXML, argon_context, argon_system


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
def argon_run() -> Any:
    """A run context over the 64-atom argon cell, on the deterministic CPU."""
    return argon_context(64, 2.4)


@pytest.fixture
def argon_scan_run() -> Any:
    """An argon cell big enough to survive an NPT equilibration and a strain.

    Sixty-four atoms reach a liquid density at an edge below twice the cutoff,
    and OpenMM refuses that outright; two hundred and sixteen do not.
    """
    return argon_context(216, 2.8)


@pytest.fixture
def dimer_argon_run() -> Any:
    """A cell of 32 bonded two-atom molecules: enough for a chain measurement."""
    return argon_context(64, 2.4, atoms_per_molecule=2)


@pytest.fixture
def dimer_trajectory(dimer_argon_run: Any) -> Any:
    """A real ten-frame XTC and its topology, written by a real stage.

    The analysis layer's job is reading what this package writes, so the
    fixture is this package writing it rather than a hand-made file. Sixty-four
    argon atoms on the CPU platform run at some thousands of steps a second, so
    a two-picosecond stage costs a quarter of a second.

    ``interval_ps`` is set explicitly: left alone, frames land at ten times the
    state-data interval, and the ten-picosecond default would put the first one
    after the stage had ended.
    """
    from openmmpolymer.reporters import TrajectoryOptions
    from openmmpolymer.simulate import run_minimise, run_nvt

    minimised = run_minimise(dimer_argon_run, "00_minimise")
    return run_nvt(
        dimer_argon_run,
        "02_nvt",
        temperature_k=120.0,
        duration_ps=2.0,
        friction_ps=20.0,
        trajectory=TrajectoryOptions("xtc", interval_ps=0.2),
        state_in=minimised.final_state,
    )


@pytest.fixture
def dimer_run_directory(dimer_trajectory: Any, tmp_path: Path) -> Path:
    """A run directory with a manifest, as ``run_protocol`` would leave one.

    Written here rather than by running a protocol because what the analysis
    needs is the manifest's shape, and a protocol run would cost every stage to
    get it.
    """
    import json

    manifest = {
        "protocol": "fixture",
        "seed": 11,
        "versions": {},
        "system": {},
        "stages": {
            "02_nvt": {
                "name": "02_nvt",
                "steps": dimer_trajectory.steps,
                "final_state": dimer_trajectory.final_state,
                "final_pdb": dimer_trajectory.final_pdb,
                "csv": dimer_trajectory.csv,
                "samples": {},
            }
        },
        "chains": None,
        "box": {
            "n_molecules": 32,
            "atoms_per_chain": 2,
            "box_nm": [2.4, 2.4, 2.4],
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return tmp_path


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


@pytest.fixture
def staged_melt(monkeypatch: pytest.MonkeyPatch, argon_run: Any) -> dict[str, Any]:
    """Stand in for the chemistry behind ``build_melt``, keeping its staging real.

    Each preparation writes the four kinds of asset ``build_melt`` records,
    adds an entry to the force-field cache it was handed, and returns the argon
    run with its force-field reference in the build directory. Set
    ``system_suffix`` to change the next preparation's Hamiltonian or ``fail``
    to make it raise; ``builds`` lists every build directory used and
    ``options`` the settings the last preparation was given.
    """
    from dataclasses import replace

    from openmmpolymer import melt
    from openmmpolymer.chain import ChainResult

    control: dict[str, Any] = {
        "system_suffix": "",
        "fail": False,
        "builds": [],
        "options": {},
    }

    def prepare(
        spec: Any, n_chains: int, build_dir: Path, cache_dir: Path, **options: Any
    ) -> Any:
        control["builds"].append(build_dir)
        control["options"] = options
        number = len(control["builds"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"entry_{number}.xml").write_text("parameters")
        build_dir.mkdir(parents=True, exist_ok=True)
        for name in ("chain_0.sdf", "chain_0.pdb", "polymer_ff.xml", "packed.pdb"):
            (build_dir / name).write_text(f"prepared artifact {number}")
        if control["fail"]:
            raise RuntimeError("preparation failed")
        chain = ChainResult(
            sdf_paths=(str(build_dir / "chain_0.sdf"),),
            pdb_paths=(str(build_dir / "chain_0.pdb"),),
            smiles="[Ar]",
            n_atoms=1,
            molar_mass_g_mol=39.948,
        )
        run = replace(
            argon_run,
            system_xml=argon_run.system_xml + control["system_suffix"],
            forcefield=replace(
                argon_run.forcefield,
                forcefield_xml=str(build_dir / "polymer_ff.xml"),
            ),
        )
        return chain, run

    monkeypatch.setattr(melt, "_prepare", prepare)
    return control
