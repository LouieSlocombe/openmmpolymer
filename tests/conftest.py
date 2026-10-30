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
import shutil
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest

# The test cells are tens of atoms: one CPU thread runs them three times faster
# than a thread per core spinning on barriers.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

from openmmpolymer.forcefield import PolymerForceField
from openmmpolymer.mdsystem import PackedBox
from openmmpolymer.simulate import StageResult

from .helpers import DIMER_FFXML, argon_cell, argon_context, write_manifest


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
    return argon_cell(64, 2.4)


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
def short_pushoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real scan preparation without a polymer-length push-off on argon.

    The lattice has no packing overlaps. Exercise all three timestep rungs in
    0.06 ps instead of spending 10 ps preparing every workflow test. A partial
    keeps the runner's signature, so budgets and resume fingerprints see the
    duration that actually runs. Tests of production defaults do not opt in.
    """
    from openmmpolymer.protocols import STAGE_RUNNERS

    monkeypatch.setitem(
        STAGE_RUNNERS, "pushoff", partial(STAGE_RUNNERS["pushoff"], duration_ps=0.06)
    )


@pytest.fixture
def dimer_argon_run() -> Any:
    """A cell of 32 bonded two-atom molecules: enough for a chain measurement."""
    return argon_context(64, 2.4, atoms_per_molecule=2)


@pytest.fixture(scope="session")
def _dimer_trajectory_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, StageResult]:
    """Generate the analysis tests' real ten-frame trajectory once.

    Each consumer gets its own copy below, so it can remove files or edit the
    manifest without changing another test's input. The trajectory still comes
    from the production writer, including its topology and final snapshot.
    """
    from openmmpolymer.reporters import TrajectoryOptions
    from openmmpolymer.simulate import run_minimise, run_nvt

    directory = tmp_path_factory.mktemp("dimer_trajectory")
    run = argon_context(64, 2.4, atoms_per_molecule=2)
    minimised = run_minimise(run, directory / "00_minimise")
    measured = run_nvt(
        run,
        directory / "02_nvt",
        temperature_k=120.0,
        duration_ps=2.0,
        friction_ps=20.0,
        trajectory=TrajectoryOptions("xtc", interval_ps=0.2),
        state_in=minimised.final_state,
    )
    return directory, measured


@pytest.fixture
def dimer_trajectory(
    _dimer_trajectory_template: tuple[Path, StageResult], tmp_path: Path
) -> StageResult:
    """A private copy of the real trajectory and its mutable stage metadata."""
    directory, measured = _dimer_trajectory_template
    assert measured.final_pdb is not None
    assert measured.csv is not None
    shutil.copytree(directory, tmp_path, dirs_exist_ok=True)
    return replace(
        measured,
        final_state=str(tmp_path / Path(measured.final_state).name),
        final_pdb=str(tmp_path / Path(measured.final_pdb).name),
        csv=str(tmp_path / Path(measured.csv).name),
        samples=deepcopy(measured.samples),
    )


@pytest.fixture
def dimer_run_directory(dimer_trajectory: Any, tmp_path: Path) -> Path:
    """A run directory with a manifest, as ``run_protocol`` would leave one.

    Written here rather than by running a protocol because what the analysis
    needs is the manifest's shape, and a protocol run would cost every stage to
    get it.
    """
    write_manifest(
        tmp_path,
        {
            "02_nvt": {
                "name": "02_nvt",
                "steps": dimer_trajectory.steps,
                "final_state": dimer_trajectory.final_state,
                "final_pdb": dimer_trajectory.final_pdb,
                "csv": dimer_trajectory.csv,
                "samples": {},
            }
        },
        protocol="fixture",
        seed=11,
        box={
            "n_molecules": 32,
            "atoms_per_chain": 2,
            "box_nm": [2.4, 2.4, 2.4],
        },
    )
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


def _build_control() -> dict[str, Any]:
    """The knobs a staged build exposes: what to break, and what it was given."""
    return {"system_suffix": "", "fail": False, "builds": [], "options": {}}


def _stage_build(
    control: dict[str, Any],
    argon_run: Any,
    build_dir: Path,
    cache_dir: Path,
    options: dict[str, Any],
    names: Sequence[str],
    *,
    extra_files: Sequence[str] = (),
) -> tuple[tuple[Any, ...], Any]:
    """Stand in for the chemistry behind a build, keeping its staging real.

    Writes the kinds of asset a build records - a conformer and a PDB per
    polymer in *names*, *extra_files*, the merged force field and the packed
    cell - adds an entry to the force-field cache it was handed, and returns
    the argon run with its force-field reference in the build directory.
    """
    from openmmpolymer.chain import ChainResult

    control["builds"].append(build_dir)
    control["options"] = options
    number = len(control["builds"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"entry_{number}.xml").write_text("parameters")
    build_dir.mkdir(parents=True, exist_ok=True)
    files = [
        *(f"{name}_0.{suffix}" for name in names for suffix in ("sdf", "pdb")),
        *extra_files,
        "polymer_ff.xml",
        "packed.pdb",
    ]
    for name in files:
        (build_dir / name).write_text(f"prepared artifact {number}")
    if control["fail"]:
        raise RuntimeError("preparation failed")
    chains = tuple(
        ChainResult(
            sdf_paths=(str(build_dir / f"{name}_0.sdf"),),
            pdb_paths=(str(build_dir / f"{name}_0.pdb"),),
            smiles="[Ar]",
            n_atoms=1,
            molar_mass_g_mol=39.948,
        )
        for name in names
    )
    run = replace(
        argon_run,
        system_xml=argon_run.system_xml + control["system_suffix"],
        forcefield=replace(
            argon_run.forcefield,
            forcefield_xml=str(build_dir / "polymer_ff.xml"),
        ),
    )
    return chains, run


@pytest.fixture
def staged_melt(monkeypatch: pytest.MonkeyPatch, argon_run: Any) -> dict[str, Any]:
    """Stand in for the chemistry behind ``build_melt``, keeping its staging real.

    Set ``system_suffix`` to change the next preparation's Hamiltonian or
    ``fail`` to make it raise; ``builds`` lists every build directory used and
    ``options`` the settings the last preparation was given.
    """
    from openmmpolymer import melt

    control = _build_control()

    def prepare(
        spec: Any, n_chains: int, build_dir: Path, cache_dir: Path, **options: Any
    ) -> Any:
        chains, run = _stage_build(
            control, argon_run, build_dir, cache_dir, options, ["chain"]
        )
        return chains[0], run

    monkeypatch.setattr(melt, "_prepare", prepare)
    return control


@pytest.fixture
def staged_blend(monkeypatch: pytest.MonkeyPatch, argon_run: Any) -> dict[str, Any]:
    """Stand in for the chemistry behind ``build_blend``, as ``staged_melt`` does.

    Each preparation writes, for every polymer, a conformer and a force field
    named after its residue, then the merged force field and the packed cell.
    ``components`` holds the polymers the last preparation was given; the
    other controls are ``staged_melt``'s.
    """
    from openmmpolymer import melt

    control = {**_build_control(), "components": ()}

    def prepare(
        components: Any, build_dir: Path, cache_dir: Path, **options: Any
    ) -> Any:
        control["components"] = components
        names = [component.spec.residue_name for component in components]
        return _stage_build(
            control,
            argon_run,
            build_dir,
            cache_dir,
            options,
            names,
            extra_files=[f"{name}_ff.xml" for name in names],
        )

    monkeypatch.setattr(melt, "_prepare_blend", prepare)
    return control


@pytest.fixture
def no_build(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Stop any melt build where it would begin, with :class:`BuildReached`.

    Returns the chain specs the build was asked for, so that a test can check
    what reached it - or that nothing did.
    """
    from openmmpolymer import melt

    from .helpers import BuildReached

    asked: list[Any] = []

    def build(spec: Any, *args: Any, **kwargs: Any) -> Any:
        asked.append(spec)
        raise BuildReached

    monkeypatch.setattr(melt, "build_chain", build)
    return asked


@pytest.fixture
def manifest_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Count manifest file reads, including readers that parse raw JSON directly."""
    reads: list[Path] = []
    original = Path.read_text

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name == "manifest.json":
            reads.append(path.resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    return reads
