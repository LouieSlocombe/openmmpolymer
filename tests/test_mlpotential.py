"""Running a cell under a TurboMACE model.

A stand-in model - argon's Lennard-Jones, evaluated in TurboMACE's units of eV
and Angstrom - goes through the real ``PythonForce``, the real pickling, the
real stages and the command line, and is held to the same potential written as
an OpenMM ``NonbondedForce``. Only TurboMACE itself is faked. One test runs
a real model through the real engine when both are available: set
``OPENMMPOLYMER_TURBOMACE_MODEL`` to a ``.tmace`` file, MACE-OFF for instance.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from collections import defaultdict
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import openmm as mm
import pytest
from openmm import app, unit

from openmmpolymer import (
    ChainSpec,
    TurboMACEError,
    TurboMACESpec,
    apply_turbomace,
    build_melt,
    prepare_run,
    run_minimise,
    run_npt,
    run_protocol,
    use_turbomace,
)
from openmmpolymer import mlpotential as ml
from openmmpolymer.__main__ import main
from openmmpolymer.forcefield import PolymerForceField
from openmmpolymer.mdsystem import make_barostat
from openmmpolymer.protocols import Protocol, Stage
from openmmpolymer.stress import stress_tensor_bar

from .helpers import argon_cell, bare_simulation, write_crystal

#: argon_system's Lennard-Jones, in the units TurboMACE returns.
SIGMA_A = 3.4
EPSILON_EV = 0.996 / ml.KJ_PER_MOL_PER_EV
CUTOFF_A = 8.0


class LennardJones:
    """Truncated Lennard-Jones under the minimum image: what the engine returns.

    The image is taken in fractional coordinates, because the stress is read
    by shearing the cell.
    """

    def __init__(self, model: Any, atomic_numbers: list[int]) -> None:
        self.atomic_numbers = atomic_numbers

    def __call__(
        self, positions: npt.NDArray[np.float64], box: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        fractional = (positions[:, None, :] - positions[None, :, :]) @ np.linalg.inv(
            box
        )
        delta = (fractional - np.round(fractional)) @ box
        squared = (delta**2).sum(axis=-1)
        np.fill_diagonal(squared, np.inf)
        inside = squared < CUTOFF_A**2
        s6 = np.where(inside, (SIGMA_A**2 / squared) ** 3, 0.0)
        energy = 0.5 * float(np.sum(4.0 * EPSILON_EV * (s6**2 - s6)))
        coefficient = 24.0 * EPSILON_EV * (2.0 * s6**2 - s6) / squared
        return energy, (coefficient[:, :, None] * delta).sum(axis=1)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """A model file, and a TurboMACE whose programs say what ``meta`` holds."""
    path = tmp_path / "argon.tmace"
    path.write_bytes(b"argon model")
    meta: dict[str, Any] = {"atomic_numbers": [18], "r_max": CUTOFF_A}
    turbomace = ModuleType("turbomace")
    polar = ModuleType("turbomace.polar")
    turbomace.Program = SimpleNamespace(  # type: ignore[attr-defined]
        load=lambda name: SimpleNamespace(meta=meta)
    )
    polar.is_polar = lambda program: "polar" in program.meta  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "turbomace", turbomace)
    monkeypatch.setitem(sys.modules, "turbomace.polar", polar)
    monkeypatch.setattr(ml, "version", lambda package: "0.1.0")
    monkeypatch.setattr(
        ml,
        "_loaded_model",
        functools.cache(lambda *description: SimpleNamespace(description=description)),
    )
    monkeypatch.setattr(ml, "_GraphEvaluator", LennardJones)
    return SimpleNamespace(path=path, meta=meta, spec=TurboMACESpec(path))


def _evaluate(
    system: Any, positions_nm: npt.ArrayLike
) -> tuple[float, npt.NDArray[np.float64]]:
    """Energy in kJ/mol and forces in kJ/mol/nm, on the Reference platform."""
    context = mm.Context(
        system,
        mm.VerletIntegrator(0.001),
        mm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(np.asarray(positions_nm) * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
    )


def _without_dispersion_correction(system: Any) -> Any:
    """*system* with the constant the stand-in model leaves out taken out too."""
    copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    for force in copy.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setUseDispersionCorrection(False)
    return copy


def _forces(system: Any) -> list[str]:
    return [type(force).__name__ for force in system.getForces()]


def test_the_ev_conversion_is_openmm_s_own() -> None:
    electron_volt = 1.0 * unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA
    assert electron_volt.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        ml.KJ_PER_MOL_PER_EV, rel=1e-12
    )


def test_the_model_s_energy_and_forces_reach_openmm_in_its_units(
    model: SimpleNamespace,
) -> None:
    """The same potential, in eV and Angstrom through the model, and in OpenMM.

    One atom is moved a whole cell along, which the model must not notice.
    """
    box, system = argon_cell(64, 2.4)
    positions = box.positions_nm + np.random.default_rng(3).normal(0.0, 0.02, (64, 3))
    positions[0, 0] += 2.4
    expected = _evaluate(_without_dispersion_correction(system), positions)

    converted = apply_turbomace(system, box.topology, model.spec)
    energy, forces = _evaluate(converted, positions)

    assert energy == pytest.approx(expected[0], rel=1e-9)
    np.testing.assert_allclose(forces, expected[1], rtol=1e-7, atol=1e-7)


def test_the_stress_is_the_model_s_through_the_flexible_barostat(
    model: SimpleNamespace,
) -> None:
    box, system = argon_cell(64, 2.4)
    positions = box.positions_nm + np.random.default_rng(5).normal(0.0, 0.02, (64, 3))
    stresses = [
        stress_tensor_bar(
            bare_simulation(
                candidate,
                box.topology,
                positions,
                barostat=make_barostat("flexible", 300.0, 1.0, 0, seed=1),
            )
        )
        for candidate in (
            _without_dispersion_correction(system),
            apply_turbomace(system, box.topology, model.spec),
        )
    ]
    np.testing.assert_allclose(stresses[1], stresses[0], rtol=1e-6, atol=1e-6)


def test_only_the_interactions_are_replaced(model: SimpleNamespace) -> None:
    """Masses, constraints and the CM remover stay; chains stay OpenMM molecules."""
    box, system = argon_cell(64, 2.4, atoms_per_molecule=2)
    system.setParticleMass(0, 2.0 * unit.dalton)
    system.addConstraint(2, 3, 0.6)
    model.meta["atomic_numbers"] = [1, 18]

    converted = apply_turbomace(system, box.topology, model.spec)

    assert _forces(system) == ["NonbondedForce", "HarmonicBondForce", "CMMotionRemover"]
    assert _forces(converted) == ["CMMotionRemover", "HarmonicBondForce", "PythonForce"]
    bonds = converted.getForce(1)
    assert bonds.getName() == ml.MOLECULE_BONDS_NAME
    assert bonds.getNumBonds() == 32
    stiffness = unit.kilojoule_per_mole / unit.nanometer**2
    assert {
        bonds.getBondParameters(index)[3].value_in_unit(stiffness)
        for index in range(32)
    } == {0.0}
    force = converted.getForce(2)
    assert force.getName() == ml.TURBOMACE_FORCE_NAME
    assert force.usesPeriodicBoundaryConditions()
    for index in range(system.getNumParticles()):
        assert converted.getParticleMass(index) == system.getParticleMass(index)
    assert converted.getNumConstraints() == 1
    simulation = bare_simulation(converted, box.topology, box.positions_nm)
    assert len(simulation.context.getMolecules()) == 32


def test_the_system_xml_fingerprints_the_model(
    model: SimpleNamespace, tmp_path: Path
) -> None:
    """The same request serialises identically; another model or precision does not."""
    box, system = argon_cell(64, 2.4)

    def serialised(spec: TurboMACESpec) -> str:
        converted = apply_turbomace(system, box.topology, spec)
        return str(mm.XmlSerializer.serialize(converted))

    first = serialised(model.spec)
    assert serialised(TurboMACESpec(model.path)) == first
    assert serialised(TurboMACESpec(model.path, precision="fp64")) != first
    other = tmp_path / "other.tmace"
    other.write_bytes(b"argon model")
    assert serialised(TurboMACESpec(other)) != first
    model.path.write_bytes(b"retrained argon model")
    assert serialised(model.spec) != first

    restored = mm.XmlSerializer.deserialize(first)
    assert mm.XmlSerializer.serialize(restored) == first


@pytest.mark.parametrize("change", ["edit", "delete"])
def test_a_changed_model_file_refuses_to_evaluate(
    model: SimpleNamespace, change: str
) -> None:
    """OpenMM re-raises what the computation raised, message and all."""
    box, system = argon_cell(64, 2.4)
    converted = apply_turbomace(system, box.topology, model.spec)
    if change == "edit":
        model.path.write_bytes(b"retrained argon model")
    else:
        model.path.unlink()
    with pytest.raises(
        mm.OpenMMException, match="changed since" if change == "edit" else "is missing"
    ):
        _evaluate(converted, box.positions_nm)


def test_a_model_is_held_to_the_cell_it_describes(model: SimpleNamespace) -> None:
    box, system = argon_cell(64, 2.4)
    model.meta["atomic_numbers"] = [1, 6]
    with pytest.raises(TurboMACEError, match=r"atomic numbers \[18\]"):
        apply_turbomace(system, box.topology, model.spec)
    model.meta.update(atomic_numbers=[18], polar={})
    with pytest.raises(TurboMACEError, match="polar model"):
        apply_turbomace(system, box.topology, model.spec)


def test_a_system_turbomace_cannot_take_over_is_refused(model: SimpleNamespace) -> None:
    box, system = argon_cell(64, 2.4)
    other, _ = argon_cell(8, 2.4)
    with pytest.raises(ValueError, match="64 atoms but the System has 8"):
        apply_turbomace(argon_cell(8, 2.4)[1], box.topology, model.spec)

    barostat = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    barostat.addForce(make_barostat("isotropic", 300.0, 1.0, 25, seed=1))
    with pytest.raises(ValueError, match="MonteCarloBarostat"):
        apply_turbomace(barostat, box.topology, model.spec)

    virtual = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    virtual.setVirtualSite(2, mm.TwoParticleAverageSite(0, 1, 0.5, 0.5))
    with pytest.raises(ValueError, match="virtual sites"):
        apply_turbomace(virtual, box.topology, model.spec)

    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    for atom in other.topology.atoms():
        topology.addAtom(atom.name, None, residue)
    with pytest.raises(ValueError, match="no element"):
        apply_turbomace(argon_cell(8, 2.4)[1], topology, model.spec)


def test_a_spec_refuses_a_missing_model_or_an_unknown_precision(
    model: SimpleNamespace,
) -> None:
    with pytest.raises(FileNotFoundError, match="turbomace-export"):
        TurboMACESpec("missing.tmace")
    with pytest.raises(ValueError, match="precision"):
        TurboMACESpec(model.path, precision="fp16")


def test_a_missing_turbomace_says_how_to_install_it(
    model: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    box, system = argon_cell(64, 2.4)
    monkeypatch.setitem(sys.modules, "turbomace", None)
    with pytest.raises(TurboMACEError, match="pip install"):
        apply_turbomace(system, box.topology, model.spec)

    def absent(package: str) -> str:
        raise PackageNotFoundError(package)

    monkeypatch.setattr(ml, "version", absent)
    with pytest.raises(TurboMACEError, match="not installed"):
        ml.turbomace_version()


def test_use_turbomace_changes_the_system_and_nothing_else(
    model: SimpleNamespace, argon_run: Any
) -> None:
    converted = use_turbomace(argon_run, model.spec)
    assert converted.system_xml != argon_run.system_xml
    assert converted.box is argon_run.box
    assert converted.spec == argon_run.spec
    assert converted.total_mass_g_mol == argon_run.total_mass_g_mol

    box, system = argon_cell(64, 2.4)
    prepared = prepare_run(
        box,
        PolymerForceField("unused.xml", (), "AR", "smirnoff"),
        platform="CPU",
        seed=11,
        system=system,
        potential=model.spec,
    )
    assert prepared.system_xml == converted.system_xml


def test_the_stages_run_and_resume_under_turbomace(
    model: SimpleNamespace, argon_run: Any
) -> None:
    """A barostat, a reloaded state and a resumed manifest, all through the model."""
    protocol = Protocol(
        name="turbomace",
        stages=(
            Stage("00_minimise", "minimise"),
            Stage(
                "01_npt",
                "npt",
                {
                    "temperature_k": 100.0,
                    "duration_ps": 0.4,
                    "barostat": "flexible",
                    "barostat_frequency": 5,
                },
            ),
        ),
    )
    first = run_protocol(protocol, use_turbomace(argon_run, model.spec), "run")
    assert first.skipped == ()
    density = first.results[-1].mean_density_g_cm3
    assert density is not None
    assert density != pytest.approx(first.results[0].mean_density_g_cm3, rel=1e-6)

    again = run_protocol(protocol, use_turbomace(argon_run, model.spec), "run")
    assert again.skipped == ("00_minimise", "01_npt")


def test_a_single_stage_runs_from_a_classical_state(
    model: SimpleNamespace, argon_run: Any
) -> None:
    classical = run_minimise(argon_run, "classical", max_iterations=50)
    result = run_npt(
        use_turbomace(argon_run, model.spec),
        "mace",
        temperature_k=100.0,
        duration_ps=0.2,
        state_in=classical.final_state,
    )
    assert result.steps > 0


def test_build_melt_hands_the_model_to_the_build(
    model: SimpleNamespace, staged_melt: dict[str, Any]
) -> None:
    build_melt(
        ChainSpec(monomer_smiles="[*]CC[*]", degree_of_polymerization=2),
        1,
        "melt",
        target_density_g_cm3=0.85,
        potential=model.spec,
    )
    assert staged_melt["options"]["potential"] == model.spec


def test_the_command_line_builds_under_the_model_and_records_it(
    model: SimpleNamespace, staged_melt: dict[str, Any]
) -> None:
    arguments = ["[*]CC[*]", "-o", "run", "--dry-run", "--turbomace", str(model.path)]
    assert main([*arguments, "--turbomace-precision", "fp64"]) == 0
    assert staged_melt["options"]["potential"] == TurboMACESpec(
        str(model.path), precision="fp64"
    )
    recorded = json.loads(Path("run/build_request.json").read_text())
    assert recorded["turbomace"] == str(model.path)
    assert recorded["turbomace_precision"] == "fp64"
    assert recorded["runtime_versions"]["turbomace"] == "0.1.0"
    # The model is part of what was asked for.
    with pytest.raises(SystemExit, match="2"):
        main(arguments)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["[*]CC[*]", "--turbomace-precision", "fp32"], "requires --turbomace"),
        (["--analyse", "run", "--turbomace", "MODEL"], "--analyse runs none"),
        (["[*]CC[*]", "--turbomace", "missing.tmace"], "does not exist"),
        (["[*]CC[*]", "--turbomace", "MODEL", "--turbomace-precision", "fp8"], "fp8"),
    ],
)
def test_the_command_line_refuses_turbomace_flags_that_cannot_apply(
    model: SimpleNamespace,
    no_build: list[Any],
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    resolved = [str(model.path) if item == "MODEL" else item for item in arguments]
    with pytest.raises(SystemExit, match="2"):
        main(resolved)
    assert message in capsys.readouterr().err
    assert no_build == []


def test_the_command_line_refuses_a_model_without_turbomace(
    model: SimpleNamespace,
    no_build: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def absent(package: str) -> str:
        raise PackageNotFoundError(package)

    monkeypatch.setattr(ml, "version", absent)
    with pytest.raises(SystemExit, match="2"):
        main(["[*]CC[*]", "--turbomace", str(model.path)])
    assert "TurboMACE is not installed" in capsys.readouterr().err
    assert no_build == []
    assert not Path("run").exists()


def test_a_crystal_is_heated_under_the_model_too(
    model: SimpleNamespace,
    argon_box: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmmpolymer import __main__ as cli

    converted: list[Any] = []

    def recording(run: Any, spec: TurboMACESpec) -> Any:
        converted.append(spec)
        return use_turbomace(run, spec)

    monkeypatch.setattr(cli, "use_turbomace", recording)
    pdb, xml = write_crystal(*argon_box, Path("."))
    crystal = ["--crystal-pdb", str(pdb), "--system-xml", str(xml)]
    assert (
        main(
            ["--protocol", "tm", *crystal, "--turbomace", str(model.path), "--dry-run"]
        )
        == 0
    )
    assert converted == [TurboMACESpec(str(model.path))]


def test_loaded_models_are_let_go_at_exit(model: SimpleNamespace) -> None:
    """Before TurboMACE's extension module is torn down, and reloaded if needed."""
    box, system = argon_cell(64, 2.4)
    simulation = bare_simulation(
        apply_turbomace(system, box.topology, model.spec),
        box.topology,
        box.positions_nm,
    )
    before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert any(computation._evaluate is not None for computation in ml._LOADED)

    ml._release_models()

    assert all(computation._evaluate is None for computation in ml._LOADED)
    assert simulation.context.getState(getEnergy=True).getPotentialEnergy() == before


def test_a_build_says_which_model_it_runs_under(model: SimpleNamespace) -> None:
    from openmmpolymer import melt

    lines: list[str] = []
    melt._say_potential(None, lines.append)
    melt._say_potential(model.spec, lines.append)
    assert lines == [f"potential: TurboMACE {model.path} (fp32)"]


#: A real model for the test below, which is skipped without one.
REAL_MODEL = os.environ.get("OPENMMPOLYMER_TURBOMACE_MODEL")


@pytest.mark.skipif(
    REAL_MODEL is None, reason="set OPENMMPOLYMER_TURBOMACE_MODEL to a .tmace file"
)
@pytest.mark.filterwarnings("ignore:radial spline stored-table sampled error")
def test_a_real_model_reaches_openmm_unchanged() -> None:
    """MACE through OpenMM gives what TurboMACE gives on its own, for methane gas."""
    turbomace = pytest.importorskip("turbomace")
    ase = pytest.importorskip("ase")
    assert REAL_MODEL is not None

    topology = app.Topology()
    chain = topology.addChain()
    carbon, hydrogen = (app.Element.getBySymbol(symbol) for symbol in ("C", "H"))
    tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * 0.063
    centres = np.random.default_rng(2).uniform(0.0, 1.6, (27, 3))
    positions = []
    system = mm.System()
    for centre in centres:
        residue = topology.addResidue("MET", chain)
        atoms = [topology.addAtom("C", carbon, residue)]
        atoms += [topology.addAtom(f"H{n}", hydrogen, residue) for n in range(4)]
        for atom in atoms[1:]:
            topology.addBond(atoms[0], atom)
        positions += [centre, *(centre + tetrahedron)]
        for atom in atoms:
            system.addParticle(atom.element.mass)
    vectors = [mm.Vec3(1.6, 0, 0), mm.Vec3(0, 1.6, 0), mm.Vec3(0, 0, 1.6)]
    topology.setPeriodicBoxVectors(vectors * unit.nanometer)
    system.setDefaultPeriodicBoxVectors(*vectors)

    converted = apply_turbomace(system, topology, TurboMACESpec(REAL_MODEL, "fp64"))
    energy, forces = _evaluate(converted, positions)

    loaded = turbomace.load(REAL_MODEL, precision="fp64")
    atoms = ase.Atoms(
        numbers=[atom.element.atomic_number for atom in topology.atoms()],
        positions=np.asarray(positions) * 10.0,
        cell=np.eye(3) * 16.0,
        pbc=True,
    )
    direct = loaded.compute(atoms, stress=False)
    assert energy == pytest.approx(direct["energy"] * ml.KJ_PER_MOL_PER_EV, rel=1e-10)
    np.testing.assert_allclose(
        forces, direct["forces"] * ml.KJ_PER_MOL_PER_EV * 10.0, rtol=1e-6, atol=1e-4
    )


EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "mace_off_melt.py"


def test_the_mace_off_example_hands_the_classical_state_to_mace_off(
    model: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Classical build and equilibration, then MACE-OFF from its final state."""
    import importlib.util
    from unittest.mock import create_autospec

    import openmmpolymer

    chain, classical, mace = (
        SimpleNamespace(backbone=(0, 1), n_atoms=8),
        object(),
        object(),
    )
    records = {"mean_density_g_cm3": 0.77, "wall_seconds": 86.4}
    results = {
        "build_melt": (chain, classical),
        "run_protocol": SimpleNamespace(final_state="classical/final.xml"),
        "use_turbomace": mace,
    }
    mocks = {}
    for name, result in results.items():
        mocks[name] = create_autospec(getattr(openmmpolymer, name), return_value=result)
        monkeypatch.setattr(openmmpolymer, name, mocks[name])

    def load(directory: Any) -> SimpleNamespace:
        return SimpleNamespace(stages=defaultdict(lambda: records))

    monkeypatch.setattr(openmmpolymer.RunManifest, "load", staticmethod(load))
    spec = importlib.util.spec_from_file_location("mace_off_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main(["--model", str(model.path), "--output", "out", "--production-ps", "1"])

    first, second = mocks["run_protocol"].call_args_list
    assert first.args[1:] == (classical, Path("out/classical"))
    assert second.args[1:] == (mace, Path("out/mace-off"))
    assert second.kwargs["state_in"] == "classical/final.xml"
    assert mocks["use_turbomace"].call_args.args == (classical, model.spec)
    output = capsys.readouterr().out
    assert "MACE-OFF:    0.7700 g/cm3, mean of 1 ps" in output
    assert "ran at 1.0 ns/day" in output


def test_the_mace_off_example_needs_a_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import runpy

    monkeypatch.setattr(sys, "argv", [str(EXAMPLE)])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert raised.value.code == 2
    assert "--model" in capsys.readouterr().err
