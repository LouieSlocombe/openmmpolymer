"""Tests for assembling a packed cell into an OpenMM System.

Several of these exist because the behaviour they pin was measured rather than
assumed, and each one is silent when it goes wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.mdsystem import (
    PackedBox,
    SystemAssemblyError,
    SystemSpec,
    assemble_box,
    barostat_kind,
    build_system,
    check_box,
    check_target_density,
    check_timestep,
    find_barostat,
    make_barostat,
    max_timestep_fs,
    minimum_mass_g_mol,
    replicate_topology,
    select_platform,
)
from openmmpolymer.packing import PackedComponent

from .helpers import build_dimer_pdb


def test_spec_rejects_a_switch_at_or_beyond_the_cutoff() -> None:
    """A switching function has to start inside the cutoff."""
    with pytest.raises(ValueError, match="must be below"):
        SystemSpec(nonbonded_cutoff_nm=1.0, switch_distance_nm=1.0)


def test_spec_rejects_an_unknown_constraint_setting() -> None:
    """A typo here would silently change the dynamics."""
    with pytest.raises(ValueError, match="constraints"):
        SystemSpec(constraints="hbond")


def test_spec_refuses_per_atom_scaling_with_constraints() -> None:
    """Measured: a polyethylene melt run this way expands under a kilobar.

    Per-atom volume moves scale atoms through their constraints, so the energy
    the barostat's Metropolis test sees is not the energy of any real
    configuration.
    """
    with pytest.raises(ValueError, match="per-atom volume move"):
        SystemSpec(scale_molecules_as_rigid=False, constraints="hbonds")


def test_per_atom_scaling_is_allowed_without_constraints() -> None:
    """It is the combination that is wrong, not the setting."""
    spec = SystemSpec(scale_molecules_as_rigid=False, constraints="none")
    assert spec.scale_molecules_as_rigid is False


def test_the_default_is_rigid_molecule_scaling() -> None:
    """OpenMM's own default, and the one that reproduces a melt density."""
    assert SystemSpec().scale_molecules_as_rigid is True


def test_the_default_switch_distance_is_none() -> None:
    """GAFF and OpenFF are fitted against a hard cutoff plus the correction."""
    assert SystemSpec().switch_distance_nm is None


def test_the_dispersion_correction_is_on_by_default() -> None:
    """It is worth one to three per cent on a melt density."""
    assert SystemSpec().use_dispersion_correction is True


def test_check_box_passes_a_cell_comfortably_larger_than_the_cutoff() -> None:
    """Twice the cutoff is OpenMM's limit; the margin above it is for the barostat."""
    check_box((4.0, 4.0, 4.0), SystemSpec(nonbonded_cutoff_nm=1.2))


def test_check_box_refuses_a_cell_below_the_margin() -> None:
    """OpenMM would raise on this, and mid-run rather than at startup."""
    with pytest.raises(SystemAssemblyError, match="shortest edge"):
        check_box((2.5, 4.0, 4.0), SystemSpec(nonbonded_cutoff_nm=1.2))


def test_check_target_density_uses_the_compressed_cell_not_the_packed_one() -> None:
    """The cell that has to satisfy the cutoff is the dense one.

    Checking only the loose packing cell passes a run that dies hours later,
    part-way through compression.
    """
    spec = SystemSpec(nonbonded_cutoff_nm=1.2)
    check_box((4.5, 4.5, 4.5), spec)
    with pytest.raises(SystemAssemblyError, match="compresses to"):
        check_target_density([20], [563.1], 0.85, spec)


def test_check_target_density_says_how_much_more_material_is_needed() -> None:
    """A message that names the fix is worth more than one that names the fault."""
    with pytest.raises(SystemAssemblyError, match="times as many chains"):
        check_target_density([20], [563.1], 0.85, SystemSpec())


def test_check_target_density_returns_the_compressed_edge() -> None:
    """The caller usually wants the number as well as the check."""
    edge = check_target_density([200], [563.1], 0.85, SystemSpec())
    assert edge == pytest.approx(6.03, abs=0.05)


def test_minimum_mass_agrees_with_the_density_check() -> None:
    """The two are the same statement, so they must not drift apart."""
    spec = SystemSpec()
    needed = minimum_mass_g_mol(0.85, spec)
    check_target_density([1], [needed * 1.001], 0.85, spec)
    with pytest.raises(SystemAssemblyError):
        check_target_density([1], [needed * 0.999], 0.85, spec)


@pytest.mark.parametrize(
    ("hydrogen_mass", "constraints", "expected"),
    [
        (None, "hbonds", 2.0),
        (1.5, "hbonds", 3.0),
        (4.0, "hbonds", 4.0),
        (None, "none", 1.0),
    ],
)
def test_timestep_limits_at_the_reference_temperature(
    hydrogen_mass: float | None, constraints: str, expected: float
) -> None:
    """The pairing between repartitioned mass and timestep, as shipped."""
    spec = SystemSpec(hydrogen_mass_amu=hydrogen_mass, constraints=constraints)
    assert max_timestep_fs(300.0, spec) == pytest.approx(expected)


def test_the_timestep_limit_derates_with_temperature() -> None:
    """The fastest velocities scale as the square root of the temperature."""
    spec = SystemSpec()
    assert max_timestep_fs(1200.0, spec) == pytest.approx(1.0)
    assert max_timestep_fs(600.0, spec) < max_timestep_fs(300.0, spec)


def test_check_timestep_refuses_a_melt_step_that_was_fine_when_cold() -> None:
    """2 fs at 300 K is ordinary; at 600 K it is the usual source of a NaN."""
    spec = SystemSpec()
    check_timestep(2.0, 300.0, spec)
    with pytest.raises(ValueError, match="too long for 600 K"):
        check_timestep(2.0, 600.0, spec)


def test_replicate_topology_repeats_the_bonds_as_well_as_the_atoms(
    tmp_path: Path,
) -> None:
    """packmol writes no CONECT records, so the bonds have to come from here."""
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    topology, positions = replicate_topology([PackedComponent(source, 5)])
    assert topology.getNumAtoms() == 10
    assert topology.getNumResidues() == 5
    assert sum(1 for _ in topology.bonds()) == 5
    assert len(positions) == 10


def _packed_pdb(
    path: Path,
    n_molecules: int,
    elements: str = "CC",
    residue_name: str = "DIM",
) -> str:
    """Write a packmol-shaped output: coordinates, and no CONECT records.

    Written through OpenMM's own writer rather than by formatting columns by
    hand, because a PDB is fixed-width and getting it wrong by one column
    produces a file that looks fine and does not parse.
    """
    from openmm import app, unit

    topology = app.Topology()
    chain = topology.addChain()
    positions = []
    for molecule in range(n_molecules):
        residue = topology.addResidue(residue_name, chain)
        for index, symbol in enumerate(elements):
            topology.addAtom(
                f"{symbol}{index + 1}", app.Element.getBySymbol(symbol), residue
            )
            positions.append([molecule * 1.0 + index * 0.153, 0.0, 0.0])

    with path.open("w") as handle:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, handle)
    # packmol writes no connectivity, and neither does this.
    path.write_text(
        "".join(
            line
            for line in path.read_text().splitlines(keepends=True)
            if not line.startswith("CONECT")
        )
    )
    return str(path)


def test_assemble_box_takes_positions_from_packmol_and_bonds_from_the_chain(
    tmp_path: Path,
) -> None:
    """The whole point of the module."""
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 3)
    box = assemble_box([PackedComponent(source, 3)], packed, (3.0, 3.0, 3.0))

    assert box.n_molecules == 3
    assert box.topology.getNumAtoms() == 6
    assert sum(1 for _ in box.topology.bonds()) == 3
    assert box.positions_nm[2][0] == pytest.approx(1.0, abs=1e-3)


def test_assemble_box_refuses_a_packed_file_with_the_wrong_atom_count(
    tmp_path: Path,
) -> None:
    """Silently mapping the wrong coordinates would not raise anywhere later."""
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 2)
    with pytest.raises(SystemAssemblyError, match="not in the same order"):
        assemble_box([PackedComponent(source, 3)], packed, (3.0, 3.0, 3.0))


def test_a_packed_file_missing_atoms_is_told_why_that_happens(
    tmp_path: Path,
) -> None:
    """Atoms going missing has one cause in practice, and it is worth naming.

    Two molecules sharing a chain identifier and a residue number read as one
    residue described twice, and the PDB parser discards the second copy.
    Measured on a real 45-conformer pack without ``resnumbers 3``: 1406 atoms
    came back instead of 1710, eight whole molecules gone, after one warning.
    """
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 2)
    with pytest.raises(SystemAssemblyError, match="dropped the duplicates"):
        assemble_box([PackedComponent(source, 3)], packed, (3.0, 3.0, 3.0))


def test_a_packed_file_with_extra_atoms_is_not_blamed_on_numbering(
    tmp_path: Path,
) -> None:
    """The duplicate-numbering explanation only fits atoms going missing."""
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 4)
    with pytest.raises(SystemAssemblyError) as raised:
        assemble_box([PackedComponent(source, 3)], packed, (3.0, 3.0, 3.0))
    assert "dropped the duplicates" not in str(raised.value)


def test_assemble_box_refuses_a_packed_file_with_the_elements_in_a_new_order(
    tmp_path: Path,
) -> None:
    """A count check alone would pass this, and every parameter would be wrong."""
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 3, elements="CN")
    with pytest.raises(SystemAssemblyError, match="different order"):
        assemble_box([PackedComponent(source, 3)], packed, (3.0, 3.0, 3.0))


def test_assemble_box_sets_the_periodic_box_vectors(tmp_path: Path) -> None:
    """Without them a System is not periodic, whatever the nonbonded method."""
    from openmm import unit

    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 2)
    box = assemble_box([PackedComponent(source, 2)], packed, (3.5, 3.5, 3.5))
    vectors = box.topology.getPeriodicBoxVectors()
    assert vectors[0][0].value_in_unit(unit.nanometer) == pytest.approx(3.5)


def test_build_system_uses_the_force_field_and_the_cell(
    tmp_path: Path, dimer_forcefield: Any
) -> None:
    """The real ForceField path, on a residue small enough to read."""
    import openmm as mm

    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 4)
    box = assemble_box([PackedComponent(source, 4)], packed, (4.0, 4.0, 4.0))
    system = build_system(box, dimer_forcefield, SystemSpec(constraints="none"))

    assert system.getNumParticles() == 8
    assert system.usesPeriodicBoundaryConditions()
    nonbonded = next(
        force for force in system.getForces() if isinstance(force, mm.NonbondedForce)
    )
    assert nonbonded.getNonbondedMethod() == mm.NonbondedForce.PME


def test_build_system_keeps_the_dispersion_correction_on(
    tmp_path: Path, dimer_forcefield: Any
) -> None:
    """Passing None instead would have turned it off, silently."""
    import openmm as mm

    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 4)
    box = assemble_box([PackedComponent(source, 4)], packed, (4.0, 4.0, 4.0))

    for wanted in (True, False):
        system = build_system(
            box,
            dimer_forcefield,
            SystemSpec(constraints="none", use_dispersion_correction=wanted),
        )
        nonbonded = next(
            force
            for force in system.getForces()
            if isinstance(force, mm.NonbondedForce)
        )
        assert nonbonded.getUseDispersionCorrection() is wanted


def test_build_system_refuses_a_cell_too_small_for_the_cutoff(
    tmp_path: Path, dimer_forcefield: Any
) -> None:
    """Checked before anything expensive happens."""
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 2)
    box = assemble_box([PackedComponent(source, 2)], packed, (2.0, 2.0, 2.0))
    with pytest.raises(SystemAssemblyError, match="shortest edge"):
        build_system(box, dimer_forcefield, SystemSpec())


def test_make_barostat_is_seeded_and_rigid_by_default() -> None:
    """A seed of zero would make OpenMM choose its own and lose the run."""
    barostat = make_barostat("isotropic", 300.0, 1.0, 25, 4242)
    assert barostat.getRandomNumberSeed() == 4242
    # 8.3's isotropic barostat is always rigid, with no configurable switch.
    assert getattr(barostat, "getScaleMoleculesAsRigid", lambda: True)() is True


@pytest.mark.parametrize("kind", ["isotropic", "anisotropic"])
def test_older_barostat_defaults_to_rigid_and_refuses_atomic_scaling(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A missing optional API must neither break defaults nor silently change physics."""
    import openmm as mm

    cls = (
        mm.MonteCarloBarostat
        if kind == "isotropic"
        else mm.MonteCarloAnisotropicBarostat
    )
    monkeypatch.delattr(cls, "setScaleMoleculesAsRigid", raising=False)
    assert make_barostat(kind, 300.0, 1.0, 25, 4242).getRandomNumberSeed() == 4242
    with pytest.raises(SystemAssemblyError, match="scale_molecules_as_rigid=False"):
        make_barostat(kind, 300.0, 1.0, 25, 4242, scale_molecules_as_rigid=False)


def test_make_barostat_refuses_a_zero_seed() -> None:
    """OpenMM reads zero as 'choose one', which is not reproducible."""
    with pytest.raises(ValueError, match="irreproducible"):
        make_barostat("isotropic", 300.0, 1.0, 25, 0)


def test_make_barostat_builds_the_anisotropic_kind_too() -> None:
    """A different Force class with different parameter names."""
    import openmm as mm

    barostat = make_barostat("anisotropic", 300.0, 1.0, 25, 7)
    assert isinstance(barostat, mm.MonteCarloAnisotropicBarostat)


def test_barostat_kind_reports_what_is_attached(argon_box: Any) -> None:
    """None before one is added, and its kind after."""
    _, system = argon_box
    assert barostat_kind(system) is None
    system.addForce(make_barostat("isotropic", 300.0, 1.0, 25, 3))
    assert barostat_kind(system) == "isotropic"


def test_two_barostats_are_refused(argon_box: Any) -> None:
    """OpenMM applies both without complaining, which is not a pressure."""
    _, system = argon_box
    system.addForce(make_barostat("isotropic", 300.0, 1.0, 25, 3))
    system.addForce(make_barostat("isotropic", 300.0, 1.0, 25, 4))
    with pytest.raises(SystemAssemblyError, match="2 barostats"):
        barostat_kind(system)


def test_select_platform_falls_back_through_the_preference_order() -> None:
    """Whatever the machine has, something is always chosen."""
    platform, properties = select_platform()
    assert platform.getName() in {"CUDA", "OpenCL", "CPU", "Reference"}
    if platform.getName() in {"CUDA", "OpenCL"}:
        assert properties["Precision"] == "mixed"


def test_select_platform_takes_a_name() -> None:
    """The deterministic reference platform is always there."""
    platform, properties = select_platform("Reference")
    assert platform.getName() == "Reference"
    assert properties == {}


def test_select_platform_names_what_is_available_when_asked_for_nonsense() -> None:
    """A typo should not read as 'nothing works'."""
    with pytest.raises(SystemAssemblyError, match="this build has"):
        select_platform("CUDA9000")


def test_packed_box_positions_carry_units(argon_box: Any) -> None:
    """OpenMM calls want a Quantity; the rest of the package wants floats."""
    from openmm import unit

    box, _ = argon_box
    assert box.positions.unit == unit.nanometer
    assert isinstance(box.positions_nm, np.ndarray)


def test_prepare_box_is_a_no_op_without_virtual_sites(
    argon_box: Any, dimer_forcefield: Any
) -> None:
    """Most force fields declare none, and then there is nothing to add."""
    from openmmpolymer.mdsystem import prepare_box

    box, _ = argon_box
    assert prepare_box(box, dimer_forcefield) is box


def test_packed_box_is_a_plain_dataclass() -> None:
    """It travels into the manifest, so it has to stay simple."""
    box = PackedBox(
        topology=None,
        positions_nm=np.zeros((1, 3)),
        box_nm=(1.0, 1.0, 1.0),
        n_molecules=1,
    )
    assert box.box_nm == (1.0, 1.0, 1.0)


def test_a_small_hydrogen_mass_does_not_earn_a_longer_step() -> None:
    """Repartitioning below 1.5 amu buys nothing, so the limit stays at 2 fs."""
    spec = SystemSpec(hydrogen_mass_amu=1.2)
    assert max_timestep_fs(300.0, spec) == pytest.approx(2.0)


def test_hydrogen_mass_and_the_switch_reach_create_system(
    tmp_path: Path, dimer_forcefield: Any
) -> None:
    """Both are optional kwargs, and both have to actually arrive."""
    import openmm as mm
    from openmm import unit

    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 4)
    box = assemble_box([PackedComponent(source, 4)], packed, (4.0, 4.0, 4.0))
    system = build_system(
        box,
        dimer_forcefield,
        SystemSpec(
            constraints="none",
            switch_distance_nm=1.0,
            hydrogen_mass_amu=1.5,
        ),
    )
    nonbonded = next(
        force for force in system.getForces() if isinstance(force, mm.NonbondedForce)
    )
    assert nonbonded.getUseSwitchingFunction()
    assert nonbonded.getSwitchingDistance().value_in_unit(
        unit.nanometer
    ) == pytest.approx(1.0)


def test_naming_residue_templates_can_be_turned_off(
    tmp_path: Path, dimer_forcefield: Any
) -> None:
    """It is a shortcut past the graph search, not a requirement."""
    source = build_dimer_pdb(tmp_path / "dimer.pdb")
    packed = _packed_pdb(tmp_path / "packed.pdb", 4)
    box = assemble_box([PackedComponent(source, 4)], packed, (4.0, 4.0, 4.0))
    system = build_system(
        box,
        dimer_forcefield,
        SystemSpec(constraints="none"),
        use_residue_templates=False,
    )
    assert system.getNumParticles() == 8


def test_a_residue_the_force_field_does_not_cover_is_reported(
    tmp_path: Path, dimer_forcefield: Any
) -> None:
    """The message names what failed rather than repeating OpenMM's internals."""
    from openmm import app, unit

    topology = app.Topology()
    residue = topology.addResidue("NIT", topology.addChain())
    nitrogen = app.Element.getBySymbol("N")
    first = topology.addAtom("N1", nitrogen, residue)
    second = topology.addAtom("N2", nitrogen, residue)
    topology.addBond(first, second)
    source = tmp_path / "nitrogen.pdb"
    with source.open("w") as handle:
        app.PDBFile.writeFile(
            topology, [[0.0, 0.0, 0.0], [0.11, 0.0, 0.0]] * unit.nanometer, handle
        )

    box = assemble_box(
        [PackedComponent(str(source), 4)],
        _packed_pdb(tmp_path / "packed.pdb", 4, elements="NN", residue_name="NIT"),
        (4.0, 4.0, 4.0),
    )
    with pytest.raises(SystemAssemblyError, match="would not build"):
        build_system(
            box,
            dimer_forcefield,
            SystemSpec(constraints="none"),
            use_residue_templates=False,
        )


def test_the_reference_platform_is_always_usable() -> None:
    """The floor of the preference order has to work, or nothing does."""
    from openmmpolymer.mdsystem import platform_is_usable

    assert platform_is_usable("Reference")


def test_auto_selection_skips_a_platform_that_cannot_build_a_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Being listed is not the same as working.

    A CUDA build compiled against a newer toolkit than the driver is listed
    and then fails at the first Context. Without this the failure arrives at
    the start of the first stage instead of at platform selection.
    """
    from openmmpolymer import mdsystem

    tried: list[str] = []

    def only_cpu_works(name: str) -> bool:
        tried.append(name)
        return name == "CPU"

    monkeypatch.setattr(mdsystem, "platform_is_usable", only_cpu_works)
    platform, properties = mdsystem.select_platform()
    assert platform.getName() == "CPU"
    assert properties == {}
    assert tried[-1] == "CPU"


def test_a_named_platform_is_used_without_being_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asked for one explicitly, the caller should see its real failure."""
    from openmmpolymer import mdsystem

    def never_usable(name: str) -> bool:
        raise AssertionError("a named platform should not be probed")

    monkeypatch.setattr(mdsystem, "platform_is_usable", never_usable)
    platform, _ = mdsystem.select_platform("Reference")
    assert platform.getName() == "Reference"


# --------------------------------------------------------------------------
# Per-axis and flexible barostats
# --------------------------------------------------------------------------


def test_an_anisotropic_barostat_takes_a_pressure_per_axis() -> None:
    """Which is what makes a uniaxial load rather than a hydrostatic one."""
    import openmm as mm
    from openmm import unit

    barostat = make_barostat(
        "anisotropic", 300.0, 1.0, 25, 7, pressures_bar=(1.0, 1.0, -50.0)
    )
    pressures = barostat.getDefaultPressure().value_in_unit(unit.bar)
    assert (pressures[0], pressures[1], pressures[2]) == pytest.approx(
        (1.0, 1.0, -50.0)
    )
    assert isinstance(barostat, mm.MonteCarloAnisotropicBarostat)


def test_an_axis_can_be_frozen_while_the_others_move() -> None:
    """The uniaxial-strain ensemble, in one flag."""
    barostat = make_barostat(
        "anisotropic", 300.0, 1.0, 25, 7, scale_axes=(True, True, False)
    )
    assert (barostat.getScaleX(), barostat.getScaleY(), barostat.getScaleZ()) == (
        True,
        True,
        False,
    )


def test_a_barostat_that_cannot_move_anything_is_refused() -> None:
    """It is not an ensemble; a frequency of zero is how to ask for a probe."""
    with pytest.raises(ValueError, match="frequency=0"):
        make_barostat(
            "anisotropic", 300.0, 1.0, 25, 7, scale_axes=(False, False, False)
        )


def test_per_axis_settings_are_refused_by_the_barostats_that_have_no_axes() -> None:
    """Silently ignoring them would give a run that is not the one asked for."""
    for kind in ("isotropic", "flexible"):
        with pytest.raises(ValueError, match="anisotropic"):
            make_barostat(kind, 300.0, 1.0, 25, 7, pressures_bar=(1.0, 1.0, 2.0))
        with pytest.raises(ValueError, match="anisotropic"):
            make_barostat(kind, 300.0, 1.0, 25, 7, scale_axes=(True, True, False))


def test_a_flexible_barostat_is_found_rather_than_invisible() -> None:
    """It is not a MonteCarloBarostat subclass, so an isinstance chain misses it.

    Which would also mean the guard against a System carrying two barostats
    could not see one of them.
    """
    import openmm as mm

    for kind, expected in (
        ("isotropic", mm.MonteCarloBarostat),
        ("anisotropic", mm.MonteCarloAnisotropicBarostat),
        ("flexible", mm.MonteCarloFlexibleBarostat),
    ):
        system = mm.System()
        system.addForce(make_barostat(kind, 300.0, 1.0, 25, 7))
        assert barostat_kind(system) == kind
        found = find_barostat(system)
        assert found is not None
        assert isinstance(found[1], expected)
    assert not issubclass(mm.MonteCarloFlexibleBarostat, mm.MonteCarloBarostat)


def test_two_barostats_are_still_refused_when_one_is_flexible() -> None:
    """The guard has to see every kind, or it only guards some of them."""
    import openmm as mm

    system = mm.System()
    system.addForce(make_barostat("anisotropic", 300.0, 1.0, 25, 7))
    system.addForce(make_barostat("flexible", 300.0, 1.0, 0, 9))
    with pytest.raises(SystemAssemblyError, match="2 barostats"):
        find_barostat(system)


def test_a_barostat_may_be_built_at_zero_frequency() -> None:
    """A probe attached only so that its pressure readout can be called."""
    barostat = make_barostat("flexible", 300.0, 1.0, 0, 7)
    assert barostat.getFrequency() == 0
