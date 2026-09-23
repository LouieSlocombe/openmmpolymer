"""Tests for reading a stress tensor and for applying a strain.

The readout is OpenMM's, so what is tested here is the adapter around it -
the sign, the units, where each component lands, and which barostat can
answer at all - plus the two exact cases that pin the physics with no
statistics in them: a force-free gas, where the pressure is a closed-form
kinetic sum, and a gas of constrained rotors, where the molecular and atomic
virials differ by a factor anyone can write down.

The strain half is tested by invariance rather than by value. An affine map
of a periodic cell commutes with the choice of image, so re-imaging a
molecule and then straining must give bit-identical energy - and does, but
only if the box is strained too. That single assertion covers the whole
contract between these functions and their callers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from openmmpolymer.mdsystem import make_barostat
from openmmpolymer.stress import (
    StressError,
    affine_scale,
    affine_shear,
    find_barostat,
    molecule_groups,
    pressure_bar,
    pressure_tensor_bar,
    shear_box_vectors,
    stress_tensor_bar,
    tensile_stress_bar,
)

from .helpers import argon_system, ideal_gas_system, rigid_rotor_system

BOLTZMANN_KJ_PER_K = 0.00831446261815324


@pytest.fixture(params=[False, True], ids=["preferred", "openmm83-fallback"])
def stress_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the stable-release fallback even when a newer API is installed."""
    import openmm as mm

    if request.param:
        monkeypatch.delattr(
            mm.MonteCarloFlexibleBarostat, "computeStressTensor", raising=False
        )


def _simulation(
    system: Any, topology: Any, positions: np.ndarray, kind: str, **barostat: Any
) -> Any:
    """A Simulation on the Reference platform with one barostat attached."""
    import openmm as mm
    from openmm import app, unit

    system.addForce(make_barostat(kind, 300.0, 1.0, 0, 7, **barostat))
    simulation = app.Simulation(
        topology,
        system,
        mm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
        ),
        mm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(positions * unit.nanometer)
    return simulation


# --------------------------------------------------------------------------
# The exact cases
# --------------------------------------------------------------------------


def test_the_kinetic_pressure_of_a_force_free_gas_is_exact() -> None:
    """No statistics anywhere: P_aa is a closed-form sum over the velocities.

    If the unit conversion, the volume or the axis order were wrong, this is
    where it shows, and it shows to every digit rather than within an error
    bar.
    """
    from openmm import unit

    n_atoms, box_nm, mass_amu = 64, 3.0, 40.0
    system, topology, positions = ideal_gas_system(n_atoms, box_nm, mass_amu=mass_amu)
    simulation = _simulation(system, topology, positions, "anisotropic")
    velocities = np.random.default_rng(0).normal(0.0, 0.3, size=(n_atoms, 3))
    simulation.context.setVelocities(velocities * unit.nanometer / unit.picosecond)

    assert simulation.context.getState(
        getEnergy=True
    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == pytest.approx(0.0)

    # sum(m v^2) / V, in kJ/mol/nm3, converted to bar.
    kilo_joule_per_mole = mass_amu * (velocities**2).sum(axis=0)
    expected = kilo_joule_per_mole / box_nm**3 * 16.605390666
    assert np.diag(pressure_tensor_bar(simulation)) == pytest.approx(expected)


@pytest.mark.usefixtures("stress_backend")
@pytest.mark.parametrize("kind", ["anisotropic", "flexible"])
def test_a_rigid_rotor_gas_pins_the_molecular_virial(kind: str) -> None:
    """The one test that can tell the two virial conventions apart.

    With no interactions the exact pressure is N_molecules kT / V: a rotor
    translates as one unit under a volume move, so its rotational kinetic
    energy does not push on the walls. Counting every atom instead - the
    atomic virial - inflates a diatomic's answer by 5/3, and with
    constraints on there is no finite difference that can see the constraint
    forces and put it back. This is why the readout must stay molecular.
    """
    from openmm import unit

    n_molecules, box_nm, temperature = 200, 4.0, 300.0
    system, topology, positions = rigid_rotor_system(n_molecules, box_nm)
    simulation = _simulation(system, topology, positions, kind)
    simulation.context.applyConstraints(1.0e-10)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, 5)
    simulation.context.applyVelocityConstraints(1.0e-10)

    exact = n_molecules * BOLTZMANN_KJ_PER_K * temperature / box_nm**3 * 16.605390666
    measured = float(np.mean(np.diag(pressure_tensor_bar(simulation))))
    # One draw of velocities, so this is the equipartition mean to within the
    # scatter of 3N samples, not an identity. A factor 5/3 would be nowhere
    # near it.
    assert measured == pytest.approx(exact, rel=0.1)
    assert measured < exact * 1.3


def test_every_molecule_openmm_finds_is_one_of_the_chains() -> None:
    """A merged pair of chains is a wrong molecular virial, silently."""
    system, topology, positions = rigid_rotor_system(50, 3.0)
    simulation = _simulation(system, topology, positions, "anisotropic")
    groups = molecule_groups(simulation)
    assert len(groups) == 50
    assert all(group.size == 2 for group in groups)


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


def test_all_three_axes_are_reported_even_with_one_frozen() -> None:
    """A uniaxial extension depends on this and OpenMM does not promise it.

    The driven axis is held by giving the barostat scaleZ=False, and its
    stress is the measurement. If freezing an axis also stopped it being
    reported there would be no measurement at all.
    """
    system, topology, positions = ideal_gas_system(64, 3.0)
    simulation = _simulation(
        system, topology, positions, "anisotropic", scale_axes=(True, True, False)
    )
    simulation.context.setVelocitiesToTemperature(300.0, 2)
    tensor = pressure_tensor_bar(simulation)
    assert np.isfinite(np.diag(tensor)).all()


def test_components_nothing_measured_come_back_as_nan() -> None:
    """Not zero. An unmeasured shear stress is not a measured zero one."""
    system, topology, positions = ideal_gas_system(64, 3.0)
    simulation = _simulation(system, topology, positions, "anisotropic")
    simulation.context.setVelocitiesToTemperature(300.0, 2)
    tensor = pressure_tensor_bar(simulation)
    assert math.isnan(tensor[0, 1])
    assert math.isnan(tensor[0, 2])
    assert math.isnan(tensor[1, 2])


def test_a_flexible_barostat_fills_the_whole_symmetric_tensor() -> None:
    """All six components, in the places OpenMM's order puts them."""
    system, topology, positions = ideal_gas_system(64, 3.0)
    simulation = _simulation(system, topology, positions, "flexible")
    simulation.context.setVelocitiesToTemperature(300.0, 2)
    tensor = pressure_tensor_bar(simulation)
    assert np.isfinite(tensor).all()
    assert tensor == pytest.approx(tensor.T)


def test_the_flexible_shear_component_matches_the_kinetic_sum() -> None:
    """Which cell a shear component lands in, checked against an exact value.

    Asserting the six numbers are merely finite cannot catch a transposed
    map; a near-isotropic cell has off-diagonals a thousand times smaller
    than its diagonal, so a swap would pass unnoticed.
    """
    from openmm import unit

    n_atoms, box_nm, mass_amu = 64, 3.0, 40.0
    system, topology, positions = ideal_gas_system(n_atoms, box_nm, mass_amu=mass_amu)
    simulation = _simulation(system, topology, positions, "flexible")
    velocities = np.random.default_rng(1).normal(0.0, 0.3, size=(n_atoms, 3))
    simulation.context.setVelocities(velocities * unit.nanometer / unit.picosecond)

    tensor = pressure_tensor_bar(simulation)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        expected = (
            mass_amu
            * float((velocities[:, first] * velocities[:, second]).sum())
            / box_nm**3
            * 16.605390666
        )
        assert tensor[first, second] == pytest.approx(expected)


@pytest.mark.usefixtures("stress_backend")
def test_interacting_triclinic_stress_is_a_consistent_strain_derivative() -> None:
    """Nonzero potential shear and tilted diagonals detect box-entry derivatives.

    Independently deform every position and box vector, with a much smaller
    difference step than the production estimator. A finite kinetic term
    also pins the sign, units and component ordering of the complete tensor.
    """
    from openmm import unit

    system, topology, positions = argon_system(64, 2.4)
    simulation = _simulation(system, topology, positions, "flexible")
    deformation = np.array([[1.0, 0.07, 0.1], [0.0, 1.0, -0.04], [0.0, 0.0, 1.0]])
    positions = positions @ deformation.T
    box = 2.4 * deformation.T
    context = simulation.context
    context.setPeriodicBoxVectors(*(box * unit.nanometer))
    context.setPositions(positions * unit.nanometer)
    velocities = np.random.default_rng(17).normal(0.0, 0.2, (64, 3))
    context.setVelocities(velocities * unit.nanometer / unit.picosecond)
    measured = stress_tensor_bar(simulation)
    assert pressure_bar(simulation) == pytest.approx(-np.trace(measured) / 3.0)
    kinetic = 39.948 * velocities.T @ velocities
    expected = np.empty((3, 3))
    delta = 1.0e-5
    try:
        for row, column in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            energies = []
            for sign in (-1, 1):
                strain = np.eye(3)
                strain[row, column] += sign * delta
                context.setPeriodicBoxVectors(*((box @ strain.T) * unit.nanometer))
                context.setPositions((positions @ strain.T) * unit.nanometer)
                energies.append(
                    context.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(unit.kilojoule_per_mole)
                )
            virial = (energies[1] - energies[0]) / (2 * delta)
            value = (
                (virial - kinetic[row, column])
                / np.linalg.det(box)
                * 16.605390671738468
            )
            expected[row, column] = expected[column, row] = value
    finally:
        context.setPeriodicBoxVectors(*(box * unit.nanometer))
        context.setPositions(positions * unit.nanometer)
    assert measured == pytest.approx(expected, rel=3.0e-4, abs=1.0e-5)


@pytest.mark.usefixtures("stress_backend")
@pytest.mark.parametrize("rigid", [False, True])
@pytest.mark.parametrize("include_bond", [False, True])
def test_flexible_stress_preserves_molecular_convention_and_force_groups(
    rigid: bool, include_bond: bool
) -> None:
    """An unequal-mass bonded dimer separates molecular and atomic virials."""
    import openmm as mm
    from openmm import unit

    system, topology, positions = ideal_gas_system(2, 3.0)
    positions[:] = [[0.4, 0.5, 0.6], [0.6, 0.6, 0.9]]
    system.setParticleMass(0, 12.0)
    system.setParticleMass(1, 3.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 0.2, 200.0)
    bond.setForceGroup(1)
    system.addForce(bond)
    simulation = _simulation(
        system, topology, positions, "flexible", scale_molecules_as_rigid=rigid
    )
    simulation.integrator.setIntegrationForceGroups(-1 if include_bond else 1)
    velocities = np.array([[0.2, 0.3, -0.1], [-0.1, 0.4, 0.2]])
    simulation.context.setVelocities(velocities * unit.nanometer / unit.picosecond)
    masses = np.array([12.0, 3.0])
    if rigid:
        momentum = (velocities * masses[:, None]).sum(axis=0)
        expected = -np.outer(momentum, momentum) / masses.sum()
    else:
        expected = -(velocities * masses[:, None]).T @ velocities
        if include_bond:
            distance = positions[1] - positions[0]
            length = np.linalg.norm(distance)
            expected += 200.0 * (length - 0.2) / length * np.outer(distance, distance)
    expected *= 16.605390671738468 / 3.0**3
    assert stress_tensor_bar(simulation) == pytest.approx(
        expected, rel=2.0e-5, abs=1.0e-7
    )


@pytest.mark.usefixtures("stress_backend")
def test_flexible_stress_preserves_context_state() -> None:
    """A measurement must leave subsequent dynamics at the same state."""
    from openmm import unit

    system, topology, positions = argon_system(64, 2.4)
    simulation = _simulation(system, topology, positions, "flexible")
    context = simulation.context
    context.setVelocitiesToTemperature(250.0, 19)
    simulation.step(4)
    before = context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, getParameters=True
    )
    stress_tensor_bar(simulation)
    after = context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, getParameters=True
    )
    assert after.getPositions(asNumpy=True).value_in_unit(
        unit.nanometer
    ) == pytest.approx(
        before.getPositions(asNumpy=True).value_in_unit(unit.nanometer), abs=1.0e-13
    )
    assert after.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    ) == pytest.approx(
        before.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        ),
        abs=1.0e-13,
    )
    assert after.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
        unit.nanometer
    ) == pytest.approx(
        before.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
        abs=1.0e-13,
    )
    assert after.getTime() == before.getTime()
    assert after.getStepCount() == before.getStepCount()
    assert dict(after.getParameters()) == dict(before.getParameters())
    assert after.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    ) == pytest.approx(
        before.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), abs=1.0e-10
    )


def test_fallback_restores_positions_and_box_on_energy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception during either side of a derivative must not strain the run."""
    import openmm as mm
    from openmm import unit

    monkeypatch.delattr(
        mm.MonteCarloFlexibleBarostat, "computeStressTensor", raising=False
    )
    system, topology, positions = argon_system(64, 2.4)
    simulation = _simulation(system, topology, positions, "flexible")
    context = simulation.context
    before = context.getState(getPositions=True)
    original = context.getState

    def failing_energy(**kwargs: Any) -> Any:
        if kwargs.get("getEnergy"):
            raise RuntimeError("energy failed")
        return original(**kwargs)

    monkeypatch.setattr(context, "getState", failing_energy)
    with pytest.raises(RuntimeError, match="energy failed"):
        stress_tensor_bar(simulation)
    after = context.getState(getPositions=True)
    assert np.array_equal(
        after.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        before.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
    )
    assert np.array_equal(
        after.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
        before.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
    )


@pytest.mark.parametrize("reader", [pressure_bar, pressure_tensor_bar])
def test_old_openmm_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch, reader: Any
) -> None:
    """An unsupported install should not surface a bare AttributeError."""
    import openmm as mm

    system, topology, positions = ideal_gas_system(8, 3.0)
    simulation = _simulation(system, topology, positions, "anisotropic")
    monkeypatch.delattr(mm.MonteCarloAnisotropicBarostat, "computeCurrentPressure")
    with pytest.raises(StressError, match=r"OpenMM >= 8.6.1.*Upgrade"):
        reader(simulation)


@pytest.mark.parametrize("release", ["8.3.0.dev-1ce5d91", "8.3.1", "8.5.2", "8.6.0"])
@pytest.mark.parametrize("reader", [pressure_bar, pressure_tensor_bar])
def test_openmm_below_supported_floor_is_rejected(
    monkeypatch: pytest.MonkeyPatch, release: str, reader: Any
) -> None:
    """Having a pressure method does not make an older release supported."""
    from openmm import version

    system, topology, positions = ideal_gas_system(8, 3.0)
    simulation = _simulation(system, topology, positions, "anisotropic")
    monkeypatch.setattr(version, "version", release)
    with pytest.raises(StressError, match=r"OpenMM >= 8.6.1.*Upgrade"):
        reader(simulation)


def test_stress_is_minus_pressure_so_tension_is_positive() -> None:
    """The sign convention a stress-strain curve is drawn in."""
    system, topology, positions = ideal_gas_system(64, 3.0)
    simulation = _simulation(system, topology, positions, "anisotropic")
    simulation.context.setVelocitiesToTemperature(300.0, 3)
    pressure = pressure_tensor_bar(simulation)
    stress = stress_tensor_bar(simulation)
    assert np.diag(stress) == pytest.approx(-np.diag(pressure))
    # A gas pushes out, so its pressure is positive and its stress negative.
    assert (np.diag(pressure) > 0.0).all()


def test_the_scalar_pressure_is_the_mean_of_the_diagonal() -> None:
    """What an isotropic barostat can report, and all it can report."""
    system, topology, positions = ideal_gas_system(64, 3.0)
    simulation = _simulation(system, topology, positions, "anisotropic")
    simulation.context.setVelocitiesToTemperature(300.0, 4)
    assert pressure_bar(simulation) == pytest.approx(
        float(np.mean(np.diag(pressure_tensor_bar(simulation))))
    )


def test_an_isotropic_barostat_refuses_the_tensor_and_says_what_to_use() -> None:
    """One number cannot say how it is distributed over three axes."""
    system, topology, positions = ideal_gas_system(64, 3.0)
    simulation = _simulation(system, topology, positions, "isotropic")
    simulation.context.setVelocitiesToTemperature(300.0, 5)
    assert math.isfinite(pressure_bar(simulation))
    with pytest.raises(StressError, match="anisotropic"):
        pressure_tensor_bar(simulation)


def test_no_barostat_is_a_message_naming_the_fix() -> None:
    """OpenMM reports pressure only through a barostat, so say so."""
    import openmm as mm
    from openmm import app, unit

    system, topology, positions = ideal_gas_system(64, 3.0)
    simulation = app.Simulation(
        topology,
        system,
        mm.VerletIntegrator(0.001 * unit.picoseconds),
        mm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(positions * unit.nanometer)
    with pytest.raises(StressError, match="frequency=0"):
        pressure_tensor_bar(simulation)
    with pytest.raises(StressError):
        find_barostat(simulation)


def test_a_probe_at_zero_frequency_never_moves_the_box() -> None:
    """What makes a barostat readable without making it an ensemble."""
    from openmm import unit

    system, topology, positions = ideal_gas_system(216, 3.0)
    simulation = _simulation(system, topology, positions, "flexible")
    simulation.context.setVelocitiesToTemperature(300.0, 6)
    before = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
    simulation.step(300)
    after = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
    assert np.asarray(before.value_in_unit(unit.nanometer)) == pytest.approx(
        np.asarray(after.value_in_unit(unit.nanometer))
    )


def test_the_tensile_stress_subtracts_the_lateral_pair() -> None:
    """Plain arithmetic, but it is arithmetic with a sign in it."""
    stress = np.diag([10.0, 20.0, 100.0])
    assert tensile_stress_bar(stress, axis=2) == pytest.approx(100.0 - 15.0)
    assert tensile_stress_bar(stress, axis=0) == pytest.approx(10.0 - 60.0)


# --------------------------------------------------------------------------
# Applying a strain
# --------------------------------------------------------------------------


def test_an_affine_scale_moves_every_atom_and_leaves_the_input_alone() -> None:
    """Per atom, not per molecule - which is the whole point of it."""
    positions = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    scaled = affine_scale(positions, (1.0, 1.0, 1.05))
    assert scaled[:, 2] == pytest.approx(positions[:, 2] * 1.05)
    assert scaled[:, :2] == pytest.approx(positions[:, :2])
    assert positions == pytest.approx(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_an_affine_shear_displaces_along_the_gradient() -> None:
    """x gains gamma times z, and nothing else moves."""
    positions = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    sheared = affine_shear(positions, 0.02, (0, 2))
    assert sheared[:, 0] == pytest.approx(positions[:, 0] + 0.02 * positions[:, 2])
    assert sheared[:, 1:] == pytest.approx(positions[:, 1:])
    with pytest.raises(ValueError, match="two different axes"):
        affine_shear(positions, 0.02, (2, 2))


def test_straining_a_cell_does_not_depend_on_which_image_a_molecule_is_in() -> None:
    """The invariance the whole strain contract rests on.

    An affine map of a periodic cell commutes with the choice of periodic
    image *provided the box is mapped too*. Move a whole molecule by one box
    length, strain both, and the energy must be identical to the bit - and
    is. Strain the positions without the box and it is not, which is the
    mistake this asserts against.
    """
    import openmm as mm
    from openmm import unit

    system, topology, positions = rigid_rotor_system(50, 3.0)
    simulation = _simulation(system, topology, positions, "anisotropic")
    box = 3.0
    shifted = positions.copy()
    shifted[0:2, 2] += box

    def energy(coordinates: np.ndarray, scale_box: bool) -> float:
        factor = 1.02
        vectors = [
            mm.Vec3(box, 0, 0),
            mm.Vec3(0, box, 0),
            mm.Vec3(0, 0, box * factor if scale_box else box),
        ]
        simulation.context.setPeriodicBoxVectors(
            *[vector * unit.nanometer for vector in vectors]
        )
        simulation.context.setPositions(
            affine_scale(coordinates, (1.0, 1.0, factor)) * unit.nanometer
        )
        return float(
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )

    assert energy(shifted, True) == pytest.approx(energy(positions, True), abs=1e-9)


def test_a_shear_past_the_reduced_form_is_refused_before_it_runs() -> None:
    """OpenMM's triclinic constraint, named here rather than several steps on."""
    import openmm as mm
    from openmm import unit

    vectors = [
        mm.Vec3(3.0, 0, 0) * unit.nanometer,
        mm.Vec3(0, 3.0, 0) * unit.nanometer,
        mm.Vec3(0, 0, 3.0) * unit.nanometer,
    ]
    tilted = shear_box_vectors(vectors, 0.02, (0, 2))
    assert tilted[2][0].value_in_unit(unit.nanometer) == pytest.approx(0.06)
    assert tilted[0][0].value_in_unit(unit.nanometer) == pytest.approx(3.0)
    with pytest.raises(StressError, match="reduced form"):
        shear_box_vectors(vectors, 0.9, (0, 2))


def test_a_sheared_cell_is_accepted_by_openmm_and_energies_stay_finite() -> None:
    """The tilt and the displacement have to agree, or the cell is nonsense."""
    from openmm import unit

    system, topology, positions = rigid_rotor_system(64, 3.0)
    simulation = _simulation(system, topology, positions, "flexible")
    original = simulation.context.getState().getPeriodicBoxVectors()
    for gamma in (0.005, 0.02, 0.05):
        simulation.context.setPeriodicBoxVectors(
            *shear_box_vectors(original, gamma, (0, 2))
        )
        simulation.context.setPositions(
            affine_shear(positions, gamma, (0, 2)) * unit.nanometer
        )
        vectors = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
        assert float(
            np.asarray(vectors.value_in_unit(unit.nanometer))[2][0]
        ) == pytest.approx(gamma * 3.0)
        assert math.isfinite(
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
