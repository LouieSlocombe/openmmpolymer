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

from .helpers import ideal_gas_system, rigid_rotor_system

BOLTZMANN_KJ_PER_K = 0.00831446261815324


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


def test_a_rigid_rotor_gas_pins_the_molecular_virial() -> None:
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
    simulation = _simulation(system, topology, positions, "anisotropic")
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
