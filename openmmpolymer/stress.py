"""The stress tensor of a running cell, and the strain that is applied to it.

OpenMM 8.3 added ``Barostat.computeCurrentPressure``. Its flexible-barostat
implementation differentiates individual box entries, which is not a physical
stress in a tilted cell. In particular, a zero tilt can incorrectly imply zero
potential shear stress. A Cauchy stress instead differentiates energy under
the same infinitesimal deformation of positions and every box vector.

For a flexible barostat we use ``computeStressTensor`` where available, and
otherwise evaluate that consistent-strain derivative here. Both paths retain
the barostat's molecular convention: translate geometric molecular centres
rigidly and use centre-of-mass kinetic energy, or deform individual atoms when
``getScaleMoleculesAsRigid()`` is false. The isotropic and anisotropic readouts
use the pressure API directly. OpenMM 8.6.1 is the minimum supported release.

Which barostat is attached decides what can be read. An anisotropic barostat
reports the three diagonal components, and reports all three even when one
axis is frozen with ``scaleZ=False`` - which is the whole basis of a uniaxial
extension, where the driven axis is held and its stress is the measurement.
Only the flexible barostat reports shear. A barostat that is not in the
Context cannot be asked at all: OpenMM raises ``getImplInContext``. So a
constant-volume shear stage attaches a flexible barostat at ``frequency=0``,
which never moves the box and exists only to be asked.

The one thing to carry away from OpenMM's own documentation of it: *the
fluctuations around the average are extremely large, and it may take a very
long simulation to compute the average accurately*. Every user of this module
averages, and :mod:`openmmpolymer.elasticity` refuses to quote a modulus whose
standard error does not support it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ._validation import require_plane

log = logging.getLogger(__name__)

#: Which barostats report which components. The anisotropic one gives the
#: diagonal; only the flexible one gives shear.
TENSOR_BAROSTATS = ("anisotropic", "flexible")

#: Recorded in shear stage samples so analysis can reject legacy box-entry
#: derivatives, which cannot be corrected from averaged stress data alone.
STRESS_ESTIMATOR_VERSION = 1.0

#: Where the flexible barostat's six numbers belong in a symmetric 3x3, in the
#: order OpenMM returns them: (XX, YY, ZZ, XY, XZ, YZ).
_FLEXIBLE_ORDER = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


class StressError(RuntimeError):
    """A stress could not be read, or a strain could not be applied."""


def _require_pressure_api(barostat: Any) -> None:
    """Give a useful failure for an old OpenMM installed with --no-deps."""
    from openmm import version

    release = re.match(r"(\d+)\.(\d+)\.(\d+)", version.version)
    unsupported = release is not None and tuple(map(int, release.groups())) < (8, 6, 1)
    if unsupported or not callable(getattr(barostat, "computeCurrentPressure", None)):
        raise StressError(
            "Stress measurement requires OpenMM >= 8.6.1 with "
            "computeCurrentPressure available. "
            "Upgrade OpenMM in this environment."
        )


def _current_pressure_bar(barostat: Any, context: Any) -> Any:
    """8.3 returns bare bar values; newer wrappers attach a pressure unit."""
    from openmm import unit

    value = barostat.computeCurrentPressure(context)
    return value.value_in_unit(unit.bar) if unit.is_quantity(value) else value


def _strain_pressure_tensor_bar(
    simulation: Any, barostat: Any
) -> npt.NDArray[np.float64]:
    """Consistent-strain pressure for releases without computeStressTensor.

    Use the same 1e-3 central difference as OpenMM's native stress estimator.
    It is large enough to resolve energy changes in mixed precision. No
    integration or constraint projection occurs, and positions and the box
    are restored even if an energy evaluation fails. The integration force
    groups and the molecular kinetic convention match the native estimator.
    """
    from openmm import unit

    context = simulation.context
    state = context.getState(getPositions=True, getVelocities=True)
    positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    )
    box = np.asarray(
        state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    )
    velocities = np.asarray(
        state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
    )
    masses = np.asarray(
        [
            simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(simulation.system.getNumParticles())
        ]
    )
    rigid = barostat.getScaleMoleculesAsRigid()
    groups = (
        molecule_groups(simulation)
        if rigid
        else tuple(np.asarray([i]) for i in range(len(masses)))
    )
    centres = np.empty_like(positions)
    kinetic = np.zeros((3, 3), dtype=np.float64)
    for group in groups:
        centres[group] = positions[group].mean(axis=0)
        mass = float(masses[group].sum())
        if mass > 0.0:
            momentum = (masses[group, None] * velocities[group]).sum(axis=0)
            kinetic += np.outer(momentum, momentum) / mass

    volume = float(np.linalg.det(box))
    force_groups = simulation.integrator.getIntegrationForceGroups()
    delta = 1.0e-3
    tensor = np.empty((3, 3), dtype=np.float64)
    try:
        for row, column in _FLEXIBLE_ORDER:
            energies = []
            for strain in (delta, -delta):
                strained_box = box.copy()
                strained_box[:, row] += strain * box[:, column]
                # A physically equivalent reduced lattice is required by OpenMM.
                strained_box[2] -= strained_box[1] * round(
                    strained_box[2, 1] / strained_box[1, 1]
                )
                strained_box[2] -= strained_box[0] * round(
                    strained_box[2, 0] / strained_box[0, 0]
                )
                strained_box[1] -= strained_box[0] * round(
                    strained_box[1, 0] / strained_box[0, 0]
                )
                displaced = positions.copy()
                displaced[:, row] += strain * centres[:, column]
                context.setPeriodicBoxVectors(*(strained_box * unit.nanometer))
                context.setPositions(displaced * unit.nanometer)
                energy = context.getState(
                    getEnergy=True, groups=force_groups
                ).getPotentialEnergy()
                energies.append(energy.value_in_unit(unit.kilojoule_per_mole))
            derivative = (energies[0] - energies[1]) / (2.0 * delta)
            value = (kinetic[row, column] - derivative) / volume * 16.605390671738468
            tensor[row, column] = tensor[column, row] = value
    finally:
        context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        context.setPositions(state.getPositions())
    return tensor


def _flexible_pressure_tensor_bar(
    simulation: Any, barostat: Any
) -> npt.NDArray[np.float64]:
    """Read Cauchy stress, never the flexible box-entry pressure derivative."""
    from openmm import unit

    if not callable(getattr(barostat, "computeStressTensor", None)):
        return _strain_pressure_tensor_bar(simulation, barostat)
    # OpenMM returns tensile stress; this function's public convention is pressure.
    numbers = barostat.computeStressTensor(simulation.context, True).value_in_unit(
        unit.bar
    )
    tensor = np.empty((3, 3), dtype=np.float64)
    for value, (row, column) in zip(numbers, _FLEXIBLE_ORDER, strict=True):
        tensor[row, column] = tensor[column, row] = -float(value)
    return tensor


def find_barostat(simulation: Any) -> tuple[str, Any]:
    """Return the kind and force of the barostat driving *simulation*.

    Args:
        simulation: The running simulation.

    Returns:
        ``(kind, force)``.

    Raises:
        StressError: There is no barostat, so there is nothing to ask.
    """
    from .mdsystem import find_barostat as _find

    found = _find(simulation.system)
    if found is None:
        raise StressError(
            "This System has no barostat, and OpenMM reports pressure only "
            "through one - Context.getState() carries no virial. Add a "
            "MonteCarloAnisotropicBarostat for the three diagonal "
            "components, or a MonteCarloFlexibleBarostat for all six. At "
            "frequency=0 either one never moves the box and serves purely as "
            "a probe."
        )
    return found


def pressure_bar(simulation: Any) -> float:
    """Return the instantaneous mean pressure, in bar.

    One third of the trace of the pressure tensor, which for an isotropic
    barostat is the only thing it reports.

    Args:
        simulation: The running simulation.

    Returns:
        The pressure in bar. Instantaneously very noisy - see the module
        docstring.
    """
    kind, barostat = find_barostat(simulation)
    _require_pressure_api(barostat)
    if kind == "flexible":
        return float(
            np.trace(_flexible_pressure_tensor_bar(simulation, barostat)) / 3.0
        )
    value = _current_pressure_bar(barostat, simulation.context)
    if kind == "isotropic":
        return float(value)
    return float(np.mean([value[axis] for axis in range(3)]))


def pressure_tensor_bar(simulation: Any) -> npt.NDArray[np.float64]:
    """Return the instantaneous pressure tensor, in bar.

    Components the attached barostat does not measure come back as NaN rather
    than zero. An unmeasured shear stress and a measured zero shear stress are
    very different statements, and a cell of zeros would let the second be
    read off a run that only ever made the first.

    Args:
        simulation: The running simulation.

    Returns:
        A symmetric ``(3, 3)`` array in bar. An anisotropic barostat fills the
        diagonal and leaves the off-diagonals NaN; a flexible one fills all
        six.

    Raises:
        StressError: There is no barostat, or it is the isotropic one, which
            reports a single number and cannot say how it is distributed over
            the axes.
    """
    kind, barostat = find_barostat(simulation)
    _require_pressure_api(barostat)
    if kind == "isotropic":
        raise StressError(
            "An isotropic barostat reports one pressure, not a tensor, so "
            "there is no way to tell P_zz from P_xx on this run. Use "
            "pressure_bar() for the scalar, or build the stage with "
            'barostat="anisotropic" for the diagonal.'
        )

    if kind == "flexible":
        return _flexible_pressure_tensor_bar(simulation, barostat)
    tensor = np.full((3, 3), np.nan, dtype=np.float64)
    numbers = _current_pressure_bar(barostat, simulation.context)
    for axis in range(3):
        tensor[axis, axis] = float(numbers[axis])
    return tensor


def stress_tensor_bar(simulation: Any) -> npt.NDArray[np.float64]:
    """Return the instantaneous stress tensor, in bar.

    Minus the pressure tensor, so that tension is positive and compression is
    negative - the sign convention a stress-strain curve is drawn in, and the
    opposite of the one a barostat is set in.

    Args:
        simulation: The running simulation.

    Returns:
        A symmetric ``(3, 3)`` array in bar, NaN where nothing was measured.
    """
    return -pressure_tensor_bar(simulation)


def tensile_stress_bar(stress: npt.NDArray[np.float64], axis: int = 2) -> float:
    """Return the tensile stress along *axis*, corrected for the lateral pair.

    ``sigma_zz - (sigma_xx + sigma_yy) / 2``. The lateral axes are held at the
    target pressure by a barostat, but only on average and only as fast as the
    cell relaxes, so subtracting what they actually did removes a drift the
    driven axis would otherwise be credited with.

    Args:
        stress: A stress tensor, from :func:`stress_tensor_bar`.
        axis: The driven axis.

    Returns:
        The tensile stress in bar.
    """
    lateral = [index for index in range(3) if index != axis]
    return float(stress[axis, axis] - 0.5 * (sum(stress[i, i] for i in lateral)))


def molecule_groups(simulation: Any) -> tuple[npt.NDArray[np.int64], ...]:
    """Return the atom indices of each molecule, as OpenMM groups them.

    Taken from ``Context.getMolecules()``, which groups by bonds and
    constraints - the same grouping the barostat scales and the same one its
    molecular virial is computed over. Worth checking against the number of
    chains that were packed: if packmol merged two of them, this is where the
    merge becomes visible, and a cell of one giant molecule has almost no
    molecular virial at all.

    Args:
        simulation: The running simulation.

    Returns:
        One array of atom indices per molecule.
    """
    return tuple(
        np.asarray(sorted(group), dtype=np.int64)
        for group in simulation.context.getMolecules()
    )


def affine_scale(
    positions_nm: npt.NDArray[np.float64], factors: Sequence[float]
) -> npt.NDArray[np.float64]:
    """Scale every atom's position affinely with the cell.

    Per atom, not per molecule, and that is the substantive choice here.
    Translating whole molecules rigidly - which is what the barostat does for
    a volume move - moves chains past each other and leaves every
    conformation exactly as it was. A polymer's stiffness above its glass
    transition is almost entirely entropic: it comes from chains being
    stretched out of their preferred shapes. A deformation that does not
    stretch them does not generate that stress, and the modulus comes out
    low. Per-atom remapping is what a tensile simulation means, and what
    every other engine does by default.

    The reason the barostat cannot do this is a different one, and it does
    not apply. Per-atom scaling inside a *volume move* poisons the
    Metropolis test: the barostat evaluates the energy of a
    constraint-violating configuration and accepts or rejects on a number
    that is not the energy of any real state. Here the strain is applied
    once and the constraints are repaired immediately afterwards, before
    anything reads an energy - see
    :func:`openmmpolymer.simulate.run_deform`, which calls
    ``applyConstraints`` and ``applyVelocityConstraints`` on the way in.
    Measured on a cell with the usual ``constraints="hbonds"``: a strain
    increment of 0.002 stretches the longest constrained bond by 0.2 pm, and
    the repair moves no atom further than 2e-4 nm.

    The stress is still read with the *molecular* virial, and those are
    independent choices. Reading the atomic virial under constraints would
    be wrong by a large factor, because a finite difference cannot see the
    constraint forces.

    Args:
        positions_nm: Every atom's position, in nanometres. Must be the
            whole-molecule positions OpenMM keeps internally, i.e. from a
            ``getState`` without ``enforcePeriodicBox``.
        factors: The scale factor on each axis.

    Returns:
        New positions, in nanometres. The input is not modified.

    Raises:
        ValueError: *factors* is not three numbers.
    """
    scale = np.asarray(factors, dtype=np.float64)
    if scale.shape != (3,):
        raise ValueError(f"factors={factors!r} must have three entries.")
    return np.asarray(positions_nm, dtype=np.float64) * scale


def affine_shear(
    positions_nm: npt.NDArray[np.float64],
    gamma: float,
    plane: tuple[int, int] = (0, 2),
) -> npt.NDArray[np.float64]:
    """Shear every atom's position affinely with the cell.

    ``x += gamma * z`` for ``plane=(0, 2)``, and the matching displacement
    for any other pair. Per atom for the reason :func:`affine_scale` is.

    The caller must tilt the box by the same amount, or the result is not a
    shear of a periodic cell but a shear of its contents inside an unchanged
    one - a different configuration with a different energy. Measured: with
    the box tilted, re-imaging a molecule leaves the energy bit-identical;
    without, it changes it.

    Args:
        positions_nm: Whole-molecule positions, in nanometres.
        gamma: The shear strain to add.
        plane: ``(driven, gradient)`` axes - the displaced direction and the
            direction it varies along.

    Returns:
        New positions, in nanometres. The input is not modified.

    Raises:
        ValueError: *plane* is not two different axes.
    """
    driven, gradient = require_plane(plane)
    sheared = np.array(positions_nm, dtype=np.float64, copy=True)
    sheared[:, driven] += gamma * sheared[:, gradient]
    return sheared


def shear_box_vectors(vectors_nm: Any, gamma: float, plane: tuple[int, int]) -> Any:
    """Tilt a set of box vectors by *gamma*, returning new ones.

    The companion to :func:`affine_shear`, so that the two cannot be applied
    inconsistently. OpenMM requires reduced-form triclinic vectors, which for
    the usual ``(0, 2)`` plane means ``|c_x| <= a_x / 2``; a tilt past that is
    refused here rather than several steps later by OpenMM.

    Args:
        vectors_nm: The current box vectors, as OpenMM returns them.
        gamma: The shear strain.
        plane: ``(driven, gradient)`` axes.

    Returns:
        New box vectors, as a list of ``openmm.Vec3`` quantities.

    Raises:
        StressError: The tilt is past OpenMM's reduced form.
    """
    import openmm as mm
    from openmm import unit

    driven, gradient = plane
    rows = [
        [vector[axis].value_in_unit(unit.nanometer) for axis in range(3)]
        for vector in vectors_nm
    ]
    rows[gradient][driven] += gamma * rows[gradient][gradient]
    limit = 0.5 * rows[driven][driven]
    if abs(rows[gradient][driven]) > limit + 1.0e-12:
        raise StressError(
            f"A shear strain of {gamma:.4f} tilts the box to "
            f"{rows[gradient][driven]:.4f} nm, past the {limit:.4f} nm that "
            "OpenMM's reduced form allows. Shear less, or use a cell that is "
            "longer along the driven axis."
        )
    return [mm.Vec3(*row) * unit.nanometer for row in rows]


def deviatoric_strain(strain: float, poisson: float) -> float:
    """Return ``e_axial - e_lateral`` for a uniaxial affine step.

    The quantity a differential stress actually measures against, and the
    reason it is worth having in one place. For an isotropic solid under an
    imposed diagonal strain, ``sigma = 2 G e + lambda tr(e) I``, so

        sigma_zz - (sigma_xx + sigma_yy) / 2 = 2 G (e_axial - e_lateral)

    and the Lame constant cancels identically. The differential stress is a
    pure deviatoric projection: it measures the shear modulus alone, with no
    dependence on the bulk modulus, on Poisson's ratio, or on whether the step
    happened to preserve the volume. Dividing it by the axial strain instead -
    which is what a textbook writes - gives Young's modulus only when the
    material's Poisson ratio is exactly one half, and is out by about one per
    cent at a four per cent strain even then, purely from linearising the
    lateral factor.

    Args:
        strain: The engineering strain applied along the driven axis.
        poisson: The lateral contraction that was imposed, as the exponent in
            ``(1 + strain) ** -poisson``. This is the deformation that was
            applied, which need not be the material's own Poisson ratio -
            nothing above depends on the two agreeing.

    Returns:
        The difference between the axial and lateral engineering strains.
    """
    return float(strain - ((1.0 + strain) ** -poisson - 1.0))
