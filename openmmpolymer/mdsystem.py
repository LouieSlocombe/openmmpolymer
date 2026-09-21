"""Assembling the packed cell into an OpenMM System.

packmol writes coordinates and nothing else: its output PDB carries no CONECT
records, so reading it with ``app.PDBFile`` gives a topology with no bonds and
every residue template fails to match. The topology therefore comes from
somewhere else - the single-chain PDB, whose bonds are correct - replicated
once per molecule, with only the *positions* taken from packmol.

That mapping rests on packmol writing each copy's atoms in the order of the
input file, which it does. It is asserted anyway, on atom count and on the
element sequence, because the failure mode if it ever stopped being true is not
an exception but a System with every parameter on the wrong atom.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._seeds import seed_random_stream
from ._validation import require_choice, require_positive
from .forcefield import PolymerForceField
from .packing import PackedComponent, read_packed_pdb, read_pdb

log = logging.getLogger(__name__)

#: Constraint settings, by the name used in configuration.
CONSTRAINTS = ("none", "hbonds", "allbonds", "hangles")

#: Barostat kinds.
BAROSTATS = ("isotropic", "anisotropic")

#: The global parameter each barostat reads its temperature from. They are not
#: the same name, and setting the isotropic one on an anisotropic barostat
#: raises rather than being ignored.
BAROSTAT_TEMPERATURE_PARAMETER = {
    "isotropic": "MonteCarloTemperature",
    "anisotropic": "AnisotropicMonteCarloTemperature",
}

#: Platforms in the order they are tried when none is named.
PLATFORM_PREFERENCE = ("CUDA", "OpenCL", "CPU", "Reference")

#: Platforms that take a ``Precision`` property.
_PRECISION_PLATFORMS = frozenset({"CUDA", "OpenCL"})


class SystemAssemblyError(RuntimeError):
    """The packed cell could not be turned into a System."""


@dataclass(frozen=True)
class SystemSpec:
    """How the System is built.

    Every field is a plain float in the unit its name gives.
    ``openmm.unit.Quantity`` is unhashable, so a quantity cannot be a dataclass
    default at all, and floats are what a run manifest can record.

    Args:
        nonbonded_cutoff_nm: Real-space cutoff. 1.2 nm rather than 1.0, which
            is short for a GAFF melt.
        switch_distance_nm: Where the switching function starts, or None.
            None is the right default for GAFF and OpenFF: they are
            parameterised against a hard cutoff plus the analytic long-range
            correction, and force switching is a CHARMM convention. Turning it
            on quietly changes the force field the charges were fitted with.
        constraints: One of :data:`CONSTRAINTS`.
        rigid_water: Whether water, if any, is rigid.
        hydrogen_mass_amu: Hydrogen mass for repartitioning, or None to leave
            masses alone. Repartitioning changes the dynamics, so it is off
            unless asked for, and :func:`check_timestep` enforces the pairing
            with the timestep.
        ewald_error_tolerance: PME accuracy.
        use_dispersion_correction: The analytic long-range correction. Worth
            one to three per cent on a melt density, which is the quantity most
            of these runs exist to measure. Never forward None here:
            ``createSystem`` coerces its argument with ``bool()``, so passing
            None turns the correction off while reading as "leave the default".
        remove_cm_motion: Whether to remove centre-of-mass motion.
        scale_molecules_as_rigid: Whether a barostat volume move translates
            each molecule rigidly rather than scaling atoms individually.
            OpenMM's documentation suggests per-atom scaling for molecules big
            enough to span the cell, which a polymer chain can be - but scaled
            positions violate any constraint they cross, and the corrupted
            energy goes straight into the Metropolis test. Measured on a
            30-chain polyethylene cell at 600 K and 1000 bar: rigid scaling
            settles at 0.71 g/cm3, per-atom scaling with ``constraints="hbonds"``
            falls to 0.14 and keeps going. Rigid is therefore the default, and
            per-atom is refused outright while constraints are on.
        minimum_box_factor: The cell edge must exceed this many cutoffs.
            OpenMM's own limit is two; the margin above it is because the
            barostat shrinks the cell during compression and hits the limit
            mid-run rather than at startup.
    """

    nonbonded_cutoff_nm: float = 1.2
    switch_distance_nm: float | None = None
    constraints: str = "hbonds"
    rigid_water: bool = True
    hydrogen_mass_amu: float | None = None
    ewald_error_tolerance: float = 5.0e-4
    use_dispersion_correction: bool = True
    remove_cm_motion: bool = True
    scale_molecules_as_rigid: bool = True
    minimum_box_factor: float = 2.5

    def __post_init__(self) -> None:
        """Reject a spec that could not build a System."""
        require_choice(self.constraints, CONSTRAINTS, name="constraints")
        require_positive(self.nonbonded_cutoff_nm, None, name="nonbonded_cutoff_nm")
        if (
            self.switch_distance_nm is not None
            and self.switch_distance_nm >= self.nonbonded_cutoff_nm
        ):
            raise ValueError(
                f"switch_distance_nm={self.switch_distance_nm} must be below "
                f"nonbonded_cutoff_nm={self.nonbonded_cutoff_nm}."
            )
        if not self.scale_molecules_as_rigid and self.constraints != "none":
            raise ValueError(
                "scale_molecules_as_rigid=False cannot be combined with "
                f"constraints={self.constraints!r}. A per-atom volume move "
                "scales atoms through their constraints, and the energy that "
                "reaches the barostat's Metropolis test is not the energy of "
                "any real configuration: a polyethylene melt run this way "
                "expands under a kilobar of pressure. Use constraints='none' "
                "with a shorter timestep, or leave rigid scaling on."
            )


@dataclass
class PackedBox:
    """A periodic cell of chains, ready to have a System built from it.

    Args:
        topology: The box topology, with bonds, box vectors set.
        positions_nm: Positions, in the topology's atom order.
        box_nm: The cell edges.
        n_molecules: How many molecules it holds.
    """

    topology: Any
    positions_nm: npt.NDArray[np.float64]
    box_nm: tuple[float, float, float]
    n_molecules: int

    @property
    def positions(self) -> Any:
        """Positions as an ``openmm.unit.Quantity``, for OpenMM calls."""
        from openmm import unit

        return self.positions_nm * unit.nanometer


def replicate_topology(components: Sequence[PackedComponent]) -> tuple[Any, Any]:
    """Build a box topology by replicating each component's own topology.

    Args:
        components: The structures packmol was given, in the same order and
            with the same counts. Order matters: it is what makes the packed
            coordinates line up.

    Returns:
        A ``(topology, positions)`` pair. The positions are the input
        conformers' own, and are replaced by packmol's.
    """
    from openmm import app

    modeller = app.Modeller(app.Topology(), [])
    for component in components:
        source = read_pdb(component.pdb_path)
        for _ in range(component.count):
            modeller.add(source.topology, source.positions)
    return modeller.topology, modeller.positions


def assemble_box(
    components: Sequence[PackedComponent],
    packed_pdb: str,
    box_nm: tuple[float, float, float],
) -> PackedBox:
    """Put the replicated topology together with packmol's coordinates.

    Args:
        components: What packmol was given, in the same order and counts.
        packed_pdb: packmol's output.
        box_nm: The periodic cell edges.

    Returns:
        The assembled cell.

    Raises:
        SystemAssemblyError: The packed file and the replicated topology disagree, so
            the coordinates cannot be trusted to belong to these atoms.
    """
    import openmm as mm
    from openmm import unit

    topology, _ = replicate_topology(components)
    packed_topology, positions = read_packed_pdb(packed_pdb)

    if positions.shape[0] != topology.getNumAtoms():
        short = topology.getNumAtoms() - positions.shape[0]
        cause = (
            # Atoms going missing has one cause in practice, and it is worth
            # naming: two molecules sharing a chain identifier and a residue
            # number, which the PDB parser reads as one residue described
            # twice and silently discards the second copy of.
            " Atoms are missing rather than extra, which means the packed "
            "file has molecules sharing a chain and residue number and the "
            "parser dropped the duplicates. Every structure block needs "
            "`resnumbers 3`, which render_packmol_input() emits - so this "
            "file was packed by something else, or by an older version."
            if short > 0
            else ""
        )
        raise SystemAssemblyError(
            f"{packed_pdb} holds {positions.shape[0]} atoms but replicating "
            f"the input structures gives {topology.getNumAtoms()}.{cause} "
            "Otherwise the components passed here are not the ones packmol "
            "was given, or not in the same order."
        )
    _check_element_order(topology, packed_topology, packed_pdb)

    vectors = [
        mm.Vec3(box_nm[0], 0.0, 0.0),
        mm.Vec3(0.0, box_nm[1], 0.0),
        mm.Vec3(0.0, 0.0, box_nm[2]),
    ] * unit.nanometer
    topology.setPeriodicBoxVectors(vectors)

    return PackedBox(
        topology=topology,
        positions_nm=positions,
        box_nm=box_nm,
        n_molecules=sum(component.count for component in components),
    )


def _check_element_order(topology: Any, packed: Any, packed_pdb: str) -> None:
    """Raise unless the packed file's elements match the topology's, in order.

    The count matching is not enough. Two structures with the same number of
    atoms placed in the wrong order would pass that and produce a System whose
    every parameter sits on the wrong atom, with no error anywhere.
    """
    expected = [atom.element for atom in topology.atoms()]
    found = [atom.element for atom in packed.atoms()]
    for index, (want, got) in enumerate(zip(expected, found, strict=True)):
        if want is not got:
            raise SystemAssemblyError(
                f"Atom {index} is {getattr(want, 'symbol', '?')} in the "
                f"replicated topology but {getattr(got, 'symbol', '?')} in "
                f"{packed_pdb}. packmol wrote the molecules in a different "
                "order from the one they were given in."
            )


def check_box(box_nm: Sequence[float], spec: SystemSpec) -> None:
    """Raise unless the cell is large enough for the cutoff.

    OpenMM refuses a cutoff over half the box - and refuses it again, mid-run,
    when the barostat has shrunk the cell far enough. That second failure is
    the expensive one, so this is checked against the density the cell is
    heading for rather than the one it starts at.

    Args:
        box_nm: The cell edges.
        spec: The System settings, for the cutoff and the margin.

    Raises:
        SystemAssemblyError: The cell is too small.
    """
    required = spec.minimum_box_factor * spec.nonbonded_cutoff_nm
    smallest = min(box_nm)
    if smallest >= required:
        return
    raise SystemAssemblyError(
        f"The cell's shortest edge is {smallest:.2f} nm but a "
        f"{spec.nonbonded_cutoff_nm} nm cutoff needs at least "
        f"{required:.2f} nm - OpenMM's own limit is twice the cutoff, and the "
        "margin above it is for the barostat shrinking the cell during "
        "compression. Pack more chains, longer chains, or lower the cutoff."
    )


def minimum_mass_g_mol(density_g_cm3: float, spec: SystemSpec) -> float:
    """Return the least material a cell needs to be big enough for the cutoff.

    Args:
        density_g_cm3: The density the cell will reach.
        spec: The System settings, for the cutoff and the margin.

    Returns:
        The total molar mass required, in g/mol.
    """
    from .packing import AVOGADRO, NM3_PER_CM3

    edge = spec.minimum_box_factor * spec.nonbonded_cutoff_nm
    return float(edge**3 * density_g_cm3 * AVOGADRO / NM3_PER_CM3)


def prepare_box(box: PackedBox, forcefield: PolymerForceField) -> PackedBox:
    """Add the force field's virtual sites to *box*, if it declares any.

    A SMIRNOFF force field with a virtual-site handler - a sigma hole on a
    halogen, say - writes ``<VirtualSite>`` into the residue template, so the
    template describes more particles than the topology has atoms and OpenMM
    matches nothing at all. The extra particles have to be added *after*
    packing, because they would otherwise break the atom-for-atom mapping onto
    packmol's coordinates, and the System and the Simulation must both be built
    from the topology this returns rather than the one passed in.

    Args:
        box: The assembled cell.
        forcefield: The force field it will be built with.

    Returns:
        The cell, with extra particles if the force field needs them.
    """
    if not forcefield.virtual_site_residues:
        return box

    import forcefill
    from openmm import app, unit

    log.info(
        "Adding virtual sites for %s.", ", ".join(forcefield.virtual_site_residues)
    )
    openmm_forcefield = app.ForceField(*forcefield.files)
    topology, positions = forcefill.add_extra_particles(
        box.topology, box.positions, openmm_forcefield
    )
    topology.setPeriodicBoxVectors(box.topology.getPeriodicBoxVectors())
    return PackedBox(
        topology=topology,
        positions_nm=np.asarray(
            positions.value_in_unit(unit.nanometer), dtype=np.float64
        ),
        box_nm=box.box_nm,
        n_molecules=box.n_molecules,
    )


def _constraint_object(name: str) -> Any:
    """Return the ``openmm.app`` constant for a constraint setting."""
    from openmm import app

    return {
        "none": None,
        "hbonds": app.HBonds,
        "allbonds": app.AllBonds,
        "hangles": app.HAngles,
    }[name]


def build_system(
    box: PackedBox,
    forcefield: PolymerForceField,
    spec: SystemSpec | None = None,
    *,
    use_residue_templates: bool = True,
) -> Any:
    """Build the ``openmm.System`` for a packed cell.

    Args:
        box: The assembled cell, already through :func:`prepare_box`.
        forcefield: The force field to build it with.
        spec: How to build it.
        use_residue_templates: Name each residue's template explicitly.
            ``ForceField`` matches residues by graph isomorphism once per
            residue with nothing cached between them, so a cell of five hundred
            chains pays five hundred backtracking searches over a
            thousand-atom graph. Naming the template skips all of it. Falls
            back to the search if the names do not line up.

    Returns:
        The System, with a barostat still to be added by whichever stage wants
        one.

    Raises:
        SystemAssemblyError: The cell is too small for the cutoff, or the System would
            not build.
    """
    from openmm import app, unit

    settings = spec or SystemSpec()
    check_box(box.box_nm, settings)

    openmm_forcefield = app.ForceField(*forcefield.files)
    kwargs: dict[str, Any] = {
        # A melt is periodic, always.
        "nonbondedMethod": app.PME,
        "nonbondedCutoff": settings.nonbonded_cutoff_nm * unit.nanometer,
        "constraints": _constraint_object(settings.constraints),
        "rigidWater": settings.rigid_water,
        "removeCMMotion": settings.remove_cm_motion,
        "ewaldErrorTolerance": settings.ewald_error_tolerance,
        # Explicitly, and never as None: createSystem coerces this with
        # bool(), so None would silently switch the correction off. It is
        # worth one to three per cent on the melt density, which is the number
        # most of these runs exist to produce.
        "useDispersionCorrection": settings.use_dispersion_correction,
    }
    if settings.switch_distance_nm is not None:
        kwargs["switchDistance"] = settings.switch_distance_nm * unit.nanometer
    if settings.hydrogen_mass_amu is not None:
        kwargs["hydrogenMass"] = settings.hydrogen_mass_amu * unit.amu
    if use_residue_templates:
        kwargs["residueTemplates"] = {
            residue: residue.name for residue in box.topology.residues()
        }

    system = _create_system(openmm_forcefield, box.topology, kwargs)
    _verify_dispersion_correction(system, settings)
    system.setDefaultPeriodicBoxVectors(*box.topology.getPeriodicBoxVectors())
    log.info(
        "Built a System of %d particles in a %.2f x %.2f x %.2f nm cell.",
        system.getNumParticles(),
        *box.box_nm,
    )
    return system


def _create_system(forcefield: Any, topology: Any, kwargs: dict[str, Any]) -> Any:
    """Call ``createSystem``, retrying past the two shortcuts that can conflict.

    Both retries are for settings this package adds for speed or for
    correctness that a particular force-field file may already have an opinion
    about. Anything else is the caller's problem, and is re-raised with the
    context OpenMM's own message leaves out.
    """
    for _ in range(3):
        try:
            return forcefield.createSystem(topology, **kwargs)
        except Exception as error:
            message = str(error)
            if (
                "useDispersionCorrection" in message
                and "useDispersionCorrection" in kwargs
            ):
                # The file states one policy and the spec another. OpenMM
                # refuses rather than choosing; the file wins, and says so.
                log.warning(
                    "The force-field file sets its own dispersion-correction "
                    "policy, which overrides the one requested here: %s",
                    error,
                )
                kwargs.pop("useDispersionCorrection")
                continue
            if "residueTemplates" in kwargs:
                log.warning(
                    "Naming residue templates explicitly did not work (%s); "
                    "falling back to matching them by graph, which is slower.",
                    error,
                )
                kwargs.pop("residueTemplates")
                continue
            raise SystemAssemblyError(
                f"The System would not build from "
                f"{topology.getNumResidues()} residues: {error}"
            ) from error
    raise SystemAssemblyError(  # pragma: no cover - the loop always returns or raises
        "The System would not build after exhausting every fallback."
    )


def _verify_dispersion_correction(system: Any, spec: SystemSpec) -> None:
    """Log the correction that is actually in force, whatever was asked for."""
    import openmm as mm

    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            actual = force.getUseDispersionCorrection()
            if actual != spec.use_dispersion_correction:
                log.warning(
                    "The long-range dispersion correction is %s, not the %s "
                    "that was asked for. Melt densities shift by one to three "
                    "per cent between the two.",
                    "on" if actual else "off",
                    "on" if spec.use_dispersion_correction else "off",
                )
            return


def make_barostat(
    kind: str,
    temperature_k: float,
    pressure_bar: float,
    frequency: int,
    seed: int,
    *,
    scale_molecules_as_rigid: bool = True,
) -> Any:
    """Build a barostat.

    Args:
        kind: One of :data:`BAROSTATS`.
        temperature_k: The temperature its Metropolis test uses. This must
            agree with the integrator's; they are set separately and nothing
            checks that they match.
        pressure_bar: The pressure.
        frequency: Steps between volume moves.
        seed: Its random seed. Must be non-zero, or OpenMM chooses its own and
            the run stops being reproducible.
        scale_molecules_as_rigid: Whether a volume move translates each
            molecule rigidly. See :class:`SystemSpec` for why this is on by
            default and why turning it off needs ``constraints="none"``.

    Returns:
        The barostat force, not yet added to a System.
    """
    import openmm as mm
    from openmm import unit

    require_choice(kind, BAROSTATS, name="kind")
    if kind == "isotropic":
        barostat = mm.MonteCarloBarostat(
            pressure_bar * unit.bar, temperature_k * unit.kelvin, frequency
        )
    else:
        barostat = mm.MonteCarloAnisotropicBarostat(
            mm.Vec3(pressure_bar, pressure_bar, pressure_bar) * unit.bar,
            temperature_k * unit.kelvin,
            True,
            True,
            True,
            frequency,
        )
    barostat.setScaleMoleculesAsRigid(scale_molecules_as_rigid)
    seed_random_stream(barostat, seed)
    return barostat


def barostat_kind(system: Any) -> str | None:
    """Return the kind of barostat in *system*, or None if it has none.

    Raises:
        SystemAssemblyError: The System carries more than one barostat. OpenMM accepts
            that without complaint and then applies both.
    """
    import openmm as mm

    found = [
        "isotropic" if isinstance(force, mm.MonteCarloBarostat) else "anisotropic"
        for force in system.getForces()
        if isinstance(force, mm.MonteCarloBarostat | mm.MonteCarloAnisotropicBarostat)
    ]
    if len(found) > 1:
        raise SystemAssemblyError(
            f"The System carries {len(found)} barostats. OpenMM applies every "
            "one of them without complaining, which is not a pressure anyone "
            "meant to simulate."
        )
    return found[0] if found else None


def platform_is_usable(name: str) -> bool:
    """Whether a Context can actually be built on the named platform.

    Being listed is not the same as working. A CUDA build compiled against a
    newer toolkit than the installed driver appears in the platform list and
    then fails with ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` the moment a
    Context is made - which, without this, is at the start of the first stage
    rather than at platform selection.

    Args:
        name: The platform to try.

    Returns:
        Whether a one-particle Context could be built on it.
    """
    import openmm as mm
    from openmm import unit

    system = mm.System()
    system.addParticle(1.0 * unit.dalton)
    try:
        mm.Context(
            system,
            mm.VerletIntegrator(0.001 * unit.picoseconds),
            mm.Platform.getPlatformByName(name),
        )
    except Exception as error:
        log.info("The %s platform is present but not usable: %s", name, error)
        return False
    return True


def select_platform(
    name: str | None = None, precision: str = "mixed"
) -> tuple[Any, dict[str, str]]:
    """Choose an OpenMM platform and its properties.

    Args:
        name: A platform name, or None to take the fastest that works. Named
            explicitly, the platform is used as asked and any failure is the
            caller's to see; chosen automatically, each candidate is tried
            before it is picked.
        precision: ``Precision`` for the GPU platforms. Mixed rather than
            single, deliberately: these runs are long enough that single
            precision drifts, and double costs more than the accuracy is
            worth.

    Returns:
        The platform and the property dictionary to pass alongside it.

    Raises:
        SystemAssemblyError: The named platform does not exist, or nothing
            available works.
    """
    import openmm as mm

    available = {
        mm.Platform.getPlatform(index).getName()
        for index in range(mm.Platform.getNumPlatforms())
    }
    if name is not None:
        if name not in available:
            raise SystemAssemblyError(
                f"Platform {name!r} is not available; this build has "
                f"{', '.join(sorted(available))}."
            )
        chosen = name
    else:
        chosen = next(
            (
                candidate
                for candidate in PLATFORM_PREFERENCE
                if candidate in available and platform_is_usable(candidate)
            ),
            "",
        )
        if not chosen:
            raise SystemAssemblyError(  # pragma: no cover - Reference always works
                "No available platform could build a Context. This build has "
                f"{', '.join(sorted(available))}."
            )

    properties = {"Precision": precision} if chosen in _PRECISION_PLATFORMS else {}
    log.info(
        "Using the %s platform%s.", chosen, f" ({precision})" if properties else ""
    )
    return mm.Platform.getPlatformByName(chosen), properties


def check_target_density(
    counts: Sequence[int],
    molar_masses_g_mol: Sequence[float],
    target_density_g_cm3: float,
    spec: SystemSpec | None = None,
) -> float:
    """Check the cell will still be big enough once it has been compressed.

    The cell that has to satisfy the cutoff is not the loose one packmol
    filled, it is the dense one the barostat is heading for. Checking the
    packing cell alone passes a run that dies hours later, mid-compression,
    with OpenMM's own cutoff-versus-box exception.

    Args:
        counts: How many of each component.
        molar_masses_g_mol: Each component's molar mass.
        target_density_g_cm3: The density the run will reach.
        spec: The System settings, for the cutoff and the margin.

    Returns:
        The compressed cell edge, in nanometres.

    Raises:
        SystemAssemblyError: The compressed cell would be too small, with the
            extra material needed named in the message.
    """
    from .packing import box_edge_nm

    settings = spec or SystemSpec()
    edge = box_edge_nm(counts, molar_masses_g_mol, target_density_g_cm3)
    required = settings.minimum_box_factor * settings.nonbonded_cutoff_nm
    if edge >= required:
        return edge

    have = sum(
        count * mass for count, mass in zip(counts, molar_masses_g_mol, strict=True)
    )
    need = minimum_mass_g_mol(target_density_g_cm3, settings)
    raise SystemAssemblyError(
        f"At {target_density_g_cm3} g/cm3 this cell compresses to "
        f"{edge:.2f} nm, below the {required:.2f} nm a "
        f"{settings.nonbonded_cutoff_nm} nm cutoff needs. The cell holds "
        f"{have:.0f} g/mol and needs at least {need:.0f} g/mol: use "
        f"{need / have:.1f} times as many chains, or longer ones, or a "
        "shorter cutoff."
    )


def check_timestep(timestep_fs: float, temperature_k: float, spec: SystemSpec) -> None:
    """Raise unless the timestep is safe for these constraints and temperature.

    Hydrogen mass repartitioning and the timestep have to be chosen together,
    and the pairing gets looser as the mass goes up and tighter as the
    temperature does. A melt stage at 600 K with the 2 fs that was fine at 300 K
    is the most likely source of a run that dies of a NaN.

    Args:
        timestep_fs: The step being proposed.
        temperature_k: The temperature it will run at.
        spec: The System settings, for the constraints and hydrogen mass.

    Raises:
        ValueError: The step is too long.
    """
    limit = max_timestep_fs(temperature_k, spec)
    if timestep_fs > limit + 1e-9:
        raise ValueError(
            f"timestep_fs={timestep_fs} is too long for {temperature_k:.0f} K "
            f"with constraints={spec.constraints!r} and "
            f"hydrogen_mass_amu={spec.hydrogen_mass_amu}: the limit is "
            f"{limit:.2f} fs. Shorten the step, repartition hydrogen mass, or "
            "run cooler."
        )


#: Reference temperature the timestep limits below are quoted at.
_TIMESTEP_REFERENCE_K = 300.0


def max_timestep_fs(temperature_k: float, spec: SystemSpec) -> float:
    """Return the longest safe timestep, in femtoseconds.

    Args:
        temperature_k: The temperature the stage runs at.
        spec: The System settings.

    Returns:
        The limit in femtoseconds. Constrained hydrogens allow 2 fs; 1.5 amu of
        repartitioning allows 3; 3.5 amu or more allows 4. Above the reference
        temperature each is derated by the square root of the ratio, because
        that is how the fastest velocities scale.
    """
    if spec.constraints == "none":
        base = 1.0
    elif spec.hydrogen_mass_amu is None:
        base = 2.0
    elif spec.hydrogen_mass_amu >= 3.5:
        base = 4.0
    elif spec.hydrogen_mass_amu >= 1.5:
        base = 3.0
    else:
        base = 2.0
    if temperature_k <= _TIMESTEP_REFERENCE_K:
        return base
    return float(base * (_TIMESTEP_REFERENCE_K / temperature_k) ** 0.5)
