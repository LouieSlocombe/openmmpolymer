"""Turning a chain SDF into an OpenMM force-field XML, through forcefill.

forcefill will not parameterise a residue bonded to its neighbours - it says so
by name, and it is right to: a stand-alone GAFF treatment of a chain-linked
residue is not valid. That is why :mod:`openmmpolymer.chain` builds the whole
chain as one molecule. Here it goes through ``build_ligand_xml`` as a single
residue, and comes back as an ordinary ffxml to load next to amber14.

Two details of that call are not optional. The ligand mapping is always passed
in its explicit ``{name: LigandSpec}`` form, because the bare-path form derives
a residue name by truncating the file stem to three characters, and two chains
whose names truncate alike is a hard failure rather than a warning. And the
default backend is ``smirnoff`` rather than ``gaff``: it is the one that
honours the charges :mod:`openmmpolymer.charges` already assigned, and it does
not route the molecule through antechamber, which reorders and renames atoms by
default.

One consequence of that pairing is worth knowing before it surprises anyone.
OpenFF Sage carries virtual-site parameters, so handing it preset charges makes
it warn - "Preset charges were provided ... alongside a force field that
includes virtual site parameters" - on every run. It is telling you that a
virtual site would take its charge from the force field rather than from the
preset set, which matters only if the polymer gets any.
:attr:`PolymerForceField.virtual_site_residues` is where to look, and
:func:`openmmpolymer.mdsystem.prepare_box` is what acts on it. Under
``python -W error`` the warning is an exception, so a caller running that way
has to filter it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._validation import require_choice

log = logging.getLogger(__name__)

#: The force fields the generated XML is meant to be loaded beside.
DEFAULT_BASE_FORCEFIELD = ("amber14-all.xml", "amber14/tip3p.xml")

#: Backends forcefill offers that make sense for a whole polymer chain.
#: ``espaloma`` is omitted deliberately: it pulls in PyTorch, and the NAGL
#: charges this package assigns already cover what it would be reached for.
BACKENDS = ("smirnoff", "gaff", "charmm")

#: The SMIRNOFF release used when the backend is ``smirnoff``.
DEFAULT_SMIRNOFF_FORCEFIELD = "openff-2.2.1"


class ForceFieldError(RuntimeError):
    """A polymer force field could not be built or does not work."""


@dataclass(frozen=True)
class PolymerForceField:
    """A force field that can build a System for this polymer.

    Args:
        forcefield_xml: Path to the generated XML. Load this *or* the
            per-residue files forcefill also wrote, never both: the duplicated
            atom-type definitions collide.
        base_forcefield: The files it is loaded beside.
        residue_name: The residue the template covers.
        backend: Which forcefill backend produced it.
        virtual_site_residues: Templates declaring virtual sites. When this is
            non-empty the box topology needs extra particles adding before
            ``createSystem``, and ``computeVirtualSites()`` calling after
            positions are set.
        cached: Whether this came from the cache rather than being built.
    """

    forcefield_xml: str
    base_forcefield: tuple[str, ...]
    residue_name: str
    backend: str
    virtual_site_residues: tuple[str, ...] = ()
    cached: bool = False

    @property
    def files(self) -> tuple[str, ...]:
        """Every file to hand ``openmm.app.ForceField``, in order."""
        return (*self.base_forcefield, self.forcefield_xml)


def _molecule_fingerprint(sdf_path: Path) -> tuple[str, tuple[float, ...]]:
    """Return the SDF's canonical SMILES and its partial charges.

    Coordinates are deliberately left out. Two conformers of the same chain
    give the same parameters, so they should share one cache entry and one
    expensive parameterisation.
    """
    from openff.toolkit import Molecule

    molecule = Molecule.from_file(str(sdf_path), allow_undefined_stereo=True)
    if isinstance(molecule, list):
        if len(molecule) != 1:
            raise ForceFieldError(
                f"{sdf_path} holds {len(molecule)} molecules; a chain SDF "
                "holds exactly one."
            )
        molecule = molecule[0]
    charges = (
        ()
        if molecule.partial_charges is None
        else tuple(round(float(value), 6) for value in molecule.partial_charges.m)
    )
    return molecule.to_smiles(), charges


def cache_key(sdf_path: str | Path, options: dict[str, Any]) -> str:
    """Return the cache key for parameterising *sdf_path* under *options*.

    Args:
        sdf_path: The chain SDF.
        options: Everything that changes the answer - backend, force-field
            selection, residue name, and the forcefill version.

    Returns:
        A hex digest.
    """
    smiles, charges = _molecule_fingerprint(Path(sdf_path))
    payload = json.dumps(
        {"smiles": smiles, "charges": charges, **options},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_polymer_forcefield(
    chain_sdf: str | Path,
    output_xml: str | Path = "polymer_ff.xml",
    *,
    residue_name: str = "POL",
    backend: str = "smirnoff",
    base_forcefield: Sequence[str] = DEFAULT_BASE_FORCEFIELD,
    smirnoff_forcefield: str = DEFAULT_SMIRNOFF_FORCEFIELD,
    charmm_files: Sequence[str | Path] = (),
    cache_dir: str | Path | None = None,
    workdir: str | Path | None = None,
    validate: bool = True,
    minimize: bool = True,
) -> PolymerForceField:
    """Parameterise one polymer chain into an OpenMM force-field XML.

    Args:
        chain_sdf: One conformer's SDF, as
            :func:`openmmpolymer.chain.build_chain` writes and
            :func:`openmmpolymer.charges.assign_charges` charges. Bond orders
            are why this is an SDF and not a PDB.
        output_xml: Where the force field is written.
        residue_name: Residue name for the template. Must match the chain PDB.
        backend: One of :data:`BACKENDS`.
        base_forcefield: The files the result is loaded beside.
        smirnoff_forcefield: SMIRNOFF release, for ``backend="smirnoff"``.
        charmm_files: CGenFF stream files, for ``backend="charmm"``.
        cache_dir: Where built force fields are kept. Parameterising a long
            chain is minutes; nothing about it depends on the run, so a cache
            hit is the difference between iterating and waiting.
        workdir: forcefill's intermediate directory. Kept on failure, because
            ``sqm.out`` is the post-mortem.
        validate: Build a System for the single chain before returning.
        minimize: Also minimise it. This is the check ``validate`` cannot make:
            a NaN charge or a zero force constant survives a System build and
            surfaces later as an exploding simulation.

    Returns:
        The force field, ready to hand to :mod:`openmmpolymer.mdsystem`.

    Raises:
        ForceFieldError: forcefill declined the chain, or the result does not
            build a System.
    """
    require_choice(backend, BACKENDS, name="backend")
    source = Path(chain_sdf)
    destination = Path(output_xml)
    base = tuple(base_forcefield)

    import forcefill

    options = {
        "backend": backend,
        "base_forcefield": list(base),
        "smirnoff_forcefield": smirnoff_forcefield,
        "residue_name": residue_name,
        "forcefill": getattr(forcefill, "__version__", "unknown"),
    }
    _warn_if_uncharged(source, backend)
    cached_path = _cache_lookup(cache_dir, source, options)
    if cached_path is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_path, destination)
        log.info("Reusing cached force field for %s.", residue_name)
        return _describe(
            destination, base, residue_name, backend, forcefill, cached=True
        )

    log.info(
        "Parameterising %s from %s with the %s backend.",
        residue_name,
        source.name,
        backend,
    )
    spec = forcefill.LigandSpec(
        file=str(source),
        backend=backend,
        forcefield=smirnoff_forcefield if backend == "smirnoff" else None,
        charmm_files=tuple(str(path) for path in charmm_files),
    )
    try:
        result = forcefill.build_ligand_xml(
            {residue_name: spec},
            str(destination),
            base_forcefield=base,
            backend=backend,
            smirnoff_forcefield=smirnoff_forcefield,
            workdir=None if workdir is None else str(workdir),
            validate=validate,
            minimize=minimize,
        )
    except Exception as error:
        raise ForceFieldError(
            f"forcefill could not parameterise {residue_name} from "
            f"{source.name}: {error}"
        ) from error

    if result.forcefield_xml is None:  # pragma: no cover - needs a stock residue
        raise ForceFieldError(
            f"forcefill matched {residue_name} against the base force field "
            "and wrote nothing. A polymer chain is not a standard residue; "
            "check residue_name."
        )

    _cache_store(cache_dir, source, options, Path(result.forcefield_xml))
    return _describe(
        Path(result.forcefield_xml), base, residue_name, backend, forcefill
    )


def _warn_if_uncharged(source: Path, backend: str) -> None:
    """Say so when a chain is about to be sent off for AM1-BCC charges.

    With no charges on the molecule, ``openmmforcefields`` falls back to
    AM1-BCC, which runs ``sqm``. On a chain that is minutes at best, hours at
    worst, and on an installation without the AmberTools toolkit wrapper
    registered it simply is not available - a failure that arrives after the
    call rather than before it. Charging the SDF first avoids all of that.
    """
    if backend != "smirnoff":
        return
    _, charges = _molecule_fingerprint(source)
    if charges:
        return
    log.warning(
        "%s carries no partial charges, so the smirnoff backend will try "
        "AM1-BCC. On a polymer chain that is slow at best, and unavailable "
        "entirely without an AmberTools toolkit wrapper. Run "
        "openmmpolymer.charges.assign_charges(..., 'nagl') on the SDF first.",
        source.name,
    )


def _describe(
    path: Path,
    base: tuple[str, ...],
    residue_name: str,
    backend: str,
    forcefill: Any,
    *,
    cached: bool = False,
) -> PolymerForceField:
    """Wrap a written ffxml, noting whether it declares virtual sites."""
    return PolymerForceField(
        forcefield_xml=str(path),
        base_forcefield=base,
        residue_name=residue_name,
        backend=backend,
        virtual_site_residues=tuple(
            forcefill.residue_templates_with_virtual_sites(str(path))
        ),
        cached=cached,
    )


def _cache_lookup(
    cache_dir: str | Path | None, source: Path, options: dict[str, Any]
) -> Path | None:
    """Return the cached force field for these inputs, if there is one."""
    if cache_dir is None:
        return None
    candidate = Path(cache_dir) / f"{cache_key(source, options)}.xml"
    return candidate if candidate.is_file() else None


def _cache_store(
    cache_dir: str | Path | None,
    source: Path,
    options: dict[str, Any],
    built: Path,
) -> None:
    """Record a freshly built force field in the cache."""
    if cache_dir is None:
        return
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    key = cache_key(source, options)
    shutil.copyfile(built, directory / f"{key}.xml")
    (directory / f"{key}.json").write_text(
        json.dumps(options, indent=2, sort_keys=True, default=str) + "\n"
    )


def check_forcefield(
    forcefield: PolymerForceField, chain_pdb: str | Path, *, minimize: bool = True
) -> Any:
    """Build, and optionally minimise, a System for one chain.

    Worth doing before packing rather than after: it costs milliseconds and it
    catches the failures that otherwise surface hours into a run.

    Args:
        forcefield: What to check.
        chain_pdb: One chain's PDB. Its CONECT records carry the bonds the
            template is matched against.
        minimize: Also minimise, and return the result.

    Returns:
        A ``forcefill.MinimizationResult`` when *minimize*, else None.

    Raises:
        ForceFieldError: The System would not build, or the geometry is not
            physical.
    """
    import forcefill
    from openmm import app

    pdb = app.PDBFile(str(chain_pdb))
    try:
        forcefill.validate_forcefield_xml(
            pdb.topology,
            forcefield.forcefield_xml,
            forcefield.base_forcefield,
        )
        if not minimize:
            return None
        return forcefill.minimize_with_forcefield_xml(
            pdb.topology,
            pdb.positions,
            forcefield.forcefield_xml,
            forcefield.base_forcefield,
        )
    except Exception as error:
        raise ForceFieldError(
            f"{forcefield.forcefield_xml} does not build a working System for "
            f"{Path(chain_pdb).name}: {error}"
        ) from error
