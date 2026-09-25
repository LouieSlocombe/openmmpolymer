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
includes virtual site parameters" - on every run. It matters only if the polymer
gets virtual sites, which :attr:`PolymerForceField.virtual_site_residues`
reports and :func:`openmmpolymer.mdsystem.prepare_box` acts on. Under
``python -W error`` the warning is an exception, so a caller running that way
has to filter it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._files import write_atomically
from ._validation import require_choice
from .charges import read_chain_molecule

log = logging.getLogger(__name__)

#: The force fields the generated XML is meant to be loaded beside.
DEFAULT_BASE_FORCEFIELD = ("amber14-all.xml", "amber14/tip3p.xml")

#: Backends forcefill offers that make sense for a whole polymer chain.
#: ``espaloma`` is omitted because it pulls in PyTorch, and the NAGL charges
#: this package assigns already cover what it would be reached for. ``charmm``
#: is omitted because it converts a CGenFF stream written for the whole chain,
#: which nothing here can produce, and takes the charges from that stream
#: rather than from :mod:`openmmpolymer.charges`.
BACKENDS = ("smirnoff", "gaff")

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


def cache_key(molecule: Any, options: Mapping[str, Any]) -> str:
    """Return the cache key for parameterising *molecule* under *options*.

    The key covers the molecule's canonical SMILES and its partial charges,
    which are parameters too. Coordinates are deliberately left out: two
    conformers of the same chain give the same parameters, so they share one
    cache entry and one expensive parameterisation.

    Args:
        molecule: The chain, as
            :func:`openmmpolymer.charges.read_chain_molecule` returns it.
        options: Everything else that changes the answer - backend,
            force-field selection, residue name, and the forcefill version.

    Returns:
        A hex digest.
    """
    charges = (
        ()
        if molecule.partial_charges is None
        else tuple(round(float(value), 6) for value in molecule.partial_charges.m)
    )
    payload = json.dumps(
        {"smiles": molecule.to_smiles(), "charges": charges, **options},
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
    smirnoff_forcefield: str = DEFAULT_SMIRNOFF_FORCEFIELD,
    cache_dir: str | Path | None = None,
    workdir: str | Path | None = None,
) -> PolymerForceField:
    """Parameterise one polymer chain into an OpenMM force-field XML.

    forcefill checks the result before it is returned, by building a System
    for the single chain and minimising it. The minimisation is what catches a
    NaN charge or a zero force constant, which a System build accepts and which
    otherwise surfaces later as an exploding simulation.

    Args:
        chain_sdf: One conformer's SDF, as
            :func:`openmmpolymer.chain.build_chain` writes and
            :func:`openmmpolymer.charges.assign_charges` charges. Bond orders
            are why this is an SDF and not a PDB.
        output_xml: Where the force field is written.
        residue_name: Residue name for the template. Must match the chain PDB.
        backend: One of :data:`BACKENDS`.
        smirnoff_forcefield: SMIRNOFF release, for ``backend="smirnoff"``.
        cache_dir: Where built force fields are kept, keyed by
            :func:`cache_key`. Parameterising a long chain is minutes; nothing
            about it depends on the run, so a cache hit is the difference
            between iterating and waiting.
        workdir: forcefill's intermediate directory. Kept on failure, because
            ``sqm.out`` is the post-mortem.

    Returns:
        The force field, ready to hand to :mod:`openmmpolymer.mdsystem`.

    Raises:
        ForceFieldError: The SDF does not hold one molecule, forcefill declined
            the chain, or the result does not build a working System.
    """
    require_choice(backend, BACKENDS, name="backend")
    source = Path(chain_sdf)
    destination = Path(output_xml)

    import forcefill

    options = {
        "backend": backend,
        "base_forcefield": list(DEFAULT_BASE_FORCEFIELD),
        "smirnoff_forcefield": smirnoff_forcefield,
        "residue_name": residue_name,
        "forcefill": forcefill.__version__,
    }
    molecule = read_chain_molecule(source, ForceFieldError)
    entry = (
        None
        if cache_dir is None
        else Path(cache_dir) / f"{cache_key(molecule, options)}.xml"
    )
    if entry is not None and entry.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry, destination)
        log.info("Reusing cached force field for %s.", residue_name)
        written, cached = destination, True
    else:
        if backend == "smirnoff" and molecule.partial_charges is None:
            log.warning(
                "%s carries no partial charges, so the smirnoff backend will "
                "try AM1-BCC. On a polymer chain that is slow at best, and "
                "unavailable entirely without an AmberTools toolkit wrapper. "
                "Run openmmpolymer.charges.assign_charges(..., 'nagl') on the "
                "SDF first.",
                source.name,
            )
        log.info(
            "Parameterising %s from %s with the %s backend.",
            residue_name,
            source.name,
            backend,
        )
        try:
            result = forcefill.build_ligand_xml(
                {residue_name: forcefill.LigandSpec(file=str(source))},
                str(destination),
                base_forcefield=DEFAULT_BASE_FORCEFIELD,
                backend=backend,
                smirnoff_forcefield=smirnoff_forcefield,
                workdir=None if workdir is None else str(workdir),
                validate=True,
                minimize=True,
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
        written, cached = Path(result.forcefield_xml), False
        if entry is not None:
            entry.parent.mkdir(parents=True, exist_ok=True)
            write_atomically(entry, written.read_text())
            write_atomically(
                entry.with_suffix(".json"),
                json.dumps(options, indent=2, sort_keys=True, default=str) + "\n",
            )

    return PolymerForceField(
        forcefield_xml=str(written),
        base_forcefield=DEFAULT_BASE_FORCEFIELD,
        residue_name=residue_name,
        backend=backend,
        virtual_site_residues=tuple(
            forcefill.residue_templates_with_virtual_sites(str(written))
        ),
        cached=cached,
    )
