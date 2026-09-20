"""Building one polymer chain as a single free-standing molecule.

forcefill refuses to parameterise a residue that is covalently bonded to its
neighbours, so a chain written as N monomer residues cannot be parameterised at
all. The whole package therefore rests on one shape: **one chain is one
molecule and one residue**, built here and handed to
:mod:`openmmpolymer.forcefield` as an SDF.

A monomer is given as a SMILES carrying exactly two ``[*]`` attachment points -
the PSMILES convention - so ``[*]CC[*]`` is polyethylene and
``[*]CC([*])c1ccccc1`` is polystyrene. The first dummy in atom order is the
head, the last is the tail, and units are joined head-to-tail.

Two embedders live here because they fail in opposite directions. RDKit's
ETKDG collapses a long chain into a globule; a hard-core self-avoiding walk
swells it past the melt's unperturbed dimensions. :func:`build_chain`
measures the characteristic ratio it actually produced and says so when it
drifts, rather than trusting either.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._seeds import derive_seed
from ._validation import require_choice, require_integer, require_positive

log = logging.getLogger(__name__)

#: How a monomer's two open valences are marked.
DUMMY_ATOMIC_NUMBER = 0

#: Accepted values for ``ChainSpec.tacticity``.
TACTICITIES = ("atactic", "isotactic", "syndiotactic")

#: The cap that means "just fill the valence with hydrogen".
HYDROGEN_CAP = "[*][H]"

#: Backbone torsion states the grown embedder draws from, in degrees: trans,
#: gauche+, gauche-. These are absolute targets - each placement measures the
#: dihedral it produced and rotates it onto one of these - so nothing depends
#: on whatever conformation the relaxed template oligomer happened to settle in.
_TORSION_TARGETS_DEG = (180.0, 60.0, -60.0)

#: Ways to produce 3D coordinates. ``auto`` grows anything long enough to
#: stamp a template down - four units - and falls back to ETKDG below that.
EMBEDDERS = ("auto", "etkdg", "grown")

#: Shortest chain the grown embedder can build: it needs a first junction, an
#: interior one and a last one to copy.
_MIN_GROWN_UNITS = 4

#: Closest approach allowed between heavy atoms of non-adjacent units during
#: growth, in angstrom. Below a real van der Waals contact, because the geometry
#: is relaxed afterwards and a stricter test rejects almost everything.
_OVERLAP_ANGSTROM = 2.6

#: Fraction by which a measured characteristic ratio may differ from the
#: expected one before :func:`build_chain` says so.
_RATIO_TOLERANCE = 0.15


class ChainError(RuntimeError):
    """A polymer chain could not be built."""


@dataclass(frozen=True)
class ChainSpec:
    """What chain to build.

    Args:
        monomer_smiles: SMILES with exactly two ``[*]`` attachment points. The
            first in atom order is the head, the last the tail.
        degree_of_polymerization: Number of repeat units.
        tacticity: One of ``atactic``, ``isotactic``, ``syndiotactic``. A no-op
            for a monomer with no backbone stereocentre.
        head_cap: SMILES fragment with one ``[*]`` closing the first unit's open
            head, or None to choose one from the attachment chemistry.
        tail_cap: The same for the last unit's open tail.
        residue_name: Residue name for the PDB and the force-field template.
            Three characters, because forcefill truncates to three and a
            collision there is a hard failure.
        characteristic_ratio: The chain's expected C-infinity, used to set the
            grown embedder's trans fraction and to check what it produced.
            The default is polyethylene's.
        seed: Master seed. Every conformer draws its own from this.
    """

    monomer_smiles: str
    degree_of_polymerization: int = 20
    tacticity: str = "atactic"
    head_cap: str | None = None
    tail_cap: str | None = None
    residue_name: str = "POL"
    characteristic_ratio: float = 7.0
    seed: int = 0xF0

    def __post_init__(self) -> None:
        """Reject a malformed spec at construction, not mid-build."""
        require_integer(self.degree_of_polymerization, name="degree_of_polymerization")
        require_choice(self.tacticity, TACTICITIES, name="tacticity")
        require_positive(self.characteristic_ratio, None, name="characteristic_ratio")
        if not 1 <= len(self.residue_name) <= 3:
            raise ValueError(
                f"residue_name={self.residue_name!r} must be one to three "
                "characters: forcefill truncates residue names to three, and "
                "two names that truncate alike are a hard failure there."
            )
        if not self.residue_name.isalnum():
            raise ValueError(
                f"residue_name={self.residue_name!r} must be alphanumeric."
            )


@dataclass(frozen=True)
class ChainResult:
    """What :func:`build_chain` produced.

    Args:
        sdf_paths: One SDF per conformer, carrying bond orders. This is what
            forcefill's smirnoff backend needs; a PDB records no bond orders.
        pdb_paths: One PDB per conformer, in the same atom order as the SDF.
            This is what packmol reads.
        smiles: Canonical SMILES of the capped chain, used as the cache key for
            parameterisation.
        n_atoms: Atoms per chain, hydrogens included.
        molar_mass_g_mol: Chain molar mass.
        radius_of_gyration_nm: Per-conformer radius of gyration.
        max_extent_nm: Per-conformer largest interatomic distance. packmol
            cannot place a conformer longer than the box.
        backbone: Backbone atom indices within one chain, head to tail. The
            run manifest measures chain dimensions with these, and nothing
            downstream can recover them: they come from the attachment points,
            which the caps consumed.
        characteristic_ratio: Measured over the conformers, or None when the
            chain is too short for the number to mean anything.
        embedder: Which embedder ran, ``"etkdg"`` or ``"grown"``.
    """

    sdf_paths: tuple[str, ...]
    pdb_paths: tuple[str, ...]
    smiles: str
    n_atoms: int
    molar_mass_g_mol: float
    radius_of_gyration_nm: tuple[float, ...] = field(default_factory=tuple)
    max_extent_nm: tuple[float, ...] = field(default_factory=tuple)
    backbone: tuple[int, ...] = field(default_factory=tuple)
    characteristic_ratio: float | None = None
    embedder: str = "etkdg"


def _chem() -> tuple[Any, Any]:
    """Return ``(rdkit.Chem, rdkit.Chem.AllChem)``."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    return Chem, AllChem


def _parse_monomer(smiles: str) -> Any:
    """Return the monomer molecule, checking it has exactly two dummies.

    Args:
        smiles: Monomer SMILES with two ``[*]`` attachment points.

    Returns:
        A sanitised RDKit molecule.

    Raises:
        ChainError: The SMILES does not parse, or does not carry exactly two
            attachment points each bonded to one real atom by a single bond.
    """
    Chem, _ = _chem()
    monomer = Chem.MolFromSmiles(smiles)
    if monomer is None:
        raise ChainError(f"monomer_smiles={smiles!r} is not valid SMILES.")

    dummies = [
        atom.GetIdx()
        for atom in monomer.GetAtoms()
        if atom.GetAtomicNum() == DUMMY_ATOMIC_NUMBER
    ]
    if len(dummies) != 2:
        raise ChainError(
            f"monomer_smiles={smiles!r} has {len(dummies)} attachment points; "
            "a linear repeat unit needs exactly two. Polyethylene is "
            "'[*]CC[*]', polystyrene is '[*]CC([*])c1ccccc1'."
        )

    for idx in dummies:
        atom = monomer.GetAtomWithIdx(idx)
        bonds = atom.GetBonds()
        if len(bonds) != 1:
            raise ChainError(
                f"Attachment point {idx} of {smiles!r} has {len(bonds)} bonds; "
                "each must have exactly one."
            )
        if bonds[0].GetBondType() != Chem.BondType.SINGLE:
            raise ChainError(
                f"Attachment point {idx} of {smiles!r} is not joined by a "
                "single bond. Repeat units are linked head-to-tail by single "
                "bonds."
            )

    if _attachment_neighbour(monomer, dummies[0]) == _attachment_neighbour(
        monomer, dummies[-1]
    ):
        raise ChainError(
            f"Both attachment points of {smiles!r} land on the same atom, so "
            "the repeat unit has no backbone bond of its own and the chain "
            "geometry cannot be built. Write the unit with at least two "
            "backbone atoms - polyisobutylene as '[*]CC(C)(C)[*]' rather than "
            "'[*]C(C)(C)[*]'."
        )
    return monomer


def _attachment_neighbour(mol: Any, dummy_idx: int) -> int:
    """Return the real atom the dummy at *dummy_idx* is bonded to."""
    neighbours = mol.GetAtomWithIdx(dummy_idx).GetNeighbors()
    return int(neighbours[0].GetIdx())


def _default_cap(mol: Any, dummy_idx: int, *, end: str) -> str:
    """Choose a cap for the open valence at *dummy_idx*.

    Hydrogen is right almost everywhere and wrong in one place that matters: on
    a carbonyl carbon it makes an aldehyde, so an H-capped polyester ends in
    ``-CHO`` rather than ``-COOH``, with the wrong atom types and the wrong
    terminal charges. Rather than guess a replacement, say what happened.

    Args:
        mol: The molecule holding the attachment point.
        dummy_idx: Index of the ``[*]`` atom.
        end: ``"head"`` or ``"tail"``, for the error message.

    Returns:
        A cap SMILES.

    Raises:
        ChainError: The attachment atom is a carbonyl carbon, where no default
            is safe.
    """
    attachment = mol.GetAtomWithIdx(_attachment_neighbour(mol, dummy_idx))
    if _is_carbonyl_carbon(attachment):
        raise ChainError(
            f"The {end} attachment point sits on a carbonyl carbon, where a "
            "hydrogen cap would make an aldehyde rather than an acid or an "
            f"ester. Pass {end}_cap explicitly: '[*]O' for the free acid, "
            "'[*]OC' for the methyl ester."
        )
    return HYDROGEN_CAP


def _is_carbonyl_carbon(atom: Any) -> bool:
    """Whether *atom* is a carbon double-bonded to an oxygen."""
    Chem, _ = _chem()
    if atom.GetAtomicNum() != 6:
        return False
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetAtomicNum() == 8
        for bond in atom.GetBonds()
    )


def _tagged_unit(monomer: Any, unit: int) -> Any:
    """Return a copy of *monomer* whose atoms carry their unit index and role."""
    Chem, _ = _chem()
    copy = Chem.RWMol(monomer)
    dummies = [
        atom.GetIdx()
        for atom in copy.GetAtoms()
        if atom.GetAtomicNum() == DUMMY_ATOMIC_NUMBER
    ]
    for atom in copy.GetAtoms():
        atom.SetIntProp("_omp_unit", unit)
    copy.GetAtomWithIdx(dummies[0]).SetProp("_omp_role", "head")
    copy.GetAtomWithIdx(dummies[-1]).SetProp("_omp_role", "tail")
    # The dummies are consumed by the joins, so the atoms they were bonded to
    # carry the memory of where the backbone runs. The grown embedder and the
    # characteristic-ratio measurement both need that path.
    copy.GetAtomWithIdx(_attachment_neighbour(copy, dummies[0])).SetIntProp(
        "_omp_head_anchor", 1
    )
    copy.GetAtomWithIdx(_attachment_neighbour(copy, dummies[-1])).SetIntProp(
        "_omp_tail_anchor", 1
    )
    return copy


def _find_role(mol: Any, role: str, *, start: int = 0, stop: int | None = None) -> int:
    """Return the index of the single atom tagged *role* in ``[start, stop)``."""
    stop = mol.GetNumAtoms() if stop is None else stop
    for idx in range(start, stop):
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp("_omp_role") and atom.GetProp("_omp_role") == role:
            return idx
    raise ChainError(f"No atom tagged {role!r} between {start} and {stop}.")


def _join(chain: Any, unit: Any) -> Any:
    """Bond *unit*'s head onto *chain*'s open tail and drop both dummies."""
    Chem, _ = _chem()
    offset = chain.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(chain, unit))

    tail_dummy = _find_role(combined, "tail", stop=offset)
    head_dummy = _find_role(combined, "head", start=offset)
    left = _attachment_neighbour(combined, tail_dummy)
    right = _attachment_neighbour(combined, head_dummy)
    combined.AddBond(left, right, Chem.BondType.SINGLE)

    for idx in sorted((tail_dummy, head_dummy), reverse=True):
        combined.RemoveAtom(idx)
    return combined


def _attach_cap(chain: Any, role: str, cap_smiles: str) -> Any:
    """Close the open valence tagged *role* with *cap_smiles*."""
    Chem, _ = _chem()
    dummy = _find_role(chain, role)

    if cap_smiles == HYDROGEN_CAP:
        # Removing the dummy leaves a free valence that RDKit fills with an
        # implicit hydrogen on the next sanitize, which is exactly the cap.
        chain.RemoveAtom(dummy)
        return chain

    cap = Chem.MolFromSmiles(cap_smiles)
    if cap is None:
        raise ChainError(f"cap {cap_smiles!r} is not valid SMILES.")
    cap_dummies = [
        atom.GetIdx()
        for atom in cap.GetAtoms()
        if atom.GetAtomicNum() == DUMMY_ATOMIC_NUMBER
    ]
    if len(cap_dummies) != 1:
        raise ChainError(
            f"cap {cap_smiles!r} has {len(cap_dummies)} attachment points; a "
            "cap needs exactly one, as in '[*]C' or '[*]O'."
        )

    unit = chain.GetAtomWithIdx(dummy).GetIntProp("_omp_unit")
    cap = Chem.RWMol(cap)
    for atom in cap.GetAtoms():
        atom.SetIntProp("_omp_unit", unit)

    offset = chain.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(chain, cap))
    cap_dummy = offset + cap_dummies[0]
    left = _attachment_neighbour(combined, dummy)
    right = _attachment_neighbour(combined, cap_dummy)
    combined.AddBond(left, right, Chem.BondType.SINGLE)
    for idx in sorted((dummy, cap_dummy), reverse=True):
        combined.RemoveAtom(idx)
    return combined


def assemble_chain(spec: ChainSpec, n_units: int | None = None) -> Any:
    """Build the capped chain molecule, without coordinates.

    Args:
        spec: What to build.
        n_units: Override the degree of polymerization, for building the short
            template oligomer the grown embedder needs.

    Returns:
        A sanitised RDKit molecule with implicit hydrogens, whose atoms carry
        an ``_omp_unit`` integer property.

    Raises:
        ChainError: The monomer or a cap is malformed.
    """
    Chem, _ = _chem()
    monomer = _parse_monomer(spec.monomer_smiles)
    units = spec.degree_of_polymerization if n_units is None else n_units

    chain = _tagged_unit(monomer, 0)
    for unit in range(1, units):
        chain = _join(chain, _tagged_unit(monomer, unit))

    head_cap = spec.head_cap or _default_cap(
        chain, _find_role(chain, "head"), end="head"
    )
    chain = _attach_cap(chain, "head", head_cap)
    tail_cap = spec.tail_cap or _default_cap(
        chain, _find_role(chain, "tail"), end="tail"
    )
    chain = _attach_cap(chain, "tail", tail_cap)

    molecule = chain.GetMol()
    Chem.SanitizeMol(molecule)
    _apply_tacticity(molecule, spec, units)
    return molecule


def _apply_tacticity(mol: Any, spec: ChainSpec, units: int) -> None:
    """Set backbone chiral tags according to ``spec.tacticity``.

    A monomer with no backbone stereocentre - polyethylene, say - has nothing
    to set, and that is not an error.
    """
    Chem, _ = _chem()
    Chem.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    centres: dict[int, list[Any]] = {}
    for atom in mol.GetAtoms():
        if atom.HasProp("_ChiralityPossible") and atom.HasProp("_omp_unit"):
            centres.setdefault(atom.GetIntProp("_omp_unit"), []).append(atom)
    if not centres:
        log.debug(
            "%s has no backbone stereocentre; tacticity is a no-op.", spec.residue_name
        )
        return

    rng = random.Random(derive_seed(spec.seed, "tacticity"))
    tags = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    for unit in range(units):
        if spec.tacticity == "isotactic":
            which = 0
        elif spec.tacticity == "syndiotactic":
            which = unit % 2
        else:
            which = rng.randint(0, 1)
        for atom in centres.get(unit, ()):
            atom.SetChiralTag(tags[which])
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def trans_fraction(characteristic_ratio: float, bond_angle_deg: float = 112.0) -> float:
    """Return the trans probability giving *characteristic_ratio*.

    Inverts the independent-rotation expression

        C = (1 - cos t) / (1 + cos t) * (1 + <cos f>) / (1 - <cos f>)

    for a symmetric three-state backbone, where ``<cos f> = 1.5 p - 0.5``. Two
    states at plus and minus 120 degrees contribute ``-0.5`` each, so the whole
    dependence collapses onto the trans fraction.

    Hard-coding a weight triplet instead is how a builder ends up producing
    chains a quarter too compact while its own gate reports them fine: the
    common ``(0.55, 0.225, 0.225)`` gives C = 4.3, not the 7.4 polyethylene has.

    Args:
        characteristic_ratio: The target C-infinity.
        bond_angle_deg: Backbone bond angle.

    Returns:
        A trans probability, clamped to ``[0.05, 0.95]``.
    """
    cos_theta = math.cos(math.radians(bond_angle_deg))
    angle_factor = (1.0 - cos_theta) / (1.0 + cos_theta)
    x = characteristic_ratio / angle_factor
    p = (3.0 * x - 1.0) / (3.0 * (1.0 + x))
    return min(0.95, max(0.05, p))


def characteristic_ratio(
    positions_nm: npt.NDArray[np.float64],
    backbone: Sequence[int],
    bond_length_nm: float,
) -> float:
    """Return the measured C = <R^2> / (n l^2) for one conformer.

    Args:
        positions_nm: All atom positions.
        backbone: Indices of the backbone atoms, in order along the chain.
        bond_length_nm: Mean backbone bond length.

    Returns:
        The characteristic ratio.
    """
    ends = positions_nm[backbone[-1]] - positions_nm[backbone[0]]
    n_bonds = len(backbone) - 1
    return float(ends @ ends) / (n_bonds * bond_length_nm**2)


def _anchor(mol: Any, unit: int, which: str) -> int:
    """Return the index of unit *unit*'s head or tail backbone anchor atom."""
    prop = f"_omp_{which}_anchor"
    for atom in mol.GetAtoms():
        if (
            atom.HasProp(prop)
            and atom.HasProp("_omp_unit")
            and atom.GetIntProp("_omp_unit") == unit
        ):
            return int(atom.GetIdx())
    raise ChainError(f"Unit {unit} has no {which} anchor.")


def backbone_path(mol: Any, n_units: int) -> tuple[int, ...]:
    """Return the backbone atom indices, in order from head to tail.

    Args:
        mol: An assembled chain.
        n_units: Its degree of polymerization.

    Returns:
        The shortest path through the molecule from the first unit's head
        anchor to the last unit's tail anchor, which for a linear chain is the
        backbone.
    """
    Chem, _ = _chem()
    start = _anchor(mol, 0, "head")
    end = _anchor(mol, n_units - 1, "tail")
    if start == end:
        return (start,)
    return tuple(int(idx) for idx in Chem.GetShortestPath(mol, start, end))


def _add_hydrogens(mol: Any) -> Any:
    """Return *mol* with explicit hydrogens, each inheriting its parent's unit."""
    Chem, _ = _chem()
    with_h = Chem.AddHs(mol)
    for atom in with_h.GetAtoms():
        if atom.HasProp("_omp_unit"):
            continue
        parent = atom.GetNeighbors()[0]
        atom.SetIntProp("_omp_unit", parent.GetIntProp("_omp_unit"))
    return with_h


def _unit_atoms(mol: Any, n_units: int) -> list[list[int]]:
    """Return each unit's atom indices, ascending.

    Both the chain and the template oligomer are assembled by the same code, so
    within a unit this ordering is the same in both: the unit's heavy atoms in
    construction order, then its hydrogens in parent order, because ``AddHs``
    appends. That correspondence is what lets a template unit's geometry be
    copied onto a chain unit by position.
    """
    groups: list[list[int]] = [[] for _ in range(n_units)]
    for atom in mol.GetAtoms():
        groups[atom.GetIntProp("_omp_unit")].append(int(atom.GetIdx()))
    return groups


def _frame(
    head: npt.NDArray[np.float64],
    tail: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the ``(origin, rotation)`` of a unit's local frame.

    The frame sits at the unit's tail anchor, points its first axis along the
    unit's own backbone bond, and takes its second axis from *reference* - the
    previous unit's head anchor, which is what makes the frame carry the
    backbone torsion rather than an arbitrary spin.
    """
    e1 = tail - head
    e1 /= np.linalg.norm(e1)
    ref = reference - tail
    ref -= (ref @ e1) * e1
    norm = float(np.linalg.norm(ref))
    if norm < 1e-6:  # pragma: no cover - collinear reference, vanishingly rare
        ref = np.cross(e1, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(ref) < 1e-6:
            ref = np.cross(e1, np.array([0.0, 1.0, 0.0]))
        norm = float(np.linalg.norm(ref))
    e2 = ref / norm
    return tail, np.column_stack((e1, e2, np.cross(e1, e2)))


#: The reference direction used for the very first unit, which has no
#: predecessor to take one from. Any fixed choice does, as long as the template
#: and the chain make the same one.
_SEED_REFERENCE = np.array([0.0, 1.0, 0.0])


def _axis_rotation(
    axis: npt.NDArray[np.float64], angle_rad: float
) -> npt.NDArray[np.float64]:
    """Return the rotation matrix about *axis* (unit length) by *angle_rad*."""
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def _dihedral(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
) -> float:
    """Return the a-b-c-d dihedral in degrees."""
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return math.degrees(math.atan2((np.cross(b1, v) @ w), v @ w))


def _to_local(
    frame: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Express world *points* in *frame*."""
    origin, rotation = frame
    return (points - origin) @ rotation


def _to_world(
    frame: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Express local *points* in world coordinates."""
    origin, rotation = frame
    return origin + points @ rotation.T


@dataclass(frozen=True)
class _Template:
    """One unit's geometry, ready to be stamped down the chain.

    Each block holds a unit's atom positions in the *previous* unit's local
    frame, so placing a unit is one rigid transform plus the sampled backbone
    torsion. The geometry itself - bond lengths, bond angles, side-group
    placement - comes from a force-field-relaxed tetramer, so none of it has to
    be tabulated here.
    """

    first: npt.NDArray[np.float64]
    second: npt.NDArray[np.float64]
    interior: npt.NDArray[np.float64]
    last: npt.NDArray[np.float64]


def _optimise(mol: Any, max_iters: int) -> None:
    """Relax *mol* in place with MMFF94, falling back to UFF."""
    _, AllChem = _chem()
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        return
    AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)


def _embed_etkdg(mol: Any, seed: int, *, optimise_iters: int = 500) -> bool:
    """Embed a single ETKDG conformer in place. Returns whether it worked."""
    _, AllChem = _chem()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.enforceChirality = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False
    _optimise(mol, optimise_iters)
    return True


def _positions(mol: Any) -> npt.NDArray[np.float64]:
    """Return the conformer's positions in angstrom."""
    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)


def build_template(spec: ChainSpec, seed: int) -> _Template:
    """Relax a capped tetramer and reduce it to per-unit local geometry.

    Four units is the shortest oligomer that shows every junction the grown
    embedder has to make: first-to-second, interior-to-interior, and
    interior-to-last.

    Args:
        spec: The chain being built; only its monomer and caps matter here.
        seed: Seed for the template's own embedding.

    Returns:
        The template.

    Raises:
        ChainError: Even a tetramer would not embed, which means the monomer
            itself is the problem.
    """
    template_mol = _add_hydrogens(assemble_chain(spec, n_units=4))
    if not _embed_etkdg(template_mol, seed, optimise_iters=2000):
        raise ChainError(
            f"A tetramer of {spec.monomer_smiles!r} would not embed in 3D. "
            "Check the monomer SMILES and the caps: if four units cannot be "
            "built, neither can the chain."
        )

    positions = _positions(template_mol)
    groups = _unit_atoms(template_mol, 4)
    heads = [positions[_anchor(template_mol, i, "head")] for i in range(4)]
    tails = [positions[_anchor(template_mol, i, "tail")] for i in range(4)]

    # The reference is the previous unit's *tail* anchor, which is the backbone
    # atom bonded to this unit's head. Using its head instead would make the
    # controlled angle a four-point dihedral that skips a backbone atom, so
    # setting it to 180 would leave the real torsion somewhere else entirely.
    frame0 = _frame(heads[0], tails[0], tails[0] + _SEED_REFERENCE)
    frame1 = _frame(heads[1], tails[1], tails[0])
    frame2 = _frame(heads[2], tails[2], tails[1])
    return _Template(
        first=_to_local(frame0, positions[groups[0]]),
        second=_to_local(frame0, positions[groups[1]]),
        interior=_to_local(frame1, positions[groups[2]]),
        last=_to_local(frame2, positions[groups[3]]),
    )


class _Grid:
    """A uniform grid over placed heavy atoms, for the overlap test."""

    def __init__(self, spacing: float) -> None:
        self._spacing = spacing
        self._cells: dict[tuple[int, int, int], list[npt.NDArray[np.float64]]] = {}

    def _cell(self, point: npt.NDArray[np.float64]) -> tuple[int, int, int]:
        x, y, z = np.floor(point / self._spacing).astype(int)
        return int(x), int(y), int(z)

    def add(self, points: npt.NDArray[np.float64]) -> None:
        """Record *points* as occupied."""
        for point in points:
            self._cells.setdefault(self._cell(point), []).append(point)

    def clashes(self, points: npt.NDArray[np.float64], cutoff: float) -> bool:
        """Whether any of *points* lies within *cutoff* of a recorded point."""
        squared = cutoff * cutoff
        for point in points:
            cx, cy, cz = self._cell(point)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for other in self._cells.get((cx + dx, cy + dy, cz + dz), ()):
                            delta = point - other
                            if delta @ delta < squared:
                                return True
        return False


def _set_torsion(
    points: npt.NDArray[np.float64],
    *,
    pivot: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
    current: float,
    target: float,
) -> npt.NDArray[np.float64]:
    """Rotate *points* about *axis* through *pivot* to move a dihedral onto *target*.

    Measuring the dihedral the placement produced and correcting it, rather
    than applying a precomputed offset, means the sampled torsion is the
    torsion regardless of what conformation the template oligomer relaxed into.

    Args:
        points: The atoms to rotate.
        pivot: A point on the rotation axis.
        axis: The rotation axis; need not be normalised.
        current: The dihedral *points* currently produce, in degrees.
        target: The dihedral wanted, in degrees.

    Returns:
        The rotated points.
    """
    unit_axis = axis / np.linalg.norm(axis)
    rotation = _axis_rotation(unit_axis, math.radians(target - current))
    return (points - pivot) @ rotation.T + pivot


def _grow_conformer(
    mol: Any,
    spec: ChainSpec,
    template: _Template,
    n_units: int,
    seed: int,
    *,
    max_tries: int = 40,
    max_restarts: int = 200,
) -> npt.NDArray[np.float64]:
    """Grow a self-avoiding conformer by stamping the template down the chain.

    Each unit is placed by one rigid transform from the previous unit's frame,
    spun about the previous unit's backbone bond by a torsion drawn from the
    trans/gauche set. A placement that puts a heavy atom within
    :data:`_OVERLAP_ANGSTROM` of any already-placed unit before last is
    rejected and redrawn; when a unit runs out of draws, growth backs up two
    units and tries again.

    Args:
        mol: The chain, with explicit hydrogens.
        spec: The chain spec, for the torsion statistics.
        template: Per-unit geometry from :func:`build_template`.
        n_units: Degree of polymerization.
        seed: Seed for the torsion draws.
        max_tries: Torsion draws per unit before backing up.
        max_restarts: Total back-ups before giving up.

    Returns:
        Positions in angstrom.

    Raises:
        ChainError: Growth could not find a self-avoiding path.
    """
    groups = _unit_atoms(mol, n_units)
    heavy_global = [
        np.array(
            [i for i in group if mol.GetAtomWithIdx(i).GetAtomicNum() > 1], dtype=int
        )
        for group in groups
    ]
    heavy_local = [
        np.array(
            [
                position
                for position, index in enumerate(group)
                if mol.GetAtomWithIdx(index).GetAtomicNum() > 1
            ],
            dtype=int,
        )
        for group in groups
    ]
    heads = [_anchor(mol, i, "head") for i in range(n_units)]
    tails = [_anchor(mol, i, "tail") for i in range(n_units)]

    for unit, block in ((1, template.second), (n_units - 1, template.last)):
        if len(groups[unit]) != len(block):
            raise ChainError(  # pragma: no cover - guards the ordering contract
                f"Unit {unit} has {len(groups[unit])} atoms but its template "
                f"block has {len(block)}. The chain and the template oligomer "
                "disagree about atom ordering."
            )

    rng = random.Random(seed)
    p_trans = trans_fraction(spec.characteristic_ratio)
    weights = (p_trans, (1.0 - p_trans) / 2.0, (1.0 - p_trans) / 2.0)

    coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float64)
    coords[groups[0]] = template.first

    grid = _Grid(_OVERLAP_ANGSTROM)
    grid_upto = -1
    unit = 1
    restarts = 0

    while unit < n_units:
        # Only units two or more back can clash: the one immediately behind is
        # bonded to this one and is meant to be in contact.
        target = unit - 2
        if grid_upto > target:
            grid = _Grid(_OVERLAP_ANGSTROM)
            grid_upto = -1
        while grid_upto < target:
            grid_upto += 1
            grid.add(coords[heavy_global[grid_upto]])

        reference = (
            coords[tails[unit - 2]]
            if unit >= 2
            else coords[tails[unit - 1]] + _SEED_REFERENCE
        )
        frame = _frame(coords[heads[unit - 1]], coords[tails[unit - 1]], reference)
        if unit == 1:
            block = template.second
        elif unit == n_units - 1:
            block = template.last
        else:
            block = template.interior
        head_in_block = groups[unit].index(heads[unit])
        tail_in_block = groups[unit].index(tails[unit])

        for _ in range(max_tries):
            trial = _to_world(frame, block)
            # Two backbone bonds are opened by adding a unit: the previous
            # unit's own head-to-tail bond, and the junction bond joining the
            # two. Sampling only one of them leaves every second torsion frozen
            # at whatever the template relaxed into, which is how a chain ends
            # up a third too compact with the statistics apparently right.
            trial = _set_torsion(
                trial,
                pivot=coords[tails[unit - 1]],
                axis=coords[tails[unit - 1]] - coords[heads[unit - 1]],
                current=_dihedral(
                    reference,
                    coords[heads[unit - 1]],
                    coords[tails[unit - 1]],
                    trial[head_in_block],
                ),
                target=(
                    rng.uniform(-180.0, 180.0)
                    if unit == 1
                    else rng.choices(_TORSION_TARGETS_DEG, weights=weights)[0]
                ),
            )
            trial = _set_torsion(
                trial,
                pivot=trial[head_in_block],
                axis=trial[head_in_block] - coords[tails[unit - 1]],
                current=_dihedral(
                    coords[heads[unit - 1]],
                    coords[tails[unit - 1]],
                    trial[head_in_block],
                    trial[tail_in_block],
                ),
                target=rng.choices(_TORSION_TARGETS_DEG, weights=weights)[0],
            )
            if target < 0 or not grid.clashes(
                trial[heavy_local[unit]], _OVERLAP_ANGSTROM
            ):
                coords[groups[unit]] = trial
                break
        else:
            restarts += 1
            if restarts > max_restarts:
                raise ChainError(
                    f"Could not grow a self-avoiding conformer of "
                    f"{spec.residue_name} past unit {unit} of {n_units} after "
                    f"{max_restarts} back-ups. Try a different seed, or a "
                    "shorter chain."
                )
            unit = max(1, unit - 2)
            continue
        unit += 1

    return coords


def _set_conformer(mol: Any, coords: npt.NDArray[np.float64]) -> None:
    """Replace *mol*'s conformers with one holding *coords* (angstrom)."""
    Chem, _ = _chem()
    from rdkit.Geometry import Point3D

    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conformer, assignId=True)


_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _base36(value: int) -> str:
    """Render *value* in base 36, so four PDB columns hold more atoms."""
    digits = ""
    while value:
        value, remainder = divmod(value, 36)
        digits = _BASE36[remainder] + digits
    return digits or "0"


def atom_names(mol: Any) -> list[str]:
    """Return unique PDB atom names, at most four characters each.

    PDB atom names are four columns wide and ``PDBFile.writeModel`` truncates
    to four without complaint, so a naive ``C1000`` scheme silently produces
    duplicates. Counting per element in base 36 fits 46656 carbons.

    Args:
        mol: The molecule to name.

    Returns:
        One name per atom, in atom order.

    Raises:
        ChainError: An element has more atoms than four columns can name.
    """
    counts: dict[str, int] = {}
    names: list[str] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol().upper()
        counts[symbol] = counts.get(symbol, 0) + 1
        name = f"{symbol}{_base36(counts[symbol])}"
        if len(name) > 4:
            raise ChainError(
                f"{counts[symbol]} {symbol} atoms cannot be given unique "
                "four-character PDB names. Build a shorter chain."
            )
        names.append(name)
    return names


def _label_atoms(mol: Any, residue_name: str) -> None:
    """Attach PDB residue information, so the whole chain is one residue."""
    Chem, _ = _chem()
    for atom, name in zip(mol.GetAtoms(), atom_names(mol), strict=True):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.ljust(4))
        info.SetResidueName(residue_name.ljust(3))
        info.SetResidueNumber(1)
        info.SetChainId("A")
        info.SetOccupancy(1.0)
        info.SetTempFactor(0.0)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)


def molar_mass_g_mol(mol: Any) -> float:
    """Return a molecule's molar mass, summed over its atoms.

    Summed here rather than taken from ``rdkit.Chem.Descriptors`` so that the
    hydrogens this package always makes explicit are counted once, and only
    once.

    Args:
        mol: The molecule, with explicit hydrogens.

    Returns:
        The molar mass in g/mol.
    """
    return float(sum(atom.GetMass() for atom in mol.GetAtoms()))


def _radius_of_gyration_nm(mol: Any, coords: npt.NDArray[np.float64]) -> float:
    """Return the mass-weighted radius of gyration, in nanometres."""
    masses = np.array([atom.GetMass() for atom in mol.GetAtoms()], dtype=np.float64)
    centre = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    offsets = coords - centre
    squared = float((masses * (offsets * offsets).sum(axis=1)).sum() / masses.sum())
    return math.sqrt(squared) / 10.0


def _extent_bound_nm(coords: npt.NDArray[np.float64]) -> float:
    """Return a conservative bound on the conformer's diameter, in nanometres.

    Twice the largest distance from the centroid. It never understates the
    extent, which is what the "does this fit in the packing cell" check needs,
    and it costs one pass rather than the N-squared of every pair.
    """
    centre = coords.mean(axis=0)
    offsets = coords - centre
    return 2.0 * float(np.sqrt((offsets * offsets).sum(axis=1)).max()) / 10.0


def _mean_bond_length_nm(
    coords: npt.NDArray[np.float64], backbone: Sequence[int]
) -> float:
    """Return the mean backbone bond length, in nanometres."""
    points = coords[list(backbone)]
    steps = points[1:] - points[:-1]
    return float(np.sqrt((steps * steps).sum(axis=1)).mean()) / 10.0


def build_chain(
    spec: ChainSpec,
    output_prefix: str = "chain",
    *,
    n_conformers: int = 1,
    output_dir: str | Path | None = None,
    embedder: str = "auto",
) -> ChainResult:
    """Build one polymer chain and write a conformer ensemble for packing.

    Every copy of the chain in the packed box should be a different
    conformation. packmol places each structure as a rigid body, so packing one
    conformer N times makes a cell of N identical coils whose memory outlives
    any nanosecond-scale equilibration. They are all the same molecule, so they
    share a single force-field template; only the coordinates differ.

    Args:
        spec: What to build.
        output_prefix: Stem for the written files. May carry a path.
        n_conformers: How many conformations to write. Set this to the number
            of chains that will be packed.
        output_dir: Directory the files are written into. Defaults to the
            working directory, which is where the rest of the package writes.
        embedder: One of :data:`EMBEDDERS`. ``auto`` grows any chain of four
            units or more. ETKDG is not the default for those because it
            collapses a chain into a globule - a twenty-unit polyethylene comes
            out with a quarter of the dimensions it should have - and a cell
            packed from collapsed coils is a long way from a melt.

    Returns:
        What was built, including the paths to feed packing and
        parameterisation.

    Raises:
        ChainError: The monomer, a cap or the embedding failed.
    """
    Chem, _ = _chem()

    require_integer(n_conformers, name="n_conformers")
    require_choice(embedder, EMBEDDERS, name="embedder")
    directory = Path(output_dir) if output_dir is not None else Path()
    directory.mkdir(parents=True, exist_ok=True)

    units = spec.degree_of_polymerization
    molecule = _add_hydrogens(assemble_chain(spec))
    _label_atoms(molecule, spec.residue_name)
    n_atoms = molecule.GetNumAtoms()
    backbone = backbone_path(molecule, units)

    if embedder == "grown" and units < _MIN_GROWN_UNITS:
        raise ChainError(
            f"embedder='grown' needs at least {_MIN_GROWN_UNITS} units; this "
            f"chain has {units}. Use embedder='etkdg' or 'auto'."
        )
    grown = embedder == "grown" or (embedder == "auto" and units >= _MIN_GROWN_UNITS)
    template = (
        build_template(spec, derive_seed(spec.seed, "template")) if grown else None
    )
    log.info(
        "Building %d conformer(s) of %s: %d units, %d atoms, %s embedder.",
        n_conformers,
        spec.residue_name,
        units,
        n_atoms,
        "grown" if grown else "ETKDG",
    )

    sdf_paths: list[str] = []
    pdb_paths: list[str] = []
    radii: list[float] = []
    extents: list[float] = []
    ratios: list[float] = []

    for index in range(n_conformers):
        seed = derive_seed(spec.seed, "conformer", str(index))
        conformer_mol = Chem.Mol(molecule)
        if template is not None:
            coords = _grow_conformer(conformer_mol, spec, template, units, seed)
            _set_conformer(conformer_mol, coords)
            # Short, so the junction torsions survive: an unconstrained vacuum
            # relaxation run to convergence collapses the coil.
            _optimise(conformer_mol, 200)
        elif not _embed_etkdg(conformer_mol, seed):
            raise ChainError(
                f"ETKDG could not embed conformer {index} of "
                f"{spec.residue_name} ({n_atoms} atoms). Pass "
                "embedder='grown' to build it unit by unit instead."
            )

        coords = _positions(conformer_mol)
        sdf_path = directory / f"{output_prefix}_{index}.sdf"
        pdb_path = directory / f"{output_prefix}_{index}.pdb"
        conformer_mol.SetProp("_Name", spec.residue_name)
        with Chem.SDWriter(str(sdf_path)) as writer:
            writer.write(conformer_mol)
        Chem.MolToPDBFile(conformer_mol, str(pdb_path))

        sdf_paths.append(str(sdf_path))
        pdb_paths.append(str(pdb_path))
        radii.append(_radius_of_gyration_nm(conformer_mol, coords))
        extents.append(_extent_bound_nm(coords))
        if len(backbone) > 2:
            ratios.append(
                characteristic_ratio(
                    coords / 10.0, backbone, _mean_bond_length_nm(coords, backbone)
                )
            )

    measured = float(np.mean(ratios)) if ratios else None
    _report_ratio(measured, spec, len(backbone) - 1, n_conformers)

    return ChainResult(
        sdf_paths=tuple(sdf_paths),
        pdb_paths=tuple(pdb_paths),
        smiles=Chem.MolToSmiles(Chem.RemoveHs(molecule)),
        n_atoms=n_atoms,
        molar_mass_g_mol=molar_mass_g_mol(molecule),
        radius_of_gyration_nm=tuple(radii),
        max_extent_nm=tuple(extents),
        backbone=backbone,
        characteristic_ratio=measured,
        embedder="grown" if grown else "etkdg",
    )


#: Below this many backbone bonds a single chain's C is dominated by its own
#: finite length and by sample-to-sample scatter, so there is nothing to judge.
_RATIO_MIN_BONDS = 30


def _report_ratio(
    measured: float | None, spec: ChainSpec, n_bonds: int, n_conformers: int
) -> None:
    """Say so when the built conformers are not the dimensions asked for.

    The two embedders err in opposite directions - ETKDG collapses a long chain
    into a globule, a hard-core self-avoiding walk swells it past the melt's
    unperturbed dimensions - so this checks both signs rather than only the one
    the embedder in use is prone to.
    """
    if measured is None:
        return
    if n_bonds < _RATIO_MIN_BONDS or n_conformers < 3:
        log.info(
            "Characteristic ratio %.2f over %d conformer(s) of %d backbone "
            "bonds; too short a chain, or too few of them, to judge against "
            "the expected %.2f.",
            measured,
            n_conformers,
            n_bonds,
            spec.characteristic_ratio,
        )
        return
    target = spec.characteristic_ratio
    drift = (measured - target) / target
    if abs(drift) <= _RATIO_TOLERANCE:
        log.info("Characteristic ratio %.2f (expected %.2f).", measured, target)
        return
    shape = "swollen" if drift > 0 else "collapsed"
    log.warning(
        "Built conformers of %s have a characteristic ratio of %.2f against an "
        "expected %.2f (%+.0f%%, i.e. %s). The high-temperature stages have to "
        "undo this before any chain-scale property is meaningful.",
        spec.residue_name,
        measured,
        target,
        100.0 * drift,
        shape,
    )
