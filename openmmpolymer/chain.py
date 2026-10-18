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

Two embedders live here, and they fail in opposite directions. RDKit's ETKDG
collapses a long chain into a globule - a twenty-unit polyethylene comes out
with a quarter of the dimensions it should have, and a cell packed from
collapsed coils is a long way from a melt - so anything of four units or more
is grown unit by unit instead. A hard-core self-avoiding walk like that errs
the other way, swelling the chain past the melt's unperturbed dimensions.
:func:`build_chain` therefore measures the characteristic ratio it produced,
and says so when it drifts either way.

Stereochemistry the monomer SMILES writes is kept in every unit; tacticity only
fills in the backbone centres it leaves open. Every conformer is checked
against its chiral tags before it is written, because forcefill and the
simulation see the coordinates, not the tags.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Geometry import Point3D

from ._seeds import DEFAULT_SEED, derive_seed
from ._validation import require_choice, require_integer, require_positive
from .packing import ANGSTROM_PER_NM, CellList

log = logging.getLogger(__name__)

#: How a monomer's two open valences are marked.
DUMMY_ATOMIC_NUMBER = 0

#: Polyethylene reference used when no characteristic ratio is supplied.
DEFAULT_CHARACTERISTIC_RATIO = 7.0

#: Accepted values for ``ChainSpec.tacticity``.
TACTICITIES = ("atactic", "isotactic", "syndiotactic")

#: The cap that means "just fill the valence with hydrogen".
HYDROGEN_CAP = "[*][H]"

#: The two tetrahedral chiral tags, in the order tacticity choices index them.
_TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)

#: Ways to produce 3D coordinates. ``auto`` grows anything long enough to
#: stamp a template down - four units - and falls back to ETKDG below that.
EMBEDDERS = ("auto", "etkdg", "grown")

#: Shortest chain the grown embedder can build: it needs a first junction, an
#: interior one and a last one to copy.
_MIN_GROWN_UNITS = 4

#: Backbone torsion states the grown embedder draws from, in degrees: trans,
#: gauche+, gauche-. These are absolute targets - each placement measures the
#: dihedral it produced and rotates it onto one of these - so nothing depends
#: on whatever conformation the relaxed template oligomer happened to settle in.
_TORSION_TARGETS_DEG = (180.0, 60.0, -60.0)

#: The backbone bond angle the trans fraction is derived for, in degrees.
_BOND_ANGLE_DEG = 112.0

#: Closest approach allowed between heavy atoms of non-adjacent units during
#: growth, in angstrom. Below a real van der Waals contact, because the geometry
#: is relaxed afterwards and a stricter test rejects almost everything.
_OVERLAP_ANGSTROM = 2.6

#: Closest approach allowed between heavy atoms of adjacent units more than
#: three bonds apart, in angstrom. Looser than :data:`_OVERLAP_ANGSTROM`
#: because rigid units joined at fixed torsion states sit closer than that in
#: a crowded backbone - poly(methyl methacrylate)'s esters, say - and what this
#: guards against is one unit grown through the next, not crowding.
_ADJACENT_OVERLAP_ANGSTROM = 2.0

#: Torsion draws per unit before growth backs up two units, and back-ups before
#: it gives up.
_MAX_TRIES = 40
_MAX_RESTARTS = 200

#: The reference direction used for the very first unit, which has no
#: predecessor to take one from. Any fixed choice does, as long as the template
#: and the chain make the same one.
_SEED_REFERENCE = np.array([0.0, 1.0, 0.0])

#: Fraction by which a measured characteristic ratio may differ from the
#: expected one before :func:`build_chain` says so.
_RATIO_TOLERANCE = 0.15

#: Below this many backbone bonds a single chain's C is dominated by its own
#: finite length and by sample-to-sample scatter, so there is nothing to judge.
_RATIO_MIN_BONDS = 30


class ChainError(RuntimeError):
    """A polymer chain could not be built."""


@dataclass(frozen=True)
class ChainSpec:
    """What chain to build.

    Args:
        monomer_smiles: SMILES with exactly two ``[*]`` attachment points. The
            first in atom order is the head, the last the tail.
        degree_of_polymerization: Number of repeat units.
        tacticity: One of ``atactic``, ``isotactic``, ``syndiotactic``. Applies
            only to backbone stereocentres the monomer SMILES leaves
            unspecified; a configuration written with ``@``/``@@`` is kept in
            every unit, on the backbone or off it, and cap stereochemistry is
            left unchanged. A no-op for a monomer with no open backbone
            stereocentre.
        head_cap: SMILES fragment with one ``[*]`` closing the first unit's open
            head, or None to choose one from the attachment chemistry.
        tail_cap: The same for the last unit's open tail.
        residue_name: Residue name for the PDB and the force-field template.
            One to three alphanumeric characters, because forcefill truncates
            to three and a collision there is a hard failure.
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
    characteristic_ratio: float = DEFAULT_CHARACTERISTIC_RATIO
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        """Reject a malformed spec at construction, not mid-build."""
        require_integer(self.degree_of_polymerization, name="degree_of_polymerization")
        require_choice(self.tacticity, TACTICITIES, name="tacticity")
        require_positive(self.characteristic_ratio, name="characteristic_ratio")
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
        smiles: Canonical SMILES of the capped chain, hydrogens implicit.
        n_atoms: Atoms per chain, hydrogens included.
        molar_mass_g_mol: Chain molar mass.
        radius_of_gyration_nm: Per-conformer mass-weighted radius of gyration.
        max_extent_nm: Per-conformer upper bound on the largest interatomic
            distance: twice the farthest atom's distance from the centroid.
            packmol cannot place a conformer longer than the box.
        backbone: Backbone atom indices within one chain, head to tail. The
            run manifest measures chain dimensions with these, and nothing
            downstream can recover them: they come from the attachment points,
            which the caps consumed.
        characteristic_ratio: Measured over the conformers, or None for a
            single-bond backbone, whose ratio is 1 whatever its geometry.
        embedder: Which embedder ran, ``"etkdg"`` or ``"grown"``.
    """

    sdf_paths: tuple[str, ...]
    pdb_paths: tuple[str, ...]
    smiles: str
    n_atoms: int
    molar_mass_g_mol: float
    radius_of_gyration_nm: tuple[float, ...] = ()
    max_extent_nm: tuple[float, ...] = ()
    backbone: tuple[int, ...] = ()
    characteristic_ratio: float | None = None
    embedder: str = "etkdg"


def _atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    """Return *mol*'s atoms in order: RDKit's stubs leave ``GetAtoms`` untyped."""
    return list(mol.GetAtoms())  # type: ignore[no-untyped-call]


def _dummies(mol: Chem.Mol) -> list[int]:
    """Return the indices of *mol*'s ``[*]`` attachment points, in atom order."""
    return [
        atom.GetIdx()
        for atom in _atoms(mol)
        if atom.GetAtomicNum() == DUMMY_ATOMIC_NUMBER
    ]


def _parse_monomer(smiles: str) -> Chem.Mol:
    """Return the monomer molecule, checking it has exactly two dummies.

    Raises:
        ChainError: The SMILES does not parse, or does not carry exactly two
            attachment points each bonded to one real atom by a single bond,
            the two on different atoms.
    """
    monomer: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    if monomer is None:
        raise ChainError(f"monomer_smiles={smiles!r} is not valid SMILES.")

    dummies = _dummies(monomer)
    if len(dummies) != 2:
        raise ChainError(
            f"monomer_smiles={smiles!r} has {len(dummies)} attachment points; "
            "a linear repeat unit needs exactly two. Polyethylene is "
            "'[*]CC[*]', polystyrene is '[*]CC([*])c1ccccc1'."
        )

    for idx in dummies:
        bonds = monomer.GetAtomWithIdx(idx).GetBonds()
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
        monomer, dummies[1]
    ):
        raise ChainError(
            f"Both attachment points of {smiles!r} land on the same atom, so "
            "the repeat unit has no backbone bond of its own and the chain "
            "geometry cannot be built. Write the unit with at least two "
            "backbone atoms - polyisobutylene as '[*]CC(C)(C)[*]' rather than "
            "'[*]C(C)(C)[*]'."
        )
    return monomer


def _attachment_neighbour(mol: Chem.Mol, dummy_idx: int) -> int:
    """Return the real atom the dummy at *dummy_idx* is bonded to."""
    neighbours = mol.GetAtomWithIdx(dummy_idx).GetNeighbors()
    return int(neighbours[0].GetIdx())


def _default_cap(mol: Chem.Mol, end: str) -> str:
    """Choose a cap for the open valence at the chain's *end*.

    Hydrogen is right almost everywhere and wrong in one place that matters: on
    a carbonyl carbon it makes an aldehyde, so an H-capped polyester ends in
    ``-CHO`` rather than ``-COOH``, with the wrong atom types and the wrong
    terminal charges. Rather than guess a replacement, say what happened.

    Args:
        mol: The chain, with its attachment point tagged *end*.
        end: ``"head"`` or ``"tail"``.

    Returns:
        A cap SMILES.

    Raises:
        ChainError: The attachment atom is a carbonyl carbon, where no default
            is safe.
    """
    attachment = mol.GetAtomWithIdx(_attachment_neighbour(mol, _find_role(mol, end)))
    if _is_carbonyl_carbon(attachment):
        raise ChainError(
            f"The {end} attachment point sits on a carbonyl carbon, where a "
            "hydrogen cap would make an aldehyde rather than an acid or an "
            f"ester. Pass {end}_cap explicitly: '[*]O' for the free acid, "
            "'[*]OC' for the methyl ester."
        )
    return HYDROGEN_CAP


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    """Whether *atom* is a carbon double-bonded to an oxygen."""
    if atom.GetAtomicNum() != 6:
        return False
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetAtomicNum() == 8
        for bond in atom.GetBonds()
    )


def _tagged_unit(monomer: Chem.Mol, unit: int) -> Chem.RWMol:
    """Return a copy of *monomer* whose atoms carry their unit index and role."""
    copy = Chem.RWMol(monomer)
    head, tail = _dummies(copy)
    for atom in _atoms(copy):
        atom.SetIntProp("_omp_unit", unit)
    copy.GetAtomWithIdx(head).SetProp("_omp_role", "head")
    copy.GetAtomWithIdx(tail).SetProp("_omp_role", "tail")
    # The dummies are consumed by the joins, so the atoms they were bonded to
    # carry the memory of where the backbone runs. The grown embedder and the
    # characteristic-ratio measurement both need that path.
    copy.GetAtomWithIdx(_attachment_neighbour(copy, head)).SetIntProp(
        "_omp_head_anchor", 1
    )
    copy.GetAtomWithIdx(_attachment_neighbour(copy, tail)).SetIntProp(
        "_omp_tail_anchor", 1
    )
    return copy


def _find_role(mol: Chem.Mol, role: str) -> int:
    """Return the index of the atom tagged *role*."""
    for atom in _atoms(mol):
        if atom.HasProp("_omp_role") and atom.GetProp("_omp_role") == role:
            return int(atom.GetIdx())
    raise ChainError(f"No atom tagged {role!r}.")


def _keep_configuration(mol: Chem.Mol, dummy_idx: int) -> None:
    """Fix the chiral tag of the dummy's neighbour before the dummy is replaced.

    RDKit reads a chiral tag against the order of the atom's bonds, counting an
    implicit hydrogen last. Whatever takes the dummy's place ends up last too:
    ``AddBond`` appends the bond to the next unit, and a hydrogen cap becomes an
    implicit hydrogen. Moving the dummy's slot to the end of the list is an odd
    permutation about half the time, depending on how the SMILES was written,
    and then the same tag means the other configuration unless it is inverted.
    """
    atom = mol.GetAtomWithIdx(_attachment_neighbour(mol, dummy_idx))
    order = [bond.GetOtherAtomIdx(atom.GetIdx()) for bond in atom.GetBonds()]
    if (len(order) - 1 - order.index(dummy_idx)) % 2:
        atom.InvertChirality()


def _splice(
    left: Chem.Mol, left_dummy: int, right: Chem.Mol, right_dummy: int
) -> Chem.RWMol:
    """Bond *left* to *right* where their dummies were, and drop both dummies."""
    combined = Chem.RWMol(Chem.CombineMols(left, right))
    right_dummy += left.GetNumAtoms()
    for dummy in (left_dummy, right_dummy):
        _keep_configuration(combined, dummy)
    combined.AddBond(
        _attachment_neighbour(combined, left_dummy),
        _attachment_neighbour(combined, right_dummy),
        Chem.BondType.SINGLE,
    )
    for idx in sorted((left_dummy, right_dummy), reverse=True):
        combined.RemoveAtom(idx)
    return combined


def _attach_cap(chain: Chem.RWMol, role: str, cap_smiles: str) -> Chem.RWMol:
    """Close the open valence tagged *role* with *cap_smiles*."""
    dummy = _find_role(chain, role)

    if cap_smiles == HYDROGEN_CAP:
        # Removing the dummy leaves a free valence that RDKit fills with an
        # implicit hydrogen on the next sanitize, which is exactly the cap -
        # except on an atom written in brackets, as every stereocentre is,
        # whose hydrogen count is fixed and which would be left a radical.
        _keep_configuration(chain, dummy)
        anchor = chain.GetAtomWithIdx(_attachment_neighbour(chain, dummy))
        if anchor.GetNoImplicit():
            anchor.SetNumExplicitHs(anchor.GetNumExplicitHs() + 1)
        chain.RemoveAtom(dummy)
        return chain

    parsed: Chem.Mol | None = Chem.MolFromSmiles(cap_smiles)
    if parsed is None:
        raise ChainError(f"cap {cap_smiles!r} is not valid SMILES.")
    cap_dummies = _dummies(parsed)
    if len(cap_dummies) != 1:
        raise ChainError(
            f"cap {cap_smiles!r} has {len(cap_dummies)} attachment points; a "
            "cap needs exactly one, as in '[*]C' or '[*]O'."
        )

    unit = chain.GetAtomWithIdx(dummy).GetIntProp("_omp_unit")
    cap = Chem.RWMol(parsed)
    for atom in _atoms(cap):
        atom.SetIntProp("_omp_unit", unit)
    return _splice(chain, dummy, cap, cap_dummies[0])


def assemble_chain(spec: ChainSpec, n_units: int | None = None) -> Chem.Mol:
    """Build the capped chain molecule, without coordinates.

    Args:
        spec: What to build.
        n_units: Override the degree of polymerization, for building the short
            template oligomer the grown embedder needs.

    Returns:
        A sanitised RDKit molecule with implicit hydrogens, whose atoms carry
        an ``_omp_unit`` integer property.

    Raises:
        ChainError: The monomer or a cap is malformed, or an end on a carbonyl
            carbon was left to the default cap.
    """
    units = spec.degree_of_polymerization if n_units is None else n_units
    return _assemble(spec, _tacticity_choices(spec, units))


def _assemble(spec: ChainSpec, choices: Sequence[int]) -> Chem.Mol:
    """Build the capped chain of ``len(choices)`` units; see :func:`assemble_chain`.

    *choices* gives each unit's configuration for the backbone stereocentres
    tacticity sets, which lets the grown embedder build a template oligomer
    whose units match any chosen units of the chain.
    """
    monomer = _parse_monomer(spec.monomer_smiles)
    units = len(choices)

    chain = _tagged_unit(monomer, 0)
    for unit in range(1, units):
        next_unit = _tagged_unit(monomer, unit)
        chain = _splice(
            chain, _find_role(chain, "tail"), next_unit, _find_role(next_unit, "head")
        )
    for end, cap in (("head", spec.head_cap), ("tail", spec.tail_cap)):
        chain = _attach_cap(chain, end, cap or _default_cap(chain, end))

    molecule = chain.GetMol()
    Chem.SanitizeMol(molecule)
    _apply_tacticity(molecule, spec, choices)
    return molecule


def _tacticity_choices(spec: ChainSpec, units: int) -> list[int]:
    """Return which of the two configurations each unit takes, per ``spec.tacticity``.

    One draw per unit, in unit order, whether or not the monomer has anything
    for tacticity to set.
    """
    rng = random.Random(derive_seed(spec.seed, "tacticity"))
    choices: list[int] = []
    for unit in range(units):
        if spec.tacticity == "isotactic":
            choices.append(0)
        elif spec.tacticity == "syndiotactic":
            choices.append(unit % 2)
        else:
            choices.append(rng.randint(0, 1))
    return choices


def _apply_tacticity(mol: Chem.Mol, spec: ChainSpec, choices: Sequence[int]) -> None:
    """Set the chiral tags the monomer leaves open on the backbone, per *choices*.

    A monomer with no backbone stereocentre - polyethylene, say - has nothing
    to set, and that is not an error. Nor does tacticity touch a centre the
    monomer SMILES writes a configuration for, on the backbone or off it: a
    sugar-derived repeat unit's stereochemistry is the point of using it.
    """
    units = len(choices)
    Chem.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    backbone = set(backbone_path(mol, units))
    centres: dict[int, list[Chem.Atom]] = {}
    written = False
    for atom in _atoms(mol):
        if atom.GetIdx() not in backbone or not atom.HasProp("_omp_unit"):
            continue
        if atom.GetChiralTag() in _TETRAHEDRAL:
            written = True
        elif atom.HasProp("_ChiralityPossible"):
            centres.setdefault(atom.GetIntProp("_omp_unit"), []).append(atom)
    if not centres:
        if written and spec.tacticity == "syndiotactic":
            log.warning(
                "%s: every backbone stereocentre has its configuration written "
                "in the monomer SMILES, so tacticity='syndiotactic' has nothing "
                "to alternate and every unit keeps the written one.",
                spec.residue_name,
            )
        else:
            log.debug(
                "%s has no open backbone stereocentre; tacticity is a no-op.",
                spec.residue_name,
            )
        return

    for unit in range(units):
        for atom in centres.get(unit, ()):
            atom.SetChiralTag(_TETRAHEDRAL[choices[unit]])
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def _trans_fraction(characteristic_ratio: float) -> float:
    """Return the trans probability giving *characteristic_ratio*.

    Inverts the independent-rotation expression

        C = (1 - cos t) / (1 + cos t) * (1 + <cos f>) / (1 - <cos f>)

    for a symmetric three-state backbone, where ``<cos f> = 1.5 p - 0.5``. Two
    states at plus and minus 120 degrees contribute ``-0.5`` each, so the whole
    dependence collapses onto the trans fraction. Hard-coding a weight triplet
    instead is how chains come out a quarter too compact: the common
    ``(0.55, 0.225, 0.225)`` gives C = 4.3, not the 7.0 polyethylene has, which
    wants a trans fraction of 0.68.

    Returns:
        A trans probability, clamped to ``[0.05, 0.95]``.
    """
    cos_theta = math.cos(math.radians(_BOND_ANGLE_DEG))
    angle_factor = (1.0 - cos_theta) / (1.0 + cos_theta)
    x = characteristic_ratio / angle_factor
    p = (3.0 * x - 1.0) / (3.0 * (1.0 + x))
    return min(0.95, max(0.05, p))


def _characteristic_ratio(
    positions_nm: npt.NDArray[np.float64],
    backbone: Sequence[int],
    bond_length_nm: float,
) -> float:
    """Return the measured C = <R^2> / (n l^2) for one conformer."""
    ends = positions_nm[backbone[-1]] - positions_nm[backbone[0]]
    n_bonds = len(backbone) - 1
    return float(ends @ ends) / (n_bonds * bond_length_nm**2)


def _anchor_units(mol: Chem.Mol, which: str) -> dict[int, int]:
    """Each unit's head or tail backbone anchor atom index, found in one pass."""
    prop = f"_omp_{which}_anchor"
    found: dict[int, int] = {}
    for atom in _atoms(mol):
        if atom.HasProp(prop) and atom.HasProp("_omp_unit"):
            found.setdefault(atom.GetIntProp("_omp_unit"), int(atom.GetIdx()))
    return found


def _anchor(mol: Chem.Mol, unit: int, which: str) -> int:
    """Return the index of unit *unit*'s head or tail backbone anchor atom."""
    try:
        return _anchor_units(mol, which)[unit]
    except KeyError:
        raise ChainError(f"Unit {unit} has no {which} anchor.") from None


def _anchors(mol: Chem.Mol, n_units: int, which: str) -> list[int]:
    """Every unit's head or tail anchor atom index, in unit order."""
    found = _anchor_units(mol, which)
    for unit in range(n_units):
        if unit not in found:
            raise ChainError(f"Unit {unit} has no {which} anchor.")
    return [found[unit] for unit in range(n_units)]


def backbone_path(mol: Chem.Mol, n_units: int) -> tuple[int, ...]:
    """Return the backbone atom indices, in order from head to tail.

    Args:
        mol: An assembled chain.
        n_units: Its degree of polymerization.

    Returns:
        The shortest path through the molecule from the first unit's head
        anchor to the last unit's tail anchor, which for a linear chain is the
        backbone.
    """
    start = _anchor(mol, 0, "head")
    end = _anchor(mol, n_units - 1, "tail")
    return tuple(int(idx) for idx in Chem.GetShortestPath(mol, start, end))


def _add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Return *mol* with explicit hydrogens, each inheriting its parent's unit."""
    with_h = Chem.AddHs(mol)
    for atom in _atoms(with_h):
        if atom.HasProp("_omp_unit"):
            continue
        parent = atom.GetNeighbors()[0]
        atom.SetIntProp("_omp_unit", parent.GetIntProp("_omp_unit"))
    return with_h


def _unit_atoms(mol: Chem.Mol, n_units: int) -> list[list[int]]:
    """Return each unit's atom indices, ascending.

    Both the chain and the template oligomer are assembled by the same code, so
    within a unit this ordering is the same in both: the unit's heavy atoms in
    construction order, then its hydrogens in parent order, because ``AddHs``
    appends. That correspondence is what lets a template unit's geometry be
    copied onto a chain unit by position.
    """
    groups: list[list[int]] = [[] for _ in range(n_units)]
    for atom in _atoms(mol):
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
    previous unit's tail anchor, the backbone atom bonded to this unit's head,
    which is what makes the frame carry the backbone torsion rather than an
    arbitrary spin.
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
    """Per-unit geometry, ready to be stamped down the chain.

    Each block but *first* holds a unit's atom positions in the *previous*
    unit's local frame, so placing a unit is one rigid transform plus the
    sampled backbone torsions; *first* is the head unit in its own frame. The
    geometry itself - bond lengths, bond angles, side-group placement - comes
    from a force-field-relaxed tetramer, so none of it has to be tabulated here.
    """

    first: npt.NDArray[np.float64]
    second: npt.NDArray[np.float64]
    interior: npt.NDArray[np.float64]
    last: npt.NDArray[np.float64]


def _optimise(mol: Chem.Mol, max_iters: int) -> None:
    """Relax *mol* in place with MMFF94, falling back to UFF."""
    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        return
    rdForceFieldHelpers.UFFOptimizeMolecule(mol, maxIters=max_iters)


def _embed_etkdg(mol: Chem.Mol, seed: int, *, optimise_iters: int = 500) -> bool:
    """Embed a single ETKDG conformer in place. Returns whether it worked."""
    # Any, because the stubs give each of these attributes the parameters' type.
    params: Any = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.enforceChirality = True
    if rdDistGeom.EmbedMolecule(mol, params) != 0:
        return False
    _optimise(mol, optimise_iters)
    return True


def _positions(mol: Chem.Mol) -> npt.NDArray[np.float64]:
    """Return the conformer's positions in angstrom."""
    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)


def _build_template(spec: ChainSpec, choices: Sequence[int], seed: int) -> _Template:
    """Relax a capped tetramer and reduce it to per-unit local geometry.

    Four units is the shortest oligomer that shows every junction the grown
    embedder has to make: first-to-second, interior-to-interior, and
    interior-to-last.

    Args:
        spec: What is being built.
        choices: The tetramer's four tacticity choices; see
            :func:`_tacticity_choices`.
        seed: Seed for the embedding.

    Raises:
        ChainError: Even a tetramer would not embed, which means the monomer
            itself is the problem.
    """
    template_mol = _add_hydrogens(_assemble(spec, choices))
    if not _embed_etkdg(template_mol, seed, optimise_iters=2000):
        raise ChainError(
            f"A tetramer of {spec.monomer_smiles!r} would not embed in 3D. "
            "Check the monomer SMILES and the caps: if four units cannot be "
            "built, neither can the chain."
        )

    positions = _positions(template_mol)
    groups = _unit_atoms(template_mol, 4)
    heads = [positions[index] for index in _anchors(template_mol, 4, "head")]
    tails = [positions[index] for index in _anchors(template_mol, 4, "tail")]

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


def _unit_blocks(
    spec: ChainSpec, mol: Chem.Mol, choices: Sequence[int]
) -> list[npt.NDArray[np.float64]]:
    """Return the template geometry each unit of *mol* is stamped from.

    A template unit's coordinates carry its configuration, so stamping one
    interior unit down the whole chain gives every interior unit the same
    stereochemistry, whatever its own chiral tags say. Each distinct set of
    tags among the interior units therefore gets a template of its own, built
    with that unit's choice in the interior and the chain's own at the ends.
    A chain whose interior units all carry the same tags - any without a
    stereocentre for tacticity to set - has the one template it always had.

    Args:
        spec: What is being built.
        mol: The chain, with explicit hydrogens and its chiral tags set.
        choices: The chain's tacticity choices, one per unit.

    Returns:
        One block per unit, in unit order, in the frame :func:`_grow_conformer`
        places it in.
    """
    units = len(choices)
    groups = _unit_atoms(mol, units)
    templates: dict[tuple[Chem.ChiralType, ...], _Template] = {}
    interior: list[npt.NDArray[np.float64]] = []
    for unit in range(2, units - 1):
        tags = tuple(mol.GetAtomWithIdx(index).GetChiralTag() for index in groups[unit])
        if tags not in templates:
            labels = ("template",) if not templates else ("template", str(unit))
            templates[tags] = _build_template(
                spec,
                (choices[0], choices[1], choices[unit], choices[-1]),
                derive_seed(spec.seed, *labels),
            )
        interior.append(templates[tags].interior)
    ends = next(iter(templates.values()))
    return [ends.first, ends.second, *interior, ends.last]


@dataclass(frozen=True)
class _TailAnchor:
    """A unit's tail anchor, and what turning the next unit past it moves.

    Args:
        anchor: The tail anchor's atom index.
        previous: The backbone atom bonded to the anchor on the head side. The
            torsion turns about the bond from it to the anchor.
        before: The backbone atom before *previous*, which the torsion is
            measured from, or None when *previous* starts the chain.
        bonded: The anchor's neighbours within its own unit, *previous*
            included: everything but the next unit's head.
        carried: Every atom of the unit on the anchor's side of that bond,
            which turns with the next unit; None when the bond is not a single
            bond free to turn - a double bond, or one closing a ring.
    """

    anchor: int
    previous: int
    before: int | None
    bonded: tuple[int, ...]
    carried: tuple[int, ...] | None


def _tail_anchors(
    mol: Chem.Mol, n_units: int, tails: Sequence[int]
) -> list[_TailAnchor | None]:
    """Describe each unit's tail anchor, if the chain has a stereocentre.

    A chain without one is grown by spinning the new unit alone, as it always
    was, and gets None throughout: there is no configuration to lose, and
    changing how it grows would change every such build on disk.
    """
    if not any(atom.GetChiralTag() in _TETRAHEDRAL for atom in _atoms(mol)):
        return [None] * len(tails)
    backbone = backbone_path(mol, n_units)
    position = {index: at for at, index in enumerate(backbone)}
    found: list[_TailAnchor | None] = []
    for unit, tail in enumerate(tails):
        at = position[tail]
        previous = backbone[at - 1]
        bonded = tuple(
            int(neighbour.GetIdx())
            for neighbour in mol.GetAtomWithIdx(tail).GetNeighbors()
            if neighbour.GetIntProp("_omp_unit") == unit
        )
        seen = {tail}
        stack = [index for index in bonded if index != previous]
        bond = mol.GetBondBetweenAtoms(previous, tail)
        closed = bond.GetBondType() != Chem.BondType.SINGLE
        while stack and not closed:
            index = stack.pop()
            closed = index == previous
            if index in seen or closed:
                continue
            seen.add(index)
            stack.extend(
                int(neighbour.GetIdx())
                for neighbour in mol.GetAtomWithIdx(index).GetNeighbors()
                if neighbour.GetIntProp("_omp_unit") == unit
            )
        found.append(
            _TailAnchor(
                anchor=tail,
                previous=previous,
                before=backbone[at - 2] if at >= 2 else None,
                bonded=bonded,
                carried=None if closed else tuple(sorted(seen - {tail})),
            )
        )
    return found


def _turn_past_anchor(
    coords: npt.NDArray[np.float64],
    trial: npt.NDArray[np.float64],
    head: int,
    tail: _TailAnchor,
    reference: npt.NDArray[np.float64],
    torsion: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Turn a new unit about the bond into the previous unit's tail anchor.

    Spinning the new unit alone about the previous unit's head-to-tail line
    leaves its head anywhere around the anchor: in the place of one of the
    anchor's own substituents, or, where that line is not a bond, against the
    backbone atom before it. The relaxation then sorts the atoms out, and
    where one of them is a stereocentre it picks the configuration. Instead
    the head goes first into the anchor's open valence - opposite the sum of
    the bonds it already has - and then the head and the anchor's
    substituents turn together about the real bond into the anchor, so the
    torsion changes and nothing else does.

    Args:
        coords: Positions placed so far.
        trial: The new unit's positions.
        head: The new unit's head anchor, as an index into *trial*.
        tail: The previous unit's tail anchor.
        reference: What to measure the torsion from when *tail* has no
            backbone atom before its previous one.
        torsion: The dihedral wanted, in degrees.

    Returns:
        The new unit's positions, and the anchor's carried atoms moved with
        it, or None when the bond has nothing to turn.
    """
    pivot = coords[tail.anchor]
    bonds = coords[list(tail.bonded)] - pivot
    valence = -(bonds / np.linalg.norm(bonds, axis=1)[:, None]).sum(axis=0)
    current = trial[head] - pivot
    axis = np.cross(current, valence)
    sine = float(np.linalg.norm(axis))
    if sine > 1e-9 * float(np.linalg.norm(current) * np.linalg.norm(valence)):
        rotation = _axis_rotation(
            axis / sine, math.atan2(sine, float(current @ valence))
        )
        trial = (trial - pivot) @ rotation.T + pivot
    if tail.carried is None:
        return trial, None

    moved = np.vstack((trial, coords[list(tail.carried)]))
    previous = coords[tail.previous]
    moved = _set_torsion(
        moved,
        pivot=pivot,
        axis=pivot - previous,
        current=_dihedral(
            coords[tail.before] if tail.before is not None else reference,
            previous,
            pivot,
            trial[head],
        ),
        target=torsion,
    )
    return moved[: len(trial)], moved[len(trial) :]


def _adjacent_pairs(
    mol: Chem.Mol,
    groups: Sequence[Sequence[int]],
    heads: Sequence[int],
    tails: Sequence[int],
) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
    """Return each unit's heavy-atom pairs with the unit behind, more than three bonds apart.

    Units meet only at the junction bond, so a pair's separation is the bond
    distance from one atom to its unit's tail anchor, one for the junction, and
    the bond distance from the next unit's head anchor to the other. Closer
    pairs are held where they are by bonds, angles and the torsions growth
    sets; further ones have no business overlapping.

    Returns:
        One pair of arrays per unit: positions in the previous unit and the
        matching positions in this one. Both are empty for the first unit.
    """

    def separations(start: int, group: Sequence[int]) -> dict[int, int]:
        members = set(group)
        found = {start: 0}
        frontier = [start]
        while frontier:
            following = []
            for index in frontier:
                for neighbour in mol.GetAtomWithIdx(index).GetNeighbors():
                    other = int(neighbour.GetIdx())
                    if other in members and other not in found:
                        found[other] = found[index] + 1
                        following.append(other)
            frontier = following
        return found

    def heavy(group: Sequence[int]) -> list[tuple[int, int]]:
        return [
            (position, index)
            for position, index in enumerate(group)
            if mol.GetAtomWithIdx(index).GetAtomicNum() > 1
        ]

    none = np.array([], dtype=int)
    pairs = [(none, none)]
    for unit in range(1, len(groups)):
        behind = separations(tails[unit - 1], groups[unit - 1])
        ahead = separations(heads[unit], groups[unit])
        rows: list[int] = []
        columns: list[int] = []
        for row, first in heavy(groups[unit - 1]):
            for column, second in heavy(groups[unit]):
                if behind[first] + 1 + ahead[second] > 3:
                    rows.append(row)
                    columns.append(column)
        pairs.append((np.array(rows, dtype=int), np.array(columns, dtype=int)))
    return pairs


def _clashes(
    grid: CellList,
    points: npt.NDArray[np.float64],
    cutoff: float,
    limit: int | None = None,
) -> bool:
    """Whether any of *points* lies within *cutoff* of a point in *grid*.

    Only the grid's first *limit* points are considered, when it is given.
    """
    squared = cutoff * cutoff
    for point in points:
        for index in grid.neighbours(point):
            if limit is not None and index >= limit:
                continue
            delta = point - grid.points[index]
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
    mol: Chem.Mol,
    spec: ChainSpec,
    blocks: Sequence[npt.NDArray[np.float64]],
    n_units: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Grow a self-avoiding conformer by stamping template units down the chain.

    Each unit is placed by one rigid transform from the previous unit's frame,
    then spun about the previous unit's backbone bond and about the junction
    bond joining the two, each by a torsion drawn from the trans/gauche set -
    except the second unit's first spin, which has nothing before it to be
    trans to and is drawn uniformly. In a chain with a stereocentre the first
    spin is :func:`_turn_past_anchor` instead, which keeps every configuration;
    see :func:`_tail_anchors` for why the rest keep the plain spin. A
    placement that puts a heavy atom within
    :data:`_OVERLAP_ANGSTROM` of a heavy atom two or more units back is
    rejected and redrawn; when a unit runs out of draws, growth backs up two
    units and tries again.

    Args:
        mol: The chain, with explicit hydrogens.
        spec: The chain spec, for the torsion statistics.
        blocks: Each unit's geometry, from :func:`_unit_blocks`.
        n_units: Degree of polymerization.
        seed: Seed for the torsion draws.

    Returns:
        Positions in angstrom.

    Raises:
        ChainError: Growth could not find a self-avoiding path.
    """
    groups = _unit_atoms(mol, n_units)
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
    heavy_global = [
        np.asarray(group)[local]
        for group, local in zip(groups, heavy_local, strict=True)
    ]
    heads = _anchors(mol, n_units, "head")
    tails = _anchors(mol, n_units, "tail")
    tail_anchors = _tail_anchors(mol, n_units, tails)
    stereo = any(tail is not None for tail in tail_anchors)
    # What turning past each unit's tail anchor moves of that unit, as
    # positions in the unit, and which of those are heavy, for clash tests.
    carried = [
        np.array(
            [group.index(index) for index in (tail.carried or ())]
            if tail is not None
            else [],
            dtype=int,
        )
        for group, tail in zip(groups, tail_anchors, strict=True)
    ]
    carried_heavy = [
        np.intersect1d(moved, heavy, assume_unique=True)
        for moved, heavy in zip(carried, heavy_local, strict=True)
    ]
    adjacent = _adjacent_pairs(mol, groups, heads, tails) if stereo else None

    for unit, block in enumerate(blocks):
        if len(groups[unit]) != len(block):
            raise ChainError(  # pragma: no cover - guards the ordering contract
                f"Unit {unit} has {len(groups[unit])} atoms but its template "
                f"block has {len(block)}. The chain and the template oligomer "
                "disagree about atom ordering."
            )

    rng = random.Random(seed)
    p_trans = _trans_fraction(spec.characteristic_ratio)
    weights = (p_trans, (1.0 - p_trans) / 2.0, (1.0 - p_trans) / 2.0)

    coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float64)
    coords[groups[0]] = blocks[0]

    grid = CellList(_OVERLAP_ANGSTROM)
    grid_upto = -1
    # How many points the grid holds once each unit is in, so a test can stop
    # short of the most recent unit.
    grid_sizes: list[int] = []
    unit = 1
    restarts = 0

    while unit < n_units:
        # Only units two or more back can clash: the one immediately behind is
        # bonded to this one and is meant to be in contact.
        target = unit - 2
        if grid_upto > target:
            grid = CellList(_OVERLAP_ANGSTROM)
            grid_upto = -1
            grid_sizes = []
        while grid_upto < target:
            grid_upto += 1
            grid.add(coords[heavy_global[grid_upto]])
            grid_sizes.append(len(grid.points))

        reference = (
            coords[tails[unit - 2]]
            if unit >= 2
            else coords[tails[unit - 1]] + _SEED_REFERENCE
        )
        frame = _frame(coords[heads[unit - 1]], coords[tails[unit - 1]], reference)
        block = blocks[unit]
        head_in_block = groups[unit].index(heads[unit])
        tail_in_block = groups[unit].index(tails[unit])
        tail_anchor = tail_anchors[unit - 1]
        # The carried atoms belong to the unit behind, so only units three or
        # more back are tested against them.
        carried_limit = grid_sizes[unit - 3] if unit >= 3 else 0

        for _ in range(_MAX_TRIES):
            trial = _to_world(frame, block)
            torsion = (
                rng.uniform(-180.0, 180.0)
                if unit == 1
                else rng.choices(_TORSION_TARGETS_DEG, weights=weights)[0]
            )
            # Two backbone bonds are opened by adding a unit: the previous
            # unit's own head-to-tail bond, and the junction bond joining the
            # two. Sampling only one of them leaves every second torsion frozen
            # at whatever the template relaxed into, which is how a chain ends
            # up a third too compact with the statistics apparently right.
            previous = coords[groups[unit - 1]]
            if tail_anchor is None:
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
                    target=torsion,
                )
            else:
                trial, turned = _turn_past_anchor(
                    coords, trial, head_in_block, tail_anchor, reference, torsion
                )
                if turned is not None:
                    previous[carried[unit - 1]] = turned
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
            if target >= 0 and _clashes(
                grid, trial[heavy_local[unit]], _OVERLAP_ANGSTROM
            ):
                continue
            if carried_limit and _clashes(
                grid,
                previous[carried_heavy[unit - 1]],
                _OVERLAP_ANGSTROM,
                carried_limit,
            ):
                continue
            # The unit behind is bonded to this one, so only pairs more than
            # three bonds apart are tested, and only in a chain with a
            # stereocentre: one caught between interpenetrating units is one
            # the relaxation can turn inside out.
            if adjacent is not None:
                behind, ahead = adjacent[unit]
                gaps = previous[behind] - trial[ahead]
                squared = np.einsum("ij,ij->i", gaps, gaps)
                if np.any(squared < _ADJACENT_OVERLAP_ANGSTROM**2):
                    continue
            coords[groups[unit]] = trial
            coords[groups[unit - 1]] = previous
            break
        else:
            restarts += 1
            if restarts > _MAX_RESTARTS:
                raise ChainError(
                    f"Could not grow a self-avoiding conformer of "
                    f"{spec.residue_name} past unit {unit} of {n_units} after "
                    f"{_MAX_RESTARTS} back-ups. Try a different seed, or a "
                    "shorter chain."
                )
            unit = max(1, unit - 2)
            continue
        unit += 1

    return coords


def _misbuilt_units(mol: Chem.Mol) -> list[int]:
    """Return the units whose conformer contradicts one of their chiral tags.

    The SDF goes to forcefill with coordinates, and the simulation runs on
    them, so a centre whose geometry disagrees with its tag is built as the
    other stereoisomer whatever the tag and the recorded SMILES say.
    """
    tagged = [atom for atom in _atoms(mol) if atom.GetChiralTag() in _TETRAHEDRAL]
    if not tagged:
        return []
    perceived = Chem.Mol(mol)
    Chem.AssignAtomChiralTagsFromStructure(perceived)
    return sorted(
        {
            atom.GetIntProp("_omp_unit")
            for atom in tagged
            if perceived.GetAtomWithIdx(atom.GetIdx()).GetChiralTag()
            != atom.GetChiralTag()
        }
    )


def _set_conformer(mol: Chem.Mol, coords: npt.NDArray[np.float64]) -> None:
    """Replace *mol*'s conformers with one holding *coords* (angstrom)."""
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conformer, assignId=True)


def _atom_names(mol: Chem.Mol) -> list[str]:
    """Return unique PDB atom names, at most four characters each.

    PDB atom names are four columns wide and ``PDBFile.writeModel`` truncates
    to four without complaint, so a naive ``C1000`` scheme silently produces
    duplicates. Counting per element in base 36 fits 46655 carbons.

    Raises:
        ChainError: An element has more atoms than four columns can name.
    """
    counts: dict[str, int] = {}
    names: list[str] = []
    for atom in _atoms(mol):
        symbol = atom.GetSymbol().upper()
        counts[symbol] = counts.get(symbol, 0) + 1
        name = f"{symbol}{np.base_repr(counts[symbol], 36)}"
        if len(name) > 4:
            raise ChainError(
                f"{counts[symbol]} {symbol} atoms cannot be given unique "
                "four-character PDB names. Build a shorter chain."
            )
        names.append(name)
    return names


def _label_atoms(mol: Chem.Mol, residue_name: str) -> None:
    """Attach PDB residue information, so the whole chain is one residue."""
    for atom, name in zip(_atoms(mol), _atom_names(mol), strict=True):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.ljust(4))
        info.SetResidueName(residue_name.ljust(3))
        info.SetResidueNumber(1)
        info.SetChainId("A")
        info.SetOccupancy(1.0)
        info.SetTempFactor(0.0)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)


def _radius_of_gyration_nm(mol: Chem.Mol, coords: npt.NDArray[np.float64]) -> float:
    """Return the mass-weighted radius of gyration, in nanometres."""
    masses = np.array([atom.GetMass() for atom in _atoms(mol)], dtype=np.float64)
    centre = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    offsets = coords - centre
    squared = float((masses * (offsets * offsets).sum(axis=1)).sum() / masses.sum())
    return math.sqrt(squared) / ANGSTROM_PER_NM


def _extent_bound_nm(coords: npt.NDArray[np.float64]) -> float:
    """Return twice the largest distance from the centroid, in nanometres.

    A bound that never understates the conformer's largest interatomic
    distance, at the cost of one pass rather than the N-squared of every pair.
    """
    centre = coords.mean(axis=0)
    offsets = coords - centre
    return 2.0 * float(np.sqrt((offsets * offsets).sum(axis=1)).max()) / ANGSTROM_PER_NM


def _mean_bond_length_nm(
    coords: npt.NDArray[np.float64], backbone: Sequence[int]
) -> float:
    """Return the mean backbone bond length, in nanometres."""
    points = coords[list(backbone)]
    steps = points[1:] - points[:-1]
    return float(np.sqrt((steps * steps).sum(axis=1)).mean()) / ANGSTROM_PER_NM


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
            units or more rather than let ETKDG collapse it; see the module
            docstring.

    Returns:
        What was built, including the paths to feed packing and
        parameterisation.

    Raises:
        ChainError: The monomer, a cap or the embedding failed, or the chain is
            too short for ``embedder="grown"`` or too long to name its atoms.
    """
    require_integer(n_conformers, name="n_conformers")
    require_choice(embedder, EMBEDDERS, name="embedder")
    directory = Path(output_dir) if output_dir is not None else Path()
    directory.mkdir(parents=True, exist_ok=True)

    units = spec.degree_of_polymerization
    choices = _tacticity_choices(spec, units)
    molecule = _add_hydrogens(_assemble(spec, choices))
    _label_atoms(molecule, spec.residue_name)
    n_atoms = molecule.GetNumAtoms()
    backbone = backbone_path(molecule, units)

    if embedder == "grown" and units < _MIN_GROWN_UNITS:
        raise ChainError(
            f"embedder='grown' needs at least {_MIN_GROWN_UNITS} units; this "
            f"chain has {units}. Use embedder='etkdg' or 'auto'."
        )
    grown = embedder == "grown" or (embedder == "auto" and units >= _MIN_GROWN_UNITS)
    blocks = _unit_blocks(spec, molecule, choices) if grown else None
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
        if blocks is not None:
            coords = _grow_conformer(conformer_mol, spec, blocks, units, seed)
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
        wrong = _misbuilt_units(conformer_mol)
        if wrong:
            raise ChainError(
                f"Conformer {index} of {spec.residue_name} contradicts the "
                f"configuration asked for at a stereocentre in unit(s) {wrong}, "
                "so its coordinates are a different stereoisomer. Try another "
                "seed, or the other embedder."
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
                _characteristic_ratio(
                    coords / ANGSTROM_PER_NM,
                    backbone,
                    _mean_bond_length_nm(coords, backbone),
                )
            )

    measured = float(np.mean(ratios)) if ratios else None
    _report_ratio(measured, spec, len(backbone) - 1, n_conformers)

    return ChainResult(
        sdf_paths=tuple(sdf_paths),
        pdb_paths=tuple(pdb_paths),
        smiles=Chem.MolToSmiles(Chem.RemoveHs(molecule)),
        n_atoms=n_atoms,
        molar_mass_g_mol=float(sum(atom.GetMass() for atom in _atoms(molecule))),
        radius_of_gyration_nm=tuple(radii),
        max_extent_nm=tuple(extents),
        backbone=backbone,
        characteristic_ratio=measured,
        embedder="grown" if grown else "etkdg",
    )


def _report_ratio(
    measured: float | None, spec: ChainSpec, n_bonds: int, n_conformers: int
) -> None:
    """Say so when the built conformers are not the dimensions asked for.

    Both signs are checked, since the two embedders err in opposite directions.
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
