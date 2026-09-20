"""Assigning partial charges to a whole polymer chain.

Charges are the practical wall in all-atom polymer work. AM1-BCC - the
reference method, and what both of forcefill's main backends reach for by
default - runs a semi-empirical QM calculation in ``sqm``, which stops being
tractable somewhere around a couple of hundred atoms. A twenty-unit
polyethylene chain is already there; anything longer is not going to finish.

The way round it is to charge the chain here, with a graph neural network that
scales with the number of atoms rather than their cube, and write the result
into the SDF. ``openmmforcefields`` checks whether the molecule it is handed
already carries charges and, if so, passes them through as
``charge_from_molecules`` instead of running AM1-BCC - so forcefill's
``smirnoff`` backend picks them up with no special handling at all.

=============  ========================  ===================================
Method         Practical chain size      Notes
=============  ========================  ===================================
``nagl``       thousands of atoms        The default. AM1-BCC-quality.
``am1bcc``     a few hundred atoms       The reference; ``sqm`` is the wall.
``gasteiger``  unbounded                 Qualitative only, see below.
``none``       -                         Leave it to the backend.
=============  ========================  ===================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._validation import require_choice

log = logging.getLogger(__name__)

#: Accepted charge methods.
CHARGE_METHODS = ("nagl", "am1bcc", "gasteiger", "none")

#: How far the charges may sum from the formal charge before it is an error.
#: PME neutralises a charged cell with a uniform background whether or not you
#: meant it, so a charge set that does not sum to the formal charge changes the
#: physics silently.
CHARGE_SUM_TOLERANCE = 1e-3


class ChargeError(RuntimeError):
    """Partial charges could not be assigned."""


@dataclass(frozen=True)
class ChargeResult:
    """What :func:`assign_charges` did.

    Args:
        sdf_path: The SDF now carrying the charges.
        method: The method used.
        model: The NAGL model file, when one was used.
        total_charge: The charges' sum, in elementary charge.
        formal_charge: The molecule's formal charge.
        n_atoms: How many atoms were charged.
    """

    sdf_path: str
    method: str
    model: str | None
    total_charge: float
    formal_charge: int
    n_atoms: int


def default_nagl_model() -> str:
    """Return the newest released NAGL AM1-BCC model installed.

    Release candidates and alphas are skipped when a final release is present,
    because pinning a package to a name like ``openff-gnn-am1bcc-0.1.0-rc.3``
    is how a workflow stops working when the model that supersedes it lands.

    Returns:
        The model file name, as ``assign_partial_charges`` wants it.

    Raises:
        ChargeError: openff-nagl-models is not installed, or ships no model.
    """
    try:
        from openff.nagl_models import get_models_by_type
    except ImportError as error:
        raise ChargeError(
            "method='nagl' needs openff-nagl and openff-nagl-models: "
            "conda install -c conda-forge openff-nagl openff-nagl-models. "
            "Without it, charge a chain of more than a couple of hundred "
            "atoms with method='gasteiger' and read the caveat."
        ) from error

    names = [path.name for path in get_models_by_type("am1bcc")]
    if not names:
        raise ChargeError("openff-nagl-models is installed but ships no AM1-BCC model.")
    released = [name for name in names if "rc" not in name and "alpha" not in name]
    return str(sorted(released or names, key=_version_key)[-1])


def _version_key(name: str) -> tuple[int, ...]:
    """Sort key over the digits in a NAGL model file name."""
    digits: list[int] = []
    current = ""
    for character in name:
        if character.isdigit():
            current += character
        elif current:
            digits.append(int(current))
            current = ""
    if current:
        digits.append(int(current))
    return tuple(digits)


def assign_charges(
    sdf_path: str | Path,
    method: str = "nagl",
    *,
    model: str | None = None,
    output_sdf: str | Path | None = None,
) -> ChargeResult:
    """Charge the molecule in *sdf_path* and write it back out.

    Args:
        sdf_path: An SDF holding exactly one molecule, as
            :func:`openmmpolymer.chain.build_chain` writes.
        method: One of :data:`CHARGE_METHODS`.
        model: NAGL model file name. Defaults to :func:`default_nagl_model`.
        output_sdf: Where to write. Defaults to overwriting *sdf_path*.

    Returns:
        What was assigned.

    Raises:
        ChargeError: The file does not hold one molecule, the method is not
            available, or the charges do not sum to the formal charge.
    """
    require_choice(method, CHARGE_METHODS, name="method")
    source = Path(sdf_path)
    destination = Path(output_sdf) if output_sdf is not None else source

    from openff.toolkit import Molecule

    molecule = Molecule.from_file(str(source), allow_undefined_stereo=True)
    if isinstance(molecule, list):
        if len(molecule) != 1:
            raise ChargeError(
                f"{source} holds {len(molecule)} molecules; a chain SDF holds "
                "exactly one."
            )
        molecule = molecule[0]

    formal_charge = round(float(molecule.total_charge.m))
    if method == "none":
        log.info("Leaving %s uncharged; the backend will charge it.", source.name)
        return ChargeResult(
            sdf_path=str(source),
            method=method,
            model=None,
            total_charge=float(formal_charge),
            formal_charge=formal_charge,
            n_atoms=molecule.n_atoms,
        )

    chosen = _assign(molecule, method, model)
    total = float(sum(molecule.partial_charges.m))
    _check_total(total, formal_charge, method)

    if not molecule.conformers:  # pragma: no cover - build_chain always embeds
        molecule.generate_conformers(n_conformers=1)
    molecule.to_file(str(destination), file_format="SDF")
    log.info(
        "Charged %d atoms of %s with %s%s (sum %+.4f e).",
        molecule.n_atoms,
        source.name,
        method,
        f" [{chosen}]" if chosen else "",
        total,
    )
    return ChargeResult(
        sdf_path=str(destination),
        method=method,
        model=chosen,
        total_charge=total,
        formal_charge=formal_charge,
        n_atoms=molecule.n_atoms,
    )


def _assign(molecule: Any, method: str, model: str | None) -> str | None:
    """Run the chosen charge method on *molecule* in place."""
    if method == "nagl":
        chosen = model or default_nagl_model()
        molecule.assign_partial_charges(chosen)
        return chosen
    if method == "gasteiger":
        log.warning(
            "Gasteiger charges have no hydrogen bonding and no dipole "
            "calibration. For a hydrocarbon they are survivable; for an "
            "ester, ether, amide or halide they put the melt density out by "
            "several per cent and the glass transition out further. Use "
            "method='nagl' unless you know why you are not."
        )
    molecule.assign_partial_charges(method)
    return None


def _check_total(total: float, formal_charge: int, method: str) -> None:
    """Raise unless the charges sum to the formal charge.

    Args:
        total: The charges' sum.
        formal_charge: What it should be.
        method: The method used, for the error message.

    Raises:
        ChargeError: The two disagree by more than
            :data:`CHARGE_SUM_TOLERANCE`.
    """
    if math.isclose(total, formal_charge, abs_tol=CHARGE_SUM_TOLERANCE):
        return
    raise ChargeError(
        f"{method} charges sum to {total:+.6f} e but the molecule's formal "
        f"charge is {formal_charge:+d}. A cell that is not neutral gets a "
        "uniform neutralising background under PME, silently, so this is not "
        "a rounding matter. Gasteiger in particular does not renormalise."
    )
