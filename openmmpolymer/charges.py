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
import re
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


def read_chain_molecule(sdf_path: Path, error: type[RuntimeError] = ChargeError) -> Any:
    """Return the one molecule a chain SDF holds, as an OpenFF ``Molecule``.

    Args:
        sdf_path: The SDF, as :func:`openmmpolymer.chain.build_chain` writes it.
        error: What to raise if it holds more than one, so each caller reports
            the fault as its own.

    Raises:
        ChargeError: Or *error*: the file holds more than one molecule.
    """
    from openff.toolkit import Molecule

    molecule = Molecule.from_file(str(sdf_path), allow_undefined_stereo=True)
    if not isinstance(molecule, list):
        return molecule
    if len(molecule) != 1:
        raise error(
            f"{sdf_path} holds {len(molecule)} molecules; a chain SDF holds "
            "exactly one."
        )
    return molecule[0]


def _default_nagl_model() -> str:
    """Return the newest released NAGL AM1-BCC model installed.

    Release candidates and alphas are skipped when a final release is present,
    because pinning a package to a name like ``openff-gnn-am1bcc-0.1.0-rc.3``
    is how a workflow stops working when the model that supersedes it lands.

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
    """Sort key over the digit runs in a NAGL model file name."""
    return tuple(int(digits) for digits in re.findall(r"\d+", name))


def assign_charges(
    sdf_path: str | Path,
    method: str = "nagl",
    *,
    output_sdf: str | Path | None = None,
) -> ChargeResult:
    """Charge the molecule in *sdf_path* and write it back out.

    Args:
        sdf_path: An SDF holding exactly one molecule, as
            :func:`openmmpolymer.chain.build_chain` writes.
        method: One of :data:`CHARGE_METHODS`. ``nagl`` uses the newest
            released model installed.
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

    molecule = read_chain_molecule(source)
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

    model = _assign(molecule, method)
    total = float(sum(molecule.partial_charges.m))
    _check_total(total, formal_charge, method)

    molecule.to_file(str(destination), file_format="SDF")
    log.info(
        "Charged %d atoms of %s with %s%s (sum %+.4f e).",
        molecule.n_atoms,
        source.name,
        method,
        f" [{model}]" if model else "",
        total,
    )
    return ChargeResult(
        sdf_path=str(destination),
        method=method,
        model=model,
        total_charge=total,
        formal_charge=formal_charge,
        n_atoms=molecule.n_atoms,
    )


def _assign(molecule: Any, method: str) -> str | None:
    """Run the chosen charge method on *molecule* in place; return the model."""
    if method == "nagl":
        model = _default_nagl_model()
        molecule.assign_partial_charges(model)
        return model
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
    """Raise unless the charges sum to the formal charge."""
    if math.isclose(total, formal_charge, abs_tol=CHARGE_SUM_TOLERANCE):
        return
    raise ChargeError(
        f"{method} charges sum to {total:+.6f} e but the molecule's formal "
        f"charge is {formal_charge:+d}. A cell that is not neutral gets a "
        "uniform neutralising background under PME, silently, so this is not "
        "a rounding matter. Gasteiger in particular does not renormalise."
    )
