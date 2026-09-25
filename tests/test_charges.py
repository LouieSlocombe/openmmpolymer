"""Tests for partial-charge assignment.

The charge step is what makes a chain of more than a few hundred atoms
possible, so most of this needs the real toolkit. What does not is the
sum-to-formal-charge guard, which is the one that stops a silently
non-neutral cell.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openmmpolymer.charges import (
    CHARGE_SUM_TOLERANCE,
    ChargeError,
    _check_total,
    _version_key,
    assign_charges,
)

from .helpers import write_two_molecule_sdf


def test_an_unknown_method_is_refused_by_name() -> None:
    """Before anything expensive is attempted."""
    with pytest.raises(ValueError, match="method"):
        assign_charges("nowhere.sdf", "am1bbc")


def test_charges_must_sum_to_the_formal_charge() -> None:
    """Agreeing with the formal charge is the test, not neutrality.

    A cell that is not neutral gets a neutralising background under PME,
    silently. So a charge set that does not add up is an error here rather
    than a surprise in the energies later.
    """
    _check_total(CHARGE_SUM_TOLERANCE / 2, 0, "nagl")
    _check_total(-1.0, -1, "nagl")
    with pytest.raises(ChargeError, match="not a rounding matter"):
        _check_total(0.35, 0, "gasteiger")
    with pytest.raises(ChargeError, match="formal charge is -1"):
        _check_total(0.0, -1, "nagl")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("openff-gnn-am1bcc-1.0.0.pt", (1, 1, 0, 0)),
        ("openff-gnn-am1bcc-0.1.0-rc.3.pt", (1, 0, 1, 0, 3)),
    ],
)
def test_model_names_sort_by_their_digits(name: str, expected: tuple[int, ...]) -> None:
    """So that 1.0.0 beats 0.1.0, rather than losing a string comparison."""
    assert _version_key(name) == expected


@pytest.fixture
def methanol_sdf(tmp_path: Path) -> str:
    """A small molecule written the way build_chain writes a chain."""
    from openff.toolkit import Molecule

    molecule = Molecule.from_smiles("CO")
    molecule.generate_conformers(n_conformers=1)
    path = tmp_path / "methanol.sdf"
    molecule.to_file(str(path), file_format="SDF")
    return str(path)


# openff.nagl_models._dynamic_fetch._get_sha256 leaves the model file unclosed.
# Filter its ResourceWarning before pytest wraps it as an unraisable exception:
# the wrapper wording differs between Python 3.12 and 3.14. Keep the exception
# scoped to NAGL AM1-BCC model files in this test; other warnings remain errors.
@pytest.mark.filterwarnings(
    r"ignore:unclosed file .*openff-gnn-am1bcc-.*\.pt['\"]>:ResourceWarning"
)
@pytest.mark.forcefield
def test_nagl_charges_a_molecule_with_a_released_model(methanol_sdf: str) -> None:
    """The whole point: the SDF comes back carrying charges.

    From a released model, because pinning a release candidate is how a
    workflow stops working when the model that supersedes it lands.
    """
    pytest.importorskip("openff.nagl_models")
    from openff.toolkit import Molecule

    result = assign_charges(methanol_sdf, "nagl")
    assert result.method == "nagl"
    assert result.total_charge == pytest.approx(0.0, abs=CHARGE_SUM_TOLERANCE)
    assert result.model is not None
    assert result.model.endswith(".pt")
    assert "rc" not in result.model
    assert "alpha" not in result.model
    assert Molecule.from_file(result.sdf_path).partial_charges is not None


def test_gasteiger_works_and_says_what_it_costs(
    methanol_sdf: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The fallback for a chain too big for anything else, with its caveat.

    It has no hydrogen bonding and no dipole calibration; that has to be said.
    """
    with caplog.at_level("WARNING"):
        result = assign_charges(methanol_sdf, "gasteiger")
    assert result.method == "gasteiger"
    assert result.model is None
    assert result.n_atoms == 6
    assert "dipole calibration" in caplog.text


def test_none_leaves_the_file_alone(methanol_sdf: str) -> None:
    """A legitimate choice: let the backend charge it."""
    before = Path(methanol_sdf).read_text()
    result = assign_charges(methanol_sdf, "none")
    assert result.model is None
    assert Path(methanol_sdf).read_text() == before


def test_charges_can_be_written_somewhere_else(
    methanol_sdf: str, tmp_path: Path
) -> None:
    """The input stays as it was when an output is named."""
    before = Path(methanol_sdf).read_text()
    destination = tmp_path / "charged.sdf"
    result = assign_charges(methanol_sdf, "gasteiger", output_sdf=destination)
    assert result.sdf_path == str(destination)
    assert destination.is_file()
    assert Path(methanol_sdf).read_text() == before


def test_an_sdf_holding_more_than_one_molecule_is_refused(tmp_path: Path) -> None:
    """A chain SDF holds exactly one, and quietly charging the first is worse."""
    with pytest.raises(ChargeError, match="holds 2 molecules"):
        assign_charges(write_two_molecule_sdf(tmp_path / "two.sdf"), "gasteiger")
