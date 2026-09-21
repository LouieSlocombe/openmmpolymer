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
    default_nagl_model,
)

openff = pytest.importorskip("openff.toolkit")


def test_an_unknown_method_is_refused_by_name() -> None:
    """Before anything expensive is attempted."""
    with pytest.raises(ValueError, match="method"):
        assign_charges("nowhere.sdf", "am1bbc")


def test_charges_must_sum_to_the_formal_charge() -> None:
    """A cell that is not neutral gets a neutralising background under PME.

    Silently. So a charge set that does not add up is an error here rather
    than a surprise in the energies later.
    """
    _check_total(0.0, 0, "nagl")
    _check_total(CHARGE_SUM_TOLERANCE / 2, 0, "nagl")
    with pytest.raises(ChargeError, match="not a rounding matter"):
        _check_total(0.35, 0, "gasteiger")


def test_a_charged_molecule_must_sum_to_its_own_charge() -> None:
    """Neutrality is not the test; agreeing with the formal charge is."""
    _check_total(-1.0, -1, "nagl")
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


@pytest.mark.forcefield
def test_the_default_model_is_a_released_one() -> None:
    """Pinning a release candidate is how a workflow stops working later."""
    pytest.importorskip("openff.nagl_models")
    model = default_nagl_model()
    assert model.endswith(".pt")
    assert "rc" not in model
    assert "alpha" not in model


@pytest.fixture
def methanol_sdf(tmp_path: Path) -> str:
    """A small molecule written the way build_chain writes a chain."""
    from openff.toolkit import Molecule

    molecule = Molecule.from_smiles("CO")
    molecule.generate_conformers(n_conformers=1)
    path = tmp_path / "methanol.sdf"
    molecule.to_file(str(path), file_format="SDF")
    return str(path)


# Observed loading a NAGL model, which reads a torch archive:
#   PytestUnraisableExceptionWarning: Exception ignored while finalizing file
#   <_io.FileIO name='.../openff-gnn-am1bcc-1.0.0.pt' mode='rb' closefd=True>
# A file handle the model loader leaves to the garbage collector. Scoped to
# the two tests that load a model rather than relaxed globally, so the
# suite-wide "warnings are errors" stays as it is.
_NAGL_LOADER_NOISE = pytest.mark.filterwarnings(
    "ignore:Exception ignored while finalizing file"
    ":pytest.PytestUnraisableExceptionWarning"
)


@_NAGL_LOADER_NOISE
@pytest.mark.forcefield
def test_nagl_charges_a_molecule_and_writes_them_back(methanol_sdf: str) -> None:
    """The whole point: the SDF comes back carrying charges."""
    pytest.importorskip("openff.nagl_models")
    from openff.toolkit import Molecule

    result = assign_charges(methanol_sdf, "nagl")
    assert result.method == "nagl"
    assert result.total_charge == pytest.approx(0.0, abs=CHARGE_SUM_TOLERANCE)

    reloaded = Molecule.from_file(result.sdf_path)
    assert reloaded.partial_charges is not None


@_NAGL_LOADER_NOISE
@pytest.mark.forcefield
def test_openmmforcefields_uses_the_charges_rather_than_running_am1bcc(
    methanol_sdf: str,
) -> None:
    """The mechanism the whole chain-length story rests on.

    ``SMIRNOFFTemplateGenerator`` checks whether the molecule it is handed
    already has charges and, if so, passes them through instead of calling
    AM1-BCC. Without that, nothing longer than a short oligomer is reachable.
    """
    pytest.importorskip("openff.nagl_models")
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    assign_charges(methanol_sdf, "nagl")
    molecule = Molecule.from_file(methanol_sdf)
    generator = SMIRNOFFTemplateGenerator(forcefield="openff-2.2.1")
    assert generator._molecule_has_user_charges(molecule)


def test_gasteiger_works_without_any_optional_toolkit(methanol_sdf: str) -> None:
    """The fallback for a chain too big for anything else."""
    result = assign_charges(methanol_sdf, "gasteiger")
    assert result.method == "gasteiger"
    assert result.model is None
    assert result.n_atoms == 6


def test_gasteiger_says_what_it_costs(
    methanol_sdf: str, caplog: pytest.LogCaptureFixture
) -> None:
    """It has no hydrogen bonding and no dipole calibration; that has to be said."""
    with caplog.at_level("WARNING"):
        assign_charges(methanol_sdf, "gasteiger")
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
    destination = tmp_path / "charged.sdf"
    result = assign_charges(methanol_sdf, "gasteiger", output_sdf=destination)
    assert result.sdf_path == str(destination)
    assert destination.is_file()


def test_an_sdf_holding_more_than_one_molecule_is_refused(tmp_path: Path) -> None:
    """A chain SDF holds exactly one, and quietly charging the first is worse."""
    from openff.toolkit import Molecule

    path = tmp_path / "two.sdf"
    with path.open("w") as handle:
        for smiles in ("CO", "CC"):
            molecule = Molecule.from_smiles(smiles)
            molecule.generate_conformers(n_conformers=1)
            handle.write(Path(_written(molecule, tmp_path)).read_text())
    with pytest.raises(ChargeError, match="holds 2 molecules"):
        assign_charges(path, "gasteiger")


def _written(molecule: object, tmp_path: Path) -> str:
    """Write one molecule to its own SDF and return the path."""
    single = tmp_path / "one.sdf"
    molecule.to_file(str(single), file_format="SDF")  # type: ignore[attr-defined]
    return str(single)
