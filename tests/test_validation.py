"""Tests for the shared argument validators.

Everything numeric that crosses into this package goes through one of these,
so that a negative temperature or a typo'd unit fails at the call site rather
than a hundred picoseconds into a run.
"""

from __future__ import annotations

import math

import pytest
from openmm import unit

from openmmpolymer._seeds import derive_seed, seed_random_stream
from openmmpolymer._validation import (
    require_choice,
    require_finite,
    require_integer,
    require_positive,
)


def test_a_bare_number_is_taken_to_be_in_the_expected_unit() -> None:
    """Public signatures take floats; the unit is in the parameter name."""
    assert require_positive(300.0, unit.kelvin, name="temperature_k") == 300.0


def test_a_quantity_is_converted_rather_than_refused() -> None:
    """Callers coming from openmmnqe reach for one, and should not be stopped."""
    assert require_positive(
        2.0 * unit.picosecond, unit.femtosecond, name="timestep_fs"
    ) == pytest.approx(2000.0)


def test_something_that_is_not_a_number_says_what_was_wanted() -> None:
    """The message names the parameter and the unit."""
    with pytest.raises(TypeError, match="temperature_k"):
        require_positive("warm", unit.kelvin, name="temperature_k")


def test_a_quantity_in_the_wrong_dimension_is_refused() -> None:
    """Nanometres are not kelvin, however willing the arithmetic."""
    with pytest.raises(TypeError, match="not a number or a quantity"):
        require_positive(1.0 * unit.nanometer, unit.kelvin, name="temperature_k")


def test_a_quantity_is_converted_between_compatible_units() -> None:
    """Whatever it arrives as, the package works in the unit it named."""
    assert require_finite(
        1.0 * unit.nanometer, unit.angstrom, name="cutoff_nm"
    ) == pytest.approx(10.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nan_and_infinity_are_refused(value: float) -> None:
    """Either would propagate silently into a System."""
    with pytest.raises(ValueError, match="must be finite"):
        require_finite(value, unit.kelvin, name="temperature_k")


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_non_positive_value_is_refused_where_one_is_needed(value: float) -> None:
    """A timestep or a density of zero is not a setting."""
    with pytest.raises(ValueError, match="greater than zero"):
        require_positive(value, unit.kelvin, name="temperature_k")


def test_an_integer_is_required_where_one_is_meant() -> None:
    """A float count of chains would silently truncate somewhere."""
    assert require_integer(5, name="n_chains") == 5
    with pytest.raises(TypeError, match="must be an int"):
        require_integer(5.0, name="n_chains")


def test_a_boolean_is_not_an_integer_here() -> None:
    """True is 1 to Python and a mistake to a caller."""
    with pytest.raises(TypeError, match="must be an int"):
        require_integer(True, name="n_chains")


def test_an_integer_below_the_minimum_is_refused() -> None:
    """Zero chains is a different mistake from a negative count."""
    with pytest.raises(ValueError, match="at least 1"):
        require_integer(0, name="n_chains")
    assert require_integer(0, name="count", minimum=0) == 0


def test_a_typo_in_a_choice_lists_the_options() -> None:
    """Rather than failing later with something cryptic."""
    assert require_choice("gaff", ("gaff", "smirnoff"), name="backend") == "gaff"
    with pytest.raises(ValueError, match="gaff, smirnoff"):
        require_choice("gaf", ("gaff", "smirnoff"), name="backend")


def test_derived_seeds_are_stable_and_distinct() -> None:
    """One master seed, several streams, and they must not collide."""
    assert derive_seed(7, "thermostat") == derive_seed(7, "thermostat")
    assert derive_seed(7, "thermostat") != derive_seed(7, "barostat")
    assert derive_seed(7, "thermostat") != derive_seed(8, "thermostat")


def test_a_derived_seed_is_never_zero() -> None:
    """OpenMM reads zero as 'choose your own', losing the reproducibility.

    Swept rather than spot-checked, because the failure would be one label in
    a thousand quietly making a run irreproducible.
    """
    for index in range(5000):
        assert derive_seed(1, "conformer", str(index)) != 0


def test_a_derived_seed_fits_what_openmm_accepts() -> None:
    """A C int, so below 2**31."""
    for index in range(1000):
        assert 0 < derive_seed(3, "stage", str(index)) < 2**31


def test_seeding_a_stream_with_zero_is_refused() -> None:
    """The one value that means the opposite of what it looks like."""

    class Recorder:
        seed = -1

        def setRandomNumberSeed(self, value: int) -> None:
            self.seed = value

    recorder = Recorder()
    seed_random_stream(recorder, 42)
    assert recorder.seed == 42
    with pytest.raises(ValueError, match="irreproducible"):
        seed_random_stream(recorder, 0)
