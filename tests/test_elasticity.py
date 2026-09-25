"""Tests for the elastic-constant fits.

Everything here is arithmetic over numbers a manifest recorded, so every
test writes the manifest by hand. The curves are exactly linear with a
planted modulus, which means a correct fit recovers it to machine precision
and an assertion can be an equality - and, more usefully, that a fit which
comes back wrong is wrong by construction rather than by noise.

The other half of these is the refusals. A modulus is a slope, and a slope
through a short stretch of a noisy curve is always *something*; what has to
be tested is that the flag says so.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from openmmpolymer.elasticity import (
    MAX_CONSISTENCY_GAP,
    bulk_modulus,
    deform_stages,
    elastic_consistency,
    load_curve,
    poisson_ratio,
    shear_modulus,
    shear_stages,
    stress_strain,
    youngs_modulus,
)
from openmmpolymer.trajectory import AnalysisError

from .helpers import write_bulk, write_deformation, write_shear

E_PLANTED = 2000.0
NU_PLANTED = 0.35


# --------------------------------------------------------------------------
# Curves that were planted
# --------------------------------------------------------------------------


def test_an_exactly_linear_curve_gives_back_the_modulus_it_was_built_from(
    tmp_path: Path,
) -> None:
    """The load-bearing one. No tolerance, because there is no noise."""
    write_deformation(tmp_path, modulus_mpa=E_PLANTED, poisson=NU_PLANTED)
    fit = youngs_modulus(stress_strain(tmp_path), strain_limit=0.05)
    assert fit.modulus_mpa == pytest.approx(E_PLANTED)
    assert fit.resolved
    assert fit.n_points == 10


def test_poissons_ratio_comes_back_too(tmp_path: Path) -> None:
    """Fitted over the window, not taken as a ratio at the last point."""
    write_deformation(tmp_path, modulus_mpa=E_PLANTED, poisson=NU_PLANTED)
    ratio = poisson_ratio(stress_strain(tmp_path), strain_limit=0.05)
    assert ratio.ratio == pytest.approx(NU_PLANTED)
    assert ratio.resolved


def test_the_strain_rate_is_read_back_from_the_holds(tmp_path: Path) -> None:
    """Total strain over total time, and it travels with every fit."""
    write_deformation(tmp_path, n_steps=10, increment=0.002, relax_ps=50.0)
    curve = stress_strain(tmp_path)
    assert curve.strain_rate_per_ns == pytest.approx(curve.strain[-1] / 500.0 * 1000.0)
    assert youngs_modulus(curve).strain_rate_per_ns == curve.strain_rate_per_ns


def test_a_bulk_ladder_gives_back_its_modulus_and_no_hysteresis(
    tmp_path: Path,
) -> None:
    """Up and back down the same curve is zero hysteresis, to rounding."""
    write_bulk(tmp_path, modulus_mpa=1500.0)
    fit = bulk_modulus(tmp_path)
    assert fit.modulus_mpa == pytest.approx(1500.0)
    assert fit.compression_mpa == pytest.approx(1500.0)
    assert fit.decompression_mpa == pytest.approx(1500.0)
    assert fit.hysteresis == pytest.approx(0.0, abs=1e-9)
    assert fit.standard_error_mpa == pytest.approx(0.0, abs=1e-9)
    assert fit.relative_standard_error == pytest.approx(0.0, abs=1e-12)
    assert fit.residual_log_volume == pytest.approx(0.0, abs=1e-12)
    assert fit.half_disagreement == pytest.approx(0.0, abs=1e-9)
    assert fit.resolved


def _write_bulk_samples(
    directory: Path, pressures: list[float], densities: list[float]
) -> None:
    path = write_bulk(directory)
    record = json.loads(path.read_text())
    samples = record["stages"]["08_bulk"]["samples"]
    samples["segment_pressure_bar"] = pressures
    samples["segment_density_g_cm3"] = densities
    path.write_text(json.dumps(record))


def test_bulk_uncertainty_rejects_a_reversible_noisy_ladder(tmp_path: Path) -> None:
    """Retracing a noisy curve does not make its slope well determined."""
    pressure = np.asarray([1.0, 11.0, 21.0, 31.0, 21.0, 11.0, 1.0])
    noise = np.asarray([0.0, -0.04, 0.04, 0.0, 0.04, -0.04, 0.0])
    log_density = 0.0001 * pressure + noise
    _write_bulk_samples(tmp_path, pressure.tolist(), np.exp(log_density).tolist())
    fit = bulk_modulus(tmp_path)
    # Calculate the slope and uncertainty independently of the shared fitter.
    centered = pressure - pressure.mean()
    slope = float(centered @ (log_density - log_density.mean()) / (centered @ centered))
    residuals = log_density - log_density.mean() - slope * centered
    slope_error = math.sqrt(
        float(residuals @ residuals) / (pressure.size - 2) / float(centered @ centered)
    )
    assert fit.standard_error_mpa == pytest.approx(0.1 * slope_error / slope**2)
    assert fit.relative_standard_error == pytest.approx(slope_error / abs(slope))
    assert fit.relative_standard_error > 1.0
    assert fit.hysteresis == pytest.approx(0.0, abs=1e-12)
    assert not fit.resolved


def test_bulk_checks_linearity_across_pressure_ranges(tmp_path: Path) -> None:
    """Smooth curvature has small fit uncertainty and no branch hysteresis."""
    pressure = np.asarray([1.0, 11.0, 21.0, 31.0, 21.0, 11.0, 1.0])
    log_density = 0.0001 * pressure + 0.00001 * pressure**2
    _write_bulk_samples(tmp_path, pressure.tolist(), np.exp(log_density).tolist())
    fit = bulk_modulus(tmp_path)
    assert fit.relative_standard_error < 0.25
    assert fit.hysteresis == pytest.approx(0.0, abs=1e-12)
    assert fit.half_disagreement > 0.5
    assert not fit.resolved


def test_bulk_resolves_a_well_supported_noisy_slope(tmp_path: Path) -> None:
    pressure = np.asarray([1.0, 11.0, 21.0, 31.0, 21.0, 11.0, 1.0])
    noise = np.asarray([0.0, -1.0, 1.0, 0.0, 1.0, -1.0, 0.0]) * 1e-6
    _write_bulk_samples(
        tmp_path, pressure.tolist(), np.exp(0.0001 * pressure + noise).tolist()
    )
    fit = bulk_modulus(tmp_path)
    assert fit.resolved
    assert 0.0 < fit.relative_standard_error < 0.25
    assert fit.modulus_mpa == pytest.approx(1000.0, rel=0.001)


@pytest.mark.parametrize(
    "pressure", ([1.0] * 4, [1.0, 11.0, 1.0, 11.0], [1.0], [1.0, 11.0])
)
def test_bulk_needs_distinct_pressures_and_residual_degrees_of_freedom(
    tmp_path: Path, pressure: list[float]
) -> None:
    density = np.exp(0.0001 * np.asarray(pressure))
    _write_bulk_samples(tmp_path, pressure, density.tolist())
    fit = bulk_modulus(tmp_path, min_points=2)
    assert not fit.resolved
    if len(set(pressure)) == 1:
        assert math.isnan(fit.modulus_mpa)
        assert math.isinf(fit.relative_standard_error)
    if len(pressure) <= 2:
        assert math.isinf(fit.standard_error_mpa)


@pytest.mark.parametrize("slope", (0.0, -0.0001))
def test_bulk_refuses_zero_or_negative_compressibility(
    tmp_path: Path, slope: float
) -> None:
    pressure = [1.0, 11.0, 21.0, 31.0]
    _write_bulk_samples(
        tmp_path, pressure, np.exp(slope * np.asarray(pressure)).tolist()
    )
    assert not bulk_modulus(tmp_path).resolved


@pytest.mark.parametrize(
    ("pressure", "density"),
    (
        ([], []),
        ([1.0, 2.0], [0.9]),
        ([1.0, math.nan], [0.9, 1.0]),
        ([1.0, math.inf], [0.9, 1.0]),
        ([1.0, 2.0], [0.9, math.nan]),
        ([1.0, 2.0], [0.9, math.inf]),
        ([1.0, 2.0], [0.9, 0.0]),
        ([1.0, 2.0], [0.9, -1.0]),
    ),
)
def test_bulk_rejects_unpaired_or_nonfinite_data(
    tmp_path: Path, pressure: list[float], density: list[float]
) -> None:
    _write_bulk_samples(tmp_path, pressure, density)
    with pytest.raises(AnalysisError, match="paired finite pressures"):
        bulk_modulus(tmp_path, "08_bulk")


def test_a_one_way_bulk_ladder_can_resolve(tmp_path: Path) -> None:
    write_bulk(tmp_path, pressures_bar=[1.0, 11.0, 21.0, 31.0])
    fit = bulk_modulus(tmp_path)
    assert fit.resolved
    assert math.isnan(fit.hysteresis)


def test_a_shear_ladder_gives_back_its_modulus(tmp_path: Path) -> None:
    """The same planting, on the off-diagonal."""
    write_shear(tmp_path, modulus_mpa=700.0)
    fit = shear_modulus(tmp_path)
    assert fit.modulus_mpa == pytest.approx(700.0)
    assert fit.resolved


@pytest.mark.parametrize("version", (None, 0.0, 2.0))
def test_shear_refuses_obsolete_or_unidentified_stress(
    tmp_path: Path, version: float | None
) -> None:
    path = write_shear(tmp_path)
    record = json.loads(path.read_text())
    samples = record["stages"]["09_shear"]["samples"]
    if version is None:
        samples.pop("stress_estimator_version")
    else:
        samples["stress_estimator_version"] = [version]
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="estimator version"):
        shear_modulus(tmp_path)


def test_shear_refuses_mixed_legacy_and_current_chunks(tmp_path: Path) -> None:
    write_shear(tmp_path, stage="09_shear_00")
    path = write_shear(tmp_path, stage="09_shear_01")
    record = json.loads(path.read_text())
    record["stages"]["09_shear_01"]["samples"].pop("stress_estimator_version")
    path.write_text(json.dumps(record))
    with pytest.raises(AnalysisError, match="estimator version"):
        shear_modulus(tmp_path)


def test_a_constant_stress_curve_measures_strain_against_its_own_zero(
    tmp_path: Path,
) -> None:
    """The zero-stress rung is the origin, in the same ensemble as the rest."""
    stages = {
        "07_load": {
            "samples": {
                "segment_applied_stress_bar": [0.0, 100.0, 200.0],
                "segment_box_x_nm": [5.0, 5.0, 5.0],
                "segment_box_y_nm": [5.0, 5.0, 5.0],
                "segment_box_z_nm": [5.0, 5.025, 5.05],
                "load_axis": [2.0],
            },
            "mean_temperature_k": 298.15,
        }
    }
    write_bulk(tmp_path, stage="08_bulk", merge=stages)
    curve = load_curve(tmp_path)
    assert curve.controlled == "stress"
    assert curve.strain == pytest.approx([0.0, 0.005, 0.01])
    # 100 bar = 10 MPa over 0.005 strain = 2000 MPa.
    assert youngs_modulus(curve, strain_limit=0.05, min_points=3).modulus_mpa == (
        pytest.approx(2000.0)
    )


# --------------------------------------------------------------------------
# What the fits refuse
# --------------------------------------------------------------------------


def test_a_curve_that_bends_inside_the_window_is_not_resolved(
    tmp_path: Path,
) -> None:
    """The check that catches the real mistake.

    A line fitted across a knee has a small residual - it sits above the data
    at both ends and below it in the middle - so residual alone does not see
    this. Comparing the two halves of the window does.
    """
    write_deformation(tmp_path, n_steps=12)
    manifest = tmp_path / "manifest.json"
    record = json.loads(manifest.read_text())
    samples = record["stages"]["06_deform_r0_00"]["samples"]
    # The recorded strains compound, so the stress is built from those rather
    # than from a linear ladder: stiff at first, then yielding, which is a
    # stress-strain curve fitted past its knee.
    knee = 0.012
    samples["segment_stress_zz_bar"] = [
        (2000.0 * value if value <= knee else 2000.0 * knee + 100.0 * (value - knee))
        / 0.1
        for value in samples["segment_strain"]
    ]
    manifest.write_text(json.dumps(record))

    inside = youngs_modulus(stress_strain(tmp_path), strain_limit=0.012)
    across = youngs_modulus(stress_strain(tmp_path), strain_limit=0.025)
    assert inside.resolved
    assert inside.modulus_mpa == pytest.approx(2000.0)
    assert not across.resolved
    assert across.half_disagreement > inside.half_disagreement


def test_too_few_points_is_not_resolved(tmp_path: Path) -> None:
    """A slope through three points is a slope, not a measurement."""
    write_deformation(tmp_path, n_steps=3)
    fit = youngs_modulus(stress_strain(tmp_path), strain_limit=0.05, min_points=5)
    assert not fit.resolved
    assert fit.n_points == 3


@pytest.mark.parametrize("fit", [youngs_modulus, poisson_ratio])
def test_a_window_that_is_not_a_strain_is_refused(
    tmp_path: Path, fit: Callable[..., object]
) -> None:
    """Both fits read the same window, so both refuse the same nonsense."""
    write_deformation(tmp_path)
    with pytest.raises(ValueError, match="strain_limit"):
        fit(stress_strain(tmp_path), strain_limit=0.0)


def test_an_unphysical_poissons_ratio_is_not_resolved(tmp_path: Path) -> None:
    """An isotropic solid cannot have one, so the cell did not relax."""
    write_deformation(tmp_path, poisson=-0.2)
    ratio = poisson_ratio(stress_strain(tmp_path), strain_limit=0.05)
    assert ratio.ratio == pytest.approx(-0.2)
    assert not ratio.resolved


def test_a_negative_modulus_is_not_resolved(tmp_path: Path) -> None:
    """Whatever the fit quality says about it."""
    write_deformation(tmp_path, modulus_mpa=-500.0)
    assert not youngs_modulus(stress_strain(tmp_path), strain_limit=0.05).resolved


def test_a_ladder_that_does_not_come_back_is_not_resolved(tmp_path: Path) -> None:
    """Hysteresis means the ladder deformed the cell rather than probing it."""
    _write_bulk_samples(
        tmp_path,
        [1.0, 100.0, 200.0, 100.0, 1.0],
        [0.90, 0.906, 0.912, 0.930, 0.950],
    )
    fit = bulk_modulus(tmp_path)
    assert fit.hysteresis > 0.25
    assert not fit.resolved


def test_a_liquid_has_no_shear_modulus_and_says_so(tmp_path: Path) -> None:
    """Which is the right answer for a melt, not a failure to measure one."""
    write_shear(tmp_path, modulus_mpa=0.0)
    fit = shear_modulus(tmp_path)
    assert not fit.resolved


# --------------------------------------------------------------------------
# Finding the stages, and the check over all four
# --------------------------------------------------------------------------


def test_chunks_of_one_ladder_read_back_as_one_curve(tmp_path: Path) -> None:
    """A ladder split for resume is one deformation, not several."""
    write_deformation(tmp_path, n_steps=5, stage="06_deform_r0_00")
    manifest = tmp_path / "manifest.json"
    record = json.loads(manifest.read_text())
    first = record["stages"]["06_deform_r0_00"]["samples"]
    second = {key: list(values) for key, values in first.items()}
    second["segment_strain"] = [value + 0.02 for value in first["segment_strain"]]
    record["stages"]["06_deform_r0_01"] = {
        "samples": second,
        "mean_temperature_k": 298.15,
    }
    manifest.write_text(json.dumps(record))

    assert deform_stages(tmp_path) == ("06_deform_r0_00", "06_deform_r0_01")
    assert stress_strain(tmp_path).n_points == 10


def test_a_directory_with_nothing_in_it_says_so(tmp_path: Path) -> None:
    """Rather than returning an empty tuple nobody checks."""
    with pytest.raises(AnalysisError, match="No manifest"):
        deform_stages(tmp_path)
    write_bulk(tmp_path)
    with pytest.raises(AnalysisError, match="shear"):
        shear_stages(tmp_path)


def test_the_strains_come_back_in_ascending_order(tmp_path: Path) -> None:
    """A fit through points in manifest order is a fit through a scribble."""
    write_deformation(tmp_path, n_steps=8)
    curve = stress_strain(tmp_path)
    assert np.all(np.diff(curve.strain) > 0.0)


def test_four_constants_that_describe_one_solid_are_consistent(
    tmp_path: Path,
) -> None:
    """K = E/3(1-2nu) and G = E/2(1+nu), planted and then checked."""
    implied_bulk = E_PLANTED / (3.0 * (1.0 - 2.0 * NU_PLANTED))
    implied_shear = E_PLANTED / (2.0 * (1.0 + NU_PLANTED))
    write_deformation(tmp_path, modulus_mpa=E_PLANTED, poisson=NU_PLANTED)
    write_bulk(tmp_path, modulus_mpa=implied_bulk)
    write_shear(tmp_path, modulus_mpa=implied_shear)

    curve = stress_strain(tmp_path)
    check = elastic_consistency(
        youngs_modulus(curve, strain_limit=0.05),
        poisson_ratio(curve, strain_limit=0.05),
        bulk=bulk_modulus(tmp_path),
        shear=shear_modulus(tmp_path),
    )
    assert check.bulk_implied_mpa == pytest.approx(implied_bulk)
    assert check.shear_implied_mpa == pytest.approx(implied_shear)
    assert check.bulk_gap == pytest.approx(0.0, abs=1e-9)
    assert check.shear_gap == pytest.approx(0.0, abs=1e-9)
    assert check.consistent


def test_constants_that_do_not_fit_together_are_not_consistent(
    tmp_path: Path,
) -> None:
    """Over-determining the pair is the point: a mismatch has to show."""
    write_deformation(tmp_path, modulus_mpa=E_PLANTED, poisson=NU_PLANTED)
    write_bulk(tmp_path, modulus_mpa=200.0)
    curve = stress_strain(tmp_path)
    check = elastic_consistency(
        youngs_modulus(curve, strain_limit=0.05),
        poisson_ratio(curve, strain_limit=0.05),
        bulk=bulk_modulus(tmp_path),
    )
    assert check.bulk_gap > MAX_CONSISTENCY_GAP
    assert not check.consistent


def test_a_check_with_nothing_to_compare_is_not_a_check_that_passed(
    tmp_path: Path,
) -> None:
    """False, not True. Nothing was checked."""
    write_deformation(tmp_path)
    curve = stress_strain(tmp_path)
    check = elastic_consistency(
        youngs_modulus(curve, strain_limit=0.05),
        poisson_ratio(curve, strain_limit=0.05),
    )
    assert math.isnan(check.bulk_gap)
    assert math.isnan(check.shear_gap)
    assert not check.consistent
