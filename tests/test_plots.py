"""Tests for the figures, asserting structure and data rather than pixels.

The mechanical figures are in ``test_plots_mechanical.py``; this module has the
thermal, structural, relaxation and rate ones, and the one test that renders
every figure the package draws.
"""

from __future__ import annotations

import io
import math
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.conformation import (
    ConformationSeries,
    centre_of_mass_msd,
    chain_conformation,
    end_to_end_relaxation,
    persistence_length,
)
from openmmpolymer.convergence import (
    relaxation_window_convergence,
    time_window_convergence,
)
from openmmpolymer.correlations import radial_distribution, structure_factor
from openmmpolymer.elasticity import poisson_ratio, youngs_modulus
from openmmpolymer.mechanical import analyse_mechanics
from openmmpolymer.plots import (
    _expected_square,
    plot_breaking_strength,
    plot_conformation,
    plot_cooling_rate,
    plot_correlations,
    plot_dynamics,
    plot_elongation_at_break,
    plot_moduli,
    plot_persistence,
    plot_quench_curve,
    plot_rate_dependence,
    plot_relaxation,
    plot_relaxation_convergence,
    plot_relaxation_spectrum,
    plot_state_data,
    plot_stress_strain,
    plot_structural_convergence,
    plot_window_convergence,
    plot_yield_strength,
)
from openmmpolymer.protocols import ChainDimensions
from openmmpolymer.rate_dependence import RateExtrapolation
from openmmpolymer.relaxation import RelaxationCurve, fit_kww, fit_prony
from openmmpolymer.strength import (
    breaking_strength,
    elongation_at_break,
    yield_strength,
)
from openmmpolymer.structural_convergence import structural_window_convergence
from openmmpolymer.timeseries import (
    QuenchCurve,
    StateData,
    cooling_rate_extrapolation,
    equilibration,
    glass_transition,
    quench_curve,
    read_state_data,
)

from .helpers import (
    STRUCTURAL_OPTIONS,
    dimer_cell,
    freely_rotating_chain,
    frozen_rods,
    lattice,
    log_linear_transitions,
    nominal_curve,
    planted_rate_report,
    planted_relaxation,
    random_walk_frames,
    rod_positions,
    rotating_dimer,
    state_data_csv,
    stationary_trace,
    synthetic_ensemble,
    two_line_curve,
    write_deformation,
    write_quench,
)


def settling_state_data(directory: Path) -> StateData:
    """A short state-data series with a settling density."""
    rows = [
        [
            index * 100,
            index * 0.2,
            -100.0 - index,
            10.0,
            -90.0,
            300.0 + 0.1 * index,
            13.8,
            0.85 - 0.05 * np.exp(-index / 5.0),
        ]
        for index in range(60)
    ]
    path = directory / "05_npt.csv"
    path.write_text(state_data_csv(rows))
    return read_state_data(path, stage="05_npt")


def quench(directory: Path, *, straight: bool = False) -> QuenchCurve:
    """A quench with a clean break at 350 K, or with none, at an unknown rate."""
    temperature, density = two_line_curve()
    if straight:
        density = 1.0 / (1.0 + 5.0e-4 * temperature)
    write_quench(directory, temperature[::-1], density[::-1], with_csv=False)
    return quench_curve(directory)


def decay(*, modulus_mpa: float = 1000.0, floor: float = 1.0) -> RelaxationCurve:
    """A planted relaxation with a known error on every bin, for the figures."""
    time_ps = np.geomspace(0.1, 1.0e4, 80)
    return planted_relaxation(
        time_ps,
        modulus_mpa * np.exp(-((time_ps / 200.0) ** 0.5)),
        error_mpa=np.full(time_ps.size, 2.0),
        floor=floor,
        mode="tensile",
    )


def diffusing_dimers() -> Any:
    """A diffusing cell of dimers, long enough for a slope."""
    frames = random_walk_frames(400, 60, 0.5, 0.5, seed=13)
    return synthetic_ensemble(frames, n_chains=60, interval_ps=0.5, box_nm=100.0)


def lattice_cell() -> Any:
    """The 64-site argon lattice, read as 32 dimers."""
    return synthetic_ensemble(lattice(64, 2.4), n_chains=32, box_nm=2.4)


@pytest.fixture
def state_data(tmp_path: Path) -> StateData:
    """A short state-data series with a settling density."""
    return settling_state_data(tmp_path)


@pytest.fixture
def curve(tmp_path: Path) -> QuenchCurve:
    """A quench curve with a clean break at 350 K."""
    return quench(tmp_path)


@pytest.fixture
def walk() -> Any:
    """A diffusing cell of dimers, long enough for a slope."""
    return diffusing_dimers()


# --------------------------------------------------------------------------
# Every figure, rendered
# --------------------------------------------------------------------------


def _moduli_figure(directory: Path) -> Any:
    write_deformation(directory)
    return plot_moduli(analyse_mechanics(directory, strain_limit=0.05))


def _rate_fit(
    form: str = "log_linear", *, target: float = 0.01, unknown: bool = False
) -> RateExtrapolation:
    fit: RateExtrapolation | None = getattr(
        planted_rate_report(target=target, unknown=unknown), form
    )
    assert fit is not None
    return fit


FAILED = nominal_curve(
    np.arange(8) * 0.1, [0.0, 20.0, 60.0, 100.0, 70.0, 35.0, 30.0, 20.0]
)
YIELDED = nominal_curve(
    [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
    [2.0, 7.0, 12.0, 17.0, 22.0, 24.0, 25.0, 26.0, 26.0],
)

FIGURES: dict[str, Callable[[Path], Any]] = {
    "state_data": lambda d: plot_state_data(settling_state_data(d)),
    "quench_curve": lambda d: plot_quench_curve(
        quench(d), transition=glass_transition(quench(d))
    ),
    "cooling_rate": lambda d: plot_cooling_rate(
        cooling_rate_extrapolation(log_linear_transitions(), form="vft")
    ),
    "rate_dependence": lambda d: plot_rate_dependence(_rate_fit()),
    "conformation": lambda d: plot_conformation(
        chain_conformation(diffusing_dimers(), (0, 1))
    ),
    "correlations": lambda d: plot_correlations(
        radial_distribution(lattice_cell(), n_bins=60, heavy_atoms_only=False),
        structure=structure_factor(
            lattice_cell(), q_max_per_nm=20.0, n_bins=40, heavy_atoms_only=False
        ),
    ),
    "dynamics": lambda d: plot_dynamics(
        centre_of_mass_msd(diffusing_dimers(), remove_box_scaling=False),
        relaxation=end_to_end_relaxation(
            rotating_dimer(200, 0.05, interval_ps=0.5), (0, 1)
        ),
    ),
    "stress_strain": lambda d: plot_stress_strain(
        FAILED,
        fit=youngs_modulus(FAILED, strain_limit=0.3),
        poisson=poisson_ratio(FAILED, strain_limit=0.3),
    ),
    "breaking_strength": lambda d: plot_breaking_strength(
        FAILED, breaking_strength(FAILED)
    ),
    "elongation_at_break": lambda d: plot_elongation_at_break(
        FAILED, elongation_at_break(FAILED)
    ),
    "yield_strength": lambda d: plot_yield_strength(YIELDED, yield_strength(YIELDED)),
    "moduli": _moduli_figure,
    "relaxation": lambda d: plot_relaxation(
        decay(), kww=fit_kww(decay()), prony=fit_prony(decay()), replicas=[decay()]
    ),
    "relaxation_spectrum": lambda d: plot_relaxation_spectrum(fit_prony(decay())),
    "persistence": lambda d: plot_persistence(
        persistence_length(
            synthetic_ensemble(rod_positions(6, 0.153), n_chains=1), range(6)
        )
    ),
    "window_convergence": lambda d: plot_window_convergence(
        time_window_convergence(
            np.arange(6000), stationary_trace(), property_name="x", value_unit="u"
        )
    ),
    "relaxation_convergence": lambda d: plot_relaxation_convergence(
        relaxation_window_convergence(
            planted_relaxation(
                np.geomspace(0.1, 1.0e4, 140),
                1000.0 * np.exp(-np.geomspace(0.1, 1.0e4, 140) / 50.0),
            )
        )
    ),
    "structural_convergence": lambda d: plot_structural_convergence(
        structural_window_convergence(frozen_rods(), range(5), **STRUCTURAL_OPTIONS)
    ),
}


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_every_figure_renders_headless_without_pyplot(
    name: str, tmp_path: Path
) -> None:
    """pyplot picks a backend at import, which in a headless job means guessing,
    and registers every figure in a global table that then leaks. A bare
    Figure is drawable with no backend set and nothing written to disk."""
    sys.modules.pop("matplotlib.pyplot", None)
    figure = FIGURES[name](tmp_path)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    assert buffer.getvalue().startswith(b"\x89PNG")
    assert "matplotlib.pyplot" not in sys.modules
    assert figure.get_layout_engine() is not None


# --------------------------------------------------------------------------
# The state data and the quench
# --------------------------------------------------------------------------


def test_state_data_panels_plot_the_recorded_quantities(state_data: StateData) -> None:
    figure = plot_state_data(state_data)
    assert len(figure.axes) == 4
    assert figure.axes[-1].get_xlabel() == "Time (ps)"
    plotted = figure.axes[1].get_lines()[0].get_xydata()
    assert plotted[:, 0] == pytest.approx(state_data.time_ps)
    assert plotted[:, 1] == pytest.approx(state_data.density_g_cm3)


def test_the_settling_point_is_drawn_when_it_is_known(state_data: Any) -> None:
    """It is the whole reason to look at a density trace."""
    settled = equilibration(state_data.time_ps, state_data.density_g_cm3)
    figure = plot_state_data(state_data, settled=settled)
    assert "settled" in figure.axes[0].get_title() or "drifting" in (
        figure.axes[0].get_title()
    )
    assert len(figure.axes[0].get_lines()) > 1


def test_the_quench_figure_carries_its_cooling_rate_in_the_title(
    curve: Any,
) -> None:
    """The transition temperature is not comparable with an experiment without
    it, so the caveat has to travel with the image and not just the dataclass."""
    figure = plot_quench_curve(curve)
    assert "cooling rate unknown" in figure.axes[0].get_title()


def test_quench_fit_branches_meet_at_the_reported_transition(
    curve: QuenchCurve,
) -> None:
    transition = glass_transition(curve)
    axis = plot_quench_curve(curve, transition=transition).axes[0]
    labels = axis.get_legend_handles_labels()[1]
    assert "glass fit" in labels
    assert "melt fit" in labels
    assert any("Tg" in label for label in labels)
    for line in axis.get_lines()[1:3]:
        x, y = line.get_xydata().T
        at_crossing = np.interp(transition.temperature_k, x, y)
        assert at_crossing == pytest.approx(transition.specific_volume_cm3_g, rel=1e-6)
    title = axis.get_title()
    assert "aV" in title
    assert "melt" in title and "glass" in title
    assert "\n" in title


def test_an_unresolved_transition_is_said_so_without_coefficients(
    tmp_path: Path,
) -> None:
    """A break fitted to a straight line draws perfectly well, and a reader
    looking at the picture should be told it means nothing - and should not
    be handed expansivities of a corner fitted into noise."""
    straight = quench(tmp_path, straight=True)
    figure = plot_quench_curve(straight, transition=glass_transition(straight))

    assert "aV" not in figure.axes[0].get_title()
    assert "no clear transition" in figure.axes[0].get_title()


def test_cooling_rate_figure_marks_measurements_and_unresolved_extrapolation() -> None:
    fit = cooling_rate_extrapolation(log_linear_transitions())
    axis = plot_cooling_rate(fit).axes[0]
    measured = next(line for line in axis.get_lines() if line.get_label() == "measured")
    assert axis.get_xscale() == "log"
    assert len(measured.get_xdata()) == 3
    assert axis.patches
    assert "not resolved" in axis.get_title()
    assert "decades" in axis.get_title()


def test_the_cooling_rate_figure_draws_the_curve_the_fit_predicts() -> None:
    """A different relation has to draw a different line, and the line drawn
    is the relation the fit reports its number from."""
    figures = {
        form: plot_cooling_rate(
            cooling_rate_extrapolation(log_linear_transitions(), form=form)
        )
        for form in ("log_linear", "vft")
    }

    def fitted(figure: Any) -> Any:
        return next(
            line
            for line in figure.axes[0].get_lines()
            if "fit" in str(line.get_label())
        )

    straight, curved = (fitted(figures[form]) for form in ("log_linear", "vft"))
    assert not np.allclose(straight.get_ydata(), curved.get_ydata())
    fit = cooling_rate_extrapolation(log_linear_transitions())
    np.testing.assert_allclose(
        straight.get_ydata(), fit.predict(np.asarray(straight.get_xdata()))
    )


# --------------------------------------------------------------------------
# Structure and dynamics
# --------------------------------------------------------------------------


def test_the_conformation_figure_labels_its_axes_with_units(walk: Any) -> None:
    """Every public signature in this package carries the unit in its name, and
    an axis without one is the same bug in a different place."""
    series = chain_conformation(walk, (0, 1))
    figure = plot_conformation(series)
    assert figure.axes[0].get_ylabel() == "<R^2> (nm2)"
    assert figure.axes[1].get_ylabel() == "Rg (nm)"
    assert figure.axes[1].get_xlabel() == "Time (ps)"


def test_a_snapshot_conformation_figure_says_it_is_a_snapshot() -> None:
    """There is no settling to draw, and a bare title would suggest there was
    a series behind the single point."""
    snapshot = synthetic_ensemble(dimer_cell(4), n_chains=4)
    figure = plot_conformation(chain_conformation(snapshot, (0, 1)))
    assert "single snapshot" in figure.axes[0].get_title()


def test_a_conformation_with_no_measured_ratio_draws_no_reference_line() -> None:
    """The expected mean square is scaled from the measured one, so a measured
    characteristic ratio of zero leaves nothing to scale and no line to draw.

    Built by hand rather than measured: chain_dimensions divides by the mean
    bond length, so a cell degenerate enough to give a ratio of zero never
    gets as far as returning one.
    """
    series = ConformationSeries(
        stage="synthetic",
        time_ps=np.zeros(1),
        mean_squared_end_to_end_nm2=np.zeros(1),
        mean_radius_of_gyration_nm=np.zeros(1),
        mean=ChainDimensions(
            mean_squared_end_to_end_nm2=0.0,
            mean_radius_of_gyration_nm=0.0,
            ratio_of_squares=0.0,
            characteristic_ratio=0.0,
            expected_characteristic_ratio=7.0,
            consistent=False,
        ),
        settled=None,
        n_chains=1,
        n_frames=1,
    )
    assert _expected_square(series) is None
    assert len(plot_conformation(series).axes[0].get_lines()) == 1


@pytest.mark.parametrize("with_structure", [False, True], ids=["rdf", "rdf-and-sq"])
def test_correlations_panels_show_reference_lines_and_resolution(
    with_structure: bool,
) -> None:
    cell = lattice_cell()
    structure = (
        structure_factor(cell, q_max_per_nm=20.0, n_bins=40, heavy_atoms_only=False)
        if with_structure
        else None
    )
    figure = plot_correlations(
        radial_distribution(cell, n_bins=60, heavy_atoms_only=False),
        structure=structure,
    )
    assert len(figure.axes) == 1 + with_structure
    assert figure.axes[0].get_xlabel() == "r (nm)"
    assert len(figure.axes[0].get_lines()) == 2
    if with_structure:
        assert figure.axes[1].get_xlabel() == "q (1/nm)"
        assert "cannot resolve" in figure.axes[1].get_title()
        assert len(figure.axes[1].patches) == 1


def test_the_dynamics_figure_is_logarithmic_with_a_slope_one_guide(
    walk: Any,
) -> None:
    """A sub-diffusive curve only looks sub-diffusive next to a straight line
    of slope one on log axes."""
    figure = plot_dynamics(centre_of_mass_msd(walk, remove_box_scaling=False))
    axis = figure.axes[0]
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert "slope 1 (diffusive)" in axis.get_legend_handles_labels()[1]
    assert axis.get_ylabel() == "Centre-of-mass MSD (nm2)"


def test_a_measurement_with_no_diffusion_coefficient_says_so_in_its_title() -> None:
    """Leaving the title blank would read as a rendering failure rather than as
    the deliberate refusal it is."""
    ballistic = np.zeros((40, 40, 3), dtype=np.float64)
    generator = np.random.default_rng(2)
    velocities = generator.normal(0.0, 0.05, size=(20, 3))
    for index in range(40):
        ballistic[index, 0::2, :] = velocities * index
        ballistic[index, 1::2, :] = velocities * index + np.array([0.0, 0.0, 0.6])
    measured = centre_of_mass_msd(
        synthetic_ensemble(ballistic, n_chains=20, interval_ps=1.0),
        remove_box_scaling=False,
    )
    assert "no diffusion" in plot_dynamics(measured).axes[0].get_title()


@pytest.mark.parametrize(
    ("radians_per_frame", "phrase"),
    [(0.0, "not decorrelated"), (0.05, "relaxes in")],
)
def test_the_relaxation_panel_says_whether_the_chains_decorrelated(
    walk: Any, radians_per_frame: float, phrase: str
) -> None:
    """Not decorrelating is every protocol this package ships, so it is the
    common case; decorrelating is the one a reader is hoping for."""
    figure = plot_dynamics(
        centre_of_mass_msd(walk, remove_box_scaling=False),
        relaxation=end_to_end_relaxation(
            rotating_dimer(200, radians_per_frame, interval_ps=0.5), (0, 1)
        ),
    )
    assert len(figure.axes) == 2
    assert phrase in figure.axes[1].get_title()
    assert figure.axes[1].get_ylabel() == "End-to-end correlation"


def test_the_persistence_figure_draws_the_threshold_and_the_fit() -> None:
    """The length is read where the correlation crosses 1/e, so both the line
    and the exponential it was fitted with are drawn through the points."""
    positions = freely_rotating_chain(60, 0.153, 0.5, n_chains=300, seed=11)
    measured = persistence_length(
        synthetic_ensemble(positions, n_chains=300), range(61)
    )
    figure = plot_persistence(measured)
    axis = figure.axes[0]
    assert axis.get_xlabel() == "Separation (bonds)"
    assert axis.get_ylabel() == "<cos theta(s)>"
    assert len(axis.get_lines()) == 3
    assert "1/e" in axis.get_legend_handles_labels()[1]
    assert "l_p =" in axis.get_title()
    assert not axis.patches


def test_a_rod_is_shaded_past_the_end_of_the_chain() -> None:
    """Nothing to fit, so no fit line, and the region the chain does not reach
    is what the figure is about."""
    measured = persistence_length(
        synthetic_ensemble(rod_positions(6, 0.153), n_chains=1), range(6)
    )
    figure = plot_persistence(measured)
    axis = figure.axes[0]
    assert len(axis.get_lines()) == 2
    assert len(axis.patches) == 1
    assert "rod-like" in axis.get_title()


def test_an_extrapolated_persistence_length_says_so() -> None:
    """A chain shorter than its own persistence length gives a number, and
    the title and the shading say it was reached for."""
    positions = freely_rotating_chain(12, 0.153, 5.0, n_chains=200, seed=5)
    measured = persistence_length(
        synthetic_ensemble(positions, n_chains=200), range(13)
    )
    assert not measured.decayed
    figure = plot_persistence(measured)
    axis = figure.axes[0]
    assert "extrapolated" in axis.get_title()
    assert len(axis.get_lines()) == 3
    assert len(axis.patches) == 1
    assert axis.get_xlim()[1] > 12.0


# --------------------------------------------------------------------------
# Relaxation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("overlay", ["none", "fits", "replicas"])
def test_relaxation_panels_show_decay_samples_and_optional_fits(
    overlay: str,
) -> None:
    curve = decay(floor=5.0 if overlay == "none" else 1.0)
    replicas = (
        [decay(modulus_mpa=value) for value in (900.0, 1100.0)]
        if overlay == "replicas"
        else []
    )
    figure = plot_relaxation(
        curve,
        kww=fit_kww(curve) if overlay == "fits" else None,
        prony=fit_prony(curve) if overlay == "fits" else None,
        replicas=replicas,
    )
    assert len(figure.axes) == 2
    axis = figure.axes[0]
    assert axis.get_ylabel() == "|G(t)| (MPa)"
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert figure.axes[1].get_ylabel() == "Readings per bin"
    assert "+0.030" in axis.get_title()
    assert "298 K" in axis.get_title()
    labels = " ".join(axis.get_legend_handles_labels()[1])
    assert "below the baseline noise" in labels
    if overlay == "fits":
        assert "KWW" in labels and "beta" in labels
        assert "Prony" in labels and "G_inf" in labels
    for line, replica in zip(axis.get_lines()[: len(replicas)], replicas, strict=True):
        np.testing.assert_allclose(line.get_xdata(), replica.time_ps)
        np.testing.assert_allclose(line.get_ydata(), replica.modulus_mpa)


def test_a_spectrum_figure_marks_what_lies_past_the_end_of_the_run() -> None:
    """A weight on the slowest term is the fit saying the decay outlasted the
    data, and the figure should not let that pass as a measurement."""
    figure = plot_relaxation_spectrum(fit_prony(decay()))
    assert figure.axes[0].get_xlabel() == "Relaxation time (ps)"
    assert figure.axes[0].get_xscale() == "log"
    labels = " ".join(
        text.get_text() for text in figure.axes[0].get_legend().get_texts()
    )
    assert "past the end of the run" in labels


# --------------------------------------------------------------------------
# Rate dependence
# --------------------------------------------------------------------------


def _shaded(axis: Any) -> list[float]:
    """The rates the extrapolated interval covers, in data coordinates."""
    patch = next(
        patch for patch in axis.patches if patch.get_label() == "extrapolated interval"
    )
    vertices = patch.get_path().transformed(patch.get_transform() - axis.transData)
    return [vertices.vertices[:, 0].min(), vertices.vertices[:, 0].max()]


@pytest.mark.parametrize("form", ["log_linear", "power_law"])
def test_the_rate_figure_draws_the_data_the_fit_and_the_target(form: str) -> None:
    """Every number on it is the analysis's own, errors included."""
    fit = _rate_fit(form)
    axis: Any = plot_rate_dependence(fit).axes[0]
    line = next(line for line in axis.get_lines() if line.get_label() == f"{form} fit")
    np.testing.assert_allclose(line.get_ydata(), fit.predict(line.get_xdata()))
    assert line.get_xdata()[0] == pytest.approx(0.01)
    assert line.get_xdata()[-1] == pytest.approx(10.0)
    measured, target = axis.containers
    np.testing.assert_allclose(measured.lines[0].get_xdata(), fit.rates)
    np.testing.assert_allclose(measured.lines[0].get_ydata(), fit.values)
    np.testing.assert_allclose(target.lines[0].get_xdata(orig=False), [0.01])
    np.testing.assert_allclose(target.lines[0].get_ydata(orig=False), [fit.value])
    for container, values, errors in (
        (measured, fit.values, fit.standard_errors),
        (target, [fit.value], [fit.standard_error]),
    ):
        segments = container.lines[2][0].get_segments()
        for segment, value, error in zip(segments, values, errors, strict=True):
            np.testing.assert_allclose(segment[:, 1], [value - error, value + error])
    assert axis.get_xscale() == "log"
    assert axis.get_xlabel() == "Rate (strain/ns)"
    assert axis.get_ylabel() == "Yield strength (MPa)"
    assert f"{form}\n1.0 decades extrapolated, " in axis.get_title()


@pytest.mark.parametrize(
    "target, expected", [(0.01, (0.01, 0.1)), (100.0, (10.0, 100.0))]
)
def test_the_rate_figure_shades_the_unmeasured_interval_on_either_side(
    target: float, expected: tuple[float, float]
) -> None:
    axis: Any = plot_rate_dependence(_rate_fit(target=target)).axes[0]
    assert _shaded(axis) == pytest.approx(expected)


def test_an_interpolated_rate_has_no_extrapolated_interval() -> None:
    axis: Any = plot_rate_dependence(_rate_fit(target=0.5)).axes[0]
    assert not axis.patches
    assert "0.0 decades extrapolated" in axis.get_title()


def test_unknown_errors_and_a_distant_target_are_visibly_unresolved() -> None:
    """A clean line cannot hide an unsupported distance or a missing error."""
    axis: Any = plot_rate_dependence(_rate_fit(target=1e-8, unknown=True)).axes[0]
    assert "7.0 decades extrapolated, not resolved" in axis.get_title()
    labels = axis.get_legend_handles_labels()[1]
    assert sum("SE unavailable" in label for label in labels) == 2
    assert not any(container.has_yerr for container in axis.containers)


def test_an_unknown_target_error_is_marked_unavailable() -> None:
    fit = replace(_rate_fit(), standard_error=math.inf, resolved=False)
    axis: Any = plot_rate_dependence(fit).axes[0]
    assert "SE unavailable" in axis.containers[1].get_label()
    assert not axis.containers[1].has_yerr


def test_a_nonfinite_target_keeps_its_rate_but_draws_no_value() -> None:
    fit = replace(_rate_fit(), value=math.inf, standard_error=math.inf, resolved=False)
    axis: Any = plot_rate_dependence(fit).axes[0]
    assert len(axis.containers) == 1
    assert any(
        "target estimate not finite" in label
        for label in axis.get_legend_handles_labels()[1]
    )


def test_prefix_and_disjoint_block_figure_retains_errors_and_verdict() -> None:
    data = stationary_trace()
    result = time_window_convergence(
        np.arange(data.size), data, property_name="density", value_unit="g/cm^3"
    )
    figure = plot_window_convergence(result)
    prefix: Any = figure.axes[0]
    blocks: Any = figure.axes[1]
    np.testing.assert_allclose(
        prefix.containers[0].lines[0].get_xdata(),
        [item.duration_ps for item in result.windows],
    )
    np.testing.assert_allclose(
        prefix.containers[0].lines[0].get_ydata(),
        [item.mean for item in result.windows],
    )
    np.testing.assert_allclose(
        blocks.containers[0].lines[0].get_ydata(), result.block_means
    )
    segments = prefix.containers[0].lines[2][0].get_segments()
    for segment, window in zip(segments, result.windows, strict=True):
        np.testing.assert_allclose(
            segment[:, 1],
            [window.mean - window.standard_error, window.mean + window.standard_error],
        )
    assert "overlapping" in prefix.get_title()
    assert "disjoint" in blocks.get_title()


def test_constant_snapshot_plot_labels_unknown_uncertainty() -> None:
    result = time_window_convergence(
        [0.0], [10.0], property_name="radius", value_unit="nm"
    )
    figure = plot_window_convergence(result)
    assert "SE unavailable" in figure.axes[0].get_legend_handles_labels()[1]
    assert any("not resolved" in item.get_text() for item in figure.texts)


def test_relaxation_parameter_plot_keeps_unobserved_tails_unresolved() -> None:
    times = np.geomspace(0.1, 10000.0, 140)
    result = relaxation_window_convergence(
        planted_relaxation(times, 1000.0 * np.exp(-times / 10000.0))
    )
    figure = plot_relaxation_convergence(result)
    assert len(figure.axes) == 4
    assert all("not resolved" in axis.get_title() for axis in figure.axes)
