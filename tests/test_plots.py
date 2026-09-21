"""Tests for the figures, asserting structure and data rather than pixels."""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.conformation import (
    centre_of_mass_msd,
    chain_conformation,
    end_to_end_relaxation,
)
from openmmpolymer.correlations import radial_distribution, structure_factor
from openmmpolymer.plots import (
    plot_conformation,
    plot_cooling_rate,
    plot_correlations,
    plot_dynamics,
    plot_quench_curve,
    plot_state_data,
)
from openmmpolymer.timeseries import (
    cooling_rate_extrapolation,
    equilibration,
    glass_transition,
    quench_curve,
    read_state_data,
)

from .helpers import (
    _lattice,
    random_walk_frames,
    state_data_csv,
    synthetic_ensemble,
    transition_at,
    write_quench,
)


@pytest.fixture
def state_data(tmp_path: Path) -> Any:
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
    path = tmp_path / "05_npt.csv"
    path.write_text(state_data_csv(rows))
    return read_state_data(path, stage="05_npt")


@pytest.fixture
def curve(tmp_path: Path) -> Any:
    """A quench curve with a clean break at 350 K."""
    temperature = np.linspace(200.0, 600.0, 21)
    volume = np.where(
        temperature <= 350.0,
        1.0 + 2.0e-4 * (temperature - 350.0),
        1.0 + 8.0e-4 * (temperature - 350.0),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "melt-quench",
                "seed": 1,
                "versions": {},
                "system": {},
                "stages": {
                    "06_quench": {
                        "name": "06_quench",
                        "csv": None,
                        "samples": {
                            "segment_temperature_k": list(temperature[::-1]),
                            "segment_density_g_cm3": list(1.0 / volume[::-1]),
                        },
                    }
                },
                "chains": None,
            }
        )
    )
    return quench_curve(tmp_path)


@pytest.fixture
def walk() -> Any:
    """A diffusing cell of dimers, long enough for a slope."""
    frames = random_walk_frames(400, 60, 0.5, 0.5, seed=13)
    return synthetic_ensemble(frames, n_chains=60, interval_ps=0.5, box_nm=100.0)


def test_building_a_figure_never_imports_pyplot(state_data: Any) -> None:
    """pyplot picks a backend at import, which in a headless job means guessing,
    and registers every figure in a global table that then leaks."""
    sys.modules.pop("matplotlib.pyplot", None)
    plot_state_data(state_data)
    assert "matplotlib.pyplot" not in sys.modules


def test_the_state_data_figure_has_one_panel_per_quantity(state_data: Any) -> None:
    """Temperature, density, potential energy and box volume."""
    figure = plot_state_data(state_data)
    assert len(figure.axes) == 4
    assert figure.axes[-1].get_xlabel() == "Time (ps)"


def test_the_state_data_figure_plots_the_numbers_it_was_given(
    state_data: Any,
) -> None:
    """The failure this catches is a helper plotting the wrong column."""
    figure = plot_state_data(state_data)
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


def test_the_quench_figure_draws_both_fitted_branches(curve: Any) -> None:
    """Reading a break off a curve means seeing the two lines it breaks between."""
    figure = plot_quench_curve(curve, transition=glass_transition(curve))
    axis = figure.axes[0]
    labels = axis.get_legend_handles_labels()[1]
    assert "glass fit" in labels
    assert "melt fit" in labels
    assert any("Tg" in label for label in labels)


def test_the_fitted_branches_pass_through_the_transition(curve: Any) -> None:
    """Drawn from the crossing volume, so both lines have to meet there."""
    transition = glass_transition(curve)
    figure = plot_quench_curve(curve, transition=transition)
    lines = figure.axes[0].get_lines()
    for line in lines[1:3]:
        x, y = line.get_xydata().T
        at_crossing = np.interp(transition.temperature_k, x, y)
        assert at_crossing == pytest.approx(transition.specific_volume_cm3_g, rel=1e-6)


def test_the_conformation_figure_labels_its_axes_with_units(walk: Any) -> None:
    """Every public signature in this package carries the unit in its name, and
    an axis without one is the same bug in a different place."""
    series = chain_conformation(walk, (0, 1))
    figure = plot_conformation(series)
    assert figure.axes[0].get_ylabel() == "<R^2> (nm2)"
    assert figure.axes[1].get_ylabel() == "Rg (nm)"
    assert figure.axes[1].get_xlabel() == "Time (ps)"


def test_the_correlations_figure_marks_the_ideal_gas_line() -> None:
    """A g(r) is read against one, so the eye needs it drawn."""
    ensemble = synthetic_ensemble(_lattice(64, 2.4), n_chains=32, box_nm=2.4)
    figure = plot_correlations(
        radial_distribution(ensemble, n_bins=60, heavy_atoms_only=False)
    )
    assert len(figure.axes) == 1
    assert figure.axes[0].get_xlabel() == "r (nm)"
    assert len(figure.axes[0].get_lines()) == 2


def test_the_structure_factor_gets_its_own_panel_and_its_resolution_floor() -> None:
    """The shaded region is where the cell cannot hold a wave at all."""
    ensemble = synthetic_ensemble(_lattice(64, 2.4), n_chains=32, box_nm=2.4)
    figure = plot_correlations(
        radial_distribution(ensemble, n_bins=60, heavy_atoms_only=False),
        structure=structure_factor(
            ensemble, q_max_per_nm=20.0, n_bins=40, heavy_atoms_only=False
        ),
    )
    assert len(figure.axes) == 2
    assert figure.axes[1].get_xlabel() == "q (1/nm)"
    assert "cannot resolve" in figure.axes[1].get_title()


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


def test_a_measurement_with_no_diffusion_coefficient_says_so_in_its_title(
    walk: Any,
) -> None:
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


def test_the_relaxation_panel_reports_a_run_that_did_not_decorrelate(
    walk: Any,
) -> None:
    """Which is every protocol this package ships, so it is the common case."""
    frames = np.zeros((50, 4, 3), dtype=np.float64)
    frames[:, 1, 2] = 0.6
    frames[:, 3, 2] = 0.6
    frames[:, 2:, 0] = 1.0
    static = synthetic_ensemble(frames, n_chains=2, interval_ps=1.0)
    figure = plot_dynamics(
        centre_of_mass_msd(walk, remove_box_scaling=False),
        relaxation=end_to_end_relaxation(static, (0, 1)),
    )
    assert len(figure.axes) == 2
    assert "not decorrelated" in figure.axes[1].get_title()
    assert figure.axes[1].get_ylabel() == "End-to-end correlation"


def test_a_figure_renders_without_a_backend_or_a_display(state_data: Any) -> None:
    """The one test that actually rasterises: proves a bare Figure is drawable
    in a headless job, with no MPLBACKEND set and nothing written to disk."""
    buffer = io.BytesIO()
    plot_state_data(state_data).savefig(buffer, format="png")
    assert buffer.getbuffer().nbytes > 1000


def test_an_unresolved_transition_is_said_so_on_the_figure(tmp_path: Path) -> None:
    """A break fitted to a straight line draws perfectly well, and a reader
    looking at the picture should be told it means nothing."""
    temperature = np.linspace(200.0, 600.0, 21)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "melt-quench",
                "seed": 1,
                "versions": {},
                "system": {},
                "stages": {
                    "06_quench": {
                        "name": "06_quench",
                        "csv": None,
                        "samples": {
                            "segment_temperature_k": list(temperature),
                            "segment_density_g_cm3": list(
                                1.0 / (1.0 + 5.0e-4 * temperature)
                            ),
                        },
                    }
                },
                "chains": None,
            }
        )
    )
    straight = quench_curve(tmp_path)
    figure = plot_quench_curve(straight, transition=glass_transition(straight))
    assert "no clear transition" in figure.axes[0].get_title()


def test_a_snapshot_conformation_figure_says_it_is_a_snapshot() -> None:
    """There is no settling to draw, and a bare title would suggest there was
    a series behind the single point."""
    snapshot = synthetic_ensemble(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6]] * 4), n_chains=4
    )
    figure = plot_conformation(chain_conformation(snapshot, (0, 1)))
    assert "single snapshot" in figure.axes[0].get_title()


def test_a_relaxation_that_did_decorrelate_reports_its_time(walk: Any) -> None:
    """The good case, which is the one a reader is hoping for."""
    rate = 0.05
    angles = np.arange(200, dtype=np.float64) * rate
    frames = np.zeros((200, 2, 3), dtype=np.float64)
    frames[:, 1, 0] = np.cos(angles) * 0.6
    frames[:, 1, 1] = np.sin(angles) * 0.6
    turning = synthetic_ensemble(frames, n_chains=1, interval_ps=0.5)
    figure = plot_dynamics(
        centre_of_mass_msd(walk, remove_box_scaling=False),
        relaxation=end_to_end_relaxation(turning, (0, 1)),
    )
    assert "relaxes in" in figure.axes[1].get_title()


def test_a_conformation_with_no_measured_ratio_draws_no_reference_line() -> None:
    """The expected mean square is scaled from the measured one, so a measured
    characteristic ratio of zero leaves nothing to scale and no line to draw.

    Built by hand rather than measured: chain_dimensions divides by the mean
    bond length, so a cell degenerate enough to give a ratio of zero never
    gets as far as returning one.
    """
    from openmmpolymer.conformation import ConformationSeries
    from openmmpolymer.plots import _expected_square
    from openmmpolymer.protocols import ChainDimensions

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


def test_the_quench_title_reports_both_expansion_coefficients(curve: Any) -> None:
    """On a second line, not in the legend: the legend names the two branches.

    A dilatometry paper quotes aV, and this is where it becomes visible
    without going back to the dataclass.
    """
    figure = plot_quench_curve(curve, transition=glass_transition(curve))
    title = figure.axes[0].get_title()

    assert "aV" in title
    assert "melt" in title and "glass" in title
    assert "\n" in title


def test_an_unresolved_transition_gets_no_coefficients_on_the_figure(
    tmp_path: Path,
) -> None:
    """A corner fitted into noise has no expansivities worth quoting."""
    temperature = np.linspace(200.0, 600.0, 21)
    straight = 1.0 / (1.0 + 5.0e-4 * temperature)
    write_quench(tmp_path, temperature, straight)
    unresolved = quench_curve(tmp_path)
    figure = plot_quench_curve(unresolved, transition=glass_transition(unresolved))

    assert "aV" not in figure.axes[0].get_title()
    assert "no clear transition" in figure.axes[0].get_title()


def rate_fit(form: str = "log_linear") -> Any:
    """A rate extrapolation over three exactly log-linear measurements."""
    fits = [
        transition_at(rate, 340.0 + 20.0 * math.log10(rate))
        for rate in (1.0, 10.0, 100.0)
    ]
    return cooling_rate_extrapolation(fits, form=form)


def test_the_cooling_rate_figure_plots_one_point_per_measured_rate() -> None:
    """Three quenches, three markers, on a log axis because rates span decades."""
    figure = plot_cooling_rate(rate_fit())
    axis = figure.axes[0]
    measured = next(line for line in axis.get_lines() if line.get_label() == "measured")

    assert axis.get_xscale() == "log"
    assert len(measured.get_xdata()) == 3


def test_the_cooling_rate_figure_shades_the_decades_it_reached_across() -> None:
    """The gap the number was carried over is something the eye can see.

    The same device the structure factor uses for the region a cell cannot
    resolve: if the data does not reach there, the figure says so.
    """
    figure = plot_cooling_rate(rate_fit())
    assert figure.axes[0].patches


def test_an_unresolved_extrapolation_says_so_on_the_figure() -> None:
    """Ten decades to an experimental rate, so this is the usual case."""
    title = plot_cooling_rate(rate_fit()).axes[0].get_title()

    assert "not resolved" in title
    assert "decades" in title


def test_the_cooling_rate_figure_draws_the_curve_a_vft_fit_predicts() -> None:
    """A different relation has to draw a different line, not the same one."""
    straight = plot_cooling_rate(rate_fit("log_linear"))
    curved = plot_cooling_rate(rate_fit("vft"))

    def fitted(figure: Any) -> Any:
        return next(
            line
            for line in figure.axes[0].get_lines()
            if "fit" in str(line.get_label())
        )

    assert not np.allclose(fitted(straight).get_ydata(), fitted(curved).get_ydata())
