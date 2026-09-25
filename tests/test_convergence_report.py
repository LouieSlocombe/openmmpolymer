"""Window reports read off saved runs, and written with every qualification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openmmpolymer.convergence import (
    relaxation_window_convergence,
    time_window_convergence,
)
from openmmpolymer.convergence_report import (
    ConvergenceReport,
    analyse_convergence,
    write_convergence_report,
)
from openmmpolymer.plots import plot_structural_convergence
from openmmpolymer.structural_convergence import structural_window_convergence
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    STRUCTURAL_OPTIONS,
    frozen_rods,
    state_data_csv,
    write_manifest,
    write_polymer_snapshot,
    write_relaxation,
)
from .helpers import planted_relaxation as planted
from .helpers import stationary_trace as stationary

# --------------------------------------------------------------------------
# Reading a saved run
# --------------------------------------------------------------------------


def test_saved_state_csv_produces_window_reports_without_coordinates(
    tmp_path: Path,
) -> None:
    data = stationary()
    csv = tmp_path / "hold.csv"
    rows = [
        [
            float(i),
            float(i),
            -1000 + value,
            500 + value,
            -500 + 2 * value,
            300 + value / 10,
            100.0,
            1 + value / 1000,
        ]
        for i, value in enumerate(data)
    ]
    csv.write_text(state_data_csv(rows))
    write_manifest(tmp_path, {"hold": {"name": "hold", "csv": str(csv), "samples": {}}})
    result = analyse_convergence(tmp_path)
    assert set(result.results) == {
        "density_g_cm3",
        "temperature_k",
        "potential_energy_kj_mol",
    }
    assert all(item.resolved for item in result.results.values())
    assert result.relaxation is None
    assert any("Structural convergence unavailable" in note for note in result.notes)


def test_saved_snapshot_cannot_report_structural_window_resolution(
    tmp_path: Path,
) -> None:
    write_polymer_snapshot(tmp_path)
    result = analyse_convergence(tmp_path, backbone=(0, 1, 2, 3, 4))
    assert not result.results["mean_radius_of_gyration_nm"].resolved
    assert not result.results["mean_squared_end_to_end_nm2"].resolved
    assert result.structural is not None
    assert all(not metric.resolved for metric in result.structural.parameters.values())
    assert any("snapshot" in note for note in result.notes)


def test_saved_relaxation_bins_are_analysed_separately(tmp_path: Path) -> None:
    write_relaxation(tmp_path)
    result = analyse_convergence(tmp_path)
    assert result.relaxation is not None
    assert len(result.relaxation.windows) == 4
    assert result.results == {}


def test_relaxation_replicas_subtract_their_own_baselines_before_averaging(
    tmp_path: Path,
) -> None:
    manifest = write_relaxation(
        tmp_path,
        stem="06_relax_r0",
        mode="shear",
        beta=1.0,
        tau_ps=50.0,
        baseline_bar=0.0,
    )
    stages = json.loads(manifest.read_text())["stages"]
    write_relaxation(
        tmp_path,
        stem="06_relax_r1",
        mode="shear",
        beta=1.0,
        tau_ps=50.0,
        baseline_bar=60.0,
        merge=stages,
    )
    result = analyse_convergence(tmp_path)
    assert result.relaxation is not None
    assert abs(result.relaxation.windows[-1].prony.equilibrium_mpa) < 1.0
    assert all("One relaxation replica" not in note for note in result.relaxation.notes)


def test_missing_manifest_and_bad_explicit_stage_raise(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="No manifest"):
        analyse_convergence(tmp_path)
    write_manifest(tmp_path, {"hold": {"name": "hold", "samples": {}}})
    with pytest.raises(AnalysisError, match="has no stage"):
        analyse_convergence(tmp_path, stage="missing")
    with pytest.raises(ValueError, match="stride"):
        analyse_convergence(tmp_path, stride=0)


def test_stage_selection_keeps_requested_state_stage(tmp_path: Path) -> None:
    write_polymer_snapshot(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["stages"]["later"] = {"name": "later", "samples": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert analyse_convergence(tmp_path, stage="later").stage == "later"


# --------------------------------------------------------------------------
# Figures and the written report
# --------------------------------------------------------------------------


def test_strict_json_retains_unknown_error_and_qualifications(tmp_path: Path) -> None:
    result = time_window_convergence(
        [0.0], [10.0], property_name="radius", value_unit="nm"
    )
    report = ConvergenceReport(
        str(tmp_path), "hold", {"radius": result}, None, ("Only a snapshot.",)
    )
    files = write_convergence_report(report, figures=False)
    assert Path(files.json) == tmp_path / "analysis" / "convergence.json"
    raw = Path(files.json).read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    data = json.loads(raw)
    assert set(data) == {
        "run_dir",
        "stage",
        "results",
        "relaxation",
        "notes",
        "structural",
        "interpretation",
    }
    assert data["results"]["radius"]["windows"][-1]["standard_error"] is None
    assert not data["results"]["radius"]["resolved"]
    assert data["relaxation"] is None
    assert data["notes"] == ["Only a snapshot."]
    assert "not independent replicas" in data["interpretation"]
    assert files.figures == ()


def test_report_writes_stationary_and_relaxation_figures(tmp_path: Path) -> None:
    data = stationary()
    result = time_window_convergence(
        np.arange(data.size), data, property_name="density", value_unit="g/cm^3"
    )
    times = np.geomspace(0.1, 10000.0, 140)
    relaxation = relaxation_window_convergence(
        planted(times, 1000.0 * np.exp(-times / 50.0))
    )
    report = ConvergenceReport(
        str(tmp_path), "hold", {"density": result}, relaxation, ()
    )
    files = write_convergence_report(report, tmp_path / "report", figure_format="svg")
    assert [Path(path).name for path in files.figures] == [
        "convergence_density.svg",
        "convergence_relaxation.svg",
    ]
    assert all("<svg" in Path(path).read_text() for path in files.figures)
    payload = json.loads(Path(files.json).read_text())
    assert (
        payload["relaxation"]["metrics"]["kww_viscosity_pa_s"]["value_unit"] == "Pa s"
    )


def test_json_only_report_does_not_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = time_window_convergence(
        [0.0], [10.0], property_name="radius", value_unit="nm"
    )
    times = np.geomspace(0.1, 10000.0, 140)
    relaxation = relaxation_window_convergence(
        planted(times, 1000.0 * np.exp(-times / 50.0))
    )
    structural = structural_window_convergence(
        frozen_rods(), range(5), **STRUCTURAL_OPTIONS
    )
    report = ConvergenceReport(
        str(tmp_path), "hold", {"radius": result}, relaxation, (), structural
    )

    def unexpected_plot(*args: object, **kwargs: object) -> None:
        pytest.fail("A JSON-only report must not construct figures.")

    for name in (
        "plot_window_convergence",
        "plot_relaxation_convergence",
        "plot_structural_convergence",
    ):
        monkeypatch.setattr(f"openmmpolymer.convergence_report.{name}", unexpected_plot)
    files = write_convergence_report(report, figures=False)
    assert Path(files.json).is_file()
    assert files.figures == ()


@pytest.mark.parametrize("figures", [True, False])
def test_unsafe_format_rejected_before_writing(tmp_path: Path, figures: bool) -> None:
    report = ConvergenceReport(str(tmp_path), "hold", {}, None, ())
    with pytest.raises(ValueError, match="figure_format"):
        write_convergence_report(report, figures=figures, figure_format="../svg")
    assert not (tmp_path / "analysis").exists()


def test_structural_report_preserves_missing_metrics_curves_and_unresolved_figures(
    tmp_path: Path,
) -> None:
    structural = structural_window_convergence(
        frozen_rods(), range(5), **STRUCTURAL_OPTIONS
    )
    figure = plot_structural_convergence(structural)
    assert all(
        "not resolved" in axis.get_title() for axis in figure.axes if axis.get_visible()
    )
    report = ConvergenceReport(str(tmp_path), "hold", {}, None, (), structural)
    files = write_convergence_report(report, tmp_path)
    assert len(files.figures) == 1
    assert Path(files.figures[0]).read_bytes().startswith(b"\x89PNG")
    payload = json.loads(Path(files.json).read_text())
    assert (
        payload["structural"]["parameters"]["diffusion_coefficient_cm2_s"]["values"]
        == [None] * 4
    )
    assert "radial_distribution" in payload["structural"]["windows"][0]["curves"]
