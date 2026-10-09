"""Shared report boundaries remain stable while their writers consolidate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer import __version__, plot_melting
from openmmpolymer._files import json_value
from openmmpolymer.conformation import (
    ConformationSeries,
    EndToEndRelaxation,
    MeanSquaredDisplacement,
    PersistenceLength,
)
from openmmpolymer.convergence_report import ConvergenceReport, write_convergence_report
from openmmpolymer.correlations import RadialDistribution, StructureFactor
from openmmpolymer.elasticity import youngs_modulus
from openmmpolymer.mechanical import ModulusReport, write_mechanical_report
from openmmpolymer.protocols import ChainDimensions, RunManifest
from openmmpolymer.rate_reports import write_rate_report
from openmmpolymer.structure import StructureReport, write_structure_report
from openmmpolymer.tensile import (
    analyse_breaking,
    analyse_elongation,
    analyse_yield,
    write_breaking_report,
    write_elongation_report,
    write_yield_report,
)
from openmmpolymer.tg import TgReport, write_tg_report
from openmmpolymer.timeseries import Equilibration, QuenchCurve, glass_transition
from openmmpolymer.tm import MeltingReport, melting_temperature, write_melting_report
from openmmpolymer.viscoelastic import RelaxationReport, write_relaxation_report

from .helpers import (
    PLANTED_TENSILE,
    nominal_curve,
    planted_curve,
    planted_rate_report,
    snapshot_files,
    two_line_curve,
    write_tensile_scan,
)

WRITERS: dict[str, Callable[..., Any]] = {
    "tm": write_melting_report,
    "breaking": write_breaking_report,
    "elongation": write_elongation_report,
    "yield": write_yield_report,
    "tg": write_tg_report,
    "mechanics": write_mechanical_report,
    "relaxation": write_relaxation_report,
    "structure": write_structure_report,
    "convergence": write_convergence_report,
    "rates": write_rate_report,
}


def _report(name: str, directory: Path) -> Any:
    """Real report records, including curves for the writers that bypass helpers."""
    directory.mkdir()
    if name in PLANTED_TENSILE:
        write_tensile_scan(directory, PLANTED_TENSILE[name])
        return {
            "breaking": analyse_breaking,
            "elongation": analyse_elongation,
            "yield": analyse_yield,
        }[name](directory)
    if name == "tm":
        curve = planted_curve()
        return MeltingReport(str(directory), curve, melting_temperature(curve), ())
    if name == "rates":
        return planted_rate_report()
    if name == "convergence":
        return ConvergenceReport(str(directory), "hold", {}, None, ())
    if name == "tg":
        return TgReport(
            run_dir=str(directory),
            stages=(),
            curves=(),
            transitions=(),
            coarse=None,
            fine=None,
            log_linear=None,
            vft=None,
            melt=None,
            temperature_k=None,
            cooling_rate_k_per_ns=None,
            resolved=False,
            notes=(),
        )
    if name == "mechanics":
        return ModulusReport(
            run_dir=str(directory),
            curves=(),
            replicas=(),
            youngs=None,
            poisson=None,
            bulk=None,
            shear=None,
            load=None,
            load_modulus=None,
            consistency=None,
            replica_spread_mpa=None,
            method_gap=float("nan"),
            resolved=False,
            notes=(),
        )
    if name == "relaxation":
        return RelaxationReport(
            run_dir=str(directory),
            curves=(),
            mean=None,
            kww=None,
            prony=None,
            replica_spread_mpa=None,
            linearity=None,
            plateau_conflict=False,
            baseline_fraction=0.0,
            resolved=False,
            notes=(),
        )
    if name == "structure":
        return StructureReport(
            run_dir=str(directory),
            stage="hold",
            stage_source="last_snapshot",
            is_snapshot=True,
            n_frames=1,
            interval_ps=0.0,
            n_chains=1,
            atoms_per_chain=2,
            stride=1,
            backbone=None,
            backbone_source=None,
            backbone_file=None,
            distribution=None,
            structure=None,
            conformation=None,
            persistence=None,
            displacement=None,
            relaxation=None,
            recorded_chains=None,
            notes=(),
        )
    raise AssertionError(f"Unknown report: {name}")


@pytest.mark.parametrize("figures", [True, False])
@pytest.mark.parametrize("name", WRITERS)
def test_bad_figure_format_writes_nothing(
    tmp_path: Path, name: str, figures: bool
) -> None:
    report = _report(name, tmp_path / "run")
    output = tmp_path / "report"
    before = snapshot_files(tmp_path)
    with pytest.raises(ValueError):
        WRITERS[name](report, output, figures=figures, figure_format="not-a-format")
    assert not output.exists()
    assert snapshot_files(tmp_path) == before


REPORT_KEYS = {
    "tm": {
        "openmmpolymer",
        "versions",
        "method",
        "run_dir",
        "curve",
        "transition",
        "notes",
        "temperature_k",
        "resolved",
        "heating_rate_k_per_ns",
        "enthalpy_units",
    },
    "breaking": {
        "openmmpolymer",
        "versions",
        "run_dir",
        "manifest_path",
        "curves",
        "replicas",
        "replica_indices",
        "strength_mpa",
        "replica_spread_mpa",
        "resolved",
        "failure_fraction",
        "confirmation_steps",
        "notes",
    },
    "elongation": {
        "openmmpolymer",
        "versions",
        "run_dir",
        "manifest_path",
        "curves",
        "replicas",
        "replica_indices",
        "elongation_percent",
        "replica_spread_percent",
        "resolved",
        "failure_fraction",
        "confirmation_steps",
        "notes",
    },
    "yield": {
        "openmmpolymer",
        "versions",
        "run_dir",
        "manifest_path",
        "curves",
        "replicas",
        "replica_indices",
        "strength_mpa",
        "replica_spread_mpa",
        "resolved",
        "offset_strain",
        "fit_min_strain",
        "fit_max_strain",
        "notes",
    },
}


@pytest.mark.parametrize("name", REPORT_KEYS)
def test_report_json_keys_match_the_recorded_schema(tmp_path: Path, name: str) -> None:
    """Convergence and rate key sets are pinned in their existing writer tests."""
    report = _report(name, tmp_path / "run")
    files = WRITERS[name](report, tmp_path / "report", figures=False)
    assert Path(files.json).name == f"{name}.json"
    assert set(json.loads(Path(files.json).read_text())) == REPORT_KEYS[name]


def test_structure_nested_schema_and_complete_json_stay_stable(tmp_path: Path) -> None:
    """Pin the explicit nested records before replacing their builders with asdict."""
    dimensions = ChainDimensions(6.0, 1.0, 6.0, 7.0, 6.7, True)
    settled = Equilibration(1, 2.0, 3, 2.5, 0.5, 0.01, 0.02, True)
    values = np.array([0.0, 1.0, 2.0])
    report = replace(
        _report("structure", tmp_path / "run"),
        stage="trajectory",
        stage_source="requested",
        is_snapshot=False,
        n_frames=3,
        interval_ps=2.0,
        n_chains=2,
        atoms_per_chain=4,
        backbone=(0, 1, 2),
        backbone_source="argument",
        distribution=RadialDistribution(
            values, values + 1, values + 2, 1.0, 2.0, 3.0, 2.5, 3, 12, True
        ),
        structure=StructureFactor(
            values + 1,
            np.array([float("nan"), 1.0, 2.0]),
            np.array([0, 2, 4]),
            2.0,
            1.0,
            3,
            False,
        ),
        conformation=ConformationSeries(
            "trajectory", values, values + 6, values + 1, dimensions, settled, 2, 3
        ),
        persistence=PersistenceLength(
            values, values / 2, 0.15, float("inf"), 3, 0.45, False
        ),
        displacement=MeanSquaredDisplacement(
            values, values / 10, float("nan"), None, 0.02, 2, 3, False
        ),
        relaxation=EndToEndRelaxation(values, values / 2, None, 4.0, 2, 3, False),
        recorded_chains=dimensions,
        notes=("Unresolved diagnostics stay null.",),
    )
    files = write_structure_report(report, tmp_path / "report", figures=False)
    record = json.loads(Path(files.json).read_text())
    schemas = {
        "radial_distribution": "r_nm g_r coordination_number first_peak_nm first_peak_height number_density_nm3 r_max_nm n_frames n_pairs heavy_atoms_only",
        "structure_factor": "q_per_nm s_q n_vectors first_peak_per_nm q_min_per_nm n_frames heavy_atoms_only",
        "conformation": "stage time_ps mean_squared_end_to_end_nm2 mean_radius_of_gyration_nm mean settled n_chains n_frames",
        "persistence": "separation correlation bond_length_nm persistence_length_nm n_bonds contour_length_nm decayed",
        "displacement": "lag_ps msd_nm2 log_slope diffusion_coefficient_cm2_s box_drift_fraction n_chains n_origins diffusive",
        "relaxation": "lag_ps correlation relaxation_time_ps trajectory_ps n_chains n_origins decorrelated",
        "recorded_chains": "mean_squared_end_to_end_nm2 mean_radius_of_gyration_nm ratio_of_squares characteristic_ratio expected_characteristic_ratio consistent",
    }
    for name, keys in schemas.items():
        assert list(record[name]) == keys.split()
    assert list(record["conformation"]["mean"]) == schemas["recorded_chains"].split()
    assert list(record["conformation"]["settled"]) == [
        "start_index",
        "start_ps",
        "n_samples",
        "n_independent_samples",
        "correlation_time_ps",
        "relative_standard_error",
        "relative_drift",
        "equilibrated",
    ]
    record["openmmpolymer"] = "VERSION"
    record["run_dir"] = "RUN_DIR"
    text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "abc8033ce6e2e8409fd19741bc2a36c625ac658f9eea01a0e0505cba0dba2ed0"
    )


@pytest.mark.parametrize("name", REPORT_KEYS)
def test_consolidated_writer_preserves_report_and_adds_provenance(
    tmp_path: Path,
    name: str,
) -> None:
    report = _report(name, tmp_path / "run")
    manifest = RunManifest.load(report.run_dir)
    if manifest is None:
        manifest = RunManifest(protocol="tm_heating", seed=42)
    manifest.versions = {"openmm": "fixture-version"}
    manifest.save(report.run_dir)
    files = WRITERS[name](report, tmp_path / "report", figures=False)
    record = json.loads(Path(files.json).read_text())
    assert record.pop("openmmpolymer") == __version__
    assert record.pop("versions") == manifest.versions
    expected = json_value(asdict(report))
    if name == "tm":
        expected.update(
            method="apparent melting from NPT heating",
            temperature_k=report.temperature_k,
            resolved=report.resolved,
            heating_rate_k_per_ns=report.curve.heating_rate_k_per_ns,
            enthalpy_units="kJ/mol of simulation cells",
        )
    assert record == expected


def test_melting_accepts_matplotlib_extension_and_keeps_paired_plot(
    tmp_path: Path,
) -> None:
    report = _report("tm", tmp_path / "run")
    figure = plot_melting(report)
    assert len(figure.axes) == 2
    assert tuple(figure.get_size_inches()) == (7.0, 7.0)
    for axis in figure.axes:
        assert len(axis.patches) == 1
        assert len(axis.lines) == 1
        np.testing.assert_array_equal(
            axis.lines[0].get_xdata(), report.curve.temperature_k
        )
    np.testing.assert_allclose(
        np.asarray(figure.axes[0].lines[0].get_ydata(), dtype=float),
        report.curve.specific_volume_cm3_g,
    )
    np.testing.assert_allclose(
        np.asarray(figure.axes[1].lines[0].get_ydata(), dtype=float),
        report.curve.enthalpy_kj_mol,
    )
    assert figure.axes[1].get_xlabel() == "Temperature (K)"
    files = write_melting_report(report, tmp_path / "report", figure_format="jpeg")
    assert Path(files.figures[0]).name == "melting.jpeg"
    assert Path(files.figures[0]).stat().st_size > 1000


@pytest.mark.parametrize("name", ["tg", "mechanics"])
def test_stage_names_cannot_create_figure_subdirectories(
    tmp_path: Path, name: str
) -> None:
    report = _report(name, tmp_path / "run")
    stage = "../../outside/curve"
    if name == "tg":
        temperatures, densities = two_line_curve()
        quench = QuenchCurve(
            stage, temperatures, densities, 1.0 / densities, 100.0, 200.0
        )
        report = replace(
            report, curves=(quench,), transitions=(glass_transition(quench),)
        )
    else:
        strains = np.linspace(0.0, 0.03, 8)
        curve = nominal_curve(strains, strains * 1000.0, stage=stage)
        report = replace(report, curves=(curve,), replicas=(youngs_modulus(curve),))
    output = tmp_path / "report"
    files = WRITERS[name](report, output)
    assert len(files.figures) == 1
    expected = (
        "quench_outside_curve.png"
        if name == "tg"
        else "stress_strain_outside_curve.png"
    )
    assert Path(files.figures[0]) == output / expected
    assert Path(files.figures[0]).is_file()
    assert not (tmp_path / "outside").exists()
