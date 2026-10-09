"""Shared report boundaries remain stable while their writers consolidate."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer.convergence_report import ConvergenceReport, write_convergence_report
from openmmpolymer.mechanical import ModulusReport, write_mechanical_report
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
from openmmpolymer.tm import MeltingReport, melting_temperature, write_melting_report
from openmmpolymer.viscoelastic import RelaxationReport, write_relaxation_report

from .helpers import (
    PLANTED_TENSILE,
    planted_curve,
    planted_rate_report,
    snapshot_files,
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
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Consolidation F4: tensile writes JSON before validating formats.",
            ),
        )
        if name in PLANTED_TENSILE
        else name
        for name in WRITERS
    ],
)
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
