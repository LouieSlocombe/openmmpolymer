"""Reader consolidation keeps one manifest snapshot and its refusal semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmmpolymer.convergence_report import analyse_convergence
from openmmpolymer.elastic_rates import analyse_elastic_rates
from openmmpolymer.elasticity import stress_strain
from openmmpolymer.melt_check import melt_equilibration
from openmmpolymer.protocols import RunManifest
from openmmpolymer.relaxation import relaxation_curve
from openmmpolymer.tm import heating_curve, heating_stages
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    planted_curve,
    write_deformation,
    write_heating,
    write_polymer_snapshot,
    write_relaxation,
)


@pytest.mark.parametrize(
    "reader",
    [
        "melt",
        "snapshot_convergence",
        "relaxation_convergence",
        "heating",
        "stress",
        "relaxation",
    ],
)
def test_one_manifest_snapshot_serves_every_part_of_a_reader(
    tmp_path: Path, manifest_reads: list[Path], reader: str
) -> None:
    if reader in {"melt", "snapshot_convergence"}:
        write_polymer_snapshot(tmp_path)
    elif reader == "heating":
        write_heating(tmp_path, planted_curve())
    elif reader == "stress":
        write_deformation(tmp_path)
    else:
        write_relaxation(tmp_path)
    manifest_reads.clear()
    if reader == "melt":
        melt_equilibration(tmp_path)
    elif reader in {"snapshot_convergence", "relaxation_convergence"}:
        analyse_convergence(tmp_path)
    elif reader == "heating":
        heating_curve(tmp_path)
    elif reader == "stress":
        stress_strain(tmp_path)
    else:
        relaxation_curve(tmp_path)
    assert manifest_reads == [tmp_path / "manifest.json"]


def test_heating_selection_keeps_empty_and_malformed_candidates_distinct(
    tmp_path: Path,
) -> None:
    manifest = RunManifest(protocol="empty", seed=7)
    manifest.save(tmp_path)
    assert heating_stages(tmp_path) == ()
    with pytest.raises(AnalysisError, match="No heating stages"):
        heating_curve(tmp_path)
    manifest.stages["malformed"] = {"samples": {"segment_enthalpy_kj_mol": [1.0]}}
    manifest.save(tmp_path)
    assert heating_stages(tmp_path) == ("malformed",)
    with pytest.raises(
        AnalysisError, match="Malformed heating samples in stage 'malformed'"
    ):
        heating_curve(tmp_path)


def test_elastic_stage_selection_does_not_add_full_manifest_schema_requirements(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text('{"stages": {}, "legacy_extra": true}')
    with pytest.raises(AnalysisError, match="has no shear_modulus measurement"):
        analyse_elastic_rates(
            [tmp_path], property_name="shear_modulus", target_rate=1.0
        )


def test_melt_missing_stage_keeps_its_original_note(tmp_path: Path) -> None:
    RunManifest(protocol="empty", seed=7).save(tmp_path)
    result = melt_equilibration(tmp_path, "missing")
    assert (
        result.unchecked[0]
        == "box volume: the manifest has no stage 'missing'. It records: nothing."
    )
