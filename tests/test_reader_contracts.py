"""Reader consolidation keeps one manifest snapshot and its refusal semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import openmm as mm
import pytest
from openmm import unit

from openmmpolymer._state import (
    box_diagonal_nm,
    box_vectors_nm,
    positions_nm,
    read_state,
)
from openmmpolymer.convergence_report import analyse_convergence
from openmmpolymer.elastic_rates import analyse_elastic_rates
from openmmpolymer.elasticity import stress_strain
from openmmpolymer.mechanical import BULK_STEM, analyse_mechanics
from openmmpolymer.melt_check import melt_equilibration
from openmmpolymer.protocols import RunManifest
from openmmpolymer.relaxation import relaxation_curve
from openmmpolymer.tm import heating_curve, heating_stages
from openmmpolymer.trajectory import AnalysisError

from .helpers import (
    bare_simulation,
    planted_curve,
    write_bulk,
    write_deformation,
    write_heating,
    write_polymer_snapshot,
    write_relaxation,
    write_shear,
)


@pytest.mark.parametrize(
    "reader",
    [
        "melt",
        "snapshot_convergence",
        "relaxation_convergence",
        "heating",
        "stress",
        "mechanics",
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
    elif reader in {"stress", "mechanics"}:
        write_deformation(tmp_path)
        if reader == "mechanics":
            write_bulk(tmp_path, stage=BULK_STEM)
            write_shear(tmp_path)
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
    elif reader == "mechanics":
        analyse_mechanics(tmp_path)
    else:
        relaxation_curve(tmp_path)
    assert manifest_reads == [tmp_path / "manifest.json"]


def test_mechanics_without_a_manifest_keeps_its_final_refusal(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError) as failure:
        analyse_mechanics(tmp_path)
    assert str(failure.value) == (
        f"Nothing in {tmp_path} was a mechanical measurement. It records: nothing."
    )


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


def test_state_readers_preserve_unwrapped_positions_and_box_components(
    argon_run: Any, tmp_path: Path
) -> None:
    positions = argon_run.box.positions_nm.copy()
    positions[0] += (3.0, 0.0, 0.0)
    vectors = np.array([[3.0, 0.0, 0.0], [0.4, 2.8, 0.0], [0.2, 0.3, 2.6]])
    simulation = bare_simulation(
        mm.XmlSerializer.deserialize(argon_run.system_xml),
        argon_run.box.topology,
        positions,
    )
    simulation.context.setPeriodicBoxVectors(*(vectors * unit.nanometer))
    path = tmp_path / "state.xml"
    simulation.saveState(str(path))

    state = read_state(path)
    np.testing.assert_array_equal(positions_nm(state), positions)
    np.testing.assert_array_equal(box_vectors_nm(state), vectors)
    # Strain references use the diagonal, not the length of each tilted vector.
    np.testing.assert_array_equal(box_diagonal_nm(state), [3.0, 2.8, 2.6])
