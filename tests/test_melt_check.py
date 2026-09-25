"""Tests for the check on whether a melt had settled before it was cooled.

Read off a real trajectory the package wrote - the dimer fixture - because
what this package writes is what the check has to be able to read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmmpolymer.melt_check import MSD_RG_MULTIPLE, melt_equilibration
from openmmpolymer.trajectory import AnalysisError


def test_the_melt_check_fails_honestly_when_the_stage_wrote_no_trajectory(
    dimer_run_directory: Path,
) -> None:
    """Half the evidence missing is not a pass.

    The NPT stage writes no trajectory unless it is asked to, so the chain
    half of the check has nothing to read. The verdict then has to say that,
    rather than quietly reporting on the volume alone.
    """
    manifest = json.loads((dimer_run_directory / "manifest.json").read_text())
    stage = manifest["stages"]["02_nvt"]
    Path(str(stage["final_pdb"])).replace(dimer_run_directory / "03_alone.pdb")
    manifest["stages"] = {
        "03_alone": {
            "name": "03_alone",
            "final_pdb": str(dimer_run_directory / "03_alone.pdb"),
            "csv": stage["csv"],
            "samples": {},
        }
    }
    (dimer_run_directory / "manifest.json").write_text(json.dumps(manifest))

    verdict = melt_equilibration(dimer_run_directory, "03_alone")

    assert verdict.displacement is None
    assert verdict.chains_moved is False
    assert verdict.equilibrated is False
    assert any("no trajectory" in reason for reason in verdict.unchecked)


def test_the_melt_check_reads_a_real_trajectory(dimer_run_directory: Path) -> None:
    """The displacement is measured against the chain size it was given."""
    verdict = melt_equilibration(
        dimer_run_directory, "02_nvt", radius_of_gyration_nm=0.02
    )

    assert verdict.displacement is not None
    assert verdict.displacement_nm2 is not None
    assert verdict.displacement_target_nm2 == pytest.approx(MSD_RG_MULTIPLE * 0.02**2)
    assert verdict.displacement_lag_ps is not None


def test_a_chain_that_moved_less_than_its_own_size_is_not_equilibrated(
    dimer_run_directory: Path,
) -> None:
    """Two picoseconds of argon does not move a chain past its own radius."""
    verdict = melt_equilibration(
        dimer_run_directory, "02_nvt", radius_of_gyration_nm=100.0
    )

    assert verdict.chains_moved is False
    assert verdict.equilibrated is False


def test_the_melt_check_says_so_when_there_is_no_radius_of_gyration(
    dimer_run_directory: Path,
) -> None:
    """There is then nothing to measure the displacement against."""
    verdict = melt_equilibration(dimer_run_directory, "02_nvt")

    assert verdict.radius_of_gyration_nm is None
    assert verdict.chains_moved is False
    assert any("radius of gyration" in reason for reason in verdict.unchecked)


def test_the_melt_check_names_a_stage_the_manifest_does_not_have(
    dimer_run_directory: Path,
) -> None:
    """Saying what is there is the difference between a hint and a dead end."""
    verdict = melt_equilibration(dimer_run_directory, "09_missing")

    assert verdict.volume is None
    assert verdict.equilibrated is False
    assert any("02_nvt" in reason for reason in verdict.unchecked)


def test_the_melt_check_needs_a_manifest(tmp_path: Path) -> None:
    """Without one there is no way to know what a directory even holds."""
    with pytest.raises(AnalysisError, match="No manifest"):
        melt_equilibration(tmp_path)
