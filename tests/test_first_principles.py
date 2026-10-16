"""Check the tutorial arithmetic against hand calculations and real run files."""

from __future__ import annotations

import runpy
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmpolymer.reporters import TrajectoryOptions
from openmmpolymer.simulate import run_minimise, run_nvt
from openmmpolymer.timeseries import read_state_data

from .helpers import edited_json, write_manifest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "first_principles.py"
# The teaching script deliberately leaves its NumPy formulas free of type scaffolding.
example = runpy.run_path(str(EXAMPLE))


@pytest.fixture
def saved_run(dimer_argon_run: Any, tmp_path: Path) -> Path:
    """A real two-row CSV and ten-frame trajectory sharing a manifest."""
    minimised = run_minimise(dimer_argon_run, tmp_path / "00_minimise")
    measured = run_nvt(
        dimer_argon_run,
        tmp_path / "02_nvt",
        temperature_k=120.0,
        duration_ps=2.0,
        friction_ps=20.0,
        report_interval_ps=1.0,
        trajectory=TrajectoryOptions("xtc", interval_ps=0.2),
        state_in=minimised.final_state,
    )
    write_manifest(
        tmp_path,
        {"02_nvt": asdict(measured)},
        box={"n_molecules": 32, "atoms_per_chain": 2, "box_nm": [2.4, 2.4, 2.4]},
    )
    return tmp_path


def _example(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_density_converts_units_before_averaging_fluctuating_volumes() -> None:
    """602.214076 amu in 1 nm³ is 1 g/cm³; averaging 1/V matters."""
    density = example["density_g_cm3"](602.214076, np.array([1.0, 2.0]))
    assert density == pytest.approx([1.0, 0.5])
    assert np.mean(density) == pytest.approx(0.75)


def test_chain_sizes_weight_all_atoms_and_use_backbone_endpoints() -> None:
    """The mass centre is x=1.25; a translated, doubled chain has fourfold Rg²."""
    positions = np.zeros((2, 3, 3))
    positions[0, :, 0] = [0.0, 1.0, 3.0]
    positions[1, :, 0] = [10.0, 12.0, 16.0]
    rg2, r2 = example["chain_dimensions"](positions, np.array([1.0, 2.0, 1.0]), (0, 1))
    assert rg2 == pytest.approx([1.1875, 4.75])
    assert r2 == pytest.approx([1.0, 4.0])


def test_the_standalone_demo_recovers_its_known_properties() -> None:
    """Exercise the documented command, including slope signs and unit conversion."""
    result = _example()
    assert result.returncode == 0, result.stderr
    expected = {
        "Density = 0.996323 g/cm³",
        "Rg² = 1.250000 nm²",
        "R² = 9.000000 nm²",
        "Tg = 350.000000 K",
        "E = 3000.000000 MPa",
        "Poisson ratio = 0.350000",
        "K = 3333.333333 MPa",
        "G = 1111.111111 MPa",
        "Nominal stress = 80.000000 MPa",
        "G(t=0 ps) = 1000.000000 MPa",
        "G(t=10 ps) = 367.879441 MPa",
        "G(t=20 ps) = 135.335283 MPa",
        "D = 0.002000 nm²/ps",
        "D = 2e-09 m²/s",
    }
    assert expected <= set(result.stdout.splitlines())
    assert "not material predictions" in result.stdout


def test_a_saved_run_uses_retained_rows_and_real_trajectory_frames(
    saved_run: Path,
) -> None:
    with edited_json(saved_run / "manifest.json") as manifest:
        manifest["chains"] = {"backbone": [0, 1]}
    state = read_state_data(saved_run / manifest["stages"]["02_nvt"]["csv"])
    result = _example("--run", str(saved_run), "--stage", "02_nvt")

    assert result.returncode == 0, result.stderr
    assert "Saved stage: 02_nvt" in result.stdout
    assert "Rows retained: 1/2" in result.stdout
    assert "Frames retained: 5/10" in result.stdout
    assert _number(result.stdout, "Mean reported density") == pytest.approx(
        state.density_g_cm3[-1], abs=5e-7
    )
    # Equal-mass dimers obey R² = 4 Rg² at every frame, including moving dimers.
    assert _number(result.stdout, "Mean R²") == pytest.approx(
        4.0 * _number(result.stdout, "Mean Rg²"), abs=3e-6
    )
    assert "does not establish equilibration" in result.stdout


def test_a_saved_snapshot_keeps_its_only_frame_and_marks_the_limitation(
    saved_run: Path,
) -> None:
    from openmm import app, unit

    for trajectory in saved_run.glob("*.xtc"):
        trajectory.unlink()
    snapshot = saved_run / "02_nvt.pdb"
    pdb = app.PDBFile(str(snapshot))
    # Half the dimers have Rg=0.1 nm, half Rg=0.3 nm. These are not one RMS.
    positions = np.zeros((64, 3))
    positions[1::2, 0] = np.tile([0.2, 0.6], 16)
    with snapshot.open("w") as handle:
        app.PDBFile.writeFile(pdb.topology, positions * unit.nanometer, handle)
    result = _example("--run", str(saved_run), "--discard-fraction", "0.9")

    assert result.returncode == 0, result.stderr
    assert "Frames retained: 1/1" in result.stdout
    assert "One saved snapshot: chain sizes only; no time average or uncertainty." in (
        result.stdout
    )
    assert "No recorded backbone: R² skipped" in result.stdout
    assert "Mean R² =" not in result.stdout
    assert _number(result.stdout, "Mean Rg") == pytest.approx(0.2, abs=5e-7)
    assert _number(result.stdout, "RMS Rg") == pytest.approx(np.sqrt(0.05), abs=5e-7)


@pytest.mark.parametrize("missing", ("coordinates", "csv"))
def test_missing_saved_data_are_reported_without_inventing_results(
    saved_run: Path, missing: str
) -> None:
    extensions = ("*.xtc", "*.pdb") if missing == "coordinates" else ("*.csv",)
    for extension in extensions:
        for artifact in saved_run.glob(extension):
            artifact.unlink()
    result = _example("--run", str(saved_run))

    if missing == "coordinates":
        assert result.returncode == 0, result.stderr
        assert "Mean reported density =" in result.stdout
        notice = (
            "No saved coordinates: mass-based density and chain dimensions skipped."
        )
        assert notice in result.stdout
        assert "Mean Rg² =" not in result.stdout
    else:
        assert result.returncode == 2
        assert "CSV" in result.stderr
        assert "Mean reported density =" not in result.stdout


def _number(output: str, label: str) -> float:
    line = next(line for line in output.splitlines() if line.startswith(f"{label} = "))
    return float(line.split(" = ", 1)[1].split()[0])
