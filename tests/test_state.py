"""Portable-state readers preserve unwrapped coordinates and triclinic boxes."""

from pathlib import Path
from typing import Any

import numpy as np
import openmm as mm
from openmm import unit

from openmmpolymer._state import (
    box_diagonal_nm,
    box_vectors_nm,
    positions_nm,
    read_state,
)

from .helpers import bare_simulation


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
