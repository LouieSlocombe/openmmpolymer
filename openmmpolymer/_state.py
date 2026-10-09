"""Read portable OpenMM states and their coordinates in nanometres."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import openmm as mm
from openmm import unit


def read_state(path: str | Path) -> Any:
    """Deserialize the saved object; callers validate its required contents."""
    return mm.XmlSerializer.deserialize(Path(path).read_text())


def positions_nm(state: Any) -> npt.NDArray[np.float64]:
    """Return a state's positions, retaining its existing wrapping convention."""
    return np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )


def box_vectors_nm(state: Any) -> npt.NDArray[np.float64]:
    """Return the three periodic box vectors as rows, in nanometres."""
    return np.asarray(
        state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )


def box_diagonal_nm(state: Any) -> npt.NDArray[np.float64]:
    """Return the diagonal box components used as the strain reference."""
    return np.diag(box_vectors_nm(state)).copy()
