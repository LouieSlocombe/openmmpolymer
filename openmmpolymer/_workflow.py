"""The machinery the measurement workflows share.

Every scan equilibrates once, records what it was asked for before any
dynamics, and branches its replicas from the one equilibrated cell. The
pieces of that which are not specific to one measurement live here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._validation import require_positive

if TYPE_CHECKING:
    from .protocols import RunSummary
    from .simulate import RunContext


def run_fingerprint(run: RunContext, *, spec_key: str = "system") -> dict[str, Any]:
    """The inputs a resumed scan has to share with the one that recorded it.

    *spec_key* names the System settings as each workflow's record already
    spells them, because those records are compared as saved.
    """
    return {
        spec_key: asdict(run.spec),
        "seed": run.seed,
        "system_sha256": hashlib.sha256(run.system_xml.encode()).hexdigest(),
        "coordinates_sha256": hashlib.sha256(
            np.asarray(run.box.positions_nm, dtype=np.float64).tobytes()
        ).hexdigest(),
        "box_nm": list(run.box.box_nm),
    }


def settled_state(
    summary: RunSummary, directory: Path, *, error: type[Exception], verb: str
) -> str:
    """The state the equilibration finished at, whether it ran or resumed.

    Taken from the summary, which threads the state through skipped stages as
    well as run ones, rather than from the last manifest entry with a state:
    on a resume that entry is whatever the previous attempt got furthest
    through, and every branch has to start from the *equilibrated* cell.
    """
    state = summary.final_state
    if state and state != "None" and Path(state).is_file():
        return str(state)
    raise error(
        f"{directory} has no finished equilibration stage to {verb} from. Run "
        "the equilibration first, or delete the manifest and start over."
    )


def equilibrated_box_nm(state_path: str | Path) -> list[float]:
    """The cell edges a saved state carries, in nanometres.

    Read from the state rather than from ``run.box``, which is the *packed*
    cell: strain measured against the packed edges would be measured against
    a cell that stopped existing at the first barostat move.
    """
    import openmm as mm
    from openmm import unit

    state = mm.XmlSerializer.deserialize(Path(state_path).read_text())
    vectors = state.getPeriodicBoxVectors()
    return [
        float(vectors[axis][axis].value_in_unit(unit.nanometer)) for axis in range(3)
    ]


def group_by_stem(names: Sequence[str]) -> list[tuple[str, ...]]:
    """Group stage names into one tuple of chunks per replica.

    Grouped on the stem a numeric chunk suffix hangs off, so a ladder split for
    resume comes back as one curve and two replicas do not come back as one. A
    name with no such suffix is a group of its own.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        stem, _, tail = name.rpartition("_")
        groups.setdefault(stem if stem and tail.isdigit() else name, []).append(name)
    return [tuple(group) for group in groups.values()]


def sample_spread(values: Sequence[float]) -> float | None:
    """The sample standard deviation of the finite values, or None below two.

    None rather than zero: one replica has no spread to report, and a zero
    would read as several runs that agreed perfectly.
    """
    usable = [value for value in values if math.isfinite(value)]
    if len(usable) < 2:
        return None
    return float(np.std(usable, ddof=1))


def validate_hold_times(
    values: Sequence[float], *, name: str = "hold_times_ps"
) -> tuple[float, ...]:
    """Require three distinct positive holds, preserving their requested order."""
    holds = tuple(require_positive(value, None, name=name) for value in values)
    if len(holds) < 3 or any(
        math.isclose(first, second, rel_tol=1e-8)
        for first, second in pairwise(sorted(holds))
    ):
        raise ValueError(f"{name} needs at least three distinct positive holds.")
    return holds


def validate_extrapolation_limit(value: float) -> None:
    """Reject an undefined or negative extrapolation allowance."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("max_extrapolation_decades must be finite and nonnegative.")
