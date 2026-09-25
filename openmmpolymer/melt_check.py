"""Whether a melt had settled before it was cooled.

Two conditions, and both have to be shown rather than assumed. The cell volume
has to have stopped drifting faster than its own noise, and the chains' centres
of mass have to have travelled further than the chains are big, diffusively.
The first says the density is a density; only the second says anything about
the chains, and a melt whose density settled in two hundred picoseconds can
still be exactly as packmol left it.

A verdict with half its evidence missing is not a pass, so the check fails
whenever either half could not be measured - and says which, and what to do
about it, so that "this melt is not equilibrated" stays distinguishable from
"the data that would say was never written".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conformation import MeanSquaredDisplacement, centre_of_mass_msd
from .protocols import RunManifest
from .timeseries import Equilibration, equilibration, read_state_data
from .trajectory import AnalysisError, open_run

log = logging.getLogger(__name__)

#: How far a chain's centre of mass has to travel, in units of its own squared
#: radius of gyration, before a melt has plausibly forgotten how it was packed.
MSD_RG_MULTIPLE = 2.0


@dataclass(frozen=True)
class MeltEquilibration:
    """Whether a melt had settled before it was cooled.

    Args:
        stage: Which stage was checked.
        volume: What the cell volume did, or None if it could not be read.
        displacement: What the chains did, or None for the same reason.
        radius_of_gyration_nm: The chain size the displacement is measured
            against.
        displacement_target_nm2: :data:`MSD_RG_MULTIPLE` times its square.
        displacement_nm2: How far the chains actually went, at the longest lag.
        displacement_lag_ps: The lag that was read at. The longest lag has the
            fewest time origins behind it, so it is worth knowing.
        volume_settled: Whether the volume stopped drifting.
        chains_moved: Whether the chains went further than
            *displacement_target_nm2*, diffusively.
        equilibrated: Both of the above, and so False whenever either could
            not be measured.
        unchecked: One sentence per thing that could not be measured, each
            naming what to do about it.
    """

    stage: str
    volume: Equilibration | None
    displacement: MeanSquaredDisplacement | None
    radius_of_gyration_nm: float | None
    displacement_target_nm2: float | None
    displacement_nm2: float | None
    displacement_lag_ps: float | None
    volume_settled: bool
    chains_moved: bool
    equilibrated: bool
    unchecked: tuple[str, ...]


def melt_equilibration(
    run_dir: str | Path,
    stage: str = "05_npt",
    *,
    radius_of_gyration_nm: float | None = None,
    max_lag_fraction: float = 0.5,
    stride: int = 1,
) -> MeltEquilibration:
    """Check whether a melt had settled before it was cooled.

    Reading a trajectory leaves MDAnalysis' offset and lock files beside it, so
    this writes into the run directory even though it is an analysis, and it
    cannot be pointed at a read-only archive.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        stage: The equilibration stage to check.
        radius_of_gyration_nm: Override the chain size recorded in the
            manifest.
        max_lag_fraction: Passed to
            :func:`~openmmpolymer.conformation.centre_of_mass_msd`.
        stride: Frames to skip when reading the trajectory.

    Returns:
        The verdict, and what could not be checked.

    Raises:
        AnalysisError: There is no manifest to read.
    """
    directory = Path(run_dir)
    manifest = RunManifest.load(directory)
    if manifest is None:
        raise AnalysisError(f"No manifest in {directory}.")

    unchecked: list[str] = []
    volume = _volume_settling(stage, manifest, unchecked)
    displacement = _chain_displacement(
        directory, stage, max_lag_fraction, stride, unchecked
    )
    radius = radius_of_gyration_nm
    if radius is None:
        radius = _recorded_float(manifest.chains, "mean_radius_of_gyration_nm")
        if radius is None:
            unchecked.append(
                "chain displacement: the manifest records no radius of "
                "gyration, so there is nothing to measure the displacement "
                "against. Give run_protocol a chain_backbone, or pass "
                "radius_of_gyration_nm."
            )
    verdict = _verdict(stage, volume, displacement, radius, tuple(unchecked))
    for reason in verdict.unchecked:
        log.info("%s", reason)
    return verdict


def _volume_settling(
    stage: str,
    manifest: RunManifest,
    unchecked: list[str],
) -> Equilibration | None:
    """What the cell volume did over the stage, or None with a reason why not."""
    recorded = manifest.stages.get(stage)
    if recorded is None:
        unchecked.append(
            f"box volume: the manifest has no stage {stage!r}. It records: "
            f"{', '.join(manifest.stages) or 'nothing'}."
        )
        return None
    csv = recorded.get("csv")
    if not csv or not Path(str(csv)).is_file():
        unchecked.append(
            f"box volume: stage {stage!r} left no state-data CSV to read, so "
            "there is no volume series."
        )
        return None
    try:
        series = read_state_data(csv, stage=stage)
        return equilibration(series.time_ps, series.volume_nm3)
    except AnalysisError as error:
        unchecked.append(f"box volume: {error}")
        return None


def _chain_displacement(
    directory: Path,
    stage: str,
    max_lag_fraction: float,
    stride: int,
    unchecked: list[str],
) -> MeanSquaredDisplacement | None:
    """How far the chains went, or None with a reason why it is not known."""
    try:
        ensemble = open_run(directory, stage)
    except AnalysisError as error:
        unchecked.append(f"chain displacement: {error}")
        return None
    if ensemble.is_snapshot:
        unchecked.append(
            f"chain displacement: stage {stage!r} wrote no trajectory, so it "
            "could not be measured. Give the equilibration stage a trajectory "
            "- standard_melt_equilibration takes npt_trajectory, and TgSpec "
            "takes npt_trajectory_ps."
        )
        return None
    try:
        return centre_of_mass_msd(
            ensemble, max_lag_fraction=max_lag_fraction, stride=stride
        )
    except AnalysisError as error:
        unchecked.append(f"chain displacement: {error}")
        return None


def _recorded_float(record: dict[str, Any] | None, key: str) -> float | None:
    """Read one float out of the manifest, if it is there and usable."""
    if not record:
        return None
    value = record.get(key)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _verdict(
    stage: str,
    volume: Equilibration | None,
    displacement: MeanSquaredDisplacement | None,
    radius_of_gyration_nm: float | None,
    unchecked: tuple[str, ...],
) -> MeltEquilibration:
    """Combine the two halves of the check into one verdict."""
    settled = volume is not None and volume.equilibrated
    target = (
        None
        if radius_of_gyration_nm is None
        else MSD_RG_MULTIPLE * radius_of_gyration_nm**2
    )
    travelled: float | None = None
    lag: float | None = None
    if displacement is not None and displacement.msd_nm2.size:
        travelled = float(displacement.msd_nm2[-1])
        lag = float(displacement.lag_ps[-1])
    moved = (
        displacement is not None
        and displacement.diffusive
        and travelled is not None
        and target is not None
        and travelled > target
    )
    return MeltEquilibration(
        stage=stage,
        volume=volume,
        displacement=displacement,
        radius_of_gyration_nm=radius_of_gyration_nm,
        displacement_target_nm2=target,
        displacement_nm2=travelled,
        displacement_lag_ps=lag,
        volume_settled=settled,
        chains_moved=moved,
        equilibrated=settled and moved,
        unchecked=unchecked,
    )
