"""Reading a finished run's structure and dynamics back, and reporting them.

The workflow modules run a scan and then read it; this one only reads. Every
stage a run finishes leaves a closing structure, and a stage asked for a
trajectory leaves frames, so any run directory has something to say about how
its chains are arranged - the intermolecular pair distribution and the
structure factor - and, given a backbone, how big the chains are and how stiff.
A trajectory adds whether they moved: the centre-of-mass displacement and the
end-to-end relaxation.

Each measurement stands on its own. A cell of one molecule has no
intermolecular pairs, a snapshot has no displacement, a run with no recorded
backbone has no chain dimensions, and in each case the report says so in a note
and carries on with what it can measure, rather than failing the whole report
over the part it cannot.

The backbone is the awkward one. It comes from the attachment points the caps
consumed when the chain was built, and a run records it in its manifest only
when it was given one to measure the chains with. So it is taken as given,
else as recorded, and failing both it is inferred from the bond graph as the
longest shortest path through one chain's heavy atoms, which for a linear
polymer is the backbone. The report says which of those it used, because an
inferred path can end on a terminal side group and put one extra bond on the
end-to-end vector.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._files import ReportFiles
from ._validation import require_integer, require_positive
from ._workflow import optional, write_report_files
from .conformation import (
    ConformationSeries,
    EndToEndRelaxation,
    MeanSquaredDisplacement,
    PersistenceLength,
    centre_of_mass_msd,
    chain_conformation,
    end_to_end_relaxation,
    persistence_length,
)
from .correlations import (
    RadialDistribution,
    StructureFactor,
    radial_distribution,
    structure_factor,
)
from .plots import (
    plot_conformation,
    plot_correlations,
    plot_dynamics,
    plot_persistence,
)
from .protocols import ChainDimensions, RunManifest
from .timeseries import Equilibration
from .trajectory import (
    AnalysisError,
    Ensemble,
    StageFiles,
    backbone_indices,
    open_run,
    stage_files,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)

#: Most frames the pair distribution averages over, whatever the stride. Past
#: a few dozen frames the curve stops changing and the cost keeps climbing.
MAX_DISTRIBUTION_FRAMES = 50

#: Most frames the structure factor sums over. It is the expensive
#: measurement - every atom against every wavevector - and neighbouring frames
#: carry almost the same structure.
MAX_STRUCTURE_FACTOR_FRAMES = 8


@dataclass(frozen=True)
class StructureReport:
    """Everything a finished run directory has to say about its structure.

    Args:
        run_dir: The directory read.
        stage: The stage measured.
        stage_source: How it was chosen: ``"requested"``,
            ``"last_trajectory"`` or ``"last_snapshot"``.
        is_snapshot: Whether only a closing structure was read.
        n_frames: Frames in the stage. One for a snapshot.
        interval_ps: Time between frames. Zero for a snapshot.
        n_chains: Molecules in the cell.
        atoms_per_chain: Atoms in each one.
        stride: The frame stride asked for. The pair distribution and the
            structure factor may have used a larger one; their own
            ``n_frames`` say how many frames they saw.
        backbone: Backbone atom indices within one chain, or None when none
            was known.
        backbone_source: Where it came from: ``"argument"``, ``"manifest"``
            or ``"inferred"``, or None.
        backbone_file: The file it was read from, when it was read from one.
        distribution: The intermolecular pair distribution, or None.
        structure: The static structure factor, or None.
        conformation: The chain dimensions, frame by frame, or None.
        persistence: The persistence length, or None.
        displacement: The centre-of-mass displacement, or None.
        relaxation: The end-to-end relaxation, or None.
        recorded_chains: The dimensions the run measured at its end and put
            in the manifest, for cross-reference, or None.
        notes: What could not be measured, in plain sentences.
    """

    run_dir: str
    stage: str
    stage_source: str
    is_snapshot: bool
    n_frames: int
    interval_ps: float
    n_chains: int
    atoms_per_chain: int
    stride: int
    backbone: tuple[int, ...] | None
    backbone_source: str | None
    backbone_file: str | None
    distribution: RadialDistribution | None
    structure: StructureFactor | None
    conformation: ConformationSeries | None
    persistence: PersistenceLength | None
    displacement: MeanSquaredDisplacement | None
    relaxation: EndToEndRelaxation | None
    recorded_chains: ChainDimensions | None
    notes: tuple[str, ...]


# --------------------------------------------------------------------------
# Finding what to read
# --------------------------------------------------------------------------


def structure_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Find the stages that left coordinates to read.

    Every stage writes its closing structure, so this is normally every stage
    the manifest records. A stage whose files have since gone, or whose
    trajectory has no topology beside it, is left out.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.

    Returns:
        The stage names, in the order the manifest records them.

    Raises:
        AnalysisError: There is no manifest, it records no stages, or none of
            them left anything to read.
    """
    directory = Path(run_dir)
    manifest = RunManifest.load(directory)
    names = list(manifest.stages) if manifest is not None else []
    if not names:
        # Refuses a directory with no manifest, or no stages, in its own words.
        stage_files(directory)

    readable: list[str] = []
    for name in names:
        try:
            files = stage_files(directory, name)
        except AnalysisError as error:
            log.info("%s: %s", name, error)
            continue
        if files.topology is not None:
            readable.append(name)
    if not readable:
        raise AnalysisError(
            f"No stage in {directory} left coordinates to read. Every stage "
            f"writes <stem>.pdb at its end; the manifest records: "
            f"{', '.join(names)}."
        )
    return tuple(readable)


def _select_stage(directory: Path, stage: str | None) -> tuple[StageFiles, str]:
    """Choose the stage to measure, and say how it was chosen.

    A trajectory beats a snapshot because it can answer the dynamic
    questions; among trajectories the last one wins, because it is the most
    equilibrated cell in the directory.
    """
    readable = structure_stages(directory)
    if stage is not None:
        if stage not in readable:
            raise AnalysisError(
                f"Stage {stage!r} left no coordinates to read. Stages that did: "
                f"{', '.join(readable)}."
            )
        return stage_files(directory, stage), "requested"

    with_trajectory = [
        files
        for files in (stage_files(directory, name) for name in readable)
        if files.trajectory is not None
    ]
    if with_trajectory:
        return with_trajectory[-1], "last_trajectory"
    return stage_files(directory, readable[-1]), "last_snapshot"


# --------------------------------------------------------------------------
# The backbone
# --------------------------------------------------------------------------


def infer_backbone(ensemble: Ensemble) -> tuple[int, ...]:
    """Infer a chain's backbone from its bond graph.

    The longest shortest path through the first chain's heavy atoms, found
    with two breadth-first searches: from any atom to the farthest one, then
    from there to the farthest again. On a tree that is the diameter exactly,
    and a linear polymer's heavy-atom graph is a tree whose diameter is the
    backbone - or the backbone plus a terminal side group, when a side group
    on the last unit reaches further than the chain end does. Nothing here
    can tell those apart, which is why the caller records that the path was
    inferred.

    Args:
        ensemble: What to read the bonds from. Every chain is a copy of the
            same molecule, so the first one stands for all.

    Returns:
        Chain-local atom indices along the path, lower index first.

    Raises:
        AnalysisError: There is no topology, it records no bonds between the
            chain's heavy atoms, or those atoms are not one connected
            molecule.
    """
    if ensemble.topology is None:
        raise AnalysisError(
            "The ensemble carries no topology, so there is no bond graph to "
            "read a backbone from."
        )
    n_atoms = ensemble.atoms_per_chain
    heavy = [index for index in range(n_atoms) if not ensemble.is_hydrogen[index]]
    if not heavy:
        raise AnalysisError("The chain has no heavy atoms, so it has no backbone.")

    adjacency: dict[int, list[int]] = {index: [] for index in heavy}
    for first, second in ensemble.topology.bonds():
        i, j = int(first.index), int(second.index)
        if i in adjacency and j in adjacency:
            adjacency[i].append(j)
            adjacency[j].append(i)
    if not any(adjacency.values()):
        raise AnalysisError(
            f"The topology records no bonds between the chain's {len(heavy)} "
            "heavy atoms, so there is no graph to walk. A PDB written by "
            "packmol has no CONECT records; the closing structures this "
            "package writes do."
        )

    far, parents = _farthest(adjacency, heavy[0])
    if len(parents) != len(heavy):
        raise AnalysisError(
            f"Only {len(parents)} of the chain's {len(heavy)} heavy atoms are "
            "bonded to the first, so they are not one connected molecule. "
            "Check atoms_per_chain, or pass the backbone."
        )
    end, parents = _farthest(adjacency, far)
    path = [end]
    while path[-1] != far:
        path.append(parents[path[-1]])
    if path[0] > path[-1]:
        path.reverse()
    return tuple(int(index) for index in backbone_indices(path, n_atoms))


def _farthest(
    adjacency: dict[int, list[int]], start: int
) -> tuple[int, dict[int, int]]:
    """Breadth-first from *start*: the last atom reached, and every parent."""
    parents = {start: start}
    queue = deque([start])
    last = start
    while queue:
        last = queue.popleft()
        for neighbour in adjacency[last]:
            if neighbour not in parents:
                parents[neighbour] = last
                queue.append(neighbour)
    return last, parents


def _is_index_list(value: object) -> bool:
    """Whether a recorded value could be a backbone at all."""
    return (
        isinstance(value, list)
        and len(value) >= 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _resolve_backbone(
    directory: Path,
    manifest: RunManifest | None,
    ensemble: Ensemble,
    explicit: Sequence[int] | None,
    infer: bool,
    notes: list[str],
) -> tuple[tuple[int, ...] | None, str | None, str | None]:
    """Find the backbone - given, recorded in the manifest, or inferred, in
    that order - and say which, and from which file."""
    n_atoms = ensemble.atoms_per_chain

    if explicit is not None:
        try:
            path = backbone_indices(explicit, n_atoms)
        except AnalysisError as error:
            notes.append(
                f"The backbone given was not used: {error} Nothing else was "
                "tried, because an explicit path that does not fit is a "
                "question to answer rather than one to guess past."
            )
            return None, None, None
        return tuple(int(index) for index in path), "argument", None

    chains = manifest.chains if manifest is not None else None
    value = chains.get("backbone") if isinstance(chains, dict) else None
    if value is not None:
        found = _recorded_backbone(value, n_atoms, "manifest.json", notes)
        if found is not None:
            return found, "manifest", "manifest.json"

    if infer:
        try:
            inferred = infer_backbone(ensemble)
        except AnalysisError as error:
            notes.append(f"No backbone could be inferred: {error}")
        else:
            n_heavy = int((~ensemble.is_hydrogen).sum())
            notes.append(
                "Backbone inferred from the bond graph as the longest shortest "
                f"path through one chain's heavy atoms ({len(inferred)} of "
                f"{n_heavy}). A side group on the last unit can be picked up as "
                "the chain end, which puts one extra bond on the end-to-end "
                "vector; record chain_backbone on the run, or pass backbone=, "
                "to say what the backbone really is."
            )
            return inferred, "inferred", None

    notes.append(
        "No backbone is recorded for this run and none was given, so the "
        "chain dimensions, persistence length and end-to-end relaxation were "
        "not measured. Pass backbone= (--backbone on the command line), or "
        "run through a workflow that records chain_backbone."
    )
    return None, None, None


def _recorded_backbone(
    value: object, n_atoms: int, source: str, notes: list[str]
) -> tuple[int, ...] | None:
    """Validate a backbone read from a file, turning a bad one into a note."""
    if not _is_index_list(value):
        notes.append(
            f"{source} records a backbone that is not a list of atom indices, "
            "so it was not used."
        )
        return None
    try:
        path = backbone_indices(value, n_atoms)  # type: ignore[arg-type]
    except AnalysisError as error:
        notes.append(f"The backbone recorded in {source} was not used: {error}")
        return None
    return tuple(int(index) for index in path)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _capped_stride(n_frames: int, stride: int, cap: int) -> int:
    """The stride that keeps a measurement to at most *cap* frames."""
    return max(stride, -(-n_frames // cap))


def _recorded_chains(manifest: RunManifest | None) -> ChainDimensions | None:
    """The dimensions the run recorded at its end, if they are all there.

    Read key by key rather than ``ChainDimensions(**chains)``, so a manifest
    that records more than the dataclass holds - a backbone, say - still
    reads.
    """
    if manifest is None or not isinstance(manifest.chains, dict):
        return None
    chains = manifest.chains
    try:
        return ChainDimensions(
            mean_squared_end_to_end_nm2=float(chains["mean_squared_end_to_end_nm2"]),
            mean_radius_of_gyration_nm=float(chains["mean_radius_of_gyration_nm"]),
            ratio_of_squares=float(chains["ratio_of_squares"]),
            characteristic_ratio=float(chains["characteristic_ratio"]),
            expected_characteristic_ratio=float(
                chains["expected_characteristic_ratio"]
            ),
            consistent=bool(chains["consistent"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def analyse_structure(
    run_dir: str | Path,
    *,
    stage: str | None = None,
    backbone: Sequence[int] | None = None,
    infer_backbone: bool = True,
    expected_characteristic_ratio: float | None = None,
    stride: int = 1,
    heavy_atoms_only: bool = True,
    q_max_per_nm: float = 40.0,
    max_lag_fraction: float = 0.5,
) -> StructureReport:
    """Read everything a finished run has to say about its structure.

    Reads and returns; writes nothing of its own. Opening a trajectory does
    leave MDAnalysis' offset and lock files beside it, so this touches the
    run directory even though it is an analysis, and cannot be pointed at a
    read-only archive.

    The stage is the last one with a trajectory, or failing that the last
    one's closing structure. Each measurement that cannot be made becomes a
    note rather than an error: one molecule has no intermolecular pairs, a
    snapshot has no displacement, an unknown backbone gives no chain
    dimensions.

    Args:
        run_dir: A directory :func:`~openmmpolymer.protocols.run_protocol`
            wrote to.
        stage: The stage to measure, or None for the rule above.
        backbone: Backbone atom indices within one chain, overriding whatever
            the run recorded.
        infer_backbone: When nothing was given or recorded, infer the backbone
            from the bond graph with :func:`infer_backbone`.
        expected_characteristic_ratio: The polymer's C-infinity, overriding
            the value saved in the manifest. If neither is available, use
            the polyethylene default of 7.0.
        stride: Measure every *stride*-th frame. The pair distribution and the
            structure factor are further capped at
            :data:`MAX_DISTRIBUTION_FRAMES` and
            :data:`MAX_STRUCTURE_FACTOR_FRAMES` frames.
        heavy_atoms_only: Drop hydrogens from the pair distribution and the
            structure factor.
        q_max_per_nm: Highest wavevector for the structure factor.
        max_lag_fraction: Longest lag for the displacement and the end-to-end
            relaxation, as a fraction of the trajectory.

    Returns:
        The report.

    Raises:
        AnalysisError: There is no manifest, nothing in it left coordinates,
            or the stage asked for did not.
        ValueError: *stride* is not a positive integer.
    """
    require_integer(stride, name="stride")
    directory = Path(run_dir)
    files, stage_source = _select_stage(directory, stage)
    ensemble = open_run(directory, files.stage)
    manifest = RunManifest.load(directory)
    notes: list[str] = []
    if expected_characteristic_ratio is None:
        recorded = (
            manifest.chains
            if manifest is not None and isinstance(manifest.chains, dict)
            else {}
        )
        expected_characteristic_ratio = recorded.get(
            "expected_characteristic_ratio", 7.0
        )
    expected_characteristic_ratio = require_positive(
        expected_characteristic_ratio, None, name="expected_characteristic_ratio"
    )

    if ensemble.is_snapshot:
        if stage_source == "last_snapshot":
            notes.append(
                f"No stage in {directory} wrote a trajectory, so the closing "
                f"snapshot of {files.stage!r} was measured: the pair "
                "distribution, structure factor and chain dimensions are one "
                "frame, and the mean-squared displacement and end-to-end "
                "relaxation were not measured. Run a stage with "
                "trajectory='xtc' to measure anything time-dependent."
            )
        else:
            notes.append(
                f"Stage {files.stage!r} is a single snapshot, so the "
                "mean-squared displacement and end-to-end relaxation were not "
                "measured."
            )

    path, backbone_source, backbone_file = _resolve_backbone(
        directory, manifest, ensemble, backbone, infer_backbone, notes
    )

    pair_stride = _capped_stride(ensemble.n_frames, stride, MAX_DISTRIBUTION_FRAMES)
    factor_stride = _capped_stride(
        ensemble.n_frames, stride, MAX_STRUCTURE_FACTOR_FRAMES
    )
    distribution = optional(
        lambda: radial_distribution(
            ensemble, heavy_atoms_only=heavy_atoms_only, stride=pair_stride
        ),
        notes,
        "No pair distribution",
    )
    structure = optional(
        lambda: structure_factor(
            ensemble,
            q_max_per_nm=q_max_per_nm,
            heavy_atoms_only=heavy_atoms_only,
            stride=factor_stride,
        ),
        notes,
        "No structure factor",
    )

    conformation = persistence = displacement = relaxation = None
    if path is not None:
        conformation = optional(
            lambda: chain_conformation(
                ensemble,
                path,
                expected_characteristic_ratio=expected_characteristic_ratio,
                stride=stride,
            ),
            notes,
            "No chain dimensions",
        )
        persistence = optional(
            lambda: persistence_length(ensemble, path, stride=stride),
            notes,
            "No persistence length",
        )
    if not ensemble.is_snapshot:
        displacement = optional(
            lambda: centre_of_mass_msd(
                ensemble, max_lag_fraction=max_lag_fraction, stride=stride
            ),
            notes,
            "No mean-squared displacement",
        )
        if path is not None:
            relaxation = optional(
                lambda: end_to_end_relaxation(
                    ensemble, path, max_lag_fraction=max_lag_fraction
                ),
                notes,
                "No end-to-end relaxation",
            )

    return StructureReport(
        run_dir=str(directory),
        stage=files.stage,
        stage_source=stage_source,
        is_snapshot=ensemble.is_snapshot,
        n_frames=ensemble.n_frames,
        interval_ps=ensemble.interval_ps,
        n_chains=ensemble.n_chains,
        atoms_per_chain=ensemble.atoms_per_chain,
        stride=stride,
        backbone=path,
        backbone_source=backbone_source,
        backbone_file=backbone_file,
        distribution=distribution,
        structure=structure,
        conformation=conformation,
        persistence=persistence,
        displacement=displacement,
        relaxation=relaxation,
        recorded_chains=_recorded_chains(manifest),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _distribution_record(distribution: RadialDistribution) -> dict[str, Any]:
    """A pair distribution as plain JSON types."""
    return {
        "r_nm": distribution.r_nm.tolist(),
        "g_r": distribution.g_r.tolist(),
        "coordination_number": distribution.coordination_number.tolist(),
        "first_peak_nm": distribution.first_peak_nm,
        "first_peak_height": distribution.first_peak_height,
        "number_density_nm3": distribution.number_density_nm3,
        "r_max_nm": distribution.r_max_nm,
        "n_frames": distribution.n_frames,
        "n_pairs": distribution.n_pairs,
        "heavy_atoms_only": distribution.heavy_atoms_only,
    }


def _structure_factor_record(structure: StructureFactor) -> dict[str, Any]:
    """A structure factor as plain JSON types."""
    return {
        "q_per_nm": structure.q_per_nm.tolist(),
        "s_q": structure.s_q.tolist(),
        "n_vectors": structure.n_vectors.tolist(),
        "first_peak_per_nm": structure.first_peak_per_nm,
        "q_min_per_nm": structure.q_min_per_nm,
        "n_frames": structure.n_frames,
        "heavy_atoms_only": structure.heavy_atoms_only,
    }


def _chain_dimensions_record(dimensions: ChainDimensions) -> dict[str, Any]:
    """Chain dimensions as plain JSON types."""
    return {
        "mean_squared_end_to_end_nm2": dimensions.mean_squared_end_to_end_nm2,
        "mean_radius_of_gyration_nm": dimensions.mean_radius_of_gyration_nm,
        "ratio_of_squares": dimensions.ratio_of_squares,
        "characteristic_ratio": dimensions.characteristic_ratio,
        "expected_characteristic_ratio": dimensions.expected_characteristic_ratio,
        "consistent": dimensions.consistent,
    }


def _equilibration_record(settled: Equilibration) -> dict[str, Any]:
    """Where a series settled, as plain JSON types."""
    return {
        "start_index": settled.start_index,
        "start_ps": settled.start_ps,
        "n_samples": settled.n_samples,
        "n_independent_samples": settled.n_independent_samples,
        "correlation_time_ps": settled.correlation_time_ps,
        "relative_standard_error": settled.relative_standard_error,
        "relative_drift": settled.relative_drift,
        "equilibrated": settled.equilibrated,
    }


def _conformation_record(series: ConformationSeries) -> dict[str, Any]:
    """A conformation series as plain JSON types."""
    return {
        "stage": series.stage,
        "time_ps": series.time_ps.tolist(),
        "mean_squared_end_to_end_nm2": series.mean_squared_end_to_end_nm2.tolist(),
        "mean_radius_of_gyration_nm": series.mean_radius_of_gyration_nm.tolist(),
        "mean": _chain_dimensions_record(series.mean),
        "settled": (
            None if series.settled is None else _equilibration_record(series.settled)
        ),
        "n_chains": series.n_chains,
        "n_frames": series.n_frames,
    }


def _persistence_record(length: PersistenceLength) -> dict[str, Any]:
    """A persistence length as plain JSON types."""
    return {
        "separation": length.separation.tolist(),
        "correlation": length.correlation.tolist(),
        "bond_length_nm": length.bond_length_nm,
        "persistence_length_nm": length.persistence_length_nm,
        "n_bonds": length.n_bonds,
        "contour_length_nm": length.contour_length_nm,
        "decayed": length.decayed,
    }


def _displacement_record(msd: MeanSquaredDisplacement) -> dict[str, Any]:
    """A mean-squared displacement as plain JSON types."""
    return {
        "lag_ps": msd.lag_ps.tolist(),
        "msd_nm2": msd.msd_nm2.tolist(),
        "log_slope": msd.log_slope,
        "diffusion_coefficient_cm2_s": msd.diffusion_coefficient_cm2_s,
        "box_drift_fraction": msd.box_drift_fraction,
        "n_chains": msd.n_chains,
        "n_origins": msd.n_origins,
        "diffusive": msd.diffusive,
    }


def _relaxation_record(relaxation: EndToEndRelaxation) -> dict[str, Any]:
    """An end-to-end relaxation as plain JSON types."""
    return {
        "lag_ps": relaxation.lag_ps.tolist(),
        "correlation": relaxation.correlation.tolist(),
        "relaxation_time_ps": relaxation.relaxation_time_ps,
        "trajectory_ps": relaxation.trajectory_ps,
        "n_chains": relaxation.n_chains,
        "n_origins": relaxation.n_origins,
        "decorrelated": relaxation.decorrelated,
    }


def write_structure_report(
    report: StructureReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``structure.json`` and its figures into ``<run_dir>/analysis``.

    Or into *output_dir*, when the run directory should not be touched.
    """
    fields: dict[str, Any] = {
        "stage": report.stage,
        "stage_source": report.stage_source,
        "is_snapshot": report.is_snapshot,
        "n_frames": report.n_frames,
        "interval_ps": report.interval_ps,
        "n_chains": report.n_chains,
        "atoms_per_chain": report.atoms_per_chain,
        "stride": report.stride,
        "backbone": None if report.backbone is None else list(report.backbone),
        "backbone_source": report.backbone_source,
        "backbone_file": report.backbone_file,
        "radial_distribution": (
            None
            if report.distribution is None
            else _distribution_record(report.distribution)
        ),
        "structure_factor": (
            None
            if report.structure is None
            else _structure_factor_record(report.structure)
        ),
        "conformation": (
            None
            if report.conformation is None
            else _conformation_record(report.conformation)
        ),
        "persistence": (
            None
            if report.persistence is None
            else _persistence_record(report.persistence)
        ),
        "displacement": (
            None
            if report.displacement is None
            else _displacement_record(report.displacement)
        ),
        "relaxation": (
            None if report.relaxation is None else _relaxation_record(report.relaxation)
        ),
        "recorded_chains": (
            None
            if report.recorded_chains is None
            else _chain_dimensions_record(report.recorded_chains)
        ),
        "notes": list(report.notes),
    }
    return write_report_files(
        report.run_dir,
        output_dir,
        "structure.json",
        fields,
        _figures(report) if figures else (),
        figure_format,
    )


def _figures(report: StructureReport) -> Iterator[tuple[str, Figure]]:
    """A figure for each measurement that was made."""
    if report.distribution is not None:
        yield (
            "correlations",
            plot_correlations(report.distribution, structure=report.structure),
        )
    if report.conformation is not None:
        yield "conformation", plot_conformation(report.conformation)
    if report.persistence is not None:
        yield "persistence", plot_persistence(report.persistence)
    if report.displacement is not None:
        yield (
            "dynamics",
            plot_dynamics(report.displacement, relaxation=report.relaxation),
        )
