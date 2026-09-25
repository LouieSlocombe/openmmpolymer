"""How the cell is arranged, in real space and in reciprocal space.

Both measurements here need the minimum image convention, and this is the one
part of the analysis where the trajectory works against you: stages write
unwrapped coordinates on purpose, because wrapping splits a chain across a cell
face and a split chain has a meaningless radius of gyration. Everything
conformational wants that. A pair distribution does not - it needs each pair's
shortest separation through the periodic boundaries - so it applies the
convention itself, through MDAnalysis's distance kernels, which handle a
triclinic cell and build a neighbour list rather than an N-by-N matrix.

The pair distribution is deliberately **inter**molecular. A polymer's
intramolecular g(r) is dominated by its own bonded geometry, which is a
property of the force field and says nothing about the melt; the interesting
question is whether chains interpenetrated, and only pairs on different
molecules can answer it. That exclusion is one integer division, because a
chain is a contiguous block of atoms.

``r_max`` defaults to half the smallest cell edge and is refused above it. Past
that distance the minimum image convention is simply wrong - and MDAnalysis
applies it anyway, with no warning - so a g(r) plotted out to the cell edge
would show structure that is an artefact of the arithmetic.

The structure factor is summed directly on the wavevectors the cell can
actually hold, ``q = 2 pi n / L``. Fourier-transforming a g(r) truncated at
half the box would put ringing into the answer; summing on commensurate
wavevectors has no truncation to ring. The cost is a resolution floor, and the
result reports it rather than drawing a curve below it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._fitting import TINY
from ._validation import require_integer, require_positive
from .trajectory import AnalysisError, Ensemble, boxes_nm, chain_positions

log = logging.getLogger(__name__)

#: Wavevectors evaluated at once. The sum over atoms is a matrix product, and
#: the whole set of wavevectors against a large cell would not fit in memory,
#: so it goes in blocks.
_Q_CHUNK = 4096

#: Most wavevectors that will be summed before giving up. A big cell at a high
#: ``q_max`` runs to millions, and the answer is to ask for less rather than
#: to wait.
MAX_WAVEVECTORS = 2_000_000

#: Wavevectors a bin needs in each frame before its value may be called the
#: peak. The longest waves a cell holds come in threes and sixes - there are
#: only so many ways to make a short integer vector - and a sum over three of
#: them follows the cell's own periodicity, not the structure inside it. On a
#: real polyethylene melt those bins reach S(q) of twenty-six while the
#: amorphous halo, averaged over a couple of hundred wavevectors, sits near
#: three. Counted per frame, because averaging more frames of the same three
#: wavevectors does not make them any more than three. A crystal is the
#: exception, where a sharp Bragg peak genuinely has few wavevectors in it, so
#: this is a parameter rather than a rule.
MIN_WAVEVECTORS_PER_BIN = 20


@dataclass(frozen=True)
class RadialDistribution:
    """The intermolecular pair distribution function.

    Args:
        r_nm: Bin centres.
        g_r: The distribution. One means "as likely as in an ideal gas of the
            same density"; a melt that has interpenetrated goes to one by a
            nanometre or so.
        coordination_number: Running count of intermolecular neighbours within
            each radius, per atom. Read straight off the pair counts, so it
            does not depend on the normalisation being right.
        first_peak_nm: Where the distribution peaks.
        first_peak_height: How high.
        number_density_nm3: Atoms per cubic nanometre, averaged over frames.
        r_max_nm: How far out it was measured.
        n_frames: Frames averaged over. A g(r) from one frame and one from
            five hundred have the same shape and very different standing.
        n_pairs: Intermolecular pairs counted, over all frames.
        heavy_atoms_only: Whether hydrogens were dropped.
    """

    r_nm: npt.NDArray[np.float64]
    g_r: npt.NDArray[np.float64]
    coordination_number: npt.NDArray[np.float64]
    first_peak_nm: float
    first_peak_height: float
    number_density_nm3: float
    r_max_nm: float
    n_frames: int
    n_pairs: int
    heavy_atoms_only: bool


@dataclass(frozen=True)
class StructureFactor:
    """The static structure factor, on the cell's own wavevectors.

    Args:
        q_per_nm: Bin centres, in inverse nanometres.
        s_q: The structure factor, spherically averaged over every wavevector
            of that magnitude the cell holds.
        n_vectors: How many wavevectors fell in each bin, over all the frames.
        first_peak_per_nm: Where it peaks - the amorphous halo. Searched only
            over the bins :func:`peak_bins` allows, which a peak the cell can
            resolve and an average worth the name have to be in. Zero when no
            bin qualifies.
        q_min_per_nm: ``2 pi / L`` for the largest cell edge. Nothing below
            this is measurable in a cell this size, whatever the curve does,
            because the cell cannot hold a longer wave.
        n_frames: Frames averaged over.
        heavy_atoms_only: Whether hydrogens were dropped.
    """

    q_per_nm: npt.NDArray[np.float64]
    s_q: npt.NDArray[np.float64]
    n_vectors: npt.NDArray[np.int64]
    first_peak_per_nm: float
    q_min_per_nm: float
    n_frames: int
    heavy_atoms_only: bool


def radial_distribution(
    ensemble: Ensemble,
    *,
    r_max_nm: float | None = None,
    n_bins: int = 150,
    heavy_atoms_only: bool = True,
    stride: int = 1,
) -> RadialDistribution:
    """Measure the intermolecular pair distribution function.

    Args:
        ensemble: A snapshot or a trajectory.
        r_max_nm: How far out to measure. None means half the smallest cell
            edge, which is as far as the minimum image convention holds.
        n_bins: Bins between zero and *r_max_nm*.
        heavy_atoms_only: Drop hydrogens. On by default: a polymer melt's
            hydrogens trace the same structure as the carbons they hang off
            and quadruple the pair count.
        stride: Use every *stride*-th frame.

    Returns:
        The distribution, with the pair count behind it.

    Raises:
        AnalysisError: The cell has only one molecule, so there are no
            intermolecular pairs, or *r_max_nm* exceeds half the smallest cell
            edge.
        ValueError: *n_bins* or *stride* is not a positive integer.
    """
    require_integer(n_bins, name="n_bins")
    require_integer(stride, name="stride")
    if ensemble.n_chains < 2:
        raise AnalysisError(
            "A cell of one molecule has no intermolecular pairs. The "
            "intramolecular distribution is a property of the force field, "
            "not of a melt, so it is not what this measures."
        )

    positions, _ = chain_positions(
        ensemble, stride=stride, heavy_atoms_only=heavy_atoms_only
    )
    per_chain = positions.shape[2]
    if per_chain == 0:
        raise AnalysisError(
            "Dropping hydrogens left no atoms. Pass heavy_atoms_only=False."
        )
    boxes = boxes_nm(ensemble, stride=stride)
    limit = _pair_limit(boxes, r_max_nm)

    edges = np.linspace(0.0, limit, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.float64)
    densities: list[float] = []
    n_atoms = positions.shape[1] * per_chain
    for frame, box in zip(positions, boxes, strict=True):
        distances = _intermolecular_distances(
            frame.reshape(n_atoms, 3), per_chain, box, limit
        )
        counts += np.histogram(distances, bins=edges)[0]
        densities.append(n_atoms / float(box.prod()))

    n_frames = positions.shape[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    shells = 4.0 / 3.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = float(np.mean(densities))

    # The reference is an ideal gas of everything that is not in the atom's own
    # molecule, which is what makes this an intermolecular g(r) rather than a
    # total one scaled wrongly.
    other = n_atoms - per_chain
    ideal = 0.5 * n_atoms * other / float(np.mean([b.prod() for b in boxes])) * shells
    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = np.where(ideal > TINY, counts / n_frames / np.maximum(ideal, TINY), 0.0)

    # Read off the counts, not off g(r): the average number of other molecules'
    # atoms within a radius is a count, and needs no normalisation to be right.
    coordination = 2.0 * np.cumsum(counts) / n_frames / n_atoms
    peak = int(np.argmax(g_r))
    return RadialDistribution(
        r_nm=centres,
        g_r=g_r,
        coordination_number=coordination,
        first_peak_nm=float(centres[peak]),
        first_peak_height=float(g_r[peak]),
        number_density_nm3=density,
        r_max_nm=limit,
        n_frames=int(n_frames),
        n_pairs=int(counts.sum()),
        heavy_atoms_only=heavy_atoms_only,
    )


def structure_factor(
    ensemble: Ensemble,
    *,
    q_max_per_nm: float = 40.0,
    n_bins: int = 100,
    heavy_atoms_only: bool = True,
    stride: int = 8,
    min_vectors_per_bin: int = MIN_WAVEVECTORS_PER_BIN,
) -> StructureFactor:
    """Measure the static structure factor on commensurate wavevectors.

    Args:
        ensemble: A snapshot or a trajectory.
        q_max_per_nm: Highest wavevector magnitude to sum.
        n_bins: Bins to spherically average into.
        heavy_atoms_only: Drop hydrogens.
        stride: Use every *stride*-th frame. Higher than elsewhere by
            default, because this is the expensive one and neighbouring frames
            carry almost the same structure.
        min_vectors_per_bin: Wavevectors a bin needs in each frame before it
            may be called the peak. Lower it to one for a crystal, whose Bragg
            peaks are sharp and genuinely thinly populated; see
            :data:`MIN_WAVEVECTORS_PER_BIN`.

    Returns:
        The structure factor, and the resolution floor of the cell.

    Raises:
        AnalysisError: *q_max_per_nm* asks for more wavevectors than
            :data:`MAX_WAVEVECTORS`.
        ValueError: A numeric argument is out of range.
    """
    require_positive(q_max_per_nm, None, name="q_max_per_nm")
    require_integer(n_bins, name="n_bins")
    require_integer(stride, name="stride")

    positions, _ = chain_positions(
        ensemble, stride=stride, heavy_atoms_only=heavy_atoms_only
    )
    per_chain = positions.shape[2]
    if per_chain == 0:
        raise AnalysisError(
            "Dropping hydrogens left no atoms. Pass heavy_atoms_only=False."
        )
    boxes = boxes_nm(ensemble, stride=stride)
    n_atoms = positions.shape[1] * per_chain

    edges = np.linspace(0.0, q_max_per_nm, n_bins + 1)
    total = np.zeros(n_bins, dtype=np.float64)
    vectors = np.zeros(n_bins, dtype=np.int64)
    for frame, box in zip(positions, boxes, strict=True):
        wavevectors = _wavevectors(box, q_max_per_nm)
        intensity = _intensity(frame.reshape(n_atoms, 3), wavevectors)
        magnitudes = np.linalg.norm(wavevectors, axis=1)
        index = np.digitize(magnitudes, edges) - 1
        inside = (index >= 0) & (index < n_bins)
        np.add.at(total, index[inside], intensity[inside])
        np.add.at(vectors, index[inside], 1)

    counted = vectors > 0
    s_q = np.zeros(n_bins, dtype=np.float64)
    s_q[counted] = total[counted] / vectors[counted]
    centres = 0.5 * (edges[:-1] + edges[1:])
    floor = 2.0 * math.pi / float(boxes.max())
    n_frames = int(positions.shape[0])
    usable = peak_bins(centres, vectors, n_frames, floor, min_vectors_per_bin)
    return StructureFactor(
        q_per_nm=centres,
        s_q=s_q,
        n_vectors=vectors,
        first_peak_per_nm=_peak_above(centres, s_q, usable),
        q_min_per_nm=floor,
        n_frames=n_frames,
        heavy_atoms_only=heavy_atoms_only,
    )


def peak_bins(
    q_per_nm: npt.NDArray[np.float64],
    n_vectors: npt.NDArray[np.int64],
    n_frames: int,
    floor_per_nm: float,
    min_vectors_per_bin: int,
) -> npt.NDArray[np.bool_]:
    """The bins of a structure factor that can carry its peak.

    Two exclusions, and the second is the one that matters. Below
    *floor_per_nm*, ``2 pi / L``, the cell cannot hold a wave at all, so a
    peak there contradicts the floor reported beside it. And a bin fed by
    fewer than *min_vectors_per_bin* wavevectors a frame is not an average;
    :data:`MIN_WAVEVECTORS_PER_BIN` says why.
    """
    return np.asarray(
        (n_vectors / n_frames >= min_vectors_per_bin) & (q_per_nm >= floor_per_nm),
        dtype=np.bool_,
    )


def _peak_above(
    q_per_nm: npt.NDArray[np.float64],
    s_q: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
) -> float:
    """Where ``S(q)`` peaks over the *usable* bins, or zero if there are none."""
    if not usable.any():
        return 0.0
    within = np.where(usable, s_q, -np.inf)
    return float(q_per_nm[int(np.argmax(within))])


def _pair_limit(boxes: npt.NDArray[np.float64], r_max_nm: float | None) -> float:
    """Return the radius to measure to, refusing one the cell cannot support.

    Half the smallest edge over the whole trajectory, because the cell
    shrinks under a barostat and a limit that was legal in the first frame
    need not be in the last.
    """
    half = float(boxes.min()) / 2.0
    if r_max_nm is None:
        return half
    limit = require_positive(r_max_nm, None, name="r_max_nm")
    if limit > half:
        raise AnalysisError(
            f"r_max_nm={limit:.3f} is more than half the smallest cell edge "
            f"({2 * half:.3f} nm, so {half:.3f} nm). Past that the minimum "
            "image convention counts the same neighbour twice, and the "
            "distance kernel applies it anyway without complaining."
        )
    return limit


def _intermolecular_distances(
    frame: npt.NDArray[np.float64],
    atoms_per_chain: int,
    box_nm: npt.NDArray[np.float64],
    r_max_nm: float,
) -> npt.NDArray[np.float64]:
    """Distances of every pair within *r_max_nm* that spans two molecules.

    The kernel is handed nanometres and a box in nanometres. It is
    unit-agnostic - it only needs the two to agree - so converting to
    Angstrom and back would be two more places for a factor of ten to go
    missing.
    """
    from MDAnalysis.lib.distances import self_capped_distance

    coordinates = np.ascontiguousarray(frame, dtype=np.float32)
    box = np.array(
        [box_nm[0], box_nm[1], box_nm[2], 90.0, 90.0, 90.0], dtype=np.float32
    )
    # The kernel checks the cutoff against its float32 box, which can round
    # half an edge down past the float64 limit by one ulp and refuse it.
    cutoff = min(float(r_max_nm), float(box[:3].min()) / 2.0)
    found, measured = self_capped_distance(
        coordinates, max_cutoff=cutoff, box=box, return_distances=True
    )
    pairs = np.asarray(found, dtype=np.int64)
    if pairs.size == 0:
        return np.zeros(0, dtype=np.float64)
    # One integer division per atom says which molecule it belongs to, so the
    # intramolecular pairs drop out without any connectivity being consulted.
    between = np.asarray(
        pairs[:, 0] // atoms_per_chain != pairs[:, 1] // atoms_per_chain,
        dtype=np.bool_,
    )
    distances = np.asarray(measured, dtype=np.float64)
    return np.asarray(distances[between], dtype=np.float64)


def _wavevectors(
    box_nm: npt.NDArray[np.float64], q_max_per_nm: float
) -> npt.NDArray[np.float64]:
    """Every wavevector the cell holds up to *q_max_per_nm*, one per pair.

    Only half of reciprocal space is generated: the intensity is even in q,
    so summing both halves would do twice the work for the same answer.

    Raises:
        AnalysisError: There are more than :data:`MAX_WAVEVECTORS` of them.
    """
    spacing = 2.0 * math.pi / box_nm
    limits = np.floor(q_max_per_nm / spacing).astype(int)
    estimate = float(np.prod(2 * limits + 1))
    if estimate > 2 * MAX_WAVEVECTORS:
        raise AnalysisError(
            f"A {box_nm.max():.1f} nm cell holds about {estimate / 2:.3g} "
            f"wavevectors below {q_max_per_nm} /nm, over the "
            f"{MAX_WAVEVECTORS:.3g} this will sum. Lower q_max_per_nm."
        )
    ranges = [np.arange(-limit, limit + 1) for limit in limits]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    # Keep one of each +/- pair, and drop the origin.
    first = np.argmax(grid != 0, axis=1)
    leading = grid[np.arange(grid.shape[0]), first]
    grid = grid[(grid != 0).any(axis=1) & (leading > 0)]
    wavevectors = grid * spacing
    magnitudes = np.linalg.norm(wavevectors, axis=1)
    return np.asarray(wavevectors[magnitudes <= q_max_per_nm], dtype=np.float64)


def _intensity(
    frame: npt.NDArray[np.float64], wavevectors: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``|sum_j exp(i q . r_j)|^2 / N`` for each wavevector, in blocks."""
    n_atoms = frame.shape[0]
    out = np.empty(wavevectors.shape[0], dtype=np.float64)
    for start in range(0, wavevectors.shape[0], _Q_CHUNK):
        block = wavevectors[start : start + _Q_CHUNK]
        phase = block @ frame.T
        real = np.cos(phase).sum(axis=1)
        imaginary = np.sin(phase).sum(axis=1)
        out[start : start + block.shape[0]] = (
            real * real + imaginary * imaginary
        ) / n_atoms
    return out
