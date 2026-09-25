"""Apparent melting from a constant-pressure heating scan of a supplied crystal.

Melting is a discontinuity, unlike the continuous change of expansivity used
to find Tg. Both specific volume and enthalpy must support a positive jump in
the same temperature interval here. A resolved result is still an apparent
heating transition: superheating, finite size and crystal morphology can shift
it away from equilibrium Tm. Confirm crystal loss in the saved structures or
trajectories and repeat at longer holds before interpreting the number.

The crystal is supplied rather than built: chains packed from a monomer make
an amorphous melt, which has no melting point to find. :func:`load_crystal`
reads a prepared cell with its System, and :func:`run_tm_scan` heats it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import openmm as mm
from openmm import app, unit

from ._files import ReportFiles, file_sha256, write_json
from ._validation import require_integer
from ._workflow import (
    require_positive_fields,
    resume_chunks,
    run_fingerprint,
    spec_request,
)
from .forcefield import PolymerForceField
from .mdsystem import PackedBox, SystemSpec, ensemble_controls
from .protocols import (
    Protocol,
    RunManifest,
    RunSummary,
    Stage,
    run_protocol,
    validate_run_inputs,
)
from .reporters import TrajectoryOptions
from .simulate import RunContext, heating_temperatures, prepare_run, safe_timestep_fs
from .trajectory import AnalysisError

PROTOCOL_NAME = "tm_heating"
WORKFLOW_NAME = "tm_workflow.json"
HEAT_STEM = "02_heat"

_LIMITATIONS = (
    "This is an apparent heating transition, not equilibrium Tm; the crystal "
    "may superheat and the result depends on hold time, size and morphology.",
    "Density and enthalpy do not establish loss of crystalline order. Check "
    "the saved structures or trajectories and repeat with longer holds.",
)


class TmError(RuntimeError):
    """A melting scan lacks a crystal or conflicts with an existing run."""


@dataclass(frozen=True)
class TmSpec:
    """Heating settings; times are ps, temperatures K and pressure bar.

    The supplied crystal is minimised and equilibrated at ``t_start_k`` before
    heating. No melt preparation or cooling is performed. ``stage_ps`` limits
    the duration of each resumable heating chunk; a temperature hold is never
    split. ``trajectory_ps`` enables trajectories for structural confirmation.
    The first half of each hold's samples is discarded.
    """

    t_start_k: float = 250.0
    t_end_k: float = 650.0
    step_k: float = 10.0
    hold_ps: float = 1000.0
    equilibration_ps: float = 1000.0
    pressure_bar: float = 1.0
    stage_ps: float = 10_000.0
    samples_per_segment: int = 30
    min_points_per_branch: int = 3
    barostat: str = "anisotropic"
    trajectory_ps: float | None = None
    max_total_ns: float | None = None

    def __post_init__(self) -> None:
        """Reject invalid schedules before any files or dynamics are created."""
        require_positive_fields(
            self,
            (
                "t_start_k",
                "t_end_k",
                "step_k",
                "hold_ps",
                "equilibration_ps",
                "pressure_bar",
                "stage_ps",
            ),
            optional=("trajectory_ps", "max_total_ns"),
        )
        require_integer(self.samples_per_segment, minimum=4, name="samples_per_segment")
        require_integer(
            self.min_points_per_branch, minimum=3, name="min_points_per_branch"
        )
        if self.barostat not in ("isotropic", "anisotropic"):
            raise ValueError("barostat must be 'isotropic' or 'anisotropic'.")
        if self.stage_ps < self.hold_ps:
            raise ValueError("stage_ps must be at least hold_ps: a hold is not split.")
        if len(self.temperatures_k) < 2 * self.min_points_per_branch:
            raise ValueError(
                "The heating ladder needs at least twice min_points_per_branch "
                "temperatures; widen the range or reduce step_k."
            )

    @property
    def temperatures_k(self) -> tuple[float, ...]:
        """The exact ascending ladder, including the requested endpoint."""
        return tuple(heating_temperatures(self.t_start_k, self.t_end_k, self.step_k))


DEFAULT_SPEC = TmSpec()


@dataclass(frozen=True)
class HeatingCurve:
    """Per-hold means in heating order, with their pressure and duration.

    Enthalpy is in OpenMM's kJ/mol units for the entire simulation cell, not
    per monomer or per chain. It includes kinetic energy and applied P times
    instantaneous volume. All arrays must describe the same constant-pressure
    heating history.
    """

    temperature_k: tuple[float, ...]
    density_g_cm3: tuple[float, ...]
    enthalpy_kj_mol: tuple[float, ...]
    hold_ps: tuple[float, ...]
    pressure_bar: tuple[float, ...]
    stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Keep malformed or mixed heating histories out of the estimator."""
        size = len(self.temperature_k)
        if size < 2:
            raise ValueError("A heating curve needs at least two temperatures.")
        for name in (
            "temperature_k",
            "density_g_cm3",
            "enthalpy_kj_mol",
            "hold_ps",
            "pressure_bar",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must have {size} finite values.")
            if name != "enthalpy_kj_mol" and np.any(values <= 0):
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if np.any(np.diff(self.temperature_k) <= 0):
            raise ValueError("Heating temperatures must be strictly increasing.")
        if not np.allclose(self.pressure_bar, self.pressure_bar[0], rtol=1e-8):
            raise ValueError("A melting curve must be measured at constant pressure.")

    @property
    def n_points(self) -> int:
        """Number of temperatures visited."""
        return len(self.temperature_k)

    @property
    def specific_volume_cm3_g(self) -> npt.NDArray[np.float64]:
        """The reciprocal of the mean density at each hold."""
        return 1.0 / np.asarray(self.density_g_cm3, dtype=np.float64)

    @property
    def heating_rate_k_per_ns(self) -> float | None:
        """Nominal step/hold rate, or None for an irregular schedule.

        A shortened final temperature step is permitted. This is the nominal
        staircase rate, not a continuous ramp or a derivative at each hold.
        """
        steps = np.diff(self.temperature_k)
        if not np.allclose(self.hold_ps, self.hold_ps[0]):
            return None
        if not np.allclose(steps[:-1], steps[0]) or steps[-1] > steps[0] * (1 + 1e-8):
            return None
        return float(steps[0] / self.hold_ps[0] * 1000.0)


@dataclass(frozen=True)
class MeltingTransition:
    """A jump supported by both observables, or an unresolved result.

    ``bracket_k`` is the adjacent pair of sampled temperatures, not a
    confidence interval. Its midpoint is ``temperature_k``; no sub-grid
    precision is inferred. Jump sizes are extrapolated to that midpoint.
    """

    temperature_k: float | None
    bracket_k: tuple[float, float] | None
    resolved: bool
    volume_jump_cm3_g: float | None
    enthalpy_jump_kj_mol: float | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class MeltingReport:
    """Read-only analysis of one heating history."""

    run_dir: str
    curve: HeatingCurve
    transition: MeltingTransition
    notes: tuple[str, ...]

    @property
    def temperature_k(self) -> float | None:
        """Resolved apparent melting temperature, if any."""
        return self.transition.temperature_k

    @property
    def resolved(self) -> bool:
        """Whether both observables locate the same positive discontinuity."""
        return self.transition.resolved


@dataclass(frozen=True)
class TmResult:
    """The completed or resumed heating scan and its analysis."""

    manifest_path: str
    summary: RunSummary
    report: MeltingReport

    @property
    def temperature_k(self) -> float | None:
        """The apparent transition at the measured heating rate."""
        return self.report.temperature_k

    @property
    def bracket_k(self) -> tuple[float, float] | None:
        """The finite-temperature-grid bracket."""
        return self.report.transition.bracket_k

    @property
    def resolved(self) -> bool:
        """Whether the heating curve resolved a transition."""
        return self.report.resolved


def melting_scan(spec: TmSpec = DEFAULT_SPEC) -> Protocol:
    """Minimise a supplied crystal, settle it cold, then heat in NPT chunks."""
    common: dict[str, object] = {
        "pressure_bar": spec.pressure_bar,
        "barostat": spec.barostat,
        "report_interval_ps": min(10.0, spec.hold_ps / spec.samples_per_segment),
        "samples_per_segment": spec.samples_per_segment,
    }
    if spec.trajectory_ps is not None:
        common["trajectory"] = TrajectoryOptions("xtc", interval_ps=spec.trajectory_ps)
    stages = [
        Stage("00_minimise", "minimise", {"temperature_k": spec.t_start_k}),
        Stage(
            "01_crystal_npt",
            "npt",
            {
                **common,
                "temperature_k": spec.t_start_k,
                "duration_ps": spec.equilibration_ps,
            },
        ),
    ]
    ladder = spec.temperatures_k
    # Unlike a quench ladder's, a trailing chunk of one temperature is left
    # alone: heating stages are found by the enthalpy they record rather than
    # by their step, so a one-point chunk still reads as part of the history.
    for index, chunk in enumerate(
        resume_chunks(len(ladder), spec.hold_ps, spec.stage_ps)
    ):
        stages.append(
            Stage(
                f"{HEAT_STEM}_{index:02d}",
                "heat",
                {
                    **common,
                    "temperatures_k": ladder[chunk.start : chunk.stop],
                    "hold_ps": spec.hold_ps,
                },
            )
        )
    protocol = Protocol(PROTOCOL_NAME, tuple(stages))
    total_ns = protocol.total_duration_ps / 1000.0
    if spec.max_total_ns is not None and total_ns > spec.max_total_ns:
        raise TmError(
            f"Heating scan needs {total_ns:g} ns, above max_total_ns={spec.max_total_ns:g}."
        )
    return protocol


def heating_stages(run_dir: str | Path) -> tuple[str, ...]:
    """Find stages carrying heating enthalpy samples, including one-point chunks."""
    manifest = RunManifest.load(run_dir)
    if manifest is None:
        raise AnalysisError(f"No manifest in {run_dir}.")
    names = []
    for name, record in manifest.stages.items():
        samples = record.get("samples", {})
        if "segment_enthalpy_kj_mol" not in samples:
            continue
        try:
            temperatures = np.asarray(samples["segment_temperature_k"], dtype=float)
            if (
                temperatures.ndim == 1
                and len(temperatures) > 1
                and np.all(np.diff(temperatures) < 0)
            ):
                continue
        except (KeyError, TypeError, ValueError):
            # Keep malformed candidates visible so heating_curve names the
            # offending stage instead of silently shortening the history.
            pass
        names.append(name)
    return tuple(names)


def heating_curve(
    run_dir: str | Path,
    stages: str | Sequence[str] | None = None,
) -> HeatingCurve:
    """Read a recorded ascending history without modifying its manifest."""
    manifest = RunManifest.load(run_dir)
    if manifest is None:
        raise AnalysisError(f"No manifest in {run_dir}.")
    names = (
        heating_stages(run_dir)
        if stages is None
        else ((stages,) if isinstance(stages, str) else tuple(stages))
    )
    if not names:
        raise AnalysisError(f"No heating stages with enthalpy samples in {run_dir}.")
    fields = (
        "segment_temperature_k",
        "segment_density_g_cm3",
        "segment_enthalpy_kj_mol",
        "segment_duration_ps",
        "segment_pressure_bar",
    )
    columns: list[list[float]] = [[] for _ in fields]
    for name in names:
        try:
            samples = manifest.stages[name]["samples"]
            size = len(samples[fields[0]])
            if size == 0:
                raise ValueError("empty heating stage")
            for column, field in zip(columns, fields, strict=True):
                values = samples[field]
                if len(values) != size:
                    raise ValueError(f"{field} has a different sample count")
                column.extend(float(value) for value in values)
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisError(
                f"Malformed heating samples in stage {name!r}: {error}"
            ) from error
    try:
        return HeatingCurve(
            temperature_k=tuple(columns[0]),
            density_g_cm3=tuple(columns[1]),
            enthalpy_kj_mol=tuple(columns[2]),
            hold_ps=tuple(columns[3]),
            pressure_bar=tuple(columns[4]),
            stages=tuple(names),
        )
    except ValueError as error:
        raise AnalysisError(f"Invalid heating curve in {run_dir}: {error}") from error


@dataclass(frozen=True)
class _Jump:
    index: int
    jump: float
    supported: bool
    multiple: bool


def _jump_fit(
    temperature: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    minimum: int,
) -> _Jump:
    """Compare a discontinuous pair of lines against smooth alternatives.

    The BIC difference charges the discontinuous model an extra parameter;
    ten is required as well as a jump exceeding three residual standard errors.
    Those errors describe the fitted curve, not independent MD sample errors.
    A continuous hinge and cubic guard against Tg and broad curvature. A
    three-branch alternative checks whether one transition is sufficient.
    """
    n = len(temperature)
    # Centre and scale both axes to keep whole-cell energies well conditioned.
    x = (temperature - temperature[0]) / np.ptp(temperature)
    scale = max(float(np.ptp(values)), float(np.max(np.abs(values))) * 1e-12, 1e-12)
    y = (values - values[0]) / scale
    candidates: list[tuple[float, int, float, float]] = []
    knots = list(x) + list((x[:-1] + x[1:]) / 2)
    for index in range(minimum, n - minimum + 1):
        midpoint = (x[index - 1] + x[index]) / 2
        dx = x - midpoint
        right = (np.arange(n) >= index).astype(float)
        design = np.column_stack((np.ones(n), dx, right, right * dx))
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coef
        sse = float(residual @ residual)
        variance = max(0.0, sse / (n - 4))
        error = float(np.sqrt(variance * np.linalg.inv(design.T @ design)[2, 2]))
        candidates.append((sse, index, float(coef[2]), error))
        # Include the fitted branches' intersection: a true continuous Tg
        # between sampled temperatures must not look like a latent jump.
        if abs(coef[3]) > 1e-12:
            intersection = float(midpoint - coef[2] / coef[3])
            if x[0] < intersection < x[-1]:
                knots.append(intersection)
    sse, index, jump, error = min(candidates)
    # A smooth cubic is also a four-parameter null: broad curved expansion
    # must not be reduced to a narrow temperature-grid melting bracket.
    cubic = np.column_stack((np.ones(n), x, x**2, x**3))
    coef, _, _, _ = np.linalg.lstsq(cubic, y, rcond=None)
    residual = y - cubic @ coef
    null_sse = float(residual @ residual)
    for knot in knots:
        design = np.column_stack((np.ones(n), x, np.maximum(x - knot, 0)))
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coef
        null_sse = min(null_sse, float(residual @ residual))
    floor = 1e-24 * n
    bic_gain = n * np.log(max(null_sse, floor) / max(sse, floor)) - np.log(n)
    supported = jump > max(3 * error, 1e-8) and bic_gain >= 10.0
    multiple = False
    if supported:
        # Three independent lines need six coefficients and two split
        # positions, against four coefficients and one position above. If
        # the extra transition clearly improves BIC, one bracket is misleading.
        for first in range(minimum, n - 2 * minimum + 1):
            for second in range(first + minimum, n - minimum + 1):
                after_first = (np.arange(n) >= first).astype(float)
                after_second = (np.arange(n) >= second).astype(float)
                design = np.column_stack(
                    (
                        np.ones(n),
                        x,
                        after_first,
                        after_first * x,
                        after_second,
                        after_second * x,
                    )
                )
                coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
                residual = y - design @ coef
                two_sse = float(residual @ residual)
                gain = n * np.log(max(sse, floor) / max(two_sse, floor)) - 3 * np.log(n)
                if gain >= 10.0:
                    multiple = True
                    break
            if multiple:
                break
    return _Jump(index, jump * scale, bool(supported and not multiple), multiple)


def melting_temperature(
    curve: HeatingCurve,
    *,
    min_points_per_branch: int = 3,
) -> MeltingTransition:
    """Locate coincident positive volume and enthalpy jumps, or report none.

    A continuous two-branch curve or smooth cubic is the null hypothesis.
    Each observable independently chooses its best split; they must select
    the same interval and both pass the evidence threshold. Curves better
    explained by three branches are left unresolved as ambiguous.
    This deliberately leaves weak or inconsistent curves unresolved.
    """
    require_integer(min_points_per_branch, minimum=3, name="min_points_per_branch")
    if curve.n_points < 2 * min_points_per_branch:
        return MeltingTransition(
            None,
            None,
            False,
            None,
            None,
            (
                "Too few temperatures for both branches; extend or refine the heating ladder.",
            ),
        )
    temperature = np.asarray(curve.temperature_k, dtype=np.float64)
    volume = _jump_fit(temperature, curve.specific_volume_cm3_g, min_points_per_branch)
    enthalpy = _jump_fit(
        temperature, np.asarray(curve.enthalpy_kj_mol), min_points_per_branch
    )
    notes = []
    if volume.multiple or enthalpy.multiple:
        notes.append(
            "More than one change is supported; a single melting bracket is ambiguous."
        )
    if not volume.supported:
        notes.append(
            "Specific volume did not resolve a positive discontinuity over a continuous slope change."
        )
    if not enthalpy.supported:
        notes.append(
            "Enthalpy did not resolve a positive discontinuity over a continuous slope change."
        )
    if volume.index != enthalpy.index:
        notes.append(
            "Volume and enthalpy locate different intervals; the transition is ambiguous."
        )
    if notes:
        return MeltingTransition(
            None, None, False, volume.jump, enthalpy.jump, tuple(notes)
        )
    bracket = (float(temperature[volume.index - 1]), float(temperature[volume.index]))
    return MeltingTransition(
        sum(bracket) / 2,
        bracket,
        True,
        volume.jump,
        enthalpy.jump,
        ("The bracket is the temperature-grid resolution, not a confidence interval.",),
    )


def analyse_melting(
    run_dir: str | Path,
    *,
    min_points_per_branch: int = 3,
) -> MeltingReport:
    """Analyse a saved heating scan; no files are written."""
    curve = heating_curve(run_dir)
    transition = melting_temperature(curve, min_points_per_branch=min_points_per_branch)
    return MeltingReport(
        str(run_dir), curve, transition, _LIMITATIONS + transition.notes
    )


def load_crystal(
    crystal_pdb: str | Path,
    system_xml: str | Path,
    *,
    state_in: str | Path | None = None,
    platform: str | None = None,
    seed: int = 0xF0,
) -> RunContext:
    """Read a prepared crystal and its System, ready for :func:`run_tm_scan`.

    Nothing is repacked or reparameterised: the System keeps its own
    parameters, masses and constraints, and the force field the run records is
    provenance only. What is checked is everything a heating scan would
    otherwise trip over later, before it writes a file: the PDB and the System
    hold the same atoms, in a periodic cell of positive volume with finite
    coordinates; the System brings no barostat or Andersen thermostat of its
    own; and its bonded molecules are equal, contiguous blocks of atoms, which
    is how the reports divide a cell into chains.

    Args:
        crystal_pdb: The crystalline or semicrystalline cell, with its box
            vectors (CRYST1) and bonds, in the System's atom order.
        system_xml: A serialized OpenMM System for exactly that cell.
        state_in: A serialized State of the cell for the scan to start from,
            checked here; the scan is handed it separately.
        platform: OpenMM platform, or None for the fastest available.
        seed: Master seed.

    Returns:
        The run context.

    Raises:
        ValueError: A file cannot be read, or does not describe such a cell.
    """
    try:
        pdb = app.PDBFile(str(crystal_pdb))
        system = mm.XmlSerializer.deserialize(Path(system_xml).read_text())
    except Exception as error:
        raise ValueError(
            f"Could not read the prepared crystal and System: {error}"
        ) from error
    if not isinstance(system, mm.System):
        raise ValueError("system_xml must contain a serialized OpenMM System")
    if system.getNumParticles() != pdb.topology.getNumAtoms():
        raise ValueError("Crystal PDB and System must contain the same number of atoms")
    _refuse_ensemble_controls(system, ValueError)
    if not system.usesPeriodicBoundaryConditions():
        raise ValueError("The supplied System must use periodic boundary conditions")
    vectors = pdb.topology.getPeriodicBoxVectors()
    if vectors is None:
        raise ValueError("Crystal PDB must contain periodic box vectors (CRYST1)")
    matrix = np.asarray(vectors.value_in_unit(unit.nanometer), dtype=float)
    if not np.all(np.isfinite(matrix)) or np.linalg.det(matrix) <= 0:
        raise ValueError("Crystal PDB must have finite, positive-volume box vectors")
    positions = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    if not np.all(np.isfinite(positions)):
        raise ValueError("Crystal PDB coordinates must be finite")
    system.setDefaultPeriodicBoxVectors(*vectors)
    if state_in is not None:
        try:
            state = mm.XmlSerializer.deserialize(Path(state_in).read_text())
            if not isinstance(state, mm.State):
                raise ValueError("state_in must contain a serialized OpenMM State")
            saved_positions = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=float,
            )
            saved_vectors = np.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=float,
            )
        except Exception as error:
            raise ValueError(
                f"Could not read the crystalline starting State: {error}"
            ) from error
        if saved_positions.shape != positions.shape or not np.all(
            np.isfinite(saved_positions)
        ):
            raise ValueError(
                "Starting State must have finite positions for every crystal atom"
            )
        if not np.all(np.isfinite(saved_vectors)) or np.linalg.det(saved_vectors) <= 0:
            raise ValueError(
                "Starting State must have finite, positive-volume box vectors"
            )
    a, b, c = np.linalg.norm(matrix, axis=1)
    box = PackedBox(
        topology=pdb.topology,
        positions_nm=positions,
        box_nm=(float(a), float(b), float(c)),
        n_molecules=_molecule_count(pdb.topology),
    )
    forcefield = PolymerForceField(
        forcefield_xml=str(Path(system_xml).resolve()),
        base_forcefield=(),
        residue_name="",
        backend="prepared-system",
    )
    return prepare_run(
        box,
        forcefield,
        SystemSpec(constraints="none", hydrogen_mass_amu=None),
        platform=platform,
        seed=seed,
        system=system,
    )


def _refuse_ensemble_controls(system: Any, error: type[Exception]) -> None:
    """Refuse a System that brings its own barostat or Andersen thermostat.

    The heating stages control temperature and pressure themselves, and
    OpenMM applies every barostat a System carries, a second one included.
    """
    controls = ensemble_controls(system)
    if controls:
        raise error(
            "The supplied System must contain no barostat or Andersen thermostat; "
            "the heating stages provide their own temperature and pressure "
            f"control. It carries {', '.join(controls)}."
        )


def _molecule_count(topology: Any) -> int:
    """Count the bonded molecules, which the reports take to be equal blocks."""
    n_atoms = topology.getNumAtoms()
    if n_atoms == 0:
        raise ValueError("Crystal PDB must contain atoms")
    neighbours: list[list[int]] = [[] for _ in range(n_atoms)]
    for left, right in topology.bonds():
        neighbours[left.index].append(right.index)
        neighbours[right.index].append(left.index)
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        visited.add(start)
        pending = [start]
        component = []
        while pending:
            atom = pending.pop()
            component.append(atom)
            for neighbour in neighbours[atom]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        components.append(component)
    if any(len(component) != len(components[0]) for component in components):
        raise ValueError(
            "Crystal PDB must contain molecules with equal atom counts; "
            "check its bonds/CONECT records against the System"
        )
    if any(
        max(component) - min(component) + 1 != len(component)
        for component in components
    ):
        raise ValueError(
            "Crystal PDB molecules must occupy contiguous atom blocks; "
            "reorder both the PDB and System consistently"
        )
    return len(components)


def run_tm_scan(
    run: RunContext,
    run_dir: str | Path = "run",
    *,
    spec: TmSpec = DEFAULT_SPEC,
    state_in: str | Path | None = None,
    crystalline: bool = False,
    resume: bool = True,
    chain_backbone: Sequence[int] | None = None,
    atoms_per_chain: int | None = None,
) -> TmResult:
    """Heat a supplied crystal and estimate its apparent melting interval.

    ``crystalline=True`` declares that the supplied coordinates (or
    ``state_in``) contain crystalline order. It is a caller assertion, not an
    automated crystallinity measurement. Packed amorphous chains and quenched
    glasses are not valid substitutes. The workflow records input fingerprints
    and settings so resume cannot silently mix different heating histories.
    """
    if not crystalline:
        raise TmError(
            "A melting scan requires a supplied crystalline or semicrystalline "
            "cell. Pass crystalline=True only after preparing that structure."
        )
    _refuse_ensemble_controls(mm.XmlSerializer.deserialize(run.system_xml), TmError)
    protocol = melting_scan(spec)
    total_ns = protocol.total_duration_ps / 1000.0
    timestep = safe_timestep_fs(spec.t_end_k, run.spec)
    protocol = replace(
        protocol,
        stages=tuple(
            stage
            if stage.kind == "minimise"
            else replace(stage, options={**stage.options, "timestep_fs": timestep})
            for stage in protocol.stages
        ),
    )
    request = spec_request(
        spec,
        drop=("max_total_ns",),
        **run_fingerprint(run, spec_key="system_spec"),
        state_sha256=None if state_in is None else file_sha256(state_in),
    )
    directory = Path(run_dir)
    path = directory / WORKFLOW_NAME
    manifest = RunManifest.load(directory)
    if manifest is not None and manifest.protocol != PROTOCOL_NAME:
        raise TmError(
            f"{directory} already contains a different protocol; use a new run directory."
        )
    if resume and manifest is not None and not path.is_file():
        raise TmError(
            "The melting workflow record is missing; use a new run directory."
        )
    if resume and path.is_file():
        previous = json.loads(path.read_text())
        if previous.get("request") != request:
            raise TmError(
                "Melting settings or starting inputs changed; use a new run directory."
            )
    if resume:
        validate_run_inputs(run, directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "request": request,
            "crystalline_supplied": True,
            "method": "apparent melting from NPT heating",
            "total_ns": total_ns,
        },
    )
    summary = run_protocol(
        protocol,
        run,
        directory,
        resume=resume,
        state_in=state_in,
        chain_backbone=chain_backbone,
        atoms_per_chain=atoms_per_chain,
    )
    report = analyse_melting(
        directory, min_points_per_branch=spec.min_points_per_branch
    )
    return TmResult(summary.manifest_path, summary, report)


def write_melting_report(
    report: MeltingReport,
    output_dir: str | Path | None = None,
    *,
    figures: bool = True,
    figure_format: str = "png",
) -> ReportFiles:
    """Write ``tm.json`` and a paired volume/enthalpy plot, outside the manifest."""
    from . import __version__

    directory = (
        Path(output_dir)
        if output_dir is not None
        else Path(report.run_dir) / "analysis"
    )
    if figure_format not in ("png", "pdf", "svg"):
        raise ValueError("figure_format must be png, pdf or svg.")
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "openmmpolymer": __version__,
        "method": "apparent melting from NPT heating",
        **asdict(report),
        "temperature_k": report.temperature_k,
        "resolved": report.resolved,
        "heating_rate_k_per_ns": report.curve.heating_rate_k_per_ns,
        "enthalpy_units": "kJ/mol of simulation cells",
    }
    path = directory / "tm.json"
    write_json(path, record)
    paths: list[str] = []
    if figures:
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7, 7), layout="constrained")
        axes = figure.subplots(2, 1, sharex=True)
        curve = report.curve
        axes[0].plot(curve.temperature_k, curve.specific_volume_cm3_g, "o-")
        axes[1].plot(curve.temperature_k, curve.enthalpy_kj_mol, "o-")
        axes[0].set_ylabel("Specific volume (cm³/g)")
        axes[1].set_ylabel("Enthalpy (kJ/mol of cells)")
        axes[1].set_xlabel("Temperature (K)")
        for axis in axes:
            if report.transition.bracket_k is not None:
                axis.axvspan(*report.transition.bracket_k, alpha=0.2, color="tab:red")
        rate = curve.heating_rate_k_per_ns
        label = "irregular heating schedule" if rate is None else f"{rate:g} K/ns"
        figure.suptitle(
            f"Apparent melting scan — {label}, {curve.pressure_bar[0]:g} bar"
        )
        figure_path = directory / f"melting.{figure_format}"
        figure.savefig(figure_path)
        paths.append(str(figure_path))
    return ReportFiles(str(path), tuple(paths))
