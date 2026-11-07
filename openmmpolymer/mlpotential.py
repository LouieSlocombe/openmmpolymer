"""Running a cell under a MACE potential evaluated by TurboMACE.

A machine-learned potential replaces the force field's interactions, not the
rest of the System. :func:`apply_turbomace` keeps a System's particles,
masses (hydrogen repartitioning included), constraints and centre-of-mass
remover, removes every interaction force, and adds one ``openmm.PythonForce``
whose computation evaluates a TurboMACE model on the whole periodic cell.
Every stage runner, barostat and stress reading then works unchanged, because
each of them only ever deserialises the System and adds to it.

Three things make that hold:

- ``PythonForce`` pickles its computation into the System XML, so the
  computation is a small top-level class carrying the model's path, its
  SHA-256, the precision and the atomic numbers - never the loaded model. It
  loads the model the first time it is called and checks the file still has
  the digest it was built with, so a replaced model file cannot quietly
  change a run's Hamiltonian. Because the XML is a run's fingerprint, that
  class's module, name and arguments are an on-disk format.
- A zero-stiffness ``HarmonicBondForce`` over the topology's bonds keeps each
  chain one OpenMM molecule. It contributes nothing, but it is how OpenMM
  decides what a molecule is, and the barostats' rigid-molecule moves, the
  molecular virial behind :mod:`openmmpolymer.stress` and the stages' check
  that the cell holds as many molecules as chains all rest on that.
- The device is chosen when the model is loaded, not recorded: CUDA if this
  TurboMACE build has it and a GPU is visible, else the CPU, like the OpenMM
  platform. ``CUDA_VISIBLE_DEVICES=`` forces the CPU for both.

The model is evaluated on TurboMACE's own neighbour graph rather than through
its ASE calculator: the same engine call, without the calculator's per-call
bookkeeping, which measured at a sixth of the evaluation on a 3100-atom
polyethylene cell. Models that take a total charge and spin read the neutral
singlet TurboMACE defaults to; polar models, which need long-range
electrostatics of the whole cell, are refused.

Constraint lengths stay the force field's equilibrium bond lengths, so with
the default ``constraints="hbonds"`` MACE sets every geometry but the X-H
bond lengths. Energies are MACE's absolute energies, including the atomic
reference energies, so a reported potential energy is large and negative; only
its differences mean anything.
"""

from __future__ import annotations

import atexit
import functools
import logging
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import openmm as mm
from openmm import unit

from ._files import file_sha256
from ._validation import require_choice
from .mdsystem import require_no_ensemble_controls

if TYPE_CHECKING:
    from .simulate import RunContext

log = logging.getLogger(__name__)

#: TurboMACE engine precisions. fp32 is the default because consumer GPUs run
#: double precision at a small fraction of their single-precision rate.
TURBOMACE_PRECISIONS = ("fp32", "fp64")

#: kJ/mol in one eV: the elementary charge times Avogadro's number, over 1000.
KJ_PER_MOL_PER_EV = 96.48533212331002

#: Angstrom in one nanometre.
ANGSTROM_PER_NM = 10.0

#: The Verlet skin, in Angstrom, past the model cutoff that the neighbour list
#: is built to. TurboMACE reuses a list only while no pair can have entered the
#: cutoff, so this changes how often the list is rebuilt and never the forces.
NEIGHBOUR_SKIN_ANGSTROM = 0.5

#: The names the two forces apply_turbomace adds go by.
TURBOMACE_FORCE_NAME = "TurboMACE"
MOLECULE_BONDS_NAME = "TurboMACEMoleculeBonds"

#: A model evaluation: positions and box rows in Angstrom in, energy in eV
#: and forces in eV/Angstrom out.
Evaluator = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[float, npt.NDArray[np.float64]],
]


class TurboMACEError(RuntimeError):
    """TurboMACE is missing, or the model cannot describe this cell."""


@dataclass(frozen=True)
class TurboMACESpec:
    """Which TurboMACE model a cell runs under, and how it is evaluated.

    Args:
        model_path: A ``.tmace`` file, converted from a mace-torch model with
            ``turbomace-export``. MACE-OFF23 covers H, C, N, O, F, P, S, Cl,
            Br and I, which is most of polymer chemistry.
        precision: One of :data:`TURBOMACE_PRECISIONS`. Positions and outputs
            are double precision either way; fp32 evaluates the network in
            single precision.
    """

    model_path: str | Path
    precision: str = "fp32"

    def __post_init__(self) -> None:
        """Refuse a missing model or an unknown precision before anything is built."""
        require_choice(self.precision, TURBOMACE_PRECISIONS, name="precision")
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"TurboMACE model {str(self.model_path)!r} does not exist. Convert "
                "a mace-torch model with turbomace-export first."
            )


class TurboMACEComputation:
    """The computation a System's ``PythonForce`` calls: one model evaluation.

    It is pickled into the System XML, which is a run's fingerprint, so its
    module, name and constructor arguments are an on-disk format: renaming any
    of them stops every run made under TurboMACE from deserialising.

    Args:
        model_path: The absolute path of the ``.tmace`` file.
        model_sha256: The digest that file had when the System was built.
        precision: One of :data:`TURBOMACE_PRECISIONS`.
        atomic_numbers: One byte per particle, in System order.
    """

    def __init__(
        self,
        model_path: str,
        model_sha256: str,
        precision: str,
        atomic_numbers: bytes,
    ) -> None:
        self.model_path = model_path
        self.model_sha256 = model_sha256
        self.precision = precision
        self.atomic_numbers = atomic_numbers
        self._evaluate: Evaluator | None = None

    def __reduce__(
        self,
    ) -> tuple[type[TurboMACEComputation], tuple[str, str, str, bytes]]:
        """Pickle the description, never the loaded model."""
        return (
            type(self),
            (self.model_path, self.model_sha256, self.precision, self.atomic_numbers),
        )

    def __call__(self, state: Any) -> tuple[float, npt.NDArray[np.float64]]:
        """Return the energy in kJ/mol and the forces in kJ/mol/nm for *state*."""
        if self._evaluate is None:
            self._evaluate = _load_evaluator(
                self.model_path,
                self.model_sha256,
                self.precision,
                list(self.atomic_numbers),
            )
            _LOADED.add(self)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        energy, forces = self._evaluate(
            np.asarray(positions, dtype=np.float64), np.asarray(box, dtype=np.float64)
        )
        return (
            energy * KJ_PER_MOL_PER_EV,
            np.asarray(forces, dtype=np.float64)
            * (KJ_PER_MOL_PER_EV * ANGSTROM_PER_NM),
        )


def apply_turbomace(system: Any, topology: Any, spec: TurboMACESpec) -> Any:
    """Return a copy of *system* whose interactions are the TurboMACE model.

    Args:
        system: A System built for *topology*, with no barostat or Andersen
            thermostat: the stages add their own.
        topology: Its topology, for the elements and the bonds.
        spec: The model.

    Returns:
        A new System with the same particles, masses, constraints, box and
        centre-of-mass remover, every interaction force replaced by the model.

    Raises:
        ValueError: The System carries a barostat or thermostat, has virtual
            sites, or does not match the topology.
        TurboMACEError: TurboMACE is not installed, or the model is polar or
            does not cover every element in the cell.
    """
    require_no_ensemble_controls(system, ValueError, stages="TurboMACE")
    numbers = _atomic_numbers(system, topology)
    path = str(Path(spec.model_path).resolve())
    _check_model(path, numbers)

    result = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    for index in reversed(range(result.getNumForces())):
        if not isinstance(result.getForce(index), mm.CMMotionRemover):
            result.removeForce(index)

    bonds = mm.HarmonicBondForce()
    bonds.setName(MOLECULE_BONDS_NAME)
    for bond in topology.bonds():
        bonds.addBond(bond[0].index, bond[1].index, 0.0, 0.0)
    result.addForce(bonds)

    force = mm.PythonForce(
        TurboMACEComputation(path, file_sha256(path), spec.precision, bytes(numbers))
    )
    force.setName(TURBOMACE_FORCE_NAME)
    force.setUsesPeriodicBoundaryConditions(True)
    result.addForce(force)
    log.info(
        "Replaced the force field's interactions with TurboMACE model %s (%s).",
        path,
        spec.precision,
    )
    return result


def use_turbomace(run: RunContext, spec: TurboMACESpec) -> RunContext:
    """Return *run* with its System's interactions replaced by the model.

    The cell, force field, settings and seed stay as they were, so a state
    saved by a classical run can start a TurboMACE one: equilibrate cheaply
    with the force field, then pass the classical run's final state as
    ``state_in`` to a protocol run with the context this returns. That run
    goes in its own directory, because its System - and so its fingerprint -
    is not the classical one.

    Args:
        run: A run context, from :func:`~openmmpolymer.build_melt` or
            :func:`~openmmpolymer.prepare_run`.
        spec: The model.

    Returns:
        The run context for running under TurboMACE.
    """
    system = mm.XmlSerializer.deserialize(run.system_xml)
    converted = apply_turbomace(system, run.box.topology, spec)
    return replace(run, system_xml=mm.XmlSerializer.serialize(converted))


def _atomic_numbers(system: Any, topology: Any) -> list[int]:
    """Each particle's atomic number, refusing anything MACE cannot describe."""
    atoms = list(topology.atoms())
    if len(atoms) != system.getNumParticles():
        raise ValueError(
            f"The topology has {len(atoms)} atoms but the System has "
            f"{system.getNumParticles()} particles, so the two do not describe "
            "the same cell."
        )
    if any(system.isVirtualSite(index) for index in range(system.getNumParticles())):
        raise ValueError(
            "The System has virtual sites, which a MACE model does not describe. "
            "Build it from a force field without them."
        )
    unknown = [atom.index for atom in atoms if atom.element is None]
    if unknown:
        raise ValueError(
            f"Atoms {unknown[:5]} have no element, so the model cannot be given "
            "their atomic numbers."
        )
    return [atom.element.atomic_number for atom in atoms]


#: What to do about a missing TurboMACE.
_INSTALL = (
    "TurboMACE is not installed. Build it from "
    "https://github.com/SMTG-Bham/TurboMACE with its ase extra: "
    "pip install '.[ase]' in a TurboMACE checkout."
)


def turbomace_version() -> str:
    """The installed TurboMACE's version, which a run's request records.

    Raises:
        TurboMACEError: TurboMACE is not installed.
    """
    try:
        return version("turbomace")
    except PackageNotFoundError as error:
        raise TurboMACEError(_INSTALL) from error


def _import_turbomace() -> Any:
    """Import TurboMACE, saying how to get it if it is missing."""
    try:
        import turbomace
    except ImportError as error:
        raise TurboMACEError(_INSTALL) from error
    return turbomace


def _check_model(path: str, atomic_numbers: Sequence[int]) -> None:
    """Refuse a model that cannot describe a cell of these elements on its own.

    Read from the file's metadata, so a build fails here rather than at the
    first step of the first stage.
    """
    turbomace = _import_turbomace()
    from turbomace.polar import is_polar

    program = turbomace.Program.load(path)
    if is_polar(program):
        raise TurboMACEError(
            f"{path} is a polar model. Its long-range electrostatics need the "
            "cell's total charge and spin, which this package does not pass; "
            "use a short-range model such as MACE-OFF."
        )
    covered = {int(number) for number in program.meta["atomic_numbers"]}
    missing = sorted(set(atomic_numbers) - covered)
    if missing:
        raise TurboMACEError(
            f"{path} does not cover atomic numbers {missing}, which this cell "
            "contains. Choose a model trained on every element of the polymer."
        )


def _device() -> str:
    """CUDA if this TurboMACE build has it and a GPU is visible, else the CPU."""
    from turbomace import _core

    if not getattr(_core, "has_cuda", False):
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.cache
def _loaded_model(path: str, sha256: str, precision: str) -> Any:
    """Load a model once per process; the digest keys out a replaced file."""
    turbomace = _import_turbomace()
    device = _device()
    log.info("Loading TurboMACE model %s on %s (%s).", path, device, precision)
    return turbomace.load(path, device=device, precision=precision)


#: Every computation that has loaded a model, for _release_models.
_LOADED: weakref.WeakSet[TurboMACEComputation] = weakref.WeakSet()


@atexit.register
def _release_models() -> None:
    """Let every loaded model go before the interpreter tears TurboMACE down.

    OpenMM keeps a ``PythonForce``'s computation alive after the stage that
    used it has finished, and a model still alive when TurboMACE's extension
    module is torn down makes nanobind print a page of leak warnings on every
    exit.
    """
    for computation in list(_LOADED):
        computation._evaluate = None
    _loaded_model.cache_clear()


def _load_evaluator(
    path: str, sha256: str, precision: str, atomic_numbers: Sequence[int]
) -> Evaluator:
    """Check the model file is the one the System was built with, then load it.

    Raises:
        TurboMACEError: The file is missing or its contents changed.
    """
    if not Path(path).is_file():
        raise TurboMACEError(
            f"TurboMACE model {path} is missing. It was recorded in the System "
            "when the run was built; restore it to resume the run."
        )
    found = file_sha256(path)
    if found != sha256:
        raise TurboMACEError(
            f"TurboMACE model {path} changed since the run was built "
            f"(SHA-256 {found}, recorded {sha256}). A different model is a "
            "different Hamiltonian; restore the original or use a fresh run "
            "directory."
        )
    return _GraphEvaluator(_loaded_model(path, sha256, precision), atomic_numbers)


class _GraphEvaluator:
    """Evaluate a loaded model on TurboMACE's reusable neighbour graph.

    The graph builder keeps one neighbour list while no pair can have entered
    the cutoff, including across barostat moves, so it is rebuilt only every
    few dozen steps. The model's per-system inputs, if it takes any, are the
    same for the whole run and are read once.
    """

    def __init__(self, model: Any, atomic_numbers: Sequence[int]) -> None:
        from ase import Atoms
        from turbomace.embedding import features_from_atoms
        from turbomace.neighbors import SkinGraphBuilder

        self._model = model
        self._atoms = Atoms(numbers=list(atomic_numbers), pbc=True)
        self._features = features_from_atoms(model.program, [self._atoms])
        self._graphs = SkinGraphBuilder(
            model.r_max,
            model.species_of,
            skin=NEIGHBOUR_SKIN_ANGSTROM,
            backend="cuda" if model.is_cuda else "native",
        )

    def __call__(
        self, positions: npt.NDArray[np.float64], box: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        self._atoms.set_cell(box)
        self._atoms.set_positions(positions)
        graph, cells, pbc = self._graphs(self._atoms)
        result = self._model.compute_graph(
            graph,
            forces=True,
            virial=False,
            features=self._features,
            cells=cells,
            pbc=pbc,
        )
        return float(result["energy"][0]), np.asarray(result["forces"])
