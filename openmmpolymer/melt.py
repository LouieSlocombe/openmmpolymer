"""Building a melt from a monomer in one call, and rebuilding it safely.

:func:`build_melt` runs the four layers in turn - chain, charges, force field,
packing - and returns the :class:`~openmmpolymer.simulate.RunContext` a
protocol takes. What it adds to calling them yourself is what a second call
into the same directory needs.

The build goes into ``<directory>/build`` together with a record of what it
produced. A rerun cannot simply reuse those files - new dependency versions can
change the Hamiltonian even when nothing was asked differently - and must not
overwrite them, because a finished run rests on them. So it rebuilds in
scratch space inside the directory, from a copy of the force-field cache, and
goes on only if the rebuilt chain, cell and System match the record and every
run already started there. The originals are never touched: a failed or
changed rebuild leaves them as they were, and an accepted one hands back file
references to them rather than to the discarded rebuild.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ._files import file_sha256, write_json
from ._validation import require_integer
from .chain import ChainResult, ChainSpec, build_chain
from .charges import assign_charges
from .forcefield import DEFAULT_SMIRNOFF_FORCEFIELD, build_polymer_forcefield
from .mdsystem import SystemSpec, assemble_box, check_target_density, prepare_box
from .packing import (
    DEFAULT_PACKING_DENSITY,
    PACKMOL_TIMEOUT_S,
    box_edge_nm,
    check_packing,
    distribute_conformers,
    pack_box,
)
from .protocols import ProtocolError, _run_identity, validate_run_inputs
from .simulate import RunContext, prepare_run

log = logging.getLogger(__name__)

#: The record of what a build produced, kept among its assets.
_RECORD_NAME = "inputs.json"


def build_melt(
    spec: ChainSpec,
    n_chains: int,
    directory: str | Path,
    *,
    target_density_g_cm3: float,
    n_conformers: int | None = None,
    charge_method: str = "nagl",
    backend: str = "smirnoff",
    smirnoff_forcefield: str = DEFAULT_SMIRNOFF_FORCEFIELD,
    pack_density_g_cm3: float = DEFAULT_PACKING_DENSITY,
    packmol_timeout_s: float | None = PACKMOL_TIMEOUT_S,
    platform: str | None = None,
    cache_dir: str | Path | None = None,
    progress: Callable[[str], object] = log.info,
) -> tuple[ChainResult, RunContext]:
    """Build a cell of *n_chains* chains of *spec*, ready for a protocol.

    Args:
        spec: The chain. Its seed also seeds the packing and the run.
        n_chains: Chains in the cell.
        directory: The run directory. The build goes in its ``build``
            subdirectory, and a rebuild has to match the runs already started
            in it or in its ``equilibration`` subdirectory, where the rate
            scans keep their shared preparation.
        target_density_g_cm3: The density the cell will be compressed to. It
            has to fit the cutoff there too, which is checked before anything
            expensive starts.
        n_conformers: Distinct conformations to build, repeated to fill the
            cell. None for one per chain, which is right and the slowest to
            pack; more than one per chain is never built.
        charge_method: One of :data:`~openmmpolymer.charges.CHARGE_METHODS`.
        backend: One of :data:`~openmmpolymer.forcefield.BACKENDS`.
        smirnoff_forcefield: SMIRNOFF release, for the smirnoff backend.
        pack_density_g_cm3: The density packmol fills the cell to.
        packmol_timeout_s: Seconds packmol is allowed, or None for no limit.
        platform: OpenMM platform, or None for the fastest available.
        cache_dir: The force-field cache, ``<directory>/cache`` by default.
            Share one to parameterise a chain once for several runs.
        progress: Called with one line of news as each layer finishes.

    Returns:
        The chain and the run context. Their file references point at the
        verified assets in ``<directory>/build``.

    Raises:
        ProtocolError: A recorded asset changed or is missing, the directory
            holds a build with no record, or the rebuild matches neither the
            record nor a run already started. Nothing is overwritten.
    """
    require_integer(n_chains, name="n_chains")
    root = Path(directory)
    build = root / "build"
    cache = root / "cache" if cache_dir is None else Path(cache_dir)
    record_path = build / _RECORD_NAME
    record = json.loads(record_path.read_text()) if record_path.is_file() else None
    if record is not None:
        for name, digest in record["artifacts"].items():
            path = build / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ProtocolError(
                    f"Existing build artifact {name!r} changed or is missing. "
                    "Restore it or use a fresh output directory."
                )

    fresh = not build.exists()
    with ExitStack() as stack:
        if fresh:
            working, working_cache = build, cache
        else:
            scratch = Path(
                stack.enter_context(
                    TemporaryDirectory(prefix=".build-check-", dir=root)
                )
            )
            working, working_cache = scratch / "build", scratch / "cache"
            if cache.is_dir():
                shutil.copytree(cache, working_cache)
        chain, run = _prepare(
            spec,
            n_chains,
            working,
            working_cache,
            n_conformers=(
                n_chains if n_conformers is None else min(n_conformers, n_chains)
            ),
            target_density_g_cm3=target_density_g_cm3,
            charge_method=charge_method,
            backend=backend,
            smirnoff_forcefield=smirnoff_forcefield,
            pack_density_g_cm3=pack_density_g_cm3,
            packmol_timeout_s=packmol_timeout_s,
            platform=platform,
            progress=progress,
        )
        # Through JSON, so the tuples in the chain compare with the record's lists.
        inputs = json.loads(
            json.dumps(
                {
                    "run": _run_identity(run),
                    "chain": {
                        key: value
                        for key, value in asdict(chain).items()
                        if key not in {"sdf_paths", "pdb_paths"}
                    },
                },
                allow_nan=False,
            )
        )
        if not fresh and (record is None or record["inputs"] != inputs):
            raise ProtocolError(
                "Prepared chemistry, force field or packed coordinates changed, "
                "or the existing build lacks verified inputs. The original build "
                "was preserved; use a fresh output directory."
            )
        for run_dir in (root, root / "equilibration"):
            validate_run_inputs(run, run_dir)

    if fresh:
        artifacts = (
            *chain.sdf_paths,
            *chain.pdb_paths,
            run.forcefield.forcefield_xml,
            str(build / "packed.pdb"),
        )
        write_json(
            record_path,
            {
                "inputs": inputs,
                "artifacts": {
                    str(Path(path).resolve().relative_to(build.resolve())): (
                        file_sha256(path)
                    )
                    for path in artifacts
                },
            },
        )
        return chain, run
    return (
        replace(
            chain,
            sdf_paths=tuple(str(build / Path(path).name) for path in chain.sdf_paths),
            pdb_paths=tuple(str(build / Path(path).name) for path in chain.pdb_paths),
        ),
        replace(
            run,
            forcefield=replace(
                run.forcefield,
                forcefield_xml=str(build / Path(run.forcefield.forcefield_xml).name),
            ),
        ),
    )


def _prepare(
    spec: ChainSpec,
    n_chains: int,
    build_dir: Path,
    cache_dir: Path,
    *,
    n_conformers: int,
    target_density_g_cm3: float,
    charge_method: str,
    backend: str,
    smirnoff_forcefield: str,
    pack_density_g_cm3: float,
    packmol_timeout_s: float | None,
    platform: str | None,
    progress: Callable[[str], object],
) -> tuple[ChainResult, RunContext]:
    """Run the four layers into *build_dir*, the cheap cell-size check first."""
    chain = build_chain(spec, "chain", n_conformers=n_conformers, output_dir=build_dir)
    progress(
        f"chain: {chain.n_atoms} atoms, {chain.molar_mass_g_mol:.1f} g/mol, "
        f"{chain.embedder} embedder"
    )

    system = SystemSpec()
    edge = check_target_density(
        [n_chains], [chain.molar_mass_g_mol], target_density_g_cm3, system
    )
    progress(f"cell: {edge:.2f} nm at {target_density_g_cm3} g/cm3 once compressed")

    assign_charges(chain.sdf_paths[0], charge_method)
    forcefield = build_polymer_forcefield(
        chain.sdf_paths[0],
        build_dir / "polymer_ff.xml",
        residue_name=spec.residue_name,
        backend=backend,
        smirnoff_forcefield=smirnoff_forcefield,
        cache_dir=cache_dir,
        workdir=build_dir / "forcefill",
    )
    progress(f"force field: {forcefield.forcefield_xml}")

    components = distribute_conformers(chain.pdb_paths, n_chains)
    packed = pack_box(
        components,
        box_edge_nm([n_chains], [chain.molar_mass_g_mol], pack_density_g_cm3),
        build_dir / "packed.pdb",
        seed=spec.seed,
        timeout_s=packmol_timeout_s,
        workdir=build_dir,
    )
    box = assemble_box(components, packed.packed_pdb, packed.box_nm)
    check_packing(box.topology, box.positions_nm)
    progress(f"packed: {box.n_molecules} chains, {box.topology.getNumAtoms()} atoms")

    run = prepare_run(
        prepare_box(box, forcefield),
        forcefield,
        system,
        platform=platform,
        seed=spec.seed,
    )
    return chain, run
