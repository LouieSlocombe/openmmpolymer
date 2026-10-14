"""Building a melt from a monomer in one call, and rebuilding it safely.

:func:`build_melt` runs the four layers in turn - chain, charges, force field,
packing - builds the System, and returns the chain with the
:class:`~openmmpolymer.simulate.RunContext` a protocol takes. What it adds to
calling them yourself is what a second call into the same directory needs.

The first build is prepared in scratch space and published to
``<directory>/build`` together with a record of what it produced. A failed
attempt leaves no partial build to block a retry. A rerun cannot simply reuse
those files - new dependency versions can change the Hamiltonian even when
nothing was asked differently - and must not overwrite them, because a finished
run rests on them. So it rebuilds in
scratch space inside the directory, from a copy of the force-field cache, and
goes on only if the rebuilt chain, cell and System match the record and every
run already started there. The originals are never touched: a failed or
changed rebuild leaves them as they were, and an accepted one hands back file
references to them rather than to the discarded rebuild.

:func:`build_blend` does the same for a cell of several polymers. Each is
built and parameterised on its own, the force fields are merged, and the
chains are packed together, one polymer after another.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ._files import file_sha256, write_json
from ._seeds import DEFAULT_SEED
from ._validation import require_integer
from .chain import ChainResult, ChainSpec, build_chain
from .charges import DEFAULT_CHARGE_METHOD, assign_charges
from .forcefield import (
    DEFAULT_BACKEND,
    DEFAULT_SMIRNOFF_FORCEFIELD,
    build_polymer_forcefield,
    merge_polymer_forcefields,
)
from .mdsystem import SystemSpec, assemble_box, check_target_density, prepare_box
from .packing import (
    DEFAULT_PACKING_DENSITY,
    PACKMOL_TIMEOUT_S,
    box_edge_nm,
    check_packing,
    distribute_conformers,
    pack_box,
)
from .protocols import ProtocolError, _canonical, _run_identity, validate_run_inputs
from .simulate import RunContext, prepare_run

log = logging.getLogger(__name__)

#: The record of what a build produced, kept among its assets.
_RECORD_NAME = "inputs.json"

#: What each kind of build holds, by the key its record keeps the chemistry under.
_KINDS = {"chain": "melt of one polymer", "blend": "blend of several polymers"}


@dataclass(frozen=True)
class BlendComponent:
    """One polymer of a blend.

    Args:
        spec: Its chain. The residue name identifies the polymer throughout:
            its files, its force-field template and its chains' residues.
        n_chains: How many chains of it the cell holds.
    """

    spec: ChainSpec
    n_chains: int

    def __post_init__(self) -> None:
        """Reject a count below one at construction, not mid-build."""
        require_integer(self.n_chains, name="n_chains")


def build_melt(
    spec: ChainSpec,
    n_chains: int,
    directory: str | Path,
    *,
    target_density_g_cm3: float,
    n_conformers: int | None = None,
    charge_method: str = DEFAULT_CHARGE_METHOD,
    backend: str = DEFAULT_BACKEND,
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
            subdirectory only after preparation and verification succeed.
            A rebuild has to match the runs already started in it or in its
            ``equilibration`` subdirectory, where the rate
            scans keep their shared preparation.
        target_density_g_cm3: The density the cell will be compressed to. It
            has to fit the cutoff there too, which is checked as soon as the
            chain is built, before charges, parameterisation and packing.
        n_conformers: Distinct conformations to build, repeated to fill the
            cell. None for one per chain, which is right and the slowest to
            pack; more than one per chain is never built.
        charge_method: One of :data:`~openmmpolymer.charges.CHARGE_METHODS`.
        backend: One of :data:`~openmmpolymer.forcefield.BACKENDS`.
        smirnoff_forcefield: SMIRNOFF release, for the smirnoff backend.
        pack_density_g_cm3: The density packmol fills the cell to.
        packmol_timeout_s: Seconds packmol is allowed, or None for no limit.
        platform: OpenMM platform, or None for the fastest that works.
        cache_dir: The force-field cache, ``<directory>/cache`` by default.
            Share one to parameterise a chain once for several runs.
        progress: Called with one line of news as each layer finishes.

    Returns:
        The chain and the run context. Their file references point at the
        verified assets in ``<directory>/build``.

    Raises:
        ProtocolError: A recorded asset changed or is missing, the directory
            holds a build with no record, or the build does not match the
            record or a run already started there. Nothing is overwritten.
    """
    require_integer(n_chains, name="n_chains")
    conformers = n_chains if n_conformers is None else min(n_conformers, n_chains)

    def prepare(working: Path, working_cache: Path) -> tuple[ChainResult, RunContext]:
        return _prepare(
            spec,
            n_chains,
            working,
            working_cache,
            n_conformers=conformers,
            target_density_g_cm3=target_density_g_cm3,
            charge_method=charge_method,
            backend=backend,
            smirnoff_forcefield=smirnoff_forcefield,
            pack_density_g_cm3=pack_density_g_cm3,
            packmol_timeout_s=packmol_timeout_s,
            platform=platform,
            progress=progress,
        )

    def describe(chain: ChainResult) -> tuple[object, tuple[str, ...]]:
        return _chain_record(chain), (*chain.sdf_paths, *chain.pdb_paths)

    chain, run, build = _verified_build(
        directory, cache_dir, "chain", prepare, describe
    )
    return _published(chain, build), run


def build_blend(
    components: Sequence[BlendComponent],
    directory: str | Path,
    *,
    target_density_g_cm3: float,
    n_conformers: int | None = None,
    charge_method: str = DEFAULT_CHARGE_METHOD,
    backend: str = DEFAULT_BACKEND,
    smirnoff_forcefield: str = DEFAULT_SMIRNOFF_FORCEFIELD,
    pack_density_g_cm3: float = DEFAULT_PACKING_DENSITY,
    packmol_timeout_s: float | None = PACKMOL_TIMEOUT_S,
    platform: str | None = None,
    cache_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    progress: Callable[[str], object] = log.info,
) -> tuple[tuple[ChainResult, ...], RunContext]:
    """Build a cell holding chains of several polymers, ready for a protocol.

    Each polymer is built, charged and parameterised on its own, exactly as
    :func:`build_melt` does for one, and its force field is cached on its own.
    The force fields are then merged, and the chains packed together in the
    order given, so each polymer's chains sit in one contiguous block of the
    topology. A rebuild is held to the record in the same way.

    Only the smirnoff backend is supported: see
    :func:`~openmmpolymer.forcefield.merge_polymer_forcefields`. And because
    the chains are not all alike, the per-chain measurements a protocol can
    take - chain dimensions, and the structure and dynamics read back from a
    trajectory - refuse a blend rather than mix the polymers up. Everything
    measured over the whole cell, from density to stress, works as for a melt.

    Args:
        components: The polymers and how many chains of each. Every residue
            name has to differ, ignoring case, because it names the polymer's
            files, its force-field template and its chains in the topology.
        directory: The run directory, as for :func:`build_melt`.
        target_density_g_cm3: The density the cell will be compressed to.
        n_conformers: Distinct conformations to build per polymer, at most
            one per chain of it. None for one per chain.
        charge_method: One of :data:`~openmmpolymer.charges.CHARGE_METHODS`.
        backend: Must be ``"smirnoff"``; the parameter exists so that a
            caller passing :func:`build_melt`'s settings is told why not.
        smirnoff_forcefield: SMIRNOFF release.
        pack_density_g_cm3: The density packmol fills the cell to.
        packmol_timeout_s: Seconds packmol is allowed, or None for no limit.
        platform: OpenMM platform, or None for the fastest that works.
        cache_dir: The force-field cache, ``<directory>/cache`` by default.
        seed: Seeds the packing and the run. Each chain's own seed is its
            spec's, as for a melt.
        progress: Called with one line of news as each layer finishes.

    Returns:
        One chain per component, in order, and the run context. Their file
        references point at the verified assets in ``<directory>/build``.

    Raises:
        ValueError: Fewer than two components, a repeated residue name, a
            chain count below one, or a backend other than smirnoff.
        ProtocolError: As for :func:`build_melt`, including a directory that
            holds a melt rather than a blend.
    """
    components = tuple(components)
    check_blend(components, backend)
    require_integer(seed, name="seed", minimum=0)

    def prepare(
        working: Path, working_cache: Path
    ) -> tuple[tuple[ChainResult, ...], RunContext]:
        return _prepare_blend(
            components,
            working,
            working_cache,
            n_conformers=n_conformers,
            target_density_g_cm3=target_density_g_cm3,
            charge_method=charge_method,
            smirnoff_forcefield=smirnoff_forcefield,
            pack_density_g_cm3=pack_density_g_cm3,
            packmol_timeout_s=packmol_timeout_s,
            platform=platform,
            seed=seed,
            progress=progress,
        )

    def describe(chains: tuple[ChainResult, ...]) -> tuple[object, tuple[str, ...]]:
        record = [
            {
                "residue_name": component.spec.residue_name,
                "n_chains": component.n_chains,
                "chain": _chain_record(chain),
            }
            for component, chain in zip(components, chains, strict=True)
        ]
        files = tuple(
            path for chain in chains for path in (*chain.sdf_paths, *chain.pdb_paths)
        )
        forcefields = tuple(
            str(
                Path(chains[0].sdf_paths[0]).parent
                / _forcefield_name(component.spec.residue_name)
            )
            for component in components
        )
        return record, (*files, *forcefields)

    chains, run, build = _verified_build(
        directory, cache_dir, "blend", prepare, describe
    )
    return tuple(_published(chain, build) for chain in chains), run


def check_blend(components: Sequence[BlendComponent], backend: str) -> None:
    """Refuse a blend :func:`build_blend` could not build safely.

    Cheap, so a caller can check a request before recording or building
    anything.

    Raises:
        ValueError: Fewer than two components, a repeated residue name, or a
            backend other than smirnoff.
    """
    if len(components) < 2:
        raise ValueError(
            "A blend needs at least two components; build one polymer with build_melt."
        )
    names = [component.spec.residue_name for component in components]
    if len({name.upper() for name in names}) != len(names):
        raise ValueError(
            f"Each polymer in a blend needs its own residue name, not {names}: "
            "it names the polymer's files and force-field template, and two "
            "that differ only in case would overwrite each other's files."
        )
    if backend != "smirnoff":
        raise ValueError(
            f"A blend needs the smirnoff backend, not {backend!r}: GAFF atom "
            "types are shared classes, so merging two chains' parameters "
            "would silently keep one chain's bonded terms for both."
        )


def _chain_record(chain: ChainResult) -> dict[str, object]:
    """What a build's record holds about one chain: all but its file paths."""
    return {
        key: value
        for key, value in asdict(chain).items()
        if key not in {"sdf_paths", "pdb_paths"}
    }


def _published(chain: ChainResult, build: Path) -> ChainResult:
    """Point *chain*'s file references at the published build."""
    return replace(
        chain,
        sdf_paths=tuple(str(build / Path(path).name) for path in chain.sdf_paths),
        pdb_paths=tuple(str(build / Path(path).name) for path in chain.pdb_paths),
    )


def _forcefield_name(residue_name: str) -> str:
    """The file one polymer of a blend is parameterised into."""
    return f"{residue_name}_ff.xml"


def _verified_build[T](
    directory: str | Path,
    cache_dir: str | Path | None,
    kind: str,
    prepare: Callable[[Path, Path], tuple[T, RunContext]],
    describe: Callable[[T], tuple[object, tuple[str, ...]]],
) -> tuple[T, RunContext, Path]:
    """Prepare a build in scratch space, hold it to the record, then publish it.

    Args:
        directory: The run directory.
        cache_dir: The force-field cache, or None for ``<directory>/cache``.
        kind: The key the record keeps the chemistry under: ``"chain"`` for a
            melt, ``"blend"`` for a blend. A directory built as one kind is
            refused by the other before anything is rebuilt.
        prepare: Builds into the working directory it is given, using the
            force-field cache it is given.
        describe: What to record of the chemistry *prepare* returned, and its
            files besides the force field and the packed cell.

    Returns:
        What *prepare* returned, the run with its force field in the published
        build, and the published build directory.
    """
    root = Path(directory)
    build = root / "build"
    cache = root / "cache" if cache_dir is None else Path(cache_dir)
    record_path = build / _RECORD_NAME
    record = json.loads(record_path.read_text()) if record_path.is_file() else None
    fresh = not build.exists()
    if not fresh and record is None:
        raise ProtocolError(
            "The existing build lacks verified inputs. Its files were preserved; "
            "restore inputs.json or use a fresh output directory."
        )
    if record is not None:
        recorded_kind = (set(record["inputs"]) - {"run"}).pop()
        if recorded_kind != kind:
            raise ProtocolError(
                f"The existing build holds a {_KINDS[recorded_kind]}, not a "
                f"{_KINDS[kind]}. It was preserved; use a fresh output directory."
            )
        for name, digest in record["artifacts"].items():
            path = build / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ProtocolError(
                    f"Existing build artifact {name!r} changed or is missing. "
                    "Restore it or use a fresh output directory."
                )

    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".build-check-", dir=root) as temporary:
        scratch = Path(temporary)
        working = scratch / "build"
        if fresh:
            # A cache inside build/ must move with the assets; creating it at
            # its final path would itself leave a partial published build.
            working_cache = (
                working / cache.resolve().relative_to(build.resolve())
                if cache.resolve().is_relative_to(build.resolve())
                else cache
            )
        else:
            working_cache = scratch / "cache"
            if cache.is_dir():
                shutil.copytree(cache, working_cache)
        prepared, run = prepare(working, working_cache)
        chemistry, files = describe(prepared)
        # Canonical lists compare with the chain tuples saved in the JSON record.
        inputs = _canonical({"run": _run_identity(run), kind: chemistry})
        if record is not None and record["inputs"] != inputs:
            raise ProtocolError(
                "Prepared chemistry, force field or packed coordinates changed. "
                "The original build was preserved; use a fresh output directory."
            )
        for run_dir in (root, root / "equilibration"):
            validate_run_inputs(run, run_dir)

        if fresh:
            artifacts = (
                *files,
                run.forcefield.forcefield_xml,
                str(working / "packed.pdb"),
            )
            write_json(
                working / _RECORD_NAME,
                {
                    "inputs": inputs,
                    "artifacts": {
                        str(Path(path).resolve().relative_to(working.resolve())): (
                            file_sha256(path)
                        )
                        for path in artifacts
                    },
                },
            )
            # Publish the complete directory, including its record, in one
            # rename. Any earlier failure only discards this attempt's scratch.
            working.rename(build)
    return (
        prepared,
        replace(
            run,
            forcefield=replace(
                run.forcefield,
                forcefield_xml=str(build / Path(run.forcefield.forcefield_xml).name),
            ),
        ),
        build,
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
    """Run the four layers into *build_dir*.

    The cheap cell-size check runs as soon as the chain's molar mass is known,
    before the charges, the force field and the packing.
    """
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


def _prepare_blend(
    components: tuple[BlendComponent, ...],
    build_dir: Path,
    cache_dir: Path,
    *,
    n_conformers: int | None,
    target_density_g_cm3: float,
    charge_method: str,
    smirnoff_forcefield: str,
    pack_density_g_cm3: float,
    packmol_timeout_s: float | None,
    platform: str | None,
    seed: int,
    progress: Callable[[str], object],
) -> tuple[tuple[ChainResult, ...], RunContext]:
    """Run the four layers for every polymer of a blend into *build_dir*.

    Every chain is built before the cell-size check, which needs all their
    masses, and that check still comes before the charges, the force fields
    and the packing.
    """
    chains = []
    for component in components:
        spec, count = component.spec, component.n_chains
        chain = build_chain(
            spec,
            spec.residue_name,
            n_conformers=count if n_conformers is None else min(n_conformers, count),
            output_dir=build_dir,
        )
        progress(
            f"chain {spec.residue_name}: {chain.n_atoms} atoms, "
            f"{chain.molar_mass_g_mol:.1f} g/mol, {chain.embedder} embedder"
        )
        chains.append(chain)
    counts = [component.n_chains for component in components]
    names = [component.spec.residue_name for component in components]
    masses = [chain.molar_mass_g_mol for chain in chains]

    system = SystemSpec()
    edge = check_target_density(counts, masses, target_density_g_cm3, system)
    progress(f"cell: {edge:.2f} nm at {target_density_g_cm3} g/cm3 once compressed")

    forcefields = []
    for name, chain in zip(names, chains, strict=True):
        assign_charges(chain.sdf_paths[0], charge_method)
        forcefields.append(
            build_polymer_forcefield(
                chain.sdf_paths[0],
                build_dir / _forcefield_name(name),
                residue_name=name,
                backend="smirnoff",
                smirnoff_forcefield=smirnoff_forcefield,
                cache_dir=cache_dir,
                workdir=build_dir / "forcefill",
            )
        )
    forcefield = merge_polymer_forcefields(forcefields, build_dir / "polymer_ff.xml")
    progress(f"force field: {forcefield.forcefield_xml}")

    packed_components = [
        packed_component
        for component, chain in zip(components, chains, strict=True)
        for packed_component in distribute_conformers(
            chain.pdb_paths, component.n_chains
        )
    ]
    packed = pack_box(
        packed_components,
        box_edge_nm(counts, masses, pack_density_g_cm3),
        build_dir / "packed.pdb",
        seed=seed,
        timeout_s=packmol_timeout_s,
        workdir=build_dir,
    )
    box = assemble_box(packed_components, packed.packed_pdb, packed.box_nm)
    check_packing(box.topology, box.positions_nm)
    mix = ", ".join(f"{n} {name}" for n, name in zip(counts, names, strict=True))
    progress(
        f"packed: {box.n_molecules} chains ({mix}), {box.topology.getNumAtoms()} atoms"
    )

    run = prepare_run(
        prepare_box(box, forcefield),
        forcefield,
        system,
        platform=platform,
        seed=seed,
    )
    return tuple(chains), run
