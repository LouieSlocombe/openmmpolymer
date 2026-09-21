"""All-atom polymer melts with OpenMM, packmol and forcefill.

The package takes a monomer SMILES to an equilibrated melt in four steps, each
of which is usable on its own::

    from openmmpolymer import (
        ChainSpec, build_chain, assign_charges, build_polymer_forcefield,
        PackedComponent, box_edge_nm, pack_box, assemble_box, prepare_box,
        prepare_run, standard_melt_equilibration, run_protocol,
    )

    chain = build_chain(
        ChainSpec(monomer_smiles="[*]CC[*]", degree_of_polymerization=20,
                  residue_name="PE"),
        n_conformers=50,
    )
    assign_charges(chain.sdf_paths[0], "nagl")
    forcefield = build_polymer_forcefield(chain.sdf_paths[0], residue_name="PE")

    components = [PackedComponent(path, 1) for path in chain.pdb_paths]
    edge = box_edge_nm([50], [chain.molar_mass_g_mol], 0.3)
    packed = pack_box(components, edge)

    box = prepare_box(assemble_box(components, packed.packed_pdb, packed.box_nm),
                      forcefield)
    run = prepare_run(box, forcefield)
    run_protocol(standard_melt_equilibration(), run, "run")

The shape of all this follows from one constraint. forcefill will not
parameterise a residue bonded to its neighbours, and it is right not to: a
stand-alone GAFF treatment of a chain-linked residue is not valid. So a chain
is built as a single molecule and a single residue, charged here rather than by
the backend - a graph neural network scales where AM1-BCC's semi-empirical QM
does not - and packed by conformer so that the cell holds fifty different
conformations rather than fifty copies of one.

It sits next to two of the same author's packages: ``forcefill`` does the
parameterisation, and ``openmmnqe`` covers nuclear quantum effects for systems
where they matter.
"""

from importlib.metadata import PackageNotFoundError, version

from .chain import (
    ChainError,
    ChainResult,
    ChainSpec,
    assemble_chain,
    atom_names,
    backbone_path,
    build_chain,
    characteristic_ratio,
    trans_fraction,
)
from .charges import (
    CHARGE_METHODS,
    ChargeError,
    ChargeResult,
    assign_charges,
    default_nagl_model,
)
from .forcefield import (
    BACKENDS,
    DEFAULT_BASE_FORCEFIELD,
    ForceFieldError,
    PolymerForceField,
    build_polymer_forcefield,
    check_forcefield,
)
from .mdsystem import (
    PackedBox,
    SystemAssemblyError,
    SystemSpec,
    assemble_box,
    barostat_kind,
    build_system,
    check_box,
    check_target_density,
    check_timestep,
    make_barostat,
    max_timestep_fs,
    minimum_mass_g_mol,
    platform_is_usable,
    prepare_box,
    replicate_topology,
    select_platform,
)
from .packing import (
    DEFAULT_PACKING_DENSITY,
    DEFAULT_TOLERANCE_NM,
    PackedComponent,
    PackmolError,
    PackResult,
    box_edge_nm,
    check_packing,
    density_g_cm3,
    distribute_conformers,
    find_packmol,
    find_rings,
    load_positions_nm,
    pack_box,
    packmol_version,
    read_packed_pdb,
    read_pdb,
    render_packmol_input,
)
from .protocols import (
    ChainDimensions,
    Protocol,
    ProtocolError,
    RunManifest,
    RunSummary,
    Stage,
    chain_dimensions,
    melt_quench,
    run_protocol,
    standard_melt_equilibration,
)
from .reporters import TrajectoryOptions, steps_for
from .simulate import (
    RunContext,
    Segment,
    SimulationError,
    StageResult,
    prepare_run,
    run_anneal,
    run_compress,
    run_minimise,
    run_npt,
    run_nvt,
    run_production,
    run_pushoff,
    run_quench,
    run_segments,
    safe_timestep_fs,
)

try:
    __version__ = version("openmmpolymer")
except PackageNotFoundError:  # pragma: no cover - uninstalled checkout
    __version__ = "0.0.0+unknown"

__all__ = [
    "BACKENDS",
    "CHARGE_METHODS",
    "DEFAULT_BASE_FORCEFIELD",
    "DEFAULT_PACKING_DENSITY",
    "DEFAULT_TOLERANCE_NM",
    "ChainDimensions",
    "ChainError",
    "ChainResult",
    "ChainSpec",
    "ChargeError",
    "ChargeResult",
    "ForceFieldError",
    "PackResult",
    "PackedBox",
    "PackedComponent",
    "PackmolError",
    "PolymerForceField",
    "Protocol",
    "ProtocolError",
    "RunContext",
    "RunManifest",
    "RunSummary",
    "Segment",
    "SimulationError",
    "Stage",
    "StageResult",
    "SystemAssemblyError",
    "SystemSpec",
    "TrajectoryOptions",
    "__version__",
    "assemble_box",
    "assemble_chain",
    "assign_charges",
    "atom_names",
    "backbone_path",
    "barostat_kind",
    "box_edge_nm",
    "build_chain",
    "build_polymer_forcefield",
    "build_system",
    "chain_dimensions",
    "characteristic_ratio",
    "check_box",
    "check_forcefield",
    "check_packing",
    "check_target_density",
    "check_timestep",
    "default_nagl_model",
    "density_g_cm3",
    "distribute_conformers",
    "find_packmol",
    "find_rings",
    "load_positions_nm",
    "make_barostat",
    "max_timestep_fs",
    "melt_quench",
    "minimum_mass_g_mol",
    "pack_box",
    "packmol_version",
    "platform_is_usable",
    "prepare_box",
    "prepare_run",
    "read_packed_pdb",
    "read_pdb",
    "render_packmol_input",
    "replicate_topology",
    "run_anneal",
    "run_compress",
    "run_minimise",
    "run_npt",
    "run_nvt",
    "run_production",
    "run_protocol",
    "run_pushoff",
    "run_quench",
    "run_segments",
    "safe_timestep_fs",
    "select_platform",
    "standard_melt_equilibration",
    "steps_for",
    "trans_fraction",
]
