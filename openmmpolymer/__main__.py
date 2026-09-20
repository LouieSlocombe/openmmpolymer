"""Command-line driver for a whole polymer melt run.

Flat flags rather than subcommands: the pipeline is one path from a monomer
SMILES to an equilibrated cell, and every stage of it wants the same handful of
facts. Anything more selective is better done from Python, where the four
layers - chain, force field, packing, protocol - are separately callable.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .chain import ChainSpec, build_chain
from .charges import CHARGE_METHODS, assign_charges
from .forcefield import BACKENDS, build_polymer_forcefield
from .mdsystem import (
    SystemSpec,
    assemble_box,
    check_target_density,
    prepare_box,
)
from .packing import (
    DEFAULT_PACKING_DENSITY,
    PackedComponent,
    box_edge_nm,
    check_packing,
    pack_box,
)
from .protocols import melt_quench, run_protocol, standard_melt_equilibration
from .simulate import prepare_run

log = logging.getLogger(__name__)

#: The protocols the command line offers.
PROTOCOLS = {
    "equilibrate": standard_melt_equilibration,
    "melt-quench": melt_quench,
}


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="openmmpolymer",
        description="Build, pack and equilibrate an all-atom polymer melt.",
    )
    parser.add_argument(
        "monomer",
        help="monomer SMILES with two [*] attachment points, e.g. '[*]CC[*]'",
    )
    parser.add_argument(
        "-n",
        "--degree-of-polymerization",
        type=int,
        default=20,
        help="repeat units per chain (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--chains",
        type=int,
        default=30,
        help="chains in the cell (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--residue-name",
        default="POL",
        help="residue name, at most three characters (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=450.0,
        help="target temperature in kelvin (default: %(default)s)",
    )
    parser.add_argument(
        "--melt-temperature",
        type=float,
        default=600.0,
        help="temperature the chains are mobilised at (default: %(default)s)",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="pressure in bar (default: %(default)s)",
    )
    parser.add_argument(
        "--pack-density",
        type=float,
        default=DEFAULT_PACKING_DENSITY,
        help="density to pack at in g/cm3, before compression (default: %(default)s)",
    )
    parser.add_argument(
        "--target-density",
        type=float,
        default=0.85,
        help="density the cell is expected to reach, checked against the "
        "cutoff before anything long starts (default: %(default)s)",
    )
    parser.add_argument(
        "--charge-method",
        default="nagl",
        choices=CHARGE_METHODS,
        help="partial-charge method (default: %(default)s)",
    )
    parser.add_argument(
        "--backend",
        default="smirnoff",
        choices=BACKENDS,
        help="forcefill parameterisation backend (default: %(default)s)",
    )
    parser.add_argument(
        "--protocol",
        default="equilibrate",
        choices=sorted(PROTOCOLS),
        help="what to run (default: %(default)s)",
    )
    parser.add_argument(
        "--tacticity",
        default="atactic",
        choices=("atactic", "isotactic", "syndiotactic"),
        help="backbone stereochemistry (default: %(default)s)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="OpenMM platform (default: the fastest available)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="run",
        help="where everything is written (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0xF0,
        help="master random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build, charge, parameterise and pack, but run no dynamics",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log what each step is doing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments to parse, or None to take them from the command line.

    Returns:
        A process exit code.
    """
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    output = Path(cast(str, arguments.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    n_chains = int(arguments.chains)

    chain = build_chain(
        ChainSpec(
            monomer_smiles=cast(str, arguments.monomer),
            degree_of_polymerization=int(arguments.degree_of_polymerization),
            residue_name=cast(str, arguments.residue_name),
            tacticity=cast(str, arguments.tacticity),
            seed=int(arguments.seed),
        ),
        "chain",
        n_conformers=n_chains,
        output_dir=output / "build",
    )
    print(
        f"chain: {chain.n_atoms} atoms, {chain.molar_mass_g_mol:.1f} g/mol, "
        f"{chain.embedder} embedder",
        flush=True,
    )

    spec = SystemSpec()
    edge = check_target_density(
        [n_chains],
        [chain.molar_mass_g_mol],
        float(arguments.target_density),
        spec,
    )
    print(
        f"cell: {edge:.2f} nm at {arguments.target_density} g/cm3 once compressed",
        flush=True,
    )

    if arguments.charge_method != "none":
        assign_charges(chain.sdf_paths[0], cast(str, arguments.charge_method))
    forcefield = build_polymer_forcefield(
        chain.sdf_paths[0],
        output / "build" / "polymer_ff.xml",
        residue_name=cast(str, arguments.residue_name),
        backend=cast(str, arguments.backend),
        cache_dir=output / "cache",
        workdir=output / "build" / "forcefill",
    )
    print(f"force field: {forcefield.forcefield_xml}", flush=True)

    components = [PackedComponent(path, 1) for path in chain.pdb_paths]
    packed = pack_box(
        components,
        box_edge_nm(
            [n_chains], [chain.molar_mass_g_mol], float(arguments.pack_density)
        ),
        output / "build" / "packed.pdb",
        seed=int(arguments.seed),
        workdir=output / "build",
    )
    box = assemble_box(components, packed.packed_pdb, packed.box_nm)
    check_packing(box.topology, box.positions_nm)
    print(
        f"packed: {box.n_molecules} chains, {box.topology.getNumAtoms()} atoms",
        flush=True,
    )

    if arguments.dry_run:
        print("dry run: stopping before dynamics", flush=True)
        return 0

    run = prepare_run(
        prepare_box(box, forcefield),
        forcefield,
        spec,
        platform=cast("str | None", arguments.platform),
        seed=int(arguments.seed),
    )
    protocol = PROTOCOLS[cast(str, arguments.protocol)](
        target_temperature_k=float(arguments.temperature),
        melt_temperature_k=float(arguments.melt_temperature),
        pressure_bar=float(arguments.pressure),
    )
    summary = run_protocol(
        protocol,
        run,
        output,
        chain_backbone=chain.backbone,
        atoms_per_chain=chain.n_atoms,
        expected_characteristic_ratio=7.0,
    )
    print(
        f"{summary.protocol}: {len(summary.results)} stages in "
        f"{summary.wall_seconds / 60:.1f} min, manifest {summary.manifest_path}",
        flush=True,
    )
    if summary.chains is not None:
        print(
            f"chains: Rg {summary.chains.mean_radius_of_gyration_nm:.3f} nm, "
            f"C {summary.chains.characteristic_ratio:.2f} "
            f"({'consistent' if summary.chains.consistent else 'not relaxed'})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
