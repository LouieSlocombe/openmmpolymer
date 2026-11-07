"""Step by step: equilibrate a polyethylene melt, then run it under MACE-OFF.

Run from the repository root in the full openmmpolymer environment, with
TurboMACE installed and MACE-OFF converted once to a .tmace file::

    turbomace-export MACE-OFF23_small.model -o MACE-OFF23_small.tmace
    python examples/mace_off_melt.py --model MACE-OFF23_small.tmace --output mace-off-example
    python examples/mace_off_melt.py --model MACE-OFF23_small.tmace --output mace-off-check --quick

The classical force field does the cheap part: packing relief, compression
from the packing density and annealing take nanoseconds, which a GPU covers
in minutes with OpenFF Sage and in hours with MACE. The equilibrated state then
starts a run whose every interaction is MACE-OFF23, evaluated by TurboMACE
inside OpenMM, while the force field still supplies the masses and the X-H
constraints. The two runs live in separate directories because they are
different Hamiltonians, and repeating the command resumes both.

The MACE-OFF density at the end is the mean over the production stage. It
starts from the classical density, so a production run shorter than the
melt's volume relaxation reports mostly where it started; --quick is for
checking the set-up, not for a number.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from openmmpolymer import (
    ChainSpec,
    Protocol,
    RunManifest,
    Stage,
    TurboMACESpec,
    build_melt,
    run_protocol,
    standard_melt_equilibration,
    use_turbomace,
)

TEMPERATURE_K = 450.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="MACE-OFF as a .tmace file"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("mace-off-example"), help="run directory"
    )
    parser.add_argument(
        "--production-ps",
        type=float,
        default=200.0,
        help="time under MACE-OFF at constant pressure (default: 200)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="shorten the classical equilibration to check the set-up",
    )
    return parser


def final_stage(directory: Path, protocol: Protocol) -> dict[str, Any]:
    """What a protocol's last stage recorded, whether it ran now or before."""
    manifest = RunManifest.load(directory)
    assert manifest is not None, "run_protocol writes the manifest"
    return manifest.stages[protocol.stages[-1].name]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    classical_dir = args.output / "classical"
    mace_dir = args.output / "mace-off"

    # 1. Build a classical cell. Forty chains of twenty ethylene units are
    # about 4900 atoms, comfortably above the cell the cutoff needs.
    print("1. Build 40 polyethylene chains of 20 repeat units with OpenFF Sage.")
    chain, classical = build_melt(
        ChainSpec(
            monomer_smiles="[*]CC[*]", degree_of_polymerization=20, residue_name="PE"
        ),
        n_chains=40,
        directory=classical_dir,
        target_density_g_cm3=0.85,
    )
    chain_options = {"chain_backbone": chain.backbone, "atoms_per_chain": chain.n_atoms}

    # 2. Equilibrate with the force field: the long, cheap part.
    print(f"2. Equilibrate the classical melt at {TEMPERATURE_K:g} K.")
    shortened = (
        {"nvt_ps": 50.0, "compress_ps_each": 20.0, "npt_ps": 200.0, "anneal_cycles": 1}
        if args.quick
        else {}
    )
    equilibration = standard_melt_equilibration(
        target_temperature_k=TEMPERATURE_K, **shortened
    )
    equilibrated = run_protocol(
        equilibration, classical, classical_dir, **chain_options
    )

    # 3. Swap the force field's interactions for MACE-OFF. Same cell, masses
    # and constraints; only the System's forces change.
    print(f"3. Replace the interactions with {args.model.name}.")
    mace = use_turbomace(classical, TurboMACESpec(args.model))

    # 4. Minimise on the new surface, then hold the pressure under MACE-OFF,
    # starting from the classical state.
    print(f"4. Run {args.production_ps:g} ps at 1 bar under MACE-OFF.")
    production = Protocol(
        name="mace_off_npt",
        stages=(
            Stage("00_minimise", "minimise"),
            Stage(
                "01_npt",
                "npt",
                {"temperature_k": TEMPERATURE_K, "duration_ps": args.production_ps},
            ),
        ),
    )
    run_protocol(
        production,
        mace,
        mace_dir,
        state_in=equilibrated.final_state,
        **chain_options,
    )

    # 5. Compare the two densities and say what the MACE run cost. The
    # manifests hold every stage's result, including any a resume skipped.
    settled = final_stage(classical_dir, equilibration)
    held = final_stage(mace_dir, production)
    print(f"5. Density at {TEMPERATURE_K:g} K and 1 bar:")
    print(f"   OpenFF Sage: {settled['mean_density_g_cm3']:.4f} g/cm3")
    print(
        f"   MACE-OFF:    {held['mean_density_g_cm3']:.4f} g/cm3, "
        f"mean of {args.production_ps:g} ps"
    )
    rate = args.production_ps / 1000.0 / (held["wall_seconds"] / 86400.0)
    print(f"   MACE-OFF ran at {rate:.1f} ns/day.")
    print(f"   Manifests: {classical_dir}/manifest.json, {mace_dir}/manifest.json")


if __name__ == "__main__":
    main()
