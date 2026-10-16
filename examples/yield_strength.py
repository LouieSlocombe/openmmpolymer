"""Measure an apparent 0.2% offset yield strength, step by step.

Run with the full openmmpolymer environment from the repository root::

    python examples/yield_strength.py --output yield-example
    python examples/yield_strength.py --output yield-example --analyse-only

The proof stress is the first later crossing of the nominal tensile curve
with a shifted elastic line. Without unloading, this does not establish
permanent deformation. Nominal stress uses the measured transverse area.
Fixed harmonic bonds cannot model covalent fracture. These short illustrative
holds and three velocity replicas do not guarantee a resolved material value.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("yield-example"))
    parser.add_argument(
        "--analyse-only", action="store_true", help="read a saved scan without dynamics"
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="write JSON without plots"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Parse first so --help works without the simulation dependencies installed.
    from openmmpolymer import (
        ChainSpec,
        YieldSpec,
        analyse_yield,
        build_melt,
        run_yield_scan,
        write_yield_report,
    )

    if not args.analyse_only:
        # 1. Define the polymer and cell. These are polyethylene repeat units.
        chain_spec = ChainSpec(
            monomer_smiles="[*]CC[*]",
            degree_of_polymerization=30,
            residue_name="PE",
        )
        n_chains = 40

        # 2. Build, charge, parameterise and pack. The density is a cell-size
        # check; the scan's NPT equilibration determines the actual density.
        chain, run = build_melt(
            chain_spec,
            n_chains=n_chains,
            directory=args.output,
            target_density_g_cm3=0.95,
            charge_method="nagl",
        )

        # 3. Specify the experiment. Strains are fractions, so 0.002 is 0.2%.
        # Fit sigma_nominal = E*strain + b in the elastic window, then find
        # its later crossing with E*(strain - offset_strain) + b.
        spec = YieldSpec(
            temperature_k=298.15,
            pressure_bar=1.0,
            axis=2,
            strain_increment=0.002,
            max_strain=0.3,
            relax_ps=50.0,
            n_replicas=3,
            samples_per_step=250,
            fit_min_strain=0.0,
            fit_max_strain=0.015,
            offset_strain=0.002,
            stage_ps=1000.0,
            max_total_ns=100.0,
        )

        # 4. Equilibrate once, then extend three velocity replicas. Increments
        # compound; the transverse box dimensions relax at the set pressure.
        # An unchanged request resumes intact stages on a repeated invocation.
        run_yield_scan(
            run,
            args.output,
            spec=spec,
            chain_backbone=chain.backbone,
            atoms_per_chain=chain.n_atoms,
        )

    # 5. Reanalyse the saved curves using their recorded offset and fit window.
    # --analyse-only reaches here without building a cell or running dynamics.
    report = analyse_yield(args.output)

    # 6. Save the full curves, criteria, diagnostics and optional figures.
    files = write_yield_report(report, figures=not args.no_figures)
    print(f"Report: {files.json}")

    # 7. Quote only a supported headline. The reader already returns MPa.
    if report.resolved and report.strength_mpa is not None:
        print(f"Apparent offset yield strength: {report.strength_mpa:.6g} MPa")
    else:
        print("Apparent offset yield strength: unresolved")
    for index, curve in zip(report.replica_indices, report.curves, strict=True):
        rate = curve.strain_rate_per_ns
        print(
            f"Replica {index} strain rate: "
            + ("unknown" if rate is None else f"{rate:.6g} strain/ns")
        )
    for note in report.notes:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()
