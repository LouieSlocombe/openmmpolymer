"""Measure apparent breaking strength from a tensile ladder, step by step.

Run with the full openmmpolymer environment from the repository root::

    python examples/breaking_strength.py --output breaking-example
    python examples/breaking_strength.py --output breaking-example --analyse-only

Breaking strength is the peak nominal tensile stress, confirmed only after a
sustained terminal stress drop. Nominal stress uses the measured transverse
area relative to its initial value. A still-rising curve does not establish a
break. Fixed harmonic bonds cannot model covalent fracture: stress loss can
reflect chain sliding or separation. The illustrative duration and maximum
strain do not guarantee a resolved break.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("breaking-example"))
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
        BreakingSpec,
        ChainSpec,
        analyse_breaking,
        build_melt,
        run_breaking_scan,
        write_breaking_report,
    )

    if not args.analyse_only:
        # 1. Define the polymer and cell: 40 polyethylene chains of 30 units.
        chain_spec = ChainSpec(
            monomer_smiles="[*]CC[*]",
            degree_of_polymerization=30,
            residue_name="PE",
        )
        n_chains = 40

        # 2. Build, charge, parameterise and pack. The density checks cell size;
        # it does not impose the density ultimately measured in NPT.
        chain, run = build_melt(
            chain_spec,
            n_chains=n_chains,
            directory=args.output,
            target_density_g_cm3=0.95,
            charge_method="nagl",
        )

        # 3. Specify the strain ladder and failure criterion. A peak is accepted
        # only with at least three final holds at or below half that peak.
        spec = BreakingSpec(
            temperature_k=298.15,
            pressure_bar=1.0,
            axis=2,
            strain_increment=0.01,
            max_strain=1.0,
            relax_ps=50.0,
            n_replicas=3,
            samples_per_step=250,
            failure_fraction=0.5,
            confirmation_steps=3,
            stage_ps=1000.0,
            trajectory_ps=10.0,
            max_total_ns=100.0,
        )

        # 4. Equilibrate once and extend three velocity replicas. Increments
        # compound; max_strain=1 requests at least 100% engineering extension.
        # Each hold's transverse area supplies sigma_nominal = sigma_true*A/A0.
        run_breaking_scan(
            run,
            args.output,
            spec=spec,
            chain_backbone=chain.backbone,
            atoms_per_chain=chain.n_atoms,
        )

    # 5. Reanalyse with the saved criterion; no dynamics occur in this call.
    report = analyse_breaking(args.output)

    # 6. Save each curve and its failure diagnostics, including unresolved ones.
    files = write_breaking_report(report, figures=not args.no_figures)
    print(f"Report: {files.json}")

    # 7. A numerical peak is not sufficient: require the supported headline.
    # strength_mpa is already MPa, so no pressure-unit conversion is needed.
    if report.resolved and report.strength_mpa is not None:
        print(f"Apparent breaking strength: {report.strength_mpa:.6g} MPa")
    else:
        print("Apparent breaking strength: unresolved")
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
