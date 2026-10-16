"""Measure apparent elongation at break, step by step.

Run with the full openmmpolymer environment from the repository root::

    python examples/elongation_at_break.py --output elongation-example
    python examples/elongation_at_break.py --output elongation-example --analyse-only

Elongation at break is 100 times the engineering strain at the FIRST hold of
a confirmed terminal loss of nominal tensile stress. It is not the strain at
the peak or the maximum requested strain. Fixed harmonic bonds cannot model
covalent fracture, so this criterion measures apparent cell failure through
stress loss. The illustrative duration and maximum strain do not guarantee
that a break will occur or that its elongation will resolve.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("elongation-example"))
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
        ElongationSpec,
        analyse_elongation,
        build_melt,
        run_elongation_scan,
        write_elongation_report,
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
        # the scan will equilibrate the cell at the measurement conditions.
        chain, run = build_melt(
            chain_spec,
            n_chains=n_chains,
            directory=args.output,
            target_density_g_cm3=0.95,
            charge_method="nagl",
        )

        # 3. Define a break as at least three terminal holds at or below half the
        # peak nominal stress. The first of those holds locates the event.
        spec = ElongationSpec(
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

        # 4. Equilibrate once, then extend three velocity replicas. Increments
        # compound. All chunks share L0, so epsilon=(L-L0)/L0 keeps its origin.
        run_elongation_scan(
            run,
            args.output,
            spec=spec,
            chain_backbone=chain.backbone,
            atoms_per_chain=chain.n_atoms,
        )

    # 5. Reanalyse the saved ladder with its recorded stress-loss criterion.
    report = analyse_elongation(args.output)

    # 6. Write the full report and optionally each replica's curve and event.
    files = write_elongation_report(report, figures=not args.no_figures)
    print(f"Report: {files.json}")

    # 7. The report already stores 100*epsilon_break as a percentage.
    # Do not multiply elongation_percent by 100 again or substitute max_strain.
    if report.resolved and report.elongation_percent is not None:
        print(f"Apparent elongation at break: {report.elongation_percent:.6g}%")
    else:
        print("Apparent elongation at break: unresolved")
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
