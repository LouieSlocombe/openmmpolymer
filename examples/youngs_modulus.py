"""Measure Young's modulus E, step by step, in MPa.

Run from the repository root in the full openmmpolymer environment::

    python examples/youngs_modulus.py --output modulus-example
    python examples/youngs_modulus.py --output modulus-example --analyse-only

The first command builds polyethylene and runs real dynamics. Edit the chain,
temperature and sampling settings below for your material, using a new output
directory for a different request. A short hold can leave E unresolved.

This example measures extension and a separate load-controlled cross-check.
Its replicas share one equilibrated configuration with fresh velocities.
See docs/main_properties.md for the numbered walkthrough and interpretation.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from openmmpolymer import (
    ChainSpec,
    ModulusSpec,
    analyse_mechanics,
    build_melt,
    mechanical_scan,
    run_modulus_scan,
    write_mechanical_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("modulus-example"))
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="read the saved run without dynamics",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="write the JSON report without plots"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Choose the measurement conditions and the small-strain fitting window.
    # Strains are fractions: 0.015 means 1.5%; temperatures are always kelvin.
    spec = ModulusSpec(
        temperature_k=298.15,
        pressure_bar=1.0,
        axis=2,
        strain_increment=0.002,
        max_strain=0.05,
        elastic_strain_limit=0.015,
        relax_ps=50.0,
        n_replicas=3,
        samples_per_step=250,
        load_stresses_bar=(0.0, 100.0, 200.0, 300.0),
        load_ps_each=1000.0,
        bulk_pressures_bar=None,
        shear_strains=None,
        max_total_ns=100.0,
    )

    if not args.analyse_only:
        # 2. Build, charge, parameterise and pack the starting cell.
        # This is a packing; the scan below performs the equilibration.
        plan = mechanical_scan(spec)
        print(f"Requested dynamics: {plan.total_duration_ps / 1000:.3f} ns")
        chain, run = build_melt(
            ChainSpec(
                monomer_smiles="[*]CC[*]",
                degree_of_polymerization=30,
                residue_name="PE",
            ),
            n_chains=40,
            directory=args.output,
            target_density_g_cm3=0.85,
        )

        # 3. Equilibrate at the measurement temperature, then extend replicas.
        # The lateral axes relax at pressure. Repeating this request resumes it.
        run_modulus_scan(
            run,
            args.output,
            spec=spec,
            chain_backbone=chain.backbone,
            atoms_per_chain=chain.n_atoms,
        )

    # 4. Read the saved curves and fit differential stress versus axial strain.
    # Keep this window consistent with elastic_strain_limit in the scan above.
    report = analyse_mechanics(args.output, strain_limit=spec.elastic_strain_limit)
    print(f"Resolved: {report.resolved}")
    if report.resolved and report.youngs is not None and report.youngs.resolved:
        print(f"E = {report.youngs.modulus_mpa:.6g} MPa")
    else:
        print("E unresolved: inspect the fit window, stress noise and replica spread.")
    if report.youngs is not None:
        print(f"Strain rate = {report.youngs.strain_rate_per_ns} strain/ns")
    if report.replica_spread_mpa is not None:
        print(
            f"Replica spread = {report.replica_spread_mpa:.6g} MPa (standard deviation)"
        )
    if math.isfinite(report.method_gap):
        print(f"Extension/load method gap = {100 * report.method_gap:.3g}%")
    for note in report.notes:
        print(f"Note: {note}")

    # 5. Save mechanics.json and figures, including unresolved diagnostics.
    files = write_mechanical_report(report, figures=not args.no_figures)
    print(f"Report: {files.json}")


if __name__ == "__main__":
    main()
