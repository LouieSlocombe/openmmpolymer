"""Step by step: build a polyethylene melt, cool it, and report Tg in Celsius.

Run from the repository root in the full openmmpolymer environment::

    python examples/glass_transition.py --output tg-example
    python examples/glass_transition.py --output tg-example --analyse-only
    python examples/glass_transition.py --output tg-example --analyse-only --no-figures

This is a real simulation, with a 150 ns total dynamics ceiling, not a quick
demo. Forty chains of thirty ethylene repeat units are an example system;
chain length, size, sampling and force-field accuracy need convergence checks.
Repeating the same command resumes recorded stages with the same inputs.

At each cooling temperature, density gives specific volume v = 1/rho. The
analysis fits glass and melt branches v = a*T + b and finds their intersection
Tg = (b_glass - b_melt)/(a_melt - a_glass). The scan first locates the change
coarsely, then cools through a finer window. A coarse scan without a resolved
transition stops instead of inventing a fine window. A resolved result remains
a transition at the simulated cooling rate, not an experimental prediction.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openmmpolymer import (
    ChainSpec,
    TgSpec,
    analyse_tg,
    build_melt,
    run_tg_scan,
    write_tg_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tg-example"), help="run directory"
    )
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="read saved results without dynamics",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="write the JSON report without plots"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.analyse_only:
        # 1. Specify chemistry and build a sparse, parameterised cell.
        # NAGL is the default charge method; build_melt handles packing and
        # force-field assignment. Packing density is not an equilibrium result.
        print("1. Build 40 polyethylene chains, each with 30 repeat units.")
        chain, run = build_melt(
            ChainSpec(
                monomer_smiles="[*]CC[*]",
                degree_of_polymerization=30,
                residue_name="PE",
            ),
            n_chains=40,
            directory=args.output,
            target_density_g_cm3=0.85,
            charge_method="nagl",
        )

        # 2. Choose the cooling schedule. Times are ps; temperatures are K.
        # Record the final melt's trajectory so analysis can check chain motion.
        print("2. Set coarse and fine cooling ladders at 1 bar.")
        spec = TgSpec(
            melt_temperature_k=650.0,
            t_floor_k=150.0,
            coarse_step_k=25.0,
            coarse_hold_ps=1000.0,
            window_k=60.0,
            fine_step_k=5.0,
            fine_hold_ps=3000.0,
            pressure_bar=1.0,
            stage_ps=10_000.0,
            samples_per_segment=30,
            min_points_per_branch=4,
            npt_trajectory_ps=10.0,
            max_total_ns=150.0,
        )

        # 3. Equilibrate first, then run the coarse and adaptive fine scans.
        # Backbone indices identify endpoints; atom order alone does not.
        print("3. Equilibrate and run the two-pass Tg scan (up to 150 ns).")
        run_tg_scan(
            run,
            args.output,
            spec=spec,
            chain_backbone=chain.backbone,
            atoms_per_chain=chain.n_atoms,
        )

    # 4. Reanalyse saved measurements, then save JSON and optional figures.
    # This same step works later, without rebuilding or running dynamics.
    print("4. Reanalyse the saved cooling curves and write the report.")
    report = analyse_tg(args.output)
    files = write_tg_report(report, figures=not args.no_figures)

    # 5. Convert a resolved temperature, preserving rate and diagnostic notes.
    # Kelvin and Celsius have the same step size, so K/ns needs no conversion.
    if report.resolved and report.temperature_k is not None:
        print(f"5. Tg = {report.temperature_k - 273.15:.2f} °C")
    else:
        print("5. Tg unresolved; no temperature is reported.")
    if report.cooling_rate_k_per_ns is not None:
        print(f"Cooling rate = {report.cooling_rate_k_per_ns:.6g} K/ns")
    for note in report.notes:
        print(f"Note: {note}")
    print(f"Report: {files.json}")


if __name__ == "__main__":
    main()
