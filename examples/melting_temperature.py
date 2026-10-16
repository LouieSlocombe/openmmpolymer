r"""Step by step: heat a prepared polymer crystal and report apparent Tm.

Run from the repository root in the full openmmpolymer environment::

    python examples/melting_temperature.py --crystal-pdb crystal.pdb \
        --system-xml system.xml --output tm-example
    python examples/melting_temperature.py --output tm-example --analyse-only
    python examples/melting_temperature.py --output tm-example --analyse-only --no-figures

Prepare a crystalline or semicrystalline periodic cell before running this
example. The PDB needs box vectors (CRYST1) and bonds, with atoms in the order
of a matching serialised OpenMM System. The System must use periodic boundary
conditions and contain no barostat or Andersen thermostat. Molecules must have
equal atom counts in contiguous blocks. Loading these files validates the cell
but does not establish crystalline order. An amorphous packed melt or quenched
glass is not a substitute for the supplied crystal.

The example performs 42 ns of dynamics. The analysis looks for coincident
positive jumps in specific volume and enthalpy during constant-pressure
heating. Its Tm is the midpoint of the two adjacent sampled temperatures that
bracket the jump. This is an apparent melting temperature: superheating, scan
rate and morphology matter. Inspect saved coordinates to verify crystal loss.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openmmpolymer import (
    TmSpec,
    analyse_melting,
    load_crystal,
    run_tm_scan,
    write_melting_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tm-example"), help="run directory"
    )
    parser.add_argument(
        "--crystal-pdb", type=Path, help="prepared periodic crystal PDB with bonds"
    )
    parser.add_argument(
        "--system-xml", type=Path, help="OpenMM System matching the crystal atom order"
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
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.analyse_only:
        if args.crystal_pdb is None or args.system_xml is None:
            parser.error("--crystal-pdb and --system-xml are required for a new scan")

        # 1. Read an independently prepared crystal and its force-field System.
        # The cell must contain crystalline order before it reaches this step.
        print("1. Load the prepared periodic crystal and matching OpenMM System.")
        run = load_crystal(args.crystal_pdb, args.system_xml)

        # 2. Start below and finish above the expected transition at 1 bar.
        # The example is 41 temperature holds plus 1 ns initial equilibration.
        # Anisotropic pressure coupling lets each crystal cell length relax.
        print("2. Set a 250-650 K heating ladder with 10 K steps and 1 ns holds.")
        spec = TmSpec(
            t_start_k=250.0,
            t_end_k=650.0,
            step_k=10.0,
            hold_ps=1000.0,
            equilibration_ps=1000.0,
            pressure_bar=1.0,
            stage_ps=10_000.0,
            samples_per_segment=30,
            min_points_per_branch=3,
            barostat="anisotropic",
            trajectory_ps=10.0,
            max_total_ns=50.0,
        )

        # 3. Minimise, settle the crystal, then heat it without melt preparation.
        # crystalline=True is our declaration, not an automated structure test.
        print("3. Equilibrate the crystal and run the 42 ns heating scan.")
        run_tm_scan(run, args.output, spec=spec, crystalline=True)

    # 4. Reanalyse the saved volume/enthalpy jumps and write the report.
    # --analyse-only enters here; the original input files are not required.
    print("4. Reanalyse the saved heating curve and write the report.")
    report = analyse_melting(args.output)
    files = write_melting_report(report, figures=not args.no_figures)

    # 5. Report only a resolved temperature and its finite-grid bracket.
    # The bracket is a sampled interval, not a statistical confidence interval.
    if report.resolved and report.temperature_k is not None:
        print(f"5. Apparent Tm = {report.temperature_k - 273.15:.2f} °C")
        if report.transition.bracket_k is not None:
            low_k, high_k = report.transition.bracket_k
            print(f"Sampled bracket = {low_k - 273.15:.2f} to {high_k - 273.15:.2f} °C")
    else:
        print("5. Apparent Tm unresolved; no temperature is reported.")
    if report.curve.heating_rate_k_per_ns is not None:
        print(f"Heating rate = {report.curve.heating_rate_k_per_ns:.6g} K/ns")
    for note in report.notes:
        print(f"Note: {note}")
    print(f"Report: {files.json}")


if __name__ == "__main__":
    main()
