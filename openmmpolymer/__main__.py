"""Command-line driver for a whole polymer melt run.

Flat flags rather than subcommands: the pipeline is one path from a monomer
SMILES to an equilibrated cell, and every stage of it wants the same handful of
facts. Anything more selective is better done from Python, where the four
layers - chain, force field, packing, protocol - are separately callable.

There is one exception, and it is ``--analyse``. Reading a finished run
directory back is a different verb over a different input: it needs no monomer,
none of the build flags apply to it, and "do it from Python" was the wrong
answer for the one thing every run ends in. So the monomer is required unless
that flag is given, which is the whole of the dispatch this command has.

The flags a protocol accepts are a table rather than a chain of conditionals,
because the alternative failed quietly: a flag that no factory took was parsed,
ignored, and never reached the run. :data:`PROTOCOLS` names what each factory
accepts, and a test checks those names against the factories themselves.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .chain import ChainSpec, build_chain
from .charges import CHARGE_METHODS, assign_charges
from .elasticity import deform_stages, load_stages, shear_stages
from .forcefield import BACKENDS, build_polymer_forcefield
from .mdsystem import (
    SystemSpec,
    assemble_box,
    check_target_density,
    prepare_box,
)
from .mechanical import (
    ModulusSpec,
    analyse_mechanics,
    mechanical_scan,
    run_modulus_scan,
    write_mechanical_report,
)
from .packing import (
    DEFAULT_PACKING_DENSITY,
    box_edge_nm,
    check_packing,
    distribute_conformers,
    pack_box,
)
from .protocols import Protocol, melt_quench, run_protocol, standard_melt_equilibration
from .simulate import prepare_run
from .tg import (
    TgSpec,
    analyse_run,
    cooling_rate_series,
    run_tg_scan,
    tg_coarse_scan,
    write_report,
)
from .timeseries import (
    DSC_COOLING_RATE_K_PER_NS,
    EXTRAPOLATION_FORMS,
    cooling_rate_extrapolation,
    quench_stages,
)
from .trajectory import AnalysisError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProtocolEntry:
    """A protocol the command line offers, and the flags it accepts.

    Args:
        factory: What builds the protocol.
        options: The factory keywords this protocol accepts. Listed rather
            than inferred so that a flag which reaches no factory is a test
            failure rather than a setting that silently never arrives, which
            is how t_end, step_k and hold_ps came to be unreachable.
    """

    factory: Callable[..., Protocol]
    options: tuple[str, ...]


#: Keywords whose argparse destination is spelled differently, because the
#: flags were named before the factories were.
_DESTS = {
    "target_temperature_k": "temperature",
    "melt_temperature_k": "melt_temperature",
    "pressure_bar": "pressure",
    # The modulus protocol has one temperature rather than a melt and a
    # target, and it is the same -t flag.
    "temperature_k": "temperature",
}

#: Keywords every protocol factory takes.
_COMMON = ("melt_temperature_k", "pressure_bar")

#: The temperature a run settles at. A two-pass scan settles at the melt
#: temperature by construction, so it does not take one.
_TARGET = ("target_temperature_k",)

#: The cooling ladder.
_QUENCH = ("t_start", "t_end", "step_k", "hold_ps")

#: The second pass, and the two things that gate a whole scan.
_TG = (
    "fine_step_k",
    "fine_hold_ps",
    "fine_window_k",
    "max_total_ns",
    "check_melt",
)


#: The extension, and the three passes that can be skipped.
_MECHANICS = (
    "temperature_k",
    "strain_increment",
    "max_strain",
    "relax_ps",
    "elastic_strain_limit",
    "replicas",
    "deform_axis",
    "load_stresses",
    "bulk_pressures",
    "shear_strains",
    "skip",
    "max_total_ns",
)


def _modulus_spec(
    *,
    temperature_k: float = 298.15,
    pressure_bar: float = 1.0,
    strain_increment: float = 0.002,
    max_strain: float = 0.05,
    relax_ps: float = 50.0,
    elastic_strain_limit: float = 0.015,
    replicas: int = 3,
    deform_axis: int = 2,
    load_stresses: tuple[float, ...] | None = None,
    bulk_pressures: tuple[float, ...] | None = None,
    shear_strains: tuple[float, ...] | None = None,
    skip: Sequence[str] | None = None,
    max_total_ns: float | None = None,
) -> ModulusSpec:
    """Turn the flat mechanics flags into the spec a scan takes.

    A pass is skipped by naming it in ``--skip``, which is clearer than
    passing an empty list to the flag that configures it.
    """
    dropped = set(skip or ())
    defaults = ModulusSpec()
    return ModulusSpec(
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        axis=deform_axis,
        strain_increment=strain_increment,
        max_strain=max_strain,
        relax_ps=relax_ps,
        elastic_strain_limit=elastic_strain_limit,
        n_replicas=replicas,
        load_stresses_bar=(
            None if "load" in dropped else (load_stresses or defaults.load_stresses_bar)
        ),
        bulk_pressures_bar=(
            None
            if "bulk" in dropped
            else (bulk_pressures or defaults.bulk_pressures_bar)
        ),
        shear_strains=(
            None if "shear" in dropped else (shear_strains or defaults.shear_strains)
        ),
        max_total_ns=max_total_ns,
    )


def _modulus_protocol(**options: Any) -> Protocol:
    """The equilibration and one extension, so --dry-run can price it."""
    return mechanical_scan(_modulus_spec(**options))


def _tg_spec(
    *,
    melt_temperature_k: float = 650.0,
    pressure_bar: float = 1.0,
    t_end: float = 150.0,
    step_k: float = 25.0,
    hold_ps: float = 1000.0,
    fine_step_k: float = 5.0,
    fine_hold_ps: float = 3000.0,
    fine_window_k: float = 60.0,
    max_total_ns: float | None = None,
    check_melt: float | None = None,
) -> TgSpec:
    """Turn the flat cooling flags into the spec a two-pass scan takes."""
    return TgSpec(
        melt_temperature_k=melt_temperature_k,
        t_floor_k=t_end,
        coarse_step_k=step_k,
        coarse_hold_ps=hold_ps,
        window_k=fine_window_k,
        fine_step_k=fine_step_k,
        fine_hold_ps=fine_hold_ps,
        pressure_bar=pressure_bar,
        npt_trajectory_ps=check_melt,
        max_total_ns=max_total_ns,
    )


def _tg_protocol(**options: Any) -> Protocol:
    """The coarse half of a two-pass scan, so --dry-run can price it."""
    return tg_coarse_scan(_tg_spec(**options))


#: The protocols the command line offers.
PROTOCOLS = {
    "equilibrate": ProtocolEntry(standard_melt_equilibration, _TARGET + _COMMON),
    "melt-quench": ProtocolEntry(melt_quench, _TARGET + _COMMON + _QUENCH),
    "tg": ProtocolEntry(_tg_protocol, _COMMON + _QUENCH[1:] + _TG),
    "modulus": ProtocolEntry(_modulus_protocol, ("pressure_bar", *_MECHANICS)),
}


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="openmmpolymer",
        description="Build, pack and equilibrate an all-atom polymer melt.",
    )
    parser.add_argument(
        "monomer",
        nargs="?",
        default=None,
        help="monomer SMILES with two [*] attachment points, e.g. '[*]CC[*]'. "
        "Required unless --analyse is given",
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
    quench = parser.add_argument_group(
        "cooling",
        "the ladder a quenching protocol walks; tg adds a second, finer pass",
    )
    quench.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="temperature cooling starts from (default: the melt temperature)",
    )
    quench.add_argument(
        "--t-end",
        type=float,
        default=200.0,
        help="temperature cooling stops at (default: %(default)s)",
    )
    quench.add_argument(
        "--step-k",
        type=float,
        default=20.0,
        help="temperature drop per step (default: %(default)s)",
    )
    quench.add_argument(
        "--hold-ps",
        type=float,
        default=200.0,
        help="time held at each temperature (default: %(default)s)",
    )
    quench.add_argument(
        "--fine-step-k",
        type=float,
        default=5.0,
        help="temperature drop per step in the tg fine pass (default: %(default)s)",
    )
    quench.add_argument(
        "--fine-hold-ps",
        type=float,
        default=3000.0,
        help="time held at each fine temperature (default: %(default)s)",
    )
    quench.add_argument(
        "--fine-window-k",
        type=float,
        default=60.0,
        help="half-width of the fine window around the coarse transition "
        "(default: %(default)s)",
    )
    quench.add_argument(
        "--tg-approx",
        type=float,
        default=None,
        help="centre the fine window here instead of on the coarse fit",
    )
    quench.add_argument(
        "--cooling-rates",
        type=_rates,
        default=None,
        help="comma-separated rates in K/ns to repeat the fine pass at, e.g. "
        "10,5,2; the transition is then extrapolated toward experiment",
    )
    quench.add_argument(
        "--max-total-ns",
        type=float,
        default=None,
        help="refuse to start a tg or modulus scan longer than this",
    )
    quench.add_argument(
        "--check-melt",
        type=float,
        nargs="?",
        const=10.0,
        default=None,
        help="write a trajectory every N ps during equilibration so the melt "
        "can be shown to have relaxed (default interval 10 ps). This is "
        "frames of the whole cell - tens of megabytes - and without it the "
        "chain half of that check has nothing to read",
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
        default=None,
        help="where everything is written (default: run). With --analyse, "
        "where the report goes instead of <RUN_DIR>/analysis",
    )
    parser.add_argument(
        "--conformers",
        type=int,
        default=None,
        help="distinct conformations to build; they are repeated to fill the "
        "cell (default: one per chain, which is the right thing and the "
        "slowest to pack)",
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
    mechanics = parser.add_argument_group(
        "mechanics",
        "the extension the modulus protocol walks, and the passes beside it",
    )
    mechanics.add_argument(
        "--strain-increment",
        type=float,
        default=0.002,
        help="engineering strain added per step (default: %(default)s)",
    )
    mechanics.add_argument(
        "--max-strain",
        type=float,
        default=0.05,
        help="strain the ladder stops at (default: %(default)s)",
    )
    mechanics.add_argument(
        "--relax-ps",
        type=float,
        default=50.0,
        help="time to relax after each increment; the mean is over the "
        "second half, so this is twice the averaging window "
        "(default: %(default)s)",
    )
    mechanics.add_argument(
        "--elastic-strain-limit",
        type=float,
        default=0.015,
        help="strain the modulus is fitted up to (default: %(default)s)",
    )
    mechanics.add_argument(
        "--replicas",
        type=int,
        default=3,
        help="extensions from the same cell with fresh velocities; their "
        "spread is the error bar (default: %(default)s)",
    )
    mechanics.add_argument(
        "--deform-axis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="axis to stretch (default: %(default)s)",
    )
    mechanics.add_argument(
        "--load-stresses",
        type=_floats,
        default=None,
        help="comma-separated stresses in bar for the constant-stress "
        "cross-check, e.g. 0,100,200,300",
    )
    mechanics.add_argument(
        "--bulk-pressures",
        type=_floats,
        default=None,
        help="comma-separated pressures in bar for the bulk modulus, up and "
        "back down so the hysteresis is measurable",
    )
    mechanics.add_argument(
        "--shear-strains",
        type=_floats,
        default=None,
        help="comma-separated shear strains for the shear modulus",
    )
    mechanics.add_argument(
        "--skip",
        nargs="*",
        default=(),
        choices=("load", "bulk", "shear"),
        help="passes to leave out; the extension always runs",
    )
    analysis = parser.add_argument_group(
        "analysis", "read a finished run directory instead of building one"
    )
    analysis.add_argument(
        "--analyse",
        nargs="+",
        default=None,
        metavar="RUN_DIR",
        help="report the transition from these finished run directories; the "
        "first owns the output. No monomer is needed",
    )
    analysis.add_argument(
        "--melt-stage",
        default="05_npt",
        help="the equilibration stage to check (default: %(default)s)",
    )
    analysis.add_argument(
        "--no-melt-check",
        action="store_true",
        help="skip the melt equilibration check",
    )
    analysis.add_argument(
        "--rate-form",
        default="log_linear",
        choices=EXTRAPOLATION_FORMS,
        help="which rate relation to headline (default: %(default)s)",
    )
    analysis.add_argument(
        "--target-rate",
        type=float,
        default=DSC_COOLING_RATE_K_PER_NS,
        help="cooling rate in K/ns to extrapolate to (default: 10 K/min)",
    )
    analysis.add_argument(
        "--min-points-per-branch",
        type=int,
        default=4,
        help="points each branch of the fit must keep (default: %(default)s)",
    )
    analysis.add_argument(
        "--rg",
        type=float,
        default=None,
        help="radius of gyration in nm, overriding the manifest",
    )
    analysis.add_argument(
        "--no-figures",
        action="store_true",
        help="write the record but no figures",
    )
    analysis.add_argument(
        "--figure-format",
        default="png",
        help="what to save figures as (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log what each step is doing",
    )
    return parser


def _floats(text: str) -> tuple[float, ...]:
    """Parse a comma-separated list of numbers, at the front door."""
    try:
        values = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of numbers, e.g. '0,100,200'."
        ) from error
    if not values:
        raise argparse.ArgumentTypeError(
            f"{text!r} is empty. Leave the flag out for the default, or name "
            "the pass in --skip to drop it."
        )
    return values


def _rates(text: str) -> tuple[float, ...]:
    """Parse a comma-separated list of cooling rates, at the front door."""
    try:
        rates = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of numbers, e.g. '10,5,2'."
        ) from error
    if len(rates) < 2:
        raise argparse.ArgumentTypeError(
            f"{text!r} gives {len(rates)} rate(s); a rate dependence needs at "
            "least two."
        )
    if any(rate <= 0.0 for rate in rates) or len(set(rates)) != len(rates):
        raise argparse.ArgumentTypeError(
            f"{text!r} must be distinct positive rates in K/ns."
        )
    return rates


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments to parse, or None to take them from the command line.

    Returns:
        A process exit code.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if arguments.analyse:
        return _analyse(arguments)
    if arguments.monomer is None:
        parser.error("a monomer SMILES is required unless --analyse is given")

    output = Path(cast("str | None", arguments.output_dir) or "run")
    output.mkdir(parents=True, exist_ok=True)
    n_chains = int(arguments.chains)
    n_conformers = min(int(arguments.conformers or n_chains), n_chains)

    chain = build_chain(
        ChainSpec(
            monomer_smiles=cast(str, arguments.monomer),
            degree_of_polymerization=int(arguments.degree_of_polymerization),
            residue_name=cast(str, arguments.residue_name),
            tacticity=cast(str, arguments.tacticity),
            seed=int(arguments.seed),
        ),
        "chain",
        n_conformers=n_conformers,
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

    components = distribute_conformers(list(chain.pdb_paths), n_chains)
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
    name = cast(str, arguments.protocol)
    entry = PROTOCOLS[name]
    options = _protocol_options(arguments, entry)
    if name == "tg":
        return _run_tg_scan(arguments, run, output, chain, options)
    if name == "modulus":
        return _run_modulus_scan(arguments, run, output, chain, options)

    summary = run_protocol(
        entry.factory(**options),
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
    _print_chains(summary.chains)
    return 0


def _protocol_options(
    arguments: argparse.Namespace, entry: ProtocolEntry
) -> dict[str, Any]:
    """Collect the flags this protocol's factory actually takes."""
    return {name: getattr(arguments, _DESTS.get(name, name)) for name in entry.options}


def _print_chains(chains: Any) -> None:
    """Report the final chain dimensions, when they were measured."""
    if chains is None:
        return
    print(
        f"chains: Rg {chains.mean_radius_of_gyration_nm:.3f} nm, "
        f"C {chains.characteristic_ratio:.2f} "
        f"({'consistent' if chains.consistent else 'not relaxed'})",
        flush=True,
    )


def _run_tg_scan(
    arguments: argparse.Namespace,
    run: Any,
    output: Path,
    chain: Any,
    options: dict[str, Any],
) -> int:
    """Run a two-pass glass-transition scan, at one rate or several."""
    spec = _tg_spec(**options)
    common = {
        "spec": spec,
        "tg_approx_k": arguments.tg_approx,
        "chain_backbone": chain.backbone,
        "atoms_per_chain": chain.n_atoms,
    }
    rates = cast("tuple[float, ...] | None", arguments.cooling_rates)
    if rates is None:
        result = run_tg_scan(run, output, **common)
        found = (
            "no clear transition"
            if result.temperature_k is None
            else f"Tg = {result.temperature_k:.0f} K"
        )
        coarse = (
            "nothing"
            if result.approximate is None
            else f"{result.approximate.temperature_k:.0f} K"
        )
        print(
            f"tg: {found} at {result.fine_schedule.cooling_rate_k_per_ns:.2f} "
            f"K/ns (coarse said {coarse}), started from the "
            f"{result.restart} state",
            flush=True,
        )
        _print_chains(result.fine_summary.chains)
        return 0

    transitions = cooling_rate_series(run, output, rates_k_per_ns=rates, **common)
    for rate, fit in zip(rates, transitions, strict=True):
        print(
            f"tg: {fit.temperature_k:.0f} K at {rate:g} K/ns"
            f"{'' if fit.resolved else ' (not resolved)'}",
            flush=True,
        )
    extrapolation = cooling_rate_extrapolation(
        transitions,
        target_rate_k_per_ns=float(arguments.target_rate),
        form=cast(str, arguments.rate_form),
    )
    print(_extrapolation_line(extrapolation), flush=True)
    return 0


def _extrapolation_line(extrapolation: Any) -> str:
    """One line for a rate extrapolation, caveat included."""
    return (
        f"{extrapolation.form}: {extrapolation.temperature_k:.0f} K at "
        f"{extrapolation.target_rate_k_per_ns:.3g} K/ns, "
        f"{extrapolation.sensitivity_k_per_decade:.1f} K per decade over "
        f"{extrapolation.n_rates} rates - extrapolated "
        f"{extrapolation.extrapolation_decades:.1f} decades"
        f"{'' if extrapolation.resolved else ', not resolved'}"
    )


def _run_modulus_scan(
    arguments: argparse.Namespace,
    run: Any,
    output: Path,
    chain: Any,
    options: dict[str, Any],
) -> int:
    """Measure the elastic constants, and say what qualifies each one."""
    result = run_modulus_scan(
        run,
        output,
        spec=_modulus_spec(**options),
        chain_backbone=chain.backbone,
        atoms_per_chain=chain.n_atoms,
    )
    for line in _modulus_lines(result):
        print(line, flush=True)
    _print_chains(None)
    return 0


def _modulus_lines(result: Any) -> list[str]:
    """One line per constant, each carrying what qualifies it."""
    lines: list[str] = []
    if result.youngs is None:
        return ["modulus: nothing was deformed"]
    spread = (
        ""
        if result.replica_spread_mpa is None
        else f" +/- {result.replica_spread_mpa:.0f} over {len(result.replicas)}"
    )
    rate = result.schedule.strain_rate_per_ns
    lines.append(
        f"E = {result.youngs.modulus_mpa:.0f} MPa{spread} at {rate:.3g} "
        f"strain/ns, {result.youngs.temperature_k:.0f} K"
        f"{'' if result.resolved else ' (not resolved)'}"
    )
    if result.poisson is not None:
        lines.append(
            f"nu = {result.poisson.ratio:.3f}"
            f"{'' if result.poisson.resolved else ' (not resolved)'}"
        )
    for label, fit in (("K", result.bulk), ("G", result.shear)):
        if fit is not None:
            lines.append(
                f"{label} = {fit.modulus_mpa:.0f} MPa"
                f"{'' if fit.resolved else ' (not resolved)'}"
            )
    if result.load_modulus is not None:
        lines.append(
            f"constant-stress cross-check: "
            f"E = {result.load_modulus.modulus_mpa:.0f} MPa"
        )
    if result.consistency is not None:
        lines.append(_consistency_line(result.consistency))
    return lines


def _consistency_line(check: Any) -> str:
    """One line for the over-determination check."""
    import math

    gaps = ", ".join(
        f"{name} {100.0 * gap:.0f}%"
        for name, gap in (("K", check.bulk_gap), ("G", check.shear_gap))
        if math.isfinite(gap)
    )
    if not gaps:
        return (
            f"E and nu imply K = {check.bulk_implied_mpa:.0f}, "
            f"G = {check.shear_implied_mpa:.0f} MPa - nothing measured to "
            "check them against"
        )
    return (
        f"E and nu imply K = {check.bulk_implied_mpa:.0f}, "
        f"G = {check.shear_implied_mpa:.0f} MPa; measured differ by {gaps}"
        f"{'' if check.consistent else ' - not consistent'}"
    )


def _has_stages(run_dir: Path, find: Any) -> bool:
    """Whether a reader finds anything of its kind in this directory.

    The readers raise rather than return empty, which is the right shape for
    a caller that asked for one thing and consistent with
    :func:`~openmmpolymer.timeseries.quench_stages`. Here the question really
    is "is there any", so the refusal is caught once, in the one place that
    is asking rather than telling.
    """
    try:
        find(run_dir)
    except AnalysisError:
        return False
    return True


def _analyse(arguments: argparse.Namespace) -> int:
    """Report whatever a finished run directory holds, and write it out.

    Dispatches on what the directory recorded rather than on a flag. A run
    that quenched gets a glass transition, a run that was deformed gets its
    elastic constants, a run that did both gets both, and a run that did
    neither is an error - which is the same "find the stages by what they
    recorded" rule the readers underneath follow.
    """
    directories = [Path(name) for name in arguments.analyse]
    first = directories[0]
    quenched = _has_stages(first, quench_stages)
    deformed = any(
        _has_stages(first, find) for find in (deform_stages, load_stages, shear_stages)
    )
    if not quenched and not deformed:
        print(
            f"nothing in {first} was a quench or a deformation, so there is "
            "nothing to report",
            flush=True,
        )
        return 1

    if quenched:
        _analyse_tg(arguments, directories)
    if deformed:
        _analyse_mechanics(arguments, first)
    return 0


def _analyse_mechanics(arguments: argparse.Namespace, run_dir: Path) -> None:
    """Report the elastic constants, and write them out."""
    report = analyse_mechanics(run_dir)
    if report.youngs is not None:
        rate = report.youngs.strain_rate_per_ns
        print(
            f"E = {report.youngs.modulus_mpa:.0f} MPa"
            + (
                ""
                if report.replica_spread_mpa is None
                else f" +/- {report.replica_spread_mpa:.0f} over "
                f"{len(report.replicas)} replicas"
            )
            + (" at rate unknown" if rate is None else f" at {rate:.3g} strain/ns")
            + f", {report.youngs.temperature_k:.0f} K"
            + ("" if report.youngs.resolved else " (not resolved)"),
            flush=True,
        )
    if report.poisson is not None:
        print(
            f"nu = {report.poisson.ratio:.3f}"
            f"{'' if report.poisson.resolved else ' (not resolved)'}",
            flush=True,
        )
    for label, fit in (("K", report.bulk), ("G", report.shear)):
        if fit is not None:
            print(
                f"{label} = {fit.modulus_mpa:.0f} MPa"
                f"{'' if fit.resolved else ' (not resolved)'}",
                flush=True,
            )
    if report.load_modulus is not None:
        print(
            f"constant-stress cross-check: "
            f"E = {report.load_modulus.modulus_mpa:.0f} MPa",
            flush=True,
        )
    if report.consistency is not None:
        print(_consistency_line(report.consistency), flush=True)
    for note in report.notes:
        print(f"note: {note}", flush=True)

    files = write_mechanical_report(
        report,
        arguments.output_dir,
        figures=not arguments.no_figures,
        figure_format=cast(str, arguments.figure_format),
    )
    print(
        f"wrote {files.json} and {len(files.figures)} figure(s)",
        flush=True,
    )


def _analyse_tg(arguments: argparse.Namespace, directories: Sequence[Path]) -> None:
    """Report the glass transition from finished run directories."""
    report = analyse_run(
        directories[0],
        extra_run_dirs=directories[1:],
        melt_stage=None if arguments.no_melt_check else arguments.melt_stage,
        min_points_per_branch=int(arguments.min_points_per_branch),
        target_rate_k_per_ns=float(arguments.target_rate),
        radius_of_gyration_nm=arguments.rg,
    )
    print(
        "quenches: "
        + ", ".join(
            f"{curve.stage} ({curve.temperature_step_k:.0f} K steps, "
            + (
                "rate unknown"
                if curve.cooling_rate_k_per_ns is None
                else f"{curve.cooling_rate_k_per_ns:.2f} K/ns"
            )
            + ")"
            for curve in report.curves
        ),
        flush=True,
    )
    if report.melt is not None:
        settled = (
            "volume settled" if report.melt.volume_settled else "volume still drifting"
        )
        moved = "chains moved" if report.melt.chains_moved else "chains have not"
        print(f"melt {report.melt.stage}: {settled}; {moved}", flush=True)
        for reason in report.melt.unchecked:
            print(f"  unchecked: {reason}", flush=True)
    for label, transition in (("coarse", report.coarse), ("fine", report.fine)):
        if transition is not None:
            print(_transition_line(label, transition), flush=True)
    for extrapolation in (report.log_linear, report.vft):
        if extrapolation is not None:
            print(_extrapolation_line(extrapolation), flush=True)
    for note in report.notes:
        print(f"note: {note}", flush=True)

    files = write_report(
        report,
        arguments.output_dir,
        figures=not arguments.no_figures,
        figure_format=cast(str, arguments.figure_format),
    )
    print(
        f"wrote {files.json} and {len(files.figures)} figure(s)",
        flush=True,
    )


def _transition_line(label: str, fit: Any) -> str:
    """One line for a fitted transition, with both expansivities."""
    rate = (
        "rate unknown"
        if fit.cooling_rate_k_per_ns is None
        else f"{fit.cooling_rate_k_per_ns:.2f} K/ns"
    )
    if not fit.resolved:
        return f"{label}: no clear transition at {rate}"
    return (
        f"{label}: Tg = {fit.temperature_k:.0f} K at {rate}, aV "
        f"{fit.melt_expansivity_per_k:.2e} / {fit.glass_expansivity_per_k:.2e} per K"
    )


if __name__ == "__main__":
    raise SystemExit(main())
