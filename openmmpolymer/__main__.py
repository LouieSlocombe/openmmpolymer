"""Command-line driver for a whole polymer melt run.

Flat flags rather than subcommands: the pipeline is one path from a monomer
SMILES to an equilibrated cell, and every stage of it wants the same handful of
facts. Anything more selective is better done from Python, where the four
layers - chain, force field, packing, protocol - are separately callable.

``--analyse`` reads a finished run without rebuilding it. ``--protocol tm``
heats an explicitly supplied crystal and serialised System: packing an
amorphous melt from a monomer cannot provide a crystalline melting point.

The flags each protocol takes are a table, :data:`PROTOCOLS`, rather than a
chain of conditionals, because the alternative failed quietly: a flag that no
protocol took was parsed, ignored, and never reached the run. From the table
every run is checked and priced before anything is built. Every flag but the
few saying where and how a run goes is also part of what a rerun must
repeat - :func:`main` records the parsed namespace beside the build - so no
destination or default can change without leaving the runs already on disk
unable to resume.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

from ._cli_reports import (
    _breaking_lines,
    _convergence_lines,
    _elongation_lines,
    _extrapolation_line,
    _melting_lines,
    _modulus_lines,
    _print_chains,
    _rate_lines,
    _relaxation_lines,
    _say,
    _structure_lines,
    _tg_lines,
    _unresolved,
    _yield_lines,
)
from ._files import ReportFiles
from ._fitting import MAX_EXTRAPOLATION_DECADES, PS_PER_NS
from ._seeds import DEFAULT_SEED
from ._workflow import chain_options
from .chain import ChainResult, ChainSpec
from .charges import CHARGE_METHODS, DEFAULT_CHARGE_METHOD
from .convergence import DEFAULT_WINDOW_FRACTIONS
from .convergence_report import analyse_convergence, write_convergence_report
from .elasticity import DEFAULT_STRAIN_LIMIT, deform_stages, load_stages, shear_stages
from .forcefield import BACKENDS, DEFAULT_BACKEND
from .mechanical import (
    ModulusSpec,
    analyse_mechanics,
    mechanical_scan,
    run_modulus_scan,
    write_mechanical_report,
)
from .melt import build_melt
from .packing import DEFAULT_PACKING_DENSITY
from .property_rates import (
    RATE_PROPERTIES,
    analyse_property_rates,
    default_rate_spec,
    run_property_rate_scan,
    validate_property_rate_scan,
)
from .protocols import (
    Protocol,
    ProtocolError,
    melt_quench,
    record_build_request,
    run_protocol,
    standard_melt_equilibration,
)
from .rate_dependence import validate_rate_request
from .rate_reports import write_rate_report
from .relaxation import relax_stages
from .reporters import TrajectoryOptions
from .simulate import RELAX_MODES, RunContext
from .strength import (
    DEFAULT_CONFIRMATION_STEPS,
    DEFAULT_FAILURE_FRACTION,
    DEFAULT_FIT_MAX_STRAIN,
    DEFAULT_FIT_MIN_STRAIN,
    DEFAULT_OFFSET_STRAIN,
)
from .structure import (
    MAX_DISTRIBUTION_FRAMES,
    MAX_STRUCTURE_FACTOR_FRAMES,
    analyse_structure,
    structure_stages,
    write_structure_report,
)
from .tensile import (
    BreakingSpec,
    ElongationSpec,
    TensileSpec,
    YieldSpec,
    analyse_breaking,
    analyse_elongation,
    analyse_yield,
    breaking_stages,
    elongation_stages,
    run_breaking_scan,
    run_elongation_scan,
    run_yield_scan,
    tensile_scan,
    write_breaking_report,
    write_elongation_report,
    write_yield_report,
    yield_stages,
)
from .tg import (
    TgSpec,
    analyse_tg,
    cooling_rate_series,
    fine_window,
    run_tg_scan,
    tg_scan_duration_ps,
    write_tg_report,
)
from .timeseries import (
    DSC_COOLING_RATE_K_PER_NS,
    EXTRAPOLATION_FORMS,
    cooling_rate_extrapolation,
    minimum_cooling_rates,
    quench_stages,
)
from .tm import (
    TmSpec,
    analyse_melting,
    heating_stages,
    load_crystal,
    melting_scan,
    run_tm_scan,
    write_melting_report,
)
from .trajectory import AnalysisError
from .viscoelastic import (
    RelaxationSpec,
    analyse_relaxation,
    relaxation_scan,
    run_relaxation_scan,
    write_relaxation_report,
)

#: What is reported as a usage error rather than a traceback: a request the
#: library refuses before running anything. Every error class the library
#: defines is a RuntimeError.
_REFUSED = (OSError, ValueError, TypeError, RuntimeError)

# --------------------------------------------------------------------------
# The protocols, and the flags each one takes
# --------------------------------------------------------------------------


def _keywords(
    arguments: argparse.Namespace, fields: Mapping[str, str]
) -> dict[str, Any]:
    """The flags *fields* names, as the keywords they are passed as.

    A flag left unset is left out, so that the library's own default applies.
    """
    values = {keyword: getattr(arguments, dest) for dest, keyword in fields.items()}
    return {keyword: value for keyword, value in values.items() if value is not None}


def _npt_trajectory(interval_ps: float | None) -> dict[str, TrajectoryOptions]:
    """The trajectory the melt's last stage keeps when --check-melt asks."""
    if interval_ps is None:
        return {}
    return {"npt_trajectory": TrajectoryOptions("xtc", interval_ps=interval_ps)}


def _settle(arguments: argparse.Namespace) -> dict[str, Any]:
    """How a scan settles its melt first: how hot, and what it keeps."""
    return {
        "melt_temperature_k": arguments.melt_temperature,
        **_npt_trajectory(arguments.check_melt),
    }


def _plain(
    factory: Callable[..., Protocol],
    *,
    check_melt: float | None = None,
    **keywords: Any,
) -> Protocol:
    """A plain protocol, made straight from its factory's keywords."""
    return factory(**keywords, **_npt_trajectory(check_melt))


#: The passes beside the extension, and the spec field that configures each.
_PASSES = {
    "load": "load_stresses_bar",
    "bulk": "bulk_pressures_bar",
    "shear": "shear_strains",
}


def _modulus_spec(*, skip: Sequence[str] = (), **fields: Any) -> ModulusSpec:
    """The mechanical scan less the passes ``--skip`` names.

    Naming a pass is clearer than handing the flag that configures it an
    empty list.
    """
    return ModulusSpec(**{**fields, **{_PASSES[name]: None for name in skip}})


def _stages_ps(protocol: Protocol, arguments: argparse.Namespace) -> float:
    """A plain protocol lists every stage it runs itself."""
    return protocol.total_duration_ps


def _listed(
    listing: Callable[[Any], Protocol],
) -> Callable[[Any, argparse.Namespace], float]:
    """Price a scan by the library's listing of every stage it runs."""
    return lambda spec, arguments: listing(spec).total_duration_ps


def _tg_ps(spec: TgSpec, arguments: argparse.Namespace) -> float:
    """The coarse pass, and each fine pass as long as its window will be.

    The window's place waits on the coarse fit, but not its size; each rate
    of ``--cooling-rates`` holds each fine temperature for its own time.
    """
    if arguments.tg_approx is not None:
        fine_window(arguments.tg_approx, spec)
    rates = arguments.cooling_rates
    if (
        rates is not None
        and arguments.rate_form == "vft"
        and len(rates) < minimum_cooling_rates(arguments.rate_form)
    ):
        raise ValueError("--rate-form vft requires at least three --cooling-rates.")
    return tg_scan_duration_ps(spec, rates_k_per_ns=rates)


class _Job(NamedTuple):
    """A protocol ready to run: its flags and settings, its cell, and where."""

    arguments: argparse.Namespace
    spec: Any
    run: RunContext
    output: Path
    options: dict[str, Any]

    def scan(self, scan: Callable[..., Any], **options: Any) -> Any:
        """Run a ``run_*_scan`` on this cell, with these settings."""
        return scan(self.run, self.output, spec=self.spec, **self.options, **options)

    def report(
        self,
        report: Any,
        lines: Callable[[Any], Iterable[str]],
        write: Callable[..., ReportFiles],
    ) -> None:
        """Say what a scan found, and write its report where the scan ran."""
        _emit(self.arguments, report, lines, write, None)


def _run_protocol(job: _Job) -> None:
    """Run a plain protocol and say what it did."""
    summary = run_protocol(job.spec, job.run, job.output, **job.options)
    print(
        f"{summary.protocol}: {len(summary.results)} stages in "
        f"{summary.wall_seconds / 60:.1f} min, manifest {summary.manifest_path}"
    )
    _print_chains(summary.chains)


def _run_tg(job: _Job) -> None:
    """Run a two-pass glass-transition scan, at one rate or several."""
    arguments = job.arguments
    rates = arguments.cooling_rates
    if rates is None:
        result = job.scan(run_tg_scan, tg_approx_k=arguments.tg_approx)
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
            f"K/ns (coarse said {coarse}), started from the {result.restart} state"
        )
        _print_chains(result.fine_summary.chains)
        return
    transitions = job.scan(
        cooling_rate_series, rates_k_per_ns=rates, tg_approx_k=arguments.tg_approx
    )
    for rate, fit in zip(rates, transitions, strict=True):
        print(
            f"tg: {fit.temperature_k:.0f} K at {rate:g} K/ns{_unresolved(fit.resolved)}"
        )
    extrapolation = cooling_rate_extrapolation(
        transitions,
        target_rate_k_per_ns=arguments.target_rate,
        form=arguments.rate_form,
    )
    print(_extrapolation_line(extrapolation))


def _run_tm(job: _Job) -> None:
    """Heat the supplied crystal, then report its apparent melting interval."""
    job.report(job.scan(run_tm_scan).report, _melting_lines, write_melting_report)


def _run_modulus(job: _Job) -> None:
    """Measure the elastic constants, and say what qualifies each one."""
    _say(_modulus_lines(job.scan(run_modulus_scan)))


def _run_breaking(job: _Job) -> None:
    """Measure the apparent tensile strength, then report it."""
    job.report(job.scan(run_breaking_scan), _breaking_lines, write_breaking_report)


def _run_elongation(job: _Job) -> None:
    """Measure the apparent elongation at break, then report it."""
    job.report(
        job.scan(run_elongation_scan), _elongation_lines, write_elongation_report
    )


def _run_yield(job: _Job) -> None:
    """Measure the apparent offset yield strength, then report it."""
    job.report(job.scan(run_yield_scan), _yield_lines, write_yield_report)


def _run_relaxation(job: _Job) -> None:
    """Strain the cell once, watch the stress decay, and say what it decayed to."""
    _say(_relaxation_lines(job.scan(run_relaxation_scan)))


@dataclass(frozen=True)
class ProtocolEntry:
    """A protocol the command line offers, and how its flags become a run.

    Args:
        spec: Makes the protocol's settings from the keywords *fields* names:
            a spec, or for the two plain protocols the protocol itself.
        fields: Each flag's destination, and the keyword it is passed as.
        defaults: The flags whose defaults depend on the protocol, filled in
            once parsing is done, so that an explicit flag always wins.
        price: The dynamics the whole run holds, in ps, known before any of
            it is built.
        run: Runs the protocol on a prepared cell and says what it found.
        rate_property: What a rate scan with this protocol measures unless
            ``--rate-property`` names another.
        settles: Whether the scan settles a melt on the way, taking the
            equilibration's keywords to do it. The plain protocols and tg
            take the same flags as settings of their own.
    """

    spec: Callable[..., Any]
    fields: Mapping[str, str]
    defaults: Mapping[str, float]
    price: Callable[[Any, argparse.Namespace], float]
    run: Callable[[_Job], None]
    rate_property: str | None = None
    settles: bool = False

    def settings(self, arguments: argparse.Namespace) -> Any:
        """The settings the flags ask for, checked as they are made."""
        return self.spec(**_keywords(arguments, self.fields))


#: Where a melt settles, for the protocols that settle one as their own work.
_SETTLED = {
    "temperature": "target_temperature_k",
    "melt_temperature": "melt_temperature_k",
    "pressure": "pressure_bar",
    "check_melt": "check_melt",
}

#: Where a measurement is made, and the budget it is made within.
_MEASURED = {
    "temperature": "temperature_k",
    "pressure": "pressure_bar",
    "deform_axis": "axis",
    "max_total_ns": "max_total_ns",
}

#: The ladder every tensile measurement walks, each under its own prefix.
_LADDER = {
    "strain_increment": "strain_increment",
    "max_strain": "max_strain",
    "relax_ps": "relax_ps",
    "replicas": "n_replicas",
    "samples_per_step": "samples_per_step",
    "stage_ps": "stage_ps",
    "trajectory_ps": "trajectory_ps",
}


def _tensile(prefix: str, **criterion: str) -> dict[str, str]:
    """A tensile measurement's flags: where it is made, its ladder, its criterion."""
    ladder = {f"{prefix}_{flag}": field for flag, field in _LADDER.items()}
    return {**_MEASURED, **ladder, **criterion}


_FAILURE = {
    "failure_fraction": "failure_fraction",
    "confirmation_steps": "confirmation_steps",
}

#: The defaults that depend on the protocol. A cooling ladder steps 20 K,
#: held 200 ps, down to 200 K; a heating ladder 10 K, held 1000 ps, from 250 K
#: to 650 K. A measurement is made at 450 K, a tensile one at 298.15 K.
_COOLING = {"t_end": 200.0, "step_k": 20.0, "hold_ps": 200.0, "temperature": 450.0}
_ROOM = {**_COOLING, "temperature": 298.15}
_HEATING = {
    "t_start": 250.0,
    "t_end": 650.0,
    "step_k": 10.0,
    "hold_ps": 1000.0,
    "temperature": 450.0,
}

#: The protocols the command line offers. The flags' defaults apply, not the
#: specs', and seven differ on purpose: a modulus or a relaxation is measured
#: at 450 K rather than 298.15 K; tg settles its melt at 600 K rather than
#: 650 K and screens it in 20 K steps held 200 ps down to 200 K rather than
#: 25 K, 1000 ps and 150 K; and a tm fit keeps four points a branch, not three.
PROTOCOLS = {
    "equilibrate": ProtocolEntry(
        partial(_plain, standard_melt_equilibration),
        _SETTLED,
        _COOLING,
        _stages_ps,
        _run_protocol,
    ),
    "melt-quench": ProtocolEntry(
        partial(_plain, melt_quench),
        {
            **_SETTLED,
            "t_start": "t_start",
            "t_end": "t_end",
            "step_k": "step_k",
            "hold_ps": "hold_ps",
        },
        _COOLING,
        _stages_ps,
        _run_protocol,
    ),
    "tg": ProtocolEntry(
        TgSpec,
        {
            "melt_temperature": "melt_temperature_k",
            "pressure": "pressure_bar",
            "t_end": "t_floor_k",
            "step_k": "coarse_step_k",
            "hold_ps": "coarse_hold_ps",
            "fine_step_k": "fine_step_k",
            "fine_hold_ps": "fine_hold_ps",
            "fine_window_k": "window_k",
            "min_points_per_branch": "min_points_per_branch",
            "check_melt": "npt_trajectory_ps",
            "max_total_ns": "max_total_ns",
        },
        _COOLING,
        _tg_ps,
        _run_tg,
        rate_property="glass_transition",
    ),
    "tm": ProtocolEntry(
        TmSpec,
        {
            "t_start": "t_start_k",
            "t_end": "t_end_k",
            "step_k": "step_k",
            "hold_ps": "hold_ps",
            "pressure": "pressure_bar",
            "tm_equilibration_ps": "equilibration_ps",
            "tm_stage_ps": "stage_ps",
            "tm_trajectory_ps": "trajectory_ps",
            "tm_barostat": "barostat",
            "min_points_per_branch": "min_points_per_branch",
            "max_total_ns": "max_total_ns",
        },
        _HEATING,
        _listed(melting_scan),
        _run_tm,
        rate_property="melting_temperature",
    ),
    "modulus": ProtocolEntry(
        _modulus_spec,
        {
            **_MEASURED,
            "strain_increment": "strain_increment",
            "max_strain": "max_strain",
            "relax_ps": "relax_ps",
            "elastic_strain_limit": "elastic_strain_limit",
            "replicas": "n_replicas",
            "load_stresses": "load_stresses_bar",
            "bulk_pressures": "bulk_pressures_bar",
            "shear_strains": "shear_strains",
            "skip": "skip",
        },
        _COOLING,
        _listed(mechanical_scan),
        _run_modulus,
        rate_property="youngs_modulus",
        settles=True,
    ),
    "breaking": ProtocolEntry(
        BreakingSpec,
        _tensile("breaking", **_FAILURE),
        _ROOM,
        _listed(tensile_scan),
        _run_breaking,
        rate_property="breaking_strength",
        settles=True,
    ),
    "elongation": ProtocolEntry(
        ElongationSpec,
        _tensile("elongation", **_FAILURE),
        _ROOM,
        _listed(tensile_scan),
        _run_elongation,
        rate_property="elongation_at_break",
        settles=True,
    ),
    "yield": ProtocolEntry(
        YieldSpec,
        _tensile(
            "yield",
            yield_offset_strain="offset_strain",
            yield_fit_min_strain="fit_min_strain",
            yield_fit_max_strain="fit_max_strain",
        ),
        _ROOM,
        _listed(tensile_scan),
        _run_yield,
        rate_property="yield_strength",
        settles=True,
    ),
    "relax": ProtocolEntry(
        RelaxationSpec,
        {
            **_MEASURED,
            "relax_mode": "mode",
            "step_strain": "step_strain",
            "baseline_ps": "baseline_ps",
            "relaxation_ps": "relax_ps",
            "relax_replicas": "n_replicas",
            "sample_every_ps": "sample_every_ps",
            "bins_per_decade": "bins_per_decade",
            "relax_stage_ps": "stage_ps",
            "linearity_strains": "linearity_strains",
        },
        _COOLING,
        _listed(relaxation_scan),
        _run_relaxation,
        settles=True,
    ),
}

#: The chain every melt protocol builds, from the flags that describe it.
_CHAIN = {
    "monomer": "monomer_smiles",
    "degree_of_polymerization": "degree_of_polymerization",
    "residue_name": "residue_name",
    "tacticity": "tacticity",
    "head_cap": "head_cap",
    "tail_cap": "tail_cap",
    "characteristic_ratio": "characteristic_ratio",
    "seed": "seed",
}

# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


class _ProtocolParser(argparse.ArgumentParser):
    """Fill in the defaults that depend on the protocol, after every flag."""

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: Any = None,
    ) -> Any:
        arguments = super().parse_args(args, namespace)
        for name, value in PROTOCOLS[arguments.protocol].defaults.items():
            if getattr(arguments, name) is None:
                setattr(arguments, name, value)
        return arguments


def _flag(
    group: argparse._ActionsContainer,
    names: str | tuple[str, ...],
    default: Any,
    help: str,
    **options: Any,
) -> None:
    """Add a flag that takes its default's type, and whose help ends with it."""
    if not isinstance(default, str):
        options.setdefault("type", type(default))
    group.add_argument(
        *((names,) if isinstance(names, str) else names),
        default=default,
        help=f"{help} (default: %(default)s)",
        **options,
    )


def _tensile_flags(
    group: argparse._ActionsContainer, prefix: str, defaults: TensileSpec
) -> None:
    """One tensile measurement's ladder flags, each defaulting to its spec's."""
    flag = f"--{prefix}"
    _flag(
        group,
        f"{flag}-strain-increment",
        defaults.strain_increment,
        "fractional extension of the current cell at each step; increments compound",
    )
    _flag(
        group,
        f"{flag}-max-strain",
        defaults.max_strain,
        f"engineering strain to reach; {defaults.max_strain:g} means "
        f"{100 * defaults.max_strain:g}%%",
    )
    _flag(
        group, f"{flag}-relax-ps", defaults.relax_ps, "hold after each extension, in ps"
    )
    _flag(
        group,
        f"{flag}-replicas",
        defaults.n_replicas,
        "extensions from the equilibrated cell with fresh velocities",
    )
    _flag(
        group,
        f"{flag}-samples-per-step",
        defaults.samples_per_step,
        "stress readings per extension",
    )
    _flag(
        group,
        f"{flag}-stage-ps",
        defaults.stage_ps,
        "duration of each resumable extension chunk, in ps",
    )
    group.add_argument(
        f"{flag}-trajectory-ps",
        type=float,
        help="save extension coordinates every this many ps; omitted by default",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = _ProtocolParser(
        prog="openmmpolymer",
        description="Build, pack and equilibrate an all-atom polymer melt.",
    )
    parser.add_argument(
        "monomer",
        nargs="?",
        help="monomer SMILES with two [*] attachment points, e.g. '[*]CC[*]'. "
        "Required unless --analyse or --protocol tm is given",
    )
    _flag(
        parser,
        ("-n", "--degree-of-polymerization"),
        20,
        "repeat units per chain",
        type=_positive_int,
    )
    _flag(parser, ("-c", "--chains"), 30, "chains in the cell", type=_positive_int)
    _flag(
        parser,
        ("-r", "--residue-name"),
        "POL",
        "residue name, at most three characters",
    )
    parser.add_argument(
        "--head-cap", help="head end-cap SMILES with one [*] attachment point"
    )
    parser.add_argument(
        "--tail-cap",
        help="tail end-cap SMILES with one [*] attachment point, e.g. '[*]O' for PLA",
    )
    parser.add_argument(
        "--characteristic-ratio",
        type=_positive_float,
        help="polymer C-infinity used for chain growth and dimension checks "
        "(default: 7.0 for building; recorded value for analysis)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        help="target temperature in kelvin (default: 298.15 for breaking, "
        "elongation and yield, 450 for other protocols)",
    )
    _flag(
        parser, "--melt-temperature", 600.0, "temperature the chains are mobilised at"
    )
    _flag(parser, "--pressure", 1.0, "pressure in bar")
    _flag(
        parser,
        "--pack-density",
        DEFAULT_PACKING_DENSITY,
        "density to pack at in g/cm3, before compression",
        type=_positive_float,
    )
    _flag(
        parser,
        "--target-density",
        0.85,
        "density the cell is expected to reach, checked against the cutoff "
        "before anything long starts",
        type=_positive_float,
    )
    _flag(
        parser,
        "--charge-method",
        DEFAULT_CHARGE_METHOD,
        "partial-charge method",
        choices=CHARGE_METHODS,
    )
    _flag(
        parser,
        "--backend",
        DEFAULT_BACKEND,
        "forcefill parameterisation backend",
        choices=BACKENDS,
    )
    _flag(parser, "--protocol", "equilibrate", "what to run", choices=sorted(PROTOCOLS))
    quench = parser.add_argument_group(
        "temperature ladder",
        "cooling for melt-quench/tg, heating from a crystal for tm",
    )
    # Their defaults depend on the protocol, so are filled in after parsing.
    for flag, text in (
        ("--t-start", "starting temperature (default: melt temperature; tm: 250 K)"),
        ("--t-end", "ending temperature (default: 200 K; tm: 650 K)"),
        ("--step-k", "temperature change per step (default: 20 K; tm: 10 K)"),
        ("--hold-ps", "time held at each temperature (default: 200 ps; tm: 1000 ps)"),
    ):
        quench.add_argument(flag, type=float, help=text)
    _flag(quench, "--fine-step-k", 5.0, "temperature drop per step in the tg fine pass")
    _flag(quench, "--fine-hold-ps", 3000.0, "time held at each fine temperature")
    _flag(
        quench,
        "--fine-window-k",
        60.0,
        "half-width of the fine window around the coarse transition",
    )
    quench.add_argument(
        "--tg-approx",
        type=float,
        help="centre the fine window here instead of on the coarse fit",
    )
    quench.add_argument(
        "--cooling-rates",
        type=_rates,
        help="comma-separated rates in K/ns to repeat the fine pass at, e.g. "
        "10,5,2; the transition is then extrapolated toward experiment",
    )
    quench.add_argument(
        "--max-total-ns",
        type=_positive_float,
        help="refuse to start a run longer than this many ns",
    )
    quench.add_argument(
        "--check-melt",
        type=_positive_float,
        nargs="?",
        const=10.0,
        help="write a trajectory every N ps during equilibration so the melt "
        "can be shown to have relaxed (default interval 10 ps). This is "
        "frames of the whole cell - tens of megabytes - and without it the "
        "chain half of that check has nothing to read",
    )
    melting = parser.add_argument_group(
        "melting",
        "tm requires a prepared crystalline or semicrystalline periodic cell",
    )
    melting.add_argument(
        "--crystal-pdb",
        help="crystalline starting PDB, with periodic box and System atom order",
    )
    melting.add_argument(
        "--system-xml",
        help="serialised OpenMM System for --crystal-pdb, without a thermostat "
        "or barostat",
    )
    melting.add_argument(
        "--state-in",
        help="optional serialised OpenMM State for the same crystalline cell",
    )
    _flag(
        melting,
        "--tm-equilibration-ps",
        1000.0,
        "equilibrate the crystal at --t-start for this long",
    )
    _flag(melting, "--tm-stage-ps", 10_000.0, "maximum heating stage duration in ps")
    melting.add_argument(
        "--tm-trajectory-ps",
        type=float,
        help="optional trajectory interval in ps, to inspect loss of crystal order",
    )
    _flag(
        melting,
        "--tm-barostat",
        "anisotropic",
        "pressure control for the crystal and heating",
        choices=("isotropic", "anisotropic"),
    )
    _flag(
        parser,
        "--tacticity",
        "atactic",
        "backbone stereochemistry",
        choices=("atactic", "isotactic", "syndiotactic"),
    )
    parser.add_argument(
        "--platform", help="OpenMM platform (default: the fastest that works)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="where everything is written (default: run). With --analyse, "
        "where the report goes instead of <RUN_DIR>/analysis",
    )
    parser.add_argument(
        "--conformers",
        type=_positive_int,
        help="distinct conformations to build; they are repeated to fill the "
        "cell (default: one per chain, which is the right thing and the "
        "slowest to pack)",
    )
    _flag(parser, "--seed", DEFAULT_SEED, "master random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build, charge, parameterise and pack, but run no dynamics",
    )
    mechanics = parser.add_argument_group(
        "mechanics",
        "the extension the modulus protocol walks, and the passes beside it",
    )
    _flag(
        mechanics,
        "--strain-increment",
        0.002,
        "fractional extension of the current cell at each step; increments compound",
    )
    _flag(
        mechanics,
        "--max-strain",
        0.05,
        "engineering strain to reach; the last increment may pass it",
    )
    _flag(
        mechanics,
        "--relax-ps",
        50.0,
        "time to relax after each increment; the mean is over the second "
        "half, so this is twice the averaging window",
    )
    mechanics.add_argument(
        "--modulus-relax-times",
        type=_floats,
        help="the same as --rate-property youngs_modulus --rate-hold-times: "
        "three or more distinct hold times in ps for a Young's modulus rate "
        "scan, e.g. 50,150,500",
    )
    mechanics.add_argument(
        "--target-strain-rate",
        type=float,
        help="the same as --target-property-rate, for a property measured in "
        "strain/ns (multiply a rate in s^-1 by 1e-9); without --rate-property "
        "or a measurement --protocol, the property is youngs_modulus",
    )
    _flag(
        mechanics,
        "--elastic-strain-limit",
        DEFAULT_STRAIN_LIMIT,
        "strain the modulus is fitted up to",
    )
    rate_analysis = parser.add_argument_group(
        "rate sensitivity",
        "compare the same measurement at several imposed loading rates",
    )
    rate_analysis.add_argument(
        "--rate-property",
        choices=sorted(RATE_PROPERTIES),
        help="property to compare; a new scan must use its matching --protocol",
    )
    rate_analysis.add_argument(
        "--rate-hold-times",
        type=_floats,
        help="three or more distinct hold times in ps; varies rate while keeping "
        "the same ladder and preparation",
    )
    rate_analysis.add_argument(
        "--target-property-rate",
        type=float,
        help="positive target in the selected property's units: strain/ns, "
        "bar/ns, or K/ns",
    )
    rate_analysis.add_argument(
        "--max-rate-extrapolation-decades",
        type=float,
        default=MAX_EXTRAPOLATION_DECADES,
        help="largest extrapolation distance allowed to resolve (default: "
        "%(default)s); this is a reporting guard",
    )
    _flag(
        rate_analysis,
        "--thermal-rate-replicas",
        3,
        "independent velocity replicas per thermal rate",
        type=_positive_int,
    )
    _flag(
        mechanics,
        "--replicas",
        3,
        "extensions from the same cell with fresh velocities; their spread is "
        "the error bar",
    )
    _flag(mechanics, "--deform-axis", 2, "axis to stretch", choices=(0, 1, 2))
    mechanics.add_argument(
        "--load-stresses",
        type=_floats,
        help="comma-separated stresses in bar for the constant-stress "
        "cross-check, e.g. 0,100,200,300",
    )
    mechanics.add_argument(
        "--bulk-pressures",
        type=_floats,
        help="comma-separated pressures in bar for the bulk modulus, up and "
        "back down so the hysteresis is measurable",
    )
    mechanics.add_argument(
        "--shear-strains",
        type=_floats,
        help="comma-separated shear strains for the shear modulus",
    )
    mechanics.add_argument(
        "--skip",
        nargs="*",
        default=(),
        choices=tuple(_PASSES),
        help="passes to leave out; the extension always runs",
    )
    breaking = parser.add_argument_group(
        "breaking strength",
        "finite tensile extension and the stress drop that qualifies its peak",
    )
    _tensile_flags(breaking, "breaking", BreakingSpec())
    elongation = parser.add_argument_group(
        "elongation at break",
        "engineering strain at the onset of a confirmed terminal stress drop",
    )
    _tensile_flags(elongation, "elongation", ElongationSpec())
    failure = parser.add_argument_group(
        "tensile failure criterion",
        "shared stress-drop criterion for breaking strength and elongation at break",
    )
    _flag(
        failure,
        "--failure-fraction",
        DEFAULT_FAILURE_FRACTION,
        "fraction of peak nominal stress below which the terminal drop must remain",
    )
    _flag(
        failure,
        "--confirmation-steps",
        DEFAULT_CONFIRMATION_STEPS,
        "consecutive terminal holds needed to confirm the stress drop",
    )
    yielding = parser.add_argument_group(
        "yield strength",
        "finite tensile extension and the offset line defining its proof stress",
    )
    _tensile_flags(yielding, "yield", YieldSpec())
    _flag(
        yielding,
        "--yield-offset-strain",
        DEFAULT_OFFSET_STRAIN,
        "strain offset for the proof-stress line; 0.002 means 0.2%%",
    )
    _flag(
        yielding,
        "--yield-fit-min-strain",
        DEFAULT_FIT_MIN_STRAIN,
        "lower engineering strain for the initial elastic fit",
    )
    _flag(
        yielding,
        "--yield-fit-max-strain",
        DEFAULT_FIT_MAX_STRAIN,
        "upper engineering strain for the initial elastic fit",
    )
    relaxation = parser.add_argument_group(
        "relaxation",
        "the step strain the relax protocol applies, and how the decay after "
        "it is sampled",
    )
    _flag(
        relaxation,
        "--relax-mode",
        "tensile",
        "whether the step is an extension or a shear; both measure G(t), and "
        "a shear imposes no lateral contraction",
        choices=RELAX_MODES,
    )
    _flag(
        relaxation, "--step-strain", 0.03, "the strain applied all at once, then held"
    )
    _flag(
        relaxation,
        "--baseline-ps",
        1000.0,
        "time at the locked box before straining; its scatter is the floor "
        "the decay is read against",
    )
    _flag(
        relaxation,
        "--relaxation-ps",
        10000.0,
        "how long the strain is held, per replica",
    )
    _flag(
        relaxation,
        "--relax-replicas",
        4,
        "independent runs from the same cell with fresh velocities. The first "
        "knob to turn: the early bins hold one reading each, so averaging "
        "replicas is what makes the fast end of the curve mean anything",
    )
    _flag(
        relaxation, "--sample-every-ps", 0.05, "time between stress readings early on"
    )
    _flag(relaxation, "--bins-per-decade", 20, "logarithmic time bins per decade")
    _flag(
        relaxation,
        "--relax-stage-ps",
        20000.0,
        "most relaxation one stage may hold before it is split for resume",
    )
    relaxation.add_argument(
        "--linearity-strains",
        type=_floats,
        help="comma-separated strains to repeat the whole measurement at, "
        "e.g. 0.01,0.05; inside the linear region the moduli coincide, and "
        "there is no other way to check from one strain alone",
    )
    analysis = parser.add_argument_group(
        "analysis", "read a finished run directory instead of building one"
    )
    analysis.add_argument(
        "--analyse",
        nargs="+",
        metavar="RUN_DIR",
        help="report what these finished run directories recorded; the first "
        "owns the output, and the rest add only to its glass transition or "
        "rate series. No monomer is needed",
    )
    _flag(analysis, "--melt-stage", "05_npt", "the equilibration stage to check")
    convergence = parser.add_argument_group(
        "time-window convergence", "check estimates as the observation window grows"
    )
    convergence.add_argument(
        "--convergence",
        action="store_true",
        help="analyse observation-window stability for one --analyse directory",
    )
    convergence.add_argument(
        "--convergence-stage",
        metavar="STAGE",
        help="saved stage to check (default: --structure-stage, else the last "
        "stage with a trajectory, else the last stage)",
    )
    convergence.add_argument(
        "--window-fractions",
        type=_floats,
        default=DEFAULT_WINDOW_FRACTIONS,
        help="increasing observed fractions ending at 1 (default: "
        + ",".join(f"{fraction:g}" for fraction in DEFAULT_WINDOW_FRACTIONS)
        + ")",
    )
    _flag(
        convergence,
        "--convergence-tolerance",
        0.1,
        "maximum relative change to resolve window stability",
    )
    _flag(
        convergence,
        "--min-effective-samples",
        20.0,
        "minimum autocorrelation-adjusted count for scalar time traces",
    )
    _flag(
        convergence,
        "--convergence-discard-fraction",
        0.1,
        "initial fraction discarded from stationary time traces",
    )
    analysis.add_argument(
        "--no-melt-check",
        action="store_true",
        help="skip the melt equilibration check",
    )
    _flag(
        analysis,
        "--rate-form",
        "log_linear",
        "which rate relation a --cooling-rates tg run headlines; --analyse "
        "reports every one the rates can fit; vft needs at least three rates",
        choices=EXTRAPOLATION_FORMS,
    )
    analysis.add_argument(
        "--target-rate",
        type=_positive_float,
        default=DSC_COOLING_RATE_K_PER_NS,
        help="cooling rate in K/ns to extrapolate to (default: 10 K/min)",
    )
    _flag(
        analysis,
        "--min-points-per-branch",
        4,
        "points each branch of the fit must keep",
    )
    analysis.add_argument(
        "--rg", type=float, help="radius of gyration in nm, overriding the manifest"
    )
    analysis.add_argument(
        "--structure-stage",
        metavar="STAGE",
        help="the stage whose coordinates to measure (default: the last stage "
        "with a trajectory, else the last stage's closing snapshot)",
    )
    analysis.add_argument(
        "--backbone",
        type=_ints,
        help="comma-separated backbone atom indices within one chain, e.g. "
        "0,1,4,5; overrides what the run recorded and the bond-graph inference",
    )
    analysis.add_argument(
        "--no-structure",
        action="store_true",
        help="skip the structure and dynamics report",
    )
    analysis.add_argument(
        "--stride",
        type=_positive_int,
        default=1,
        help="measure every Nth frame (default: %(default)s); g(r) and S(q) "
        f"are further capped at {MAX_DISTRIBUTION_FRAMES} and "
        f"{MAX_STRUCTURE_FACTOR_FRAMES} frames",
    )
    analysis.add_argument(
        "--no-figures",
        action="store_true",
        help="write the record but no figures",
    )
    _flag(analysis, "--figure-format", "png", "what to save figures as")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log what each step is doing"
    )
    return parser


def _split[T](
    text: str, number: Callable[[str], T], kind: str, example: str
) -> tuple[T, ...]:
    """A comma-separated list, parsed at the front door."""
    try:
        return tuple(number(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of {kind}, e.g. {example!r}."
        ) from error


def _floats(text: str) -> tuple[float, ...]:
    """A comma-separated list of numbers."""
    values = _split(text, float, "numbers", "0,100,200")
    if not values:
        raise argparse.ArgumentTypeError(
            f"{text!r} is empty. Leave the flag out for the default; to drop a "
            "modulus pass, name it in --skip."
        )
    return values


def _ints(text: str) -> tuple[int, ...]:
    """A comma-separated list of atom indices."""
    values = _split(text, int, "integers", "0,1,4,5")
    if len(values) < 2:
        raise argparse.ArgumentTypeError(
            f"{text!r} names {len(values)} atom(s); a backbone of one atom has no "
            "end-to-end vector."
        )
    return values


def _rates(text: str) -> tuple[float, ...]:
    """A comma-separated list of cooling rates."""
    rates = _split(text, float, "numbers", "10,5,2")
    if len(rates) < 2:
        raise argparse.ArgumentTypeError(
            f"{text!r} gives {len(rates)} rate(s); a rate dependence needs at "
            "least two."
        )
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in rates) or len(
        set(rates)
    ) != len(rates):
        raise argparse.ArgumentTypeError(
            f"{text!r} must be distinct positive finite rates in K/ns."
        )
    return rates


def _positive_float(text: str) -> float:
    """An explicitly finite, positive number."""
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _positive_int(text: str) -> int:
    """A count that has to be at least one."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer.")
    return value


# --------------------------------------------------------------------------
# Rate scans
# --------------------------------------------------------------------------


class _RateRequest(NamedTuple):
    """A rate scan or analysis, resolved from the command line.

    An analysis runs no dynamics and has no hold times.
    """

    property_name: str
    target_rate: float
    hold_times_ps: tuple[float, ...]


#: The flags that ask for a rate scan, or for the analysis of one.
_RATE_FLAGS = (
    "rate_property",
    "rate_hold_times",
    "modulus_relax_times",
    "target_property_rate",
    "target_strain_rate",
)


def _property_rates_requested(arguments: argparse.Namespace) -> bool:
    return any(getattr(arguments, name) is not None for name in _RATE_FLAGS)


def _either(new: Any, old: Any, flags: str) -> Any:
    """A setting given by its flag or by the older one it replaced, never both ways."""
    if new is not None and old is not None and new != old:
        raise ValueError(f"{flags} give different values; pass one of them.")
    return old if new is None else new


def _rate_protocol(property_name: str) -> str:
    """The protocol a rate scan of *property_name* varies the loading rate of.

    It is the one whose own rate property is measured with the same settings.
    """
    settings = type(default_rate_spec(property_name))
    return next(
        name
        for name, entry in PROTOCOLS.items()
        if entry.rate_property is not None
        and type(default_rate_spec(entry.rate_property)) is settings
    )


def _property_rate_request(arguments: argparse.Namespace) -> _RateRequest:
    """Resolve a physical rate unit before allowing any build or dynamics.

    ``--modulus-relax-times`` and ``--target-strain-rate`` predate the other
    properties. They stand for ``--rate-property youngs_modulus
    --rate-hold-times`` and for ``--target-property-rate`` in strain/ns, and
    the target alone still selects Young's modulus when neither a property
    nor a measurement protocol does.
    """
    property_name = arguments.rate_property
    if arguments.modulus_relax_times is not None:
        if property_name not in (None, "youngs_modulus"):
            raise ValueError(
                "--modulus-relax-times sets Young's modulus holds; use "
                f"--rate-hold-times for {property_name}."
            )
        property_name = "youngs_modulus"
    property_name = (
        property_name
        or PROTOCOLS[arguments.protocol].rate_property
        or ("youngs_modulus" if arguments.target_strain_rate is not None else None)
    )
    if property_name is None:
        raise ValueError(
            "Choose --rate-property for rate analysis, or a measurement --protocol for a scan."
        )
    unit = RATE_PROPERTIES[property_name].rate_unit
    if arguments.target_strain_rate is not None and unit != "strain/ns":
        raise ValueError(f"{property_name} uses {unit}; use --target-property-rate.")
    target = _either(
        arguments.target_property_rate,
        arguments.target_strain_rate,
        "--target-property-rate and --target-strain-rate",
    )
    holds = _either(
        arguments.rate_hold_times,
        arguments.modulus_relax_times,
        "--rate-hold-times and --modulus-relax-times",
    )
    if target is None:
        raise ValueError("A finite positive --target-property-rate is required.")
    target, _ = validate_rate_request(target, arguments.max_rate_extrapolation_decades)
    if arguments.analyse:
        if holds is not None:
            raise ValueError(
                "--rate-hold-times and --modulus-relax-times start new dynamics "
                "and cannot be used with --analyse."
            )
        return _RateRequest(property_name, target, ())
    protocol = _rate_protocol(property_name)
    if arguments.protocol != protocol or holds is None:
        raise ValueError(
            f"{property_name} scans require --protocol {protocol} and --rate-hold-times."
        )
    return _RateRequest(property_name, target, holds)


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------

#: What the build request leaves out: where things go, which device runs
#: them, how much is said, and whether - or under what budget - dynamics
#: start. None of them changes the physical system.
_UNRECORDED = frozenset(
    {
        "output_dir",
        "platform",
        "dry_run",
        "verbose",
        "no_figures",
        "figure_format",
        "max_total_ns",
    }
)

#: The dependencies whose versions can change a rebuilt Hamiltonian.
_RUNTIME = ("openmm", "rdkit", "forcefill", "openff-toolkit", "numpy")


@contextmanager
def _refusals(parser: argparse.ArgumentParser) -> Iterator[None]:
    """Show a refused CLI request as a usage error at the dispatch boundary."""
    try:
        yield
    except _REFUSED as error:
        parser.error(str(error))


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
    if isinstance(sys.stdout, io.TextIOWrapper):
        # A long run's progress has to reach a redirected log as it happens.
        sys.stdout.reconfigure(line_buffering=True)

    if arguments.convergence:
        if not arguments.analyse or len(arguments.analyse) != 1:
            parser.error("--convergence requires exactly one --analyse directory")
        if _property_rates_requested(arguments):
            parser.error("run --convergence separately from imposed-rate analysis")
        with _refusals(parser):
            windows = analyse_convergence(
                arguments.analyse[0],
                stage=arguments.convergence_stage or arguments.structure_stage,
                backbone=arguments.backbone,
                stride=arguments.stride,
                window_fractions=arguments.window_fractions,
                relative_tolerance=arguments.convergence_tolerance,
                min_effective_samples=arguments.min_effective_samples,
                discard_fraction=arguments.convergence_discard_fraction,
            )
            _emit(
                arguments,
                windows,
                _convergence_lines,
                write_convergence_report,
                arguments.output_dir,
            )
        return 0

    rates = None
    if _property_rates_requested(arguments):
        with _refusals(parser):
            rates = _property_rate_request(arguments)
            if arguments.analyse:
                saved = analyse_property_rates(
                    arguments.analyse,
                    property_name=rates.property_name,
                    target_rate=rates.target_rate,
                    strain_limit=arguments.elastic_strain_limit,
                    max_extrapolation_decades=arguments.max_rate_extrapolation_decades,
                )
                _emit(
                    arguments,
                    saved,
                    _rate_lines,
                    write_rate_report,
                    arguments.output_dir,
                )
                return 0
    if arguments.analyse:
        if arguments.protocol == "modulus" and len(arguments.analyse) > 1:
            parser.error(
                "analysing multiple modulus rates requires --target-property-rate"
            )
        with _refusals(parser):
            return _analyse(arguments)

    crystalline = arguments.protocol == "tm"
    if crystalline:
        if arguments.monomer is not None:
            parser.error("tm starts from --crystal-pdb, not a monomer SMILES")
        if arguments.crystal_pdb is None or arguments.system_xml is None:
            parser.error("tm requires both --crystal-pdb and --system-xml")
        if arguments.check_melt is not None:
            parser.error("--check-melt watches a melt settle; tm heats a crystal")
    else:
        if any((arguments.crystal_pdb, arguments.system_xml, arguments.state_in)):
            parser.error(
                "--crystal-pdb, --system-xml and --state-in require --protocol tm"
            )
        if arguments.monomer is None:
            parser.error("a monomer SMILES is required unless --analyse is given")
    entry = PROTOCOLS[arguments.protocol]
    # Everything is checked and priced before a file is read or written.
    with _refusals(parser):
        chain = None if crystalline else ChainSpec(**_keywords(arguments, _CHAIN))
        spec = entry.settings(arguments)
        if rates is None:
            total_ns = entry.price(spec, arguments) / PS_PER_NS
            cost = f"{arguments.protocol} run: {total_ns:.3g} ns of dynamics in total"
            budget = arguments.max_total_ns
            if budget is not None and total_ns > budget:
                raise ValueError(
                    f"The {arguments.protocol} run is {total_ns:.3g} ns, over the "
                    f"{budget:g} ns budget. Shorten it or raise --max-total-ns."
                )
        else:
            total_ns = validate_property_rate_scan(
                spec,
                rates.hold_times_ps,
                property_name=rates.property_name,
                target_rate=rates.target_rate,
                n_replicas=arguments.thermal_rate_replicas,
                max_extrapolation_decades=arguments.max_rate_extrapolation_decades,
            ).total_ns
            cost = (
                f"{rates.property_name} rate scan: {total_ns:.3g} ns total "
                "including preparation and all replicas"
            )
    print(cost)

    output = Path(arguments.output_dir or "run")
    if chain is None:
        with _refusals(parser):
            run = load_crystal(
                arguments.crystal_pdb,
                arguments.system_xml,
                state_in=arguments.state_in,
                platform=arguments.platform,
                seed=arguments.seed,
            )
        options: dict[str, Any] = {"state_in": arguments.state_in, "crystalline": True}
    else:
        built, run = _build(parser, arguments, chain, output)
        options = chain_options(
            built.backbone, built.n_atoms, chain.characteristic_ratio
        )
        if entry.settles:
            options.update(_settle(arguments))
    if arguments.dry_run:
        print(
            "dry run: crystal and heating schedule validated; no dynamics"
            if crystalline
            else "dry run: stopping before dynamics"
        )
        return 0

    if rates is None:
        with _refusals(parser):
            entry.run(_Job(arguments, spec, run, output, options))
        return 0
    with _refusals(parser):
        report = run_property_rate_scan(
            run,
            output,
            property_name=rates.property_name,
            target_rate=rates.target_rate,
            spec=spec,
            hold_times_ps=rates.hold_times_ps,
            n_replicas=arguments.thermal_rate_replicas,
            max_extrapolation_decades=arguments.max_rate_extrapolation_decades,
            **options,
        )
        _emit(arguments, report, _rate_lines, write_rate_report, output / "analysis")
    return 0


def _build(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    chain: ChainSpec,
    output: Path,
) -> tuple[ChainResult, RunContext]:
    """Record what was asked for, then build the melt or check a rebuild.

    The request is recorded first: a rebuild into a finished run could
    otherwise overwrite its chemistry before anything refused to resume it.
    """
    request = {
        key: value for key, value in vars(arguments).items() if key not in _UNRECORDED
    }
    request["conformers"] = min(
        arguments.conformers or arguments.chains, arguments.chains
    )
    request["characteristic_ratio"] = chain.characteristic_ratio
    request["runtime_versions"] = {package: version(package) for package in _RUNTIME}
    try:
        record_build_request(output, request)
        return build_melt(
            chain,
            arguments.chains,
            output,
            target_density_g_cm3=arguments.target_density,
            n_conformers=arguments.conformers,
            charge_method=arguments.charge_method,
            backend=arguments.backend,
            pack_density_g_cm3=arguments.pack_density,
            platform=arguments.platform,
            progress=print,
        )
    except ProtocolError as error:
        parser.error(str(error))


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def _emit(
    arguments: argparse.Namespace,
    report: Any,
    lines: Callable[[Any], Iterable[str]],
    write: Callable[..., ReportFiles],
    output_dir: str | Path | None,
) -> None:
    """Print what a report found and its notes, then write it and say where."""
    _say(lines(report))
    _say(f"note: {note}" for note in report.notes)
    files = write(
        report,
        output_dir,
        figures=not arguments.no_figures,
        figure_format=arguments.figure_format,
    )
    print(f"wrote {files.json} and {len(files.figures)} figure(s)")


def _has_stages(run_dir: Path, find: Any) -> bool:
    """Whether a reader finds anything of its kind in this directory.

    Readers can return an empty collection or raise when none are found.
    Both mean that this report does not apply to the directory.
    """
    try:
        return bool(find(run_dir))
    except AnalysisError:
        return False


def _analyse(arguments: argparse.Namespace) -> int:
    """Report whatever a finished run directory holds, and write it out.

    Dispatches on what the directory recorded rather than on a flag. A run
    that quenched gets a glass transition, one that was heated a melting
    interval, a tensile ladder its own measurement, any other deformation its
    elastic constants, a relaxation its decay, and any stage that left
    coordinates a structure report. A run gets every report that applies, and
    a run with none is an error - which is the same "find the stages by what
    they recorded" rule the readers underneath follow. Only a glass transition
    reads the further directories.
    """
    directories = [Path(name) for name in arguments.analyse]
    first = directories[0]
    quenched = _has_stages(first, quench_stages)
    heated = _has_stages(first, heating_stages)
    tensile = [
        (analyse, lines, write)
        for find, analyse, lines, write in (
            (breaking_stages, analyse_breaking, _breaking_lines, write_breaking_report),
            (
                elongation_stages,
                analyse_elongation,
                _elongation_lines,
                write_elongation_report,
            ),
            (yield_stages, analyse_yield, _yield_lines, write_yield_report),
        )
        if _has_stages(first, find)
    ]
    # A tensile ladder deforms the cell too, but measures no elastic constant.
    deformed = not tensile and any(
        _has_stages(first, find) for find in (deform_stages, load_stages, shear_stages)
    )
    relaxed = _has_stages(first, relax_stages)
    structured = not arguments.no_structure and _has_stages(first, structure_stages)
    if not (quenched or heated or tensile or deformed or relaxed or structured):
        print(
            f"nothing in {first} was a quench, a heating scan, a deformation or a relaxation, "
            "and no stage left coordinates to measure, so there is nothing to "
            "report"
        )
        return 1

    def emit(report: Any, lines: Callable[[Any], Iterable[str]], write: Any) -> None:
        _emit(arguments, report, lines, write, arguments.output_dir)

    if quenched:
        tg = analyse_tg(
            first,
            extra_run_dirs=directories[1:],
            melt_stage=None if arguments.no_melt_check else arguments.melt_stage,
            min_points_per_branch=arguments.min_points_per_branch,
            target_rate_k_per_ns=arguments.target_rate,
            radius_of_gyration_nm=arguments.rg,
        )
        emit(tg, _tg_lines, write_tg_report)
    if heated:
        melting = analyse_melting(
            first, min_points_per_branch=arguments.min_points_per_branch
        )
        emit(melting, _melting_lines, write_melting_report)
    for analyse, lines, write in tensile:
        emit(analyse(first), lines, write)
    if deformed:
        emit(analyse_mechanics(first), _modulus_lines, write_mechanical_report)
    if relaxed:
        emit(analyse_relaxation(first), _relaxation_lines, write_relaxation_report)
    if structured:
        structure = analyse_structure(
            first,
            stage=arguments.structure_stage,
            backbone=arguments.backbone,
            expected_characteristic_ratio=arguments.characteristic_ratio,
            stride=arguments.stride,
        )
        emit(structure, _structure_lines, write_structure_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
