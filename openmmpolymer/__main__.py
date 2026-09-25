"""Command-line driver for a whole polymer melt run.

Flat flags rather than subcommands: the pipeline is one path from a monomer
SMILES to an equilibrated cell, and every stage of it wants the same handful of
facts. Anything more selective is better done from Python, where the four
layers - chain, force field, packing, protocol - are separately callable.

``--analyse`` reads a finished run without rebuilding it. ``--protocol tm``
heats an explicitly supplied crystal and serialized System: packing an
amorphous melt from a monomer cannot provide a crystalline melting point.

The flags a protocol accepts are a table rather than a chain of conditionals,
because the alternative failed quietly: a flag that no factory took was parsed,
ignored, and never reached the run. :data:`PROTOCOLS` names what each factory
accepts, and a test checks those names against the factories themselves.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from ._files import ReportFiles, file_sha256, write_json
from .chain import ChainResult, ChainSpec, build_chain
from .charges import CHARGE_METHODS, assign_charges
from .convergence import DEFAULT_WINDOW_FRACTIONS
from .convergence_report import analyse_convergence, write_convergence_report
from .elasticity import deform_stages, load_stages, shear_stages
from .forcefield import BACKENDS, PolymerForceField, build_polymer_forcefield
from .mdsystem import (
    PackedBox,
    SystemSpec,
    assemble_box,
    check_target_density,
    prepare_box,
)
from .mechanical import (
    MechanicalError,
    ModulusSpec,
    analyse_mechanics,
    mechanical_scan,
    run_modulus_scan,
    write_mechanical_report,
)
from .modulus_rate_report import write_modulus_rate_report
from .modulus_rates import (
    ModulusRateReport,
    analyse_modulus_rates,
    run_modulus_rate_scan,
    validate_modulus_rate_scan,
)
from .packing import (
    DEFAULT_PACKING_DENSITY,
    box_edge_nm,
    check_packing,
    distribute_conformers,
    pack_box,
)
from .property_rates import (
    RATE_PROPERTIES,
    RateScanSpec,
    analyse_property_rates,
    default_rate_spec,
    run_property_rate_scan,
    validate_property_rate_scan,
)
from .protocols import (
    Protocol,
    ProtocolError,
    _run_identity,
    melt_quench,
    record_build_request,
    run_protocol,
    standard_melt_equilibration,
    validate_run_inputs,
)
from .rate_dependence import RateReport
from .rate_reports import write_rate_report
from .relaxation import relax_stages
from .simulate import RELAX_MODES, RunContext, prepare_run
from .structure import analyse_structure, structure_stages, write_structure_report
from .tensile import (
    BreakingError,
    BreakingSpec,
    ElongationError,
    ElongationSpec,
    YieldError,
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
    run_tg_scan,
    tg_coarse_scan,
    write_tg_report,
)
from .timeseries import (
    DSC_COOLING_RATE_K_PER_NS,
    EXTRAPOLATION_FORMS,
    cooling_rate_extrapolation,
    quench_stages,
)
from .tm import (
    TmError,
    TmSpec,
    analyse_melting,
    heating_stages,
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


#: Finite extension and the sustained stress drop used to qualify its peak.
_BREAKING = (
    "temperature_k",
    "pressure_bar",
    "deform_axis",
    "breaking_strain_increment",
    "breaking_max_strain",
    "breaking_relax_ps",
    "breaking_replicas",
    "breaking_samples_per_step",
    "breaking_stage_ps",
    "breaking_trajectory_ps",
    "failure_fraction",
    "confirmation_steps",
    "max_total_ns",
)


#: Finite extension and the sustained stress drop defining apparent elongation at break.
_ELONGATION = (
    "temperature_k",
    "pressure_bar",
    "deform_axis",
    "elongation_strain_increment",
    "elongation_max_strain",
    "elongation_relax_ps",
    "elongation_replicas",
    "elongation_samples_per_step",
    "elongation_stage_ps",
    "elongation_trajectory_ps",
    "failure_fraction",
    "confirmation_steps",
    "max_total_ns",
)


#: Finite extension and the elastic fit defining an offset proof stress.
_YIELD = (
    "temperature_k",
    "pressure_bar",
    "deform_axis",
    "yield_strain_increment",
    "yield_max_strain",
    "yield_relax_ps",
    "yield_replicas",
    "yield_samples_per_step",
    "yield_stage_ps",
    "yield_trajectory_ps",
    "yield_offset_strain",
    "yield_fit_min_strain",
    "yield_fit_max_strain",
    "max_total_ns",
)


#: The step strain, and the pass that checks it was small enough.
_RELAXATION = (
    "temperature_k",
    "pressure_bar",
    "relax_mode",
    "deform_axis",
    "step_strain",
    "baseline_ps",
    "relaxation_ps",
    "relax_replicas",
    "sample_every_ps",
    "bins_per_decade",
    "relax_stage_ps",
    "linearity_strains",
    "max_total_ns",
)


_TM = (
    *_QUENCH,
    "pressure_bar",
    "tm_equilibration_ps",
    "tm_stage_ps",
    "tm_trajectory_ps",
    "tm_barostat",
    "min_points_per_branch",
    "max_total_ns",
)


def _tm_spec(
    *,
    t_start: float = 250.0,
    t_end: float = 650.0,
    step_k: float = 10.0,
    hold_ps: float = 1000.0,
    pressure_bar: float = 1.0,
    tm_equilibration_ps: float = 1000.0,
    tm_stage_ps: float = 10_000.0,
    tm_trajectory_ps: float | None = None,
    tm_barostat: str = "anisotropic",
    min_points_per_branch: int = 3,
    max_total_ns: float | None = None,
) -> TmSpec:
    """Map the heating controls onto a crystalline melting scan."""
    return TmSpec(
        t_start_k=t_start,
        t_end_k=t_end,
        step_k=step_k,
        hold_ps=hold_ps,
        pressure_bar=pressure_bar,
        equilibration_ps=tm_equilibration_ps,
        stage_ps=tm_stage_ps,
        trajectory_ps=tm_trajectory_ps,
        barostat=tm_barostat,
        min_points_per_branch=min_points_per_branch,
        max_total_ns=max_total_ns,
    )


def _tm_protocol(**options: Any) -> Protocol:
    """The complete heating ladder, including crystal equilibration."""
    return melting_scan(_tm_spec(**options))


def _relaxation_spec(
    *,
    temperature_k: float = 298.15,
    pressure_bar: float = 1.0,
    relax_mode: str = "tensile",
    deform_axis: int = 2,
    step_strain: float = 0.03,
    baseline_ps: float = 1000.0,
    relaxation_ps: float = 10_000.0,
    relax_replicas: int = 4,
    sample_every_ps: float = 0.05,
    bins_per_decade: int = 20,
    relax_stage_ps: float = 20_000.0,
    linearity_strains: tuple[float, ...] | None = None,
    max_total_ns: float | None = None,
) -> RelaxationSpec:
    """Turn the flat relaxation flags into the spec a scan takes."""
    return RelaxationSpec(
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        mode=relax_mode,
        axis=deform_axis,
        step_strain=step_strain,
        baseline_ps=baseline_ps,
        relax_ps=relaxation_ps,
        n_replicas=relax_replicas,
        sample_every_ps=sample_every_ps,
        bins_per_decade=bins_per_decade,
        stage_ps=relax_stage_ps,
        linearity_strains=linearity_strains,
        max_total_ns=max_total_ns,
    )


def _relax_protocol(**options: Any) -> Protocol:
    """Every stage of the relaxation scan, so --dry-run can price it."""
    return relaxation_scan(_relaxation_spec(**options))


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
    """Every stage of the mechanical scan, so --dry-run can price it."""
    return mechanical_scan(_modulus_spec(**options))


def _breaking_spec(
    *,
    temperature_k: float = 298.15,
    pressure_bar: float = 1.0,
    deform_axis: int = 2,
    breaking_strain_increment: float = 0.01,
    breaking_max_strain: float = 1.0,
    breaking_relax_ps: float = 50.0,
    breaking_replicas: int = 3,
    breaking_samples_per_step: int = 250,
    breaking_stage_ps: float = 1000.0,
    breaking_trajectory_ps: float | None = None,
    failure_fraction: float = 0.5,
    confirmation_steps: int = 3,
    max_total_ns: float | None = None,
) -> BreakingSpec:
    """Map the finite-extension controls onto the breaking workflow."""
    return BreakingSpec(
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        axis=deform_axis,
        strain_increment=breaking_strain_increment,
        max_strain=breaking_max_strain,
        relax_ps=breaking_relax_ps,
        n_replicas=breaking_replicas,
        samples_per_step=breaking_samples_per_step,
        stage_ps=breaking_stage_ps,
        trajectory_ps=breaking_trajectory_ps,
        failure_fraction=failure_fraction,
        confirmation_steps=confirmation_steps,
        max_total_ns=max_total_ns,
    )


def _breaking_protocol(**options: Any) -> Protocol:
    """Build the equilibration and every replica of the tensile ladder."""
    spec = _breaking_spec(**options)
    return _check_scan_budget(
        tensile_scan(spec), spec.max_total_ns, "breaking", BreakingError
    )


def _elongation_spec(
    *,
    temperature_k: float = 298.15,
    pressure_bar: float = 1.0,
    deform_axis: int = 2,
    elongation_strain_increment: float = 0.01,
    elongation_max_strain: float = 1.0,
    elongation_relax_ps: float = 50.0,
    elongation_replicas: int = 3,
    elongation_samples_per_step: int = 250,
    elongation_stage_ps: float = 1000.0,
    elongation_trajectory_ps: float | None = None,
    failure_fraction: float = 0.5,
    confirmation_steps: int = 3,
    max_total_ns: float | None = None,
) -> ElongationSpec:
    """Map the finite-extension controls onto the elongation workflow."""
    return ElongationSpec(
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        axis=deform_axis,
        strain_increment=elongation_strain_increment,
        max_strain=elongation_max_strain,
        relax_ps=elongation_relax_ps,
        n_replicas=elongation_replicas,
        samples_per_step=elongation_samples_per_step,
        stage_ps=elongation_stage_ps,
        trajectory_ps=elongation_trajectory_ps,
        failure_fraction=failure_fraction,
        confirmation_steps=confirmation_steps,
        max_total_ns=max_total_ns,
    )


def _elongation_protocol(**options: Any) -> Protocol:
    """Build the equilibration and every replica of the tensile ladder."""
    spec = _elongation_spec(**options)
    return _check_scan_budget(
        tensile_scan(spec), spec.max_total_ns, "elongation", ElongationError
    )


def _yield_spec(
    *,
    temperature_k: float = 298.15,
    pressure_bar: float = 1.0,
    deform_axis: int = 2,
    yield_strain_increment: float = 0.002,
    yield_max_strain: float = 0.3,
    yield_relax_ps: float = 50.0,
    yield_replicas: int = 3,
    yield_samples_per_step: int = 250,
    yield_stage_ps: float = 1000.0,
    yield_trajectory_ps: float | None = None,
    yield_offset_strain: float = 0.002,
    yield_fit_min_strain: float = 0.0,
    yield_fit_max_strain: float = 0.02,
    max_total_ns: float | None = None,
) -> YieldSpec:
    """Map the tensile and proof-stress controls onto the yield workflow."""
    return YieldSpec(
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        axis=deform_axis,
        strain_increment=yield_strain_increment,
        max_strain=yield_max_strain,
        relax_ps=yield_relax_ps,
        n_replicas=yield_replicas,
        samples_per_step=yield_samples_per_step,
        stage_ps=yield_stage_ps,
        trajectory_ps=yield_trajectory_ps,
        offset_strain=yield_offset_strain,
        fit_min_strain=yield_fit_min_strain,
        fit_max_strain=yield_fit_max_strain,
        max_total_ns=max_total_ns,
    )


def _yield_protocol(**options: Any) -> Protocol:
    """Validate equilibration and every tensile replica before building a cell."""
    spec = _yield_spec(**options)
    return _check_scan_budget(
        tensile_scan(spec), spec.max_total_ns, "yield", YieldError
    )


def _check_scan_budget(
    protocol: Protocol,
    max_total_ns: float | None,
    name: str,
    error_type: type[Exception],
) -> Protocol:
    """Reject a tensile schedule that exceeds its budget before building a cell."""
    duration_ns = protocol.total_duration_ps / 1000.0
    if max_total_ns is not None and duration_ns > max_total_ns:
        raise error_type(
            f"The {name} scan is {duration_ns:.3g} ns, over the "
            f"{max_total_ns:g} ns budget. Shorten the scan or raise max_total_ns."
        )
    return protocol


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
    "tm": ProtocolEntry(_tm_protocol, _TM),
    "modulus": ProtocolEntry(_modulus_protocol, ("pressure_bar", *_MECHANICS)),
    "breaking": ProtocolEntry(_breaking_protocol, _BREAKING),
    "elongation": ProtocolEntry(_elongation_protocol, _ELONGATION),
    "yield": ProtocolEntry(_yield_protocol, _YIELD),
    "relax": ProtocolEntry(_relax_protocol, _RELAXATION),
}


class _ProtocolParser(argparse.ArgumentParser):
    """Choose protocol-specific defaults after parsing every explicit flag."""

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: Any = None,
    ) -> Any:
        arguments = super().parse_args(args, namespace)
        defaults = {"t_end": 200.0, "step_k": 20.0, "hold_ps": 200.0}
        if arguments.protocol == "tm":
            defaults = {
                "t_start": 250.0,
                "t_end": 650.0,
                "step_k": 10.0,
                "hold_ps": 1000.0,
            }
        defaults["temperature"] = (
            298.15
            if arguments.protocol in ("breaking", "elongation", "yield")
            else 450.0
        )
        for name, value in defaults.items():
            if getattr(arguments, name) is None:
                setattr(arguments, name, value)
        return arguments


def _add_tensile_arguments(
    group: argparse._ArgumentGroup,
    prefix: str,
    defaults: BreakingSpec | ElongationSpec | YieldSpec,
) -> None:
    """Expose the common tensile controls using each workflow's own defaults."""
    group.add_argument(
        f"--{prefix}-strain-increment",
        type=float,
        default=defaults.strain_increment,
        help="fractional extension of the current cell at each step "
        "(default: %(default)s)",
    )
    group.add_argument(
        f"--{prefix}-max-strain",
        type=float,
        default=defaults.max_strain,
        help=f"engineering strain to reach; {defaults.max_strain:g} means "
        f"{100 * defaults.max_strain:g}%% (default: %(default)s)",
    )
    group.add_argument(
        f"--{prefix}-relax-ps",
        type=float,
        default=defaults.relax_ps,
        help="hold after each extension, in ps (default: %(default)s)",
    )
    group.add_argument(
        f"--{prefix}-replicas",
        type=int,
        default=defaults.n_replicas,
        help="extensions from the equilibrated cell with fresh velocities "
        "(default: %(default)s)",
    )
    group.add_argument(
        f"--{prefix}-samples-per-step",
        type=int,
        default=defaults.samples_per_step,
        help="stress readings per extension (default: %(default)s)",
    )
    group.add_argument(
        f"--{prefix}-stage-ps",
        type=float,
        default=defaults.stage_ps,
        help="duration of each resumable extension chunk, in ps (default: %(default)s)",
    )
    group.add_argument(
        f"--{prefix}-trajectory-ps",
        type=float,
        default=defaults.trajectory_ps,
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
        default=None,
        help="monomer SMILES with two [*] attachment points, e.g. '[*]CC[*]'. "
        "Required unless --analyse or --protocol tm is given",
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
        "--head-cap",
        help="head end-cap SMILES with one [*] attachment point",
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
        default=None,
        help="target temperature in kelvin (default: 298.15 for breaking, elongation "
        "and yield, "
        "450 for other protocols)",
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
        "temperature ladder",
        "cooling for melt-quench/tg, heating from a crystal for tm",
    )
    quench.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="starting temperature (default: melt temperature; tm: 250 K)",
    )
    quench.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="ending temperature (default: 200 K; tm: 650 K)",
    )
    quench.add_argument(
        "--step-k",
        type=float,
        default=None,
        help="temperature change per step (default: 20 K; tm: 10 K)",
    )
    quench.add_argument(
        "--hold-ps",
        type=float,
        default=None,
        help="time held at each temperature (default: 200 ps; tm: 1000 ps)",
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
        help="refuse to start a scan longer than this",
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
        help="serialized OpenMM System for --crystal-pdb, without a thermostat or barostat",
    )
    melting.add_argument(
        "--state-in",
        help="optional serialized OpenMM State for the same crystalline cell",
    )
    melting.add_argument(
        "--tm-equilibration-ps",
        type=float,
        default=1000.0,
        help="equilibrate the crystal at --t-start for this long (default: %(default)s)",
    )
    melting.add_argument(
        "--tm-stage-ps",
        type=float,
        default=10_000.0,
        help="maximum heating stage duration in ps (default: %(default)s)",
    )
    melting.add_argument(
        "--tm-trajectory-ps",
        type=float,
        default=None,
        help="optional trajectory interval in ps, to inspect loss of crystal order",
    )
    melting.add_argument(
        "--tm-barostat",
        choices=("isotropic", "anisotropic"),
        default="anisotropic",
        help="pressure control for the crystal and heating (default: %(default)s)",
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
        "--modulus-relax-times",
        type=_floats,
        default=None,
        help="comma-separated hold times in ps for a Young's modulus rate scan; "
        "at least three distinct values, e.g. 50,150,500. Requires --protocol "
        "modulus and --target-strain-rate; runs extension replicas at each rate",
    )
    mechanics.add_argument(
        "--target-strain-rate",
        type=float,
        default=None,
        help="positive target rate in strain/ns for modulus extrapolation "
        "(multiply a rate in s^-1 by 1e-9); also enables rate analysis of "
        "--analyse directories or a saved modulus rate scan",
    )
    mechanics.add_argument(
        "--elastic-strain-limit",
        type=float,
        default=0.015,
        help="strain the modulus is fitted up to (default: %(default)s)",
    )
    rate_analysis = parser.add_argument_group(
        "rate sensitivity",
        "compare the same measurement at several imposed loading rates",
    )
    rate_analysis.add_argument(
        "--rate-property",
        choices=sorted(RATE_PROPERTIES),
        default=None,
        help="property to compare; a new scan must use its matching --protocol",
    )
    rate_analysis.add_argument(
        "--rate-hold-times",
        type=_floats,
        default=None,
        help="three or more distinct hold times in ps; varies rate while keeping the same ladder and preparation",
    )
    rate_analysis.add_argument(
        "--target-property-rate",
        type=float,
        default=None,
        help="positive target in the selected property's units: strain/ns, bar/ns, or K/ns",
    )
    rate_analysis.add_argument(
        "--max-rate-extrapolation-decades",
        type=float,
        default=2.0,
        help="largest extrapolation distance allowed to resolve (default: %(default)s); this is a reporting guard",
    )
    rate_analysis.add_argument(
        "--thermal-rate-replicas",
        type=_positive_int,
        default=3,
        help="independent velocity replicas per thermal rate (default: %(default)s)",
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
    breaking = parser.add_argument_group(
        "breaking strength",
        "finite tensile extension and the stress drop that qualifies its peak",
    )
    _add_tensile_arguments(breaking, "breaking", BreakingSpec())
    elongation = parser.add_argument_group(
        "elongation at break",
        "engineering strain at the onset of a confirmed terminal stress drop",
    )
    _add_tensile_arguments(elongation, "elongation", ElongationSpec())
    failure = parser.add_argument_group(
        "tensile failure criterion",
        "shared stress-drop criterion for breaking strength and elongation at break",
    )
    failure.add_argument(
        "--failure-fraction",
        type=float,
        default=0.5,
        help="fraction of peak nominal stress below which the terminal "
        "drop must remain (default: %(default)s)",
    )
    failure.add_argument(
        "--confirmation-steps",
        type=int,
        default=3,
        help="consecutive terminal holds needed to confirm the stress drop "
        "(default: %(default)s)",
    )
    yielding = parser.add_argument_group(
        "yield strength",
        "finite tensile extension and the offset line defining its proof stress",
    )
    _add_tensile_arguments(yielding, "yield", YieldSpec())
    yielding.add_argument(
        "--yield-offset-strain",
        type=float,
        default=0.002,
        help="strain offset for the proof-stress line; 0.002 means 0.2%% "
        "(default: %(default)s)",
    )
    yielding.add_argument(
        "--yield-fit-min-strain",
        type=float,
        default=0.0,
        help="lower engineering strain for the initial elastic fit "
        "(default: %(default)s)",
    )
    yielding.add_argument(
        "--yield-fit-max-strain",
        type=float,
        default=0.02,
        help="upper engineering strain for the initial elastic fit "
        "(default: %(default)s)",
    )
    relaxation = parser.add_argument_group(
        "relaxation",
        "the step strain the relax protocol applies, and how the decay after "
        "it is sampled",
    )
    relaxation.add_argument(
        "--relax-mode",
        default="tensile",
        choices=RELAX_MODES,
        help="whether the step is an extension or a shear; both measure G(t), "
        "and a shear imposes no lateral contraction (default: %(default)s)",
    )
    relaxation.add_argument(
        "--step-strain",
        type=float,
        default=0.03,
        help="the strain applied all at once, then held (default: %(default)s)",
    )
    relaxation.add_argument(
        "--baseline-ps",
        type=float,
        default=1000.0,
        help="time at the locked box before straining; its scatter is the "
        "floor the decay is read against (default: %(default)s)",
    )
    relaxation.add_argument(
        "--relaxation-ps",
        type=float,
        default=10000.0,
        help="how long the strain is held, per replica (default: %(default)s)",
    )
    relaxation.add_argument(
        "--relax-replicas",
        type=int,
        default=4,
        help="independent runs from the same cell with fresh velocities. The "
        "first knob to turn: the early bins hold one reading each, so "
        "averaging replicas is what makes the fast end of the curve mean "
        "anything (default: %(default)s)",
    )
    relaxation.add_argument(
        "--sample-every-ps",
        type=float,
        default=0.05,
        help="time between stress readings early on (default: %(default)s)",
    )
    relaxation.add_argument(
        "--bins-per-decade",
        type=int,
        default=20,
        help="logarithmic time bins per decade (default: %(default)s)",
    )
    relaxation.add_argument(
        "--relax-stage-ps",
        type=float,
        default=20000.0,
        help="most relaxation one stage may hold before it is split for "
        "resume (default: %(default)s)",
    )
    relaxation.add_argument(
        "--linearity-strains",
        type=_floats,
        default=None,
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
        default=None,
        metavar="STAGE",
        help="saved stage to check (default: --structure-stage or the last available stage)",
    )
    convergence.add_argument(
        "--window-fractions",
        type=_floats,
        default=DEFAULT_WINDOW_FRACTIONS,
        help="increasing observed fractions ending at 1 (default: 0.25,0.5,0.75,1)",
    )
    convergence.add_argument(
        "--convergence-tolerance",
        type=float,
        default=0.1,
        help="maximum relative change to resolve window stability (default: %(default)s)",
    )
    convergence.add_argument(
        "--min-effective-samples",
        type=float,
        default=20.0,
        help="minimum autocorrelation-adjusted count for scalar time traces (default: %(default)s)",
    )
    convergence.add_argument(
        "--convergence-discard-fraction",
        type=float,
        default=0.1,
        help="initial fraction discarded from stationary time traces (default: %(default)s)",
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
        "--structure-stage",
        default=None,
        metavar="STAGE",
        help="the stage whose coordinates to measure (default: the last stage "
        "with a trajectory, else the last stage's closing snapshot)",
    )
    analysis.add_argument(
        "--backbone",
        type=_ints,
        default=None,
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
        "are further capped at 50 and 8 frames",
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


def _ints(text: str) -> tuple[int, ...]:
    """Parse a comma-separated list of atom indices, at the front door."""
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of integers, e.g. '0,1,4,5'."
        ) from error
    if len(values) < 2:
        raise argparse.ArgumentTypeError(
            f"{text!r} names {len(values)} atom(s); a backbone of one atom has no "
            "end-to-end vector."
        )
    return values


def _positive_float(text: str) -> float:
    """An explicitly finite, positive numeric command-line value."""
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _build_characteristic_ratio(arguments: argparse.Namespace) -> float:
    """Keep the polyethylene build default separate from recorded analysis values."""
    return (
        7.0
        if arguments.characteristic_ratio is None
        else float(arguments.characteristic_ratio)
    )


def _positive_int(text: str) -> int:
    """Parse a count that has to be at least one, at the front door."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer.")
    return value


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


_RATE_PROTOCOLS = {
    "youngs_modulus": "modulus",
    "poisson_ratio": "modulus",
    "shear_modulus": "modulus",
    "bulk_modulus": "modulus",
    "load_modulus": "modulus",
    "yield_strength": "yield",
    "yield_strain": "yield",
    "breaking_strength": "breaking",
    "elongation_at_break": "elongation",
    "glass_transition": "tg",
    "melting_temperature": "tm",
}
_PROTOCOL_RATE_DEFAULTS = {
    "modulus": "youngs_modulus",
    "yield": "yield_strength",
    "breaking": "breaking_strength",
    "elongation": "elongation_at_break",
    "tg": "glass_transition",
    "tm": "melting_temperature",
}


def _property_rates_requested(arguments: argparse.Namespace) -> bool:
    return any(
        (
            arguments.rate_property is not None,
            arguments.rate_hold_times is not None,
            arguments.target_property_rate is not None,
            arguments.target_strain_rate is not None
            and arguments.protocol in ("yield", "breaking", "elongation"),
        )
    )


def _property_rate_request(
    arguments: argparse.Namespace,
) -> tuple[str, float, RateScanSpec]:
    """Resolve a physical rate unit before allowing any build or dynamics."""
    property_name = arguments.rate_property or _PROTOCOL_RATE_DEFAULTS.get(
        arguments.protocol
    )
    if property_name is None:
        raise ValueError(
            "Choose --rate-property for rate analysis, or a measurement --protocol for a scan."
        )
    if arguments.modulus_relax_times is not None:
        raise ValueError(
            "Use --rate-hold-times with --rate-property; --modulus-relax-times belongs to the original Young's modulus interface."
        )
    if (
        arguments.target_property_rate is not None
        and arguments.target_strain_rate is not None
    ):
        raise ValueError(
            "Specify one of --target-property-rate or --target-strain-rate."
        )
    target = arguments.target_property_rate
    if target is None and arguments.target_strain_rate is not None:
        if RATE_PROPERTIES[property_name].rate_unit != "strain/ns":
            raise ValueError(
                f"{property_name} uses {RATE_PROPERTIES[property_name].rate_unit}; use --target-property-rate."
            )
        target = arguments.target_strain_rate
    if target is None or not math.isfinite(target) or target <= 0:
        raise ValueError("A finite positive --target-property-rate is required.")
    maximum = arguments.max_rate_extrapolation_decades
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError(
            "--max-rate-extrapolation-decades must be finite and nonnegative."
        )
    protocol_name = _RATE_PROTOCOLS[property_name]
    if arguments.analyse:
        if arguments.rate_hold_times is not None:
            raise ValueError(
                "--rate-hold-times starts new dynamics and cannot be used with --analyse."
            )
        return property_name, float(target), default_rate_spec(property_name)
    elif arguments.protocol != protocol_name or arguments.rate_hold_times is None:
        raise ValueError(
            f"{property_name} scans require --protocol {protocol_name} and --rate-hold-times."
        )
    factories: dict[str, Callable[..., RateScanSpec]] = {
        "modulus": _modulus_spec,
        "yield": _yield_spec,
        "breaking": _breaking_spec,
        "elongation": _elongation_spec,
        "tg": _tg_spec,
        "tm": _tm_spec,
    }
    spec = factories[protocol_name](
        **_protocol_options(arguments, PROTOCOLS[protocol_name])
    )
    return property_name, float(target), spec


def _write_property_rate_result(
    arguments: argparse.Namespace, report: RateReport, output_dir: Path | None = None
) -> None:
    """A unit-aware headline for each model, including unavailable predictions."""
    property = report.property
    for form in ("log_linear", "power_law"):
        fit = getattr(report, form)
        if fit is None:
            print(f"{property.label}, {form}: unavailable (not resolved)", flush=True)
            continue
        uncertainty = (
            f"{fit.standard_error:.3g}"
            if math.isfinite(fit.standard_error)
            else "unknown"
        )
        print(
            f"{property.label}, {form}: {fit.value:.5g} +/- {uncertainty} {property.value_unit} (fit SE) at "
            f"{fit.target_rate:.4g} {property.rate_unit}; {fit.n_rates} rates, extrapolated {fit.extrapolation_decades:.2f} decades"
            f"{'' if fit.resolved else ' (not resolved)'}",
            flush=True,
        )
        for note in fit.notes:
            print(f"note ({form}): {note}", flush=True)
    if report.log_linear is not None and report.power_law is not None:
        print(
            f"model difference at target: {abs(report.log_linear.value - report.power_law.value):.4g} {property.value_unit}",
            flush=True,
        )
    for note in report.notes:
        print(f"note: {note}", flush=True)
    files = write_rate_report(
        report,
        output_dir if output_dir is not None else arguments.output_dir,
        figures=not arguments.no_figures,
        figure_format=arguments.figure_format,
    )
    print(f"wrote {files.json} and {len(files.figures)} figure(s)", flush=True)


def _analyse_convergence(arguments: argparse.Namespace) -> int:
    report = analyse_convergence(
        arguments.analyse[0],
        stage=arguments.convergence_stage or arguments.structure_stage,
        backbone=arguments.backbone,
        stride=arguments.stride,
        window_fractions=arguments.window_fractions,
        relative_tolerance=arguments.convergence_tolerance,
        min_effective_samples=arguments.min_effective_samples,
        discard_fraction=arguments.convergence_discard_fraction,
    )
    print(f"observation-window convergence: stage {report.stage}", flush=True)
    statuses: list[tuple[str, bool, Sequence[str]]] = [
        (name, result.resolved, result.notes) for name, result in report.results.items()
    ]
    if report.relaxation is not None:
        statuses.extend(
            (name, result.resolved, result.notes)
            for name, result in report.relaxation.metrics.items()
        )
    if report.structural is not None:
        statuses.extend(
            (name, parameter.resolved, parameter.notes)
            for name, parameter in report.structural.parameters.items()
        )
    for name, resolved, notes in statuses:
        print(f"{name}: {'resolved' if resolved else 'unresolved'}", flush=True)
        for note in notes:
            print(f"  {note}", flush=True)
    for note in report.notes:
        print(f"note: {note}", flush=True)
    files = write_convergence_report(
        report,
        arguments.output_dir,
        figures=not arguments.no_figures,
        figure_format=arguments.figure_format,
    )
    print(f"wrote {files.json} and {len(files.figures)} figure(s)", flush=True)
    return 0


def _build_cli_melt(
    arguments: argparse.Namespace, build_dir: Path, cache_dir: Path
) -> tuple[ChainResult, RunContext]:
    """Prepare and validate a physical cell without starting dynamics."""
    n_chains = int(arguments.chains)
    n_conformers = min(int(arguments.conformers or n_chains), n_chains)
    chain = build_chain(
        ChainSpec(
            monomer_smiles=cast(str, arguments.monomer),
            degree_of_polymerization=int(arguments.degree_of_polymerization),
            residue_name=cast(str, arguments.residue_name),
            tacticity=cast(str, arguments.tacticity),
            head_cap=cast("str | None", arguments.head_cap),
            tail_cap=cast("str | None", arguments.tail_cap),
            characteristic_ratio=_build_characteristic_ratio(arguments),
            seed=int(arguments.seed),
        ),
        "chain",
        n_conformers=n_conformers,
        output_dir=build_dir,
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
        build_dir / "polymer_ff.xml",
        residue_name=cast(str, arguments.residue_name),
        backend=cast(str, arguments.backend),
        cache_dir=cache_dir,
        workdir=build_dir / "forcefill",
    )
    print(f"force field: {forcefield.forcefield_xml}", flush=True)

    components = distribute_conformers(list(chain.pdb_paths), n_chains)
    packed = pack_box(
        components,
        box_edge_nm(
            [n_chains], [chain.molar_mass_g_mol], float(arguments.pack_density)
        ),
        build_dir / "packed.pdb",
        seed=int(arguments.seed),
        workdir=build_dir,
    )
    box = assemble_box(components, packed.packed_pdb, packed.box_nm)
    check_packing(box.topology, box.positions_nm)
    print(
        f"packed: {box.n_molecules} chains, {box.topology.getNumAtoms()} atoms",
        flush=True,
    )

    run = prepare_run(
        prepare_box(box, forcefield),
        forcefield,
        spec,
        platform=cast("str | None", arguments.platform),
        seed=int(arguments.seed),
    )
    return chain, run


def _prepare_cli_melt(
    arguments: argparse.Namespace, output: Path
) -> tuple[ChainResult, RunContext]:
    """Stage repeat preparations so failed validation cannot alter old assets."""
    build = output / "build"
    record_path = build / "inputs.json"
    previous = json.loads(record_path.read_text()) if record_path.is_file() else None
    if previous is not None:
        for name, digest in previous["artifacts"].items():
            path = build / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ProtocolError(
                    f"Existing build artifact {name!r} changed or is missing. "
                    "Restore it or use a fresh output directory."
                )
    repeated = build.exists()
    with ExitStack() as stack:
        if repeated:
            temporary = Path(
                stack.enter_context(
                    TemporaryDirectory(prefix=".build-check-", dir=output)
                )
            )
            working = temporary / "build"
            cache = temporary / "cache"
            if (output / "cache").is_dir():
                shutil.copytree(output / "cache", cache)
        else:
            working = build
            cache = output / "cache"
        chain, run = _build_cli_melt(arguments, working, cache)
        inputs = {
            "run": _run_identity(run),
            "chain": {
                key: value
                for key, value in asdict(chain).items()
                if key not in {"sdf_paths", "pdb_paths"}
            },
        }
        # JSON normalization makes tuple-valued chain metadata comparable.
        inputs = json.loads(json.dumps(inputs, allow_nan=False))
        if repeated and (previous is None or previous["inputs"] != inputs):
            raise ProtocolError(
                "Prepared chemistry, force field or packed coordinates changed, "
                "or the existing build lacks verified inputs. The original build "
                "was preserved; use a fresh output directory."
            )
        for directory in (output, output / "equilibration"):
            validate_run_inputs(run, directory)
        if not repeated:
            artifacts = [
                *chain.sdf_paths,
                *chain.pdb_paths,
                run.forcefield.forcefield_xml,
                str(working / "packed.pdb"),
            ]
            record = {
                "inputs": inputs,
                "artifacts": {
                    str(
                        Path(path).resolve().relative_to(working.resolve())
                    ): file_sha256(path)
                    for path in artifacts
                },
            }
            write_json(record_path, record)
        else:
            # Everything needed for dynamics is now in memory. Returned file
            # references point at the verified persistent originals.
            chain = replace(
                chain,
                sdf_paths=tuple(
                    str(build / Path(path).name) for path in chain.sdf_paths
                ),
                pdb_paths=tuple(
                    str(build / Path(path).name) for path in chain.pdb_paths
                ),
            )
            run.forcefield = replace(
                run.forcefield,
                forcefield_xml=str(build / Path(run.forcefield.forcefield_xml).name),
            )
        return chain, run


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

    if arguments.convergence:
        if not arguments.analyse or len(arguments.analyse) != 1:
            parser.error("--convergence requires exactly one --analyse directory")
        if (
            _property_rates_requested(arguments)
            or arguments.target_strain_rate is not None
            or arguments.modulus_relax_times is not None
        ):
            parser.error("run --convergence separately from imposed-rate analysis")
        try:
            return _analyse_convergence(arguments)
        except (OSError, ValueError, AnalysisError) as error:
            parser.error(str(error))

    property_request: tuple[str, float, RateScanSpec] | None = None
    if _property_rates_requested(arguments):
        try:
            property_request = _property_rate_request(arguments)
            property_name, target, rate_spec = property_request
            if arguments.analyse:
                report = analyse_property_rates(
                    arguments.analyse,
                    property_name=property_name,
                    target_rate=target,
                    strain_limit=arguments.elastic_strain_limit,
                    max_extrapolation_decades=arguments.max_rate_extrapolation_decades,
                )
                _write_property_rate_result(arguments, report)
                return 0
            plan = validate_property_rate_scan(
                rate_spec,
                arguments.rate_hold_times,
                property_name=property_name,
                target_rate=target,
                n_replicas=arguments.thermal_rate_replicas,
                max_extrapolation_decades=arguments.max_rate_extrapolation_decades,
            )
            print(
                f"{property_name} rate scan: {plan.total_ns:.3g} ns total including preparation and all replicas",
                flush=True,
            )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            parser.error(str(error))

    if property_request is None and arguments.modulus_relax_times is not None:
        if arguments.analyse or arguments.protocol != "modulus":
            parser.error("--modulus-relax-times requires a new --protocol modulus scan")
        if arguments.target_strain_rate is None:
            parser.error("--modulus-relax-times requires --target-strain-rate")
    if property_request is None and arguments.target_strain_rate is not None:
        if (
            not math.isfinite(arguments.target_strain_rate)
            or arguments.target_strain_rate <= 0.0
        ):
            parser.error("--target-strain-rate must be finite and positive (strain/ns)")
        if not arguments.analyse and arguments.modulus_relax_times is None:
            parser.error(
                "--target-strain-rate requires --modulus-relax-times or --analyse"
            )
    if arguments.analyse:
        if arguments.target_strain_rate is not None:
            try:
                modulus_report = analyse_modulus_rates(
                    arguments.analyse,
                    target_rate_per_ns=arguments.target_strain_rate,
                    strain_limit=arguments.elastic_strain_limit,
                )
            except (OSError, ValueError, AnalysisError) as error:
                parser.error(str(error))
            _write_modulus_rate_result(arguments, modulus_report)
            return 0
        if arguments.protocol == "modulus" and len(arguments.analyse) > 1:
            parser.error(
                "analysing multiple modulus rates requires --target-strain-rate"
            )
        return _analyse(arguments)
    if arguments.protocol == "tm":
        if arguments.monomer is not None:
            parser.error("tm starts from --crystal-pdb, not a monomer SMILES")
        if arguments.crystal_pdb is None or arguments.system_xml is None:
            parser.error("tm requires both --crystal-pdb and --system-xml")
        try:
            tm_spec = _tm_spec(**_protocol_options(arguments, PROTOCOLS["tm"]))
            # Validate the complete scan before reading coordinates or creating files.
            if property_request is None:
                melting_scan(tm_spec)
            crystal_run = _prepared_crystal(arguments)
        except (OSError, ValueError, TmError) as error:
            parser.error(str(error))
        if arguments.dry_run:
            print(
                "dry run: crystal and heating schedule validated; no dynamics",
                flush=True,
            )
            return 0
        if property_request is not None:
            property_name, target, rate_spec = property_request
            report = run_property_rate_scan(
                crystal_run,
                Path(arguments.output_dir or "run"),
                property_name=property_name,
                target_rate=target,
                spec=rate_spec,
                hold_times_ps=arguments.rate_hold_times,
                n_replicas=arguments.thermal_rate_replicas,
                max_extrapolation_decades=arguments.max_rate_extrapolation_decades,
                state_in=arguments.state_in,
                crystalline=True,
            )
            _write_property_rate_result(
                arguments, report, Path(arguments.output_dir or "run") / "analysis"
            )
            return 0
        result = run_tm_scan(
            crystal_run,
            Path(arguments.output_dir or "run"),
            spec=tm_spec,
            state_in=arguments.state_in,
            crystalline=True,
        )
        print(_melting_line(result), flush=True)
        for note in result.report.notes:
            print(f"note: {note}", flush=True)
        files = write_melting_report(
            result.report,
            figures=not arguments.no_figures,
            figure_format=cast(str, arguments.figure_format),
        )
        print(f"wrote {files.json} and {len(files.figures)} figure(s)", flush=True)
        return 0
    if any((arguments.crystal_pdb, arguments.system_xml, arguments.state_in)):
        parser.error("--crystal-pdb, --system-xml and --state-in require --protocol tm")
    if arguments.monomer is None:
        parser.error("a monomer SMILES is required unless --analyse is given")
    if property_request is None and arguments.protocol == "breaking":
        try:
            _breaking_protocol(**_protocol_options(arguments, PROTOCOLS["breaking"]))
        except (ValueError, BreakingError) as error:
            parser.error(str(error))
    if property_request is None and arguments.protocol == "elongation":
        try:
            _elongation_protocol(
                **_protocol_options(arguments, PROTOCOLS["elongation"])
            )
        except (ValueError, ElongationError) as error:
            parser.error(str(error))
    if property_request is None and arguments.protocol == "yield":
        try:
            _yield_protocol(**_protocol_options(arguments, PROTOCOLS["yield"]))
        except (ValueError, YieldError) as error:
            parser.error(str(error))
    if arguments.modulus_relax_times is not None:
        try:
            plan = validate_modulus_rate_scan(
                _modulus_spec(**_protocol_options(arguments, PROTOCOLS["modulus"])),
                arguments.modulus_relax_times,
                target_rate_per_ns=arguments.target_strain_rate,
            )
        except (ValueError, MechanicalError) as error:
            parser.error(str(error))
        print(
            f"modulus rate scan: {len(plan.schedules)} rates, "
            f"{plan.n_replicas} replicas each, {plan.total_ns:.3g} ns total "
            "including equilibration",
            flush=True,
        )

    output = Path(cast("str | None", arguments.output_dir) or "run")
    n_chains = int(arguments.chains)
    n_conformers = min(int(arguments.conformers or n_chains), n_chains)
    # Guard the preparation too: rebuilding into a completed run can overwrite
    # its chemistry before the dynamics runner has a chance to reject resume.
    request = {
        key: value
        for key, value in vars(arguments).items()
        if key
        not in {
            "output_dir",
            "platform",
            "dry_run",
            "verbose",
            "no_figures",
            "figure_format",
            "max_total_ns",
        }
    }
    request["conformers"] = n_conformers
    request["characteristic_ratio"] = _build_characteristic_ratio(arguments)
    request["runtime_versions"] = {
        package: version(package)
        for package in ("openmm", "rdkit", "forcefill", "openff-toolkit", "numpy")
    }
    try:
        record_build_request(output, request)
    except ProtocolError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)

    try:
        chain, run = _prepare_cli_melt(arguments, output)
    except ProtocolError as error:
        parser.error(str(error))
    if arguments.dry_run:
        print("dry run: stopping before dynamics", flush=True)
        return 0

    if property_request is not None:
        property_name, target, rate_spec = property_request
        report = run_property_rate_scan(
            run,
            output,
            property_name=property_name,
            target_rate=target,
            spec=rate_spec,
            hold_times_ps=arguments.rate_hold_times,
            n_replicas=arguments.thermal_rate_replicas,
            max_extrapolation_decades=arguments.max_rate_extrapolation_decades,
            chain_backbone=chain.backbone,
            atoms_per_chain=chain.n_atoms,
            expected_characteristic_ratio=_build_characteristic_ratio(arguments),
        )
        _write_property_rate_result(arguments, report, output / "analysis")
        return 0
    name = cast(str, arguments.protocol)
    entry = PROTOCOLS[name]
    options = _protocol_options(arguments, entry)
    if name == "tg":
        return _run_tg_scan(arguments, run, output, chain, options)
    if name == "modulus":
        return _run_modulus_scan(arguments, run, output, chain, options)
    if name == "breaking":
        return _run_breaking_scan(arguments, run, output, chain, options)
    if name == "elongation":
        return _run_elongation_scan(arguments, run, output, chain, options)
    if name == "yield":
        return _run_yield_scan(arguments, run, output, chain, options)
    if name == "relax":
        return _run_relaxation_scan(arguments, run, output, chain, options)

    summary = run_protocol(
        entry.factory(**options),
        run,
        output,
        chain_backbone=chain.backbone,
        atoms_per_chain=chain.n_atoms,
        expected_characteristic_ratio=_build_characteristic_ratio(arguments),
    )
    print(
        f"{summary.protocol}: {len(summary.results)} stages in "
        f"{summary.wall_seconds / 60:.1f} min, manifest {summary.manifest_path}",
        flush=True,
    )
    _print_chains(summary.chains)
    return 0


def _prepared_crystal(arguments: argparse.Namespace) -> RunContext:
    """Load a crystal without repacking it or rebuilding its force field."""
    import numpy as np
    import openmm as mm
    from openmm import app, unit

    try:
        pdb = app.PDBFile(arguments.crystal_pdb)
        system = mm.XmlSerializer.deserialize(Path(arguments.system_xml).read_text())
    except Exception as error:
        raise ValueError(
            f"Could not read the prepared crystal and System: {error}"
        ) from error
    if not isinstance(system, mm.System):
        raise ValueError("--system-xml must contain a serialized OpenMM System")
    if system.getNumParticles() != pdb.topology.getNumAtoms():
        raise ValueError("Crystal PDB and System must contain the same number of atoms")
    if any("Barostat" in type(force).__name__ for force in system.getForces()):
        raise ValueError("The supplied System must not contain a barostat")
    if any(isinstance(force, mm.AndersenThermostat) for force in system.getForces()):
        raise ValueError(
            "The supplied System must not contain an Andersen thermostat; "
            "the heating scan controls temperature with its Langevin integrator"
        )
    if not system.usesPeriodicBoundaryConditions():
        raise ValueError("The supplied System must use periodic boundary conditions")
    vectors = pdb.topology.getPeriodicBoxVectors()
    if vectors is None:
        raise ValueError("Crystal PDB must contain periodic box vectors (CRYST1)")
    matrix = np.asarray(vectors.value_in_unit(unit.nanometer), dtype=float)
    if not np.all(np.isfinite(matrix)) or np.linalg.det(matrix) <= 0:
        raise ValueError("Crystal PDB must have finite, positive-volume box vectors")
    positions = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    if not np.all(np.isfinite(positions)):
        raise ValueError("Crystal PDB coordinates must be finite")
    system.setDefaultPeriodicBoxVectors(*vectors)
    if arguments.state_in is not None:
        try:
            state = mm.XmlSerializer.deserialize(Path(arguments.state_in).read_text())
            if not isinstance(state, mm.State):
                raise ValueError("--state-in must contain a serialized OpenMM State")
            saved_positions = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=float,
            )
            saved_vectors = np.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=float,
            )
        except Exception as error:
            raise ValueError(
                f"Could not read the crystalline starting State: {error}"
            ) from error
        if saved_positions.shape != positions.shape or not np.all(
            np.isfinite(saved_positions)
        ):
            raise ValueError(
                "Starting State must have finite positions for every crystal atom"
            )
        if not np.all(np.isfinite(saved_vectors)) or np.linalg.det(saved_vectors) <= 0:
            raise ValueError(
                "Starting State must have finite, positive-volume box vectors"
            )
    box = PackedBox(
        topology=pdb.topology,
        positions_nm=positions,
        box_nm=cast(
            tuple[float, float, float],
            tuple(float(value) for value in np.linalg.norm(matrix, axis=1)),
        ),
        n_molecules=_crystal_molecule_count(pdb.topology),
    )
    # The supplied System owns its parameters. This descriptor is provenance,
    # never passed to app.ForceField; masses and constraints are left untouched.
    forcefield = PolymerForceField(
        forcefield_xml=str(Path(arguments.system_xml).resolve()),
        base_forcefield=(),
        residue_name="",
        backend="prepared-system",
    )
    return prepare_run(
        box,
        forcefield,
        SystemSpec(constraints="none", hydrogen_mass_amu=None),
        platform=arguments.platform,
        seed=arguments.seed,
        system=system,
    )


def _crystal_molecule_count(topology: Any) -> int:
    """Count bonded components and validate the molecule layout used by reports."""
    n_atoms = topology.getNumAtoms()
    if n_atoms == 0:
        raise ValueError("Crystal PDB must contain atoms")
    neighbours: list[list[int]] = [[] for _ in range(n_atoms)]
    for left, right in topology.bonds():
        neighbours[left.index].append(right.index)
        neighbours[right.index].append(left.index)
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        visited.add(start)
        pending = [start]
        component = []
        while pending:
            atom = pending.pop()
            component.append(atom)
            for neighbour in neighbours[atom]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        components.append(component)
    if any(len(component) != len(components[0]) for component in components):
        raise ValueError(
            "Crystal PDB must contain molecules with equal atom counts; "
            "check its bonds/CONECT records against the System"
        )
    if any(
        max(component) - min(component) + 1 != len(component)
        for component in components
    ):
        raise ValueError(
            "Crystal PDB molecules must occupy contiguous atom blocks; "
            "reorder both the PDB and System consistently"
        )
    return len(components)


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
        "expected_characteristic_ratio": _build_characteristic_ratio(arguments),
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
    if arguments.modulus_relax_times is not None:
        report = run_modulus_rate_scan(
            run,
            output,
            relax_ps=arguments.modulus_relax_times,
            target_rate_per_ns=arguments.target_strain_rate,
            spec=_modulus_spec(**options),
            chain_backbone=chain.backbone,
            atoms_per_chain=chain.n_atoms,
            expected_characteristic_ratio=_build_characteristic_ratio(arguments),
        )
        _write_modulus_rate_result(arguments, report, output / "analysis")
        return 0
    result = run_modulus_scan(
        run,
        output,
        spec=_modulus_spec(**options),
        chain_backbone=chain.backbone,
        atoms_per_chain=chain.n_atoms,
        expected_characteristic_ratio=_build_characteristic_ratio(arguments),
    )
    for line in _modulus_lines(result):
        print(line, flush=True)
    _print_chains(None)
    return 0


def _write_modulus_rate_result(
    arguments: argparse.Namespace,
    report: ModulusRateReport,
    output_dir: Path | None = None,
) -> None:
    """Keep each model's target, uncertainty and extrapolation distance visible."""
    for fit in (report.log_linear, report.power_law):
        print(
            f"{fit.form}: E = {fit.modulus_mpa:.4g} +/- "
            f"{fit.standard_error_mpa:.3g} MPa (fit SE) at "
            f"{fit.target_rate_per_ns:.3g} strain/ns, {fit.temperature_k:.1f} K; "
            f"{fit.sensitivity_mpa_per_decade:.3g} MPa per decade at "
            f"{fit.reference_rate_per_ns:.3g} strain/ns; "
            f"{fit.n_rates} measured rates, extrapolated "
            f"{fit.extrapolation_decades:.2f} decades"
            f"{'' if fit.resolved else ' (not resolved)'}",
            flush=True,
        )
        for note in fit.notes:
            print(f"note ({fit.form}): {note}", flush=True)
    print(
        "model difference at target: "
        f"{abs(report.log_linear.modulus_mpa - report.power_law.modulus_mpa):.4g} MPa",
        flush=True,
    )
    printed_notes = set(report.log_linear.notes) | set(report.power_law.notes)
    for note in report.notes:
        if note not in printed_notes:
            print(f"note: {note}", flush=True)
    files = write_modulus_rate_report(
        report,
        output_dir if output_dir is not None else arguments.output_dir,
        figures=not arguments.no_figures,
        figure_format=arguments.figure_format,
    )
    print(f"wrote {files.json} and {len(files.figures)} figure(s)", flush=True)


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
    rate = result.youngs.strain_rate_per_ns
    lines.append(
        f"E = {result.youngs.modulus_mpa:.0f} MPa{spread} at "
        + ("rate unknown" if rate is None else f"{rate:.3g} strain/ns")
        + f", {result.youngs.temperature_k:.0f} K"
        f"{'' if result.resolved else ' (not resolved)'}"
    )
    return lines + _additional_modulus_lines(result)


def _additional_modulus_lines(result: Any) -> list[str]:
    """Format the elastic cross-checks shared by fresh scans and saved reports."""
    lines: list[str] = []
    if result.poisson is not None:
        lines.append(
            f"nu = {result.poisson.ratio:.3f}"
            f"{'' if result.poisson.resolved else ' (not resolved)'}"
        )
    for label, fit in (("K", result.bulk), ("G", result.shear)):
        if fit is not None:
            lines.append(
                f"{label} = {fit.modulus_mpa:.0f} +/- "
                f"{fit.standard_error_mpa:.2g} MPa (fit SE)"
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


def _breaking_lines(report: Any) -> list[str]:
    """Keep an unconfirmed peak distinct from an apparent tensile strength."""
    if report.strength_mpa is None or not report.resolved:
        lines = ["breaking: apparent tensile strength not resolved"]
    else:
        spread = (
            ""
            if report.replica_spread_mpa is None
            else f" +/- {report.replica_spread_mpa:.3g} over {len(report.replicas)} replicas"
        )
        lines = [
            f"breaking: apparent ultimate nominal tensile strength = "
            f"{report.strength_mpa:.4g} MPa{spread}"
        ]
    for index, result in zip(report.replica_indices, report.replicas, strict=True):
        rate = (
            "unknown strain rate"
            if result.strain_rate_per_ns is None
            else f"{result.strain_rate_per_ns:.3g} strain/ns"
        )
        lines.append(
            f"  replica {index}: peak {result.peak_stress_mpa:.4g} MPa at "
            f"strain {result.strain_at_peak:.4g}, {result.temperature_k:.0f} K, "
            f"{rate}{'' if result.resolved else ' (not resolved)'}"
        )
        if result.failure_strain is not None:
            lines.append(
                f"    stress drop at strain {result.failure_strain:.4g}, "
                f"stress {result.failure_stress_mpa:.4g} MPa"
            )
    return lines


def _write_tensile_result(
    arguments: argparse.Namespace,
    report: Any,
    lines: Sequence[str],
    writer: Callable[..., ReportFiles],
) -> None:
    """Print and save a tensile report consistently after a scan or a reread."""
    for line in lines:
        print(line, flush=True)
    for note in report.notes:
        print(f"note: {note}", flush=True)
    files = writer(
        report,
        arguments.output_dir if arguments.analyse else None,
        figures=not arguments.no_figures,
        figure_format=cast(str, arguments.figure_format),
    )
    print(f"wrote {files.json} and {len(files.figures)} figure(s)", flush=True)


def _write_breaking_result(arguments: argparse.Namespace, report: Any) -> None:
    """Print and save the same result for a run and a later analysis."""
    _write_tensile_result(
        arguments, report, _breaking_lines(report), write_breaking_report
    )


def _run_breaking_scan(
    arguments: argparse.Namespace,
    run: Any,
    output: Path,
    chain: Any,
    options: dict[str, Any],
) -> int:
    """Measure the apparent tensile strength and write its report."""
    report = run_breaking_scan(
        run,
        output,
        spec=_breaking_spec(**options),
        chain_backbone=chain.backbone,
        atoms_per_chain=chain.n_atoms,
        expected_characteristic_ratio=_build_characteristic_ratio(arguments),
    )
    _write_breaking_result(arguments, report)
    return 0


def _elongation_lines(report: Any) -> list[str]:
    """Report the confirmed break strain separately from the stress maximum."""
    if report.elongation_percent is None or not report.resolved:
        lines = ["elongation: apparent elongation at break not resolved"]
    else:
        spread = (
            ""
            if report.replica_spread_percent is None
            else f" +/- {report.replica_spread_percent:.3g} percentage points "
            f"over {len(report.replicas)} replicas"
        )
        lines = [
            f"elongation: apparent elongation at break = "
            f"{report.elongation_percent:.4g}%{spread}"
        ]
    for index, result in zip(report.replica_indices, report.replicas, strict=True):
        rate = (
            "unknown strain rate"
            if result.strain_rate_per_ns is None
            else f"{result.strain_rate_per_ns:.3g} strain/ns"
        )
        elongation = (
            f"{result.elongation_percent:.4g}% at engineering strain "
            f"{result.strain_at_break:.4g}"
            if result.resolved
            and result.elongation_percent is not None
            and result.strain_at_break is not None
            else "not resolved"
        )
        lines.append(
            f"  replica {index}: elongation at break {elongation}, "
            f"{result.temperature_k:.0f} K, {rate}"
        )
        lines.append(
            f"    peak {result.peak_stress_mpa:.4g} MPa at "
            f"strain {result.strain_at_peak:.4g}"
        )
        if result.resolved and result.break_stress_mpa is not None:
            lines.append(f"    stress at break {result.break_stress_mpa:.4g} MPa")
    return lines


def _write_elongation_result(arguments: argparse.Namespace, report: Any) -> None:
    """Print and save the same result for a run and a later analysis."""
    _write_tensile_result(
        arguments, report, _elongation_lines(report), write_elongation_report
    )


def _run_elongation_scan(
    arguments: argparse.Namespace,
    run: Any,
    output: Path,
    chain: Any,
    options: dict[str, Any],
) -> int:
    """Measure apparent elongation at break and write its report."""
    report = run_elongation_scan(
        run,
        output,
        spec=_elongation_spec(**options),
        chain_backbone=chain.backbone,
        atoms_per_chain=chain.n_atoms,
        expected_characteristic_ratio=_build_characteristic_ratio(arguments),
    )
    _write_elongation_result(arguments, report)
    return 0


def _yield_lines(report: Any) -> list[str]:
    """Print the proof stress together with its offset, temperature and rate."""
    if report.strength_mpa is None or not report.resolved:
        lines = ["yield: apparent offset yield strength not resolved"]
    else:
        spread = (
            ""
            if report.replica_spread_mpa is None
            else f" +/- {report.replica_spread_mpa:.3g} over {len(report.replicas)} replicas"
        )
        lines = [
            f"yield: apparent offset yield strength = "
            f"{report.strength_mpa:.4g} MPa{spread}"
        ]
    for index, result in zip(report.replica_indices, report.replicas, strict=True):
        rate = (
            "unknown strain rate"
            if result.strain_rate_per_ns is None
            else f"{result.strain_rate_per_ns:.3g} strain/ns"
        )
        strength = (
            f"{result.strength_mpa:.4g} MPa at strain {result.yield_strain:.4g}"
            if result.resolved
            and result.strength_mpa is not None
            and result.yield_strain is not None
            else "not resolved"
        )
        lines.append(
            f"  replica {index}: {100.0 * result.offset_strain:g}% offset "
            f"proof stress {strength}, {result.temperature_k:.0f} K, {rate}"
        )
        if result.modulus_mpa is not None:
            lines.append(
                f"    initial elastic slope {result.modulus_mpa:.4g} MPa "
                f"over strain {result.fit_min_strain:g} to {result.fit_max_strain:g}"
            )
    return lines


def _write_yield_result(arguments: argparse.Namespace, report: Any) -> None:
    """Print and save the same proof-stress report after a run or a reread."""
    _write_tensile_result(arguments, report, _yield_lines(report), write_yield_report)


def _run_yield_scan(
    arguments: argparse.Namespace,
    run: Any,
    output: Path,
    chain: Any,
    options: dict[str, Any],
) -> int:
    """Measure apparent offset yield strength and write its report."""
    report = run_yield_scan(
        run,
        output,
        spec=_yield_spec(**options),
        chain_backbone=chain.backbone,
        atoms_per_chain=chain.n_atoms,
        expected_characteristic_ratio=_build_characteristic_ratio(arguments),
    )
    _write_yield_result(arguments, report)
    return 0


def _run_relaxation_scan(
    arguments: argparse.Namespace,
    run: Any,
    output: Path,
    chain: Any,
    options: dict[str, Any],
) -> int:
    """Strain the cell once, watch the stress decay, and say what it decayed to."""
    result = run_relaxation_scan(
        run,
        output,
        spec=_relaxation_spec(**options),
        chain_backbone=chain.backbone,
        atoms_per_chain=chain.n_atoms,
        expected_characteristic_ratio=_build_characteristic_ratio(arguments),
    )
    for line in _relaxation_lines(result):
        print(line, flush=True)
    return 0


def _relaxation_lines(result: Any) -> list[str]:
    """One line per fitted quantity, each carrying what qualifies it.

    Takes the :class:`~openmmpolymer.viscoelastic.RelaxationReport` a scan
    returns or ``--analyse`` reads back; both carry the overall verdict.
    """
    if result.mean is None:
        return ["relax: nothing was strained"]
    mean = result.mean
    spread = (
        ""
        if result.replica_spread_mpa is None
        else f" +/- {result.replica_spread_mpa:.3g} over {len(result.curves)}"
    )
    lines = [
        f"G(0) = {mean.initial_modulus_mpa:.4g} MPa{spread} at "
        f"{mean.step_strain:+.3f} strain, {mean.temperature_k:.0f} K, over "
        f"{mean.decades:.1f} decades"
        f"{'' if result.resolved else ' (not resolved)'}"
    ]
    if result.kww is not None:
        lines.append(
            f"KWW: beta = {result.kww.beta:.3f}, tau = {result.kww.tau_ps:.4g} ps, "
            f"<tau> = {result.kww.mean_tau_ps:.4g} ps"
            f"{'' if result.kww.resolved else ' (not resolved)'}"
        )
    if result.prony is not None:
        lines.append(
            f"Prony: G_inf = {result.prony.equilibrium_mpa:.4g} MPa over "
            f"{result.prony.n_active} of {result.prony.n_terms} terms"
            f"{'' if result.prony.plateau_reached else ' - still decaying'}"
        )
    if result.linearity is not None:
        lines.append(
            f"linearity: strains {[round(v, 4) for v in result.linearity.strains]} "
            f"differ by {100.0 * result.linearity.gap:.0f}%"
            f"{'' if result.linearity.linear else ' - outside the linear region'}"
        )
    return lines


def _analyse_relaxation(arguments: argparse.Namespace, run_dir: Path) -> None:
    """Report the relaxation modulus, and write it out."""
    report = analyse_relaxation(run_dir)
    for line in _relaxation_lines(report):
        print(line, flush=True)
    for note in report.notes:
        print(f"note: {note}", flush=True)

    files = write_relaxation_report(
        report,
        arguments.output_dir,
        figures=not arguments.no_figures,
        figure_format=cast(str, arguments.figure_format),
    )
    print(f"wrote {files.json} and {len(files.figures)} figure(s)", flush=True)


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
    that quenched gets a glass transition, a run that was deformed gets its
    elastic constants, a run that did both gets both, and a run that did
    neither is an error - which is the same "find the stages by what they
    recorded" rule the readers underneath follow.
    """
    directories = [Path(name) for name in arguments.analyse]
    first = directories[0]
    quenched = _has_stages(first, quench_stages)
    heated = _has_stages(first, heating_stages)
    broken = _has_stages(first, breaking_stages)
    elongated = _has_stages(first, elongation_stages)
    yielded = _has_stages(first, yield_stages)
    deformed = any(
        _has_stages(first, find) for find in (deform_stages, load_stages, shear_stages)
    )
    relaxed = _has_stages(first, relax_stages)
    structured = not arguments.no_structure and _has_stages(first, structure_stages)
    if not any(
        (quenched, heated, broken, elongated, yielded, deformed, relaxed, structured)
    ):
        print(
            f"nothing in {first} was a quench, a heating scan, a deformation or a relaxation, "
            "and no stage left coordinates to measure, so there is nothing to "
            "report",
            flush=True,
        )
        return 1

    if quenched:
        _analyse_tg(arguments, directories)
    if heated:
        _analyse_melting(arguments, first)
    if broken:
        _write_breaking_result(arguments, analyse_breaking(first))
    if elongated:
        _write_elongation_result(arguments, analyse_elongation(first))
    if yielded:
        _write_yield_result(arguments, analyse_yield(first))
    if deformed and not (broken or elongated or yielded):
        _analyse_mechanics(arguments, first)
    if relaxed:
        _analyse_relaxation(arguments, first)
    if structured:
        _analyse_structure(arguments, first)
    return 0


def _melting_line(result: Any) -> str:
    """Report the finite heating bracket without implying equilibrium Tm."""
    transition = getattr(result, "transition", result)
    if not transition.resolved or transition.temperature_k is None:
        return "tm: no clear melting transition (not resolved)"
    low, high = transition.bracket_k
    return (
        f"tm: apparent Tm = {transition.temperature_k:g} K "
        f"(heating bracket {low:g}-{high:g} K)"
    )


def _analyse_melting(arguments: argparse.Namespace, run_dir: Path) -> None:
    """Read the density and enthalpy discontinuity of a recorded heating scan."""
    report = analyse_melting(
        run_dir, min_points_per_branch=int(arguments.min_points_per_branch)
    )
    print(_melting_line(report), flush=True)
    for note in report.notes:
        print(f"note: {note}", flush=True)
    files = write_melting_report(
        report,
        arguments.output_dir,
        figures=not arguments.no_figures,
        figure_format=cast(str, arguments.figure_format),
    )
    print(f"wrote {files.json} and {len(files.figures)} figure(s)", flush=True)


def _analyse_structure(arguments: argparse.Namespace, run_dir: Path) -> None:
    """Report the structure and dynamics, and write them out."""
    report = analyse_structure(
        run_dir,
        stage=arguments.structure_stage,
        backbone=arguments.backbone,
        expected_characteristic_ratio=arguments.characteristic_ratio,
        stride=int(arguments.stride),
    )
    for line in _structure_lines(report):
        print(line, flush=True)
    for note in report.notes:
        print(f"note: {note}", flush=True)

    files = write_structure_report(
        report,
        arguments.output_dir,
        figures=not arguments.no_figures,
        figure_format=cast(str, arguments.figure_format),
    )
    print(f"wrote {files.json} and {len(files.figures)} figure(s)", flush=True)


def _structure_lines(report: Any) -> list[str]:
    """One line per measurement, each carrying its own caveat."""
    frames = (
        "single snapshot"
        if report.is_snapshot
        else f"{report.n_frames} frames at {report.interval_ps:g} ps"
    )
    lines = [
        f"structure: stage {report.stage} ({frames}), {report.n_chains} chains "
        f"of {report.atoms_per_chain} atoms"
    ]
    if report.backbone is None:
        lines.append("backbone: unknown, so no chain measurements")
    else:
        origin = report.backbone_source + (
            "" if report.backbone_file is None else f" from {report.backbone_file}"
        )
        lines.append(f"backbone: {len(report.backbone)} atoms, {origin}")

    distribution = report.distribution
    if distribution is not None:
        lines.append(
            f"g(r): first peak {distribution.first_peak_height:.2f} at "
            f"{distribution.first_peak_nm:.3f} nm, {distribution.n_pairs:,} "
            f"intermolecular pairs over {distribution.n_frames} frame(s)"
        )
    structure = report.structure
    if structure is not None:
        if structure.first_peak_per_nm > 0.0:
            lines.append(
                f"S(q): peak at {structure.first_peak_per_nm:.1f} /nm; nothing "
                f"below {structure.q_min_per_nm:.1f} /nm is resolvable in this cell"
            )
        else:
            lines.append(
                f"S(q): no resolvable peak above {structure.q_min_per_nm:.1f} /nm"
            )

    conformation = report.conformation
    if conformation is not None:
        mean = conformation.mean
        line = (
            f"chains: <R^2> = {mean.mean_squared_end_to_end_nm2:.3f} nm2, "
            f"Rg = {mean.mean_radius_of_gyration_nm:.3f} nm, "
            f"C = {mean.characteristic_ratio:.2f} against an expected "
            f"{mean.expected_characteristic_ratio:.2f}"
        )
        if not mean.consistent:
            line += " - not consistent with a relaxed melt"
        if conformation.settled is not None:
            line += (
                ", <R^2> settled"
                if conformation.settled.equilibrated
                else ", <R^2> still moving"
            )
        lines.append(line)

    persistence = report.persistence
    if persistence is not None:
        lines.append(_persistence_line(persistence))

    displacement = report.displacement
    if displacement is not None:
        if displacement.diffusion_coefficient_cm2_s is None:
            lines.append(
                f"MSD: slope {displacement.log_slope:.2f}, not diffusive, so no "
                "diffusion coefficient"
            )
        else:
            lines.append(
                f"MSD: slope {displacement.log_slope:.2f}, "
                f"D = {displacement.diffusion_coefficient_cm2_s:.3e} cm2/s"
            )

    relaxation = report.relaxation
    if relaxation is not None:
        if relaxation.relaxation_time_ps is None:
            lines.append(
                f"end-to-end: not decorrelated in {relaxation.trajectory_ps:.0f} "
                "ps; the relaxation time is longer than the run"
            )
        else:
            lines.append(
                f"end-to-end: relaxes in {relaxation.relaxation_time_ps:.0f} ps"
            )

    recorded = report.recorded_chains
    if recorded is not None:
        lines.append(
            "manifest recorded at the end of the run: "
            f"<R^2> = {recorded.mean_squared_end_to_end_nm2:.3f} nm2, "
            f"Rg = {recorded.mean_radius_of_gyration_nm:.3f} nm"
        )
    return lines


def _persistence_line(persistence: Any) -> str:
    """One line for a persistence length, with the extrapolation caveat."""
    if not math.isfinite(persistence.persistence_length_nm):
        return (
            "persistence length: no decay along the chain "
            f"({persistence.contour_length_nm:.2f} nm contour), rod-like"
        )
    line = (
        f"persistence length: {persistence.persistence_length_nm:.3f} nm over "
        f"{persistence.n_bonds} bonds ({persistence.contour_length_nm:.2f} nm "
        "contour)"
    )
    if not persistence.decayed:
        line += " - never decayed to 1/e within the chain, so this is an extrapolation"
    return line


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
    for line in _additional_modulus_lines(report):
        print(line, flush=True)
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
    report = analyse_tg(
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

    files = write_tg_report(
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
