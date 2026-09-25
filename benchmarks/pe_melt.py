"""Build and measure C44H90 melts against a published density reference.

Run ``python -m benchmarks.pe_melt --help`` from the repository root.
The short smoke mode exercises real chemistry and dynamics, but cannot pass
the scientific comparison. Reference tolerances are fixed before running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from openmmpolymer import (
    ChainSpec,
    Protocol,
    Stage,
    TrajectoryOptions,
    build_melt,
    end_to_end_relaxation,
    equilibration,
    open_run,
    read_state_data,
    run_protocol,
    standard_melt_equilibration,
)
from openmmpolymer._files import json_value
from openmmpolymer.packing import PACKMOL_TIMEOUT_S
from openmmpolymer.protocols import RunManifest, record_build_request

REFERENCE_PATH = Path(__file__).with_name("polyethylene.json")

#: A density above any the melt reaches at either reference temperature, for
#: checking the compressed cell still fits the cutoff.
_MAX_DENSITY_G_CM3 = 0.9


def _specific_volumes(case: dict[str, Any]) -> dict[int, float]:
    """The reference's specific volumes, in cm3/g, by temperature in kelvin."""
    return {
        int(kelvin): float(volume)
        for kelvin, volume in case["reference"]["specific_volume_cm3_g"].items()
    }


def comparison(
    measurements: list[dict[str, Any]], reference: dict[str, Any], *, smoke: bool
) -> dict[str, Any]:
    """Keep agreement with a reference separate from adequate sampling."""
    target = float(reference["density_g_cm3"])
    tolerance = float(reference["relative_density_tolerance"])
    temperature = float(reference["temperature_k"])
    temperature_tolerance = float(reference["relative_temperature_tolerance"])
    if not all(math.isfinite(value) and value > 0 for value in (target, temperature)):
        raise ValueError(
            "Reference density and temperature must be positive and finite."
        )
    if not all(
        math.isfinite(value) and 0 < value < 1
        for value in (tolerance, temperature_tolerance)
    ):
        raise ValueError(
            "Comparison tolerances must be finite fractions between 0 and 1."
        )
    if not measurements:
        raise ValueError("At least one measured replica is required.")
    values = np.asarray([row["density_g_cm3"] for row in measurements], dtype=float)
    temperatures = np.asarray(
        [row["temperature_k"] for row in measurements], dtype=float
    )
    if not all(
        np.all(np.isfinite(value)) and np.all(value > 0)
        for value in (values, temperatures)
    ):
        raise ValueError(
            "Measured densities and temperatures must be positive and finite."
        )
    mean = float(values.mean())
    error = (
        float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else None
    )
    within_range = bool(np.all(np.abs(values / target - 1.0) <= tolerance))
    at_temperature = bool(
        np.all(np.abs(temperatures / temperature - 1.0) <= temperature_tolerance)
    )
    sampled = bool(
        len(measurements) >= 3
        and len({row["seed"] for row in measurements}) == len(measurements)
        and all(
            row["density_equilibration"]["equilibrated"] and row["chains_decorrelated"]
            for row in measurements
        )
    )
    if smoke:
        status = "smoke_only"
    elif not sampled:
        status = "unresolved"
    elif within_range and at_temperature:
        status = "passed"
    else:
        status = "failed"
    return {
        "density_g_cm3": mean,
        "replica_standard_error_g_cm3": error,
        "reference_density_g_cm3": target,
        "accepted_density_range_g_cm3": [
            target * (1 - tolerance),
            target * (1 + tolerance),
        ],
        "relative_difference": mean / target - 1,
        "all_replicas_within_range": within_range,
        "all_replicas_at_target_temperature": at_temperature,
        "sampling_sufficient": sampled,
        "status": status,
    }


def _protocol(
    temperature_k: float, pressure_bar: float, smoke: bool, *, production_ps: float
) -> Protocol:
    preparation: tuple[Stage, ...]
    if smoke:
        preparation = (
            Stage("00_minimise", "minimise", {"max_iterations": 500}),
            Stage("01_pushoff", "pushoff", {"duration_ps": 0.03}),
            Stage(
                "02_nvt",
                "nvt",
                {
                    "temperature_k": temperature_k,
                    "duration_ps": 0.1,
                    "timestep_fs": 0.5,
                    "friction_ps": 10.0,
                },
            ),
            Stage(
                "03_compress",
                "compress",
                {
                    "temperature_k": temperature_k,
                    "duration_ps_each": 0.05,
                    "pressures_bar": (pressure_bar, 100.0, pressure_bar),
                    "timestep_fs": 0.5,
                },
            ),
        )
    else:
        preparation = standard_melt_equilibration(
            target_temperature_k=temperature_k,
            pressure_bar=pressure_bar,
        ).stages
    production = Stage(
        "06_measure",
        "npt",
        {
            "temperature_k": temperature_k,
            "pressure_bar": pressure_bar,
            "duration_ps": 0.5 if smoke else production_ps,
            "timestep_fs": 0.5 if smoke else 2.0,
            "report_interval_ps": 0.05 if smoke else 10.0,
            "trajectory": TrajectoryOptions("xtc", interval_ps=0.05 if smoke else 10.0),
        },
    )
    return Protocol(
        "pe_density_smoke" if smoke else "pe_density_reference",
        (*preparation, production),
    )


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "uninstalled checkout or unavailable metadata"


def run_benchmark(
    output: Path,
    *,
    temperature_k: int = 400,
    smoke: bool = False,
    seeds: tuple[int, ...] | None = None,
    platform: str | None = None,
) -> dict[str, Any]:
    """Run independent packings, recording inputs, provenance and diagnostics."""
    case = json.loads(REFERENCE_PATH.read_text())
    packing_density = case[
        "smoke_packing_density_g_cm3" if smoke else "packing_density_g_cm3"
    ]
    charge_method = "gasteiger" if smoke else case["charge_method"]
    seeds = tuple(case["seeds"]) if seeds is None else seeds
    volumes = _specific_volumes(case)
    if temperature_k not in volumes:
        raise ValueError(
            "The reference gives specific volumes only at "
            f"{', '.join(f'{kelvin} K' for kelvin in sorted(volumes))}."
        )
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("Use at least one positive seed, with no duplicates.")
    output = output.resolve()
    request = {
        "case": case,
        "temperature_k": temperature_k,
        "smoke": smoke,
        "seeds": seeds,
    }
    record_build_request(output, request)
    protocol = _protocol(
        temperature_k,
        case["pressure_bar"],
        smoke,
        production_ps=case["production_ps"],
    )
    observations = []
    for seed in seeds:
        directory = output / f"seed_{seed}"
        chain, run = build_melt(
            ChainSpec(
                monomer_smiles=case["monomer_smiles"],
                degree_of_polymerization=case["degree_of_polymerization"],
                residue_name="PE",
                seed=seed,
                characteristic_ratio=case["characteristic_ratio"],
            ),
            case["chains"],
            directory,
            target_density_g_cm3=_MAX_DENSITY_G_CM3,
            charge_method=charge_method,
            smirnoff_forcefield=case["smirnoff_forcefield"],
            pack_density_g_cm3=packing_density,
            packmol_timeout_s=120.0 if smoke else PACKMOL_TIMEOUT_S,
            platform=platform,
            cache_dir=output / "cache",
        )
        run_protocol(
            protocol,
            run,
            directory,
            chain_backbone=chain.backbone,
            atoms_per_chain=chain.n_atoms,
            expected_characteristic_ratio=case["characteristic_ratio"],
        )
        manifest = RunManifest.load(directory)
        assert manifest is not None
        series = read_state_data(manifest.stages["06_measure"]["csv"])
        density_check = equilibration(series.time_ps, series.density_g_cm3)
        ensemble = open_run(directory, "06_measure")
        try:
            chains = end_to_end_relaxation(ensemble, chain.backbone)
        finally:
            ensemble.universe.trajectory.close()
        observations.append(
            {
                "seed": seed,
                "n_atoms": run.box.topology.getNumAtoms(),
                "n_chains": run.box.n_molecules,
                "frames": ensemble.n_frames,
                "density_g_cm3": float(
                    density_check.window(series.density_g_cm3).mean()
                ),
                "temperature_k": float(
                    density_check.window(series.temperature_k).mean()
                ),
                "density_equilibration": asdict(density_check),
                "chains_decorrelated": chains.decorrelated,
                "end_to_end_correlation_final": float(chains.correlation[-1]),
                "end_to_end_relaxation_time_ps": chains.relaxation_time_ps,
                "end_to_end_max_lag_ps": float(chains.lag_ps[-1]),
                "manifest": str(Path(f"seed_{seed}") / "manifest.json"),
            }
        )
    reference = {
        **case["reference"],
        "density_g_cm3": 1.0 / volumes[temperature_k],
        "temperature_k": temperature_k,
    }
    report = {
        "case_id": case["id"],
        "smoke": smoke,
        "packing_density_g_cm3": packing_density,
        "temperature_k": temperature_k,
        "pressure_bar": case["pressure_bar"],
        "case_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
        "charge_method": charge_method,
        "smirnoff_forcefield": case["smirnoff_forcefield"],
        "protocol": asdict(protocol),
        "system": asdict(run.spec),
        "requested_platform": run.platform_name,
        "versions": {
            package: _version(package)
            for package in (
                "openmmpolymer",
                "openmm",
                "forcefill",
                "rdkit",
                "numpy",
                "openff-toolkit",
            )
        },
        "reference": reference,
        "replicas": observations,
        "comparison": comparison(observations, reference, smoke=smoke),
    }
    (output / "benchmark.json").write_text(
        json.dumps(json_value(report), indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--temperature",
        type=int,
        choices=sorted(_specific_volumes(json.loads(REFERENCE_PATH.read_text()))),
        default=400,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="short integration run; never a scientific pass",
    )
    parser.add_argument("--platform", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    arguments = parser.parse_args()
    report = run_benchmark(
        arguments.output,
        temperature_k=arguments.temperature,
        smoke=arguments.smoke,
        platform=arguments.platform,
        seeds=None if arguments.seeds is None else tuple(arguments.seeds),
    )
    print(json.dumps(report["comparison"], indent=2))
    return 0 if report["comparison"]["status"] in ("smoke_only", "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
