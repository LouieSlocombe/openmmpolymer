"""Calculate polymer properties directly from definitions and small arrays.

From the repository root, with NumPy installed::

    python examples/first_principles.py

With the full openmmpolymer environment and a completed run::

    python examples/first_principles.py --run tutorial-run --stage 06_measure
    python examples/first_principles.py --run run --discard-fraction 0.25

The default data are synthetic, noiseless examples, not material predictions.
The optional run mode reads state data and coordinates; it does not fit thermal
or mechanical scan results. See docs/first_principles.md for assumptions,
sampling limitations, stress conventions, and production analysis workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

AVOGADRO = 6.02214076e23  # particles / mol


def density_g_cm3(mass_amu, volume_nm3):
    """rho = M/V; use 1 amu approximately 1/N_A g and 1 nm³ = 1e-21 cm³."""
    return mass_amu * 1.0e21 / AVOGADRO / np.asarray(volume_nm3)


def chain_dimensions(positions_nm, masses_amu, backbone=None):
    """Return each chain's mass-weighted Rg² and optional backbone R² (nm²).

    ``positions_nm`` has shape (chains, atoms, 3). Coordinates must keep each
    molecule whole across periodic boundaries. ``backbone`` lists chain-local
    atom indices in backbone order; the first and last define the endpoints.
    """
    positions = np.asarray(positions_nm, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    centre = (positions * masses[None, :, None]).sum(axis=1) / masses.sum()
    displacement = positions - centre[:, None, :]
    distances2 = (displacement**2).sum(axis=2)
    rg2 = (distances2 * masses).sum(axis=1) / masses.sum()
    r2 = None
    if backbone is not None:
        end_to_end = positions[:, backbone[-1]] - positions[:, backbone[0]]
        r2 = (end_to_end**2).sum(axis=1)
    return rg2, r2


def synthetic_results():
    """Work each definition on transparent data with known answers."""
    # Density: total cell mass, not molecular weight of just one chain.
    rho = density_g_cm3(60_000.0, 100.0)

    # Four equal-mass beads along x; all y and z coordinates are zero.
    positions = np.zeros((1, 4, 3))
    positions[0, :, 0] = [0.0, 1.0, 2.0, 3.0]
    rg2, r2 = chain_dimensions(positions, np.ones(4), [0, 1, 2, 3])

    # Specific volume (cm³/g) on two KNOWN branches at a fixed scan rate.
    # Their intersection, not a chosen breakpoint temperature, is this Tg.
    glass_t = np.array([250.0, 275.0, 300.0, 325.0])  # K
    melt_t = np.array([375.0, 400.0, 425.0, 450.0])
    glass_v = 1.0 + 2.0e-4 * (glass_t - 350.0)
    melt_v = 1.0 + 6.0e-4 * (melt_t - 350.0)
    slope_glass, intercept_glass = np.polyfit(glass_t, glass_v, 1)
    slope_melt, intercept_melt = np.polyfit(melt_t, melt_v, 1)
    tg = (intercept_glass - intercept_melt) / (slope_melt - slope_glass)

    # Small engineering strains; stress is tension-positive, in MPa.
    strain = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    axial_stress = 2.0 + 3000.0 * strain
    lateral_stress = np.full(strain.size, 2.0)
    tensile_stress = axial_stress - lateral_stress
    young = np.polyfit(strain, tensile_stress, 1)[0]
    transverse_strain = -0.35 * strain
    poisson = -np.polyfit(strain, transverse_strain, 1)[0]

    # Bulk and shear slopes measured separately. Positive pressure compresses.
    pressure_mpa = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    volume_ratio = np.exp(-pressure_mpa / (10_000.0 / 3.0))  # V(P)/V(0)
    log_volume_slope = np.polyfit(pressure_mpa, np.log(volume_ratio), 1)[0]
    bulk = -1.0 / log_volume_slope
    shear_strain = strain  # engineering gamma, not tensor epsilon_xy = gamma/2
    shear_stress = (10_000.0 / 9.0) * shear_strain
    shear = np.polyfit(shear_strain, shear_stress, 1)[0]

    # Nominal stress = F/A0 = true stress * A/A0. Same force, different area.
    true_stress_mpa, area_nm2, reference_area_nm2 = 100.0, 8.0, 10.0
    nominal = true_stress_mpa * area_nm2 / reference_area_nm2

    # Step shear strain gamma0: remove the baseline before dividing by strain.
    times_ps = np.array([0.0, 10.0, 20.0])
    gamma0, baseline_mpa = 0.01, 2.0
    stress_mpa = baseline_mpa + gamma0 * 1000.0 * np.exp(-times_ps / 10.0)
    relaxation = (stress_mpa - baseline_mpa) / gamma0

    # Einstein relation in 3 dimensions: long-time MSD = 6 D t + intercept.
    # Here the MSD is supplied, not estimated from one short displacement.
    lag_ps = np.array([100.0, 200.0, 300.0, 400.0])
    msd_nm2 = 0.1 + 6.0 * 0.002 * lag_ps
    diffusion = np.polyfit(lag_ps, msd_nm2, 1)[0] / 6.0

    return {
        "density_g_cm3": float(rho),
        "rg2_nm2": float(rg2[0]),
        "end_to_end2_nm2": float(r2[0]),
        "tg_k": float(tg),
        "young_mpa": float(young),
        "poisson": float(poisson),
        "bulk_mpa": float(bulk),
        "shear_mpa": float(shear),
        "nominal_mpa": nominal,
        "relaxation_mpa": relaxation,
        "diffusion_nm2_ps": float(diffusion),
    }


def run_demo():
    """Print the worked answers; no simulation or openmmpolymer import needed."""
    result = synthetic_results()
    print("Synthetic examples (noiseless; not material predictions)")
    print(f"Density = {result['density_g_cm3']:.6f} g/cm³")
    print(f"Rg² = {result['rg2_nm2']:.6f} nm²")
    print(f"Rg = {np.sqrt(result['rg2_nm2']):.6f} nm")
    print(f"R² = {result['end_to_end2_nm2']:.6f} nm²")
    print(f"Tg = {result['tg_k']:.6f} K")
    print(f"E = {result['young_mpa']:.6f} MPa")
    print(f"Poisson ratio = {result['poisson']:.6f}")
    print(f"K = {result['bulk_mpa']:.6f} MPa")
    print(f"G = {result['shear_mpa']:.6f} MPa")
    print(f"Nominal stress = {result['nominal_mpa']:.6f} MPa")
    for time, modulus in zip((0, 10, 20), result["relaxation_mpa"], strict=True):
        print(f"G(t={time} ps) = {modulus:.6f} MPa")
    print(f"D = {result['diffusion_nm2_ps']:.6f} nm²/ps")
    print(f"D = {result['diffusion_nm2_ps'] * 1e-6:.6g} m²/s")


def run_saved(directory, stage, discard_fraction):
    """Use public file readers, then calculate density and sizes directly."""
    # These imports are lazy so that the default demonstration needs only NumPy.
    from openmmpolymer.timeseries import read_state_data
    from openmmpolymer.trajectory import (
        AnalysisError,
        backbone_indices,
        load_manifest,
        open_run,
        stage_files,
    )

    manifest = load_manifest(directory)
    files = stage_files(directory, stage, manifest=manifest)
    if files.csv is None:
        raise AnalysisError(f"Stage {files.stage!r} did not record a state-data CSV.")
    state = read_state_data(files.csv, stage=files.stage)
    start = int(state.n_rows * discard_fraction)
    volumes = state.volume_nm3[start:]
    reported = state.density_g_cm3[start:]
    if not np.all(np.isfinite(volumes) & (volumes > 0)):
        raise AnalysisError("Retained CSV volumes must be finite and positive.")
    if not np.all(np.isfinite(reported) & (reported > 0)):
        raise AnalysisError("Retained CSV densities must be finite and positive.")
    print(f"Saved stage: {files.stage}")
    print(f"Rows retained: {volumes.size}/{state.n_rows}")
    print(f"Mean reported density = {reported.mean():.6f} g/cm³")
    print("Fixed discard is illustrative; it does not establish equilibration.")
    if files.topology is None:
        print("No saved coordinates: mass-based density and chain dimensions skipped.")
        return

    ensemble = open_run(directory, files.stage)
    mass_amu = ensemble.n_chains * ensemble.masses_amu.sum()
    density = density_g_cm3(mass_amu, volumes)
    print(f"Mean mass/volume density = {density.mean():.6f} g/cm³")
    print("Mass and Rg weights are topology element-mass estimates, not System masses.")
    print("Average M/V per row; M divided by the average V is a different estimator.")

    recorded_backbone = (manifest.chains or {}).get("backbone")
    backbone = (
        None
        if recorded_backbone is None
        else backbone_indices(recorded_backbone, ensemble.atoms_per_chain)
    )
    first_frame = int(ensemble.n_frames * discard_fraction)
    rg_frames, rg2_frames, r2_frames = [], [], []
    for frame in ensemble.frames(start=first_frame):
        chains = ensemble.per_chain(frame.positions_nm)
        rg2, r2 = chain_dimensions(chains, ensemble.masses_amu, backbone)
        rg_frames.append(np.sqrt(rg2).mean())
        rg2_frames.append(rg2.mean())
        if r2 is not None:
            r2_frames.append(r2.mean())
    mean_rg2 = np.mean(rg2_frames)
    print(f"Frames retained: {len(rg2_frames)}/{ensemble.n_frames}")
    print(f"Mean Rg = {np.mean(rg_frames):.6f} nm")
    print(f"Mean Rg² = {mean_rg2:.6f} nm²")
    print(f"RMS Rg = {np.sqrt(mean_rg2):.6f} nm")
    if r2_frames:
        print(f"Mean R² = {np.mean(r2_frames):.6f} nm²")
    else:
        print(
            "No recorded backbone: R² skipped (atom order does not define endpoints)."
        )
    if ensemble.is_snapshot:
        print("One saved snapshot: chain sizes only; no time average or uncertainty.")
    else:
        print(
            "Frames and chains can be correlated; these means have no uncertainty estimate."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", type=Path, help="read a completed run instead of toy data"
    )
    parser.add_argument(
        "--stage", help="stage name; default is the last completed stage"
    )
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.5,
        help="discard this initial fraction of CSV rows and frames (default: 0.5)",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.discard_fraction < 1.0:
        parser.error("--discard-fraction must be finite and in [0, 1)")
    if args.stage is not None and args.run is None:
        parser.error("--stage requires --run")
    if args.run is None:
        run_demo()
    else:
        from openmmpolymer.trajectory import AnalysisError

        try:
            run_saved(args.run, args.stage, args.discard_fraction)
        except AnalysisError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
