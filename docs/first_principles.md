# Calculating polymer properties from first principles

[Quickstart](../README.md) · [Workflow guide](guide.md) ·
[Runnable example](../examples/first_principles.py)

For complete numbered workflows for Tg, Tm, breaking strength, yield strength,
elongation at break and Young's modulus, use the
[step-by-step property examples](main_properties.md).

This tutorial starts with masses, coordinates, box vectors and stresses and
shows how they become the properties reported by openmmpolymer. Here, **first
principles means deriving the observables from their definitions in classical
statistical mechanics**. The potential energy and charges still come from a
parameterised force field; this is not an ab initio electronic-structure
calculation. Every result depends on that model and on the preparation,
temperature, pressure, size and time scale of the simulation.

Start with the small numerical example, then use a saved melt to calculate
density and chain dimensions. The later sections explain which additional
experiments supply thermal, mechanical and dynamic properties: one equilibrium
trajectory cannot supply them all.

## 1. Run the arithmetic without running dynamics

From the repository root:

```bash
python examples/first_principles.py
```

This takes seconds, needs only NumPy, and uses deterministic **synthetic data**.
It calculates the following known answers. These are checks of the arithmetic,
not predictions for polyethylene.

| Quantity | Input or construction | Expected result |
| --- | --- | --- |
| Density | 60,000 amu in 100 nm³ | 0.99632344 g/cm³ |
| Radius of gyration | Four equal masses at x = 0, 1, 2, 3 nm | Rg² = 1.25 nm²; Rg = 1.118034 nm |
| Squared end-to-end distance | Endpoints at 0 and 3 nm | R² = 9 nm² |
| Glass transition | Two specific-volume branches meeting at 350 K | Tg = 350 K |
| Young's modulus and Poisson's ratio | Linear stress and transverse strain | E = 3000 MPa; ν = 0.35 |
| Bulk and shear moduli | Independent synthetic pressure and shear curves | K = 3333.333333 MPa; G = 1111.111111 MPa |
| Nominal tensile stress | Differential true stress 100 MPa; A/A₀ = 0.8 | 80 MPa |
| Relaxation modulus | G(t) = 1000 exp(−t/10 ps) MPa | 1000, 367.879441, 135.335283 MPa at 0, 10, 20 ps |
| Diffusion coefficient | MSD slope 0.012 nm²/ps in three dimensions | 0.002 nm²/ps = 2 × 10⁻⁵ cm²/s |

The example deliberately exposes the calculations rather than calling the
property-fitting functions. It does not reproduce their uncertainty estimates
or their checks for whether a result is resolved.

## 2. Make a trajectory with the information you need

For a real polymer, install the [full environment](../README.md#installation).
The following is a complete preparation followed by a 2 ns NPT measurement at
450 K and 1 bar. Save it as a Python script and run it from the repository root.
Unlike the arithmetic example, this builds and parameterises 40 chains and runs
several nanoseconds of dynamics; runtime depends on your hardware. The chosen
durations illustrate a workflow and do not guarantee equilibration.

```python
from openmmpolymer import (
    ChainSpec,
    Protocol,
    Stage,
    TrajectoryOptions,
    build_melt,
    run_protocol,
    standard_melt_equilibration,
)

chain, run = build_melt(
    ChainSpec(
        monomer_smiles="[*]CC[*]", degree_of_polymerization=30, residue_name="PE"
    ),
    n_chains=40,
    directory="tutorial-run",
    target_density_g_cm3=0.85,
)
preparation = standard_melt_equilibration(
    target_temperature_k=450.0,
    npt_trajectory=TrajectoryOptions("xtc", interval_ps=10.0),
)
protocol = Protocol(
    name="first_principles",
    stages=(
        *preparation.stages,
        Stage(
            "06_measure",
            "npt",
            {
                "temperature_k": 450.0,
                "pressure_bar": 1.0,
                "duration_ps": 2000.0,
                "report_interval_ps": 5.0,
                "trajectory": TrajectoryOptions("xtc", interval_ps=10.0),
            },
        ),
    ),
)
print(f"Requested dynamics: {protocol.total_duration_ps / 1000:.3f} ns")
summary = run_protocol(
    protocol,
    run,
    "tutorial-run",
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
print(summary.manifest_path)
```

The same request resumes intact completed stages. Choose a new directory if
you change the polymer or preparation. `target_density_g_cm3` is used to check
the cell size; the NPT measurement determines the density.

Now run the example on the saved measurement:

```bash
python examples/first_principles.py --run tutorial-run --stage 06_measure \
  --discard-fraction 0.5
```

This computes density and chain dimensions directly and prints the retained
sample counts. It uses element masses from the topology, so its reconstructed
density can differ from the reporter if the System has modified masses. It
uses the recorded backbone for R²; without that record it skips R². A closing
snapshot can supply dimensions, but cannot establish their convergence. This
mode does not run dynamics or write property reports.

The discard fraction is an explicit demonstration choice, not an equilibration
detector. Inspect the time series and change the retained window when needed.
Use the [convergence tools](guide.md#observation-window-convergence) to test
whether estimates stabilise as the observation time grows.

| Raw information | Where to find it | Units / meaning |
| --- | --- | --- |
| Time, density, volume, energy, temperature | `06_measure.csv`, read with `read_state_data` | ps, g/cm³, nm³, kJ/mol, K |
| Coordinates and full periodic cell | `06_measure.xtc` plus `06_measure_topology.pdb`, read with `open_run` | The reader converts Å to nm |
| Backbone and chain block size | `manifest.json`, `chains` and `box` | Indices within one chain; atoms in contiguous chain blocks |
| Thermal or mechanical hold averages | Each stage's `samples` in `manifest.json` | Named arrays such as `segment_density_g_cm3` |
| Stress components | Mechanical stage `samples` | bar, **tension positive** |

Use the per-hold manifest samples when reproducing a scan report. They average
the measurement portion of each hold, normally its second half. CSV reports
may have a different sampling cadence and are not guaranteed to reproduce
those averages exactly. Analyse each replica separately before combining its
result with the others.

## 3. Density: divide mass by volume, then average

For row cell vectors assembled into a matrix H, the volume is
`V = abs(det(H))`. Multiplying edge lengths is valid only for an orthogonal
box. Let M be the sum of the System's particle masses, numerically in amu.
Using the molar-mass convention used by the simulation:

$$
\rho(t) = \frac{M}{N_A V(t)\,10^{-21}}
         = 0.001660539067\,\frac{M}{V(t)}\quad\mathrm{g/cm^3},
\qquad N_A = 6.02214076\times10^{23}.
$$

Here `1 nm³ = 10⁻²¹ cm³`. Continue in the Python session that built `run` to
compare an independent mass/volume calculation with the reporter:

```python
import numpy as np
import openmm as mm
from openmm import unit

from openmmpolymer import read_state_data, stage_files

files = stage_files("tutorial-run", "06_measure")
assert files.csv is not None
data = read_state_data(files.csv)
system = mm.XmlSerializer.deserialize(run.system_xml)
mass_amu = sum(
    system.getParticleMass(i).value_in_unit(unit.dalton)
    for i in range(system.getNumParticles())
)
start = data.n_rows // 2
rho = (1e21 / 6.02214076e23) * mass_amu / data.volume_nm3[start:]
print("Mean density (g/cm³):", rho.mean())
print("Largest reporter difference:", np.max(np.abs(rho - data.density_g_cm3[start:])))
```

The estimator is `mean(M/V)`, which generally differs from `M/mean(V)`.
Likewise, `mean(1/rho)` and `1/mean(rho)` differ in a fluctuating cell. The
thermal scan reports use **1/mean(rho)** for each temperature hold; preserve
that convention when comparing results.

Saved rows are correlated. For a stationary observable x, a simple uncertainty
estimate is to divide its retained series into B equal, contiguous blocks
longer than its correlation time, calculate each block mean, and use
`std(block_means, ddof=1) / sqrt(B)`. Increase the block length and look for a
stable uncertainty while retaining enough blocks. A small error from treating
every MD step as independent is misleading; a drifting mean needs more
equilibration, not a smaller error bar.

## 4. Chain dimensions: use whole, unwrapped chains

For a chain with atom masses mᵢ and coordinates rᵢ:

$$
\mathbf r_{\rm cm}=\frac{\sum_i m_i\mathbf r_i}{\sum_i m_i},\qquad
R_g^2=\frac{\sum_i m_i|\mathbf r_i-\mathbf r_{\rm cm}|^2}{\sum_i m_i},\qquad
\mathbf R=\mathbf r_{\rm backbone,last}-\mathbf r_{\rm backbone,first}.
$$

Rg uses **all atoms**, including hydrogens; R uses the endpoints of the
recorded backbone, which need not be the first and last atoms in storage.
For one retained frame:

```python
from openmmpolymer import open_run

ensemble = open_run("tutorial-run", "06_measure")
frame = next(ensemble.frames(start=ensemble.n_frames // 2))
positions = ensemble.per_chain(frame.positions_nm)  # (chains, atoms, xyz)
masses = ensemble.masses_amu  # topology element masses, as in chain_conformation
centres = (positions * masses[None, :, None]).sum(axis=1) / masses.sum()
distance_sq = ((positions - centres[:, None, :]) ** 2).sum(axis=2)
rg_sq = (distance_sq * masses[None, :]).sum(axis=1) / masses.sum()
vectors = positions[:, chain.backbone[-1]] - positions[:, chain.backbone[0]]
r_sq = (vectors**2).sum(axis=1)
print("Mean Rg (nm):", np.sqrt(rg_sq).mean())
print("Mean R² (nm²):", r_sq.mean())
print("Ratio of mean squares:", r_sq.mean() / rg_sq.mean())
```

Repeat over retained frames and pool chains and frames with equal weight, as
the runnable example does. The reported mean Rg is `mean(sqrt(Rg²))`, not
`sqrt(mean(Rg²))`. The ideal long Gaussian-chain comparison is
`mean(R²)/mean(Rg²) ≈ 6`; it is a model diagnostic, not a proof of equilibrium.

The finite-chain characteristic ratio is
`C_n = mean(R²) / (n_bonds * mean(bond_length)²)` in this package. Pool bond
lengths across frames and chains before taking this ratio. It approaches C∞
only in the long-chain limit. For persistence length, form unit backbone bond
vectors uᵢ, average `uᵢ · uᵢ₊ₖ` over bonds, chains and frames, and fit the
initial positive correlation to `exp(−s/l_p)`, where
`s = k * mean(bond_length)`. The package fits log correlation through its
first 1/e crossing; local bond oscillations can make a single exponential a
poor description. A chain too short to show decay cannot determine l_p.

Keep chains whole for shape measurements and continuously unwrapped for
displacements. The trajectory settings above leave wrapping off. Applying a
minimum-image convention independently to each atom does not repair a chain
that crosses the box. The [structure workflow](guide.md#structure) provides
`chain_conformation` and `persistence_length` for the production calculations.

## 5. Thermal transitions: change the temperature

**Glass transition.** Cool a prepared melt at fixed pressure, hold each
temperature, average the settled density, and form `v(T) = 1/mean(rho)` in
cm³/g. In the glass and melt branches approximate:

$$
v_g(T)=a_g+b_gT,\qquad v_m(T)=a_m+b_mT,\qquad
T_g=\frac{a_g-a_m}{b_m-b_g}.
$$

This follows by setting the two specific volumes equal. The volumetric thermal
expansion coefficient on either branch is `alpha = (1/v) dv/dT = b/v`, in K⁻¹;
the fitted slope b alone is not an expansivity.

The numerical example constructs two branches and fits them with NumPy. To
generate a simple cooling run using the `chain` and `run` from section 2:

```python
from openmmpolymer import melt_quench, run_protocol

run_protocol(
    melt_quench(
        t_start=600.0,
        t_end=200.0,
        step_k=20.0,
        hold_ps=200.0,
        target_temperature_k=600.0,
        anneal_t_low_k=450.0,
    ),
    run,
    "cooling-run",
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
```

This performs its own preparation and a coarse cooling ladder. On that saved
run, this code exposes the same least-squares search used by `glass_transition`:

```python
import numpy as np

from openmmpolymer import glass_transition, quench_curve

# For a different Tg scan, select one complete cooling pass.
# Do not mix coarse/fine passes or replicas.
curve = quench_curve("cooling-run", stage="06_quench")
t = curve.temperature_k  # already sorted in increasing order
v = curve.specific_volume_cm3_g
if len(t) < 8:
    raise ValueError("Need at least four points on each branch")
candidates = []
for split in range(4, len(t) - 3):
    glass = np.polyfit(t[:split], v[:split], 1)  # slope, intercept
    melt = np.polyfit(t[split:], v[split:], 1)
    residual = ((v[:split] - np.polyval(glass, t[:split])) ** 2).sum()
    residual += ((v[split:] - np.polyval(melt, t[split:])) ** 2).sum()
    candidates.append((residual, glass, melt))
_, glass, melt = min(candidates, key=lambda candidate: candidate[0])
if abs(melt[0] - glass[0]) < 1e-12:
    raise ValueError("Parallel branches do not define a transition")
tg_k = (glass[1] - melt[1]) / (melt[0] - glass[0])
checked = glass_transition(curve)
print("Intersection (K):", tg_k, "Resolved:", checked.resolved)
```

The [adaptive Tg workflow](guide.md#glass-transition) refines a candidate region.
A minimum residual always exists, even for data with no transition: require a
meaningful slope change and an intersection near the sampled change of slope.
Record the cooling rate `1000 * step_k / hold_ps` in K/ns. This is an apparent,
history-dependent Tg at that rate, not a zero-rate prediction.

**Melting temperature.** Start from a separately prepared crystal and heat it
at fixed external pressure. Its enthalpy is

$$
H = U + K_{\rm kinetic} + P_{\rm ext}V,
\qquad
H_{\rm kJ/mol} = U_{\rm kJ/mol}+K_{\rm kJ/mol}
                 +0.0602214076\,P_{\rm bar}V_{\rm nm^3}.
$$

The factor converts bar nm³ into kJ/mol of **simulation cells**; divide by the
number of chains or repeat units before quoting energy per mole of those
objects. Seek a coincident positive jump in v and H across a heating interval
`[T_i, T_{i+1}]`. The report gives its midpoint as the apparent Tm and retains
the bracket. This is a discontinuity criterion, unlike the Tg slope change.
The [melting workflow](guide.md#melting-temperature) checks both signals;
inspect loss of crystalline order as well. Heating an amorphous packed melt
does not establish a melting point.

## 6. Elastic constants: apply a small perturbation

Stress is the mechanical response conjugate to strain. In a simple classical
particle description the pressure tensor contains kinetic and virial terms,
`P = (sum m v⊗v + W)/V`, and the tension-positive stress is `sigma = −P`.
For a periodic constrained system, the virial must include the force and
constraint contributions consistently. Do not replace it with a sum of wrapped
positions times forces. This package reads OpenMM's barostat estimators; the
flexible barostat supplies physical shear stress via `computeStressTensor`
with its recorded molecular-scaling convention.
See [OpenMM's stress estimator documentation](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.MonteCarloFlexibleBarostat.html#openmm.openmm.MonteCarloFlexibleBarostat.computeStressTensor).

For uniaxial extension along z, with x and y allowed to relax at pressure:

$$
\epsilon_z=L_z/L_{z0}-1,\quad
\epsilon_\perp=\tfrac12(\epsilon_x+\epsilon_y),\quad
s=\sigma_{zz}-\tfrac12(\sigma_{xx}+\sigma_{yy}),\quad
E=\frac{ds}{d\epsilon_z},\quad
\nu=-\frac{d\epsilon_\perp}{d\epsilon_z}.
$$

Subtracting the lateral stress removes the hydrostatic background. Fit an
intercept as well as a slope; forcing the line through zero would turn
pre-existing stress into apparent stiffness. Stored stresses are already
tension positive: multiply bar by **0.1** to obtain MPa, without another sign
change.

Generate a measurement with the [modulus workflow](guide.md#elastic-constants).
For example, the following uses one replica to keep the data path simple;
increase the replica count for production uncertainty estimates:

```python
from openmmpolymer import ModulusSpec, run_modulus_scan

report = run_modulus_scan(
    run,
    "modulus-run",
    spec=ModulusSpec(temperature_k=298.15, n_replicas=1, max_strain=0.05),
)
```

Read that replica's curve and calculate the two slopes directly:

```python
import numpy as np

from openmmpolymer import poisson_ratio, stress_strain, youngs_modulus

curve = stress_strain("modulus-run")  # this example has only one replica
inside = (curve.strain >= 0.0) & (curve.strain <= 0.015)
if inside.sum() < 5:
    raise ValueError("Need at least five points in the elastic window")
strain = curve.strain[inside]
stress = curve.stress_mpa - curve.lateral_stress_mpa.mean(axis=1)
youngs_mpa, intercept_mpa = np.polyfit(strain, stress[inside], 1)
nu = -np.polyfit(strain, curve.lateral_strain[inside].mean(axis=1), 1)[0]
print("E (MPa), nu:", youngs_mpa, nu)
print("Resolved:", youngs_modulus(curve).resolved, poisson_ratio(curve).resolved)
```

The window must lie in a linear regime and contain enough stress signal. Try
smaller windows and longer holds; a neatly fitted slope alone is insufficient.
At 298 K a particular polymer may still be above Tg. A melt need not show a
solid-like modulus over the chosen measurement window.

The other two moduli need separate perturbations:

| Measurement | Definition | Direct calculation from hold averages |
| --- | --- | --- |
| Bulk K at fixed temperature | `K = −V dP/dV = −1/(d ln V/dP)` | Fit `−ln(rho)` against pressure in bar, then `K_MPa = −0.1/slope`; compare compression and decompression |
| Shear G | `G = d sigma_xy/d gamma` | Fit shear stress in MPa against engineering shear gamma; use the recorded physical stress estimator |
| Load-controlled E | `E = d s/d epsilon_z` | Impose differential stress with the anisotropic barostat and measure the axial strain |

For a homogeneous isotropic linear elastic solid,
`G = E/[2(1+nu)]` and `K = E/[3(1−2nu)]`. Compare these with independently
measured G and K. Deriving all four from just E and ν would remove that check.
The synthetic example constructs compatible curves so these identities hold
exactly. Real data can disagree because of noise, anisotropy or incomplete
relaxation.

## 7. Yield, breaking and elongation: keep track of area

The [tensile workflows](guide.md#tensile-strength-yield-breaking-and-elongation-at-break)
continue extension beyond the elastic window. The recorded stress is Cauchy
stress (force per **current** area). The reported strength uses nominal stress
(force per **initial** area), after removing the lateral background:

$$
s_{\rm nominal}=s\,\frac{A}{A_0}
 =s(1+\epsilon_x)(1+\epsilon_y).
$$

For example, a differential true stress of 100 MPa acting over 80% of the
initial transverse area corresponds to 80 MPa nominal stress. Use the measured
lateral dimensions; replacing `A/A0` with `1/(1+epsilon_z)` assumes constant
volume, which need not hold.

For offset yield, fit the elastic nominal curve as `s = E_fit * epsilon + b`,
then find the first later crossing with
`s_offset = E_fit * (epsilon − 0.002) + b`. Interpolate within its sampled
bracket. This is a 0.2% proof stress; without unloading it does not demonstrate
permanent deformation.

For breaking strength, take the maximum nominal stress only after a terminal
drop is confirmed: by default, at least three final consecutive holds at or
below half that peak. Elongation at break is `100 * epsilon_z` at the **first
hold of the confirmed terminal drop**, not at the peak or the maximum imposed
strain. No confirmed drop means no resolved break.

The supported force fields have fixed bonded connectivity and harmonic bond
terms, so stress loss here describes chain sliding or separation, not covalent
bond fracture. The model's bond energy is described in the
[OpenMM force documentation](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#harmonicbondforce).

## 8. Relaxation and diffusion: observe the time dependence

**Stress relaxation.** Apply a small step strain, hold the box fixed and
subtract the measured pre-strain baseline. For simple shear:

$$
G(t)=\frac{\sigma_{xy}(t)-\sigma_{xy,\rm before}}{\gamma_0}.
$$

For a tensile step, isotropic linear elasticity gives
`sigma = 2G epsilon + lambda tr(epsilon) I`; subtracting transverse from axial
stress cancels the lambda term. Therefore:

$$
G(t)=\frac{s(t)-s_{\rm before}}{2(\epsilon_z-\epsilon_\perp)}.
$$

The [relaxation workflow](guide.md#stress-relaxation) records that denominator
as `relax_strain_measure`. Use it rather than dividing by the axial step alone.
The imposed lateral strain can differ from the material's relaxed Poisson
response. The linear constitutive relation is an approximation at finite
strain: repeat with smaller steps to check it.

A stretched exponential fits `G(t) = G0 exp[−(t/tau)^beta]`; its integral time
is `tau * Gamma(1/beta) / beta`. A Prony representation fits
`G(t) = G_inf + sum G_i exp(−t/tau_i)` with nonnegative weights. A decaying
fluid has `eta = integral G(t) dt`; in a solid with nonzero G_inf this total
integral diverges, although the transient excess has a finite integral.
`1 MPa ps = 10⁻⁶ Pa s`. A short observation window cannot determine an
unobserved tail or distinguish a slow decay from an equilibrium plateau.

**Chain diffusion.** Compute each whole chain's centre of mass, remove the
cell's collective drift, and average displacements over chains and time
origins:

$$
\mathrm{MSD}(\tau)=\left\langle|\mathbf r_{\rm cm}(t+\tau)
 -\mathbf r_{\rm cm}(t)|^2\right\rangle,\qquad
D=\frac16\frac{d\,\mathrm{MSD}}{d\tau}.
$$

Only fit a regime with `MSD proportional to tau`, beyond ballistic or
subdiffusive motion and before too few time origins remain. A slope in nm²/ps
becomes D in cm²/s by multiplying by `1e-2 / 6`. The synthetic example starts
from an exactly diffusive MSD curve. Real trajectories need continuous
unwrapped coordinates; see the
[MDAnalysis Einstein-MSD derivation](https://docs.mdanalysis.org/stable/documentation_pages/analysis/msd.html).

For a clean transport measurement, equilibrate the density with NPT and then
collect a separate NVT trajectory. The package's `centre_of_mass_msd` can
remove isotropic box scaling from NPT data, but this is not a general
correction for a shearing or strongly changing cell. It returns no diffusion
coefficient when the measured log-log slope fails its diffusive-regime check.

The end-to-end relaxation function uses the **raw** backbone vectors:
`C_R(tau) = mean(R(t) · R(t+tau)) / mean(R²)`. The package reports the first
interpolated crossing of `1/e`; a trajectory without that crossing has no
resolved relaxation time. This is different from correlating unit vectors,
which discards changes in chain extension.

## 9. Pair structure and deciding whether a number is supported

For an intermolecular radial distribution, count unique pairs on different
chains in a shell of volume `dV = 4*pi/3 * (r_outer³ − r_inner³)`. With N
selected atoms in the cell and n selected atoms per chain, the ideal-gas
reference count per frame at fixed volume V is
`N * (N−n) * dV / (2V)`. Divide the measured count by that reference to obtain
g(r). The coordination number follows directly from
`2 * cumulative_pair_count / (n_frames * N)`.

The implementation uses the mean volume for this reference across frames.
It excludes same-chain pairs and selects heavy atoms by default. Distances
need the full periodic minimum-image geometry; shells must stay within half
the smallest cell face spacing.

The number structure factor is a separate observable:

$$
S(\mathbf q)=\frac1N\left|\sum_j e^{i\mathbf q\cdot\mathbf r_j}\right|^2,
\qquad \mathbf q=2\pi\mathbf n H^{-T},\quad \mathbf n\in\mathbb Z^3\setminus\{0\}.
$$

Average over frames and reciprocal vectors of similar magnitude. This
includes both intra- and intermolecular contributions of the selected atoms,
so it is not the Fourier transform of the intermolecular g(r) above. It also
uses no X-ray or neutron scattering weights. The finite cell limits the
smallest q and how many vectors support each bin. Use `radial_distribution`
and `structure_factor` from the [structure workflow](guide.md#structure) for
these periodic calculations.

For a result you intend to interpret, report the actual temperature, pressure,
chain length, cell size, force field, preparation, retained window, and rate
or step size. Check stability against longer holds, smaller perturbations,
more frames and independent initial structures as appropriate. Workflow
replicas branch from one prepared structure with fresh velocities; they
measure trajectory variability, not uncertainty over morphologies.

Keep `resolved` and `notes` beside every headline. A successful numerical fit
does not prove that a transition, diffusive regime, linear response or break
was observed. The [rate](guide.md#rate-dependence) and
[observation-window](guide.md#observation-window-convergence) checks address
different limits; neither turns a fast molecular-dynamics result into an
experimental equilibrium value automatically.
