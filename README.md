# openmmpolymer

Build, pack, parameterise, equilibrate and measure all-atom polymer melts with
[OpenMM](https://openmm.org), [packmol](https://m3g.github.io/packmol) and
[forcefill](https://github.com/LouieSlocombe/forcefill).

Give it a monomer SMILES and it grows chains with the right dimensions,
charges them, turns them into an OpenMM force field, packs them into a periodic
cell and takes the cell through minimisation, push-off, high-temperature
equilibration, compression, annealing and whatever measurement you ask for -
a glass transition, a melting point, elastic constants, yield, breaking,
elongation at break, stress relaxation, structure - recording every stage in a
manifest so a run the queue killed picks up where it stopped.

## Installation

Python 3.12 or newer and OpenMM 8.6.1 or newer, from conda-forge:

```bash
conda env create -f build_tools/environment.yml
conda activate openmmpolymer
python -m pip install -e . --no-deps
```

conda-forge is the only route that resolves: AmberTools, which supplies the
`packmol` executable, is not a Python package, and `openmmforcefields >= 0.16`
has never been published to PyPI. The figures are built as
`matplotlib.figure.Figure` objects without `pyplot`, so no display is needed.

## A polyethylene melt

```python
from openmmpolymer import (
    ChainSpec,
    build_melt,
    run_protocol,
    standard_melt_equilibration,
)

chain, run = build_melt(
    ChainSpec(
        monomer_smiles="[*]CC[*]", degree_of_polymerization=30, residue_name="PE"
    ),
    n_chains=40,
    directory="run",
    target_density_g_cm3=0.85,
)
summary = run_protocol(
    standard_melt_equilibration(target_temperature_k=450.0),
    run,
    "run",
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
print(summary.manifest_path)
```

`build_melt` runs the four layers in turn - `build_chain`, `assign_charges`,
`build_polymer_forcefield`, then `pack_box`, `assemble_box`, `check_packing`
and `prepare_run` - each of which is usable on its own. What it adds is safe
rebuilding: the build and a record of it go in `run/build`, and a second call
rebuilds in scratch space and goes on only if the chain, cell and System match
the record and every run already started there.

Or from the command line:

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE -t 450 -o run -v
```

Run it again and it picks up from the last stage that finished. Resume checks
the System, topology, starting coordinates, seed, settings and each stage's
inputs before reusing anything, and a missing or changed upstream state
invalidates everything downstream of it. The CLI records its whole request in
`build_request.json` before building, so a different request cannot overwrite
an existing run; use a new output directory for a different simulation.

End caps and chain statistics are part of the chain. An acid-terminated PLA
chain needs a hydroxyl cap on its carbonyl end:

```bash
openmmpolymer '[*]OC(C)C(=O)[*]' -n 20 -c 40 -r PLA \
  --tail-cap '[*]O' --charge-method nagl --dry-run -o pla-build
```

`--head-cap` and `--tail-cap` each take a fragment with one `[*]`.
`--characteristic-ratio` sets the C-infinity the chains are grown to and
checked against; the default, 7.0, is polyethylene's.

## The constraints everything follows from

**One molecule, one residue.** forcefill will not parameterise a residue
bonded to its neighbours, and it is right not to: a stand-alone GAFF treatment
of a chain-linked residue is not valid. So a chain is built, charged and
parameterised whole.

**Charges are assigned here, not by the backend.** AM1-BCC runs a
semi-empirical QM calculation that stops being tractable at a few hundred
atoms - a twenty-unit polyethylene chain is already there - so the default is a
graph neural network that scales with the number of atoms. openmmforcefields
uses charges a molecule already carries instead of running AM1-BCC.

| Method | Practical chain size | Notes |
|---|---|---|
| `nagl` | thousands of atoms | The default. AM1-BCC quality. |
| `am1bcc` | a few hundred atoms | The reference; `sqm` is the wall. |
| `gasteiger` | unbounded | Qualitative: fine for a polyolefin, several per cent out for an ester or an amide. |

**One conformer per chain, grown rather than embedded.** packmol places each
structure as a rigid body, so packing one conformer forty times gives forty
identical coils: build `n_conformers=n_chains`. RDKit's ETKDG collapses a long
chain into a globule, so chains of four units or more are grown unit by unit
from torsions drawn to match the characteristic ratio, with a self-avoidance
check, and `build_chain` says when the result drifts from it.

**Equilibration is reported, not claimed.** packmol places chains that do not
interpenetrate, and the Rouse time of a melt is tens of nanoseconds - longer
than the high-temperature stages here. The manifest records `<R²>`, the radius
of gyration and the measured characteristic ratio against the expected one.
`melt_equilibration` gives a verdict: the volume must have stopped drifting
faster than its own noise *and* the chains' centres of mass must have moved
further than the chains are big, diffusively. It needs a trajectory from the
equilibration (`npt_trajectory` on the protocol, `npt_trajectory_ps` on a
`TgSpec`, `--check-melt` on the command line); without one the verdict is
False and `unchecked` says which half was missing.

**The backbone is recorded with the run.** The backbone indices come from the
attachment points the caps consumed, and nothing downstream can recover them
from the structure alone. A protocol given `chain_backbone` records it in the
manifest; analysis uses a backbone you give it, then the recorded one, and
otherwise infers the longest shortest path through one chain's heavy atoms and
says so in `backbone_source`.

## Measuring properties

Every measurement follows the same pattern:

- It equilibrates the cell once, at the measurement conditions, and branches
  every pass and replica from that one equilibrated cell with fresh
  velocities. Replica spread is trajectory variability at one starting
  structure, not uncertainty over morphologies.
- It records what it was asked for before any dynamics. A rerun resumes only
  with the same request and intact states; `resume=False` reruns everything.
  Long ladders are split into stages, so an interrupted scan resumes mid-way.
- `max_total_ns` prices the whole scan and refuses before anything is written.
  The command line prints that price first, and `--dry-run` stops once the
  cell is built.
- Every result carries `resolved` and `notes`. Unresolved means the data did
  not support a number - a fit that found a corner in noise, a curve that
  never failed, a replica that is missing - and the headline is then `None`.
- `analyse_*` rereads a finished directory without running dynamics, and
  `write_*_report` writes its JSON and figures to `<run>/analysis`.
  `openmmpolymer --analyse RUN_DIR` works out from what a directory recorded
  which reports it can produce - a quench gets a glass transition, a heating
  scan a melting report, a deformation its mechanical report, any stage with
  coordinates a structure report - and never modifies the manifest.
- Molecular dynamics rates are some ten decades faster than experiment, so
  every number is an apparent one at its rate, and the rate travels with it.

### Glass transition

Resolving a transition to a few kelvin needs small steps and long holds, and
paying for that from the melt all the way down is most of the cost for none of
the answer. `run_tg_scan` goes down twice: a coarse ladder to find the break,
then a fine one across a window centred on it.

```python
from openmmpolymer import TgSpec, analyse_tg, run_tg_scan, write_tg_report

result = run_tg_scan(
    run,
    "run",
    spec=TgSpec(melt_temperature_k=650.0, t_floor_k=150.0, npt_trajectory_ps=10.0),
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
print(result.temperature_k, result.fine_schedule.cooling_rate_k_per_ns)
write_tg_report(analyse_tg("run"))
```

The fine pass starts from a state the coarse pass saved on the way past, not
from the melt and not from the bottom of the coarse ladder: a glass remembers
how it was cooled, and a window entered by reheating a solid measures a
different thermal history. The scan refuses to guess - if the coarse fit found
a corner in noise it stops and says so, and `tg_approx_k` names the window
yourself. `cooling_rate_series` repeats the fine window at several rates from
that one configuration, and `cooling_rate_extrapolation` fits how the
transition moves, as a straight line in log rate or in the
Vogel-Fulcher-Tammann form (`wlf_constants` converts to WLF). Extrapolating to
a calorimeter's 10 K/min is about ten decades, so it is always unresolved; the
measured shift per decade of rate is the robust number.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol tg --check-melt -o run -v
```

### Melting temperature

Melting needs a crystalline or semicrystalline starting cell, which the
monomer-to-melt workflow does not make. Prepare the crystal separately, with a
force field appropriate for its solid phase, and `run_tm_scan` equilibrates it
below the expected melting range and heats it through a temperature ladder at
fixed pressure.

```python
from openmmpolymer import (
    TmSpec,
    analyse_melting,
    load_crystal,
    run_tm_scan,
    write_melting_report,
)

crystal_run = load_crystal("crystal.pdb", "system.xml")
result = run_tm_scan(
    crystal_run,
    "melting",
    crystalline=True,
    spec=TmSpec(t_start_k=250.0, t_end_k=650.0, step_k=10.0, hold_ps=1000.0),
)
print(result.temperature_k, result.bracket_k, result.resolved)
write_melting_report(analyse_melting("melting"))
```

`crystalline=True` is your declaration about the starting structure, not a
check of it. The scan looks for a coincident jump in specific volume and
enthalpy; smooth expansion, a glass-transition kink or conflicting signals
stay unresolved. The temperature reported is the midpoint of a sampled heating
bracket - an apparent melting point at that heating rate, which superheating,
cell size, morphology and the force field can all shift. Inspect the loss of
order (`trajectory_ps` saves coordinates) and repeat with other starting
configurations.

`load_crystal` takes the prepared PDB, with its periodic box and bonds, and an
OpenMM `XmlSerializer` System in the same atom order with no thermostat or
barostat, and checks them before anything runs. Molecules must have equal
atom counts in contiguous blocks. The command line takes the same two files:

```bash
openmmpolymer --protocol tm --crystal-pdb crystal.pdb --system-xml system.xml \
  --t-start 250 --t-end 650 --step-k 10 --hold-ps 1000 \
  --tm-trajectory-ps 10 --max-total-ns 50 -o melting
openmmpolymer --analyse melting
```

`--state-in` starts from a saved State of the same crystal. Pressure control is
anisotropic by default, so the three box lengths relax independently;
`--tm-barostat isotropic` holds their ratios.

### Elastic constants

OpenMM has no continuous deformation, so a strain rate is a staircase: scale
the cell by one increment, let it relax under a barostat holding the other two
axes at pressure, read the stress, repeat. `run_modulus_scan` walks that
ladder, and three more passes beside it.

```python
from openmmpolymer import (
    ModulusSpec,
    analyse_mechanics,
    run_modulus_scan,
    write_mechanical_report,
)

report = run_modulus_scan(
    run,
    "run",
    spec=ModulusSpec(temperature_k=298.15, max_strain=0.05),
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
print(report.youngs.modulus_mpa, report.replica_spread_mpa, report.resolved)
write_mechanical_report(analyse_mechanics("run"))
```

`E` and `nu` come from the extension, `K` from a pressure ladder up and back
down, and `G` from a shear ladder, each measured rather than derived. For an
isotropic solid those four constants are two, so the gap between the measured
`K` and `G` and the ones `E` and `nu` imply checks all of them at once. The
load pass is an independent estimate of `E`: it imposes a known stress with an
anisotropic barostat and measures the box, with no virial anywhere, and
`method_gap` reports how far the two methods disagree.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol modulus -t 298 -o run -v
```

`--skip bulk shear` leaves out the passes you do not need. Which side of Tg
298 K falls on is a property of the polymer, and nothing here knows it: above
Tg the stiffness is entropic, far smaller and far slower to relax than a 50 ps
hold allows, and an unresolved modulus on a melt is the honest answer.

### Tensile strength: yield, breaking and elongation at break

`run_yield_scan`, `run_breaking_scan` and `run_elongation_scan` share one
engine. Each extends replicas from the equilibrated cell in compounding
increments while the transverse dimensions relax at the set pressure; a
replica's ladder is split into resumable chunks that share one unstrained
reference box. The recorded stress is the instantaneous Cauchy stress, so the
reports convert the differential stress (axial minus mean transverse) to
nominal stress with the measured transverse area ratio `A / A0` rather than
assuming constant volume. The supported force fields have
[fixed harmonic bonds](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#harmonicbondforce),
so none of this models covalent fracture: a loss of stress is chains sliding
or separating. Report temperature, strain rate, criterion, force field, chain
length and cell size with every number.

```python
from openmmpolymer import BreakingSpec, ElongationSpec, YieldSpec
from openmmpolymer import run_breaking_scan, run_elongation_scan, run_yield_scan

yield_report = run_yield_scan(run, "yield_run", spec=YieldSpec(temperature_k=298.15))
breaking = run_breaking_scan(run, "breaking_run", spec=BreakingSpec(trajectory_ps=10.0))
elongation = run_elongation_scan(run, "elongation_run", spec=ElongationSpec())
print(yield_report.strength_mpa, breaking.strength_mpa, elongation.elongation_percent)
```

- **Yield** is an apparent offset proof stress: an elastic line fitted between
  `fit_min_strain` and `fit_max_strain` (at least five points, passing its
  quality checks), shifted by `offset_strain` (0.2% by default); the yield
  point is its first later crossing with the curve, interpolated within the
  sampled bracket. Without an unloading measurement this does not establish
  permanent deformation - see
  [Instron's offset yield definition](https://www.instron.com/en/resources/glossary/offset-yield-strength/).
- **Breaking strength** is the peak nominal stress, confirmed only when the
  curve ends with `confirmation_steps` consecutive holds below
  `failure_fraction` of the peak (three below half, by default). A curve still
  rising, one that recovers, or one that never drops that far stays
  unresolved.
- **Elongation at break** is `100 (L_break - L0) / L0` at the *first* hold of
  that confirmed terminal drop, with `L0` the equilibrated length - distinct
  from the strain at the peak. The preceding hold and the first low hold form
  `break_bracket`; nothing is interpolated, and the maximum imposed strain is
  never substituted for a break that did not happen.

A missing or incomplete replica keeps its curve and diagnostics but leaves the
headline `None`, and reanalysis always uses the criterion the scan recorded.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol yield -t 298 \
  --yield-max-strain 0.3 --yield-replicas 3 -o yield_run -v
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol breaking -t 298 \
  --breaking-max-strain 1.0 --failure-fraction 0.5 --confirmation-steps 3 \
  --breaking-trajectory-ps 10 -o breaking_run -v
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol elongation -t 298 \
  --elongation-max-strain 1.0 -o elongation_run -v
openmmpolymer --analyse breaking_run
```

Each protocol has its own `--<name>-strain-increment`, `-max-strain`,
`-relax-ps`, `-replicas`, `-samples-per-step`, `-stage-ps` and
`-trajectory-ps`; strains are fractions, and increments compound.

### Stress relaxation

A modulus says how hard the cell pushes back; a relaxation modulus says how
long it keeps pushing. `run_relaxation_scan` equilibrates, applies one affine
step strain, locks the box and watches the stress decay.

```python
from openmmpolymer import (
    RelaxationSpec,
    analyse_relaxation,
    run_relaxation_scan,
    write_relaxation_report,
)

report = run_relaxation_scan(
    run,
    "run",
    spec=RelaxationSpec(temperature_k=298.15, step_strain=0.03, relax_ps=50_000.0),
)
print(report.kww.beta, report.kww.mean_tau_ps, report.prony.equilibrium_mpa)
write_relaxation_report(analyse_relaxation("run"))
```

What comes back is `G(t)`. For an isotropic solid the differential stress
`σ_zz − (σ_xx + σ_yy)/2` is exactly `2G(ε_axial − ε_lateral)`, so a tensile
step measures the shear modulus with no assumption about Poisson's ratio;
`E(t) = 2(1 + ν)G(t)` is derived, and `mode="shear"` reads `G(t)` off the
off-diagonal directly. Two fits read it, both in numpy: a stretched exponential
(KWW) and a Prony series with non-negative weights. The decay is read against
the scatter of a baseline measured before the strain (`noise_floor_mpa`), a
run that stops before the decay does reports `plateau_reached` False, and
replicas are what make the fast end mean anything - turn `n_replicas` up
before `sample_every_ps` down. A relaxation modulus is a material property only
inside the linear region, and `linearity_strains` repeats the measurement at
other strains to check.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol relax -t 298 \
  --step-strain 0.03 --linearity-strains 0.01,0.06 -o run -v
```

### Structure

```python
from openmmpolymer import analyse_structure, write_structure_report

report = analyse_structure("run", backbone=chain.backbone)
print(report.distribution.first_peak_nm, report.conformation.mean.characteristic_ratio)
write_structure_report(report)
```

Every stage leaves its closing structure, and a stage asked for a trajectory
leaves frames. `analyse_structure` reads the last stage with a trajectory, or
failing that the last closing snapshot, and measures what that stage supports:
the intermolecular `g(r)` and the structure factor always; `<R²>`, `Rg`, the
characteristic ratio and the persistence length when the backbone is known;
centre-of-mass displacement and end-to-end relaxation only from a trajectory.
What cannot be measured becomes a note. `g(r)` and `S(q)` are capped at 50 and
8 frames, because past that they stop changing and `S(q)` is expensive.

```bash
openmmpolymer --analyse run --backbone 0,1,4,5
openmmpolymer --analyse run --structure-stage 05_npt --stride 4
```

## Rate dependence

Fast deformation leaves less time for stress to relax, and fast cooling traps
a glass higher. `run_property_rate_scan` measures one property at three or more
rates from one common prepared state - keeping the ladder, criterion and
preparation fixed and changing only the hold per step - and
`analyse_property_rates` fits how it moves and evaluates that at a target rate.

| `property_name` | Measured quantity | Value unit | Rate unit | Scan spec / CLI protocol |
| --- | --- | --- | --- | --- |
| `youngs_modulus` | Young's modulus from extension | MPa | strain/ns | `ModulusSpec` / `modulus` |
| `poisson_ratio` | Transverse/axial strain ratio | dimensionless | strain/ns | `ModulusSpec` / `modulus` |
| `shear_modulus` | Shear modulus | MPa | strain/ns | `ModulusSpec` / `modulus` |
| `bulk_modulus` | Bulk modulus | MPa | bar/ns | `ModulusSpec` / `modulus` |
| `load_modulus` | Young's modulus under applied stress | MPa | bar/ns | `ModulusSpec` / `modulus` |
| `yield_strength` | Apparent offset yield strength | MPa | strain/ns | `YieldSpec` / `yield` |
| `yield_strain` | Strain at the offset yield event | strain | strain/ns | `YieldSpec` / `yield` |
| `breaking_strength` | Apparent ultimate tensile strength | MPa | strain/ns | `BreakingSpec` / `breaking` |
| `elongation_at_break` | Apparent elongation at break | % | strain/ns | `ElongationSpec` / `elongation` |
| `glass_transition` | Glass transition on cooling | K | K/ns | `TgSpec` / `tg` |
| `melting_temperature` | Apparent melting on heating | K | K/ns | `TmSpec` / `tm` |

```python
from openmmpolymer import (
    ModulusSpec,
    analyse_property_rates,
    run_property_rate_scan,
    validate_property_rate_scan,
    write_rate_report,
)

spec = ModulusSpec(temperature_k=298.15, n_replicas=3)
plan = validate_property_rate_scan(
    spec, (100.0, 500.0, 2000.0), property_name="bulk_modulus", target_rate=10.0
)
print(plan.total_ns)  # the common preparation and every rate and replica
report = run_property_rate_scan(
    run,
    "bulk_rates",
    property_name="bulk_modulus",
    hold_times_ps=(100.0, 500.0, 2000.0),
    target_rate=10.0,
    spec=spec,
)
write_rate_report(report)

# Finished runs, or a scan root that expands to its rate runs, can be reanalysed.
report = analyse_property_rates(
    ["bulk_rates"], property_name="bulk_modulus", target_rate=10.0
)
```

Both `log_linear` (`value = v_ref + b log10(rate / rate_ref)`) and
`power_law` (`value = v_ref (rate / rate_ref)^b`) fits are attempted. Each reports its value at the target
rate with a standard error that propagates the measurement errors and the fit
scatter, and the sensitivity per decade of rate at the reference rate - the
number that was measured rather than extrapolated. Neither is a zero-rate or
equilibrium value, and logarithmic rate dependence can fail at the fastest
rates ([Nazarychev et al., Soft Matter
(2016)](https://pubs.rsc.org/en/content/articlehtml/2016/sm/c6sm00230g)).
Unresolved inputs, unknown uncertainty, poor fits, implausible bounds or
trends, and extrapolation beyond `max_extrapolation_decades` (2.0) remain
unresolved, so a laboratory target many decades away is always unresolved.
Choose a target near the slowest rate you measured.

A yield or break that never happened is kept as missing rather than dropped,
so a fit cannot quietly use only the rates that succeeded. The mechanical
scans take their replica count from the spec; the thermal ones from an
`n_replicas` argument. A Tg rate scan walks the full coarse ladder at every
rate - the adaptive fine window has no place in a comparison across rates.

For thermal scans the rate is `1000 * step_k / hold_ps` in K/ns; cooling-rate
effects need material-specific validation before comparison with experiment
([an epoxy network's specific volume against cooling
rate](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.7b01303)), and heating
may superheat a crystal, so a heating-rate correction cannot establish
equilibrium melting ([a molecular-dynamics study of superheating against
heating rate](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.68.134206)).
Pressure ramps count total ladder distance over total hold time. Rates are in
the units in the table: multiply a strain rate in s^-1 by `1e-9`.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol modulus -t 298 \
  --rate-property bulk_modulus --rate-hold-times 100,500,2000 \
  --target-property-rate 10 -o bulk_rates
openmmpolymer --analyse elongation_fast elongation_medium elongation_slow \
  --rate-property elongation_at_break --target-property-rate 0.0002
```

## Observation-window convergence

A quantity read off a trajectory or a relaxation curve also needs a window
check: `analyse_convergence` compares estimates from increasing fractions of a
saved run.

| Measurement | Checked quantities | Evidence required |
| --- | --- | --- |
| State-data series | Density, temperature, potential energy | Stable prefix means and tail blocks; enough autocorrelation-adjusted samples |
| Chain dimensions | Mean `Rg`, mean `<R²>` | The same |
| Structural refits | Persistence length, characteristic ratio | Stable refits; the backbone correlation must decay |
| Chain dynamics | Diffusion coefficient, end-to-end relaxation time | Stable refits; diffusive MSD or decorrelation actually observed |
| Pair and reciprocal structure | `g(r)` and `S(q)` peak position and height | Fixed grids and sampling; agreement of disjoint tail blocks |
| Stress relaxation | Equilibrium modulus, KWW mean time, viscosities | Stable decay refits; the relevant tail observed |

```python
from openmmpolymer import analyse_convergence, write_convergence_report

convergence = analyse_convergence("run", stage="05_npt", backbone=chain.backbone)
write_convergence_report(convergence, output_dir="run/convergence_analysis")
```

```bash
openmmpolymer --analyse run --convergence --convergence-stage 05_npt \
  --window-fractions .25,.5,.75,1 -o convergence_analysis
```

`time_window_convergence` and `relaxation_window_convergence` check a raw
series or an existing `RelaxationCurve`, and `structural_window_convergence`
takes an `open_run` ensemble with explicit frame caps when the defaults leave
`S(q)` unresolved. Window differences are stability diagnostics, not standard
errors; a snapshot cannot establish convergence, and none of these checks
proves equilibrium.

## Things worth knowing

**Strain is applied per atom, not per molecule.** A barostat moves whole
molecules for a volume change, but a polymer's stiffness comes from chains
being stretched, so a deformation scales every atom and `applyConstraints`
repairs the constrained bonds before anything reads an energy.

**The barostat scales molecules rigidly.** Per-atom scaling with constraints
on violates them: on a 30-chain polyethylene cell at 600 K and 1000 bar, rigid
scaling settles at 0.71 g/cm³ and per-atom scaling falls to 0.14 and keeps
going, so it is refused while constraints are on.

**The instantaneous pressure fluctuates enormously**, and that is the dominant
source of error in every mechanical measurement here. Stress is sampled
densely through each hold, averaged over its second half and repeated across
replicas; where the spread does not support the number, `resolved` is False.

**Cell size is checked against the compressed density.** OpenMM refuses a
cutoff over half the box, and refuses it again mid-run once the barostat has
shrunk the cell; `check_target_density` catches that before anything long
starts and says how much more material is needed.

**OpenFF Sage warns about preset charges** on every run, because it carries
virtual-site parameters. It matters only if the polymer gets virtual sites -
`PolymerForceField.virtual_site_residues` says whether it did.

## How it fits together

| Layer | Modules |
|---|---|
| Building a cell | `chain` (monomer SMILES to grown chains), `charges`, `forcefield` (forcefill to an ffxml, cached), `packing` (packmol and the checks that catch a bad cell), `mdsystem` (topology, System, barostats, platforms), `melt` (all of it in one call) |
| Running it | `simulate` (the stages), `protocols` (named stage sequences, the manifest and resume), `reporters`, `stress` |
| Reading it back | `trajectory`, `timeseries`, `conformation`, `correlations`, `elasticity`, `strength`, `relaxation`, `melt_check`, `plots` |
| Workflows | `tg`, `tm`, `mechanical`, `tensile`, `viscoelastic`, `structure` |
| Rate and window checks | `rate_dependence`, `property_rates`, `elastic_rates`, `tensile_rates`, `thermal_rates`, `rate_reports`, `convergence`, `structural_convergence`, `convergence_report` |

## Reference benchmark

[The polyethylene benchmark](benchmarks/README.md) builds independent C44H90
melts, runs their preparation and NPT measurement, and compares density with
the published specific volumes of Lee, Frank and Yoon,
[Polymers 12 (2020), Table 2](https://doi.org/10.3390/polym12051059).

## Development

The environment file installs the test and lint tools. Arm the hooks once;
they are the same checks CI's lint job runs:

```bash
pre-commit install
```

```bash
pre-commit run --all-files
mypy
pytest
```

The heavy tests are marked: `forcefield` needs forcefill and the OpenFF stack,
`packmol` the binary on `PATH`, `slow` more than a second of dynamics.

```bash
pytest -m "not forcefield and not packmol and not slow"
```

## License

Released under the [MIT License](LICENSE).
