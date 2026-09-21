# openmmpolymer

Build, pack, parameterise and equilibrate all-atom polymer melts with
[OpenMM](https://openmm.org), [packmol](https://m3g.github.io/packmol) and
[forcefill](https://github.com/LouieSlocombe/forcefill).

Give it a monomer SMILES and it will grow chains with the right dimensions,
charge them, turn them into an OpenMM force field, pack them into a periodic
cell, and take that cell through minimisation, push-off, high-temperature
equilibration, compression, annealing and a quench — with a manifest that lets
the run be picked up again after the queue kills it.

## Requirements

conda-forge, and not by preference. AmberTools is not a Python package at all
(and is what supplies the `packmol` executable), and `openmmforcefields >=
0.16` — forcefill's floor — has never been published to PyPI. `pip install
openmmpolymer` will not give you a working install.

```bash
conda env create -f build_tools/environment.yml
conda activate openmmpolymer
python -m pip install -e . --no-deps
```

Python 3.12 or newer.

The analysis layer reads trajectories with MDAnalysis and draws them with
matplotlib, both of which the environment file installs. The plotting helpers
build a `matplotlib.figure.Figure` directly and never touch `pyplot`, so they
need no display and no backend.

## A polyethylene melt

```python
from openmmpolymer import (
    ChainSpec,
    PackedComponent,
    assemble_box,
    assign_charges,
    box_edge_nm,
    build_chain,
    build_polymer_forcefield,
    check_packing,
    pack_box,
    prepare_box,
    prepare_run,
    run_protocol,
    standard_melt_equilibration,
)

n_chains = 40

chain = build_chain(
    ChainSpec(
        monomer_smiles="[*]CC[*]", degree_of_polymerization=30, residue_name="PE"
    ),
    n_conformers=n_chains,
)
assign_charges(chain.sdf_paths[0], "nagl")
forcefield = build_polymer_forcefield(
    chain.sdf_paths[0], residue_name="PE", cache_dir="ff-cache"
)

components = [PackedComponent(path, 1) for path in chain.pdb_paths]
packed = pack_box(components, box_edge_nm([n_chains], [chain.molar_mass_g_mol], 0.3))

box = assemble_box(components, packed.packed_pdb, packed.box_nm)
check_packing(box.topology, box.positions_nm)

run = prepare_run(prepare_box(box, forcefield), forcefield)
summary = run_protocol(
    standard_melt_equilibration(target_temperature_k=450.0),
    run,
    "run",
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
print(summary.manifest_path)
```

Or from the command line:

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE -t 450 -o run -v
```

Run it again and it picks up from the last stage that finished.

## Finding a glass transition

Resolving a transition to a few kelvin needs small temperature steps and long
holds, and paying for that resolution from the melt temperature all the way
down is most of the cost for none of the answer. So `run_tg_scan` goes down
the ladder twice: a coarse scan to find roughly where the break is, then a
fine one across a window centred on it.

```python
from openmmpolymer import TgSpec, analyse_run, run_tg_scan, write_report

result = run_tg_scan(
    run,
    "run",
    spec=TgSpec(melt_temperature_k=650.0, t_floor_k=150.0, npt_trajectory_ps=10.0),
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
print(result.temperature_k, result.fine_schedule.cooling_rate_k_per_ns)

report = analyse_run("run")
write_report(report)
```

The second pass starts from a state the first one saved as it went past, not
from the equilibrated melt and not from the bottom of the coarse ladder. A
glass remembers how it was cooled, so a fine window entered by reheating a
solid is measuring a different thermal history from the one that located it.

The scan refuses to guess. If the coarse fit reports it found a corner in
noise rather than a transition, it stops and says so, because a fine pass is
tens of nanoseconds and a window derived from that fit produces a curve with
nothing in it. Pass `tg_approx_k` to name the window yourself.

`cooling_rate_series` runs the fine window several times at different rates,
all from that one configuration, and `cooling_rate_extrapolation` fits how the
transition moves. Both passes are chunked into stages of a few nanoseconds, so
an interrupted run resumes at the stage it stopped in rather than at the top of
the ramp.

From the command line:

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol tg --check-melt -o run -v
```

## Reading a finished run

```bash
openmmpolymer --analyse run
```

No monomer and no chemistry: a directory in, a number out. It finds the quench
stages by what they recorded rather than by what they were called, tells the
coarse pass from the fine one by its temperature step, checks whether the melt
had settled before cooling started, and writes `run/analysis/tg.json` beside
the figures. Give it several run directories to pool a cooling-rate series
across them. The manifest is not touched.

It works out what to report from what the directory recorded: a run that
quenched gets a glass transition, a run that was deformed gets its elastic
constants, a run that did both gets both, and any run whose stages left
coordinates gets its structure read back as well.

## Measuring a modulus

OpenMM has no continuous deformation, so a strain rate here is a staircase:
scale the cell by one increment, let it relax under a barostat holding the
other two axes at pressure and leaving the driven one alone, read the stress,
repeat. `run_modulus_scan` walks that ladder and three more passes beside it.

```python
from openmmpolymer import (
    ModulusSpec,
    analyse_mechanics,
    run_modulus_scan,
    write_mechanical_report,
)

result = run_modulus_scan(
    run,
    "run",
    spec=ModulusSpec(temperature_k=298.15, max_strain=0.05),
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
print(result.youngs.modulus_mpa, result.replica_spread_mpa)

write_mechanical_report(analyse_mechanics("run"))
```

Four constants come back, and the fourth is what the other three are for. `E`
and `nu` come from the extension; `K` from a gentle pressure ladder and `G`
from a shear ladder, each measured rather than derived. For an isotropic solid
those four are two, so the gap between the measured `K` and `G` and the ones
`E` and `nu` imply checks all of them at once — and unlike each of them, it is
not a straight line fitted through a window someone chose the ends of.

Every pass branches from the *same* equilibrated cell. A cell that has just
been stretched to five per cent is not the cell the next measurement wants, so
they are not run one after another. The replicas branch from it too, with
fresh velocities: inheriting the equilibrated state's velocities as well as
its positions gives the same trajectory every time, and a spread computed over
those would be zero dressed up as an error bar.

There is a second, independent estimate of `E` in there. `run_load` imposes a
known stress with an anisotropic barostat and measures the box, so no virial
is involved anywhere in it. The two methods share no machinery, which is what
makes their agreement worth something; `method_gap` reports it.

From the command line:

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol modulus -t 298 -o run -v
```

and `--skip bulk shear` if `E` and `nu` are all you want.

## Watching a stress relax

A modulus says how hard the cell pushes back. A *relaxation* modulus says how
long it keeps pushing, which for a polymer is the more interesting half.
`run_relaxation_scan` equilibrates a cell, applies one affine step strain,
locks the box, and watches the stress decay.

```python
from openmmpolymer import (
    RelaxationSpec,
    analyse_relaxation,
    run_relaxation_scan,
    write_relaxation_report,
)

result = run_relaxation_scan(
    run,
    "run",
    spec=RelaxationSpec(temperature_k=298.15, step_strain=0.03, relax_ps=50_000.0),
    chain_backbone=chain.backbone,
    atoms_per_chain=chain.n_atoms,
)
print(result.kww.beta, result.kww.mean_tau_ps, result.prony.equilibrium_mpa)

write_relaxation_report(analyse_relaxation("run"))
```

What comes back is `G(t)`, and that is the measurement rather than a
conversion. For an isotropic solid the differential stress
`σ_zz − (σ_xx + σ_yy)/2` is exactly `2G(ε_axial − ε_lateral)`: the Lamé
constant cancels, so a tensile step measures the shear modulus with no
assumption about Poisson's ratio, the bulk modulus, or whether the deformation
preserved the volume. `E(t) = 2(1 + ν)G(t)` is the derived number, and
`youngs_modulus_mpa` takes the material's own ν rather than the one the box was
scaled by. Writing `E(t) = σ_diff/ε₀` instead — the form the textbook gives —
quietly assumes both are one half, and is a per cent out at a four per cent
strain even when they are. A shear step (`mode="shear"`) skips the question
entirely and reads `G(t)` off the off-diagonal.

The box is locked for the whole hold and a barostat is attached anyway, at
`frequency=0`. That is not a contradiction: the applied strain *is* the
measurement so nothing may relax it away, but OpenMM reports a pressure only
through a barostat and refuses it for a force that is not in the Context. One
that never moves the box exists purely to be asked — the same trick `run_shear`
uses. The stage checks the cell is where it left it and raises if not.

Two fits, both numpy, because scipy is not a dependency. The stretched
exponential separates: for a fixed exponent `ln G` is linear in `t^β`, so a
three-parameter fit collapses to a bracketed search with an exact solve inside,
the way the VFT fit already does. The Prony series separates differently — fix
the time constants on a log grid and the weights are linear, subject to being
non-negative, which is what the Lawson–Hanson solve is for. Plain least squares
returns a negative weight in about four runs out of five, and a spectrum with
negative weight is not a spectrum.

From the command line:

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol relax -t 298 --step-strain 0.03 -o run -v
```

and `--linearity-strains 0.01,0.06` to check the strain was small enough to
mean anything.

## Looking at the structure

```python
from openmmpolymer import analyse_structure, write_structure_report

report = analyse_structure("run", backbone=chain.backbone)
print(report.distribution.first_peak_nm, report.conformation.mean.characteristic_ratio)
write_structure_report(report)
```

Every finished stage leaves its closing structure, and a stage asked for a
trajectory leaves frames, so any run has something to say about how its chains
are arranged. `analyse_structure` reads the last stage that wrote a trajectory,
or failing that the last stage's closing snapshot, and measures what that stage
can support: the intermolecular `g(r)` and the structure factor always; `⟨R²⟩`,
`Rg`, the characteristic ratio and the persistence length when the backbone is
known; the centre-of-mass displacement and the end-to-end relaxation only from
a trajectory. Whatever cannot be measured becomes a note rather than an error,
and `structure.json` records which stage was read and why, where the backbone
came from, and every curve. `g(r)` and `S(q)` are capped at 50 and 8 frames
however long the trajectory is, because past that they stop changing and the
structure factor is the expensive one.

```bash
openmmpolymer --analyse run --backbone 0,1,4,5
openmmpolymer --analyse run --structure-stage 05_npt --stride 4
openmmpolymer --analyse run --no-structure
```

## How it fits together

| Layer | Module | What it does |
|---|---|---|
| Chain | `chain` | Monomer SMILES → one 3D molecule per chain copy, as SDF and PDB |
| Charges | `charges` | Partial charges, written into the SDF |
| Force field | `forcefield` | forcefill → an ffxml, cached |
| Packing | `packing` | Box arithmetic, packmol, and the checks that catch a bad cell |
| System | `mdsystem` | Replicated topology, packmol's coordinates, `createSystem` |
| Stages | `simulate` | minimise, push-off, NVT, compress, NPT, anneal, quench, production, deform, load, shear, relax |
| Protocol | `protocols` | Named stage sequences, the manifest, and resume |
| Trajectory | `trajectory` | A finished run directory → frames, per chain, in nanometres |
| Time series | `timeseries` | State-data CSVs, equilibration detection, the quench curve, the rate extrapolation |
| Conformation | `conformation` | `⟨R²⟩`, `Rg`, persistence length, end-to-end relaxation, COM displacement |
| Correlations | `correlations` | Intermolecular `g(r)` and the static structure factor |
| Plots | `plots` | A figure per result, returned rather than written |
| Stress | `stress` | The pressure tensor of a running cell, and the strain applied to it |
| Elasticity | `elasticity` | Stress-strain curves, and the four elastic constants read off them |
| Relaxation | `relaxation` | `G(t)` from a step strain, and the Prony and KWW fits read off it |
| Workflow | `tg` | The two-pass glass-transition scan, and reading a finished run back |
| Workflow | `mechanical` | The extension, the load, bulk and shear passes, and the report |
| Workflow | `viscoelastic` | The step-strain scan, its replicas, and the report |
| Workflow | `structure` | Reading a finished run's `g(r)`, `S(q)`, chain dimensions, persistence length, displacement and end-to-end relaxation back, and the report |

## The constraint everything follows from

forcefill will not parameterise a residue bonded to its neighbours, and it is
right not to: a stand-alone GAFF treatment of a chain-linked residue is not
valid. So a chain here is **one molecule and one residue**, and the whole
design follows: chains are built whole, charged whole, and parameterised whole
through `build_ligand_xml`.

## Chain length and charges

AM1-BCC runs a semi-empirical QM calculation, which stops being tractable at a
few hundred atoms — a twenty-unit polyethylene chain is already there. So
charges are assigned here rather than by the backend, with a graph neural
network that scales with the number of atoms rather than their cube.
`openmmforcefields` checks whether the molecule already carries charges and
passes them through instead of running AM1-BCC.

| Method | Practical chain size | Notes |
|---|---|---|
| `nagl` | thousands of atoms | The default. AM1-BCC quality. |
| `am1bcc` | a few hundred atoms | The reference; `sqm` is the wall. |
| `gasteiger` | unbounded | Qualitative only: no hydrogen bonding, no dipole calibration. Fine for a polyolefin, several per cent out for an ester or an amide. |

## Things worth knowing

**One conformer per chain.** packmol places each structure as a rigid body, so
packing one conformer forty times gives a cell of forty identical coils. Build
`n_conformers=n_chains`; they are the same molecule, so they share one force
field.

**Chains are grown, not embedded.** RDKit's ETKDG collapses a long chain into a
globule — a twenty-unit polyethylene comes out with a quarter of the dimensions
it should have. Anything of four units or more is instead grown unit by unit,
sampling backbone torsions drawn from the polymer's characteristic ratio, with
a self-avoidance check. `build_chain` measures what it produced and says so
when it drifts.

**Equilibration is reported, not claimed.** packmol places chains that do not
interpenetrate, and the Rouse time of a melt is tens of nanoseconds — longer
than the high-temperature stages here. The manifest records `⟨R²⟩`, the radius
of gyration and the measured characteristic ratio against the expected one, so
you can see whether the chains have relaxed at their own scale. It does not
assert that they have.

What can settle the question is a trajectory. `end_to_end_relaxation` measures
how fast the end-to-end vector decorrelates and reports `decorrelated` — False,
for every protocol shipped here, because none of them runs for a Rouse time.
`equilibration` does the same for any single series, and says how many
genuinely independent samples sit behind a mean rather than how many rows do.
`centre_of_mass_msd` refuses to divide by six until the log-log slope says the
chains are diffusing, because a melt short of its entanglement time is
sub-diffusive and a coefficient fitted to that is not a diffusion coefficient.

`melt_equilibration` puts those together into one verdict: the cell volume has
to have stopped drifting faster than its own noise, *and* the chains' centres
of mass have to have travelled further than the chains are big, diffusively.
It needs a trajectory from the equilibration stage, which is not written unless
asked for — `npt_trajectory` on the protocol, `npt_trajectory_ps` on a `TgSpec`,
`--check-melt` on the command line. Without one the verdict is False and
`unchecked` says which half was missing, because a verdict with half its
evidence absent is not a pass.

**The manifest does not know the backbone.** The backbone atom indices come
from the attachment points the caps consumed when the chain was built, and
nothing downstream can recover them from the structure alone. The workflow
drivers record them as `chain_backbone` in their own `*_workflow.json`; a plain
protocol run records them nowhere. So `analyse_structure` looks for a backbone
in that order — given, recorded by a workflow, recorded in the manifest — and
failing all three infers one from the bond graph as the longest shortest path
through one chain's heavy atoms. For a linear polymer that is the backbone,
unless a side group on the last unit reaches further than the chain end does,
in which case the inferred path ends on it and the end-to-end vector carries
one extra bond. The report says which source it used in `backbone_source`,
and `--backbone` overrides all of them. Opening a trajectory also leaves
MDAnalysis' offset files beside it, so reading a run touches its directory.

**A quench is not a Tg measurement.** `melt_quench` records a density at every
temperature on the way down, which is the specific-volume curve a glass
transition is read off. Every all-atom cooling rate is many orders of magnitude
faster than any experiment, so the transition sits well above the measured one.
The shape is informative; the number is not directly comparable. `quench_curve`
reads that curve back and `glass_transition` fits the two straight lines it is
read off, reporting the cooling rate alongside the temperature so the caveat
travels with the number, and `resolved` False when the fit found a corner in
noise — which is what fitting two lines to a straight one always finds.

`melt_expansivity_per_k` and `glass_expansivity_per_k` report the same two
branches as thermal expansion coefficients, `(1/v)(∂v/∂T)` in 1/K, which is
what a dilatometry paper quotes. They are not a second check on the fit: both
divide the same crossing volume, so the melt expanding faster than the glass is
the slope condition `resolved` already requires.

**And a rate extrapolation is an extrapolation.** `cooling_rate_extrapolation`
fits how the transition moves with cooling rate and evaluates that fit wherever
you ask, including at the 10 K/min a calorimeter scans at. The gap is about ten
decades. `extrapolation_decades` reports it and `resolved` is False past two,
so an extrapolation to an experimental rate is *always* unresolved — that is
the design working rather than failing. The robust number is the other one:
how far the transition moves per decade of rate, which was measured rather
than extrapolated. Two relations are offered because over that gap they
disagree by more than a hundred kelvin: a straight line in log rate runs away,
and the Vogel–Fulcher–Tammann form, which has a finite limit, does not. WLF is
not a third option — it is VFT reparameterised, and `wlf_constants` converts.

**A strain rate is not quasi-static.** An increment of 0.002 relaxed for 50 ps
is about 4x10^7 per second, against 10^-3 in a tensile test. That is the same
ten-decade gap a cooling rate has, and the same caveat applies: the shape of
the stress-strain curve is informative, the number is not directly comparable.
`strain_rate_per_ns` is carried on every fit, drawn on every figure and
printed on every line, so it cannot be quoted without it.

**A modulus above Tg is a rubber modulus.** Which side of the transition
298 K falls on is a property of the polymer, and nothing in `mechanical`
knows it. Run `run_tg_scan` first. Below Tg the response is local and
enthalpic and this protocol measures it; above it the stiffness is entropic,
comes from chains being pulled out of shape, and is both far smaller and far
slower to relax than a 50 ps window allows. A `resolved` of False on a melt is
the honest answer rather than a failure.

**The instantaneous pressure fluctuates enormously.** That is OpenMM's own
warning about `computeCurrentPressure`, and it is the dominant source of
error here, not a detail to work around. So the stress is sampled densely
through each relaxation window - the pressure decorrelates in well under a
picosecond, and readings spaced picoseconds apart throw away almost all of
the statistics the window already paid for - averaged over the second half,
and repeated across replicas. Where the spread does not support the number,
`resolved` is False.

**Strain is applied per atom, not per molecule.** The barostat translates
whole molecules for a volume move, and this package insists on that; a
deformation is the other case. Moving molecules rigidly leaves every chain
conformation exactly as it was, and a polymer's stiffness comes from chains
being stretched, so a rigid deformation would generate almost none of it. The
reason the barostat cannot scale per atom - that it would evaluate a
Metropolis energy for a configuration violating its own constraints - does
not apply to a strain applied once and repaired immediately: `applyConstraints`
puts the constrained bonds back before anything reads an energy. Measured at
an increment of 0.002, the longest constrained bond stretches by 0.2 pm and
the repair moves no atom further than 2x10^-4 nm.

**A relaxation modulus is read against a floor.** The stage measures the cell's
deviatoric stress for a while *before* straining it, and the scatter of that
window is the level below which a decaying cell and a fully relaxed one are the
same measurement. `noise_floor_mpa` carries it, the figure shades it, and both
fits stop where the signal does — the window is truncated at the end rather than
filtered point by point, because dropping the bins where `G` came out negative
would drop the downward half of the noise and keep the upward half, biasing the
tail up and `β` down. The mean stress is *not* checked: freezing the box at an
NPT snapshot leaves the pressure wherever that fluctuation was, which is
expected and harmless. A deviatoric stress is not, and a large one is reported.

**Replicas are what make the fast end of the decay mean anything.** The stress is
pooled into logarithmic time bins, so the earliest bins hold one reading each —
a single run resolves about the decade around the largest stress and loses the
rest in noise. Independent runs add bin for bin, which is the whole reason the
bin edges come from the settings rather than from the data: a relaxation split
across stages for resume and eight replicas of it merge by the *same* addition.
Turn `n_replicas` up before `sample_every_ps` down.

**A run that stops before the decay does has not measured a plateau.** `E_inf`
comes out of the Prony fit whatever happens, and a spectrum that has piled its
weight onto the slowest time constant is telling you the decay was still going
when the data stopped. `plateau_reached` says so and `resolved` goes False. The
grid deliberately stops at a third of the run, because an exponential as slow as
the run is 0.99 collinear with a constant and the split between the two becomes
arbitrary — reaching the whole window recovered 567 against a planted 400, a
third of it recovered 404.

**And a step strain is only a material property if it was small enough.** A
relaxation modulus is one inside the linear viscoelastic region and a property of
the deformation outside it, and no amount of looking at one strain says which you
have. `linearity_strains` repeats the whole measurement at another; inside the
region the curves coincide. It is off by default because it doubles the scan.

**Cell size is checked against the compressed density, not the packed one.**
OpenMM refuses a cutoff over half the box — and refuses it again mid-run once
the barostat has shrunk the cell. `check_target_density` catches that before
anything long starts, and says how much more material is needed.

**The barostat scales molecules rigidly.** OpenMM's documentation suggests
per-atom scaling for molecules big enough to span the cell, which a polymer
chain can be, but scaled positions violate any constraint they cross. Measured
on a 30-chain polyethylene cell at 600 K and 1000 bar: rigid scaling settles at
0.71 g/cm³, per-atom scaling with `constraints="hbonds"` falls to 0.14 and
keeps going. Per-atom scaling is therefore refused while constraints are on.

**OpenFF Sage warns about preset charges.** It carries virtual-site parameters,
so handing it charges makes it say so on every run. It matters only if the
polymer gets virtual sites — `PolymerForceField.virtual_site_residues` is where
to look.

## Development

```bash
ruff check .
ruff format --check .
mypy
pytest
```

The heavy legs are marked and can be left out:

```bash
pytest -m "not forcefield and not packmol and not slow"
```

`forcefield` needs forcefill and the OpenFF stack, `packmol` needs the binary
on `PATH`, `cuda` needs a working CUDA platform.

## License

Released under the [MIT License](LICENSE).
