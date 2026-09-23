# openmmpolymer

Build, pack, parameterise and equilibrate all-atom polymer melts with
[OpenMM](https://openmm.org), [packmol](https://m3g.github.io/packmol) and
[forcefill](https://github.com/LouieSlocombe/forcefill).

Give it a monomer SMILES and it will grow chains with the right dimensions,
charge them, turn them into an OpenMM force field, pack them into a periodic
cell, and take that cell through minimisation, push-off, high-temperature
equilibration, compression, annealing and a quench — with a manifest that lets
the run be picked up again after the queue kills it.

## Installation

Python 3.12 or newer and OpenMM 8.3.1 or newer, with the dependencies from
conda-forge:

```bash
conda env create -f build_tools/environment.yml
conda activate openmmpolymer
python -m pip install -e . --no-deps
```

conda-forge is the only route that works, not a preference. AmberTools is not
a Python package and is what supplies the `packmol` executable, and
`openmmforcefields >= 0.16` has never been published to PyPI. `pip install
openmmpolymer` on its own will not give you a working install.

The plotting helpers build a `matplotlib.figure.Figure` directly and never
touch `pyplot`, so they need no display and no backend.

OpenMM 8.3.0 has a kinetic-pressure calculation bug and is not supported.
CI exercises the pressure and system adapters against 8.3.1 as well as the
current environment. Flexible-cell stress uses `computeStressTensor` when
available, with a consistent-strain finite-difference implementation on older
supported releases. Older isotropic and anisotropic barostats support only
rigid molecular scaling; requesting unavailable atomic scaling is an error.

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

Resume checks the serialized System, topology, initial coordinates, seed,
settings and stage inputs before reusing results. Missing or changed upstream
states invalidate their dependent stages. The CLI also records the effective
request and dependency versions in `build_request.json` before preparing the
cell, so changed inputs cannot overwrite an existing run's build assets.
Use a new output directory for a different simulation. Legacy manifests remain
readable, but cannot be resumed without provenance; the Python API can
explicitly rerun them with `resume=False`.

End caps and chain dimensions can be specified for other polymers. For example,
an acid-terminated PLA chain needs an explicit hydroxyl cap on its carbonyl end:

```bash
openmmpolymer '[*]OC(C)C(=O)[*]' -n 20 -c 40 -r PLA \
  --tail-cap '[*]O' --charge-method nagl --dry-run -o pla-build
```

`--head-cap` and `--tail-cap` each accept a fragment with one `[*]`. Set
`--characteristic-ratio` to a material-appropriate C-infinity for chain growth
and every subsequent dimension check. Structure analysis reuses the recorded
value unless an explicit override is supplied. The build default, 7.0, is for
polyethylene; the PLA example above demonstrates end-cap construction and does
not calibrate PLA chain statistics.

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

## Estimating a melting temperature

Melting needs a crystalline or semicrystalline starting cell. The ordinary
monomer-to-melt workflow produces an amorphous cell, so it cannot supply that
starting point. Prepare and check the crystal separately, with a force field
appropriate for its solid phase, then use `run_tm_scan` to equilibrate it below
the expected melting range and heat it through a temperature ladder at fixed
pressure.

```python
from openmmpolymer import TmSpec, analyse_melting, run_tm_scan, write_melting_report

# crystal_run is a prepared RunContext for your crystalline periodic cell.
result = run_tm_scan(
    crystal_run,
    "melting",
    crystalline=True,
    spec=TmSpec(t_start_k=250.0, t_end_k=650.0, step_k=10.0, hold_ps=1000.0),
)
print(result.temperature_k, result.bracket_k, result.resolved)
write_melting_report(analyse_melting("melting"))
```

`crystalline=True` is your declaration about the starting structure, not an
automatic crystallinity check. The scan looks for a coincident upward jump in
specific volume and enthalpy. Smooth thermal expansion, a glass-transition
slope change, or conflicting signals leave the result unresolved. The reported
temperature is the midpoint of a sampled heating bracket; this is an apparent
melting temperature at that heating rate, not an equilibrium melting point.
Superheating, cell size, crystal morphology and the force field can shift it.
Inspect the loss of crystalline order, repeat with longer holds and smaller
steps, and use independent starting configurations to assess that uncertainty.
`trajectory_ps` can save coordinates for that inspection.

The CLI takes the prepared PDB and an OpenMM `XmlSerializer` System file in
exactly the same atom order. The PDB must include its periodic box (`CRYST1`),
and the System must be periodic and contain no thermostat or barostat. The scan
supplies its own temperature and pressure controls. This is a serialized System,
not a force-field XML template. Parameterisation and packing are skipped.
The PDB must include molecular bonds (`CONECT` records where needed) consistent
with the System: bonded connected components define the molecules, independently
of PDB chain identifiers. Molecules must have equal atom counts and occupy
contiguous atom blocks, as required by the structural reports. Mixed molecule
sizes and interleaved atom ordering are rejected.

```bash
openmmpolymer --protocol tm --crystal-pdb crystal.pdb --system-xml system.xml \
  --t-start 250 --t-end 650 --step-k 10 --hold-ps 1000 \
  --tm-trajectory-ps 10 --max-total-ns 50 -o melting
openmmpolymer --analyse melting
```

`--state-in crystal-state.xml` can supply positions, velocities and box vectors
from the same prepared crystal instead of starting at the PDB coordinates.
`--dry-run` checks the inputs and schedule without running dynamics. The
default pressure control is anisotropic so the crystal's three box lengths can
relax independently; `--tm-barostat isotropic` holds their ratios fixed. Heating
is split into resumable stages (`--tm-stage-ps`), and rerunning the same command
resumes completed work. Reports and figures go to `melting/analysis/tm.json`
and its neighbouring image files.

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
quenched gets a glass transition, a heating scan gets a melting report, a run
that was deformed gets its elastic constants, yield-strength, breaking-strength
or elongation-at-break report, and any run whose stages left coordinates gets
its structure read back as well.

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

The bulk fit reports `standard_error_mpa`, `relative_standard_error`,
`residual_log_volume` and `half_disagreement`. A resolved fit needs distinct
pressure levels, a positive modulus, relative fit uncertainty below 25%,
approximately linear response and acceptable compression/decompression
hysteresis. This uncertainty describes the fitted ladder; replica and
preparation variability remain separate checks.

Shear samples now record `stress_estimator_version`. Older shear and shear
relaxation results used box-entry derivatives that are not the physical stress
in a tilted cell. Their analysis is refused, including mixtures of old and new
chunks; rerun those measurements with the corrected estimator. Reanalysing
their saved stress values cannot repair them.

### Reducing strain-rate bias in Young's modulus

Fast deformation leaves less time for stress to relax and can overestimate the
modulus at a slower rate. Measure at least three distinct rates, keeping the
temperature, strain increment, elastic fitting window and sample preparation
fixed. `run_modulus_rate_scan` varies the relaxation time between increments;
longer holds give a slower nominal rate, approximately
`1000 * strain_increment / relax_ps` in strain/ns. The reported rate uses the
final compounded strain divided by the total deformation time. Keep the strain
increment small enough to remain in the initial linear response and use
replicas to quantify stress noise. The scan equilibrates once, then branches
each rate and replica from that common configuration. It runs extension passes
only; the load, bulk and shear passes are skipped.

```python
from openmmpolymer import (
    ModulusSpec,
    analyse_modulus_rates,
    run_modulus_rate_scan,
    write_modulus_rate_report,
)

report = run_modulus_rate_scan(
    run,
    "modulus_rates",
    relax_ps=(50.0, 200.0, 1000.0),
    target_rate_per_ns=0.0002,
    spec=ModulusSpec(temperature_k=298.15, max_strain=0.03),
)
write_modulus_rate_report(report)

# Existing modulus runs can also be analysed together, without more dynamics.
report = analyse_modulus_rates(
    ["modulus_fast", "modulus_medium", "modulus_slow"],
    target_rate_per_ns=0.0002,
    strain_limit=0.015,
)
write_modulus_rate_report(report, output_dir="rate_analysis")
```

Without `output_dir`, the report writer puts `modulus_rates.json` and the
figures in the first measured run's `analysis` directory (for a rate scan,
`modulus_rates/rate_00/analysis`).

For already fitted `ElasticModulus` results, use
`strain_rate_extrapolation(fits, target_rate_per_ns=0.0002)`. The default
`form="log_linear"` fits `E = E_ref + b * log10(rate / rate_ref)`;
`form="power_law"` fits `E = E_ref * (rate / rate_ref)**b` as an alternative.
Both are local empirical relations evaluated at a **positive, finite target
rate**. Neither determines a zero-rate or equilibrium modulus. Logarithmic
rate dependence has been observed in atomistic polyimide simulations, but
the fastest deformations in that study also departed from it; inspect the
measured trend before extending either fit. See
[Nazarychev et al., Soft Matter (2016)](https://pubs.rsc.org/en/content/articlehtml/2016/sm/c6sm00230g).

The results retain each measured modulus and its standard error, propagate
measurement uncertainty and rate-fit scatter to the target estimate, and
report `sensitivity_mpa_per_decade` at `reference_rate_per_ns`. This uncertainty
does not include force-field bias, sample-history differences or the error of
extending an empirical relation beyond its measured range. Compare both forms
and add slower measurements when their predictions diverge. Temperature and
elastic fitting window must match across fits. Unresolved input fits, poor
rate fits or unsupported predictions remain unresolved; the default limit is
two extrapolated decades. A conservative guard also refuses fits whose RMS
residual exceeds 10% of the mean measured modulus or, in the fitted response
scale, three times the RMS input error when any input errors are nonzero.
A target many decades below molecular-dynamics rates is therefore an
unresolved estimate, even when the fitted line is clean.
`plot_strain_rate` shows the measured errors and target uncertainty, and shades
the extrapolated interval.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol modulus -t 298 \
  --modulus-relax-times 50,200,1000 --target-strain-rate 0.0002 \
  -o modulus_rates -v
openmmpolymer --analyse modulus_fast modulus_medium modulus_slow \
  --protocol modulus --target-strain-rate 0.0002
```

Rates are in strain/ns: multiply a rate in s^-1 by `1e-9` before passing it.
Choose a target near the slowest sampled rate to evaluate a controlled
reduction in rate sensitivity; extending directly to a laboratory rate does
not by itself remove kinetic stiffness.

### Rate sensitivity across measured properties

The common `run_property_rate_scan`, `analyse_property_rates` and
`write_rate_report` APIs extend the same measured-rate comparison to the
other loading and thermal workflows. `RATE_PROPERTIES` records each quantity's
units. The target rate must be positive and use the units in the last column;
pressure ramps, heating and cooling are not strain rates.

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

Each new scan prepares one common starting state and branches every rate and
replica from it, with fresh velocities and independent random streams. Only
the requested measurement runs. The scan keeps the loading/temperature
ladder, fitting criterion and preparation fixed while changing the hold per
step. Settings are recorded before dynamics; changed requests are refused on
resume. `validate_property_rate_scan` returns a plan with `total_ns`, counting
the common preparation and every rate and replica. The spec's
`max_total_ns` applies to that whole budget.

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
    spec,
    (100.0, 500.0, 2000.0),
    property_name="bulk_modulus",
    target_rate=10.0,  # bar/ns
)
print(plan.total_ns)
report = run_property_rate_scan(
    run,
    "bulk_rates",
    property_name="bulk_modulus",
    hold_times_ps=(100.0, 500.0, 2000.0),
    target_rate=10.0,
    spec=spec,
)
write_rate_report(report, output_dir="bulk_rates/analysis")

# A scan root expands to its recorded rate runs. Existing compatible runs
# can instead be supplied as ["bulk_fast", "bulk_medium", "bulk_slow"].
report = analyse_property_rates(
    ["bulk_rates"],
    property_name="bulk_modulus",
    target_rate=10.0,
)
```

The nominal pressure/stress and shear rates use total absolute distance along
the imposed ladder divided by the time in all its holds. Reversing a pressure
ladder therefore counts both branches, and the initial hold remains in the
time denominator. Tensile rates retain the compounded engineering-strain
convention above. Mechanical replica counts come from the relevant spec.

For thermal scans, the hold determines `1000 * temperature_step_k / hold_ps`
in K/ns. A Tg rate scan uses the fixed full ladder given by
`melt_temperature_k`, `t_floor_k` and `coarse_step_k`; adaptive fine-window
settings are unused. The existing two-pass `run_tg_scan`,
`cooling_rate_series` and Tg log-linear/VFT analysis remain available.
Thermal replica counts use the scan's `n_replicas` argument.

```python
from openmmpolymer import TgSpec, TmSpec

tg = run_property_rate_scan(
    run,
    "tg_rates",
    property_name="glass_transition",
    hold_times_ps=(500.0, 1500.0, 5000.0),
    target_rate=1.0,  # K/ns
    spec=TgSpec(melt_temperature_k=650, t_floor_k=150, coarse_step_k=10),
    n_replicas=3,
)
write_rate_report(tg, output_dir="tg_rates/analysis")

# crystal_run must contain a prepared crystalline or semicrystalline cell.
tm = run_property_rate_scan(
    crystal_run,
    "tm_rates",
    property_name="melting_temperature",
    hold_times_ps=(500.0, 1500.0, 5000.0),
    target_rate=1.0,  # K/ns
    spec=TmSpec(t_start_k=250, t_end_k=650, step_k=10),
    n_replicas=3,
    crystalline=True,
)
write_rate_report(tm, output_dir="tm_rates/analysis")

elongation = analyse_property_rates(
    ["elongation_fast", "elongation_medium", "elongation_slow"],
    property_name="elongation_at_break",
    target_rate=0.0002,  # strain/ns
)
if elongation.log_linear is not None:
    print(elongation.log_linear.value)  # percentage, not fractional strain
write_rate_report(elongation, output_dir="elongation_rate_analysis")
```

Tm preparation minimises and settles the supplied crystal at its starting
temperature; it never substitutes a melt preparation. `crystalline=True`
asserts the supplied structure's order, including an optional `state_in`.
Common starting coordinates do not establish melt equilibration or independent
crystal morphologies. Cooling-rate effects require material-specific validation
when comparing with experiments, as illustrated by
[the specific-volume/cooling-rate analysis of an epoxy network](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.7b01303).
Heating may superheat a crystal, so a fitted heating-rate correction cannot
establish equilibrium melting; see
[the molecular-dynamics study of superheating versus heating rate](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.68.134206).
Check crystalline-order loss in saved structures or trajectories as well.

The shared analysis requires at least three distinct measured rates. It
retains each observation, source and qualification, pools same-rate replicas,
and attempts both `log_linear` and `power_law` fits. `value`, `standard_error`
and `sensitivity_per_decade` use the property's value unit; the report records
the reference and target rates, extrapolation distance, fit residuals and
resolution status. Physical bounds and expected trends are property-specific;
no universal monotonic correction is imposed on every quantity.

Unknown single-history uncertainty stays unknown. Where available, input
errors and excess rate-fit scatter propagate to the target, with
between-replica standard deviation as a conservative floor. Tg's current
single-history fit supplies no temperature standard error. A Tm bracket is
the adjacent sampled-temperature interval, not an error bar; replicas provide
temperature variability when their transitions differ. These errors do not
cover force-field bias, shared morphology, model choice or experimental
calibration. A missing yield or break event is retained as missing/censored
and prevents an extrapolation from silently fitting only successful events.

Unresolved observations, unknown uncertainty, poor fits, unsupported bounds
or trends, excessive relative target uncertainty and extrapolations beyond
`max_extrapolation_decades=2.0` remain unresolved. A model that cannot be fitted
is `None`, with the reason retained. JSON writes missing/nonfinite fields as
`null`, preserves Tm bracket notes and reports the disagreement between model
predictions. Figures label the property's units and unknown errors. Files are
`<property_name>_rates.json` plus one figure per available model; the default
destination is the first measured run's `analysis` directory.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol modulus -t 298 \
  --rate-property bulk_modulus --rate-hold-times 100,500,2000 \
  --target-property-rate 10 -o bulk_rates
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol tg \
  --rate-property glass_transition --rate-hold-times 500,1500,5000 \
  --target-property-rate 1 -o tg_rates
openmmpolymer --protocol tm --crystal-pdb crystal.pdb --system-xml system.xml \
  --rate-property melting_temperature --rate-hold-times 500,1500,5000 \
  --target-property-rate 1 -o tm_rates
openmmpolymer --analyse elongation_fast elongation_medium elongation_slow \
  --rate-property elongation_at_break --target-property-rate 0.0002
```

New scans require the matching protocol from the table. Saved-run analysis
needs the property and target rate; it does not require a protocol selection.
The older Young's-modulus API and `--modulus-relax-times` /
`--target-strain-rate` options retain their existing behavior.

### Observation-window convergence for other measures

Quantities measured from a trajectory or a relaxation curve need an
observation-window check as well. `analyse_convergence` compares estimates
from increasing fractions of a saved run and writes a separate report; it
does not reinterpret observation time as an imposed strain or thermal rate.

| Measurement family | Checked quantities | Evidence required |
| --- | --- | --- |
| State-data time series | Density, temperature, potential energy | Stable prefix means and disjoint tail blocks; enough autocorrelation-adjusted samples |
| Chain-dimension time series | Mean radius of gyration, mean squared end-to-end distance | The same sampling and stability checks |
| Structural refits | Persistence length, characteristic ratio, ratio of squares | Stable refits and disjoint tail blocks; backbone correlation must decay for persistence length |
| Chain dynamics | COM diffusion coefficient, end-to-end relaxation time | Stable refits; observed diffusive MSD or observed orientational decorrelation |
| Pair and reciprocal structure | RDF peak position/height, S(q) peak position/height | Fixed grids and sampling policy; enough sampled frames and agreement of disjoint tail blocks |
| Stress relaxation | Equilibrium modulus, KWW mean relaxation time, KWW and Prony viscosities | Stable decay refits and evidence that the relevant tail/plateau was observed |

```python
from openmmpolymer import analyse_convergence, write_convergence_report

convergence = analyse_convergence(
    "run",
    stage="05_npt",
    backbone=chain.backbone,
    window_fractions=(0.25, 0.5, 0.75, 1.0),
    relative_tolerance=0.1,
    min_effective_samples=20,
    discard_fraction=0.1,
)
write_convergence_report(convergence, output_dir="run/convergence_analysis")
```

For raw stationary observations, `time_window_convergence(time_ps, values,
property_name="density", value_unit="g/cm^3")` supplies the same check.
`relaxation_window_convergence(curve)` refits an existing `RelaxationCurve`.
Relaxation is a physical decay, so its model parameters are checked through
longer fitted windows rather than treating G(t) as a stationary trace.

The structural report retains the complete RDF, S(q), backbone-correlation,
MSD and end-to-end-correlation curves for each window. One RDF radius remains
legal for every observed box, and the same histogram grids and global strides
apply throughout. S(q) peak comparisons share a bin mask with sufficient
wavevectors per frame. A changing bin population cannot silently change the
set of allowed peak locations.

Pair-distribution and structure-factor sampling defaults to the ordinary
structural-report caps of 50 and 8 frames. Those caps can leave the window
check unresolved, especially S(q). Increase sampling explicitly when the
cost is acceptable:

```python
from openmmpolymer import open_run, structural_window_convergence

ensemble = open_run("run", "05_npt")
structure_windows = structural_window_convergence(
    ensemble,
    backbone=chain.backbone,
    min_frames=20,
    max_distribution_frames=None,
    max_structure_factor_frames=None,
)
print(structure_windows.parameters["diffusion_coefficient_cm2_s"].resolved)
```

Structural refit differences and differences between overlapping windows are
stability diagnostics, not standard errors or independent replicas. Inspect
individual parameter verdicts: a trajectory can have stable pair structure
without observing chain diffusion. A snapshot cannot establish convergence;
repeated frozen coordinates or uniform translation alone also remain
unresolved. An unobserved relaxation time, diffusive regime or backbone decay stays
missing/censored. None of these window checks proves equilibrium or supplies
a zero-rate correction.

```bash
openmmpolymer --analyse run --convergence --convergence-stage 05_npt \
  --window-fractions .25,.5,.75,1 --convergence-tolerance .1 \
  --min-effective-samples 20 --convergence-discard-fraction .1 \
  -o convergence_analysis
```

`--structure-stage` can select the stage when `--convergence-stage` is omitted.
The existing `--backbone`, `--stride` and `--no-figures` options also apply.

## Calculating a yield strength

`run_yield_scan` measures an **apparent offset yield strength** from a tensile
stress-strain curve. It equilibrates the cell, then extends replicas from that
same state with fresh velocities while the transverse dimensions relax at the
specified pressure. The default criterion is a 0.2% offset proof stress. This
is a configurable, operational definition; it is not a universal polymer yield
criterion or a test of permanent deformation after unloading.

```python
from openmmpolymer import (
    YieldSpec,
    analyse_yield,
    run_yield_scan,
    write_yield_report,
)

report = run_yield_scan(
    run,
    "yield_run",
    spec=YieldSpec(
        temperature_k=298.15,
        strain_increment=0.002,
        max_strain=0.3,
        relax_ps=50.0,
        n_replicas=3,
        offset_strain=0.002,
        fit_min_strain=0.0,
        fit_max_strain=0.02,
    ),
)
print(report.strength_mpa, report.replica_spread_mpa, report.resolved)
write_yield_report(report)
# Reanalyse with the saved criterion, without running more dynamics:
write_yield_report(analyse_yield("yield_run"))
```

The workflow uses engineering strain and nominal tensile stress. It subtracts
the mean transverse stress from the axial stress and multiplies by the measured
transverse area ratio `A / A0`, as in the breaking workflow below. It fits an
initial elastic line `sigma = E * strain + intercept`, then finds the first
crossing after that fit window with the parallel offset line
`sigma = E * (strain - offset_strain) + intercept`. Linear interpolation within
the sampled strain bracket gives the proof stress and yield strain.

The elastic fit needs at least five points and must pass its fit-quality checks.
Choose the fit window inside the material's initial linear response; a window
that includes yielding cannot define its elastic slope reliably. A missing
crossing, an invalid elastic fit or an incomplete replica leaves the pooled
strength unresolved (`strength_mpa` is `None`). Reports retain each replica's
curve, fit, crossing bracket and reasons for unresolved results. Figures show
the initial elastic fit and offset line alongside nominal stress.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol yield -t 298 \
  --yield-strain-increment 0.002 --yield-max-strain 0.3 \
  --yield-relax-ps 50 --yield-replicas 3 \
  --yield-offset-strain 0.002 --yield-fit-min-strain 0 \
  --yield-fit-max-strain 0.02 -o yield_run -v
openmmpolymer --analyse yield_run
```

`--yield-stage-ps` controls resumable chunk duration,
`--yield-samples-per-step` controls stress sampling, and
`--yield-trajectory-ps` optionally saves coordinates. Increments extend the
current cell and compound; the maximum strain is relative to the initial cell.
`--max-total-ns` limits the complete equilibration and replica schedule before
the monomer is built. JSON and figures are written to `yield_run/analysis`;
`--no-figures` writes only JSON. Automatic analysis recognises yield runs and
uses their saved fit window and offset.

Report the temperature, strain rate, offset and elastic fit window with the
strength. The rapid rates, force field, chain length and periodic cell size can
shift this apparent value relative to a macroscopic experiment. Replica spread
reflects fresh velocities at one starting structure. For the distinction
between an offset proof stress and a physical onset of yielding, see
[Instron's offset yield strength definition](https://www.instron.com/en/resources/glossary/offset-yield-strength/).

## Calculating a breaking strength

`run_breaking_scan` measures an **apparent ultimate nominal tensile strength**
from a finite extension. It equilibrates the cell, then starts each tensile
replica from that same state with fresh velocities. The extension proceeds in
increments while the two transverse dimensions relax at the specified
pressure. Chunks retain the unstrained reference box so an interrupted scan
can resume without resetting its strain origin.

```python
from openmmpolymer import (
    BreakingSpec,
    analyse_breaking,
    run_breaking_scan,
    write_breaking_report,
)

report = run_breaking_scan(
    run,
    "breaking_run",
    spec=BreakingSpec(
        temperature_k=298.15,
        strain_increment=0.01,
        max_strain=1.0,
        relax_ps=50.0,
        n_replicas=3,
        trajectory_ps=10.0,
    ),
)
print(report.strength_mpa, report.replica_spread_mpa, report.resolved)
write_breaking_report(report)
# Reanalyse the recorded holds later, without rerunning dynamics:
write_breaking_report(analyse_breaking("breaking_run"))
```

The recorded stress tensor uses the instantaneous cell. The workflow subtracts
the mean transverse stress from the axial stress, then multiplies this
differential true stress by the transverse area ratio `A / A0` to obtain
nominal stress in MPa. It uses the recorded transverse dimensions; it does not
assume constant volume. The reported strength is the **peak** nominal stress.
The stress at the subsequent drop and its strain are reported separately.
The conversion uses each hold's mean stress and final transverse area, so it
approximates the mean force when that area fluctuates during the hold.

A peak is confirmed only when the curve ends with a sustained drop below a
specified fraction of that peak. The defaults require three consecutive
terminal holds below half the peak. A curve that is still rising, recovers
after a temporary dip, or does not reach that drop remains unresolved;
`strength_mpa` is `None` and the observed peak remains available per replica.
Incomplete replicas or stages also leave the pooled result unresolved. The
threshold is an operational definition of loss of load-bearing capacity, not
an observation of a crack.

The supported force fields have [fixed harmonic bonds](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#harmonicbondforce): this workflow does **not** model
chemical bond scission or measure covalent fracture strength. A loss of stress
can reflect chain sliding or separation. Interpret it alongside saved
coordinates, and report the temperature, strain rate, force field, chain
length and periodic cell size. These small periodic cells and rapid molecular
dynamics strain rates can give strengths different from a macroscopic tensile
test. The replica spread measures the variation from fresh velocities at one
starting structure.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol breaking -t 298 \
  --breaking-strain-increment 0.01 --breaking-max-strain 1.0 \
  --breaking-relax-ps 50 --breaking-replicas 3 \
  --failure-fraction 0.5 --confirmation-steps 3 \
  --breaking-trajectory-ps 10 -o breaking_run -v
openmmpolymer --analyse breaking_run
```

Strain increments compound: `0.01` extends the current cell by one percent,
and `--breaking-max-strain 1.0` requests at least 100% engineering strain.
`--breaking-stage-ps` controls the resumable chunk duration;
`--breaking-samples-per-step` controls stress sampling. `--max-total-ns`
limits the full equilibration and replica schedule. Reports contain the
stress-strain curves, individual peaks and drop criteria, and any reasons the
strength remained unresolved. Use `--no-figures` for a JSON-only report.

## Calculating elongation at break

`run_elongation_scan` measures **apparent engineering elongation at break** from
the onset of a confirmed terminal loss of nominal tensile stress. It uses the
same finite-extension schedule as the breaking-strength workflow: equilibrate
the cell, start tensile replicas with fresh velocities, and relax the transverse
dimensions at the specified pressure between strain increments.

```python
from openmmpolymer import (
    ElongationSpec,
    analyse_elongation,
    run_elongation_scan,
    write_elongation_report,
)

report = run_elongation_scan(
    run,
    "elongation_run",
    spec=ElongationSpec(
        temperature_k=298.15,
        strain_increment=0.01,
        max_strain=1.0,
        relax_ps=50.0,
        n_replicas=3,
        failure_fraction=0.5,
        confirmation_steps=3,
        trajectory_ps=10.0,
    ),
)
print(report.elongation_percent, report.replica_spread_percent, report.resolved)
write_elongation_report(report)
# Reanalyse recorded holds with the saved criterion, without more dynamics:
write_elongation_report(analyse_elongation("elongation_run"))
```

The reported value is `100 * (L_break - L0) / L0`, where `L0` is the equilibrated
axial cell length. This is percent engineering strain, with the original cell
serving as the reference length. Experimental strain at break likewise uses
the change in gauge length divided by its original length; see
[Instron's explanation of strain at break](https://instron.com/en/resources/blog/2013/november/question-from-a-customer-on-how-to-report-strain-at-break/).
The workflow's break criterion is operational: by default, the curve must end
with at least three consecutive holds at or below half its peak nominal stress.
Elongation is taken at the **first** hold in that confirmed terminal drop. It
is distinct from the strain at ultimate tensile stress and from the last hold
that confirms failure. The preceding hold and that first low-stress hold form
the reported `break_bracket`; no interpolation is used. Smaller strain
increments narrow this sampling interval.

A rising curve, a temporary stress drop followed by recovery, an insufficient
number of confirming holds, or a missing or incomplete replica leaves the
pooled elongation unresolved (`elongation_percent` is `None`). The workflow
does not substitute the maximum imposed strain for a missing break. Reports
retain each replica's curve, sampled peak, break bracket and unresolved notes.
Figures show nominal stress against percent elongation and mark the break
onset separately from the peak. `replica_spread_percent` is a spread in
percentage points from fresh velocities at one starting structure.

```bash
openmmpolymer '[*]CC[*]' -n 30 -c 40 -r PE --protocol elongation -t 298 \
  --elongation-strain-increment 0.01 --elongation-max-strain 1.0 \
  --elongation-relax-ps 50 --elongation-replicas 3 \
  --failure-fraction 0.5 --confirmation-steps 3 \
  --elongation-trajectory-ps 10 -o elongation_run -v
openmmpolymer --analyse elongation_run
```

The strain options use fractions: `0.01` extends the current cell by one
percent, and `--elongation-max-strain 1.0` requests at least 100% engineering
strain relative to the initial cell. Increments compound.
`--elongation-stage-ps` controls resumable chunks,
`--elongation-samples-per-step` controls stress sampling, and
`--max-total-ns` limits the full equilibration and replica schedule. Reports
are written under `elongation_run/analysis`; `--no-figures` writes only JSON.
Automatic analysis recognises elongation runs and restores their saved failure
criterion.

The supported force fields have fixed harmonic bonds, so this workflow cannot
model covalent bond rupture. A stress loss can reflect chain sliding or
separation. Inspect saved coordinates and report the temperature, strain rate,
failure criterion, force field, chain length and periodic cell size alongside
the result. Molecular dynamics rates and small periodic cells limit comparison
with macroscopic elongation-at-break measurements.

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
| Strain rate | `strain_rate` | Young's modulus versus strain rate, local empirical fits and finite-rate extrapolation |
| Relaxation | `relaxation` | `G(t)` from a step strain, and the Prony and KWW fits read off it |
| Workflow | `tg` | The two-pass glass-transition scan, and reading a finished run back |
| Workflow | `mechanical` | The extension, the load, bulk and shear passes, and the report |
| Workflow | `modulus_rates` | Modulus scans at several relaxation times, shared analysis and rate reports |
| Workflow | `breaking` | Finite tensile extension, nominal stress peaks and sustained stress drops, replicas and reporting |
| Workflow | `elongation` | Percent engineering elongation at a confirmed terminal stress loss, replicas and reporting |
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

**Record the backbone with the run.** The backbone atom indices come
from the attachment points the caps consumed when the chain was built, and
nothing downstream can recover them from the structure alone. The workflow
drivers record them as `chain_backbone` in their own `*_workflow.json`; a plain
protocol run given `chain_backbone` records them in `manifest.chains.backbone`
alongside the final dimensions. `analyse_structure` looks for a backbone
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

**CHARMM cache entries include the parameter streams.** Both stream order and
file contents are fingerprinted. Editing a stream in place or replacing it
invalidates the entry; a missing stream is an error even if a previous result
was cached.

## Reference benchmarks

[The polyethylene benchmark](benchmarks/README.md) builds independent C44H90
melts, runs their preparation and NPT measurement, and compares density with
the published specific volumes in Lee, Frank and Yoon,
[Polymers 12 (2020), Table 2](https://doi.org/10.3390/polym12051059).
The checked-in reference records conditions, sources and a fixed comparison
tolerance. Reports include replica variation, density settling and chain
decorrelation; insufficient sampling cannot pass the comparison.

The reference uses a united-atom model, while this benchmark uses all-atom
OpenFF. Agreement is a cross-model check, not experimental validation of every
property. A short `--smoke` run tests the actual chemistry-to-analysis pipeline
and always reports `smoke_only`. The full comparison requires long simulations;
see the benchmark documentation for commands and the scope of recorded results.

## Development

The environment file already installs the test and lint tools. Arm the
pre-commit hooks once, then run the same checks CI does:

```bash
pre-commit install
```

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
