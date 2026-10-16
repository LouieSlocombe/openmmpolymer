# Step-by-step examples for the six main properties

[Quickstart](../README.md) · [First-principles calculations](first_principles.md) ·
[Workflow reference](guide.md)

These examples take you from a starting structure to **Tg, Tm, breaking
strength, yield strength, elongation at break and Young's modulus**. Each
linked script is standalone, with numbered comments showing the inputs,
preparation, simulation, analysis and report writing. The commands below run
real simulations; for a fast numerical demonstration with known answers, use
[`first_principles.py`](../examples/first_principles.py).

| Property | Standalone script | Report file under the output directory |
| --- | --- | --- |
| Tg, glass transition temperature | [glass_transition.py](../examples/glass_transition.py) | `analysis/tg.json` |
| Tm, melting temperature | [melting_temperature.py](../examples/melting_temperature.py) | `analysis/tm.json` |
| σb, breaking strength | [breaking_strength.py](../examples/breaking_strength.py) | `analysis/breaking.json` |
| σy, yield strength | [yield_strength.py](../examples/yield_strength.py) | `analysis/yield.json` |
| εb, elongation at break | [elongation_at_break.py](../examples/elongation_at_break.py) | `analysis/elongation.json` |
| E, Young's modulus | [youngs_modulus.py](../examples/youngs_modulus.py) | `analysis/mechanics.json` |

## Before you start

1. From the repository root, install and activate the full environment:

   ```bash
   conda env create -f build_tools/environment.yml
   conda activate openmmpolymer
   python -m pip install -e . --no-deps
   ```

2. Read the relevant script and edit its `ChainSpec` and measurement `Spec`
   for your material. The melt-based examples use 40 polyethylene chains of
   30 repeat units, with NAGL charges and the default force field. These are
   illustrative inputs, not validated predictions for every polymer.
   `build_melt` builds, charges, parameterises and packs the chains; the
   subsequent `run_*_scan` performs the preparation and measurement.

3. Check the units before editing: all `*_k` settings are **kelvin**; times
   are **picoseconds**; pressure is **bar**; strains are **fractions**.
   `298.15 K = 25 °C`, `1000 ps = 1 ns`, and `0.01` strain means 1%.
   The scripts convert Tg/Tm to °C for display. Their JSON reports retain K.
   Strength and modulus reports already use MPa, and elongation reports
   already use percent.

4. Choose a fresh output directory for each method and each changed set of
   inputs. Repeat the same command to resume intact stages. Do not run
   different methods into one directory. The examples include a
   `max_total_ns` budget; this limits simulated time, not elapsed wall time,
   and does not guarantee convergence. Full scans can take substantial time.

5. Each script supports `--help`, `--analyse-only` and `--no-figures`.
   `--analyse-only` reads the existing run and regenerates its report without
   building a cell or running dynamics. `--no-figures` keeps JSON output and
   skips plots. A resolved result depends on the evidence in the run; an
   unresolved result is not replaced with zero or a plausible-looking fit.

## 1. Tg: glass transition temperature in °C

Use [glass_transition.py](../examples/glass_transition.py).

1. **Choose the temperature range.** The example equilibrates at 650 K and
   cools to 150 K. The coarse spacing is 25 K with 1000 ps holds. A fine
   window extends 60 K either side of the candidate transition, with 5 K
   spacing and 3000 ps holds. Edit these in `TgSpec` so the range brackets
   your polymer's transition. The budget is 150 ns.

2. **Build and prepare the melt.** The script calls `build_melt` and then
   `run_tg_scan`, passing the chain backbone and atom count so the run records
   its chain geometry. Equilibration trajectories are enabled for the melt
   check. A settled density alone does not establish that the chains relaxed.

3. **Run the scan.**

   ```bash
   python examples/glass_transition.py --output tg-example
   ```

   Each temperature hold supplies an averaged density. Analysis converts it
   to specific volume, fits glass and melt branches, and locates their
   intersection. The fine scan starts from a saved point on the coarse cooling
   path. If no supported coarse transition locates a fine window, the scan
   can stop with a `TgError`; inspect the coarse evidence and improve the
   range or sampling. A manually chosen window needs independent justification.

4. **Read Tg and its cooling rate.** The script rereads the saved data with
   `analyse_tg`, prints the temperature in °C only when resolved, and writes
   `analysis/tg.json` and figures. To extract the same headline yourself:

   ```python
   from openmmpolymer import analyse_tg

   report = analyse_tg("tg-example")
   tg_c = (
       report.temperature_k - 273.15
       if report.resolved and report.temperature_k is not None
       else None
   )
   print("Tg (°C):", tg_c)
   print("Cooling rate (K/ns):", report.cooling_rate_k_per_ns)
   print("Notes:", report.notes)
   ```

5. **Inspect and repeat.** Check the two branches, their intersection and the
   melt-equilibration notes. Repeat with smaller temperature steps or longer
   holds in a new directory. The reported Tg is at the recorded cooling rate.
   To reread a completed run:

   ```bash
   python examples/glass_transition.py --output tg-example --analyse-only
   ```

The [thermal derivation](first_principles.md#5-thermal-transitions-change-the-temperature)
shows the intersection calculation explicitly.

## 2. Tm: melting temperature in °C

Use [melting_temperature.py](../examples/melting_temperature.py).

1. **Supply a crystal.** Prepare `crystal.pdb` with periodic box vectors,
   elements and bonds, and a matching serialized OpenMM `system.xml` with
   the same atom ordering. The System must have an appropriate force field
   and no thermostat or barostat. Molecules must occupy equal-sized,
   contiguous atom blocks. `load_crystal` validates these inputs. The normal
   monomer-to-melt builder does not prepare a crystalline morphology.

2. **Choose the heating experiment.** `TmSpec` starts at 250 K, ends at
   650 K, steps by 10 K and holds each temperature for 1000 ps. It also
   equilibrates the crystal at the starting temperature. Choose the endpoints
   below and above the expected transition. The example records coordinates
   every 10 ps and uses anisotropic pressure control at 1 bar.

3. **Run with the two input files.**

   ```bash
   python examples/melting_temperature.py --output tm-example \
     --crystal-pdb crystal.pdb --system-xml system.xml
   ```

   The script's `crystalline=True` declares the intended starting morphology;
   it is not an automated crystallinity test. The scan looks for coincident
   positive jumps in specific volume and enthalpy.

4. **Read the temperature and bracket.** The headline is the midpoint of the
   sampled heating interval, whose endpoints remain available. Convert both
   the headline and bracket to °C:

   ```python
   from openmmpolymer import analyse_melting

   report = analyse_melting("tm-example")
   tm_c = (
       report.temperature_k - 273.15
       if report.resolved and report.temperature_k is not None
       else None
   )
   bracket = report.transition.bracket_k
   bracket_c = (
       tuple(value - 273.15 for value in bracket)
       if report.resolved and bracket is not None
       else None
   )
   print("Tm (°C):", tm_c, "Sampled bracket (°C):", bracket_c)
   print("Notes:", report.notes)
   ```

5. **Check the physical event.** Inspect the density/enthalpy plots and the
   saved structures for loss of crystalline order. The bracket is a sampling
   interval, not a confidence interval. Slower heating and different crystal
   preparations test rate and morphology effects. Reanalysis needs only the
   saved directory:

   ```bash
   python examples/melting_temperature.py --output tm-example --analyse-only
   ```

## 3. E: Young's modulus in MPa

Use [youngs_modulus.py](../examples/youngs_modulus.py).

1. **Choose temperature and the elastic window.** The example uses 298.15 K,
   1 bar and z-axis extension. It applies 0.002 strain increments up to 0.05,
   with 50 ps holds, and fits only through 0.015 axial strain. These are
   fractions: the fitted window ends at 1.5%, not 0.015%.
   Each increment extends the current length, so the accumulated strain is
   `(1.002 ** n) - 1` and can slightly exceed the requested endpoint.

2. **Choose sampling and the cross-check.** `ModulusSpec` requests three
   replicas and 250 stress readings per hold. A separate imposed-stress
   experiment checks E against the extension result. This example skips the
   bulk and shear passes to focus on E; notes about their absence are expected.

3. **Build, equilibrate and extend.**

   ```bash
   python examples/youngs_modulus.py --output modulus-example
   ```

   Each replica branches from the equilibrated structure with fresh
   velocities. The transverse axes relax at pressure. The modulus is the
   slope of axial-minus-mean-transverse stress against engineering strain,
   with a fitted intercept.

4. **Read the fitted modulus and spread.**

   ```python
   from openmmpolymer import analyse_mechanics

   report = analyse_mechanics("modulus-example", strain_limit=0.015)
   fit = report.youngs
   e_mpa = (
       fit.modulus_mpa if report.resolved and fit is not None and fit.resolved else None
   )
   print("E (MPa):", e_mpa)
   print("Replica standard deviation (MPa):", report.replica_spread_mpa)
   print("Extension/load relative gap:", report.method_gap)
   ```

5. **Verify linearity and relaxation.** Inspect the stress–strain curve,
   early/late slopes in the fitting window, replica spread and load-method
   agreement. Keep the `strain_limit` used for reanalysis consistent with
   `elastic_strain_limit` in the script. Try a smaller fitting window and
   longer holds. Whether 298.15 K is above or below Tg depends on the polymer.

   ```bash
   python examples/youngs_modulus.py --output modulus-example --analyse-only
   ```

See the [elastic derivation](first_principles.md#6-elastic-constants-apply-a-small-perturbation)
for stress signs, units and the E/ν/K/G relations.

## 4. σy: yield strength in MPa

Use [yield_strength.py](../examples/yield_strength.py).

1. **Set a fine strain ladder.** `YieldSpec` uses 0.002 increments up to 0.3
   strain, with three replicas at 298.15 K. Small increments are needed to
   supply at least five points in the elastic fitting interval.

2. **Set the proof-stress criterion.** The example uses an offset of 0.002
   strain, or 0.2%. It fits the initial nominal stress–strain line, shifts
   that line by the offset and searches for the first later crossing.
   `fit_min_strain` and `fit_max_strain` select the elastic fit; keep that
   interval below the onset of nonlinearity.

3. **Run the measurement.**

   ```bash
   python examples/yield_strength.py --output yield-example
   ```

   The workflow prepares the cell and extends each replica. Recorded Cauchy
   stress is converted to nominal stress using the measured transverse area.
   The yield strength is interpolated within the crossing bracket.

4. **Read σy and the evidence.**

   ```python
   from openmmpolymer import analyse_yield

   report = analyse_yield("yield-example")
   sigma_y_mpa = report.strength_mpa if report.resolved else None
   print("Yield strength (MPa):", sigma_y_mpa)
   for fit in report.replicas:
       print("Replica resolved:", fit.resolved, "Yield strain:", fit.yield_strain)
   print("Notes:", report.notes)
   ```

5. **Inspect the offset crossing.** An unreliable elastic slope or a curve
   without a crossing leaves yield unresolved. This is an offset proof
   stress; an unloading experiment would be needed to establish permanent
   deformation. Reanalysis uses the saved criterion:

   ```bash
   python examples/yield_strength.py --output yield-example --analyse-only
   ```

## 5. σb: breaking strength in MPa

Use [breaking_strength.py](../examples/breaking_strength.py).

1. **Set an extension range capable of containing failure.** The example
   requests increments of 0.01 up to 1.0 strain, three replicas, 50 ps holds
   and 250 stress readings per hold at 298.15 K. A maximum strain of 1.0
   requests 100% extension. The increments compound, so the final imposed
   strain can slightly exceed the target.

2. **Choose the failure criterion.** `failure_fraction=0.5` and
   `confirmation_steps=3` require at least three final consecutive holds at
   or below half the peak nominal stress. A single low point does not
   confirm a break, and a curve that recovers does not satisfy a terminal drop.

3. **Run and record the deformation.**

   ```bash
   python examples/breaking_strength.py --output breaking-example
   ```

   Saved trajectories help distinguish chain separation from numerical or
   structural problems. The force field keeps bonded connectivity fixed;
   the observed failure is a loss of load-bearing stress, not covalent scission.

4. **Read σb.** Each replica's breaking strength is its peak nominal stress,
   accepted only after the terminal drop has been confirmed. The headline
   averages the confirmed strengths across replicas:

   ```python
   from openmmpolymer import analyse_breaking

   report = analyse_breaking("breaking-example")
   sigma_b_mpa = report.strength_mpa if report.resolved else None
   print("Breaking strength (MPa):", sigma_b_mpa)
   print("Notes:", report.notes)
   ```

5. **Check each replica.** Inspect the peak and final low-stress holds. If
   failure has not happened, the largest stress reached is not substituted
   for resolved breaking strength. Increase the extension range or sampling
   in a new run when justified.

   ```bash
   python examples/breaking_strength.py --output breaking-example --analyse-only
   ```

## 6. εb: elongation at break in percent

Use [elongation_at_break.py](../examples/elongation_at_break.py).

1. **Choose the same kind of extension experiment.** `ElongationSpec` uses
   the same strain ladder and terminal stress-loss criterion as breaking
   strength. The example has three replicas, 0.01 increments, a target of
   1.0 strain and a three-hold confirmation below half the peak.

2. **Keep the original reference length.** Every resumable chunk uses the
   same equilibrated, unstrained length L₀. The relevant strain is
   `(L_break − L0) / L0`, at the first hold of the confirmed terminal drop.
   It is distinct from the strain at peak stress.

3. **Run the elongation workflow.**

   ```bash
   python examples/elongation_at_break.py --output elongation-example
   ```

4. **Read εb without converting it twice.** The report already multiplies
   engineering strain by 100:

   ```python
   from openmmpolymer import analyse_elongation

   report = analyse_elongation("elongation-example")
   epsilon_b_percent = report.elongation_percent if report.resolved else None
   print("Elongation at break (%):", epsilon_b_percent)
   print("Notes:", report.notes)
   ```

5. **Inspect the break bracket.** The previous hold and first confirmed low
   hold bracket the event. There is no interpolation of break strain and no
   substitution of the maximum imposed strain when failure was not observed.
   Smaller increments can narrow this bracket.

   ```bash
   python examples/elongation_at_break.py --output elongation-example --analyse-only
   ```

The [tensile derivation](first_principles.md#7-yield-breaking-and-elongation-keep-track-of-area)
explains nominal stress, the offset line and the distinction between the
peak and the break strain.

## Reading the reports together

Keep the headline's units, temperature, pressure, force field, chain size,
preparation, rate and criterion with every exported value. When a scan reaches
analysis, the scripts save JSON and figures even when a property is unresolved,
so you can inspect the curves and diagnostics that explain the missing result.
The files retain the native K/MPa/% conventions; the console conversion to °C
does not change them.

Yield, breaking and elongation headlines are means of the individual replica
estimates, and remain unresolved if a requested replica is missing, incomplete
or unresolved. The Young's modulus headline fits the pooled replica curves
and reports their separate fitted-modulus spread beside it.

The three mechanical replicas share a prepared structure and differ in their
velocities. Their spread describes trajectory variability at that structure;
independently prepared morphologies test a different source of uncertainty.
For more evidence, follow the [rate-dependence](guide.md#rate-dependence) and
[observation-window](guide.md#observation-window-convergence) examples.
