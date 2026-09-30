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

## Measurements and analysis

Each measurement prepares the cell, records its request and resumes from intact
stages. Results carry `resolved` and `notes`; an unsupported estimate stays
`None`. Finished runs can be analysed again without running dynamics:

```bash
openmmpolymer --analyse run
```

The [user guide](docs/guide.md) covers the workflows, their assumptions and
complete Python and CLI examples:

| Task | Guide |
| --- | --- |
| Cooling and heating | [Glass transition](docs/guide.md#glass-transition), [melting temperature](docs/guide.md#melting-temperature) |
| Mechanical properties | [Elastic constants](docs/guide.md#elastic-constants), [tensile strength](docs/guide.md#tensile-strength-yield-breaking-and-elongation-at-break), [stress relaxation](docs/guide.md#stress-relaxation) |
| Structure and dynamics | [Chain dimensions, correlations and relaxation](docs/guide.md#structure) |
| Sampling checks | [Rate dependence](docs/guide.md#rate-dependence), [observation-window convergence](docs/guide.md#observation-window-convergence) |

Before interpreting a result, read the [model assumptions](docs/guide.md#the-constraints-everything-follows-from)
and [simulation conventions](docs/guide.md#things-worth-knowing). Equilibration
is measured rather than assumed, replicas share a prepared structure, and
molecular-dynamics rates are far faster than experimental ones.

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
