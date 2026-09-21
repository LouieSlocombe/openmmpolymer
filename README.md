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

## How it fits together

| Layer | Module | What it does |
|---|---|---|
| Chain | `chain` | Monomer SMILES → one 3D molecule per chain copy, as SDF and PDB |
| Charges | `charges` | Partial charges, written into the SDF |
| Force field | `forcefield` | forcefill → an ffxml, cached |
| Packing | `packing` | Box arithmetic, packmol, and the checks that catch a bad cell |
| System | `mdsystem` | Replicated topology, packmol's coordinates, `createSystem` |
| Stages | `simulate` | minimise, push-off, NVT, compress, NPT, anneal, quench, production |
| Protocol | `protocols` | Named stage sequences, the manifest, and resume |
| Trajectory | `trajectory` | A finished run directory → frames, per chain, in nanometres |
| Time series | `timeseries` | State-data CSVs, equilibration detection, the quench curve |
| Conformation | `conformation` | `⟨R²⟩`, `Rg`, persistence length, end-to-end relaxation, COM displacement |
| Correlations | `correlations` | Intermolecular `g(r)` and the static structure factor |
| Plots | `plots` | A figure per result, returned rather than written |

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

**A quench is not a Tg measurement.** `melt_quench` records a density at every
temperature on the way down, which is the specific-volume curve a glass
transition is read off. Every all-atom cooling rate is many orders of magnitude
faster than any experiment, so the transition sits well above the measured one.
The shape is informative; the number is not directly comparable. `quench_curve`
reads that curve back and `glass_transition` fits the two straight lines it is
read off, reporting the cooling rate alongside the temperature so the caveat
travels with the number, and `resolved` False when the fit found a corner in
noise — which is what fitting two lines to a straight one always finds.

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
