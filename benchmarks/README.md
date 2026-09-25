# Polyethylene melt benchmark

This benchmark adds a real polymer chemistry-to-analysis check alongside the
synthetic and argon tests. It builds 30 independently grown C44H90 molecules
(22 ethylene repeat units per molecule), assigns charges, parameterises them
with OpenFF 2.2.1, packs a periodic cell, prepares the melt and measures its
density under NPT. Each replica starts from its own packing and random seed,
and is built by `openmmpolymer.build_melt`.

## Reference and acceptance

The source is Lee, Frank and Yoon, *Interface Characteristics of Neat Melts and
Binary Mixtures of Polyethylenes from Atomistic Molecular Dynamics Simulations*,
[Polymers 12 (2020), 1059, Table 2](https://doi.org/10.3390/polym12051059).
For C44H90 it reports specific volumes of 1.232 cm³/g at 350 K and
1.269 cm³/g at 400 K. The machine-readable source and conditions are in
[polyethylene.json](polyethylene.json), and the temperatures it covers are the
only ones the benchmark accepts.

Those numbers come from a united-atom simulation. This project's comparison
uses all-atom OpenFF, NAGL charges and 1 atm. The cited table does not specify
the bulk NPT pressure. The ±10% density interval is a deliberately broad,
fixed project tolerance for comparing models; it is not the source's error
bar or a claim of experimental accuracy. These limitations travel with the
JSON report.

A scientific pass needs all replicas within that density interval, their
retained mean temperatures within 2% of the requested temperature, at least
three independently seeded replicas, settled density with sufficient
independent samples, and decorrelated chain end-to-end vectors. A density near
the target without adequate sampling remains `unresolved`. This benchmark does
not validate Tg, Tm, tensile strength or elastic constants; each requires its
own material, state and observation-time reference.

## Running

Use the repository's conda environment, with Packmol available on PATH. Run
from the repository root:

```bash
# Real build, parameterisation, packing, dynamics, trajectory reading and reporting.
OPENMM_CPU_THREADS=1 python -m benchmarks.pe_melt \
  --output /tmp/pe-smoke --smoke --seeds 11 --platform CPU

# Full preparation plus 10 ns of NPT measurement per replica.
python -m benchmarks.pe_melt --output pe-reference-400 --temperature 400
python -m benchmarks.pe_melt --output pe-reference-350 --temperature 350
```

The full commands default to seeds 11, 29 and 47. They can take substantial
time on a CPU. Device selection uses the package's ordinary platform logic;
pass `--platform CUDA` when that platform is available. Repeat the exact
command to resume: each replica is rebuilt in scratch space and checked against
its recorded build before its run continues, so a dependency upgrade cannot
silently change the Hamiltonian under a half-finished run. Use a different
output directory for altered settings.

The short smoke protocol uses Gasteiger charges, an initial packing density of
0.1 g/cm³ instead of 0.3 g/cm³, a Packmol attempt bounded to two minutes, and
sub-picosecond dynamics to keep the integration test affordable. It always
reports `smoke_only`, even if its density happens to fall inside the reference
interval. It must never be substituted for the NAGL/long-trajectory
comparison.

Each output directory contains `benchmark.json`, the effective build request,
and per replica its build, manifest and trajectories. The report includes
dependency versions, a checksum of the fixed reference, temperature, density,
replica standard error, equilibration diagnostics and reference provenance. A
full command exits with status 1 when the comparison fails or remains
unresolved.

## Recorded results

[`results/pe_c44_smoke.json`](results/pe_c44_smoke.json) is the unmodified
report of a real CPU execution, on 2026-09-23, of:

```bash
OPENMM_CPU_THREADS=1 python -m benchmarks.pe_melt \
  --output /tmp/openmmpolymer-pe-c44-audit-smoke-loose \
  --smoke --seeds 11 --platform CPU
```

It took the 4,020-atom C44H90 system through chain construction, Gasteiger
charges, OpenFF parameterisation, Packmol, minimisation, dynamics, trajectory
reading and analysis, and records the package versions, the effective
protocol, the case digest and the relative manifest path. Build intermediates,
trajectories and manifests are not checked in; rerun the command to regenerate
them and a new report.

This is integration evidence only: the initial packing is deliberately sparse,
production lasts 0.5 ps, and the report correctly states `smoke_only`, with
insufficient sampling and no reference-density agreement. No full 10 ns,
three-replica reference result is claimed - not from this report, and not from
passing the unit tests. The full comparison should be run and its report
reviewed before relying on quantitative material predictions.
