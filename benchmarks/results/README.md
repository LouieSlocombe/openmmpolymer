# Recorded smoke evidence

`pe_c44_smoke.json` is the unmodified report from a real CPU execution on
2026-09-23 of:

```bash
OPENMM_CPU_THREADS=1 python -m benchmarks.pe_melt \
  --output /tmp/openmmpolymer-pe-c44-audit-smoke-loose \
  --smoke --seeds 11 --platform CPU
```

It exercised the 4,020-atom C44H90 system through chain construction, Gasteiger
charges, OpenFF force-field parameterization, Packmol, minimization, dynamics,
trajectory reading and analysis. The report records the package versions,
effective protocol, case digest and relative manifest path. Build intermediates,
trajectories and manifests are not checked in; rerun the command to regenerate
them and a new report.

This is integration evidence only. Initial packing is deliberately sparse,
production lasts 0.5 ps, and the report correctly states `smoke_only` with
insufficient sampling and no reference-density agreement. No equilibrated
three-replica reference result is claimed by this artifact.
