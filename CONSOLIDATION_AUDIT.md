# Consolidation audit: follow-up plan

Audit of duplicated and overlapping functionality in openmmpolymer, 2026-10-09, `main` at `9e22802`.
This file is the work order; implementation progress and dispositions are recorded in section 10. Line numbers are at that commit and drift as work lands, so re-grep the symbol before editing.

How to use it: do the stages in section 4 in order, tick the boxes, and keep the gates in section 8 green after every commit. Each box points at a finding (F1-F18, T1-T8) in sections 5 and 6, which carry the locations, evidence, constraints and tests.

Keep this file at the repository root. Do not move it into `docs/`: `tests/test_readme.py` parses every code fence in `docs/*.md`, and `docs/` ships in the sdist.

## 1. Summary

The recent consolidation passes (#24-#30) removed the copy-paste: a scan of repeated 3- and 4-statement blocks finds only three in 29.7k lines (the `mechanical.py`/`viscoelastic.py` driver, the heavy-atom load in `correlations.py`, and a four-line setup block shared by `run_deform` and `run_relax`). What remains is the same job written several ways, mostly in the scan/resume, report-writing and validation layers.

| | Source lines | Share |
|---|---|---|
| Recommended, no owner decision | about -430 | 1.4 % of 29.7k |
| With the owner decisions in section 3 | about -730 | 2.5 % |
| Tests (test-only) | about -430, of which about -135 is worth doing | 2.0 % of 21.5k |

The case for doing it is drift prevention, not size. Three duplicates have already drifted into visible differences (each reproduced, scripts in Appendix B):

- A Tg scan resumes after its `tg_workflow.json` is deleted. Mechanical, viscoelastic and tensile refuse.
- Tensile's report writer leaves a half-written `breaking.json` on an unsupported `figure_format`; tm's refuses first; neither stamps version provenance.
- `run_nvt(duration_ps=-1)`, `run_quench(hold_ps=-1)`, `run_quench(pressure_bar=nan)`, `run_anneal(window_ps=-1, hold_ps=-1)`, `run_compress(duration_ps_each=-1)`, `run_compress(pressures_bar=(nan,))` and `run_load(stresses_bar=[nan])` are accepted silently. Only `run_heat` validates.

Start with Stage 0 (section 4): it changes no production code and makes every later stage checkable.

## 2. Constraints (read before touching anything)

Resume compares recorded fingerprints, so these are on-disk formats.

Pinned by tests today:

- every stage runner's recorded defaults: `RECORDED_DEFAULTS_SHA256`, `tests/test_protocols.py:82`
- the CLI request namespaces: `RECORDED_REQUEST_SHA256`, `tests/test_cli.py:56` (dest, flags, default and const only; not type, choices, nargs or help)

Not pinned by anything (Stage 0 adds the pins):

- what each protocol builder puts on its stages: names (they seed the RNG through `_seeds.derive_seed(..., stage_name)`) and options (compared on resume)
- the `request` dict each scan saves in `*_workflow.json` (compared whole)
- `build/inputs.json`: `melt.py:149-167` refuses a rebuild unless `asdict(ChainResult)` (less paths) and `_run_identity` are exactly equal as floats and the packed cell's `positions_sha256` matches. One ulp in `chain.py` Rg, bond length or C, or any change to RNG draw order or floating-point order in `_frame`, `_set_torsion`, `_dihedral` or `_grow_conformer`, makes old build directories refuse to resume. Removing or renaming a `ChainResult` field breaks it too. `melt.py` passes `spec.seed` raw to packmol on purpose: do not tidy it into `derive_seed`.

Never change: parameter names and defaults of the `run_*` stage functions and `run_segments`; stage names and kinds; the options builders put on stages; request dicts; CLI dests and defaults.

Other facts that bound the work:

- `quench_temperatures` (repeated subtraction) and `heating_temperatures` (`t_start + i*step`) differ in the last bits for decimal steps (0.1, 0.3, 0.7, 7.3 K), and the lists are recorded as `temperatures_k` options. Do not unify them (Appendix B5).
- mechanical and viscoelastic requests include `max_total_ns` and the chain options (changing only the budget blocks a resume); tensile, tm and tensile_rates drop the budget, and tensile also omits the chain options. Any shared request builder must reproduce each caller's current dict.
- Tests monkeypatch `melt._prepare`, `melt.build_chain` and `<module>.run_protocol` (tensile, tensile_rates, elastic_rates, thermal_rates). Moving a symbol needs a test edit, not an on-disk change.
- `pyproject.toml` runs pytest with warnings as errors and `fail_under = 90`.

## 3. Decisions needed from the owner

| ID | Decision | Recommendation | Blocks |
|---|---|---|---|
| D1 | Make Tg's pre-flight as strict as the others (refuse a manifest with no workflow record, and a foreign protocol)? | Yes: it closes a real gap; needs a test. | F1, Stage 3 |
| D2 | Leave mechanical/viscoelastic requests recording `max_total_ns`, or migrate (old records would need a fallback in `check_request`)? | Leave. Shared code must keep both shapes. | F1 |
| D3 | `--protocol modulus/relax`: write the report files at run time like tm/breaking (option A), or only unify the verdict logic (option B)? | A, if you accept analysis files appearing at run time. | F14 |
| D4 | Replace `structure.py`'s eight record builders with `asdict` (JSON identical today) and accept layout coupling for plain dataclasses? | Yes for these nine dataclasses. | F15 |
| D5 | Public-API prunes (0.1.0 alpha): `ModulusSchedule`/`TensileSchedule`; 16 rate-family re-exports in `__init__`; `chain_positions`; `make_barostat(pressures_bar=...)`; `AtomicStateReporter` to `CheckpointReporter(writeState=True)`; `stress.pressure_bar` | Independent of each other. `AtomicStateReporter` removes a user-visible artefact (`*.state.a/b.xml`, `*.state.which`), so decide it explicitly. | F16 |
| D6 | What should a zero or negative duration do? Today a negative one runs one step (`steps_for` floors at 1). | Refuse non-positive; first check nobody relies on `hold_ps=0`. | F5 |
| D7 | Accept one strictness for Young's recorded-ladder verification? | Only if no legacy run directories depend on the looser check. | U2 |
| D8 | After F4, tm's figure formats: keep png/pdf/svg only, or any alphanumeric extension like the shared writer? | Accept the shared rule. | F4 |

## 4. Follow-up checklist

Gates (section 8) must be green before and after every item. A diff of the builder digests (Appendix A) must be empty after every stage.

### Stage 0: safety net (no production change)

- [x] Add `tests/test_recorded_builders.py` from Appendix A: 26 digests of builder stage names plus fully filled-in options, and of four request dicts. Baseline recorded at HEAD and identical on a second run.
- [x] Pin the whole parser (type, choices, nargs, help), not only dest/default.
- [x] Add the resume refusal matrix (F1) for mechanical, viscoelastic, tg, tm and tensile x3: foreign protocol, changed request, record missing, changed inputs, state missing. Tg's "record missing" case fails until D1 is applied; mark it `xfail` or apply D1 in the same stage.
- [x] Add "a bad `figure_format` writes nothing" for every report writer, and key-set snapshots of `tm.json`, `breaking/elongation/yield.json`, `convergence.json` and the rate report JSON.
- [x] Add direct tests for `_files.write_report/write_json/json_value/write_atomically`, `_validation.require_axis/require_plane` and `mdsystem.ensemble_controls`.
- [x] Add a CSV write-then-read round trip (`reporters.CSV_COLUMNS` vs `timeseries.CSV_FIELDS`; today's guard compares the length only).
- [x] Add golden `ChainResult` floats for one or two seeds (guards F17 and any `chain.py` change).

### Stage 1: leaf helpers (risk L, about -190 lines)

Reuse or extract: named constants (F7); `mdsystem.require_no_ensemble_controls` (F11); the F12 items; `Ensemble.first_frame` (F6); the F9 pieces (centre x in `fit_line` first, then swap the four `np.polyfit` calls; `standard_error_from_moments`; shared clustering, decades, `rms`, `finite_or_none`; `_sse` in `tm._jump_fit`); the F18 bypasses.
Callers changed: about 25 call sites across `chain`, `packing`, `trajectory`, `simulate`, `stress`, `melt`, `tm`, `thermal_rates`, `conformation`, `timeseries`, `relaxation`, `rate_dependence`, `elastic_rates`, `structural_convergence`, `convergence`, `benchmarks/pe_melt`, `forcefield`.
Removed: `set_pressure`, `_BlockEnsemble`, the four `polyfit` calls, the duplicate `_finite`/`_relative_change`/`_spacing_ps`, the local hash and sidecar code.

- [x] F7 constants
- [x] F11 ensemble-control refusal
- [x] F12 engine micro-duplicates
- [x] F6 `first_frame`
- [x] F9 numerics (centre x in `fit_line` before any `polyfit` swap)
- [x] F18 small bypasses (completed across Stages 1–3; exceptions recorded below)
- [x] F5 validation (separate behavior-fix commit; D6 resolved below)

### Stage 2: reports and shared readers (about -100 lines)

Reuse: `write_report_files`, `trajectory.load_manifest/stage_files/open_stage`, `_workflow.optional`.
Callers changed: `tm.write_melting_report` (plus a new `plots.plot_melting`), `tensile._write_report`, `convergence_report`, `rate_reports`, `melt_check`, `tm.heating_*`, `structural_convergence._measure`, the rate modules' manifest reads.
Removed: two bypass writers, five spellings of the `analysis` directory default, three of the four manifest parses in `analyse_convergence`.

- [x] F4 report writers (needs D8)
- [x] F10 manifest helpers and `optional`
- [x] F14 renderer verdicts (needs D3)
- [x] F15 `asdict` records (needs D4)

### Stage 3: scan scaffolding (highest value, highest risk, about -150 lines)

Order matters. Reuse: `run_branches`, `equilibrate`, `run_branched_scan`, `check_request`, `spec_request`; add a `precheck` hook to `run_branched_scan`.
Callers changed: `elastic_rates`, `tensile_rates`, `tensile` (all three scans through `_run_scan`), `tg._approach`, `tm.run_tm_scan`, `mechanical`, `viscoelastic`; test helpers `helpers.fake_scan_dynamics` and `tests/test_tensile.py::_interrupt`, which must patch `_workflow.run_protocol`.
Removed: the `tensile._run_scan` body and most of `_check_resume`, two replica loops, `tm`'s inline pre-flight, the duplicated mechanical/viscoelastic driver bodies.

- [x] F2 `run_branches` in `elastic_rates` and `tensile_rates`
- [x] F1 shared pre-flight; tensile onto `run_branched_scan` (needs D1, D2)
- [x] F1 tg and tm onto the pre-flight
- [x] F3 mechanical/viscoelastic driver
- [x] F8 budget helper (optional; do it while F1/F3 touch the same functions)
- [x] F2 optional `prepare_rate_scan` reviewed; retained explicit preparation (see section 10)

### Stage 4: drift guards (add a test or one shared table; do not merge)

- [x] F13: `run_heat`/`run_production` options table shared by `simulate.py` and `protocols._stage_options`; stage file-name helpers; `tg` pricing function used by the CLI; one tensile record-schema builder; help text formatted from constants

### Stage 5: owner decisions on the public surface

- [ ] F16 (needs D5)
- [ ] F17: cross-check test; extract only if a third user appears

### Stage 6: tests

- [ ] T1, T2, T4, T5 first (about -135 lines)
- [ ] T3, T6-T8 only when touching those tests anyway; add the missing direct tests (Stage 0) before trimming any per-caller test matrix

## 5. Findings reference (confirmed)

Net lines are approximate and net of new helpers. Risk: L = bit-identical or output-only, M = touches resume or recorded values, H = changes scientific behaviour.

| # | Finding | Priority | Net lines | Risk |
|---|---|---|---|---|
| F1 | Five pre-flights decide whether a scan may resume; Tg skips one rule | HIGH | -45 to -75 | M |
| F2 | Replica loops and the prepare-once skeleton bypass `run_branches`/`equilibrate` | MED | -20 (-45 with helper) | L / M |
| F3 | `mechanical.py`/`viscoelastic.py` driver scaffold | MED | -30 to -40 | L-M |
| F4 | tm and tensile report writers bypass the shared writer | MED | -25 to -40 | L |
| F5 | Segment validation in one of seven builders; about 15 hand-rolled scalar checks | MED (correctness) | about -14 | L-M |
| F6 | `_BlockEnsemble.frames` re-implements `Ensemble.frames`, wrong for negative bounds | MED (latent bug) | -16 | L |
| F7 | Defaults and unit constants written many times | MED (drift) | about -10 | L |
| F8 | `max_total_ns` guard written eight times | LOW-MED | -8 to -10 | L |
| F9 | Fitting numerics written several ways | LOW-MED | about -45 | L-M |
| F10 | Manifest helpers bypassed; reader toolkit in four places | LOW-MED | -45 to -60 | L-M |
| F11 | Ensemble-control refusal worded twice | LOW | -9 | L |
| F12 | Engine micro-duplicates | LOW | about -50 | L |
| F13 | Writer/reader pairs that can drift | LOW-MED | about 0 | L-M |
| F14 | CLI and library renderers decide the same verdicts | MED, decision | -10 / -90 | L-M |
| F15 | `structure.py` record builders equal `asdict` | decision | -105 | L |
| F16 | Public-API prune candidates | decision | about -120 | L-M |
| F17 | Chain Rg and bond-length kernels in two places | LOW | about 0 | L / H |
| F18 | Small bypasses of existing helpers | LOW | about -55 | L |

### F1. Five pre-flights decide whether a scan may resume

- **Responsibility:** before any dynamics, compare this request with the saved `*_workflow.json`; refuse another protocol's manifest, a manifest with no recorded request, changed inputs and missing states; then record the request.
- **Where:**
  - `_workflow.run_branched_scan:454` (with `check_request:315`, `record_scan_request:418`, `scan_request:442`), used by `mechanical.run_modulus_scan:536` and `viscoelastic.run_relaxation_scan:595`.
  - `_workflow.resumable_record:554` (with `start_fingerprint:621`), used by `elastic_rates.py:336`, `thermal_rates.py:319`, `tensile_rates.py:197`.
  - Inline: `tg._approach:668-675`; `tm.run_tm_scan:760-794`; `tensile._run_scan:588-683` with `_read_record:507` and `_check_resume:522-585`.
- **Evidence:**
  - Refusal matrix read from the code. Foreign protocol: branched yes, rates and tg left to `run_protocol`, tm yes, tensile yes. Manifest without a record: branched yes, rates yes, **tg no**, tm yes, tensile yes. Missing state files: rates and tensile refuse, the rest repair through `run_protocol`. Prefix integrity: tensile only.
  - Appendix B1: delete a finished scan's workflow record and resume. Mechanical, viscoelastic and tensile raise; **tg resumes silently**.
  - Tensile's request equals `spec_request(spec, drop=("max_total_ns",), equilibration=[asdict(s) for s in settle.stages], **run_fingerprint(run))` (dict and JSON text identical for the three specs). `scan_request()` is not equal: it adds `chain_backbone`/`atoms_per_chain`/`expected_characteristic_ratio` and keeps `max_total_ns`.
  - `tensile.py:661-667` is `_workflow.equilibrate` minus a log line; `:670-682` is `run_branches`; `:539-541` is word-for-word `_workflow.py:481-484`.
- **Preserve:** every saved request and record byte-for-byte; error classes and pinned tokens ("different settings", "fresh directory", "Cannot resume: ... missing predecessors / before equilibration is complete / unexpected stages"); tensile's prefix-integrity checks (keep as a hook); `_read_record`'s "Cannot read" wrapping of a corrupt record; budget and wrong-spec `TypeError` stay before any write.
- **Do:** one pre-flight in `_workflow` (protocol-name refusal, `check_request`, "manifest but no record", `validate_run_inputs`, optional missing-state check) used by `run_branched_scan`, `tg._approach` and `tm.run_tm_scan`. Tensile delegates its whole body to `run_branched_scan(request=its own, metadata={replica_stages, steps_per_replica, timestep_fs}, precheck=...)`. Rate scans keep `resumable_record` but call the same primitives. Do not "fix" the budget-in-request difference here (D2).
- **Lines / risk:** tensile -35, tm -15, tg -3, helper +10 to +25. M: touches resume of every scan, and Tg becomes stricter (D1).
- **Tests:** Stage-0 digests; the refusal matrix; add tensile to `tests/test_workflow_resume.py::scan` (covers only mechanical and relaxation today); patch `_workflow.run_protocol` in `tests/test_tensile.py:152`; existing `test_tg.py:253`, `test_tm.py:538-593`, tensile resume tests.

### F2. Replica loops and the prepare-once skeleton bypass the shared helpers

- **Where:** loops at `elastic_rates.py:373-392`, `tensile_rates.py:260-268`, `tensile.py:670-682`; prepare-once sequence at `elastic_rates.py:336-372`, `tensile_rates.py:197-224`, `thermal_rates.py:317-355` (which uses `run_protocol` plus `settled_state` instead of `equilibrate`, `:340-350`).
- **Evidence:** `_workflow.run_branches` is `run_protocol(..., resume=True, state_in=..., **chains)` per protocol. The loops' `resume=resume or replica > 0` is equivalent because `record_scan_request(resume=False)` has already unlinked the manifests (`_workflow.py:434-436`). Elastic and tensile_rates sequences are identical apart from `error`, `verb`, `fingerprinted`, `timestep_fs`.
- **Preserve:** stage names, per-rate directories, per-rate seeds (`derive_seed`), `state_in=start`. Thermal's per-(rate, replica) directories are an on-disk layout and cannot use `run_branches`.
- **Do:** `run_branches` in elastic_rates and tensile_rates. Optional `prepare_rate_scan` for those two only; adding thermal needs knobs (`state_in`, no box, `fingerprinted`) and becomes flag soup.
- **Lines / risk:** -20 for the loops (L); -25 more for the helper (M, three resume flows).
- **Tests:** `test_elastic_rates.py:396,904`, `test_tensile_rates.py:162`. `helpers.fake_scan_dynamics` does `monkeypatch.setattr(module, "run_protocol", ...)`, which raises `AttributeError` once the module stops importing it.

### F3. `mechanical.py` and `viscoelastic.py` carry one scan-driver scaffold

- **Where:** `equilibration_protocol` (`mechanical.py:333` / `viscoelastic.py:434`; `tensile._equilibration:481` is the same one-liner), `mechanical_scan:425` / `relaxation_scan:476`, `_report_cost:456` / `:509`, `run_modulus_scan:490` / `run_relaxation_scan:546`.
- **Evidence:** `difflib` on the bodies: the equilibration and listing functions are identical; the two 26-line driver bodies differ in exactly four things (error class, verb `"deform"`/`"strain"`, the `analyse_*` call, the `_log_result` call); `_report_cost` shares its skeleton.
- **Preserve:** both public signatures; the saved request (`scan_request(...)` unchanged); each scan's `_branches`, cost text and advice; protocol names.
- **Do:** extend the descriptor pattern `tensile.py` already uses, or a thin driver around `run_branched_scan` taking `branches`, the cost text, `analyse` and `log`.
- **Lines / risk:** -30 to -40; L-M. Do it with F1 and F8, which edit the same functions.
- **Tests:** `tests/test_workflow_resume.py::scan` already parametrises both; `test_mechanical.py`, `test_viscoelastic.py`; `test_workflow_schedules.py` ("what a dry run quotes is what the scan runs").

### F4. tm and tensile report writers bypass the shared writer

- **Where:** bypass in `tm.py:810 write_melting_report` and `tensile.py:984 _write_report`. Shared path: `_workflow.py:517 write_report_files` -> `_files.py:34 write_report` (used by tg, mechanical, viscoelastic, structure). `convergence_report.py:255` and `rate_reports.py:58` call `write_report` but recompute the `analysis` directory (five spellings: `_workflow.py:538`, `tensile.py:993`, `tm.py:827`, `convergence_report.py:244`, `rate_reports.py:38`).
- **Evidence:** Appendix B3: with `figure_format="not-a-format"`, tensile raises after writing `analysis/breaking.json`; tm refuses first. Only four writers record `openmmpolymer`/`versions`. tm draws the only `Figure(...)` outside `plots.py` (dpi 100, no `bbox_inches="tight"`). `write_report` reproduces tensile's output byte-identically (JSON and PNG) on planted scans.
- **Preserve:** `tm.json` keys (`method`, `enthalpy_units`, `heating_rate_k_per_ns`, `temperature_k`, `resolved`) and `melting.<fmt>`; tensile stems `<name>_r<i>`; strict JSON; tm accepts only png/pdf/svg today (D8).
- **Do:** `write_report_files` with a `_figures` generator per writer; tm's figure becomes `plots.plot_melting`; one `analysis_directory()` helper in `_files.py`.
- **Lines / risk:** -25 to -40; L (outputs only; JSON gains header keys).
- **Tests:** `test_tm.py:340-375`, `test_tensile.py:491-530`, `test_viscoelastic.py:343` (pins the key set including `versions`); key-set snapshots first.

### F5. Segment validation in one of seven builders; scalar validation hand-rolled about 15 times

- **Where:** `simulate.py`: `_heat_segments:1544-1545` is the only builder validating `hold_ps`/`pressure_bar`; none in `_compress_segments:1187`, `_anneal_segments:1253`, `run_nvt:1139`, `run_npt:1173`, `run_production:1636`; `_quench_segments:1413` checks endpoints only; `run_load:2166` and `run_shear:2300` use a bare `float()`. Hand-rolled elsewhere (no `require_nonnegative` or range helper exists): `strength.py:78,97,101,153,349-353`, `convergence.py:260-263,516`, `structural_convergence.py:536`, `rate_dependence.py:57,189,297,395`, `viscoelastic.py:211`, `simulate.py:2742-2749`, `tm.py:113,165`, `tensile.py:208`, `elasticity.py:599`.
- **Evidence:** Appendix B2: silently accepted: the seven calls listed in section 1. `run_nvt(duration_ps=-1)` runs one step because `steps_for` floors at 1. `run_shear(strains=[nan])` dies inside OpenMM. `run_heat(hold_ps=-1)` is refused.
- **Preserve:** no `run_*` parameter name or default; `_heat_segments` keeps option-named messages (`test_simulate.py:684-690`); check that nothing relies on `hold_ps=0` (D6). Pinned phrases: "finite and zero or more" (`test_simulate.py:1078`), "nonnegative" (`test_strength.py:153`); other tests match on the parameter name.
- **Do:** `Segment.__post_init__` (positive finite temperature and duration, finite pressure), reached by both the estimate and execution paths (`test_protocols.py:661-690` shows both build `Segment`s before touching `run`); `require_finite` in the two ladder comprehensions; `_validation.require_nonnegative` and `require_in_range` for the hand-rolled sites. No "empty sequence" helper (wording is partly pinned).
- **Lines / risk:** about -14 net (+6 checks, -20 migrated); L-M.
- **Tests:** `test_simulate.py:413,435-440,595-700,1078`, `test_protocols.py:661-690`; one new test per newly refused input.

### F6. `_BlockEnsemble.frames` re-implements `Ensemble.frames` and is wrong for negative bounds

- **Where:** `structural_convergence.py:165-195` (`_BlockEnsemble.frames`, `_block`) vs `trajectory.py:174-203` (`Ensemble.frames`).
- **Evidence:** with a defaulted `first_frame` field on `Ensemble`, the shared path matches on 300/300 non-negative calls; the copy is wrong on 10/10 negative cases (block (5,17), `start=-3` returns frames 2..16 instead of 14..16). No live caller passes negative bounds: a latent bug.
- **Preserve:** absolute `Frame.index`, `time_ps=(index+1)*interval_ps`, `is_snapshot`, `replace(ensemble, n_frames=...)` at `structural_convergence.py:571`.
- **Do:** `Ensemble.first_frame: int = 0`; `_block` becomes `replace(ensemble, n_frames=stop-start, first_frame=ensemble.first_frame+start)`; delete `_BlockEnsemble`.
- **Lines / risk:** -16; L (`Ensemble` is built by keyword).
- **Tests:** `test_frame_slices_keep_absolute_indices_and_times`, `test_disjoint_blocks_read_tail_frames_and_have_no_overlap`; parametrise the first over a block.

### F7. Defaults and unit constants spelled many times

- **Where:**
  - master seed `0xF0` x6: `chain.py:129`, `packing.py:263`, `simulate.py:172,183`, `tm.py:562`, `__main__.py:845`
  - characteristic ratio `7.0` x13: `chain.py:128`, `conformation.py:199`, `protocols.py:899`, `tg.py:928,1026`, `mechanical.py:498`, `viscoelastic.py:554`, `tensile.py:694,723,752`, `tensile_rates.py:160`, `elastic_rates.py:312`, `thermal_rates.py:247`
  - `"nagl"`/`"smirnoff"` x5: `charges.py:132`, `forcefield.py:130`, `melt.py:60-61`, `__main__.py:728,735`
  - `max_extrapolation_decades=2.0` x15 (plus `timeseries.MAX_EXTRAPOLATION_DECADES` and the CLI default at `__main__.py:918`); `strain_limit=0.015` x6; criterion defaults in both `tensile.py` and `strength.py`
  - units: `ANGSTROM_PER_NM` at `packing.py:46` and `trajectory.py:73` (whose comment is false); `_BAR_NM3_TO_KJ_MOL` (`simulate.py:81`) is `AVOGADRO*1e-25`; ps-to-ns `1000.0` at about 22-25 sites; `1.0e-12` at `mechanical.py:705` vs `NEGLIGIBLE`; `64*eps` at `strength.py:390` vs `_fitting.ROUNDING`; `MAX_REPLICA_SPREAD=0.3` in `mechanical.py:100` and `viscoelastic.py:97`; "edge >= 2x cutoff" `2.0` at `simulate.py:2020` and `mdsystem.py:173`
- **Evidence:** grep and read; `_BAR_NM3_TO_KJ_MOL == AVOGADRO*1e-25` bit-identical; none of these are `run_*` parameters, so `_stage_options` provenance is unaffected.
- **Preserve:** the values (a named constant with the same value changes nothing recorded); the benchmark's pins stay literal (independent harness); do not rename parameters; CLI defaults go through `vars(arguments)`.
- **Do:** `_seeds.DEFAULT_SEED`, `chain.DEFAULT_CHARACTERISTIC_RATIO`, `DEFAULT_CHARGE_METHOD`/`DEFAULT_BACKEND`, `PS_PER_NS` and `MAX_RELATIVE_STANDARD_ERROR` in `_fitting.py`, one `ANGSTROM_PER_NM`.
- **Lines / risk:** about 45 literals become about 12 names; roughly zero net lines; L for correctness, M for churn (about 12 modules).
- **Tests:** `RECORDED_REQUEST_SHA256`, `RECORDED_DEFAULTS_SHA256`, Stage-0 digests.

### F8. The `max_total_ns` guard is written eight times

- **Where:** `mechanical.py:482`, `viscoelastic.py:538`, `tg.py:595`, `tm.py:306`, `tensile.py:617`, `elastic_rates.py:253`, `tensile_rates.py:141`, `thermal_rates.py:195`.
- **Evidence:** eight wordings, five exception types, totals in ps or ns. Tests pin different fragments: `"max_total_ns"` (`test_tg.py:228`, `test_mechanical.py:193`, `test_tensile.py:557`, `test_tensile_rates.py:148`), `"budget"` (`test_elastic_rates.py:304,976`), `"all rates and replicas"` (the budget test in `test_thermal_rates.py`), `"over the"` (`test_viscoelastic.py:204`).
- **Preserve:** error classes, units, message fragments. The CLI's pre-build check (`__main__.py:1478-1485`) is separate and not redundant: it is the only pre-build guard for equilibrate, melt-quench, tg's fine pass and the tensile/modulus/relax scans.
- **Do:** `enforce_budget(total_ns, max_total_ns, *, error, message)` in `_workflow`, each caller supplying its own text.
- **Lines / risk:** about -10; L. The value is one definition of "None means unlimited, strict `>`, nanoseconds". One reviewer rated it marginal because of the test pins; do it only when F1/F3 touch these functions.

### F9. Fitting numerics written several ways

- **Where:**
  - `np.polyfit(x, y, 1)[0]` x4 beside `_fitting.fit_line`: `conformation.py:532,548,564`, `timeseries.py:990`
  - lstsq, residual, SSE: `_fitting.py:45-53`, `:89-94`, `tm.py:422-471` (x5 in `_jump_fit`)
  - standard error from running sums: `simulate.py:2863-2867` and `relaxation.py:420`
  - rate clustering at 1e-8: `rate_dependence.py:316-327` and `elastic_rates.py:900-909`; extrapolation decades: `rate_dependence.py:462-466` and `timeseries.py:927-936`; `_rms` imported privately by `convergence.py:34`; `structural_convergence._relative_change` vs `convergence._relative_span`; `timeseries._spacing_ps` vs `_temperature_step`; `_finite` (2 definitions plus 4 inline copies in `strength.py:446-458`)
- **Evidence:** polyfit vs `fit_line` differ by 4.6e-15 to 1.1e-12 at three sites and 1.5e-8 at `_relative_drift` (where both are equally accurate against an exact reference); SE-from-moments identical to 0.0; clustering identical on 3000 random trials; decades equal to 4e-16; `_relative_span` equals `_relative_change` on 5006 cases. Caveat (Appendix B4): `fit_line` is less robust than `polyfit` for badly scaled x (offset 1e5, span 1e-6: 100 % wrong and silent vs 6e-5); `polyfit` warns on constant x, which is an error under this suite's warning filter.
- **Preserve:** the TINY guards at the call sites; `fit_line`'s return shape; the ddof=0 vs ddof=1 distinction (SE-from-moments is a population variance and stays apart from `_fitting.standard_error`); `_workflow.sample_spread` is not overflow-safe (inf at 1e200), so it cannot replace the replica scatter in `rate_dependence`.
- **Do:** centre x inside `fit_line` first, then swap the four calls; `standard_error_from_moments` in `_fitting.py`; shared clustering, decades and `rms`; `finite_or_none`; a local `_sse` in `tm._jump_fit`.
- **Lines / risk:** about -45 in total; the polyfit swap alone is about +2. L-M (centring changes the shared fit at 1e-16).
- **Tests:** `tests/test_fitting.py` plus a new offset case; `test_conformation.py`, `test_timeseries.py` drift tests; `test_relaxation.py`; one unit test for the moments helper.

### F10. Manifest helpers bypassed; `optional` hand-written; reader toolkit in four places

- **Where:** `melt_check.py:102-183`, `convergence_report.py:108-176`, `tm.py:313-387` (hand-rolled `heating_stages`/`heating_curve`, parallel to `timeseries.quench_stages`/`quench_curve`) vs `trajectory.load_manifest/stage_record/stages_holding/stage_files/open_stage`; `elasticity._gather/_ladder` private-imported by `relaxation.py:74`; `_workflow.optional` vs hand-written copies at `structural_convergence.py:256`, `melt_check.py:148-183`, `convergence_report.py:115-137`, `structure.py:350-390` (and five more outside the slice); rate modules re-read manifests 3-5 times (`elastic_rates.py:402`, `thermal_rates.py:375,514`, `tensile_rates.py:316,373,438`).
- **Evidence:** instrumented `RunManifest.load`: `analyse_convergence` parses `manifest.json` 4 times and `melt_equilibration` twice (#25 touched only `structure.py`); `optional` yields the same string character for character; `tm.heating_curve` loads the manifest twice when `stages is None`. `tg._figures` (`tg.py:1396-1409`) re-opens the manifest and the CSV the melt check already read (uncertain: `MeltEquilibration` does not carry the series).
- **Preserve:** `melt_check` turns a missing stage or CSV into notes (use `optional` around `stage_record`); `tm.heating_stages` keeps stages with malformed samples visible so `heating_curve` can name them (`tm.py:331-334`); `quench_curve`'s per-stage refusal text; `match="No manifest"` tests.
- **Do:** one `load_manifest`, then `manifest=` through `select_stage/stage_files/open_stage/resolve_backbone`; `heating_stages = stages_holding(...)`; `optional` everywhere; move `_gather/_ladder` beside `stages_holding` under public names.
- **Lines / risk:** -45 to -60; L for melt_check and convergence_report, M for tm and the `_gather` move.
- **Tests:** `test_melt_check.py:96`, `test_timeseries.py:257`, `test_tm.py:253-335`, `test_structure.py:279-341`, `test_convergence_report.py:68`.

### F11. Ensemble-control refusal worded twice

- **Where:** `tm.py:664 _refuse_ensemble_controls` (callers `tm.py:601,744`) and `thermal_rates.py:203 _check_system`; identical text except "heating"/"thermal"; both call `mdsystem.ensemble_controls:670`.
- **Preserve:** error classes; tests pin only "barostat or Andersen thermostat" (`test_tm.py:455-459,522`, `test_thermal_rates.py:370`).
- **Do:** `mdsystem.require_no_ensemble_controls(system, error, stages=...)`; add one direct test of `ensemble_controls` (none exists).
- **Lines / risk:** -9; L.

### F12. Engine micro-duplicates

- `run_deform`'s inline stretch `simulate.py:1909-1920` equals `_strain_increment(mode="tensile", poisson=0.0)` `:2544-2582`: box vectors, positions and velocities differ by exactly 0.0 on a constrained triclinic cell. -6.
- `set_pressure:530` equals `set_pressures:542` with three equal entries: identical `setParameter` calls for all 3 barostat kinds x 5 pressures; private. -10.
- The four per-window `segment_*` lists declared at `simulate.py:896,1886,2193,2332`; `_Live.hold` appends to them: let `hold` use `setdefault`. -14.
- `stress.pressure_bar:86-110` equals trace(`pressure_tensor_bar`)/3 (difference 0.0). -6.
- `run_pushoff:1100-1112` re-lists ten `StageResult` fields: use `dataclasses.replace(last, ...)`. -5.
- `melt.py:148-161` JSON normalisation equals `protocols._canonical:594` (a strict superset; both refuse NaN). -3.
- State read, box edges and positions idioms at `simulate.py:490,1669,1684`, `tm.py:616-624`, `_workflow.py:102`, `protocols.py:1023-1027`. -7. Cell-size rule `minimum_box_factor * cutoff` at `mdsystem.py:329,352,800`: about 0.
- **Preserve:** bitwise results; key order in `StageResult.samples` (resume compares stage requests and digests, not samples); no `run_*` signature changes.
- **Risk / tests:** L. `test_simulate.py:898-912,960-975,1020-1051`, `test_stress.py:201,327,337`, `test_melt.py`.

### F13. Writer/reader pairs that can drift: add guards, do not merge

- `run_heat` forces `measure_enthalpy` and `run_production` derives `barostat` in `simulate.py:1601,1647`, and `protocols._stage_options:229-232` states both again. It is the fingerprint source, so risk M if touched: export one table or function both read, or add a test that `_stage_options` equals what the runner applies.
- `Stage.duration_ps` (`protocols.py:151-163`) re-derives four runners' totals (`simulate.py:1876,2186,2326,2787`). Units differ on purpose (ps vs rounded steps); parity is pinned by `test_protocols.py` (about lines 551 and 680).
- Stage file naming is written by `reporters.py:203-222` and `simulate.py:658-683,2833` and re-spelled by `trajectory.py:403-470`: expose `topology_path`/`trajectory_path` in `reporters.py`.
- `reporters.CSV_COLUMNS` vs `timeseries.CSV_FIELDS`: the only guard compares the length (`test_timeseries.py:41`); add a round-trip test.
- The CLI's `_tg_ps` (`__main__.py:218`) re-derives tg's price (equal to 0.0 ps on four configurations) and hard-codes "3 rates" for VFT where `timeseries._FORM_PARAMETERS` is the source: expose one pricing function in `tg.py`.
- The tensile workflow-record schema (`replica_stages`, `steps_per_replica`, `timestep_fs`, `start_state`, `reference_box_nm`) is written by both `tensile.py:638-645` and `tensile_rates.py:240-258`: one `_record(...)` builder owns it.
- Numbers restated in help text and docs (`--window-fractions` vs `DEFAULT_WINDOW_FRACTIONS`; "capped at 50 and 8 frames" vs `MAX_DISTRIBUTION_FRAMES`/`MAX_STRUCTURE_FACTOR_FRAMES`; the guide's 11-row rate table vs `RATE_PROPERTIES`): format the help strings from the constants (help text is not in `vars(arguments)` or the digest) and add a short test that the guide's rate table equals `RATE_PROPERTIES`.

### F14. CLI and library renderers decide the same verdicts (decision D3)

- **Where:** `mechanical._log_result:556`, `viscoelastic._log_result:615`, `tg.py:992`, `protocols.py:1044` vs `_cli_reports._modulus_lines:134`, `_consistency_line:166`, `_relaxation_lines:285`, `_run_tg` (`__main__.py:271`), `_print_chains:24`; verdicts re-decided in `plots.py` and the CLI (`_persistence_line` vs `plots._persistence_title`; the K/G gap list x4; SE and strain-rate wording).
- **Evidence:** with NaN gaps the log prints "gaps nan% and nan%" while the CLI guards `isfinite`; a persistence length of 0.0 prints "0.000 nm ... extrapolation" in the console but "no persistence length could be fitted" in the figure; `_modulus_lines` prints "+/- inf"; `--protocol modulus/relax` never prints notes or writes the report at run time, unlike tm/breaking/elongation/yield. Nothing pins the library log wording.
- **Preserve:** INFO/WARNING levels for library users; CLI wording pinned (`tests/test_cli.py:927 PRINTED`); per-medium wording (`_unresolved` differs by design).
- **Do:** put interpretation on the dataclasses (`PersistenceLength.regime`, `ElasticConsistency.gaps`) and keep the wording per medium; option A: modulus/relax use `job.report(...)` and the two `_log_result` shrink to one headline line.
- **Lines / risk:** -10 for the verdict half (fixes two visible output bugs); about -90 more with option A, which changes run-time CLI behaviour. L-M.
- **Tests:** `test_cli.py:721,927-1061`, `test_viscoelastic.py:298-352`; add 0.0/NaN persistence cases and a `--protocol relax` plateau-note test.

### F15. `structure.py` record builders equal `asdict` (decision D4)

- `structure.py:601-709` (eight `*_record` functions, 109 lines) and `:723-769` (14 `None if x is None else f(x)`). For all nine dataclass instances `json_value(asdict(x))` equals the hand-built dict key for key, in order, with identical JSON text; none has a computed property the record includes.
- The cost is the documented decoupling rule in `_workflow.write_report_files` (fields spelled out so the on-disk layout does not follow the dataclass layout). The repo is already mixed: `convergence_report`, `rate_reports` and `RunManifest.save` use `asdict`; `tg._transition_record` genuinely needs explicit fields.
- -105 lines; L. `test_structure.py:373-400` pins keys and would still hold.

### F16. Public-API prune candidates (decision D5)

- `ModulusSchedule` (`mechanical.py:224`) and `TensileSchedule` (`tensile.py:217`) add nothing to `StrainSchedule`; tests assert type identity (`test_workflow_schedules.py:127`, `test_tensile.py:717`). -10.
- 16 rate-family names re-exported from `__init__` beside the umbrella `run_property_rate_scan` (docs teach only the umbrella; tests import the family API from the family modules). -32; add `set(imported) == set(__all__)` to `test_package_metadata.py`.
- `chain_positions` (22 lines, two callers) and `boxes_nm`, plus the private `_load_chain_frames` imported by two modules: make the loader public. -22.
- `make_barostat(pressures_bar=...)` (`mdsystem.py:541`): no production caller (`run_load` uses `set_pressures`); its validation is also duplicated with `run_heat:1600`. -12 source, -25 tests.
- `AtomicStateReporter` (`reporters.py:106-160`): its own docstring says OpenMM >= 8.6.1's atomic `saveState` made the two-slot alternation redundant; the slot file is byte-identical to `CheckpointReporter(writeState=True)` output; nothing in the package or docs reads `*.state.a/b.xml`, `*.state.which` or `current_state`. -45 source, -25 tests.
- `stress.pressure_bar` is exported public API with no production caller (F12).

### F17. Chain Rg and bond-length kernels in two places

- `chain.py:918-924`, `:938-944` vs `protocols.py:514-524`. Same formula and operation order: 2e-16 and 0.0 relative difference on a random conformer, 3.8e-16 on 200.
- Routing `chain.py` through the protocols function is not bit-neutral (1-3 ulp on real grown chains), and `build/inputs.json` compares those floats exactly: do not re-route. A unit-agnostic numpy helper both import is bit-safe (1200 random conformers, 0 mismatches), but the net is about 0 lines.
- Do: add a test that the two routes agree to 1e-12, plus the golden `ChainResult` floats (Stage 0). Extract only if a third user appears.

### F18. Small bypasses of existing helpers (about -55 lines, all L)

- `pe_melt.py:290` hand-hashes (use `file_sha256`; identical digest); `pe_melt.py:250-254` reads the manifest twice (`stage_files` plus `open_stage`).
- `forcefield.py:233-236` writes the cache sidecar by hand (`write_json`; only `sort_keys` differs; `cache_key` sorts itself).
- `tm.py:591` uses `PDBFile` instead of `packing.read_pdb`.
- `TensileSpec.__post_init__` (`tensile.py:130-149`) is `require_positive_fields`; criterion validation and defaults are repeated between the `tensile.py` specs and the `strength.py` functions.
- `thermal_rates.py:107-116` chunk rule equals `tg._chunks:334-350` (400 random ladders, 0 mismatches); `thermal_rates.py:186` slices the Tm preparation as `stages[:2]` (positional) where the Tg branch filters by kind; `_replica_protocol` timestep injection equals `tm.py:750-759`.
- Four property-name lookups raising `ValueError("Unknown ...")` (`require_choice` exists) and parallel dispatch tables (`property_rates._SPECS`, `tensile_rates._EVENTS`); `elastic_rates` `_KIND`/`_HOLD`/`_PATH_KEYS` plus `_expected_path` are four tables over the same keys.
- The request JSON round trip is spelled five ways with different `default=`/`allow_nan` flags (`_workflow.py:309`, `elastic_rates.py:294`, `thermal_rates.py:296`, `tensile_rates.py:185`, `tensile.py:625`); every recorded dict must stay byte-equal.
- The stage-to-filename stem sanitiser is two idioms at four sites (`tg.py:1391`, `mechanical.py:889`, `convergence_report.py:273`, `rate_reports.py:42`), and the `.replace` form lets a stage name containing `/` write a figure outside `analysis/`.
- `correlations.py:175-182` and `:263-270` repeat the heavy-atom load plus the "Dropping hydrogens left no atoms" refusal, and `_load_chain_frames` has no guard, so `chain_positions(..., heavy_atoms_only=True)` silently returns a zero-atom array (raise inside the loader, -4).
- The convergence figures and `plot_moduli`'s legend (`plots.py:1387-1503`, `:658-660`) bypass the module's own `_error_bars/_point/_measured/_guide/_legend` helpers (consistency, about -10).
- Four `try/except _REFUSED -> parser.error` stanzas in `main()` (`__main__.py:1406-1514`); the `--analyse` and run paths are unprotected, so a refused `--figure-format` there is a traceback (a 6-line context manager saves 4 lines).

## 6. Test-suite findings (all test-only; none touches an on-disk contract)

985 tests, 232 `parametrize` decorators, no dead helper. About -430 lines in all; T1, T2, T4 and T5 (about -135) have real maintenance value, T3 is another -70 at the cost of deeper nesting.

| # | Duplicate | Where | Canonical | Lines |
|---|---|---|---|---|
| T1 | Manifest writer hand-spelled about 11 times (three mechanisms in `helpers.py`) | `helpers.py:782`, `:282-291`, `:948-954`, `:1035`, `:1416`; `conftest.py:122-144`; `test_trajectory.py:29` (15 calls); `test_mechanical.py:372`, `test_viscoelastic.py:327`, `test_relaxation.py:61`, `test_cli.py:1107`, `test_timeseries.py:237` | `helpers.write_manifest(run_dir, stages, **extra)`, merging into an existing manifest. Keep it hand-written: it is the de-facto "legacy manifest stays readable" pin for about 150 tests, so do not route it through `RunManifest.save` | -75 |
| T2 | Planted load ladder x3 plus a dead `merge=` parameter | `test_elasticity.py:243-255`, `test_mechanical.py:394-404`, `test_elastic_rates.py:97-120`; `helpers.py:740,747` | `helpers.write_load`; delete `merge=`; `densities_g_cm3=None` on `write_bulk`; write `load_axis` as `[float(axis)]` | -38 |
| T3 | Read-mutate-write JSON idiom, 39 sites in 14 files | `test_elastic_rates` (13), `test_tensile_rates` (6), `test_elasticity` (5), `test_structure` (4), ... | `helpers.edited_json(path)` context manager and `first_stage_samples` | -70 |
| T4 | AR(1) generator x3 (bit-identical) | `test_fitting.py:19`, `test_timeseries.py:145`, `:354-358` | `helpers.ar1(n, memory, *, seed)` | -12 |
| T5 | Planted tensile profile x3 | `helpers.py:1008`, `test_plots.py:187-193`, `test_plots_mechanical.py:40-44` | move `failure_curve`/`yield_curve` into `helpers`; keep expected numbers hand-stated | -10 |
| T6 | `run_protocol` stand-ins: 8 `pytest.fail` closures, about 7 raise-to-interrupt, 4 record-and-fake | `test_tensile_cli.py:158`, `test_cli_validation.py:162`, `test_tm.py:381,530`, `test_workflow_resume.py:205,237,292`, ... | plain helpers `forbidden(message)` and an interrupt helper (a conftest fixture would not help: the patch target differs per module) | -35 |
| T7 | Argon cell recipe x4; snapshot helper x3; ensemble-control tables x2; stand-in `RunContext` x2; `unexpected_plot` x2; `frames_once` x2; descending-quench entries (17 plus 9 calls) | `helpers.py:101-119`, `conftest.py:50-59`, `test_simulate.py:149,473`; `helpers.snapshot_files:59` vs `test_workflow_resume._files:83` (mtime matters); `test_tm.py:389` vs `test_thermal_rates.py:351-365` | `argon_cell()`, `snapshot_files(dir, *, mtimes=False)`, `ENSEMBLE_CONTROLS`, `quench_entry` (keep `write_quench` accepting ascending arrays) | about -120 |
| T8 | Parametrisation candidates | `test_correlations.py:139`/`:297`, `test_timeseries.py` refusal groups, `test_conformation.py` | only the first is clearly worth it (-9); ids change on merge | -30 |
| gap | No direct tests for `_files.write_report/write_json/json_value/write_atomically`, `_validation.require_axis/require_plane`, `mdsystem.ensemble_controls`; every caller re-proves their contract (bad axes enumerated per spec at `test_tensile.py:247-249`; `ModulusSpec` has none) | | add direct tests first, then trim per-caller matrices | adds tests |

## 7. Uncertain, and candidates that stay separate

### Uncertain: investigate before acting

| # | Candidate | What is known | What is left |
|---|---|---|---|
| U1 | Two Tg-vs-cooling-rate extrapolators: `timeseries.CoolingRateExtrapolation`/`cooling_rate_extrapolation` (log-linear, VFT, WLF; about 280 lines) vs `rate_dependence.RateExtrapolation` (log-linear, power law, error propagation) | Log-linear estimates identical (1982 random sets: 9e-13 K, 6e-13 K/decade, 4e-16 decades). Contracts differ: minimum rates 2 vs 3, duplicate rates rejected vs pooled, no uncertainty vs propagated SE, `predict` returns -inf/nan vs raises. tg and the CLI use the first, thermal_rates the second | The owner's intent for the two flows; adding VFT to `rate_dependence` is a design decision (risk H). Safe now: share the decades function and the `2.0` constant |
| U2 | Young's legacy recorded-ladder check (`elastic_rates.py:778-845`) vs the generic check (`:409-473`), plus `tensile_rates._validate_recorded_ladder` and `thermal_rates._check_completed` | Same walk at two strictness levels; 11 `allclose` comparisons with 6 tolerance pairs and about 25 message spellings; potentially -40 to -60 lines | Whether legacy run directories relying on the looser check exist (D7); risk H for legacy resume |
| U3 | Residual guard `convergence.py:540-553` vs `rate_dependence.py:425-432,503-513` | Constants shared, predicate copied | Same RMS? windowed vs all points, log vs linear scale, NaN and zero-error edges |
| U4 | JSON normalisation by round trip (6 sites) plus `protocols._canonical` and `_files.json_value` | Three policies (fail-closed, NaN to null, stringify) | Byte-equality of every recorded request before any unification |
| U5 | Bond-graph adjacency built in `packing.py:636`, `structure.py:264`, `tm.py:684` | About -6 lines | Traversals differ; further copies |
| U6 | `Stage.duration_ps` vs runner totals (F13) | One-line products; parity pinned | Only worth it if drift appears |
| U7 | ps-to-ns `1000.0` at about 22-25 sites | Grep only | A constant is cheap, but not every site was read |
| U8 | Test candidates: two near-identical 6000-row noise CSV builders; `QUICK`/`SCAN` specs shared by `test_workflow_resume.py`; three "writes nothing" tests moved to `snapshot_files` could expose a latent rewrite | | Needs a run |

### Stay separate (examined and rejected)

| Look-alike | Why it stays |
|---|---|
| `quench_temperatures` (`simulate.py:1347`) vs `heating_temperatures` (`:1502`) | Repeated subtraction vs `t_start + i*step`: bits differ for decimal steps in 314/1008 and 180/576 start/end/step combinations; the lists are recorded as `temperatures_k` options and compared on resume; tolerances differ and quench deliberately accepts offset ladders (`tg.py:553`) |
| `run_fingerprint` (`_workflow.py:56`) vs `_run_identity` (`protocols.py:617`) | Two different recorded values (workflow request vs manifest provenance), both compared on resume; unifying changes an on-disk format |
| `_canonical` vs `json_value` vs `spec_request`; `check_build_request` vs `check_request` | Different nonfinite policies and record shapes; all recorded |
| `run_deform`/`run_load`/`run_shear`/`run_relax` bodies; the seven thin wrappers `run_nvt` to `run_production` | Four to five identical statements per pair (AST count); the rest is the physics; wrappers have frozen signatures and a shared helper would hide which segments each passes |
| `CellList` vs MDAnalysis `self_capped_distance` | Identical pair sets, but `CellList` is incremental, float64 and non-periodic; a float32 borderline flip would change accepted trials and break `build/inputs.json` |
| `chain.py` rotation and dihedral helpers; its twin frame and torsion calls (`:667-669` vs `:799-804`, `:821-848`) | The only such code in the package, no scipy dependency; a prototype merging the twins was bit-identical on 12 builds but +10 lines net |
| `benchmarks/pe_melt.py` | Legitimately independent harness (runs standalone, pins published numbers and a digest of `polyethylene.json`); only the two bypasses in F18 are real; its smoke protocol has a different shape from `standard_melt_equilibration` and the recorded result embeds it |
| `deform_stages`, `load_stages`, `shear_stages`, `bulk_stages`, `relax_stages` | The AST "five identical copies" are one-line wrappers over the already shared `stages_holding(_ladder(key))`; each carries its own key and public docstring |
| `_mean_stress`, `_fitting.standard_error`, `_workflow.sample_spread` | Three different statistics on different shapes; direct reuse changes last bits, and `sample_spread` is not overflow-safe |
| `rate_dependence._regression` vs `fit_line` | Different statistic (anchored mean, so a flat response gives an exactly zero slope) |
| `simulate._LogBins` vs `relaxation._merge_bins`/`_standard_error` | Writer and reader of one record; the engine must not import analysis |
| `relax_bin_edges_ps` vs `prony_times_ps` | Identical arithmetic (180 settings, 0.0), different roles and guards; a 3-line formula |
| Strain bookkeeping in `run_deform` (iterative) vs `_workflow.strain_ladder`/`strain_after` (closed form) | 90/120 ladders differ in the last bits (3.7e-12); `simulate` cannot import `_workflow` (cycle); verifiers use `allclose` |
| tm and tg two-pass flows; the tensile `run_breaking/elongation/yield_scan` wrappers | Different structure; typed public API (`docs/guide.md:215-234`) |
| Per-dataclass report-record builders (tg, mechanical, viscoelastic) and the six `_figures` generators | Deliberate explicit schema; different content; a registry would be flag soup |
| `_unresolved` (`_cli_reports.py:19` vs `plots.py:1339`) and about 30 resolved/unresolved spellings | Wording per medium, test-pinned |
| Dependency versions recorded by three snippets (`pe_melt.py:177-181`, `__main__.py:1379,1567`, `protocols._versions`) | The CLI list is part of `build_request.json`; the benchmark must stay independent |
| `pyproject` / `environment.yml` / CI dependency lists; README and docs examples | Guarded by `test_package_metadata.py:41-72` and `test_readme.py` |
| Test oracles: `write_deformation`, `write_tensile_scan`, `deformation_rate_per_ns` (independent ladder), `planted_*`, closed-form generators, `RECORDED_*_SHA256`, `test_workflow_schedules.py` | An oracle that shares code with the code under test stops being one |

## 8. Gates and commands

Run the CPU environment's binaries directly and put its `bin` on `PATH` inline, or the packmol-marked tests fail or skip. The full suite takes about two minutes. GPU runs belong in the `openmmnqe` env, not here.

```bash
ENV=/home/louie/miniconda3/envs/openmmpolymer/bin
PATH=$ENV:$PATH pytest -q
PATH=$ENV:$PATH ruff check openmmpolymer tests benchmarks
PATH=$ENV:$PATH ruff format --check openmmpolymer tests benchmarks
PATH=$ENV:$PATH mypy
# builder and request digests must not move (Appendix A; run from the repo root)
PATH=$ENV:$PATH python golden_builders.py | diff - golden_builders.baseline.txt
```

Name explicit paths for ruff while any `.claude/worktrees` exist, or it scans the worktrees.

Per-stage gate: the full suite, the digests above, `RECORDED_DEFAULTS_SHA256`, `RECORDED_REQUEST_SHA256` and the refusal matrix all green and unchanged.

## 9. Coverage and limits

- Every production module was read in full by at least one reader (`tg.py` and `tm.py` apart from their import blocks), plus `docs/guide.md`, `README.md`, CI and packaging config, `benchmarks/`, `tests/helpers.py` and `tests/conftest.py`. About 27 test files were read in full and 19 more in part; `test_structural_convergence.py` was only scanned.
- The test suite was not run (the audit was read-only), so every "tests that pin it" statement comes from reading, not from a green run. No profiling was done.
- Re-run by the lead: the Tg resume gap (B1), tensile request equality and the `scan_request` mismatch, the budget in mechanical's request, `_BlockEnsemble` with negative bounds, the segment-builder acceptances (B2), tensile's half-written JSON (B3), `AtomicStateReporter` byte-identity, the `fit_line` large-offset failure (B4), Rg and bond-length agreement, the ladder arithmetic (B5), and the determinism of the digests. Taken from the reviewers' scripts without re-running: the `set_pressure`, `pressure_bar`, stretch, SE-from-moments, clustering, decades, `_relative_span`, chunk-rule and `CellList` equivalences.
- Line estimates are approximate and net of new helpers.

## 10. Implementation record for issue #40

Work is on local branch `codex/issue-40-consolidation`, starting from `9e22802`.
Each implementation commit references #40; nothing is pushed. The original findings
remain above so the changes and decisions can be reviewed against the audit.

### Compatibility decisions

- D2: preserve each existing request shape, including the differing treatment of
  budgets and chain options. No workflow-record migration is part of this work.
- D3: use shared interpretation properties for persistence regimes and finite
  elastic-consistency gaps. Keep runtime analysis-file generation unchanged;
  making scans automatically write reports is a separate product choice.
- D4: nested structure records use `asdict`, guarded by ordered schemas and a
  baseline full-JSON digest. Top-level report layouts remain explicit.
- D8: use the common filename-extension validation for all report writers.
  Tm gains dependency versions; tensile reports gain package/dependency headers.
- D5/D7: public API removals, checkpoint artifact removal, and stricter legacy
  Young's-ladder verification wait until the final review. Existing user data and
  external API use cannot be established from this checkout.
- All entries in “Stay separate” remain separate. Shared numerical primitives
  must not collapse different scientific or serialization contracts.

### Stages and verification

- Baseline: 1,942 tests passed, coverage 95.94%; ruff, formatting and mypy clean.
- Stage 0: all 26 audit digests match twice; complete parser schema, exact chain
  floats, seven-scan refusal matrix, report schemas, CSV round-trip, and helper
  contracts are pinned. Eight strict expected failures expose the two Tg
  preflight gaps and the three tensile writers with/without figures. Existing
  convergence/rate writer tests already pin their report key sets. F17 also
  cross-checks the chain kernels to 1e-12 without rerouting their arithmetic.
  Gate: 2,043 passed, eight expected failures; ruff, formatting and mypy clean.

- Stage 1: shared constants preserve all recorded defaults; `Ensemble.first_frame`
  fixes block slicing for negative bounds; shared ensemble refusal and engine
  state/pressure/strain helpers preserve the existing scientific operations.
  Keep the explicit `run_segments` sample lists to preserve their key order.
  F18 file/hash/PDB/chunk/timestep/CLI-refusal bypasses are consolidated; remaining
  report and scan bypasses are tracked with their later stages. The public scalar
  `pressure_bar` keeps its isotropic support and delegates tensor-compatible cases.
  F9 centres/scales line fits and shares moments, clustering, RMS, finite-value and
  spacing helpers. Boundary regressions showed two audit equivalence claims were
  not bit-identical: rate extrapolation retains ratio-log versus log-difference
  arithmetic per caller; structural relative change retains `ptp(values)/scale`
  rather than `ptp(values/scale)`. This avoids changing scientific verdicts at
  exactly two decades or 10% relative change. U1/U3 numerical sharing stops at
  these primitives; the different fitted models and residual guards stay apart.
  Gate: 2,075 passed, eight expected failures; ruff, formatting and mypy clean.

- F5/D6: no repository caller depended on zero holds. Segment holds now require
  positive finite temperatures/durations and finite pressures before writing;
  load/shear ladders reject nonfinite entries. Dry-run duration estimates use the
  same checks. Shared scalar and tensile-criterion validation keeps intentional
  zero baseline/ramp/time-offset values and finite zero/negative pressures.
  Gate: 2,255 passed, eight expected failures; ruff, formatting and mypy clean.

- Stage 2: report writers share output-directory/format validation and safe
  figure stems. Six tensile format-validation expected failures are now passing.
  Tm gains dependency versions, tensile gains package/dependency headers; all
  measured report fields remain unchanged. Melting uses the common tight bounding
  box when saving its existing paired plot. Nested structure records preserve
  their ordered schema and full normalized JSON digest under `asdict`.
  Manifest snapshots flow through reader calls without global caching, preserving
  permissive legacy elastic reads and malformed heating candidates. Persistence
  regimes and finite consistency gaps now drive console/figure/log verdicts;
  unavailable fit errors display as unknown. Runtime report-file generation is
  unchanged (D3 option B); adding automatic analysis reports remains deferred.
  F18 report directories, safe figure stems and plotting-helper bypasses are done.
  Gate: 2,302 passed, two expected Tg failures; ruff, formatting and mypy clean.

- Stage 3: shared preflight covers protocol/request/input checks, with tensile
  prefix and state checks retained as domain hooks. Tg now refuses foreign
  protocols and missing workflow records before writing (D1). Tensile still
  permits an empty manifest without a record; Tm still refuses foreign protocols
  even on a forced rerun. Mechanical/relaxation share their driver; tensile and
  the two mechanical rate families share branch execution. The first rate replica
  retains its original resume flag on forced reruns, despite the audit's proposed
  unconditional-resume equivalence. Budget checks share strict `>` semantics and
  lazy caller-specific messages. Tensile initial/final record field order is
  preserved by a shared schema builder, also used by its rate family (F13).
  All eleven rate requests match the captured 31,355 serialized bytes. Only
  their identical strict/stringifying normalization is shared. The audit's
  tensile `spec_request` equivalence does not hold for unsupported scientific
  scalar types: its original non-stringifying serializer remains, with a
  regression requiring refusal before writing. Missing-state checks preserve
  rate scans' short-circuit behavior. Property dispatch shares descriptors and
  choice validation without tightening legacy Young's-ladder verification.
  The optional `prepare_rate_scan` would need numerous policies and would change
  metadata-update ordering without simplifying the existing four helpers, so
  the explicit preparation sequence stays. Thermal keeps its directory loop
  and `run_protocol`/`settled_state` pair: using `equilibrate` would unnecessarily
  parse/log a reference cell. F18's remaining dispatch and compatible rate
  normalization bypasses are completed.
  Gate: 2,387 passed with no expected failures, coverage 96.10%; ruff,
  formatting and mypy clean.

- Stage 4: reporter and reader share trajectory/topology filename helpers,
  including suffix-bearing stems and existing snapshot fallback/format priority.
  Direct execution-versus-provenance tests guard heat's forced enthalpy and
  production's pressure-dependent barostat without merging their implementations.
  The CLI uses Tg's shared pricing function and the model's minimum-rate source;
  six pre-change prices match exactly. Help/criterion defaults come from their
  constants while the complete parser digest remains unchanged. The guide's
  eleven rate-property names/units and explicit package exports now have drift
  guards. CSV round-trip and duration parity were already covered in earlier
  stages. F10's five remaining prefix-only error-to-note loops use `optional`,
  and a mechanical report reuses its manifest snapshot. Distinct tuple-return,
  logging-only, exception-conversion and bespoke-suffix handlers stay separate.
  Tg figures still reread the volume CSV: carrying that series through
  `MeltEquilibration` would change the report model for a minor plotting shortcut.
  Gate: 2,409 passed, coverage 96.13%; ruff, formatting and mypy clean.

## Appendix A. Builder and request digest harness (Stage 0)

Save as `golden_builders.py` in the repo root and run from there. It prints one digest per builder or request. Written against `9e22802`; the 26 digests below were identical on two runs. If a builder's name or signature has changed since, adjust the call, not the expectation.

```python
"""Digest what every protocol builder and workflow request records.

Stage names seed the RNG, stage options are compared on resume, and the
`request` dicts in *_workflow.json are compared whole. test_protocols.py pins
each runner's defaults and test_cli.py pins the CLI namespaces; nothing pins
what the *builders* put on their stages or what each scan writes as its request.
Run before and after a refactor; the digests must not move.
"""

import hashlib, json, os, sys
from dataclasses import replace

os.environ.setdefault("OPENMM_CPU_THREADS", "1")
sys.path.insert(0, ".")

from openmmpolymer import mechanical, viscoelastic, tg, tm, tensile
from openmmpolymer import elastic_rates, thermal_rates
from openmmpolymer import _workflow
from openmmpolymer.protocols import (
    _canonical,
    _stage_options,
    standard_melt_equilibration,
    melt_quench,
)
from tests.helpers import argon_context


def digest(value):
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:16]


def stages(protocol):
    """Name, kind and the FULL recorded options (runner defaults filled in) of every stage."""
    return [(s.name, s.kind, _stage_options(s)) for s in protocol.stages]


out = {}
out["standard_melt_equilibration"] = digest(stages(standard_melt_equilibration()))
out["melt_quench"] = digest(stages(melt_quench()))

m, v, t = mechanical.ModulusSpec(), viscoelastic.RelaxationSpec(), tg.TgSpec()
out["mechanical_scan"] = digest(stages(mechanical.mechanical_scan(m)))
out["mechanical.deform_protocol r1"] = digest(
    stages(
        mechanical.deform_protocol(
            m, timestep_fs=2.0, replica=1, reference_box_nm=(3.0, 3.1, 3.2)
        )
    )
)
out["relaxation_scan"] = digest(stages(viscoelastic.relaxation_scan(v)))
out["relaxation_scan +linearity"] = digest(
    stages(viscoelastic.relaxation_scan(replace(v, linearity_strains=(0.01, 0.06))))
)
out["tg_coarse_scan"] = digest(stages(tg.tg_coarse_scan(t)))
out["tg_fine_scan"] = digest(
    stages(tg.tg_fine_scan(tg.nominal_fine_schedule(t), t, timestep_fs=2.0))
)
out["tm.melting_scan"] = digest(stages(tm.melting_scan(tm.TmSpec())))
for spec in (tensile.BreakingSpec(), tensile.ElongationSpec(), tensile.YieldSpec()):
    out[f"tensile_scan {type(spec).__name__}"] = digest(
        stages(tensile.tensile_scan(spec))
    )
    out[f"tensile_protocol {type(spec).__name__} r2"] = digest(
        stages(
            tensile.tensile_protocol(
                spec, timestep_fs=2.0, replica=2, reference_box_nm=(3.0, 3.0, 3.0)
            )
        )
    )

run = argon_context(64, 2.4)
holds = (50.0, 100.0, 200.0)
for prop in (
    "youngs_modulus",
    "poisson_ratio",
    "shear_modulus",
    "bulk_modulus",
    "load_modulus",
):
    plan = elastic_rates.validate_elastic_rate_scan(
        m, holds, property_name=prop, target_rate=0.01
    )
    out[f"elastic_rate_plan {prop}"] = digest(
        [stages(plan.equilibration)] + [[stages(p) for p in g] for g in plan.protocols]
    )
for prop, spec in (("glass_transition", t), ("melting_temperature", tm.TmSpec())):
    plan = thermal_rates.validate_thermal_rate_scan(
        spec, holds, property_name=prop, target_rate=0.01, n_replicas=2
    )
    out[f"thermal_rate_plan {prop}"] = digest(
        [
            stages(plan.preparation)
            if hasattr(plan, "preparation")
            else stages(plan.equilibration)
        ]
        + [stages(p) for p in plan.protocols]
    )

# Requests: drop the parts that depend on the OpenMM build (system xml hash etc.)
FINGERPRINT = {
    "system",
    "system_spec",
    "seed",
    "system_sha256",
    "coordinates_sha256",
    "box_nm",
}


def request_digest(request):
    return digest({k: v for k, v in request.items() if k not in FINGERPRINT})


chains = _workflow.chain_options(None, None, 7.0)
out["request mechanical"] = request_digest(
    _workflow.scan_request(run, m, mechanical.equilibration_protocol(m), **chains)
)
out["request viscoelastic"] = request_digest(
    _workflow.scan_request(run, v, viscoelastic.equilibration_protocol(v), **chains)
)
out["request tg"] = digest(_workflow.spec_request(t, tg_approx_k=None))
out["request tm"] = request_digest(
    _workflow.spec_request(
        tm.TmSpec(),
        drop=("max_total_ns",),
        **_workflow.run_fingerprint(run, spec_key="system_spec"),
        state_sha256=None,
    )
)
for k, val in sorted(out.items()):
    print(f"{val}  {k}")
```

Baseline (`golden_builders.baseline.txt`):

```text
9308b0c4cf7dc357  elastic_rate_plan bulk_modulus
691cd44028f5ff9d  elastic_rate_plan load_modulus
5f68e777ee009d1d  elastic_rate_plan poisson_ratio
7096f07711d8ef63  elastic_rate_plan shear_modulus
5f68e777ee009d1d  elastic_rate_plan youngs_modulus
ba5e1a8d6d39ab97  mechanical.deform_protocol r1
9308a799398744e8  mechanical_scan
71c740e121869fa9  melt_quench
7950cb1ec7ef20fb  relaxation_scan
73d0c3391e3f75e4  relaxation_scan +linearity
b23d209ba672543b  request mechanical
4e87700207f313a4  request tg
8d129746f33b9eaa  request tm
b791323344b0b56b  request viscoelastic
0f0aa1230f5b8744  standard_melt_equilibration
4a90b874340b79d1  tensile_protocol BreakingSpec r2
68f095a751e8e871  tensile_protocol ElongationSpec r2
cfffa477b4eb1b54  tensile_protocol YieldSpec r2
41bb7e24e5b7d663  tensile_scan BreakingSpec
b1648611c011e43d  tensile_scan ElongationSpec
9058ea3bae5fcf35  tensile_scan YieldSpec
ab806f52805ad1bf  tg_coarse_scan
eac763867bc77d7e  tg_fine_scan
95dd619e68775ac6  thermal_rate_plan glass_transition
cdcd436d5a4de410  thermal_rate_plan melting_temperature
fedb89b842b91c28  tm.melting_scan
```

When turning this into `tests/test_recorded_builders.py`, keep the same shape as `RECORDED_DEFAULTS_SHA256` in `tests/test_protocols.py`: a dict of expected digests and one assertion, with a comment that changing one refuses the resume of every run already on disk.

## Appendix B. Reproductions

Each was run at `9e22802` with `PATH` set as in section 8 and `OPENMM_CPU_THREADS=1`, from the repo root, in the CPU environment. They write only to temporary directories.

### B1. Tg resumes without its workflow record; the other scans refuse (F1)

```python
import logging, os, sys, tempfile

os.environ.setdefault("OPENMM_CPU_THREADS", "1")
sys.path.insert(0, os.getcwd())  # the repo root; the script changes directory below
from pathlib import Path
from tests.helpers import argon_context, QUICK_EQUILIBRATION
from openmmpolymer import mechanical, viscoelastic, tg, tensile

logging.disable(logging.CRITICAL)


def trial(label, runner, record_name):
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        runner()  # a complete scan
        Path("run", record_name).unlink()  # lose the workflow record, keep the manifest
        try:
            runner()  # resume=True is the default
            print(f"{label:<20} ACCEPTED (no refusal)")
        except Exception as error:
            print(f"{label:<20} {type(error).__name__}: {str(error)[:80]}")


ctx = argon_context(216, 2.8)

spec_m = mechanical.ModulusSpec(
    temperature_k=120.0,
    strain_increment=0.004,
    max_strain=0.012,
    relax_ps=0.3,
    elastic_strain_limit=0.012,
    n_replicas=1,
    samples_per_step=4,
    stage_ps=1.0,
    load_stresses_bar=None,
    bulk_pressures_bar=None,
    shear_strains=None,
)
trial(
    "mechanical",
    lambda: mechanical.run_modulus_scan(ctx, "run", spec=spec_m, **QUICK_EQUILIBRATION),
    mechanical.WORKFLOW_NAME,
)

spec_v = viscoelastic.RelaxationSpec(
    temperature_k=120.0,
    step_strain=0.04,
    baseline_ps=0.5,
    relax_ps=3.0,
    stage_ps=3.0,
    n_replicas=1,
    sample_every_ps=0.05,
    late_sample_every_ps=0.2,
    late_after_ps=1.0,
    bins_per_decade=8,
)
trial(
    "viscoelastic",
    lambda: viscoelastic.run_relaxation_scan(
        ctx, "run", spec=spec_v, **QUICK_EQUILIBRATION
    ),
    viscoelastic.WORKFLOW_NAME,
)

spec_g = tg.TgSpec(
    melt_temperature_k=150.0,
    t_floor_k=90.0,
    coarse_step_k=10.0,
    coarse_hold_ps=0.4,
    window_k=20.0,
    fine_step_k=5.0,
    fine_hold_ps=0.4,
    stage_ps=2.0,
    samples_per_segment=4,
    min_points_per_branch=2,
)
trial(
    "tg",
    lambda: tg.run_tg_scan(
        ctx, "run", spec=spec_g, tg_approx_k=120.0, **QUICK_EQUILIBRATION
    ),
    tg.WORKFLOW_NAME,
)

spec_t = tensile.BreakingSpec(
    temperature_k=120.0,
    strain_increment=0.01,
    max_strain=0.03,
    relax_ps=0.3,
    n_replicas=1,
    samples_per_step=4,
    stage_ps=1.0,
)
trial(
    "tensile (breaking)",
    lambda: tensile.run_breaking_scan(ctx, "run", spec=spec_t, **QUICK_EQUILIBRATION),
    tensile.BREAKING.workflow_name,
)
```

Observed: mechanical, viscoelastic and tensile raise ("... holds runs without a request in ...", "... stages without a matching ... workflow record"); tg prints `ACCEPTED (no refusal)`.

### B2. Segment builders accept negative and NaN inputs (F5)

```python
import math, os, sys, tempfile

os.environ.setdefault("OPENMM_CPU_THREADS", "1")
sys.path.insert(0, ".")
from tests.helpers import argon_context
from openmmpolymer import simulate as sim

run = argon_context(64, 2.4)
tmp = tempfile.mkdtemp()


def attempt(label, fn):
    try:
        res = fn()
        print(f"{label:46s} -> ACCEPTED (steps={res.steps})")
    except Exception as e:
        print(f"{label:46s} -> {type(e).__name__}: {str(e)[:70]}")


p = lambda name: os.path.join(tmp, name)
attempt(
    "run_nvt(duration_ps=-1)",
    lambda: sim.run_nvt(run, p("a"), temperature_k=100.0, duration_ps=-1.0),
)
attempt(
    "run_quench(hold_ps=-1)",
    lambda: sim.run_quench(
        run, p("c"), t_start=120.0, t_end=100.0, step_k=10.0, hold_ps=-1.0
    ),
)
attempt(
    "run_quench(pressure_bar=nan)",
    lambda: sim.run_quench(
        run,
        p("e"),
        t_start=120.0,
        t_end=100.0,
        step_k=10.0,
        hold_ps=0.1,
        pressure_bar=math.nan,
    ),
)
attempt(
    "run_heat(hold_ps=-1)",
    lambda: sim.run_heat(
        run, p("f"), t_start=100.0, t_end=120.0, step_k=10.0, hold_ps=-1.0
    ),
)
attempt(
    "run_anneal(window_ps=-1, hold_ps=-1)",
    lambda: sim.run_anneal(
        run,
        p("g"),
        t_low=100.0,
        t_high=120.0,
        n_cycles=1,
        ramp_windows=1,
        window_ps=-1.0,
        hold_ps=-1.0,
    ),
)
attempt(
    "run_compress(duration_ps_each=-1)",
    lambda: sim.run_compress(
        run,
        p("i"),
        temperature_k=100.0,
        pressures_bar=(1.0, 5.0),
        duration_ps_each=-1.0,
    ),
)
attempt(
    "run_compress(pressures_bar=(nan,))",
    lambda: sim.run_compress(
        run,
        p("j"),
        temperature_k=100.0,
        pressures_bar=(math.nan,),
        duration_ps_each=0.1,
    ),
)
attempt(
    "run_load(stresses_bar=[nan])",
    lambda: sim.run_load(
        run, p("k"), temperature_k=100.0, stresses_bar=[math.nan], duration_ps_each=0.1
    ),
)
loaded = sim.run_load(
    run, p("m"), temperature_k=100.0, stresses_bar=[math.nan], duration_ps_each=0.1
)
print(loaded.samples["segment_applied_stress_bar"])
```

Observed: every call is `ACCEPTED` except `run_heat` (refused with "hold_ps=-1.0 must be greater than zero"). The last line prints `[nan]`: `run_load` records the NaN as its applied stress.

### B3. A bad `figure_format`: tensile half-writes, tm refuses first (F4)

```python
import sys, tempfile

sys.path.insert(0, ".")
from pathlib import Path
from tests.helpers import (
    write_tensile_scan,
    PLANTED_TENSILE,
    write_heating,
    planted_curve,
)
from openmmpolymer.tensile import analyse_breaking, write_breaking_report
from openmmpolymer.tm import analyse_melting, write_melting_report

for label, build in (
    (
        "tensile",
        lambda d: (
            write_tensile_scan(d, PLANTED_TENSILE["breaking"]),
            analyse_breaking(d),
            write_breaking_report,
        )[1:],
    ),
    (
        "tm",
        lambda d: (
            write_heating(d, planted_curve()),
            analyse_melting(d),
            write_melting_report,
        )[1:],
    ),
):
    d = Path(tempfile.mkdtemp())
    report, writer = build(d)
    try:
        writer(report, figure_format="not-a-format")
        outcome = "no error"
    except Exception as error:
        outcome = f"{type(error).__name__}: {str(error)[:60]}"
    left = (
        sorted(p.name for p in (d / "analysis").glob("*"))
        if (d / "analysis").exists()
        else []
    )
    print(f"{label:8} -> {outcome}; files left in analysis/: {left}")
```

Observed: tensile raises a matplotlib `ValueError` and leaves `['breaking.json']`; tm raises `figure_format must be png, pdf or svg.` and leaves `[]`.

### B4. `fit_line` fails silently for badly scaled x where `polyfit` survives (F9)

```python
import warnings, numpy as np
from openmmpolymer._fitting import fit_line

rng = np.random.default_rng(3)
for offset, span in [(1e5, 100.0), (1e6, 0.2), (1e7, 1.0), (1e5, 1e-6)]:
    x = offset + np.linspace(0.0, span, 12)
    y = 3.0 * (x - offset) + 5.0 + 1e-3 * rng.normal(size=x.size)
    exact = np.polyfit(x - x.mean(), y, 1)[0]  # centred reference
    (slope, _), _ = fit_line(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = np.polyfit(x, y, 1)[0]
    print(
        f"offset {offset:8.0e} span {span:8.0e}: fit_line rel err {abs(slope - exact) / abs(exact):9.2e}   polyfit rel err {abs(pf - exact) / abs(exact):9.2e}"
    )
```

Observed on the last row (offset 1e5, span 1e-6): `fit_line` relative error 1.0, `polyfit` 5.7e-5.

### B5. The quench and heating ladders cannot share one arithmetic (section 2)

```python
import math, sys

sys.path.insert(0, ".")
from openmmpolymer.simulate import quench_temperatures


def indexed_down(t_start, t_end, step):
    out, i = [t_start], 1
    while True:
        t = t_start - i * step
        if t <= t_end + 1e-9:
            break
        out.append(t)
        i += 1
    out.append(t_end)
    return out


for start, end, step in [
    (650.0, 150.0, 25.0),
    (600.0, 200.0, 20.0),
    (450.0, 300.0, 0.1),
    (650.0, 150.0, 7.3),
    (500.0, 100.0, 0.7),
    (650.0, 600.0, 0.3),
]:
    a, b = quench_temperatures(start, end, step), indexed_down(start, end, step)
    diff = (
        max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        if len(a) == len(b)
        else math.nan
    )
    print(
        f"{start:>6} -> {end:<6} step {step:<5}: n={len(a)} identical={a == b}  max|diff|={diff:.3g}"
    )
```

Observed: identical for the 25 K and 20 K steps; different by 3e-11, 8e-13, 6e-12 and 8e-12 K for the 0.1, 7.3, 0.7 and 0.3 K steps.
