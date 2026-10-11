# Corr-FEP agent handoff — B2.5: bounded, production-style GitHub Actions benchmark

Date: 2026-10-11
Repository: https://github.com/georgpou/corr-fep
Preparation branch: `stage-b/extended-cloud-benchmark-handoff`
Scientific base: `main` at B2.4 squash merge `284da956f6387ef9e77b10f751cb0c9e6be5e3a7`.
Previous work: [PR #13](https://github.com/georgpou/corr-fep/pull/13), merged by owner authorization. Final PR head `c39cddff34ff91d1b9c3d3aecd179742e883ba84`; exact-head [successful real QM/MM run 38132439002](https://github.com/georgpou/corr-fep/actions/runs/38132439002), [artifact 11700270251](https://github.com/georgpou/corr-fep/actions/runs/38132439002/artifacts/11700270251). Main contains B2.4 code, workflow, and the report at `docs/STAGE_B_FIRST_FLOW_TRAINING.md`.

## Mission and scope

Implement, review, and conduct **one** deliberate, roughly three-hour, GitHub-hosted CPU scientific validation run of real upstream one-layer MixedMAF training against archived real QM/MM. Goal: determine whether increasing the number of distinct training frames yields lower mapped-work variance **on a time-separated held-out sample**, without claiming rigorous free energies or hiding deterioration. Do not start a broad parameter sweep, parallel compute farm, or multiple seeds on GitHub. Do not rerun original alchemical MD; use existing deposited trajectory for `p38a_2g`, water, repeat-0.

User explicitly requests responsible use of GitHub resources. GitHub Actions is principally for development and testing of repo software: keep this a **bounded integration/validation benchmark of Corr-FEP**, not free general-purpose HPC. Reference: https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features (Actions terms prohibit disproportionate burden and activities unrelated to production, testing, deployment, or publication of associated repo software). No one can guarantee an account will never be rate-limited/flagged. A single manual, scientifically justified test with caps, no automatic retry and no repetitive runs is the chosen conservative approach.

As before: work on a separate branch, create a PR, review/fix it, and **always ask owner approval before merging**. No need to burden the repo with a large testing framework.

## Established chemistry and implementation

The B2.4 implementation `scripts/stage_b_first_flow_training.py` reuses `SequentialTrainingPotential` in `scripts/stage_b_training_interface.py`, `MechanicalEmbedding`, original pinned `tfep.app.mixedmaf.MixedMAFMap`, proper ligand unwrap/center and environment wrap, original XTC TPR atom order, correct per-frame boxes, and inferred atom elements. GROMACS backend shares work directories, so **perform MM/QM calls serially**. No GPU assumption.

```
U_ref(x) = U_MM_full(x)
U_target(y) = U_MM_full(y) - U_MM_ligand(y_L) + U_GFN1-xTB(y_L)
w(x) = (U_target(M(x)) - U_ref(x)) / kT - log|det J_M(x)|
loss = mean(U_target(M(x))/kT - log|det J_M(x)|)
kT = R * 298 K = 2.477709860209666 kJ/mol
```

Positions Å; energy kJ/mol. **Recompute U_ref on the same quantized XTC inputs**, not the original high-precision `ener.xvg`/EDR. Cache reference values per input once; target varies with parameters. The deposited compressed XTC is a lossy coordinate representation, so re-evaluated reference energies do NOT establish that those inputs were sampled from the corresponding exact FF equilibrium measure. This invalidates a claim of unbiased importance reweighting / publication-level free-energy correction. Single deposited repeat also does not replicate paper's 25 repeats.

Keep one MAF layer and ligand-only mapping (46 ligand atoms, 8420 full atoms) for B2.5. Full paper had six layers and solvent conditioning; complexity should not grow before this data-size benchmark. Pinned versions, upstream source SHAs, and GROMACS double build are in B2.4 workflow; reuse rather than floating package upgrades. H200 ARM64 portability is later.

## Evidence-based runtime budget

Final exact-head B2.4 experiment `runtime_seconds=171.602930899` (includes preparation but not installation). Its measured phase costs on 4-vCPU standard Ubuntu runner were approximately:

- reference FF: **0.84 s/frame**;
- mapped QM/MM forward with reduced work: **2.16 s/frame**;
- QM/MM training forward+backward: **2.25 s per frame occurrence** (3 steps x batch-2 = 6 frame occurrences over ~13.48 s);
- setup, environment provisioning, trajectory download, etc. add variable overhead, potentially appreciable. CPU benchmark is **not** guaranteed constant under expanded batches/training.

Official GitHub runner: standard public Ubuntu 24.04 4 vCPU / 16 GB RAM / 14 GB SSD; hard maximum GitHub-hosted job duration six hours, but **deliberately stay far below that**. Sources:
https://docs.github.com/en/actions/reference/runners/github-hosted-runners
https://docs.github.com/en/actions/reference/limits

**Design for ~2.5–3 hours normal runtime** with an internal soft stop and no assumption that the upper bound will be achieved.

## Fixed recommended experiment (one job, one seed initially)

Approximately 2000 archived production frames remain after 2 ns equilibration; verify the actual trajectory length and 5-ps spacing before running. Suggested disjoint chronological blocks:

- **Training:** archive frames **401–1050 inclusive** (650 distinct).
- **Temporal gap:** 1051–1150 (100 frames; ~500 ps).
- **Validation:** 1151–1400 inclusive (250 distinct).
- **Temporal gap:** 1401–1500 (100 frames; ~500 ps).
- **Final test:** 1501–1800 inclusive (300 distinct).
- Remaining frames 1801–2400 are unused, not additional opportunistic test frames after seeing results.

This is a pragmatic baseline, not proof of statistical independence. Analyze autocorrelation at least for identity-map work and optionally representative ligand coordinates. The chronological gaps might be shorter than relaxation times. Document effective limitations.

- RNG seed `20261011` for comparability; consider alternative single seed if scientifically motivated, but **do not run a multi-seed sweep** on Actions. Save Python/NumPy/PyTorch RNG states in checkpoint if resuming.
- Initialize Z-matrix/spline ranges using **training-only** frames; never include validation or test in setup. Use a representative subset of training frames for range calibration (e.g. 24–48 uniformly dispersed training frames) rather than all 650 if setup scales poorly; record exactly which, confirm stability and finite mappings.
- Mini-batch 2 initially (measured speed), AdamW `lr=1e-4`, weight decay=0, 3 epochs, deterministic training-frame shuffle within each epoch, no multiple workers writing the same GROMACS directory. At 650 examples, approximately 325 optimizer steps/epoch, 975 total.
- Reference FF: calculate for all 1200 selected train/validation/test frames *once*, in serial. Prefer not calculating training reference for optimization because it is parameter independent; nevertheless needed for diagnostic training work and variance.
- Evaluation: initial **validation and test** before training; validation after each epoch; final test **once after checkpoint selection**. The final test must not influence learning rate, epoch count, checkpoint selection, or other hyperparameter decisions. For a fair comparison, test initial identity map vs the chosen checkpoint, using same frames and cached reference energies. Note that if stopping/selection is made from validation, test work is appropriately once-off.
- One optional tiny preflight **inside the same job**: short batch showing frame/box replay, finite target and gradients, fixed solvent, one reversible inverse/log-J probe. Do not launch a separate matrix of CI runs.

Estimated target calls (conservative assuming initial baseline and each evaluation):
Training 650*3 = **1950** target frame calls (~73 min).
Validation initial plus 3 epochs: 250*4 = **1000** target calls (~37 min).
Test initial and selected-final once: 300*2 = **600** target calls (~22 min).
Reference 1200*0.84 s = **~17 min**.
Sum roughly **150 min of molecular work** plus setup, I/O and checkpoints (plan ~165–190 min total, variable). Revisit this projection after in-run preflight and first 50–100 measured frame calls; adapt by **reducing epochs or sample counts** if needed, without changing held-out/test data based on results. If run cannot complete scientifically, save a partial `validated=false` result and report why; never mislabel partial benchmark as final.

**Hard engineering guardrails:** workflow only `workflow_dispatch`, not push/PR/schedule; one standard Ubuntu runner, no matrix and no parallel jobs; use `concurrency` unique to B2.5 to avoid overlapping runs with `cancel-in-progress: false`. Set workflow `timeout-minutes: 210` (3.5h including installation). In code track wall clock and enforce a **~165-minute scientific budget** with orderly checkpoint and exit before GitHub job termination; reserve ~15–30 minutes for cleanup/metrics/artifact upload. Avoid work after budget exhausted. If timing materially differs from forecast, stop earlier. Never automatically retry on timeout/failure, and do not trigger a run on every code push. Obtain user approval before manually dispatching the substantial benchmark if the agent's working mandate is ambiguous; user's stated objective authorizes preparation, but human oversight of expensive runs is preferable. Distinguish configuration of the workflow (cheap) from actual intentional dispatch (one longer compute run).

If validation/test dominate budget, a defensible fallback is 2 epochs with predefined full evaluation at start/end; avoid selectively omitting hard frames. For efficiency, evaluate validation every epoch but do **not** reevaluate the 300-frame test after each epoch. Early stopping and checkpoint selection must use validation exclusively.

## Scientific checks before launching full run

1. **Independent forward/inverse and logdet closure on an actual nonidentity trained flow** (B2.4 lacks independent post-update Jacobian/invertibility check). Use original upstream inverse; check coordinate error and forward + inverse log determinants within numerical tolerance. Check finite ligand bonds and no unreasonable short contacts. Avoid full 3N Cartesian determinant, which is prohibitive. Prefer targeted finite-difference directional derivative / small reduced-DOF check if practical; distinguish inverse closure from independent determinant validation.
2. **Fresh checkpoint reconstruction and load.** Restore model in new process with deterministic architecture and setup, recover same mapped positions and log-J for same input. Ensure optimizer and RNG states save/restore if resuming. Current B2.4 has matching state keys but no independent reload validation.
3. Make sure original frame transforms remain identical for all partitions, and boxes remain frame-specific. Fail on holdout calibration leakage or overlapping frame indices.
4. Check finite potentials/gradients, gradient components and decomposition at start and selected intervals, model parameter movement, no solvent movement, and large displacement/geometry anomalies.
5. Confirm output JSON starts `validated=false`; only set true after all specified scientific gates pass, including complete test evaluation and audit. Export CSV per-frame before/after reduced work, Jacobians, physical target energies, cached same-input reference energies; diagnostics should be algebraically reproducible.

Avoid expanding into a broad synthetic edge-case testing suite. Priority is one trustworthy molecular benchmark.

## Scientific output and interpretation

Produce:
- training and validation objective/work variance **vs epoch** (with time, counts, gradient norms); train/validation comparisons must use the actual objective/work for the same archived inputs;
- final locked test identity vs trained map: all 300 per-frame reduced works, mean/sample variance, percent variance reduction **or deterioration**, quantiles, finite values and log-J terms;
- geometry stats and inverse/log-J closure;
- chronological-block autocorrelation or conservative block-bootstrap uncertainty (clearly note limited independent repeats); no naïve independent-frame confidence intervals;
- run walltime per stage, per-frame throughput, memory, cache status, input SHA256 hashes, GROMACS/TFEP/tblite/torch pins, seed, optimizer and flow hyperparameters;
- saved chosen model, optimizer (if continuation intended), RNG state, selected index manifest, source SHA, and compact JSON/CSV logs. Upload artifacts **even on failure**; avoid massive caches and binaries. A larger XTC may be reused from existing download, but no redownload of the entire Zenodo ZIP.
- Explicit comparison to B2.4 tiny proof (3 train, 2 heldout, 3 optimizer steps, `85.033861 -> 77.306550` held-out two-point variance). Do not use the B2.4 tiny sample as a statistical benchmark.

**Success gate is not a required positive variance result.** Require completed real training, coherent independent physical gradients, invertibility/geometry within agreed tolerances, finite no-leakage held-out work and honest independent evaluation. If variance worsens, record the result and diagnose rather than alter test partitions post hoc. No ESS or unbiased ΔG claim from compressed-XTC source. A sensible benchmark may establish *failure of present architecture to generalize*, which is scientifically informative.

## Operational and review workflow

1. Create dedicated `stage-b/extended-cloud-benchmark` implementation branch from `main`; reference this handoff; leave handoff branch separate if desired. Reuse old code instead of rewriting whole molecular layer.
2. Add one configured, **manual-only** workflow with concurrency guard and 210-minute timeout; one bounded training script/config and a short scientific results report. No other automatic heavy CI on code push. The existing normal repository CI can continue.
3. Review source for physical/gradient correctness and preflight checkpoint reload/inverse consistency before substantial dispatch; ensure no invisible high-cost tasks.
4. Run the one intentional benchmark. Observe runtime budget and persist partial results on failure. Do not automatically rerun because a result is unfavorable.
5. Fetch and audit the resulting scientific artifact independently: recompute work and variances from per-frame CSV, check chronological split, checkpoint changes/restore, numerical finiteness, geometry, elapsed time, and actual run SHA. Distinguish verified from unverified gates.
6. Open a PR, review/fix, and obtain explicit user approval **before each merge**. If new full-run source changes after benchmark, revalidate affected scientific assertions with justified workload; do not burn several hours on indiscriminate repeats. A reproducible preflight and unchanged molecular source can help limit cost.

## GitHub resource stewardship

The standard public repo runner is free to use but **not an unrestricted HPC allocation**. Keep one run, low concurrency, fixed small dataset, bounded walltime and artifacts, no schedule, no matrix, no auto retries, no cryptomining, no background service, and no unrelated external workloads. No claim that any proposed cadence is guaranteed to be accepted by GitHub; usage policies remain authoritative. For larger multi-repeat / multi-seed / six-layer studies, transition to the user's ARM64 H200 research cluster.

## Final status at handoff

B2.4 was squash merged into `main` on 2026-10-11 after explicit authorization and a successful audit. **B2.5 is NOT yet implemented or run.** This document is a design/handoff on its own branch, not evidence of a completed benchmark. The next agent should perform the next stage and preserve complete scientific transparency.
