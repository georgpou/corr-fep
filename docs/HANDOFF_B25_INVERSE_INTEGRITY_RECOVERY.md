# Corr-FEP Stage B2.5 — Scientific Integrity Recovery Handoff

**Date:** 2026-10-11 (Europe/Copenhagen). **Branch:** `stage-b/extended-cloud-benchmark`.  
**Repository:** https://github.com/georgpou/corr-fep  
**Draft PR:** https://github.com/georgpou/corr-fep/pull/14 (target `main`; do not merge without explicit user approval).

## Mission and operating rules

Continue **on the existing development branch** and restore a scientifically valid real-QM/MM MixedMAF training benchmark. Find the root cause of the trained flow's inverse/log-Jacobian closure error, implement a documented correction, and produce sufficient *fresh, independent numerical evidence* to recommend PR #14 for the user's review. Do **not** declare success based on passing CI or weakening a scientific gate.

The user permits careful fixes and bounded cloud experiments, but prioritizes scientific checkpoints over generic test coverage. Do not add extensive edge-case testing or infrastructure unrelated to the method. **Never merge without explicit approval for that merge.** Keep `main` unchanged. If a correction would change the Hamiltonian, target labels, data selection, model objective, or acceptance definition, document it as a methodological change, justify it scientifically, and seek the user's decision rather than silently changing the experiment. Do not retry failed multi-hour experiments blindly.

## Existing factual evidence

1. Stage B2.4, already on `main`: three AdamW steps with real QM/MM energies and MixedMAF. Tiny pilot (3 train frames, 2 test frames) reduced sample work variance 85.033861 to 77.306550; **not statistically meaningful**. Do not conflate this with the extended B2.5 study.
2. B2.5 benchmark at https://github.com/georgpou/corr-fep/actions/runs/38138876548 ran 2026-10-11 12:20–13:18 UTC (~58 min) and **failed** the scientific inverse gate after all **325 training steps in epoch 1**. The first 1,200 MM reference single points (650 train / 250 validation / 300 test), initial 300-frame test and initial 250-frame validation completed.
3. The full B2.5 artifact is downloadable at https://github.com/georgpou/corr-fep/actions/runs/38138876548/artifacts/11703033396 . Read `extended-cloud-benchmark/benchmark.json`, `frame-metrics.csv`, and `experiment.log`. The JSON has `validated: false`, `stage: incomplete`, `history: []` because epoch 1 failed before history/checkpoint update.
4. Baseline (untrained identity map) reduced-work sample variances: **46.72648230520545 validation (250)** and **53.35795513498891 test (300)**. Baseline inverse error **1.4210854715202004e-14 Å**, log-Jacobian closure **1.1368683772161603e-13**.
5. After 325 steps, `check_geometry` failed with **max inverse Cartesian error 0.0021159731524242886 Å**, **log-Jacobian closure 0.02828200723749319**, fixed-environment displacement **0.0 Å**, bonds **1.0291782161021685–1.5417886202799 Å**, max ligand displacement **0.1488533852700904 Å**. Current hard gates in `scripts/stage_b_extended_cloud_benchmark.py`: inverse error <= 1e-5 Å; Jacobian closure <= 1e-5; fixed environment <= 1e-10 Å; bonds in [0.6, 2.5] Å; displacement <= 2.0 Å. **Retain these gates while diagnosing**, unless independent numerical evidence establishes a different justified tolerance *before* evaluating improvements.
6. Full training uses 650 frames 401..1050, validation 250 frames 1151..1400, test 300 frames 1501..1800, with 100 omitted frames between blocks; only 32 train-era calibration frames choose spline ranges. One archived repeat and **quantized XTC** (original work used TRR). Chronological separation is not a guarantee of independence.
7. Original full run's `selected.pt` is **the initial identity checkpoint**, not a trained valid checkpoint. It cannot reconstruct the failed epoch. Do not claim the failure was from a specific subcomponent until the new diagnostic supplies evidence.
8. Prior infrastructure failures were fixed: first preflight scratch-directory reuse; second GROMACS 99-backup-file limit. The benchmark job sets `GMX_MAXBACKUP=-1`, which does not modify molecular physics. Do not mistake these resolved execution issues for the current scientific failure.

## New diagnostic: examine FIRST

The new one-off **train-only inverse-localization run** is https://github.com/georgpou/corr-fep/actions/runs/38148045379 (branch SHA when dispatched: `680a15dff8023e2b174ab9b31d0d1f1242ef98c0`). **Completed:** all 325 steps and both frozen checkpoints are archived. See the recovery update below before initiating any new computation. It does not run full baseline validation/test or claim a benchmark success.

Relevant files:
- `scripts/stage_b_extended_cloud_benchmark.py`: `--diagnose-inverse`, `detailed_inverse_probe`, step hook, and `inverse-diagnostic.json`.
- `.github/workflows/stage-b-inverse-diagnostic.yml`: 120-minute cap, 90-minute scientific soft cap, original training optimizer/data and pinned GROMACS.
- `.github/b25-diagnostics/run-request.txt`: **one-time push marker**. Avoid modifying the marker for automatic retries.
- `docs/STAGE_B_EXTENDED_CLOUD_BENCHMARK.md`: full benchmark protocol.

The diagnostic is designed to measure every 25 steps (and at step 325): (a) complete Cartesian inverse and closure, (b) mixed-coordinate MAF inverse and closure, (c) direct rational-quadratic spline inverse with the **true forward conditioner parameters**, (d) per-transformer width/height/slope ranges and largest errors. It saves `first-invalid.pt` when a checkpoint first violates the original gates and `last-epoch1.pt`. **Verify that the instrumentation itself works and that a checkpoint actually exists**; a CI-green or job-success label is not proof. Beware that a `first_failure_step` at a 25-step probe is the *first observed failure*, not necessarily the precise first bad optimization step.

Diagnostic interpretation:
- Direct spline inverse inaccurate but autoregressive error similar: likely numerically ill-conditioned spline/root formula/parameters. Audit tfep spline `neural_spline_transformer_inverse` and conditioning, particularly quadratic discriminant/cancellation, minimum bin heights/widths/slopes, periodic discontinuities.
- Direct spline inversion accurate but MAF inversion wrong: inspect autoregressive degree masks, inverse iteration/parameter conditioning, and periodic torsion branch behavior.
- Mixed-space inverse accurate but Cartesian inverse wrong: inspect `_CartesianToMixedFlow`, relative internal coordinates, reference frame reconstruction and collinearity/angle branches in pinned bgflow.
- Jacobian closure poor even where positional inversion is good: audit logdet calculation and mixed-coordinate Jacobian conversion, not just physical coordinates.
- Capture and inspect gradient norms/learning-rate/parameter evolution before proposing clipping or LR adjustments.

## Direct comparison to the *original research* (verified source)

Original paper code: https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026/blob/b8dfd897ebd914b4e592221710b397a2758353ce/run_tfep.py .  
Pinned TFEP library: https://github.com/andrrizzi/tfep/tree/0bc369a1194c688bb6ac997cc6b7c2db2b063c9e .  
Pinned bgflow: https://github.com/noegroup/bgflow/tree/fbba56fac3eb88f6825d2bd4f745ee75ae9715e1 .

In `run_tfep.py`, the paper's **production SQM/MM protocol** uses **six** MAF layers (line ~993), **batch size 32**, **one epoch**, **AdamW lr 1e-4 with default weight decay 0.01**, **ExponentialLR gamma 0.997 per optimizer step** (lines ~1075–1084), and Lightning **gradient_clip_val=3000** (line ~1231). It reads original **TRR trajectories**, excludes first 2 ns of each repeat, and preprocesses mapped/conditioning atoms and a stable origin/axes, including optional nearby water or protein atoms. The original code also has a 1-step *debug* configuration — do not confuse that with its production training. B2.5 intentionally differs: **one** MAF layer; batch **2**; up to **three** epochs; no LR scheduler; **zero** weight decay; **no** gradient clipping; 32-frame initial spline calibration; only the ligand mapped, no conditioning atoms, and XTC rather than TRR.

Crucial: these are **documented deviations and candidate contributors, not an experimentally proven root cause**. A one-layer ligand-only proof-of-concept can be scientifically legitimate but is not an exact reproduction. Do not simply switch to six layers or pretend a configuration change resolves an inverse error without rechecking reversibility. Original `tfep/nn/transformers/spline.py` uses 5-bin rational-quadratic splines with `min_bin_size=1e-4`, `min_slope=1e-4`, and a closed-form inverse `epsilon = 2*c / (-b - sqrt(b² - 4ac))`; inspect its conditioning under trained parameters. TFEP's own tests use `torch.allclose` on untrained/random transformers, which does not guarantee numerical stability after our sustained optimization. Our benchmark runs torch default float64 already.

The original setup uses GFN1-xTB in SQM/MM mode with mechanical embedding and an FF reference; the current project uses pinned GROMACS and tblite via validated real QM/MM integration. Do not replace real energies/gradients with mock data or change the FF/QM model merely to force a passing benchmark.

## Prioritized work plan

**1. Audit** the diagnostic run and confirm exactly which level fails, when, on which DOFs, and with what parameter conditioning. Compare with the original expected result. Reproduce using the archived model weights from the diagnostic; preserve raw logs and numerical error vectors.

**2. Make the smallest justified fix**, preferably with a **low-cost, no-QM offline inverse test** loading the trained `*.pt` flow weights and sample coordinates. If numerical spline inversion is failing, compare the existing analytic inverse against a stable mathematical reference on the *same frozen parameters* and report both Cartesian and Jacobian errors; establish that any correction preserves the intended invertible function and its exact Jacobian. If optimizer instability, examine gradient distribution first, then test original-paper safeguards (grad clip and scheduler, possibly weight decay) as explicitly documented ablations. All such changes must be fixed using training/validation only, not held-out test data.

**3. Scientific gates before new full benchmark:** deterministic forward/inverse and Jacobian closure after nontrivial trained steps on **multiple** train and validation configurations, finite forces and target energies, exact fixed-environment invariance, reasonable bond geometry/displacements, nonzero physical gradient, agreement between independently reconstructed checkpoint and original map, unchanged source potential/temperature/units. Preserve the original gate thresholds. Do not use test metrics to select hyperparameters or checkpoints.

**4. Only then** run a single bounded extended benchmark on the development branch; keep 650/250/300 partitions and pinned provenance, checkpoint by validation only, preserve diagnostics on failure, and produce final 300-frame held-out test only for a committed, gate-passing checkpoint. One run's work variance improvement is exploratory, not an unbiased free-energy/ESS claim.

**5. Before merge readiness**, independently audit full workflow artifacts, numerical CSV recomputation `w = (U_QMMM(T(x))-U_MM(x))/kT - log|J|`, partition indices, finite outputs, geometry/inversion on several frames, checkpoint reload, and uncertainty/trajectory autocorrelation limitations. Update README/protocol/PR description, explicitly state any deviations from the paper, and complete targeted review. Avoid a broad unrelated test suite.

**6. Stop at PR review**: mark PR #14 ready for user inspection only after objective scientific gates and actual artifacts support it. **Ask the user for separate explicit approval before any merge.** If no credible fix meets inverse closure, retain draft status and explain the scientific blocker and next controlled experiment.

## Important caution on leakage and interpretation

The current full script evaluates initial test work *before* model selection but records it as non-selection-only. Although this is not training leakage by itself, the **cleaner protocol** is to lock model selection first, then evaluate the identity baseline and selected model on the held-out test once at the end. Consider arranging that in a transparent follow-up without using prior test results to choose or optimize the model. The already observed 300-frame identity baseline is historical, and the prior run produced **no final trained-model test result**. Correlated samples, one repeat, and lossy XTC limit scientific inference.

## Existing hourly task

The user has an active hourly status automation initially tracking the failed full benchmark. Update its focus to the inverse diagnosis/new current run when appropriate, without allowing the automation itself to run compute or merge code. Do not create duplicate hourly tasks. Report actual observations, not estimates.

**Definition of done:** A scientifically justified and reproducible correction, concrete inverse/Jacobian evidence *after optimization*, fresh benchmark results from real pinned QM/MM, independent artifact audit, clear limitations, and a reviewable **unmerged** PR #14. No claim of success solely from a green GitHub Actions badge.



## Recovery update — 2026-10-11, after coordinate review

The original inverse diagnostic completed with exact reproduction of the
failed epoch and machine-precision mixed/direct-spline inverses. Frozen audit
[38149445011](https://github.com/georgpou/corr-fep/actions/runs/38149445011)
replayed both actual trained checkpoints exactly. Disabling only six reference
shifts restores positional inversion on six sampled train/validation frames
per checkpoint, but the true Cartesian logdet remains wrong (step-325
absolute discrepancy 0.01790556). This ablation must not be adopted as a fix.

The diagnostic fixed-solvent-reference proposal in run
[38150245773](https://github.com/georgpou/corr-fep/actions/runs/38150245773)
passes sampled inverse/geometry and independent derivative checks at identity
and noise amplitude 0.001; amplitude 0.03 fails the unchanged geometry gates,
so the overall run is a scientific failure. No 0.03 determinant was measured.
Both coordinate experiments used zero QM evaluations and zero physical updates.
The proposal has not been adopted into the benchmark.

See `docs/B25_COORDINATE_RECOVERY.md` for artifact identifiers, exact evidence,
review caveats and the concrete next action: a bounded three-update real-QM/MM
pilot using locked zero-based solvent references 4747/4748/4749, subject to
the user's methodological decision on conditioning atom selection. First
check reference continuity/geometry across train and validation without
validation-driven reselection. Keep original Hamiltonian, objective, data
partitions, calibration, optimizer and scientific gates. Test remains untouched
by these recovery experiments. No full rerun or merge is authorized by the
proposal. PR #14 remains draft; main is unchanged. The original hourly status
task tracks these completed results and the pending decision; it does not
execute computations or modify/merge code.


## User steering and source audit — 2026-10-11

The user requested building upon the original coding foundation and logging
suspected upstream bugs for discussion. Read `docs/UPSTREAM_IMPLEMENTATION_AUDIT.md`
next. It supersedes the exact single-water O/H/H pilot proposal: the released
code matches spatial water roles and chooses stable noncollinear conditioning
heavy atoms. No adoption or physical run has been approved. Two focused
bookkeeping checks support an ordering issue in reference-stability statistics
and a conditional candidate-loop bounds issue; permutation output restoration
has an additional static concern when solvent is mapped. Publication impact
is unknown. Preserve these classifications and discuss any correction before
porting it. The current driver and MixedMAF file contents match the inspected
pins. Keep our original physics, clean held-out split and scientific gates;
prepare the original preprocessing pattern with explicit attribution and
training-only fitting before revising the physical pilot.
