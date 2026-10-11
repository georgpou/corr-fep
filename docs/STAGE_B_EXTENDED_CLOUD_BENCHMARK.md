# B2.5 — bounded cloud integration benchmark

Implementation branch: `stage-b/extended-cloud-benchmark`.
Design handoff: [B2.5 bounded benchmark](https://github.com/georgpou/corr-fep/blob/stage-b/extended-cloud-benchmark-handoff/docs/HANDOFF_B25_BOUNDED_CLOUD_BENCHMARK.md).

**Preparation only.** This branch introduces a single, manual-only GitHub Actions
scientific integration experiment. Merging/dispatching is separate and requires
owner approval. The 210-minute workflow is not configured to run on push or PR.
One `ubuntu-24.04` runner, no matrix, no retries, unique concurrency group,
and a 165-minute internal molecular cutoff with failure artifact preservation.

## Model and evaluation

Reuses B2.4's pinned, original upstream `MixedMAFMap` (one MAF layer,
46-atom mapped ligand and fixed environment) and serial double-precision
GROMACS FF / GFN1-xTB QM/MM mechanical embedding. Archive `p38a_2g`,
water, repeat-0; no new alchemical MD.

- Training: 401–1050 (650 frames), shuffled once per epoch.
- Spline-range calibration: 32 uniformly spaced training-only indices.
- Training progress probe: 48 uniformly spaced training-only indices.
- Validation: 1151–1400 (250 frames), evaluated at epoch 0 and after every
  epoch; choose lowest **validation sample mapped-work variance** (including
  epoch 0). Only validation selects the checkpoint.
- Locked test: 1501–1800 (300 frames). Measure epoch-0 baseline and chosen
  checkpoint **once each**; test metrics do not inform model selection.
- Index gaps: 100 excluded frames between each block (~500 ps).
- AdamW 1e-4, weight decay 0, seed 20261011, batch 2, up to 3 epochs.

The apparent *initial* reference state is the freshly initialized, untrained
flow, not a separately evaluated mathematical identity map. This distinction
must be made in interpretation. Similarly, the on-the-fly training objective
is not a matched evaluation of a single fixed model over the full training
set; use the 48-frame fixed training probe for epochwise interpretation.

Per frame: `U_ref(x) = U_MM_full(x)`;
`U_target(y) = U_MM_full(y) - U_MM_ligand(y_L) + U_GFN1-xTB(y_L)`;
`w = (U_target(M(x))-U_ref(x))/kT - log|det J_M(x)|`.
Reference and mapped energies must be evaluated on **the same quantized
XTC frame inputs**. Never combine source XVG with quantized XTC.

## Scientific checks and deliverables

- Real trajectory alignment, finite/ordered 5-ps frame times and partition
  non-overlap; training-only initialization and dataset replay.
- Double precision, serial potential calls, unit check and explicit
  physical/Jacobian parameter-gradient decomposition with closure.
- Forward/inverse coordinate consistency and log-J closure on untrained,
  post-update and selected model; basic mapped ligand-bond geometry and
  fixed environment checks. Inverse closure is not an independent numerical
  determinant proof; that remains a publication-grade follow-up.
- Fresh upstream-flow reconstruction and strict portable checkpoint reload;
  training RNG, optimizer and model checkpoint saved for completed epochs.
- `benchmark.json` begins `validated=false` and is true only after final
  test completion. `frame-metrics.csv` includes reference, target, Jacobian,
  reduced work and geometry statistics. `experiment.log`, provenance,
  environment, timing, checkpoints are uploaded even on failure.
- Lag-one autocorrelation is a descriptive diagnostic, not an independent
  frame uncertainty interval. Future block-bootstrap/multi-repeat work
  belongs on research compute rather than additional cloud CI jobs.

If the time budget is exceeded mid-phase, `benchmark.json` stays unvalidated,
the job fails, and partial JSON/CSV/logs remain as evidence. No automatic
scientific accept/retry or test-set cherry-picking is permitted.

**Limits of scientific interpretation:** 650/250/300 archive frames are
chronologically correlated; the source compressed XTC is lossy, therefore
a correct re-evaluation of FF energies on those inputs does *not* establish
the original equilibrium sampling measure. This experiment cannot supply
unbiased ΔG, ESS, or a publication-level correction. One original
repeat and one MAF layer are not a replication of the full paper.

## Launch and independent review

First review PR changes and workflow resource caps. After owner-approved merge
onto `main`, manually dispatch `Stage B2.5 - Bounded cloud benchmark` once.
Evaluate the downloaded artifact independently by reconstructing all CSV works
and per-block sample variances, comparing frame splits, confirming model reload
and numerical finite values, and checking the actual execution SHA. A green
CI indicator alone is not evidence of scientific success.

B2.4's small two-frame held-out variance decrease (85.033861 to 77.306550)
is only an implementation pilot and not the baseline for this experiment.
