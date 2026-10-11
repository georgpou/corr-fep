# Stage B2.4 — first real MixedMAF optimization

This checkpoint runs actual AdamW updates of the pinned original one-layer
tfep.app.mixedmaf.MixedMAFMap on p38a_2g, water, repeat-0. It reuses the
B2.3 SequentialTrainingPotential and validated GROMACS double-precision
full-MM and ligand-MM plus tblite GFN1-xTB mechanical embedding. The
46-atom ligand is mapped; the environment is held fixed. No alchemical
trajectories are rerun.

For each archived XTC input x and flow output y=M(x),

    U_ref(x) = U_MM_full(x)
    U_target(y) = U_MM_full(y) - U_MM_ligand(y_L) + U_GFN1-xTB(y_L)
    w(x) = (U_target(M(x)) - U_ref(x))/kT - log|det J_M(x)|

Reference FF energies are recomputed at the actual quantized XTC inputs
and cached by selected frame. Never use source XVG/EDR energies here.

The flow is initialized with 12 early-time frames spanning archive indices
401..2000, excluding all held-out samples from spline-range calibration.
The training block is frames 401, 582, 764; held-out block
is 2218, 2400 (time-separated). Fixed RNG seed 20261011, three AdamW
steps at 1e-4 (weight decay 0), serial 2-frame batches (401/582,
582/764, 401/764). The held-out frames never enter the optimizer.

Run via .github/workflows/stage-b-first-flow-training.yml.
Outputs in results/stage-b/first-flow-training:
- first-flow-training.json: starts validated=false; pins, input hashes,
  provenance, energies, work statistics and all training gradients/steps
- frame-metrics.csv: same-frame before/after reference, mapped target,
  reduced work, displacement, log-J and fixed environment
- selected-inputs.npz: transformed input positions, boxes, frames and times
- flow-before.pt and flow-after.pt: PyTorch flow model checkpoints
- experiment.log and environment.txt: full numerical execution context

Acceptance requires observed parameter change, finite nonidentity mapping,
nonzero post-update log-J, unchanged water, and independent nonzero
real-energy and Jacobian gradient contributions with closure. Inspect
the JSON/CSV artifact, not merely workflow status. Report deterioration
just as faithfully as any improvement.

**Limitations:** This is a small temporally correlated sample of only one
archived repeat; held-out variance is only a two-point sample variance.
XTC is lossy and the true sampled distribution is not demonstrated to be
the same as re-evaluated FF on its quantized coordinates. It is neither
an unbiased free-energy estimate nor an ESS calculation, and it does not
reproduce the paper's six-layer solvent-conditioned, 25-repeat study.

## Measured clean temporal-holdout result (2026-10-11)

The corrected [Actions run 38131687881](https://github.com/georgpou/corr-fep/actions/runs/38131687881)
passed (code at commit `13daa645db18e2260b8acaa8b3e828709c7ebcde`).
The [full numerical artifact 11699552120](https://github.com/georgpou/corr-fep/actions/runs/38131687881/artifacts/11699552120)
has ZIP SHA256 `cb801a28dd53b8df47f01ee1428c37ceb86e149ac567cb87e1bac54f1e4e1d9b`.
An independent artifact audit confirmed a valid ZIP CRC, JSON
`validated=true`, *no overlap* between initialization/training and held-out
frame indices, recomputation of every stored reduced work from its own
reference and target energies/log-J, consistent ddof=1 sample variance,
strictly fixed environmental atoms and saved model parameter changes.

Train frames 401, 582, 764; held-out later frames 2218 and 2400.
All 12 initialization/spline-range frames are 401..2000, entirely
earlier than the held-out block. Original transformations and exact
per-frame cells were retained.

| Quantity | Before | After |
| --- | ---: | ---: |
| Training mean reduced work (3 frames) | -86908.883624 | -86910.913643 |
| Training sample work variance | 40.609532 | 39.501711 |
| Held-out mean reduced work (2 frames) | -86910.326416 | -86910.504122 |
| Held-out sample work variance | 85.033861 | 77.306550 |
| Maximum training ligand displacement (Å) | ~0 | 0.075418 |
| Maximum held-out ligand displacement (Å) | ~0 | 0.091444 |
| Maximum training absolute log-J | ~0 | 0.043730 |
| Maximum held-out absolute log-J | ~0 | 0.033948 |

Held-out work-variance delta was -7.727311 (about 9.1% smaller),
and held-out mean-work delta was -0.177706. The two held-out frames
**do not individually both improve**: frame 2218 work decreases from
-86903.805915 to -86904.286946, whereas frame 2400 increases from
-86916.846916 to -86916.721298. All five fixed-environment maximum
displacements were exactly zero in the recorded CSV.

Three actual AdamW steps changed the original model parameters by at most
0.000300482 (independently verified from `flow-before.pt` and
`flow-after.pt`); the corresponding physical-gradient L2 norms
were 695.994, 682.247 and 660.743, and Jacobian-gradient norms
14.932, 14.935 and 14.919. The physical and Jacobian contributions
sum to the total gradient within 5.7e-13. Post-training map and
log-J are clearly nonidentity. Total experiment time was 122.258 s
on a GitHub x86 CPU runner and peak Python RSS was 675.0 MiB
(excludes child GROMACS processes).

This **passes B2.4 feasibility**, not a statistically convincing
out-of-sample variance improvement. Two temporally correlated validation
frames, quantized XTC sampling-measure mismatch, one available repeat
instead of the study's 25, and a one-layer ligand-only flow preclude
publication-scale TFEP and unbiased free-energy claims.

