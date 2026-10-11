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

The flow is initialized with 12 frames uniformly spanning archive indices
401..2400. The training block is frames 401, 582, 764; held-out block
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

**Run status:** Implementation initially submitted, numerical acceptance
pending a real uploaded Actions artifact audit.
