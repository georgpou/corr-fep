# Actual-trajectory water-matching invariance experiment — 2026-10-12

## Result

**Completed successfully:** [GitHub Actions run 38208740661](https://github.com/georgpou/corr-fep/actions/runs/38208740661), job 114677945750, commit `a7814a1c146f82313b700ed002232ba97f78fd35`. The run executed the pinned upstream ligand-aligned/Hungarian matching helper through Corr-FEP's existing small adapter, against archived `lig_p38a_2g` repeat-0 water trajectory. **No QM/MM energies or model training were performed.**

The hypothesis that the current water assignment is a function of unchanged environment coordinates only is **falsified by actual trajectory configurations**. With the environment positions fixed, moving the ligand changes the assignments, including at a ligand translation of 0.02 Å.

This observation is *not* a demonstration of numerical bias in the authors' reported free energies or the current Corr-FEP estimator. The global injectivity/surjectivity question remains unproven, and a model-specific or expanded-state proof might, in principle, justify some state-dependent schemes.

## Protocol

Script: [`scripts/stage_b_water_matching_invariance.py`](../scripts/stage_b_water_matching_invariance.py).

- Use the original archived `eq.tpr` and `traj_comp.xtc`, with frame indices `401, 409, 550, 725, 950, 1050, 1151, 1275, 1400` spanning the existing train/validation blocks.
- Use original frame 409 as the fixed reference for all comparisons.
- For each source frame, create nine configurations from the original physical positions: unmodified, ligand translations by +0.02/+0.10/+0.30/+0.75 Å along x, ligand translation +0.10 Å along y, ligand rigid rotations 5/15/30 degrees about its center around the z axis.
- Check that the physical environment coordinate array is **identical** to that of the source frame before passing the variant into the matcher. Keep box and topology the same. Original positional data comes from the quantized XTC; trial frames are held in float32 for MDAnalysis `MemoryReader`.
- Pass the resulting configurations through the same authors' `match_waters` implementation, with the adapter's source-preserving U1 ordering fix and supplied-memory-universe integration. No replacement matching algorithm or learned map was used.
- Compare each perturbed assignment to the **same frame's unmodified** assignment. Distinguish oxygen and hydrogen slots; separately check the three reference atom identities recorded by the successful six-layer baseline.
- The helper's required distance-probe atom argument uses one ligand atom; the distance output is not used in assignment comparisons. The earlier run [38208455799](https://github.com/georgpou/corr-fep/actions/runs/38208455799) failed before any matching result because an empty distance-probe atom list was rejected by MDAnalysis. The successful run uses a nonempty one-element index, without modifying the matching logic.
- The resulting detailed per-frame assignments, RMSDs and change counts were uploaded in [artifact 11728094348](https://github.com/georgpou/corr-fep/actions/runs/38208740661/artifacts/11728094348). Artifact retention is subject to GitHub Actions policies.

## Measured results

| Perturbation | Oxygen-changing frames / 9 | Changed O slots across frames | Changed H slots across frames | Frames with change to any of 3 baseline reference atoms |
|---|---:|---:|---:|---:|
| Original | 0 | 0 | 0 | 0 |
| x +0.02 Å | 1 | 2 | 4 | 0 |
| x +0.10 Å | 5 | 12 | 28 | 1 |
| x +0.30 Å | 7 | 34 | 94 | 3 |
| x +0.75 Å | 8 | 98 | 234 | 6 |
| y +0.10 Å | 7 | 14 | 34 | 2 |
| z rotation 5° | 8 | 76 | 174 | 3 |
| z rotation 15° | 9 | 199 | 428 | 5 |
| z rotation 30° | 9 | 298 | 612 | 9 |

The **original** control returns its own assignments identically, and the source matcher returns physically meaningful assignments for all 81 trials. The strong response to ligand-only rigid rotations is expected: the original matching helper aligns snapshots on the ligand scaffold and searches in a ligand-dependent solvent neighborhood.

The table reports assignment *indices*, not independent water transitions in equilibrium trajectories. Summed changed slots are comparisons of 9 configurations per trial, not separate molecules or statistically independent observations.

## Interpretation and next decision

Let `x` denote ligand coordinates, `e` the unchanged environment and `A(x,e)` the source matching rule. The experiment finds pairs `x != x'` with **identical `e`** and `A(x,e) != A(x',e)`. Therefore the presently used conditioning rule cannot be justified as `A(e)`, which would have made the ligand-only map triangular. The existing inverse check uses a frozen `A(x,e)`, so it cannot test that alternative.

A **single observed reassignment** suffices to refute universal assignment invariance. This is a stronger and much cheaper finding than waiting for a physical QM/MM convergence comparison, but it is **not** a proof of actual estimator bias.

Next: develop a conditional assignment based solely on a genuinely unmoved environment subset, or derive a mathematically correct estimator for dynamically selected branches. To keep this tractable, first attempt an environment-only rule with no QM, then check physical reference-frame geometry, independent forward/inverse reconstruction **without trajectory identifiers**, and ligand-block Jacobian on train and validation. Do not claim authors' numerical free energies are wrong. Do not launch another full 650/250 QM/MM training run until the corrected mathematical interpretation is verified.

Note: the archived historical environment-only anchor candidate had geometry failures, so this recommendation is not to switch blindly to that earlier implementation.

**PR #15's baseline remains unchanged.** The experimental script and results are confined to draft PR #16.
