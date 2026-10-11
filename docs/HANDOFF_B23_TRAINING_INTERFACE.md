# Corr-FEP agent handoff — Stage B2.3: training-compatible QM/MM backend

**Date:** 2026-10-11  
**Repository:** https://github.com/georgpou/corr-fep  
**Starting branch:** `stage-b/training-interface`  
**Base:** `main` at B2.2 squash-merge commit `679b44fe18e7a87f88df4ea6210602f6062022c8`  
**Completed:** [PR #11](https://github.com/georgpou/corr-fep/pull/11) was reviewed, fixed, numerically audited, and **merged with explicit user authorization**. All eight inline review threads resolved.

## User mandate and workflow rules

The immediate task is **B2.3: make the real published GROMACS + tblite mechanical-embedding potential training-compatible**. Next, B2.4 should run a small *actual* molecular normalizing-flow training smoke experiment, not another synthetic demo. Keep work lean: prioritize physical correctness and measured scientific checkpoints over general test scaffolding, defensive abstractions and speculative edge cases.

You are authorized to edit branches, commit, create PRs, run GitHub Actions, and conduct scientific/code reviews. **Ask for explicit user approval before each merge.** Do not merge without approval. Do not execute the original alchemical A→B simulations. Start with **one ligand, water, repeat-0**. Never claim full 25-repeat reproduction or publication-quality free-energy accuracy from this dataset.

## Why B2.3 is next

The prior checks demonstrated real molecular energy and gradient evaluations at isolated configurations. They have **not yet shown** a robust and efficient batch/sequential interface for a flow training loop, or optimized a molecular flow.

Published subtractive mechanical-embedding potential:

```text
U_QMMM(x) = U_MM_full(x) - U_MM_ligand(x_L) + U_GFN1-xTB_ligand(x_L)
```

The full reference potential is `U_FF(x) = U_MM_full(x)`. For the *same* input configuration, `U_QMMM(x)-U_FF(x)=U_GFN1-xTB(x_L)-U_MM_ligand(x_L)`, but in flow mapping `U_QMMM(M(x))` and `U_FF(x)` are **different configurations**. Do not falsely cancel their full-system MM terms when evaluating mapped TFEP work. Proper reduced map work: `w(x) = beta*(U_QMMM(M(x))-U_FF(x)) - log|det J_M(x)|` with a consistent coordinate/energy convention.

Critically, the deposited `ener.xvg` comes from original high-precision MD coordinates, while the only deposited full trajectories are **lossy XTC**. DO NOT naively mix `ener.xvg` with target evaluations on XTC/map coordinates. Evaluate `U_FF` consistently on the saved input frames (including their own boxes/periodic images). This defines internally consistent work on archived configurations, **not proof of unbiased sampling with respect to the original continuous-coordinate FF measure**. Identify consequences and preserve this qualification before free-energy inference.

## Accepted experiments and evidence

- **B1:** genuine GROMACS double-precision full FF and isolated-ligand FF plus tblite GFN1-xTB ligand, with combined force/gradient checking. See [docs/STAGE_B_MECHANICAL_EMBEDDING.md](STAGE_B_MECHANICAL_EMBEDDING.md).
- **B2.1:** real trajectory XTC, ligand transforms `unwrap → center_in_box → wrap(environment)`, four production frames (index 401/1067/1733/2400, times 2005/5335/8665/12000 ps), finite energies/gradients and selective central finite differences. See [docs/STAGE_B_MULTIFRAME.md](STAGE_B_MULTIFRAME.md).
- **B2.2:** [docs/STAGE_B_REFERENCE_ENERGY.md](STAGE_B_REFERENCE_ENERGY.md); the original deposited `ener.edr` and `ener.xvg` agree within ~5e-7 kJ/mol. Time-matched full-MM reruns on lossy XTC differ from original high-precision published potentials by +258.514/+303.235/+261.643/+317.729 kJ/mol; 95.4% of mean discrepancy is bond+angle. Archived and rebuilt TPR rerun potentials on identical inputs agree ~1e-8 kJ/mol. Original full-precision TRR is unavailable, original 2021.6 double versus pinned newer 2023.3 double.
- **B2.2b paired experiment:** new short native TRR/XTC controls prove coordinate quantization can give similar-magnitude biased energies, primarily in bonded terms, but do not reconstruct original coordinates. The final correctly *seeded* control uses `gen-seed=1729`, `ld-seed=271828`, mean +270.929644 kJ/mol over 10 noninitial frames. [Successful run](https://github.com/georgpou/corr-fep/actions/runs/38094626889).
- **B2.2c:** original EDR matched four saved XTC frames, numerical term accounting closes ~3e-11 kJ/mol; the underlying EDR components `Improper Dih.` versus `Per. Imp. Dih.` across engine versions are **bookkeeping** and not proof of equivalent force-field terms. Final audited [run](https://github.com/georgpou/corr-fep/actions/runs/38094626826) with preserved G96, original/rebuilt EDRs and individually checked SHA256 hashes.
- **B2.2a final:** [run](https://github.com/georgpou/corr-fep/actions/runs/38094626858).
- **Issue #9 remains open**: https://github.com/georgpou/corr-fep/issues/9. The special terminal XTC-vs-GRO ~1 kJ/mol difference is localized primarily to water oxygen 463, but exact encoding history unknown; do not confound with bulk 260–320 kJ/mol original-EDR-vs-XTC loss. The future choice of reference observable and sampling implications remains an explicit decision.

## Pinned inputs and code

- Research repo `andrrizzi/paper-alchemical-tfep-qmmm-2026` commit `b8dfd897ebd914b4e592221710b397a2758353ce`.
- TFEP `andrrizzi/tfep` commit `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e`.
- GROMACS double CPU v2023.3-dev commit `5e5ea27d8e594124d229a726616b7f6473f2c935`.
- Python 3.11, torch 2.4.0, NumPy 1.26.4, MDAnalysis 2.7.0, tblite 0.3.0, pyedr 0.8.0. GROMACS is installed/cached in GitHub Actions with an identifier bound to commit, precision and FFT library. An eventual ARM64/H200 migration is later; don't assume the original linux-64 Conda environment works on ARM.
- Zenodo https://doi.org/10.5281/zenodo.21919704, `lig_p38a_2g/eq/water/repeat-0/`. Full ZIP ~49.2 GB: use `remotezip` HTTP Range ZIP-member reads (never full download). Water trajectory `traj_comp.xtc` ~73 MB compressed ZIP member, with `eq.tpr`, `eqout.gro`, `ener.edr`, `ener.xvg`, plus ligand-specific `topology/` files. The archive contains only **repeat-0 for 32 ligands in two phases**; paper trained 25 repeats (50,000 production frames per system), so only ~2,000 post-2ns production frames are available per phase.
- Actual working scripts to reuse:
  - `scripts/stage_b_qm_singlepoint.py`: archive file retrieval and tblite single points.
  - `scripts/stage_b_mechanical_embedding.py`: published topology, GROMACS full/ligand TPRs, `MechanicalEmbedding` torch module, `grompp`.
  - `scripts/stage_b_multiframe.py`: XTC trajectory and validated transformations, production-frame selection.
  - `scripts/stage_b_reference_energy.py`: original reference/XVG matching and archived-vs-rebuilt MM reruns.
  - `scripts/stage_b_original_edr_decomposition.py`: original component accounting and provenance.
  - `scripts/stage_b_precision_roundtrip.py`: controlled lossy versus full-precision experiment.
- `docs/REPRODUCIBILITY.md` has upstream author's original `run_tfep.py preprocess/train/estimate` commands and **known path mismatches**: deposited `traj_comp.xtc` instead of expected `traj.trr`; missing 24 repeats; older expected `eq_nvt/water/eqnvtout.gro`; `topol_shared/ligand_only.top` lives within archive topology. Training `--potential sqm` invokes real GFN1-xTB mechanical-embedding scheme; `--potential dft` would incorrectly use CPMD/MiMiC.

## Proposed concrete B2.3 scientific checkpoints

1. **Read source before changing it.** Inspect actual upstream `run_tfep.py` map/preprocess/train integration and `GROMACSPotential`, `TBLitePotential` APIs. Preserve the exact published Hamiltonian, embedding and map definitions.
2. **Small real-data batch:** choose a handful of distinct deposited production frames, preserve actual atom identities and per-frame box, and demonstrate that a training-callable interface returns finite properly shaped `U_FF`, `U_QMMM`, gradients (especially w.r.t. mapped ligand Cartesian coordinates), plus correct units and differentiability across repeated calls. Verify both sequential and batch behavior or choose deliberate sequential fallback if direct batching isn't supported; quantify walltime/throughput.
3. **Coordinate consistency and map semantics:** input reference FF and mapped target evaluations must use the right frame/box and shared atom ordering; reversible mapping and log-Jacobian sign must be carefully checked. Avoid accidental cancellation of `U_MM_full(M(x))` versus `U_MM_full(x)`.
4. **Minimal independent check:** compare a batch element with the accepted B2.1 single-point/gradient and verify a small finite-difference gradient or derivative through an actual map; no sprawling edge-case suites.
5. **Document measured runtime, memory, real output energies/gradients, hashes, precise commands and accepted limits** in one scientific JSON/artifact plus concise `docs/STAGE_B_TRAINING_INTERFACE.md`. Use a dedicated GitHub Actions workflow or the repository's existing cached GROMACS CPU workflow. Prefer economical sampling (a few frames), not a 2,000-frame baseline initially.
6. **Then B2.4:** run a genuine limited optimization on water repeat-0 via the upstream TFEP architecture with a small subset, verify loss decreases and finite transforms/work/Jacobians, compare raw versus mapped work on held-out temporal blocks. Only after stable baseline, implement the new Corr-FEP training objective; never claim ΔG/ESS or reproduced original paper results from a pilot.

## Important open scientific question

**Do not mistake a successful optimizer step for a trustworthy free-energy estimate.** With lossy XTC, the reference MD sampling distribution is known only through rounded saved coordinates; coherent `U_FF(x)` on those positions is essential, but not automatically the exact sampled FF density. This limitation can be tolerated during model-development smoke tests, not silently ignored in final statistical conclusions. Only one repeat prevents independent-repeat splits.

## Working approach for the successor

Start on `stage-b/training-interface` based on the **merged main SHA** above. If additional PRs exist, inspect them before diverging. Follow the repository's scientific checkpoint style, audit the real data/artifacts, and open a focused PR when B2.3 passes. **Review and repair the PR; ask the user before every merge.** Do not make large framework reorganizations, run original alchemical FEP, or add dozens of defensive tests before obtaining the first real model-training result.
