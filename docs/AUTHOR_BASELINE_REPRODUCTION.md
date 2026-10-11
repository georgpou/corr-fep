# Authors' QM/MM TFEP baseline — reproduction record

## Scope

This PR ports the **successful author-based six-layer MixedMAF baseline** from [original PR #14](https://github.com/georgpou/corr-fep/pull/14) into a clean branch from `main`, without the rejected one-layer, frozen-reference, or environment-only experimental paths. This is **not** a modified Corr-FEP model or ESS-aware loss.

## Reproduction provenance and method

- Tested training commit: [`efa2e84`](https://github.com/georgpou/corr-fep/commit/efa2e84bd59a953515b821374a4218e903326db3), [completed cloud run](https://github.com/georgpou/corr-fep/actions/runs/38173132836).
- Original tested author-style six-layer code was copied into `scripts/stage_b_author_map.py` and `scripts/stage_b_author_full_baseline.py`, with only imports changed.
- Successful `prepare`, `sample`, `reference_gate`, `load_driver`, `save_weights` and `batch_geometry` function bodies were extracted without changing their calculations into `scripts/stage_b_author_baseline_setup.py` and `scripts/stage_b_author_baseline_utils.py`. The original experimental modules containing those helpers are *not* imported.
- Checkpoint replay code was copied from its [verified legacy branch](https://github.com/georgpou/corr-fep/actions/runs/38179406749) with only import paths and an optional original-run identifier changed.
- The remaining GROMACS, trajectory preparation and physical-potential modules already exist in `main` unchanged.

The run uses the pinned upstream authors' TFEP MixedMAF (six MAF layers), original ligand-scaffold water matching/Hungarian assignment and reference-atom selection, GFN1-xTB subtractive mechanical embedding, AdamW learning rate 1e-4, decay 0.01, exponential gamma 0.997 per update, batch size 32, one epoch, gradient clipping 3000.

## Original author pipeline versus this adaptation

| Aspect | Authors | Reproduced baseline |
|---|---|---|
| Model and objective | Six-layer MixedMAF, mean QM/MM energy/kT minus log Jacobian | Same |
| Potential | GFN1-xTB/MM mechanical embedding | Same intended Hamiltonian |
| Source data | 25 repeats/~50,000 training frames used in publication | One publicly deposited compressed XTC repeat: 650 training frames |
| Validation / estimator | Online multimap estimator | Frozen trained map on 250 separate chronological validation frames |
| Calibration | Original production distribution | 32 training-era frames |
| Software | Authors' Lightning driver and GROMACS 2021.6 | Thin Python wrapper, CPU double-precision GROMACS 2023.3 |
| Water matching | Ligand alignment and environment-neighbor matching | Same algorithm; supplied Universe for XTC, corrected intermediate O/H ordering and bounded reference-candidate loop |
| Mapped atoms | Authors expose several cutoff configurations | Ligand only, mapped water cutoff 0 Å; environment reference cutoff 3.5 Å; three conditioning references |
| Water permutation | Original upstream `PermutationFlow` | Thin `ConditioningContextFlow` restoring fixed MM environment and reusing cached frame-wise matches |

The original code has been audited for water-slot ordering and other restricted edge cases; the adapter only changes pathways needed for the ligand-only author baseline. The cached frame-dependent water matching is **not proven globally bijective**, although its fixed-assignment inverse and Jacobian consistency checks pass. No direct estimator bias has been demonstrated.

## Previous verified results (original branch)

- 650/650 frames, 21 physical gradient updates.
- 250/250 validation configurations: unmapped work SD 6.836, mapped SD 5.705; unmapped ESS 1.116, mapped ESS 1.951. **ESS remains very low; these are not converged free energies.**
- Preprocessing reference geometry passed over 900 training/validation frames.
- Five optimizer checkpoints at updates 5, 10, 15, 20, 21 reloaded and re-evaluated successfully on 16 fixed validation frames in [checkpoint replay](https://github.com/georgpou/corr-fep/actions/runs/38179406749). Final checkpoint replay error was zero.

## Clean-branch acceptance

1. Compare clean branch files with their preserved original functions and verify no rejected experiments are included.
2. Rerun full 650-training/250-validation author baseline with unchanged numerical settings.
3. Compare resulting metrics and provenance against the original successful run.
4. Retain PR #14 for traceability; do not merge any PR without explicit owner approval.

This PR does **not** claim exact reproduction of the paper's full multi-repeat multimap free-energy results. Neither the novel Corr-FEP hypothesis nor an ESS penalty is implemented here.
