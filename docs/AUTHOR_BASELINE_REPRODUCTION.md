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
2. **Passed:** Rerun full 650-training/250-validation author baseline with unchanged numerical settings.
3. **Passed:** Compare results and provenance against the original successful run.
4. Retain PR #14 for traceability; do not merge any PR without explicit owner approval.

This PR does **not** claim exact reproduction of the paper's full multi-repeat multimap free-energy results. Neither the novel Corr-FEP hypothesis nor an ESS penalty is implemented here.

## Independent rerun on the clean branch — passed

[Clean GitHub Actions run #38180943734](https://github.com/georgpou/corr-fep/actions/runs/38180943734) completed successfully on the new branch in ~52 minutes. It completed all **21** updates on the **same 650 training frames in exactly the same shuffled order**, selected the same matched-water reference slots `[8299, 487, 6148]`, and evaluated all **250** validation frames.

| Diagnostic | Original run #38173132836 | Clean rerun #38180943734 |
|---|---:|---:|
| Identity reduced-work SD | 6.835677165012 | 6.835677165015 |
| Mapped reduced-work SD | 5.705048691078 | 5.705048499720 |
| Identity ESS / 250 | 1.116307030578 | 1.116307030570 |
| Mapped ESS / 250 | 1.951064263632 | 1.951064281095 |
| Mapped free-energy correction (kJ/mol; not converged) | -215367.048445 | -215367.048444 |

The largest difference in the 21 corresponding training-batch objectives was `6.85e-7` reduced units; the largest difference in gradient L2 norms was `4.21e-5`. The mapped free-energy estimates differ by only `1.91e-6` kJ/mol. Small platform-level numerical variation is expected. **These are equivalent physical results.**

The original and transplanted `stage_b_author_map.py` and `stage_b_author_full_baseline.py` were compared line-for-line: **only import lines changed** (2 and 3, respectively). Reused numerical helper function bodies were copied to small self-contained modules. The clean PR contains no abandoned environment-only, frozen-reference or one-layer experimental source files/workflows. The earlier five-checkpoint numerical replay remains recorded at [run #38179406749](https://github.com/georgpou/corr-fep/actions/runs/38179406749).

**Acceptance:** The clean implementation's reproducibility checkpoint is satisfied, and PR #15 is ready for code review as a partial author-pipeline reproduction. No novel Corr-FEP improvement or ESS-aware training is included; neither statistical convergence nor the paper's complete 25-repeat multimap results are claimed.
