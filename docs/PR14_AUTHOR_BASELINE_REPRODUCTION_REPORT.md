# PR #14 — author-method reproduction, checkpoint audit and deviations

## Decision and scope

This PR is a **baseline reproduction and research infrastructure PR**, not the proposed Corr-FEP statistical improvement. The authors' released source is authoritative for the software behavior, with explicit adaptations for the archived data and bounded validation. Do not claim exact publication reproduction, estimator unbiasedness or an improved loss function.

## Observed successful molecular runs

- Pinned TFEP MixedMAF with six autoregressive layers, original ligand-aligned Hungarian water matching, conditioning reference atoms and GFN1-xTB / subtractive mechanical QM/MM.
- [Three-update pilot](https://github.com/georgpou/corr-fep/actions/runs/38154651608): 96 training configurations; energy/gradient, inverse/Jacobian, checkpoint reload and sampled derivative checks passed.
- [Full baseline](https://github.com/georgpou/corr-fep/actions/runs/38173132836): one epoch, **650 training frames / 21 updates**, validation on **250 separate frames**, reference geometry valid, all inversion/environment checks passed, ~56-minute CPU workflow. **300 held-out test frames not evaluated in this full run.**
- Validation work standard deviation **6.836 identity / 5.705 mapped**, importance-weight ESS **1.116 / 1.951**. Frame 1158 has **94.6% / 69.5%** of normalized weight, respectively. These are poor-ESS diagnostics, not reliable converged free energies.
- [Checkpoint replay run](https://github.com/georgpou/corr-fep/actions/runs/38179406749) **completed successfully**. All five checkpoints at updates **5, 10, 15, 20, 21** strictly reloaded and passed 16-frame physical QM/MM mapped-work and numerical geometry evaluation. Original author preprocessing arrays were reproduced exactly. The final checkpoint's work and target energies matched the previous full baseline to **zero recorded error** on this subset.

## Differences from original published/released pipeline

| Dimension | Original authors | This PR |
|---|---|---|
| Core model | Six-layer MixedMAF from TFEP | Same pinned model |
| Objective | Mean target energy/kT minus Jacobian logdet | Same; **no ESS penalty/optimizer** |
| QM/MM | Subtractive FF(full)-FF(ligand)+GFN1-xTB(ligand) | Same intended mechanical embedding |
| Conditioning / water match | Preprocess matched waters, ligand-scaffold alignment and reference atoms | Same upstream helpers, except explicit narrow fixes below |
| Input trajectories | Up to 25 independent repeats with full-precision TRR, ~50,000 usable frames | **One deposited repeat** of compressed XTC, 650 train /250 validation frames, one 300-frame test reservation |
| Dataset input formats | Full archived production directory and original TFEP/Lightning source | Rebuilt physical topology/TPRs, narrow byte-range archive retrieval, Python training wrapper and GitHub Actions |
| Run settings | 6 layers, 32 batch, 1 epoch, AdamW 1e-4 with 0.01 decay, gamma .997, gradient clip 3000 | Match these settings in recovery and full baseline |
| Conditioning cutoff | Authors expose multiple options | Fixed 0 Å mapped water /3.5 Å water conditioning; only 3 stable reference atoms conditioned |
| Estimation | Published multimap online-work estimator over author repeat ensemble | Single frozen trained map, independent chronological validation; no exact multimap reproduction |
| Calibration | Original full-paper dataset calibration | 32 training-era sample subset |
| GROMACS | Publication reports double-precision 2021.6 | Pinned double-precision CPU 2023.3 |
| Experimental scientific change | Not applicable | **None**: ESS-aware loss belongs to subsequent PR |

## Specifically documented code adaptations

1. Upstream `match_waters()`: allow a supplied MDAnalysis Universe for the deposited XTC and frame partitioning; fix an oxygen-then-hydrogen versus O/H/H slot ordering error in the intermediate reference-stability calculation (U1), while retaining the original Hungarian matching.
2. Upstream reference selection: bound the number of candidate heavy atoms by the actual candidate count to avoid out-of-range combinations (U2).
3. Introduce `ConditioningContextFlow` to gather per-frame author water-match roles, run the original TFEP PartialFlow/MixedMAF and restore unchanged MM environment coordinates. The base author's `PermutationFlow` restoration can overwrite actively mapped waters (U3); our baseline **does not map solvent**, so we bypass that code path instead of claiming a full upstream correction.
4. Keep each per-frame water matching assignment unchanged during both forward and inverse operations. This makes the numerical inverse well defined for a fixed assignment, **but does not mathematically establish global bijectivity for a ligand-coordinate-dependent assignment**.
5. Create reproducible CPU/cloud workflows with double-precision GROMACS and pinned dependencies, physically consistent GROMACS/tblite coupling, train-only fitting of reference roles, per-frame test-isolated validation and checkpoint provenance.

## Exclusions / known limitations

- The environment-anchor water matching candidate failed the scientific geometry test and is **not used in the author baseline**; it is preserved only as research history on this draft branch. A later cleanup PR or PR scope trim can remove its workflow/scripts without affecting the working baseline.
- No proof of unbiased TFEP or paper-level publication reproduction; only one deposited MD repeat was available. The source and manuscript differ in how they describe selecting the relative frame; this implementation follows the source.
- The old one-layer moving-reference experimental benchmark failed trained inverse/Jacobian checks. It remains historical diagnostic code and must not be represented as the adopted training path.
- **Intermediate checkpoint diagnostics (same fixed 16 validation frames):** update 5 ESS **1.643**; update 10 **2.475**; update 15 **1.290**; update 20 **1.039**; update 21 **1.060**. The 16-frame sample is too small and weight-dominated to select a new checkpoint or claim a reproducible convergence trend. The 250-frame final-checkpoint ESS is **1.951**. We did not evaluate the other checkpoints on all 250 frames.
- Checkpoint replay verifies saved model consistency, but does not establish exact replication of the paper's published 25-repeat multimap free-energy estimate.

## PR acceptance / handoff

The bounded checkpoint-replay and numerical-integrity gates have now passed. The PR is **ready for human code review as baseline infrastructure**, not for claims of full published free-energy reproduction. Before merging, review its scope and remove or quarantine superseded experimental workflows if desired. Exact replication of the paper's 25-repeat multimap free energies is outside the scope and must be stated, not implied. No changes to the training loss or hypothesis should enter PR #14. **Do not merge without owner approval.**
