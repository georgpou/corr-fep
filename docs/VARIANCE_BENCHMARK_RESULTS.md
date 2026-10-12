# Corr-FEP work-variance benchmark — first matched physical comparison

Date: 2026-10-12. This is a **single-seed validation diagnostic**, not a thermodynamically converged or independent-replicate result.

## Provenance and reproducibility

- [Completed physical cloud run 38277490360](https://github.com/georgpou/corr-fep/actions/runs/38277490360) (physical smoke; expanded control; expanded variance-aware model); all three jobs passed.
- [Completed compact analysis run 38302476494](https://github.com/georgpou/corr-fep/actions/runs/38302476494), [compact JSON artifact 11770962094](https://github.com/georgpou/corr-fep/actions/runs/38302476494/artifacts/11770962094). The analysis script reads the **actual** two physical `report.json` artifacts and rejects mismatched initial parameters, frame order, training exposures, validation frames, or Hamiltonian evaluation counts. It passed.
- Model: pinned six-layer author-inspired MixedMAF; p38a_2g water; GFN1-xTB ligand QM; subtractive mechanical embedding; ligand-only map; original reference matching/conditioning. AdamW, LR 1e-4, decay 0.01, gamma 0.997 per optimizer step, batch 32, float64.
- Each: 1450 unique training frames in two epochs, 92 AdamW updates, 2900 mapped QM/MM target evaluations and 1450 cached full-MM reference evaluations; frozen identity and mapped QM/MM validation on the identical 250 frames (250 evaluations each); 300 reserved-test frames excluded. Identical initial parameter SHA256 `94766553bd1b243438154689cdec5ac57062d1211ee4760ecfc8b231eb10d335`, training order and training batches across models.
- The physical smoke at nonzero regularization finished one real eight-frame backward/update, showed finite variance=31.321144, weighted penalty=0.313211, un-clipped gradient L2=1182.05 and inverse max error 8.60e-13 Å.

## Results on the frozen 250-frame validation set

| Statistic | Identity | Expanded control λ=0 | Expanded variance λ=0.01 |
|---|---:|---:|---:|
| Work SD (dimensionless) | 6.835677165 | 5.162085775 | 5.161809319 |
| Importance-weight ESS / 250 | 1.116307031 | 1.635498788 | 1.801947153 |
| Largest normalized weight | 0.946210064 | 0.773319698 | 0.733802887 |
| Largest five weight fraction | 0.985316875 | 0.951464578 | 0.939565570 |
| Diagnostic free-energy correction (kJ/mol) | -215353.869860 | -215376.975451 | -215376.460189 |
| Work minimum | -86921.966491 | -86931.090101 | -86930.829690 |
| Work median | -86904.184332 | -86916.244759 | -86916.351771 |
| Work maximum | -86887.039306 | -86903.135787 | -86902.994650 |

The variance-aware map reduces the validation work SD by **0.000276456 dimensionless units** (about **0.00536%**) relative to the matched control while raising ESS by **0.166448366** (**10.177% relative**). The largest-weight share drops by 3.952 percentage points. The work distributions are broadly similar: the 10th-percentile changes from -86923.09419 (control) to -86923.18522 (regularized), while the minimum increases by 0.26041. Paired per-frame variance-minus-control work differences have median -0.00162 and range [-0.72336, +0.67028]. Small changes in the low-work tail can materially change exponential weights without changing work SD appreciably.

The older 650-frame, one-epoch flow had mapped work SD 5.7050485 and ESS 1.9510643 on these same validation frames. The expanded control improves the work SD (5.705 → 5.162) but **not** the ESS (1.951 → 1.635), establishing that additional optimization does not uniformly improve all diagnostics. It is invalid to attribute the variance-aware improvement to longer training without comparison to the expanded control.

Recorded end-to-end script runtimes were **7552.4 s for control** and **10586.0 s for variance**, a difference of 40%. The training workloads had identical nominal physical evaluation counts; these wall times also reflect unequal runner/system overhead and cannot by themselves establish a 40% algorithmic overhead from the penalty.

## Scientific interpretation

The new loss **trains correctly** and shows a modest, favorable ESS change under an exactly matched QM/MM evaluation budget. However, work SD is essentially unchanged. This is **not** strong evidence for the proposed mechanism that lowering work variance improves statistical efficiency; the ESS change likely reflects modest reshaping of the extreme low-work tail. We cannot say this change is robust from one initialization or trajectory repeat, and neither ESS <2/250 provides a reliable equilibrium free-energy correction.

The two diagnostic free-energy corrections differ by 0.51526 kJ/mol, but this is not an accuracy claim: there is no independent converged ground truth or uncertainty estimate. Adjacent held-out temporal blocks share correlation with the same trajectory; they are not independent thermodynamic samples. The frame-conditioned solvent assignment's global invertibility is unresolved (issue #17), and the reserved test is not yet touched.

**Smallest justified next experiment:** repeat the *same* matched control/regularizer comparison for two additional independent initialization/shuffle seeds (three total), evaluating frozen models on exactly the same validation frames. Compare per-seed ESS and low-work tail, not just SD; do not tune λ on the reserved test or launch a broad sweep. Stronger thermodynamic conclusions later need independent trajectories and explicit estimator validation.
