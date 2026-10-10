# Scientific roadmap and acceptance gates

## M1: Reproducibility (weeks 1–4)
- [x] Minimal installable Python package, tests, CI, layout-check job.
- [ ] Pin upstream source SHAs and record environment provenance.
- [ ] Verify selected Zenodo files and extract 2 representative ligand systems.
- [ ] Reproduce original endpoint delta G and work-sigma values, including energy units.
- [ ] Measure wall time, CPU/GPU allocation, target energy/force evaluation counts.

**Go/no-go:** baseline agrees with reported values within reported uncertainty, with explainable deviations.

## M2: Training method (weeks 5–8)
- [ ] Implement optional ESS-aware objective and Jacobian regularization in the original PyTorch pipeline.
- [ ] Unit-test derivatives, invertible maps, and exact free-energy invariances on toy systems.
- [ ] Reproduce original loss as special case (regularization coefficients = 0).
- [ ] Compare equal-budget training on independent repeats and random seeds.

**Go/no-go:** improvement in independent importance ESS and/or uncertainty, without contradictory free energies.

## M3: Statistical validation (weeks 9–12)
- [ ] Freeze hyperparameters; independent held-out benchmark.
- [ ] Run block/trajectory bootstrap, convergence curves, outlier and tail diagnostics.
- [ ] Report negative results and computational budgets transparently.

## M4–M6: Extend only if validated
- [ ] Wider p38 benchmark.
- [ ] ARM64 and H200 environment validation on the target cluster.
- [ ] Electrostatic-embedding *pilot* only after energy/force and double-counting tests.

## Main risks
- A reduced Jacobian variance does not by itself imply unbiased finite-sample free energy.
- The original mapping training data can overfit; sample independence is crucial.
- H200 accelerates training, but many QM/MM evaluations may remain CPU-bound or poorly GPU-utilized.
- A changed Hamiltonian is a separate question from a better estimator.
