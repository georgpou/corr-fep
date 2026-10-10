# Scientific checkpoints — keep this project lean

The first priority is a working published QM/MM endpoint correction, not exhaustive tests or a general-purpose framework.

## Checkpoint 1 — Working baseline (now)
- [x] Minimal Python project, numerical smoke tests, lightweight CI.
- [ ] Inspect selected files in the authors' Zenodo archive and choose **one ligand in one phase**.
- [ ] Pin upstream code revisions and set up its actual Python/TFEP/GROMACS/tblite dependencies.
- [ ] Run the author's original `run_tfep.py` training and estimation for that endpoint.
- [ ] Compare its estimated correction and generalized-work spread against the reported result.
- [ ] Record input file manifest, reproducible command, environment, output, and wall time.

**Scientific gate:** one reproduced result, not just successful imports.

## Checkpoint 2 — Controlled methodology change
- [ ] Make the smallest possible change to the original training objective.
- [ ] Train baseline and modified flows on the same configurations at matched QM/MM evaluation cost.
- [ ] Compare held-out free energies, work spread, and importance ESS.

**Scientific gate:** show numerical improvement without changing the Hamiltonian or obtaining incompatible free energies.

## Checkpoint 3 — Replication across systems
- [ ] Add water/protein for selected ligands.
- [ ] Compare independent random seeds and independent MD repeat blocks.
- [ ] Expand only after finding stable improvement on the difficult cases.

## Checkpoint 4 — Cluster migration
- [ ] Verify PyTorch CUDA on Linux ARM64/H200.
- [ ] Validate native GROMACS/tblite and cluster modules.
- [ ] Run the same baseline and improved commands under SLURM.

## Scope decisions
- Avoid full alchemical A→B reruns.
- Don't download the entire trajectory archive into GitHub Actions.
- Keep automated tests to essential numerical smoke checks; expand only when a real bug or scientific requirement justifies them.
- Do not spend the initial weeks on electrostatic embedding, extensive regularization grids, or elaborate CI.
- Document what **ran** separately from what is merely **planned**.
