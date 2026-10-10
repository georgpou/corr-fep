# Reproduction protocol

## The authoritative implementation

The paper's [research scripts](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026) include:
- `run_tfep.py` — training and estimation CLI.
- `run_ff_calculations.py` — classical trajectories and alchemical setup; do not run for phase-one reproduction unless needed.
- `mdp/`, `p38/`, and `topol_shared/` — simulation inputs.
- `conda-env.yaml` — historical **linux-64-specific** environment. Do not install it unchanged on Linux ARM64.

The [tfep package](https://github.com/andrrizzi/tfep) is a separate dependency; its `pyproject.toml` currently lists PyTorch, MDAnalysis, NumPy <2, Lightning, pint, and h5py. The author script imports GROMACS, MiMiC, and tblite interfaces; the installed backends need to be validated independently.

## Stage 0: Environment and provenance

1. `python -m pip install -e ".[dev]"`; run `pytest` and `corr-fep doctor`.
2. Run GitHub Actions **Upstream layout smoke test** (clones only public source, no data).
3. Record exact SHA of both upstream repositories and versions/builds of Python, PyTorch, GROMACS, tblite, and tfep.
4. Verify unit conventions: reduced work, kJ/mol, Jacobian, and energy/force sign conventions.

## Stage 1: Obtain inputs, NOT new alchemical FEP

Study the data manifest at https://doi.org/10.5281/zenodo.21919704. Prefer selected files for two ligands, water and bound environments. Place them in `data/` or institutional scratch space; **do not commit them**. Confirm they contain the trajectories and reference energies required by `run_tfep.py`. The script expects per-repeat energies in `p38/calculations/lig_<id>/eq/<phase>/repeat-<n>/ener.xvg` (or `ener.edr` convertible using GROMACS). Other trajectory/topology paths must be confirmed by examining the full upstream script and archive.

The authors' README states their FF driver was adapted from pmx and requires their simulation directories. Initial reproduction targets the published endpoint corrections rather than rerunning `run_ff_calculations.py`.

## Stage 2: Original baseline

In the *upstream* directory and validated upstream environment, start with the paper's documented CLI, adapting ligand names and environment paths to your downloaded data:

```bash
python run_tfep.py train --ligand p38a_2g --phase protein --potential dft
python run_tfep.py estimate --ligand p38a_2g --phase protein
```

These are upstream documented examples, **not yet tested on our platform**. Ensure `--potential dft` maps to the intended configured target before using it as GFN1-xTB baseline. Prefer exact authors' commands from their reproducibility material after reviewing CLI argument definitions.

Baseline gate: reproduce one ligand in both phases, report delta G, work sigma, training budget, and uncertainty before changing a loss function.

## Stage 3: Controlled improvements

- Split data by independent MD repeat (never random adjacent frames).
- Keep reference and target Hamiltonians and mapped region identical.
- Compare original and modified losses using matched numbers of target QM/MM calls.
- Use held-out trajectories, several random seeds, stable log-sum-exp free energies, importance-weight ESS, and repeat/block bootstrap.
- For multimap estimators, retain original estimator bookkeeping: naively concatenating work from maps trained on their own samples is not an unbiased substitute for the published multimap construction.
- Evaluate larger mapped regions only after ligand-only baseline succeeds.

## Caveats

The repository's NumPy diagnostics are **diagnostics, not an implementation of multimap TFEP**; a synthetic numerical smoke test is not scientific reproduction. Importance-weight ESS omits MD autocorrelation and never proves target support coverage. No full data download is run automatically in CI.
