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

## Verified provenance (inspected October 10, 2026)

- Paper implementation: [`andrrizzi/paper-alchemical-tfep-qmmm-2026`](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026), commit `b8dfd897ebd914b4e592221710b397a2758353ce`.
- TFEP v0.2: tag `0.2` at commit `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e` (upstream environment exports `tfep==1+unknown`, **not** an exact source pin).
- Published environment: Python 3.11.10, PyTorch 2.4.0, Lightning 2.4.0, MDAnalysis 2.7.0, NumPy 1.26.4, tblite and tblite-python 0.3.0. The archived Conda builds are linux-64-specific; GROMACS 2021.6 is reported in the paper.
- Zenodo [10.5281/zenodo.21919704](https://zenodo.org/records/21919704) contains a **single** 49.2 GB `alchemical_tfep.zip`, described as *representative* MD trajectories, FEP and TFEP data. **The verified ZIP central directory has 9,093 entries and contains only `repeat-0` for every one of the 32 ligands in both water and protein (64 ligand/phase combinations). The remaining 24 repeats used by the paper are not archived.** A published 50,000-frame retraining cannot proceed from this archive alone. [Archive inventory run](https://github.com/georgpou/corr-fep/actions/runs/38062133720).
- To inspect ZIP members via HTTP byte ranges (without transferring the archive): `pip install requests remotezip`; `python scripts/inspect_zenodo_archive.py`. An optional GitHub Actions workflow also records the archive inventory in an artifact. This depends on Zenodo honoring byte-range requests; failure must not trigger a full ZIP download.

## Stage 1: Obtain inputs, NOT new alchemical FEP

Study the data manifest at https://doi.org/10.5281/zenodo.21919704. Prefer selected files for two ligands, water and bound environments. Place them in `data/` or institutional scratch space; **do not commit them**. Confirm they contain the trajectories and reference energies required by `run_tfep.py`. The script expects per-repeat energies in `p38/calculations/lig_<id>/eq/<phase>/repeat-<n>/ener.xvg` (or `ener.edr` convertible using GROMACS). Other trajectory/topology paths must be confirmed by examining the full upstream script and archive.

The authors' README states their FF driver was adapted from pmx and requires their simulation directories. Initial reproduction targets the published endpoint corrections rather than rerunning `run_ff_calculations.py`.

## Stage 2: Original baseline

In the *upstream* directory and validated upstream environment, start with the paper's documented CLI, adapting ligand names and environment paths to your downloaded data:

```bash
# Execute from a checkout of the authors' repository, after installing compatible
# dependencies and restoring the actual input-data directory tree.
# Ligand-only mapped region, with the 0.0–3.5 Å environment shell conditioning.
python run_tfep.py preprocess --ligand p38a_2g --phase water --mapcut 0.0 --condcut 3.5 --outdir tfep-mixedmaf/map00-cond35
python run_tfep.py train --ligand p38a_2g --phase water --potential sqm --outdir tfep-mixedmaf/map00-cond35
python run_tfep.py estimate --ligand p38a_2g --phase water --outdir tfep-mixedmaf/map00-cond35
```

**Important correction**: the upstream docstring's `--potential dft` command invokes CPMD/MiMiC, **not** the paper's GFN1-xTB mechanical-embedding target. The actual `sqm` code calls `GROMACSPotential(full) - GROMACSPotential(ligand) + TBLitePotential(method='GFN1-xTB')`. With the argparse CLI, omitting `--onlyref` sets `only_reference_conditioning=False` and uses the environment conditioning region. Adding `--onlyref` selects reference-frame atoms only. Use the same run configuration when comparing to the paper's results. These commands have **not been executed** with the published trajectories.

The original preprocessing and data-loader code assume each repeat contains **2,401 frames (12 ns at 5 ps intervals)**; training discards the first 2 ns, leaving **2,000 frames per repeat**. The paper used 25 repeats (50,000 samples per ligand/phase), and its training used **one epoch with batch size 32**. Reducing the repeat count changes the estimator's sampling and does **not** constitute exact reproduction.

Required paths include `p38/calculations/lig_p38a_2g/eq/water/repeat-*/{eq.tpr,traj.trr,ener.xvg}` (or `ener.edr` converted to `ener.xvg`), `eq_nvt/water/eqnvtout.gro`, ligand-specific topology files in `topology/`, and `p38/ligands_gaff2/lig_p38a_2g/mol_gmx.pdb`. The upstream `SQMMMPotential` additionally copies `topol_shared/ligand_only.top`, **which is absent from the public source tree at the inspected commit**. The Zenodo ZIP **does contain** `lig_p38a_2g/topology/ligand_only.top`; recover this exact published file and verify include-relative paths if staging it at `topol_shared/ligand_only.top`. Do not invent force-field topology parameters.

The training script expects GROMACS executable `gmx_mpi_d` (double-precision MPI) and GROMACS inputs compatible with the published **2021.6** setup. It was written for a SLURM/MPI environment; no claim of CPU desktop or aarch64 compatibility has yet been established.

**Additional archive/script mismatches** verified from ZIP member names: the deposited trajectories are `eq/<phase>/repeat-0/traj_comp.xtc`, not `traj.trr`; the archived one-repeat structures are `eq/<phase>/repeat-0/eqout.gro`, not the upstream `eq_nvt/<phase>/eqnvtout.gro` expected for SQM single-point GROMACS initialization. The `.xtc` compressed trajectory is ~73 MB in water and ~1.01 GB in protein for this ligand (compressed ZIP member sizes), so target water first. A physically consistent one-repeat adaptation may use `.xtc` and the archived `eqout.gro` after validating topology, atom order, box, and coordinate alignment; it will **not reproduce the original full-data TFEP result**.

A lightweight independent-data check is implemented by `scripts/check_published_endpoint.py` and GitHub Actions `published-endpoint-probe.yml`. It accesses small saved `tfep_logs/analysis.json` and `tfep_logs/train/epoch-0.npz` files using byte-range reads; any attempt to recreate the full work trajectory from only one archived `ener.xvg` should report incompatible reference-energy indices rather than pretend to validate the 50,000-frame free energy. Preserve the original training/run-provenance context when interpreting the archived estimates.

The `run_tfep.py estimate` implementation uses work `(target[kT]-log_det_J)*kT - FF[kJ/mol]`, selects FF potentials by original `trajectory_sample_index`, and computes FEP on training-logged samples. This on-training estimator is the authors' published estimator, not independent held-out validation. The paper's reported work-standard-deviation diagnostic discards high-energy outliers and uses the last 40,000 frames, so the unfiltered `analysis.json` value `std work` is not necessarily the same plotted metric.

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
