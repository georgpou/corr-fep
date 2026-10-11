# corr-fep

**Reproducible FF → QM/MM endpoint correction research using multimap targeted free-energy perturbation (TFEP).**

[![CI](https://github.com/georgpou/corr-fep/actions/workflows/ci.yml/badge.svg)](https://github.com/georgpou/corr-fep/actions/workflows/ci.yml)

## Scientific objective

Reproduce the published *mechanical-embedding* GFN1-xTB/p38 kinase benchmark, then test whether improved training and independent, trajectory-aware diagnostics make FF-to-QM/MM free-energy corrections more statistically reliable. In phase one we **do not rerun alchemical A → B simulations**. Existing classical trajectories and published relative binding free energies are inputs.

For a ligand in a given phase, the reduced work of an invertible coordinate map is

```
w(x) = u_QMMM(M(x)) - u_FF(x) - log|det J_M(x)|
Delta f = -log <exp(-w)>_FF
```

We will compare estimators at equal QM/MM evaluation budgets and on *held-out trajectories*. Changing a mapping must not change the exact free-energy difference.

## Status

**Stage B2.5: author-method training recovery passed its bounded integration checks.** The failed extended benchmark was traced to the moving ligand reference frame. A source-informed adapter now uses the authors’ matched-water conditioning references and original six-layer MixedMAF, with an attributed intermediate water-slot ordering correction. A real GROMACS/GFN1-xTB pilot completed three updates over 96 training frames; sampled trained inverses, independent Cartesian determinants, fixed environment, physical gradient and checkpoint reload checks passed. The frozen 16-frame validation comparison is mixed: work spread decreased, while weight ESS worsened (1.83 to 1.56). This is not convergence or a demonstrated TFEP improvement. The recovered full 650/250/300 benchmark, hypothesis comparison, complete 25-repeat paper experiment and ARM64/H200 deployment remain outstanding. See the [detailed audit log](docs/AUTHOR_RECOVERY_AUDIT_LOG_2026-10-11.md), [report snapshots](docs/evidence/2026-10-11-author-recovery), [source audit](docs/UPSTREAM_IMPLEMENTATION_AUDIT.md), and [historical recovery record](docs/B25_COORDINATE_RECOVERY.md).

## Get started (no dataset required)

Python >=3.10:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m corr_fep.cli doctor
python -m corr_fep.cli demo --seed 17
```

The demo evaluates mathematically constructed work values; it is **not** a QM/MM reproduction.

## Upstream research software

- [Gay et al. reproducibility scripts](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026): `run_tfep.py`, `run_ff_calculations.py`, simulation inputs, and historical Conda environment.
- [TFEP library](https://github.com/andrrizzi/tfep) (MIT): flow implementations and molecular-potential interfaces.
- [Paper-associated data](https://doi.org/10.5281/zenodo.21919704): inspect the archive before downloading; do not commit trajectories.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [docs/ROADMAP.md](docs/ROADMAP.md), and [docs/HPC_ARM64.md](docs/HPC_ARM64.md).

## Suggested layout

```
src/corr_fep/       lightweight independently testable research utilities
tests/              numerical and CLI regression tests
docs/               scientific protocol, milestones, HPC notes
scripts/            opt-in data/source preparation
configs/            experiment configuration
.github/workflows/  cheap CI and explicitly triggered smoke checks
data/               ignored local datasets
runs/               ignored results and checkpoints
external/           ignored clones of upstream sources
```

## Provenance and scope

The original paper's QM/MM scheme is *subtractive mechanical embedding*: the ligand's FF intramolecular energy is replaced with GFN1-xTB, while ligand–environment interactions remain classical. Electrostatic embedding is a future, separately validated extension. Read the upstream licenses before redistributing data or code.

No measured scientific improvements are asserted by this scaffold.
