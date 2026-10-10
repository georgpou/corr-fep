# Checkpoint 1: p38a_2g published-data verification (2026-10-10)

**Status: archival inspection completed; original 25-repeat TFEP training NOT reproduced.**

## Source provenance

- Paper workflow: `andrrizzi/paper-alchemical-tfep-qmmm-2026` at `b8dfd897ebd914b4e592221710b397a2758353ce`.
- TFEP v0.2: tag `0.2`, commit `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e`.
- Dataset DOI: [10.5281/zenodo.21919704](https://doi.org/10.5281/zenodo.21919704); ZIP `alchemical_tfep.zip`, 49,205,427,069 bytes.
- Author's backend: `--potential sqm`, with `GFN1-xTB` from tblite v0.3.0 and GROMACS MM full-minus-isolated-ligand; *not* `--potential dft`.
- Archived experiment: `lig_p38a_2g/tfep-mixedmaf/map00-cond35/{water,protein}` (ligand-only mapped region; 0–3.5 Å conditioning region).

## Authors' saved endpoint statistics

The first GitHub Actions numerical probe [successfully retrieved](https://github.com/georgpou/corr-fep/actions/runs/38062406270) the original `analysis.json` and 50,000-entry `epoch-0.npz` tensors for **each** phase. These are **archived outputs of the authors**, not a reproduction by Corr-FEP:

| Quantity | Water | Protein complex |
| --- | ---: | ---: |
| Archived FF→QM/MM ΔG (kJ/mol) | −215380.22037162536 | −215388.78090839947 |
| Archived 95% bootstrap CI (kJ/mol) | [−215381.51166, −215378.96117] | [−215389.98646, −215387.52238] |
| Archived unfiltered `std work` (kJ/mol) | 11.33895 | 8.68209 |
| Saved mapped target/work log entries | 50,000 | 50,000 |

Authors' archived phase correction difference (protein minus water):

```
delta_delta_G = -215388.78090839947 - (-215380.22037162536)
              = -8.5605367741 kJ/mol = -2.0460173934 kcal/mol
```

The absolute corrections include a large electronic-energy reference offset; the phase difference cancels it. This difference is computed algebraically from archived authors' outputs, not an independent estimate.

The paper's overlap plots use *late 40,000 generalized works with the 10 highest excluded*, whereas `analysis.json`'s `std work` is unfiltered; do **not** compare these SDs directly to the plotted metric.

## Independently recomputed **first-repeat subset** (not retraining)

GitHub Actions [successfully recomputed](https://github.com/georgpou/corr-fep/actions/runs/38062725060) reduced-data FF→QM/MM work with the **authors' stored** mapped QM/MM potentials and Jacobians, paired only to published `repeat-0/ener.xvg` FF reference energies. The 2,000 selected `trajectory_sample_index` values match 2,000 of the 50,000 logged training configurations for each phase; all are finite. These are **new calculations of the archived work estimator**, not new QM/MM single-point calculations or new training.

| Metric | Water | Protein |
| --- | ---: | ---: |
| Matched first-repeat training frames | 2,000 | 2,000 |
| Reduced-data ΔG (kJ/mol) | −215384.72179921376 | −215385.28259668284 |
| Raw work SD (kJ/mol) | 11.42985348849 | 8.77948341345 |
| Raw importance ESS, 2,000 weights | 1.356963412 | 32.638211873 |
| ESS fraction | 0.000678482 | 0.016319106 |
| Difference from archived full-data ΔG (kJ/mol) | −4.5014275884 | +3.4983117166 |

The reduced-data phase correction difference is **−0.5607974691 kJ/mol (−0.1340 kcal/mol)**. This differs by **+7.9997393050 kJ/mol** from the archived 50,000-sample phase difference (−8.5605367741 kJ/mol). These figures must **not** be interpreted as a controlled demonstration of convergence because the 2,000 samples use the authors' progressive 50,000-step training maps, not a flow trained exclusively on 2,000 configurations.

The ESS indicates severe weight concentration in the aqueous subset: 1.36 effective importance samples out of 2,000, versus 32.64 in the complex. It is **raw importance ESS only**, with no autocorrelation correction and no held-out validation. The work SD is untrimmed and not directly the published late-frame, outlier-excluded diagnostic.

## Why a full-data rerun is blocked

The successful [ZIP central-directory inventory](https://github.com/georgpou/corr-fep/actions/runs/38062133720) recorded 9,093 entries. For **every** ligand/phase among 32 ligands × 2 phases, Zenodo includes *only* `eq/<phase>/repeat-0/`. It lacks the other 24 repeats necessary for the original 25 × 2,000 = 50,000 analyzed frames. For `p38a_2g`, the single `ener.xvg` contains 2,401 total FF energy frames; the saved mapped work tensor refers to original trajectory indices **401 through 60024** (all 25 repeats). Only the 2,000 mapped samples whose indices are within repeat-0 can be paired to archived FF reference energies.

Additional issues before a reduced-data retraining: the deposited trajectory is `traj_comp.xtc` rather than `traj.trr`, the deposited structure is `eqout.gro` rather than `eq_nvt/.../eqnvtout.gro`, and `topol_shared/ligand_only.top` must be supplied from archived `lig_p38a_2g/topology/ligand_only.top` (not included in public code source). The water trajectory ZIP member is ~73 MB versus ~1.01 GB protein.

## Next decision

1. **Data-level validation — DONE:** matched and recalculated 2,000 first-repeat archived TFEP weights in both phases (see table above), with ESS and work dispersion. This does *not* constitute full 25-repeat reproduction or a new QM/MM calculation.
2. **Physical retraining:** adapt upstream input paths and test a **reduced-data** original `sqm` training on `repeat-0` water using the same flow and physical Hamiltonian, followed by a controlled numerical comparison.
3. **Full-reproduction gate:** obtain all 25 original FF reference trajectories/energies from the authors (or recreate and label them as new sampling). The authors' archived 50,000-entry target potential tensors alone cannot reconstruct missing FF reference energies.

Do not modify the original loss until the reduced baseline is executable and scientifically diagnosed; no original QM/MM training is claimed here.
