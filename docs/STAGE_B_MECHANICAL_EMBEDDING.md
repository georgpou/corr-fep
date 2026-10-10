# Stage B1 — GFN1-xTB mechanical-embedding QM/MM

**Status: executed on the original archived `p38a_2g` water configuration.** The target reproduces the author's *subtractive* mechanical-embedding prescription:

```
U_QM/MM(x) = U_MM(full; x) - U_MM(ligand; x_L) + U_GFN1-xTB(ligand; x_L)
```

Ligand–environment forces remain classical. All three contributions use precisely the same ligand geometry and atom indices. No MM proxy is mislabeled as QM.

## Input provenance and algorithm

The existing Stage B0 `prepare_structure` reconstructs coordinates and atom identity from the original `lig_p38a_2g/eq/water/repeat-0/{eq.tpr,eqout.gro}` and `lig_p38a_2g/topology/merged.itp`. It checks the 46-atom neutral ligand, connected bonds, topology atom names, types, masses, charges, and applies `unwrap(MOL)`, `center_in_box(MOL)`, `wrap(environment)`. Additional published `lig_p38a_2g/topology/` members, including `ligand_only.top`, `topol_water.top`, and `ffMOL.itp`, are obtained via HTTP byte-range access to [Zenodo 21919704](https://doi.org/10.5281/zenodo.21919704); the ~49 GB archive is not fully downloaded.

The original force field `topol_shared/amber99sb-star-ildn-mut.ff` and the two single-point MDPs come from [research source](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026) commit `b8dfd897ebd914b4e592221710b397a2758353ce`. Only the include search path is rebased, without changing physical settings. Full and ligand-only GROMACS TPRs are constructed from the archived topologies. The original `SQMMMPotential` expands the ligand-only periodic cell lengths 1000-fold to approximate vacuum; this protocol is reproduced. Pinned [TFEP v0.2](https://github.com/andrrizzi/tfep) at `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e` supplies `GROMACSPotential` and `TBLitePotential` with precomputed PyTorch gradients. GROMACS returns forces `F=-∇U`; the TFEP wrapper handles the sign and converts coordinate units from Å to nm and forces back to kJ/mol/Å.

## Executed single-point energies

[Full 18-probe gradient run](https://github.com/georgpou/corr-fep/actions/runs/38071996006) and [compiled-TPR audit run](https://github.com/georgpou/corr-fep/actions/runs/38072104953) both succeeded.

| Component | kJ/mol |
| --- | ---: |
| Full-system GROMACS MM | -116801.421875 |
| Isolated-ligand GROMACS MM | -971.0215454101562 |
| Ligand GFN1-xTB (tblite 0.3.0) | -216325.75103968772 |
| **Combined QM/MM** | **-332156.15136927756** |

Direct tblite versus TFEP quantum discrepancies were `9.8953e-10 kJ/mol` in energy and `1.3812e-8 kJ/mol/Å` for the maximum gradient component. For every environmental atom the full-MM and combined QM/MM gradient tensors agreed exactly within the reported calculation (`max error = 0`).

**Complete-Hamiltonian central finite differences:** six atom/axis pairs (five ligand elements H/C/N/O/F and one solvent atom), each with displacements `h=0.002, 0.005, 0.010 Å`, for **18 probes total**. Maximum absolute analytical-versus-FD errors in kJ/mol/Å:

| Central-difference step (Å) | Maximum error (kJ/mol/Å) |
| --- | ---: |
| 0.002 | 2.84616 |
| 0.005 | 0.45640 |
| 0.010 | 0.63276 |

The accepted strict bound is **0.8 kJ/mol/Å at 0.005 and 0.010 Å**; all relevant probes passed. Smaller 0.002 Å derivatives are retained to show loss of precision. The complete GROMACS-minus-GROMACS-plus-QM energy is reevaluated on every perturbed structure (not a surrogate). The agreement supports correct gradient signs and atom correspondence at this state point, **not** universal smoothness or force-field interchangeability.

## Compiled TPR physical audit

The Ubuntu workflow uses **GROMACS 2023.3 mixed precision**, rather than the authors' GROMACS 2021.6 double-precision MPI build. Its `grompp` translates historical `vdw-type = switch` to `vdw-type = Cut-off` and `vdw-modifier = Potential-switch`; the latter describes the same switched potential, and the **compiled TPR** is inspected with `gmx dump -s`. The final script asserts the following settings (not just the input MDP names):

| Parameter | Water/full MM | Isolated ligand MM |
| --- | --- | --- |
| PBC | xyz | xyz, cell lengths x1000 |
| Electrostatics | PME, `rcoulomb=1.1 nm` | Cut-off, `rcoulomb=5.0 nm` |
| Lennard-Jones | Potential-switch, 1.0 to 1.1 nm | Potential-switch, 4.9 to 5.0 nm |
| Dispersion correction | EnerPres | EnerPres |

The larger h=0.005–0.010 Å probes mitigate energy quantization from mixed-precision full-system MM energies; for example, one solvent 0.002 Å numerical derivative rounds to zero. GROMACS reports neighbor-list adjustments without changing interaction cutoffs. Bitwise energy/force equivalence with the authors' double-precision build is **not established** and should be revisited before extensive molecular flow training if precision problems appear.

## Reproducibility and limitations

The workflow `stage-b-mechanical-embedding.yml` is manually dispatchable and runs on an Actions CPU runner. It installs the real GROMACS CPU engine and pinned Python molecular stack, downloads only required remote archive entries, preprocesses the original inputs, asserts compiled physical interaction settings and audits gradient components. Failed and successful runs upload text diagnostics. The [scientific artifact](https://github.com/georgpou/corr-fep/actions/runs/38072104953/artifacts/11677555522) contains `preparation.json` (SHA256 input hashes, versions, upstream source revision, generated TPR hashes, ligand indices), `components.json` (all energies and per-probe derivatives), MDPs, compiled-TPR parameter extracts, rerun/grompp logs and `environment.txt`. Original input topologies, molecular coordinates, trajectories and binaries are not committed.

The first attempt failed only because the optional `pyedr` EDR reader was missing; it was installed and the real engine run subsequently passed. The present result validates **one published molecular configuration**, not equilibration, statistical overlap, free energy, or real-molecule objective comparisons. The available archive contains only `repeat-0`, not all 25 original sampling repeats, and Stage B0's neutral closed-shell singlet assumption is retained. Do **not** initiate molecular flow training until this precise checkpoint is accepted and multi-frame feasibility has been considered.
