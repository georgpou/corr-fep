# Stage B1 — GFN1-xTB mechanical-embedding QM/MM

**Status: accepted real Hamiltonian and autograd validation using pinned double-precision GROMACS 2023.3 and an interfacial environment probe on one archived `p38a_2g` water configuration.** The target reproduces the author's *subtractive* mechanical-embedding prescription:

```
U_QM/MM(x) = U_MM(full; x) - U_MM(ligand; x_L) + U_GFN1-xTB(ligand; x_L)
```

Ligand–environment forces remain classical. All three contributions use precisely the same ligand geometry and atom indices. No MM proxy is mislabeled as QM.

## Input provenance and algorithm

The existing Stage B0 `prepare_structure` reconstructs coordinates and atom identity from the original `lig_p38a_2g/eq/water/repeat-0/{eq.tpr,eqout.gro}` and `lig_p38a_2g/topology/merged.itp`. It checks the 46-atom neutral ligand, connected bonds, topology atom names, types, masses, charges, and applies `unwrap(MOL)`, `center_in_box(MOL)`, `wrap(environment)`. Additional published `lig_p38a_2g/topology/` members, including `ligand_only.top`, `topol_water.top`, and `ffMOL.itp`, are obtained via HTTP byte-range access to [Zenodo 21919704](https://doi.org/10.5281/zenodo.21919704); the ~49 GB archive is not fully downloaded.

The original force field `topol_shared/amber99sb-star-ildn-mut.ff` and the two single-point MDPs come from [research source](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026) commit `b8dfd897ebd914b4e592221710b397a2758353ce`. Only the include search path is rebased, without changing physical settings. Full and ligand-only GROMACS TPRs are constructed from the archived topologies. The original `SQMMMPotential` expands the ligand-only periodic cell lengths 1000-fold to approximate vacuum; this protocol is reproduced. Pinned [TFEP v0.2](https://github.com/andrrizzi/tfep) at `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e` supplies `GROMACSPotential` and `TBLitePotential` with precomputed PyTorch gradients. GROMACS returns forces `F=-∇U`; the TFEP wrapper handles the sign and converts coordinate units from Å to nm and forces back to kJ/mol/Å.

## Executed single-point energies

The accepted [double-precision interfacial-gradient run](https://github.com/georgpou/corr-fep/actions/runs/38073286507) succeeded on source commit `f7505065cb77c87d0ca8e054352a050d0377af6c` and was independently inspected via [artifact 11678021501](https://github.com/georgpou/corr-fep/actions/runs/38073286507/artifacts/11678021501). Its source code is unchanged by the subsequent documentation-only commit.

| Component | kJ/mol |
| --- | ---: |
| Full-system GROMACS MM | -116802.61791405082 |
| Isolated-ligand GROMACS MM | -971.022166844934 |
| Ligand GFN1-xTB (tblite 0.3.0) | -216325.75103968772 |
| **Combined QM/MM** | **-332157.3467868936** |

Direct tblite versus TFEP quantum discrepancies were `1.0768e-9 kJ/mol` in energy and `1.38145e-8 kJ/mol/Å` for the maximum gradient component. For every environmental atom the full-MM and combined QM/MM gradient tensors agreed exactly within the reported calculation (`max error = 0`).

**Complete-Hamiltonian central finite differences:** six atom/axis pairs (five ligand elements H/C/N/O/F and one solvent atom), each with displacements `h=0.002, 0.005, 0.010 Å`, for **18 probes total**. Maximum absolute analytical-versus-FD errors in kJ/mol/Å:

| Central-difference step (Å) | Maximum error (kJ/mol/Å) |
| --- | ---: |
| 0.002 | 0.01220 |
| 0.005 | 0.08413 |
| 0.010 | 0.35612 |

The module supports **one configuration at a time**, not a batched training backend. Its TFEP-style `(batch_positions, batch_cell)` interface has shapes `(1, 25260)` and `(1, 6)` and returns energy shape `(1,)`; its complete-module backward gradient differs from component assembly by at most `1.4211e-14 kJ/mol/Å` (energy error zero). **All 18 finite-difference probes** passed the fixed acceptance bound **0.8 kJ/mol/Å at each of 0.002, 0.005 and 0.010 Å**, including an atom 2.676 Å from the ligand with an interfacial gradient component of `292.103 kJ/mol/Å`. Its per-step errors were `0.00105`, `0.00659`, and `0.02641 kJ/mol/Å`. The complete GROMACS-minus-GROMACS-plus-QM energy is reevaluated on every perturbed structure (not a surrogate). The agreement supports correct gradient signs and atom correspondence at this state point, **not** universal smoothness or force-field interchangeability.

## Compiled TPR physical audit

The final successful workflow builds **GROMACS 2023.3 in double precision**, pinned to upstream commit `5e5ea27d8e594124d229a726616b7f6473f2c935`; the original paper used **GROMACS 2021.6 double-precision MPI**, so bitwise equivalence across versions is not established. Earlier Ubuntu mixed-precision runs are retained only as precision diagnostics. Its `grompp` translates historical `vdw-type = switch` to `vdw-type = Cut-off` and `vdw-modifier = Potential-switch`; the latter describes the same switched potential, and the **compiled TPR** is inspected with `gmx dump -s`. The final script asserts the following settings (not just the input MDP names):

| Parameter | Water/full MM | Isolated ligand MM |
| --- | --- | --- |
| PBC | xyz | xyz, cell lengths x1000 |
| Electrostatics | PME, `rcoulomb=1.1 nm` | Cut-off, `rcoulomb=5.0 nm` |
| Lennard-Jones | Potential-switch, 1.0 to 1.1 nm | Potential-switch, 4.9 to 5.0 nm |
| Dispersion correction | EnerPres | EnerPres |

A repeated mixed-precision run exhibited an error of `2.013 kJ/mol/Å` at 0.005 Å, consistent with quantization of the large full-system MM energy; no tolerances were relaxed. Switching to the pinned double-precision build reduced the maximum 0.002 Å error to `0.01220 kJ/mol/Å`. The compiled TPR confirms unchanged physical electrostatic/LJ cutoffs; GROMACS may still adjust neighbor-list scheduling. GROMACS 2021.6 double-precision equivalence remains an unresolved version-comparison question, not a blocker for this single-point Stage B1 checkpoint.

## Reproducibility and limitations

The workflow `stage-b-mechanical-embedding.yml` is manually dispatchable and runs on an Actions CPU runner. It builds the real GROMACS double-precision CPU engine and pinned Python molecular stack, downloads only required remote archive entries, preprocesses the original inputs, asserts compiled physical interaction settings and audits gradient components. Failed and successful runs upload text diagnostics. The [final scientific artifact](https://github.com/georgpou/corr-fep/actions/runs/38073286507/artifacts/11678021501) contains `preparation.json` (SHA256 input hashes, versions, upstream source revision, generated TPR hashes, ligand indices), `components.json` (all energies and per-probe derivatives), MDPs, compiled-TPR parameter extracts, rerun/grompp logs and `environment.txt`. Original input topologies, molecular coordinates, trajectories and binaries are not committed.

The initial attempt exposed a missing `pyedr` EDR reader; a later mixed-precision run exposed real energy quantization; and the first source build encountered an upstream CMake test-target issue. These failures were corrected without changing force-field parameters. The final double-precision experiment passed the full scientific gates. The present result validates **one published molecular configuration**, not equilibration, statistical overlap, free energy, or real-molecule objective comparisons. The available archive contains only `repeat-0`, not all 25 original sampling repeats, and Stage B0's neutral closed-shell singlet assumption is retained. Do **not** initiate molecular flow training until this precise checkpoint is accepted and multi-frame feasibility has been considered.

## Final scientific review and merge readiness (October 10, 2026)

The accepted final scientific source is `f7505065cb77c87d0ca8e054352a050d0377af6c` and its verified [GitHub Actions execution](https://github.com/georgpou/corr-fep/actions/runs/38073286507) produced `components.json` with `validated=true`, the three genuine Hamiltonian energies, energy/gradient autograd checks, and 18 full finite-difference evaluations. The stronger environmental probe was selected **before computing any numerical derivatives**, by maximum absolute analytical force within a 4.5 Å periodic interfacial shell, removing the weak-gradient loophole.

The archived input SHA256 digests, exact published atom indices, original FF hashes, processed MDP and generated TPR hashes, pinned upstream software commits, GROMACS version/precision, per-step analytic and finite-difference gradients, and diagnostic engine logs are recorded in the artifact. The current branch's later `docs/` update changes documentation only.

**Scientific limitations:** one archived water configuration and one repeat; no QM/MM equilibrium sampling, TFEP target distribution, map optimization or free energy result; the `MechanicalEmbedding` implementation is single-frame and must be generalized/validated separately before molecular training. Do not merge without explicit user approval.

## Reusing the GROMACS CPU build

The Actions workflow caches the **installed** pinned GROMACS 2023.3 double-precision
binary (not generated TPRs, force-field inputs, coordinates, or results) using
`actions/cache/restore@v4` and `actions/cache/save@v4`. The cache key includes
the original GROMACS source commit, important CMake choices (double precision,
CPU/thread MPI, FFTW3 and SIMD), and runner OS/architecture. A cache miss builds
GROMACS and saves it **before** evaluating the molecular Hamiltonian, so a
failed scientific gradient check does not discard a successful build. A cache
hit skips source checkout, CMake and compilation. Each run still installs the
small FFTW runtime package and verifies the cached build fingerprint,
executable, double precision and runtime-library resolution before using it.

The first successful run with this cache configuration must populate it;
subsequent runs on the same branch/platform can restore it. GitHub cache
eviction or changes to the pinned build ID may require recompilation. Cache
restoration is a performance optimization only and does not change the
Hamiltonian, coordinates, precision, MDPs or scientific acceptance thresholds.
