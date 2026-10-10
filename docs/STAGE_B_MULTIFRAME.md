# Stage B2 — multiframe QM/MM feasibility

**Scope:** B2.1 only. This experiment tests whether the *already validated* B1 mechanical-embedding
energy and its analytic/autograd derivatives remain finite and internally consistent on
several real archived molecular configurations. It is **not** a TFEP training run, an
independent validation set, a statistical convergence analysis, or a binding free-energy
estimate.

## Physical model and input

Same Hamiltonian and fixed parameters as
[Stage B1](STAGE_B_MECHANICAL_EMBEDDING.md):

```
U_QMMM(x) = U_MM_full(x) - U_MM_isolated_ligand(x_L) + U_GFN1-xTB(x_L)
```

- Full-system classical term: GROMACS 2023.3, double precision, PME and published
  Lennard-Jones settings audited from compiled TPR, from pinned original topology/MDP.
- Isolated ligand classical term: GROMACS on identical ligand Cartesian positions,
  with the original 1000-fold enlarged vacuum-like periodic box.
- Quantum term: genuine GFN1-xTB as evaluated by tblite 0.3.0; closed-shell,
  neutral 46-atom ligand, following validated B0/B1 assumptions.
- Source: `lig_p38a_2g/eq/water/repeat-0/traj_comp.xtc`, the **single deposited
  repeat**, and its deposited `eq.tpr`, `eqout.gro`, `merged.itp`,
  and topology include tree. ZIP access uses HTTP Range and downloads just the
  required member (~73 MB water trajectory), *not* the ~49 GB entire ZIP.
- Original research source commit `b8dfd897ebd914b4e592221710b397a2758353ce`,
  TFEP commit `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e`,
  GROMACS source commit `5e5ea27d8e594124d229a726616b7f6473f2c935`.

The XTC records coordinates, time and cell, **not independent atom names/types**.
Those identities come from its matching archived TPR/ITP. The code validates the
XTC atom count against the reconstructed GRO/TPR and checks the actual ligand
bond lengths after the published transformations. It does not invent a
trajectory-native atom-identity check.

## Protocol

1. Verify source archive byte-range support; retrieve only the necessary
   trajectory and small published topology members.
2. Recreate and audit original B1 full and ligand GROMACS TPRs. Validate the
   ligand identity, periodicity and original atom index mapping.
3. Read the original XTC through MDAnalysis, apply
   `unwrap(ligand) -> center_in_box(ligand) -> wrap(environment)` per frame,
   and retain the actual frame-dependent periodic box.
4. Take **four uniformly index-spaced frames including both trajectory endpoints**
   (smoke sample, deliberately **not** claimed independent).
5. For each frame, evaluate all three *real* energies plus total PyTorch
   autograd gradient. Check finiteness and environmental
   `grad(QMMM) == grad(MM_full)`, and compare the quantum energy and
   ligand gradient to an independent direct tblite single point.
6. Record component energies, frame indices/times, boxes, ligand bond-length
   range, numerical costs, versions, provenance hashes and gradients in
   `multiframe.json` and `gradients.npz`. Preserve diagnostics even if
   any frame fails.

The B1 18-probe complete-Hamiltonian central-difference audit remains the
baseline for gradient correctness. B2 adds validation on changed geometries
without multiplying expensive finite differences.

Run through [the B2 workflow](../.github/workflows/stage-b-multiframe.yml)
on GitHub Actions. The installed double-precision GROMACS binary is restored
from the same keyed cache if accessible; a cache miss builds pinned GROMACS
from source, and the workflow saves it before the molecular experiment.
GitHub Actions cache access is scoped by branches, so a newly created branch
**may not** be able to restore a cache originally generated only on a sibling
branch, even when its key matches.

## Scientific acceptance and subsequent gates

The pilot passes only if each of the four genuine-QM/MM frames has finite
components/gradients, valid ligand geometry, correct dimensionality and
periodic box, identical environmental gradients to full MM (1e-7
kJ/mol/Å), and agreement of the independent tblite calls with the
quantum wrapper (0.02 kJ/mol and 0.03 kJ/mol/Å, matching B1).

A passed smoke check establishes multiframe *evaluation feasibility*, not
force-field correction accuracy, sampling coverage or stable training.
B2.2 will compare compatible classical reference potentials to published
`ener.xvg` and explicitly verify time alignment and energy definitions.
B2.3 will separately validate the chosen batching/sequential interface
under training conditions. B2.4 can then attempt a small, real, reverse-KL
normalizing-flow training smoke check.

Caution: the author's 25-repeat training set is not deposited. A four-frame
sample from one repeat is temporally correlated to an unknown degree. XTC
coordinates are lossy; absolute GROMACS energies can differ from archived
reference evaluations and must not be compared without matching frame
times, coordinate precision, treatment of constraints and selected energy
terms. In particular the QM/MM **absolute** energy includes a large
ligand electronic-energy offset and is not a binding free energy.

## Execution record

The workflow stores live results as GitHub Actions artifacts; scientific
acceptance must be based on the actual run output (including any failures),
not on the existence of code or a green unrelated CI job.
