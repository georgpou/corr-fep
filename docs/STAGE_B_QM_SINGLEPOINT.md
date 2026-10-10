# Stage B0 — Real published ligand GFN1-xTB single-point checkpoint

The experiment runs the *actual* semiempirical QM engine on the archived
`p38a_2g` ligand structure, verify charge from a matching archived
GROMACS ligand topology and finite-difference check its Cartesian gradient.
This is explicitly a **ligand-only QM component**, not yet the subtractive
mechanical-embedding QM/MM Hamiltonian.

The full next-stage energy must eventually be evaluated on the *same*
published trajectory configuration as

```
U_QM/MM = U_MM(full) - U_MM(isolated ligand) + U_GFN1-xTB(ligand)
```

with gradient consistency and atom indexing confirmed across both
GROMACS and tblite. This initial checkpoint must not be represented as
successful full-system QM/MM or TFEP training.

Inputs are accessed from the 49 GB Zenodo archive using byte-range
requests. The script refuses normal/full ZIP transfer, selects the exact
ligand rather than the distinct p38a_2gg, parses the published water
\`eqout.gro\` and cross-checks all 46 ligand atom names, order, mass-based
element identities and net charge against archived \`merged.itp\`.
The probe refuses to proceed on inconsistencies. GROMACS force-field
energy/gradient bookkeeping remains unverified. Only metadata, energy and gradient diagnostics are uploaded;
no structure/trajectories are committed to Git.

For a reproducible finite-difference check, displace the largest
magnitude analytic gradient coordinate by ±0.001 bohr and compare
the central numerical derivative in hartree/bohr.

## Measured findings (2026-10-10)

Real [successful cloud execution](https://github.com/georgpou/corr-fep/actions/runs/38069425210)
and [scientific artifact](https://github.com/georgpou/corr-fep/actions/runs/38069425210/artifacts/11675814139)
at commit `f709b15e73c711e7ea792baebc0c6e2708c89c8b`:

| Quantity | Measured |
| --- | ---: |
| Archived structure | `lig_p38a_2g/eq/water/repeat-0/eqout.gro` |
| Ligand topology | `lig_p38a_2g/topology/merged.itp` |
| Ligand atoms / formal net charge | 46 / 0 |
| GFN1-xTB single-point energy | -82.39412697356606 Hartree |
| Gradient Euclidean norm | 0.2242969425404303 Hartree/Bohr |
| Finite difference atom/axis (zero-based) | 7 / 2 (z) |
| Finite-difference step | 0.001 Bohr |
| Analytic derivative | 0.06287449267463131 Hartree/Bohr |
| Central numerical derivative | 0.06287437096830217 Hartree/Bohr |
| Absolute derivative discrepancy | 1.217063291397702e-7 Hartree/Bohr |
| Relative derivative discrepancy | 1.9357027621612353e-6 |
| Engine | `tblite==0.3.0`, GFN1-xTB |

The archived structure contains the full solvated system, but the QM engine
is applied **only to its MOL residue**; this is a ligand *component* energy.
The absolute energy is not a QM/MM free energy or binding affinity.

Two initial cloud probes were scientifically unsuccessful because the archive
does not contain the expected ligand `mol_gmx.pdb` and loose string selection
confused `p38a_2g` with the separate `p38a_2gg`. Both were diagnosed from
real job logs and corrected. Crucially, the initial workflow's `tee` pipeline
was not propagating Python failures; the workflow now uses
`set -o pipefail` to ensure the scientific step cannot appear green on
a Python error.

### Remaining checkpoint

Evaluate GROMACS `U_MM(full)` and `U_MM(ligand)` on the **same**
archived configuration, preserve ligand atom and coordinate indexing,
match periodic box and topology inputs, and compare combined forces with
numerical displacement derivatives. No full QM/MM single-point or flow
training is claimed at this stage.

