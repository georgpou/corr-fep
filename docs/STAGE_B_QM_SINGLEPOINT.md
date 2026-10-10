# Stage B0 — Real published ligand GFN1-xTB single-point checkpoint

First run the *actual* semiempirical QM engine on the archived
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
requests. The script refuses normal/full ZIP transfer, checks explicit PDB
elements, and refuses to assume a molecular charge in the absence of a
matching ligand ITP. The published coordinates must be inspected and
atom-order correspondence verified before combining with force-field
components. Only metadata, energy and gradient diagnostics are uploaded;
no structure/trajectories are committed to Git.

For a reproducible finite-difference check, displace the largest
magnitude analytic gradient coordinate by ±0.001 bohr and compare
the central numerical derivative in hartree/bohr.

## Measured findings

Not yet established until a successful real GitHub Actions execution.
