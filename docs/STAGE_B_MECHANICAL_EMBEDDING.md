# Stage B1 — Mechanical-embedding GFN1-xTB QM/MM

## Physics checkpoint

This workflow attempts the genuine subtractive target
`U_QMMM = U_MM(full) - U_MM(ligand) + U_GFN1-xTB(ligand)`
on one archived p38a_2g water configuration.

The full GROMACS system retains ligand–solvent MM interactions; the vacuum
ligand MM is removed and the same ligand coordinates are evaluated using tblite
GFN1-xTB. The GROMACS terms are evaluated using the pinned TFEP v0.2
`GROMACSPotential` with `precompute_gradient=True` and the QM term through
`TBLitePotential`. No classical proxy is labeled quantum.

## Provenance and methodology

Read only the exact `lig_p38a_2g` archive prefix from Zenodo 21919704
using HTTP byte ranges. The initial GRO/TPR/ITP is validated by the existing B0
identity/geometry method. Additional topology files are read under
`lig_p38a_2g/topology/` and checked for absent required members.
The ligand force-field directory and two MDP files come from the research
implementation at `b8dfd897ebd914b4e592221710b397a2758353ce`.
The original MDP physical parameters are not changed. Only the preprocessing
include path is rebased to the staged published topology directory, with hashes
of source and staged MDP files recorded.

The original `SQMMMPotential` uses a 1000-fold enlargement of each water
cell axis for ligand vacuum MM, copied here. TFEP's original GROMACS wrapper
is used directly to read GROMACS energies and forces; PyTorch autograd returns
minus the engine forces, as required. The full coordinate tensor uses the
published atom order; ligand slicing uses global indices from the TPR.
The quantum term is independently cross-checked against direct tblite energy
and analytical gradient at the identical Cartesian geometry.

The first exploratory CI environment uses the distribution's CPU GROMACS
rather than claiming the paper's double-precision GROMACS 2021.6 binary.
Actual GROMACS precision/version, preprocessing logs, checksums, FF inputs,
and package versions are written to the artifact. A source-version or
preprocessing incompatibility is an **unresolved scientific blocker**,
not justification for silently modifying the Hamiltonian.

## Acceptance

The success marker requires genuine finite energies from all three engines,
agreement between direct and wrapped QM energies/gradients,
environment-gradient equality with full MM, and Cartesian central finite
differences of the *combined* QM/MM energy at several ligand element/axis
coordinates and one solvent coordinate. Both 0.002 and 0.005 Å steps are
tested to diagnose finite precision. The exploratory bound is
4.0 kJ/mol/Å per probe; the observed discrepancies, not just pass/fail,
must be examined to decide whether the precision is adequate for training.

Artifacts include `preparation.json`, `components.json`,
`full-grompp.log`, `ligand-grompp.log`, processed MDP files,
GROMACS version, `environment.txt`, and `experiment.log`.
Generated topologies, binary TPRs and trajectories are not committed.
Do not begin training until this checkpoint is actually executed and reviewed.

## Status

Implementation proposed; successful numerical execution has not yet
been established. This section must be updated from actual Actions results.
