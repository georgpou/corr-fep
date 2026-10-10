# Stage B0 — Validated published-ligand GFN1-xTB component

This checkpoint is **not** a full QM/MM or a free-energy calculation.
It verifies the ligand-only quantum term on a published p38a_2g water
configuration and checks the coordinate/gradient interfaces required later
for differentiable mechanical embedding.

## Physical Hamiltonian and scope

The eventual subtractive mechanical-embedding target is

```text
U_QM/MM = U_MM(full) - U_MM(ligand) + U_GFN1-xTB(ligand)
```

This PR evaluates **only the final term**, preserving the distinction between
a genuine single-point quantum calculation and a complete MM-subtracted QM/MM
target. The ligand's charge and topology are taken from the published files.
No invented ligand geometry, topology parameters or surrogate energy is used.

## Input reconstruction and chemical identity

Only three exact ZIP members are read using HTTP byte ranges from
[Zenodo 21919704](https://doi.org/10.5281/zenodo.21919704):

- `lig_p38a_2g/eq/water/repeat-0/eq.tpr` (authoritative GROMACS bonded
  topology, atom types, masses and charges)
- `lig_p38a_2g/eq/water/repeat-0/eqout.gro` (water-box coordinates)
- `lig_p38a_2g/topology/merged.itp` (original atom/charge parameter table)

The TPR is loaded with MDAnalysis 2.7.0 and paired with the corresponding GRO.
We explicitly require one MOL residue and one bonded connected fragment,
and cross-check **every** MOL atom's name, type, mass and charge against the
ITP record in the exact source ordering. Atomic numbers are derived using
the **same atom-type first-letter convention** as the pinned upstream
`TFEPMap.setup`, but restricted to its documented H/C/N/O/F/S elements
and cross-checked independently against masses. Unsupported/ambiguous types
fail rather than silently guessing. Charge must be near an integer and
the assumed singlet model must have an even electron count.

To handle periodic images, apply the original paper's exact MDAnalysis
transformation sequence: `unwrap(MOL)`, `center_in_box(MOL)`,
`wrap(not MOL)`. Then verify that all ligand bonds are chemically plausible
in Cartesian distances (0.65–2.2 Å is a **sanity interval**, not a chemical
identity proof); record the pre- and post-transform maximum bond lengths,
periodic cell, transformations and coordinate shift. This explicitly
addresses the possibility of presenting an artificially split molecule to
vacuum GFN1-xTB.

The isolated-ligand electronic state is taken as **neutral closed-shell
singlet**, matching the archived neutral MM charge and even electron count.
This is an explicit assumption, **not** an independent quantum
ground-state spin or protonation-state determination.

## Independent numerical checks

1. Call the genuine `tblite.interface.Calculator("GFN1-xTB", ..., uhf=0)`
   using coordinates in Bohr and obtain energy in Hartree and analytic
   gradients in Hartree/Bohr.
2. Check Cartesian central differences of energy at two step sizes
   (0.0005 and 0.001 Bohr) across representative atoms/elements and all
   three axes, including the maximum-magnitude analytical gradient coordinate.
   Each checked derivative must agree within 5e-5 Hartree/Bohr.
3. Pass the **same geometry** into the pinned
   `tfep.potentials.tblite.TBLitePotential` with
   `positions_unit=angstrom`, `energy_unit=kJ/mol`,
   `precompute_gradient=True`, and execute PyTorch backward.
   Check returned energy against the direct tblite energy in kJ/mol and
   compare every returned tensor gradient component with direct
   GFN1-xTB gradients converted into kJ/mol/Å.
4. Independently finite-difference the actual TFEP energy output,
   to verify numerical consistency of the whole wrapper/autograd route,
   not merely the algebraic unit factors.

Note that the pinned TFEP wrapper does not expose the QM charge or UHF
parameters, so this narrow wrapper check is restricted to the observed
**neutral singlet** ligand. Charged or open-shell ligand extension needs
an explicit general solution, not an implicit assumption of neutrality.

## Reproducibility and outputs

The script logs per-member SHA-256 checksums, detailed atom and topology
information, periodic box, transformation protocol, resulting energy and
gradient metrics, pinned upstream SHAs and dependency versions. The
workflow has `set -o pipefail`; Python failures cannot produce a
false-green action from `tee`. It saves `geometry.json` even if later
QM evaluation fails; successful runs also save `singlepoint.json`.
Archived molecular structure/topology files are not committed or uploaded.

GitHub Actions on CPU uses Python 3.11, PyTorch 2.4.0, MDAnalysis 2.7.0,
tblite 0.3.0, NumPy 1.26.4 and TFEP pinned at
`0bc369a1194c688bb6ac997cc6b7c2db2b063c9e`.
No unnecessary generalized experiment framework is introduced.

## Executed results

The initial ligand-only single-point returned -82.39412697356606 Hartree
with one checked direct gradient derivative differing by ~1.22e-7
Hartree/Bohr. It was *not* geometry- or wrapper-validated, which motivated
the improved protocol above. The later topology-aware run verified 46
atoms, 49 bonds, one molecular fragment, matching TPR/ITP atom records,
neutral charge and 194 electrons, with a post-transform bond-length range
of approximately 1.019–1.584 Å. It failed while serializing a NumPy integer
in the final results, *after* the gradient checks; that output bug was
corrected without relaxing the scientific acceptance thresholds.

The fully validated [successful GitHub Actions run](https://github.com/georgpou/corr-fep/actions/runs/38070629961)
at commit `a940277e379b025cc6d11e0c0da2988973e82401` generated
[artifact 11677155856](https://github.com/georgpou/corr-fep/actions/runs/38070629961/artifacts/11677155856).
The machine-readable artifact was downloaded and inspected independently:

| Quantity | Validated value |
| --- | ---: |
| Connected MOL ligand | 46 atoms, 49 bonds, one fragment |
| Elemental composition | C19 H19 F1 N4 O3 |
| Formal net charge / electrons | 0 / 194 (assumed singlet) |
| Unprocessed maximum bonded distance | 1.58401531 Å |
| Processed bonded distance range | 1.018773–1.584016 Å |
| GFN1-xTB energy | −82.39412711652093 Hartree |
| Number of central-difference probes | 32 (16 atom/axis combinations × 2 steps) |
| Probe coverage | H, C, N, O, F; x, y, z |
| Maximum direct QM gradient discrepancy | 1.268891235374145e-6 Hartree/Bohr |
| Wrapper versus direct QM energy discrepancy | 1.0477378964424133e-9 kJ/mol |
| Max wrapper versus converted gradient discrepancy | 1.381167180625198e-8 kJ/mol/Å |
| Wrapper gradient versus wrapper-energy finite difference | 0.0008278344386667413 kJ/mol/Å |

Source file SHA-256 values are saved in the result artifact, for the
GRO, TPR and ITP files. The geometry is consistent with an intact ligand;
although the maximum coordinate shift under the published centering protocol
was 17.56 Å, the bond lengths changed only by rounding-level differences.
A large absolute coordinate shift alone is not evidence of an erroneous
periodic-boundary correction or changed internal structure.

The initial wrapper integration and multiple numerical derivatives now pass.
The smoke test's thresholds were fixed prior to interpreting the completed
scientific run; no tolerance was loosened to obtain this outcome. The
chemical charge and closed-shell model are still model assumptions, not an
experimental determination of the ligand's actual solution protonation
or electronic state.

## Remaining scope

We must still validate GROMACS `U_MM(full)` and
`U_MM(ligand)` on the same saved configuration, including compatible
PBC/bond topology, FF–QM coordinate correspondence and a finite-difference
check of the full combined Hamiltonian. No QM/MM TFEP training has yet
been performed.
