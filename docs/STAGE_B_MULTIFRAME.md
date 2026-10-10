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
4. Register the same transformations on both GRO and XTC **before** checking
   terminal coordinate alignment (the previous ordering compared a processed
   GRO with an unprocessed XTC);
   select four uniformly spaced frames from the deposited **production window
   401–2400**, omitting the first 2 ns equilibration interval. The sample
   remains deliberately **not** claimed statistically independent.
5. For the first selected production frame, independently check a ligand
   and an interfacial environment component of the **complete QM/MM gradient**
   by central differences (0.002 Å; B1 acceptance threshold 0.8 kJ/mol/Å).
6. For each frame, evaluate all three *real* energies plus total PyTorch
   autograd gradient. Check finiteness and environmental
   `grad(QMMM) == grad(MM_full)`, and compare the quantum energy and
   ligand gradient to an independent direct tblite single point.
7. Record component energies, frame indices/times, boxes, ligand bond-length
   range, numerical costs, versions, provenance hashes and gradients in
   `multiframe.json` and `gradients.npz`. Preserve diagnostics even if
   any frame fails.

The B1 18-probe complete-Hamiltonian central-difference audit remains the
baseline for gradient correctness. B2 extends it to a new production configuration
using one ligand and one interfacial-environment coordinate without repeating
all 18 checks.

Run through [the B2 workflow](../.github/workflows/stage-b-multiframe.yml)
on GitHub Actions. The installed double-precision GROMACS binary is restored
from the same keyed cache if accessible; a cache miss builds pinned GROMACS
from source, and the workflow saves it before the molecular experiment.
GitHub Actions cache access is scoped by branches, so a newly created branch
**may not** be able to restore a cache originally generated only on a sibling
branch, even when its key matches.

## Scientific acceptance and subsequent gates

The pilot passes only if the XTC-to-GRO coordinate anchor passes, the two
complete-Hamiltonian numerical gradient checks pass, and each of the four genuine-QM/MM frames has finite
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

## Recovered B2.1 scientific checkpoint — 2026-10-10

The corrected dedicated Actions job **passed** at
[run 38087599814](https://github.com/georgpou/corr-fep/actions/runs/38087599814)
(commit `8f2343d`, [artifact 11683300215](https://github.com/georgpou/corr-fep/actions/runs/38087599814/artifacts/11683300215)).
The focused endpoint-outlier repeat likewise **passed** at
[run 38088018062](https://github.com/georgpou/corr-fep/actions/runs/38088018062)
(commit `c6f2492`, [artifact 11682861447](https://github.com/georgpou/corr-fep/actions/runs/38088018062/artifacts/11682861447)).
Both artifacts were inspected; their `multiframe.json` files record
`validated: true`, and each contains finite full-system gradients for
all four production samples.

- Sampled XTC indices 401 / 1067 / 1733 / 2400 at 2005 / 5335 /
  8665 / 12000 ps; original 0--400 interval excluded.
- Correctly transformed terminal XTC versus GRO: minimum-image RMS
  0.00011077 Å, max 0.01001359 Å; after robust translation registration
  RMS 0.00010984 Å, max 0.01000404 Å. The median common translation was
  only ~0.00000954 Å per component. Full box-length difference
  0.0000191 Å. Translation was **not** applied to energy inputs.
- 46-ligand-atom periodic pairwise distance maximum discrepancy
  0.00000502 Å; nearest-ligand distance RMS discrepancy across
  environment atoms 0.00006086 Å, max 0.00546 Å. The GRO title is
  `Ligand in water`, **with no explicit recorded time**; XTC endpoint
  is 12000 ps. The exceptional close structural match supports, but does
  not prove from the GRO timestamp, same-step correspondence.
- Full-Hamiltonian central differences on frame 401: ligand atom 0,
  y axis error 0.024428 kJ mol^-1 Å^-1; environment atom 7618,
  y axis error 0.003195 kJ mol^-1 Å^-1, both well below the
  predeclared B1 tolerance 0.8. Environmental QM/MM gradient equals
  full-MM gradient exactly to recorded numerical precision. All four
  quantum direct-versus-wrapper gradient errors are below
  1.70e-8 kJ mol^-1 Å^-1.

**Interpretation:** B2.1 now establishes real multiframe evaluation and
two production-frame derivative probes for the published subtractive
Hamiltonian. It does not establish statistically independent samples,
training stability, free-energy accuracy, or reproduction of 25 repeats.

## High-priority special-case discrepancy — quantified, still tracked

[Issue #9](https://github.com/georgpou/corr-fep/issues/9) is separate
from B2.1 acceptance. Under the **same generated TPRs and runtime**, the
terminal XTC minus GRO MM-full energy remains **−1.01205337 kJ/mol**
and QM/MM combined energy **−1.01180347 kJ/mol**. This is not a
difference between the two former workflow environments.

The endpoint coordinate audit pinpoints exactly **one** atom above
0.001 Å residual: **zero-based atom 463, water oxygen OW in SOL resid
141**. Its XTC-minus-GRO minimum-image displacement is
[+0.00000763, **+0.01001358**, +0.00001144] Å; all remaining atoms'
post-registration displacements are below ~0.000033 Å. At the XTC
endpoint its y derivative is −99.414 kJ mol^-1 Å^-1, giving
`gradient · (XTC - GRO)` ~−0.995 kJ/mol for this atom.

A **diagnostic-only** hybrid XTC configuration with atom 463 reset to
its GRO position (no other coordinate, force field, or quantum parameter
changed) changes MM-full energy by **+1.06320969 kJ/mol**, leaving
a hybrid-minus-GRO difference **+0.05115632 kJ/mol**. The ligand-only
MM and isolated quantum components remain unchanged by the
environmental-atom substitution. Thus the single 0.01 Å water-oxygen
difference is causally responsible for the overwhelming majority of the
apparent ~1 kJ/mol offset. The precise file-encoding/rounding mechanism
for that one coordinate and the remaining ~0.051 kJ/mol energy difference
are not yet independently decomposed.

No per-frame offset is applied to real QM/MM energies. The hybrid is
**not** an archived molecular configuration and must never be used as
training data. Issue #9 remains open until the next focused B2.2 task
matches published `ener.xvg` definitions and frame times and evaluates
how precision-related differences vary across configurations; a single
terminal snapshot cannot establish a constant reweighting offset.

## Execution record and provenance

The two scientific artifacts contain `multiframe.json`,
`gradients.npz`, archived-source hashes, generated TPR hashes, GROMACS
double-precision version, installed package versions, compiled parameter
audits and full experiment logs. The dedicated workflow rather than
generic CI determines scientific acceptance.
