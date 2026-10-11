# Stage B2.3 — training-callable real QM/MM potential

This checkpoint adapts the validated B1/B2.1 *published* mechanical-embedding
Hamiltonian to sequential minibatches, without fitting a model or rerunning FEP:

    U_target(y) = U_MM_full(y) - U_MM_ligand(y_L) + U_GFN1-xTB(y_L)
    U_reference(x) = U_MM_full(x)

The two configurations x and y=M(x) are different. The full-system MM terms
must **not** be cancelled when constructing mapped work:

    w(x) = beta * [U_target(M(x)) - U_reference(x)] - log|det J_M(x)|
    beta = 1 / (0.00831446261815324 * 298.0) mol/kJ

Energy in kJ/mol, positions in Å, gradients in kJ/mol/Å, boxes in Å/degrees.
GROMACS is pinned double-precision 2023.3, with original published
topology/MDPs and 1000-fold expanded ligand-only box, and tblite GFN1-xTB.

## Checkpoint implementation

- scripts/stage_b_training_interface.py defines SequentialTrainingPotential.
  It takes a (B,3*N) PyTorch positions tensor and (B,6) frame-dependent box;
  forward evaluates U_target, and reference evaluates U_reference, with shape
  (B,) for both. It serializes B1 backend calls, because both GROMACS reruns
  otherwise share filesystem paths. The pinned TFEP potential caches calculated
  forces for backward, preserving differentiability after multiple calls.
- Uses p38a_2g, water, only repeat-0, and B2.1 production frames
  401/1067/1733 at 2005/5335/8665 ps. The original archived TPR supplies
  atom ordering and identities. Frame transforms are unwrap ligand, center
  ligand in box, wrap environment. Each frame keeps its own cell.
- Compares the first batch element and gradients to the independently
  accepted B2.1 single-point implementation; runs two B2.1-style finite
  differences at frame 401. Checks finite batch energies/gradients and
  consecutive repeated-call consistency.
- Uses a tiny reversible ligand-centroid affine scaling/translation map ONLY
  as a Jacobian, work-convention and map-gradient diagnostic. Its analytic
  log determinant = 3*(N_lig-1)*log(scale), independently verified against
  a 2-atom explicit Jacobian determinant. It is NOT the published MixedMAF.
- Records real evaluations, original input and generated TPR hashes,
  walltime/throughput, peak RSS, reference/target energy and gradient norms,
  map work, inverse/Jacobian/chain-rule checks, and package versions in
  results/stage-b/training-interface/training-interface.json.

Execute using .github/workflows/stage-b-training-interface.yml. Its Actions
artifact includes the JSON and full logs even on failure. See the accepted
run record below; scientific acceptance requires the JSON plus logs, not CI alone.

## Sampling caveats and next gate

The deposited XTC coordinates are quantized; original ener.xvg is based on
different, high-precision coordinates. Thus U_reference(x) MUST be
re-evaluated for each archived input x. This yields coherent energies at
the archived coordinates, but does NOT establish their sampling measure as
the exact high-precision MM equilibrium distribution. Only one deposited
repeat is available, with time-correlated frames. No free energy, ESS or
25-repeat reproduction claims can follow from this pilot.

B2.4 must replace the diagnostic map with the original molecular MixedMAF,
run a small real train optimization, and evaluate held-out temporally blocked
samples while tracking work, loss, Jacobians and the sampling limitation.


## Accepted molecular execution — 2026-10-11

The dedicated [GitHub Actions run 38127130676](https://github.com/georgpou/corr-fep/actions/runs/38127130676)
**passed** on commit `3e09c5c86cafa917726cc239849bdd77b2a81d94`.
The archived [scientific artifact 11696804319](https://github.com/georgpou/corr-fep/actions/runs/38127130676/artifacts/11696804319)
was inspected independently: the ZIP has a valid CRC and its
`training-interface/training-interface.json` reports `validated: true`.
This is a passed B2.3 *interface* checkpoint, **not** a trained flow.

- Input: 8,420 full-system atoms, 46 ligand atoms; Cartesian batch
  `(3,25260)`; frames 401, 1067, 1733 at 2005, 5335, 8665 ps, with
  measured box sides 43.66759, 43.65424, 43.68644 Å, respectively.
- Full FF reference energies (kJ/mol): −117062.438646,
  −116604.925596, −117393.872495.
- Full target QM/MM energies (kJ/mol): −332413.311513,
  −331926.731450, −332747.896869.
- The baseline complete FF + QM/MM energy/gradient minibatch took
  **6.5678 s for 3 frames**, or **0.4568 frames/s** on GitHub Actions
  CPU; peak process RSS was **477.97 MiB**. This excludes the archive
  retrieval, environment installation, and TPR compilation.
- Frame-401 direct scalar-backend versus batch-backend energies/gradients:
  recorded maximum discrepancies **0** within reported precision.
  The independent central-difference errors (kJ/mol/Å) were **0.024428**
  for the ligand and **0.003195** for the interfacial solvent, both
  below the predeclared B2.1 threshold of 0.8.
- Diagnostic ligand affine map: scale 1.0002, shift
  (0.001, −0.001, 0.0005) Å. Analytical `log|J|` **0.02699730**
  per 46-atom ligand; the separate two-atom determinant check differed
  by **1.18e−16**. Inversion error **3.55e−15 Å**, forward/inverse
  log-determinant closure **6.25e−17**; mapped-work parameter-gradient
  chain-rule discrepancies **1.14e−13** (scale) and **2.13e−14**
  (shift). Repeated target energies were exactly equal at the recorded
  precision.
- The three illustrative dimensionless reduced map works were
  −86914.9921, −86903.3389, −86916.5635, each independently
  recalculated from its archived numerical reference energy, mapped
  target energy, beta and log-J. These large negative values contain
  the absolute QM electronic energy offset; **they are not estimates
  of free energy and should not be compared to the original paper's
  free-energy values.**
- The original XTC ZIP member hash was
  `1d9451c4f969d23791676e971fdb358394917ea7c51dae43e7390fe604e99362`.
  Generated GROMACS full/ligand TPR hashes were
  `302012826952186547ef1326bc49e9d998f866b5c0ba7357f8e1b5707ce7219d`
  and `682d9a45e59888639684a82409a9186653253bcde10c4888a3ad2c89a66fb370`.
  The artifact preserves the full per-file archive and topology hashes.

**Acceptance:** B2.3 establishes finite real minibatch energy/force
autograd, a consistent reference/mapped-target work convention, and
numerically checked invertible mapping semantics. It does **not** establish
an optimized published MixedMAF, statistical overlap,
meaningful ESS, unbiased MD sampling, or a free-energy result. Those are
separate scientific gates in B2.4 and beyond.

## B2.3 upstream TFEP interface repair and revalidation

An independent review discovered the serial adapter lacked the
`positions_unit` and `energy_unit` properties read by the pinned
`TFEPMapBase` and `MixedMAFMap` constructors. The adapter now delegates
both to its existing full-MM GROMACS potential, preserving its exact pint
registry, Å and kJ/mol conventions. The dedicated Actions workflow also now
watches **its own** workflow YAML in its push path filters.

The real-trajectory checkpoint now includes a *genuine pinned-upstream
`MixedMAFMap`* constructed on the archived TPR and a 12-frame temporal
subset of the same transformed XTC trajectory. It tests thermal-energy
units, actual coordinate/box correspondence, ligand-only mapping (all water
atoms fixed), finite mapped-target energies and log-Jacobians, and one actual
flow-parameter gradient of the upstream reverse-KL target-plus-Jacobian
objective through the real QM/MM autograd backend. One MAF layer is used
to keep the interface test inexpensive. No optimizer, loss convergence,
held-out validation, or QM/MM flow-training result is claimed here; those
belong in B2.4.

**Accepted revalidation (2026-10-11):** [run #38128849456](https://github.com/georgpou/corr-fep/actions/runs/38128849456)
passed with the repaired source at commit `9f093b8379b7bd6d44bad3206fad03fe07494dab`.
[Artifact #11698756379](https://github.com/georgpou/corr-fep/actions/runs/38128849456/artifacts/11698756379)
ZIP SHA256 `74a489a8149ce13de77bffc66373092d7a39ece9e5082d99f9dd7d167604fc92` was independently checked along with ZIP CRC, all scalar delta-energy and mapped-work arithmetic, and `validated=true` in its JSON.

The first repaired run [#38128412180](https://github.com/georgpou/corr-fep/actions/runs/38128412180) revealed missing TPR element labels for the upstream Z-matrix. Reconstructing them with exactly the pinned upstream research `fix_universe_elements` logic addressed this. Run [#38128616297](https://github.com/georgpou/corr-fep/actions/runs/38128616297) then exposed an optional `bgflow` dependency. Final successful run pins `noegroup/bgflow` to `fbba56fac3eb88f6825d2bd4f745ee75ae9715e1` and `einops==0.8.0`.

**Real upstream integration:** One-layer original `tfep.app.mixedmaf.MixedMAFMap` initialized from 12 archived production frames and ran a 2-frame forward/backward on frames 401 and 582. Position and box replay errors were 0; the mapped tensor had shape `(2,25260)`, environment displacement was 0, and `kT=2.477709860209666` kJ/mol. Real QM/MM energies were `[-332413.3115125408,-332471.1247856832]` kJ/mol; log-J was `[0,0]` at the initial identity flow. All nine trainable parameter-gradient tensors were finite, two nonzero, largest absolute gradient 270.5911. This proves a gradient path through the original flow to the real target. It does **not** prove a learning step, non-identity Jacobian correctness, loss decrease, or statistical overlap.

Existing B2.3 molecular energies/gradients and mapped-work arithmetic remain numerically consistent. Repaired baseline took 10.417 s for 3 frames (versus 6.568 s in the prior run), with peak Python RSS 495 MiB; CPU speed should not be compared without accounting for runtime variation.

**Accepted scope:** The missing unit contract, original atom-element metadata, optional mixed-coordinate dependency and workflow trigger are repaired. B2.4 must still execute real flow optimization and held-out temporal diagnostics. Lossy XTC and one deposited repeat remain unresolved sampling limitations.

