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
artifact includes the JSON and full logs even on failure. **The protocol
is not itself evidence that B2.3 passed**: require validated=true in the
measured JSON and inspect execution logs before accepting the checkpoint.

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
