# B2.5 Cartesian coordinate recovery

## Initial evidence

The original epoch-1 diagnostic is Actions run 38148045379 at commit
680a15dff8023e2b174ab9b31d0d1f1242ef98c0. Its completed 13 probes (steps 25 through 325) show accurate direct spline and autoregressive inverses
(about 1e-16 position error and 1e-15 logdet closure), while the full
Cartesian inverse fails the unchanged 1e-5 gates. The first observed
failure is at step 25, not necessarily the first failing optimizer step.

At step 175, Cartesian inverse error is 0.0013095971502004033 Angstrom
and logdet closure error is 0.05727789340141953. The environment stays
exactly fixed. The splines remain close to identity with ordinary bin sizes
and slopes. These observations do not support changing spline inversion
or the optimizer to address this failure.

The pinned MixedMAF converts each configuration to an atom-based relative
frame, stores six reference-coordinate residuals near zero, shifts those
residuals, and reconstructs Cartesian positions in the original frame.
Its inverse then constructs a new frame from the moved reference atoms.
The frozen-coordinate experiment below confirms that this recoding changes
the internal/reference coordinates supplied to the inverse MAF. The precise
missing Jacobian formula remains unresolved; the independently measured
determinant discrepancy establishes that inverse closure alone is insufficient.

## One bounded frozen-coordinate experiment

`scripts/stage_b_frozen_coordinate_audit.py` loads `first-invalid.pt` and
`last-epoch1.pt` from the completed original diagnostic. It rebuilds the
original map with the same pinned libraries, calibration frames, real
TPR/XTC coordinates, and actual saved weights. No optimizer steps or
MM/QM energy evaluations are made. Existing real-potential setup is reused
only to preserve units and map reconstruction.

For three training and three validation configurations per checkpoint:

- Replay the original spline, mixed-space, Cartesian, and geometry checks.
- Measure mixed -> Cartesian -> mixed recoding before any inverse MAF.
- Temporarily replace only the six reference shifts by identity as a causal
  ablation; restore the original module immediately afterward.
- At train frame 726, independently differentiate the complete ligand
  Cartesian map and compare its 138-by-138 determinant with the reported
  logdet. The atom-dependent frame is included in this derivative.
- Verify the original frame-726 results reproduce the archived checkpoint
  probe to 1e-8 absolute tolerance.

The Actions workflow has a 60-minute cap and a 25-minute execution cap,
and waits only for the existing run's artifact. It never retries training.
The push marker is a one-time request for this distinct no-QM experiment.

## Decision record

Ruling: use the existing authorized development branch, with a separate
local snapshot; keep `main` unchanged. Do not create unrelated worktrees or
run local scientific calculations because the user requested Actions.

Ruling: use one numerical experiment as the targeted scientific check,
instead of adding general edge-case tests. No production fix is being
introduced in this commit.

Ruling: the reference-shift identity is diagnostic only. It is not a
production model change, a new acceptance threshold, or a trained
checkpoint eligible for held-out testing. Any ensuing correction must
address inverse and actual Cartesian Jacobian, and receive fresh physical
validation. No held-out test frames enter this diagnostic.

PR #14 remains draft and unmerged. Complete recovery and benchmark success
have not been established.

## Completed frozen-checkpoint evidence

The original diagnostic finished all 325 updates and saved both checkpoints.
The final Cartesian errors reproduce the first failed benchmark to about
1e-13 absolute precision; all 13 mixed-space probes remain at machine
precision. The frozen audit completed as Actions run
[38149445011](https://github.com/georgpou/corr-fep/actions/runs/38149445011),
artifact 11706690747. Its JSON confirms zero QM evaluations and exact
checkpoint replay (all four compared differences are zero).

Across three training and three validation frames per checkpoint:

| Checkpoint | Reference shifts | Max inverse error (Angstrom) | Max logdet closure | Geometry gates |
|---|---|---:|---:|---:|
| Step 25 | Original | 8.297181e-5 | 1.474078e-3 | 0/6 pass |
| Step 25 | Disabled, diagnostic only | 3.197442e-14 | 8.526513e-14 | 6/6 pass |
| Step 325 | Original | 6.325858e-3 | 8.295727e-2 | 0/6 pass |
| Step 325 | Disabled, diagnostic only | 6.394885e-14 | 1.136868e-13 | 6/6 pass |

The coordinate bridge isolates the recoding error to the reference triangle
shape coordinates (indices 129, 130, 131); the 129 other internal
coordinates recode at machine precision. Six shifted reference residuals
are reset to approximately zero when the new atom-based frame is built.
Disabling those shifts removes the positional closure failure. This is
causal evidence, not merely a correlation with optimizer steps.

However, inverse closure alone misses a second defect. At train frame 726,
the independent 138-by-138 full Cartesian determinant differs from the
reported value:

| Checkpoint | Reference shifts | Numerical logabsdet | Reported logabsdet | Absolute discrepancy |
|---|---|---:|---:|---:|
| Step 25 | Original | 0.03194662516534237 | 0.033921474432190735 | 0.001974849266848364 |
| Step 25 | Disabled | 0.03643947769728424 | 0.03392147443216231 | 0.002518003265121929 |
| Step 325 | Original | 0.8855554488581434 | 0.9067553656517191 | 0.02119991679357569 |
| Step 325 | Disabled | 0.9246609243899414 | 0.9067553656516907 | 0.017905558738250726 |

All numerical determinant signs are positive. Thus merely disabling the
six shifts is **not a scientifically valid production fix**. The true
Cartesian derivative includes the dependence of the molecular frame on
mapped coordinates; the reported coordinate-conversion determinant does
not capture the complete transformation in this ligand-based setup.

## Concrete next diagnostic and proposed representation

The pinned paper `run_tfep.py:run_preprocess` chooses its origin and axes
from **conditioning atoms**, which stay fixed under the map. Our pilot
instead obtains its frame from mapped ligand atoms. The proposed minimal
alternative retains a one-layer MixedMAF and the ligand-only mapped set,
while adding three solvent atoms as fixed conditioning/reference atoms.
The Hamiltonian, target labels, temperature, partitions, and acceptance
thresholds remain the same. This changes the model representation and
requires a methodological decision before adoption and physical training.

`scripts/stage_b_fixed_reference_probe.py` is an explicitly diagnostic
proposal, not a benchmark change. It chooses one O/H/H water identity
using only the original 32 training calibration frames, rejects wrapped
or near-collinear reference triples there, and locks that identity before
validation. It checks original geometry gates plus independent full
Cartesian determinants and a central-difference directional derivative
on training and validation. Identity and deterministic parameter
perturbations of 0.001 and 0.03 are examined. **Perturbation is not trained
optimization**; no QM/MM energies, mock energies, or training are used.
The experiment is capped at 40 workflow minutes and 25 execution minutes.
No full benchmark retry is authorized by a successful diagnostic.


## Completed fixed-reference probe and limits

Actions run [38150245773](https://github.com/georgpou/corr-fep/actions/runs/38150245773),
at commit `55c217c6779ccae7d4a6d5ca7b35ef5c750bdfc2`, completed with an
**overall scientific failure** at the 0.03 perturbation. Artifact
11706637983 contains the successful smaller-perturbation rows and the raw
failure log. The original JSON did not persist the failed row; the later
reporting repair writes failures before re-raising and does not alter gates
or retroactively modify that artifact. No experiment was rerun for this repair.

The locked reference identity is zero-based atoms **4747, 4748, 4749**
(O/H/H): origin 4747, axes 4748 and 4749. Selection used only the original
32 training calibration frames. Initial oxygen-to-ligand-centroid distance
was 2.959003 Angstrom; calibration O-H lengths were 0.916407–1.010001
Angstrom and minimum angle sine was 0.950529. These atoms remain unchanged
within each map evaluation; they still move across trajectory frames.

| Parameter noise amplitude | Geometry/inverse gates | Max inverse error (Angstrom) | Max independent logdet discrepancy | Max directional finite-difference error |
|---|---|---:|---:|---:|
| 0 | 6/6 sampled frames pass | 1.899636e-11 | 8.248957e-14 | 2.712980e-9 |
| 0.001 | 6/6 sampled frames pass | 1.903899e-11 | 4.552955e-14 | 3.230351e-9 |
| 0.03 | First frame (train 401) fails; remaining frames not evaluated | 3.197442e-14 at failure | Not evaluated | Not evaluated |

Independent determinants and finite differences were measured only at
train 726 and validation 1275 for each completed amplitude. At amplitude
0.03, bond maximum was **3.103457 Angstrom** and ligand displacement was
**8.676873 Angstrom**, exceeding the unchanged 2.5 and 2.0 Angstrom gates.
Inverse/logdet closure alone at this failed geometry is not acceptance.
Random parameter stress is not optimizer training and neither predicts
nor establishes physical training stability.

The focused independent review found no fundamental error in derivative
scope or training-only calibration. For a map `(ligand, environment) ->
(f(ligand, environment), environment)`, the full Jacobian is block triangular;
its determinant equals the ligand block differentiated with environment
held constant. All results remain sampled/local. The original-map derivative
has not received an independent finite-difference check. Water proximity,
periodic continuity, and useful conditioning throughout actual training
remain unestablished; this is not a reproduction of the paper's complete
conditioning setup. No physical training steps or QM energies occurred
in either new coordinate experiment. No held-out test frames were used.

## Reviewable next action: three real-QM/MM updates

**Superseded proposal:** after the user requested grounding in the original
code, adoption of these O/H/H indices was withdrawn as the recommended next
action. See [upstream implementation audit](UPSTREAM_IMPLEMENTATION_AUDIT.md)
for the original matched-water/heavy-atom pipeline, issue log and revised
proposal. The evidence above remains unchanged; no physical run is approved.


**Decision pending; do not dispatch yet.** Add the locked three solvent
conditioning/reference atoms above for a separate three-update pilot on
this development branch. This changes conditioning atom selection/model
representation and requires the methodological decision in the recovery
handoff. Existing trained weights cannot be treated as the new map's checkpoint.

After approval:

1. Lock the exact reference identity and original trajectory transformations.
   Check unsplit/noncollinear reference geometry across all 650 training and
   250 validation configurations before optimization. Abort if this convention
   fails; do not choose another water using validation outcomes. Preserve the
   original 32-frame training-only spline calibration and seed.
2. Use fresh one-layer MixedMAF weights, the first three original training
   batches (batch size 2), AdamW lr 1e-4, weight decay 0, float64, the same
   temperature/objective, and actual pinned GROMACS/tblite mechanical-embedding
   QM/MM energies and forces. No Hamiltonian or loss changes.
3. Before and after each update, enforce original geometry, fixed-environment,
   inverse and closure gates on the same six training/validation frames.
   After the third update, independently check full Cartesian determinants
   and directional finite differences at train 726 and validation 1275.
   Record finite energies/forces, nonzero physical gradient, parameter changes,
   checkpoint reload agreement, provenance and all failures. Cap the pilot
   at 60 workflow minutes / 40 execution minutes with no automatic retry.

No full epoch, held-out test evaluation, benchmark success claim, or merge is
included in that decision. A successful three-update pilot would support
considering a bounded extended benchmark, not establish convergence or
free-energy accuracy. PR #14 stays draft and `main` stays unchanged.
