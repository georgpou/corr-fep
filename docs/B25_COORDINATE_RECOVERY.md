# B2.5 Cartesian coordinate recovery

## Evidence and hypothesis

The original epoch-1 diagnostic is Actions run 38148045379 at commit
680a15dff8023e2b174ab9b31d0d1f1242ef98c0. Its live step-25, -50, -75,
and -175 probes show accurate direct spline and autoregressive inverses
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
Our working hypothesis is that this recoding changes the internal/reference
coordinates supplied to the inverse MAF. This is a hypothesis until the
frozen-coordinate experiment below confirms the causal link.

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
