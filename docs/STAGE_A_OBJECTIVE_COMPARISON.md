# Stage A3 — Equal-budget baseline versus Hellinger-affinity training

## Prior evidence and frozen scientific choice

Stage A2 found reverse-KL optimization could lower training mean work while
deteriorating held-out ESS and concentrating mapped coordinates in one of
two target Gaussian basins. The experimental objective here is **mode coverage,
ESS and analytical free-energy accuracy simultaneously**, not loss reduction
alone. The single modification is an affinity objective with exponent 1/2,
chosen before seeing A3 results; no hyperparameter sweep is planned.

Let (q_\theta\) be the normalized mapped reference and (p_B\) the normalized target.
For an invertible map (y=M_\theta(x)\), generalized work is
(w=u_B(y)-u_A(x)-\log|\det J_\theta(x)|\).
The **Bhattacharyya affinity** obeys

```text
BC(q,p_B) = int sqrt(q(y)*p_B(y)) dy
          = sqrt(Z_A/Z_B) * E_A[exp(-w(x)/2)].
```

Thus minimizing `L_H = -log E_A[exp(-w/2)]` maximizes affinity;
`sqrt(Z_A/Z_B)` is independent of theta, so no target samples, target
partition functions or extra energy evaluations are needed. The empirical
objective is computed using logsumexp for stability.

Unlike reverse KL, the population Hellinger objective penalizes missing
target mass even if mapped mass lies within a single target peak. However,
**a finite minibatch cannot detect a completely unsampled basin**, so this
does not guarantee multimodal recovery. Nonlinear minibatch estimation is
also biased; gradients and noisy weights may destabilize training. There is
no claim that a mathematically less mode-seeking divergence guarantees better
finite-budget optimizers.

## Paired design

- Identical 2D standard-normal reference and symmetric 2D Gaussian-mixture
  target (centers x=±5, sigma=0.7, offset=1.25), exact Δf=1.96334988787746.
- Three seed pairs: 17, 29, 43; same 768 training positions, 4096
  validation positions, plus 8192 independently generated **final-only test**
  positions per seed. Test RNG is separate to preserve original A2 minibatches.
- Byte-identical initial MAF parameters, architecture (2 affine MAFs, hidden
  width 32), batch size 128, 350 optimizer steps, AdamW, gamma=0.997 schedule.
- One target-energy call per minibatch under either loss. Identical validation
  and test schedule and input sizes; counts include all target evaluations.
- Baseline loss uses the pinned upstream `BoltzmannKLDivLoss` unchanged.
  The modified loss uses **full** generalized work including `-u_A(x)`
  in the nonlinear exponent. The reference potential is not optional here,
  unlike for the baseline gradient.
- Reports raw untrimmed ESS, work variance and distribution, Δf/error,
  mapped halfspace and both fixed-radius target basins, Jacobian summary,
  actual minibatch losses, gradient norm, train and wall time.
- SHA-256 digests for paired source configurations and initial weights
  prevent accidental mismatches. `comparison.md` and `comparison.csv`
  summarize the untouched independent-test results.

The experimental script is a conservative extension of the original A2 script.
The previous A2 invocation still defaults to baseline with no extra test set.
This comparison is triggered on the feature-branch push and supports manual
GitHub Actions dispatch; scientific artifacts are uploaded, not committed.

## Decision rule

The modification is worth advancing only if both target modes are better
covered across the seeds, test ESS improves, test Δf error is no worse, and
Jacobian and gradients are numerically stable at similar evaluation cost.
A higher ESS with worse missing-mode coverage is not a success.

## Results

To be populated **only after reviewing the executed cloud-run artifacts**.
