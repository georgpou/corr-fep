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

## Executed experiment: 2026-10-10

The unchanged objective and fixed-beta alternative were both genuinely trained
with upstream TFEP MAFs on independent source draws. The
[first executed run](https://github.com/georgpou/corr-fep/actions/runs/38068031057)
reproduced A2 baseline behavior, including seed 29's optimization/overlap
divergence. A subsequent [diagnostic-only rerun](https://github.com/georgpou/corr-fep/actions/runs/38068195481)
added normalized importance-weighted target-basin occupancies and fuller
optimizer/Jacobian reporting; it reproduced the same training and
independent-test estimates. Both runs and the lightweight CI passed.

The final [review-corrected execution](https://github.com/georgpou/corr-fep/actions/runs/38068360022)
verified the same scientific results, with an additional plot of the
objective-specific actual minibatch loss. Scientific artifacts (raw
configurations, mapped positions, work arrays, metrics, optimization logs,
plots and dependency freeze) are available at
[GitHub Actions artifact 11675822058](https://github.com/georgpou/corr-fep/actions/runs/38068360022/artifacts/11675822058).

Each row below is based on an independent **8192-configuration final test**.
All energy/free-energy/work numbers are reduced (kBT). The exact free energy is
1.96334988787746. Each model per seed used **125,952 target evaluations**,
including training minibatches, training/validation diagnostics and the final
test. All pairs passed SHA-256 checks on identical initial parameters,
training draws, validation draws and test draws.

| Seed | Training loss | Test ESS / 8192 | Test Δf | Absolute error | Raw left basin | Raw right basin | Raw outside |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | Reverse KL | 1276.0 | 1.9779 | 0.0145 | 0.441 | 0.112 | 0.448 |
| 17 | Hellinger | 1804.5 | 1.9659 | 0.0025 | 0.105 | 0.104 | 0.791 |
| 29 | Reverse KL | 194.0 | 2.3192 | 0.3558 | 0.605 | 0.005 | 0.390 |
| 29 | Hellinger | 1918.5 | 1.9234 | 0.0399 | 0.110 | 0.123 | 0.766 |
| 43 | Reverse KL | 1002.5 | 2.1034 | 0.1400 | 0.056 | 0.412 | 0.531 |
| 43 | Hellinger | 1372.3 | 1.9955 | 0.0321 | 0.105 | 0.118 | 0.778 |

The reference target has ~0.432 probability in each radius-2σ disk and
~0.135 outside both. **Hellinger does not learn the target's geometric
occupancy.** Instead, it produces a more diffuse map, with ~77–79% of
unweighted mapped configurations outside the target-peak disks. Nevertheless,
importance reweighting can recover physical target observables if
both basins are sufficiently represented. The following independent-test
observables are computed using normalized weights
`exp(-w_i)/sum_j exp(-w_j)`:

| Seed | Loss | Weighted left basin | Weighted right basin | Weighted outside |
| --- | --- | ---: | ---: | ---: |
| 17 | Reverse KL | 0.414 | 0.448 | 0.138 |
| 17 | Hellinger | 0.424 | 0.441 | 0.135 |
| 29 | Reverse KL | 0.616 | 0.196 | 0.189 |
| 29 | Hellinger | 0.442 | 0.419 | 0.139 |
| 43 | Reverse KL | 0.397 | 0.505 | 0.097 |
| 43 | Hellinger | 0.418 | 0.443 | 0.139 |
| Target | Exact (approx. disks) | 0.432 | 0.432 | 0.135 |

This is more persuasive than a high ESS alone: the Hellinger model's
reweighted observables reproduce both known target mode probabilities
within finite-sample fluctuations, whereas the original seed-29 model
substantially underestimates the right basin. These weighted observables
are nevertheless derived from the *same importance weights* as the
free-energy estimator, and are not an independent proof of convergence.

### Optimization and Jacobian audit

| Seed | Loss | Train mean work | Validation mean work | Test work SD | Test log-J mean ± SD | Log-J min/max | Final gradient norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | Reverse KL | 4.857 | 5.180 | 5.737 | 0.117 ± 0.574 | -1.543 / 1.411 | 39.120 |
| 17 | Hellinger | 9.959 | 10.120 | 11.207 | 1.350 ± 0.044 | 1.218 / 1.517 | 0.792 |
| 29 | Reverse KL | 4.549 | 4.853 | 4.862 | -0.001 ± 0.683 | -2.065 / 1.325 | 13.811 |
| 29 | Hellinger | 11.971 | 13.735 | 22.306 | 1.369 ± 0.147 | 0.914 / 2.001 | 6.368 |
| 43 | Reverse KL | 5.779 | 6.276 | 7.266 | 0.310 ± 0.746 | -2.841 / 1.689 | 9.125 |
| 43 | Hellinger | 9.799 | 9.727 | 10.048 | 0.854 ± 0.113 | 0.474 / 1.270 | 4.257 |

The objective-specific actual minibatch losses (e.g., final seed-29 losses:
baseline 5.663, Hellinger 1.446) are **not directly comparable** across
loss definitions. They are recorded stepwise in `batch_losses.csv` and
plotted separately in `minibatch_loss.png`. The legacy `loss.png`
instead displays mean work on complete source splits.

One counterintuitive result is that **work standard deviation increases** for
every Hellinger-trained map even though held-out ESS and free-energy accuracy
improve. Raw variance therefore cannot be substituted for importance-weight
efficiency: the exponential estimator cares about the `exp(-w)`-weighted
tail, not the unweighted second moment alone. Final measured gradients
were finite and log-J values remained bounded across these test draws.
Near-constant Hellinger log-J is consistent with a relatively smooth,
diffuse transport rather than a density-matched bimodal mapping; this
is a hypothesis from summary statistics, not a proven architectural mechanism.

Optimizer-only CPU training time was ~0.41–0.44 seconds per model/seed
in the final runner (the end-to-end seeded experiment including diagnostics,
NPZ output and figures took ~1.2–1.5 seconds per model, excluding dependency
installation). Per-run comparisons should use evaluation counts first because
CPU timing at this scale is noisy.

### Stage A3 scientific decision

**Preliminary positive result for practical importance sampling**, not a
demonstration of full density matching or QM/MM transferability.
At matched energy budgets, the Hellinger alternative improves raw test ESS
and analytical Δf accuracy in all three seeds, avoids gross one-basin
concentration, and reproduces the target's reweighted basin observables.

Limitations: three seeds, a two-dimensional symmetric Gaussian mixture,
a comparatively small MAF, independent synthetic source configurations,
and no QM/MM Hamiltonian or MD autocorrelation. Both methods were trained
on the same seed-specific data and selected at the fixed final checkpoint,
without strength sweeps or validation-tuned checkpoint selection.
This is insufficient for a strong statistical or molecular conclusion.
The next *scientific* checkpoint is a verified real QM/MM potential/gradient
single-point calculation before transferring the objective to molecular data.
