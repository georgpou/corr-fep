# Stage A / checkpoint A1 — Gaussian MAF baseline

This benchmark is **genuine differentiable training**, but **not QM/MM**. It
deliberately isolates the correctness of the original TFEP v0.2 MAF layer,
Boltzmann-KL objective, map Jacobian and importance estimator from expensive
physical energy evaluations.

## Distribution and exact answer

Reference: `u_A(x) = x^T x / 2` in two dimensions (normalized by its
partition function). Target:

`u_B(y) = (y-mu)^T K (y-mu)/2 + c`,

with `mu = (2, -1.5)`, `K = [[2.5, 0.7], [0.7, 0.9]]`, and `c=1.25`.
With identical integration domains, the exact reduced free energy is

`Delta f = c + 0.5 log det K`.

The two Gaussians have displaced means, unequal widths, and nonzero
correlations. The original identity mapping consequently has poor overlap.

The map uses **upstream** `tfep.nn.flows.MAF` with the same alternating
MADE ordering and affine transformer used by the authors'
`CartesianMAFMap.configure_flow()`. The optimization uses **upstream**
`tfep.loss.BoltzmannKLDivLoss`, equivalent to mean generalized work up
to the fixed reference term, plus AdamW / ExponentialLR. Two MAF layers
and 32-unit hidden layers are used instead of the original six; the
learning rate and optimizer steps are reduced-system choices, **not a
reproduction of the molecular hyperparameters**.

Samples are iid by construction, with independent seeds for the reference
training and validation arrays (both obtained from successive draws with the
same generator). The validation array is never used for gradients.
Training rows summarize a fixed subset of the training pool.

## Run

GitHub Actions: **Stage A - MAF Gaussian diagnostics**, or locally:

```bash
python scripts/stage_a_gaussian.py --steps 250 --seed 17 --out results/stage-a/baseline
```

The manually dispatchable action runs on an Ubuntu CPU runner, storing
`config.json`, `environment.txt`, `metrics.csv`,
`batch_losses.csv`, `summary.json`, `timing.json`,
`validation_work.npz`, and the four requested plots.

`metrics.csv` reports train and independent validation mean work,
variance, log-domain raw ESS, free-energy estimate/error, Jacobian
quantiles and standard deviation, and last-batch gradient norm.
The final flow is additionally checked for inverse consistency and
agreement with an autograd full Jacobian determinant at a sample.

The raw ESS is **not** an autocorrelation-adjusted effective sample size,
though the synthetic inputs are iid. It cannot by itself exclude missing
target modes. A correct exact free-energy estimate on this deliberately easy
model is a *numerical checkpoint*, not evidence of an improvement to the
molecular problem.

Next: once this baseline has produced inspectable cloud artifacts, test
**one** training-loss modification at matched source samples, architecture,
steps, and target evaluations.

## First cloud execution (2026-10-10)

[Successful Stage A CPU run](https://github.com/georgpou/corr-fep/actions/runs/38065373970)
([full experiment artifact](https://github.com/georgpou/corr-fep/actions/runs/38065373970/artifacts/11674626734)), seed 17, 250 AdamW steps,
2,048 independent held-out reference samples. Results are untrimmed:

| Validation metric | Step 0 | Step 250 |
| --- | ---: | ---: |
| Raw importance ESS / 2,048 | 56.5247 | 2042.1550 |
| Mean generalized work (kBT) | 5.80673 | 1.53468 |
| Generalized-work SD (kBT) | 4.10718 | 0.05318 |
| Delta f estimate (kBT) | 1.72541 | 1.53327 |
| Abs. error from analytical Delta f (kBT) | 0.19276 | 0.0006105 |
| Mean log Jacobian | 0 | -0.28662 |
| SD log Jacobian | 0 | 0.00479 |

Analytical Delta f = **1.53265690452503 kBT**. Forward/inverse coordinates
agree within 3.1e-16 and the autograd determinant agrees with the logged
Jacobian to 1.2e-16. The run evaluated the differentiable target for **109,056
configurations**, including training and all diagnostics; the timed training
plus diagnostics and plot generation took approximately one second excluding
Python initialization, environment setup and installation. The numerical
training is fast; startup dominates this particular GitHub Actions job.

This is a **positive implementation check** on a tractable Gaussian. It is
not an observed training pathology and is not evidence of an improved loss.
The source sampling and target both have full Gaussian support, but finite
held-out ESS is not a completeness guarantee for more difficult targets.
