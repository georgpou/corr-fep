# Stage A2 — Known-answer bimodal target, original loss only

This is the **diagnostic** preceding any modified optimization objective.
The reference is a two-dimensional standard Gaussian. The target is an
equal-weight mixture of two Gaussian components, centered at (-5, 0) and
(5, 0), with isotropic component width sigma = 0.7 and energy offset c=1.25.

The target reduced potential is
`u_B(y) = c - log[(exp(-|y-mu1|^2/(2*sigma^2)) + exp(-|y-mu2|^2/(2*sigma^2)))/2]`.
Its partition function is known without integration:
`Delta f = c - 2 log(sigma)` (both densities integrate over R^2).

For mapped density q_theta and target p_B, minimizing the original mean
generalized work minimizes **reverse KL**, D_KL(q_theta || p_B), up to
the constant exact free energy. Reverse KL is mode-seeking and may
preferentially concentrate mapped mass around one target peak. A large
importance ESS among sampled mapped configurations can therefore coexist
with an incorrect finite-sample free-energy estimate because unsampled
target regions do not contribute to those empirical weights.

The symmetry plane y[0]=0 divides the target probability exactly in half,
but the mapped *left-halfspace fraction* is not itself a measure of mode
occupancy. We separately count mapped points within a disk of radius
2*sigma around each Gaussian center, and outside both disks. At the
default separation these disks are disjoint. For a well-separated mixture,
the target probability within either disk is approximately
(1-exp(-2))/2 = 0.4323, with approximately exp(-2) = 0.1353 outside
both regions. These geometrical diagnostics expose target-peak undercoverage
without falsely assigning intermodal configurations to either mode.
Monitor them alongside held-out ESS, work spread, free-energy error and
Jacobian evolution.
All diagnostics are untrimmed. There is **no** penalty or altered architecture
relative to A1; the MAF and original BoltzmannKLDivLoss are reused.

Run three fixed independent seeds with disjoint iid training and held-out
draws per seed, 350 optimizer steps, 768 training and 4096 validation
configurations. The 0.5 target *halfspace* weight is an exact benchmark,
while each 2*sigma disk has an approximate target occupancy of 0.4323.
Neither statistic alone proves complete distributional overlap.
The experiment saves held-out source and mapped coordinates before and
after training, allowing the reported occupancy counts to be audited.

All substantial computation runs in the GitHub Actions workflow
`stage-a-non-gaussian.yml`. Scientific outputs live in workflow artifacts,
not in Git.

Interpret results before choosing a single loss modification. A poor
free-energy estimate together with high ESS and strong concentration in a
single target basin is consistent with practical missing-mode sampling,
not necessarily a failure of Jacobian regularization. An overlap/importance penalty could reinforce one-mode
collapse. Such penalties must not be chosen simply because they improve
training ESS.

## Executed three-seed cloud experiment (2026-10-10)

[Successful workflow](https://github.com/georgpou/corr-fep/actions/runs/38066112518),
[scientific artifacts](https://github.com/georgpou/corr-fep/actions/runs/38066112518/artifacts/11674893516).
All figures below are calculated from **genuinely trained upstream TFEP MAFs** and
4096 held-out iid samples per seed. Values are dimensionless (kBT).

Analytical exact Delta f = **1.96334988787746**.

| Seed | Step 100 validation ESS | Step 350 validation ESS | Step 350 Delta f | Step 350 left-halfspace fraction | Step 350 training mean work |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 819.4 | 910.2 | 2.0776 | 0.748 | 4.8567 |
| 29 | 830.2 | 144.9 | 2.3754 | 0.932 | 4.5486 |
| 43 | 754.5 | 61.0 | 1.8970 | 0.250 | 5.7791 |

The **seed-29** run demonstrates the intended pathology. Its training
mean work decreases steadily from ~9.15 at step 100 to ~4.55 by step 350;
meanwhile, held-out ESS deteriorates from ~830 to ~145 (out of 4096),
the mapped left-*halfspace* population grows to ~93%, and free-energy error
increases to ~0.412 kBT. The original run did not measure whether
mapped configurations actually occupied either Gaussian peak; the
corrected workflow adds that diagnostic and retains this initial
observation only as historical context. Seed 43 also exhibits deterioration of ESS and
mode balance, with a smaller final energy error. Seed 17 is less severe.

The seed-to-seed variability is scientifically relevant: asymmetric mapped
density can favor either halfspace, despite an exactly symmetric target.
The results establish a **controlled synthetic optimization/overlap divergence**,
not reproduction of the molecular pathology or evidence that ESS penalties
or Jacobian regularization will fix it. Invertible maps have full mathematical support; empirical
mode undercoverage here is finite-sample/practical, not literal zero support.

**Next modification to test:** a single objective-level intervention
against mode concentration with identical inputs and evaluated at matched
target-energy budgets; keep architecture unchanged. An ESS-only penalty
may improve within-mode weights without recovering the missing mode,
so success must require simultaneous improvement in held-out ESS,
mode balance and independently known free-energy accuracy.


## Follow-up review corrections

The first cloud execution is preserved above for provenance. The current
workflow reports both mapped **left/right halfspace balance** and **three-way
target-basin occupancy** (left disk, right disk, neither disk), plots mapped
validation coordinates, and saves paired source and mapped coordinates in
`validation_work.npz`. The two radii are fixed at 2*sigma, not fitted to
the observed data. Console `train_mean_w` describes the mean generalized
work over the training pool; `batch_losses.csv` contains the actual
optimized mini-batch loss. Every sample evaluated with `u_B` is included
in the target-evaluation counter. The config records the GitHub head SHA.
The original objective, architecture, seeds, training samples, and
optimization-step counts remain unchanged.
