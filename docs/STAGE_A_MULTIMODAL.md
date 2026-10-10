# Stage A2 — Known-answer bimodal target, original loss only

This is the **diagnostic** preceding any modified optimization objective.
The reference is a two-dimensional standard Gaussian. The target is an
equal-weight mixture of two Gaussian components, centered at (-5, 0) and
(5, 0), with isotropic component width sigma = 0.7 and energy offset c=1.25.

The target reduced potential is
`u_B(y) = c - log[(exp(-|y-mu1|^2/(2*sigma^2)) + exp(-|y-mu2|^2/(2*sigma^2)))/2]`.
Its partition function is known without integration:
`Delta f = c - 2 log(sigma)` (both densities integrate over R^2).

A flexible map minimizing the forward KL may preferentially populate one
mode. Large importance ESS within mapped configurations can then coexist
with an incorrect target free-energy estimate: absent modes are invisible
to a finite-sample ESS metric. For the symmetric target the correct left-
versus-right half-space probability is 0.5. Monitor held-out map occupancy,
ESS, work spread, free-energy error and Jacobian evolution concurrently.
All diagnostics are untrimmed. There is **no** penalty or altered architecture
relative to A1; the MAF and original BoltzmannKLDivLoss are reused.

Run three fixed independent seeds with disjoint iid training and held-out
draws per seed, 350 optimizer steps, 768 training and 4096 validation
configurations. The 0.5 target mode weight is an analytical benchmark; a
map's empirical left fraction is a diagnostic, not proof of support.

All substantial computation runs in the GitHub Actions workflow
`stage-a-non-gaussian.yml`. Scientific outputs live in workflow artifacts,
not in Git.

Interpret results before choosing a single loss modification. A poor
free-energy estimate together with high ESS and one-mode mapping is
diagnostic of missing target modes, not necessarily a failure of Jacobian
regularization. An overlap/importance penalty could reinforce one-mode
collapse. Such penalties must not be chosen simply because they improve
training ESS.
