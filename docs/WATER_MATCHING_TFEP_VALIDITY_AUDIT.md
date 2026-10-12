# Water-matching dependency and TFEP change-of-variables audit

Status: **source trace completed; global estimator validity not established**. This is a scientific acceptance gate, **not** evidence that the observed free-energy estimate is biased. No change to the merged training baseline is made by this audit.

## Question and scope

Does the authors' ligand-aligned matched-water conditioning, as retained by Corr-FEP's six-layer ligand-only MixedMAF, define a change of variables valid for targeted free-energy perturbation (TFEP)? The numerical training and fixed-index reconstruction succeed, but those properties alone do not answer the question.

The uploaded *Multimap Targeted Free Energy Estimation: A Mechanical Embedding QM/MM Scheme for Protein–Ligand Affinities* preprint, §2.1, defines an **invertible** map \(M\), the Jacobian-weighted work (eqs. 2–3), and the learning loss (eq. 4). Section 2.5 describes ligand-centered water selection and matching by the Hungarian algorithm. These descriptions are consistent with the released driver but do not supply a proof for configuration-dependent map selection.

We inspected the pinned sources used by the completed training workflows, rather than assuming that today's repository HEAD is the publication code:

- [Authors' driver, pinned at b8dfd897](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026/blob/b8dfd897ebd914b4e592221710b397a2758353ce/run_tfep.py)
- [TFEP MixedMAF, pinned at 0bc369a](https://github.com/andrrizzi/tfep/blob/0bc369a1194c688bb6ac997cc6b7c2db2b063c9e/tfep/app/mixedmaf.py)
- [Merged Corr-FEP map](https://github.com/georgpou/corr-fep/blob/79da7f7915ac4929bcbdde6817a58fe9d87d831e/scripts/stage_b_author_map.py)
- [PR #15 numerical reproduction](https://github.com/georgpou/corr-fep/pull/15), including completed 650/250 training and original five-checkpoint replay
- [PR #14 archived mathematical analysis](https://github.com/georgpou/corr-fep/blob/archive/pr14-author-recovery-2026-10-11/docs/B25_CACHED_MATCHING_ESTIMATOR_VALIDITY.md)

## Exact source dependency trace

| Source location | Observed behavior | Mathematical consequence |
|---|---|---|
| [`find_ligand_neighbors` L211–329](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026/blob/b8dfd897ebd914b4e592221710b397a2758353ce/run_tfep.py#L211-L329) | Counts/selects water oxygens using `around cutoff (ligand_selection)`; determines a ligand-dependent reference configuration | The original candidate selection depends on ligand coordinates |
| [`match_waters` L369–401, L461–521](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026/blob/b8dfd897ebd914b4e592221710b397a2758353ce/run_tfep.py#L369-L521) | Selects waters around the **whole ligand**, aligns each frame by the ligand scaffold (`alignto`), assigns oxygens/hydrogens with Hungarian matching, and stores per-frame atom indices | The selected assignment is in general a function \(A(x,e)\), not demonstrably a function of unchanged solvent coordinates \(e\) alone |
| [Authors' `PermutationFlow` L884–921](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026/blob/b8dfd897ebd914b4e592221710b397a2758353ce/run_tfep.py#L884-L921) | Looks up `matched_solvent_dof_indices[trajectory_sample_index]` instead of recomputing the assignment from the current positions | The supplied archive label is needed to reconstruct the chosen map branch |
| [Authors' `run_estimate` L1436–1468](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026/blob/b8dfd897ebd914b4e592221710b397a2758353ce/run_tfep.py#L1436-L1468) | Uses target energies, the Jacobian and indexed MM reference energies in the ordinary FEP estimator | No explicit conditional-source-density, branch multiplicity or variable-assignment correction is apparent in this estimator |
| [Corr-FEP `preprocess` L75–129](https://github.com/georgpou/corr-fep/blob/79da7f7915ac4929bcbdde6817a58fe9d87d831e/scripts/stage_b_author_map.py#L75-L129) | Calls the adapted upstream ligand-aligned matching helper and freezes a water-atom table for training and validation | The original dependence survives the clean transplant |
| [Corr-FEP `ConditioningContextFlow` L133–165](https://github.com/georgpou/corr-fep/blob/79da7f7915ac4929bcbdde6817a58fe9d87d831e/scripts/stage_b_author_map.py#L133-L165) | Both forward and inverse select a **saved frame index**; ligand mapping is applied using gathered water context; output environment is restored without displacement | A fixed-assignment conditional inverse is checked. An inverse on physical coordinates alone, with recomputed assignment, is not provided |
| [Pinned MixedMAF `configure_flow` L275–357](https://github.com/andrrizzi/tfep/blob/0bc369a1194c688bb6ac997cc6b7c2db2b063c9e/tfep/app/mixedmaf.py#L275-L357) | Uses a Cartesian/mixed-coordinate invertible flow conditioned on origin/axes atoms | Invertibility of the smooth **fixed-context** flow does not establish invertibility of a source-dependent switch between such flows |

The Corr-FEP branch **maps all ligand atoms** and restores the MM environment. The matching algorithm uses ligand atoms (including the scaffold) to select/align water contexts; the implementation does not constrain those ligand atoms to remain fixed under the map. Hence the key sufficient condition “assignment depends only on untransformed coordinates” is **not established**.

Important nuance: the chosen water identity may differ between physical configurations without invalidating TFEP **if** the selection depends only on unchanged coordinates. Statistical correlation between ligand and solvent is not itself the problem; **functional dependence on the coordinates being mapped** is.

## What the existing numerical gates actually establish

- The original baseline and clean PR #15 reproducibly completed **21 physical updates on the same 650 training frames** and evaluated **250 held-out validation frames**, with essentially identical results.
- For the cached assignment, the inverse closes numerically (maximum training-batch error approximately 3.5 × 10⁻¹² Å); log-Jacobian closure is approximately 2.3 × 10⁻¹³ and fixed-environment displacement is zero.
- Independent local Cartesian determinant checks and a strict five-checkpoint replay were documented on the preceding branch.
- The 250-frame mapped importance-weight ESS is about **1.95** (not an autocorrelation-corrected effective sample size); the free-energy estimate is not statistically converged.

None of these establishes injectivity/surjectivity of the physical map that would **select a branch again from a transformed configuration**. Treating the original sample number as an external conditioning variable is not a substitute: its matching table was constructed from the original samples.

## A decisive analytical counterexample (not molecular evidence)

Consider source and target **both** the standard normal density \(\phi\) on \(\mathbb{R}\). The true partition-function ratio is \(Z_2/Z_1=1\). Let the individually invertible smooth branches be

\[
F_0(x)=x,\qquad F_1(x)=x-1,
\qquad A(x)=\begin{cases}0,&x<0,\\1,&x\geq0.\end{cases}
\]

Each branch is a globally invertible diffeomorphism with unit Jacobian. But choosing the branch from the source configuration gives

\[
T(x)=F_{A(x)}(x)=
\begin{cases}x,&x<0,\\x-1,&x\geq0,\end{cases}
\]

which double-covers the interval \([-1,0)\). Supplying \(A(x)\) to the inverse produces exact reconstruction and perfect Jacobian closure, yet the ordinary TFEP expectation is

\[
\mathbb{E}_{\phi}\!\left[\frac{\phi(T(x))}{\phi(x)}|T'(x)|\right]
=\int_{-\infty}^0\!\phi(x)\,dx+\int_0^\infty\!\phi(x-1)\,dx
=\frac12+\Phi(1)
\approx1.3413447461\ne1.
\]

Thus **even exact inverse and Jacobian diagnostics for every frozen branch can coexist with a biased usual estimator**. This refutes an inference from those checks to unbiasedness; it does **not** establish a numerical bias in Corr-FEP or the paper. The toy branch switch is not intended as a molecular model.

Reproduce the number with Python's standard library:

```python
from math import erf, sqrt
ratio = 0.5 + 0.5 * (1.0 + erf(1.0 / sqrt(2.0)))
print(f"{ratio:.10f}")  # 1.3413447461, not 1.0
```

## Sufficient resolution, and what remains to be tested

Write the full coordinates as mapped ligand \(x\) and unchanged coordinates \(e\). If a deterministic assignment \(A(e)\) depends **only** on \(e\), and every conditional flow \(F_{A(e)}(\cdot;e)\) is bijective on the relevant ligand domain, then

\[
T(x,e)=(F_{A(e)}(x;e),e)
\]

is a triangular conditional map. At almost every point its relevant Jacobian determinant is \(|\det\partial_xF_{A(e)}|\), and the standard TFEP identity follows by changing variables in \(x\) separately for each fixed \(e\). Discontinuities in \(e\) at assignment boundaries need appropriate measure-zero regularity, but do not by themselves introduce a ligand-branch multiplicity.

This is a **sufficient** condition, not the only possible solution. If selection is truly \(A(x,e)\), the options are a mathematical proof that the resulting piecewise map is bijective almost everywhere, or a rigorously derived expanded-state/conditional estimator. The released code and inspected preprint do not provide either. Solvent indistinguishability alone does not establish the needed property in the fixed-labelled coordinates used by this ligand-only implementation.

### Next low-cost scientific checkpoint — *before QM/MM retraining*

1. **Dependency/invariance probe on actual archived frames (no QM/MM):** for selected train and validation frames, recompute the upstream ligand-scaffold/Hungarian assignment before and after representative ligand-only perturbations at *identical unchanged environment positions* (including controlled cases near assignment boundaries). Record switching and alignment choices. Observed switches would falsify the assumption that the current assignment is invariant; absence of observed switches is not a global proof.
2. **Construct one genuinely invariant conditioning rule:** it must be computable solely from coordinates that the flow demonstrably leaves unchanged, including after trajectory centering/wrapping. Either use a physically stable environment-only reference or explicitly leave a defined ligand anchor/scaffold fixed and match relative to that fixed anchor. These are **new experimental variants**, not a claim of faithful author reproduction. The earlier environment-anchor candidate in archived PR #14 failed a geometry probe and must not be silently adopted.
3. **No-QM acceptance gates:** independently recompute the rule in forward and inverse without a source frame identifier; test diverse train/validation configurations, non-collinearity, physical environment invariance, ligand inverse closure and an independent Cartesian ligand-block Jacobian. Document which degrees of freedom remain mapped. If these pass, then consider a small physical pilot and equal-budget baseline comparison.
4. **Alternative if retaining ligand-dependent assignments:** present a complete partition-of-domain/bijectivity proof or correctly weighted lifted-state estimator, including the induced conditional source law, *before* treating its output as a free-energy estimate.

## Decision

**Keep PR #15's merged baseline unchanged and honestly labeled as a reproducible training-pipeline integration.** Do not claim that its ESS increase proves an unbiased or converged TFEP correction. The **original paper/source does not currently provide an identifiable rescue for the global change-of-variables question**, but we have also not shown that the observed molecular estimator is biased. The next implementation should be a small, no-QM invariance experiment; do not expend another 650/250 physical training run until the mathematical structure is defensible.

Historical experiments and full raw investigation record remain on [the preserved PR #14 archive](https://github.com/georgpou/corr-fep/tree/archive/pr14-author-recovery-2026-10-11).
