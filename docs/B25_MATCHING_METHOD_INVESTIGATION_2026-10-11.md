# B2.5 investigation: paper water matching vs environment-only candidate

Date: 2026-10-11. Scientific report; **no source-code changes or new QM/MM calculations**. PR #14 remains draft. Sources: the attached preprint Gay et al., *Multimap Targeted Free Energy Estimation: A Mechanical Embedding QM/MM Scheme for Protein-Ligand Affinities*, ChemRxiv 2026 (especially §2.5); pinned authors' `run_tfep.py` at `b8dfd897ebd914b4e592221710b397a2758353ce`; Corr-FEP scripts on PR #14; completed Actions artifact `11713412567` from run `38170360939`; previously committed author pilot report.

## Findings

**These are different procedures.** The published paper and `run_tfep.py` select waters within ligand-centered mapped/conditioning cutoffs, **align the ligand scaffold to a reference frame**, solve a Hungarian assignment for oxygen positions, then separately pair hydrogens. Stable reference atoms are selected by small fluctuations of distances to the ligand and checked for collinearity. Paper §2.5 explicitly tested `(R_m,R_c)=(0,3.5) Å`, corresponding to our ligand-only *mapped* region with water conditioning. Its other paper cutoffs differ, and the source's default invocation is `(3.5,6.0) Å`, not this chosen option.

The recovered Corr-FEP *author-conditioned* adapter invokes the upstream match helper with small, auditable modifications and uses its ligand-aligned cached assignments; the 900-frame author-reference geometry gate reported **minimum reference-angle sine 0.383362, zero collinear frames**. It completed three actual physical QM/MM training updates, but its 16-frame validation ESS did not improve. This is a successful bounded numerical integration gate, not completed TFEP unbiasedness validation.

By contrast, the exploratory `stage_b_environment_only_map.py` **does not use ligand alignment or ligand-relative candidate selection**. It freezes 3 template-site offsets from an arbitrarily selected *non-solvent* environment atom (index **8404** in the 900-frame artifact) in one training configuration, then selects any 3 waters in the box closest to those template offsets relative to that anchor. It is deliberately a different proposal attempting to ensure assignment depends only on unchanged environment coordinates.

The 900-frame artifact makes the failure mode tangible:

| Metric | Experimental environment-only candidate |
|---|---:|
| frames analyzed | 650 training + 250 validation |
| median largest reference-oxygen-to-ligand nearest distance | **17.83 Å** |
| frames with at least one selected reference oxygen > 6 Å from ligand | **885/900** |
| frames with one > 10 Å from ligand | **804/900** |
| maximum selected reference oxygen-to-ligand distance | **31.18 Å** |
| raw Cartesian edge of reference triangle | up to **58.96 Å** |
| frames where raw Cartesian triangle crosses periodic images | **250/900** |
| minimum raw Cartesian reference-angle sine | **0.01204**, compared with **0.38336** for author-matched map |
| raw reference-angle sine <0.05 | **20/900, all 20 periodic-image-spanning** |
| change in selected water *identity* between adjacent archived frames | **898/898 comparisons** |

The six-frame forward/inverse/Jacobian checks and the script's weak full-scan threshold passed. However, **the candidate is usually referencing waters well outside the intended ligand shell**. Assignment is strongly determined by the relative motion of anchor 8404 and the ligand: a good water assignment *relative to the environment anchor* is not a good assignment *near the ligand*. The long Cartesian edges are in part **image-representation artifacts**, rather than physical 59 Å oxygen separations. They matter to the current model because raw Cartesian reference coordinates are fed directly to MixedMAF. The near-collinear count is also a raw-coordinate statistic, and **all 20 such cases coincide with image spanning**; it must not be described as physical 3-point collinearity without unwrapping. Water identity changes are expected for equivalent solvent molecules and **are not by themselves evidence of failure**.

The preprint describes the reference frame as determined by the first three ligand atoms (§2.5), while the checked released `run_tfep.py` implementation explicitly calls `select_reference_frame_atoms(... atom_indices=conditioning_atom_indices, ...)`, then passes those conditioning atoms as `origin_atom` and `axes_atoms` to MixedMAF. **This is a source/paper description discrepancy worth recording**, not automatically proof that the paper's results are incorrect. The actual pinned executable code is our primary implementation reference.

## Recommendation

**Do not 'fix' the authors' stable ligand-aligned matching with the environment-anchored experimental candidate.** Remove that candidate from any path that might be treated as the validated baseline. Preserve the author-conditioned implementation and its existing training/pilot evidence separately.

For the first estimator-valid and broadly reusable benchmark, prefer the original paper's **ligand-only, no-water-conditioning control** implemented with the **authors' six-layer MixedMAF**, mapping the QM/ligand atoms and preserving the MM environment. No water identity matching or frame-index-conditioned map is needed. The TFEP map is then a single explicitly composed invertible coordinate transformation, subject to ordinary spline/inverse and Jacobian checks. This is a scientifically meaningful *paper-supported control* but may be less efficient than solvent-conditioned maps; verify actual ESS, not only reduced-work standard deviation. Do **not** reuse the earlier one-layer ligand-only runner whose trained inverse failed; use the six-layer architecture and a bounded real-energy update gate.

For **general data-set coverage**, distinguish mathematical validity from statistical efficiency. Extend the flow only as needed using (i) a **permutation-invariant, smooth solvent/environment descriptor** that does not depend on actively transformed coordinates of a given invertible layer; (ii) an invertible **masked coupling / autoregressive** update with context computed from the **unchanged coordinate partition of that layer**, followed by complementary layers to allow all ligand coordinates to move; and (iii) a reversible Cartesian fallback for linear, monoatomic, small, covalent, or metal-containing QM regions for which the original Z-matrix/reference frame is undefined. Never silently use arbitrary ion/water reference anchors; never depend on frame IDs to define a production estimator. A more conservative intermediate implementation freezes a non-collinear ligand scaffold triad as context and transforms remaining ligand DOFs, recomputing any local water assignment relative **only to the frozen scaffold**, but this sacrifices those fixed triad DOFs in a single layer.

**Research caveat about author's conditional model:** their mapping uses cached per-frame assignments derived from ligand coordinates. For a fixed assignment, numerical inverse and logdet checks are valid. These checks alone do not prove that the composite assignment-dependent mapping is globally bijective in the full labeled phase space. This is a theoretical question requiring proof, a quotient/symmetry-aware justification, or a different estimator; **we have not demonstrated bias** in the authors' data or our pilot. Do not conflate this question with the experimentally rejected environment-only candidate.

## Next bounded checkpoints before full QM/MM training

1. Run six-layer *ligand-only, no-water-conditioning* initialization and geometry/inverse/Jacobian checks across representative frames; no environment anchor or water reordering.
2. Run up to three physical QM/MM updates with the **same** correctly validated Hamiltonian and optimizer settings; recheck trained inverse/force derivatives. Compare paired 16+ validation work values/ESS with identity and author-conditioned controls.
3. If the broad-control works but has poor overlap, introduce layerwise masked, permutation-invariant environment conditioning while preserving bijectivity by construction. Fail explicitly on unsupported topology rather than silently picking degenerate reference triples.

No full 650/250 training, untested global estimator claim, or PR merge at this stage.
