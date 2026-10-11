# Issue 1: Is cached matched-water conditioning a valid TFEP change of variables?

Status (2026-10-11): **open scientific blocker for estimator claims**, not a confirmed failure of the three-update pilot. This note separates established numerical results from the unproved change-of-variables assumption. PR #14 remains a draft.

## Existing implementation and upstream comparison

- `scripts/stage_b_author_map.py::preprocess` selects/matches waters by aligning to the ligand scaffold, using the pinned paper driver's `find_ligand_neighbors` and `match_waters`. The selected row for each archived frame is saved in `matched_water`.
- `ConditioningContextFlow._pass` gathers coordinates from that frame's stored row, transforms the ligand, and restores the physical environment exactly. The same **stored** row is supplied for the checked inverse. The six-layer flow's measured 138-dimensional ligand Jacobian is therefore a **fixed-assignment conditional derivative**.
- The pinned paper source `run_tfep.py` similarly caches matching by trajectory sample index in `PermutationFlow`; this is not uniquely a Corr-FEP design choice.
- Verified numerical facts: sampled fixed-row forward/inverse, Jacobian, geometry and force gates passed after three QM/MM updates. The data do not show whether a freshly selected row at the mapped ligand coordinates would agree, whether a consistent globally invertible map exists, or whether an estimator with frame-correlated rows is unbiased.

## Mathematical issue

Let `q(x,e)` denote a source density for ligand coordinates `x` and fixed environment `e`. Write `F_a(x;e)` for the differentiable invertible ligand flow using assignment `a`, with Jacobian `J_a`. A normal TFEP estimator based on one invertible map `T` relies on

`E_q[ exp(-u_target(T(x,e)) + u_ref(x,e)) |det D_x T(x,e)| ] = Z_target / Z_ref`

with consistent reduced-energy units, and with environment coordinates unchanged.

If the assignment is `a=A(e)` and does **not** depend on the ligand coordinates, then `T(x,e)=(F_{A(e)}(x;e),e)` is a valid *conditional / triangular* map, provided each fiber `F_{A(e)}(.;e)` is bijective, measurable, and uses consistent reference geometry. Different environments may choose different water assignments; smoothness across environment assignment boundaries is not required for the ligand-block Jacobian if those boundaries have zero measure.

If instead `a=A(x,e)`, then `T(x,e)=(F_{A(x,e)}(x;e),e)` is a piecewise-defined transformation. Every `F_a` being separately bijective, and the inverse with the **original** `a` passing, does **not** prove that the combined `T` is bijective: branch images may overlap or leave gaps. Reassigning waters after moving the ligand may select a different branch; the reported block Jacobian alone cannot account for overlaps/gaps or associated multiplicities.

Equally, treating the archived frame index as an external label is not automatically a rescue. When the label/assignment is derived from the same source configuration, the conditional source distribution given the label is generally not the original `q`. Standard importance weights assuming full `q` cannot silently ignore that selection.

### Explicit counterexample (analytical; not a claim about the observed molecular frames)

Let the reference and target distributions both be uniform on [0,1], so the exact free-energy ratio is 1. Take two individually invertible maps `F_0(x)=x/2` and `F_1(x)=(x+1)/2` on [0,1], each with `|J|=1/2`. Choose label `A(x)=0` for `x<1/2` and `A(x)=1` otherwise. Their combined output occupies only [0,1/4) and [3/4,1], leaving a gap. The usual mapped importance factor `exp(-u_target+u_ref)*|J|` is 1/2 on every sample, giving ratio 1/2 instead of 1. Inverse and Jacobian checks *with the old label provided* pass exactly. This establishes that such checks are insufficient, not that the actual trained molecular map fails.

## Minimal next scientific checkpoints

1. **Trace assignment dependencies**: separate selection based on ligand-scaffold alignment/relative distances from any selection solely based on immutable environment, and document the exact source and target random variables. Confirm whether the same assignment can be computed using only the unchanged environment and a frozen reference template.
2. **Check independent-of-ligand conditioning**: if feasible, define a deterministic matched-water rule `A(e)` independent of mapped ligand coordinates, keep the original author's MixedMAF and energy engine unchanged, and verify forward/inverse by *independently recomputing* `A(e)` on each side (no index lookup). Verify ligand Jacobian and noncollinear reference geometry. If a water matching rule requires ligand coordinates, it does **not** satisfy this checkpoint.
3. **If retaining source-dependent assignment**: derive and implement a correct extended-state / conditional-density or multiplicity-aware estimator, or prove that the piecewise map is bijective almost everywhere. Simply retaining the original frame index in the inverse is insufficient.
4. **Then** compare on the existing bounded frames and only afterwards advance to 650/250 training/validation. Record the physical mapping/assignment differences separately from the model's learning effects. No Corr-FEP hypothesis inference until the estimator is justified.

## Decision

Do **not** silently replace author matching or claim bias has been measured. Prefer (2) if it preserves stable molecular geometry without costly engineering; otherwise stop and justify (3) before more expensive QM/MM training. The current three-update result remains a useful **fixed-assignment conditional model-integration gate**, not a proof of a valid global TFEP estimator.

## Follow-up: environment-only candidate (2026-10-11)

A **separate, unvalidated candidate** is now implemented in `scripts/stage_b_environment_only_map.py` and has a small six-frame no-QM diagnostic in `scripts/stage_b_environment_only_probe.py`. It does **not** replace the executed author baseline.

Important correction to the first proposed design: the archived trajectory is ligand-centered and solvent-wrapped by MDAnalysis. Matching water oxygens against absolute Cartesian sites would therefore retain *indirect ligand-coordinate dependence*. The candidate instead freezes the three author-selected water-oxygen reference sites on training frame 409 as **minimum-image offsets from a fixed non-solvent environment atom**. On each frame, candidate water oxygens are assigned to these three offsets using minimum-image squared distance and a one-to-one Hungarian match. The rule receives only unchanged environment coordinates and orthorhombic box lengths. It is recomputed in forward and inverse, rather than loaded by frame number; the six-layer TFEP architecture remains unchanged. This is a new, deliberately labelled methodological option, not a claim that the authors used this assignment.

**Caveats not yet resolved:** (1) the chosen non-solvent anchor might not be a physically good reference in an isolated ligand+water system (e.g. an ion can diffuse), (2) ligand-centered/wrapped *representation* and the reference geometry still need a coordinate-level demonstration of invariance, (3) nearest-site water switches may destabilize training or select nearly collinear oxygen triples, (4) the periodic rule assumes orthorhombic cells, (5) the six-frame numerical probe has not yet been run in Actions, and (6) no free-energy/ESS claim follows from introducing this class. The existing pilot is unchanged. These checks must precede physical training, and a failed probe should reject the option rather than forcing an interpretation.
