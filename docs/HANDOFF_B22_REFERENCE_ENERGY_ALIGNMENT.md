# Corr-FEP — Agent handoff: Stage B2.2 reference-energy and coordinate-precision audit

> **Historical handoff, written 2026-10-10; superseded.** The statement below that B2.2 was not validated describes the state *when this handoff was written*. The B2.2a/b/c scientific checks subsequently passed. For current methods, run links, remaining limits, and PR #11 review state, consult [the completed Stage B2.2 audit](STAGE_B_REFERENCE_ENERGY.md). This document is retained as a dated research handoff, not an up-to-date status report.

**Date:** 2026-10-10
**Repository:** https://github.com/georgpou/corr-fep
**Base / merged main SHA:** `12d34abd32ea3f7312b0f7bd98c065046ec833df`
**Prepared successor branch:** `stage-b/reference-energy-alignment`
**Completed PR:** https://github.com/georgpou/corr-fep/pull/10 (squash merged with explicit user permission)
**Scientific tracking issue (KEEP OPEN):** https://github.com/georgpou/corr-fep/issues/9
**B2.1 source and accepted evidence:** [documentation](STAGE_B_MULTIFRAME.md), [scientific run 38089140014](https://github.com/georgpou/corr-fep/actions/runs/38089140014), [its artifact](https://github.com/georgpou/corr-fep/actions/runs/38089140014/artifacts/11683012876).

## Mission and boundaries

Complete Stage **B2.2: time-matched classical reference-energy alignment** against the deposited `ener.xvg` and determine whether XTC/GRO coordinate precision induces a configuration-dependent bias of practical consequence. Establish what energy observable in the dataset is comparable to the implemented GROMACS full-MM potential **before** training or free-energy comparisons.

Do not repeat B2.1 as if still broken: four genuine subtractive MM/QM-MM evaluations and gradients, and production-frame finite differences **passed**. B2.2 has NOT yet been validated. Do not infer free energies, effective sample sizes, force-field correction accuracy, or full reproduction from B2.1.

The user's research preference is scientific checkpoints using actual Hamiltonians, interpretable limited experiments, and avoiding sprawling unit-test suites, speculative abstractions, or unnecessary refactoring. You may commit, open PRs, review/fix code, and run real Actions workflows. **Ask user for explicit permission before merging each PR.** Do not automatically merge.

## Physics and pinned experiment

The subtractive mechanical embedding Hamiltonian is

```text
U_QMMM(x) = U_MM_full(x) - U_MM_ligand(x_L) + U_GFN1-xTB_ligand(x_L)
```

- GROMACS 2023.3 double-precision CPU pinned upstream commit `5e5ea27d8e594124d229a726616b7f6473f2c935`; original research commit `b8dfd897ebd914b4e592221710b397a2758353ce`.
- TFEP commit `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e`; tblite 0.3.0, GFN1-xTB, original neutral 46-atom ligand.
- Zenodo: https://doi.org/10.5281/zenodo.21919704, archived ZIP `https://zenodo.org/records/21919704/files/alchemical_tfep.zip?download=1`, `lig_p38a_2g` water repeat-0.
- Full system: 8420 atoms. Deposited archive is approximately 49 GB; use HTTP Range / `remotezip` to read individual selected ZIP members. **Never download the entire archive.**
- Original paper used 25 repeats; deposited data available to this project is only repeat-0. Never claim full paper reproduction.
- Production XTC samples: original indices 401 / 1067 / 1733 / 2400 at 2005 / 5335 / 8665 / 12000 ps; first 2 ns are excluded.
- Existing scripts: `scripts/stage_b_qm_singlepoint.py`, `scripts/stage_b_mechanical_embedding.py`, `scripts/stage_b_multiframe.py`. Scientific workflow: `.github/workflows/stage-b-multiframe.yml`.
- Existing documentation: `docs/STAGE_B_MULTIFRAME.md`, `docs/STAGE_B_MECHANICAL_EMBEDDING.md`, `docs/REPRODUCIBILITY.md`; historical B2 failure handoff: `docs/HANDOFF_B2_FRAME_ENERGY_DISCREPANCIES.md`.

## Evidence already established by B2.1

The original failed ~17.556 Å terminal XTC/GRO comparison was a **code-order bug**: reference GRO was transformed with `unwrap(ligand) → center_in_box(ligand) → wrap(environment)`, whereas XTC was previously audited *before* installing the same transforms. Fixed and verified.

After consistent transforms, GRO versus terminal XTC at 12000 ps has:
- Minimum-image RMS displacement 0.00011077 Å; post-registration RMS 0.00010984 Å.
- Post-registration maximum discrepancy 0.01000404 Å, with exactly **one atom above 0.001 Å**: **zero-based atom 463**, `SOL` resid 141, oxygen `OW`.
- Its XTC-minus-GRO y difference is +0.01001358 Å; every other atom differs by less than ~0.000033 Å.
- Box-length mismatch ~0.0000191 Å; largest periodic pair-distance mismatch within ligand ~0.00000502 Å.
- The GRO title is `Ligand in water` and has **no timestamp**; XTC endpoint metadata is 12000 ps. Same physical configuration is strongly supported by geometry but GRO time is not independently recorded.

Under **the same generated full/ligand TPRs and runtime**, endpoint energy values (kJ/mol):

| Term | GRO | XTC endpoint | XTC − GRO |
| --- | ---: | ---: | ---: |
| Full MM | -116802.61791405082 | -116803.62996742321 | -1.01205337240 |
| Ligand MM | -971.022166844934 | -971.023118905871 | -0.00095206094 |
| GFN1-xTB | -216325.75103968772 | -216325.75174184845 | -0.00070216073 |
| Combined | -332157.3467868936 | -332158.3585903658 | -1.01180347218 |

A **diagnostic-only** counterfactual resets atom 463 in the XTC endpoint to its GRO position without touching other positions and shifts full-MM energy by +1.06320969 kJ/mol, leaving hybrid-minus-GRO +0.05115632 kJ/mol. Its y gradient in original XTC gives a first-order change of ~−0.9955 kJ/mol for +0.0100 Å. Thus one atom causally dominates the discrepancy. The precise file-generation/rounding origin and small residual are **not independently explained**. Do NOT use the hybrid as a real trajectory sample or training input; do NOT insert compensating energy offsets.

Four real production-frame evaluations of full MM − ligand MM + GFN1-xTB passed. Frame-401 complete-Hamiltonian central FD errors (h=0.002 Å): ligand 0.02443, interfacial environmental atom 0.00320 kJ/mol/Å; both below B1 threshold 0.8. Environmental combined gradients equal full-MM environment gradients to stored precision. Direct vs wrapper quantum gradient errors below 1.7e-8 kJ/mol/Å.

The PR #10 deep audit tightened the translation-aware terminal alignment gate to post-registration RMS 0.001 Å, max individual displacement 0.02 Å, ligand-pair deviation 0.001 Å, interfacial nearest-ligand distance max 0.02 Å, and periodic box tolerance 0.01 (Å/degrees). These are **coordinate correspondence tolerances**, NOT acceptance of energy/weight accuracy. Final stricter scientific run [38089140014](https://github.com/georgpou/corr-fep/actions/runs/38089140014) passed. Run provenance and scientific artifact should be inspected rather than treating generic CI as proof.

## Next scientific work — execute in priority order

### B2.2a — establish the exact `ener.xvg` observable and time correspondence

1. Locate archived `lig_p38a_2g/eq/water/repeat-0/ener.xvg` (verify exact member name via ranged ZIP listing; do not assume the file exists or contains any specific terms). Retrieve only that small member and archive provenance hashes. Inspect header legends, energy units, times, record counts, recorded energy fields, and trajectory time sequence.
2. Inspect upstream code and original MD configurations to determine what `ener.xvg` actually represents (total potential vs individual terms, whether bias/restrained energies, kinetic terms, or energy re-evaluation). GROMACS `gmx energy` outputs and `ener.xvg` are not necessarily equivalent to the re-evaluated energies from `GROMACSPotential`. Explicitly specify/justify term mapping and units.
3. Check whether times match selected original XTC frames exactly (including 2005 ps and terminal 12000 ps). Avoid nearest-neighbor frame misalignment and be explicit about any missing values. Clarify equilibration vs production, and the fact GRO title has no independent time.
4. Where possible, independently calculate the matching potential/terms using the pinned generated full-system TPR and consistent frame box and precision; compare apples-to-apples. Consider `gmx energy`, `gmx mdrun -rerun` and energy-term extraction as the narrowest useful experiment; record definitions, exact commands and build versions. Distinguish deposited GROMACS 2021.6 double vs project 2023.3 double, and differences in compiled TPR or runs when relevant.

### B2.2b — determine precision sensitivity and numerical implications

1. Use a **small, interpretable sample** (e.g. the same four archived production frames plus available relevant reference endpoints). Do not manufacture matching GRO counterparts for intermediate XTC frames. If accurate alternate coordinate snapshots are unavailable, say so.
2. Test within one engine/runtime and fixed TPR whether the observed endpoint difference is attributable to input precision, coordinate formatting/rounding, one-atom anomaly or box differences; separate these perturbations carefully. The corrected atom-463 hybrid is *diagnostic only*.
3. Where data permit, decompose full-MM energy into bonded, Coulomb, Lennard-Jones, PME correction and dispersion contributions; distinguish true physical energy changes from mismatched reference observables. Quantify residual after the single-atom change and whether it is plausibly rounding or box-related; avoid claiming unmeasured certainty.
4. Evaluate whether observed changes affect **configuration-dependent reduced potential differences** (`Δu=βΔU`), which would influence reweighting, rather than assuming an absolute constant shift. Do not compute ESS, ΔG, or reweighting corrections until exact comparable energy definitions and sufficient paired data exist.
5. If uncertainty remains unresolvable because of archive precision/missing independent snapshots, report this as a *scientific limitation with a quantified bound*, rather than calibrating an offset to force agreement.

### B2.2c — checkpoint and documentation

Produce a compact `results/stage-b/reference-energy/` artifact with:
- archive member names/hashes and retrieval method;
- reference legends, units, energy terms and matched times;
- GROMACS/TFEP versions, archived and newly generated TPR hashes, precise energy command definitions and configuration provenance;
- a readable per-frame table of deposited reference observables vs re-evaluated comparable MM values and residuals;
- selected precision/energy-term sensitivity diagnostics with interpretation;
- explicit scientific pass/fail and outstanding limitations.

Implement only minimal code and a focused GitHub Actions workflow using the existing double-precision GROMACS cache. **New branch workflows will not automatically run on push unless branch filters are updated**; either add `stage-b/reference-energy-alignment` to a dedicated workflow trigger or use `workflow_dispatch` at this branch and confirm the workflow SHA. Maintain artifacts even if scientific checks fail.

Keep issue #9 **open** until exact time/energy meaning and precision effects are sufficiently measured and documented. Update issue #9 with verifiable run/artifact results, not conjecture. When the checkpoint is scientifically defensible, update `docs/`, open a small PR targeting `main`, critically review it, and request user approval **before every merge**. B2.3 backend/training work should not be advanced on unverified B2.2 claims.

## Explicit pitfalls from prior work

- Do not compare already-transformed GRO to untransformed XTC again.
- XTC does not contain atom names; loading both files with the same TPR and comparing names is tautological.
- Minimum-image displacement alone does not remove arbitrary common translation; use robust registration and periodic pair-distance checks where needed.
- Do not equate 0.01 Å with an inconsequential MM energy error: **one water O at 0.01 Å caused >1 kJ/mol sensitivity** here.
- Different absolute quantum reference offsets are not free energies.
- Do not make the workflow green by weakening acceptance thresholds, reusing a stale artifact, or silently substituting proxy MM/QM.
- Do not overtest peripheral edge cases; spend time on real physical numerical comparisons.

## Source links

- [Main repository](https://github.com/georgpou/corr-fep)
- [Merged PR #10](https://github.com/georgpou/corr-fep/pull/10)
- [B2.1 audited scientific run](https://github.com/georgpou/corr-fep/actions/runs/38089140014)
- [Issue #9](https://github.com/georgpou/corr-fep/issues/9)
- [Previous handoff](HANDOFF_B2_FRAME_ENERGY_DISCREPANCIES.md)
- [Stage B2 multiframe documentation](STAGE_B_MULTIFRAME.md)
- [Published archive](https://doi.org/10.5281/zenodo.21919704)
- [Original QM/MM TFEP source](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026)
