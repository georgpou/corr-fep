# Corr-FEP — Agent handoff: Stage B2 frame alignment and energy discrepancies

**Date:** 2026-10-10  
**Repository:** https://github.com/georgpou/corr-fep  
**Main after PR #8 squash merge:** `5360db791671d47195408c9d3a77397dfedd6c4f`  
**Work branch opened for successor:** `stage-b/frame-energy-discrepancies`  
**Merged PR:** https://github.com/georgpou/corr-fep/pull/8  
**High-priority scientific issue (keep open):** https://github.com/georgpou/corr-fep/issues/9

## Mission

Investigate and resolve (1) a failed terminal XTC-versus-GRO frame correspondence audit and (2) an approximately 1.012 kJ/mol terminal-frame full-system MM/QM-MM energy discrepancy. These are **two related but distinct questions**. Restore a passing, physically meaningful Stage B2.1 multi-frame scientific checkpoint before moving to B2.2 reference-energy matching or molecular TFEP training.

PR #8 was **explicitly merged by the user despite the failing revised scientific workflow**, as an accepted experimental code baseline, NOT as a completed scientific validation. Do not retroactively represent B2.1 as scientifically passed. Do not close issue #9 simply because a structural alignment check begins to pass.

User priorities: small, genuine, interpretable experiments with actual Hamiltonian; avoid excessive tests/abstractions/refactoring; prioritize scientific checkpoints. You may create branches, commits, workflows, PRs, review/fix them; **ask user approval before each new PR merge**. Do not automatically merge work on this follow-up branch.

## Ground truth / current implementation

The physical model is subtractive mechanical embedding:

```
U_QMMM(x) = U_MM_full(x) - U_MM_ligand(x_L) + U_GFN1-xTB_ligand(x_L)
```

- GROMACS double-precision CPU 2023.3, pinned source `5e5ea27d8e594124d229a726616b7f6473f2c935`.
- Original research source `b8dfd897ebd914b4e592221710b397a2758353ce`.
- TFEP `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e`; tblite 0.3.0 GFN1-xTB.
- Published Zenodo data: https://doi.org/10.5281/zenodo.21919704, selected ligand `lig_p38a_2g`, water repeat-0. Data archive ~49 GB; use HTTP byte ranges and retrieve ONLY necessary members, never the full ZIP.
- 8420 full-system atoms, 46 ligand atoms, neutral closed-shell assumed, from archived TPR/GRO/ITP.
- Study used 25 repeats but Zenodo deposited only repeat-0; do not claim full reproduction.
- Relevant files: `scripts/stage_b_multiframe.py`, `scripts/stage_b_mechanical_embedding.py`, `scripts/stage_b_qm_singlepoint.py`, `docs/STAGE_B_MULTIFRAME.md`, `docs/STAGE_B_MECHANICAL_EMBEDDING.md`, `docs/REPRODUCIBILITY.md`, workflow `.github/workflows/stage-b-multiframe.yml`.

## Evidence and exact run state

**Successful original B2 smoke run (before stricter audit fixes):**
- https://github.com/georgpou/corr-fep/actions/runs/38081235242
- Artifact: https://github.com/georgpou/corr-fep/actions/runs/38081235242/artifacts/11680069765
- Four genuine MM(full) − MM(ligand) + GFN1-xTB evaluations and gradients succeeded on original trajectory frames **0/800/1600/2400** (0/4/8/12 ns).
- Quantum direct-vs-TFEP gradient discrepancy max ~2.08e-8 kJ/mol/Å, environmental gradient identity exact to stored precision; arrays finite. **Frame 0 was inside the initial 2 ns equilibration interval and outside paper training production window.**

**Revised B2 workflow after audit fixes:**
- https://github.com/georgpou/corr-fep/actions/runs/38086054070
- **FAILURE** in added `check_terminal_alignment` before any per-frame numerical-gradient probes.
- Reported terminal XTC/GRO minimum-image displacement: RMS **17.55595 Å**, max **17.55598 Å**, near-constant across atoms; periodic boxes agree to ~0.00002 Å.
- Interpretation: nearly equal per-atom displacement magnitudes strongly *suggest* a global translation or different coordinate origins/PBC centering. This is an **unconfirmed hypothesis**, not proof. The code used `MDAnalysis.lib.distances.minimize_vectors(terminal - reference_xyz, box)` but did NOT first register/recenter the structures; minimum-image alone does not remove an arbitrary common non-lattice translation. Compute actual displacement vectors and their residuals before concluding.
- The new source chooses production frames from **401..2400** and implements two complete-Hamiltonian central-difference probes at the first production frame. **Those FD checks never ran because alignment raised first.**
- General synthetic `CI` on PR head passed; it does not validate the physical experiment.

**Previously accepted B1 real-QM/MM singlepoint:**
- https://github.com/georgpou/corr-fep/actions/runs/38073286507
- Artifact: https://github.com/georgpou/corr-fep/actions/runs/38073286507/artifacts/11678021501
- Real 18 central differences (5 ligand atoms and 1 interfacial solvent, three step sizes), maximum errors at 0.002/0.005/0.010 Å of 0.01220/0.08413/0.35612 kJ/mol/Å; accepted max 0.8.
- Original B1 and first B2 artifacts match published input hashes and relevant compiled physical settings, but generated TPR hashes differed across runs; not direct evidence of changed force-field physics.

## HIGH PRIORITY — discrepancy #1: terminal XTC/GRO positional mismatch

Question: Are terminal `eq/water/repeat-0/traj_comp.xtc` frame 2400 and deposited `eqout.gro` the same physical 12 ns configuration in different coordinate representations?

Plan — work from original archived inputs and avoid guessing:

1. Read actual GRO time title (if present), last XTC time (12000 ps in prior run), atom count/order as determined by TPR and independent structural checks, and both periodic cells. XTC does not contain independent atom names/types; comparing MDAnalysis `Universe(tpr,gro).atoms.names` against `Universe(tpr,xtc).atoms.names` is **tautological**.
2. Inspect how `setup_coordinates` transforms the GRO via `unwrap(ligand); center_in_box(ligand); wrap(environment)` and apply the same transformations to the XTC endpoint. Record for all atoms the displacement vectors under minimum-image and the vector mean, vector norm dispersion, mean-median shift, ligand/environment RMS and max deviations **before and after** subtraction of an estimated common rigid translation.
3. A robust translation estimator can be coordinate-wise robust median of minimum-image displacement vectors, but check for branch-cut problems and re-apply PBC after translation. Verify relative pair distances, ligand bonded geometry, interfacial distances, molecular identity and periodic cell, not simply RMS scalar displacement norms. If relative deviations remain large, investigate timing/indexing/wrapping, not suppressing thresholds.
4. Check time alignment and exact coordinate correspondence beyond residual translation. If same configuration, implement a scientifically meaningful, translation/PBC invariant correspondence check (or canonicalization) with recorded diagnostics and transparent threshold.
5. Do not silently shift the coordinates used for physical energy evaluations unless the transformations are proved physically equivalent in the periodic system and consistent with GROMACS; the initial task is *comparison*, not manipulation of the Hamiltonian inputs.

## HIGH PRIORITY — discrepancy #2: ~1.012 kJ/mol energy difference (issue #9)

Accepted B1 `eqout.gro` versus first B2 XTC terminal frame 2400:

| component (kJ/mol) | B1 GRO | B2 terminal XTC | B2-B1 |
|---|---:|---:|---:|
| full MM | -116802.61791405082 | -116803.62996742321 | -1.01205337239 |
| isolated MM ligand | -971.022166844934 | -971.023118905871 | -0.00095206094 |
| GFN1-xTB ligand | -216325.75103968772 | -216325.75174184845 | -0.00070216073 |
| combined | -332157.3467868936 | -332158.3585903658 | -1.0118034722 |

Large absolute values include ligand electronic reference offset; differences are *not* binding free energies. This small absolute offset could matter for future exponential reweighting **if configuration dependent**. Near equality of ligand terms suggests the discrepancy mainly resides in full-system MM interactions. Causes not established: lossy XTC vs GRO precision, distinct frames, image handling, neighbor-list choices or energy-term differences.

Plan:

1. Correlate coordinate residual diagnostics with the above full-MM energy difference; a truly common periodic translation should not change the energy of an identical periodic system. **A common translation by itself does NOT explain a 1 kJ/mol difference.**
2. Under the *same* generated TPR/runtime, evaluate the terminal XTC coordinates versus GRO coordinates (and a translated-to-match equivalent) in the same execution, with exact parameter/cell/coordinate provenance. Decompose energy terms if feasible to isolate where ~1 kJ/mol originates. Avoid comparing independent builds prematurely.
3. Compare deposited `ener.xvg` at exact frame/time to equivalent GROMACS full-MM potential and term definitions; account for XTC lossy compression, GRO rounding, constraints, precision, PME, dispersion corrections, and GROMACS version. Never manually offset or 'calibrate' energies to manufacture a match.
4. Quantify whether the difference is constant across a meaningful small sample or configuration-dependent; a constant additive potential shift cancels in many relative differences but a variable shift can distort work weights.
5. Update issue #9 with measured root cause, evidence, and acceptance threshold only after actually resolved. Continue tracking as HIGH until then.

## Recovery criteria for B2.1

- Production samples selected from the 2000-frame range **401–2400** (first 2 ns excluded), no frame-0 production claim.
- Independent, translation/PBC-aware XTC/GRO correspondence: valid and quantified, not a same-TPR atom-name comparison.
- Four real QM/MM component energies/gradients finite and consistent under audited TPR, with proper frame-dependent box.
- Two new finite differences of the complete Hamiltonian (one ligand and one interfacial environment Cartesian coordinate) on a production frame, central h=0.002 Å, with fixed B1 tolerance **0.8 kJ/mol/Å**. Check correctly computed gradient sign/units.
- Archive hashes, transformation protocol, alignment diagnostics, energy components, FD probe numerical outputs, engine versions and workflow SHA stored in run artifacts.
- Dedicated scientific Actions run fully successful, **inspect actual artifact**, not just generic CI green.

Do not advance to B2.2 FF reference alignment, B2.3 differentiated training backend, or B2.4 actual molecular flow training on the basis of failed B2.1.

## Workflow guidance

- Start from merged `main` commit `5360db791671d47195408c9d3a77397dfedd6c4f` on `stage-b/frame-energy-discrepancies`.
- Reproduce the alignment diagnostic in a minimal focused job or edit existing `stage_b_multiframe.py` to dump rich mismatch evidence before raising.
- Preserve the original real-QM/MM Hamiltonian and the double-precision GROMACS build cache; do not substitute an MM proxy.
- On current workflow, `push` science trigger is restricted to `stage-b/multiframe-qmmm`. When creating the follow-up branch, **update `.github/workflows/stage-b-multiframe.yml` trigger** to include the new branch, or manually dispatch via `workflow_dispatch` at its ref, and verify runs use the expected commit.
- Favor few purposeful scientific checks rather than endless unit/edge-case testing. Diagnose failure, change one physical assumption at a time, run the real scientific job, evaluate artifacts and document.
- Once scientifically accepted, update `docs/STAGE_B_MULTIFRAME.md` and issue #9 with evidence, open a PR, review it critically, then **ask user for explicit approval before merging**.

## References

- https://github.com/georgpou/corr-fep
- https://github.com/georgpou/corr-fep/pull/8
- https://github.com/georgpou/corr-fep/issues/9
- https://github.com/georgpou/corr-fep/actions/runs/38081235242
- https://github.com/georgpou/corr-fep/actions/runs/38086054070
- https://github.com/georgpou/corr-fep/actions/runs/38073286507
- https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026
- https://github.com/andrrizzi/tfep
- https://doi.org/10.5281/zenodo.21919704
