# Author-method coordinate recovery implementation plan

> **For agentic workers:** Use superpowers:executing-plans task by task. The user authorized all proposed investigations and physical training on 2026-10-11; execute inline and ask only at a material unresolved methodological decision or before merging.

**Goal:** Recover correct author-method real-QM/MM training on a bounded archived dataset.
**Architecture:** Reuse pinned paper preprocessing and MixedMAF. Verify narrowly logged upstream issues, then introduce an attributed matched-water conditioning adapter preserving physical environment coordinates and a cached-assignment inverse. Cloud coordinate and physical checks gate a small benchmark.
**Tech stack:** Existing pinned Python/TFEP/bgflow/MDAnalysis/GROMACS/tblite GitHub Actions environment.
**Spec:** `docs/UPSTREAM_IMPLEMENTATION_AUDIT.md`, as authorized and refined by the user's 2026-10-11 continuation.

## Global constraints

- Development branch only; main unchanged; separate explicit merge approval.
- Actual GFN1-xTB subtractive mechanical embedding, temperature 298 K, original units/topology.
- Original 650/250/300 chronological blocks and training-only fitting; no test-based choices.
- Scientific gates: inverse/logdet <=1e-5, fixed environment <=1e-10 Angstrom, bonds 0.6–2.5 Angstrom, coordinate displacement <=2 Angstrom.
- Cloud scientific calculations; lean targeted checks, persist failed evidence, no blind retry.
- Initial author-configured physical pilot: six MAF layers, 32 batch, three updates from 96 training frames; AdamW lr1e-4, weight decay0.01, ExponentialLR0.997 per update, gradient clipping3000; float64.

## Review focus

- Atom matching must not corrupt reference-stability atom identities.
- Fixed environment must be restored exactly and cached matching reused for inverse.
- Only training determines reference roles/calibration; validation applies frozen rules.
- Cartesian determinant must agree independently after parameter updates.
- Historical upstream environment and statistical limitations remain explicit.

### Task 1: Focused upstream issue audit

Create `scripts/stage_b_author_issue_audit.py` and a bounded Actions workflow. Use actual archived topology and three training snapshots (temporary TRR reencoding of XTC, precision not improved). Execute pinned `match_waters` and `PermutationFlow`. Preserve original ordering defect and mapped-water restoration reproduction. Record candidate-loop issue separately; do not conflate it with present data. Upload JSON/log on failure. Expected: precise identified source-level defects, no energies or training.

### Task 2: Author preprocessing and coordinate adapter

Create `scripts/stage_b_author_map.py`, reusing helpers from the pinned driver with minimal attributed fixes after Task1 confirms triggers. Fit matching reference/counts/heavy-atom stability on training only; apply to validation. Retain ligand-only mapping and reference-only conditioning, a reported author option. Keep actual water identities changing by cached spatial assignment. Wrap PartialFlow with an inverse that preserves original environment exactly. Cloud test: original geometry gates, independent Cartesian determinant/directional derivative and strict checkpoint reload, zero energies. Do not adopt random stress exceeding geometry gates.

### Task 3: Physical pilot and small benchmark

Create bounded author-recovery runner/workflow. First run three real physical updates with full author architecture/optimizer settings. Preserve online target energies/logdet/sample IDs and actual weights. Probe train/validation after updates and independent derivative after final update. Verify energy-force/physical-gradient finiteness and decomposition. Only a valid checkpoint permits a small final held-out evaluation with identity baseline; reuse already fixed partitions, never select by test. If physical geometry fails, diagnose frozen weights before any optimizer or gate change.

### Task 4: Evidence and review

Inspect artifacts, independently recompute work and recorded determinants, request focused final code/science review, update recovery docs/README/draft PR/hourly task. Explain deviations and blockers. Never merge without separate user approval.

## Ledger

- Ruling: prior in-chat source-informed design is explicitly authorized; do not add another approval cycle.
- Ruling: 96 training frames and 3 updates retain author batch32/one-epoch pattern at cloud pilot scale; not full-paper replication.
- Ruling: inspect U3 but keep environment-only conditioning now; mapped-solvent extension is outside this recovery.

- Task 1 execution: Actions 38152838415 failed before source audit because TPR was not materialized. Corrected raw TPR write; no numerical/source conclusion from that run.
