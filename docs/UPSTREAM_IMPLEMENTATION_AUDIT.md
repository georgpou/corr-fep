# Upstream implementation audit — 2026-10-11

## Decision and scope

The user's direction is to build Corr-FEP on the authors' coding foundation,
with departures justified explicitly. The user subsequently authorized all
proposed investigations and author-method physical training on 2026-10-11.
Coordinate validity remains the gate before physical training; merging still
requires separate approval. The previous single-water O/H/H candidate is diagnostic
evidence only; its proposed adoption is superseded by the source-informed
proposal below. It is not an implementation of the authors' preprocessing.

Primary source:
[paper driver](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026/blob/b8dfd897ebd914b4e592221710b397a2758353ce/run_tfep.py),
commit `b8dfd897ebd914b4e592221710b397a2758353ce`.
Pinned TFEP commit `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e`,
[MixedMAF](https://github.com/andrrizzi/tfep/blob/0bc369a1194c688bb6ac997cc6b7c2db2b063c9e/tfep/app/mixedmaf.py),
[base map](https://github.com/andrrizzi/tfep/blob/0bc369a1194c688bb6ac997cc6b7c2db2b063c9e/tfep/app/base.py),
and bgflow commit `fbba56fac3eb88f6825d2bd4f745ee75ae9715e1` were traced.
On this audit date, the paper driver on main and MixedMAF on master have
identical Git blob hashes to the inspected pins: respectively
`6a40c3e3c01e3cb0005ef007ff0137a4f7cca525` and
`34b604a11bc0c774d93dbdf227fbf0d625feb7af`.
This does not prove the exact historical environment used for the publication:
the released conda environment lists `tfep==1+unknown`, not a source SHA.

## What the released driver actually does

1. `find_ligand_neighbors` (lines 211–328) counts waters inside mapped and
   conditioning cutoffs, chooses a reference with the modal mapped-water count
   and a compatible frequent conditioning count, and retains stable protein
   neighbors. The source uses a **mode**, whereas the attached manuscript
   describes an average count. Heavy protein backbone atoms are conditioning,
   never mapped (`run_preprocess`, lines 1308–1316).
2. `match_waters` (331–521) unwraps/centers the ligand, handles environment
   images, aligns the ligand scaffold, solves a Hungarian oxygen assignment,
   then the two hydrogen assignments within each water. It caches a frame-indexed
   atom table: the same input role can correspond to a different physical water
   in a later snapshot. It is not a permanently selected water molecule.
3. `select_reference_frame_atoms` (524–610) ranks conditioning atoms by the
   standard deviation of distances to the ligand scaffold, excludes hydrogens,
   considers at most ten candidates, and tests triples for collinearity over
   the trajectory. In bulk water these heavy atoms are oxygens of different
   waters, not O/H/H atoms of a single water.
4. `run_preprocess` (1385–1405) explicitly selects origin and axes **from
   conditioning atoms**. `--onlyref` retains just those three atoms as the
   conditioning input. This is not the same as zero reference conditioning.
   The callable defaults `only_reference_conditioning=True`; the CLI's
   `store_true` flag defaults false unless `--onlyref` is supplied. The actual
   published invocation/configuration is not present in the inspected files.
5. `TFEPMap` (923–1090) loads the cached selection into the original
   `MixedMAFMap`, installs `PermutationFlow`, uses six MAF layers, applies
   unwrap/center/environment-wrap transformations, and discards the first 2 ns
   of each repeat. Conditioning reference atoms are automatically excluded
   from mapping in `MixedMAFMap.configure_flow` (282–313). Its reference-shift
   transformer is therefore absent for those six frame residual coordinates.
6. The target is subtractive mechanical embedding:
   `U_MM(full) - U_MM(ligand) + U_GFN1-xTB(ligand)` (driver 709–720).
   TFEP base training minimizes mean target energy/kT minus logdet for an
   unbiased trajectory (base 780–840), consistent with our current objective.
   The reference energy is parameter-independent and need not appear in the
   training gradient; it enters the generalized work for estimation.
7. Production training uses six layers, batch 32, one epoch, float64,
   AdamW lr 1e-4 with default weight decay 0.01, ExponentialLR gamma 0.997 per
   update, and Lightning gradient clipping 3000. Energies/logdets generated
   throughout training are saved for the multimap estimator; this is different
   from our current held-out assessment of one selected checkpoint.

The attached manuscript, section 2.5 (pages 3–4), describes a ligand-defined
local frame. The released driver's executable preprocessing instead selects
conditioning references. Treat this as a source/manuscript discrepancy to
resolve, not evidence that either published results or the coordinate library
are generally incorrect. The driver is the concrete baseline for our port.

## Differences relevant to our failure

| Aspect | Released driver | Current B2.5 / diagnostic proposal | Consequence |
|---|---|---|---|
| Frame reference | Explicit stable conditioning heavy atoms | B2.5: implicit mapped ligand atoms | B2.5 activates the moving-reference path that failed after training |
| Solvent identity | Matched spatial roles, cached per frame | Proposal: one locked physical water identity | Water exchange and long-term proximity are handled differently |
| Reference selection | Ligand-distance fluctuations; heavy atoms; trajectory collinearity | Proposal: nearest admissible O/H/H water on 32 calibration frames | Our proposal does not implement upstream selection |
| Coordinate engine | Original MixedMAF/Z-matrix/splines | Same pinned engine | Retain this engine; avoid rewriting it without evidence |
| Coordinate transformations | Unwrap ligand, center, wrap environment | Same operation sequence | No identified discrepancy in the sequence; permutation is missing |
| Training | Six layers/batch32/scheduler/weight decay/clipping | One layer/batch2/no scheduler/zero weight decay/no clipping | Explicit bounded-pilot deviations; not established causes of the frame defect |
| Data/calibration | Multiple TRR repeats; dataset analysis | One XTC repeat; 650/250/300 split; 32 train calibration frames | Keep our partitions clean and document the adaptation |
| Estimation | Training-time multimap work | Validation-selected final map pilot | Method development must distinguish these estimators |

The frozen audit proves a failure in our configuration of the pinned library,
not a failure of the paper's conditioning-reference configuration. Even with
shifts disabled, the measured true determinant mismatch means that the mapped
ligand-reference path cannot be used as our correction. No exact algebraic
patch to that path has been established. The library's conversion test
`test_cartesian_to_mixed_flow_conversion` (397–454) retains the original pose
when immediately reconstructing unchanged mixed coordinates; that check does
not exercise reconstruction after learned reference shifts or independently
verify a trained full Cartesian determinant.

## Upstream issue log

These are local source/bookkeeping findings. No report has been sent to the
authors, no upstream patch has been applied, and no published trajectory or
free-energy result has been reproduced to assess impact. The first two checks
below involve integer labels/array indexing, zero molecular calculations,
zero energies and zero optimizer steps.

### U1 — Atom ordering in reference-stability distance preprocessing

**Priority: resolve before reusing this selection path. Source-statement
reproduction supports the ordering defect; publication impact unknown.**

In `match_waters`, `matched_water_atom_indices` is initially constructed as
all oxygens followed by all hydrogens (lines 479, 497). Line 505 assigns this
row into `ref_atom_groups['water'].ix`, which follows topology order, typically
O/H/H per water. The row is sorted into topology order only at lines 516–519,
after computing `environment_ligand_distances` at lines 508–513.

A two-water index example executed the original AST assignments at lines
504–505. Reference slots `[0,1,2,3,4,5]` in O/H/H,O/H/H order should receive
`[6,7,8,9,10,11]`, but the temporary array receives `[6,9,7,8,10,11]`.
Thus reference slot 3, an oxygen, receives hydrogen slot 8. The returned
matching table is subsequently sorted correctly, but the earlier stability
statistics can be computed from misidentified atoms. A focused full-helper
reproduction with actual topology is still needed before a patch or any
claim about effect on the published selection/results. A possible local
correction is to use consistent oxygen-then-hydrogen reference slot order
for this temporary assignment; discuss and validate before applying.

### U2 — Reference-candidate loop exceeds available candidates

**Priority: low; conditional error-path bug, not the current training failure.**

The selector truncates actual candidates to at most `n_max_atoms` (default
10), checks only that at least three remain, then iterates combinations of
`range(n_max_atoms)` (lines 571–604). If fewer than ten remain and the first
valid-index triples are rejected, later triples index beyond the collected
positions. With three candidates and a rejected `(0,1,2)`, the next `(0,1,3)`
raises `IndexError`, rather than the intended diagnostic `RuntimeError`.
A small array check reproduced this loop-bound failure. Bound iteration by
the actual candidate count if this path is ported; no extensive edge-case
suite is warranted at this stage.

### U3 — Permutation restoration can overwrite mapped water output

**Priority: relevant before mapping solvent; static finding, runtime
reproduction pending. Not exercised by ligand-only mapping.**

`PermutationFlow.forward` scatters transformed reference slots into physical
matched indices (line 915–916), then restores every reference slot from the
input (918–919). If a reference slot is also a matched output slot, the last
assignment can overwrite the transformed coordinate. For an identity
assignment of mapped waters, all such mapped output would be restored.
This must be checked with a focused executable wrapper case before copying
that restoration logic into a solvent-mapping experiment. With only ligand
mapped and all solvent conditioning/fixed, restoring the environment is the
intended behavior; this finding does not establish a defect in that subcase.
The wrapper also exposes no inverse method. Our explicit inverse gate needs
a reviewed adapter that reuses the same cached atom assignment and restores
original physical atom order; do not claim that copying it supplies an inverse.

## Revised next experiment — proposal only

Build a thin, attributed adapter around the original preprocessing helpers
and pinned `MixedMAFMap`, rather than a second coordinate implementation.
Before any physical updates:

- Fit neighbor counts, reference snapshot, fluctuation ranking and heavy-atom
  reference slots on training coordinates only. Use the author's ligand-only
  mapped-region option (mapped cutoff 0), and initially retain only the three
  conditioning reference roles (`--onlyref` pattern). Conditioning cutoff
  3.5 Angstrom is one reported ligand-only setting, to be fixed explicitly
  before evaluation; do not tune it from validation outcomes.
- Apply the frozen matching rule to validation without refitting its reference
  or ranking. Preserve physical atom order for energy evaluation, keep all
  environment coordinates unchanged in ligand-only mapping, and reuse cached
  assignment in the inverse and derivative checks. Record assignment tables,
  atom elements, geometry and provenance.
- Verify U1 on actual selected topology before deciding its minimal local
  correction. Log any patch with source attribution and a focused reproduction.
  Keep U2/U3 separate and proportional to the code paths actually used.
- Run a bounded no-QM coordinate probe in GitHub Actions using the original
  coordinate engine, independent full Cartesian determinants, nontrivial
  parameters, train/validation geometry and checkpoint reconstruction. A
  successful execution label alone is not sufficient.
- Only after the representation is understood and explicitly approved,
  consider the bounded three-update real-QM/MM pilot. Keep physical models,
  objective and existing acceptance thresholds. Keep the held-out test unused.

The earlier proposed exact O/H/H indices 4747/4748/4749 are not the new
selection. No new indices or matched roles have been chosen yet. Do not
silently dispatch either pilot, change the benchmark or mark PR #14 ready.
The original six-layer production settings should subsequently form a
separate documented baseline before interpreting improvements from our own
hypothesis; do not change optimizer, architecture and coordinate convention
simultaneously and attribute the outcome to a single cause.

Substantial reuse of the driver must retain its MIT copyright/license notice
(Copyright 2021 Andrea Rizzi). Its hardware-specific SLURM/paths can be adapted
for Actions and the later H200 cluster without changing the method.


## Actual-topology confirmation and authorized recovery

Actions [38153394950](https://github.com/georgpou/corr-fep/actions/runs/38153394950),
commit `715c727a0de88441cb19db36182278f666dfb99a`, completed the focused source audit.
On three archived training snapshots, the unmodified helper selected 44 reference
waters (132 atoms); its pre-sort assignment placed the wrong element into 58
reference slots. This confirms U1 on the actual topology. Its returned matching
table is sorted correctly; the defective intermediate positions affect the
stability statistics used for reference selection. The local adapter corrects
that intermediate slot ordering, retains the upstream algorithm, and archives
the adapted helper with the original MIT notice.

The actual released `PermutationFlow` also erased a proposed 0.001 Angstrom
reference-water displacement under identity matching (observed output zero),
confirming U3's mapped-water overwrite. This recovery maps only ligand atoms,
so it does not require a mapped-solvent rewrite. U2 does not trigger on these
44 heavy candidates; its conditional small-candidate defect remains logged.
No energy evaluations or physical updates occurred in this source audit.

Reference counts inherit the author's exclusion of local frame zero: 649
training frames determine neighbor counts; all 650 training frames determine
matching/stability and triple admissibility. Validation only applies frozen
training selections. Cached assignments define per-frame conditional inverse
and determinant checks; they do not establish global invertibility under
recomputed matching. Historical publication impact is not established.

The next authorized gate is the six-layer author-conditioned coordinate probe.
A passing gate permits the 96-sample, three-update physical pilot with the
author optimizer settings. This is a bounded integration experiment, not
convergence or reproduction of the publication's complete multimap estimator.
