# Author-method recovery: work log and audit handoff

Date: 2026-10-11, Europe/Copenhagen. Repository: [georgpou/corr-fep](https://github.com/georgpou/corr-fep). Development branch: `stage-b/extended-cloud-benchmark`. [PR #14](https://github.com/georgpou/corr-fep/pull/14) remains draft and unmerged. The executed physical pilot is pinned to commit `7d59c1ce2021edf81928611ba68b59d80b6971fd`; subsequent documentation does not change that executed code.

## Outcome and scope

Recovered a source-informed, ligand-only, author-conditioned six-layer MixedMAF and completed three real QM/MM optimizer updates over 96 archived training frames. Sampled trained inverse, environment, geometry, independent Cartesian derivative, energy-gradient and strict checkpoint reconstruction checks passed. A frozen final-map comparison on 16 validation frames completed. Its statistical outcome is mixed: reduced-work standard deviation decreased, but weight ESS decreased too. This establishes a bounded integration result, not convergence, unbiased estimation, a complete paper reproduction, or a successful Corr-FEP hypothesis benchmark.

The full recovered 650-training/250-validation/300-test benchmark has **not** been run. The original extended runner is retained as historical/diagnostic code and still contains the failed moving-ligand-reference configuration. It is not automatically replaced by the new adapter. The next baseline must explicitly use the author adapter. No larger case study or H200/ARM64 deployment was performed.

## Authorization and operating choices

The user requested building on the original authors' source, logging suspected bugs rather than silently rewriting the method, retaining scientific checkpoints without a generic edge-case suite, and using GitHub Actions for molecular calculations. The user subsequently authorized all proposed recovery steps and physical training. Every merge still requires separate approval. This documentation request authorizes updating the PR and audit record, not a merge.

All new molecular calculations ran in Actions. Local work consisted of source inspection, syntax/static review, artifact handling, and recomputation of arithmetic from already recorded energies. The final documentation update launches no molecular experiment and changes no workflow trigger marker. No authors were contacted and no upstream issues were filed. Source attribution and the original MIT notice accompany substantial copied helper code in the run artifacts.

## Sources and fixed physical model

| Component | Inspected/executed source |
|---|---|
| Paper driver | [run_tfep.py](https://github.com/andrrizzi/paper-alchemical-tfep-qmmm-2026/blob/b8dfd897ebd914b4e592221710b397a2758353ce/run_tfep.py), commit `b8dfd897ebd914b4e592221710b397a2758353ce` |
| TFEP | `0bc369a1194c688bb6ac997cc6b7c2db2b063c9e` |
| bgflow | `fbba56fac3eb88f6825d2bd4f745ee75ae9715e1` |
| GROMACS | double-precision CPU 2023.3, commit `5e5ea27d8e594124d229a726616b7f6473f2c935` |
| Python dependencies | Python3.11, torch2.4.0 CPU, numpy1.26.4, MDAnalysis2.7.0, tblite0.3.0; complete pins/build flags in the three author workflows |
| Trajectory | archive member `lig_p38a_2g/eq/water/repeat-0/traj_comp.xtc`, 73,093,240 bytes, SHA256 `1d9451c4f969d23791676e971fdb358394917ea7c51dae43e7390fe604e99362` |

The target Hamiltonian remains subtractive mechanical embedding: `U_target = U_MM(full) - U_MM(ligand) + U_GFN1-xTB(ligand)`. Ligand/environment interactions remain classical. Temperature is 298 K and the executed `kT` is 2.477709860209666 kJ/mol. The training objective is mean target energy/kT minus mean Cartesian logdet. Reference MM energy is parameter-independent; it is retained in the estimation work, not required in the training gradient.

The attached ChemRxiv preprint and released driver were compared. The manuscript describes a ligand frame, while executable preprocessing chooses conditioning references. We followed the concrete driver. Its historical conda file does not fully pin TFEP (`1+unknown`), so the released source pins do not prove the exact publication environment or result provenance.

## Investigation ledger

| Run | What was executed | Observation and decision |
|---|---|---|
| [38138876548](https://github.com/georgpou/corr-fep/actions/runs/38138876548) | Original extended B2.5: one layer, batch2, 325 epoch-1 updates | Failed Cartesian inverse (0.0021159731524242886 Å) and logdet closure (0.02828200723749319); no valid selected trained test. Retained as failure evidence. |
| [38148045379](https://github.com/georgpou/corr-fep/actions/runs/38148045379) | Repeat original optimizer/data with probes every25 steps; actual first-invalid/last-epoch weights saved | Completed325 updates; first **observed** failure at25. Direct spline and mixed-space inversion remained near machine precision. Do not call25 the exact onset. |
| [38149445011](https://github.com/georgpou/corr-fep/actions/runs/38149445011) | Frozen trained-coordinate audit, no QM or updates | Exact replay; disabling six reference shifts repaired sampled positional closure but left true Cartesian logdet discrepancy0.017905558738250726 at325. Rejected as a production fix. |
| [38150245773](https://github.com/georgpou/corr-fep/actions/runs/38150245773) | Diagnostic single-water O/H/H reference proposal, no QM or updates | Identity and0.001 parameter noise passed sampled checks;0.03 failed geometry (bond3.1034568Å, displacement8.6768727Å). Overall scientific failure. Superseded by authors' matched heavy-atom references. |
| [38152838415](https://github.com/georgpou/corr-fep/actions/runs/38152838415) | First author-source issue audit attempt | Setup failed because archived TPR was not written to expected path. No source/scientific conclusion. Corrected materialization before rerunning the same bounded audit. |
| [38153394950](https://github.com/georgpou/corr-fep/actions/runs/38153394950) | Unmodified source helpers on actual topology/3 training frames; zero energies/updates | Reproduced U1 wrong-element slots and U3 erased mapped-water displacement; U2 did not trigger on this data. |
| [38153970197](https://github.com/georgpou/corr-fep/actions/runs/38153970197) | Author-conditioned six-layer coordinate gate, zero QM/updates | Identity and0.001 parameter stress passed sampled geometry, true138×138 ligand Cartesian determinant and checkpoint reload. All900 train/validation reference triples non-collinear. |
| [38154651608](https://github.com/georgpou/corr-fep/actions/runs/38154651608) | Author-configured real QM/MM pilot and16-frame frozen validation comparison | Completed3 updates/96 frames; all recorded gates passed, checkpoint saved; validation outcome mixed. Scientific completion marker at16:26:43UTC /18:26:43 Copenhagen. |

Earlier scratch-path reuse and GROMACS backup-limit failures were execution problems; they are separate from molecular-map integrity. `GMX_MAXBACKUP=-1` prevents excessive backup generations without changing physics. Historical documentation remains available, but this log supersedes old statements that author recovery was awaiting authorization or that the single-water proposal was the next selected method.

## What changed in the source-informed adapter

The original driver's `find_ligand_neighbors`, `match_waters` and `select_reference_frame_atoms` determine spatial water roles and stable conditioning heavy atoms. We reuse these routines rather than reconstructing their matching algorithm. The adapter supplies chronological in-memory Universe subsets from the actual XTC, and archives the adapted matching helper with the upstream license.

U1: before its final topology-order sort, the released matcher assigns an oxygen-then-hydrogen source row into interleaved O/H/H reference slots while computing reference-stability distances. The actual-topology audit selected44 waters/132 atoms and found58 wrong-element slots. The returned matching table is sorted correctly; the defective intermediate coordinates affect the stability statistics. The minimal local patch assigns into oxygen-then-hydrogen reference slots at that intermediate step. Publication impact has not been established.

U2: the reference selector truncates candidates to at most10 but iterates triples over a fixed10 even if fewer candidates exist. A three-candidate rejected triple can be followed by an out-of-range index. The adapter bounds the call by the actual heavy candidate count. This edge does not trigger on the present many-water data.

U3: the released `PermutationFlow` scatters transformed water slots, then restores every reference slot, potentially overwriting mapped water. The actual released wrapper erased a proposed0.001Å displacement under identity matching (output0). Recovery maps only ligand atoms and restores **all physical environment coordinates exactly**, so it avoids the mapped-solvent path rather than claiming to fix solvent mapping.

`ConditioningContextFlow` gathers cached matched-water coordinates into reference roles, calls the original PartialFlow/MixedMAF, and restores the original environment. Its inverse reuses the identical cached assignment. The context wrapper also preserves the temporary `return_partial=True` setup path. `BoundFrame` supplies sample identity to the existing single-frame geometry/derivative probes. Original coordinate, Z-matrix and spline implementations were retained; the six moving-reference residuals are absent because references are conditioning atoms.

## Data selection and training protocol

Preprocessing fits on train401..1050 (650 frames). The inherited neighbor-count helper excludes local frame zero, so counts use649; matching/stability/reference admissibility use all650. Validation1151..1400 (250 frames) only applies the frozen reference/ranking. No validation-based reselection occurred. Spline calibration retains the existing32 training-era frames: a pilot deviation from full-paper dataset calibration.

Full-training preprocessing chose reference snapshot global frame409 and three oxygen slots `[8299, 487, 6148]`. It matched41 reference waters. This differs from44 in the three-frame issue reproduction because selection used a different reference and dataset. Roles may refer to different physical waters in different snapshots; this is not one permanently chosen water molecule. Mapped cutoff0Å and conditioning cutoff3.5Å implement the authors' ligand-only/reference-only option. The exact published CLI invocation is unavailable; this is a source-supported option, not a proven exact paper configuration.

Physical training used96 unique `linspace` frames spanning the training block, deterministic shuffled order, six MAF layers, batch32, one epoch/three updates, float64, AdamW lr1e-4 with weight decay0.01, ExponentialLR gamma0.997 each update, and gradient clipping3000. These settings come from the production driver. The loop is an explicit bounded Python integration loop rather than the author's Lightning trainer. It saves online pre-update target/reference energies and logdet for every frame, actual weights after each update, and optimizer/scheduler state in update checkpoints.

Geometry checks were batched for the32 configurations of each update to avoid32 serial inverses. The same per-frame thresholds were retained: inverse/logdet<=1e-5, environment<=1e-10Å, bond lengths0.6..2.5Å, max Cartesian component displacement<=2Å. Six train/validation configurations were also probed at initialization and after each update. Full independent Cartesian derivatives were measured at train726 and validation1275 after update3. The16 validation comparison frames span the validation block; the final update3 was fixed beforehand, with no fitting/model choice during comparison.

**Test provenance:** the author-recovery experiments performed zero evaluations on test1501..1800. However, the original failed extended run had already evaluated the300-frame identity baseline and MM reference energies. These frames are reserved from further recovery/model fitting, but must not be called a pristine unseen test set. There is no valid recovered trained300-frame test result.

## Numerical evidence from actual trained weights

| Check | Recorded result |
|---|---:|
| Physical updates / unique online rows | 3 /96 |
| Flow parameter change L2 | 0.28623542315504735 |
| Maximum physical-gradient decomposition error | 8.285928583887103e-12 |
| Recorded geometry rows across batch/init/post-update/validation checks | 136 |
| Maximum inverse error across those rows | 3.3786307085392764e-12Å |
| Maximum inverse logdet closure | 2.2737367544323206e-13 |
| Maximum fixed environment displacement | 0Å |
| Final derivative logdet discrepancy at train726 | 1.767336277325171e-14 |
| Final derivative logdet discrepancy at validation1275 | 6.59472476627343e-14 |
| Final directional finite-difference errors | 4.717575767532178e-9;2.074201727042535e-9 |
| Complete-Hamiltonian gradient FD error, ligand selected component | 0.0053877908420219 kJ/mol/Å |
| Complete-Hamiltonian gradient FD error, environment selected component | 0.0037656912288639433 kJ/mol/Å |
| Fresh strict checkpoint reconstruction: positions/logdet | 0 /0 |

These are sampled numerical gates, not a proof over every configuration/parameter value. The force FD check covers two selected components at one physical configuration, not every force component. All900 reference triples passed the collinearity check, with minimum angle sine0.38336204490831605. The independent determinant differentiates the138-dimensional ligand Cartesian block while the environment and cached assignment are fixed. Since the environment output is identity, this is the determinant block of that local conditional map.

## Frozen validation comparison

| Metric,16 validation frames | Identity | Trained final map |
|---|---:|---:|
| Reduced-work mean | -86905.0561796652 | -86907.80936250226 |
| Reduced-work sample standard deviation | 8.670024961747218 | 6.906032363474717 |
| Normalized-weight ESS | 1.8310877701480854 | 1.5573630298742975 |
| Phase correction estimate,kJ/mol | -215351.66685255262 | -215354.55149783444 |

The work is `(U_target(T(x))-U_MM(x))/kT-logdet`; identity has logdet0. The estimator is `-kT*log(mean(exp(-w)))`, evaluated stably after subtracting the largest log weight. Weight ESS is `1/sum(normalized_weight**2)`. It is not an autocorrelation-corrected number of independent trajectory samples. These large absolute phase corrections include the different absolute energy zeros; they are not binding free energies and their absolute size does not establish convergence. A lower mean or smaller work standard deviation alone is not evidence of better estimation. Here weight ESS worsened, with roughly one or two weights dominating16 correlated samples. No confidence intervals or claimed improvement were produced.

## Evidence files and reconstruction

The authoritative files are the completed Actions artifacts; molecular data/weights were not committed to Git. The small report snapshots and arithmetic audit are under [docs/evidence/2026-10-11-author-recovery](evidence/2026-10-11-author-recovery).

| Evidence | Identifier |
|---|---|
| Source issue artifact | 11707841295, run38153394950, `stage-b25-author-issue-audit`,346350bytes |
| Coordinate artifact | 11707524666, run38153970197, `stage-b25-author-coordinate-probe`,87643246bytes |
| Physical artifact | 11707374603, run38154651608, `stage-b25-author-training-probe`,348045722bytes |
| Coordinate checkpoint SHA256 | `98dd06991cb909ac4bc86ace540ca25b9cd91b514ebda8f2d768022aa60c0d15` |
| Trained checkpoint SHA256 | `eea72d729a02d21f7a4cbee7d7f70d901f620f4dc1426b4a96c95e5c25441fb4` |

The physical JSON snapshot is copied byte-for-byte from its downloaded artifact. The coordinate JSON snapshot is a normalized prior extraction; its file hash is for that normalized snapshot, not the original byte formatting. The arithmetic audit records both snapshot hashes. A cached coordinate ZIP was found truncated during this documentation pass and was not used for new validation; the canonical Actions artifact is the source for fresh weight inspection. The physical ZIP was complete and its parsed report agrees with the earlier saved result. Audit agents should freshly download artifacts; GitHub retention is finite (reported expiry2027-01-09).

To reproduce, use the tested commits with the pins in each workflow, original archived topology/XTC, `author-preprocess.npz`, the logged references/assignments, and the correct weights. `update-N.pt` preserves optimizer/scheduler state; `trained-map.pt` is the strict-reload map snapshot. The preprocessing table is required: weights alone do not supply frame assignments. The archived adapted matcher includes its original license. Workflow names, trigger markers and output paths are explicit in `.github/workflows/stage-b-author-*.yml`; editing a push marker requests a new run. Do not accidentally launch the older extended runner when auditing.

A local arithmetic audit recomputed all96 online work values and both work values on each16 validation frame from the recorded energies/kT/logdet: maximum discrepancy0. It recomputed validation mean/std/ESS/estimate with at most2.220446049250313e-16 discrepancy from reported metrics and checked partition bounds, frame uniqueness and zero recovery test evaluations. This is an independent calculation from stored numbers, not a new molecular/force simulation or independent validation of the energy engine.

## Static review, known issues and limitations for the next auditor

Static review of the four author scripts found no blocking source/API issue. A follow-up review found the batched geometry gates and stabilized validation formulas consistent. Neither review ran additional molecular calculations. The cloud artifacts, rather than the reviewer opinion or CI badge, support the observed numerical gates.

The follow-up reviewer noted a finite-check ordering issue: training checks `logj` finiteness after the target energy call, whereas it could check before batch geometry/QM. A NaN comparison can otherwise fail open in a closure threshold until the later explicit finite check. The recorded run contains finite values, and the later check prevents successful training acceptance of a nonfinite logdet. This documentation-only pass leaves tested code unchanged and flags the early guard for audit/follow-up.

Cached frame-dependent matching supports the measured conditional inverse and Jacobian. It does **not** prove global invertibility if matching is recomputed from mapped coordinates or establish an unbiased single-map held-out estimator. The original source also caches assignments. The auditor should assess this mathematical issue before stronger TFEP claims; measured local determinants and inverse closure do not settle it.

Other remaining limits: one archived repeat; quantized XTC instead of original TRR;32-frame calibration;96 training samples/three updates;16 correlated validation frames; no uncertainty/autocorrelation analysis; no successful recovered300-frame trained test; no full original multimap estimator reproduction; no Corr-FEP intervention comparison; no larger real case. Do not infer causation separately for architecture, optimizer and reference changes from this recovery, which changed them together to follow the driver.

The next proposed work is a one-epoch recovered author baseline on the full650 training frames and250 frozen validation frames, followed by an explicitly defined, budget-matched hypothesis comparison and controlled final test. That work is not represented as completed by this PR. The independent auditor should read the code and raw artifacts, check reference matching/U1 correction, cached-map mathematical scope, physical gradient/units, full ligand determinant, checkpoint reconstruction, split provenance and mixed validation interpretation. Merge remains a separate user decision.
