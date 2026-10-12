# Learned six-layer MixedMAF: actual-map water-matching diagnostic (2026-10-12)

## Result (successful physical-data no-QM replay)

- [GitHub Actions successful run #38210407853](https://github.com/georgpou/corr-fep/actions/runs/38210407853), job 114683070073.
- [Raw per-frame numerical artifact](https://github.com/georgpou/corr-fep/actions/runs/38210407853/artifacts/11728234014) (GitHub retention applies).
- [Executable experiment](../scripts/stage_b_learned_water_matching.py) and [workflow](../.github/workflows/stage-b-learned-water-matching.yml).
- Frozen trained baseline artifact from [successful run #38180943734](https://github.com/georgpou/corr-fep/actions/runs/38180943734), 21 updates, 650 training frames, 250 evaluation frames.
- This run loaded the archived `update-21.pt`, exact archived preprocessing `author-preprocess.npz`, `full-baseline.json`, original water XTC and TPR; no QM/MM engine, gradients or new training.

### Reproducible observed outcome

The action log contains:

```text
LEARNED_MATCHING_COMPLETE {"changed_oxygen_frames": 13, "changed_reference_frames": 1, "source_preprocess_agree_frames": 16, "max_ligand_displacement_A": 0.23190633750655598}
```

**13 of 16** spread-out validation frames exhibited at least one changed matched-water oxygen assignment between the source and the *learned model's mapped output*. **1 of 16** had a changed atom identity among the three reference-water slots defining the model's coordinate frame. All **16 of 16** source-frame recalculations exactly recovered the original cached oxygen assignments: the comparison is not caused by a mismatch of our re-run matching setup to the training preprocessing. Maximum single-atom ligand displacement in this subset was 0.23190634 Å. Per-frame assignments, full counts, RMSDs and displacement values are retained in the associated numerical artifact.

### Interpretation

This is a stronger *relevance test* than the earlier artificial ligand-translation/rotation probe: water-assignment switching actually occurs on outputs of the **real, frozen, trained six-layer map**, under untouched held-out inputs. It **does not establish a defect in the paper** or actual TFEP bias. Neither switching nor fixed-cache inverse/Jacobian closure is, by itself, a proof of failure or success of global piecewise invertibility, and the original paper employs other mapped-atom/cutoff regimes beyond this ligand-only implementation.

A second nuance: the input cached assignment is how this specific baseline's map is implemented; in a full transformed configuration, recomputing the matching rule at that output gives a different selected assignment in most of these cases. This establishes a *source-dependent branch selection phenomenon* along actual trained-model displacements. Showing two different source states map to the same output, or a demonstrable coverage/Jacobian/weight discrepancy, remains a **separate scientific exercise**.

### Decision

Preserve the working baseline unmodified. Record in [deferred Issue #17](https://github.com/georgpou/corr-fep/issues/17). Prioritize the actual Corr-FEP scientific benchmark rather than stating that authors made an error; revisit a full bijectivity or estimator proof before making claims of rigorous free-energy improvement.

Experiment does not include QM/MM energies or calculate any new free-energy estimate.
