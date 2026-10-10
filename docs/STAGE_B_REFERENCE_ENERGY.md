# Stage B2.2 — FF reference-energy alignment

**Status: all three numerical B2.2 checks completed; latest source revisions and evidence-preservation changes require fresh workflow validation.**

## Objective
Check the deposited repeat-0 `ener.xvg` potential definition and exact time alignment with selected XTC frames. Compare its numeric Potential values (kJ/mol) with the original archived dynamics TPR and the rebuilt GROMACS full-MM single-point TPR on identical coordinates.

The upstream pinned `run_tfep.py` reads XVG column 1 as FF potential. The name and definition of that column, and correspondence of frames, must be checked rather than assumed. These comparisons are not a free-energy calculation.

## Procedure
Run the dedicated `.github/workflows/stage-b-reference-energy.yml` workflow, or execute `python -u scripts/stage_b_reference_energy.py` in the pinned molecular environment after staging the original research checkout.

1. Retrieve only the bounded `lig_p38a_2g/eq/water/repeat-0/ener.xvg` member via HTTP Range, record hashes, header legends, energy units and full time axis.
2. Use real B2.1 XTC frames 401, 1067, 1733 and 2400, each transformed with the established ligand unwrap/center and environment wrap sequence. Require a unique reference energy row at the exact same time and index. Never interpolate or silently use the nearest frame.
3. Rebuild the accepted B1 GROMACS 2023.3 double-precision full-system TPR. Evaluate true TFEP full-MM energies, preserve their original G96 rerun inputs, and independently rerun those identical inputs against the archived production eq.tpr in the same executable. Preserve EDR potential-energy component dictionaries and both TPR hashes.
4. Make one symmetric seeded +/-0.005 Angstrom all-atom coordinate perturbation per frame. This is diagnostic only, and neither recreates missing full-precision TRR frames nor proves a bound on reweighting errors.
5. Compare terminal GRO, terminal XTC and a diagnostic-only hybrid in which water oxygen 463 adopts its GRO coordinates. Never treat this hybrid as a production/training sample.
6. Save a machine-readable JSON report, per-frame CSV, source header and hashes, GROMACS version/TPR parameter snippets and rerun logs in `results/stage-b/reference-energy/`. Upload artifacts even on failure.

## Scientific gate
The script's `validated=true` means ONLY that the XVG has an identifiable Potential observable and that all four selected frame times match. It does NOT mean the deposited and recomputed energies agree. Interpret numerical residuals, per-term differences, original versus rebuilt TPR settings, and the paired +/- perturbations before any training or free-energy claims.

Original dynamics used GROMACS 2021.6 double precision, whereas this rerun is GROMACS 2023.3 double; the deposited trajectory is lossy XTC rather than full-precision TRR. The archived GRO title does not prove its time. The four configurations belong to one repeat and are not independent. The first-repeat subset cannot reproduce the authors' 25-repeat training experiment. No offsets, fitted corrections, ESS or Delta G are established here.

Keep [issue #9](https://github.com/georgpou/corr-fep/issues/9) open until a reviewed numerical run establishes which conclusions are supported.

## Follow-up: direct native coordinate-precision experiment (B2.2b)

The first B2.2 audit established that the deposited Potential has 2401 exactly spaced records and selected XTC times match. Yet the same-TPS re-evaluated energies exceed the deposited original by **258.51, 303.24, 261.64 and 317.73 kJ/mol**. Original archived and rebuilt full-system TPR reruns agree to ~1e-8 kJ/mol.

An eight-evaluation symmetric perturbation diagnostic (+/-0.005 Å independently per coordinate on four XTC frames) has mean total-energy increase **283.76 kJ/mol**, mainly **bond +240.54**, **angle +35.13**, **short-range LJ +8.04** kJ/mol. This is suggestive, **not** a proof that actual deposited XTC precision produced the discrepancy. Because both + and - perturbations increase energy, strong local curvature/bond stiffness is a plausible mechanism.

The script `scripts/stage_b_precision_roundtrip.py` and workflow `.github/workflows/stage-b-precision-roundtrip.yml` run one **new** 0.5 ps NVT trajectory from the real archived water model, recording uncompressed TRR and lossy XTC on identical simulation timesteps. Double-precision GROMACS then reruns *both native trajectories under the same TPR* and compares their EDR Potential and components (bond, angle, dihedrals, LJ, Coulomb and dispersion). It also compares each rerun with the energy recorded during MD.

This is a **mechanistic test on newly generated frames**, NOT restoration of the original full-precision TRR or a substitute for the original dataset. An exact numerical match between a new compressed-minus-uncompressed bias and the observed original residual would support compression as the dominant cause; matching only scale does not uniquely establish causation. No fitting, offset, flow training, or free-energy inference is permitted.

**Status: completed successfully** in [run 38092197810](https://github.com/georgpou/corr-fep/actions/runs/38092197810) ([artifact 11684014638](https://github.com/georgpou/corr-fep/actions/runs/38092197810/artifacts/11684014638)). The 11-frame all-time mean includes the initial t=0 frame (only +0.2925 kJ/mol), whereas the **10 noninitial frames** give a **+263.338 kJ/mol** mean, with bond +231.793 and angle +38.019 kJ/mol. The JSON and CSV now explicitly distinguish these two summary populations. Keep issue #9 open.


## B2.2c completed: original deposited EDR decomposition (2026-10-11)

**Scientific check PASSED** for four strictly time-matched production frames.
Workflow: https://github.com/georgpou/corr-fep/actions/runs/38092879790
Artifact: https://github.com/georgpou/corr-fep/actions/runs/38092879790/artifacts/11684787880
Input archive member: `lig_p38a_2g/eq/water/repeat-0/ener.edr` (2,948,820 B).
Machine-readable result: `results/stage-b/original-edr-decomposition/original-edr-decomposition.json` (Actions artifact).
CSV: `original-edr-components.csv`.

The original GROMACS **2021.6 double** energy EDR and selected `ener.xvg` Potential differ by at most **4.95e-7 kJ/mol**. The four timestamp matches are exact. Original archived `eq.tpr` versus rebuilt single-point `full.tpr`, both rerun in **2023.3 double** with the same XTC-derived G96 input, differ by at most **1.31e-8 kJ/mol**. All original-vs-rerun EDR energy component differences account for their total potential differences to **2.59e-11 kJ/mol**.

| Time (ps) | Original Potential (kJ/mol) | Rerun minus original (kJ/mol) | Bond | Angle | Other net |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2005 | -117320.952573 | +258.513927 | +251.214173 | +51.505982 | -44.206228 |
| 5335 | -116908.160902 | +303.235306 | +250.172854 | +15.686662 | +37.375790 |
| 8665 | -117655.515194 | +261.642699 | +187.720759 | +16.570971 | +57.350969 |
| 12000 | -117121.358510 | +317.728542 | +252.878021 | +62.944590 | +1.905931 |
| **Mean** | | **+285.280119** | **+235.496452** | **+36.677051** | **+13.106616** |

**95.41%** of mean Potential difference is bond plus angle increase (272.1735/285.2801 kJ/mol). This is consistent with and substantially strengthens the independent paired native TRR/XTC experiment (new ten noninitial snapshots: mean +263.34 kJ/mol, mainly bond/angle). Terms aside from bond/angle have frame-dependent values of either sign; they must not be silently subtracted as a constant.

The original EDR reports `Improper Dih.` while the rebuilt rerun reports `Per. Imp. Dih.`. The audit reports these as distinct EDR labels and accounts for each separately, **without asserting equality of their underlying functional forms**. The mean separate contributions are -5.375 and +5.291 kJ/mol; their net is small. The reported closure proves numeric energy accounting, not functional-form equivalence between GROMACS versions.

**Scientific interpretation:** XTC-coordinate quantization is the predominant, experimentally demonstrated and physically plausible explanation for the magnitude/sign of the discrepancy, but the original full-precision TRR was not deposited, so its exact frame-by-frame contribution cannot be reconstructed; the 2021.6/2023.3 version difference also remains. Thus this is **not** a proof of exact energy equivalence or a license to use an arbitrary offset, corrected absolute energies, or inferred importance weights.

### Implication for Corr-FEP training

The next method-design checkpoint is explicit definition of `U_FF(x)` **on exactly the coordinates used to evaluate `U_QMMM(x)`**. An internally consistent rerun of the complete classical reference potential on each deposited XTC configuration is preferable to pairing XTC QM/MM evaluations with the authors' original `ener.xvg` energies at higher-precision positions. This will define a computationally consistent correction observable, but because rounded XTC differs from the actual original trajectory and original sampling measure, **the resulting weights cannot automatically be claimed to reproduce the original TFEP estimator or unbiased free energies**.

No flow training or free-energy results are asserted here. Keep scientific issue #9 open until the chosen consistent reference definition and its sampling implications are documented as a deliberate modeling decision. PR #11 remains unmerged pending user approval.

## PR #11 independent review corrections

After review, the method now preserves separately named rebuilt and archived GROMACS EDR files and the exact G96 inputs for each frame, with SHA-256 hashes and artifact inclusion. All three workflow push path filters include the scientific helper modules so changes to coordinate or energy evaluation code trigger reevaluation. The paired TRR/XTC precision result reports 11-frame and noninitial 10-frame statistics separately. The original EDR component summation is named **reported-term bookkeeping** rather than shared-term equality because 2021.6 and 2023.3 give different improper-dihedral labels. Documentation links above reference successful experimental runs, not historical setup failures. Latest rerun status must be checked on GitHub; these corrections do not retroactively validate new commits.
