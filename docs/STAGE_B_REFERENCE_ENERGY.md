# Stage B2.2 — FF reference-energy alignment

**Status: numerical acceptance pending scientific artifacts.**

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

**Status**: committed, numerical result pending GitHub Actions run https://github.com/georgpou/corr-fep/actions/runs/38091870934 . Inspect paired artifacts and document measured results before releasing PR #11 for review. Keep issue #9 open.
