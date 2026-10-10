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
