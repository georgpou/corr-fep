# Corr-FEP agent handoff — B2.4: first REAL molecular-flow training

Date: 2026-10-11
Repo: https://github.com/georgpou/corr-fep
Branch: `stage-b/first-real-flow-training`
Base: `main` at `a471bce5dec56b16d8ae524a0c0b1c4a7fc150f5`, B2.3 squash merge
Completed: [PR #12](https://github.com/georgpou/corr-fep/pull/12) merged following explicit owner approval, multiple scientific audits and exact-head CI verification.

## User mandate and engineering rules

**Start actual neural-network optimization now**, not another synthetic exercise or interface-only test. Begin with one ligand `p38a_2g`, water, **repeat-0**. Do not rerun the original A→B alchemical dynamics. Favor focused scientific checkpoints, not a sprawling test harness; code should remain simple and maintainable. You may implement, commit, open/review/fix PRs, and run GitHub Actions, but **always ask before each merge**. The user plans eventual ARM64 H200 cluster use; present GitHub CPU smoke runs are not GPU benchmarks.

## Accepted scientific basis

B1: pinned double-precision GROMACS full MM, ligand-only MM, and tblite GFN1-xTB combined into real subtractive mechanical embedding.

B2.1: actual archived water trajectories and finite-difference-validated gradients. B2.2 and B2.2b/c: deposited XTC is lossy; source `ener.xvg` and EDR evaluate unrounded original MD coordinates, so they must NOT be used directly as FF energies for XTC/flow evaluations. See `docs/STAGE_B_REFERENCE_ENERGY.md` and scientific issue #9.

B2.3: real sequential differentiable batch potential with pinned upstream MixedMAF integration. Script: `scripts/stage_b_training_interface.py`. Documentation: `docs/STAGE_B_TRAINING_INTERFACE.md`. Final source validation: [Actions run #38130123043](https://github.com/georgpou/corr-fep/actions/runs/38130123043), [artifact #11698848503](https://github.com/georgpou/corr-fep/actions/runs/38130123043/artifacts/11698848503), tested exact B2.3 source SHA `7605a0b18700f21436861c27e3e69655ec1e9dd4`; then merged as `a471bce5dec56b16d8ae524a0c0b1c4a7fc150f5`. Both CI checks passed. JSON `validated=true`; archived ZIP SHA256 `8ac433170fd9887f2d595ca579174642086e1460e7fe00f1d8a273ae4e2545dc`.

Real molecular Hamiltonian:
```
U_ref(x) = U_MM_full(x)
U_target(y) = U_MM_full(y) - U_MM_ligand(y_L) + U_GFN1-xTB(y_L)
w(x) = beta*(U_target(M(x))-U_ref(x)) - log|det J_M(x)|
beta = 1/(R*298 K)
```
Coordinates Å, energies kJ/mol, gradients kJ/mol/Å, reduced work dimensionless. Correct per-frame periodic cells. Never cancel full-MM terms at distinct x and M(x). During learning, U_ref is independent of network parameters and may be omitted from optimizer loss, but compute it on **the same archived XTC inputs** for mapped-work diagnostics. Do not substitute high-precision original XVG energies.

Upstream TFEP pin `andrrizzi/tfep@0bc369a1194c688bb6ac997cc6b7c2db2b063c9e`; paper source `andrrizzi/paper-alchemical-tfep-qmmm-2026@b8dfd897ebd914b4e592221710b397a2758353ce`; bgflow pin `noegroup/bgflow@fbba56fac3eb88f6825d2bd4f745ee75ae9715e1`; GROMACS double 2023.3 and tblite 0.3.0. Upstream full experiment uses 6-layer mixed MAF and conditioning/water-neighbor permutation; B2.3 proven only **one-layer ligand-only** upstream MixedMAF as sensible first real training smoke. Do not present it as the full paper architecture.

B2.3 actual upstream model: 12 archived production frames for initial coordinate range/Z-matrix setup; 2 selected samples (frames 401 and 582) in true upstream `tfep.app.mixedmaf.MixedMAFMap` forward/backward. 8420 full-system atoms, 46 ligand atoms, kT=2.477709860209666 kJ/mol. At identity initial map, all water remains fixed, log-J=0, correct coordinate/box replay, finite energies/forces. Nine parameter tensors have gradients, 2 physically nonzero; separate physical-target and log-J gradients both nonzero and sum closes to ~5.63e-13. **This establishes differentiability, not learning or nontrivial Jacobian correctness.**

The adapter exposes `positions_unit` and `energy_unit` from the real GROMACS backend. For GROMACS TPR missing element names, the upstream paper's `fix_universe_elements` approach is applied (including Na/Cl special atom types). Workflow uses pinned `bgflow` and `einops`. GROMACS uses shared working directories and must run **serially** until separate directories are introduced.

## B2.4 immediate experiment

1. On `stage-b/first-real-flow-training`, reuse `SequentialTrainingPotential` and the real upstream `MixedMAFMap` setup from `scripts/stage_b_training_interface.py`. Write ONE focused training script and ONE GitHub Actions workflow. Use fixed seeds, one-layer ligand-only initial flow, a small, time-separated training subset of production frames and a distinct **temporal held-out block**. Preserve original per-frame box, atom order, original transforms (`unwrap ligand → center ligand → wrap solvent`) and atom-element inference. Start at most ~3–10 optimizer steps on a small serial CPU batch; do not extrapolate 3-frame throughput blindly.
2. Compute reference FF on the archived **input x**, preferably cache per selected frame. Use original upstream MAF forward to produce y/log-J and real QM/MM target on y. Optimize mean `U_target(y)/kT - log-J` with explicit PyTorch AdamW and conservative learning rate (original paper LR 1e-4). Use an actual `optimizer.step()`; show model parameters change. Keep all energies, gradients, log-J, mapped coordinates finite; solvent atoms should remain fixed. Record displacement and nonzero post-update log-J, physical/Jacobian gradient contributions, measured runtime.
3. Save before/after on training and held-out temporal frames: reduced map work with recomputed reference FF, mean/variance, map displacement, log-J, losses and gradient norms. Report improvements **or deterioration** as observed. Do not invent success, unbiased ΔG or meaningful ESS from a few correlated frames. Explicitly discuss quantized-XTC sampling-measure mismatch, only one repeat and missing 24 publication repeats.
4. Save JSON/CSV metrics with `validated=false` until checkpoints pass, software pins/hashes, frame indices and times, RNG seeds, optimizer/layer settings, model checkpoint state_dict if manageable, logs/inputs for scientific replay. Upload artifacts on failure too. Audit the numerical artifact (not just green CI); inspect real objective changes. If instability, use a smaller step size/displacement and accurately document it.
5. Write a concise scientific results document, open a PR, review/fix, and **request approval before merging**. No new original FEP runs.

## Acceptance criteria

At least one real parameter update and a finite nonidentity mapped outcome using published MixedMAF + real QM/MM; independent physical force reaches the model, explicit mapped-target/reference work and log-J assessed on time-disjoint held-out input, numerical result and limitations documented. Improvement in held-out variance is desirable but not guaranteed and must be measured honestly. No publication-level free-energy claim.

## Cautions

- Quantized XTC does not demonstrably sample exactly the same reference measure defined by re-evaluated FF; published original `ener.xvg` is from higher-precision coordinates. Do not derive unbiased importance weights or publication-level ΔG from these limited inputs.
- 12 frames for upstream setup and 2-frame identity forward/backward do **not** certify full training dynamics.
- Control real QM/MM calculation expense; previous CPU B2.3 ~0.3–0.46 combined frames/sec and ~0.5 GiB Python peak RSS, excluding child GROMACS memory. ARM64/H200 portability and acceleration are separate checkpoints.
- Scientific checks first, small amount of lean code second; no comprehensive synthetic test scaffolding.

**Status:** B2.3 has been squash-merged and verified on `main`. B2.4 branch has been created at that merge. Proceed to genuine flow training, not more B2.3 demonstrations.
