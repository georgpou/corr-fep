#!/usr/bin/env python3
"""B2.3: real serial-batch differentiable QM/MM and reduced mapped work."""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import time

import MDAnalysis as mda
import numpy as np
import torch

from stage_b_mechanical_embedding import (
    OUT as B1_OUT, ROOT, GMX, MechanicalEmbedding, digest, grompp,
    prepare_files, retrieve_topology, run, setup_coordinates)
from stage_b_multiframe import (
    load_trajectory, register_trajectory_transformations, sampled_frames,
    check_terminal_alignment, complete_gradient_fd)
from stage_b_qm_singlepoint import ARCHIVE, GRO_NAME, load_archive

OUT = Path("results/stage-b/training-interface").resolve()
TEMPERATURE_K = 298.0  # Upstream run_tfep.py
BETA = 1.0 / (0.00831446261815324 * TEMPERATURE_K)


class SequentialTrainingPotential(torch.nn.Module):
    """(B,3N),(B,6) -> (B,) using the validated B1 potential and autograd.

    GROMACS has one working directory per potential. Serial iteration avoids
    overlapping reruns; TFEP caches GROMACS forces in each autograd context.
    """
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    @property
    def positions_unit(self):
        """Use the existing GROMACS potential's pint registry and Å units."""
        return self.embedding.mm_full.positions_unit

    @property
    def energy_unit(self):
        """Match the GROMACS/tblite output unit used in training loss."""
        return self.embedding.mm_full.energy_unit

    def _call(self, positions, cells, target):
        if (positions.ndim != 2 or positions.shape[1] % 3
                or positions.shape[0] == 0 or cells.shape != (len(positions), 6)):
            raise ValueError("Expected (B,3N) coordinates, (B,6) cells")
        if not torch.isfinite(positions).all() or not torch.isfinite(cells).all():
            raise ValueError("Nonfinite input")
        func = self.embedding if target else self.embedding.mm_full
        return torch.cat([func(positions[i:i+1], cells[i:i+1])
                          for i in range(len(positions))], dim=0)

    def forward(self, positions, cells):
        return self._call(positions, cells, target=True)

    def reference(self, positions, cells):
        return self._call(positions, cells, target=False)


def ligand_affine_map(flat, indices, scale, shift):
    """Invertible diagnostic ligand map, NOT the published MixedMAF.

    Translation mode is unchanged under scaling about the ligand centroid.
    Its log|J| is therefore 3*(n_ligand-1)*log(scale).
    """
    xyz = flat.reshape(len(flat), -1, 3)
    ligand = xyz[:, indices]
    centroid = ligand.mean(dim=1, keepdim=True)
    mapped = xyz.clone()
    mapped[:, indices] = centroid + scale*(ligand-centroid) + shift
    logdet = 3*(len(indices)-1)*torch.log(scale)
    return mapped.reshape_as(flat), logdet.expand(len(flat))


def verify_jacobian():
    x = torch.tensor([1., 0., 0., 2., 1., 1.], dtype=torch.float64)
    idx = torch.tensor([0, 1])
    scale = torch.tensor(1.013, dtype=torch.float64)
    shift = torch.tensor([0.002, -0.003, 0.001], dtype=torch.float64)
    def f(v):
        return ligand_affine_map(v[None, :], idx, scale, shift)[0][0]
    jac = torch.autograd.functional.jacobian(f, x)
    sign, computed = torch.linalg.slogdet(jac)
    predicted = 3*torch.log(scale)
    err = abs(float(computed-predicted))
    if float(sign) != 1.0 or err > 1e-10:
        raise AssertionError("Independent affine-map Jacobian check failed")
    return dict(sign=float(sign), logabsdet=float(computed),
                analytic=float(predicted), abs_error=err)



def real_mixedmaf_integration(potential, trajectory_path, ligand_indices,
                              reference_positions, reference_box):
    """Small *real* upstream MixedMAF forward/backward checkpoint, no optimizer.

    Trainability here means actual flow-parameter gradients through real
    mechanical-embedding forces; a stable learning experiment is B2.4.
    """
    import tfep.app.mixedmaf
    import tfep.io

    # Analyze only 12 distinct archived production frames for fast flow
    # initialization. The full original XTC and TPR supply real atom identities,
    # topology, coordinates and original periodic cells, with B2.1 transforms.
    subset = np.linspace(401, 2400, 12, dtype=int)
    class RealWaterMixedMAF(tfep.app.mixedmaf.MixedMAFMap):
        def create_universe(self):
            u = mda.Universe(str(ROOT / "archived-eq.tpr"),
                             str(trajectory_path))
            register_trajectory_transformations(u, ligand_indices)
            return u

        def create_dataset(self):
            return tfep.io.TrajectorySubset(super().create_dataset(), subset)

    registry = potential.energy_unit._REGISTRY
    mapper = RealWaterMixedMAF(
        potential_energy_func=potential,
        topology_file_path=str(ROOT / "archived-eq.tpr"),
        coordinates_file_path=str(trajectory_path),
        temperature=TEMPERATURE_K * registry.kelvin,
        mapped_atoms=ligand_indices.tolist(),
        batch_size=2,
        n_maf_layers=1,  # Only an upstream architectural smoke, not paper training.
        tfep_logger_dir_path=str(OUT / "mixedmaf-logs"),
    )
    t0 = time.perf_counter()
    mapper.setup("fit")
    thermal_energy = float(mapper._kT)
    if abs(thermal_energy - 1/BETA) > 1e-9:
        raise AssertionError("TFEP thermal energy is not kJ/mol")
    samples = [mapper.dataset[i] for i in range(2)]
    positions = torch.stack([a["positions"] for a in samples]).to(torch.float64)
    cells = torch.stack([a["dimensions"] for a in samples]).to(torch.float64)
    input_error = float((positions[0] - reference_positions[0].detach()).abs().max())
    box_error = float((cells[0] - reference_box[0]).abs().max())
    if input_error > 1e-5 or box_error > 1e-4:
        raise AssertionError("Upstream dataset changed archived frame coordinates/box")
    output = mapper({"positions": positions})
    mapped = output["positions"]
    logdet = output["log_det_J"]
    if mapped.shape != positions.shape or logdet.shape != (2,):
        raise AssertionError("Original MixedMAF output shape mismatch")
    # Only ligand atoms can move; untouched solvent/environment must survive
    # original full-coordinate ordering.
    env = torch.ones(positions.shape[1] // 3, dtype=torch.bool)
    env[torch.as_tensor(ligand_indices, dtype=torch.long)] = False
    fixed_error = float((mapped.reshape(2, -1, 3)[:, env] -
                         positions.reshape(2, -1, 3)[:, env]).abs().max())
    if fixed_error > 1e-10:
        raise AssertionError("Flow moved nominally fixed water atoms")
    target = potential(mapped, cells)
    loss = (target / mapper._kT - logdet).mean()
    parameters = [p for p in mapper._flow.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    finite_gradients = [g for g in gradients if g is not None]
    if (len(parameters) == 0 or len(finite_gradients) == 0
            or not torch.isfinite(target).all() or not torch.isfinite(logdet).all()
            or not torch.isfinite(loss)
            or not all(torch.isfinite(g).all() for g in finite_gradients)):
        raise AssertionError("Real upstream MixedMAF forward/loss/backward failed")
    nonzero_gradients = sum(int(torch.count_nonzero(g) > 0) for g in finite_gradients)
    if nonzero_gradients == 0:
        raise AssertionError("MixedMAF did not receive any trainable gradient")
    return dict(
        architecture="pinned tfep.app.mixedmaf.MixedMAFMap; 1 layer, ligand-only",
        claim="One upstream forward/backward, no optimizer, no convergence/DF claim",
        dataset_archive_indices=subset.tolist(),
        train_batch_archive_indices=subset[:2].tolist(),
        input_shape=list(positions.shape), output_shape=list(mapped.shape),
        logdet_shape=list(logdet.shape),
        thermal_energy_kj_mol=thermal_energy,
        coordinate_replay_max_error_angstrom=input_error,
        cell_replay_max_error_angstrom_degrees=box_error,
        fixed_environment_max_error_angstrom=fixed_error,
        target_kj_mol=target.detach().tolist(),
        logdet=logdet.detach().tolist(),
        mean_reduced_target_minus_logdet=float(loss.detach()),
        trainable_parameter_tensors=len(parameters),
        gradient_tensors=len(finite_gradients),
        nonzero_gradient_tensors=nonzero_gradients,
        largest_abs_parameter_gradient=max(float(g.detach().abs().max())
                                           for g in finite_gradients),
        seconds=time.perf_counter() - t0,
    )

def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    for path in (ROOT, OUT, B1_OUT):
        path.mkdir(parents=True, exist_ok=True)
    raw = load_archive()
    archived_top = retrieve_topology()
    upstream = prepare_files()
    numbers, charge, gro_xyz, idx, gro_box, atom_audit = setup_coordinates(raw)
    if charge != 0:
        raise AssertionError("Expected neutral ligand")
    xtc_path, xtc_info = load_trajectory()
    version = run(GMX, "--version")
    if not re.search(r"^\s*Precision:\s+double\b", version, re.M | re.I):
        raise RuntimeError("Requires double-precision GROMACS")
    grompp()
    universe = mda.Universe(str(ROOT / "archived-eq.tpr"), str(xtc_path))
    register_trajectory_transformations(universe, idx)
    alignment, _, _ = check_terminal_alignment(
        universe, gro_xyz, idx, gro_box, raw[GRO_NAME].decode().splitlines()[0])
    if (alignment["after_translation"]["rms_displacement_angstrom"] > 0.001
            or alignment["after_translation"]["max_displacement_angstrom"] > 0.02):
        raise AssertionError("B2.1 XTC/GRO alignment gate failed")
    frames = list(sampled_frames(universe, idx, gro_xyz))[:3]
    if [f[0] for f in frames] != [401, 1067, 1733]:
        raise AssertionError("Unexpected B2.1 production frame selection")
    positions = torch.tensor(np.stack([f[3].reshape(-1) for f in frames]),
                             dtype=torch.float64, requires_grad=True)
    cells = torch.tensor(np.stack([f[4] for f in frames]), dtype=torch.float64)
    embedding = MechanicalEmbedding(idx, numbers, gro_box)
    potential = SequentialTrainingPotential(embedding)
    idx_t = torch.tensor(idx, dtype=torch.long)
    result = dict(
        checkpoint="B2.3 training interface on archived real QM/MM",
        validated=False, archive=ARCHIVE,
        frame_indices=[f[0] for f in frames], frame_times_ps=[f[2] for f in frames],
        frame_boxes_angstrom_degrees=[f[4].tolist() for f in frames],
        ligand_indices=idx.tolist(), atom_count=len(gro_xyz),
        batch_positions_shape=list(positions.shape), batch_cells_shape=list(cells.shape),
        units={"position": "angstrom", "energy": "kJ/mol",
               "gradient": "kJ/mol/angstrom", "reduced_work": "dimensionless"},
        temperature_K=TEMPERATURE_K, beta_mol_per_kj=BETA,
        warning="One repeat only; XTC is lossy; not a free-energy estimate",
        provenance=dict(
            archive_members={k: digest(v) for k, v in raw.items()},
            topology=archived_top, upstream=upstream, trajectory=xtc_info,
            original_atom_audit=atom_audit, gromacs_version=version,
            tpr_sha256={s: digest((ROOT / (s+".tpr")).read_bytes())
                        for s in ("full", "ligand")},
            versions={p: importlib.metadata.version(p)
                      for p in ("torch", "numpy", "MDAnalysis", "tblite", "tfep")},
            git_sha=os.environ.get("GITHUB_SHA", "local")))
    def save():
        (OUT/"training-interface.json").write_text(json.dumps(
            result, indent=2, allow_nan=False)+"\n")
    save()

    start = time.perf_counter()
    reference = potential.reference(positions, cells)
    target = potential(positions, cells)
    grad_ref, = torch.autograd.grad(reference.sum(), positions, retain_graph=True)
    grad_target, = torch.autograd.grad(target.sum(), positions)
    batch_seconds = time.perf_counter()-start
    if (reference.shape != (3,) or target.shape != (3,)
            or not all(torch.isfinite(x).all() for x in
                       (reference, target, grad_ref, grad_target))):
        raise AssertionError("Nonfinite or misshapen batch")
    result["unmapped"] = dict(
        reference_ff_kj_mol=reference.detach().tolist(),
        target_qmmm_kj_mol=target.detach().tolist(),
        same_x_delta_kj_mol=(target-reference).detach().tolist(),
        reference_gradient_norms_kj_mol_A=torch.linalg.vector_norm(
            grad_ref, dim=1).detach().tolist(),
        target_gradient_norms_kj_mol_A=torch.linalg.vector_norm(
            grad_target, dim=1).detach().tolist(),
        seconds=batch_seconds, frames_per_second=3/batch_seconds,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save()

    # Accepted B2.1 implementation as independent one-frame cross-check.
    first = positions[:1].detach().clone().requires_grad_(True)
    first_ref = embedding.mm_full(first, cells[:1])
    first_target = embedding(first, cells[:1])
    first_grad, = torch.autograd.grad(first_target.sum(), first)
    energy_ref_err = abs(float(first_ref.detach()-reference[0].detach()))
    energy_target_err = abs(float(first_target.detach()-target[0].detach()))
    gradient_err = float((first_grad-grad_target[:1]).abs().max())
    fd = complete_gradient_fd(
        embedding, frames[0][3], frames[0][4],
        first_grad.detach().numpy().reshape(-1, 3), idx)
    if max(energy_ref_err, energy_target_err, gradient_err) > 1e-5:
        raise AssertionError("B2.1 single-frame / serial-batch mismatch")
    result["crosscheck"] = dict(
        frame_index=401, reference_energy_error_kj_mol=energy_ref_err,
        target_energy_error_kj_mol=energy_target_err,
        target_gradient_max_error_kj_mol_A=gradient_err,
        finite_difference_probes=fd)
    save()

    # Diagnostic reversible transform, with a nonzero, independently checked J.
    scale = torch.tensor(1.0002, dtype=torch.float64, requires_grad=True)
    shift = torch.tensor([0.001, -0.001, 0.0005], dtype=torch.float64,
                         requires_grad=True)
    x0 = positions.detach()
    mapped, logdet = ligand_affine_map(x0, idx_t, scale, shift)
    inverse, inv_logdet = ligand_affine_map(
        mapped, idx_t, scale.reciprocal(), -shift)
    inv_error = float((inverse-x0).abs().max())
    logdet_error = float((logdet+inv_logdet).abs().max())
    if inv_error > 1e-9 or logdet_error > 1e-12:
        raise AssertionError("Affine map inverse/Jacobian failed")
    start = time.perf_counter()
    mapped_target = potential(mapped, cells)
    work = BETA*(mapped_target-reference.detach())-logdet
    d_scale, d_shift = torch.autograd.grad(
        work.mean(), (scale, shift), retain_graph=True)
    d_target_dy, = torch.autograd.grad(mapped_target.sum(), mapped)
    ligand_forces = d_target_dy.reshape(3, -1, 3)[:, idx_t]
    expected_shift = BETA*ligand_forces.sum(dim=(0, 1))/len(frames)
    x_lig = x0.reshape(3, -1, 3)[:, idx_t]
    expected_scale = BETA*(ligand_forces*(x_lig-x_lig.mean(
        dim=1, keepdim=True))).sum()/len(frames)-3*(len(idx)-1)/scale
    shift_error = float((d_shift-expected_shift).abs().max())
    scale_error = float((d_scale-expected_scale).abs())
    if (not all(torch.isfinite(t).all() for t in
                (mapped_target, work, d_scale, d_shift, d_target_dy))
            or max(shift_error, scale_error) > 1e-5):
        raise AssertionError("Mapped work/flow-parameter autograd failed")
    result["mapped"] = dict(
        map_type="diagnostic ligand affine (not published MixedMAF)",
        scale=float(scale.detach()), shift_angstrom=shift.detach().tolist(),
        logdet=logdet.detach().tolist(),
        jacobian_check=verify_jacobian(),
        inverse_max_abs_error_angstrom=inv_error,
        inverse_logdet_error=logdet_error,
        mapped_target_kj_mol=mapped_target.detach().tolist(),
        reduced_map_work=work.detach().tolist(),
        gradient_scale=float(d_scale), gradient_shift=d_shift.detach().tolist(),
        gradient_scale_chain_rule_error=scale_error,
        gradient_shift_chain_rule_error=shift_error,
        seconds=time.perf_counter()-start)
    save()

    start = time.perf_counter()
    repeat = potential(x0, cells)
    repeat_error = float((repeat-target.detach()).abs().max())
    result["repeat"] = dict(max_energy_error_kj_mol=repeat_error,
                            seconds=time.perf_counter()-start)
    if repeat_error > 1e-5:
        raise AssertionError("Repeated-call mismatch")
    result["upstream_mixedmaf"] = real_mixedmaf_integration(
        potential, xtc_path, idx, positions, cells)
    save()
    result["validated"] = True
    save()
    print("B23_TRAINING_INTERFACE_VALIDATED", json.dumps(
        {"indices": result["frame_indices"], "unmapped": result["unmapped"],
         "crosscheck": result["crosscheck"], "mapped": result["mapped"],
         "repeat": result["repeat"]}), flush=True)


if __name__ == "__main__":
    main()
