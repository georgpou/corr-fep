#!/usr/bin/env python3
"""No-QM audit of the actual B2.5 trained weights and Cartesian map.

The reference-shift ablation is diagnostic only. It is never checkpoint
selection, production training, or permission to change the benchmark model.
"""
import argparse
import json
import os
from pathlib import Path
import random

import MDAnalysis as mda
import numpy as np
import torch

from stage_b_extended_cloud_benchmark import (
    SEED, make_map, detailed_inverse_probe, check_geometry, record,
)
from stage_b_training_interface import SequentialTrainingPotential
from stage_b_mechanical_embedding import (
    ROOT, MechanicalEmbedding, digest, grompp, prepare_files,
    retrieve_topology, setup_coordinates,
)
from stage_b_multiframe import load_trajectory, register_trajectory_transformations
from stage_b_qm_singlepoint import load_archive

OUT = Path("results/stage-b/frozen-coordinate-audit").resolve()
FRAMES = {"train": [401, 726, 1050], "validation": [1151, 1275, 1400]}


class IdentityReference(torch.nn.Module):
    """Disable only the six reference shifts in this diagnostic ablation."""
    def forward(self, x, parameters):
        return x, x.new_zeros(len(x))

    def inverse(self, y, parameters):
        return self.forward(y, parameters)


def coordinate_bridge(mapper, x):
    """Measure C(D(f(C(x)))) against f(C(x)), before any inverse MAF."""
    with torch.no_grad():
        partial = mapper._flow
        mixed = partial.flow
        ligand = x[None, partial._propagated_indices]
        z, _, origin, rotation = mixed.cartesian_to_mixed(ligand)
        zout, _ = mixed.flow(z)
        cart, _ = mixed.mixed_to_cartesian(zout, origin, rotation)
        recoded, _, new_origin, new_rotation = mixed.cartesian_to_mixed(cart)
        refs = mixed.get_dof_indices_by_type(None)["reference"]
        mask = torch.ones(z.shape[1], dtype=torch.bool)
        mask[refs] = False
        error = (recoded - zout).abs()[0]
        v, ix = torch.topk(error[mask], min(8, int(mask.sum())))
        all_indices = torch.arange(z.shape[1])[mask]
        return dict(
            reference_dof_indices=refs.tolist(),
            input_reference_values=z[0, refs].tolist(),
            shifted_reference_values=zout[0, refs].tolist(),
            recoded_reference_values=recoded[0, refs].tolist(),
            nonreference_recoding_max_abs=float(error[mask].max()),
            worst_nonreference_dof_indices=all_indices[ix].tolist(),
            worst_nonreference_dof_errors=v.tolist(),
            origin_change_max_angstrom=float((new_origin-origin).abs().max()),
            rotation_matrix_change_max_abs=float((new_rotation-rotation).abs().max()),
        )


def cartesian_jacobian(mapper, x):
    """Independent derivative of the full ligand Cartesian map (138 by 138).

    The environment is copied unchanged by PartialFlow, so its identity block
    contributes zero log determinant. Do not hold the molecular frame fixed:
    its coordinate dependence is part of the derivative being audited.
    """
    mixed = mapper._flow.flow
    ligand = x[mapper._flow._propagated_indices].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        jac = torch.autograd.functional.jacobian(
            lambda v: mixed(v[None, :])[0][0], ligand, vectorize=False)
        sign, numerical = torch.linalg.slogdet(jac)
        _, reported = mixed(ligand[None, :])
    if not torch.isfinite(jac).all() or not torch.isfinite(numerical):
        raise AssertionError("Nonfinite independent Cartesian derivative")
    return dict(sign=float(sign), numerical_logabsdet=float(numerical.detach()),
                reported_logabsdet=float(reported[0].detach()),
                absolute_logdet_discrepancy=float((numerical-reported[0]).abs().detach()),
                n_ligand_cartesian_dofs=ligand.numel())


def measure(mapper, x, fixed, bonds, ligand_idx, derivative=False):
    result = detailed_inverse_probe(mapper, x, fixed)
    result["coordinate_bridge"] = coordinate_bridge(mapper, x)
    try:
        result["geometry"] = check_geometry(mapper, x, fixed, bonds, ligand_idx)
        result["original_geometry_gate_passes"] = True
    except AssertionError as exc:
        result["original_geometry_gate_passes"] = False
        result["geometry_gate_error"] = str(exc)
    if derivative:
        result["independent_cartesian_jacobian"] = cartesian_jacobian(mapper, x)
    return result


def main(artifact_dir):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    report = dict(diagnostic_only=True, benchmark_validated=False,
                  qm_energy_evaluations=0, source_run_id=38148045379,
                  git_sha=os.environ.get("GITHUB_SHA", "local"),
                  frames=FRAMES, checkpoints=[], stage="initializing")
    destination = OUT / "coordinate-audit.json"
    record(destination, report)
    raw = load_archive()
    retrieve_topology()
    prepare_files()
    numbers, _, xyz, ligand_idx, box, _ = setup_coordinates(raw)
    xtc, trajectory_info = load_trajectory()
    grompp()
    potential = SequentialTrainingPotential(MechanicalEmbedding(ligand_idx, numbers, box))
    # This is the real potential and original setup; no energy call is made.
    mapper = make_map(potential, xtc, ligand_idx)
    u = mda.Universe(str(ROOT / "archived-eq.tpr"), str(xtc))
    register_trajectory_transformations(u, ligand_idx)
    fixed = torch.ones(len(xyz), dtype=torch.bool)
    fixed[torch.tensor(ligand_idx)] = False
    bonds = torch.tensor(u.select_atoms("resname MOL").bonds.indices, dtype=torch.long)
    samples = []
    for block, indices in FRAMES.items():
        for frame in indices:
            u.trajectory[frame]
            x = torch.tensor(np.asarray(u.atoms.positions, dtype=np.float64).copy().reshape(-1))
            samples.append((block, frame, x))
    mapper({"positions": samples[0][2][None, :]})  # materialize derived buffer
    report["trajectory"] = trajectory_info
    reference_modules = mapper._flow.flow.flow[0]._transformer._transformers
    reference_indices = [i for i, t in enumerate(reference_modules)
                         if t.__class__.__name__ == "VolumePreservingShiftTransformer"]
    if len(reference_indices) != 1:
        raise AssertionError("Expected one original reference-shift transformer")
    iref = reference_indices[0]
    original_ref = reference_modules[iref]
    for name in ("first-invalid.pt", "last-epoch1.pt"):
        paths = list(artifact_dir.rglob(name))
        if len(paths) != 1:
            raise AssertionError(f"Expected one actual trained checkpoint: {name}")
        checkpoint = torch.load(paths[0], map_location="cpu", weights_only=False)
        mapper._flow.load_state_dict(checkpoint["state_dict"], strict=True)
        item = dict(name=name, sha256=digest(paths[0].read_bytes()),
                    optimizer_step=checkpoint["step"], original=[], reference_shift_ablation=[])
        report["checkpoints"].append(item)
        for mode in ("original", "reference_shift_ablation"):
            if mode == "reference_shift_ablation":
                reference_modules[iref] = IdentityReference()
            try:
                for block, frame, x in samples:
                    metrics = measure(mapper, x, fixed, bonds, ligand_idx, derivative=(frame == 726))
                    metrics.update(block=block, frame_index=frame)
                    item[mode].append(metrics)
                    record(destination, report)
                    print("B25_FROZEN_COORDINATE_AUDIT", name, mode, frame,
                          json.dumps(metrics, allow_nan=False), flush=True)
            finally:
                reference_modules[iref] = original_ref
        # The untouched frozen map must still reproduce the archived probe.
        archived = next(artifact_dir.rglob("inverse-diagnostic.json"))
        stages = json.loads(archived.read_text())["stages"]
        prior = next(s for s in stages if s["optimizer_step"] == checkpoint["step"])
        current = next(s for s in item["original"] if s["frame_index"] == 726)
        diffs = {key: abs(current[key]-prior[key]) for key in (
            "cartesian_inverse_max_abs_angstrom", "cartesian_logjac_closure",
            "mixed_inverse_max_abs", "direct_spline_inverse_max_abs")}
        item["archived_probe_replay_absolute_differences"] = diffs
        if max(diffs.values()) > 1e-8:
            raise AssertionError("Frozen map does not reproduce original diagnostic")
    report["stage"] = "complete"
    report["interpretation"] = (
        "Reference-shift ablation and full Cartesian autodiff are independent "
        "diagnostics. No production correction, energy improvement, or benchmark "
        "validation is claimed. Any proposed model/Jacobian change requires a "
        "documented mathematical justification and fresh real-QM/MM experiment.")
    record(destination, report)
    print("B25_FROZEN_COORDINATE_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.artifact_dir)
