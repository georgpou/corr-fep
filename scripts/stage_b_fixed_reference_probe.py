#!/usr/bin/env python3
"""Diagnostic proposal: fixed solvent reference atoms, no energies/training.

This probes a documented representation change; it does not adopt that
change in the benchmark, transfer old weights, or claim trained validity.
"""
import json
import os
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import torch

from stage_b_extended_cloud_benchmark import CALIBRATION, SEED, BATCH, make_map, check_geometry, record
from stage_b_frozen_coordinate_audit import FRAMES
from stage_b_training_interface import SequentialTrainingPotential, TEMPERATURE_K
from stage_b_mechanical_embedding import (
    ROOT, MechanicalEmbedding, grompp, prepare_files, retrieve_topology, setup_coordinates,
)
from stage_b_multiframe import load_trajectory, register_trajectory_transformations
from stage_b_qm_singlepoint import load_archive

OUT = Path("results/stage-b/fixed-reference-probe").resolve()


def select_reference(u, ligand_idx):
    """One fixed water identity, chosen and checked on calibration frames only."""
    u.trajectory[int(CALIBRATION[0])]
    center = u.atoms.positions[ligand_idx].mean(axis=0)
    candidates = []
    for residue in u.select_atoms("resname SOL").residues:
        atoms = residue.atoms
        oxygen = [int(a.index) for a in atoms if 15.0 < float(a.mass) < 17.0]
        hydrogen = [int(a.index) for a in atoms if 0.8 < float(a.mass) < 1.2]
        if len(atoms) == 3 and len(oxygen) == 1 and len(hydrogen) == 2:
            ids = oxygen + sorted(hydrogen)
            candidates.append((float(np.linalg.norm(u.atoms.positions[ids[0]]-center)), ids))
    candidates.sort(key=lambda pair: pair[0])
    # All reference identity/geometry decisions are locked before validation.
    for distance, ids in candidates[:32]:
        checks = []
        for frame in CALIBRATION:
            u.trajectory[int(frame)]
            p = np.asarray(u.atoms.positions[ids], dtype=np.float64)
            a, b = p[1]-p[0], p[2]-p[0]
            lengths = np.linalg.norm([a, b], axis=1)
            sine = np.linalg.norm(np.cross(a,b))/np.prod(lengths)
            checks.append((float(lengths.min()), float(lengths.max()), float(sine)))
        if all(0.7 <= lo <= hi <= 1.2 and sine > 0.5 for lo,hi,sine in checks):
            return ids, dict(selection="nearest admissible water at train frame 401",
                             initial_oxygen_ligand_centroid_distance_angstrom=distance,
                             calibration_indices=CALIBRATION.tolist(),
                             calibration_min_oh_angstrom=min(c[0] for c in checks),
                             calibration_max_oh_angstrom=max(c[1] for c in checks),
                             calibration_min_angle_sine=min(c[2] for c in checks))
    raise AssertionError("No stable unsplit calibration-only water reference found")


def derivative(mapper, x, indices):
    flat_indices = torch.tensor((np.asarray(indices)[:,None]*3+np.arange(3)).reshape(-1))
    v = x[flat_indices].detach().clone().requires_grad_(True)
    def forward(ligand):
        full = x.clone()
        full[flat_indices] = ligand
        return mapper({"positions": full[None,:]})["positions"][0,flat_indices]
    # Solvent reference atoms stay fixed in this full-system derivative.
    jac = torch.autograd.functional.jacobian(forward, v, vectorize=False)
    sign, numerical = torch.linalg.slogdet(jac)
    reported = mapper({"positions": x[None,:]})["log_det_J"][0]
    direction = torch.cos(torch.arange(len(v), dtype=v.dtype))
    direction = direction/torch.linalg.vector_norm(direction)
    h = 1e-5
    fd = (forward(v.detach()+h*direction)-forward(v.detach()-h*direction))/(2*h)
    tangent_error = (fd-jac@direction).abs().max()
    result = dict(sign=float(sign), numerical_logabsdet=float(numerical.detach()),
                  reported_logabsdet=float(reported.detach()),
                  absolute_logdet_discrepancy=float((numerical-reported).abs().detach()),
                  finite_difference_direction_max_error=float(tangent_error.detach()))
    if float(sign) != 1 or result["absolute_logdet_discrepancy"] > 1e-5 or result["finite_difference_direction_max_error"] > 1e-5:
        raise AssertionError("Fixed-reference Cartesian determinant check failed: "+str(result))
    return result


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    report = dict(diagnostic_only=True, benchmark_validated=False, physical_training_steps=0,
                  qm_energy_evaluations=0, git_sha=os.environ.get("GITHUB_SHA","local"),
                  frames=FRAMES, stages=[])
    path = OUT/"fixed-reference-probe.json"
    record(path,report)
    raw=load_archive()
    retrieve_topology()
    prepare_files()
    numbers,_,xyz,ligand_idx,box,_=setup_coordinates(raw)
    xtc,trajectory_info=load_trajectory()
    grompp()
    potential=SequentialTrainingPotential(MechanicalEmbedding(ligand_idx,numbers,box))
    original=make_map(potential,xtc,ligand_idx)
    u=mda.Universe(str(ROOT/"archived-eq.tpr"),str(xtc))
    register_trajectory_transformations(u,ligand_idx)
    refs,selection=select_reference(u,ligand_idx)
    torch.manual_seed(SEED)
    mapper=type(original)(potential_energy_func=potential,
        topology_file_path=str(ROOT/"archived-eq.tpr"),coordinates_file_path=str(xtc),
        temperature=TEMPERATURE_K*potential.energy_unit._REGISTRY.kelvin,
        mapped_atoms=ligand_idx.tolist(),conditioning_atoms=sorted(refs),
        origin_atom=refs[0],axes_atoms=refs[1:],batch_size=BATCH,n_maf_layers=1,
        tfep_logger_dir_path=str(OUT/"tfep-logs"))
    mapper.setup("fit")
    report.update(reference_atom_indices=refs,reference_selection=selection,trajectory=trajectory_info,
                  methodological_proposal="three fixed solvent conditioning/reference atoms; ligand remains the only mapped molecule")
    fixed=torch.ones(len(xyz),dtype=torch.bool)
    fixed[torch.tensor(ligand_idx)]=False
    bonds=torch.tensor(u.select_atoms("resname MOL").bonds.indices,dtype=torch.long)
    initial=[p.detach().clone() for p in mapper._flow.parameters()]
    # This is numerical stress, explicitly not optimizer learning or fake QM.
    for amplitude in (0.0,0.001,0.03):
        generator=torch.Generator().manual_seed(SEED+77)
        with torch.no_grad():
            for p,start in zip(mapper._flow.parameters(),initial):
                p.copy_(start+amplitude*torch.randn(p.shape,generator=generator,dtype=p.dtype))
        stage=dict(parameter_noise_amplitude=amplitude,rows=[])
        report["stages"].append(stage)
        for block,frames in FRAMES.items():
            for frame in frames:
                u.trajectory[frame]
                x=torch.tensor(np.asarray(u.atoms.positions,dtype=np.float64).copy().reshape(-1))
                row=dict(block=block,frame_index=frame,
                         geometry=check_geometry(mapper,x,fixed,bonds,ligand_idx))
                if frame in (726,1275):
                    row["independent_cartesian_jacobian"]=derivative(mapper,x,ligand_idx)
                stage["rows"].append(row)
                record(path,report)
                print("B25_FIXED_REFERENCE_PROBE",amplitude,frame,json.dumps(row,allow_nan=False),flush=True)
    report["stage"]="complete"
    report["limitation"]="Nontrivial parameter perturbations are not physical training; adoption requires a methodological decision and real-QM/MM post-optimization validation."
    record(path,report)
    print("B25_FIXED_REFERENCE_PROBE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
