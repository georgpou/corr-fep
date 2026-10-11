#!/usr/bin/env python3
"""Bounded no-QM gate of the author-conditioned six-layer molecular map."""
import json
import os
from pathlib import Path
import random
import traceback

import MDAnalysis as mda
import numpy as np
import torch
import tfep.app.mixedmaf

from stage_b_author_map import AuthorMixedMAF, BoundFrame, preprocess
from stage_b_extended_cloud_benchmark import SEED, TRAIN, VALIDATION, check_geometry, record
from stage_b_fixed_reference_probe import derivative
from stage_b_frozen_coordinate_audit import FRAMES
from stage_b_training_interface import SequentialTrainingPotential
from stage_b_mechanical_embedding import ROOT, MechanicalEmbedding, grompp, prepare_files, retrieve_topology, setup_coordinates, digest
from stage_b_multiframe import load_trajectory, register_trajectory_transformations
from stage_b_qm_singlepoint import load_archive

OUT=Path('results/stage-b/author-coordinate-probe').resolve()


def prepare(output_dir):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    output_dir.mkdir(parents=True,exist_ok=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    raw=load_archive(); retrieve_topology(); prepare_files()
    numbers,_,xyz,ligand_idx,box,_=setup_coordinates(raw)
    xtc,trajectory=load_trajectory(); grompp()
    potential=SequentialTrainingPotential(MechanicalEmbedding(ligand_idx,numbers,box))
    preprocessing,summary=preprocess(xtc,output_dir)
    torch.manual_seed(SEED)
    mapper=AuthorMixedMAF(potential,xtc,ligand_idx,preprocessing,output_dir)
    mapper.setup('fit')
    if abs(float(mapper._kT)-0.00831446261815324*298.0)>1e-9:
        raise AssertionError('Author-map kT mismatch')
    u=mda.Universe(str(ROOT/'archived-eq.tpr'),str(xtc))
    register_trajectory_transformations(u,ligand_idx)
    fixed=torch.ones(len(xyz),dtype=torch.bool); fixed[torch.tensor(ligand_idx)]=False
    bonds=torch.tensor(u.select_atoms('resname MOL').bonds.indices,dtype=torch.long)
    return mapper,potential,u,ligand_idx,fixed,bonds,preprocessing,dict(trajectory=trajectory,preprocessing=summary)


def sample(u,frame):
    u.trajectory[int(frame)]
    return dict(frame_index=int(frame),positions=torch.tensor(u.atoms.positions.copy().reshape(-1),dtype=torch.float64),
                cell=torch.tensor(u.dimensions.copy(),dtype=torch.float64),time_ps=float(u.trajectory.ts.time))


def reference_gate(u,preprocessing,indices):
    # Stable non-water conditioning atoms (e.g. an ion) retain physical identity.
    slots=[int(np.flatnonzero(preprocessing['ref_water']==r)[0]) if r in preprocessing['ref_water'] else None
           for r in preprocessing['references']]
    minimum_sine=1.0; failed=[]
    for frame in indices:
        u.trajectory[int(frame)]
        actual=[int(preprocessing['matched_water'][frame,slot]) if slot is not None else ref
                for slot,ref in zip(slots,preprocessing['references'])]
        points=torch.tensor(u.atoms.positions[actual].copy(),dtype=torch.float64)
        if tfep.app.mixedmaf.is_collinear(points[None,:]):
            failed.append(int(frame))
        a,b=points[1]-points[0],points[2]-points[1]
        sine=float(torch.linalg.vector_norm(torch.cross(a,b,dim=-1))/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)))
        minimum_sine=min(minimum_sine,sine)
    return dict(frame_count=len(indices),minimum_angle_sine=minimum_sine,collinear_frames=failed)


def probe(mapper,u,ligand_idx,fixed,bonds,include_derivative=True,rows=None,persist=None):
    rows=[] if rows is None else rows
    for block,frames in FRAMES.items():
        for frame in frames:
            s=sample(u,frame); bound=BoundFrame(mapper,frame)
            row=dict(block=block,frame_index=frame)
            rows.append(row)
            if persist is not None: persist()
            row['geometry']=check_geometry(bound,s['positions'],fixed,bonds,ligand_idx)
            if persist is not None: persist()
            if include_derivative and frame in (726,1275):
                row['independent_cartesian_jacobian']=derivative(bound,s['positions'],ligand_idx)
                if persist is not None: persist()
            print('B25_AUTHOR_MAP_PROBE',json.dumps(row,allow_nan=False),flush=True)
    return rows


def checkpoint_reload(mapper,potential,u,ligand_idx,preprocessing,output_dir,path):
    first=sample(u,726)
    batch=dict(positions=first['positions'][None,:],trajectory_sample_index=torch.tensor([726]))
    before=mapper(batch)
    saved=dict(state_dict={k:v.detach().cpu().clone() for k,v in mapper._flow.state_dict().items()},
               seed=SEED,reference_slots=preprocessing['references'],n_maf_layers=6)
    torch.save(saved,path)
    torch.manual_seed(SEED)
    fresh=AuthorMixedMAF(potential,Path(u.trajectory.filename),ligand_idx,preprocessing,output_dir/'reload')
    fresh.setup('fit'); fresh(batch)  # Materialize PartialFlow's derived index buffer.
    fresh._flow.load_state_dict(torch.load(path,weights_only=False)['state_dict'],strict=True)
    after=fresh(batch)
    result=dict(position_max_abs=float((before['positions']-after['positions']).abs().max()),
                logdet_max_abs=float((before['log_det_J']-after['log_det_J']).abs().max()),
                checkpoint_sha256=digest(path.read_bytes()))
    if max(result['position_max_abs'],result['logdet_max_abs'])>1e-10:
        raise AssertionError('Strict author checkpoint reload differs')
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'coordinate-probe.json'
    report=dict(stage='initializing',diagnostic_only=True,benchmark_validated=False,
                qm_energy_evaluations=0,physical_updates=0,git_sha=os.environ.get('GITHUB_SHA','local'),stages=[])
    record(path,report)
    mapper=None
    try:
        mapper,potential,u,ligand_idx,fixed,bonds,preprocessing,metadata=prepare(OUT)
        report.update(metadata)
        report['reference_geometry']=reference_gate(u,preprocessing,np.concatenate([TRAIN,VALIDATION]))
        record(path,report)
        if report['reference_geometry']['collinear_frames']:
            raise AssertionError('Frozen training-selected reference becomes collinear: '+str(report['reference_geometry']['collinear_frames']))
        # Conditional reference DOFs must be absent from the mapped residual path.
        mixed=mapper._flow.flow.flow
        reference_dofs=mixed.get_dof_indices_by_type(mapper.get_conditioning_indices(idx_type='atom',remove_fixed=True))['reference']
        report['mapped_reference_residual_count']=len(reference_dofs)
        if len(reference_dofs):
            raise AssertionError('Author conditioning references still activate residual shifts')
        initial=[p.detach().clone() for p in mapper._flow.parameters()]
        for amplitude in (0.0,0.001):
            stage=dict(parameter_noise_amplitude=amplitude,rows=[])
            report['stages'].append(stage); record(path,report)
            generator=torch.Generator().manual_seed(SEED+77)
            with torch.no_grad():
                for p,start in zip(mapper._flow.parameters(),initial):
                    p.copy_(start+amplitude*torch.randn(p.shape,generator=generator,dtype=p.dtype))
            probe(mapper,u,ligand_idx,fixed,bonds,rows=stage['rows'],persist=lambda:record(path,report))
            record(path,report)
        report['checkpoint_reload']=checkpoint_reload(mapper,potential,u,ligand_idx,preprocessing,OUT,OUT/'perturbed-map.pt')
        report.update(stage='complete',coordinate_gate_passed=True,
            limitation='Sampled parameter stress is not trained validity; real author-configured QM/MM updates are the next gate.')
        record(path,report)
        print('B25_AUTHOR_COORDINATE_GATE_COMPLETE',flush=True)
    except Exception as exc:
        report.update(stage='incomplete',coordinate_gate_passed=False,error=str(exc),traceback=traceback.format_exc())
        record(path,report)
        if mapper is not None:
            torch.save(dict(state_dict=mapper._flow.state_dict(),stages=report['stages']),OUT/'failure-map.pt')
        raise


if __name__=='__main__':
    main()
