#!/usr/bin/env python3
"""Original trained author baseline setup (function bodies copied verbatim)."""
from pathlib import Path
import random
import MDAnalysis as mda
import numpy as np
import torch
import tfep.app.mixedmaf
from stage_b_author_map import AuthorMixedMAF, preprocess
from stage_b_author_baseline_utils import SEED, TRAIN, VALIDATION
from stage_b_training_interface import SequentialTrainingPotential
from stage_b_mechanical_embedding import ROOT, MechanicalEmbedding, grompp, prepare_files, retrieve_topology, setup_coordinates
from stage_b_multiframe import load_trajectory, register_trajectory_transformations
from stage_b_qm_singlepoint import load_archive

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

