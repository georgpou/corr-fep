#!/usr/bin/env python3
"""Verbatim helpers and archived data partitions for author baseline."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from stage_b_first_flow_training import check_finite

SEED = 20261011
TRAIN = np.arange(401, 1051, dtype=int)
VALIDATION = np.arange(1151, 1401, dtype=int)
TEST = np.arange(1501, 1801, dtype=int)
CALIBRATION = np.unique(np.linspace(TRAIN[0], TRAIN[-1], 32, dtype=int))

def record(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_driver():
    path = Path('upstream-stage-b/run_tfep.py').resolve()
    spec = importlib.util.spec_from_file_location('pinned_paper_driver', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def save_weights(mapper,optimizer,scheduler,path,step,metadata):
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in mapper._flow.state_dict().items()},
        optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
        step=step,seed=SEED,metadata=metadata),path)


def batch_geometry(mapper,x,y,logj,indices,fixed,bonds,ligand_idx):
    """Same per-frame gates in one batched inverse; avoids 32 serial inverses."""
    with torch.no_grad():
        reverse=mapper.inverse(dict(positions=y.detach(),trajectory_sample_index=torch.as_tensor(indices)))
        check_finite(reverse['positions'],'batch inverse positions')
        check_finite(reverse['log_det_J'],'batch inverse logdet')
        p=y.detach().reshape(len(indices),-1,3); delta=(y.detach()-x).reshape(len(indices),-1,3)
        lengths=torch.linalg.vector_norm(p[:,bonds[:,0]]-p[:,bonds[:,1]],dim=-1)
        values=dict(inverse_max_abs_error_angstrom=(reverse['positions']-x).abs().amax(dim=1),
            inverse_logdet_closure=(reverse['log_det_J']+logj.detach()).abs(),
            fixed_environment_error_angstrom=delta[:,fixed].abs().flatten(1).amax(dim=1),
            min_bond_angstrom=lengths.amin(dim=1),max_bond_angstrom=lengths.amax(dim=1),
            max_ligand_displacement_angstrom=delta[:,ligand_idx].abs().flatten(1).amax(dim=1))
        rows=[{key:float(value[i]) for key,value in values.items()} for i in range(len(indices))]
    for frame,row in zip(indices,rows):
        if (row['inverse_max_abs_error_angstrom']>1e-5 or row['inverse_logdet_closure']>1e-5
            or row['fixed_environment_error_angstrom']>1e-10 or row['min_bond_angstrom']<0.6
            or row['max_bond_angstrom']>2.5 or row['max_ligand_displacement_angstrom']>2.0):
            raise AssertionError('Training batch geometry failed at frame '+str(frame)+': '+str(row))
    return rows

