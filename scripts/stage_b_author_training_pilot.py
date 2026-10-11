#!/usr/bin/env python3
"""Three real-QM/MM updates with the authors' production model/optimizer.

One epoch over 96 training-era samples. This is a bounded integration pilot,
not the complete paper multimap estimator or extended held-out benchmark.
"""
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from stage_b_author_map import BoundFrame
from stage_b_author_coordinate_probe import prepare, sample, probe, reference_gate, checkpoint_reload
from stage_b_extended_cloud_benchmark import SEED, TRAIN, VALIDATION, check_geometry, record
from stage_b_first_flow_training import check_finite, grad_norm
from stage_b_multiframe import complete_gradient_fd

OUT=Path('results/stage-b/author-training-pilot').resolve()
PILOT_TRAIN=np.unique(np.linspace(TRAIN[0],TRAIN[-1],96,dtype=int))


def save_weights(mapper,optimizer,scheduler,path,step,metadata):
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in mapper._flow.state_dict().items()},
        optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
        step=step,seed=SEED,metadata=metadata),path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'training-pilot.json'
    report=dict(stage='initializing',benchmark_validated=False,physical_updates=0,
        git_sha=os.environ.get('GITHUB_SHA','local'),training_indices=PILOT_TRAIN.tolist(),
        heldout_test_evaluations=0,steps=[],online_rows=[],geometry_probes=[],
        author_settings=dict(n_maf_layers=6,batch_size=32,epochs=1,learning_rate=1e-4,
            weight_decay=0.01,exponential_lr_gamma=0.997,gradient_clip_norm=3000.0,dtype='float64'))
    record(path,report)
    mapper=optimizer=scheduler=None
    try:
        mapper,potential,u,ligand_idx,fixed,bonds,preprocessing,metadata=prepare(OUT)
        report.update(metadata)
        report['reference_geometry']=reference_gate(u,preprocessing,np.concatenate([TRAIN,VALIDATION]))
        record(path,report)
        if report['reference_geometry']['collinear_frames']:
            raise AssertionError('Frozen author reference fails train/validation geometry')
        report['initial_probes']=[]
        probe(mapper,u,ligand_idx,fixed,bonds,rows=report['initial_probes'],persist=lambda:record(path,report))
        record(path,report)
        params=[p for p in mapper._flow.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.01)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.997)
        initial=[p.detach().clone() for p in params]
        generator=torch.Generator().manual_seed(SEED)
        order=PILOT_TRAIN[torch.randperm(len(PILOT_TRAIN),generator=generator).numpy()]
        report['training_order']=order.tolist()
        for step in range(1,4):
            indices=order[(step-1)*32:step*32]
            batch=[sample(u,int(f)) for f in indices]
            x=torch.stack([s['positions'] for s in batch]); cells=torch.stack([s['cell'] for s in batch])
            optimizer.zero_grad(set_to_none=True)
            mapped=mapper(dict(positions=x,trajectory_sample_index=torch.as_tensor(indices)))
            y,logj=mapped['positions'],mapped['log_det_J']
            # Gate every configuration before its first target energy call.
            geometry=[check_geometry(BoundFrame(mapper,int(f)),s['positions'],fixed,bonds,ligand_idx) for f,s in zip(indices,batch)]
            target=potential(y,cells)
            physical=(target/mapper._kT).mean(); jacobian=-logj.mean(); loss=physical+jacobian
            for value,label in ((target,'QM/MM target'),(logj,'Cartesian logdet'),(loss,'physical training objective')):
                check_finite(value,label)
            if step==1:
                forces=torch.autograd.grad(target[0],y,retain_graph=True)[0][0]
                check_finite(forces,'complete Hamiltonian gradient')
                report['force_check']=dict(max_abs_kj_mol_angstrom=float(forces.abs().max()),
                    ligand_gradient_l2=float(torch.linalg.vector_norm(forces.reshape(-1,3)[ligand_idx])),
                    finite_difference=complete_gradient_fd(potential.embedding,
                        y[0].detach().numpy().reshape(-1,3).copy(),cells[0].numpy(),
                        forces.detach().numpy().reshape(-1,3).copy(),ligand_idx))
                record(path,report)
            eg=torch.autograd.grad(physical,params,retain_graph=True,allow_unused=True)
            jg=torch.autograd.grad(jacobian,params,retain_graph=True,allow_unused=True)
            loss.backward()
            grads=[p.grad for p in params]
            for grad in grads:
                if grad is not None: check_finite(grad,'flow parameter gradient')
            closure=max(float((g-(e if e is not None else torch.zeros_like(g))
                -(j if j is not None else torch.zeros_like(g))).abs().max())
                for g,e,j in zip(grads,eg,jg) if g is not None)
            if grad_norm(eg)<=0 or closure>1e-7:
                raise AssertionError('Physical gradient or physical/Jacobian decomposition invalid')
            entry=dict(step=step,frame_indices=indices.tolist(),objective=float(loss.detach()),
                physical_gradient_l2=grad_norm(eg),jacobian_gradient_l2=grad_norm(jg),
                total_gradient_l2=grad_norm(grads),gradient_decomposition_max_abs=closure,
                learning_rate=float(optimizer.param_groups[0]['lr']),batch_geometry=geometry)
            # Preserve all pre-update energies/J values like the authors' online log.
            with torch.no_grad():
                reference=potential.reference(x,cells)
                work=(target-reference)/mapper._kT-logj
            check_finite(reference,'MM reference'); check_finite(work,'online reduced work')
            for i,f in enumerate(indices):
                report['online_rows'].append(dict(step=step,frame_index=int(f),
                    reference_mm_kj_mol=float(reference[i]),mapped_target_qmmm_kj_mol=float(target[i].detach()),
                    log_jacobian=float(logj[i].detach()),reduced_map_work=float(work[i].detach()),kT=float(mapper._kT)))
            torch.nn.utils.clip_grad_norm_(params,max_norm=3000.0,error_if_nonfinite=True)
            entry['clipped_gradient_l2']=grad_norm([p.grad for p in params])
            optimizer.step(); scheduler.step()
            for p in params: check_finite(p,'updated flow weights')
            report['physical_updates']=step
            report['steps'].append(entry)
            save_weights(mapper,optimizer,scheduler,OUT/f'update-{step}.pt',step,metadata)
            record(path,report)
            # Real target energetics are not used to tune parameters from validation.
            row=dict(step=step,rows=[])
            report['geometry_probes'].append(row); record(path,report)
            probe(mapper,u,ligand_idx,fixed,bonds,include_derivative=(step==3),
                  rows=row['rows'],persist=lambda:record(path,report))
            print('B25_AUTHOR_PHYSICAL_UPDATE',json.dumps(entry,allow_nan=False),flush=True)
        change=float(sum(float((p.detach()-start).square().sum()) for p,start in zip(params,initial))**0.5)
        if change<=0: raise AssertionError('Physical optimization did not change parameters')
        report['parameter_change_l2']=change
        report['checkpoint_reload']=checkpoint_reload(mapper,potential,u,ligand_idx,preprocessing,OUT,OUT/'trained-map.pt')
        report.update(stage='complete',physical_training_gate_passed=True,
            limitation='96 samples / 3 updates establishes a bounded integration gate, not convergence, unbiased free energy or paper reproduction. No held-out test used.')
        record(path,report)
        print('B25_AUTHOR_PHYSICAL_TRAINING_GATE_COMPLETE',flush=True)
    except Exception as exc:
        report.update(stage='incomplete',physical_training_gate_passed=False,error=str(exc),traceback=traceback.format_exc())
        record(path,report)
        if mapper is not None:
            torch.save(dict(state_dict=mapper._flow.state_dict(),physical_updates=report['physical_updates']),OUT/'failure-map.pt')
        raise


if __name__=='__main__':
    main()
