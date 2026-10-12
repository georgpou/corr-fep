#!/usr/bin/env python3
"""Full train/validation baseline using the recovered authors' six-layer MixedMAF.

One epoch: 650 chronological training frames, evaluated on 250 distinct validation
frames. The untouched 300-frame test set is NEVER loaded/evaluated in this stage.
This is a QM/MM baseline diagnostic, not proof of an unbiased TFEP estimator or
a paper-reproduction of multi-map statistics.
"""
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from stage_b_author_baseline_setup import prepare, sample, reference_gate
from stage_b_author_baseline_utils import batch_geometry, save_weights
from stage_b_author_baseline_utils import SEED, TRAIN, VALIDATION, TEST, record
from stage_b_first_flow_training import check_finite, grad_norm

OUT = Path('results/stage-b/author-full-baseline').resolve()
BATCH = 32
EPOCHS = 1
CHECKPOINT_EVERY = 5
VALIDATION_BATCH = 16


def weighted_summary(rows, key, kt):
    w = np.asarray([r[key] for r in rows], dtype=np.float64)
    if w.size == 0 or not np.isfinite(w).all():
        raise FloatingPointError('No finite validation works for ' + key)
    logw = -w
    shift = float(logw.max())
    weights = np.exp(logw - shift)
    probs = weights / weights.sum()
    return dict(n=int(w.size), mean_reduced_work=float(w.mean()),
                reduced_work_std=float(w.std(ddof=1)) if len(w)>1 else None,
                weight_ess=float(1.0/np.square(probs).sum()),
                delta_f_kj_mol=float(-kt*(shift+np.log(weights.mean()))),
                min_reduced_work=float(w.min()), max_reduced_work=float(w.max()))


def evaluate_validation(mapper, potential, u, ligand_idx, fixed, bonds, report, dest):
    """Frozen final checkpoint: paired identity/mapped work on every validation frame."""
    rows = []
    report['validation'] = dict(completed=False, rows=rows, frame_indices=VALIDATION.tolist(),
        checkpoint_selection='final update of epoch 1; never choose weights using validation')
    record(dest, report)
    before = [p.detach().clone() for p in mapper._flow.parameters()]
    with torch.no_grad():
        for offset in range(0, len(VALIDATION), VALIDATION_BATCH):
            indices = VALIDATION[offset:offset+VALIDATION_BATCH]
            frames = [sample(u, int(f)) for f in indices]
            x = torch.stack([s['positions'] for s in frames])
            cells = torch.stack([s['cell'] for s in frames])
            mapped = mapper(dict(positions=x,
                                 trajectory_sample_index=torch.as_tensor(indices,dtype=torch.long)))
            y, logj = mapped['positions'], mapped['log_det_J']
            check_finite(y, 'validation mapped positions')
            check_finite(logj, 'validation Cartesian logdet')
            geometry = batch_geometry(mapper,x,y,logj,indices,fixed,bonds,ligand_idx)
            reference = potential.reference(x,cells)
            identity = potential(x,cells)
            target = potential(y,cells)
            iw = (identity-reference)/mapper._kT
            mw = (target-reference)/mapper._kT-logj
            for label, tensor in [('MM reference validation',reference),
                                  ('identity QM/MM validation',identity),
                                  ('mapped QM/MM validation',target),
                                  ('identity reduced work',iw),
                                  ('mapped reduced work',mw)]:
                check_finite(tensor,label)
            for i,f in enumerate(indices):
                rows.append(dict(frame_index=int(f),geometry=geometry[i],
                    reference_mm_kj_mol=float(reference[i]),
                    identity_target_qmmm_kj_mol=float(identity[i]),
                    mapped_target_qmmm_kj_mol=float(target[i]),
                    log_jacobian=float(logj[i]),
                    identity_reduced_work=float(iw[i]),
                    mapped_reduced_work=float(mw[i])))
            record(dest,report)
            print('B25_AUTHOR_BASELINE_VAL_PROGRESS',len(rows),len(VALIDATION),flush=True)
    if any(not torch.equal(p.detach(),original) for p,original in zip(mapper._flow.parameters(),before)):
        raise AssertionError('Validation changed frozen model weights')
    if [r['frame_index'] for r in rows] != VALIDATION.tolist():
        raise AssertionError('Incomplete or misordered validation')
    report['validation'].update(identity=weighted_summary(rows,'identity_reduced_work',mapper._kT),
                                mapped=weighted_summary(rows,'mapped_reduced_work',mapper._kT),
                                completed=True, parameters_unchanged=True,
                                note='Chronologically correlated frames. ESS and work estimates are diagnostics, not independent error bars or a binding ΔΔG.')
    record(dest, report)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dest = OUT/'full-baseline.json'
    if len(TRAIN)!=650 or len(VALIDATION)!=250 or len(TEST)!=300:
        raise AssertionError('Unexpected dataset split sizes')
    if len(set(TRAIN)&set(VALIDATION)) or len(set(TRAIN)&set(TEST)) or len(set(VALIDATION)&set(TEST)):
        raise AssertionError('Dataset split leakage')
    report=dict(stage='initializing',git_sha=os.environ.get('GITHUB_SHA','local'),
        author_method='original ligand-aligned matched-water reference; six-layer MixedMAF',
        physical_updates=0,train_frames=len(TRAIN),validation_frames=len(VALIDATION),
        heldout_test_evaluations=0,train_indices=TRAIN.tolist(),validation_indices=VALIDATION.tolist(),
        optimizer=dict(name='AdamW',learning_rate=1e-4,weight_decay=0.01,
                       exponential_lr_gamma=0.997,gradient_clip_norm=3000.0,
                       epochs=EPOCHS,batch_size=BATCH,maf_layers=6),
        steps=[],validation=None,benchmark_validated=False,
        limitation='One epoch of a single archived repeat; correlated validation and frozen per-frame ligand-matched solvent permutations. Mathematical global estimator invertibility remains unresolved.')
    record(dest,report)
    mapper=optimizer=scheduler=None
    try:
        mapper,potential,u,ligand_idx,fixed,bonds,preprocessing,metadata=prepare(OUT)
        report.update(metadata)
        reference_geometry = reference_gate(u,preprocessing,np.concatenate([TRAIN,VALIDATION]))
        report['reference_geometry']=reference_geometry
        record(dest,report)
        if reference_geometry['collinear_frames']:
            raise AssertionError('Authors ligand-aligned reference is collinear')
        params=[p for p in mapper._flow.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.01)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.997)
        generator=torch.Generator().manual_seed(SEED)
        order=TRAIN[torch.randperm(len(TRAIN),generator=generator).numpy()]
        report['training_order']=order.tolist()
        report['stage']='training'
        record(dest,report)
        for step,offset in enumerate(range(0,len(order),BATCH),1):
            indices=order[offset:offset+BATCH]
            frames=[sample(u,int(f)) for f in indices]
            x=torch.stack([s['positions'] for s in frames])
            cells=torch.stack([s['cell'] for s in frames])
            optimizer.zero_grad(set_to_none=True)
            mapped=mapper(dict(positions=x,
                               trajectory_sample_index=torch.as_tensor(indices,dtype=torch.long)))
            y,logj=mapped['positions'],mapped['log_det_J']
            check_finite(y,'training mapped coordinates')
            check_finite(logj,'training Cartesian logdet')
            geo=batch_geometry(mapper,x,y,logj,indices,fixed,bonds,ligand_idx)
            target=potential(y,cells)
            physical=(target/mapper._kT).mean()
            jacobian=-logj.mean()
            loss=physical+jacobian
            check_finite(target,'training physical QM/MM energy')
            check_finite(loss,'training objective')
            if step==1:
                eg=torch.autograd.grad(physical,params,retain_graph=True,allow_unused=True)
                jg=torch.autograd.grad(jacobian,params,retain_graph=True,allow_unused=True)
            loss.backward()
            grads=[p.grad for p in params]
            for g in grads:
                if g is not None: check_finite(g,'training parameter gradient')
            if step==1:
                closure=max(float((g-(e if e is not None else torch.zeros_like(g))
                                  -(j if j is not None else torch.zeros_like(g))).abs().max())
                            for g,e,j in zip(grads,eg,jg) if g is not None)
                report['initial_gradient_decomposition']=dict(
                    physical_gradient_l2=grad_norm(eg),jacobian_gradient_l2=grad_norm(jg),
                    max_abs_closure=closure)
                if grad_norm(eg)<=0 or closure>1e-7:
                    raise AssertionError('Physical and Jacobian gradient mismatch')
            unclipped=grad_norm(grads)
            torch.nn.utils.clip_grad_norm_(params,max_norm=3000.0,error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            for p in params: check_finite(p,'updated parameter')
            entry=dict(step=step,training_frame_indices=indices.tolist(),n=len(indices),
                objective=float(loss.detach()),physical_term=float(physical.detach()),
                jacobian_term=float(jacobian.detach()),gradient_l2=unclipped,
                learning_rate=float(optimizer.param_groups[0]['lr']),
                max_inverse_error_angstrom=max(r['inverse_max_abs_error_angstrom'] for r in geo),
                max_logdet_closure=max(r['inverse_logdet_closure'] for r in geo),
                max_environment_error_angstrom=max(r['fixed_environment_error_angstrom'] for r in geo))
            report['steps'].append(entry)
            report['physical_updates']=step
            record(dest,report)
            print('B25_AUTHOR_BASELINE_UPDATE',json.dumps(entry,allow_nan=False),flush=True)
            if step%CHECKPOINT_EVERY==0 or offset+BATCH>=len(order):
                save_weights(mapper,optimizer,scheduler,OUT/f'update-{step}.pt',step,metadata)
        if sorted(int(x) for row in report['steps'] for x in row['training_frame_indices']) != TRAIN.tolist():
            raise AssertionError('Not all 650 training frames were consumed exactly once')
        report['stage']='validation'
        record(dest,report)
        evaluate_validation(mapper,potential,u,ligand_idx,fixed,bonds,report,dest)
        report.update(stage='complete',baseline_run_complete=True,
                      benchmark_validated=bool(report['validation']['completed']),
                      heldout_test_evaluations=0)
        record(dest,report)
        print('B25_AUTHOR_FULL_BASELINE_COMPLETE',json.dumps({
            'updates':report['physical_updates'],
            'validation_identity':report['validation']['identity'],
            'validation_mapped':report['validation']['mapped']},allow_nan=False),flush=True)
    except Exception as exc:
        report.update(stage='incomplete',error=str(exc),traceback=traceback.format_exc(),
                      baseline_run_complete=False,benchmark_validated=False)
        record(dest,report)
        if mapper is not None:
            torch.save(dict(state_dict=mapper._flow.state_dict(),
                            physical_updates=report['physical_updates']),
                       OUT/'interrupted-map.pt')
        raise


if __name__=='__main__':
    main()
