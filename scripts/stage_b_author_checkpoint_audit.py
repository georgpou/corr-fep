#!/usr/bin/env python3
"""Re-evaluate 5 archived author-model checkpoints on fixed validation frames.

Scientific purpose: verify save/reload and that parameter updates produce valid
physical mapped work. This is a small diagnostic, not checkpoint selection for
the full validation set or a replacement for the authors' multimap estimator.
"""
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from stage_b_author_coordinate_probe import prepare, sample
from stage_b_author_training_pilot import batch_geometry
from stage_b_extended_cloud_benchmark import VALIDATION, TRAIN, record
from stage_b_first_flow_training import check_finite

OUT=Path('results/stage-b/checkpoint-audit').resolve()
SAVED=Path('saved-baseline').resolve()
CHECKPOINTS=[5,10,15,20,21]
FRAMES=np.unique(np.linspace(VALIDATION[0],VALIDATION[-1],16,dtype=int))


def describe(rows, field):
    w=np.asarray([r[field] for r in rows],dtype=np.float64)
    v=np.exp(-w+np.min(w))
    p=v/v.sum()
    return dict(n=len(w), work_std=float(w.std(ddof=1)),
                weight_ess=float(1/(p*p).sum()),max_weight_fraction=float(p.max()),
                mean_work=float(w.mean()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'checkpoint-audit.json'
    report=dict(stage='initializing',source_run_id=38173132836,
                frames=FRAMES.tolist(),checkpoints=CHECKPOINTS,
                rows=[],complete=False,heldout_test_evaluations=0,
                limitation='16-frame fixed subset only; not complete 250-frame checkpoint selection or exact paper multimap estimator.')
    record(dest,report)
    try:
        baseline=json.loads((SAVED/'full-baseline.json').read_text())
        expected={int(r['frame_index']):r for r in baseline['validation']['rows']}
        if baseline.get('stage')!='complete' or len(expected)!=250:
            raise AssertionError('Source full baseline missing 250 validation frames')
        original=np.load(SAVED/'author-preprocess.npz')
        mapper,potential,u,ligand_idx,fixed,bonds,prep,metadata=prepare(OUT)
        for key in ['ref_water','matched_water','references','conditioning_environment']:
            actual=np.asarray(prep[key])
            cached=np.asarray(original[key])
            if not np.array_equal(actual,cached):
                raise AssertionError('Checkpoint preprocessing does not exactly replay: '+key)
        report['original_preprocessing_exact_replay']=True
        record(dest,report)
        snapshots=[sample(u,int(f)) for f in FRAMES]
        x=torch.stack([s['positions'] for s in snapshots])
        cells=torch.stack([s['cell'] for s in snapshots])
        batch=dict(positions=x,trajectory_sample_index=torch.as_tensor(FRAMES,dtype=torch.long))
        # Materialize lazy flow propagated-indices buffers before strict load.
        with torch.no_grad():
            mapper(batch)
        for step in CHECKPOINTS:
            path=SAVED/f'update-{step}.pt'
            saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            if int(saved['step'])!=step or int(saved['seed'])!=int(baseline['seed'] if 'seed' in baseline else 2025):
                # The recorded baseline report may not include the seed; compare
                # against the experiment's actual saved seed when available.
                if int(saved['step'])!=step:
                    raise AssertionError('Checkpoint step metadata mismatch')
            missing,extra=mapper._flow.load_state_dict(saved['state_dict'],strict=True)
            if missing or extra:
                raise AssertionError('Non-strict checkpoint reconstruction')
            del saved
            rows=[]
            with torch.no_grad():
                result=mapper(batch)
                y,logj=result['positions'],result['log_det_J']
                check_finite(y,'reloaded checkpoint mapped positions')
                check_finite(logj,'reloaded checkpoint Jacobian')
                geometry=batch_geometry(mapper,x,y,logj,FRAMES,fixed,bonds,ligand_idx)
                target=potential(y,cells)
                reference=torch.as_tensor([expected[int(f)]['reference_mm_kj_mol'] for f in FRAMES],dtype=torch.float64)
                reduced=(target-reference)/mapper._kT-logj
                check_finite(target,'checkpoint target QM/MM')
                check_finite(reduced,'checkpoint reduced mapped work')
                for i,f in enumerate(FRAMES):
                    rr=expected[int(f)]
                    row=dict(frame=int(f),mapped_reduced_work=float(reduced[i]),
                        mapped_qmmm_kj_mol=float(target[i]),
                        log_jacobian=float(logj[i]),
                        geometry=geometry[i])
                    if step==21:
                        row['previous_run_work_error']=abs(float(reduced[i])-float(rr['mapped_reduced_work']))
                        row['previous_run_target_error_kj_mol']=abs(float(target[i])-float(rr['mapped_target_qmmm_kj_mol']))
                    rows.append(row)
            outcome=dict(checkpoint=step,analysis=describe(rows,'mapped_reduced_work'),
                rows=rows)
            if step==21:
                outcome['final_checkpoint_max_work_replay_error']=max(r['previous_run_work_error'] for r in rows)
                outcome['final_checkpoint_max_energy_replay_error_kj_mol']=max(r['previous_run_target_error_kj_mol'] for r in rows)
                # XTC, matching and QM/MM must reproduce actual saved full baseline numerically.
                if outcome['final_checkpoint_max_work_replay_error']>0.05 or outcome['final_checkpoint_max_energy_replay_error_kj_mol']>0.12:
                    raise AssertionError('Final checkpoint does not replay original validation work')
            report['rows'].append(outcome)
            record(dest,report)
            print('B25_AUTHOR_CHECKPOINT_VALIDATED',json.dumps({k:v for k,v in outcome.items() if k!='rows'}),flush=True)
        report.update(stage='complete',complete=True)
        record(dest,report)
        print('B25_AUTHOR_CHECKPOINT_AUDIT_COMPLETE',flush=True)
    except Exception as e:
        report.update(stage='incomplete',complete=False,error=str(e),traceback=traceback.format_exc())
        record(dest,report)
        raise


if __name__=='__main__':
    main()
