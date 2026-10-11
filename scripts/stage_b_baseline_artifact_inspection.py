#!/usr/bin/env python3
"""Read-only inspection of the completed 650/250 baseline Actions artifact.

Works on the already-run model's saved output, not new QM/MM simulations.
Emits a compact JSON report and a one-line machine-readable log summary.
"""
import gc
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np


def weight_stats(rows, key, kt):
    w = np.array([r[key] for r in rows], dtype=np.float64)
    lw = -w
    weights = np.exp(lw - lw.max())
    p = weights / weights.sum()
    ids = np.argsort(-p)
    shift = float(lw.max())
    def estimate_without(idx):
        remaining = np.delete(w, idx)
        z = -remaining
        zmax = z.max()
        return float(-kt*(zmax+np.log(np.exp(z-zmax).mean())))
    full = float(-kt*(shift+np.log(weights.mean())))
    dominant = int(ids[0])
    return dict(n=len(w), std=float(w.std(ddof=1)), ess=float(1/(p*p).sum()),
                fraction_in_top_one=float(p[ids[0]]),
                fraction_in_top_three=float(p[ids[:3]].sum()),
                fraction_in_top_ten=float(p[ids[:10]].sum()),
                maximum_weight_frame=int(rows[dominant]['frame_index']),
                max_weight_reduced_work=float(w[dominant]),
                df_kj_mol=full,
                df_without_largest_weight_kj_mol=estimate_without(dominant),
                df_shift_without_largest_kj_mol=float(estimate_without(dominant)-full),
                top_five=[dict(frame=int(rows[i]['frame_index']),
                               reduced_work=float(w[i]),normalized_weight=float(p[i]))
                          for i in ids[:5]],
                halves=[dict(block=j, count=len(sel), std=float(sel.std(ddof=1)),
                             ess=float((lambda x:1/(x*x).sum())(
                                 np.exp(-sel+sel.min())/np.exp(-sel+sel.min()).sum())))
                        for j,sel in enumerate([w[:len(w)//2],w[len(w)//2:]],1)])


def main(zip_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        json_names = [x for x in names if x.endswith('full-baseline.json')]
        if len(json_names)!=1:
            raise AssertionError('Exactly one expected baseline JSON, saw '+str(json_names))
        data = json.loads(z.read(json_names[0]))
        files = [dict(filename=Path(x.filename).name, bytes=int(x.file_size))
                 for x in z.infolist() if x.filename.endswith('.pt')]
    rows = data['validation']['rows']
    if (data['stage']!='complete' or not data['validation']['completed']
            or data['physical_updates']!=21 or len(rows)!=250
            or data['heldout_test_evaluations']!=0):
        raise AssertionError('Incomplete baseline output or touched test set')
    kt = 0.00831446261815324 * 298.0
    x = np.array([r['identity_reduced_work'] for r in rows])
    y = np.array([r['mapped_reduced_work'] for r in rows])
    checkpoints = {f['filename']:f['bytes'] for f in files
                   if f['filename'].startswith('update-')}
    expected = {'update-5.pt','update-10.pt','update-15.pt','update-20.pt','update-21.pt'}
    if set(checkpoints) != expected:
        raise AssertionError('Incomplete checkpoint set: '+str(checkpoints))
    report = dict(
        baseline_git_sha=data['git_sha'], train_samples=data['train_frames'],
        validation_samples=len(rows), checkpoint_files=checkpoints,
        identity=weight_stats(rows,'identity_reduced_work',kt),
        mapped=weight_stats(rows,'mapped_reduced_work',kt),
        reduced_work_correlation=float(np.corrcoef(x,y)[0,1]),
        paired_reduced_work_change=dict(mean=float(np.mean(y-x)),std=float(np.std(y-x,ddof=1))),
        training_updates=int(data['physical_updates']),
        final_gradient_decomposition=data.get('initial_gradient_decomposition'),
        last_validation_frame=int(rows[-1]['frame_index']),
        note='Inspection of saved output only. Intermediate checkpoints were not evaluated on validation and cannot establish optimal epoch. ESS ignores time correlations.'
    )
    (out/'inspection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('B25_BASELINE_INSPECTION',json.dumps(report,allow_nan=False),flush=True)
    # Confirm saved checkpoint PyTorch archives have the required binary entries.
    # Full torch.load intentionally avoided to keep the inspection lightweight.
    with zipfile.ZipFile(zip_path) as outer:
        import io
        validations = {}
        for path in outer.namelist():
            if Path(path).name in expected:
                with zipfile.ZipFile(io.BytesIO(outer.read(path))) as inner:
                    members = inner.namelist()
                    validations[Path(path).name] = dict(
                        valid_zip=True,
                        has_pickle=any(n.endswith('/data.pkl') for n in members),
                        tensor_storage_parts=sum('/data/' in n for n in members),
                        size_bytes=outer.getinfo(path).file_size)
        if set(validations)!=expected or not all(v['has_pickle'] and v['tensor_storage_parts']>0 for v in validations.values()):
            raise AssertionError('PyTorch checkpoint structure invalid')
    (out/'checkpoint-manifest.json').write_text(json.dumps(validations,indent=2)+'\n')
    print('B25_BASELINE_CHECKPOINT_MANIFEST',json.dumps(validations),flush=True)


if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])
