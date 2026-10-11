#!/usr/bin/env python3
"""No-QM candidate gate. Runs using the author coordinate workflow environment.

The candidate is evaluated on actual train/validation snapshots; it is not a
replacement or a new training run. No parameter updates or energy evaluations.
"""
import json
from pathlib import Path
import numpy as np
import torch

from stage_b_author_coordinate_probe import prepare, sample, reference_gate
from stage_b_author_map import BoundFrame
from stage_b_environment_only_map import EnvironmentOnlyMixedMAF, environmental_sites
from stage_b_extended_cloud_benchmark import TRAIN, VALIDATION, SEED, check_geometry, record
from stage_b_fixed_reference_probe import derivative

OUT = Path('results/stage-b/environment-only-probe').resolve()
FRAMES = [401, 726, 1050, 1151, 1275, 1400]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report = dict(stage='initializing', diagnostic_only=True, qm_energy_evaluations=0,
                  physical_updates=0, frames=FRAMES, rows=[])
    dest = OUT/'environment-only-probe.json'
    record(dest,report)
    legacy,potential,u,ligand_idx,fixed,bonds,preprocessing,metadata=prepare(OUT)
    candidate_context = environmental_sites(u, preprocessing,
        metadata['preprocessing']['reference_training_frame'])
    candidate_context['references'] = preprocessing['references']
    torch.manual_seed(SEED)
    candidate = EnvironmentOnlyMixedMAF(potential,legacy.xtc,ligand_idx,
                                        candidate_context,OUT/'candidate')
    candidate.setup('fit')
    for frame in FRAMES:
        s = sample(u,frame)
        x = s['positions'][None,:]
        cell = s['cell'][None,:]
        batch = dict(positions=x,cell=cell)
        with torch.no_grad():
            mapped = candidate(batch)
            reversed_map = candidate.inverse(dict(positions=mapped['positions'],cell=cell))
            component = candidate._flow
            a = component.assignments(x,cell[:,:3])
            b = component.assignments(mapped['positions'],cell[:,:3])
        row = dict(frame=frame, assignment_identical=bool(torch.equal(a,b)),
                   physical_oxygen_indices=a[0].tolist(),
                   inverse_error=float((reversed_map['positions']-x).abs().max()),
                   logdet_closure=float((reversed_map['log_det_J']+mapped['log_det_J']).abs().max()),
                   environment_error=float((mapped['positions']-x).reshape(1,-1,3)[:,fixed].abs().max()))
        report['rows'].append(row)
        record(dest,report)
        if not row['assignment_identical'] or max(row['inverse_error'],row['logdet_closure'])>1e-5 or row['environment_error']>1e-10:
            raise AssertionError('Candidate environmental inverse/assignment failed: '+str(row))
    report.update(stage='complete', candidate_gate_passed=True,
                  limitation='Six selected archived configurations, random initial map only; no QM or trained-weight/estimator proof.')
    record(dest,report)


if __name__ == '__main__':
    main()
