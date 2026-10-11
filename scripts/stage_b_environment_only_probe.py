#!/usr/bin/env python3
"""No-QM candidate gate. Runs using the author coordinate workflow environment.

The candidate is evaluated on actual train/validation snapshots; it is not a
replacement or a new training run. No parameter updates or energy evaluations.
"""
import json
from pathlib import Path
import numpy as np
import torch

from stage_b_author_coordinate_probe import prepare, sample
from stage_b_environment_only_map import EnvironmentOnlyMixedMAF, environmental_sites
from stage_b_extended_cloud_benchmark import TRAIN, VALIDATION, SEED, record

OUT = Path('results/stage-b/environment-only-probe').resolve()
FRAMES = [401, 726, 1050, 1151, 1275, 1400]



def complete_reference_scan(u, candidate, ligand_idx, frames, report, dest):
    """Scan every train/validation snapshot without QM/MM or flow evaluations.

    Record assignment switches and actual Cartesian reference-triangle geometry.
    Report, rather than hide, image discontinuities / remote-water selection.
    """
    selected = sorted({int(f) for f in frames})
    component = candidate._flow
    template = component.template_sites.detach().cpu().numpy()
    anchor = int(component.anchor_atom)
    selected_oxygen = []
    rows = []
    previous = {}
    per_block_switches = {'train': 0, 'validation': 0}
    per_block_comparisons = {'train': 0, 'validation': 0}
    for frame in selected:
        s = sample(u, frame)
        xyz = s['positions'].reshape(-1, 3)
        box = s['cell'][:3].cpu().numpy()
        xyz_np = xyz.cpu().numpy()
        phys = component.assignments(s['positions'][None, :], s['cell'][None, :3])[0].cpu().numpy()
        points = xyz_np[phys]
        e1, e2 = points[1] - points[0], points[2] - points[0]
        norms = np.linalg.norm([e1, e2], axis=1)
        sine = float(np.linalg.norm(np.cross(e1, e2)) / np.prod(norms)) if np.all(norms > 0) else 0.0
        edges = np.linalg.norm(points[np.array([0,0,1])] - points[np.array([1,2,2])], axis=1)
        relative = points - xyz_np[anchor]
        relative -= box * np.rint(relative / box)
        to_sites = relative - template
        to_sites -= box * np.rint(to_sites / box)
        site_dist = np.linalg.norm(to_sites, axis=1)
        lig = xyz_np[ligand_idx]
        delta = points[:,None,:] - lig[None,:,:]
        delta -= box * np.rint(delta / box)
        nearest_lig = np.min(np.linalg.norm(delta,axis=2),axis=1)
        block = 'train' if frame in TRAIN else 'validation'
        if block in previous:
            per_block_comparisons[block] += 1
            per_block_switches[block] += int(np.any(phys != previous[block]))
        previous[block] = phys.copy()
        row = dict(frame=int(frame), block=block, oxygens=phys.tolist(),
                   reference_sine=sine,
                   edge_max_angstrom=float(edges.max()),
                   edge_min_angstrom=float(edges.min()),
                   reference_crosses_image=bool(np.any(edges > 0.5 * box.min())),
                   largest_site_offset_angstrom=float(site_dist.max()),
                   nearest_ligand_distance_min_angstrom=float(nearest_lig.min()),
                   nearest_ligand_distance_max_angstrom=float(nearest_lig.max()))
        rows.append(row)
    min_sine = min(x['reference_sine'] for x in rows)
    report['full_reference_scan'] = dict(completed=True, frames=len(rows),
        train_frames=sum(x['block']=='train' for x in rows),
        validation_frames=sum(x['block']=='validation' for x in rows),
        minimum_sine=min_sine,
        min_sine_frame=min(rows,key=lambda x:x['reference_sine'])['frame'],
        near_collinear_001=sum(x['reference_sine'] < 0.01 for x in rows),
        near_collinear_005=sum(x['reference_sine'] < 0.05 for x in rows),
        image_spanning_frames=sum(x['reference_crosses_image'] for x in rows),
        maximal_edge_angstrom=max(x['edge_max_angstrom'] for x in rows),
        maximal_site_offset_angstrom=max(x['largest_site_offset_angstrom'] for x in rows),
        maximal_ligand_distance_angstrom=max(x['nearest_ligand_distance_max_angstrom'] for x in rows),
        water_switches=per_block_switches,
        adjacent_comparisons=per_block_comparisons,
        rows=rows)
    record(dest, report)
    print('B25_ENVIRONMENT_FULL_SCAN',json.dumps({k:v for k,v in report['full_reference_scan'].items() if k!='rows'}),flush=True)
    if len(rows)!=len(TRAIN)+len(VALIDATION):
        raise AssertionError('900-frame coverage incomplete')
    if not np.isfinite([x['reference_sine'] for x in rows]).all() or min_sine<0.01:
        raise AssertionError('Ill-conditioned or nonfinite environmental reference triangle')
    return report['full_reference_scan']

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
    report['candidate_context'] = {k:(v.tolist() if isinstance(v,np.ndarray) else v)
                                   for k,v in candidate_context.items() if k!='candidate_oxygens'}
    report['candidate_context']['candidate_oxygen_count'] = len(candidate_context['candidate_oxygens'])
    record(dest,report)
    torch.manual_seed(SEED)
    candidate = EnvironmentOnlyMixedMAF(potential,legacy.xtc,ligand_idx,
                                        candidate_context,OUT/'candidate')
    candidate.setup('fit')
    complete_reference_scan(u,candidate,ligand_idx,np.concatenate([TRAIN,VALIDATION]),report,dest)
    # Compare exactly the same assignment using original and randomly changed ligand positions.
    # This is a zero-QM, direct check that no mapped-coordinate leakage remains.
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
            changed_ligand = x.clone().reshape(1,-1,3)
            changed_ligand[:,ligand_idx] += 3.0
            c = component.assignments(changed_ligand.reshape_as(x),cell[:,:3])
            reference_xyz = x.reshape(1,-1,3)[0,a[0]]
            v1 = reference_xyz[1]-reference_xyz[0]
            v2 = reference_xyz[2]-reference_xyz[0]
            sine = torch.linalg.vector_norm(torch.cross(v1,v2,dim=0))/(torch.linalg.vector_norm(v1)*torch.linalg.vector_norm(v2))
        row = dict(frame=frame, assignment_identical=bool(torch.equal(a,b)),
                   assignment_ligand_independent=bool(torch.equal(a,c)),
                   reference_angle_sine=float(sine),
                   physical_oxygen_indices=a[0].tolist(),
                   inverse_error=float((reversed_map['positions']-x).abs().max()),
                   logdet_closure=float((reversed_map['log_det_J']+mapped['log_det_J']).abs().max()),
                   environment_error=float((mapped['positions']-x).reshape(1,-1,3)[:,fixed].abs().max()))
        report['rows'].append(row)
        record(dest,report)
        if not row['assignment_identical'] or not row['assignment_ligand_independent'] or not np.isfinite(row['reference_angle_sine']) or row['reference_angle_sine']<1e-3 or max(row['inverse_error'],row['logdet_closure'])>1e-5 or row['environment_error']>1e-10:
            raise AssertionError('Candidate environmental inverse/assignment failed: '+str(row))
    report.update(stage='complete', candidate_gate_passed=True,
                  limitation='900 training/validation reference geometries plus six random-initial-map inversion checks; no QM or trained-weight/estimator proof.')
    record(dest,report)


if __name__ == '__main__':
    main()
