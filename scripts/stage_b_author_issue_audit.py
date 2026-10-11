#!/usr/bin/env python3
"""Execute narrowly scoped released-code bookkeeping checks on real topology.

Zero energy evaluations or optimizer steps. Temporary TRR contains reencoded
XTC coordinates; it does not recover original TRR precision.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import traceback

import MDAnalysis as mda
import MDAnalysis.analysis.align
import numpy as np
import torch

from stage_b_qm_singlepoint import load_archive, TPR_NAME
from stage_b_mechanical_embedding import ROOT, prepare_files, retrieve_topology, digest
from stage_b_multiframe import load_trajectory
from stage_b_extended_cloud_benchmark import SEED, record

OUT = Path('results/stage-b/author-issue-audit').resolve()
FRAMES = [401, 726, 1050]


def load_driver():
    path = Path('upstream-stage-b/run_tfep.py').resolve()
    spec = importlib.util.spec_from_file_location('pinned_paper_driver', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(stage='initializing', diagnostic_only=True, benchmark_validated=False,
                  qm_energy_evaluations=0, physical_updates=0, frames=FRAMES,
                  git_sha=os.environ.get('GITHUB_SHA', 'local'))
    destination = OUT/'issue-audit.json'
    record(destination, report)
    try:
        raw = load_archive()
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT/'archived-eq.tpr').write_bytes(raw[TPR_NAME])
        retrieve_topology()
        prepare_files()
        xtc, info = load_trajectory()
        report['trajectory'] = info
        driver = load_driver()
        report['upstream_driver_sha256'] = digest(Path('upstream-stage-b/run_tfep.py').read_bytes())
        work = OUT/'three-frame-input'
        work.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'archived-eq.tpr', work/'eq.tpr')
        u = mda.Universe(str(work/'eq.tpr'), str(xtc))
        with mda.Writer(str(work/'traj.trr'), n_atoms=len(u.atoms)) as writer:
            for frame in FRAMES:
                u.trajectory[frame]
                writer.write(u.atoms)
        ref = mda.Universe(str(work/'eq.tpr'), str(work/'traj.trr'))
        driver.fix_universe_elements(ref)
        scaffold = '(name C or name C1 or name C2 or name C3 or name C4 or name C5 or name C6 or name C7) and resname MOL'
        ref.trajectory[0]
        env = ref.select_atoms('same resid as (resname SOL and element O and around 3.5 resname MOL)').indices
        if len(env) < 9:
            raise AssertionError('Need at least three nearby waters for actual-topology audit')
        refs, matched, rmsd, legacy_distances = driver.match_waters(
            str(work), ref, 0, scaffold, 'resname SOL', 3.5, env)
        oxygen = refs[ref.atoms[refs].elements == 'O']
        hydrogen = refs[ref.atoms[refs].elements == 'H']
        unsorted = np.concatenate([oxygen, hydrogen])
        sorter = np.argsort(unsorted)
        legacy_row = matched[0, np.argsort(sorter)]
        correct_row = matched[0]
        mismatches = np.flatnonzero(ref.atoms[refs].elements != ref.atoms[legacy_row].elements)
        report['U1'] = dict(status='actual_topology_trigger_reproduced',
            n_reference_atoms=len(refs), n_reference_waters=len(oxygen),
            wrong_element_slots=len(mismatches),
            examples=[dict(reference_index=int(refs[i]), intended_element=str(ref.atoms[refs[i]].element),
                legacy_source_index=int(legacy_row[i]), legacy_source_element=str(ref.atoms[legacy_row[i]].element),
                correct_source_index=int(correct_row[i])) for i in mismatches[:10]],
            returned_table_in_topology_order=True,
            legacy_environment_distance_shape=list(legacy_distances.shape),
            precision_note='Three original XTC snapshots reencoded as TRR; precision not improved')
        record(destination, report)
        if len(mismatches) == 0:
            raise AssertionError('Expected U1 trigger absent: inspect actual order rather than patching blindly')

        # Released wrapper, arithmetic displacement only: not an energy model.
        class ReferenceDisplacement(torch.nn.Module):
            def forward(self, x):
                y = x.clone()
                y[:, torch.as_tensor((refs[:,None]*3+np.arange(3)).reshape(-1))] += 0.001
                return y, x.new_zeros(len(x))
        ref.trajectory[0]
        x = torch.tensor(ref.atoms.positions.copy().reshape(1,-1), dtype=torch.float64)
        original = driver.PermutationFlow(ReferenceDisplacement(), refs, refs[None,:], True)
        y, _ = original(x, torch.tensor([0]))
        flat = torch.as_tensor((refs[:,None]*3+np.arange(3)).reshape(-1))
        observed = float((y[:,flat]-x[:,flat]).abs().max())
        report['U3'] = dict(status='released_wrapper_identity_assignment_reproduced',
            proposed_reference_displacement_angstrom=0.001,
            actual_output_displacement_angstrom=observed,
            mapped_water_output_overwritten=(observed == 0.0),
            relevance='Solvent mapping only; recovery retains ligand-only mapping')
        if observed != 0.0:
            raise AssertionError('U3 differs from predicted overwrite; reassess source')
        # U2 is already a loop-bound reproduction; actual bulk-water candidates
        # exceed ten, so it is not the current dataset failure.
        report['U2'] = dict(status='conditional_loop_bound_issue',
            actual_heavy_candidates=len(oxygen), current_data_trigger=(len(oxygen)<10),
            planned_call_bound=min(10,len(oxygen)))
        report['stage'] = 'complete'
        record(destination, report)
        print('B25_AUTHOR_ISSUE_AUDIT', json.dumps(report,allow_nan=False), flush=True)
    except Exception as exc:
        report.update(stage='incomplete',error=str(exc),traceback=traceback.format_exc())
        record(destination,report)
        raise


if __name__ == '__main__':
    main()
