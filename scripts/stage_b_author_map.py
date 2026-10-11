#!/usr/bin/env python3
"""Thin author-method adapter: matched-water conditioning, original MixedMAF.

The authors' driver is imported from the pinned checkout. The match helper is
loaded with two explicit source adaptations: accepting a supplied Universe
for clean partitions/XTC, and consistent oxygen-then-hydrogen slot ordering
for reference-stability statistics (U1). Original source/license stays in
upstream-stage-b; substantially reused source is also archived with the run.
"""
import inspect
from pathlib import Path
import textwrap

import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np
import torch
import tfep.app.mixedmaf
import tfep.io
from tfep.utils.misc import atom_to_flattened_indices

from stage_b_author_issue_audit import load_driver
from stage_b_extended_cloud_benchmark import CALIBRATION, TRAIN, VALIDATION, SEED
from stage_b_training_interface import TEMPERATURE_K
from stage_b_mechanical_embedding import ROOT
from stage_b_multiframe import register_trajectory_transformations

SCAFFOLD = '(name C or name C1 or name C2 or name C3 or name C4 or name C5 or name C6 or name C7) and resname MOL'
CUTOFF = 3.5


def adapted_match(driver, output_dir):
    """Keep the actual upstream matching algorithm; local patch is auditable."""
    source = textwrap.dedent(inspect.getsource(driver.match_waters))
    source = source.replace('        environment_atom_indices,\n',
                            '        environment_atom_indices,\n        matched_universe=None,\n',1)
    old = """    matched_universe = MDAnalysis.Universe(
        os.path.join(repeat_dir_path, 'eq.tpr'),
        os.path.join(repeat_dir_path, 'traj.trr')
    )"""
    new = """    if matched_universe is None:
        matched_universe = MDAnalysis.Universe(
            os.path.join(repeat_dir_path, 'eq.tpr'),
            os.path.join(repeat_dir_path, 'traj.trr')
        )"""
    if old not in source:
        raise AssertionError('Pinned match helper load-site changed')
    source = source.replace(old,new,1)
    old_order = "exchanged_positions[ref_atom_groups['water'].ix] = ts.positions[matched_water_atom_indices[frame_idx]]"
    new_order = """reference_slots = np.concatenate([ref_atom_groups['oxygen'].ix, ref_atom_groups['hydrogen'].ix])
        exchanged_positions[reference_slots] = ts.positions[matched_water_atom_indices[frame_idx]]"""
    if old_order not in source:
        raise AssertionError('Pinned U1 assignment changed')
    source = source.replace(old_order,new_order,1)
    # Original notice accompanies the archived substantial helper copy.
    license_text = Path('upstream-stage-b/LICENSE').read_text()
    (output_dir/'adapted-match-waters.py').write_text(''.join('# '+line+'\n' for line in license_text.splitlines())+source)
    namespace = dict(driver.__dict__)
    exec(compile(source,'<pinned match_waters: U1 and supplied-Universe adapter>','exec'),namespace)
    return namespace['match_waters']


def memory_subset(topology, xtc, indices):
    """Raw selected frames only, before the authors' helper transformations."""
    u = mda.Universe(str(topology),str(xtc))
    coordinates, cells = [], []
    for frame in indices:
        u.trajectory[int(frame)]
        coordinates.append(u.atoms.positions.copy())
        cells.append(u.dimensions.copy())
    u.load_new(np.asarray(coordinates),format=MemoryReader,dimensions=np.asarray(cells))
    return u


def preprocess(xtc, output_dir):
    """Fit all representation choices on train; only apply them on validation."""
    driver = load_driver()
    topology = ROOT/'archived-eq.tpr'
    match = adapted_match(driver,output_dir)
    train = memory_subset(topology,xtc,TRAIN)
    driver.fix_universe_elements(train)
    (mapped_other,conditioning_other,mapped_water,conditioning_water,ref_local) = driver.find_ligand_neighbors(
        train, n_repeats=1,step=1,stability_threshold=0.5,
        mapped_distance_cutoff=0.0,conditioning_distance_cutoff=CUTOFF,
        ligand_selection='resname MOL',water_selection='resname SOL')
    if len(mapped_water) or len(mapped_other):
        raise AssertionError('Ligand-only option unexpectedly selects mapped water')
    conditioning_environment=np.concatenate([conditioning_other,conditioning_water])
    if len(conditioning_water)<9:
        raise AssertionError('Insufficient conditioning water heavy atoms')
    ref_water, train_matched, rmsd, distances = match(
        '',train.copy(),ref_local,SCAFFOLD,'resname SOL',CUTOFF,
        conditioning_environment,matched_universe=train.copy())
    fluctuations = distances.std(axis=0).mean(axis=-1)
    n_heavy = sum(train.atoms[int(i)].element!='H' for i in conditioning_environment)
    origin, axes = driver.select_reference_frame_atoms(
        train.copy(),conditioning_environment,fluctuations,'resname MOL','resname SOL',
        CUTOFF,ref_water,train_matched,n_max_atoms=min(10,n_heavy))
    references = [int(origin),*[int(i) for i in axes]]
    if any(train.atoms[i].element=='H' for i in references):
        raise AssertionError('Author reference selection must use heavy atoms')
    print('B25_AUTHOR_REFERENCE_SELECTION',references,flush=True)
    validation = memory_subset(topology,xtc,VALIDATION)
    driver.fix_universe_elements(validation)
    val_water,val_matched,val_rmsd,_ = match(
        '',train.copy(),ref_local,SCAFFOLD,'resname SOL',CUTOFF,
        conditioning_environment,matched_universe=validation)
    if not np.array_equal(val_water,ref_water):
        raise AssertionError('Validation changed reference water slots')
    n_frames = len(mda.Universe(str(topology),str(xtc)).trajectory)
    table = np.full((n_frames,len(ref_water)),-1,dtype=np.int64)
    table[TRAIN] = train_matched
    table[VALIDATION] = val_matched
    np.savez_compressed(output_dir/'author-preprocess.npz',
        ref_water=ref_water,matched_water=table,references=references,
        fit_indices=TRAIN,validation_indices=VALIDATION,
        conditioning_water=conditioning_water,conditioning_environment=conditioning_environment,fluctuations=fluctuations)
    summary = dict(fit_indices=TRAIN.tolist(),apply_only_indices=VALIDATION.tolist(),
        reference_training_frame=int(TRAIN[ref_local]),reference_atom_slots=references,
        reference_elements=[str(train.atoms[i].element) for i in references],
        n_reference_waters=len(ref_water)//3,conditioning_cutoff_angstrom=CUTOFF,
        mapped_cutoff_angstrom=0.0,only_reference_conditioning=True,
        calibration_indices=CALIBRATION.tolist(),
        training_matching_rmsd_max=float(rmsd.max()),validation_matching_rmsd_max=float(val_rmsd.max()),
        local_source_adaptations=['U1: consistent O-then-H reference slots for stability statistics',
            'supplied Universe for chronological subsets/XTC',
            'U2: call selector with actual candidate count bounded by ten'],
        trajectory_precision='Original quantized XTC; no reconstructed TRR precision claim')
    return dict(ref_water=ref_water,matched_water=table,references=references,
                conditioning_environment=conditioning_environment,ref_local=ref_local), summary


class ConditioningContextFlow(torch.nn.Module):
    """Author gathered-water context with exactly restored physical environment.

    Deliberately supports ligand-only mapping, not mapped solvent. This avoids
    copying the original mapped-water restoration bug (U3). The cached context
    assignment is identical in forward and inverse; no reselection is performed.
    """
    def __init__(self,flow,preprocessing,environment,return_partial=False):
        super().__init__()
        self.flow=flow
        self.return_partial=return_partial
        self.register_buffer('ref_dofs',atom_to_flattened_indices(torch.as_tensor(preprocessing['ref_water'])))
        self.register_buffer('matched_dofs',atom_to_flattened_indices(torch.as_tensor(preprocessing['matched_water'])))
        self.register_buffer('environment_dofs',atom_to_flattened_indices(torch.as_tensor(environment)))

    def _pass(self,x,indices,inverse):
        assignments=self.matched_dofs[indices]
        if bool((assignments<0).any()):
            raise AssertionError('Frame has no frozen author water assignment')
        context=x.clone()
        context[:,self.ref_dofs]=torch.gather(x,dim=1,index=assignments)
        result=self.flow.inverse(context) if inverse else self.flow(context)
        if self.return_partial:
            return result
        y=result[0].clone()
        y[:,self.environment_dofs]=x[:,self.environment_dofs]
        return (y,*result[1:])

    def forward(self,x,trajectory_sample_index):
        return self._pass(x,trajectory_sample_index,False)

    def inverse(self,y,trajectory_sample_index):
        return self._pass(y,trajectory_sample_index,True)


class AuthorMixedMAF(tfep.app.mixedmaf.MixedMAFMap):
    def __init__(self,potential,xtc,ligand_idx,preprocessing,output_dir,n_layers=6):
        self.preprocessing=preprocessing
        self.xtc=xtc
        self.ligand_idx=ligand_idx
        registry=potential.energy_unit._REGISTRY
        super().__init__(potential_energy_func=potential,topology_file_path=str(ROOT/'archived-eq.tpr'),
            coordinates_file_path=str(xtc),temperature=TEMPERATURE_K*registry.kelvin,
            mapped_atoms=ligand_idx.tolist(),conditioning_atoms=sorted(preprocessing['references']),
            origin_atom=preprocessing['references'][0],axes_atoms=preprocessing['references'][1:],
            n_maf_layers=n_layers,batch_size=32,tfep_logger_dir_path=str(output_dir/'tfep-logs'))

    def create_universe(self):
        u=mda.Universe(str(ROOT/'archived-eq.tpr'),str(self.xtc))
        register_trajectory_transformations(u,self.ligand_idx)
        load_driver().fix_universe_elements(u)
        return u

    def create_dataset(self):
        return tfep.io.TrajectorySubset(super().create_dataset(),CALIBRATION)

    def create_partial_flow(self,flow,return_partial=False):
        partial=super().create_partial_flow(flow,return_partial)
        environment=np.setdiff1d(np.arange(self.dataset.n_atoms),self.ligand_idx)
        return ConditioningContextFlow(partial,self.preprocessing,environment,return_partial)

    def forward(self,batch):
        y,logj=self._flow(batch['positions'],batch['trajectory_sample_index'])
        return dict(positions=y,log_det_J=logj)

    def inverse(self,batch):
        y,logj=self._flow.inverse(batch['positions'],batch['trajectory_sample_index'])
        return dict(positions=y,log_det_J=logj)


class BoundFrame:
    """Bind cached sample identity for existing geometry/derivative probes."""
    def __init__(self,mapper,frame):
        self.mapper=mapper
        self.frame=int(frame)

    def __call__(self,batch):
        return self.mapper({**batch,'trajectory_sample_index':torch.full((len(batch['positions']),),self.frame,dtype=torch.long)})

    def inverse(self,batch):
        return self.mapper.inverse({**batch,'trajectory_sample_index':torch.full((len(batch['positions']),),self.frame,dtype=torch.long)})
