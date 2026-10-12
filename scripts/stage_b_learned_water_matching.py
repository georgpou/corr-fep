#!/usr/bin/env python3
"""Frozen trained-flow water-matching probe: no energies or training."""
import json, traceback
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np
import pint, torch
from stage_b_author_baseline_utils import VALIDATION, SEED, load_driver
from stage_b_author_map import AuthorMixedMAF, adapted_match, memory_subset, SCAFFOLD, CUTOFF
from stage_b_mechanical_embedding import ROOT
from stage_b_multiframe import load_trajectory, register_trajectory_transformations
from stage_b_qm_singlepoint import load_archive, TPR_NAME

OUT=Path("results/stage-b/learned-water-matching")
SAVED=Path("saved-baseline")
FRAMES=np.unique(np.linspace(VALIDATION[0],VALIDATION[-1],16,dtype=int))
class UnitsOnly(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.registry=pint.UnitRegistry()
    @property
    def energy_unit(self): return self.registry.kilojoule/self.registry.mole
    @property
    def positions_unit(self): return self.registry.angstrom
    def forward(self,*a,**kw): raise RuntimeError("QM/MM not allowed")

def main():
    OUT.mkdir(parents=True,exist_ok=True); dest=OUT/"results.json"
    report=dict(status="started",frames=FRAMES.tolist(),qmmm_calls=0,training_updates=0)
    try:
        torch.set_default_dtype(torch.float64); torch.set_num_threads(1); torch.manual_seed(SEED)
        ROOT.mkdir(parents=True,exist_ok=True)
        (ROOT/"archived-eq.tpr").write_bytes(load_archive()[TPR_NAME])
        xtc,provenance=load_trajectory(); driver=load_driver()
        baseline=json.loads((SAVED/"full-baseline.json").read_text())
        assert baseline["stage"]=="complete" and baseline["physical_updates"]==21
        data=np.load(SAVED/"author-preprocess.npz")
        prep={k:data[k].copy() for k in ("ref_water","matched_water","references","conditioning_environment")}
        assert np.array_equal(data["validation_indices"],VALIDATION)
        ref_frame=int(baseline["preprocessing"]["reference_training_frame"])
        topology=ROOT/"archived-eq.tpr"
        u=mda.Universe(str(topology),str(xtc)); ligand_idx=u.select_atoms("resname MOL").indices.copy()
        register_trajectory_transformations(u,ligand_idx)
        # Transform exactly as baseline.
        snapshots=[]
        for f in FRAMES:
            u.trajectory[int(f)]; snapshots.append(u.atoms.positions.copy())
        x=torch.tensor(np.stack(snapshots).reshape(len(FRAMES),-1),dtype=torch.float64)
        ids=torch.as_tensor(FRAMES,dtype=torch.long)
        mapper=AuthorMixedMAF(UnitsOnly(),xtc,ligand_idx,prep,OUT)
        mapper.setup("fit")
        with torch.no_grad(): mapper(dict(positions=x,trajectory_sample_index=ids))
        checkpoint=torch.load(SAVED/"update-21.pt",map_location="cpu",weights_only=True,mmap=True)
        assert checkpoint["step"]==21 and checkpoint["seed"]==SEED
        mapper._flow.load_state_dict(checkpoint["state_dict"],strict=True)
        with torch.no_grad():
            result=mapper(dict(positions=x,trajectory_sample_index=ids))
            inverse=mapper.inverse(dict(positions=result["positions"],trajectory_sample_index=ids))
        y=result["positions"].numpy().reshape(len(FRAMES),-1,3)
        original=x.numpy().reshape(len(FRAMES),-1,3)
        inverse_error=float((inverse["positions"]-x).abs().max())
        env=np.setdiff1d(np.arange(len(u.atoms)),ligand_idx)
        env_error=float(np.max(np.abs(y[:,env]-original[:,env])))
        assert inverse_error<1e-5 and env_error<1e-10
        displacement=np.linalg.norm(y[:,ligand_idx]-original[:,ligand_idx],axis=2)
        # Match source and learned mapped configurations using same pinned
        # upstream matcher and same reference and box. Original source frames
        # are re-read from XTC to avoid double-applied transformations.
        raw=mda.Universe(str(topology),str(xtc))
        src=[]; cells=[]
        for f in FRAMES:
            raw.trajectory[int(f)]; src.append(raw.atoms.positions.copy()); cells.append(raw.dimensions.copy())
        pairs=np.concatenate([np.array(src),y.astype(np.float32)])
        cellpairs=np.concatenate([np.array(cells),np.array(cells)])
        trial=mda.Universe(str(topology))
        trial.load_new(pairs,format=MemoryReader,dimensions=cellpairs)
        driver.fix_universe_elements(trial)
        reference=memory_subset(topology,xtc,[ref_frame]); driver.fix_universe_elements(reference)
        matcher=adapted_match(driver,OUT)
        slots,assigned,rmsds,_=matcher("",reference,0,SCAFFOLD,"resname SOL",CUTOFF,
                np.array([int(ligand_idx[0])]),matched_universe=trial)
        assert np.array_equal(slots,prep["ref_water"])
        oxy=np.array([i for i,a in enumerate(slots) if reference.atoms[int(a)].element=="O"])
        hyd=np.array([i for i,a in enumerate(slots) if reference.atoms[int(a)].element=="H"])
        ref_slots=[int(np.where(slots==a)[0][0]) for a in prep["references"]]
        per_frame=[]
        for k,f in enumerate(FRAMES):
            source=assigned[k]; mapped=assigned[k+len(FRAMES)]
            cached=prep["matched_water"][int(f)]
            per_frame.append(dict(frame=int(f),
                original_vs_cached_oxygen_changes=int(np.count_nonzero(source[oxy]!=cached[oxy])),
                original_vs_mapped_oxygen_changes=int(np.count_nonzero(source[oxy]!=mapped[oxy])),
                original_vs_mapped_hydrogen_changes=int(np.count_nonzero(source[hyd]!=mapped[hyd])),
                baseline_reference_changes=int(np.count_nonzero(source[ref_slots]!=mapped[ref_slots])),
                max_ligand_displacement_A=float(displacement[k].max()),
                rms_ligand_displacement_A=float(np.sqrt(np.mean(displacement[k]**2)))))
        report.update(status="complete",source_run=38180943734,source_reference_frame=ref_frame,
            source_trajectory=provenance,inverse_max_error_A=inverse_error,
            env_error_A=env_error,frame_rows=per_frame,
            changed_oxygen_frames=sum(z["original_vs_mapped_oxygen_changes"]>0 for z in per_frame),
            changed_reference_frames=sum(z["baseline_reference_changes"]>0 for z in per_frame),
            source_preprocess_agree_frames=sum(z["original_vs_cached_oxygen_changes"]==0 for z in per_frame),
            max_ligand_displacement_A=float(displacement.max()),
            limitation="Assignment changes are not evidence of nonbijectivity or free-energy bias.")
        print("LEARNED_MATCHING_COMPLETE",json.dumps({k:report[k] for k in
            ("changed_oxygen_frames","changed_reference_frames","source_preprocess_agree_frames","max_ligand_displacement_A")}),flush=True)
    except Exception as exc:
        report.update(status="failed",error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        dest.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
if __name__=="__main__": main()
