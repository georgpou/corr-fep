#!/usr/bin/env python3
"""Matched physical test of mean-work and mean-work-plus-variance training.

Uses the validated six-layer author MixedMAF, water matching, GFN1-xTB/MM
potential, and 250-frame frozen validation from stage_b_author_full_baseline.
Set CORR_FEP_EXPANDED_BENCHMARK=1 before importing author helper modules.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

# These helpers resolve their frame partitions at module import.
if os.environ.get("CORR_FEP_EXPANDED_BENCHMARK") != "1":
    raise RuntimeError("Expanded benchmark requires CORR_FEP_EXPANDED_BENCHMARK=1")

from stage_b_author_baseline_setup import prepare, reference_gate, sample
from stage_b_author_baseline_utils import TRAIN, VALIDATION, TEST, SEED
from stage_b_author_baseline_utils import batch_geometry, record, save_weights
from stage_b_author_full_baseline import evaluate_validation
from stage_b_first_flow_training import check_finite, grad_norm

BATCH = 32
EPOCHS = 2
CLIP = 3000.0
CHECKPOINT_EVERY = 10


def digest_parameters(mapper):
    h = hashlib.sha256()
    for name, value in sorted(mapper._flow.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cache_reference(potential, universe, indices, out, report):
    """Evaluate full-MM U_FF(x) once per distinct immutable training frame.

    Full-MM evaluations do not enter the PyTorch gradient graph because x is
    not trainable. Target QM/MM and the map/Jacobian remain differentiable.
    """
    energies = np.full(2401, np.nan, dtype=np.float64)
    for offset in range(0, len(indices), BATCH):
        frames = indices[offset:offset+BATCH]
        batch = [sample(universe, int(i)) for i in frames]
        x = torch.stack([s["positions"] for s in batch])
        cells = torch.stack([s["cell"] for s in batch])
        with torch.no_grad():
            values = potential.reference(x, cells)
        check_finite(values, "cached FF reference")
        energies[frames] = values.detach().cpu().numpy()
        report["physical_evaluations"]["train_full_mm_reference"] += len(frames)
        if (offset // BATCH) % 5 == 0:
            record(out / "report.json", report)
            print("REFERENCE_CACHE_PROGRESS", min(offset+BATCH,len(indices)),
                  len(indices), flush=True)
    if not np.isfinite(energies[indices]).all():
        raise FloatingPointError("Incomplete training reference cache")
    np.savez_compressed(out / "training-reference.npz",
                        frame_indices=indices, reference_mm_kj_mol=energies[indices])
    return energies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variance-weight", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true",
                        help="One real eight-frame physical QM/MM gradient update")
    args = parser.parse_args()
    if not np.isfinite(args.variance_weight) or args.variance_weight < 0:
        parser.error("variance-weight must be finite and nonnegative")
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if not (len(TRAIN)==1450 and len(VALIDATION)==250 and len(TEST)==300):
        raise AssertionError("Unexpected expanded partition")
    if len(set(TRAIN) & set(VALIDATION)) or len(set(TRAIN) & set(TEST)):
        raise AssertionError("Train/held-out leakage")
    if len(set(VALIDATION) & set(TEST)):
        raise AssertionError("Validation/test leakage")

    tag = "smoke" if args.smoke else ("control" if args.variance_weight==0 else "variance")
    out = (Path("results/stage-b/variance-benchmark") / tag).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    report = dict(stage="setup",variant=tag,variance_weight=args.variance_weight,
                  git_sha=os.environ.get("GITHUB_SHA","local"),
                  train_indices=TRAIN.tolist(),validation_indices=VALIDATION.tolist(),
                  reserved_test_count=len(TEST),reserved_test_evaluations=0,
                  architecture="Six-layer author MixedMAF, ligand only, frozen water context",
                  physical_system="p38a_2g; water; GFN1-xTB/MM subtractive mechanical embedding",
                  objective="mean(U_QMMM(mapped)/kT - logdet) + lambda * population_var((U_QMMM(mapped)-U_FF(x))/kT - logdet)",
                  optimizer=dict(name="AdamW",lr=1e-4,weight_decay=0.01,
                                 scheduler_gamma_per_update=0.997,
                                 clip_gradient_l2=CLIP,epochs=args.epochs,
                                 batch_size=(8 if args.smoke else BATCH)),
                  physical_evaluations=dict(train_qmmm=0,train_full_mm_reference=0,
                                            validation_qmmm_identity=0,
                                            validation_qmmm_mapped=0,
                                            validation_full_mm_reference=0),
                  training_steps=[],completed_updates=0,validation=None)
    dest=out / "report.json"
    record(dest,report)
    mapper=None
    try:
        mapper,potential,u,ligand_idx,fixed,bonds,preprocessing,metadata=prepare(out)
        report.update(metadata)
        report["initial_parameter_sha256"]=digest_parameters(mapper)
        gate=reference_gate(u,preprocessing,np.concatenate((TRAIN,VALIDATION)))
        report["reference_geometry"]=gate
        if gate["collinear_frames"]:
            raise AssertionError("Collinear conditioning references")
        record(dest,report)

        params=[p for p in mapper._flow.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.01)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.997)
        rng=torch.Generator().manual_seed(SEED)
        batches=[]
        for epoch in range(args.epochs):
            order=TRAIN[torch.randperm(len(TRAIN),generator=rng).numpy()]
            if args.smoke:
                order=order[:8]
            batches.append(order)
        report["training_order_by_epoch"]=[order.tolist() for order in batches]
        reference_indices=TRAIN if not args.smoke else np.unique(batches[0])
        report["stage"]="reference_cache"
        record(dest,report)
        references=cache_reference(potential,u,reference_indices,out,report)

        report["stage"]="training"
        record(dest,report)
        update=0
        for epoch,order in enumerate(batches,1):
            for offset in range(0,len(order),(8 if args.smoke else BATCH)):
                tick=time.perf_counter()
                indices=order[offset:offset+(8 if args.smoke else BATCH)]
                samples=[sample(u,int(f)) for f in indices]
                x=torch.stack([s["positions"] for s in samples])
                cells=torch.stack([s["cell"] for s in samples])
                ref=torch.as_tensor(references[indices],dtype=torch.float64)
                optimizer.zero_grad(set_to_none=True)
                mapped=mapper(dict(positions=x,trajectory_sample_index=
                                   torch.as_tensor(indices,dtype=torch.long)))
                y,logj=mapped["positions"],mapped["log_det_J"]
                check_finite(y,"mapped training positions")
                check_finite(logj,"mapped training logdet")
                geometry=batch_geometry(mapper,x,y,logj,indices,fixed,bonds,ligand_idx)
                target=potential(y,cells)
                report["physical_evaluations"]["train_qmmm"] += len(indices)
                check_finite(target,"real QM/MM training target")
                physical=(target/mapper._kT).mean()
                jacobian=-logj.mean()
                base_loss=physical+jacobian  # Identical to historical lambda=0 loss.
                work=(target-ref)/mapper._kT-logj
                variance=work.var(unbiased=False)
                loss=base_loss+args.variance_weight*variance if args.variance_weight else base_loss
                check_finite(work,"training generalized work")
                check_finite(variance,"training work variance")
                check_finite(loss,"total objective")
                if update==0:
                    eg=torch.autograd.grad(physical,params,retain_graph=True,allow_unused=True)
                    jg=torch.autograd.grad(jacobian,params,retain_graph=True,allow_unused=True)
                    vg=(torch.autograd.grad(variance,params,retain_graph=True,
                                            allow_unused=True)
                        if args.variance_weight else None)
                loss.backward()
                gradients=[p.grad for p in params]
                for g in gradients:
                    if g is not None:
                        check_finite(g,"QM/MM-flow parameter gradient")
                if update==0:
                    closure=max(
                        float((g - (e if e is not None else torch.zeros_like(g))
                               - (j if j is not None else torch.zeros_like(g))
                               - (args.variance_weight*v if v is not None
                                  else torch.zeros_like(g))).abs().max())
                        for g,e,j,v in zip(gradients,eg,jg,
                                           vg if vg is not None else [None]*len(params))
                        if g is not None)
                    report["initial_gradient_decomposition"]=dict(
                        physical_gradient_l2=grad_norm(eg),
                        jacobian_gradient_l2=grad_norm(jg),
                        work_variance_gradient_l2=(grad_norm(vg) if vg is not None else None),
                        weighted_variance_gradient_l2=(
                            args.variance_weight*grad_norm(vg) if vg is not None else 0.0),
                        maximum_absolute_closure=closure)
                    if grad_norm(eg)<=0 or (vg is not None and grad_norm(vg)<=0):
                        raise AssertionError("Missing QM/MM or variance gradient")
                    if closure > 1e-6*(1+grad_norm(gradients)):
                        raise AssertionError("Training gradient decomposition mismatch")
                unclipped=grad_norm(gradients)
                torch.nn.utils.clip_grad_norm_(params,CLIP,error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                for p in params:
                    check_finite(p,"updated flow parameter")
                update+=1
                entry=dict(update=update,epoch=epoch,frame_indices=indices.tolist(),
                           n=len(indices),original_loss=float(base_loss.detach()),
                           physical_term=float(physical.detach()),
                           jacobian_term=float(jacobian.detach()),
                           work_variance=float(variance.detach()),
                           work_std_population=float(torch.sqrt(variance.detach())),
                           weighted_variance_penalty=float((args.variance_weight*variance).detach()),
                           total_loss=float(loss.detach()),unclipped_gradient_l2=unclipped,
                           gradient_clipped=unclipped>CLIP,
                           lr=float(optimizer.param_groups[0]["lr"]),
                           duration_seconds=time.perf_counter()-tick,
                           max_inverse_error_angstrom=max(r["inverse_max_abs_error_angstrom"] for r in geometry),
                           max_logdet_closure=max(r["inverse_logdet_closure"] for r in geometry),
                           max_fixed_environment_error_angstrom=max(
                               r["fixed_environment_error_angstrom"] for r in geometry))
                report["training_steps"].append(entry)
                report["completed_updates"]=update
                report["runtime_seconds"]=time.perf_counter()-start
                record(dest,report)
                print("VARIANCE_TRAIN_UPDATE",json.dumps(entry,allow_nan=False),flush=True)
                if update%CHECKPOINT_EVERY==0 or offset+(8 if args.smoke else BATCH)>=len(order):
                    save_weights(mapper,optimizer,scheduler,out / f"update-{update}.pt",update,metadata)
            if args.smoke:
                break
        if not args.smoke:
            expected=args.epochs*len(TRAIN)
            consumed=sum(r["n"] for r in report["training_steps"])
            if consumed!=expected:
                raise AssertionError("Training frame exposure mismatch")
            report["stage"]="validation"
            record(dest,report)
            evaluate_validation(mapper,potential,u,ligand_idx,fixed,bonds,report,dest)
            report["validation"]["checkpoint_selection"]=(
                f"frozen final optimizer update {update}; no validation model selection")
            report["physical_evaluations"]["validation_qmmm_identity"]=len(VALIDATION)
            report["physical_evaluations"]["validation_qmmm_mapped"]=len(VALIDATION)
            report["physical_evaluations"]["validation_full_mm_reference"]=len(VALIDATION)
        report.update(stage="complete",runtime_seconds=time.perf_counter()-start,
                      model_final_parameter_sha256=digest_parameters(mapper),
                      complete=True)
        record(dest,report)
        print("VARIANCE_BENCHMARK_COMPLETE",
              json.dumps(dict(variant=tag,updates=update,
                              validation=report["validation"],
                              seconds=report["runtime_seconds"]),allow_nan=False),
              flush=True)
    except Exception as exc:
        report.update(stage="incomplete",error=str(exc),traceback=traceback.format_exc(),
                      complete=False,runtime_seconds=time.perf_counter()-start)
        record(dest,report)
        if mapper is not None:
            torch.save(dict(state_dict=mapper._flow.state_dict(),
                            completed_updates=report["completed_updates"]),
                       out / "interrupted-map.pt")
        raise


if __name__=="__main__":
    main()
