#!/usr/bin/env python3
"""Compare two completed physical Corr-FEP variance benchmark artifacts.

Usage: python scripts/stage_b_variance_compare.py CONTROL/report.json VARIANCE/report.json
Reports are written only after verifying equal optimizer exposure, map
initialization, identical frozen validation frames, and untouched reserved test.
"""
import argparse
import json

import numpy as np


def summarize(values, kt):
    work = np.asarray(values, dtype=np.float64)
    if not work.size or not np.isfinite(work).all():
        raise AssertionError("Generalized-work array is empty or nonfinite")
    a = -work
    m = float(a.max())
    shifted = np.exp(a - m)
    p = shifted / shifted.sum()
    return {
        "count": int(work.size),
        "work_mean": float(work.mean()),
        "work_standard_deviation": float(work.std(ddof=1)),
        "work_quantiles_0_10_25_50_75_90_100": np.quantile(
            work, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist(),
        "weight_ess": float(1.0 / np.square(p).sum()),
        "largest_normalized_weight": float(p.max()),
        "largest_five_normalized_weight_sum": float(
            np.sort(p)[-min(5, len(p)):].sum()),
        "delta_f_kj_mol": float(-kt * (m + np.log(shifted.mean()))),
    }


def check_compatible(a, b):
    for key in ("train_indices", "validation_indices",
                "initial_parameter_sha256", "training_order_by_epoch"):
        if a[key] != b[key]:
            raise AssertionError("Not matched on " + key)
    for key in ("completed_updates", "reserved_test_count",
                "reserved_test_evaluations"):
        if a[key] != b[key]:
            raise AssertionError("Different counts on " + key)
    if not a["complete"] or not b["complete"] or a["stage"] != b["stage"] or a["stage"] != "complete":
        raise AssertionError("One or both runs did not complete")
    if a["variant"] != "control" or b["variant"] != "variance":
        raise AssertionError("Expected control then variance experiment")
    if a["variance_weight"] != 0 or b["variance_weight"] <= 0:
        raise AssertionError("Incorrect objective comparison")
    for key in ("train_qmmm", "train_full_mm_reference",
                "validation_qmmm_identity", "validation_qmmm_mapped",
                "validation_full_mm_reference"):
        if a["physical_evaluations"][key] != b["physical_evaluations"][key]:
            raise AssertionError("Different physical calculation counts: " + key)
    if len(a["training_steps"]) != len(b["training_steps"]):
        raise AssertionError("Different numbers of optimizer steps")
    for x,y in zip(a["training_steps"],b["training_steps"]):
        for key in ("epoch", "frame_indices", "n"):
            if x[key] != y[key]:
                raise AssertionError("Training batch mismatch: " + key)
    for report in (a,b):
        val=report["validation"]
        if not val["completed"] or not val["parameters_unchanged"]:
            raise AssertionError("Missing frozen validation")
        if [row["frame_index"] for row in val["rows"]] != report["validation_indices"]:
            raise AssertionError("Invalid validation frame order")
        if report["reserved_test_evaluations"] != 0:
            raise AssertionError("Reserved-test frames were used")
    ia=np.array([r["identity_reduced_work"] for r in a["validation"]["rows"]])
    ib=np.array([r["identity_reduced_work"] for r in b["validation"]["rows"]])
    if not np.allclose(ia,ib,rtol=0,atol=1e-5):
        raise AssertionError("Identity-map works differ between runs")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("control")
    p.add_argument("variance")
    args=p.parse_args()
    with open(args.control) as f: a=json.load(f)
    with open(args.variance) as f: b=json.load(f)
    check_compatible(a,b)
    kt=0.00831446261815324*298.0
    identity=np.array([r["identity_reduced_work"] for r in a["validation"]["rows"]])
    wa=np.array([r["mapped_reduced_work"] for r in a["validation"]["rows"]])
    wb=np.array([r["mapped_reduced_work"] for r in b["validation"]["rows"]])
    control=summarize(wa,kt)
    variance=summarize(wb,kt)
    result={
        "status":"completed matched physical diagnostics; not converged free energies",
        "training_updates_each":a["completed_updates"],
        "physical_evaluations_each":a["physical_evaluations"],
        "initial_parameter_sha256":a["initial_parameter_sha256"],
        "variance_weight":b["variance_weight"],
        "identity":summarize(identity,kt),
        "control":control,
        "variance":variance,
        "variance_minus_control_work_per_frame_quantiles":
            np.quantile(wb-wa,[0,0.1,0.25,0.5,0.75,0.9,1]).tolist(),
        "variance_over_control_work_std":
            variance["work_standard_deviation"]/control["work_standard_deviation"],
        "variance_over_control_weight_ess":
            variance["weight_ess"]/control["weight_ess"],
        "variance_minus_control_largest_weight_share":
            variance["largest_normalized_weight"]-control["largest_normalized_weight"],
        "training_runtime_seconds":{
            "control":a["runtime_seconds"],"variance":b["runtime_seconds"]},
        "limitations":[
            "One temporally correlated simulation repeat, not independent MD sampling",
            "250-frame exponential reweighting ESS remains a finite-sample diagnostic",
            "No reserved-test model selection or full multimap estimator",
            "Frame-wise solvent matching invertibility remains unresolved (issue #17)"],
    }
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
