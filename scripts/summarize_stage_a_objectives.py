#!/usr/bin/env python3
"""Paired final-only independent-test comparison of the Stage A3 objectives."""
import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "importance_ess", "importance_ess_fraction", "delta_f_reduced", "df_error",
    "mean_work_reduced", "work_var", "mapped_left_halfspace_fraction",
    "mapped_left_mode_fraction", "mapped_right_mode_fraction",
    "mapped_outside_mode_fraction", "jac_mean", "jac_std", "jac_min", "jac_max",
    "jac_p01", "jac_p99", "grad_norm",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    baseline = {s["seed"]: s for s in json.loads((root / "baseline/summary.json").read_text())}
    modified = {s["seed"]: s for s in json.loads((root / "hellinger/summary.json").read_text())}
    if baseline.keys() != modified.keys():
        raise ValueError("Unpaired random seeds")
    rows = []
    for seed in sorted(baseline):
        a, b = baseline[seed], modified[seed]
        for key in ("initial_parameters_sha256", "training_sha256", "validation_sha256",
                    "test_sha256", "parameters", "optimizer_steps", "target_evaluations"):
            if a[key] != b[key]:
                raise ValueError(f"Seed {seed}: unequal {key}")
        if a["objective"] != "baseline" or b["objective"] != "hellinger":
            raise ValueError("Unexpected objectives")
        for name, s in [("baseline", a), ("hellinger", b)]:
            t = s["final_test"]
            if t is None or t["n"] != 8192:
                raise ValueError("Independent final test missing or incorrect size")
            rows.append({
                "seed": seed, "objective": name,
                "abs_df_error": abs(t["df_error"]),
                "target_evaluations": s["target_evaluations"],
                "optimizer_steps": s["optimizer_steps"],
                "training_s": s["training_s"], "elapsed_s": s["elapsed_s"],
                **{key: t[key] for key in FIELDS}
            })

    with (root / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# Stage A3 independent-test paired comparison",
        "",
        "All numbers are on independent 8,192-draw test sets, evaluated once after",
        "350 training steps. Validation sets were recorded throughout training.",
        "The baseline/modified pairing was validated by SHA-256 hashes of initial",
        "parameters and all source datasets, and identical target-evaluation counts.",
        "",
        "| Seed | Objective | Test ESS/8192 | Δf | |Δf error| | Left basin | Right basin | Outside | Target evaluations | Train time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['objective']} | {r['importance_ess']:.1f} | "
            f"{r['delta_f_reduced']:.4f} | {r['abs_df_error']:.4f} | "
            f"{r['mapped_left_mode_fraction']:.3f} | "
            f"{r['mapped_right_mode_fraction']:.3f} | "
            f"{r['mapped_outside_mode_fraction']:.3f} | "
            f"{r['target_evaluations']} | {r['training_s']:.2f} |"
        )
    lines += [
        "",
        "Reference: exact Δf = 1.96334988787746, left and right basin ≈0.432 each,",
        "outside ≈0.135. ESS alone does not demonstrate both modes are covered.",
        "See each seed's metrics.csv, batch_losses.csv, mapped_positions.png,",
        "mode_coverage.png, validation_work.npz, and test_work.npz.",
        "",
    ]
    (root / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
