#!/usr/bin/env python3
"""Stage A: train upstream TFEP v0.2 MAF on a known-answer Gaussian target.

Reduced potentials and work are dimensionless (kBT units). No QM/MM is used.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tfep.nn.conditioners.made import generate_degrees
from tfep.nn.flows.maf import MAF
from tfep.nn.flows.sequential import SequentialFlow
from tfep.loss import BoltzmannKLDivLoss

from corr_fep.diagnostics import summarize_work

UPSTREAM_TFEP_SHA = "0bc369a1194c688bb6ac997cc6b7c2db2b063c9e"


class GaussianTarget:
    def __init__(self):
        self.mean = torch.tensor([2.0, -1.5], dtype=torch.float64)
        self.precision = torch.tensor(
            [[2.5, 0.7], [0.7, 0.9]], dtype=torch.float64
        )
        self.offset = 1.25
        self.evaluations = 0
        sign, logdet = torch.linalg.slogdet(self.precision)
        assert sign.item() > 0
        # Both distributions use u = -ln(unnormalized density);
        # the Gaussian integral gives Z_B/Z_A = exp(-offset)/sqrt(det K).
        self.exact_df = self.offset + 0.5 * logdet.item()

    def __call__(self, y):
        self.evaluations += len(y)
        d = y - self.mean
        return 0.5 * torch.sum((d @ self.precision) * d, dim=-1) + self.offset


def reference_potential(x):
    return 0.5 * (x * x).sum(dim=-1)


def make_flow(layers=2, width=32):
    # Same upstream MAF + alternating MADE ordering as CartesianMAFMap.
    return SequentialFlow(*[
        MAF(
            degrees_in=generate_degrees(
                n_features=2, order="ascending" if i % 2 == 0 else "descending"
            ),
            hidden_layers=[width, width],
            weight_norm=True,
            initialize_identity=True,
        )
        for i in range(layers)
    ]).double()


def calculate_work(flow, target, x):
    y, log_j = flow(x)
    work = target(y) - reference_potential(x) - log_j
    return work, log_j


def summarize(flow, target, positions, step, split, grad_norm, elapsed):
    with torch.no_grad():
        w, log_j = calculate_work(flow, target, positions)
    work = w.cpu().numpy()
    jac = log_j.cpu().numpy()
    row = {
        "step": step, "split": split, "wall_s": elapsed,
        "grad_norm": grad_norm, "exact_df_reduced": target.exact_df,
        **summarize_work(work),
        "work_var_reduced": float(np.var(work, ddof=1)),
        "df_abs_error_reduced": abs(summarize_work(work)["delta_f_reduced"] - target.exact_df),
        "jac_mean": float(jac.mean()),
        "jac_std": float(jac.std(ddof=1)),
        "jac_min": float(jac.min()),
        "jac_max": float(jac.max()),
        "jac_p01": float(np.quantile(jac, 0.01)),
        "jac_p99": float(np.quantile(jac, 0.99)),
    }
    return row, work, jac


def verify_mapping(flow):
    # Independent numerical check of the density-change convention at one point.
    x = torch.tensor([[0.19, -0.41]], dtype=torch.float64)
    with torch.no_grad():
        y, log_j = flow(x)
        reconstructed, log_j_inv = flow.inverse(y)
    inverse_error = float((x - reconstructed).abs().max())
    log_j_inverse_error = float((log_j + log_j_inv).abs().max())

    def f(v):
        return flow(v.unsqueeze(0))[0].squeeze(0)

    jac_matrix = torch.autograd.functional.jacobian(f, x.squeeze(0))
    sign, numerical_log_j = torch.linalg.slogdet(jac_matrix)
    jac_error = float((numerical_log_j - log_j.squeeze()).abs())
    assert sign.abs().item() == 1
    assert inverse_error < 1e-7, f"Inverse error {inverse_error}"
    assert log_j_inverse_error < 1e-7, f"Inverse Jacobian error {log_j_inverse_error}"
    assert jac_error < 1e-6, f"Jacobian error {jac_error}"
    return {
        "inverse_error": inverse_error,
        "inverse_logdet_error": log_j_inverse_error,
        "autograd_logdet_error": jac_error,
    }


def verify_exact_estimator(target, x):
    """A known-answer test independent of the *trained* model.

    A linear Gaussian transport makes generalized work exactly constant;
    failure therefore indicates a potential, determinant, or estimator bug.
    """
    cholesky = torch.linalg.cholesky(target.precision)
    linear_map = torch.linalg.inv(cholesky.T)
    sign, log_j = torch.linalg.slogdet(linear_map)
    assert sign.item() > 0
    mapped = target.mean + x @ linear_map.T
    work = target(mapped) - reference_potential(x) - log_j
    work_np = work.detach().cpu().numpy()
    estimate = summarize_work(work_np)["delta_f_reduced"]
    largest_error = float(np.max(np.abs(work_np - target.exact_df)))
    assert abs(estimate - target.exact_df) < 1e-10, (
        f"Known-answer free-energy estimator mismatch: {estimate} != {target.exact_df}"
    )
    assert largest_error < 1e-10, f"Exact map has nonconstant work: {largest_error}"
    return {
        "estimated_df_reduced": estimate,
        "exact_df_reduced": target.exact_df,
        "max_work_error_reduced": largest_error,
    }


def save_plots(rows, first_work, last_work, final_jac, folder):
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]

    def curve(filename, keys, ylabel):
        fig, ax = plt.subplots(figsize=(6.3, 4))
        for name, data in [("train", train), ("validation", validation)]:
            ax.plot([r["step"] for r in data], [r[keys] for r in data], label=name)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=140)
        plt.close(fig)

    curve("loss.png", "mean_work_reduced", "Mean generalized work (kBT)")
    curve("ess.png", "importance_ess_fraction", "Raw importance ESS / N")
    curve("jacobian.png", "jac_std", "Std(log |det J|)")

    fig, ax = plt.subplots(figsize=(6.3, 4))
    ax.hist(first_work, bins=55, density=True, alpha=0.55, label="Untrained (step 0)")
    ax.hist(last_work, bins=55, density=True, alpha=0.55, label="Trained (final)")
    ax.set_xlabel("Held-out generalized work (kBT)")
    ax.set_ylabel("Probability density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "work_distribution.png", dpi=140)
    plt.close(fig)

    np.savez_compressed(folder / "validation_work.npz",
                        work_initial=first_work, work_final=last_work,
                        jacobian_final=final_jac)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/stage-a/baseline"))
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-validation", type=int, default=2048)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.n_train, args.n_validation, args.eval_every) <= 0:
        p.error("All sizes/steps must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    g = torch.Generator(device="cpu").manual_seed(args.seed + 91)
    train_x = torch.randn(args.n_train, 2, dtype=torch.float64, generator=g)
    val_x = torch.randn(args.n_validation, 2, dtype=torch.float64, generator=g)
    eval_train_x = train_x[:min(args.n_validation, args.n_train)]
    target = GaussianTarget()
    flow = make_flow(args.layers, args.width)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
    loss_fn = BoltzmannKLDivLoss()

    config = vars(args).copy()
    config["out"] = str(args.out)
    config.update({
        "target": "shifted correlated Gaussian",
        "target_mean": target.mean.tolist(),
        "target_precision": target.precision.tolist(),
        "target_energy_offset": target.offset,
        "architecture": "tfep.nn.flows.MAF (affine), alternating MADE degrees",
        "objective": "tfep.loss.BoltzmannKLDivLoss (target u - logJ)",
        "tfep_commit": UPSTREAM_TFEP_SHA,
        "repo_commit": os.environ.get("GITHUB_SHA", "local"),
        "units": "reduced kBT",
    })
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    env = "\n".join([
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"Torch: {torch.__version__}",
        f"NumPy: {np.__version__}",
        f"TFEP source commit: {UPSTREAM_TFEP_SHA}",
        subprocess.run([sys.executable, "-m", "pip", "freeze"],
                       text=True, capture_output=True, check=True).stdout,
    ])
    (args.out / "environment.txt").write_text(env)

    rows = []
    batch_losses = []
    gradient_norm = None  # No gradient exists before the first optimizer step.
    start = time.perf_counter()
    first_val_work = None
    final_val_work = None
    final_val_jac = None
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            for label, x in [("train", eval_train_x), ("validation", val_x)]:
                row, work, log_j = summarize(
                    flow, target, x, step, label, gradient_norm, time.perf_counter() - start
                )
                rows.append(row)
                if label == "validation":
                    if first_val_work is None:
                        first_val_work = work
                    final_val_work, final_val_jac = work, log_j
            v = rows[-1]
            print(f"step={step} train_mean_w={rows[-2]['mean_work_reduced']:.4f} "
                  f"val_ESS={v['importance_ess']:.1f}/{v['n']} "
                  f"val_dF={v['delta_f_reduced']:.4f} "
                  f"exact_dF={target.exact_df:.4f}", flush=True)
        if step == args.steps:
            break
        idx = torch.randint(len(train_x), (args.batch_size,), generator=g)
        batch_x = train_x[idx]
        optimizer.zero_grad()
        y, log_j = flow(batch_x)
        loss = loss_fn(target_potentials=target(y), log_det_J=log_j)
        assert torch.isfinite(loss).item(), f"Nonfinite training loss at step {step}"
        loss.backward()
        gradient_norm = math.sqrt(sum(
            float(torch.sum(p.grad.detach() ** 2))
            for p in flow.parameters() if p.grad is not None
        ))
        assert math.isfinite(gradient_norm)
        optimizer.step()
        scheduler.step()
        batch_losses.append(float(loss.detach()))

    checks = verify_mapping(flow)
    estimator_check = verify_exact_estimator(target, val_x[:32])
    # Do not impose a training-convergence threshold on the fitted MAF:
    # its held-out error is a reported *result*, not a software-pass condition.
    final_val = next(r for r in reversed(rows) if r["split"] == "validation")
    with (args.out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "batch_losses.csv").write_text(
        "step,training_loss_reduced\n" +
        "".join(f"{i + 1},{v:.12g}\n" for i, v in enumerate(batch_losses))
    )
    save_plots(rows, first_val_work, final_val_work, final_val_jac, args.out)
    elapsed = time.perf_counter() - start
    result = {
        "initial_validation": next(r for r in rows if r["split"] == "validation"),
        "final_validation": final_val,
        "exact_df_reduced": target.exact_df,
        "jacobian_and_inverse_checks": checks,
        "exact_estimator_check": estimator_check,
        "target_potential_evaluations": target.evaluations,
        "elapsed_s": elapsed,
        "model_parameters": sum(p.numel() for p in flow.parameters()),
        "warning": "Raw ESS only; finite held-out source sampling cannot rule out missing target modes.",
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.out / "timing.json").write_text(json.dumps({
        "elapsed_s": elapsed, "optimization_steps": args.steps,
        "target_potential_evaluations": target.evaluations,
        "mean_wall_s_per_step_including_evaluations": elapsed / args.steps,
    }, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
