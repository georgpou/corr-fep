#!/usr/bin/env python3
"""Stage A2: two-mode target with a known free energy and mode coverage."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tfep.loss import BoltzmannKLDivLoss

from stage_a_gaussian import make_flow, reference_potential
from corr_fep.diagnostics import summarize_work


class TwoModeTarget:
    """Equal-weight Gaussian mixture with analytically calculable partition function."""

    def __init__(self, separation=5.0, sigma=0.7):
        self.centers = torch.tensor([[-separation, 0.0], [separation, 0.0]], dtype=torch.float64)
        self.sigma = sigma
        self.offset = 1.25
        self.evaluations = 0
        # Z_B = exp(-c) * (2*pi*sigma**2), Z_A = 2*pi.
        self.exact_df = self.offset - 2.0 * math.log(sigma)

    def __call__(self, y):
        self.evaluations += len(y)
        sqdist = ((y[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        terms = -0.5 * sqdist / self.sigma**2 - math.log(2.0)
        return -torch.logsumexp(terms, dim=1) + self.offset


@torch.no_grad()
def evaluate(flow, target, x, step, split):
    y, logj = flow(x)
    work = (target(y) - reference_potential(x) - logj).numpy()
    jac = logj.numpy()
    stats = summarize_work(work)
    # The half-space split alone does not establish residence in either mode.
    # Two non-overlapping disks (radius 2*sigma) also count configurations
    # genuinely near each target peak, or outside *both* peak regions.
    left_halfspace_fraction = float((y[:, 0] < 0).double().mean())
    radius = 2.0 * target.sigma
    distance_squared = ((y[:, None, :] - target.centers[None, :, :]) ** 2).sum(-1)
    near_left = distance_squared[:, 0] <= radius**2
    near_right = distance_squared[:, 1] <= radius**2
    if torch.any(near_left & near_right).item():
        raise ValueError("Mode regions overlap: the diagnostic requires separated modes.")
    left_mode_fraction = float(near_left.double().mean())
    right_mode_fraction = float(near_right.double().mean())
    outside_mode_fraction = float((~(near_left | near_right)).double().mean())
    assert abs(left_mode_fraction + right_mode_fraction + outside_mode_fraction - 1) < 1e-12
    result = {
        "step": step, "split": split, **stats,
        "df_error": stats["delta_f_reduced"] - target.exact_df,
        "work_var": float(np.var(work, ddof=1)),
        "jac_mean": float(jac.mean()), "jac_std": float(jac.std(ddof=1)),
        "jac_min": float(jac.min()), "jac_max": float(jac.max()),
        "jac_p01": float(np.quantile(jac, 0.01)),
        "jac_p99": float(np.quantile(jac, 0.99)),
        "mapped_left_halfspace_fraction": left_halfspace_fraction,
        "mapped_halfspace_imbalance": abs(left_halfspace_fraction - 0.5),
        "mapped_left_mode_fraction": left_mode_fraction,
        "mapped_right_mode_fraction": right_mode_fraction,
        "mapped_outside_mode_fraction": outside_mode_fraction,
        "mode_region_radius": radius,
    }
    return result, work, y.numpy()


def single_experiment(args, seed, out):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 27)
    train = torch.randn(args.n_train, 2, generator=generator, dtype=torch.float64)
    validation = torch.randn(args.n_val, 2, generator=generator, dtype=torch.float64)
    # Separate generator: the A2 baseline training batches remain unchanged.
    test = (torch.randn(args.n_test, 2,
                        generator=torch.Generator().manual_seed(seed + 100027),
                        dtype=torch.float64) if args.n_test else None)
    sample_hashes = {
        "training_sha256": hashlib.sha256(train.numpy().tobytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(validation.numpy().tobytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test.numpy().tobytes()).hexdigest() if test is not None else None,
    }
    target = TwoModeTarget(args.separation, args.sigma)
    flow = make_flow(args.layers, args.width)
    initial_parameters = sum(p.numel() for p in flow.parameters())
    initial_parameters_sha256 = hashlib.sha256(b"".join(
        p.detach().numpy().tobytes() for p in flow.parameters()
    )).hexdigest()
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
    objective = BoltzmannKLDivLoss()
    rows, losses = [], []
    last_grad_norm = None
    initial_work, final_work = None, None
    initial_mapped, final_mapped = None, None
    t0 = time.perf_counter()
    training_s = 0.0

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            for name, x in [("train", train), ("validation", validation)]:
                row, work, mapped = evaluate(flow, target, x, step, name)
                row["grad_norm"] = last_grad_norm
                rows.append(row)
                if name == "validation":
                    if initial_work is None:
                        initial_work, initial_mapped = work, mapped
                    final_work, final_mapped = work, mapped
            v = rows[-1]
            print(f"seed={seed} step={step} train_mean_w={rows[-2]['mean_work_reduced']:.4f} "
                  f"val_ess={v['importance_ess']:.1f}/{v['n']} "
                  f"val_df={v['delta_f_reduced']:.4f} target_df={target.exact_df:.4f} "
                  f"left_halfspace={v['mapped_left_halfspace_fraction']:.3f} "
                  f"left_mode={v['mapped_left_mode_fraction']:.3f} "
                  f"right_mode={v['mapped_right_mode_fraction']:.3f} "
                  f"outside={v['mapped_outside_mode_fraction']:.3f}", flush=True)

        if step == args.steps:
            break
        idx = torch.randint(len(train), (args.batch_size,), generator=generator)
        xb = train[idx]
        tick = time.perf_counter()
        optimizer.zero_grad()
        y, logj = flow(xb)
        target_energies = target(y)  # One target evaluation per minibatch for both objectives.
        if args.objective == "baseline":
            loss = objective(target_potentials=target_energies, log_det_J=logj)
        else:
            # Hellinger affinity = const * E_A[exp(-w/2)]. Partition function
            # is unknown but its contribution to -log(affinity) is constant.
            # The source potential MUST be retained in these nonlinear weights.
            w = target_energies - reference_potential(xb) - logj
            loss = -(torch.logsumexp(-args.beta * w, dim=0) - math.log(len(w)))
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite loss at seed={seed} step={step}")
        loss.backward()
        last_grad_norm = math.sqrt(sum(
            float((p.grad.detach() ** 2).sum()) for p in flow.parameters() if p.grad is not None
        ))
        if not math.isfinite(last_grad_norm):
            raise FloatingPointError(f"Nonfinite gradient at seed={seed} step={step}")
        optimizer.step()
        scheduler.step()
        training_s += time.perf_counter() - tick
        losses.append(float(loss.detach()))

    test_row = None
    if test is not None:
        test_row, test_work, mapped_test = evaluate(flow, target, test, args.steps, "test")
        test_row["grad_norm"] = last_grad_norm
        rows.append(test_row)
    out.mkdir(parents=True, exist_ok=True)
    if test is not None:
        np.savez_compressed(out / "test_work.npz",
                            source_test=test.numpy(), mapped_test=mapped_test,
                            work_test=test_work)
    with (out / "metrics.csv").open("w", newline="") as file:
        w = csv.DictWriter(file, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / "batch_losses.csv").open("w", newline="") as file:
        w = csv.writer(file)
        w.writerow(["step", "batch_loss"])
        w.writerows(enumerate(losses, start=1))

    for file, metric, ylabel in [
        ("loss.png", "mean_work_reduced", "Mean generalized work (kBT)"),
        ("ess.png", "importance_ess_fraction", "Raw ESS fraction"),
        ("jacobian.png", "jac_std", "SD of log |det J|"),
        ("halfspace_balance.png", "mapped_left_halfspace_fraction", "Mapped left-halfspace fraction"),
        ("free_energy.png", "df_error", "Estimated Delta f - exact (kBT)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for split in ("train", "validation"):
            r = [v for v in rows if v["split"] == split]
            ax.plot([v["step"] for v in r], [v[metric] for v in r], label=split)
        if file == "halfspace_balance.png":
            ax.axhline(0.5, color="black", linestyle="--", label="Target halfspace fraction")
        elif file == "free_energy.png":
            ax.axhline(0, color="black", linestyle="--", label="Exact")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / file, dpi=140)
        plt.close(fig)

    # Mode-basin occupancy is *not* the same as left/right halfspace balance.
    # For well-separated 2D Gaussians, the 2*sigma disk contains 1-exp(-2)
    # of its own component. Contributions from the opposite peak are negligible.
    fig, ax = plt.subplots(figsize=(6, 4))
    for metric, label in [
        ("mapped_left_mode_fraction", "Near left mode"),
        ("mapped_right_mode_fraction", "Near right mode"),
        ("mapped_outside_mode_fraction", "Outside both"),
    ]:
        r = [v for v in rows if v["split"] == "validation"]
        ax.plot([v["step"] for v in r], [v[metric] for v in r], label=label)
    expected_per_mode = 0.5 * (1 - math.exp(-2))
    ax.axhline(expected_per_mode, color="gray", linestyle="--",
               label="Target near each mode (approx.)")
    ax.axhline(math.exp(-2), color="black", linestyle=":",
               label="Target outside (approx.)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Held-out mapped fraction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mode_coverage.png", dpi=140)
    plt.close(fig)

    # Save data as well as a visual check so halfspace imbalance cannot be
    # mistaken for actual occupancy of the high-probability target basins.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, mapped, name in [
        (axes[0], initial_mapped, "Before training"),
        (axes[1], final_mapped, "After training"),
    ]:
        ax.scatter(mapped[:, 0], mapped[:, 1], s=3, alpha=0.2, rasterized=True)
        for center in target.centers.numpy():
            ax.scatter(center[0], center[1], s=50, marker="x", color="black")
            ax.add_patch(plt.Circle(center, 2 * target.sigma, color="black",
                                    fill=False, linestyle="--"))
        ax.set_title(name)
        ax.set_xlabel("Mapped x")
        ax.set_xlim(-args.separation - 3, args.separation + 3)
        ax.set_ylim(-4, 4)
    axes[0].set_ylabel("Mapped y")
    fig.tight_layout()
    fig.savefig(out / "mapped_positions.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(initial_work, bins=70, density=True, alpha=0.5, label="initial")
    ax.hist(final_work, bins=70, density=True, alpha=0.5, label="trained")
    ax.set_xlabel("Held-out generalized work (kBT)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "work_distribution.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(out / "validation_work.npz",
                        initial=initial_work, final=final_work,
                        source_validation=validation.numpy(),
                        mapped_initial=initial_mapped, mapped_final=final_mapped)

    first = next(r for r in rows if r["split"] == "validation")
    last = next(r for r in reversed(rows) if r["split"] == "validation")
    summary = {
        "seed": seed, "objective": args.objective, "beta": args.beta if args.objective == "hellinger" else None,
        "initial_validation": first, "final_validation": last, "final_test": test_row,
        "exact_df": target.exact_df,
        "parameters": initial_parameters,
        "initial_parameters_sha256": initial_parameters_sha256,
        **sample_hashes,
        "target_evaluations": target.evaluations, "optimizer_steps": args.steps,
        "training_s": training_s, "elapsed_s": time.perf_counter() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/stage-a/non-gaussian"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--n-train", type=int, default=768)
    parser.add_argument("--n-val", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=0,
                        help="Independent final-only test; zero preserves the A2 protocol.")
    parser.add_argument("--objective", choices=["baseline", "hellinger"], default="baseline")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Power in -log mean(exp(-beta*w)); A3 fixes beta=0.5.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--separation", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=0.7)
    args = parser.parse_args()
    if args.n_test < 0 or not 0 < args.beta < 1:
        parser.error("Require n-test >= 0 and 0 < beta < 1.")
    if args.sigma <= 0 or args.separation <= 2 * args.sigma:
        parser.error("Require sigma > 0 and separation > 2*sigma for disjoint mode disks.")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({
        **vars(args), "out": str(args.out),
        "tfep_commit": "0bc369a1194c688bb6ac997cc6b7c2db2b063c9e",
        "repo_commit": os.environ.get("GITHUB_SHA", "local"),
        "mode_diagnostic": "Euclidean disks radius 2*sigma around each mode",
        "units": "reduced kBT",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "note": ("Pinned upstream mean-work objective" if args.objective == "baseline"
                 else "Hellinger affinity, normalized minibatch exponent beta=0.5; no target samples"),
    }, indent=2, allow_nan=False))
    summaries = [single_experiment(args, seed, args.out / f"seed-{seed}") for seed in args.seeds]
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
