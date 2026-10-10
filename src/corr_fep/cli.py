"""Conservative CLI for dependency checks and numerical smoke tests."""

import argparse
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np

from corr_fep.diagnostics import summarize_work


def doctor():
    modules = ("numpy", "torch", "tfep", "MDAnalysis", "tblite", "lightning", "scipy")
    report = {
        "python_architecture": platform.machine(),
        "python": platform.python_version(),
        "modules_available": {name: importlib.util.find_spec(name) is not None for name in modules},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def demo(seed, n):
    rng = np.random.default_rng(seed)
    work = rng.normal(loc=2.0, scale=0.8, size=n)
    report = {"mode": "synthetic_demo", "seed": seed, **summarize_work(work)}
    print(json.dumps(report, indent=2, sort_keys=True))


def check_data(root):
    """Only checks expected upstream directory names, not scientific completeness."""
    expected = ("p38", "mdp", "topol_shared", "run_tfep.py")
    report = {
        "root": str(root),
        "expected_paths": {item: (root / item).exists() for item in expected},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(report["expected_paths"].values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="corr-fep reproducibility helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="Report local scientific Python dependencies")
    demo_p = sub.add_parser("demo", help="Run a synthetic numerical smoke test")
    demo_p.add_argument("--seed", type=int, default=17)
    demo_p.add_argument("--n", type=int, default=2000)
    data_p = sub.add_parser("check-data", help="Check upstream source layout (not trajectory data)")
    data_p.add_argument("--root", type=Path, default=Path("external/paper"))
    args = parser.parse_args()
    if args.cmd == "doctor":
        doctor()
    elif args.cmd == "demo":
        if args.n < 1:
            parser.error("--n must be positive")
        demo(args.seed, args.n)
    else:
        check_data(args.root)


if __name__ == "__main__":
    main()
