#!/usr/bin/env python3
"""Probe one archived published endpoint using a few ZIP members only.

This does NOT train a flow or evaluate new QM/MM energies.
Reads authors' saved analysis and per-sample tensors without fetching the ZIP.
"""
import io
import json
from pathlib import Path

import numpy as np
from remotezip import RemoteZip

from inspect_zenodo_archive import ARCHIVE_URL, make_session, require_range_support

LIGAND = "p38a_2g"
CONFIG = "map00-cond35"
KT_KJ_PER_MOL = 8.31446261815324e-3 * 298.0


def logsumexp(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        raise ValueError("Empty work array")
    return np.logaddexp.reduce(a)


def energy_from_xvg(archive, path):
    text = archive.read(path).decode("utf-8")
    lines = [s for s in text.splitlines() if s.strip() and not s.startswith(("@", "#"))]
    return np.loadtxt(io.StringIO("\n".join(lines)), usecols=(1,), ndmin=1)


def numeric_result(archive, phase, tensor_arrays):
    prefix = f"lig_{LIGAND}/eq/{phase}/"
    energy_paths = sorted(
        (name for name in archive.namelist()
         if name.startswith(prefix) and name.endswith("/ener.xvg")),
        key=lambda name: int(name.split("/repeat-")[1].split("/")[0])
    )
    if not energy_paths:
        return {"status": "no per-repeat ener.xvg files", "found": []}

    expected = {"potential", "log_det_J", "trajectory_sample_index"}
    if not expected.issubset(tensor_arrays):
        return {"status": "train-tensor field names do not match expected keys",
                "found": list(tensor_arrays)}

    reference = np.concatenate([energy_from_xvg(archive, p) for p in energy_paths])
    indices = np.asarray(tensor_arrays["trajectory_sample_index"], dtype=np.int64).ravel()
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= reference.size):
        return {"status": "trajectory indices inconsistent with available reference data",
                "n_reference": int(reference.size), "n_indices": int(indices.size),
                "min_index": int(indices.min()) if indices.size else None,
                "max_index": int(indices.max()) if indices.size else None}

    potential = np.asarray(tensor_arrays["potential"], dtype=np.float64).ravel()
    logdet = np.asarray(tensor_arrays["log_det_J"], dtype=np.float64).ravel()
    if len(potential) != len(indices) or len(logdet) != len(indices):
        return {"status": "tensor lengths do not match"}
    work = (potential - logdet) * KT_KJ_PER_MOL - reference[indices]
    valid = np.isfinite(work)
    work = work[valid]
    log_weights = -work / KT_KJ_PER_MOL
    log_total = logsumexp(log_weights)
    log_neff = 2 * log_total - logsumexp(2 * log_weights)
    return {
        "status": "calculated from archived per-sample tensors and FF XVGs",
        "reference_files": energy_paths,
        "reference_samples": int(len(reference)),
        "valid_samples": int(len(work)),
        "invalid_samples": int(len(valid) - valid.sum()),
        "delta_g_kj_mol": float(-KT_KJ_PER_MOL * (log_total - np.log(len(work)))),
        "work_std_kj_mol": float(np.std(work, ddof=1)),
        "raw_importance_ess": float(np.exp(log_neff)),
        "raw_importance_ess_fraction": float(np.exp(log_neff) / len(work)),
        "note": "ESS ignores trajectory autocorrelation; work uses on-training-run multimap tensors."
    }


def main():
    output_dir = Path("archive-metadata")
    output_dir.mkdir(exist_ok=True)
    result = {
        "dataset_doi": "10.5281/zenodo.21919704",
        "ligand": LIGAND,
        "configuration": CONFIG,
        "kT_kj_mol": KT_KJ_PER_MOL,
        "phases": {},
    }
    with make_session() as session:
        require_range_support(ARCHIVE_URL, session=session)
        with RemoteZip(ARCHIVE_URL, session=session, initial_buffer_size=512 * 1024, timeout=120) as archive:
            names = archive.namelist()
            root = f"lig_{LIGAND}/"
            result["input_candidates"] = [
                {"name": n, "compressed_bytes": archive.getinfo(n).compress_size}
                for n in names if n.startswith(root) and n.endswith(
                    (".xtc", ".trr", ".edr", ".xvg", ".tpr", ".top", ".gro"))
            ]
            for phase in ("water", "protein"):
                subdir = f"{root}tfep-mixedmaf/{CONFIG}/{phase}/tfep_logs/"
                analysis_name = subdir + "analysis.json"
                npz_name = subdir + "train/epoch-0.npz"
                published = json.loads(archive.read(analysis_name))
                with np.load(io.BytesIO(archive.read(npz_name)), allow_pickle=False) as arrays:
                    tensors = {key: arrays[key] for key in arrays.files}
                phase_result = {
                    "archive_analysis": published,
                    "train_tensor_shapes": {
                        k: list(v.shape) for k, v in tensors.items()
                    },
                    "independent_recalculation": numeric_result(archive, phase, tensors),
                }
                result["phases"][phase] = phase_result
                print(phase, "published df:", published.get("df", {}).get("true"))
                print(phase, "archive tensor shapes:", phase_result["train_tensor_shapes"])
                print(phase, "recalculation:", phase_result["independent_recalculation"])

    dest = output_dir / "p38a_2g-map00-cond35.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("FF trajectory/topology candidates:", len(result["input_candidates"]))
    for item in result["input_candidates"]:
        print(" ", item["name"], item["compressed_bytes"])
    print("Analysis file:", dest)


if __name__ == "__main__":
    main()
