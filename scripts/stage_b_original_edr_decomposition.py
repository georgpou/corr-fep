#!/usr/bin/env python3
"""B2.2c: compare the ORIGINAL MD EDR components to XTC-recalculated MM terms.

Use only deposited members via HTTP byte-range ZIP access. Original energies
are from authors' 2021.6 double simulation; rerun energies use pinned 2023.3
double on compressed XTC frames. The term-by-term residual is measured, not
corrected. Neither the original uncompressed TRR nor unseen repeats are inferred.
"""
import csv
import json
import os
from pathlib import Path
import re

import MDAnalysis as mda
import numpy as np
import pyedr
from remotezip import RemoteZip
import torch

from inspect_zenodo_archive import make_session, require_range_support
from stage_b_mechanical_embedding import (
    OUT as B1_OUT, ROOT, GMX, MechanicalEmbedding, digest,
    grompp, prepare_files, retrieve_topology, run, setup_coordinates,
)
from stage_b_multiframe import (
    load_trajectory, register_trajectory_transformations, sampled_frames,
)
from stage_b_qm_singlepoint import ARCHIVE, PREFIX, load_archive
from stage_b_reference_energy import (
    OUT as REFERENCE_OUT, load_reference, match_reference, evaluate_pair,
)

OUT = Path("results/stage-b/original-edr-decomposition").resolve()
ARCHIVED_EDR_NAME = PREFIX + "eq/water/repeat-0/ener.edr"
MAX_EDR_BYTES = 8_000_000

# The GROMACS potential-energy terms common to the original source and the
# single-point run. Exact names are independently verified from each EDR.
CORE = ("Bond", "Angle", "Proper Dih.", "Improper Dih.", "Per. Imp. Dih.",
        "LJ-14", "Coulomb-14", "LJ (SR)", "Coulomb (SR)",
        "Coul. recip.", "Disper. corr.")
BONDED = ("Bond", "Angle", "Proper Dih.", "Per. Imp. Dih.")


def save(data):
    (OUT / "original-edr-decomposition.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def retrieve_original_edr():
    with make_session() as session:
        require_range_support(ARCHIVE, session)
        with RemoteZip(ARCHIVE, session=session, initial_buffer_size=512 * 1024,
                       timeout=180) as z:
            info = z.getinfo(ARCHIVED_EDR_NAME)
            if info.file_size <= 0 or info.file_size > MAX_EDR_BYTES:
                raise RuntimeError(
                    f"Unexpected original EDR size {info.file_size}; refusing download")
            raw = z.read(ARCHIVED_EDR_NAME)
    if len(raw) != info.file_size:
        raise IOError("Incomplete source EDR member")
    path = OUT / "deposited-ener.edr"
    path.write_bytes(raw)
    return path, {
        "member": ARCHIVED_EDR_NAME, "sha256": digest(raw),
        "bytes": len(raw), "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
        "method": "individual ZIP member via verified HTTP byte ranges",
    }


def read_edr(path):
    """Read all native energy series; never interpolate time or map aliases."""
    raw = pyedr.edr_to_dict(str(path))
    data = {str(k): np.asarray(v, dtype=np.float64).reshape(-1)
            for k, v in raw.items()}
    if "Time" not in data or "Potential" not in data:
        raise ValueError(f"Missing Time/Potential from original EDR: {sorted(data)}")
    count = len(data["Time"])
    if count < 4 or any(len(v) != count for v in data.values()):
        raise ValueError("Malformed EDR series lengths")
    if not np.all(np.isfinite(data["Time"])) or not np.all(np.diff(data["Time"]) > 0):
        raise ValueError("EDR time axis invalid/nonmonotonic")
    if not np.isfinite(data["Potential"]).all():
        raise ValueError("EDR Potential contains nonfinite values")
    return data


def exactly_at(times, timestamp_ps):
    indices = np.flatnonzero(np.isclose(times, timestamp_ps, rtol=0, atol=0.001))
    if len(indices) != 1:
        raise AssertionError(
            f"EDR timestamp {timestamp_ps} ps has {len(indices)} matching records")
    return int(indices[0])


def aggregate(report):
    frames = report["frames"]
    terms = frames[0]["energy_terms_kj_mol"]
    means = {term: float(np.mean([
        frame["energy_terms_kj_mol"][term]["rerun_minus_original"]
        for frame in frames])) for term in terms}
    report["aggregate"] = {
        "num_exactly_matched_frames": len(frames),
        "mean_rerun_minus_original_by_term_kj_mol": means,
        "potential_discrepancy_min_kj_mol": float(min(
            f["potential_difference_kj_mol"] for f in frames)),
        "potential_discrepancy_max_kj_mol": float(max(
            f["potential_difference_kj_mol"] for f in frames)),
        "mean_potential_discrepancy_kj_mol": means["Potential"],
        "mean_bond_plus_angle_kj_mol": means["Bond"] + means["Angle"],
        "mean_remaining_energy_difference_kj_mol":
            means["Potential"] - means["Bond"] - means["Angle"],
        "mean_bond_plus_angle_fraction_of_potential_difference":
            ((means["Bond"] + means["Angle"]) / means["Potential"]
             if means["Potential"] != 0 else None),
        "max_abs_term_sum_residual_kj_mol": float(max(
            abs(f["decomposition"]["potential_minus_sum_reported_terms_kj_mol"])
            for f in frames)),
        "max_abs_original_edr_minus_xvg_kj_mol": float(max(
            abs(f["original_edr_minus_xvg_kj_mol"]) for f in frames)),
        "archived_tpr_minus_rebuilt_tpr_max_abs_kj_mol": float(max(
            abs(f["archived_tpr_minus_rebuilt_tpr_potential_kj_mol"])
            for f in frames)),
    }


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    B1_OUT.mkdir(parents=True, exist_ok=True)
    REFERENCE_OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": "B2.2c deposited EDR potential decomposition",
        "validated": False, "frames": [],
        "conditions": [
            "Original ener.edr is from 2021.6 double MD; reruns are 2023.3 double",
            "Rerun coordinates come from lossy XTC; original uncompressed TRR unavailable",
            "No assumption of zero force-field changes between simulation engines",
            "Identical snapshots are selected by timestamp, not by nearest row",
            "This is a diagnostic, not a correction scheme, importance weight or Delta G",
        ],
    }
    save(report)
    original_path, original_meta = retrieve_original_edr()
    edr = read_edr(original_path)
    xvg, xvg_meta = load_reference()
    report["deposited"] = {
        "edr": original_meta,
        "edr_time_records": int(len(edr["Time"])),
        "edr_first_time_ps": float(edr["Time"][0]),
        "edr_last_time_ps": float(edr["Time"][-1]),
        "edr_energy_term_names": sorted(edr.keys()),
        "xvg": xvg_meta,
    }
    save(report)
    raw = load_archive()
    topology = retrieve_topology()
    upstream = prepare_files()
    numbers, charge, gro_xyz, idx, gro_box, provenance = setup_coordinates(raw)
    if charge != 0:
        raise AssertionError("Ligand charge changed")
    traj, xtc_meta = load_trajectory()
    gmx_version = run(GMX, "--version")
    if not re.search(r"^\s*Precision:\s+double\b", gmx_version, re.M | re.I):
        raise RuntimeError("Requires pinned double-precision GROMACS")
    grompp()
    report["runtime"] = {
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "gromacs_version": gmx_version,
        "archived_production_tpr_sha256": digest((ROOT/"archived-eq.tpr").read_bytes()),
        "rebuilt_singlepoint_tpr_sha256": digest((ROOT/"full.tpr").read_bytes()),
        "xtc": xtc_meta,
        "topology": topology,
        "upstream_commit": upstream["upstream_sha"],
        "archive_input_hashes": {k: digest(v) for k, v in raw.items()},
    }
    save(report)
    universe = mda.Universe(str(ROOT/"archived-eq.tpr"), str(traj))
    register_trajectory_transformations(universe, idx)
    model = MechanicalEmbedding(idx, numbers, gro_box)
    for frame_idx, n_frames, time_ps, xyz, cell, bond_limits in sampled_frames(
            universe, idx, gro_xyz):
        edr_idx = exactly_at(edr["Time"], time_ps)
        xvg_idx, xvg_t, xvg_potential = match_reference(xvg, time_ps, frame_idx)
        original_potential = float(edr["Potential"][edr_idx])
        if abs(original_potential - xvg_potential) > 0.001:
            raise AssertionError(
                f"Original EDR and XVG Potential do not agree at {time_ps} ps: "
                f"{original_potential} vs {xvg_potential}")
        rerun = evaluate_pair(model, xyz, cell, f"original-edr-frame-{frame_idx}")
        archived = rerun["archived_production_tpr_terms_kj_mol"]
        rebuilt = rerun["rebuilt_singlepoint_tpr_terms_kj_mol"]
        original_keys, rerun_keys = set(edr), set(archived)
        required = {"Bond", "Angle", "Potential"}
        missing = required - (original_keys & rerun_keys)
        if missing:
            raise AssertionError("Missing indispensable energy terms: " + repr(sorted(missing)))
        terms = {
            term: {
                "original": float(edr[term][edr_idx]) if term in edr else 0.0,
                "rerun": float(archived[term]) if term in archived else 0.0,
                "original_term_present": term in edr,
                "rerun_term_present": term in archived,
                "rerun_minus_original": float(
                    (archived[term] if term in archived else 0.0) -
                    (edr[term][edr_idx] if term in edr else 0.0)),
            }
            for term in (*CORE, "Potential")
        }
        delta = terms["Potential"]["rerun_minus_original"]
        accounted = sum(terms[k]["rerun_minus_original"] for k in CORE)
        entry = {
            "xtc_frame_index": frame_idx, "xtc_time_ps": time_ps,
            "original_edr_row": edr_idx,
            "original_edr_time_ps": float(edr["Time"][edr_idx]),
            "xvg_row": xvg_idx, "xvg_time_ps": xvg_t,
            "original_edr_minus_xvg_kj_mol":
                original_potential - xvg_potential,
            "potential_difference_kj_mol": float(delta),
            "energy_terms_kj_mol": terms,
            "decomposition": {
                "sum_reported_term_differences_kj_mol": float(accounted),
                "potential_minus_sum_reported_terms_kj_mol": float(delta-accounted),
                "bond_plus_angle_kj_mol": float(
                    terms["Bond"]["rerun_minus_original"] +
                    terms["Angle"]["rerun_minus_original"]),
                "remaining_after_bond_angle_kj_mol": float(
                    delta-terms["Bond"]["rerun_minus_original"] -
                    terms["Angle"]["rerun_minus_original"]),
                "original_edr_only_terms": sorted(original_keys-rerun_keys),
                "rerun_only_terms": sorted(rerun_keys-original_keys),
                "unequal_improper_dihedral_labels_not_assumed_equivalent": True,
                "missing_term_convention": "Absent term contributes zero to the enumerated GROMACS EDR sum; this is an accounting convention, not a cross-version functional-form mapping",
                "term_sum_interpretation": "Numerical closure, not cross-engine physical equivalence of individually labeled components",
            },
            "archived_tpr_minus_rebuilt_tpr_potential_kj_mol":
                float(archived["Potential"] - rebuilt["Potential"]),
            "original_and_replayed_box": {
                "original_box_xyz_nm": {
                    axis: float(edr[f"Box-{axis}"][edr_idx])
                    for axis in "XYZ" if f"Box-{axis}" in edr
                },
                "xtc_box_abc_nm": (cell[:3]/10).tolist(),
            },
            "g96_sha256": rerun["input_g96_sha256"],
            "ligand_bond_range_angstrom": bond_limits,
        }
        report["frames"].append(entry)
        save(report)
        print("ORIGINAL_EDR_FRAME", json.dumps({
            "frame": frame_idx, "time_ps": time_ps,
            "deposited_potential": original_potential,
            "rerun_potential": archived["Potential"],
            "rerun_minus_deposited": delta,
            "bond": terms["Bond"]["rerun_minus_original"],
            "angle": terms["Angle"]["rerun_minus_original"],
            "other_net": entry["decomposition"]["remaining_after_bond_angle_kj_mol"],
            "decomposition_sum_residual":
                entry["decomposition"]["potential_minus_sum_reported_terms_kj_mol"],
        }), flush=True)
    if len(report["frames"]) != 4:
        raise AssertionError("Exactly four archived production frames expected")
    aggregate(report)
    with (OUT / "original-edr-components.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["xtc_index", "time_ps", "deposited_original_potential",
                         "xtc_rerun_potential", "rerun_minus_original_potential",
                         *["delta_" + k for k in CORE],
                         "sum_accounting_residual"])
        for f in report["frames"]:
            terms = f["energy_terms_kj_mol"]
            writer.writerow([
                f["xtc_frame_index"], f["xtc_time_ps"],
                terms["Potential"]["original"],
                terms["Potential"]["rerun"],
                terms["Potential"]["rerun_minus_original"],
                *[terms[k]["rerun_minus_original"] for k in CORE],
                f["decomposition"]["potential_minus_sum_reported_terms_kj_mol"],
            ])
    report["validated"] = bool(
        report["deposited"]["xvg"]["potential_identity_from_xvg"] and
        report["aggregate"]["max_abs_original_edr_minus_xvg_kj_mol"] < 0.001 and
        report["aggregate"]["max_abs_term_sum_residual_kj_mol"] < 0.1 and
        report["aggregate"]["archived_tpr_minus_rebuilt_tpr_max_abs_kj_mol"] < 0.001
    )
    report["validated_meaning"] = (
        "Deposited EDR agrees with XVG, four frame times match exactly, "
        "reported components account for Potential, and same-XTC-input TPR reruns agree. "
        "This does NOT establish that missing original TRR coordinates are recovered "
        "or that FF/QM reweighting is unbiased."
    )
    save(report)
    if not report["validated"]:
        raise AssertionError("Original EDR decomposition inconsistency; inspect JSON report")
    print("B22_ORIGINAL_EDR_DECOMPOSITION_VALIDATED", flush=True)


if __name__ == "__main__":
    main()
