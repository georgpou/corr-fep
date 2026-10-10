#!/usr/bin/env python3
"""Stage B2.2: compare time-matched published FF energies to real GROMACS reruns.

Scientific audit only: no corrections, fitting, ESS, or free energies.
Only individual members of the ~49 GB Zenodo ZIP are downloaded.
"""
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil

import MDAnalysis as mda
import numpy as np
import pyedr
from remotezip import RemoteZip
import torch

from inspect_zenodo_archive import make_session, require_range_support
from stage_b_mechanical_embedding import (
    OUT as B1_OUT, ROOT, GMX, MechanicalEmbedding, digest, grompp,
    prepare_files, retrieve_topology, run, setup_coordinates,
)
from stage_b_multiframe import (
    XTC_NAME, load_trajectory, register_trajectory_transformations,
    sampled_frames,
)
from stage_b_qm_singlepoint import ARCHIVE, PREFIX, GRO_NAME, load_archive

OUT = Path("results/stage-b/reference-energy").resolve()
REFERENCE_NAME = PREFIX + "eq/water/repeat-0/ener.xvg"
MAX_REFERENCE_BYTES = 4_000_000
# 0.01 Å nominal XTC grid (xtc-precision=1000 in nm); hypothetical
# perturbations are NOT a reconstruction of unavailable TRR coordinates.
HALF_TICK_ANGSTROM = 0.005
TEMPERATURE_K = 298.0
RT_KJ_MOL = 8.31446261815324e-3 * TEMPERATURE_K


def save_json(data):
    (OUT / "reference-energy.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_reference():
    with make_session() as session:
        require_range_support(ARCHIVE, session)
        with RemoteZip(ARCHIVE, session=session,
                       initial_buffer_size=512 * 1024, timeout=180) as z:
            info = z.getinfo(REFERENCE_NAME)  # exact archive member; fail if absent
            if info.file_size > MAX_REFERENCE_BYTES:
                raise RuntimeError("Refusing oversized reference-energy member")
            data = z.read(REFERENCE_NAME)
    if len(data) != info.file_size:
        raise IOError("Incomplete reference-energy member")
    (OUT / "deposited-ener.xvg").write_bytes(data)
    text = data.decode("utf-8")
    headers = [line for line in text.splitlines() if line.lstrip().startswith(("@", "#"))]
    rows = [line for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith(("@", "#", "&"))]
    if not rows:
        raise ValueError("Deposited ener.xvg contains no numeric observations")
    numeric = np.loadtxt(io.StringIO("\n".join(rows)), ndmin=2)
    if numeric.ndim != 2 or numeric.shape[1] < 2 or not np.isfinite(numeric).all():
        raise ValueError("Invalid XVG data; expected finite time plus energy columns")
    if not np.all(np.diff(numeric[:, 0]) > 0):
        raise ValueError("XVG timestamps are not strictly increasing")

    legend_matches = re.findall(
        r'^\s*@\s+s(\d+)\s+legend\s+"([^"]+)"', text, flags=re.M | re.I
    )
    legends = {int(i): label for i, label in legend_matches}
    ylabel = re.findall(r'^\s*@\s+yaxis\s+label\s+"([^"]+)"',
                       text, flags=re.M | re.I)
    xlabel = re.findall(r'^\s*@\s+xaxis\s+label\s+"([^"]+)"',
                       text, flags=re.M | re.I)
    # The original run_tfep.py explicitly loads column 1 as FF potential.
    # We independently require the XVG to identify that column as Potential.
    label = legends.get(0, "")
    identity_verified = label.strip().lower() == "potential" or (
        numeric.shape[1] == 2 and "potential" in (ylabel[0].lower() if ylabel else "")
    )
    if numeric.shape[1] != 2:
        # Multiple curves require a specific s0 label; a generic y-axis cannot suffice.
        identity_verified = label.strip().lower() == "potential"
    manifest = {
        "member": REFERENCE_NAME, "sha256": digest(data),
        "uncompressed_bytes": len(data), "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}", "retrieval": "HTTP byte-range ZIP member",
        "header_lines": headers, "x_axis_label": xlabel, "y_axis_label": ylabel,
        "legends": legends, "columns": int(numeric.shape[1]),
        "records": int(len(numeric)), "time_start_ps": float(numeric[0, 0]),
        "time_end_ps": float(numeric[-1, 0]),
        "time_min_step_ps": float(np.min(np.diff(numeric[:, 0]))),
        "time_max_step_ps": float(np.max(np.diff(numeric[:, 0]))),
        "potential_column": 1,
        "potential_identity_from_xvg": bool(identity_verified),
        "upstream_reader": "run_tfep.py: np.loadtxt(..., usecols=1), interpreted as FF potential",
        "units": "kJ/mol if identified as GROMACS Potential; time in ps",
    }
    return numeric, manifest


def edr_terms(edr_path):
    """Copy all available single-point GROMACS EDR terms without guessing labels."""
    raw = pyedr.edr_to_dict(str(edr_path))
    terms = {}
    for key, array in raw.items():
        values = np.asarray(array)
        if values.size == 1 and np.issubdtype(values.dtype, np.number):
            terms[str(key)] = float(values.ravel()[0])
    if "Potential" not in terms or not np.isfinite(terms["Potential"]):
        raise ValueError("GROMACS EDR does not contain a finite Potential term")
    return terms


def evaluate_pair(model, xyz, box, name):
    """Evaluate identical G96 precision/coordinates under both real TPRs."""
    x = torch.as_tensor(xyz.reshape(1, -1), dtype=torch.float64)
    cell = torch.as_tensor(box.reshape(1, 6), dtype=torch.float64)
    with torch.no_grad():
        rebuilt_energy = float(model.mm_full(x, cell).item())

    # TFEP's GROMACSPotential writes its rerun input as a 9-decimal-nm G96.
    # Reuse EXACTLY that file with the archived production TPR, rather than
    # introducing a second writer or any coordinate reformatting.
    g96 = ROOT / "rerun-full" / "configuration.g96"
    rebuilt_edr = ROOT / "rerun-full" / "energy.edr"
    if not g96.exists() or not rebuilt_edr.exists():
        raise FileNotFoundError("Missing actual TFEP G96/EDR rerun files")
    workdir = OUT / "reruns"
    workdir.mkdir(exist_ok=True)
    input_path = workdir / (name + ".g96")
    shutil.copyfile(g96, input_path)
    new_terms = edr_terms(rebuilt_edr)
    if abs(new_terms["Potential"] - rebuilt_energy) > 0.001:
        raise AssertionError("TFEP scalar and rebuilt EDR Potential disagree")
    # Keep the raw EDR for each frame; TFEP overwrites its rerun-full/energy.edr.
    rebuilt_edr_copy = workdir / (name + "-rebuilt-tpr.edr")
    shutil.copyfile(rebuilt_edr, rebuilt_edr_copy)

    prefix = workdir / (name + "-archived-tpr")
    run(GMX, "mdrun", "-s", ROOT / "archived-eq.tpr", "-rerun", input_path,
        "-deffnm", prefix, "-ntmpi", "1", "-ntomp", "1",
        cwd=workdir, logfile=workdir / (name + "-archived-tpr.log"))
    archived_edr = Path(str(prefix) + ".edr")
    original_terms = edr_terms(archived_edr)
    return {
        "input_g96_sha256": digest(input_path.read_bytes()),
        "rebuilt_tpr_edr_sha256": digest(rebuilt_edr_copy.read_bytes()),
        "archived_production_tpr_edr_sha256": digest(archived_edr.read_bytes()),
        "rerun_files_relative_to_reference_output": [
            "reruns/" + name + ".g96",
            "reruns/" + name + "-rebuilt-tpr.edr",
            "reruns/" + name + "-archived-tpr.edr",
        ],
        "archived_production_tpr_terms_kj_mol": original_terms,
        "rebuilt_singlepoint_tpr_terms_kj_mol": new_terms,
        "archived_minus_rebuilt_potential_kj_mol":
            original_terms["Potential"] - rebuilt_energy,
    }


def extract_tpr_settings():
    selected = {}
    for label, path in (
        ("archived_production", ROOT / "archived-eq.tpr"),
        ("rebuilt_singlepoint", ROOT / "full.tpr"),
    ):
        dump = run(GMX, "dump", "-s", path)
        terms = ("coulombtype", "pbc", "rvdw", "rcoulomb", "vdw-",
                 "dispcorr", "constraint", "refcoord", "free-energy",
                 "define", "integrator", "tcoupl", "pcoupl")
        matches = [s.strip() for s in dump.splitlines()
                   if any(token in s.lower() for token in terms)]
        (OUT / (label + "-tpr-settings.txt")).write_text(
            "\n".join(matches[:200]) + "\n", encoding="utf-8")
        selected[label] = {"sha256": digest(path.read_bytes()),
                           "settings_file": label + "-tpr-settings.txt"}
    return selected


def match_reference(table, time_ps, frame_index):
    """No interpolation or silent nearest-frame matching is permitted."""
    matches = np.flatnonzero(np.isclose(
        table[:, 0], time_ps, rtol=0, atol=0.001))
    if len(matches) != 1:
        raise AssertionError(
            f"Time {time_ps} ps matched {len(matches)} deposited XVG records")
    row = int(matches[0])
    if row != frame_index:
        raise AssertionError(
            f"XTC index {frame_index}, energy row {row}: shifted sampling")
    return row, float(table[row, 0]), float(table[row, 1])


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    B1_OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    table, reference = load_reference()
    report = {
        "checkpoint": "B2.2 reference observable + time matching + precision sensitivity",
        "validated": False, "reference": reference, "frames": [],
        "cautions": [
            "GROMACS 2021.6 double production versus 2023.3 double rerun",
            "G96 replay has no original frame time/velocities; time-dependent terms must be excluded",
            "XTC stores lossy coordinates; deposited full-precision TRR is unavailable",
            "Uniform jitter is a controlled sensitivity probe, not recovered original coordinates",
            "A four-frame repeat-0 experiment is not a statistical estimate of reweighting error",
        ],
        "rt_kj_mol_at_298_k": RT_KJ_MOL,
    }
    save_json(report)
    print("REFERENCE", json.dumps(reference), flush=True)
    raw = load_archive()
    topology = retrieve_topology()
    upstream = prepare_files()
    numbers, charge, gro_xyz, ligand_idx, gro_box, b0 = setup_coordinates(raw)
    if charge != 0:
        raise AssertionError("Ligand charge assumption changed")
    trajectory_path, trajectory_meta = load_trajectory()
    gromacs_version = run(GMX, "--version")
    if not re.search(r"^\s*Precision:\s+double\b", gromacs_version, re.M | re.I):
        raise RuntimeError("Expected pinned double-precision GROMACS")
    grompp()
    report["provenance"] = {
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "gromacs_version": gromacs_version,
        "archived_inputs_sha256": {k: digest(v) for k, v in raw.items()},
        "topology_members": topology, "upstream": upstream,
        "trajectory": trajectory_meta, "tprs": extract_tpr_settings(),
        "gro_title": raw[GRO_NAME].decode("utf-8").splitlines()[0],
        "gro_has_explicit_timestamp": False,
        "atom_count": int(len(gro_xyz)),
        "ligand_atom_count": int(len(ligand_idx)),
    }
    save_json(report)
    universe = mda.Universe(str(ROOT / "archived-eq.tpr"), str(trajectory_path))
    register_trajectory_transformations(universe, ligand_idx)
    model = MechanicalEmbedding(ligand_idx, numbers, gro_box)
    final_xyz, final_box = None, None
    for index, total, time_ps, xyz, box, bond_range in sampled_frames(
            universe, ligand_idx, gro_xyz):
        row, matched_time, deposited = match_reference(table, time_ps, index)
        print(f"COMPARE index={index} time={time_ps} ps", flush=True)
        base = evaluate_pair(model, xyz, box, f"xtc-{index}")
        original = base["archived_production_tpr_terms_kj_mol"]["Potential"]
        rebuilt = base["rebuilt_singlepoint_tpr_terms_kj_mol"]["Potential"]
        rng = np.random.default_rng(1729 + index)
        jitter = rng.uniform(-HALF_TICK_ANGSTROM, HALF_TICK_ANGSTROM, xyz.shape)
        perturbed = {}
        for label, sign in (("plus", 1), ("minus", -1)):
            detail = evaluate_pair(model, xyz + sign * jitter, box,
                                   f"xtc-{index}-jitter-{label}")
            perturbed[label] = {
                "archived_tpr_delta_kj_mol":
                    detail["archived_production_tpr_terms_kj_mol"]["Potential"] - original,
                "rebuilt_tpr_delta_kj_mol":
                    detail["rebuilt_singlepoint_tpr_terms_kj_mol"]["Potential"] - rebuilt,
                "archived_tpr_terms_kj_mol":
                    detail["archived_production_tpr_terms_kj_mol"],
                "input_g96_sha256": detail["input_g96_sha256"],
            }
        entry = {
            "xtc_frame_index": index, "xtc_total_frames": total,
            "xtc_time_ps": time_ps, "xvg_row": row, "xvg_time_ps": matched_time,
            "time_difference_ps": time_ps - matched_time,
            "periodic_box_angstrom_degrees": box.tolist(),
            "ligand_bond_range_angstrom": bond_range,
            "deposited_xvg_potential_kj_mol": deposited,
            "rerun": base,
            "archived_tpr_minus_deposited_kj_mol": original - deposited,
            "rebuilt_tpr_minus_deposited_kj_mol": rebuilt - deposited,
            "controlled_jitter": {
                "definition": "All atoms perturbed by identical seeded uniform [-0.005,+0.005] Å vector, plus/minus; NOT original TRR data",
                "seed": 1729 + index,
                "max_abs_coordinate_change_angstrom": float(np.max(np.abs(jitter))),
                "plus": perturbed["plus"], "minus": perturbed["minus"],
                "archived_tpr_plus_minus_span_reduced":
                    (perturbed["plus"]["archived_tpr_delta_kj_mol"] -
                     perturbed["minus"]["archived_tpr_delta_kj_mol"]) / RT_KJ_MOL,
            },
        }
        report["frames"].append(entry)
        save_json(report)
        print("MATCHED_FRAME", json.dumps({
            "index": index, "time_ps": time_ps, "xvg": deposited,
            "archived_rerun": original, "rebuilt_rerun": rebuilt,
            "jitter_archived_plus_minus_kj_mol": [
                perturbed[key]["archived_tpr_delta_kj_mol"] for key in ("plus", "minus")
            ],
        }), flush=True)
        if index == total - 1:
            final_xyz, final_box = xyz, box

    if final_xyz is None:
        raise AssertionError("Expected the terminal production frame")
    gro_result = evaluate_pair(model, gro_xyz, gro_box, "gro-diagnostic")
    # Precisely the B2.1 outlier; do not introduce this hybrid as trajectory data.
    hybrid_xyz = final_xyz.copy()
    hybrid_xyz[463] = gro_xyz[463]
    hybrid_result = evaluate_pair(model, hybrid_xyz, final_box, "outlier463-hybrid")
    for tpr_key in ("archived_production_tpr_terms_kj_mol",
                    "rebuilt_singlepoint_tpr_terms_kj_mol"):
        actual = report["frames"][-1]["rerun"][tpr_key]["Potential"]
        report.setdefault("endpoint_diagnostic_only", {})[tpr_key] = {
            "xtc_minus_gro_kj_mol":
                actual - gro_result[tpr_key]["Potential"],
            "hybrid_minus_xtc_kj_mol":
                hybrid_result[tpr_key]["Potential"] - actual,
            "hybrid_minus_gro_kj_mol":
                hybrid_result[tpr_key]["Potential"] - gro_result[tpr_key]["Potential"],
        }
    report["endpoint_diagnostic_only"]["description"] = (
        "GRO lacks independent timestamp; diagnostic hybrid replaces only water oxygen 463; neither is an extra production sample")
    report["all_four_times_exactly_matched"] = len(report["frames"]) == 4
    report["validated"] = bool(
        report["all_four_times_exactly_matched"] and
        report["reference"]["potential_identity_from_xvg"]
    )
    with (OUT / "frame-comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["xtc_index", "time_ps", "xvg_potential_kj_mol",
                         "archived_tpr_potential_kj_mol", "rebuilt_tpr_potential_kj_mol",
                         "archived_minus_xvg_kj_mol", "rebuilt_minus_xvg_kj_mol",
                         "jitter_archived_plus_kj_mol", "jitter_archived_minus_kj_mol"])
        for row in report["frames"]:
            jitter = row["controlled_jitter"]
            writer.writerow([
                row["xtc_frame_index"], row["xtc_time_ps"],
                row["deposited_xvg_potential_kj_mol"],
                row["rerun"]["archived_production_tpr_terms_kj_mol"]["Potential"],
                row["rerun"]["rebuilt_singlepoint_tpr_terms_kj_mol"]["Potential"],
                row["archived_tpr_minus_deposited_kj_mol"],
                row["rebuilt_tpr_minus_deposited_kj_mol"],
                jitter["plus"]["archived_tpr_delta_kj_mol"],
                jitter["minus"]["archived_tpr_delta_kj_mol"],
            ])
    report["interpretation"] = (
        "validated=True confirms XVG labeling and exact time alignment ONLY; "
        "energy equivalence, systematic precision bias, and suitability for "
        "free-energy reweighting require interpretation of numerical residuals."
    )
    save_json(report)
    if not report["validated"]:
        raise AssertionError("Missing Potential identity or time matching; inspect saved artifact")
    print("B22_REFERENCE_TIME_AND_IDENTITY_VALIDATED; ENERGY_EQUIVALENCE_NOT_ASSERTED", flush=True)


if __name__ == "__main__":
    main()
