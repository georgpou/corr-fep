#!/usr/bin/env python3
"""B2.2b: isolate native XTC precision bias using paired GROMACS TRR/XTC frames.

Run a short *new* NVT trajectory at the original FF Hamiltonian. The original
simulation EDR, uncompressed TRR rerun, and compressed XTC rerun are compared
at identical times with the SAME GROMACS TPR and executable. This validates
the mechanism; it cannot recover the original authors' missing TRR snapshots.
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

from inspect_zenodo_archive import make_session, require_range_support
from stage_b_mechanical_embedding import (
    OUT as B1_OUT, ROOT, TOPOLOGY, MDP_DIR, GMX, digest, grompp,
    prepare_files, retrieve_topology, run, setup_coordinates,
)
from stage_b_qm_singlepoint import ARCHIVE, PREFIX, load_archive

OUT = Path("results/stage-b/precision-roundtrip").resolve()
NUM_STEPS = 1000
WRITE_STEPS = 100
# Independent, explicitly pinned RNG seeds for velocity initialization and
# the stochastic v-rescale thermostat. Both must be fixed for rerun parity.
VELOCITY_SEED = 1729
THERMOSTAT_SEED = 271828
# With dt=.0005 ps this gives a small 0.5 ps control trajectory, 11 frames.
# This is NOT a statistically independent trajectory for free-energy inference.
DT_PS = 0.0005
XYZ_TERMS = ("Bond", "Angle", "Proper Dih.", "Per. Imp. Dih.",
             "LJ-14", "Coulomb-14", "LJ (SR)", "Coulomb (SR)",
             "Coul. recip.", "Disper. corr.", "Potential")


def write_json(obj):
    (OUT / "precision-roundtrip.json").write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n")


def inspect_archived_members():
    """Confirm which original uncompressed files actually exist via ZIP index."""
    prefix = PREFIX + "eq/water/repeat-0/"
    with make_session() as session:
        require_range_support(ARCHIVE, session)
        with RemoteZip(ARCHIVE, session=session, initial_buffer_size=512 * 1024,
                       timeout=180) as archive:
            items = [m for m in archive.infolist() if m.filename.startswith(prefix)
                     and m.filename.endswith((".edr", ".trr", ".xtc", ".xvg", ".gro"))]
    return [{"member": m.filename, "uncompressed_bytes": m.file_size,
             "compressed_bytes": m.compress_size}
            for m in items]


def write_mdp():
    # Start from the pinned original, already adapted for the archived topology.
    original = (MDP_DIR / "singlepoint.mdp").read_text()
    overrides = {
        "integrator": "md", "nsteps": str(NUM_STEPS), "dt": str(DT_PS),
        "gen-vel": "yes", "gen-temp": "298",
        "gen-seed": str(VELOCITY_SEED), "ld-seed": str(THERMOSTAT_SEED),
        "continuation": "no",
        "pcoupl": "no", "tcoupl": "v-rescale",
        "nstxout": str(WRITE_STEPS), "nstvout": "0", "nstfout": "0",
        "nstxout-compressed": str(WRITE_STEPS),
        "xtc-precision": "1000", "nstenergy": str(WRITE_STEPS),
        "nstcalcenergy": str(WRITE_STEPS), "nstlog": str(WRITE_STEPS),
    }
    seen = set()
    new_lines = []
    for line in original.splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=", line)
        if match:
            # GROMACS normalizes underscore/hyphen aliases to the same MDP key.
            key = match.group(1).lower().replace("_", "-")
            if key in ("nstxtcout", "unconstrained-start", "ld-seed"):
                # Remove legacy trajectory keys and pre-existing thermostat seed;
                # append one unambiguous ld-seed below, not a duplicate.
                continue
            if key in overrides:
                if key in seen:
                    raise ValueError("Duplicated MDP parameter: " + key)
                new_lines.append(f"{key} = {overrides[key]}")
                seen.add(key)
                continue
        new_lines.append(line)
    for key, value in overrides.items():
        if key not in seen:
            new_lines.append(f"{key} = {value}")
    path = OUT / "precision-nvt.mdp"
    path.write_text("\n".join(new_lines) + "\n")
    return path


def energy_series(path):
    raw = pyedr.edr_to_dict(str(path))
    if "Time" not in raw or "Potential" not in raw:
        raise ValueError(f"EDR missing Time/Potential: {path}")
    result = {str(k): np.asarray(v, dtype=np.float64).reshape(-1)
              for k, v in raw.items()}
    n = len(result["Time"])
    if n == 0 or any(len(v) != n for v in result.values()):
        raise ValueError("EDR observations are inconsistent")
    if not np.all(np.diff(result["Time"]) > 0):
        raise ValueError("EDR time axis not increasing")
    if not np.isfinite(result["Potential"]).all():
        raise FloatingPointError("Nonfinite EDR potential")
    return result


def time_index(time_array, target):
    matches = np.flatnonzero(np.isclose(time_array, target, rtol=0, atol=0.00005))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected unique exact EDR time {target} ps, got {len(matches)}")
    return int(matches[0])


def compare_native_coordinates(tpr_path, trr_path, xtc_path, frame_times):
    """Coordinate error is measured only; both energy reruns remain native GROMACS."""
    trr = mda.Universe(str(tpr_path), str(trr_path))
    xtc = mda.Universe(str(tpr_path), str(xtc_path))
    if len(trr.atoms) != len(xtc.atoms):
        raise AssertionError("TRR/XTC atom counts differ")
    if len(trr.trajectory) != len(xtc.trajectory):
        raise AssertionError("TRR/XTC frame counts differ")
    summary = {}
    for k in range(len(trr.trajectory)):
        frame_trr = trr.trajectory[k]
        frame_xtc = xtc.trajectory[k]
        t = float(frame_trr.time)
        if abs(float(frame_xtc.time) - t) > 0.00005:
            raise AssertionError("TRR and XTC sampling times diverge")
        if not np.isclose(frame_times, t, rtol=0, atol=0.00005).any():
            raise AssertionError("Unmatched coordinate frame time")
        a = np.asarray(frame_trr.positions, dtype=np.float64).copy()
        b = np.asarray(frame_xtc.positions, dtype=np.float64).copy()
        delta = a - b
        # For this new NVT control trajectory, matching stream images must agree.
        absmax = float(np.max(np.abs(delta)))
        box_error = float(np.max(np.abs(
            np.asarray(frame_trr.dimensions, dtype=np.float64) -
            np.asarray(frame_xtc.dimensions, dtype=np.float64))))
        summary[f"{t:.5f}"] = {
            "rms_atom_displacement_angstrom":
                float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
            "max_axis_displacement_angstrom": absmax,
            "mean_axis_abs_displacement_angstrom": float(np.mean(np.abs(delta))),
            "box_max_difference_angstrom_degrees": box_error,
        }
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    B1_OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": "B2.2b paired native high-precision vs compressed trajectory",
        "validated": False, "frames": [],
        "limits": [
            "NEW short 2023.3 double NVT trajectory, not original 2021.6 trajectory",
            "TRR and XTC are recorded on identical timesteps from ONE test simulation",
            "No inference that the inaccessible original TRR coordinates were recovered",
            "Newly generated velocities; this is a mechanistic test, not statistical sampling",
        ],
    }
    write_json(report)
    report["archive_candidate_members"] = inspect_archived_members()
    report["archived_full_precision_trr_available"] = any(
        m["member"].endswith(".trr") for m in report["archive_candidate_members"])
    report["archived_energy_edr_available"] = any(
        m["member"].endswith(".edr") for m in report["archive_candidate_members"])
    raw = load_archive()
    topology = retrieve_topology()
    upstream = prepare_files()
    numbers, charge, xyz, idx, box, _ = setup_coordinates(raw)
    if charge != 0:
        raise AssertionError("Expected neutral ligand")
    version = run(GMX, "--version")
    if not re.search(r"^\s*Precision:\s+double\b", version, re.M | re.I):
        raise RuntimeError("Need the accepted double precision GROMACS executable")
    grompp()
    mdp = write_mdp()
    tpr = OUT / "precision-nvt.tpr"
    run(GMX, "grompp", "-f", mdp, "-c", ROOT / "full.gro",
        "-p", TOPOLOGY / "topol_water.top", "-o", tpr,
        "-po", OUT / "precision-processed.mdp",
        cwd=ROOT, logfile=OUT / "precision-grompp.log")
    report["provenance"] = {
        "workflow_sha": os.environ.get("GITHUB_SHA", "local"),
        "gromacs_version": version,
        "nvt_mdp_sha256": digest(mdp.read_bytes()),
        "control_rng": {
            "velocity_initialization_gen_seed": VELOCITY_SEED,
            "stochastic_v_rescale_ld_seed": THERMOSTAT_SEED,
        },
        "nvt_tpr_sha256": digest(tpr.read_bytes()),
        "rebuilt_ff_tpr_sha256": digest((ROOT / "full.tpr").read_bytes()),
        "topology_members": topology, "upstream_commit": upstream["upstream_sha"],
        "archived_inputs_sha256": {k: digest(v) for k, v in raw.items()},
        "atom_count": len(xyz), "ligand_atoms": len(idx),
        "trajectory_duration_ps": NUM_STEPS * DT_PS,
        "xtc_nominal_coordinate_tick_angstrom": 0.01,
    }
    write_json(report)
    start = OUT / "precision-nvt"
    run(GMX, "mdrun", "-s", tpr, "-deffnm", start,
        "-ntmpi", "1", "-ntomp", "1",
        cwd=OUT, logfile=OUT / "precision-md-run.log")
    full_precision = Path(str(start) + ".trr")
    lossy = Path(str(start) + ".xtc")
    md_edr = Path(str(start) + ".edr")
    if not full_precision.exists() or not lossy.exists():
        raise FileNotFoundError("GROMACS did not generate both native TRR and XTC")
    for label, trajectory in (("trr", full_precision), ("xtc", lossy)):
        prefix = OUT / ("replay-" + label)
        run(GMX, "mdrun", "-s", tpr, "-rerun", trajectory,
            "-deffnm", prefix, "-ntmpi", "1", "-ntomp", "1",
            cwd=OUT, logfile=OUT / f"replay-{label}-run.log")
    md = energy_series(md_edr)
    trr = energy_series(OUT / "replay-trr.edr")
    xtc = energy_series(OUT / "replay-xtc.edr")
    report["stream_counts"] = {
        "md_edr": len(md["Time"]), "trr_rerun_edr": len(trr["Time"]),
        "xtc_rerun_edr": len(xtc["Time"]),
    }
    keys = sorted(set(md).intersection(trr).intersection(xtc))
    if any(term not in keys for term in XYZ_TERMS):
        raise AssertionError(f"Required EDR energy terms missing: {set(XYZ_TERMS)-set(keys)}")
    report["energy_components"] = list(XYZ_TERMS)
    report["coordinate_comparison"] = compare_native_coordinates(
        tpr, full_precision, lossy, trr["Time"])
    for i, time_ps in enumerate(trr["Time"]):
        im = time_index(md["Time"], time_ps)
        ix = time_index(xtc["Time"], time_ps)
        values = {
            key: {
                "md": float(md[key][im]),
                "trr": float(trr[key][i]),
                "xtc": float(xtc[key][ix]),
                "xtc_minus_trr": float(xtc[key][ix] - trr[key][i]),
                "trr_minus_md": float(trr[key][i] - md[key][im]),
            }
            for key in XYZ_TERMS
        }
        entry = {
            "time_ps": float(time_ps), "time_matched": True,
            "terms_kj_mol": values,
            "quantization_geometry": report["coordinate_comparison"][f"{time_ps:.5f}"],
        }
        report["frames"].append(entry)
        write_json(report)
        print("NATIVE_PAIR", json.dumps({
            "time_ps": entry["time_ps"],
            "xtc_minus_trr_potential_kj_mol":
                values["Potential"]["xtc_minus_trr"],
            "trr_minus_md_potential_kj_mol":
                values["Potential"]["trr_minus_md"],
            "xtc_minus_trr_bond_kj_mol":
                values["Bond"]["xtc_minus_trr"],
            "xtc_minus_trr_angle_kj_mol":
                values["Angle"]["xtc_minus_trr"],
        }), flush=True)
    errors = [x["terms_kj_mol"]["Potential"]["xtc_minus_trr"]
              for x in report["frames"]]
    # t=0 starts from the prepared GRO and is not comparable to post-integration
    # outputs for a quantization-bias summary. Keep it in the raw table.
    if abs(report["frames"][0]["time_ps"]) > 0.00005:
        raise AssertionError("Control roundtrip first frame must be t=0")
    noninitial = report["frames"][1:]
    if not noninitial:
        raise AssertionError("No postinitial snapshots for precision analysis")
    postinitial_errors = [
        f["terms_kj_mol"]["Potential"]["xtc_minus_trr"] for f in noninitial
    ]
    trr_err = [x["terms_kj_mol"]["Potential"]["trr_minus_md"]
               for x in report["frames"]]
    report["summary"] = {
        "num_paired_frames": len(errors),
        "xtc_minus_trr_mean_kj_mol": float(np.mean(errors)),
        "all_frame_mean_includes_t0": True,
        "noninitial_frame_count": len(noninitial),
        "noninitial_exclusion_rule": "Exclude the t=0 starting GRO frame; use all frames at t>0 (not independent equilibrium samples)",
        "noninitial_xtc_minus_trr_mean_kj_mol": float(np.mean(postinitial_errors)),
        "noninitial_xtc_minus_trr_median_kj_mol": float(np.median(postinitial_errors)),
        "noninitial_xtc_minus_trr_min_kj_mol": float(np.min(postinitial_errors)),
        "noninitial_xtc_minus_trr_max_kj_mol": float(np.max(postinitial_errors)),
        "noninitial_component_mean_xtc_minus_trr_kj_mol": {
            key: float(np.mean([f["terms_kj_mol"][key]["xtc_minus_trr"]
                                for f in noninitial]))
            for key in XYZ_TERMS
        },
        "xtc_minus_trr_median_kj_mol": float(np.median(errors)),
        "xtc_minus_trr_min_kj_mol": float(np.min(errors)),
        "xtc_minus_trr_max_kj_mol": float(np.max(errors)),
        "trr_minus_md_max_abs_kj_mol": float(np.max(np.abs(trr_err))),
        "component_mean_xtc_minus_trr_kj_mol": {
            key: float(np.mean([f["terms_kj_mol"][key]["xtc_minus_trr"]
                                for f in report["frames"]]))
            for key in XYZ_TERMS
        },
        "positive_bias_in_all_test_frames": bool(np.all(np.asarray(errors) > 0)),
    }
    with (OUT / "paired-energy-components.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["time_ps", "included_in_noninitial_summary",
                    "xtc_minus_trr_potential", "trr_minus_md_potential",
                    *["xtc_minus_trr_" + key for key in XYZ_TERMS if key != "Potential"],
                    "coordinate_rms_angstrom"])
        for entry in report["frames"]:
            terms = entry["terms_kj_mol"]
            w.writerow([
                entry["time_ps"], entry["time_ps"] > 0,
                terms["Potential"]["xtc_minus_trr"],
                terms["Potential"]["trr_minus_md"],
                *[terms[key]["xtc_minus_trr"] for key in XYZ_TERMS if key != "Potential"],
                entry["quantization_geometry"]["rms_atom_displacement_angstrom"],
            ])
    print("PRECISION_SUMMARY", json.dumps({
        "all_11_mean_kj_mol": report["summary"]["xtc_minus_trr_mean_kj_mol"],
        "noninitial_10_mean_kj_mol": report["summary"]["noninitial_xtc_minus_trr_mean_kj_mol"],
    }), flush=True)
    report["validated"] = True  # Paired experiment completed, NOT full original replication.
    write_json(report)
    print("B22_PAIRED_NATIVE_PRECISION_MECHANISM_MEASURED", flush=True)


if __name__ == "__main__":
    main()
