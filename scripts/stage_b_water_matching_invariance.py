#!/usr/bin/env python3
"""Actual-trajectory, no-QM invariance probe for the pinned author's water matcher.

Change only ligand coordinates (rigid translation/rotation) while keeping
the physical environment and periodic box bitwise unchanged in each source
snapshot. Recompute full ligand-aligned/Hungarian assignments independently
on the perturbed configurations. No ML mapping, QM/MM or optimization.
"""
import json
import math
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np

from stage_b_qm_singlepoint import load_archive, TPR_NAME
from stage_b_multiframe import load_trajectory
from stage_b_mechanical_embedding import ROOT
from stage_b_author_baseline_utils import load_driver
from stage_b_author_map import adapted_match, memory_subset, SCAFFOLD, CUTOFF

OUT = Path("results/stage-b/water-matching-invariance")
FRAMES = (401, 409, 550, 725, 950, 1050, 1151, 1275, 1400)
REFERENCE_FRAME = 409  # Authors-style full baseline's reference snapshot.
# Every trial is a rigid ligand-only operation in Angstrom or degrees.
TRIALS = (
    ("original", "none", 0.0),
    ("tx_0.02A", "tx", 0.02),
    ("tx_0.10A", "tx", 0.10),
    ("tx_0.30A", "tx", 0.30),
    ("tx_0.75A", "tx", 0.75),
    ("ty_0.10A", "ty", 0.10),
    ("rz_5deg", "rz", 5.0),
    ("rz_15deg", "rz", 15.0),
    ("rz_30deg", "rz", 30.0),
)

def perturb(x, ligand_indices, operation, value):
    result = x.copy()
    if operation == "none":
        return result
    if operation.startswith("t"):
        axis = 0 if operation == "tx" else 1
        result[ligand_indices, axis] += value
        return result
    if operation == "rz":
        points = x[ligand_indices]
        center = points.mean(axis=0)
        a = math.radians(value)
        c, s = math.cos(a), math.sin(a)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        result[ligand_indices] = (points-center) @ rotation.T + center
        return result
    raise ValueError(operation)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(status="initializing", method="pinned adapted original match_waters",
                  source_reference_global_frame=REFERENCE_FRAME,
                  source_frames=FRAMES, trials=[t[0] for t in TRIALS],
                  qm_evaluations=0, training_updates=0)
    dest = OUT / "water-assignment-invariance.json"
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        raw = load_archive()
        topo = ROOT / "archived-eq.tpr"
        topo.write_bytes(raw[TPR_NAME])
        trajectory, provenance = load_trajectory()
        driver = load_driver()
        original = mda.Universe(str(topo), str(trajectory))
        driver.fix_universe_elements(original)
        ligand = original.select_atoms("resname MOL")
        lig_indices = np.asarray(ligand.indices, dtype=int)
        if len(lig_indices) == 0 or not len(ligand.bonds):
            raise AssertionError("No bonded ligand in archived topology")
        env = np.setdiff1d(np.arange(len(original.atoms)), lig_indices)
        if len(original.trajectory) <= max(FRAMES):
            raise AssertionError("Dataset too short for original 650/250 split")
        ref = memory_subset(topo, trajectory, [REFERENCE_FRAME])
        driver.fix_universe_elements(ref)
        # Important: matching (including source's ligand centering/wrapping) is
        # applied inside match_waters, not to the saved unperturbed environment.
        scenarios = []
        coords = []
        cells = []
        for f in FRAMES:
            original.trajectory[int(f)]
            snapshot = np.asarray(original.atoms.positions, dtype=np.float64).copy()
            cell = np.asarray(original.dimensions).copy()
            if not np.isfinite(snapshot).all() or not np.isfinite(cell).all():
                raise AssertionError("Non-finite archived frame")
            for label, operation, value in TRIALS:
                xyz = perturb(snapshot, lig_indices, operation, value)
                if not np.array_equal(xyz[env], snapshot[env]):
                    raise AssertionError("A ligand-only perturbation moved the environment")
                coords.append(xyz.astype(np.float32))
                cells.append(cell)
                scenarios.append((f, label))
        # All trials use the same original topology and raw XTC frame representation;
        # one source matcher invocation ensures identical alignment and assignment.
        matched = mda.Universe(str(topo))
        matched.load_new(np.stack(coords), format=MemoryReader,
                         dimensions=np.stack(cells))
        driver.fix_universe_elements(matched)
        matcher = adapted_match(driver, OUT)
        ref_water, matched_rows, rmsds, _ = matcher(
            "", ref, 0, SCAFFOLD, "resname SOL", CUTOFF,
            np.asarray([], dtype=int), matched_universe=matched)
        if len(matched_rows) != len(scenarios):
            raise AssertionError("Matcher did not return all configurations")
        elements = np.asarray([str(ref.atoms[i].element) for i in ref_water])
        oxygen_slots = np.flatnonzero(elements == "O")
        hydrogen_slots = np.flatnonzero(elements == "H")
        if len(oxygen_slots) < 3 or len(oxygen_slots)*2 != len(hydrogen_slots):
            raise AssertionError("Reference water atom roles unexpected")
        origin_ref_atoms = [8299, 487, 6148]  # Successful 650/250 run's three O slots.
        anchor_slots = [int(np.flatnonzero(ref_water == x)[0])
                        for x in origin_ref_atoms if np.count_nonzero(ref_water == x) == 1]
        rows = []
        for k, (frame, label) in enumerate(scenarios):
            base = FRAMES.index(frame)*len(TRIALS)
            old, new = matched_rows[base], matched_rows[k]
            rows.append(dict(
                frame=int(frame), trial=label,
                changed_oxygen_slots=int(np.count_nonzero(old[oxygen_slots] != new[oxygen_slots])),
                changed_hydrogen_slots=int(np.count_nonzero(old[hydrogen_slots] != new[hydrogen_slots])),
                changed_baseline_reference_atoms=int(np.count_nonzero(old[anchor_slots] != new[anchor_slots])),
                matched_water_oxygen_indices=[int(i) for i in new[oxygen_slots]],
                matched_baseline_reference_indices=[int(i) for i in new[anchor_slots]],
                rmsd_angstrom=float(rmsds[k])))
        groups = {}
        for label, _, _ in TRIALS:
            rr = [row for row in rows if row["trial"] == label]
            groups[label] = dict(
                frames=len(rr),
                switched_frames=sum(r["changed_oxygen_slots"] > 0 for r in rr),
                oxygen_slot_changes=sum(r["changed_oxygen_slots"] for r in rr),
                hydrogen_slot_changes=sum(r["changed_hydrogen_slots"] for r in rr),
                reference_atom_switch_frames=sum(r["changed_baseline_reference_atoms"] > 0 for r in rr),
                max_oxygen_changes=max(r["changed_oxygen_slots"] for r in rr))
        if groups["original"]["oxygen_slot_changes"] or groups["original"]["hydrogen_slot_changes"]:
            raise AssertionError("Identity test unexpectedly changed a matched row")
        report.update(status="complete", archived_trajectory=provenance,
                      atom_count=len(original.atoms), ligand_atom_count=len(lig_indices),
                      matched_reference_waters=len(oxygen_slots),
                      baseline_reference_atom_slots=origin_ref_atoms,
                      baseline_reference_slots_located=anchor_slots,
                      frame_trials=rows, summary=groups,
                      interpretation="Only matching under prescribed ligand-only perturbations; neither full map bias nor global map invertibility is measured.")
        dest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("WATER_MATCHING_INVARIANCE_COMPLETE " + json.dumps(groups, sort_keys=True), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        dest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise

if __name__ == "__main__":
    main()
