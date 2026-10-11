#!/usr/bin/env python3
"""B2.4: genuine three-step original MixedMAF training on archived real QM/MM."""
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import resource
import time

import MDAnalysis as mda
import numpy as np
import torch

from stage_b_training_interface import SequentialTrainingPotential, TEMPERATURE_K
from stage_b_mechanical_embedding import (
    ROOT, GMX, MechanicalEmbedding, digest, grompp, prepare_files,
    retrieve_topology, run, setup_coordinates)
from stage_b_multiframe import (
    check_terminal_alignment, load_trajectory, register_trajectory_transformations)
from stage_b_qm_singlepoint import ARCHIVE, GRO_NAME, load_archive

OUT = Path("results/stage-b/first-flow-training").resolve()
SEED, LR, STEPS = 20261011, 1e-4, 3
SETUP_FRAMES = np.array([401, 582, 764, 946, 1128, 1310, 1492, 1674,
                         1855, 1930, 1980, 2000], dtype=int)
TRAIN_ROWS, HELDOUT_FRAMES = (0, 1, 2), (2218, 2400)
STEP_ROWS = ((0, 1), (1, 2), (0, 2))


def save(result):
    (OUT / "first-flow-training.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")


def check_finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Nonfinite " + label)


def grad_norm(grads):
    return float(sum(float(g.detach().square().sum()) for g in grads
                     if g is not None)**0.5)


def evaluate(mapper, potential, samples, reference, fixed, phase):
    """Evaluate mapped target on y and cached FF reference on identical XTC x."""
    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for s in samples:
            x, cell = s["positions"], s["cell"]
            output = mapper({"positions": x[None, :]})
            y, log_j = output["positions"], output["log_det_J"]
            target = potential(y, cell[None, :])[0]
            work = (target - reference[s["row"]])/mapper._kT - log_j[0]
            for val, label in ((y, "mapped positions"), (log_j, "log J"),
                               (target, "target energy"), (work, "mapped work")):
                check_finite(val, label)
            displacement = (y-x[None, :]).reshape(1, -1, 3)
            fixed_error = float(displacement[:, fixed].abs().max())
            if fixed_error > 1e-10:
                raise AssertionError("Flow moved fixed water/environment")
            ligand = displacement[:, ~fixed]
            rows.append({
                "phase": phase, "block": s["block"],
                "frame_index": s["frame_index"], "time_ps": s["time_ps"],
                "reference_mm_kj_mol": float(reference[s["row"]]),
                "mapped_target_qmmm_kj_mol": float(target),
                "log_jacobian": float(log_j[0]),
                "reduced_map_work": float(work),
                "ligand_displacement_rms_angstrom": float(ligand.square().mean().sqrt()),
                "ligand_displacement_max_angstrom": float(ligand.abs().max()),
                "fixed_environment_max_angstrom": fixed_error,
            })
    return rows, time.perf_counter()-started


def stats(rows):
    output = {}
    for block in ("train", "heldout"):
        selected = [r for r in rows if r["block"] == block]
        w = np.array([r["reduced_map_work"] for r in selected])
        output[block] = {
            "n": len(w), "mean_reduced_work": float(w.mean()),
            "sample_variance_reduced_work": float(w.var(ddof=1)),
            "max_abs_log_jacobian": max(abs(r["log_jacobian"]) for r in selected),
            "max_ligand_displacement_angstrom": max(
                r["ligand_displacement_max_angstrom"] for r in selected),
        }
    return output


def main():
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)

    result = {
        "checkpoint": "B2.4 real MixedMAF molecular optimization", "validated": False,
        "archive": ARCHIVE, "seed": SEED,
        "architecture": "pinned original MixedMAFMap; one MAF layer; ligand mapped; fixed water",
        "optimizer": {"name": "AdamW", "learning_rate": LR,
                      "weight_decay": 0, "steps": STEPS, "batch_size": 2},
        "setup_frame_indices": SETUP_FRAMES.tolist(),
        "training_frame_indices": [int(SETUP_FRAMES[i]) for i in TRAIN_ROWS],
        "heldout_frame_indices": list(HELDOUT_FRAMES),
        "step_frame_indices": [
            [int(SETUP_FRAMES[i]) for i in pair] for pair in STEP_ROWS],
        "warning": (
            "Quantized XTC measure differs from original high-precision MD coordinates; "
            "re-evaluated FF on XTC is not proven to match exact sampled reference. "
            "Only one repeat, temporally correlated frames, one-layer pilot: "
            "no free-energy/ESS inference or publication-level claim."),
        "provenance": {
            "git_sha": os.environ.get("GITHUB_SHA", "local"),
            "upstream_tfep_commit": "0bc369a1194c688bb6ac997cc6b7c2db2b063c9e",
            "upstream_paper_commit": "b8dfd897ebd914b4e592221710b397a2758353ce",
            "bgflow_commit": "fbba56fac3eb88f6825d2bd4f745ee75ae9715e1",
            "gromacs_commit": "5e5ea27d8e594124d229a726616b7f6473f2c935",
            "units": {"position": "angstrom", "energy": "kJ/mol",
                      "gradient": "kJ/mol/angstrom", "reduced_work": "dimensionless"},
        }
    }
    save(result)
    raw = load_archive()
    topology = retrieve_topology()
    upstream = prepare_files()
    numbers, charge, gro_xyz, ligand_idx, gro_box, atom_audit = setup_coordinates(raw)
    if charge != 0:
        raise AssertionError("Expected neutral ligand")
    xtc_path, xtc_info = load_trajectory()
    version = run(GMX, "--version")
    if not re.search(r"^\s*Precision:\s+double\b", version, re.M | re.I):
        raise RuntimeError("Requires double-precision GROMACS")
    grompp()

    universe = mda.Universe(str(ROOT / "archived-eq.tpr"), str(xtc_path))
    register_trajectory_transformations(universe, ligand_idx)
    alignment, _, _ = check_terminal_alignment(
        universe, gro_xyz, ligand_idx, gro_box, raw[GRO_NAME].decode().splitlines()[0])
    if (alignment["after_translation"]["rms_displacement_angstrom"] > 0.001
            or alignment["after_translation"]["max_displacement_angstrom"] > 0.02):
        raise AssertionError("Archived XTC/GRO mismatch")
    embedding = MechanicalEmbedding(ligand_idx, numbers, gro_box)
    potential = SequentialTrainingPotential(embedding)
    import tfep.app.mixedmaf
    import tfep.io

    class RealMixedMAF(tfep.app.mixedmaf.MixedMAFMap):
        def create_universe(self):
            u = mda.Universe(str(ROOT / "archived-eq.tpr"), str(xtc_path))
            register_trajectory_transformations(u, ligand_idx)
            special_types = {"amber99_338": "Na", "amber99_333": "Cl"}
            elements = []
            for atom in u.atoms:
                atom_type = str(atom.type)
                if "DUM" in atom_type:
                    raise ValueError("Unexpected dummy atom")
                elements.append(special_types.get(atom_type, atom_type[0].upper()))
            if hasattr(u.atoms, "elements"):
                u.del_TopologyAttr("element")
            u.add_TopologyAttr("element", elements)
            return u

        def create_dataset(self):
            return tfep.io.TrajectorySubset(super().create_dataset(), SETUP_FRAMES)

    registry = potential.energy_unit._REGISTRY
    mapper = RealMixedMAF(
        potential_energy_func=potential,
        topology_file_path=str(ROOT / "archived-eq.tpr"),
        coordinates_file_path=str(xtc_path),
        temperature=TEMPERATURE_K*registry.kelvin,
        mapped_atoms=ligand_idx.tolist(), batch_size=2, n_maf_layers=1,
        tfep_logger_dir_path=str(OUT / "tfep-logs"))
    mapper.setup("fit")
    if abs(float(mapper._kT)-0.00831446261815324*TEMPERATURE_K) > 1e-9:
        raise AssertionError("Unexpected kT/energy units")

    # Only the early training-era frames calibrate Z-matrix/spline ranges.
    # Holdout is read separately, never through the mapper's setup dataset.
    samples, position_error, box_error = [], 0.0, 0.0
    for row in TRAIN_ROWS:
        item = mapper.dataset[row]
        x = item["positions"].to(torch.float64).detach()
        cell = item["dimensions"].to(torch.float64).detach()
        ts = universe.trajectory[int(SETUP_FRAMES[row])]
        independent_x = torch.as_tensor(
            np.asarray(universe.atoms.positions, dtype=np.float64).reshape(-1))
        independent_box = torch.as_tensor(
            np.asarray(ts.dimensions, dtype=np.float64))
        position_error = max(position_error, float((x-independent_x).abs().max()))
        box_error = max(box_error, float((cell-independent_box).abs().max()))
        check_finite(x, "training XTC"); check_finite(cell, "training box")
        samples.append({
            "row": row, "positions": x, "cell": cell,
            "frame_index": int(SETUP_FRAMES[row]), "time_ps": float(ts.time),
            "block": "train",
        })
    for frame_index in HELDOUT_FRAMES:
        # Same upstream B2.1 unwrap/center/wrap transforms; no holdout
        # coordinates inform flow initialization or optimizer gradients.
        ts = universe.trajectory[frame_index]
        x = torch.tensor(np.asarray(universe.atoms.positions, dtype=np.float64).copy()
                         .reshape(-1), dtype=torch.float64)
        cell = torch.tensor(np.asarray(ts.dimensions, dtype=np.float64).copy(),
                            dtype=torch.float64)
        check_finite(x, "held-out XTC"); check_finite(cell, "held-out box")
        samples.append({
            "row": frame_index, "positions": x, "cell": cell,
            "frame_index": frame_index, "time_ps": float(ts.time),
            "block": "heldout",
        })
    if position_error > 1e-5 or box_error > 1e-4:
        raise AssertionError("Upstream coordinate/cell replay mismatch")
    if max(s["time_ps"] for s in samples if s["block"] == "train") >= min(
        s["time_ps"] for s in samples if s["block"] == "heldout"
    ):
        raise AssertionError("Training/held-out time blocks overlap")
    np.savez_compressed(
        OUT / "selected-inputs.npz",
        positions=np.stack([s["positions"].numpy() for s in samples]),
        cells=np.stack([s["cell"].numpy() for s in samples]),
        frame_indices=np.array([s["frame_index"] for s in samples]),
        times_ps=np.array([s["time_ps"] for s in samples]))
    result["provenance"].update({
        "archive_member_sha256": {k: digest(v) for k, v in raw.items()},
        "archive_topology": topology, "upstream": upstream, "trajectory": xtc_info,
        "atom_audit": atom_audit, "gromacs_version": version,
        "generated_tpr_sha256": {
            key: digest((ROOT/(key+".tpr")).read_bytes()) for key in ("full", "ligand")},
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "numpy", "MDAnalysis",
                                          "tblite", "tfep", "pint")}})
    result["frame_times_ps"] = {str(s["frame_index"]): s["time_ps"] for s in samples}
    result["frame_boxes_angstrom_degrees"] = {
        str(s["frame_index"]): s["cell"].tolist() for s in samples}
    result["atom_count"] = len(gro_xyz)
    result["ligand_atom_count"] = len(ligand_idx)
    result["kT_kj_mol"] = float(mapper._kT)
    result["coordinate_replay_error_angstrom"] = position_error
    result["cell_replay_error"] = box_error
    save(result)

    reference = {}
    reference_started = time.perf_counter()
    with torch.no_grad():
        for s in samples:
            energy = potential.reference(s["positions"][None, :], s["cell"][None, :])[0]
            check_finite(energy, "reference FF")
            reference[s["row"]] = energy.detach().clone()
    result["reference_seconds"] = time.perf_counter()-reference_started
    result["reference_ff_kj_mol"] = {
        str(s["frame_index"]): float(reference[s["row"]]) for s in samples}
    save(result)

    params = [p for p in mapper._flow.parameters() if p.requires_grad]
    if not params:
        raise AssertionError("No flow parameters")
    initial = [p.detach().clone() for p in params]
    initial_flow_state = mapper._flow.state_dict()
    initial_state_keys = tuple(initial_flow_state)
    torch.save({"state_dict": initial_flow_state, "seed": SEED},
               OUT / "flow-before.pt")
    fixed = torch.ones(len(gro_xyz), dtype=torch.bool)
    fixed[torch.as_tensor(ligand_idx, dtype=torch.long)] = False
    before_rows, before_seconds = evaluate(
        mapper, potential, samples, reference, fixed, "before")
    result["before"], result["before_seconds"] = stats(before_rows), before_seconds
    save(result)
    print("B24_BEFORE", json.dumps(result["before"]), flush=True)

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    step_results = []
    for n, pair in enumerate(STEP_ROWS, 1):
        chosen = [next(s for s in samples if s["row"] == row) for row in pair]
        x = torch.stack([s["positions"] for s in chosen])
        cell = torch.stack([s["cell"] for s in chosen])
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        mapped_result = mapper({"positions": x})
        y, log_j = mapped_result["positions"], mapped_result["log_det_J"]
        target = potential(y, cell)
        physical, jacobian = (target/mapper._kT).mean(), -log_j.mean()
        loss = physical + jacobian
        for val, label in ((y, "mapped positions"), (log_j, "log-J"),
                           (target, "target QM/MM"), (loss, "loss")):
            check_finite(val, label)
        energy_grads = torch.autograd.grad(
            physical, params, retain_graph=True, allow_unused=True)
        jacobian_grads = torch.autograd.grad(
            jacobian, params, retain_graph=True, allow_unused=True)
        loss.backward()
        grads = [p.grad for p in params]
        for grad in (*energy_grads, *jacobian_grads, *grads):
            if grad is not None:
                check_finite(grad, "parameter gradient")
        closure = max(
            float((total - (e if e is not None else torch.zeros_like(total))
                   - (j if j is not None else torch.zeros_like(total))).abs().max())
            for total, e, j in zip(grads, energy_grads, jacobian_grads)
            if total is not None)
        e_norm, j_norm, norm = (grad_norm(a) for a in
                               (energy_grads, jacobian_grads, grads))
        if e_norm <= 0 or j_norm <= 0 or norm <= 0 or closure > 1e-7:
            raise AssertionError("Physical/Jacobian gradient failed")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError("Nonfinite updated parameter")
        changed = max(float((p.detach()-init).abs().max())
                      for p, init in zip(params, initial))
        info = {
            "step": n, "frame_indices": [s["frame_index"] for s in chosen],
            "target_over_kT": float(physical.detach()),
            "minus_mean_log_j": float(jacobian.detach()),
            "training_objective": float(loss.detach()),
            "physical_gradient_l2": e_norm, "jacobian_gradient_l2": j_norm,
            "combined_gradient_l2": norm, "gradient_closure_max_abs": closure,
            "nonzero_physical_gradient_tensors": sum(
                int(g is not None and bool(torch.count_nonzero(g))) for g in energy_grads),
            "max_parameter_change_from_initial": changed,
            "seconds": time.perf_counter()-step_start,
        }
        step_results.append(info)
        result["steps"] = step_results
        save(result)
        print("B24_STEP", json.dumps(info), flush=True)

    after_rows, after_seconds = evaluate(
        mapper, potential, samples, reference, fixed, "after")
    result["after"], result["after_seconds"] = stats(after_rows), after_seconds
    result["heldout_mean_work_change"] = (
        result["after"]["heldout"]["mean_reduced_work"]
        - result["before"]["heldout"]["mean_reduced_work"])
    result["heldout_work_variance_change"] = (
        result["after"]["heldout"]["sample_variance_reduced_work"]
        - result["before"]["heldout"]["sample_variance_reduced_work"])
    result["max_parameter_change"] = max(
        float((p.detach()-init).abs().max()) for p, init in zip(params, initial))
    result["runtime_seconds"] = time.perf_counter()-started
    result["python_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    with (OUT / "frame-metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=before_rows[0].keys())
        writer.writeheader()
        writer.writerows(before_rows+after_rows)
    # PartialFlow populates a large deterministic index cache on first forward.
    # Omit this derived buffer to keep both checkpoints strictly reloadable
    # into freshly initialized upstream MixedMAF flows.
    updated_flow_state = mapper._flow.state_dict()
    portable_state = {key: updated_flow_state[key] for key in initial_state_keys}
    torch.save({"state_dict": portable_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "seed": SEED, "setup_frames": SETUP_FRAMES.tolist(),
                "train_frames": result["training_frame_indices"],
                "heldout_frames": result["heldout_frame_indices"]},
               OUT / "flow-after.pt")
    save(result)
    if (result["max_parameter_change"] <= 0
            or result["after"]["train"]["max_ligand_displacement_angstrom"] <= 1e-10
            or result["after"]["train"]["max_abs_log_jacobian"] <= 1e-9):
        raise AssertionError("No nonidentity mapped result after optimizer")
    result["validated"] = True
    save(result)
    print("B24_REAL_TRAINING_VALIDATED", json.dumps({
        "steps": STEPS, "parameter_change": result["max_parameter_change"],
        "before": result["before"], "after": result["after"],
        "heldout_mean_change": result["heldout_mean_work_change"],
        "heldout_variance_change": result["heldout_work_variance_change"],
        "seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
