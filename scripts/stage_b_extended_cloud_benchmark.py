#!/usr/bin/env python3
"""B2.5: one bounded, real-QM/MM, chronological-holdout MixedMAF benchmark.

Reuse B2.4's upstream map and serial GROMACS/tblite potential. This is a
software/scientific integration benchmark, not an unbiased free-energy estimate.
The final test is never used to select model parameters or a checkpoint.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import resource
import sys
import time
import traceback

import MDAnalysis as mda
import numpy as np
import torch

from stage_b_first_flow_training import check_finite, grad_norm
from stage_b_training_interface import SequentialTrainingPotential, TEMPERATURE_K
from stage_b_mechanical_embedding import (
    ROOT, GMX, MechanicalEmbedding, digest, grompp, prepare_files,
    retrieve_topology, run, setup_coordinates,
)
from stage_b_multiframe import (
    check_terminal_alignment, load_trajectory, register_trajectory_transformations,
)
from stage_b_qm_singlepoint import ARCHIVE, GRO_NAME, load_archive

OUT = Path("results/stage-b/extended-cloud-benchmark").resolve()
SEED, LR, EPOCHS, BATCH = 20261011, 1e-4, 3, 2
TRAIN = np.arange(401, 1051, dtype=int)
VALIDATION = np.arange(1151, 1401, dtype=int)
TEST = np.arange(1501, 1801, dtype=int)
CALIBRATION = np.unique(np.linspace(TRAIN[0], TRAIN[-1], 32, dtype=int))
TRAIN_PROBE = np.unique(np.linspace(TRAIN[0], TRAIN[-1], 48, dtype=int))
COLUMNS = [
    "phase", "block", "epoch", "frame_index", "time_ps",
    "reference_mm_kj_mol", "mapped_target_qmmm_kj_mol",
    "log_jacobian", "reduced_map_work",
    "ligand_displacement_rms_angstrom", "ligand_displacement_max_angstrom",
    "fixed_environment_max_angstrom",
]


class BudgetExceeded(RuntimeError):
    """Soft scientific cutoff; results remain unvalidated."""


def record(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state_dict(mapper, keys):
    # PartialFlow creates a large, derived _propagated_indices buffer on first
    # use. Keep exactly the setup-time keys so fresh setup can load strictly.
    state = mapper._flow.state_dict()
    return {key: state[key].detach().cpu().clone() for key in keys}


def capture_checkpoint(mapper, optimizer, keys, epoch, path):
    torch.save({
        "state_dict": state_dict(mapper, keys),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "seed": SEED,
        "calibration_indices": CALIBRATION.tolist(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }, path)


def summaries(rows):
    w = np.asarray([r["reduced_map_work"] for r in rows], dtype=float)
    if len(w) < 2 or not np.isfinite(w).all():
        raise AssertionError("Incomplete or nonfinite evaluation")
    return {
        "n": int(len(w)),
        "mean_reduced_work": float(w.mean()),
        "sample_variance_reduced_work": float(w.var(ddof=1)),
        "q05": float(np.quantile(w, 0.05)),
        "median": float(np.median(w)),
        "q95": float(np.quantile(w, 0.95)),
        "max_abs_log_jacobian": max(abs(r["log_jacobian"]) for r in rows),
        "max_ligand_displacement_angstrom": max(
            r["ligand_displacement_max_angstrom"] for r in rows),
    }


def lag_one_correlation(rows):
    w = np.asarray([r["reduced_map_work"] for r in rows], dtype=float)
    a, b = w[:-1], w[1:]
    if len(w) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def make_map(potential, xtc_path, ligand_idx):
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
            # Critically: only training-era coordinates inform spline ranges.
            return tfep.io.TrajectorySubset(super().create_dataset(), CALIBRATION)

    registry = potential.energy_unit._REGISTRY
    mapper = RealMixedMAF(
        potential_energy_func=potential,
        topology_file_path=str(ROOT / "archived-eq.tpr"),
        coordinates_file_path=str(xtc_path),
        temperature=TEMPERATURE_K * registry.kelvin,
        mapped_atoms=ligand_idx.tolist(),
        batch_size=BATCH, n_maf_layers=1,
        tfep_logger_dir_path=str(OUT / "tfep-logs"),
    )
    mapper.setup("fit")
    if abs(float(mapper._kT) - 0.00831446261815324 * TEMPERATURE_K) > 1e-9:
        raise AssertionError("kT/energy units mismatch")
    return mapper


def check_geometry(mapper, x, fixed, bonds, ligand_idx):
    with torch.no_grad():
        forward = mapper({"positions": x[None, :]})
        y, log_j = forward["positions"], forward["log_det_J"]
        reverse = mapper.inverse({"positions": y})
        check_finite(y, "inverse probe positions")
        check_finite(log_j, "inverse probe Jacobian")
        check_finite(reverse["positions"], "inverse positions")
        check_finite(reverse["log_det_J"], "inverse Jacobian")
        inv_error = float((reverse["positions"] - x[None, :]).abs().max())
        det_error = float((reverse["log_det_J"] + log_j).abs().max())
        fixed_error = float((y - x[None, :]).reshape(1, -1, 3)[:, fixed].abs().max())
        p = y.reshape(-1, 3)
        lengths = torch.linalg.vector_norm(p[bonds[:, 0]] - p[bonds[:, 1]], dim=1)
        bond_min, bond_max = float(lengths.min()), float(lengths.max())
        displacement = (y - x[None, :]).reshape(-1, 3)[ligand_idx]
        max_displacement = float(displacement.abs().max())
    if (inv_error > 1e-5 or det_error > 1e-5 or fixed_error > 1e-10
            or bond_min < 0.6 or bond_max > 2.5 or max_displacement > 2.0):
        raise AssertionError(
            f"Flow inverse/geometry failed: x={inv_error} J={det_error} "
            f"env={fixed_error} bonds={bond_min},{bond_max} "
            f"displacement={max_displacement}")
    return dict(inverse_max_abs_error_angstrom=inv_error,
                inverse_logdet_closure=det_error,
                fixed_environment_error_angstrom=fixed_error,
                min_bond_angstrom=bond_min, max_bond_angstrom=bond_max,
                max_ligand_displacement_angstrom=max_displacement)


def evaluate(mapper, potential, samples, ref, fixed, phase, epoch, csv_writer,
             csv_file, deadline):
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        for number, sample in enumerate(samples, 1):
            if time.perf_counter() >= deadline:
                raise BudgetExceeded(f"Budget during {phase}: {number-1}/{len(samples)}")
            x, cell = sample["positions"], sample["cell"]
            out = mapper({"positions": x[None, :]})
            y, log_j = out["positions"], out["log_det_J"]
            target = potential(y, cell[None, :])[0]
            work = (target - ref[sample["frame_index"]]) / mapper._kT - log_j[0]
            for val, label in ((y, "mapped coordinates"), (log_j, "log J"),
                               (target, "target energy"), (work, "mapped work")):
                check_finite(val, label)
            delta = (y - x[None, :]).reshape(-1, 3)
            fixed_error = float(delta[fixed].abs().max())
            if fixed_error > 1e-10:
                raise AssertionError("Moved fixed solvent/environment")
            ligand = delta[~fixed]
            row = dict(
                phase=phase, block=sample["block"], epoch=epoch,
                frame_index=sample["frame_index"], time_ps=sample["time_ps"],
                reference_mm_kj_mol=float(ref[sample["frame_index"]]),
                mapped_target_qmmm_kj_mol=float(target),
                log_jacobian=float(log_j[0]),
                reduced_map_work=float(work),
                ligand_displacement_rms_angstrom=float(ligand.square().mean().sqrt()),
                ligand_displacement_max_angstrom=float(ligand.abs().max()),
                fixed_environment_max_angstrom=fixed_error)
            csv_writer.writerow(row)
            rows.append(row)
            if number % 50 == 0:
                csv_file.flush()
                print("B25_EVAL_PROGRESS", phase, number, len(samples), flush=True)
    csv_file.flush()
    return rows, time.perf_counter() - start


def train_epoch(mapper, potential, samples, optimizer, deadline, epoch):
    rng = np.random.default_rng(SEED + epoch)
    order = rng.permutation(len(samples))
    params = [p for p in mapper._flow.parameters() if p.requires_grad]
    values, first_grad = [], None
    started = time.perf_counter()
    for offset in range(0, len(order), BATCH):
        if time.perf_counter() >= deadline:
            raise BudgetExceeded(f"Budget during epoch {epoch}: batch {offset//BATCH}")
        batch = [samples[int(k)] for k in order[offset:offset+BATCH]]
        if len(batch) != BATCH:
            raise AssertionError("Expected fixed-size batch")
        x = torch.stack([s["positions"] for s in batch])
        cells = torch.stack([s["cell"] for s in batch])
        optimizer.zero_grad(set_to_none=True)
        mapped = mapper({"positions": x})
        y, log_j = mapped["positions"], mapped["log_det_J"]
        target = potential(y, cells)
        physical, jacobian = (target / mapper._kT).mean(), -log_j.mean()
        loss = physical + jacobian
        for val, label in ((y, "training map"), (target, "training energy"),
                           (log_j, "training log-J"), (loss, "training objective")):
            check_finite(val, label)
        if offset == 0:
            # Independent physical and Jacobian contributions, not merely a
            # nonzero overall gradient that might be Jacobian-only.
            eg = torch.autograd.grad(physical, params, retain_graph=True,
                                     allow_unused=True)
            jg = torch.autograd.grad(jacobian, params, retain_graph=True,
                                     allow_unused=True)
        loss.backward()
        grads = [p.grad for p in params]
        if not all(bool(torch.isfinite(g).all()) for g in grads if g is not None):
            raise FloatingPointError("Nonfinite optimizer gradient")
        if offset == 0:
            closure = max(
                float((g - (e if e is not None else torch.zeros_like(g))
                       - (j if j is not None else torch.zeros_like(g))).abs().max())
                for g, e, j in zip(grads, eg, jg) if g is not None)
            first_grad = {
                "physical_gradient_l2": grad_norm(eg),
                "jacobian_gradient_l2": grad_norm(jg),
                "combined_gradient_l2": grad_norm(grads),
                "closure_max_abs": closure,
            }
            if (min(first_grad[k] for k in ("physical_gradient_l2",
                    "jacobian_gradient_l2", "combined_gradient_l2")) <= 0
                    or closure > 1e-7):
                raise AssertionError("Physical/Jacobian gradient decomposition failed")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError("Nonfinite model parameter")
        values.append(float(loss.detach()))
        if len(values) % 50 == 0:
            print("B25_TRAIN_PROGRESS", epoch, len(values), len(order)//BATCH,
                  "last_objective", values[-1], flush=True)
    return dict(epoch=epoch, steps=len(values), mean_online_objective=float(np.mean(values)),
                gradient_audit=first_grad, seconds=time.perf_counter()-started)


def benchmark(max_minutes, preflight=False):
    global OUT
    if preflight:
        OUT = OUT / "preflight"

    started = time.perf_counter()
    deadline = started + 60 * max_minutes
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    arrays = ({"train": np.array([401, 582, 764, 1050]),
               "validation": np.array([1151, 1400]),
               "test": np.array([1501, 1800])}
              if preflight else
              {"train": TRAIN, "validation": VALIDATION, "test": TEST})
    all_indices = np.concatenate(list(arrays.values()))
    if len(set(all_indices)) != len(all_indices) or len(set(CALIBRATION)-set(TRAIN)):
        raise AssertionError("Overlapping partitions or calibration leakage")
    result = dict(
        checkpoint="B2.5 bounded real-QM/MM MixedMAF",
        validated=False, preflight=preflight, stage="initializing",
        seed=SEED, archive=ARCHIVE, max_scientific_minutes=max_minutes,
        index_partitions={key: values.tolist() for key, values in arrays.items()},
        calibration_frame_indices=CALIBRATION.tolist(),
        training_probe_frame_indices=TRAIN_PROBE.tolist(),
        optimizer={"name": "AdamW", "lr": LR, "weight_decay": 0, "epochs": (1 if preflight else EPOCHS),
                   "batch_size": BATCH, "layers": 1},
        limitations=("One archived repeat and quantized XTC; no unbiased free-energy "
                     "or ESS interpretation; chronological gaps do not guarantee "
                     "independence; one-layer ligand-only model."),
        provenance={
            "git_sha": os.environ.get("GITHUB_SHA", "local"),
            "tfep_commit": "0bc369a1194c688bb6ac997cc6b7c2db2b063c9e",
            "paper_commit": "b8dfd897ebd914b4e592221710b397a2758353ce",
            "gromacs_commit": "5e5ea27d8e594124d229a726616b7f6473f2c935",
            "bgflow_commit": "fbba56fac3eb88f6825d2bd4f745ee75ae9715e1",
        }, history=[])
    destination = OUT / "benchmark.json"
    record(destination, result)
    try:
        raw = load_archive()
        topology = retrieve_topology()
        upstream = prepare_files()
        numbers, charge, xyz, ligand_idx, gro_box, atom_audit = setup_coordinates(raw)
        if charge != 0:
            raise AssertionError("Expected neutral ligand")
        xtc, xtc_info = load_trajectory()
        version = run(GMX, "--version")
        if not re.search(r"^\s*Precision:\s+double\b", version, re.M | re.I):
            raise AssertionError("Double-precision GROMACS required")
        grompp()
        universe = mda.Universe(str(ROOT / "archived-eq.tpr"), str(xtc))
        register_trajectory_transformations(universe, ligand_idx)
        alignment, _, _ = check_terminal_alignment(
            universe, xyz, ligand_idx, gro_box, raw[GRO_NAME].decode().splitlines()[0])
        if (alignment["after_translation"]["rms_displacement_angstrom"] > 0.001
                or alignment["after_translation"]["max_displacement_angstrom"] > 0.02):
            raise AssertionError("Trajectory alignment failed")
        if len(universe.trajectory) <= int(TEST[-1]):
            raise AssertionError("Insufficient deposited trajectory frames")
        for label, indices in arrays.items():
            if not preflight and np.any(np.diff(indices) != 1):
                raise AssertionError(f"Noncontiguous {label} index block")
        if np.min(VALIDATION) - np.max(TRAIN) - 1 != 100:
            raise AssertionError("Training/validation gap not 100 frames")
        if np.min(TEST) - np.max(VALIDATION) - 1 != 100:
            raise AssertionError("Validation/test gap not 100 frames")
        result["provenance"].update(
            archive_member_sha256={key: digest(value) for key, value in raw.items()},
            archive_topology=topology, upstream=upstream, trajectory=xtc_info,
            atom_audit=atom_audit, gromacs_version=version,
            tpr_sha256={key: digest((ROOT / (key+".tpr")).read_bytes())
                        for key in ("full", "ligand")},
            package_versions={name: importlib.metadata.version(name)
                              for name in ("torch", "numpy", "MDAnalysis",
                                           "tblite", "tfep", "pint")})
        embedding = MechanicalEmbedding(ligand_idx, numbers, gro_box)
        potential = SequentialTrainingPotential(embedding)
        mapper = make_map(potential, xtc, ligand_idx)
        setup_keys = tuple(mapper._flow.state_dict())
        params = [p for p in mapper._flow.parameters() if p.requires_grad]
        first_params = [p.detach().clone() for p in params]
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0)
        result["kT_kj_mol"] = float(mapper._kT)
        result["n_atoms"] = int(len(xyz))
        result["n_ligand_atoms"] = int(len(ligand_idx))
        fixed = torch.ones(len(xyz), dtype=torch.bool)
        fixed[torch.tensor(ligand_idx, dtype=torch.long)] = False
        bonds = torch.tensor(universe.select_atoms("resname MOL").bonds.indices,
                             dtype=torch.long)
        if len(bonds) == 0:
            raise AssertionError("No ligand bonds")
        samples, prev_time = {}, -float("inf")
        for block, indices in arrays.items():
            samples[block] = []
            for index in indices:
                ts = universe.trajectory[int(index)]
                frame_time = float(ts.time)
                if not np.isfinite(frame_time) or frame_time <= prev_time:
                    raise AssertionError("Unexpected archive frame times")
                prev_time = frame_time
                x = torch.tensor(np.asarray(universe.atoms.positions,
                                 dtype=np.float64).copy().reshape(-1))
                cell = torch.tensor(np.asarray(ts.dimensions,
                                    dtype=np.float64).copy())
                check_finite(x, "selected XTC"); check_finite(cell, "selected cell")
                samples[block].append(dict(
                    frame_index=int(index), block=block, positions=x,
                    cell=cell, time_ps=frame_time))
        # Spot-check upstream setup coordinate/box replay on a calibration frame.
        ds = mapper.dataset[0]
        first_sample = samples["train"][int(CALIBRATION[0]-TRAIN[0])]
        x_err = float((ds["positions"].to(torch.float64) -
                       first_sample["positions"]).abs().max())
        cell_err = float((ds["dimensions"].to(torch.float64) -
                          first_sample["cell"]).abs().max())
        if x_err > 1e-5 or cell_err > 1e-4:
            raise AssertionError("Setup/sample replay differs")
        result["input_replay_max_error"] = dict(position=x_err, cell=cell_err)
        result["frame_spacing_ps"] = dict(
            train=float(np.median(np.diff([s["time_ps"] for s in samples["train"]]))),
            validation=float(np.median(np.diff([s["time_ps"] for s in samples["validation"]]))),
            test=float(np.median(np.diff([s["time_ps"] for s in samples["test"]]))))
        if not preflight and any(abs(t-5.0) > 0.1 for t in result["frame_spacing_ps"].values()):
            raise AssertionError("Expected 5-ps archive frame spacing")
        record(destination, result)

        ref = {}
        reference_started = time.perf_counter()
        for block in ("train", "validation", "test"):
            for n, s in enumerate(samples[block], 1):
                if time.perf_counter() >= deadline:
                    raise BudgetExceeded(f"Budget during {block} reference: {n}")
                with torch.no_grad():
                    energy = potential.reference(
                        s["positions"][None, :], s["cell"][None, :])[0]
                check_finite(energy, "same-input MM reference")
                ref[s["frame_index"]] = energy.detach().clone()
                if n % 100 == 0:
                    print("B25_REFERENCE_PROGRESS", block, n, flush=True)
        result["reference_completed_frames"] = len(ref)
        result["reference_seconds"] = time.perf_counter()-reference_started
        result["stage"] = "training_and_validation"
        record(destination, result)

        with (OUT / "frame-metrics.csv").open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
            writer.writeheader()
            # Identity/initial-map test is measured once, never used for selection.
            initial_test, _ = evaluate(
                mapper, potential, samples["test"], ref, fixed, "initial_test",
                0, writer, csv_file, deadline)
            result["initial_test"] = summaries(initial_test)
            result["test_initial_lag1_correlation"] = lag_one_correlation(initial_test)
            result["test_initial_is_selection_blind"] = True
            # Preflight inverse after initial and after actual parameter updates.
            probe = samples["train"][len(samples["train"])//2]["positions"]
            result["initial_inverse_geometry"] = check_geometry(
                mapper, probe, fixed, bonds, ligand_idx)
            probe_samples = (samples["train"] if preflight else
                             [samples["train"][int(i - TRAIN[0])] for i in TRAIN_PROBE])
            initial_train, _ = evaluate(
                mapper, potential, probe_samples, ref, fixed,
                "initial_train_probe", 0, writer, csv_file, deadline)
            initial_validation, _ = evaluate(
                mapper, potential, samples["validation"], ref, fixed,
                "initial_validation", 0, writer, csv_file, deadline)
            result["initial_train_probe"] = summaries(initial_train)
            result["initial_validation"] = summaries(initial_validation)
            best_variance = result["initial_validation"]["sample_variance_reduced_work"]
            selected_epoch = 0
            capture_checkpoint(mapper, optimizer, setup_keys, 0, OUT / "selected.pt")
            record(destination, result)
            for epoch in range(1, (1 if preflight else EPOCHS)+1):
                epoch_info = train_epoch(
                    mapper, potential, samples["train"], optimizer, deadline, epoch)
                epoch_info["inverse_geometry"] = check_geometry(
                    mapper, probe, fixed, bonds, ligand_idx)
                probe_rows, _ = evaluate(
                    mapper, potential, probe_samples, ref, fixed,
                    "train_probe", epoch, writer, csv_file, deadline)
                val_rows, _ = evaluate(
                    mapper, potential, samples["validation"], ref, fixed,
                    "validation", epoch, writer, csv_file, deadline)
                epoch_info["train_probe"] = summaries(probe_rows)
                epoch_info["validation"] = summaries(val_rows)
                variance = epoch_info["validation"]["sample_variance_reduced_work"]
                if variance < best_variance:
                    best_variance, selected_epoch = variance, epoch
                    capture_checkpoint(mapper, optimizer, setup_keys,
                                       epoch, OUT / "selected.pt")
                capture_checkpoint(mapper, optimizer, setup_keys,
                                   epoch, OUT / "latest.pt")
                epoch_info["elapsed_seconds"] = time.perf_counter()-started
                result["history"].append(epoch_info)
                result["selected_epoch"] = selected_epoch
                record(destination, result)
                print("B25_EPOCH", json.dumps(epoch_info, allow_nan=False), flush=True)

            result["max_parameter_change"] = max(
                float((p.detach()-initial).abs().max())
                for p, initial in zip(params, first_params))
            if result["max_parameter_change"] <= 0:
                raise AssertionError("Optimizer did not change model parameters")
            # Reconstruct model independently of optimizer and reload the saved
            # selected flow. Exact keys (excluding derived cache) are required.
            chosen = torch.load(OUT / "selected.pt", map_location="cpu",
                                weights_only=False)
            fresh = make_map(potential, xtc, ligand_idx)
            fresh._flow.load_state_dict(chosen["state_dict"], strict=True)
            for name, s in (("train", samples["train"][0]),
                            ("test", samples["test"][0])):
                forward = fresh({"positions": s["positions"][None, :]})
                previous = mapper._flow.state_dict()
                mapper._flow.load_state_dict(chosen["state_dict"], strict=False)
                other = mapper({"positions": s["positions"][None, :]})
                coord_error = float((forward["positions"]-other["positions"]).abs().max())
                log_error = float((forward["log_det_J"]-other["log_det_J"]).abs().max())
                if coord_error > 1e-8 or log_error > 1e-8:
                    raise AssertionError("Fresh checkpoint reconstruction mismatch")
                result.setdefault("checkpoint_reload", {})[name] = dict(
                    positions_max_error_angstrom=coord_error,
                    logdet_error=log_error)
            result["selected_epoch"] = selected_epoch
            result["selected_inverse_geometry"] = check_geometry(
                fresh, probe, fixed, bonds, ligand_idx)
            result["stage"] = "locked_final_test"
            record(destination, result)
            final_test, _ = evaluate(
                fresh, potential, samples["test"], ref, fixed,
                "selected_final_test", selected_epoch,
                writer, csv_file, deadline)
            result["final_test"] = summaries(final_test)
            result["test_final_lag1_correlation"] = lag_one_correlation(final_test)
            v0 = result["initial_test"]["sample_variance_reduced_work"]
            v1 = result["final_test"]["sample_variance_reduced_work"]
            result["test_variance_change"] = v1-v0
            result["test_variance_reduction_percent"] = 100.0*(v0-v1)/v0
            result["test_mean_change"] = (
                result["final_test"]["mean_reduced_work"]
                - result["initial_test"]["mean_reduced_work"])
            result["stage"] = "complete"
            result["runtime_seconds"] = time.perf_counter()-started
            result["python_peak_rss_mib"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            result["validated"] = not preflight
            result["preflight_validated"] = bool(preflight)
            record(destination, result)
            print("B25_PREFLIGHT_PASSED" if preflight else "B25_VALIDATED", json.dumps({
                "selected_epoch": selected_epoch,
                "test_before": result["initial_test"],
                "test_after": result["final_test"],
                "variance_change": result["test_variance_change"],
                "elapsed_seconds": result["runtime_seconds"]}), flush=True)
    except Exception as exc:
        result["stage"] = "incomplete"
        result["validated"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["runtime_seconds"] = time.perf_counter()-started
        result["python_peak_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        record(destination, result)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scientific-minutes", type=float, default=165)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.scientific_minutes <= 165:
        parser.error("scientific budget must lie in [1, 165] minutes")
    benchmark(args.scientific_minutes, preflight=args.preflight)
