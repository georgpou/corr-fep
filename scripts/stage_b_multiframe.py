#!/usr/bin/env python3
"""Stage B2.1: real QM/MM energies and gradients on separated archived water frames.

The calculation is deliberately sequential (batch size one) and reuses the
validated Stage B1 mechanical-embedding Hamiltonian and physical inputs.
A trajectory excerpt is evidence of feasibility, not independent sampling.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import MDAnalysis as mda
from MDAnalysis import transformations as trans
import numpy as np
import requests
from remotezip import RemoteZip
import torch

from stage_b_mechanical_embedding import (
    OUT as B1_OUT, ROOT, GMX, MechanicalEmbedding, digest, grompp,
    prepare_files, retrieve_topology, run, setup_coordinates,
)
from stage_b_qm_singlepoint import (
    ARCHIVE, PREFIX, load_archive, BOHR_PER_ANGSTROM,
    HARTREE_KJMOL, direct_singlepoint,
)

OUT = Path("results/stage-b/multiframe").resolve()
XTC_NAME = PREFIX + "eq/water/repeat-0/traj_comp.xtc"
# The deposited water XTC is ~73 MB. Read that ZIP member only, never the archive.
MAX_XTC_BYTES = 160_000_000
SAMPLE_COUNT = 4
PRODUCTION_START = 401  # Published training removes first 2 ns (frames 0..400).
FD_STEP_ANGSTROM = 0.002
FD_TOLERANCE = 0.8  # B1's double-precision complete-Hamiltonian tolerance


def load_trajectory():
    """Fetch one named ZIP member over byte ranges, refusing oversized inputs."""
    with requests.Session() as session:
        with session.get(ARCHIVE, headers={"Range": "bytes=-32"},
                         stream=True, timeout=90) as response:
            if response.status_code != 206 or "Content-Range" not in response.headers:
                raise RuntimeError("Zenodo ignored Range; refusing 49 GB archive download")
        with RemoteZip(ARCHIVE, session=session,
                       initial_buffer_size=512 * 1024, timeout=180) as archive:
            info = archive.getinfo(XTC_NAME)
            if info.file_size > MAX_XTC_BYTES or info.compress_size > MAX_XTC_BYTES:
                raise RuntimeError(
                    f"Trajectory ZIP member too large for B2 pilot: "
                    f"{info.file_size} uncompressed / {info.compress_size} compressed"
                )
            data = archive.read(XTC_NAME)
    if len(data) != info.file_size:
        raise IOError("Incomplete archived XTC")
    path = ROOT / "traj_comp.xtc"
    path.write_bytes(data)
    return path, {
        "member": XTC_NAME, "sha256": digest(data),
        "size_bytes": len(data), "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
        "retrieval": "ZIP member via HTTP byte ranges, no whole-archive download",
    }


def sampled_frames(universe, ligand_indices, reference_xyz):
    """Yield transformed coordinates and boxes after topology/geometry validation."""
    ligand = universe.select_atoms("resname MOL")
    environment = universe.select_atoms("not resname MOL")
    if not np.array_equal(ligand.indices, ligand_indices):
        raise AssertionError("Trajectory ligand indices differ from B1 topology")
    if len(universe.atoms) != len(reference_xyz):
        raise AssertionError("XTC atom count differs from archived GRO/TPR")
    if len(ligand.fragments) != 1 or len(ligand.bonds) == 0:
        raise AssertionError("Ligand must be one bonded fragment")
    local = {int(i): n for n, i in enumerate(ligand.indices)}
    bonds = np.asarray([[local[int(i)], local[int(j)]]
                        for i, j in ligand.bonds.indices], dtype=int)
    if len(bonds) == 0:
        raise AssertionError("Missing ligand bonds")
    universe.trajectory.add_transformations(
        trans.unwrap(ligand), trans.center_in_box(ligand), trans.wrap(environment))
    n_frames = len(universe.trajectory)
    if n_frames - PRODUCTION_START < SAMPLE_COUNT:
        raise RuntimeError(f"Only {n_frames - PRODUCTION_START} production frames")
    indices = np.linspace(PRODUCTION_START, n_frames - 1, SAMPLE_COUNT, dtype=int)
    if len(np.unique(indices)) != SAMPLE_COUNT:
        raise AssertionError("Frame selection not unique")
    previous_time = -float("inf")
    for index in indices:
        ts = universe.trajectory[int(index)]
        xyz = np.asarray(universe.atoms.positions, dtype=np.float64).copy()
        box = np.asarray(ts.dimensions, dtype=np.float64).copy()
        frame_time = float(ts.time)  # MDAnalysis time convention: ps
        if not np.isfinite(frame_time) or frame_time <= previous_time:
            raise AssertionError("Nonfinite or non-increasing frame times")
        previous_time = frame_time
        if box.shape != (6,) or not np.isfinite(box).all() or np.any(box[:3] <= 0):
            raise AssertionError("Invalid trajectory box")
        if xyz.shape != reference_xyz.shape or not np.isfinite(xyz).all():
            raise AssertionError("Invalid trajectory coordinate array")
        lengths = np.linalg.norm(xyz[ligand_indices[bonds[:, 0]]] -
                                 xyz[ligand_indices[bonds[:, 1]]], axis=1)
        if lengths.min() < 0.65 or lengths.max() > 2.2:
            raise AssertionError(
                f"Frame {index} has broken ligand: bond range {lengths.min()}--{lengths.max()} A"
            )
        yield int(index), n_frames, frame_time, xyz, box, [
            float(lengths.min()), float(lengths.max())
        ]


def check_terminal_alignment(universe, reference_xyz, ligand_indices, reference_box):
    """Independent positional anchor: XTC endpoint versus deposited GRO snapshot.

    This check does not assume identity of the coordinates: distinct or lossy
    snapshots are a recorded observation, not silently treated as identical.
    """
    from MDAnalysis.lib.distances import minimize_vectors
    ts = universe.trajectory[-1]
    terminal = np.asarray(universe.atoms.positions, dtype=np.float64).copy()
    box = np.asarray(ts.dimensions, dtype=np.float64)
    if terminal.shape != reference_xyz.shape:
        raise AssertionError("Terminal XTC and GRO have different atom counts")
    if not np.allclose(box, reference_box, rtol=0, atol=0.01):
        raise AssertionError("Terminal XTC/GRO periodic boxes differ")
    displacement = minimize_vectors(terminal - reference_xyz, box)
    norms = np.linalg.norm(displacement, axis=1)
    ligand_mask = np.zeros(len(norms), dtype=bool)
    ligand_mask[ligand_indices] = True
    results = {
        "reference_assumption": "XTC terminal frame and archived eqout.gro correspond to the same time; positional comparison tests this",
        "terminal_time_ps": float(ts.time),
        "terminal_box_angstrom_degrees": box.tolist(),
        "gro_box_angstrom_degrees": reference_box.tolist(),
        "rms_displacement_angstrom": float(np.sqrt(np.mean(norms**2))),
        "max_displacement_angstrom": float(norms.max()),
        "ligand_rms_displacement_angstrom": float(np.sqrt(np.mean(norms[ligand_mask]**2))),
        "environment_rms_displacement_angstrom": float(np.sqrt(np.mean(norms[~ligand_mask]**2))),
    }
    # A wrong trajectory pairing should not be accepted just because XTC
    # and GRO both have the expected atom count. Tight enough to flag
    # neighboring frames, loose enough for compressed XTC/GRO rounding.
    if results["rms_displacement_angstrom"] > 0.05 or results["max_displacement_angstrom"] > 0.3:
        raise AssertionError("Terminal XTC/GRO positional mismatch: " + json.dumps(results))
    return results


def complete_gradient_fd(model, xyz, cell, grad, ligand_indices):
    """Two independent central differences of the full subtractive Hamiltonian."""
    from MDAnalysis.lib.distances import distance_array
    ligand_atom = int(ligand_indices[0])
    env = np.setdiff1d(np.arange(len(xyz)), ligand_indices)
    shell_distance = distance_array(xyz[env], xyz[ligand_indices], box=cell).min(axis=1)
    shell = env[shell_distance < 4.5]
    if not len(shell):
        raise AssertionError("No solvent/environment atom inside ligand interface shell")
    best_atom, best_axis = np.unravel_index(np.abs(grad[shell]).argmax(), grad[shell].shape)
    probes = [(ligand_atom, int(np.abs(grad[ligand_atom]).argmax()), "ligand"),
              (int(shell[best_atom]), int(best_axis), "environment")]
    checks = []
    with torch.no_grad():
        for atom, axis, region in probes:
            plus, minus = xyz.copy(), xyz.copy()
            plus[atom, axis] += FD_STEP_ANGSTROM
            minus[atom, axis] -= FD_STEP_ANGSTROM
            def energy(positions):
                x = torch.tensor(positions.reshape(1, -1), dtype=torch.float64)
                cell_tensor = torch.tensor(cell.reshape(1, 6), dtype=torch.float64)
                return float(model(x, cell_tensor).item())
            e_plus, e_minus = energy(plus), energy(minus)
            numeric = (e_plus - e_minus) / (2 * FD_STEP_ANGSTROM)
            analytic = float(grad[atom, axis])
            error = abs(numeric - analytic)
            entry = {"region": region, "atom": atom, "axis": axis,
                     "step_angstrom": FD_STEP_ANGSTROM,
                     "numeric_kj_mol_angstrom": numeric,
                     "analytic_kj_mol_angstrom": analytic,
                     "absolute_error_kj_mol_angstrom": error}
            checks.append(entry)
            if not np.isfinite(error) or error > FD_TOLERANCE:
                raise AssertionError("Complete-Hamiltonian finite difference failed: " + json.dumps(entry))
    return checks


def evaluate(model, xyz, cell, ligand_indices, atomic_numbers):
    """One genuine full-MM - ligand-MM + GFN1-xTB evaluation and backward."""
    x = torch.tensor(xyz.reshape(1, -1), dtype=torch.float64, requires_grad=True)
    box = torch.tensor(cell.reshape(1, 6), dtype=torch.float64)
    start = time.perf_counter()
    full, lig, qm = model.components(x, box)
    total = full - lig + qm
    if any(t.shape != (1,) for t in (full, lig, qm, total)):
        raise AssertionError("Expected single-frame scalar potential components")
    full_gradient, = torch.autograd.grad(full.sum(), x, retain_graph=True)
    total.sum().backward()
    grad = x.grad.detach().numpy().reshape(-1, 3).copy()
    env = np.ones(len(xyz), dtype=bool)
    env[ligand_indices] = False
    env_error = float(np.max(np.abs(
        grad[env] - full_gradient.detach().numpy().reshape(-1, 3)[env]
    )))
    values = np.asarray([float(t.detach().item()) for t in
                         (full, lig, qm, total)], dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(grad).all():
        raise FloatingPointError("Nonfinite real QM/MM energy/gradient")
    if abs(values[0] - values[1] + values[2] - values[3]) > 1e-7:
        raise AssertionError("Mechanical-embedding sum identity failed")
    if env_error > 1e-7:
        raise AssertionError("Environmental QM/MM gradient differs from full MM")
    quantum_energy, quantum_grad = direct_singlepoint(
        atomic_numbers, xyz[ligand_indices] * BOHR_PER_ANGSTROM, 0)
    direct_e_error = abs(values[2] - quantum_energy * HARTREE_KJMOL)
    # B1 already verified numerical derivatives; here check an independent
    # tblite call at each distinct frame, including units and atom ordering.
    quantum_x = torch.tensor(xyz.reshape(1, -1),
                             dtype=torch.float64, requires_grad=True)
    qm_tensor = model.qm(
        quantum_x.reshape(1, -1, 3)[:, model.idx, :].reshape(1, -1))
    quantum_g, = torch.autograd.grad(qm_tensor.sum(), quantum_x)
    direct_g_error = float(np.max(np.abs(
        quantum_g.detach().numpy().reshape(-1, 3)[ligand_indices] -
        quantum_grad * HARTREE_KJMOL * BOHR_PER_ANGSTROM
    )))
    if direct_e_error > 0.02 or direct_g_error > 0.03:
        raise AssertionError(
            f"Direct vs wrapped tblite discrepancy: {direct_e_error}, {direct_g_error}"
        )
    return {
        "mm_full": values[0], "mm_ligand": values[1],
        "qm_ligand_gfn1xtb": values[2], "qmmm_combined": values[3],
    }, grad, {
        "environment_gradient_identity_max_kj_mol_angstrom": env_error,
        "quantum_energy_direct_error_kj_mol": float(direct_e_error),
        "quantum_gradient_direct_error_max_kj_mol_angstrom": direct_g_error,
        "gradient_l2_kj_mol_angstrom": float(np.linalg.norm(grad)),
        "gradient_max_abs_kj_mol_angstrom": float(np.max(np.abs(grad))),
        "elapsed_seconds": time.perf_counter() - start,
    }


def write_summary(summary):
    (OUT / "multiframe.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    B1_OUT.mkdir(parents=True, exist_ok=True)
    print("Stage B2: retrieving actual archived p38a_2g water configurations", flush=True)
    raw = load_archive()
    topology = retrieve_topology()
    source = prepare_files()
    numbers, charge, reference_xyz, ligand_idx, reference_box, b0_info = setup_coordinates(raw)
    if charge != 0:
        raise AssertionError("Stage B1 neutral-ligand assumption changed")
    trajectory_path, trajectory_meta = load_trajectory()
    gromacs_version = run(GMX, "--version")
    if "Precision:       double" not in gromacs_version and not __import__("re").search(
            r"^\s*Precision:\s+double\b", gromacs_version, flags=__import__("re").M | __import__("re").I):
        raise RuntimeError("Must use double-precision GROMACS")
    grompp()
    initial = mda.Universe(str(ROOT / "archived-eq.tpr"), str(ROOT / "archived-eqout.gro"))
    universe = mda.Universe(str(ROOT / "archived-eq.tpr"), str(trajectory_path))
    # XTC does not contain names or types. Both Universe instances would
    # inherit identical metadata from the same TPR, so comparing it is vacuous.
    # Use actual coordinate correspondence as the independent alignment check.
    model = MechanicalEmbedding(ligand_idx, numbers, reference_box)
    result = {
        "checkpoint": "B2.1, genuine molecular multiframe QM/MM feasibility only",
        "source": {"archive_url": ARCHIVE, "trajectory": trajectory_meta,
                   "small_inputs": {k: {"sha256": digest(v), "bytes": len(v)}
                                    for k, v in raw.items()},
                   "topology_members": topology, "upstream": source,
                   "gromacs_version": gromacs_version,
                   "generated_tpr_sha256": {
                       label: digest((ROOT / f"{label}.tpr").read_bytes())
                       for label in ("full", "ligand")
                   },
                   "versions": {p: importlib.metadata.version(p) for p in (
                       "MDAnalysis", "torch", "numpy", "tblite", "tfep", "pyedr")},
                   "commit": os.environ.get("GITHUB_SHA", "local")},
        "reference": {"ligand_indices": ligand_idx.tolist(),
                      "ligand_elements": numbers.tolist(),
                      "atom_count": int(len(reference_xyz)),
                      "topology_validation": b0_info},
        "selection": {"rule": "four uniformly index-spaced production frames 401..2400",
                      "not_statistically_independent": True},
        "frames": [], "validated": False,
    }
    # Keep GRO/XTC discrepancy for high-priority issue #9 investigation.
    result["xtc_gro_alignment"] = check_terminal_alignment(universe, reference_xyz, ligand_idx, reference_box)
    write_summary(result)
    gradient_arrays = {}
    for frame_index, count, time_ps, xyz, box, bond_range in sampled_frames(
            universe, ligand_idx, reference_xyz):
        print(f"Evaluating frame {frame_index}/{count} at t={time_ps} ps", flush=True)
        energies, grad, diagnostics = evaluate(model, xyz, box, ligand_idx, numbers)
        fd_checks = complete_gradient_fd(model, xyz, box, grad, ligand_idx) if frame_index == 401 else []
        entry = {"frame_index": frame_index, "total_frames": count,
                 "time_ps": time_ps, "periodic_box_angstrom_degrees": box.tolist(),
                 "bond_length_range_angstrom": bond_range,
                 "energies_kj_mol": energies, "diagnostics": diagnostics,
                 "complete_hamiltonian_fd_checks": fd_checks}
        result["frames"].append(entry)
        gradient_arrays[f"gradient_frame_{frame_index}_kj_mol_A"] = grad
        print("FRAME_QMMM", json.dumps(entry, allow_nan=False), flush=True)
        write_summary(result)
    if len(result["frames"]) != SAMPLE_COUNT:
        raise AssertionError("Incomplete trajectory sample")
    np.savez_compressed(OUT / "gradients.npz", **gradient_arrays)
    result["gradient_archive_sha256"] = hashlib.sha256(
        (OUT / "gradients.npz").read_bytes()).hexdigest()
    result["validated"] = True
    write_summary(result)
    print("B2_MULTIFRAME_QMMM_VALIDATED", flush=True)


if __name__ == "__main__":
    main()
