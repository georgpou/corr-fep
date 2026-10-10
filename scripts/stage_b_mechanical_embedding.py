#!/usr/bin/env python3
"""Stage B1: published p38a_2g water mechanical-embedding single point.

Scientific gate: fails on missing published parameters, incompatible GROMACS
settings, nonfinite components, or inconsistent analytical/autograd gradients.
No flow training or surrogate potentials are involved.
"""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.lib.distances import distance_array
import numpy as np
import pint
import requests
from remotezip import RemoteZip
import torch
from tfep.potentials.gromacs import GROMACSPotential
from tfep.potentials.tblite import TBLitePotential

from stage_b_qm_singlepoint import (
    ARCHIVE, PREFIX, GRO_NAME, TPR_NAME, ITP_NAME,
    BOHR_PER_ANGSTROM, HARTREE_KJMOL, digest,
    load_archive, prepare_structure, direct_singlepoint,
)

OUT = Path("results/stage-b/mechanical-embedding").resolve()
UPSTREAM = Path("upstream-stage-b").resolve()
ROOT = OUT / "working"
TOPOLOGY = ROOT / "topology"
MDP_DIR = ROOT / "mdp"
GMX = os.environ.get("GMX", "gmx")
REQUIRED = ("topol_water.top", "ligand_only.top", "merged.itp", "ffMOL.itp")


def run(*args, cwd=None, logfile=None):
    result = subprocess.run(
        list(map(str, args)), cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "GMX_MAXBACKUP": "-1"},
    )
    if logfile is not None:
        Path(logfile).write_text("COMMAND: " + " ".join(map(str, args))
                                 + "\nEXIT: " + str(result.returncode)
                                 + "\n" + result.stdout)
    if result.returncode:
        raise RuntimeError("Command failed (see log): " + " ".join(map(str, args))
                           + "\n" + result.stdout[-3000:])
    return result.stdout


def retrieve_topology():
    """Read exact small archived topology members by byte range, never full ZIP."""
    with requests.Session() as session:
        with session.get(ARCHIVE, headers={"Range": "bytes=-32"},
                         stream=True, timeout=90) as res:
            if res.status_code != 206 or not res.headers.get("Content-Range"):
                raise RuntimeError("Archive does not support byte ranges")
        with RemoteZip(ARCHIVE, session=session,
                       initial_buffer_size=512*1024, timeout=120) as archive:
            names = {info.filename: info for info in archive.infolist()
                     if info.filename.startswith(PREFIX + "topology/")
                     and not info.is_dir()}
            selected = {}
            for name, info in names.items():
                rel = name.removeprefix(PREFIX + "topology/")
                if (not rel or Path(rel).is_absolute() or ".." in Path(rel).parts
                        or info.file_size > 2_000_000):
                    raise ValueError("Unsafe/oversized archived topology member: " + name)
                selected[rel] = archive.read(name)
    absent = set(REQUIRED) - set(selected)
    if absent:
        raise FileNotFoundError("Published topology files missing: " + repr(sorted(absent)))
    TOPOLOGY.mkdir(parents=True, exist_ok=True)
    for rel, data in selected.items():
        dst = TOPOLOGY / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return {PREFIX + "topology/" + k: {"bytes": len(v), "sha256": digest(v)}
            for k, v in sorted(selected.items())}


def prepare_files():
    if not UPSTREAM.is_dir():
        raise FileNotFoundError("Pinned upstream source must be checked out at upstream-stage-b")
    revision = run("git", "-C", UPSTREAM, "rev-parse", "HEAD").strip()
    if revision != "b8dfd897ebd914b4e592221710b397a2758353ce":
        raise RuntimeError("Upstream research commit mismatch: " + revision)
    mdps = {}
    MDP_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("singlepoint.mdp", "singlepoint-vacuum.mdp"):
        original = (UPSTREAM / "mdp" / filename).read_bytes()
        mdps[filename] = {"upstream_sha256": digest(original)}
        # Only rebase the topological include search path; do not alter physics.
        text = original.decode()
        matches = re.findall(r"^\s*include\s*=.*$", text, flags=re.M)
        if len(matches) != 1:
            raise ValueError("Expected one MDP include line in " + filename)
        replacement = "include = -I" + str(TOPOLOGY)
        text = text.replace(matches[0], replacement)
        (MDP_DIR / filename).write_text(text)
        mdps[filename]["staged_sha256"] = digest(text.encode())
    ff = "amber99sb-star-ildn-mut.ff"
    ff_src = UPSTREAM / "topol_shared" / ff
    ff_dst = TOPOLOGY / ff
    if ff_dst.exists():
        raise RuntimeError("Published archive already supplies forcefield directory; audit it")
    shutil.copytree(ff_src, ff_dst)
    forcefield_hashes = {}
    for src in ff_dst.rglob("*"):
        if src.is_file():
            forcefield_hashes[str(src.relative_to(TOPOLOGY))] = digest(src.read_bytes())
    return dict(upstream_sha=revision, mdp=mdps, forcefield_sha256=forcefield_hashes)


def setup_coordinates(raw):
    """Use Stage B0's independently verified atom identities and transformations."""
    numbers, ligand_xyz, charge, b0_info = prepare_structure(raw)
    tpr = ROOT / "archived-eq.tpr"
    gro = ROOT / "archived-eqout.gro"
    tpr.write_bytes(raw[TPR_NAME])
    gro.write_bytes(raw[GRO_NAME])
    universe = mda.Universe(str(tpr), str(gro))
    ligand = universe.select_atoms("resname MOL")
    env = universe.select_atoms("not resname MOL")
    universe.trajectory.add_transformations(
        trans.unwrap(ligand), trans.center_in_box(ligand), trans.wrap(env))
    universe.trajectory[0]
    xyz = np.asarray(universe.atoms.positions, dtype=np.float64).copy()
    idx = ligand.indices.astype(int).copy()
    if not np.allclose(xyz[idx], ligand_xyz, rtol=0, atol=2e-5):
        raise AssertionError("Full/GFN1-xTB ligand coordinate reconstruction differs")
    box = universe.dimensions.astype(np.float64).copy()
    full_gro = ROOT / "full.gro"
    ligand_gro = ROOT / "ligand.gro"
    universe.atoms.write(str(full_gro))
    # grompp requires cutoff < half the box; runtime ligand cell is 1000x full.
    universe.dimensions = box * np.array([1000., 1000., 1000., 1., 1., 1.])
    ligand.write(str(ligand_gro))
    universe.dimensions = box
    check_full = mda.Universe(str(full_gro))
    check_lig = mda.Universe(str(ligand_gro))
    if len(check_full.atoms) != len(xyz) or len(check_lig.atoms) != len(idx):
        raise AssertionError("GRO writer changed atom count")
    if not np.array_equal(np.asarray(check_full.atoms.names), universe.atoms.names):
        raise AssertionError("Full GRO reordered atoms")
    if not np.array_equal(np.asarray(check_lig.atoms.names), ligand.names):
        raise AssertionError("Ligand GRO reordered atoms")
    return numbers, charge, xyz, idx, box, b0_info


def grompp():
    for key, gro, top, mdp in (
        ("full", "full.gro", "topol_water.top", "singlepoint.mdp"),
        ("ligand", "ligand.gro", "ligand_only.top", "singlepoint-vacuum.mdp"),
    ):
        run(GMX, "grompp", "-f", MDP_DIR / mdp, "-c", ROOT / gro,
            "-p", TOPOLOGY / top, "-o", ROOT / (key + ".tpr"),
            "-po", OUT / (key + "-processed.mdp"),
            cwd=ROOT, logfile=OUT / (key + "-grompp.log"))
        # The -po MDP can retain legacy input labels even when grompp translates
        # their runtime meaning; inspect the compiled TPR rather than guessing.
        dump = run(GMX, "dump", "-s", ROOT / (key + ".tpr"))
        terms = ("vdw", "rlist", "rcoulomb", "coulombtype", "dispcorr",
                 "pbc", "constraint", "ewald", "fourier", "rvdw")
        selected = [line.strip() for line in dump.splitlines()
                    if any(term in line.lower() for term in terms)]
        if not selected:
            raise RuntimeError("Cannot audit compiled GROMACS interaction settings")
        # Do not assume that grompp's legacy MDP translation preserved physics.
        expected = {
            "full": {
                "pbc": "xyz", "coulombtype": "PME", "rcoulomb": "1.1",
                "vdw-type": "Cut-off", "vdw-modifier": "Potential-switch",
                "rvdw-switch": "1", "rvdw": "1.1", "DispCorr": "EnerPres",
            },
            "ligand": {
                "pbc": "xyz", "coulombtype": "Cut-off", "rcoulomb": "5",
                "vdw-type": "Cut-off", "vdw-modifier": "Potential-switch",
                "rvdw-switch": "4.9", "rvdw": "5", "DispCorr": "EnerPres",
            },
        }[key]
        for field, value in expected.items():
            pattern = (r"^\s*" + re.escape(field) + r"\s*=\s*" +
                       re.escape(value) + r"\s*$")
            if not any(re.match(pattern, line, flags=re.I)
                       for line in dump.splitlines()):
                raise AssertionError(f"Compiled {key} TPR changed {field}; "
                                     f"expected {value!r}")
        (OUT / (key + "-tpr-parameters.txt")).write_text(
            "\n".join(selected[:120]) + "\n")


class MechanicalEmbedding(torch.nn.Module):
    def __init__(self, idx, numbers, box):
        super().__init__()
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))
        self.box = torch.tensor(box, dtype=torch.float64).reshape(1, 6)
        unit = pint.UnitRegistry()
        kw = dict(positions_unit=unit.angstrom,
                  energy_unit=unit.kilojoule / unit.mole,
                  precompute_gradient=True, on_mdrun_error="raise",
                  mdrun_kwargs={"n_omp_threads_per_mpi_rank": 1})
        self.mm_full = GROMACSPotential(
            tpr_file_path=str(ROOT / "full.tpr"),
            working_dir_path=str(ROOT / "rerun-full"),
            **kw)
        self.mm_lig = GROMACSPotential(
            tpr_file_path=str(ROOT / "ligand.tpr"),
            working_dir_path=str(ROOT / "rerun-ligand"),
            **kw)
        self.qm = TBLitePotential(
            method="GFN1-xTB", numbers=numbers, positions_unit=unit.angstrom,
            energy_unit=unit.kilojoule / unit.mole, precompute_gradient=True)
        for label in ("rerun-full", "rerun-ligand"):
            (ROOT / label).mkdir(exist_ok=True)

    def components(self, flat, batch_cell=None):
        if flat.ndim != 2 or flat.shape[0] != 1 or flat.shape[1] % 3:
            raise ValueError("Expected a batch of one flattened full-system coordinate")
        if flat.shape[1] < 3 * (self.idx.max().item() + 1):
            raise ValueError("Full tensor does not contain all ligand indices")
        ligand_flat = flat.reshape(1, -1, 3)[:, self.idx, :].reshape(1, -1)
        box = self.box.to(flat) if batch_cell is None else batch_cell
        if box.shape != (1, 6) or not torch.isfinite(box).all():
            raise ValueError("Invalid batch cell: expected finite (1, 6) in angstrom/degrees")
        if not torch.all(box[:, :3] > 0):
            raise ValueError("Nonpositive periodic cell length")
        large_box = box.clone()
        large_box[:, :3] *= 1000.
        return (self.mm_full(flat, box),
                self.mm_lig(ligand_flat, large_box),
                self.qm(ligand_flat))

    def forward(self, flat, batch_cell=None):
        full, lig, qm = self.components(flat, batch_cell)
        return full - lig + qm


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def measure(model, xyz, gradients):
    x = torch.tensor(xyz.reshape(1, -1), dtype=torch.float64,
                     requires_grad=gradients)
    if gradients:
        terms = model.components(x)
        grads = []
        for term in terms:
            grad, = torch.autograd.grad(term.sum(), x, retain_graph=True)
            grads.append(grad.detach().cpu().numpy().reshape(-1, 3))
        e = [float(term.detach().item()) for term in terms]
        combined_grad = grads[0] - grads[1] + grads[2]
        full_grad = grads[0]
        return e, combined_grad, full_grad, grads
    return [float(t.item()) for t in model.components(x)], None, None, None


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(exist_ok=True)
    print("Reading published molecular inputs", flush=True)
    raw = load_archive()
    archive_top = retrieve_topology()
    source = prepare_files()
    numbers, charge, xyz, ligand_idx, box, topology_info = setup_coordinates(raw)
    if charge != 0:
        raise ValueError("Upstream TBLitePotential assumes neutral ligand")
    vers = run(GMX, "--version")
    (OUT / "gromacs-version.txt").write_text(vers)
    if not re.search(r"^\s*Precision:\s+double\b", vers, flags=re.M | re.I):
        raise RuntimeError("Stage B1 strict gradient audit requires double-precision GROMACS")
    provenance = dict(
        source=source,
        published_inputs={**{k: {"sha256": digest(v), "bytes":len(v)}
                             for k, v in raw.items()}, **archive_top},
        gromacs_version=vers,
        dependencies={key: importlib.metadata.version(key) for key in
                      ("MDAnalysis", "numpy", "pint", "tblite", "torch", "tfep", "pyedr")},
        workflow_commit=os.environ.get("GITHUB_SHA", "local"),
        atom_count=int(len(xyz)),
        ligand_indices=[int(i) for i in ligand_idx],
        box_angstrom=box.tolist(),
        stage_b0_topology=topology_info,
    )
    save_json("preparation.json", provenance)
    print("Preprocessing original singlepoint MDP files", flush=True)
    grompp()
    tpr_hash = {k: digest((ROOT / (k + ".tpr")).read_bytes())
                for k in ("full", "ligand")}
    provenance["generated_tpr_sha256"] = tpr_hash
    save_json("preparation.json", provenance)

    model = MechanicalEmbedding(ligand_idx, numbers, box)
    print("Running genuine full MM, isolated MM, and GFN1-xTB", flush=True)
    e, grad, full_grad, component_grads = measure(model, xyz, True)
    energy = e[0] - e[1] + e[2]
    if not np.all(np.isfinite(e)) or not np.isfinite(energy) or not np.isfinite(grad).all():
        raise FloatingPointError("Nonfinite QM/MM components")
    # Check the isolated quantum term is exactly evaluated at the full-system ligand positions.
    qm_direct, qm_direct_grad = direct_singlepoint(
        numbers, xyz[ligand_idx] * BOHR_PER_ANGSTROM, charge)
    qm_energy_error = abs(e[2] - qm_direct * HARTREE_KJMOL)
    qm_grad_error = np.max(np.abs(
        component_grads[2][ligand_idx] -
        qm_direct_grad * HARTREE_KJMOL * BOHR_PER_ANGSTROM))
    if qm_energy_error > 0.02 or qm_grad_error > 0.03:
        raise AssertionError("Quantum tensor component does not match direct GFN1-xTB")

    # Ligand-only and QM energies depend exclusively on ligand coordinates.
    env_idx = np.setdiff1d(np.arange(len(xyz)), ligand_idx)
    env_error = float(np.max(np.abs(grad[env_idx] - full_grad[env_idx])))
    if env_error > 1e-7:
        raise AssertionError("QM/MM environmental gradient differs from full MM")
    # Verify the exact (flattened coordinates, periodic cell) module signature
    # used by TFEPMap, not only gradients assembled by hand.
    x_tfep = torch.tensor(xyz.reshape(1, -1), dtype=torch.float64,
                          requires_grad=True)
    box_tfep = torch.tensor(box.reshape(1, 6), dtype=torch.float64)
    total_tfep = model(x_tfep, box_tfep)
    if total_tfep.shape != (1,):
        raise AssertionError(f"Wrong target tensor shape: {tuple(total_tfep.shape)}")
    total_tfep.sum().backward()
    interface_e_error = abs(float(total_tfep.detach().item()) - energy)
    interface_g_error = float(np.max(np.abs(x_tfep.grad.detach().numpy().reshape(-1, 3) - grad)))
    if interface_e_error > 0.05 or interface_g_error > 0.05:
        raise AssertionError(f"TFEP-style QM/MM autograd mismatch: "
                             f"energy={interface_e_error}, gradient={interface_g_error}")
    stage = dict(energies_kj_mol={
        "mm_full":e[0], "mm_ligand":e[1],
        "qm_ligand_gfn1xtb":e[2], "qmmm_combined":energy},
        gradient_norm_kj_mol_angstrom=float(np.linalg.norm(grad)),
        quantum_energy_error_kj_mol=qm_energy_error,
        quantum_gradient_max_error_kj_mol_angstrom=float(qm_grad_error),
        environment_gradient_identity_error_kj_mol_angstrom=env_error,
        tfep_interface={"positions_shape": [1, int(xyz.size)],
                        "cell_shape": [1, 6], "energy_shape": [1],
                        "energy_discrepancy_kj_mol": interface_e_error,
                        "gradient_max_discrepancy_kj_mol_angstrom": interface_g_error},
        finite_difference_checks=[], provenance=provenance)
    save_json("components.json", stage)
    print("Energy components:", json.dumps(stage["energies_kj_mol"]), flush=True)

    # Test all ligand elements and an independently selected strongly forced
    # solvent coordinate at the ligand interface. A weak arbitrary solvent x
    # component can produce a zero mixed-precision numerical derivative.
    # Calculate ligand--environment proximity with the actual periodic box.
    probes = []
    for z in (1, 6, 7, 8, 9):
        local = np.flatnonzero(numbers == z)
        if len(local):
            probes.append((int(ligand_idx[local[0]]), len(probes) % 3))
    environment_distance = distance_array(
        xyz[env_idx], xyz[ligand_idx], box=box).min(axis=1)
    within_shell = env_idx[environment_distance < 4.5]
    if not len(within_shell):
        raise AssertionError("No solvent atom within 4.5 A of ligand; inspect coordinates")
    # Use the maximum absolute Cartesian analytical gradient component,
    # not a manually chosen atom or one selected using numerical derivatives.
    shell_magnitude = np.abs(grad[within_shell])
    best_shell_atom, best_shell_axis = np.unravel_index(
        int(np.argmax(shell_magnitude)), shell_magnitude.shape)
    solvent_atom = int(within_shell[best_shell_atom])
    solvent_axis = int(best_shell_axis)
    probes.append((solvent_atom, solvent_axis))
    stage["environment_probe"] = {
        "atom": solvent_atom,
        "axis": solvent_axis,
        "distance_to_nearest_ligand_angstrom":
            float(environment_distance[np.where(env_idx == solvent_atom)[0][0]]),
        "analytic_gradient_kj_mol_angstrom":
            float(grad[solvent_atom, solvent_axis]),
        "selection": "maximum absolute force component among environment atoms within 4.5 A of ligand",
    }
    save_json("components.json", stage)
    for atom, axis in probes:
        for h in (0.002, 0.005, 0.010):  # Angstrom; convergence and roundoff diagnostic
            plus = xyz.copy()
            minus = xyz.copy()
            plus[atom, axis] += h
            minus[atom, axis] -= h
            ep = measure(model, plus, False)[0]
            em = measure(model, minus, False)[0]
            fd = ((ep[0] - ep[1] + ep[2]) -
                  (em[0] - em[1] + em[2])) / (2*h)
            analytic = float(grad[atom, axis])
            err = abs(fd - analytic)
            item = dict(atom=int(atom), region="ligand" if atom in ligand_idx else "environment",
                        axis=int(axis), h_angstrom=h, analytic_kj_mol_angstrom=analytic,
                        fd_kj_mol_angstrom=float(fd), abs_error_kj_mol_angstrom=float(err))
            stage["finite_difference_checks"].append(item)
            save_json("components.json", stage)
            print("FD_PROBE", json.dumps(item), flush=True)
            # With double-precision energies, all three FD scales must pass.
            if err > 0.8:
                raise AssertionError("Combined QM/MM derivative mismatch; "
                                     + json.dumps(item))
    stage["max_fd_abs_error_kj_mol_angstrom"] = max(
        t["abs_error_kj_mol_angstrom"] for t in stage["finite_difference_checks"])
    stage["max_fd_error_by_h_kj_mol_angstrom"] = {
        str(h): max(t["abs_error_kj_mol_angstrom"] for t in stage["finite_difference_checks"]
                    if t["h_angstrom"] == h)
        for h in (0.002, 0.005, 0.010)
    }
    stage["precision_limitation"] = (
        "The pinned GROMACS 2023.3 run is double precision, unlike the initial "
        "Ubuntu mixed-precision exploratory runs. This does not establish bitwise "
        "equivalence to the authors\u0027 GROMACS 2021.6 double-precision build."
    )
    stage["validated"] = True
    save_json("components.json", stage)
    print("B1_MECHANICAL_EMBEDDING_VALIDATED", flush=True)


if __name__ == "__main__":
    main()
