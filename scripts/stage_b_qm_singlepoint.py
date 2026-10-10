#!/usr/bin/env python3
"""Stage B0: genuine GFN1-xTB ligand energy/gradient from published coordinates.

Downloads ONLY small archived structure/topology members through HTTP byte ranges.
No surrogate potential, no full 49 GB ZIP transfer, no molecular-flow training.
This is a QM-component checkpoint, NOT yet a complete QM/MM calculation.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import requests
from remotezip import RemoteZip
from tblite.interface import Calculator
import tblite

ARCHIVE = "https://zenodo.org/records/21919704/files/alchemical_tfep.zip?download=1"
BOHR_ANGSTROM = 0.529177210903
HARTREE_KJMOL = 2625.4996394799
SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
).split()
Z = {symbol.upper(): i + 1 for i, symbol in enumerate(SYMBOLS)}


def read_pdb(data):
    lines = data.decode("utf-8").splitlines()
    entries = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    atoms, positions = [], []
    for line in entries:
        if len(line) < 78:
            raise ValueError("PDB atom line missing element columns")
        symbol = line[76:78].strip().upper()
        if symbol not in Z:
            raise ValueError(f"Missing or unsupported explicit PDB element {symbol!r}: {line}")
        atoms.append(Z[symbol])
        positions.append([float(line[k:k+8]) for k in (30, 38, 46)])
    if not atoms:
        raise ValueError("No atoms in published ligand structure")
    return np.asarray(atoms, dtype=np.int32), np.asarray(positions, dtype=np.float64)


def ligand_atoms_from_itp(data):
    """Read ligand atom names, charges, masses from GROMACS [ atoms ]."""
    in_atoms = False
    rows = []
    for line in data.decode("utf-8").splitlines():
        row = line.split(";", 1)[0].strip()
        if row.startswith("["):
            in_atoms = bool(re.fullmatch(r"\[\s*atoms\s*\]", row, re.I))
            continue
        if in_atoms and row and not row.startswith("#"):
            parts = row.split()
            if len(parts) >= 8 and parts[0].isdigit():
                rows.append((parts[4], float(parts[6]), float(parts[7])))
    return rows


def atoms_from_gro(data):
    """Read MOL residue atoms and nm coordinates, retaining source atom order."""
    lines = data.decode("utf-8").splitlines()
    n = int(lines[1].strip())
    if len(lines) < n+3:
        raise ValueError("Truncated archived GRO")
    selected = []
    for line in lines[2:2+n]:
        if line[5:10].strip().upper() == "MOL":
            name = line[10:15].strip()
            coords = [float(line[k:k+8])*10 for k in (20, 28, 36)]
            selected.append((name, coords))
    if not selected:
        raise ValueError("No residue named MOL in archived GRO structure")
    return selected


def atomic_number_from_mass(mass):
    masses = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
              15: 30.974, 16: 32.06, 17: 35.45, 35: 79.904}
    candidates = [z for z, reference in masses.items() if abs(mass-reference) < 0.25]
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous element mass {mass}")
    return candidates[0]


def main():
    out = Path("results/stage-b/qm-singlepoint")
    out.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        with session.get(ARCHIVE, headers={"Range": "bytes=-32"},
                         stream=True, timeout=90) as response:
            if response.status_code != 206 or not response.headers.get("Content-Range"):
                raise RuntimeError("Zenodo does not support required byte-range request")
        with RemoteZip(ARCHIVE, session=session, initial_buffer_size=512*1024,
                       timeout=120) as archive:
            info = archive.infolist()
            selected = [v for v in info if "p38a_2g" in v.filename.lower()]
            inventory = [{"name":v.filename,"bytes":v.file_size} for v in selected
                         if v.file_size < 1500000 and not v.is_dir()]
            (out / "small_member_inventory.json").write_text(json.dumps(inventory, indent=2))
            structures = [v for v in selected
                          if v.filename.lower() == "lig_p38a_2g/eq/water/repeat-0/eqout.gro"]
            itps = [v for v in selected
                    if v.filename.lower() == "lig_p38a_2g/topology/merged.itp"]
            print("Candidate water structures:", [(v.filename,v.file_size) for v in structures], flush=True)
            print("Candidate ligand topologies:", [(v.filename,v.file_size) for v in itps], flush=True)
            if len(structures) != 1 or len(itps) != 1:
                raise RuntimeError("Expected one water eqout.gro and one ligand merged.itp")
            gro = archive.read(structures[0])
            lig = atoms_from_gro(gro)
            records = ligand_atoms_from_itp(archive.read(itps[0]))
            print("GRO MOL atoms:", len(lig), "ITP atoms:", len(records), flush=True)
            if len(lig) != len(records):
                raise RuntimeError("Ligand GRO/ITP atom count mismatch")
            for index, ((gro_name,_),(itp_name,_,_)) in enumerate(zip(lig,records)):
                if gro_name != itp_name:
                    raise RuntimeError(f"Atom order mismatch at {index}: {gro_name} vs {itp_name}")
            net_charge = sum(row[1] for row in records)
            charge = round(net_charge)
            if abs(net_charge-charge) > 0.02:
                raise RuntimeError(f"Nonintegral ligand topology net charge {net_charge}")
            numbers = np.asarray([atomic_number_from_mass(mass) for _,_,mass in records], dtype=np.int32)
            angstrom = np.asarray([coords for _,coords in lig], dtype=np.float64)
            charge_options = [(charge, itps[0].filename, net_charge)]
    positions = angstrom / BOHR_ANGSTROM
    start = time.perf_counter()
    def calc_energy(x, gradient):
        engine = Calculator("GFN1-xTB", numbers, x, charge=charge)
        engine.set("verbosity", 0)
        result = engine.singlepoint()
        energy = float(result.get("energy"))
        if not math.isfinite(energy):
            raise FloatingPointError("Nonfinite GFN1-xTB energy")
        if gradient:
            grad = np.asarray(result.get("gradient"), dtype=float)
            if grad.shape != positions.shape or not np.isfinite(grad).all():
                raise FloatingPointError("Invalid GFN1-xTB gradients")
            return energy, grad
        return energy
    energy, gradient = calc_energy(positions, True)
    idx = int(np.argmax(np.abs(gradient)))
    atom, axis = np.unravel_index(idx, gradient.shape)
    h = 1e-3  # bohr
    xp = positions.copy()
    xm = positions.copy()
    xp[atom,axis] += h
    xm[atom,axis] -= h
    plus = calc_energy(xp, False)
    minus = calc_energy(xm, False)
    fd = (plus-minus)/(2*h)
    analytic = float(gradient[atom,axis])
    absolute_error = abs(fd-analytic)
    relative_error = absolute_error/max(abs(fd),abs(analytic),1e-6)
    summary = {
        "system":"p38a_2g ligand geometry from published archive (NOT full QM/MM)",
        "method":"GFN1-xTB", "embedding":"none: isolated ligand QM component",
        "charge":charge, "n_atoms":len(numbers),
        "gro_archive_member":structures[0].filename,
        "gro_sha256":hashlib.sha256(gro).hexdigest(),
        "itp_charge_candidates":charge_options,
        "coordinate_units":"angstrom input -> bohr for tblite",
        "energy_hartree":energy, "energy_kj_mol":energy*HARTREE_KJMOL,
        "gradient_units":"hartree/bohr",
        "gradient_norm":float(np.linalg.norm(gradient)),
        "fd_atom_zero_based":int(atom), "fd_axis_zero_based":int(axis),
        "fd_step_bohr":h, "fd_analytic_gradient":analytic,
        "fd_numerical_gradient":fd, "fd_abs_error":absolute_error,
        "fd_relative_error":relative_error,
        "elapsed_seconds":time.perf_counter()-start,
        "tblite_version":getattr(tblite,"__version__","unknown"),
        "repo_commit":os.getenv("GITHUB_SHA","local"),
        "original_research_commit":"b8dfd897ebd914b4e592221710b397a2758353ce",
        "full_mechanical_embedding_implemented":False
    }
    (out / "singlepoint.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)
    if absolute_error > 1e-3 or relative_error > 0.05:
        raise AssertionError("Finite-difference gradient inconsistent with tblite gradient")


if __name__ == "__main__":
    main()
