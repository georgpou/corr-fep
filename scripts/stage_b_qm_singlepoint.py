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


def charge_from_itp(data):
    """Read standard GROMACS atom charge column; fail rather than guess net charge."""
    in_atoms = False
    charges = []
    for line in data.decode("utf-8").splitlines():
        row = line.split(";", 1)[0].strip()
        if row.startswith("["):
            in_atoms = bool(re.fullmatch(r"\[\s*atoms\s*\]", row, re.I))
            continue
        if in_atoms and row and not row.startswith("#"):
            tokens = row.split()
            if len(tokens) < 8:
                raise ValueError(f"Malformed [ atoms ] entry: {row}")
            charges.append(float(tokens[6]))
    if not charges:
        raise ValueError("No [ atoms ] charges found in ligand ITP")
    total = sum(charges)
    value = round(total)
    if abs(total-value) > 0.02:
        raise ValueError(f"Force-field net charge {total:.5f} not near integer")
    return value, len(charges), total


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
            pdbs = [v for v in selected if v.filename.lower().endswith("/mol_gmx.pdb")]
            itps = [v for v in selected if v.filename.lower().endswith(".itp")
                    and "lig" in v.filename.lower() and v.file_size < 1000000]
            print("Candidate PDBs:", [(v.filename,v.file_size) for v in pdbs], flush=True)
            print("Candidate ITPs:", [(v.filename,v.file_size) for v in itps], flush=True)
            if len(pdbs) != 1:
                raise RuntimeError(f"Require unique archived ligand mol_gmx.pdb, found {len(pdbs)}")
            if not itps:
                raise RuntimeError("No ligand ITP candidate; cannot verify net charge")
            pdb = archive.read(pdbs[0])
            numbers, angstrom = read_pdb(pdb)
            charge_options = []
            for member in itps:
                try:
                    q, count, total = charge_from_itp(archive.read(member))
                    if count == len(numbers):
                        charge_options.append((q,member.filename,total))
                except (ValueError,UnicodeDecodeError):
                    pass
            if not charge_options:
                raise RuntimeError("No ligand ITP with matching atom count/charge; refuse guessed charge")
            if len(set(x[0] for x in charge_options)) != 1:
                raise RuntimeError(f"Ambiguous ligand net charges: {charge_options}")
            charge = charge_options[0][0]
    # Atomic-order correspondence with ITP must still be checked before full QM/MM.
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
        "pdb_archive_member":pdbs[0].filename,
        "pdb_sha256":hashlib.sha256(pdb).hexdigest(),
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
