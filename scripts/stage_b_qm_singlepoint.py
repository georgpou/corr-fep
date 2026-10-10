#!/usr/bin/env python3
"""Validate published p38a_2g ligand geometry and true GFN1-xTB gradients.

Reads ONLY selected small ZIP members via HTTP ranges. This validates the
isolated ligand QM component, NOT the complete mechanical-embedding Hamiltonian.
"""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
from collections import Counter

import MDAnalysis as mda
from MDAnalysis import transformations as trans
import numpy as np
import pint
import requests
from remotezip import RemoteZip
from tblite.interface import Calculator
import torch
from tfep.potentials.tblite import TBLitePotential

ARCHIVE = "https://zenodo.org/records/21919704/files/alchemical_tfep.zip?download=1"
PREFIX = "lig_p38a_2g/"
GRO_NAME = PREFIX + "eq/water/repeat-0/eqout.gro"
TPR_NAME = PREFIX + "eq/water/repeat-0/eq.tpr"
ITP_NAME = PREFIX + "topology/merged.itp"
OUT = Path("results/stage-b/qm-singlepoint")
BOHR_PER_ANGSTROM = 1 / 0.529177210903
HARTREE_KJMOL = 2625.4996394799
# Pinned upstream maps atom.type[0] to exactly these elements.
ELEMENTS = {"H": (1, 1.008), "C": (6, 12.011), "N": (7, 14.007),
            "O": (8, 15.999), "F": (9, 18.998), "S": (16, 32.06)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_ligand_itp(raw):
    """GROMACS [atoms]: nr type resnr residue atom cgnr charge mass ..."""
    active = False
    result = []
    for line in raw.decode("utf-8").splitlines():
        stripped = line.split(";", 1)[0].strip()
        if stripped.startswith("["):
            active = stripped.lower().replace(" ", "") == "[atoms]"
            continue
        if active and stripped and not stripped.startswith("#"):
            words = stripped.split()
            if len(words) < 8 or not words[0].isdigit():
                raise ValueError(f"Malformed ligand [atoms] line: {line}")
            result.append(dict(index=int(words[0]), type=words[1],
                               residue=words[3], name=words[4],
                               charge=float(words[6]), mass=float(words[7])))
    if not result or [r["index"] for r in result] != list(range(1, len(result)+1)):
        raise ValueError("Missing/invalid ordered ligand ITP atom table")
    return result


def load_archive():
    OUT.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        with session.get(ARCHIVE, headers={"Range": "bytes=-32"},
                         stream=True, timeout=90) as response:
            if response.status_code != 206 or "Content-Range" not in response.headers:
                raise RuntimeError("Zenodo ignored HTTP Range; refusing whole ZIP download")
        with RemoteZip(ARCHIVE, session=session, initial_buffer_size=512*1024,
                       timeout=120) as archive:
            members = {x.filename: x for x in archive.infolist()
                       if x.filename.startswith(PREFIX)}
            for name in (GRO_NAME, TPR_NAME, ITP_NAME):
                if name not in members:
                    raise FileNotFoundError(f"Missing published input: {name}")
                if members[name].file_size > 2_000_000:
                    raise RuntimeError(f"Unexpectedly large input {name}: refusing")
            result = {name: archive.read(name) for name in (GRO_NAME, TPR_NAME, ITP_NAME)}
    return result


def prepare_structure(raw):
    """Use exactly the original paper's ligand unwrap/center transformations."""
    with tempfile.TemporaryDirectory() as tmp:
        tpr = Path(tmp) / "eq.tpr"
        gro = Path(tmp) / "eqout.gro"
        tpr.write_bytes(raw[TPR_NAME])
        gro.write_bytes(raw[GRO_NAME])
        u = mda.Universe(str(tpr), str(gro))
        ligand = u.select_atoms("resname MOL")
        if len(ligand) == 0 or len(ligand.residues) != 1:
            raise ValueError(f"Expected one MOL residue; got {len(ligand.residues)}")
        if len(ligand.fragments) != 1:
            raise ValueError(f"Expected one covalently connected ligand, got {len(ligand.fragments)} fragments")
        if len(ligand.bonds) == 0:
            raise ValueError("TPR has no ligand bonds; cannot validate PBC geometry")
        if not np.isfinite(u.dimensions).all() or (u.dimensions[:3] <= 0).any():
            raise ValueError("Invalid periodic box dimensions")
        raw_xyz = np.asarray(ligand.positions, dtype=np.float64).copy()
        raw_box = np.asarray(u.dimensions, dtype=np.float64).copy()
        records = read_ligand_itp(raw[ITP_NAME])
        if len(records) != len(ligand):
            raise ValueError(f"ITP/TPR ligand atom counts differ: {len(records)}/{len(ligand)}")
        if not np.array_equal(ligand.names, [x["name"] for x in records]):
            raise ValueError("ITP and TPR ligand atom ordering/names disagree")
        if not np.array_equal(ligand.resnames, [x["residue"] for x in records]):
            raise ValueError("ITP and TPR ligand residue labels disagree")
        mm_charges = np.asarray([x["charge"] for x in records])
        mm_masses = np.asarray([x["mass"] for x in records])
        if np.max(np.abs(mm_charges - ligand.charges)) > 2e-3:
            raise ValueError("ITP/TPR ligand partial charges disagree")
        if np.max(np.abs(mm_masses - ligand.masses)) > 0.25:
            raise ValueError("ITP/TPR ligand masses disagree")
        if not np.array_equal(ligand.types, [x["type"] for x in records]):
            raise ValueError("ITP/TPR ligand atom types disagree")
        numbers, elements = [], []
        for atom in ligand:
            symbol = atom.type[0].upper()  # as in pinned upstream TFEPMap.setup
            if symbol not in ELEMENTS:
                raise ValueError(f"Unsupported or ambiguous atom type {atom.type}")
            atomic_number, standard_mass = ELEMENTS[symbol]
            if abs(atom.mass - standard_mass) > 0.25:
                raise ValueError(f"Atom type/mass inconsistent: {atom.type} {atom.mass}")
            numbers.append(atomic_number)
            elements.append(symbol)
        charge_float = float(mm_charges.sum())
        charge = round(charge_float)
        if abs(charge_float - charge) > 0.02:
            raise ValueError(f"Unphysical non-integral ligand charge {charge_float}")
        electrons = sum(numbers) - charge
        if electrons % 2:
            raise ValueError("Odd electron count: cannot justify singlet uhf=0")
        # Published workflow: unwrap ligand, center it and wrap environment.
        environment = u.select_atoms("not resname MOL")
        u.trajectory.add_transformations(
            trans.unwrap(ligand),
            trans.center_in_box(ligand),
            trans.wrap(environment),
        )
        u.trajectory[0]
        fixed_xyz = np.asarray(ligand.positions, dtype=np.float64).copy()
        if not np.isfinite(fixed_xyz).all():
            raise FloatingPointError("Transformed ligand coordinates nonfinite")
        atom_to_local = {int(global_index): i for i, global_index in enumerate(ligand.indices)}
        edges = np.asarray([[atom_to_local[int(i)], atom_to_local[int(j)]]
                            for i, j in ligand.bonds.indices], dtype=int)
        def lengths(x):
            return np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
        raw_bonds = lengths(raw_xyz)
        fixed_bonds = lengths(fixed_xyz)
        if fixed_bonds.min() < 0.65 or fixed_bonds.max() > 2.2:
            raise ValueError(f"Implausible unwrapped bonded distances: "
                             f"{fixed_bonds.min():.3f} to {fixed_bonds.max():.3f} Angstrom")
        # The bond-length criterion is a physical sanity check, not proof of chemistry.
        return np.asarray(numbers, dtype=np.int32), fixed_xyz, charge, dict(
            n_atoms=int(len(ligand)),
            atom_names=list(ligand.names.astype(str)),
            atom_types=list(ligand.types.astype(str)),
            atomic_numbers=[int(x) for x in numbers],
            elemental_composition=dict(sorted(Counter(elements).items())),
            net_charge=charge,
            itp_net_partial_charge=charge_float,
            tpr_net_partial_charge=float(ligand.charges.sum()),
            electrons=int(electrons),
            assumed_unpaired_electrons=0,
            spin_assumption="closed-shell singlet; parity verified, ground-state spin not independently established",
            bond_count=int(len(edges)), connected_fragments=int(len(ligand.fragments)),
            periodic_box_dimensions_angstrom=raw_box.tolist(),
            raw_max_bond_length_angstrom=float(raw_bonds.max()),
            processed_max_bond_length_angstrom=float(fixed_bonds.max()),
            processed_min_bond_length_angstrom=float(fixed_bonds.min()),
            max_coordinate_shift_angstrom=float(np.linalg.norm(fixed_xyz-raw_xyz, axis=1).max()),
            coordinate_protocol="MDAnalysis 2.7.0: unwrap(MOL), center_in_box(MOL), wrap(non-MOL)",
        )


def direct_singlepoint(numbers, x_bohr, charge):
    calc = Calculator("GFN1-xTB", numbers, x_bohr, charge=charge, uhf=0)
    calc.set("verbosity", 0)
    res = calc.singlepoint()
    e = float(res.get("energy"))
    g = np.asarray(res.get("gradient"), dtype=np.float64)
    if not np.isfinite(e) or not np.isfinite(g).all() or g.shape != x_bohr.shape:
        raise FloatingPointError("Nonfinite QM energy or gradient")
    return e, g


def gradient_probes(numbers, bohr, charge, gradient):
    """Central differences on multiple chemical species and axes at two step sizes."""
    choice = [tuple(np.unravel_index(np.abs(gradient).argmax(), gradient.shape))]
    seen = {choice[0]}
    # Representative elements, spreading the axes deterministically.
    for z in sorted(set(numbers.tolist())):
        atom = int(np.flatnonzero(numbers == z)[0])
        for axis in (0, 1, 2):
            item = (atom, axis)
            if item not in seen:
                choice.append(item)
                seen.add(item)
    # Limit expense while covering 3 axes and distinct element identities.
    choice = choice[:16]
    tests = []
    for atom, axis in choice:
        for h in (0.0005, 0.001):
            plus, minus = bohr.copy(), bohr.copy()
            plus[atom, axis] += h
            minus[atom, axis] -= h
            eplus = direct_singlepoint(numbers, plus, charge)[0]
            eminus = direct_singlepoint(numbers, minus, charge)[0]
            numeric = (eplus - eminus) / (2*h)
            analytic = float(gradient[atom, axis])
            err = abs(numeric-analytic)
            tests.append(dict(atom=atom, axis=axis, atomic_number=int(numbers[atom]),
                              h_bohr=h, analytic_hartree_per_bohr=analytic,
                              finite_diff_hartree_per_bohr=numeric,
                              absolute_error_hartree_per_bohr=err))
            if err > 5e-5:
                raise AssertionError(f"GFN1-xTB gradient mismatch atom={atom} axis={axis} h={h}: {err}")
    return tests


def tfep_wrapper_probe(numbers, angstrom, charge, direct_energy, direct_grad):
    """Confirm the actual TFEP torch autograd interface and unit conversions."""
    if charge != 0:
        raise RuntimeError("Pinned TFEP TBLitePotential wrapper has no charge parameter; "
                           "requires explicit upstream modification for ions")
    ureg = pint.UnitRegistry()
    potential = TBLitePotential(
        method="GFN1-xTB", numbers=numbers,
        positions_unit=ureg.angstrom,
        energy_unit=ureg.kilojoule / ureg.mole,
        precompute_gradient=True,
    )
    x = torch.tensor(angstrom.reshape(1, -1), dtype=torch.float64, requires_grad=True)
    value = potential(x)
    value.sum().backward()
    e_wrapped = float(value.detach().item())
    g_wrapped = x.grad.detach().numpy().reshape(-1, 3)
    reference = direct_grad * HARTREE_KJMOL * BOHR_PER_ANGSTROM
    gradient_max_diff = float(np.max(np.abs(g_wrapped-reference)))
    energy_diff = abs(e_wrapped-direct_energy*HARTREE_KJMOL)
    if energy_diff > 0.02 or gradient_max_diff > 0.03:
        raise AssertionError(f"TFEP unit conversion mismatch: energy {energy_diff}, gradient {gradient_max_diff}")
    # The complete forward+autograd path must itself agree with numerical differences.
    atom, axis = tuple(np.unravel_index(np.abs(g_wrapped).argmax(), g_wrapped.shape))
    h = 0.001  # angstrom
    probes = []
    for sign in (-1, 1):
        displaced = angstrom.copy()
        displaced[atom, axis] += sign*h
        probes.append(float(potential(torch.tensor(displaced.reshape(1, -1), dtype=torch.float64)).item()))
    fd = (probes[1]-probes[0])/(2*h)
    error = abs(fd-g_wrapped[atom, axis])
    if error > 0.2:  # kJ/mol/angstrom; independent of E and gradient conversion
        raise AssertionError(f"TFEP autograd finite difference mismatch {error}")
    return dict(
        energy_kj_mol=e_wrapped,
        direct_energy_discrepancy_kj_mol=energy_diff,
        max_gradient_discrepancy_kj_mol_angstrom=gradient_max_diff,
        finite_difference_atom=int(atom), finite_difference_axis=int(axis),
        fd_step_angstrom=h,
        fd_gradient_kj_mol_angstrom=float(fd),
        autograd_gradient_kj_mol_angstrom=float(g_wrapped[atom, axis]),
        fd_abs_error_kj_mol_angstrom=float(error),
        precompute_gradient=True,
    )


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_archive()
    numbers, angstrom, charge, topology = prepare_structure(raw)
    provenance = dict(
        input_members={k: dict(bytes=len(v), sha256=digest(v)) for k, v in raw.items()},
        archive_url=ARCHIVE,
        repository_commit=os.environ.get("GITHUB_SHA", "local"),
        upstream_paper_commit="b8dfd897ebd914b4e592221710b397a2758353ce",
        upstream_tfep_commit="0bc369a1194c688bb6ac997cc6b7c2db2b063c9e",
        dependency_versions={name: importlib.metadata.version(name) for name in
                             ("MDAnalysis", "tblite", "tfep", "torch", "numpy", "pint")},
    )
    # Write physically validated preparation evidence even if QM fails.
    (OUT / "geometry.json").write_text(json.dumps(dict(provenance=provenance, topology=topology),
                                                   indent=2, allow_nan=False))
    print("Geometry validated:", json.dumps(topology, sort_keys=True), flush=True)
    bohr = angstrom * BOHR_PER_ANGSTROM
    energy, grad = direct_singlepoint(numbers, bohr, charge)
    direct_checks = gradient_probes(numbers, bohr, charge, grad)
    wrapped = tfep_wrapper_probe(numbers, angstrom, charge, energy, grad)
    result = dict(
        method="GFN1-xTB", target="isolated ligand QM term only, NOT full QM/MM",
        energy_hartree=energy,
        energy_kj_mol=energy*HARTREE_KJMOL,
        gradient_norm_hartree_per_bohr=float(np.linalg.norm(grad)),
        gradient_probes=direct_checks,
        max_direct_fd_abs_error_hartree_per_bohr=max(x["absolute_error_hartree_per_bohr"]
                                                    for x in direct_checks),
        tfep_wrapper=wrapped,
        full_mechanical_embedding_implemented=False,
        geometry=topology,
        provenance=provenance,
    )
    (OUT / "singlepoint.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("QM_CHECKPOINT_SUCCESS:", json.dumps({
        "energy_hartree": energy,
        "atom_count": len(numbers),
        "max_direct_fd_error": result["max_direct_fd_abs_error_hartree_per_bohr"],
        "tfep_gradient_diff": wrapped["max_gradient_discrepancy_kj_mol_angstrom"],
        "tfep_fd_error": wrapped["fd_abs_error_kj_mol_angstrom"],
    }), flush=True)


if __name__ == "__main__":
    main()
