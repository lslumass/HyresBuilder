"""
AlphaFold2 Structure Relaxation with OpenMM
============================================
Pipeline:
  1. PDBFixer  — fix missing atoms, add hydrogens
  2. Energy minimization — remove clashes
  3. Restrained MD heating (0 → 300 K) — gentle equilibration
  4. Restrained MD production — further relaxation with backbone restraints
  5. Final energy minimization — clean up

Backbone heavy atoms (CA, N, C, O) are harmonically restrained throughout MD
so the overall fold is preserved while side-chains and loops relax freely.

FIX vs previous version:
  OpenMM >= 8.x does NOT accept implicitSolvent= in createSystem().
  Instead, pass 'implicit/gbn2.xml' directly to the ForceField constructor.

Usage:
    python relax_alphafold.py input.pdb [output.pdb]
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple

from pdbfixer import PDBFixer
from openmm.app import (
    PDBFile, ForceField, Simulation,
    CutoffNonPeriodic, HBonds, StateDataReporter
)
from openmm import (
    LangevinMiddleIntegrator, CustomExternalForce, Platform
)
from openmm.unit import (
    kelvin, picosecond, picoseconds, nanometer,
    kilojoules_per_mole
)


# ─── tuneable parameters ────────────────────────────────────────────────────

BACKBONE_ATOMS      = {"CA", "N", "C", "O"}   # atoms to restrain
RESTRAINT_K         = 1000.0                    # kJ/mol/nm² — spring constant
MINIMIZATION_TOL    = 10.0                      # kJ/mol/nm  — convergence tol
MINIMIZATION_ITER   = 0                         # 0 = run until convergence

HEATING_STEPS       = 5_000                     # steps: 0 → 300 K
HEATING_TEMP_START  = 10                        # K
HEATING_TEMP_END    = 300                       # K
HEATING_N_STAGES    = 10                        # number of temperature ramps

PRODUCTION_STEPS    = 25_000                    # ~100 ps at 4 fs timestep
REPORT_INTERVAL     = 1_000                     # log every N steps

TIMESTEP            = 0.004 * picoseconds       # 4 fs (with HBond constraints)
TEMPERATURE         = 300 * kelvin
FRICTION            = 1 / picosecond
NONBONDED_CUTOFF    = 1.0 * nanometer

# ────────────────────────────────────────────────────────────────────────────


def get_platform() -> Platform:
    """Pick the fastest available platform."""
    for name in ["CUDA", "OpenCL", "CPU"]:
        try:
            p = Platform.getPlatformByName(name)
            print(f"    → Using platform: {name}")
            return p
        except Exception:
            continue
    raise RuntimeError("No OpenMM platform found.")


def prepare_with_pdbfixer(input_pdb: str) -> tuple:
    """Fix structure: non-standard residues, missing heavy atoms, hydrogens."""
    print(f"  Loading:  {input_pdb}")
    fixer = PDBFixer(filename=input_pdb)

    fixer.findMissingResidues()
    fixer.missingResidues = {}          # AF2: skip inserting missing residues

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    fixer.removeHeterogens(keepWater=False)

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    fixer.addMissingHydrogens(7.0)

    n_atoms = fixer.topology.getNumAtoms()
    n_res   = fixer.topology.getNumResidues()
    print(f"  Prepared: {n_res} residues, {n_atoms} atoms")
    return fixer.topology, fixer.positions


def build_system(topology, forcefield: ForceField):
    """
    Create OpenMM system with GBn2 implicit solvent.

    KEY FIX: In OpenMM >= 8.x, implicit solvent is specified by including
    'implicit/gbn2.xml' in the ForceField() constructor — NOT via an
    implicitSolvent= kwarg in createSystem(). The old kwarg raises:
        ValueError: argument 'implicitSolvent' was never used.

    Available implicit solvent XMLs in OpenMM:
        implicit/hct.xml      — Hawkins-Cramer-Truhlar GB
        implicit/obc1.xml     — OBC GB model 1
        implicit/obc2.xml     — OBC GB model 2  (most common)
        implicit/gbn.xml      — GBn
        implicit/gbn2.xml     — GBn2  (most accurate, recommended)
    """
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=NONBONDED_CUTOFF,
        constraints=HBonds,
        # ← implicitSolvent= removed; handled by 'implicit/gbn2.xml' in ForceField
    )
    return system


def add_backbone_restraints(system, topology, positions, k: float = RESTRAINT_K):
    """
    Add harmonic positional restraints on backbone heavy atoms.
    k_restr is a global parameter — can be reduced mid-simulation if desired.
    """
    restraint = CustomExternalForce(
        "k_restr * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    restraint.addGlobalParameter("k_restr", k)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    pos = positions.value_in_unit(nanometer)
    n_restrained = 0

    for atom in topology.atoms():
        if atom.name in BACKBONE_ATOMS:
            restraint.addParticle(
                atom.index,
                [pos[atom.index][0], pos[atom.index][1], pos[atom.index][2]]
            )
            n_restrained += 1

    system.addForce(restraint)
    print(f"  Restrained {n_restrained} backbone atoms  (k = {k} kJ/mol/nm²)")
    return restraint


def report_energy(simulation, label: str):
    state = simulation.context.getState(getEnergy=True)
    e = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    print(f"    [{label}]  E_pot = {e:>14.2f} kJ/mol")
    return e


def run_heating(simulation, start_T, end_T, n_stages, total_steps):
    """Gradually heat from start_T to end_T in n_stages ramps."""
    temps = np.linspace(start_T, end_T, n_stages)
    steps = max(1, total_steps // n_stages)
    print(f"  Heating {start_T} K → {end_T} K  ({n_stages} stages × {steps} steps)")
    for T in temps:
        simulation.integrator.setTemperature(T * kelvin)
        simulation.step(steps)
    simulation.integrator.setTemperature(end_T * kelvin)


def save_pdb(topology, positions, path: str):
    with open(path, "w") as f:
        PDBFile.writeFile(topology, positions, f)
    print(f"  Saved → {path}")


# ─── RMSD analysis ──────────────────────────────────────────────────────────

def _get_positions_angstrom(pdb_path: str) -> Tuple[object, np.ndarray]:
    """Load a PDB and return (topology, positions in Angstrom as Nx3 array)."""
    pdb = PDBFile(pdb_path)
    pos = np.array(pdb.positions.value_in_unit(
        __import__("openmm.unit", fromlist=["angstrom"]).angstrom
    ))
    return pdb.topology, pos


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute RMSD between two Nx3 arrays after optimal superposition (Kabsch).

    Steps:
      1. Translate both sets to their centroids.
      2. Compute the covariance matrix H = P^T Q.
      3. SVD of H → rotation matrix R.
      4. Rotate P onto Q, compute RMSD.
    """
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection (det < 0 means improper rotation)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    P_rot = P @ R.T
    diff  = P_rot - Q
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def _collect_atoms(
    topology,
    positions: np.ndarray,
    atom_names: set,
    heavy_only: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, str, str]]]:
    """
    Extract a subset of atom positions and their labels.

    Returns:
        coords : (M, 3) array in Angstrom
        labels : list of (residue_index, residue_name, atom_name)
    """
    coords, labels = [], []
    for atom in topology.atoms():
        if heavy_only and atom.element is not None and atom.element.symbol == "H":
            continue
        if atom_names and atom.name not in atom_names:
            continue
        coords.append(positions[atom.index])
        labels.append((atom.residue.index, atom.residue.name, atom.name))
    return np.array(coords), labels


def compute_rmsd(
    ref_pdb: str,
    relaxed_pdb: str,
    per_residue: bool = True,
    save_csv: str = None,
) -> Dict[str, float]:
    """
    Compare a relaxed structure to its AlphaFold2 reference.

    Metrics computed (all after Kabsch superposition):
      - CA-RMSD       : Cα atoms only  (global fold deviation)
      - backbone-RMSD : CA + N + C + O (backbone deviation)
      - heavy-RMSD    : all non-hydrogen atoms (all-atom deviation)
      - per-residue   : Cα RMSD for every residue (optional)

    Parameters
    ----------
    ref_pdb     : original AlphaFold2 structure (prepared, with H)
    relaxed_pdb : output of the relaxation pipeline
    per_residue : if True, compute and print per-residue Cα RMSD
    save_csv    : if given, write per-residue RMSD table to this CSV path

    Returns
    -------
    dict with keys 'ca_rmsd', 'backbone_rmsd', 'heavy_rmsd',
    and optionally 'per_residue' (list of dicts)
    """
    print(f"\n  Reference : {ref_pdb}")
    print(f"  Relaxed   : {relaxed_pdb}")

    top_ref, pos_ref = _get_positions_angstrom(ref_pdb)
    top_rel, pos_rel = _get_positions_angstrom(relaxed_pdb)

    if top_ref.getNumAtoms() != top_rel.getNumAtoms():
        raise ValueError(
            f"Atom count mismatch: ref={top_ref.getNumAtoms()}, "
            f"relaxed={top_rel.getNumAtoms()}. "
            "Both PDBs must come from the same prepared topology."
        )

    results = {}

    # ── global RMSDs ─────────────────────────────────────────────
    for label, atom_set, heavy in [
        ("CA-RMSD",       {"CA"},                    False),
        ("backbone-RMSD", {"CA", "N", "C", "O"},     False),
        ("heavy-RMSD",    set(),                      True),   # empty set = all atoms
    ]:
        c_ref, _ = _collect_atoms(top_ref, pos_ref, atom_set, heavy_only=heavy)
        c_rel, _ = _collect_atoms(top_rel, pos_rel, atom_set, heavy_only=heavy)
        rmsd = _kabsch_rmsd(c_ref, c_rel)
        results[label.replace("-", "_").lower()] = rmsd
        print(f"    {label:<18s}  {rmsd:.4f} Å  ({len(c_ref)} atoms)")

    # ── per-residue Cα RMSD ──────────────────────────────────────
    if per_residue:
        c_ref_all, labels_ref = _collect_atoms(top_ref, pos_ref, {"CA"})
        c_rel_all, _          = _collect_atoms(top_rel, pos_rel, {"CA"})

        # Superpose globally first (same rotation for all residues)
        c_ref_c = c_ref_all - c_ref_all.mean(axis=0)
        c_rel_c = c_rel_all - c_rel_all.mean(axis=0)
        H = c_ref_c.T @ c_rel_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        c_ref_rot = c_ref_c @ R.T

        per_res = []
        print(f"\n    {'ResIdx':>6}  {'ResName':>7}  {'CA-RMSD (Å)':>12}")
        print("    " + "-" * 30)
        for i, (res_idx, res_name, _) in enumerate(labels_ref):
            diff = c_ref_rot[i] - (c_rel_all[i] - c_rel_all.mean(axis=0))
            res_rmsd = float(np.sqrt((diff ** 2).sum()))
            per_res.append({
                "residue_index": res_idx,
                "residue_name":  res_name,
                "ca_rmsd_A":     round(res_rmsd, 4),
            })
            print(f"    {res_idx:>6}  {res_name:>7}  {res_rmsd:>12.4f}")

        results["per_residue"] = per_res

        # Optional CSV export
        if save_csv:
            import csv
            with open(save_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["residue_index", "residue_name", "ca_rmsd_A"]
                )
                writer.writeheader()
                writer.writerows(per_res)
            print(f"\n    Per-residue RMSD saved → {save_csv}")

    return results


# ─── main pipeline ──────────────────────────────────────────────────────────

def relax_alphafold_structure(input_pdb: str, output_pdb: str = "relaxed.pdb"):
    print("\n" + "="*60)
    print(" AlphaFold2 Structure Relaxation Pipeline")
    print("="*60)

    # ── 1. Prepare structure ─────────────────────────────────────
    print("\n[1/5] Preparing structure with PDBFixer...")
    topology, positions = prepare_with_pdbfixer(input_pdb)

    # ── 2. Build force field & system ───────────────────────────
    print("\n[2/5] Building OpenMM system (AMBER14 + GBn2 implicit solvent)...")
    # Pass implicit solvent XML here, not in createSystem()
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system     = build_system(topology, forcefield)
    restraint  = add_backbone_restraints(system, topology, positions)
    platform   = get_platform()

    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # ── 3. Initial energy minimization ──────────────────────────
    print("\n[3/5] Energy minimization (remove clashes)...")
    report_energy(simulation, "before minimization")
    simulation.minimizeEnergy(
        tolerance=MINIMIZATION_TOL * kilojoules_per_mole / nanometer,
        maxIterations=MINIMIZATION_ITER,
    )
    report_energy(simulation, "after  minimization")

    base = os.path.splitext(output_pdb)[0]
    minimized_pdb = base + "_minimized.pdb"
    state = simulation.context.getState(getPositions=True)
    # save_pdb(topology, state.getPositions(), minimized_pdb)

    # ── 4. Restrained MD — heating + production ─────────────────
    print(f"\n[4/5] Restrained MD  (backbone k = {RESTRAINT_K} kJ/mol/nm²)...")

    simulation.reporters.append(
        StateDataReporter(
            sys.stdout,
            REPORT_INTERVAL,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            totalSteps=HEATING_STEPS + PRODUCTION_STEPS,
            separator="\t",
        )
    )

    # 4a. Heating 10 K → 300 K
    run_heating(simulation, HEATING_TEMP_START, HEATING_TEMP_END,
                HEATING_N_STAGES, HEATING_STEPS)
    report_energy(simulation, "after  heating")

    # 4b. Production MD at 300 K
    print(f"  Production MD: {PRODUCTION_STEPS} steps "
          f"({PRODUCTION_STEPS * 0.004 / 1000:.1f} ns)  @ {HEATING_TEMP_END} K")
    simulation.step(PRODUCTION_STEPS)
    report_energy(simulation, "after  production MD")

    # ── 5. Final minimization ────────────────────────────────────
    print("\n[5/5] Final energy minimization...")
    simulation.minimizeEnergy(
        tolerance=MINIMIZATION_TOL * kilojoules_per_mole / nanometer,
        maxIterations=MINIMIZATION_ITER,
    )
    report_energy(simulation, "final")

    state = simulation.context.getState(getPositions=True, getEnergy=True)
    save_pdb(topology, state.getPositions(), output_pdb)

    print("\n" + "="*60)
    print(f" Relaxation complete!")
    print(f"   Minimized structure : {minimized_pdb}")
    print(f"   Fully relaxed       : {output_pdb}")
    print("="*60 + "\n")

    return minimized_pdb, output_pdb


# ─── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "alphafold_model.pdb"
    out = sys.argv[2] if len(sys.argv) > 2 else "relaxed.pdb"
    rmsd_analysis = sys.argv[3] if len(sys.argv) > 3 else 0


    if not os.path.exists(inp):
        print(f"ERROR: Input file not found: {inp}")
        sys.exit(1)

    minimized_pdb, relaxed_pdb = relax_alphafold_structure(inp, out)

    if rmsd_analysis == 0:
        sys.exit(0)

    # ── RMSD analysis ────────────────────────────────────────────
    # Uses the PDBFixer-prepared (minimized) structure as the AF2 reference
    # so both PDBs share identical topology (same atoms, same order).
    base = os.path.splitext(out)[0]
    csv_out = base + "_per_residue_rmsd.csv"

    print("\n" + "="*60)
    print(" RMSD Analysis  (relaxed vs. AF2 input)")
    print("="*60)

    rmsd_results = compute_rmsd(
        ref_pdb     = minimized_pdb,   # AF2 structure after PDBFixer (pre-MD)
        relaxed_pdb = relaxed_pdb,     # fully relaxed structure
        per_residue = True,
        save_csv    = csv_out,
    )

    print("\n  ── Summary ──────────────────────────────")
    print(f"  Cα RMSD       : {rmsd_results['ca_rmsd']:.4f} Å")
    print(f"  Backbone RMSD : {rmsd_results['backbone_rmsd']:.4f} Å")
    print(f"  Heavy-atom RMSD: {rmsd_results['heavy_rmsd']:.4f} Å")
    if "per_residue" in rmsd_results:
        rmsds = [r["ca_rmsd_A"] for r in rmsd_results["per_residue"]]
        worst = max(rmsd_results["per_residue"], key=lambda r: r["ca_rmsd_A"])
        print(f"  Max per-res Cα: {max(rmsds):.4f} Å  "
              f"(residue {worst['residue_index']} {worst['residue_name']})")
        print(f"  Mean per-res Cα: {np.mean(rmsds):.4f} Å")
    print(f"\n  Per-residue CSV : {csv_out}")
    print("="*60 + "\n")
