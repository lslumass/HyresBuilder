#!/usr/bin/env python3
"""
build_missing_residues_assembly.py

Rebuild missing residues of a biological assembly (or ASU):

  Phase 1  ASSEMBLY + ALIGNMENT
           Download the mmCIF (or use --cif), expand the biological assembly
           with gemmi, give every chain a unique single-character ID, write
           <PDBID>_<assembly|asu>.pdb and a MODELLER-style .ali (template =
           observed residues, target = full entity sequence). Manual mode
           (--pdb/--ali) skips this phase.

  Phase 2  MISSING ATOMS OF DEPOSITED RESIDUES (PDBFixer, cropped)
           PDBFixer only completes incomplete deposited residues (and swaps
           MSE -> MET, keeping the Se position for SD). It runs on a cropped
           neighbourhood of each incomplete residue. Every backbone O it
           creates is rebuilt exactly in the peptide plane (PDBFixer's soft
           placement can leave it on the wrong side); an O next to a gap is
           placed in Phase 3 together with the new peptide bond.

  Phase 3  LOOPS AND TAILS (one NeRF engine, everything else rigid)
           Existing atoms (deposited residues, other chains, DNA, ligands,
           waters, segments built earlier) form a rigid background.
             * Tails grow forward (C-terminal) or backward (N-terminal) from
               the anchor with NeRF; loops grow from the N-side anchor towards
               the C-side anchor and are closed exactly by CCD followed by a
               Levenberg-Marquardt torsion fit (closure RMSD < --closure-tol).
             * Growth samples Ramachandran phi/psi in vectorised batches,
               keeps clash-free candidates, and backtracks on dead ends.
             * Remaining clashes: Metropolis MC over phi/psi (loops re-closed
               after every move), then side chains from a precomputed rotamer
               library, with a side-chain-aware MC pass if needed.
             * HINGE (--hinge, default 1): only when a segment cannot be made
               clean otherwise, the phi/psi of the deposited residue(s) next to
               the gap become MC / closure variables, restrained to their
               deposited values (+/- --hinge-cap). Everything past the hinge
               stays fixed; the log reports how far hinge atoms moved.
           New residues are L, trans and have ideal geometry by construction.

  Phase 4  CLASH RELAXATION (optional, off by default; enable with --relax local)
           Soft-core minimisation, heavy atoms only. The repulsion k(sigma-r)^2
           stays finite even for overlapping atoms, so it cannot produce NaN.
           Ideal bond lengths/angles (PDBFixer templates, realistic
           stiffness), template handedness for every chiral centre, planar
           groups kept planar, built peptide bonds trans, misplaced backbone O
           snapped back into the peptide plane. --relax local: only
           residues in a contact < 2.5 A (+1 sequence neighbour) move -- built
           ones freely, deposited ones under a positional restraint; metal-ion
           coordination is not treated as a clash. Also works on any PDB:
               --relax-only in.pdb --out relaxed.pdb

  Speed-ups: background KD-tree built once and cropped per segment; all
  segments are first built independently in parallel (--jobs), then accepted
  in a fixed order -- a segment that collides with (or shares hinge residues
  with) an accepted one is rebuilt against the updated structure. Results are
  identical for any --jobs value.

Usage:
  python build_missing_residues_assembly.py 6WG6
  python build_missing_residues_assembly.py 6WG6 --cif 6wg6.cif --assembly-id 1 --jobs 8
  python build_missing_residues_assembly.py --pdb model.pdb --ali model.ali --out filled.pdb
  python build_missing_residues_assembly.py --relax-only 7v96_assembly_filled.pdb --out relaxed.pdb
"""

import argparse
import csv
import itertools
import math
import multiprocessing
import os
import random
import string
import sys
import tempfile
import time
import urllib.request
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import cKDTree

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Atom import Atom as BioAtom
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue as BioResidue

warnings.simplefilter("ignore", PDBConstructionWarning)

try:
    import gemmi
    HAVE_GEMMI = True
except ImportError:
    HAVE_GEMMI = False

import pdbfixer
from pdbfixer import PDBFixer
from openmm import app, unit, Vec3


# ==========================================================================
# Constants
# ==========================================================================
BOND_C_N, BOND_N_CA, BOND_CA_C, BOND_C_O = 1.329, 1.458, 1.525, 1.231
ANG_CA_C_N, ANG_C_N_CA, ANG_N_CA_C, ANG_CA_C_O = 116.2, 121.7, 111.2, 120.5
OMEGA_TRANS = 180.0

DEFAULT_ROTAMER_BUDGET = 4000
SELF_CLASH_CUTOFF = 2.7

# (phi, psi, weight) basins used for sampling AND for the MC Ramachandran prior
RAMA_DEFAULT = [(-120, 130, 0.55), (-75, 145, 0.35), (-90, 0, 0.08), (-60, -45, 0.02)]
RAMA_EXTENDED = [(-125, 135, 0.70), (-78, 148, 0.25), (-95, 20, 0.05)]
RAMA_SIGMA = 12.0          # sampling spread (deg)
RAMA_PRIOR_SIGMA = 25.0    # prior width used by the MC energy (deg)

# Soft "room for the side chain" radius around CB, by residue type.
CB_BUFFER = {**{a: 4.5 for a in 'WYFR'}, **{a: 3.8 for a in 'KMQEHL'},
             **{a: 3.2 for a in 'IVTPNDC'}}
CB_BUFFER_DEFAULT = 2.5

# A tail atom can be at most ~3.8 A/residue from the anchor CA, plus side chain.
REPORT_TOL = 0.1          # overlaps below this (A) are reported as clean
REACH_PER_RES = 3.8
REACH_MARGIN = 10.0

AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}
AA1TO3 = {v: k for k, v in AA3TO1.items() if k not in ('MSE', 'SEC', 'PYL')}


def aa3(aa1):
    return AA1TO3.get(aa1, 'ALA')


def log(msg=''):
    print(msg, flush=True)


def warn(msg):
    print(msg, file=sys.stderr, flush=True)


# ==========================================================================
# Geometry (hand-written cross/norm: np.cross is very slow on tiny arrays)
# ==========================================================================
def _cross(a, b):
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def _unit(v):
    return v / np.sqrt((v * v).sum(axis=-1, keepdims=True))


def nerf(a, b, c, bond, angle_deg, torsion_deg):
    """Place D bonded to C with |CD| = bond, angle B-C-D = angle and dihedral
    A-B-C-D = torsion. Broadcasts over leading dimensions (batched candidates)."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    t = np.radians(np.asarray(torsion_deg, float))[..., None]
    ang = math.radians(angle_deg)
    bc = _unit(c - b)
    n = _unit(_cross(b - a, bc))
    m = _cross(n, bc)
    return c + (-bond * math.cos(ang)) * bc + (bond * math.sin(ang)) * (np.cos(t) * m + np.sin(t) * n)


def place_cb(n, ca, c):
    b = ca - n
    cc = c - ca
    a = _cross(b, cc)
    return ca + (-0.58273431 * a + 0.56802827 * b - 0.54067466 * cc)


def dihedral_deg(p0, p1, p2, p3):
    b0 = np.asarray(p0, float) - np.asarray(p1, float)
    b1 = np.asarray(p2, float) - np.asarray(p1, float)
    b2 = np.asarray(p3, float) - np.asarray(p2, float)
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def rotation_matrix(axis, angle_deg):
    """Right-handed rotation about `axis` (Rodrigues)."""
    k = np.asarray(axis, float)
    k = k / math.sqrt(float(k.dot(k)))
    th = math.radians(angle_deg)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K.dot(K)


def residue_frame(N, CA, C):
    """Orthonormal frame (rows) of a residue from N, CA, C."""
    e1 = _unit(np.asarray(C, float) - CA)
    v = np.asarray(N, float) - CA
    e2 = _unit(v - v.dot(e1) * e1)
    return np.array([e1, e2, np.cross(e1, e2)])


def wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


# ==========================================================================
# .ali parsing and gap mapping
# ==========================================================================
class AliRecord:
    def __init__(self, code, kind, seq):
        self.code, self.kind, self.seq = code, kind, seq


def parse_ali(path):
    records, lines = {}, open(path).read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('>P1;'):
            code = lines[i][4:].strip()
            kind = lines[i + 1].split(':')[0].strip() if i + 1 < len(lines) else ''
            i += 2
            chunks = []
            while i < len(lines) and not lines[i].startswith('>P1;'):
                chunks.append(lines[i].strip())
                i += 1
            records[code] = AliRecord(code, kind, ''.join(chunks).rstrip('*'))
        else:
            i += 1
    return records


def pick_template_and_target(records):
    template = target = None
    for rec in records.values():
        if rec.kind.lower().startswith('structure') and template is None:
            template = rec
        elif rec.kind.lower().startswith('sequence') and target is None:
            target = rec
    if template is None or target is None:
        vals = list(records.values())
        if len(vals) != 2:
            raise ValueError("Could not identify template/target in the .ali: expected one 'structure' "
                             "and one 'sequence' record.")
        template, target = vals
    return template, target


class MissingSeg:
    """A run of missing residues. prev_id / next_id are Bio.PDB residue ids
    (hetflag, number, icode) of the flanking resolved residues (None at a terminus)."""

    def __init__(self, chain_id, aa1_list, prev_id, next_id):
        self.chain_id, self.aa1_list = chain_id, list(aa1_list)
        self.prev_id, self.next_id = prev_id, next_id
        if prev_id is not None:
            self.start_resnum = prev_id[1] + 1
        elif next_id is not None:
            # N-terminal: count straight back from the anchor, never renumber
            # (0 / negative numbers are written as-is).
            self.start_resnum = next_id[1] - len(self.aa1_list)
        else:
            self.start_resnum = 1

    @property
    def kind(self):
        if self.prev_id is None and self.next_id is None:
            return 'unanchored'
        if self.prev_id is None:
            return 'N-tail'
        if self.next_id is None:
            return 'C-tail'
        return 'loop'

    def label(self):
        end = self.start_resnum + len(self.aa1_list) - 1
        return f"{self.chain_id}:{self.kind} {self.start_resnum}..{end} ({len(self.aa1_list)} aa)"


def is_polymer_residue(res):
    """Amino-acid residue of the polymer (standard or modified, e.g. MSE written
    as HETATM). Waters, ions and ligands are excluded."""
    if res.id[0] == 'W':
        return False
    if res.id[0] == ' ':
        return True
    return is_aa(res, standard=False) and 'CA' in res


def default_polymer_chain_ids(model):
    return [c.id for c in model if any(is_polymer_residue(r) for r in c)]


def build_missing_segments(structure, template_rec, target_rec, chain_ids=None):
    tmpl_chains = template_rec.seq.split('/')
    tgt_chains = target_rec.seq.split('/')
    if len(tmpl_chains) != len(tgt_chains):
        raise ValueError("Template and target have different chain counts in .ali")

    model = structure[0]
    chain_ids = chain_ids or default_polymer_chain_ids(model)
    out = {}
    for idx, cid in enumerate(chain_ids):
        if idx >= len(tmpl_chains):
            break
        tstr, gstr = tmpl_chains[idx], tgt_chains[idx]
        if len(tstr) != len(gstr):
            raise ValueError(f"Chain {cid}: template/target length mismatch in .ali")
        resolved = [r for r in model[cid] if is_polymer_residue(r)]
        it = iter(resolved)
        cur = next(it, None)
        segs, pending, prev_id, mismatches = [], [], None, 0
        for t_ch, g_ch in zip(tstr, gstr):
            if g_ch == '-':
                if t_ch != '-':          # residue present in PDB but not in target
                    cur = next(it, None)
                continue
            if t_ch != '-':
                if cur is None:
                    raise ValueError(f"Chain {cid}: .ali lists more resolved residues than the PDB has")
                if AA3TO1.get(cur.get_resname(), 'X') != t_ch.upper() and t_ch.upper() != 'X':
                    mismatches += 1
                if pending:
                    segs.append(MissingSeg(cid, pending, prev_id, cur.id))
                    pending = []
                prev_id = cur.id
                cur = next(it, None)
            else:
                pending.append(g_ch)
        if pending:
            segs.append(MissingSeg(cid, pending, prev_id, None))
        if mismatches:
            warn(f"  WARNING: chain {cid}: {mismatches} template residue(s) differ from the PDB residue names")
        out[cid] = segs
    return out


# ==========================================================================
# Phase 1: assembly + alignment (gemmi)
# ==========================================================================
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
CHAIN_ID_POOL = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)


def _require_gemmi():
    if not HAVE_GEMMI:
        raise RuntimeError("pdb_id mode requires gemmi: pip install gemmi")


def download_cif(pdb_id, outdir):
    path = os.path.join(outdir, f"{pdb_id.lower()}.cif")
    if not os.path.exists(path):
        log(f"[Phase 1] downloading {pdb_id} from RCSB")
        urllib.request.urlretrieve(RCSB_CIF_URL.format(pdb_id=pdb_id.lower()), path)
    return path


def _gemmi_three_to_one(code):
    info = gemmi.find_tabulated_residue(code)
    if info is not None and info.one_letter_code.isalpha():
        return info.one_letter_code.upper()
    return 'X'


def _wrap(seq, width=75):
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def list_assemblies(st):
    if not st.assemblies:
        log("No biological assemblies defined.")
    for a in st.assemblies:
        log(f"- assembly '{a.name}' ({a.oligomeric_details or 'n/a'})"
            f"{'  [author-defined]' if a.author_determined else ''}")


def collect_full_sequences(st):
    table = {}
    for chain in st[0]:
        polymer = chain.get_polymer()
        if not len(polymer):
            continue
        entity = st.get_entity_of(polymer)
        if entity is None or not entity.full_sequence:
            continue
        table[chain.name] = (list(entity.full_sequence), entity.polymer_type)
    return table


def assign_unique_labels(model):
    origin, counts = {}, {}
    for chain in model:
        original = chain.name
        n = counts.get(original, 0) + 1
        label = original if n == 1 else f"{original}-{n}"
        while label in origin:
            n += 1
            label = f"{original}-{n}"
        counts[original] = n
        chain.name = label
        origin[label] = original
    return origin


def load_structure_for_pdb_id(cif_path, assembly_id=None, use_assembly=True):
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.assign_label_seq_id()
    seq_table = collect_full_sequences(st)
    if not use_assembly or not st.assemblies:
        if use_assembly:
            warn("  no assemblies defined in the mmCIF; using the ASU")
        return st, seq_table, assign_unique_labels(st[0]), None

    available = [a.name for a in st.assemblies]
    if assembly_id is None:
        auth = [a.name for a in st.assemblies if a.author_determined]
        assembly_id = auth[0] if auth else available[0]
    elif assembly_id not in available:
        warn(f"  assembly '{assembly_id}' not found; using '{available[0]}'")
        assembly_id = available[0]
    st.transform_to_assembly(assembly_id, gemmi.HowToNameCopiedChain.Dup)
    return st, seq_table, assign_unique_labels(st[0]), assembly_id


def build_gapped_sequences(st, seq_table, origin, nucleic_mode='copy'):
    scoring = gemmi.AlignmentScoring()
    chain_order, observed, full, drop = [], {}, {}, []
    for chain in st[0]:
        key = origin.get(chain.name, chain.name)
        if key not in seq_table:
            continue
        full_sequence, ptype = seq_table[key]
        if ptype not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            if nucleic_mode == 'drop' and ptype in (gemmi.PolymerType.Dna, gemmi.PolymerType.Rna,
                                                    gemmi.PolymerType.DnaRnaHybrid):
                drop.append(chain.name)
            continue
        polymer = chain.get_polymer()
        res = gemmi.align_sequence_to_polymer(full_sequence, polymer, ptype, scoring)
        fg = res.add_gaps("".join(_gemmi_three_to_one(c) for c in full_sequence), 1)
        og = res.add_gaps("".join(_gemmi_three_to_one(r.name) for r in polymer), 2)
        if len(fg) != len(og) or not og.replace('-', ''):
            continue
        chain_order.append(chain.name)
        observed[chain.name], full[chain.name] = og, fg
    return chain_order, observed, full, drop


def prepare_from_pdb_id(pdb_id, workdir, assembly_id=None, asu=False, nucleic_mode='copy',
                        cif_path=None, list_only=False):
    _require_gemmi()
    os.makedirs(workdir, exist_ok=True)
    cif_path = cif_path or download_cif(pdb_id, workdir)
    if list_only:
        list_assemblies(gemmi.read_structure(cif_path))
        return None, None, None

    st, seq_table, origin, used_asm = load_structure_for_pdb_id(cif_path, assembly_id, not asu)
    chain_order, obs, full, drop = build_gapped_sequences(st, seq_table, origin, nucleic_mode)
    for name in drop:
        st[0].remove_chain(name)
    if not chain_order:
        raise ValueError(f"{pdb_id}: no protein chain with a usable entity sequence.")

    model = st[0]
    if len(model) > len(CHAIN_ID_POOL):
        raise ValueError("Too many chains for single-character PDB chain IDs.")
    id_map, rename = {}, {}
    for chain, new_id in zip(model, CHAIN_ID_POOL):
        id_map[new_id], rename[chain.name] = chain.name, new_id
        chain.name = new_id
    order = [rename[c] for c in chain_order]

    tag = f"{pdb_id}_{'asu' if asu else 'assembly'}"
    pdb_path = os.path.join(workdir, f"{tag}.pdb")
    st.write_pdb(pdb_path)

    ali_path = os.path.join(workdir, f"{tag}_alignment.ali")
    with open(ali_path, 'w') as fh:
        fh.write(f">P1;{tag}_template\nstructureX:{tag}_template:FIRST:@:END:@::::\n")
        fh.write(_wrap("/".join(obs[c] for c in chain_order) + "*") + "\n\n")
        fh.write(f">P1;{tag}_target\nsequence:{tag}_target::::::::\n")
        fh.write(_wrap("/".join(full[c] for c in chain_order) + "*") + "\n\n")

    map_path = os.path.join(workdir, f"{pdb_id}_chain_map.csv")
    with open(map_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pdb_chain_id', 'original_label', 'role'])
        for new_id, old in sorted(id_map.items()):
            w.writerow([new_id, old, 'modeled (protein)' if new_id in order else 'passthrough'])

    log(f"[Phase 1] {'ASU' if asu or used_asm is None else 'assembly ' + used_asm}: "
        f"{len(model)} chain(s), {len(order)} protein chain(s) modeled")
    log(f"[Phase 1] wrote {pdb_path}\n[Phase 1] wrote {ali_path}\n[Phase 1] wrote {map_path}")
    return pdb_path, ali_path, order


# ==========================================================================
# Phase 2: complete missing atoms of deposited residues (PDBFixer, cropped)
# ==========================================================================
def _find_topology_residue(fixer, chain_id, bio_id):
    """(chain, index_in_chain) of the residue with the given Bio.PDB id."""
    num, icode = str(bio_id[1]), (bio_id[2] or ' ').strip()
    for chain in fixer.topology.chains():
        if chain.id != chain_id:
            continue
        for i, res in enumerate(chain.residues()):
            if res.id.strip() == num and (res.insertionCode or '').strip() == icode:
                return chain, i
    return None, None


def _positions_A(fixer):
    return np.array(fixer.positions.value_in_unit(unit.angstrom), float).reshape(-1, 3)


def _pdb_atom_line(serial, atom, resname, chain_id, resid, icode, xyz):
    name = atom.name
    el = atom.element.symbol if atom.element is not None else name[0]
    fname = name if len(name) >= 4 or len(el) == 2 else ' ' + name
    rec = 'ATOM  ' if resname in AA1TO3.values() else 'HETATM'
    return (f"{rec}{serial % 100000:5d} {fname:<4} {resname[:3]:>3} {chain_id[:1] or 'A'}"
            f"{resid[-4:]:>4}{(icode or ' ')[:1]}   {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"  1.00  0.00          {el:>2}")


def _cluster_sites(sites):
    """Union-find over (center, radius) spheres that overlap."""
    parent = list(range(len(sites)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if np.linalg.norm(sites[i][0] - sites[j][0]) < sites[i][1] + sites[j][1]:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(sites)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _add_missing_cropped(fixer, seed, margin, workdir):
    """Equivalent of fixer.addMissingAtoms() for incomplete deposited residues,
    but PDBFixer only sees a cropped neighbourhood around each spatial cluster."""
    pos = _positions_A(fixer)
    chains = list(fixer.topology.chains())
    chain_res = [list(c.residues()) for c in chains]
    res_atoms = {r: [a.index for a in r.atoms()] for c in chain_res for r in c}
    sites = [(pos[res_atoms[r]].mean(axis=0), margin + 4.0, r)
             for r in set(fixer.missingAtoms) | set(fixer.missingTerminals)]
    if not sites:
        return 0
    tree = cKDTree(pos)
    atom_res = {i: r for c in chain_res for r in c for i in res_atoms[r]}
    res_pos = {r: (ci, k) for ci, c in enumerate(chain_res) for k, r in enumerate(c)}
    new_xyz = {}
    clusters = _cluster_sites([(c, rad) for c, rad, _ in sites])
    for cl in clusters:
        members = [sites[i][2] for i in cl]
        chosen = set(members)
        for i in cl:
            chosen.update(atom_res[a] for a in tree.query_ball_point(sites[i][0], sites[i][1]))
        frags = []
        for ci in range(len(chain_res)):
            ks = sorted(res_pos[r][1] for r in chosen if res_pos[r][0] == ci)
            run = []
            for k in ks:
                if run and k != run[-1] + 1:
                    frags.append((ci, run))
                    run = []
                run.append(k)
            if run:
                frags.append((ci, run))
        fd, crop_path = tempfile.mkstemp(suffix='.pdb', dir=workdir)
        os.close(fd)
        serial, lines = 0, []
        for ci, run in frags:
            for k in run:
                r = chain_res[ci][k]
                for a in r.atoms():
                    serial += 1
                    lines.append(_pdb_atom_line(serial, a, r.name, chains[ci].id, r.id, r.insertionCode, pos[a.index]))
            lines.append('TER')
        lines.append('END')
        with open(crop_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        try:
            crop = PDBFixer(filename=crop_path)
        finally:
            os.remove(crop_path)
        crop_chains = list(crop.topology.chains())
        if len(crop_chains) != len(frags) or any(
                len(list(cc.residues())) != len(run) for cc, (_, run) in zip(crop_chains, frags)):
            raise RuntimeError("crop topology does not match the fragments written")
        crop_res = [list(cc.residues()) for cc in crop_chains]
        to_crop = {chain_res[ci][k]: crop_res[f][j] for f, (ci, run) in enumerate(frags) for j, k in enumerate(run)}
        crop.missingResidues = {}
        crop.findMissingAtoms()
        want = {to_crop[p] for p in members if p in fixer.missingAtoms}
        crop.missingAtoms = {r: v for r, v in crop.missingAtoms.items() if r in want}
        crop.missingTerminals = {to_crop[p]: list(fixer.missingTerminals[p]) for p in members
                                 if p in fixer.missingTerminals}
        crop.addMissingAtoms(seed=seed)
        cpos = _positions_A(crop)
        frag_chain = [ci for ci, _ in frags]
        for f, cc in enumerate(crop.topology.chains()):
            for r in cc.residues():
                for a in r.atoms():
                    new_xyz[(frag_chain[f], r.id.strip(), (r.insertionCode or '').strip(), a.name)] = cpos[a.index]

    new_top, new_pos, new_atoms, _ = fixer._addAtomsToTopology(False, False)
    positions = list(new_pos.value_in_unit(unit.nanometer))
    missing = 0
    for atom in new_atoms:
        key = (atom.residue.chain.index, atom.residue.id.strip(), (atom.residue.insertionCode or '').strip(), atom.name)
        if key not in new_xyz:
            missing += 1
            continue
        x = new_xyz[key]
        positions[atom.index] = Vec3(float(x[0]) / 10, float(x[1]) / 10, float(x[2]) / 10)
    if missing:
        raise RuntimeError(f"{missing} new atom(s) were not produced by the cropped runs")
    fixer.topology, fixer.positions = new_top, unit.Quantity(positions, unit.nanometer)
    return len(clusters)


def complete_missing_atoms(pdb_in, segments_by_chain, out_path, seed=0, crop=True, margin=10.0):
    """PDBFixer only fills missing heavy atoms of DEPOSITED residues (and swaps
    nonstandard residues such as MSE -> MET). Loops and tails are built in
    Phase 3. Returns the set of residues whose backbone O PDBFixer created."""
    log("\n[Phase 2] PDBFixer: completing missing atoms of deposited residues")
    t0 = time.time()
    fixer = PDBFixer(filename=pdb_in)
    fixer.findNonstandardResidues()
    se_xyz = {}   # MSE -> MET: keep the deposited Se position for the new SD
    if fixer.nonstandardResidues:
        log(f"  replacing {len(fixer.nonstandardResidues)} nonstandard residue(s) "
            f"({', '.join(sorted({r.name for r, _ in fixer.nonstandardResidues}))})")
        pos0 = fixer.positions.value_in_unit(unit.nanometer)
        for r, _ in fixer.nonstandardResidues:
            if r.name == 'MSE':
                for a in r.atoms():
                    if a.name == 'SE':
                        se_xyz[(r.chain.index, r.id.strip(), (r.insertionCode or '').strip())] = pos0[a.index]
        fixer.replaceNonstandardResidues()

    # residues that will get a new neighbour in Phase 3 are not chain termini
    no_oxt = set()
    for cid, segs in segments_by_chain.items():
        for sg in segs:
            if sg.prev_id is not None:
                chain, idx = _find_topology_residue(fixer, cid, sg.prev_id)
                if chain is not None:
                    no_oxt.add((chain.index, idx))
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    for res in list(fixer.missingTerminals):
        if (res.chain.index, list(res.chain.residues()).index(res)) in no_oxt:
            fixer.missingTerminals[res] = [t for t in fixer.missingTerminals[res] if t != 'OXT']
            if not fixer.missingTerminals[res]:
                del fixer.missingTerminals[res]
    added_O = {(r.chain.id, r.id.strip()) for r, missing in fixer.missingAtoms.items()
               if any(a.name == 'O' for a in missing)}
    n_atoms = sum(len(v) for v in fixer.missingAtoms.values())
    if fixer.missingAtoms or fixer.missingTerminals:
        log(f"  completing {n_atoms} missing heavy atom(s) in {len(fixer.missingAtoms)} residue(s)")
        done = False
        if crop:
            try:
                n_cl = _add_missing_cropped(fixer, seed, margin, os.path.dirname(os.path.abspath(out_path)))
                log(f"  PDBFixer ran on {n_cl} cropped cluster(s)")
                done = True
            except Exception as e:  # never lose a run to the optimisation
                warn(f"  cropped PDBFixer failed ({e}); falling back to the full structure")
        if not done:
            fixer.addMissingAtoms(seed=seed)
    else:
        log("  no missing atoms in deposited residues")

    if se_xyz:
        positions = list(fixer.positions.value_in_unit(unit.nanometer))
        for r in fixer.topology.residues():
            key = (r.chain.index, r.id.strip(), (r.insertionCode or '').strip())
            if r.name == 'MET' and key in se_xyz:
                for a in r.atoms():
                    if a.name == 'SD':
                        positions[a.index] = se_xyz[key]
        fixer.positions = unit.Quantity(positions, unit.nanometer)

    with open(out_path, 'w') as fh:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    log(f"  done in {time.time() - t0:.1f} s")
    return added_O


def ideal_carbonyl_O(CA, C, N_next):
    """Backbone O in the peptide plane, pointing away from CA(i) and N(i+1)."""
    CA, C, N_next = (np.asarray(x, float) for x in (CA, C, N_next))
    u = (C - CA) / np.linalg.norm(C - CA) + (C - N_next) / np.linalg.norm(C - N_next)
    return C + BOND_C_O * u / np.linalg.norm(u)


def repair_backbone_oxygens(pdb_path, added_O):
    """PDBFixer places a missing carbonyl O by a single-residue template fit and
    a very soft minimisation, which can leave it on the wrong side of the
    peptide bond. Rebuild each such O exactly in the peptide plane. An O whose
    next residue is still missing is removed: Phase 3 places it together with
    the new peptide bond."""
    if not added_O:
        return 0, 0
    structure = PDBParser(QUIET=True).get_structure('x', pdb_path)
    fixed = dropped = 0
    for chain in structure[0]:
        poly = [r for r in chain if is_polymer_residue(r)]
        for k, r in enumerate(poly):
            if (chain.id, str(r.id[1])) not in added_O or 'O' not in r or not all(a in r for a in ('CA', 'C')):
                continue
            nx = poly[k + 1] if k + 1 < len(poly) else None
            if nx is not None and 'N' in nx and (r['C'] - nx['N']) < 2.0:
                r['O'].set_coord(ideal_carbonyl_O(r['CA'].coord, r['C'].coord, nx['N'].coord).astype('f'))
                fixed += 1
            else:
                r.detach_child('O')
                dropped += 1
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    return fixed, dropped


def report_loop_closure(structure, loops):
    """Peptide C-N distances across each built loop (ideal 1.33 A)."""
    bad = 0
    for seg in loops:
        chain = structure[0][seg.chain_id]
        polymer = [r for r in chain if is_polymer_residue(r)]
        nums = [(r.id[1], r.id[2]) for r in polymer]
        try:
            i0 = nums.index((seg.prev_id[1], seg.prev_id[2]))
            i1 = nums.index((seg.next_id[1], seg.next_id[2]))
        except ValueError:
            continue
        worst = 0.0
        for a, b in zip(polymer[i0:i1], polymer[i0 + 1:i1 + 1]):
            if 'C' in a and 'N' in b:
                worst = max(worst, abs((a['C'] - b['N']) - BOND_C_N))
        if worst > 0.15:
            bad += 1
            warn(f"  WARNING: loop {seg.label()}: peptide bond off ideal by {worst:.2f} A")
    return bad


# ==========================================================================
# Side-chain templates (PDBFixer's bundled ideal residues, no download)
# ==========================================================================
BACKBONE_NAMES = ('N', 'CA', 'C', 'O', 'OXT')

CHI_DEFS = {
    'ARG': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'), ('CB', 'CG', 'CD', 'NE'), ('CG', 'CD', 'NE', 'CZ')],
    'ASN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')],
    'ASP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')],
    'CYS': [('N', 'CA', 'CB', 'SG')],
    'GLN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'), ('CB', 'CG', 'CD', 'OE1')],
    'GLU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'), ('CB', 'CG', 'CD', 'OE1')],
    'HIS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'ND1')],
    'ILE': [('N', 'CA', 'CB', 'CG1'), ('CA', 'CB', 'CG1', 'CD1')],
    'LEU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
    'LYS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'), ('CB', 'CG', 'CD', 'CE'), ('CG', 'CD', 'CE', 'NZ')],
    'MET': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'SD'), ('CB', 'CG', 'SD', 'CE')],
    'PHE': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
    'SER': [('N', 'CA', 'CB', 'OG')],
    'THR': [('N', 'CA', 'CB', 'OG1')],
    'TRP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
    'TYR': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
    'VAL': [('N', 'CA', 'CB', 'CG1')],
}
SP3 = (-65.0, -177.0, 62.0)
SP3_TER = (180.0, -60.0, 60.0)
PLANAR_SYM = (0.0, -30.0, 30.0, 60.0, -60.0, 90.0, -90.0)
PLANAR_ASYM = tuple(float(v) for v in range(-180, 180, 30))
GUANIDINIUM = (180.0, -90.0, 90.0, -120.0, 120.0, 0.0)
CHI_SAMPLES = {
    'ARG': (SP3, SP3, SP3, GUANIDINIUM), 'ASN': (SP3, PLANAR_ASYM), 'ASP': (SP3, PLANAR_SYM),
    'CYS': (SP3,), 'GLN': (SP3, SP3, PLANAR_ASYM), 'GLU': (SP3, SP3, PLANAR_SYM),
    'HIS': (SP3, PLANAR_ASYM), 'ILE': (SP3, SP3), 'LEU': (SP3, SP3),
    'LYS': (SP3, SP3, SP3, SP3_TER), 'MET': (SP3, SP3, SP3_TER), 'PHE': (SP3, PLANAR_SYM),
    'SER': (SP3,), 'THR': (SP3,), 'TRP': (SP3, PLANAR_ASYM), 'TYR': (SP3, PLANAR_SYM), 'VAL': (SP3,),
}


class ResidueTemplate:
    def __init__(self, resname, coords, elements, bonds):
        self.resname, self.coords, self.elements = resname, coords, elements
        self.adj = {n: set() for n in coords}
        for a, b in bonds:
            if a in self.adj and b in self.adj:
                self.adj[a].add(b)
                self.adj[b].add(a)
        self.order = [n for n in ('N', 'CA', 'C', 'O') if n in coords] + \
                     [n for n in coords if n not in BACKBONE_NAMES]
        self._distal = {}

    def distal(self, b, c):
        """Atoms on the c-side of bond b-c (moved by rotating that bond)."""
        if (b, c) not in self._distal:
            seen, stack = {c}, [c]
            while stack:
                cur = stack.pop()
                for nb in self.adj[cur]:
                    if (cur == c and nb == b) or nb in seen:
                        continue
                    seen.add(nb)
                    stack.append(nb)
            self._distal[(b, c)] = None if b in seen else sorted(seen - {c})
        return self._distal[(b, c)]

    def graph_distance(self, a):
        dist, queue = {a: 0}, [a]
        while queue:
            cur = queue.pop(0)
            for nb in self.adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        return dist


_TEMPLATES = {}


def get_template(resname):
    if resname not in _TEMPLATES:
        path = os.path.join(os.path.dirname(pdbfixer.__file__), 'templates', f'{resname}.pdb')
        pdb = app.PDBFile(path)
        coords, elements = {}, {}
        for atom, pos in zip(pdb.topology.atoms(), pdb.positions.value_in_unit(unit.angstrom)):
            if atom.element is None or atom.element.symbol in ('H', 'D') or atom.name == 'OXT':
                continue
            coords[atom.name] = np.array([pos[0], pos[1], pos[2]])
            elements[atom.name] = atom.element.symbol
        bonds = [(a.name, b.name) for a, b in pdb.topology.bonds()]
        _TEMPLATES[resname] = ResidueTemplate(resname, coords, elements, bonds)
    return _TEMPLATES[resname]


def _apply_chi(coords, quad, moving, value):
    a, b, c, d = quad
    delta = value - dihedral_deg(coords[a], coords[b], coords[c], coords[d])
    M = rotation_matrix(coords[c] - coords[b], delta)
    origin = coords[b]
    for name in moving:
        coords[name] = origin + M.dot(coords[name] - origin)


class RotamerLibrary:
    """All chi combinations of one residue type, precomputed ONCE in the
    residue's local N/CA/C frame. Placing every rotamer on a new backbone is a
    single matrix multiply."""

    def __init__(self, resname, budget=DEFAULT_ROTAMER_BUDGET, self_cut=SELF_CLASH_CUTOFF):
        tpl = get_template(resname)
        self.resname = resname
        self.side = [n for n in tpl.order if n not in BACKBONE_NAMES]
        self.elements = [tpl.elements[n] for n in self.side]
        if not self.side:
            self.local = np.zeros((1, 0, 3))
            self.static = np.zeros(1)
            return
        base = dict(tpl.coords)
        rotatable, samples = [], CHI_SAMPLES.get(resname, ())
        for k, quad in enumerate(CHI_DEFS.get(resname, [])):
            moving = tpl.distal(quad[1], quad[2])
            if moving and all(x in base for x in quad):
                vals = samples[k] if k < len(samples) else SP3
                if k < 2:   # +/-15 deg satellites around chi1/chi2 rotamers
                    vals = tuple(v + d for v in vals for d in (0.0, -15.0, 15.0))
                rotatable.append((quad, moving, vals))
        grids = [range(len(s)) for _, _, s in rotatable] or [range(1)]
        combos = list(itertools.islice(itertools.product(*grids), budget))

        R = residue_frame(base['N'], base['CA'], base['C'])
        ca = base['CA']
        local = np.empty((len(combos), len(self.side), 3))
        for ci, idxs in enumerate(combos):
            trial = dict(base)
            for (quad, moving, values), k in zip(rotatable, idxs):
                _apply_chi(trial, quad, moving, values[k])
            local[ci] = [(trial[n] - ca).dot(R.T) for n in self.side]
        self.local = local

        # intra-residue clashes: side-side and side-(N, CA, C) are backbone
        # independent -> precomputed; pairs with O depend on psi -> at pack time
        names = self.side
        bb_local = {n: (base[n] - ca).dot(R.T) for n in ('N', 'CA', 'C')}
        static = 0.02 * np.array([sum(c) for c in combos], float)
        self.o_pairs = []
        for i, a in enumerate(names):
            gd = tpl.graph_distance(a)
            for j in range(i + 1, len(names)):
                b = names[j]
                if gd.get(b, 99) >= 4 and np.linalg.norm(tpl.coords[a] - tpl.coords[b]) >= self_cut:
                    d = np.linalg.norm(local[:, i] - local[:, j], axis=1)
                    static += np.clip(self_cut - d, 0, None)
            for bn, bl in bb_local.items():
                if gd.get(bn, 99) >= 4:
                    d = np.linalg.norm(local[:, i] - bl, axis=1)
                    static += np.clip(self_cut - d, 0, None)
            if gd.get('O', 99) >= 4:
                self.o_pairs.append(i)
        self.static = static
        self.self_cut = self_cut

    def place(self, N, CA, C):
        R = residue_frame(N, CA, C)
        return self.local.dot(R) + CA


_ROTLIBS = {}


def rotamer_library(resname, budget):
    key = (resname, budget)
    if key not in _ROTLIBS:
        _ROTLIBS[key] = RotamerLibrary(resname, budget)
    return _ROTLIBS[key]


class Environment:
    """Rigid atoms for clash scoring: one or more KD-trees."""

    def __init__(self, *trees):
        self.trees = [t for t in trees if t is not None and t.n > 0]

    def penetration(self, pts, radius, k=12, squared=False):
        """Per point: sum over neighbours of p = max(0, radius - d) (or p**2)."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        radius = np.broadcast_to(np.asarray(radius, float), (len(pts),))
        pen = np.zeros(len(pts))
        if not len(pts):
            return pen
        rmax = float(radius.max())
        for tree in self.trees:
            d, _ = tree.query(pts, k=min(k, tree.n), distance_upper_bound=rmax)
            p = np.clip(radius[:, None] - d.reshape(len(pts), -1), 0, None)
            pen += (p * p if squared else p).sum(axis=1)
        return pen

    def worst(self, pts, radius):
        pts = np.asarray(pts, float).reshape(-1, 3)
        w = 0.0
        for tree in self.trees:
            if len(pts):
                d, _ = tree.query(pts, k=1)
                w = max(w, float(np.clip(radius - d, 0, None).max()))
        return w

    def burial(self, pts, r):
        out = np.zeros(len(pts))
        for tree in self.trees:
            out += np.asarray(tree.query_ball_point(pts, r, return_length=True), float)
        return out


def pack_sidechain(resname, N, CA, C, O, env, radius, budget=DEFAULT_ROTAMER_BUDGET):
    """Best rotamer against the rigid environment, all rotamers scored at once.
    Returns ([(name, xyz, element)], residual_penetration)."""
    lib = rotamer_library(resname, budget)
    if not lib.side:
        return [], 0.0
    xyz = lib.place(N, CA, C)                                  # (n_rot, n_side, 3)
    raw = env.penetration(xyz.reshape(-1, 3), radius).reshape(len(xyz), -1).sum(axis=1)
    cost = raw + lib.static
    if lib.o_pairs:
        d = np.linalg.norm(xyz[:, lib.o_pairs] - O, axis=2)
        cost += np.clip(lib.self_cut - d, 0, None).sum(axis=1)
    j = int(np.argmin(cost))
    return [(n, xyz[j, i], lib.elements[i]) for i, n in enumerate(lib.side)], float(raw[j])


# ==========================================================================
# Phase 3: loops and tails -- NeRF growth + CCD closure + MC with hinges
# ==========================================================================
def rama_energy(phi, psi, basins):
    phi, psi = np.atleast_1d(phi), np.atleast_1d(psi)
    p = np.zeros(len(phi))
    wmax = max(b[2] for b in basins)
    for bphi, bpsi, w in basins:
        d2 = wrap_deg(phi - bphi) ** 2 + wrap_deg(psi - bpsi) ** 2
        p += (w / wmax) * np.exp(-0.5 * d2 / RAMA_PRIOR_SIGMA ** 2)
    return -np.log(p + 1e-4)


_IDEAL_BOND = {frozenset(('C', 'N')): BOND_C_N, frozenset(('N', 'CA')): BOND_N_CA, frozenset(('CA', 'C')): BOND_CA_C}
_IDEAL_ANGLE = {'N': ANG_C_N_CA, 'CA': ANG_N_CA_C, 'C': ANG_CA_C_N}   # keyed by the middle atom


def _frames(a, b, c):
    """Orthonormal frames (rows e1, e2, e3; origin b), batched over leading dims."""
    e1 = _unit(c - b)
    v = a - b
    e2 = _unit(v - (v * e1).sum(-1, keepdims=True) * e1)
    return np.stack([e1, e2, _cross(e1, e2)], axis=-2)


def _to_local(a, b, c, x):
    a, b, c, x = (np.asarray(v, float) for v in (a, b, c, x))
    return _frames(a, b, c).dot(x - b)


_LOCAL = {}


def _ideal_locals():
    """Ideal positions of O / OXT (peptide frame CA, C, N+1) and CB (N, CA, C)."""
    if not _LOCAL:
        N0, CA0 = np.zeros(3), np.array([BOND_N_CA, 0.0, 0.0])
        C0 = nerf(np.array([0.0, 1.0, 0.0]), N0, CA0, BOND_CA_C, ANG_N_CA_C, 60.0)
        N1 = nerf(N0, CA0, C0, BOND_C_N, ANG_CA_C_N, 140.0)
        _LOCAL['O'] = _to_local(CA0, C0, N1, nerf(N0, CA0, C0, BOND_C_O, ANG_CA_C_O, 320.0))
        _LOCAL['OXT'] = _to_local(CA0, C0, N1, C0 + 1.25 * _unit(N1 - C0))
        _LOCAL['CB'] = _to_local(N0, CA0, C0, place_cb(N0, CA0, C0))
    return _LOCAL


def _place_py(a, b, c, bond, cos_a, sin_a, t):
    """Scalar NeRF in plain Python (fast for one atom at a time)."""
    bx, by, bz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    ln = math.sqrt(bx * bx + by * by + bz * bz)
    bx, by, bz = bx / ln, by / ln, bz / ln
    ax, ay, az = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    mx, my, mz = ny * bz - nz * by, nz * bx - nx * bz, nx * by - ny * bx
    x = -bond * cos_a
    y, z = bond * sin_a * math.cos(t), bond * sin_a * math.sin(t)
    return (c[0] + x * bx + y * mx + z * nx, c[1] + x * by + y * my + z * ny, c[2] + x * bz + y * mz + z * nz)


def _angle_deg(a, b, c):
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    return math.degrees(math.acos(max(-1.0, min(1.0, v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))))


# atom pairs between sequence neighbours (earlier, later) that are >= 4 bonds apart
_ADJ_PAIRS = {('O', 'O'), ('O', 'CB'), ('O', 'C'), ('CB', 'CB'), ('CB', 'CA'), ('N', 'CA'),
              ('N', 'C'), ('CA', 'C'), ('CA', 'CB'), ('N', 'CB')}


def _pairs_allowed(res_a, names_a, res_b, names_b, forward):
    """Which atom pairs are scored for clashes: residues >= 2 apart, plus the
    >= 4-bond pairs between sequence neighbours."""
    ra, rb = np.asarray(res_a)[:, None], np.asarray(res_b)[None, :]
    d = rb - ra if forward else ra - rb          # > 0: b is later in sequence
    ok = np.abs(d) >= 2
    adj = np.abs(d) == 1
    if adj.any():
        na, nb = np.asarray(names_a, object), np.asarray(names_b, object)
        for i, j in zip(*np.nonzero(adj)):
            pair = (na[i], nb[j]) if d[i, j] > 0 else (nb[j], na[i])
            ok[i, j] = pair in _ADJ_PAIRS
    return ok


class SegmentModel:
    """One loop or tail in internal coordinates.

    Backbone ("chain") atoms -- N, CA, C per residue, or C, CA, N when building
    towards the N-terminus -- are placed by NeRF from the three backbone atoms
    of a fixed BASE residue. Residues in build order:
        hinge (deposited, restrained) -> new -> far hinge (loops) ->
        virtual TARGET (loops: must land on the fixed residue after the gap)
        or virtual next N (C-terminal tail: defines the last psi, O and OXT).
    Bond lengths and angles are the deposited values wherever every atom
    involved is deposited, ideal otherwise. omega is fixed (trans for new
    bonds). phi/psi are DOFs: free for new residues, restrained to the
    deposited values for hinge residues (these only move when enabled).
    Other atoms (O, CB, side chains of hinge residues) ride on local frames.
    """

    def __init__(self, spec, env, rng, args):
        self.spec, self.env, self.rng, self.args = spec, env, rng, args
        self.nprng = np.random.default_rng(rng.randrange(2 ** 32))
        self.R = args.clash_radius
        self.is_loop = spec['kind'] == 'loop'
        self.F = spec['dir'] == 'F'
        self.basins = RAMA_DEFAULT if (self.is_loop or args.tail_mode != 'extended') else RAMA_EXTENDED
        self.cap = math.radians(args.hinge_cap)
        self.flat = math.radians(3.0)
        self.k_h = args.hinge_k
        loc = _ideal_locals()

        res = [dict(kind='hinge', **h) for h in spec['hinge_lo']]
        res += [dict(kind='new', aa=aa, num=num, atoms={}, elem={}) for aa, num in spec['new']]
        res += [dict(kind='hinge', **h) for h in spec['hinge_hi']]
        self.m = len(res)
        if self.is_loop:
            res.append(dict(kind='target', aa=spec['target']['aa'], num=None,
                            atoms=spec['target']['atoms'], elem=spec['target']['elem']))
        elif spec['c_terminal']:
            res.append(dict(kind='vnext', aa='G', num=None, atoms={}, elem={}))
        self.res = res
        base = spec['base']
        order = ('N', 'CA', 'C') if self.F else ('C', 'CA', 'N')

        ext = [(-1, n) for n in order]
        for ri, r in enumerate(res):
            for n in (('N',) if r['kind'] == 'vnext' else order):
                ext.append((ri, n))
        self.ext = ext
        self.row = {key: k for k, key in enumerate(ext)}
        self.L = len(ext) - 3

        def dep(ri, n):
            if ri == -1:
                return base['atoms'].get(n)
            return res[ri]['atoms'].get(n) if res[ri]['kind'] in ('hinge', 'target') else None

        def aa_of(ri):
            return base['aa'] if ri == -1 else res[ri]['aa']
        self._dep, self._aa_of = dep, aa_of

        L = self.L
        self.bond, self.cosA, self.sinA = np.zeros(L), np.zeros(L), np.zeros(L)
        self.tors = np.zeros(L)
        self.kind = [''] * L
        self.ref = np.zeros(L)
        self.owner = np.full(L, -2)
        self.ttype = [''] * L
        self.phi_idx, self.psi_idx = {}, {}
        for j in range(L):
            k = j + 3
            (r0, n0), (r1, n1), (r2, n2), (r3, n3) = ext[k], ext[k - 1], ext[k - 2], ext[k - 3]
            d0, d1, d2, d3 = dep(r0, n0), dep(r1, n1), dep(r2, n2), dep(r3, n3)
            self.bond[j] = np.linalg.norm(d0 - d1) if d0 is not None and d1 is not None \
                else _IDEAL_BOND[frozenset((n1, n0))]
            ang = _angle_deg(d2, d1, d0) if all(x is not None for x in (d0, d1, d2)) else _IDEAL_ANGLE[n1]
            self.cosA[j], self.sinA[j] = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            all4 = all(x is not None for x in (d0, d1, d2, d3))
            meas = dihedral_deg(d3, d2, d1, d0) if all4 else None
            if {n2, n1} == {'C', 'N'}:
                self.kind[j], self.tors[j], self.ttype[j] = 'omega', meas if all4 else 180.0, 'omega'
                continue
            typ = 'phi' if {n2, n1} == {'N', 'CA'} else 'psi'
            self.ttype[j], self.owner[j] = typ, r1
            (self.phi_idx if typ == 'phi' else self.psi_idx)[r1] = j
            if all4:
                self.kind[j], self.ref[j] = 'restr', meas
            else:
                self.kind[j] = 'free'
                own = base['atoms'] if r1 == -1 else res[r1]['atoms']
                if typ == 'psi' and (r1 == -1 or res[r1]['kind'] == 'hinge') and \
                        all(x in own for x in ('N', 'CA', 'C', 'O')):
                    # psi of a deposited residue whose next N is new: its own O fixes it
                    self.kind[j] = 'restr'
                    self.ref[j] = dihedral_deg(own['N'], own['CA'], own['C'], own['O']) - 180.0
            if r1 == -1 and self.kind[j] == 'restr':
                self.kind[j] = 'fixed'          # torsion of the fixed base: its atoms depend on it
            if typ == 'phi' and aa_of(r1) == 'P' and self.kind[j] == 'free':
                self.kind[j], self.ref[j] = 'pro', -65.0
            if self.kind[j] in ('restr', 'fixed', 'pro'):
                self.tors[j] = self.ref[j]
            else:
                self.tors[j] = -70.0 if typ == 'phi' else 140.0
        self.tors = np.array(wrap_deg(self.tors), float)
        self.ref = np.array(wrap_deg(self.ref), float)

        # ---- branch atoms (O, CB / side chains, OXT, missing base O)
        br = []   # (res index, name, element, (row a, row b, row c), local)
        for ri in range(self.m):
            r = res[ri]
            rN, rCA, rC = self.row[(ri, 'N')], self.row[(ri, 'CA')], self.row[(ri, 'C')]
            nxt = (ri + 1, 'N') if self.F else ((ri - 1, 'N') if ri > 0 else (-1, 'N'))
            nN = self.row.get(nxt)
            depN = dep(*nxt) if nN is not None else None
            if nN is not None:
                if r['kind'] == 'hinge' and 'O' in r['atoms'] and depN is not None:
                    lo = _to_local(r['atoms']['CA'], r['atoms']['C'], depN, r['atoms']['O'])
                else:
                    lo = loc['O']
                br.append((ri, 'O', 'O', (rCA, rC, nN), lo))
                if spec['c_terminal'] and ri == self.m - 1:
                    br.append((ri, 'OXT', 'O', (rCA, rC, nN), loc['OXT']))
            if r['kind'] == 'hinge':
                for name, xyz in r['atoms'].items():
                    if name not in ('N', 'CA', 'C', 'O', 'OXT'):
                        br.append((ri, name, r['elem'].get(name, name[0]), (rN, rCA, rC),
                                   _to_local(r['atoms']['N'], r['atoms']['CA'], r['atoms']['C'], xyz)))
            elif r['aa'] != 'G':
                br.append((ri, 'CB', 'C', (rN, rCA, rC), loc['CB']))
        self.base_O_missing = self.F and 'O' not in base['atoms']
        if self.base_O_missing:
            br.append((-1, 'O', 'O', (self.row[(-1, 'CA')], self.row[(-1, 'C')], self.row[(0, 'N')]), loc['O']))
        self.br = br
        self.br_rows = np.array([b[3] for b in br], int).reshape(-1, 3)
        self.br_local = np.array([b[4] for b in br], float).reshape(-1, 3)
        self.br_res = np.array([b[0] for b in br], int)

        # ---- atoms that are scored
        self.X = np.zeros((len(ext), 3))
        for i, n in enumerate(order):
            self.X[i] = base['atoms'][n]
        real_rows = [k for k, (ri, n) in enumerate(ext) if 0 <= ri < self.m]
        self.mob_rows = np.array(real_rows, int)
        self.mob_res = np.array([ext[k][0] for k in real_rows] + list(self.br_res), int)
        self.mob_is_cb_new = np.array([False] * len(real_rows) +
                                      [b[1] == 'CB' and res[b[0]]['kind'] == 'new' for b in br])
        self.mob_bulk = np.array([0.0] * len(real_rows) +
                                 [max(self.R, CB_BUFFER.get(res[b[0]]['aa'], CB_BUFFER_DEFAULT) * args.cb_scale)
                                  if b[1] == 'CB' and b[0] >= 0 and res[b[0]]['kind'] == 'new' else 0.0 for b in br])
        fixed_pts, fixed_res = [], []
        for n, x in base['atoms'].items():
            fixed_pts.append(x)
            fixed_res.append(-1)
        if self.is_loop:
            for n, x in spec['target']['atoms'].items():
                fixed_pts.append(x)
                fixed_res.append(self.m)
        self.fixed_pts = np.array(fixed_pts, float).reshape(-1, 3)
        self.fixed_res = np.array(fixed_res, int)
        self.fixed_names = list(base['atoms']) + (list(spec['target']['atoms']) if self.is_loop else [])
        self.mob_names = [ext[k][1] for k in real_rows] + [b[1] for b in br]
        allres = np.concatenate([self.mob_res, self.fixed_res])
        nm = len(self.mob_res)
        self.allowed = _pairs_allowed(self.mob_res, self.mob_names, allres, self.mob_names + self.fixed_names, self.F)
        self.allowed[np.arange(nm), np.arange(nm)] = False
        self.tgt_rows = [self.row[(self.m, n)] for n in ('N', 'CA', 'C')] if self.is_loop else []
        self.tgt_xyz = np.array([spec['target']['atoms'][n] for n in ('N', 'CA', 'C')]) if self.is_loop else None
        self.new_res = [ri for ri in range(self.m) if res[ri]['kind'] == 'new']
        self.rama_pairs = [(self.phi_idx[ri], self.psi_idx[ri]) for ri in self.new_res
                           if ri in self.phi_idx and ri in self.psi_idx]
        self.restr = [j for j in range(L) if self.kind[j] == 'restr']
        self.tors0 = self.tors.copy()

    # ------------------------------------------------------------ geometry
    def rebuild(self, upto=None):
        X = self.X
        P = [tuple(X[0]), tuple(X[1]), tuple(X[2])]
        t = np.radians(self.tors)
        n = self.L if upto is None else upto
        for j in range(n):
            P.append(_place_py(P[-3], P[-2], P[-1], self.bond[j], self.cosA[j], self.sinA[j], t[j]))
        if n:
            X[3:3 + n] = P[3:]

    def branch_xyz(self, rows_ok=None):
        X = self.X
        a, b, c = X[self.br_rows[:, 0]], X[self.br_rows[:, 1]], X[self.br_rows[:, 2]]
        R = _frames(a, b, c)
        return np.einsum('ni,nij->nj', self.br_local, R) + b

    def mobile_xyz(self):
        return np.vstack([self.X[self.mob_rows], self.branch_xyz()])

    # -------------------------------------------------------------- energy
    def closure_rmsd(self):
        if not self.is_loop:
            return 0.0
        d = self.X[self.tgt_rows] - self.tgt_xyz
        return math.sqrt((d * d).sum() / 3)

    def energy(self, hinge_on, w_bulk=None, w_rama=None):
        w_bulk = self.args.w_bulk if w_bulk is None else w_bulk
        w_rama = self.args.w_rama if w_rama is None else w_rama
        M = self.mobile_xyz()
        bg = self.env.penetration(M, self.R, squared=True)
        allp = np.vstack([M, self.fixed_pts])
        d = np.sqrt(((M[:, None, :] - allp[None, :, :]) ** 2).sum(axis=2))
        pen = np.where(self.allowed, np.clip(self.R - d, 0, None), 0.0)
        nm = len(M)
        sq = pen ** 2
        self_e = sq[:, :nm].sum() / 2 + sq[:, nm:].sum()
        per_atom = bg + sq.sum(axis=1)
        bulk = 0.0
        if self.mob_is_cb_new.any():
            idx = np.nonzero(self.mob_is_cb_new)[0]
            bulk = self.env.penetration(M[idx], self.mob_bulk[idx], squared=True).sum()
        rama = sum(float(rama_energy(self.tors[a], self.tors[b], self.basins)[0]) for a, b in self.rama_pairs)
        restr = 0.0
        if self.restr:
            dev = np.abs(np.radians(wrap_deg(self.tors[self.restr] - self.ref[self.restr])))
            restr = self.k_h * float((np.clip(dev - self.flat, 0, None) ** 2).sum())
        close = 0.0
        if self.is_loop:
            dd = self.X[self.tgt_rows] - self.tgt_xyz
            close = 200.0 * float((dd * dd).sum())
        clash = 10.0 * (bg.sum() + self_e)
        worst = max(self.env.worst(M, self.R), float(pen.max(initial=0.0)))
        per_res = np.bincount(np.clip(self.mob_res, 0, None), weights=per_atom, minlength=self.m)
        E = clash + w_bulk * bulk + w_rama * rama + restr + close
        return E, clash, self.closure_rmsd(), per_res, worst

    # --------------------------------------------------------------- growth
    def _rama(self, m, phi_given=None, psi_given=None):
        w = np.array([b[2] for b in self.basins], float)
        if psi_given is not None:
            w = w * np.exp(-0.5 * (wrap_deg(np.array([b[1] for b in self.basins]) - psi_given) / 30.0) ** 2) + 1e-12
        if phi_given is not None:
            w = w * np.exp(-0.5 * (wrap_deg(np.array([b[0] for b in self.basins]) - phi_given) / 30.0) ** 2) + 1e-12
        j = self.nprng.choice(len(self.basins), size=m, p=w / w.sum())
        b = np.array(self.basins, float)[j]
        return b[:, 0] + self.nprng.normal(0, RAMA_SIGMA, m), b[:, 1] + self.nprng.normal(0, RAMA_SIGMA, m)

    def _sample(self, j, m):
        """Candidate values for torsion j (m samples)."""
        kd = self.kind[j]
        if kd in ('fixed', 'restr', 'omega'):
            return np.full(m, self.tors[j])
        if kd == 'pro':
            return -65.0 + self.nprng.normal(0, 10, m)
        own = self.owner[j]
        if self.ttype[j] == 'phi':
            pj = self.psi_idx.get(own)
            given = self.tors[pj] if (pj is not None and pj < j) else None
            return self._rama(m, psi_given=given)[0]
        pj = self.phi_idx.get(own)
        given = self.tors[pj] if (pj is not None and pj < j) else None
        return self._rama(m, phi_given=given)[1]

    def grow(self, n_candidates=12, retries=40, max_backtrack=3, escape_radius=8.0):
        """Place new residues one at a time (vectorised candidates), keeping
        clash-free ones; on a dead end step back 1..max_backtrack residues."""
        self.rebuild()
        if not self.new_res:
            return True
        first_row = {ri: min(self.row[(ri, n)] for n in ('N', 'CA', 'C')) for ri in range(self.m)}
        # atoms that already exist while growing: fixed self + far-hinge deposited atoms
        static_pts, static_res, static_names = [self.fixed_pts], [self.fixed_res], list(self.fixed_names)
        for ri in range(self.new_res[-1] + 1, self.m):
            pts = np.array(list(self.res[ri]['atoms'].values()), float)
            static_pts.append(pts)
            static_res.append(np.full(len(pts), ri))
            static_names += list(self.res[ri]['atoms'])
        static_pts, static_res = np.vstack(static_pts), np.concatenate(static_res)
        goal = None
        if self.is_loop:
            nxt = self.new_res[-1] + 1
            goal = self.res[nxt]['atoms']['N'] if nxt < self.m else self.tgt_xyz[0]
        kpos, clean, backtracks = 0, True, 0
        budget = 3 * len(self.new_res) + 3
        while kpos < len(self.new_res):
            ri = self.new_res[kpos]
            r0 = first_row[ri]
            js = [r0 - 3, r0 - 2, r0 - 1]          # torsion indices of the residue's 3 chain atoms
            self.rebuild(upto=r0 - 3)
            best, soft, soft_pen = None, None, np.inf
            rounds = retries
            while rounds > 0 and best is None:
                m = n_candidates * min(8, rounds)
                rounds -= 8
                tv = [self._sample(j, m) for j in js]
                A = [self.X[r0 - 3], self.X[r0 - 2], self.X[r0 - 1]]
                pos = []
                for q, j in enumerate(js):
                    p = nerf(A[-3], A[-2], A[-1], self.bond[j],
                             math.degrees(math.atan2(self.sinA[j], self.cosA[j])), tv[q])
                    pos.append(p)
                    A.append(p)
                byname = dict(zip(('N', 'CA', 'C') if self.F else ('C', 'CA', 'N'), pos))
                cand = [byname['N'], byname['CA'], byname['C']]
                cres = [ri, ri, ri]
                cnames = ['N', 'CA', 'C']
                loc = _ideal_locals()
                if self.res[ri]['aa'] != 'G':
                    R = _frames(byname['N'], byname['CA'], byname['C'])
                    cand.append(np.einsum('i,nij->nj', loc['CB'], R) + byname['CA'])
                    cres.append(ri)
                    cnames.append('CB')
                if self.F and ri > 0:        # O of the previous residue now has its frame
                    pr = ri - 1
                    Rf = _frames(np.broadcast_to(self.X[self.row[(pr, 'CA')]], byname['N'].shape),
                                 np.broadcast_to(self.X[self.row[(pr, 'C')]], byname['N'].shape), byname['N'])
                    lo = [b for b in self.br if b[0] == pr and b[1] == 'O'][0][4]
                    cand.append(np.einsum('i,nij->nj', lo, Rf) + self.X[self.row[(pr, 'C')]])
                    cres.append(pr)
                    cnames.append('O')
                if not self.F:               # O of this residue (frame uses the inner N)
                    nN = self.X[self.row[(ri - 1, 'N')] if ri > 0 else self.row[(-1, 'N')]]
                    Rf = _frames(byname['CA'], byname['C'], np.broadcast_to(nN, byname['C'].shape))
                    lo = [b for b in self.br if b[0] == ri and b[1] == 'O'][0][4]
                    cand.append(np.einsum('i,nij->nj', lo, Rf) + byname['C'])
                    cres.append(ri)
                    cnames.append('O')
                C3 = np.stack(cand, axis=1)                         # (m, k, 3)
                cres = np.array(cres)
                flat = C3.reshape(-1, 3)
                pen = self.env.penetration(flat, self.R, squared=True).reshape(m, -1).sum(axis=1)
                # already placed chain atoms + their branch atoms (residues before ri)
                prev_rows = [k for k in range(3, r0) if 0 <= self.ext[k][0] < ri]
                others = [static_pts] + ([self.X[prev_rows]] if prev_rows else [])
                ores = [static_res] + ([np.array([self.ext[k][0] for k in prev_rows])] if prev_rows else [])
                onames = static_names + [self.ext[k][1] for k in prev_rows]
                B = self.branch_xyz()
                sel = (self.br_res >= 0) & (self.br_res < ri) & (self.br_rows.max(axis=1) < r0)
                if sel.any():
                    others.append(B[sel])
                    ores.append(self.br_res[sel])
                    onames += [self.br[q][1] for q in np.nonzero(sel)[0]]
                O_ = np.vstack(others)
                Or = np.concatenate(ores)
                d = np.sqrt(((C3[:, :, None, :] - O_[None, None, :, :]) ** 2).sum(-1))
                ok_pair = _pairs_allowed(cres, cnames, Or, onames, self.F)
                # the candidate's own atoms vs each other (O of previous residue vs this residue)
                if len(set(cres.tolist())) > 1:
                    dd = np.sqrt(((C3[:, :, None, :] - C3[:, None, :, :]) ** 2).sum(-1))
                    okc = np.triu(_pairs_allowed(cres, cnames, cres, cnames, self.F), 1)
                    pen += (np.where(okc[None], np.clip(self.R - dd, 0, None), 0.0) ** 2).sum(axis=(1, 2))
                pen += (np.where(ok_pair[None], np.clip(self.R - d, 0, None), 0.0) ** 2).sum(axis=(1, 2))
                bulk = np.zeros(m)
                if self.res[ri]['aa'] != 'G':
                    bulk = self.env.penetration(C3[:, 3], max(self.R, CB_BUFFER.get(self.res[ri]['aa'],
                                                                                   CB_BUFFER_DEFAULT) * self.args.cb_scale),
                                                squared=True)
                feas = np.zeros(m)
                if goal is not None:
                    remaining = len(self.new_res) - kpos - 1
                    reach = REACH_PER_RES * (remaining + 1)   # CCD does the exact closure
                    feas = np.clip(np.linalg.norm(byname['C'] - goal, axis=1) - reach, 0, None)
                okm = np.nonzero((pen <= 1e-12) & (feas <= 1e-9))[0]
                if len(okm):
                    score = bulk[okm]
                    if not self.is_loop:
                        score = score + 0.002 * self.env.burial(byname['CA'][okm], escape_radius)
                    else:
                        score = score + 0.3 * self.nprng.random(len(okm))
                    best = [tv[q][okm[int(np.argmin(score))]] for q in range(3)]
                else:
                    tot = pen + 5 * feas ** 2
                    q0 = int(np.argmin(tot))
                    if tot[q0] < soft_pen:
                        soft_pen, soft = tot[q0], [tv[q][q0] for q in range(3)]
            if best is None and backtracks < budget and kpos > 0:
                backtracks += 1
                kpos = max(0, kpos - self.rng.randint(1, max_backtrack))
                continue
            if best is None:
                best, clean = soft, False
            for q, j in enumerate(js):
                self.tors[j] = wrap_deg(best[q])
            kpos += 1
        self.rebuild()
        return clean

    # ------------------------------------------------------------------ CCD
    def dofs(self, hinge_on):
        return [j for j in range(self.L) if self.kind[j] == 'free' or (hinge_on and self.kind[j] == 'restr')]

    def ccd(self, dofs, sweeps=300, tol=0.05, skip=None):
        """Cyclic coordinate descent: rotate one torsion at a time (from the
        target end backwards) to bring the virtual target N/CA/C onto the real
        one. Restrained (hinge) torsions stay within +/- hinge_cap."""
        if not self.is_loop or not dofs:
            return self.closure_rmsd()
        dofs = sorted(j for j in dofs if j != skip)
        T = self.tgt_xyz
        self.rebuild()
        for _ in range(sweeps):
            if self.closure_rmsd() < tol:
                break
            E = self.X[self.tgt_rows].copy()
            for j in reversed(dofs):
                o = self.X[j + 2]
                u = self.X[j + 2] - self.X[j + 1]
                u = u / np.linalg.norm(u)
                r = E - o
                rp = r - np.outer(r @ u, u)
                s = np.linalg.norm(rp, axis=1)
                keep = s > 1e-6
                if not keep.any():
                    continue
                rh = rp[keep] / s[keep, None]
                sh = np.cross(u, rh)
                f = (T - o)[keep]
                th = math.atan2(float((s[keep] * (f * sh).sum(1)).sum()), float((s[keep] * (f * rh).sum(1)).sum()))
                if self.kind[j] == 'restr':
                    new = self.ref[j] + np.clip(wrap_deg(self.tors[j] + math.degrees(th) - self.ref[j]),
                                                -self.args.hinge_cap, self.args.hinge_cap)
                    th = math.radians(float(wrap_deg(new - self.tors[j])))
                if abs(th) < 1e-9:
                    continue
                self.tors[j] = float(wrap_deg(self.tors[j] + math.degrees(th)))
                E = (E - o) @ rotation_matrix(u, math.degrees(th)).T + o
            self.rebuild()
        return self.closure_rmsd()

    def close_ls(self, dofs, iters=30, tol=0.01, skip=None):
        """Finish closure with Levenberg-Marquardt on the torsions: the
        derivative of a downstream atom x w.r.t. torsion j is u_j x (x - o_j),
        so the 9 x n Jacobian is exact. Small steps => torsions stay close to
        their current values; hinge torsions stay within +/- hinge_cap."""
        if not self.is_loop:
            return 0.0
        dofs = [j for j in dofs if j != skip]
        self.rebuild()
        cur = self.closure_rmsd()
        if not dofs:
            return cur
        lam = 1e-2
        wts = np.array([4.0 if self.kind[j] == 'restr' else 1.0 for j in dofs])
        for _ in range(iters):
            if cur < tol:
                break
            E = self.X[self.tgt_rows]
            J = np.zeros((9, len(dofs)))
            for c, j in enumerate(dofs):
                o = self.X[j + 2]
                u = o - self.X[j + 1]
                u = u / np.linalg.norm(u)
                J[:, c] = np.cross(u, E - o).ravel()
            r = (self.tgt_xyz - E).ravel()
            JTJ = J.T @ J
            scale = np.trace(JTJ) / len(dofs) + 1e-9
            step = np.degrees(np.linalg.solve(JTJ + np.diag(lam * scale * wts), J.T @ r))
            mx = np.abs(step).max()
            if mx > 20:
                step *= 20 / mx
            saved = self.tors.copy()
            for c, j in enumerate(dofs):
                new = self.tors[j] + step[c]
                if self.kind[j] == 'restr':
                    new = self.ref[j] + np.clip(wrap_deg(new - self.ref[j]), -self.args.hinge_cap, self.args.hinge_cap)
                self.tors[j] = float(wrap_deg(new))
            self.rebuild()
            nxt = self.closure_rmsd()
            if nxt < cur:
                cur, lam = nxt, max(lam / 3, 1e-6)
            else:
                self.tors = saved
                self.rebuild()
                lam *= 5
                if lam > 1e3:
                    break
        return cur

    def close(self, dofs, sweeps, tol):
        """CCD from far away, least squares to finish."""
        if not self.is_loop:
            return 0.0
        if self.closure_rmsd() > 0.5:
            self.ccd(dofs, sweeps, 0.3)
        return self.close_ls(dofs, 20, tol * 0.5)

    # ------------------------------------------------------------------- MC
    def mc(self, steps, hinge_on, t_start=1.0, t_end=0.01, sigma=15.0, w_bulk=None, until_clean=True):
        """Metropolis MC over phi/psi (and restrained hinge torsions when
        enabled). Loops are re-closed by a few CCD sweeps after every move."""
        dofs = self.dofs(hinge_on)
        if not dofs:
            return self.energy(hinge_on, w_bulk)[4], 0
        tol = self.args.closure_tol
        E, clash, cerr, per_res, worst = self.energy(hinge_on, w_bulk)
        key = lambda c, ce, e: (c > 1e-9 or ce > tol, c, e)
        best = (key(clash, cerr, E), self.tors.copy(), worst)
        if until_clean and clash <= 1e-9 and cerr <= tol:
            return worst, 0
        accepted = 0
        dof_res = {j: int(self.owner[j]) for j in dofs}
        for step in range(steps):
            T = t_start * (t_end / t_start) ** (step / max(1, steps - 1))
            bad = np.nonzero(per_res > 1e-12)[0]
            if len(bad) and self.rng.random() < 0.8:
                r = int(bad[self.rng.randrange(len(bad))])
                pool = [j for j in dofs if r - 4 <= dof_res[j] <= r] or dofs
            else:
                pool = dofs
            j = pool[self.rng.randrange(len(pool))]
            if self.kind[j] == 'restr':
                delta = self.rng.gauss(0, sigma / 3)
                if abs(wrap_deg(self.tors[j] + delta - self.ref[j])) > self.args.hinge_cap:
                    continue
            else:
                delta = self.rng.uniform(-180, 180) if self.rng.random() < 0.1 else self.rng.gauss(0, sigma)
            saved = self.tors.copy()
            self.tors[j] = wrap_deg(self.tors[j] + delta)
            if self.is_loop:
                self.close_ls(dofs, iters=4, tol=tol / 2, skip=j)
            else:
                self.rebuild()
            E2, clash2, cerr2, per2, worst2 = self.energy(hinge_on, w_bulk)
            if E2 <= E or self.rng.random() < math.exp(-(E2 - E) / T):
                E, clash, cerr, per_res, worst = E2, clash2, cerr2, per2, worst2
                accepted += 1
                k2 = key(clash, cerr, E)
                if k2 < best[0]:
                    best = (k2, self.tors.copy(), worst)
                    if until_clean and not k2[0]:
                        break
            else:
                self.tors = saved
        self.tors = best[1].copy()
        self.rebuild()
        return best[2], accepted

    # ----------------------------------------------------------- reporting
    def hinge_shift(self):
        if not self.restr:
            return 0.0, 0.0
        dt = float(np.abs(wrap_deg(self.tors[self.restr] - self.tors0[self.restr])).max())
        return dt, 0.0


def find_residue(chain, bio_id):
    for r in chain:
        if r.id[1] == bio_id[1] and r.id[2] == bio_id[2] and is_polymer_residue(r):
            return r
    return None


def sort_chain_residues(chain):
    """Sequence order for output; residue numbers are never changed."""
    chain.child_list.sort(key=lambda r: (0 if is_polymer_residue(r) else 2 if r.id[0] == 'W' else 1,
                                         r.id[1], r.id[2]))
    chain.child_dict = {r.id: r for r in chain.child_list}


class Background:
    """All rigid heavy atoms, indexed ONCE; each segment gets a cropped KD-tree
    of the atoms it can possibly reach."""

    def __init__(self, structure, include_water=True):
        atoms = [a for a in structure.get_atoms() if a.element not in ('H', 'D') and
                 (include_water or a.get_parent().id[0] != 'W')]
        self.xyz = np.array([a.get_coord() for a in atoms], float).reshape(-1, 3)
        self.index = {id(a): i for i, a in enumerate(atoms)}
        self.tree = cKDTree(self.xyz) if len(self.xyz) else None

    def crop(self, center, radius, exclude=(), added=None):
        idx = np.array(self.tree.query_ball_point(center, radius), int) if self.tree is not None else np.zeros(0, int)
        if len(exclude):
            idx = np.setdiff1d(idx, np.asarray(list(exclude), int))
        pts = self.xyz[idx]
        if added is not None and len(added):
            near = added[np.linalg.norm(added - center, axis=1) <= radius]
            pts = np.vstack([pts, near])
        return cKDTree(pts) if len(pts) else None


def _residue_atoms(res):
    atoms = {a.get_id(): a.get_coord().astype(float) for a in res if a.element not in ('H', 'D')}
    elem = {a.get_id(): (a.element.capitalize() if len(a.element) > 1 else a.element) for a in res
            if a.element not in ('H', 'D')}
    return atoms, elem


def make_segment_spec(model0, seg, hinge, bg):
    """Base / hinge / target residues of one gap, read from the current structure."""
    chain = model0[seg.chain_id]
    poly = [r for r in chain if is_polymer_residue(r) and all(k in r for k in ('N', 'CA', 'C'))]
    pos = {(r.id[1], r.id[2]): k for k, r in enumerate(poly)}
    bonded = [('C' in a and 'N' in b and (a['C'] - b['N']) < 2.0) for a, b in zip(poly, poly[1:])]

    def run_ok(k0, k1):   # poly[k0..k1] consecutive and peptide-bonded
        return 0 <= k0 <= k1 < len(poly) and all(bonded[k0:k1])

    def pack(r):
        atoms, elem = _residue_atoms(r)
        return {'aa': AA3TO1.get(r.get_resname(), 'A'), 'num': r.id[1], 'atoms': atoms, 'elem': elem}
    n = len(seg.aa1_list)
    nums = [seg.start_resnum + i for i in range(n)]
    spec = {'kind': seg.kind, 'label': seg.label(), 'chain_id': seg.chain_id, 'hinge_hi': [],
            'c_terminal': seg.kind == 'C-tail', 'target': None}
    if seg.kind in ('C-tail', 'loop'):
        a = pos[(seg.prev_id[1], seg.prev_id[2])]
        h = hinge
        while h > 0 and not run_ok(a - h, a):
            h -= 1
        base, lo = poly[a - h], poly[a - h + 1: a + 1]
        spec.update(dir='F', base=pack(base), hinge_lo=[pack(r) for r in lo],
                    new=list(zip(seg.aa1_list, nums)))
        touched = [base] + lo
        if seg.kind == 'loop':
            q = pos[(seg.next_id[1], seg.next_id[2])]
            h2 = hinge
            while h2 > 0 and not run_ok(q, q + h2):
                h2 -= 1
            hi, tgt = poly[q: q + h2], poly[q + h2]
            spec['hinge_hi'] = [pack(r) for r in hi]
            spec['target'] = pack(tgt)
            touched += hi + [tgt]
            c0, c1 = base['CA'].get_coord().astype(float), tgt['CA'].get_coord().astype(float)
            spec['center'] = (c0 + c1) / 2
            spec['reach'] = float(np.linalg.norm(c1 - c0)) / 2 + REACH_PER_RES * (n + len(lo) + len(hi)) / 2 + REACH_MARGIN
        else:
            spec['center'] = base['CA'].get_coord().astype(float)
            spec['reach'] = REACH_PER_RES * (n + len(lo)) + REACH_MARGIN
    else:   # N-tail: build towards the N-terminus
        a = pos[(seg.next_id[1], seg.next_id[2])]
        h = hinge
        while h > 0 and not run_ok(a, a + h):
            h -= 1
        base, lo = poly[a + h], list(reversed(poly[a: a + h]))
        spec.update(dir='R', base=pack(base), hinge_lo=[pack(r) for r in lo],
                    new=list(reversed(list(zip(seg.aa1_list, nums)))))
        touched = [base] + lo
        spec['center'] = base['CA'].get_coord().astype(float)
        spec['reach'] = REACH_PER_RES * (n + len(lo)) + REACH_MARGIN
    spec['exclude'] = [bg.index[id(at)] for r in touched for at in r if id(at) in bg.index]
    spec['touched'] = {(seg.chain_id, r.id[1]) for r in touched}
    return spec


def _pack_new_sidechains(model, env_tree, args):
    """Side chains of new residues, outward from the base, against everything."""
    M = model.mobile_xyz()
    packed, sc_worst = {}, 0.0
    names = {}
    for k, (ri, n) in enumerate(model.ext):
        names[k] = (ri, n)
    for ri in model.new_res:
        own = model.mob_res == ri
        extra = [M[~own], model.fixed_pts] + [np.array([a[1] for a in packed[q]]).reshape(-1, 3) for q in packed]
        env = Environment(env_tree, cKDTree(np.vstack(extra)))
        X = model.X
        N, CA, C = X[model.row[(ri, 'N')]], X[model.row[(ri, 'CA')]], X[model.row[(ri, 'C')]]
        Oi = [i for i, b in enumerate(model.br) if b[0] == ri and b[1] == 'O']
        O = M[len(model.mob_rows) + Oi[0]] if Oi else C
        atoms, _ = pack_sidechain(aa3(model.res[ri]['aa']), N, CA, C, O, env, args.clash_radius, args.rotamer_budget)
        packed[ri] = atoms
        side = [a[1] for a in atoms if a[0] != 'CB']
        if side:
            sc_worst = max(sc_worst, env.worst(side, args.clash_radius))
    return packed, sc_worst


def build_one_segment(spec, env_tree, args, rng):
    """Grow (+ close) + MC + pack one loop or tail. Returns a result dict."""
    env = Environment(env_tree)
    is_loop = spec['kind'] == 'loop'
    tol = args.closure_tol
    has_hinge = any(True for _ in spec['hinge_lo']) or any(True for _ in spec['hinge_hi'])
    way = 'loop NeRF+CCD' if is_loop else ('forward' if spec['dir'] == 'F' else 'backward')

    def attempt_model():
        m = SegmentModel(spec, env, rng, args)
        m.grow(args.candidates, args.retries, args.backtrack)
        if is_loop:
            if m.close(m.dofs(False), args.ccd_sweeps, tol) > tol and has_hinge:
                m.close(m.dofs(True), args.ccd_sweeps, tol)
        return m

    best_bb, best_packed, n_att = None, None, 0
    for attempt in range(max(1, args.attempts)):
        n_att = attempt + 1
        m = attempt_model()
        _, clash, cerr, _, worst = m.energy(True)
        if clash <= 1e-9 and cerr <= tol:
            packed, scw = _pack_new_sidechains(m, env_tree, args)
            if best_packed is None or scw < best_packed[2]:
                best_packed = (m, packed, scw)
            if scw <= REPORT_TOL:
                break
        elif best_bb is None or (cerr > tol, clash) < (best_bb[2] > tol, best_bb[1]):
            best_bb = (m, clash, cerr, worst)

    notes = []
    if best_packed is not None:
        model, packed, sc_worst = best_packed
    else:
        model = best_bb[0]
        w0 = best_bb[3]
        worst, _ = model.mc(args.mc_steps // 2 if has_hinge else args.mc_steps, False,
                            args.mc_t_start, args.mc_t_end, args.mc_sigma)
        stage = f"MC {w0:.2f} -> {worst:.2f} A"
        if (worst > 1e-9 or model.closure_rmsd() > tol) and has_hinge:
            worst, _ = model.mc(args.mc_steps // 2, True, args.mc_t_start, args.mc_t_end, args.mc_sigma)
            stage += f", with hinge -> {worst:.2f} A"
        notes.append(stage)
        packed, sc_worst = _pack_new_sidechains(model, env_tree, args)
    how = f"{way}, {n_att} attempt(s)"
    if sc_worst > REPORT_TOL:
        # backbone leaves no room for a side chain: wiggle with a strong
        # side-chain-room term (hinge allowed) and repack
        before = sc_worst
        trial = SegmentModel(spec, env, rng, args)
        trial.tors = model.tors.copy()
        trial.rebuild()
        _, bclash, _, _, _ = model.energy(True)
        bbw, _ = trial.mc(args.mc_steps // 2, has_hinge, args.mc_t_start, args.mc_t_end, args.mc_sigma,
                          w_bulk=10 * args.w_bulk, until_clean=False)
        p2, sc2 = _pack_new_sidechains(trial, env_tree, args)
        _, tclash, _, _, _ = trial.energy(True)
        if tclash <= bclash + 1e-9 and trial.closure_rmsd() <= max(tol, model.closure_rmsd()) and sc2 < sc_worst:
            model, packed, sc_worst = trial, p2, sc2
        notes.append(f"side-chain MC {before:.2f} -> {sc_worst:.2f} A")

    E, clash, cerr, _, worst = model.energy(True)
    # ---- collect coordinates
    M = model.mobile_xyz()
    nrow = len(model.mob_rows)
    per_res = {}
    for k, row in enumerate(model.mob_rows):
        ri, n = model.ext[row]
        per_res.setdefault(ri, {})[n] = (M[k], n[0])
    for i, b in enumerate(model.br):
        per_res.setdefault(b[0], {})[b[1]] = (M[nrow + i], b[2])
    new_records, hinge_records = [], []
    for ri in range(model.m):
        r = model.res[ri]
        atoms = per_res.get(ri, {})
        if r['kind'] == 'new':
            lst = [(n, atoms[n][0], atoms[n][1]) for n in ('N', 'CA', 'C', 'O') if n in atoms]
            lst += [(n, np.asarray(x, float), el) for n, x, el in packed.get(ri, [])]
            if 'OXT' in atoms:
                lst.append(('OXT', atoms['OXT'][0], 'O'))
            new_records.append((r['num'], aa3(r['aa']), lst))
        else:
            hinge_records.append((r['num'], {n: v[0] for n, v in atoms.items()}))
    new_records.sort(key=lambda x: x[0])
    base_O = per_res.get(-1, {}).get('O', (None,))[0]
    moved = [x for _, _, lst in new_records for _, x, _ in lst] + \
            [x for _, d in hinge_records for x in d.values()] + ([base_O] if base_O is not None else [])
    dt, _ = model.hinge_shift()
    hinge_shift = 0.0
    for num, d in hinge_records:
        src = [h for h in spec['hinge_lo'] + spec['hinge_hi'] if h['num'] == num][0]['atoms']
        hinge_shift = max([hinge_shift] + [float(np.linalg.norm(d[n] - src[n])) for n in d if n in src])
    if hinge_shift > 0.01:
        notes.append(f"hinge moved {hinge_shift:.2f} A (max torsion change {dt:.0f} deg)")
    if is_loop:
        notes.append(f"closure {cerr:.3f} A")
    status = 'clean' if worst <= REPORT_TOL and sc_worst <= REPORT_TOL and (not is_loop or cerr <= tol) else \
        f"worst backbone overlap {worst:.2f} A, worst side-chain overlap {sc_worst:.2f} A" + \
        (f", CLOSURE GAP {cerr:.2f} A" if is_loop and cerr > tol else '')
    line = f"  {spec['label']}: {how}; {'; '.join(notes) + '; ' if notes else ''}{status}"
    report = {'segment': spec['label'], 'backbone_worst_A': round(worst, 3), 'sidechain_residual_A': round(sc_worst, 3),
              'closure_A': round(cerr, 3), 'hinge_shift_A': round(hinge_shift, 3)}
    return {'spec_label': spec['label'], 'chain_id': spec['chain_id'], 'new': new_records, 'hinge': hinge_records,
            'base_num': spec['base']['num'], 'base_O': base_O, 'moved': np.array(moved, float).reshape(-1, 3),
            'touched': spec['touched'], 'report': report, 'line': line}


_BG = None   # Background shared with worker processes (inherited through fork)


def _seg_seed(seed, label, round_no):
    return zlib.crc32(f"{seed}:{label}:{round_no}".encode())


def _build_job(job):
    spec, args, round_no = job
    t0 = time.time()
    rng = random.Random(_seg_seed(args.seed, spec['label'], round_no))
    tree = _BG.crop(spec['center'], spec['reach'], spec['exclude'])
    res = build_one_segment(spec, tree, args, rng)
    res['seconds'] = round(time.time() - t0, 2)
    res['line'] += f" [{res['seconds']:.1f} s]"
    return res


def _apply_result(model0, res):
    chain = model0[res['chain_id']]
    for num, resname, atoms in res['new']:
        r = BioResidue((' ', num, ' '), resname, ' ')
        for name, xyz, el in atoms:
            r.add(BioAtom(name, np.asarray(xyz, 'f'), 0.0, 1.0, ' ', f' {name:<3}', 0, element=el))
        chain.add(r)
    for num, atoms in res['hinge']:
        r = find_residue(chain, (' ', num, ' '))
        for a in [a for a in r if a.element in ('H', 'D')]:
            r.detach_child(a.get_id())
        for name, xyz in atoms.items():
            if name in r:
                r[name].set_coord(np.asarray(xyz, 'f'))
            else:
                r.add(BioAtom(name, np.asarray(xyz, 'f'), 0.0, 1.0, ' ', f' {name:<3}', 0, element=name[0]))
    if res['base_O'] is not None:
        r = find_residue(chain, (' ', res['base_num'], ' '))
        if r is not None and 'O' not in r:
            r.add(BioAtom('O', np.asarray(res['base_O'], 'f'), 0.0, 1.0, ' ', ' O  ', 0, element='O'))
            log(f"  placed missing O of {res['chain_id']}:{r.get_resname()}{r.id[1]} in the new peptide plane")


def build_segments(pdb_in, segments_by_chain, out_path, args):
    """All loops and tails. Round 1 builds every segment independently in
    parallel against the rigid structure; results are accepted in a fixed
    order, and any segment that collides with (or shares hinge residues with)
    an accepted one is rebuilt in round 2 against the updated structure."""
    global _BG
    log("\n[Phase 3] Loops and tails: NeRF growth + CCD closure + MC (hinge on demand)")
    t_start = time.time()
    structure = PDBParser(QUIET=True).get_structure('model', pdb_in)
    model0 = structure[0]
    segs = [s for v in segments_by_chain.values() for s in v if s.kind in ('loop', 'N-tail', 'C-tail')]
    ok = []
    for seg in segs:
        chain = model0[seg.chain_id]
        if seg.kind == 'loop' and seg.next_id[1] - seg.prev_id[1] - 1 != len(seg.aa1_list):
            warn(f"  SKIPPED {seg.label()}: residue-number gap {seg.next_id[1] - seg.prev_id[1] - 1} does not fit "
                 f"{len(seg.aa1_list)} residues; refusing to renumber")
            continue
        anchors = [x for x in (seg.prev_id, seg.next_id) if x is not None]
        if any(find_residue(chain, x) is None or not all(k in find_residue(chain, x) for k in ('N', 'CA', 'C'))
               for x in anchors):
            warn(f"  FAILED {seg.label()}: anchor residue missing N/CA/C")
            continue
        taken = [seg.start_resnum + i for i in range(len(seg.aa1_list))
                 if any(r.id[1] == seg.start_resnum + i and r.id[2] == ' ' and is_polymer_residue(r) for r in chain)]
        if taken:
            warn(f"  SKIPPED {seg.label()}: residue number(s) {taken} already used; refusing to renumber")
            continue
        if seg.prev_id is not None:
            r = find_residue(chain, seg.prev_id)
            for name in ('OXT', 'HXT'):
                if name in r:
                    r.detach_child(name)
        if seg.kind == 'N-tail':
            r = find_residue(chain, seg.next_id)
            for name in ('H1', 'H2', 'H3'):
                if name in r:
                    r.detach_child(name)
        ok.append(seg)
    order_key = lambda s: (0 if s.kind == 'loop' else 1, len(s.aa1_list), s.chain_id, s.start_resnum)
    ok.sort(key=order_key)

    _BG = Background(structure, include_water=not args.ignore_water)
    specs = [make_segment_spec(model0, s, args.hinge, _BG) for s in ok]
    n_jobs = max(1, min(args.jobs, len(specs)))
    log(f"  round 1: {len(specs)} segment(s) built independently on {n_jobs} worker(s)")
    jobs = [(sp, args, 1) for sp in sorted(specs, key=lambda sp: -len(sp['new']))]
    results = {}
    if n_jobs > 1 and len(jobs) > 1 and 'fork' in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
            for f in as_completed([pool.submit(_build_job, j) for j in jobs]):
                r = f.result()
                results[r['spec_label']] = r
    else:
        for j in jobs:
            r = _build_job(j)
            results[r['spec_label']] = r

    accepted, redo = [], []
    acc_pts, acc_touched = np.zeros((0, 3)), set()
    for s in ok:
        r = results[s.label()]
        clash = False
        if len(acc_pts) and len(r['moved']):
            d, _ = cKDTree(acc_pts).query(r['moved'], k=1)
            clash = bool((d < args.clash_radius).any())
        if clash or (r['touched'] & acc_touched):
            redo.append(s)
            continue
        accepted.append(r)
        acc_pts = np.vstack([acc_pts, r['moved']])
        acc_touched |= r['touched']
        log(r['line'])
    for r in accepted:
        _apply_result(model0, r)
    if redo:
        log(f"  round 2: {len(redo)} segment(s) collided or share hinge residues; rebuilding against the update")
        for s in redo:
            _BG = Background(structure, include_water=not args.ignore_water)
            sp = make_segment_spec(model0, s, args.hinge, _BG)
            r = _build_job((sp, args, 2))
            _apply_result(model0, r)
            accepted.append(r)
            log(r['line'])
    for chain in model0:
        sort_chain_residues(chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    log(f"  segments built in {time.time() - t_start:.1f} s")
    return [r['report'] for r in accepted], structure, ok


# ==========================================================================
# Phase 4: clash relaxation (soft-core, geometry-preserving minimisation)
# ==========================================================================
def _bond_graph(topology, pos_nm):
    """Covalent graph: OpenMM template/CONECT bonds + distance bonds inside
    residues without templates (ligands, modified residues)."""
    n = topology.getNumAtoms()
    adj = [set() for _ in range(n)]
    for a, b in topology.bonds():
        adj[a.index].add(b.index)
        adj[b.index].add(a.index)
    big = {'S', 'P', 'Se', 'Br', 'I', 'Cl'}
    for res in topology.residues():
        atoms = list(res.atoms())
        if len(atoms) < 2 or any(adj[a.index] for a in atoms):
            continue
        for i, a in enumerate(atoms):
            for b in atoms[i + 1:]:
                cut = 0.23 if ({getattr(a.element, 'symbol', ''), getattr(b.element, 'symbol', '')} & big) else 0.195
                if np.linalg.norm(pos_nm[a.index] - pos_nm[b.index]) < cut:
                    adj[a.index].add(b.index)
                    adj[b.index].add(a.index)
    return adj


def _within_three_bonds(adj):
    """Set of (i, j), i < j, separated by 1, 2 or 3 bonds."""
    pairs = set()
    for i in range(len(adj)):
        seen, frontier = {i}, {i}
        for _ in range(3):
            nxt = set()
            for u in frontier:
                nxt |= adj[u] - seen
            seen |= nxt
            frontier = nxt
        pairs.update((i, j) for j in seen if j > i)
    return pairs


METALS = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Mg', 'Ca', 'Sr', 'Ba', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Cd', 'Hg'}


def _close_contacts(pos_nm, excluded, cutoff_nm, subset=None, metal=None):
    tree = cKDTree(pos_nm)
    pairs = tree.query_pairs(cutoff_nm, output_type='ndarray')
    out = []
    for i, j in pairs:
        i, j = int(min(i, j)), int(max(i, j))
        if (i, j) in excluded or (metal is not None and (metal[i] or metal[j])):
            continue
        if subset is not None and i not in subset and j not in subset:
            continue
        out.append((i, j, float(np.linalg.norm(pos_nm[i] - pos_nm[j]))))
    return sorted(out, key=lambda x: x[2])


def _soft_forcefield_system(topology):
    """Bonded terms with ideal parameters from PDBFixer's heavy-atom soft force
    field (standard protein / nucleic residues; templates are generated for
    anything else). Returns an OpenMM System or None if unavailable."""
    import openmm as mm
    ff = PDBFixer._createForceField(None, topology, False)   # private PDBFixer helper
    system = ff.createSystem(topology)
    for i in reversed(range(system.getNumForces())):
        f = system.getForce(i)
        if isinstance(f, (mm.CustomNonbondedForce, mm.NonbondedForce, mm.CMMotionRemover, mm.PeriodicTorsionForce)):
            system.removeForce(i)
        # soft.xml is deliberately floppy (bonds k=1e4, angles k=10): keep its
        # ideal lengths/angles but give them realistic stiffness
        elif isinstance(f, mm.HarmonicBondForce):
            for b in range(f.getNumBonds()):
                a1, a2, r0, _ = f.getBondParameters(b)
                f.setBondParameters(b, a1, a2, r0, 2.0e5)
        elif isinstance(f, mm.HarmonicAngleForce):
            for b in range(f.getNumAngles()):
                a1, a2, a3, t0, _ = f.getAngleParameters(b)
                f.setAngleParameters(b, a1, a2, a3, t0, 400.0)
    return system


_CENTERS = {}


def _template_centers(resname):
    """{centre atom: ((n1, n2, n3), ideal improper (deg), planar?)} for every
    atom with >= 3 heavy neighbours in the ideal template of a standard residue."""
    if resname not in _CENTERS:
        out = {}
        if resname in AA1TO3.values():
            tpl = get_template(resname)
            for c, nbs in tpl.adj.items():
                if len(nbs) >= 3:
                    q = tuple(sorted(nbs))[:3]
                    t = dihedral_deg(tpl.coords[q[0]], tpl.coords[q[1]], tpl.coords[c], tpl.coords[q[2]])
                    planar = abs(t) > 160 or abs(t) < 20
                    out[c] = (q, (0.0 if abs(t) < 90 else 180.0) if planar else t, planar)
        _CENTERS[resname] = out
    return _CENTERS[resname]


def relax_clashes(pdb_in, pdb_out, built=frozenset(), scope='local', sigma=2.7, detect=2.5,
                  k_pos=1000.0, shell=1, max_iter=20000):
    """Remove close contacts with a soft-core minimisation that keeps covalent
    geometry. Heavy atoms only (hydrogens are removed).

    Energy:
      * bonds, angles, torsions with IDEAL parameters from PDBFixer's soft force
        field for standard protein/DNA/RNA residues (this also repairs
        distorted geometry, e.g. at a strained loop junction); bonds and angles
        of anything without a template are held at their current values,
      * impropers at every atom with >= 3 heavy neighbours (planar centres are
        kept planar, chiral centres keep their handedness), trans/cis peptide
        omega kept,
      * soft repulsion k (sigma - r)^2 between atoms more than 3 bonds apart --
        finite even for fully overlapping atoms, so it cannot produce NaN.
    Which atoms move (`scope`):
      built : only residues built by this script.
      local : only residues in a contact < `detect` A (plus `shell` sequence
              neighbours) move: built ones freely, deposited ones under a
              positional restraint (k_pos, kJ/mol/nm^2); everything else fixed.
      all   : every deposited atom moves under the positional restraint.
    `built` is a set of (chain_id, residue_number_string)."""
    import openmm as mm
    log(f"\n[Phase 4] Relaxing close contacts (scope: {scope}, soft radius {sigma:.2f} A)")
    t0 = time.time()
    pdb = app.PDBFile(pdb_in)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    hyd = [a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol in ('H', 'D')]
    if hyd:
        warn(f"  removing {len(hyd)} hydrogen(s) before relaxation (they would not follow their heavy atoms)")
        modeller.delete(hyd)
    top = modeller.topology
    pos = np.array(modeller.positions.value_in_unit(unit.nanometer), float)
    atoms = list(top.atoms())
    n = len(atoms)

    adj = _bond_graph(top, pos)
    excl = _within_three_bonds(adj)
    # a backbone O far from the peptide plane is a model error (e.g. a PDBFixer
    # placement): its O...N(i+1) pair is bonded-excluded, so no contact would
    # ever flag it. Put it back in the plane first.
    n_snap = 0
    for c in atoms:
        if c.name != 'C':
            continue
        own = {a.name: a.index for a in c.residue.atoms()}
        nxt = [j for j in adj[c.index] if atoms[j].name == 'N' and atoms[j].residue is not c.residue]
        if 'CA' in own and 'O' in own and nxt:
            ideal = ideal_carbonyl_O(pos[own['CA']] * 10, pos[c.index] * 10, pos[nxt[0]] * 10) / 10
            if np.linalg.norm(pos[own['O']] - ideal) > 0.1:
                pos[own['O']] = ideal
                n_snap += 1
    if n_snap:
        log(f"  moved {n_snap} misplaced backbone O (> 1 A off the peptide plane) back into place")
    sig, det = sigma / 10.0, detect / 10.0
    # metal ions coordinate at 2.0-2.4 A: never treated as clashes
    metal = np.array([a.element is not None and a.element.symbol in METALS for a in atoms])
    for x, y in top.bonds():   # OpenMM links any two Cys SG < 3 A: say so when a built Cys is involved
        if x.name == 'SG' and y.name == 'SG' and x.residue is not y.residue and \
                ((x.residue.chain.id, x.residue.id.strip()) in built or (y.residue.chain.id, y.residue.id.strip()) in built):
            log(f"  note: built residue in disulfide {x.residue.chain.id}:CYS{x.residue.id.strip()}-"
                f"{y.residue.chain.id}:CYS{y.residue.id.strip()} (SG-SG < 3 A); check it is intended")
    before = _close_contacts(pos, excl, det, metal=metal)

    res_key = [(a.residue.chain.id, a.residue.id.strip()) for a in atoms]
    is_built = np.array([k in built for k in res_key])
    free = is_built.copy() if scope in ('built', 'all') else np.zeros(n, bool)
    restrained = np.zeros(n, bool)
    if scope == 'all':
        restrained = ~is_built
    elif scope == 'local':
        chain_res = {c.index: list(c.residues()) for c in top.chains()}
        res_index = {r: k for c in chain_res.values() for k, r in enumerate(c)}
        hot = set()
        for i, j, _ in before:
            for k in (i, j):
                r = atoms[k].residue
                lst = chain_res[r.chain.index]
                idx = res_index[r]
                hot.update(lst[max(0, idx - shell): idx + shell + 1])
        in_hot = np.array([a.residue in hot for a in atoms])
        free = in_hot & is_built              # built residues near a clash move freely
        restrained = in_hot & ~is_built       # deposited ones move under restraint
    mobile = free | restrained
    log(f"  {len(before)} contact(s) < {detect:.1f} A; mobile atoms: {int(free.sum())} built + "
        f"{int(restrained.sum())} restrained deposited; {int((~mobile).sum())} fixed")
    if not mobile.any():
        with open(pdb_out, 'w') as fh:
            app.PDBFile.writeFile(top, modeller.positions, fh, keepIds=True)
        log("  nothing to relax")
        return {'before': len(before), 'after': len(before)}

    # break exact overlaps (a zero distance has no direction)
    rng = np.random.default_rng(0)
    for i, j, d in before:
        if d < 1e-3:
            k = j if mobile[j] else i
            if mobile[k]:
                pos[k] += rng.normal(0, 0.005, 3)

    # Only atoms near the mobile ones matter: build the system on that subset.
    tree = cKDTree(pos)
    near = set(np.nonzero(mobile)[0].tolist())
    for lst in tree.query_ball_point(pos[mobile], sig + 0.35):   # reach + how far atoms may move
        near.update(lst)
    keep_res = {atoms[i].residue for i in near}           # whole residues for templates
    sub_atoms = [a for a in atoms if a.residue in keep_res]
    sub = app.Modeller(top, unit.Quantity([mm.Vec3(*map(float, p)) for p in pos], unit.nanometer))
    sub.delete([a for a in atoms if a.residue not in keep_res])
    stop = sub.topology
    g2s = {a.index: k for k, a in enumerate(sub_atoms)}  # Modeller keeps atom order
    s2g = np.array([a.index for a in sub_atoms])
    m = len(s2g)
    spos = pos[s2g]
    smob = mobile[s2g]
    sadj = [set(g2s[j] for j in adj[i] if j in g2s) for i in s2g]
    sexcl = {(g2s[i], g2s[j]) if g2s[i] < g2s[j] else (g2s[j], g2s[i])
             for i, j in excl if i in g2s and j in g2s}

    ideal = True
    try:
        system = _soft_forcefield_system(stop)
    except Exception as e:
        warn(f"  ideal-geometry force field unavailable ({e}); holding current geometry instead")
        system, ideal = mm.System(), False
        for _ in range(m):
            system.addParticle(12.0)
    for k in range(m):
        system.setParticleMass(k, system.getParticleMass(k) if smob[k] else 0.0)
        if smob[k] and system.getParticleMass(k).value_in_unit(unit.dalton) == 0:
            system.setParticleMass(k, 12.0)
    touches = lambda *ix: any(smob[i] for i in ix)

    covered_b, covered_a = set(), set()
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for b in range(f.getNumBonds()):
                i, j, _, _ = f.getBondParameters(b)
                covered_b.add((min(i, j), max(i, j)))
        elif isinstance(f, mm.HarmonicAngleForce):
            for b in range(f.getNumAngles()):
                i, j, k, _, _ = f.getAngleParameters(b)
                covered_a.add((min(i, k), j, max(i, k)))
    bonds, angles = mm.HarmonicBondForce(), mm.HarmonicAngleForce()
    for i in range(m):
        for j in sadj[i]:
            if i < j and touches(i, j) and (i, j) not in covered_b:
                bonds.addBond(i, j, float(np.linalg.norm(spos[i] - spos[j])), 2.0e5)
        nb = sorted(sadj[i])
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                a, c = nb[x], nb[y]
                if touches(a, i, c) and (a, i, c) not in covered_a:
                    v1, v2 = spos[a] - spos[i], spos[c] - spos[i]
                    th = math.acos(max(-1.0, min(1.0, v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))))
                    angles.addAngle(a, i, c, th, 400.0)
    system.addForce(bonds)
    system.addForce(angles)

    tors = mm.CustomTorsionForce("0.5*kt*d^2; d = dt - 6.283185307*floor((dt + 3.141592654)/6.283185307); dt = theta - t0")
    tors.addPerTorsionParameter('t0')
    tors.addPerTorsionParameter('kt')
    satoms = list(stop.atoms())

    def add_tors(a, b, c, d, target, k):
        if touches(a, b, c, d):
            tors.addTorsion(a, b, c, d, [math.radians(target), k])
    inverted = []
    for i in range(m):
        nb = sorted(sadj[i])
        if len(nb) < 3:
            continue
        a = satoms[i]
        tpl = _template_centers(a.residue.name)
        nbn = {satoms[x].name: x for x in nb}
        if a.name in tpl and all(q in nbn for q in tpl[a.name][0]):
            # standard residue: handedness / planarity from the ideal template
            (q1, q2, q3), target, planar = tpl[a.name]
            quad = (nbn[q1], nbn[q2], i, nbn[q3])
            t = dihedral_deg(*(spos[x] for x in quad))
            if not planar and abs(wrap_deg(t - target)) > 90:
                # inverted centre in the INPUT: a minimiser cannot carry an
                # inversion through, and forcing it jams everything else. Keep
                # the handedness, report it.
                target = -target
                if smob[i]:
                    inverted.append(f"{a.residue.chain.id}:{a.residue.name}{a.residue.id.strip()}.{a.name}")
            add_tors(*quad, target, 1000.0 if planar else 20000.0)
            continue
        t = dihedral_deg(spos[nb[0]], spos[nb[1]], spos[i], spos[nb[2]])
        if abs(t) > 160 or abs(t) < 20:          # other planar centre: keep it planar
            add_tors(nb[0], nb[1], i, nb[2], 0.0 if abs(t) < 90 else 180.0, 1000.0)
        else:                                    # other chiral centre: keep its handedness
            add_tors(nb[0], nb[1], i, nb[2], t, 20000.0)

    sbuilt = is_built[s2g]
    n_cis = 0
    for r in stop.residues():                           # peptide omega: trans (or cis) exactly
        names = {a.name: a.index for a in r.atoms()}
        if 'C' in names and 'CA' in names:
            for nbr in sadj[names['C']]:
                if satoms[nbr].name == 'N' and satoms[nbr].residue is not r:
                    other = {a.name: a.index for a in satoms[nbr].residue.atoms()}
                    if 'CA' in other:
                        w = dihedral_deg(spos[names['CA']], spos[names['C']], spos[nbr], spos[other['CA']])
                        target = 0.0 if abs(w) < 90 else 180.0
                        if target == 0.0 and (sbuilt[names['CA']] or sbuilt[other['CA']]):
                            target = 180.0              # built peptide bonds should be trans
                            n_cis += bool(smob[names['CA']] or smob[other['CA']])
                        add_tors(names['CA'], names['C'], nbr, other['CA'], target, 1000.0)
    system.addForce(tors)

    rep = mm.CustomNonbondedForce("on1*on2*krep*(sig - r)^2*step(sig - r)")
    rep.addGlobalParameter('krep', 1.0e4)
    rep.addGlobalParameter('sig', sig)
    rep.addPerParticleParameter('on')
    rep.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    rep.setCutoffDistance(sig)
    smetal = metal[s2g]
    for k in range(m):
        rep.addParticle([0.0 if smetal[k] else 1.0])
    for i, j in sexcl:
        rep.addExclusion(i, j)
    rep.addInteractionGroup([int(i) for i in np.nonzero(smob)[0]], list(range(m)))
    system.addForce(rep)

    srest = restrained[s2g]
    if srest.any():
        posr = mm.CustomExternalForce("0.5*kpos*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        posr.addGlobalParameter('kpos', k_pos)
        for p in ('x0', 'y0', 'z0'):
            posr.addPerParticleParameter(p)
        for i in np.nonzero(srest)[0]:
            posr.addParticle(int(i), [float(v) for v in spos[i]])
        system.addForce(posr)

    ctx = mm.Context(system, mm.VerletIntegrator(0.001))
    ctx.setPositions([mm.Vec3(*map(float, p)) for p in spos])
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, max_iter)
    st = ctx.getState(getPositions=True, getEnergy=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    snew = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer), float)
    if not np.isfinite(snew).all():
        raise RuntimeError("relaxation produced non-finite coordinates")
    newpos = pos.copy()
    newpos[s2g[smob]] = snew[smob]

    after = _close_contacts(newpos, excl, det, metal=metal)
    shift = np.linalg.norm(newpos - pos, axis=1) * 10
    with open(pdb_out, 'w') as fh:
        app.PDBFile.writeFile(top, unit.Quantity([mm.Vec3(*map(float, p)) for p in newpos], unit.nanometer),
                              fh, keepIds=True)

    def fmt(c):
        i, j, d = c
        a, b = atoms[i], atoms[j]
        return (f"{a.residue.chain.id}:{a.residue.name}{a.residue.id.strip()}.{a.name} - "
                f"{b.residue.chain.id}:{b.residue.name}{b.residue.id.strip()}.{b.name} {d * 10:.2f} A")
    log(f"  {'ideal' if ideal else 'current'} covalent geometry; {m} of {n} atoms in the relaxed region"
        f""
        f"{f'; {n_cis} built cis peptide(s) restrained to trans' if n_cis else ''}")
    if inverted:
        warn(f"  WARNING: {len(inverted)} inverted chiral centre(s) in the input were kept as they are "
             f"(minimisation cannot invert them; rebuild those residues): {', '.join(inverted[:12])}"
             f"{' ...' if len(inverted) > 12 else ''}")
    log(f"  energy {e0:.4g} -> {e1:.4g} kJ/mol in {time.time() - t0:.1f} s")
    log(f"  contacts < {detect:.1f} A: {len(before)} -> {len(after)}"
        f"{'  (closest before: ' + fmt(before[0]) + ')' if before else ''}")
    for c in after[:5]:
        log(f"    remaining: {fmt(c)}")
    if restrained.any():
        log(f"  deposited atoms moved: max {shift[restrained].max():.2f} A, "
            f"RMS {math.sqrt((shift[restrained] ** 2).mean()):.2f} A")
    if free.any():
        log(f"  built atoms moved: max {shift[free].max():.2f} A")
    return {'before': len(before), 'after': len(after)}


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdb_id', nargs='?', help='PDB ID (Phase 1 builds assembly + .ali). Omit for --pdb/--ali mode.')

    p1 = ap.add_argument_group('Phase 1: assembly + alignment (pdb_id mode)')
    p1.add_argument('--cif', help='use this local mmCIF instead of downloading')
    p1.add_argument('--assembly-id', default=None)
    p1.add_argument('--asu', action='store_true', help='use the asymmetric unit instead of an assembly')
    p1.add_argument('--list-assemblies', action='store_true')
    p1.add_argument('--nucleic-mode', choices=['copy', 'drop'], default='copy')
    p1.add_argument('--workdir', default='.')

    man = ap.add_argument_group('manual mode')
    man.add_argument('--pdb', help='existing PDB file')
    man.add_argument('--ali', help='MODELLER-style .ali for --pdb')
    man.add_argument('--chains', help='comma-separated chain IDs in .ali order (default: protein chains in file order)')

    ap.add_argument('--out', help='output PDB (default <pdbid>_<assembly|asu>_filled.pdb)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1,
                    help='parallel workers for independent tails (default: all CPUs)')
    ap.add_argument('--keep-intermediate', action='store_true', help='keep the Phase-2 (loops only) PDB')

    p2 = ap.add_argument_group('Phase 2: missing atoms of deposited residues (PDBFixer)')
    p2.add_argument('--no-crop', action='store_true', help='run PDBFixer on the whole structure (slow)')
    p2.add_argument('--crop-margin', type=float, default=10.0, help='neighbourhood kept around each incomplete residue (A)')

    p3 = ap.add_argument_group('Phase 3: loops and tails')
    p3.add_argument('--tail-mode', choices=['extended', 'default'], default='extended',
                    help='backbone preference for tails (loops always use the default basins)')
    p3.add_argument('--hinge', type=int, default=1,
                    help='deposited residues next to each gap whose phi/psi the MC / closure may bend, '
                         'restrained to their deposited values (0 = strictly rigid anchors)')
    p3.add_argument('--hinge-cap', type=float, default=30.0, help='max hinge torsion change (deg)')
    p3.add_argument('--hinge-k', type=float, default=40.0, help='hinge restraint strength (per rad^2)')
    p3.add_argument('--closure-tol', type=float, default=0.08, help='loop closure RMSD tolerance (A)')
    p3.add_argument('--ccd-sweeps', type=int, default=300, help='max CCD sweeps per closure')
    p3.add_argument('--clash-radius', type=float, default=2.2, help='heavy-atom clash distance (A)')
    p3.add_argument('--cb-scale', type=float, default=1.0, help='scale for the CB side-chain room radii')
    p3.add_argument('--attempts', type=int, default=8, help='full growth restarts per loop/tail')
    p3.add_argument('--candidates', type=int, default=12, help='torsion candidates per sampling round')
    p3.add_argument('--retries', type=int, default=40, help='sampling rounds per residue before backtracking')
    p3.add_argument('--backtrack', type=int, default=3, help='max residues to step back on a dead end')
    p3.add_argument('--mc-steps', type=int, default=5000)
    p3.add_argument('--mc-t-start', type=float, default=1.0)
    p3.add_argument('--mc-t-end', type=float, default=0.01)
    p3.add_argument('--mc-sigma', type=float, default=15.0, help='torsion step size (deg)')
    p3.add_argument('--w-rama', type=float, default=0.1, help='Ramachandran prior weight in the MC energy')
    p3.add_argument('--w-bulk', type=float, default=1.0, help='weight of the soft CB side-chain-room term')
    p3.add_argument('--ignore-water', action='store_true', help='do not treat waters as rigid background')
    p3.add_argument('--rotamer-budget', type=int, default=DEFAULT_ROTAMER_BUDGET)

    p4 = ap.add_argument_group('Phase 4: clash relaxation')
    p4.add_argument('--relax', choices=['none', 'local', 'built', 'all'], default='none',
                    help='none: no minimisation (default); local: only residues in close contacts move -- built '
                         'ones freely, deposited ones restrained; built: only built residues move; all: every '
                         'atom, deposited restrained')
    p4.add_argument('--relax-only', metavar='PDB',
                    help='only relax this existing PDB (needs --out; uses --relax, or local if --relax is none); '
                         'skips Phases 1-3')
    p4.add_argument('--relax-sigma', type=float, default=2.7, help='soft repulsion radius (A)')
    p4.add_argument('--relax-detect', type=float, default=2.5, help='contact distance that counts as a clash (A)')
    p4.add_argument('--relax-k', type=float, default=1000.0,
                    help='positional restraint on deposited atoms (kJ/mol/nm^2)')
    p4.add_argument('--relax-shell', type=int, default=1,
                    help='sequence neighbours of a clashing residue that may also move')
    args = ap.parse_args()
    t_all = time.time()

    if args.relax_only:
        if not args.out:
            ap.error("--relax-only needs --out")
        relax_clashes(args.relax_only, args.out, scope='local' if args.relax in ('built', 'none') else args.relax,
                      sigma=args.relax_sigma, detect=args.relax_detect, k_pos=args.relax_k, shell=args.relax_shell)
        log(f"Wrote {args.out}")
        return

    # ---------------- Phase 1
    chain_ids = None
    if args.pdb_id:
        pdb_id = args.pdb_id.upper()
        if args.list_assemblies:
            prepare_from_pdb_id(pdb_id, args.workdir, cif_path=args.cif, list_only=True)
            return
        log("[Phase 1] Building assembly and alignment")
        pdb_path, ali_path, chain_ids = prepare_from_pdb_id(
            pdb_id, args.workdir, args.assembly_id, args.asu, args.nucleic_mode, cif_path=args.cif)
        out_path = args.out or os.path.join(args.workdir, f"{pdb_id.lower()}_{'asu' if args.asu else 'assembly'}_filled.pdb")
    elif args.pdb and args.ali:
        pdb_path, ali_path = args.pdb, args.ali
        chain_ids = args.chains.split(',') if args.chains else None
        if not args.out:
            ap.error("--out is required in manual mode")
        out_path = args.out
        log("[Phase 1] Skipped (manual --pdb/--ali)")
    else:
        ap.error("provide a pdb_id, or both --pdb and --ali")

    structure = PDBParser(QUIET=True).get_structure('input', pdb_path)
    template_rec, target_rec = pick_template_and_target(parse_ali(ali_path))
    segments = build_missing_segments(structure, template_rec, target_rec, chain_ids)
    counts = {k: sum(s.kind == k for v in segments.values() for s in v) for k in ('loop', 'N-tail', 'C-tail')}
    log(f"  gaps: {counts['loop']} internal loop(s), {counts['N-tail']} N-tail(s), {counts['C-tail']} C-tail(s)")

    # ---------------- Phase 2
    out_dir = os.path.dirname(os.path.abspath(out_path))
    stem = os.path.splitext(os.path.basename(out_path))[0]
    completed_pdb = os.path.join(out_dir, stem + '_completed.pdb')
    added_O = complete_missing_atoms(pdb_path, segments, completed_pdb, seed=args.seed,
                                     crop=not args.no_crop, margin=args.crop_margin)
    n_fix, n_drop = repair_backbone_oxygens(completed_pdb, added_O)
    if n_fix or n_drop:
        log(f"  rebuilt {n_fix} PDBFixer-placed backbone O in the peptide plane"
            f"{f'; {n_drop} O next to a gap left to Phase 3' if n_drop else ''}")

    # ---------------- Phase 3
    built_pdb = os.path.join(out_dir, stem + '_unrelaxed.pdb') if args.relax != 'none' else out_path
    report, final, built_segs = build_segments(completed_pdb, segments, built_pdb, args)
    loops = [s for s in built_segs if s.kind == 'loop']
    bad_loops = report_loop_closure(final, loops)
    if not args.keep_intermediate and os.path.exists(completed_pdb):
        os.remove(completed_pdb)

    # ---------------- Phase 4
    if args.relax != 'none':
        built = {(sg.chain_id, str(sg.start_resnum + i)) for sg in built_segs for i in range(len(sg.aa1_list))}
        relax_clashes(built_pdb, out_path, built, scope=args.relax, sigma=args.relax_sigma,
                      detect=args.relax_detect, k_pos=args.relax_k, shell=args.relax_shell)
        if not args.keep_intermediate:
            os.remove(built_pdb)

    flagged = [r for r in report if r['backbone_worst_A'] > REPORT_TOL or r['sidechain_residual_A'] > REPORT_TOL
               or r['closure_A'] > args.closure_tol]
    n_tails = len(built_segs) - len(loops)
    log(f"\nDone in {time.time() - t_all:.1f} s: {len(loops)} loop(s) and {n_tails} tail(s) built"
        f"{f', {len(flagged)} with residual overlap or closure gap before relaxation' if flagged else ''}"
        f"{f', {bad_loops} loop(s) with a distorted junction' if bad_loops else ''}")
    log(f"Wrote {out_path}")


if __name__ == '__main__':
    main()