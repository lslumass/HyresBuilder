"""
PSF file generation for HyRes and iConRNA coarse-grained systems.

This module constructs CHARMM-style PSF topology files from coarse-grained
PDB structures produced by the HyRes (protein) and iConRNA (RNA) force fields.
It handles mixed systems containing any combination of protein, RNA, DNA,
Mg²⁺, and Ca²⁺ chains in a single input PDB, automatically detecting molecule
types by chain identity and assigning structured segment IDs before invoking
``psfgen`` to build and write the topology.

Workflow
--------
1. Parse the input CG PDB and split it into per-chain temporary files,
   detecting molecule type from residue names (:func:`split_chains`).
2. Assign segment IDs following the convention below and register each chain
   with ``psfgen`` using the appropriate force-field topology (:func:`genpsf`).
3. Optionally set terminus charge states for protein chains
   (:func:`set_terminus`).
4. Write the PSF file and remove all intermediate temporary PDB files.

Segment ID convention
---------------------
Segment IDs are four characters: a single type prefix followed by a
three-character hybrid-36 counter encoded by :func:`encode_segid`.

========  ===========  ===============
Prefix    Molecule     Example IDs
========  ===========  ===============
``P``     Protein      P001, P002, …
``R``     RNA          R001, R002, …
``D``     DNA          D001, D002, …
``I``     Mg²⁺,Ca²⁺    I001, I002, …
``S``     PolyP, PEG   S001, S002, …
``AGs``   Antibiotics  AGs001, AGs002, …
``M``     Metabolites  M001, M002, …
========  ===========  ===============

The hybrid-36 counter supports up to 68,391 chains per molecule type before
overflowing.

A command-line interface is exposed via :func:`main` and registered as the
``GenPsf`` entry point.

Dependencies
------------
* `psfgen <https://github.com/MDAnalysis/psfgen>`_ (``psfgen.PsfGen``)
* HyresBuilder force-field topology files, loaded via ``utils.load_ff``.
"""

from __future__ import annotations
import re
import argparse
import os
import glob
import tempfile
import shutil
import sys
from importlib.resources import files
from psfgen import PsfGen
from HyresBuilder import utils

# ===========================================================================
# SECTION 1: PSF Fast Replication Engine
# ===========================================================================

class PSF:
    """In-memory representation of a single PSF file's contents."""
    def __init__(self):
        self.flags = []
        self.title_lines = []
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.donors = []
        self.acceptors = []
        self.nnb = []
        self.nnb_label = "NNB"
        self.groups = []
        self.ngrp_nst2 = 0
        self.crossterms = []
        self.section_order = []

    @property
    def natom(self):
        return len(self.atoms)

def _read_int_block(lines, i, n_ints):
    vals = []
    while len(vals) < n_ints:
        vals.extend(int(x) for x in lines[i].split())
        i += 1
    assert len(vals) == n_ints, f"expected {n_ints} ints, got {len(vals)}"
    return vals, i

_SECTION_RE = re.compile(r"!([A-Z0-9:]+)")

def parse_psf(path: str) -> PSF:
    with open(path) as f:
        lines = f.read().splitlines()

    i = 0
    header_tokens = lines[i].split()
    if not header_tokens or header_tokens[0] != "PSF":
        raise ValueError(f"{path}: does not start with 'PSF' header")
    psf = PSF()
    psf.flags = header_tokens[1:]
    i += 1

    while lines[i].strip() == "":
        i += 1
    ntitle = int(lines[i].split()[0])
    i += 1
    psf.title_lines = lines[i:i + ntitle]
    i += ntitle

    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        m = _SECTION_RE.search(line)
        if not m:
            i += 1
            continue

        tag = m.group(1).split(":")[0]
        tokens = line.split()
        count = int(tokens[0])
        i += 1

        if tag == "NATOM":
            psf.section_order.append("NATOM")
            atoms = []
            for _ in range(count):
                atoms.append(lines[i].split())
                i += 1
            psf.atoms = atoms
        elif tag == "NBOND":
            psf.section_order.append("NBOND")
            vals, i = _read_int_block(lines, i, count * 2)
            psf.bonds = list(zip(vals[0::2], vals[1::2]))
        elif tag == "NTHETA":
            psf.section_order.append("NTHETA")
            vals, i = _read_int_block(lines, i, count * 3)
            psf.angles = [tuple(vals[k:k + 3]) for k in range(0, len(vals), 3)]
        elif tag == "NPHI":
            psf.section_order.append("NPHI")
            vals, i = _read_int_block(lines, i, count * 4)
            psf.dihedrals = [tuple(vals[k:k + 4]) for k in range(0, len(vals), 4)]
        elif tag == "NIMPHI":
            psf.section_order.append("NIMPHI")
            vals, i = _read_int_block(lines, i, count * 4)
            psf.impropers = [tuple(vals[k:k + 4]) for k in range(0, len(vals), 4)]
        elif tag == "NDON":
            psf.section_order.append("NDON")
            vals, i = _read_int_block(lines, i, count * 2)
            psf.donors = [tuple(vals[k:k + 2]) for k in range(0, len(vals), 2)]
        elif tag == "NACC":
            psf.section_order.append("NACC")
            vals, i = _read_int_block(lines, i, count * 2)
            psf.acceptors = [tuple(vals[k:k + 2]) for k in range(0, len(vals), 2)]
        elif tag == "NNB":
            psf.section_order.append("NNB")
            psf.nnb_label = m.group(1)
            vals, i = _read_int_block(lines, i, count) if count else ([], i)
            psf.nnb = vals
        elif tag == "NGRP":
            psf.section_order.append("NGRP")
            ngrp = count
            nst2 = int(tokens[1]) if len(tokens) > 1 and not tokens[1].startswith("!") else 0
            psf.ngrp_nst2 = nst2
            vals, i = _read_int_block(lines, i, ngrp * 3) if ngrp else ([], i)
            psf.groups = [tuple(vals[k:k + 3]) for k in range(0, len(vals), 3)]
        elif tag == "NCRTERM":
            psf.section_order.append("NCRTERM")
            vals, i = _read_int_block(lines, i, count * 8) if count else ([], i)
            psf.crossterms = [tuple(vals[k:k + 8]) for k in range(0, len(vals), 8)]
        else:
            raise NotImplementedError(f"{path}: unsupported PSF section '!{tag}'")

    return psf

def _offset(v, by):
    return v if v == 0 else v + by

def replicate_segment(template: PSF, n_copies: int, segid_for_copy, start_offset: int = 0):
    natom = template.natom
    for c in range(n_copies):
        atom_offset = start_offset + c * natom
        new_segid = segid_for_copy(c)

        atoms = []
        for tok in template.atoms:
            tok = list(tok)
            tok[0] = str(int(tok[0]) + atom_offset) 
            tok[1] = new_segid                       
            atoms.append(tok)

        bonds = [tuple(_offset(v, atom_offset) for v in b) for b in template.bonds]
        angles = [tuple(_offset(v, atom_offset) for v in a) for a in template.angles]
        dihedrals = [tuple(_offset(v, atom_offset) for v in d) for d in template.dihedrals]
        impropers = [tuple(_offset(v, atom_offset) for v in d) for d in template.impropers]
        donors = [tuple(_offset(v, atom_offset) for v in d) for d in template.donors]
        acceptors = [tuple(_offset(v, atom_offset) for v in d) for d in template.acceptors]
        nnb = [_offset(v, atom_offset) for v in template.nnb]
        groups = [(g[0] + atom_offset, g[1], g[2]) for g in template.groups]
        crossterms = [tuple(_offset(v, atom_offset) for v in ct) for ct in template.crossterms]

        yield dict(
            atoms=atoms, bonds=bonds, angles=angles, dihedrals=dihedrals,
            impropers=impropers, donors=donors, acceptors=acceptors,
            nnb=nnb, groups=groups, crossterms=crossterms,
        )

def _write_int_section(out, label, count, flat_ints, per_line):
    out.write(f"{count:>8d} !{label}\n")
    for k in range(0, len(flat_ints), per_line):
        out.write("".join(f"{v:>8d}" for v in flat_ints[k:k + per_line]))
        out.write("\n")
    out.write("\n")

def write_merged_psf(out_path, title_lines, flags, atom_blocks, bond_blocks,
                      angle_blocks, dihedral_blocks, improper_blocks,
                      donor_blocks, acceptor_blocks, nnb_blocks, nnb_label,
                      group_blocks, ngrp_nst2, crossterm_blocks):
    """Write a single merged PSF from lists of per-segment-group blocks."""

    natom = sum(len(b) for b in atom_blocks)
    nbond = sum(len(b) for b in bond_blocks)
    ntheta = sum(len(b) for b in angle_blocks)
    nphi = sum(len(b) for b in dihedral_blocks)
    nimphi = sum(len(b) for b in improper_blocks)
    ndon = sum(len(b) for b in donor_blocks)
    nacc = sum(len(b) for b in acceptor_blocks)
    nnb_total = sum(len(b) for b in nnb_blocks)
    ncrterm = sum(len(b) for b in crossterm_blocks)

    # Automatically promote format to EXT if atom count exceeds standard limits
    is_ext = natom > 99999
    if is_ext and "EXT" not in flags:
        flags.append("EXT")

    with open(out_path, "w") as out:
        header = "PSF"
        if flags:
            header += " " + " ".join(flags)
        out.write(header + "\n\n")
        out.write(f"{len(title_lines):>8d} !NTITLE\n")
        for t in title_lines:
            out.write(t + "\n")
        out.write("\n")

        out.write(f"{natom:>8d} !NATOM\n")
        for block in atom_blocks:
            for tok in block:
                # Reconstruct an atom line using strictly enforced standard/extended column layout.
                if is_ext:
                    out.write(
                        f"{int(tok[0]):10d} {tok[1]:<8s} {tok[2]:<8s} {tok[3]:<8s} "
                        f"{tok[4]:<8s} {tok[5]:<6s} {float(tok[6]):10.6f} "
                        f"{float(tok[7]):14.4f} {int(tok[8]):11d}\n"
                    )
                else:
                    out.write(
                        f"{int(tok[0]):8d} {tok[1]:<4s} {tok[2]:<4s} {tok[3]:<4s} "
                        f"{tok[4]:<4s} {tok[5]:<4s} {float(tok[6]):10.6f} "
                        f"{float(tok[7]):13.4f} {int(tok[8]):11d}\n"
                    )
        out.write("\n")

        flat = [v for block in bond_blocks for pair in block for v in pair]
        _write_int_section(out, "NBOND: bonds", nbond, flat, 8)

        flat = [v for block in angle_blocks for tri in block for v in tri]
        _write_int_section(out, "NTHETA: angles", ntheta, flat, 9)

        flat = [v for block in dihedral_blocks for q in block for v in q]
        _write_int_section(out, "NPHI: dihedrals", nphi, flat, 8)

        flat = [v for block in improper_blocks for q in block for v in q]
        _write_int_section(out, "NIMPHI: impropers", nimphi, flat, 8)

        flat = [v for block in donor_blocks for pair in block for v in pair]
        _write_int_section(out, "NDON: donors", ndon, flat, 8)

        flat = [v for block in acceptor_blocks for pair in block for v in pair]
        _write_int_section(out, "NACC: acceptors", nacc, flat, 8)

        flat = [v for block in nnb_blocks for v in block]
        out.write(f"{nnb_total:>8d} !{nnb_label}\n")
        for k in range(0, len(flat), 8):
            out.write("".join(f"{v:>8d}" for v in flat[k:k + 8]))
            out.write("\n")
        out.write("\n")

        # --- HARDCODED 1 NGRP BLOCK ---
        out.write(f"{1:>8d} {0:>8d} !NGRP NST2\n")
        out.write("       0       0       0\n\n")

        if ncrterm:
            flat = [v for block in crossterm_blocks for ct in block for v in ct]
            _write_int_section(out, "NCRTERM: cross-terms", ncrterm, flat, 8)

# ===========================================================================
# SECTION 2: Main PSF Generation Logic
# ===========================================================================

aas = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
rnas = ["ADE", "GUA", "CYT", "URA", "A", "G", "C", "U"]
dnas = ["DAD", "DCY", "DTH", "DA", "DG", "DC", "DT"]
ions = ["MG+", "SMG", "CA+"]
polymer = ['PHO', 'PEG']
AGs = ['KAN']
metabolites = ['UN1', 'AYA', 'ACA', 'NLG', 'C3C', 'C4C', 'C5C', 'Y52', 'CHT', 'CIT',
               'CTT', 'ABU', 'CH5', 'GSH', 'MTA', 'SHR', 'TAU', 'BET', '3PG', 'G6P',
               'COA', 'FAD', 'NCA', 'PAU', 'ADN', 'ADP', 'AMP', 'ATP', 'C5P', 'CTN',
               'UGA', 'GMP', 'UD1', 'NOS', 'NAD', 'NAI', 'NAD', 'UDP', 'U5P', 'UPG',
               '2PG', '13P', 'PEP', 'SAM', '2HG', 'FUM', 'AKG', 'LMR', 'MCT', 'SIN', 'DGU']

segtypes = ['P', 'R', 'D', 'I', 'S', 'AGs', 'M']

def get_type(resname):
    chaintype = (
        'P' if resname in aas else
        'R' if resname in rnas else
        'D' if resname in dnas else
        'I' if resname in ions else
        'S' if resname in polymer else
        'AGs' if resname in AGs else
        'M' if resname in metabolites else
        None
    )
    return chaintype

def split_chains(pdb):
    currentKey = None
    atoms = []
    chains = []
    types = []
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chainid = line[21]
                segid = line[72:76].strip()
                resname = line[17:20].strip()
                key = (chainid, segid)

                if key != currentKey:
                    if atoms:
                        chains.append(atoms)
                    currentKey = key
                    types.append(get_type(resname))
                    atoms = [line]
                else:
                    atoms.append(line)
        if atoms:
            chains.append(atoms)

    pre_type = None
    for i, (t, chain) in enumerate(zip(types, chains)):
        if t in ['P', 'R', 'D', 'S', 'AGs', 'M']:
            tmp_pdb = f"psfgentmp_{i}.pdb"
        elif t in ['I']:
            if t == pre_type:
                continue
            else:
                tmp_pdb = f"psfgentmp_{t}.pdb"
        else:
            print('Unknown molecule type')
            exit(1)
        pre_type = t

        with open(tmp_pdb, 'w') as f:
            for line in chain:
                f.write(line)
            f.write('END\n')
    return types

def set_terminus(gen, segid, charge_status):
    if segid.startswith("P"):
        nter, cter = gen.get_resids(segid)[0], gen.get_resids(segid)[-1]
        if charge_status == 'charged':
            gen.set_charge(segid, nter, "N", 1.00)
            gen.set_charge(segid, cter, "O", -1.00)
        elif charge_status == 'NT':
            gen.set_charge(segid, nter, "N", 1.00)
        elif charge_status == 'CT':
            gen.set_charge(segid, cter, "O", -1.00)
        elif charge_status == 'positive':
            gen.set_charge(segid, nter, "N", -1.00)
            gen.set_charge(segid, cter, "O", -1.00)
        else:
            print("Error: Only 'neutral', 'charged', 'NT', and 'CT' charge status are supported.")
            exit(1)

def encode_segid(n: int) -> str:
    BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    LEAD_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n > 68391:
        return str(n)
    if n < 1000:
        return f"{n:03d}"
    n -= 1000
    lead = LEAD_CHARS[n // 1296]
    rem = n % 1296
    return f"{lead}{BASE36[rem // 36]}{BASE36[rem % 36]}"

def genpsf(pdb_in, psf_out, terminal='neutral', RNA='mix'):
    if RNA == 'mix':
        RNA_topology, _ = utils.load_ff('RNA')
    elif RNA == 'icon':
        path1 = files("HyresBuilder") / "forcefield" / "top_RNA.inp"
        RNA_topology = path1.as_posix()
    protein_topology, _ = utils.load_ff('Protein')
    AGs_topology, _ = utils.load_ff('AGs')
    Mats_topology, _ = utils.load_ff('Metabolite')

    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)
    gen.read_topology(AGs_topology)
    gen.read_topology(Mats_topology)

    counts = {'P': 1, 'R': 1, 'D': 1, 'I': 1, 'S': 1, 'AGs': 1, 'M': 1}
    types = split_chains(pdb_in)
    for i, t in enumerate(types):
        if t in ["P", "R", "D", "S", "AGs", "M"]:
            tmp_pdb = f"psfgentmp_{i}.pdb"
        else:
            tmp_pdb = f"psfgentmp_{t}.pdb"

        segid = f"{t}{encode_segid(counts[t])}"
        counts[t] += 1
        if t == 'P':
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False)
        else:
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False, auto_dihedrals=False)

    for segid in gen.get_segids():
        if terminal != "neutral":
            set_terminus(gen, segid, terminal)

    gen.write_psf(filename=psf_out)
    for file_path in glob.glob("psfgentmp_*.pdb"):
        os.remove(file_path)

def custom_genpsf(pdb_list, num_list, psf_out, terminal='neutral', RNA='mix'):
    if RNA == 'mix':
        RNA_topology, _ = utils.load_ff('RNA')
    elif RNA == 'icon':
        path1 = files("HyresBuilder") / "forcefield" / "top_RNA.inp"
        RNA_topology = path1.as_posix()
    protein_topology, _ = utils.load_ff('Protein')
    AGs_topology, _ = utils.load_ff('AGs')
    Mats_topology, _ = utils.load_ff('Metabolite')

    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)
    gen.read_topology(AGs_topology)
    gen.read_topology(Mats_topology)

    for pdb, num in zip(pdb_list, num_list):
        num = int(num)
        with open(pdb, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    resname = line[17:20].strip()
                    chaintype = get_type(resname)
                    if chaintype is None:
                        print(f"Unknown molecule type for residue {resname} in file {pdb}")
                        exit(1)
                    elif chaintype == 'P':
                        for i in range(num):
                            segid = f"{chaintype}{encode_segid(i+1)}"
                            gen.add_segment(segid=segid, pdbfile=pdb, auto_angles=False)
                    elif chaintype == 'S':
                        for i in range(num):
                            segid = f"{chaintype}{encode_segid(i+1)}"
                            gen.add_segment(segid=segid, pdbfile=pdb)
                    else:
                        for i in range(num):
                            segid = f"{chaintype}{encode_segid(i+1)}"
                            gen.add_segment(segid=segid, pdbfile=pdb, auto_angles=False, auto_dihedrals=False)
                    break 

    for segid in gen.get_segids():
        if terminal != "neutral":
            set_terminus(gen, segid, terminal)

    gen.write_psf(filename=psf_out)

def _apply_terminus_to_template(atoms, charge_status):
    resid_order = []
    for a in atoms:
        if a[2] not in resid_order:
            resid_order.append(a[2])
    nter, cter = resid_order[0], resid_order[-1]

    def set_charge(resid, atomname, charge):
        hit = False
        for a in atoms:
            if a[2] == resid and a[4] == atomname:
                a[6] = f"{charge:.6f}"
                hit = True
        if not hit:
            print(f"Warning: terminus atom '{atomname}' not found in resid {resid}")

    if charge_status == 'charged':
        set_charge(nter, "N", 1.00)
        set_charge(cter, "O", -1.00)
    elif charge_status == 'NT':
        set_charge(nter, "N", 1.00)
    elif charge_status == 'CT':
        set_charge(cter, "O", -1.00)
    elif charge_status == 'positive':
        set_charge(nter, "N", -1.00)
        set_charge(cter, "O", -1.00)
    else:
        print("Error: Only 'neutral', 'charged', 'NT', and 'CT' charge status are supported.")
        exit(1)

def custom_genpsf_fast(pdb_list, num_list, psf_out, terminal='neutral', RNA='mix', verbose=True):
    if RNA == 'mix':
        RNA_topology, _ = utils.load_ff('RNA')
    elif RNA == 'icon':
        path1 = files("HyresBuilder") / "forcefield" / "top_RNA.inp"
        RNA_topology = path1.as_posix()
    protein_topology, _ = utils.load_ff('Protein')
    AGs_topology, _ = utils.load_ff('AGs')
    Mats_topology, _ = utils.load_ff('Metabolite')

    atom_blocks, bond_blocks, angle_blocks = [], [], []
    dihedral_blocks, improper_blocks = [], []
    donor_blocks, acceptor_blocks, nnb_blocks = [], [], []
    group_blocks, crossterm_blocks = [], []
    nnb_label = "NNB"
    title_lines, flags = None, None
    global_offset = 0

    counts = {'P': 0, 'R': 0, 'D': 0, 'I': 0, 'S': 0, 'AGs': 0, 'M': 0}

    workdir = tempfile.mkdtemp(prefix="genpsf_fast_")
    try:
        for pdb, num in zip(pdb_list, num_list):
            num = int(num)
            if num <= 0:
                continue

            chaintype = None
            with open(pdb, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        resname = line[17:20].strip()
                        chaintype = get_type(resname)
                        break
            if chaintype is None:
                print(f"Unknown molecule type for residue in file {pdb}")
                exit(1)

            gen = PsfGen()
            gen.read_topology(RNA_topology)
            gen.read_topology(protein_topology)
            gen.read_topology(AGs_topology)
            gen.read_topology(Mats_topology)

            tmpl_segid = f"{chaintype}{encode_segid(counts[chaintype] + 1)}"
            if chaintype == 'P':
                gen.add_segment(segid=tmpl_segid, pdbfile=pdb, auto_angles=False)
            elif chaintype == 'S':
                gen.add_segment(segid=tmpl_segid, pdbfile=pdb)
            else:
                gen.add_segment(segid=tmpl_segid, pdbfile=pdb, auto_angles=False, auto_dihedrals=False)

            if verbose:
                print(f"[fast] built template for {pdb} (type {chaintype}); replicating x{num}...")

            tmpl_psf_path = os.path.join(workdir, f"tmpl_{chaintype}_{os.path.basename(pdb)}.psf")
            gen.write_psf(filename=tmpl_psf_path)
            del gen 

            tmpl = parse_psf(tmpl_psf_path)
            
            if title_lines is None:
                title_lines = [line for line in tmpl.title_lines if "REMARKS segment" not in line]
                flags = tmpl.flags
            nnb_label = tmpl.nnb_label

            remark_template = f" REMARKS segment {{segid}} {{ first none; last none; auto none  }}"
            for line in tmpl.title_lines:
                if line.startswith(f" REMARKS segment {tmpl_segid}"):
                    remark_template = line.replace(f" {tmpl_segid} ", " {segid} ")
                    break

            if chaintype == 'P' and terminal != 'neutral':
                _apply_terminus_to_template(tmpl.atoms, terminal)

            def segid_for(c, chaintype=chaintype, start_idx=counts[chaintype]):
                return f"{chaintype}{encode_segid(start_idx + c + 1)}"

            for rep in replicate_segment(tmpl, num, segid_for, start_offset=global_offset):
                atom_blocks.append(rep["atoms"])
                bond_blocks.append(rep["bonds"])
                angle_blocks.append(rep["angles"])
                dihedral_blocks.append(rep["dihedrals"])
                improper_blocks.append(rep["impropers"])
                donor_blocks.append(rep["donors"])
                acceptor_blocks.append(rep["acceptors"])
                nnb_blocks.append(rep["nnb"])
                group_blocks.append(rep["groups"])
                crossterm_blocks.append(rep["crossterms"])
                
            for c in range(num):
                new_segid = segid_for(c)
                # Fixed line:
                title_lines.append(remark_template.replace("{segid}", new_segid))

            global_offset += num * tmpl.natom
            counts[chaintype] += num
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        for file_path in glob.glob("psfgentmp_*.pdb"):
            if os.path.exists(file_path):
                os.remove(file_path)

    write_merged_psf(
        psf_out,
        title_lines=title_lines or ["fast-generated PSF"],
        flags=flags or [],
        atom_blocks=atom_blocks, bond_blocks=bond_blocks, angle_blocks=angle_blocks,
        dihedral_blocks=dihedral_blocks, improper_blocks=improper_blocks,
        donor_blocks=donor_blocks, acceptor_blocks=acceptor_blocks,
        nnb_blocks=nnb_blocks, nnb_label=nnb_label,
        group_blocks=group_blocks, ngrp_nst2=0,
        crossterm_blocks=crossterm_blocks,
    )
    if verbose:
        total_atoms = sum(len(b) for b in atom_blocks)
        print(f"[fast] wrote {psf_out}: {total_atoms} atoms total")

# ===========================================================================
# SECTION 3: Verification Utility
# ===========================================================================

def verify_fast_replication(pdbs, num=10, terminal='neutral'):
    num_list = [num] * len(pdbs)

    print(f"Running ORIGINAL custom_genpsf on {pdbs} x{num} each...")
    custom_genpsf(pdbs, num_list, "verify_slow.psf", terminal=terminal)

    print(f"Running FAST custom_genpsf_fast on {pdbs} x{num} each...")
    custom_genpsf_fast(pdbs, num_list, "verify_fast.psf", terminal=terminal)

    slow = parse_psf("verify_slow.psf")
    fast = parse_psf("verify_fast.psf")

    checks = [
        ("natom", slow.natom, fast.natom),
        ("nbond", len(slow.bonds), len(fast.bonds)),
        ("nangle", len(slow.angles), len(fast.angles)),
        ("ndihedral", len(slow.dihedrals), len(fast.dihedrals)),
        ("nimproper", len(slow.impropers), len(fast.impropers)),
    ]

    ok = True
    for name, a, b in checks:
        status = "OK" if a == b else "MISMATCH"
        if a != b:
            ok = False
        print(f"  {name:12s} slow={a:>8d}  fast={b:>8d}  {status}")

    slow_charges = sorted(float(a[6]) for a in slow.atoms)
    fast_charges = sorted(float(a[6]) for a in fast.atoms)
    charge_ok = slow_charges == fast_charges
    print(f"  {'charges':12s} {'matched' if charge_ok else 'MISMATCH'}")
    ok = ok and charge_ok

    slow_masses = sorted(float(a[7]) for a in slow.atoms)
    fast_masses = sorted(float(a[7]) for a in fast.atoms)
    mass_ok = slow_masses == fast_masses
    print(f"  {'masses':12s} {'matched' if mass_ok else 'MISMATCH'}")
    ok = ok and mass_ok

    if ok:
        print("\nPASSED: fast and original methods agree on this small case.")
    else:
        print("\nFAILED: outputs differ -- do NOT use the fast path until this is resolved.")

    return ok

# ===========================================================================
# SECTION 4: Command-Line Interface
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="generate PSF for Hyres/iCon systems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pdb", help="CG PDB file(s)", default='conf.pdb')
    parser.add_argument("psf", help="output name/path for PSF", default='conf.psf')
    parser.add_argument("-t", "--ter",
                        choices=['neutral', 'charged', 'NT', 'CT', 'positive'],
                        help="Terminal charged status (choose from ['neutral', 'charged', 'NT', 'CT', 'positive'])",
                        default='neutral')
    parser.add_argument("--icon", action='store_true',
                        help="Use iConRNA topologies instead of HyRes_iConRNA topologies")
    parser.add_argument("--custom", action='store_true',
                        help="Custom model with specified pdb files and numbers")
    parser.add_argument("-p", "--pdb_list", nargs='+',
                        help="List of PDB files for custom model (ignored if --custom not set)")
    parser.add_argument("-n", "--num_list", nargs='+',
                        help="List of numbers of each molecule type for custom model (ignored if --custom not set)")
    parser.add_argument("--fast", action='store_true',
                        help="Use the fast replication path for --custom mode")
    parser.add_argument("--verify", action='store_true',
                        help="Run verification: compare fast vs original on a small case (n=10).")
    args = parser.parse_args()

    if args.verify:
        if not args.pdb_list:
            print("Error: --verify requires -p (pdb_list) to be set.")
            sys.exit(1)
        success = verify_fast_replication(args.pdb_list, num=10, terminal=args.ter)
        sys.exit(0 if success else 1)

    if args.icon:
        if args.custom:
            if args.fast:
                custom_genpsf_fast(args.pdb_list, args.num_list, args.psf,
                                   terminal=args.ter, RNA='icon')
            else:
                custom_genpsf(args.pdb_list, args.num_list, args.psf,
                              terminal=args.ter, RNA='icon')
        else:
            genpsf(args.pdb, args.psf, terminal=args.ter, RNA='icon')
    else:
        if args.custom:
            if args.fast:
                custom_genpsf_fast(args.pdb_list, args.num_list, args.psf,
                                   terminal=args.ter)
            else:
                custom_genpsf(args.pdb_list, args.num_list, args.psf,
                              terminal=args.ter)
        else:
            genpsf(args.pdb, args.psf, terminal=args.ter)

    for file_path in glob.glob("psfgentmp_*.pdb"):
        os.remove(file_path)

if __name__ == '__main__':
    main()