"""
PSF file generation for HyRes and iCon coarse-grained systems.

This module builds CHARMM-style PSF topology files from coarse-grained PDB
structures for the HyRes (protein) and iCon (RNA, DNA) force fields, plus
ions, CG polymers, metabolites (including aminoglycosides such as KAN) and
user-supplied custom metabolites. Molecule types are detected from residue
names, each chain is assigned a structured segment ID, and ``psfgen`` is used
to build and write the topology.

Two generation paths are provided:

* :func:`genpsf` (default) -- one mixed PDB containing any number of chains
  of any supported type. Every chain is added to a single ``psfgen`` session.
* :func:`custom_genpsf_fast` (``--fast``) -- a list of single-molecule PDBs
  and a copy number for each. One template PSF is built per PDB with
  ``psfgen`` and then replicated N times by text-level PSF parsing and index
  offsetting, which is much faster for systems with many identical copies.

Workflow (default path)
-----------------------
1. Split the input CG PDB into chains keyed by (chainID, segID), type each
   chain from its first residue name, merge consecutive ion chains into one
   segment and write each block to ``psfgentmp_{i}.pdb``
   (:func:`split_chains`).
2. Load all force-field topologies (RNA, Protein, DNA, AGs, Metabolite,
   Polymer and any custom topologies) and add each block to ``psfgen`` with
   a segment ID following the convention below (:func:`genpsf`).
3. Optionally set terminus charges on protein segments (:func:`set_terminus`).
4. Write the PSF and delete the ``psfgentmp_*.pdb`` files.

Segment ID convention
---------------------
Segment IDs are a single type prefix followed by a counter (starting at 1 for
each type) encoded by :func:`encode_segid`.

========  ==================================  ===============
Prefix    Molecule                            Example IDs
========  ==================================  ===============
``P``     Protein                             P001, P002, ...
``R``     RNA                                 R001, R002, ...
``D``     DNA                                 D001, D002, ...
``I``     Ions (MG+, SMG, CA+)                I001, I002, ...
``S``     Polymers (PHO, PEG, QDM, BZM)       S001, S002, ...
``M``     Metabolites, incl. AGs (e.g. KAN)   M001, M002, ...
========  ==================================  ===============

The counter is ``001``-``999``, then ``A00``-``Z99``, then lowercase-led
base62 codes (``a00``...), keeping segment IDs at 4 characters for up to
103,543 segments per type; beyond that a plain decimal number is used, which
makes the segment ID longer than 4 characters.

Custom metabolites
------------------
``--custom ABC,XYZ`` appends the codes to the module-level ``metabolites``
list (so they are typed as ``M``) and, for each code, converts ``ABC.itp`` in
the current directory into ``ABC.top`` and ``ABC.par`` with
``utils.itp2charmm``; the ``.top`` files are then loaded into ``psfgen``
(:func:`prepare_custom_metabolites`).

Command line
------------
Exposed via :func:`main` as the ``genpsf`` console script (also runnable as
``python -m HyresBuilder.GenPsf``)::

    genpsf conf.pdb conf.psf [-t neutral|charged|NT|CT|positive] [--icon]
           [--custom ABC,XYZ]
    genpsf unused.pdb conf.psf --fast -p a.pdb b.pdb -n 10 200 [...]

Dependencies
------------
* `psfgen <https://github.com/MDAnalysis/psfgen>`_ (``psfgen.PsfGen``)
* HyresBuilder force-field topology files, loaded via ``utils.load_ff``.
* For custom metabolites: ``.itp`` files converted via ``utils.itp2charmm``.
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
    """In-memory representation of a single PSF file's contents.

    Filled by :func:`parse_psf`. Atoms are stored as lists of whitespace-split
    tokens (``[id, segid, resid, resname, name, type, charge, mass, imove]``);
    bonded terms, donors/acceptors, groups and cross-terms are stored as tuples
    of 1-based atom indices (groups keep their raw three-integer records).
    ``section_order`` records the order in which sections were read.
    """
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
        """int: Number of atoms in the PSF."""
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
    """Parse a PSF file into a :class:`PSF` object.

    Reads the ``PSF`` header flags and title, then the NATOM, NBOND, NTHETA,
    NPHI, NIMPHI, NDON, NACC, NNB, NGRP and NCRTERM sections, identified by
    their ``!TAG`` labels.

    Args:
        path (str): Path to the PSF file.

    Returns:
        PSF: The parsed contents.

    Raises:
        ValueError: If the first line does not start with ``PSF``.
        NotImplementedError: If a section with any other tag is encountered.
    """
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
    """Generate offset copies of a template PSF.

    For copy ``c`` every atom index is shifted by
    ``start_offset + c * template.natom`` (zero entries are left unchanged)
    and every atom's segment ID is replaced by ``segid_for_copy(c)``.

    Args:
        template (PSF): Template, typically a single segment.
        n_copies (int): Number of copies to generate.
        segid_for_copy (callable): Maps the copy index ``c`` (0-based) to the
            segment ID string for that copy.
        start_offset (int): Number of atoms preceding the first copy in the
            merged PSF. Defaults to 0.

    Yields:
        dict: Per-copy blocks with keys ``atoms``, ``bonds``, ``angles``,
        ``dihedrals``, ``impropers``, ``donors``, ``acceptors``, ``nnb``,
        ``groups`` and ``crossterms``, ready for :func:`write_merged_psf`.
    """
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
    """Write a single merged PSF from lists of per-segment blocks.

    Each ``*_blocks`` argument is a list with one entry per segment copy (as
    yielded by :func:`replicate_segment`); entries are concatenated in order
    and section counts are recomputed. Atom indices must already be offset.

    Notes:
        * If the concatenated NNB list does not have one entry per atom, it is
          replaced by ``natom`` zeros.
        * ``group_blocks`` and ``ngrp_nst2`` are accepted but ignored: a single
          NGRP group (``0 0 0``) is always written.
        * The NCRTERM section is written only if there are cross-terms.

    Args:
        out_path (str): Output PSF path (overwritten).
        title_lines (list of str): Title (REMARKS) lines.
        flags (list of str): Header flags written after ``PSF``
            (e.g. ``EXT``).
        atom_blocks (list): Lists of atom token lists.
        bond_blocks, angle_blocks, dihedral_blocks, improper_blocks,
        donor_blocks, acceptor_blocks, crossterm_blocks (list): Lists of
            index tuples for the corresponding sections.
        nnb_blocks (list): Lists of NNB integers.
        nnb_label (str): Label written after the NNB count (e.g. ``NNB``).
        group_blocks (list): Unused.
        ngrp_nst2 (int): Unused.
    """

    natom = sum(len(b) for b in atom_blocks)
    nbond = sum(len(b) for b in bond_blocks)
    ntheta = sum(len(b) for b in angle_blocks)
    nphi = sum(len(b) for b in dihedral_blocks)
    nimphi = sum(len(b) for b in improper_blocks)
    ndon = sum(len(b) for b in donor_blocks)
    nacc = sum(len(b) for b in acceptor_blocks)
    nnb_total = sum(len(b) for b in nnb_blocks)
    ncrterm = sum(len(b) for b in crossterm_blocks)

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
        if len(flat) != natom:
            flat = [0] * natom
            nnb_total = natom

        out.write(f"{nnb_total:>8d} !{nnb_label}\n")
        for k in range(0, len(flat), 8):
            out.write("".join(f"{v:>8d}" for v in flat[k:k + 8]))
            out.write("\n")
        out.write("\n")

        out.write(f"{1:>8d} {0:>7d} !NGRP\n")
        out.write("       0       0       0\n\n")

        if ncrterm:
            flat = [v for block in crossterm_blocks for ct in block for v in ct]
            _write_int_section(out, "NCRTERM: cross-terms", ncrterm, flat, 8)

# ===========================================================================
# SECTION 2: Custom Metabolite Topology Preparation
# ===========================================================================

def prepare_custom_metabolites(metabolite_names, verbose=True):
    """Convert custom metabolite ``.itp`` files to CHARMM topology files.

    For each code (e.g. ``'ABC'``), ``ABC.itp`` in the current working
    directory is converted with ``utils.itp2charmm``, which writes
    ``<RESI>.top`` and ``<RESI>.par`` named after the residue in the itp's
    ``[ RESI ]`` section; ``ABC.top`` must exist afterwards.

    Args:
        metabolite_names (list of str): Metabolite codes, e.g.
            ``['ABC', 'UVW']``.
        verbose (bool): Print status messages. Defaults to True.

    Returns:
        list of str: Paths of the generated ``.top`` files (relative to the
        current directory), ready to load into ``psfgen``.

    Note:
        Calls ``sys.exit(1)`` if an ``.itp`` file is missing, if the expected
        ``.top`` file is not produced, or if the conversion raises.
    """
    custom_top_files = []
    
    for met_name in metabolite_names:
        itp_file = f"{met_name}.itp"
        top_file = f"{met_name}.top"
        par_file = f"{met_name}.par"
        
        if not os.path.exists(itp_file):
            print(f"Error: {itp_file} not found for metabolite '{met_name}'")
            sys.exit(1)
        
        try:
            if verbose:
                print(f"Converting {itp_file} to CHARMM format...")
            
            # Convert .itp to .top and .par using utils.itp2charmm
            utils.itp2charmm(itp_file)
            
            if not os.path.exists(top_file):
                print(f"Error: Failed to generate {top_file}")
                sys.exit(1)
            
            custom_top_files.append(top_file)
            
            if verbose:
                print(f"Generated: {top_file} and {par_file}")
        
        except Exception as e:
            print(f"Error converting {itp_file}: {e}")
            sys.exit(1)
    
    return custom_top_files

# ===========================================================================
# SECTION 3: Main PSF Generation Logic
# ===========================================================================

aas = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
rnas = ["ADE", "GUA", "CYT", "URA", "A", "G", "C", "U"]
dnas = ["DAD", "DCY", "DTH", "DGU", "DA", "DG", "DC", "DT"]
ions = ["MG+", "SMG", "CA+"]
polymer = ['PHO', 'PEG', 'QDM', 'BZM']
auto_polymer = ['PHO']     # polymers relying on psfgen auto angles/dihedrals in the fast path
metabolites = ['KAN', 'LLL', 'SRY',
               'UN1', 'AYA', 'ACA', 'NLG', 'C3C', 'C4C', 'C5C', '152', 'CHT', 'CIT',
               'CTT', 'ABU', 'CH5', 'GSH', 'MTA', 'SHR', 'TAU', 'BET', '3PG', 'G6P',
               'COA', 'FAD', 'NCA', 'PAU', 'ADN', 'ADP', 'AMP', 'ATP', 'C5P', 'CTN',
               'UGA', '5GP', 'UD1', 'NOS', 'NAD', 'NAI', 'NAD', 'UDP', 'U5P', 'UPG',
               '2PG', '13P', 'PEP', 'SAM', '2HG', 'FUM', 'AKG', 'LMR', 'MCT', 'SIN',
               'DGU', "DMG", "MG7", "HIC", "AD0", "GRS", "CRN", "AOR", "4UO", "GUN",
               "NLQ", "GNG", "HYP", "ICO", "HFA", "MLA", "URI", "KIV", "FLN", "TRA",
               "X5A", "U0",  "PC",  "BTN", "ALY", "DCM", "1AL", "G3P", "D5M", "PPY",
               "ORN", "2ND", "CHD", "HPA", "RBF", "XAN", "DXC", "VIB", "3D1",
               ]

segtypes = ['P', 'R', 'D', 'I', 'S', 'M']

def get_type(resname):
    """Return the molecule-type code for a residue name.

    Args:
        resname (str): Residue name (as in PDB columns 18-20, stripped).

    Returns:
        str or None: ``'P'`` protein (``aas``), ``'R'`` RNA (``rnas``),
        ``'D'`` DNA (``dnas``), ``'I'`` ion (``ions``: MG+, SMG, CA+),
        ``'S'`` polymer (``polymer``: PHO, PEG, QDM, BZM), ``'M'`` metabolite
        (``metabolites``, including AGs such as KAN and any ``--custom``
        codes), or None if unrecognised. Lists are checked in that order.
    """
    chaintype = (
        'P' if resname in aas else
        'R' if resname in rnas else
        'D' if resname in dnas else
        'I' if resname in ions else
        'S' if resname in polymer else
        'M' if resname in metabolites else
        None
    )
    return chaintype

def _encode_resseq(n):
    """Encode residue number n into the 4-char PDB resSeq field (hybrid-36 beyond 9999)."""
    if n < 10000:
        return f"{n:4d}"
    n -= 10000
    if n < 26 * 36**3:
        digits, first = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'A'
    else:
        n -= 26 * 36**3
        digits, first = '0123456789abcdefghijklmnopqrstuvwxyz', 'a'
    result = []
    for _ in range(3):
        n, remainder = divmod(n, 36)
        result.append(digits[remainder])
    result.append(chr(ord(first) + n))
    return ''.join(reversed(result))

def _renumber_residues(lines):
    """Renumber residues in ATOM lines sequentially from 1, in file order.

    A new residue starts whenever the (chainID, segID, resSeq+iCode) key
    changes. The new number is written with :func:`_encode_resseq` and the
    insertion-code column is blanked.

    Args:
        lines (list of str): PDB ATOM lines.

    Returns:
        list of str: The renumbered lines.
    """
    new_lines = []
    old_key = None
    new_resid = 0
    for line in lines:
        key = (line[21], line[72:76], line[22:27])
        if key != old_key:
            new_resid += 1
            old_key = key
        new_lines.append(line[:22] + _encode_resseq(new_resid) + ' ' + line[27:])
    return new_lines

def split_chains(pdb):
    """Split a CG PDB into per-segment temporary PDB files.

    ATOM records are grouped into chains whenever the (chainID, segID) key
    (PDB columns 22 and 73-76) changes; each chain is typed by
    :func:`get_type` from its first residue name. Consecutive ion (``'I'``)
    chains are merged into one block whose residues are renumbered from 1
    (:func:`_renumber_residues`, hybrid-36 beyond 9999). Block ``i`` is
    written to ``psfgentmp_{i}.pdb`` in the current directory, terminated by
    ``END``. Non-ATOM records (including HETATM) are ignored.

    Args:
        pdb (str): Input CG PDB path.

    Returns:
        list of str: Type code of each block, in the order of the
        ``psfgentmp_{i}.pdb`` files.

    Note:
        Prints ``Unknown molecule type`` and exits with status 1 if any
        chain's first residue is not recognised.
    """
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

    # merge adjacent ion chains into one segment; each block -> psfgentmp_{i}.pdb
    blocks = []
    merged = []
    for t, chain in zip(types, chains):
        if t not in segtypes:
            print('Unknown molecule type')
            exit(1)
        if t == 'I' and blocks and blocks[-1][0] == 'I':
            blocks[-1][1].extend(chain)
            merged[-1] = True
        else:
            blocks.append((t, list(chain)))
            merged.append(False)

    types = []
    for i, ((t, chain), is_merged) in enumerate(zip(blocks, merged)):
        types.append(t)
        if is_merged:
            # merged ion chains may repeat resids, renumber them within the segment
            chain = _renumber_residues(chain)
        tmp_pdb = f"psfgentmp_{i}.pdb"
        with open(tmp_pdb, 'w') as f:
            for line in chain:
                f.write(line)
            f.write('END\n')
    return types

def set_terminus(gen, segid, charge_status):
    """Set terminus charges on a protein segment in a ``psfgen`` session.

    Only segments whose ID starts with ``P`` are modified; others are left
    unchanged. The N-terminus is atom ``N`` of the first residue and the
    C-terminus is atom ``O`` of the last residue.

    Args:
        gen (psfgen.PsfGen): Session containing the segment.
        segid (str): Segment ID.
        charge_status (str): One of

            - ``'charged'``: N-terminal N = +1.00, C-terminal O = -1.00
            - ``'NT'``: N-terminal N = +1.00 only
            - ``'CT'``: C-terminal O = -1.00 only
            - ``'positive'``: N-terminal N = -1.00 and C-terminal O = -1.00
              (both set to -1.00 as currently implemented)

    Note:
        Any other value, including ``'neutral'``, prints an error and exits
        with status 1 for protein segments; callers skip this function for
        ``'neutral'``.
    """
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
    """Encode a segment counter as a (normally 3-character) string.

    Tier 1 (n < 1000):        zero-padded decimal  ``"001"``..``"999"``
    Tier 2 (n=1000..3599):    letter + 2 digits    ``"A00"``..``"Z99"``
    Tier 3 (n=3600..103543):  lowercase letter + 2 base62 characters
        (``"a00"``..``"zZZ"``); cannot collide with tiers 1/2, which start
        with a digit or an uppercase letter.
    n > 103543: ``str(n)`` (6+ characters, so the resulting segment ID
        exceeds the 4-character PSF limit).

    Args:
        n (int): Segment counter (1-based in this module).

    Returns:
        str: Encoded counter.
    """
    if n < 1000:
        return f"{n:03d}"

    if n < 3600:
        m = n - 1000
        letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[m // 100]
        rest = m % 100
        return f"{letter}{rest:02d}"

    BASE62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    m = n - 3600
    if m >= 26 * 62 * 62:
        return str(n)

    c1 = "abcdefghijklmnopqrstuvwxyz"[m // 3844]
    m = m % 3844
    c2 = BASE62[m // 62]
    c3 = BASE62[m % 62]
    return f"{c1}{c2}{c3}"

def genpsf(pdb_in, psf_out, terminal='neutral', RNA='mix', custom_top_files=None):
    """Generate a PSF for a mixed CG system from a single PDB.

    Loads the RNA, Protein, DNA, AGs, Metabolite and Polymer topologies (plus
    any custom ones), splits the PDB with :func:`split_chains`, and adds
    each block as a segment named ``<type><encode_segid(k)>`` with ``k``
    counted from 1 per type. Protein segments use ``auto_angles=False``;
    all other types use ``auto_angles=False, auto_dihedrals=False``. If
    ``terminal`` is not ``'neutral'``, :func:`set_terminus` is applied to
    every segment (it only affects protein segments). The temporary
    ``psfgentmp_*.pdb`` files are deleted after the PSF is written.

    Args:
        pdb_in (str): Input CG PDB path.
        psf_out (str): Output PSF path.
        terminal (str): Protein terminus charge status: ``'neutral'``,
            ``'charged'``, ``'NT'``, ``'CT'`` or ``'positive'`` (see
            :func:`set_terminus`). Defaults to ``'neutral'``.
        RNA (str): RNA topology: ``'mix'`` uses ``utils.load_ff('RNA')``
            (``top_RNA_mix.inp``, HyRes-compatible iConRNA); ``'icon'`` uses
            ``forcefield/top_RNA.inp``. Defaults to ``'mix'``.
        custom_top_files (list of str, optional): Extra topology files (e.g.
            from :func:`prepare_custom_metabolites`).
    """
    if RNA == 'mix':
        RNA_topology, _ = utils.load_ff('RNA')
    elif RNA == 'icon':
        path1 = files("HyresBuilder") / "forcefield" / "top_RNA.inp"
        RNA_topology = path1.as_posix()
    protein_topology, _ = utils.load_ff('Protein')
    DNA_topology, _ = utils.load_ff('DNA')
    AGs_topology, _ = utils.load_ff('AGs')
    Mats_topology, _ = utils.load_ff('Metabolite')
    polymer_topology, _ = utils.load_ff('Polymer')

    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)
    gen.read_topology(DNA_topology)
    gen.read_topology(AGs_topology)
    gen.read_topology(Mats_topology)
    gen.read_topology(polymer_topology)
    
    # Load custom metabolite topologies if provided
    if custom_top_files:
        for top_file in custom_top_files:
            gen.read_topology(top_file)

    counts = {'P': 1, 'R': 1, 'D': 1, 'I': 1, 'S': 1, 'M': 1}
    types = split_chains(pdb_in)
    for i, t in enumerate(types):
        tmp_pdb = f"psfgentmp_{i}.pdb"
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

def _apply_terminus_to_template(atoms, charge_status):
    """Set terminus charges directly on parsed PSF atom tokens (in place).

    Text-level equivalent of :func:`set_terminus` used by the fast path: the
    N-terminus is atom ``N`` of the first resid and the C-terminus atom
    ``O`` of the last resid, in order of appearance. Prints a warning if a
    target atom is missing; prints an error and exits with status 1 for an
    unsupported ``charge_status`` (including ``'neutral'``).

    Args:
        atoms (list of list of str): ``PSF.atoms`` of a protein template.
        charge_status (str): ``'charged'``, ``'NT'``, ``'CT'`` or
            ``'positive'`` (same charges as :func:`set_terminus`).
    """
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

def custom_genpsf_fast(pdb_list, num_list, psf_out, terminal='neutral', RNA='mix', custom_top_files=None, verbose=True):
    """Generate a PSF for many copies of a few molecules (fast path).

    For each ``(pdb, num)`` pair with ``num > 0``, the molecule type is taken
    from the first ATOM residue of ``pdb`` (:func:`get_type`); the whole file
    is treated as one segment. A template PSF is built with a fresh
    ``psfgen`` session (all topologies loaded, as in :func:`genpsf`) and
    written to a temporary ``genpsf_fast_*`` directory, parsed with
    :func:`parse_psf`, and replicated ``num`` times with
    :func:`replicate_segment`. Segment IDs continue the per-type counters
    across inputs (``<type><encode_segid(k)>``), and a ``REMARKS segment``
    title line is added for each copy. All copies are written to ``psf_out``
    with :func:`write_merged_psf`.

    Template ``psfgen`` options: proteins use ``auto_angles=False``;
    polymers listed in ``auto_polymer`` (PHO) keep psfgen's automatic angles
    and dihedrals; all other types use ``auto_angles=False,
    auto_dihedrals=False``. For protein templates, a non-``'neutral'``
    ``terminal`` is applied via :func:`_apply_terminus_to_template`.

    The temporary directory and any ``psfgentmp_*.pdb`` files in the current
    directory are removed afterwards.

    Args:
        pdb_list (list of str): Single-molecule CG PDB files.
        num_list (list of int or str): Copy number for each PDB (converted
            with ``int``); pairs beyond the shorter list are ignored.
        psf_out (str): Output PSF path.
        terminal (str): Protein terminus charge status. Defaults to
            ``'neutral'``.
        RNA (str): ``'mix'`` or ``'icon'``, as in :func:`genpsf`. Defaults to
            ``'mix'``.
        custom_top_files (list of str, optional): Extra topology files.
        verbose (bool): Print progress. Defaults to True.

    Note:
        Exits with status 1 if a PDB's first residue type is unknown.
    """
    if RNA == 'mix':
        RNA_topology, _ = utils.load_ff('RNA')
    elif RNA == 'icon':
        path1 = files("HyresBuilder") / "forcefield" / "top_RNA.inp"
        RNA_topology = path1.as_posix()
    protein_topology, _ = utils.load_ff('Protein')
    DNA_topology, _ = utils.load_ff('DNA')
    AGs_topology, _ = utils.load_ff('AGs')
    Mats_topology, _ = utils.load_ff('Metabolite')
    polymer_topology, _ = utils.load_ff('Polymer')

    atom_blocks, bond_blocks, angle_blocks = [], [], []
    dihedral_blocks, improper_blocks = [], []
    donor_blocks, acceptor_blocks, nnb_blocks = [], [], []
    group_blocks, crossterm_blocks = [], []
    nnb_label = "NNB"
    title_lines, flags = None, None
    global_offset = 0

    counts = {'P': 0, 'R': 0, 'D': 0, 'I': 0, 'S': 0, 'M': 0}

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
            gen.read_topology(DNA_topology)
            gen.read_topology(AGs_topology)
            gen.read_topology(Mats_topology)
            gen.read_topology(polymer_topology)
            
            # Load custom metabolite topologies if provided
            if custom_top_files:
                for top_file in custom_top_files:
                    gen.read_topology(top_file)

            tmpl_segid = f"{chaintype}{encode_segid(counts[chaintype] + 1)}"
            if chaintype == 'P':
                gen.add_segment(segid=tmpl_segid, pdbfile=pdb, auto_angles=False)
            elif chaintype == 'S' and resname in auto_polymer:
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
# SECTION 4: Command-Line Interface
# ===========================================================================

def main():
    """Command-line interface for PSF generation (``genpsf`` console script).

    Positional arguments are ``pdb`` (input CG PDB) and ``psf`` (output PSF);
    both are required, although ``pdb`` is unused with ``--fast``.

    Options:
        -t/--ter: Protein terminus charge status (``neutral`` [default],
            ``charged``, ``NT``, ``CT``, ``positive``).
        --icon: Use the iConRNA topology ``top_RNA.inp`` instead of the
            HyRes-compatible ``top_RNA_mix.inp``.
        --fast: Use :func:`custom_genpsf_fast`; requires ``-p/--pdb_list``
            and ``-n/--num_list`` (otherwise exits with status 1).
        -p/--pdb_list: Single-molecule PDB files for ``--fast``.
        -n/--num_list: Copy number of each PDB for ``--fast``.
        --custom: Comma-separated custom metabolite codes (e.g. ``ABC,UVW``).
            Codes are added to ``metabolites`` and ``<code>.itp`` files in the
            current directory are converted via
            :func:`prepare_custom_metabolites`; the resulting ``.top`` files
            are loaded in either mode.

    Without ``--fast``, :func:`genpsf` is run on ``pdb``. Any remaining
    ``psfgentmp_*.pdb`` files are removed at the end.

    Example::

        genpsf conf.pdb conf.psf -t charged
        genpsf conf.pdb conf.psf --custom ABC
        genpsf x.pdb system.psf --fast -p protein.pdb kan.pdb -n 10 500
    """
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
    parser.add_argument("--fast", action='store_true',
                        help="Use fast replication path with custom PDB files and numbers (requires -p and -n)")
    parser.add_argument("-p", "--pdb_list", nargs='+',
                        help="List of PDB files for custom model (required when --fast is set)")
    parser.add_argument("-n", "--num_list", nargs='+',
                        help="List of numbers of each molecule type for custom model (required when --fast is set)")
    parser.add_argument("--custom", type=str,
                        help="Add custom metabolite residues (comma-separated, e.g., 'ABC,UVW')")
    args = parser.parse_args()

    # Handle custom metabolites if --custom flag is provided
    custom_top_files = None
    if args.custom:
        custom_mets = [m.strip() for m in args.custom.split(',')]
        for met in custom_mets:
            if met not in metabolites:
                metabolites.append(met)
                if len(custom_mets) <= 5:
                    print(f"Added metabolite: '{met}'")
            else:
                print(f"Note: '{met}' already in metabolites list")
        
        # Convert .itp files to CHARMM topology files
        print(f"Preparing topology files for custom metabolites...")
        custom_top_files = prepare_custom_metabolites(custom_mets, verbose=True)
        print(f"Successfully prepared {len(custom_top_files)} custom topology files\n")

    if args.fast:
        # Validate required arguments
        if not args.pdb_list or not args.num_list:
            print("Error: --fast requires -p/--pdb_list and -n/--num_list arguments")
            sys.exit(1)
        
        # Determine RNA mode
        rna_mode = 'icon' if args.icon else 'mix'
        
        # Run fast custom mode
        custom_genpsf_fast(args.pdb_list, args.num_list, args.psf,
                          terminal=args.ter, RNA=rna_mode, 
                          custom_top_files=custom_top_files)
    else:
        # Standard mode: single PDB file
        rna_mode = 'icon' if args.icon else 'mix'
        genpsf(args.pdb, args.psf, terminal=args.ter, RNA=rna_mode,
               custom_top_files=custom_top_files)

    for file_path in glob.glob("psfgentmp_*.pdb"):
        os.remove(file_path)

if __name__ == '__main__':
    main()