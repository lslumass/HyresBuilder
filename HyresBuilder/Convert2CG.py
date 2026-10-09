"""
Conversion utilities for all-atom to coarse-grained (CG) structure preparation.

This module converts all-atom protein, RNA, DNA and aminoglycoside (AGs)
structures into coarse-grained representations compatible with the HyRes
(protein), iConRNA (RNA), iConDNA (DNA) and AGs CG models. It handles the full
pipeline -- optional backbone hydrogen addition, CG bead placement, topology
generation with ``psfgen`` and PSF writing -- and can process mixed systems in
a single call.

Conversion models
-----------------
* **HyRes (protein)** -- backbone atoms N, H, CA, C, O are kept at their
  original positions; sidechain heavy atoms are collapsed into one to five
  geometric-center beads named CB, CC, CD, CE, CF depending on residue type.
  Glycine has no sidechain bead. Histidine variants are written as HIS
  (:func:`at2hyres`).
* **iConRNA (RNA)** -- each nucleotide is mapped to a phosphate bead (P), two
  sugar beads (C1 at C4', C2 at C1') and three or four base beads (NA-ND),
  each at the geometric center of its contributing atoms. Supported
  nucleotides: ADE, GUA, CYT, URA; one-letter A, G, C, U are renamed to these
  during chain splitting (:func:`at2RNA`).
* **iConDNA (DNA)** -- the same bead topology (P, C1, C2, NA-ND) applied to
  deoxyribonucleotides DA, DG, DC, DT and the aliases DAD, DGU, DCY, DTH
  (:func:`at2DNA`).
* **AGs (aminoglycosides)** -- residue-specific bead mappings; currently
  kanamycin A (KAN, 11 beads K1-K11) (:func:`at2AGs`).

Pipeline overview
-----------------
The top-level function :func:`at2cg` orchestrates the workflow:

1. (CLI ``--hydrogen`` only) add backbone amide hydrogens to the all-atom
   input (:func:`add_backbone_hydrogen`).
2. Optionally rename CHARMM-style DNA residue names so DNA chains are
   distinguishable from RNA (:func:`fix_charmm_dna_resnames`).
3. Split the input PDB into per-chain temporary files
   ``aa2cgtmp_{i}_aa.pdb``, detect molecule types and assign segment IDs
   (:func:`split_chains`), optionally renumbering residues from 1.
4. Apply the CG mapping per chain (:func:`at2hyres`, :func:`at2RNA`,
   :func:`at2DNA` or :func:`at2AGs`) and add each chain to ``psfgen``.
5. Write the CG PDB, set protein terminus charges (:func:`set_terminus`),
   write the PSF, remove the ``aa2cgtmp_*.pdb`` files (unless
   ``cleanup=False``) and renumber the PDB atom serials, using hybrid-36
   above 99,999 (:func:`fix_pdb_serial`).

A command-line interface is exposed via :func:`main` and registered as the
``convert2cg`` console script.

CHARMM residue names
--------------------
CHARMM PDB files use the same residue names for RNA and DNA bases (ADE, GUA,
CYT, THY), so a DNA chain cannot be distinguished from an RNA chain by residue
name alone. Passing ``charmm=True`` to :func:`at2cg` (or ``--charmm`` on the
command line) renames ADE->DA, GUA->DG, CYT->DC, THY->DT before chain
splitting, so those chains are typed as DNA and routed to :func:`at2DNA`.

Hybrid-36 serial encoding
-------------------------
The PDB format allows at most 5 characters for atom serials (99,999). Larger
serials are written in hybrid-36: ``A0000``-``ZZZZZ`` for atoms
100,000-43,770,015, then ``a0000``-``zzzzz`` beyond that.

Dependencies
------------
* `psfgen <https://github.com/MDAnalysis/psfgen>`_ (``psfgen.PsfGen``)
* `NumPy <https://numpy.org>`_ (``numpy``)
* HyresBuilder force-field topology files, loaded via ``utils.load_ff``.
"""

from psfgen import PsfGen
import numpy as np
import os
import warnings
from .utils import load_ff


def add_backbone_hydrogen(pdb_file, output_file):
    """
    Add backbone amide hydrogen atoms (H) to peptide chains in a PDB file.

    All ATOM records are kept (other records such as HETATM, TER and END are
    dropped) and atom serials are renumbered from 1 (hybrid-36 above 99,999).
    An ``H`` atom is inserted directly after the ``N`` atom of every residue
    except PRO, residues that already have ``H`` or ``HN``, and residues
    lacking ``CA`` or any usable ``C``. It is placed 1.01 A from N along the
    bisector of the C(prev)->N and CA->N directions, where C(prev) is the
    previous residue's C within the same segment; the residue's own C is used
    for the first residue of a segment. A new segment starts when the chain
    ID changes or the residue number jumps by more than 1.

    Args:
        pdb_file (str): Path to the input all-atom PDB file.
        output_file (str): Path to the output PDB file (overwritten).

    Returns:
        str: ``output_file``.
    """
    
    def parse_atom_line(line):
        """Parse a PDB ATOM line and extract relevant information."""
        def parse_serial(s):
            """Handle both decimal and hybrid-36/hex encoded serial numbers."""
            s = s.strip()
            try:
                return int(s)
            except ValueError:
                # Try hexadecimal (used when serial > 99999)
                try:
                    return int(s, 16)
                except ValueError:
                    # Full hybrid-36: uppercase letters start at 100000, lowercase at 1316736
                    if s[0].isupper():
                        return (ord(s[0]) - ord('A')) * 36**4 + int(s[1:], 36) + 100000
                    else:
                        return (ord(s[0]) - ord('a')) * 36**4 + int(s[1:], 36) + 1316736
                    
        atom_serial = parse_serial(line[6:11])
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_seq = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        occupancy = line[54:60].strip() if len(line) > 54 else "1.00"
        temp_factor = line[60:66].strip() if len(line) > 60 else "0.00"
        element = line[76:78].strip() if len(line) > 76 else ""
        
        return {
            'serial': atom_serial,
            'name': atom_name,
            'residue': residue_name,
            'chain': chain_id,
            'res_seq': residue_seq,
            'coords': np.array([x, y, z]),
            'occupancy': occupancy,
            'temp_factor': temp_factor,
            'element': element,
            'line': line
        }
    
    def encode_serial(n):
        """Encode integer to hybrid-36 format for PDB serial number field (5 chars)."""
        if n < 100000:
            return f"{n:5d}"

        n -= 100000
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        if n < 26 * (36**4):  # uppercase range
            result = []
            for _ in range(4):
                n, remainder = divmod(n, 36)
                result.append(chars[remainder])
            result.append(chr(ord('A') + n))
            return ''.join(reversed(result))

        n -= 26 * (36**4)  # lowercase range
        chars_lower = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = []
        for _ in range(4):
            n, remainder = divmod(n, 36)
            result.append(chars_lower[remainder])
        result.append(chr(ord('a') + n))
        return ''.join(reversed(result))


    def encode_resseq(n):
        """Encode integer to hybrid-36 format for residue sequence field (4 chars)."""
        if n < 10000:
            return f"{n:4d}"

        n -= 10000
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        if n < 26 * (36**3):
            result = []
            for _ in range(3):
                n, remainder = divmod(n, 36)
                result.append(chars[remainder])
            result.append(chr(ord('A') + n))
            return ''.join(reversed(result))

        n -= 26 * (36**3)
        chars_lower = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = []
        for _ in range(3):
            n, remainder = divmod(n, 36)
            result.append(chars_lower[remainder])
        result.append(chr(ord('a') + n))
        return ''.join(reversed(result))

    def format_atom_line(serial, atom_name, residue_name, chain_id, residue_seq,
                         coords, occupancy="1.00", temp_factor="0.00", element="H"):
        """Format an ATOM line in PDB format."""
        serial_str = encode_serial(serial)
        resseq_str = encode_resseq(residue_seq)
        # PDB columns 13-16 are a fixed 4-character atom-name field. Names of
        # 1-3 characters are written with a leading blank (column 13) and
        # left-justified after it; names of exactly 4 characters fill the
        # whole field. This keeps every column after the name (altLoc,
        # resName, chainID, resSeq, ...) at a fixed position no matter how
        # long the atom name is - previously a 3-char-wide field silently
        # let 4-character names (e.g. hydrogens like "HD11") overflow by one
        # column, shifting resName and corrupting downstream parsing.
        if len(atom_name) >= 4:
            name_field = atom_name[:4]
        else:
            name_field = f" {atom_name:<3s}"
        return (f"ATOM  {serial_str} {name_field} {residue_name:3s} {chain_id}{resseq_str}    "
                f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}{occupancy:>6s}{temp_factor:>6s}"
                f"          {element:>2s}\n")
    
    def calculate_h_position(n_coord, ca_coord, c_prev_coord):
        """
        Calculate the position of backbone H atom bonded to N.
        
        The H is placed along the N-C(previous) direction with proper geometry:
        - N-H bond length: 1.01 Å
        - C-N-H angle: ~120° (sp2 hybridization)
        
        Parameters:
        -----------
        n_coord : np.array
            Coordinates of N atom
        ca_coord : np.array
            Coordinates of CA atom (current residue)
        c_prev_coord : np.array
            Coordinates of C atom from previous residue (or current for first residue)
        """
        # Vector from C(prev) to N
        v_cn = n_coord - c_prev_coord
        v_cn = v_cn / np.linalg.norm(v_cn)
        
        # Vector from N to CA
        v_nca = ca_coord - n_coord
        v_nca = v_nca / np.linalg.norm(v_nca)
        
        # Bisector direction (for ideal geometry)
        # The H should be opposite to the peptide bond direction
        # but also considering the CA position
        bisector = v_cn - v_nca
        bisector = bisector / np.linalg.norm(bisector)
        
        # N-H bond length (standard: 1.01 Å)
        nh_bond_length = 1.01
        
        # Position H atom
        h_coord = n_coord + bisector * nh_bond_length
        
        return h_coord
    
    # Read PDB file and store all lines
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    # First pass: organize atoms by residue to find N, CA, C positions
    # and detect chain segments
    residue_data = []  # List of (line_idx, atom_dict) to maintain order
    residue_lookup = {}  # For quick lookup: (chain, res_seq, segment_id) -> {atom_name: atom_dict}
    
    current_segment_id = 0
    prev_chain = None
    prev_res_seq = None
    
    for idx, line in enumerate(lines):
        if not line.startswith('ATOM'):
            continue
        
        atom = parse_atom_line(line)
        chain = atom['chain']
        res_seq = atom['res_seq']
        
        # Detect chain break: different chain ID OR non-consecutive residue numbers
        is_new_segment = False
        if prev_chain is None:
            is_new_segment = True
        elif chain != prev_chain:
            is_new_segment = True
        elif abs(res_seq - prev_res_seq) > 1:
            is_new_segment = True
        
        if is_new_segment and prev_chain is not None:
            current_segment_id += 1
        
        # Store atom with its original line index and segment info
        atom['line_idx'] = idx
        atom['segment_id'] = current_segment_id
        residue_data.append((idx, atom))
        
        # Also store in lookup dictionary
        key = (chain, res_seq, current_segment_id)
        if key not in residue_lookup:
            residue_lookup[key] = {}
        residue_lookup[key][atom['name']] = atom
        
        prev_chain = chain
        prev_res_seq = res_seq
    
    # Second pass: build output with H atoms inserted after N atoms
    output_lines = []
    current_serial = 1
    
    # Group residues by segment
    segments = {}
    for idx, atom in residue_data:
        seg_id = atom['segment_id']
        if seg_id not in segments:
            segments[seg_id] = []
        key = (atom['chain'], atom['res_seq'], seg_id)
        if key not in [s['key'] for s in segments[seg_id]]:
            segments[seg_id].append({'key': key, 'first_idx': idx})
    
    # Track which residues we've added H to
    h_added = set()
    
    # Process all original atoms in order
    for idx, atom in residue_data:
        chain = atom['chain']
        res_seq = atom['res_seq']
        seg_id = atom['segment_id']
        key = (chain, res_seq, seg_id)
        
        # Write the current atom
        output_lines.append(format_atom_line(
            current_serial, atom['name'], atom['residue'],
            atom['chain'], atom['res_seq'], atom['coords'],
            atom['occupancy'], atom['temp_factor'], 
            atom['element'] if atom['element'] else atom['name'][0]
        ))
        current_serial += 1
        
        # If this is an N atom and we haven't added H yet for this residue
        if atom['name'] == 'N' and key not in h_added:
            res_atoms = residue_lookup[key]
            
            # Skip proline (PRO) residues - they don't have backbone H
            if atom['residue'] == 'PRO':
                continue

            # Skip residues that already have a backbone amide hydrogen
            # (some structures have H/HN on some residues but not others,
            # e.g. partially-protonated or mixed-source PDBs)
            if 'H' in res_atoms or 'HN' in res_atoms:
                h_added.add(key)
                continue

            # Check if we have necessary atoms (N, CA, C)
            if 'CA' in res_atoms:
                h_added.add(key)
                
                # Find if this is the first residue in the segment
                segment_residues = [s['key'] for s in segments[seg_id]]
                is_first_in_segment = (key == segment_residues[0])
                
                c_coord = None
                if not is_first_in_segment:
                    # Try to use previous residue's C atom
                    res_index = segment_residues.index(key)
                    if res_index > 0:
                        prev_key = segment_residues[res_index - 1]
                        if prev_key in residue_lookup and 'C' in residue_lookup[prev_key]:
                            c_coord = residue_lookup[prev_key]['C']['coords']
                
                # If no previous C or first residue, use current residue's C
                if c_coord is None and 'C' in res_atoms:
                    c_coord = res_atoms['C']['coords']
                
                # Calculate and add H atom
                if c_coord is not None:
                    h_coord = calculate_h_position(
                        atom['coords'],
                        res_atoms['CA']['coords'],
                        c_coord
                    )
                    
                    output_lines.append(format_atom_line(
                        current_serial, 'H', atom['residue'],
                        atom['chain'], atom['res_seq'], h_coord,
                        atom['occupancy'], atom['temp_factor'], 'H'
                    ))
                    current_serial += 1
    
    # Write output file
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"Added backbone hydrogen atoms. Output saved to {output_file}")
    return output_file


# CHARMM uses the same residue names for RNA and DNA bases (ADE/GUA/CYT/THY).
# Map them onto the DNA names recognized by split_chains() and at2DNA().
_CHARMM_DNA_RESNAMES = {'ADE': 'DA', 'GUA': 'DG', 'CYT': 'DC', 'THY': 'DT'}


def fix_charmm_dna_resnames(pdb_file, output_file=None):
    """
    Rewrite CHARMM-style nucleic acid residue names to explicit DNA names.

    CHARMM PDB files use the same residue names for RNA and DNA bases, so a
    DNA chain is written as ADE/GUA/CYT/THY rather than DA/DG/DC/DT. This
    helper renames them, in every ATOM/HETATM record, to the DNA names
    understood by :func:`split_chains` and :func:`at2DNA`::

        ADE -> DA    GUA -> DG    CYT -> DC    THY -> DT

    Note:
        The mapping is unconditional, so apply this only to files whose nucleic
        acid chains are DNA. A CHARMM file containing genuine RNA chains would
        have those chains mis-typed as DNA.

    Args:
        pdb_file (str): Path to the input PDB file.
        output_file (str, optional): Path to the output PDB file. If None,
            the input file is overwritten in place.

    Returns:
        str: Path to the written PDB file.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> Convert2CG.fix_charmm_dna_resnames("charmm_dna.pdb", "dna_fixed.pdb")
    """
    if output_file is None:
        output_file = pdb_file

    out_lines = []
    n_renamed = 0

    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            resname = line[17:20].strip().upper()
            new_resname = _CHARMM_DNA_RESNAMES.get(resname)
            if new_resname is not None:
                # resName occupies PDB columns 18-20 (0-indexed 17:20)
                line = line[:17] + f"{new_resname:<3s}" + line[20:]
                n_renamed += 1
        out_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(out_lines)

    print(f"CHARMM DNA residue names fixed for {n_renamed} atoms. "
          f"Output saved to {output_file}")
    return output_file


def _renumber_chain(chain_lines):
    """Renumber residues in a single chain's ATOM lines to start from 1.

    Checks the first residue's resSeq; if it's already 1, the lines are
    returned unchanged. Otherwise every residue is renumbered sequentially
    starting from 1, using each unique (resSeq, iCode) pair encountered (in
    file order) to detect residue boundaries so insertion codes are handled
    correctly. Residue counts beyond 9999 fall back to hybrid-36 encoding,
    consistent with the rest of this module.
    """
    def encode_resseq(n):
        if n < 10000:
            return f"{n:4d}"
        n -= 10000
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if n < 26 * (36**3):
            result = []
            for _ in range(3):
                n, remainder = divmod(n, 36)
                result.append(chars[remainder])
            result.append(chr(ord('A') + n))
            return ''.join(reversed(result))
        n -= 26 * (36**3)
        chars_lower = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = []
        for _ in range(3):
            n, remainder = divmod(n, 36)
            result.append(chars_lower[remainder])
        result.append(chr(ord('a') + n))
        return ''.join(reversed(result))

    if not chain_lines:
        return chain_lines

    try:
        first_resseq = int(chain_lines[0][22:26].strip())
    except ValueError:
        first_resseq = None

    if first_resseq == 1:
        # Already starts from 1 - leave this segment untouched
        return chain_lines

    new_lines = []
    old_key = None
    new_resid = 0
    for line in chain_lines:
        key = (line[22:26], line[26])
        if key != old_key:
            new_resid += 1
            old_key = key
        new_lines.append(line[:22] + encode_resseq(new_resid) + line[26:])
    return new_lines


def split_chains(pdb, renumber=False):
    """Split an all-atom PDB into chains, identify their types and segids.

    Chains are delimited by changes in either the chain ID (column 22) or
    the segment ID (columns 73-76) of ATOM records. If only one of them is
    present it is used; if both are present the chain ID is used when it
    changes and the segment ID either never changes or changes the same
    number of times, otherwise the segment ID is used. Before writing,
    one-letter RNA names A/G/C/U are renamed to ADE/GUA/CYT/URA and histidine
    variants (HSD, HSE, HSP, HID, HIE, HIP) to HIS. Each chain is typed from
    its first residue and written to ``aa2cgtmp_{i}_aa.pdb`` (ending with
    ``END``) in the current directory.

    Args:
        pdb (str): Path to the input PDB file.
        renumber (bool, optional): If ``True``, checks each segment's
            starting residue number and, if it doesn't already start from 1,
            renumbers that segment's residues sequentially from 1. Segments
            that already start from 1 are left untouched. Default ``False``.

    Returns:
        tuple: ``(types, segids)`` -- per-chain type codes (``'P'`` protein,
        ``'R'`` RNA, ``'D'`` DNA, ``'A'`` AGs/KAN) and segment IDs of the
        form ``<type><counter:03d>`` (e.g. ``P001``, ``R002``, ``A001``),
        counted per type.

    Raises:
        ValueError: If no ATOM record has a chain ID or segment ID, or if a
            chain's first residue name is not recognised.
    """
    aas = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    rnas = ["ADE", "GUA", "CYT", "URA", "A", "G", "C", "U"]
    dnas = ["DAD", "DGU", "DCY", "DTH", "DA", "DG", "DC", "DT"]
    ags = ["KAN"]
    counts = {'P': 1, 'R': 1, 'D': 1, 'A': 1}

    # HIS names
    HISs = ['HSD', 'HSE', 'HSP', 'HID', 'HIE', 'HIP']

    def get_type(resname):
        if resname in aas:
            return 'P'
        elif resname in rnas:
            return 'R'
        elif resname in dnas:
            return 'D'
        elif resname in ags:
            return 'A'
        return None

    # Variables to track current and previous chain identifiers
    current_chain_id = None
    current_segid = None
    prev_chain_id = None
    prev_segid = None
    
    chain_atoms = []
    chains = []
    types = []
    segids = []
    
    # First pass: check if chain_id and segid exist
    has_chain_id = False
    has_segid = False
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain_id = line[21].strip()
                segid = line[72:76].strip()
                if chain_id:
                    has_chain_id = True
                if segid:
                    has_segid = True
                if has_chain_id and has_segid:
                    break
    
    # Determine which identifier to use
    if has_chain_id and has_segid:
        # Check which one changes to determine usage
        chain_id_changes = []
        segid_changes = []
        prev_cid = None
        prev_sid = None
        
        with open(pdb, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    cid = line[21].strip()
                    sid = line[72:76].strip()
                    
                    if prev_cid is not None and cid != prev_cid:
                        chain_id_changes.append(True)
                    if prev_sid is not None and sid != prev_sid:
                        segid_changes.append(True)
                    
                    prev_cid = cid
                    prev_sid = sid
        
        # Use chain_id if it changes but segid doesn't, or if they change together
        use_chain_id = len(chain_id_changes) > 0 and (len(segid_changes) == 0 or len(chain_id_changes) == len(segid_changes))
    elif has_chain_id:
        use_chain_id = True
    elif has_segid:
        use_chain_id = False
    else:
        raise ValueError("Neither chain_id nor segid found in PDB file")
    
    # Second pass: split chains based on the determined identifier
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain_id = line[21].strip()
                segid = line[72:76].strip()
                resname = line[17:20].strip()
                resname = {"A": "ADE", "G": "GUA", "C": "CYT", "U": "URA"}.get(resname, resname)
                if resname in HISs:
                    resname = 'HIS'
                
                # Write the mapped resname back into the line
                line = line[:17] + resname.ljust(3) + line[20:]
                
                # Select the identifier to use
                identifier = chain_id if use_chain_id else segid
                
                if identifier != (current_chain_id if use_chain_id else current_segid):
                    if chain_atoms:
                        chains.append(chain_atoms)
                    
                    if use_chain_id:
                        current_chain_id = identifier
                    else:
                        current_segid = identifier
                    
                    mol_type = get_type(resname)
                    if mol_type is None:
                        raise ValueError(f'Unknown residue type: {resname}')
                    types.append(mol_type)
                    new_segid = f"{mol_type}{counts[mol_type]:03d}"
                    counts[mol_type] += 1
                    segids.append(new_segid)
                    chain_atoms = [line]
                else:
                    chain_atoms.append(line)
        
        if chain_atoms:
            chains.append(chain_atoms)

    # Save each chain to temporary file
    for i, chain in enumerate(chains):
        if renumber:
            chain = _renumber_chain(chain)
        with open(f"aa2cgtmp_{i}_aa.pdb", 'w') as f:
            for line in chain:
                f.write(line)
            f.write('END\n')
    
    return types, segids


def set_terminus(gen, segid, terminal):
    """Set terminus charges on a protein segment in a ``psfgen`` session.

    Segments whose ID does not start with ``P`` are left unchanged. The
    N-terminus is atom ``N`` of the first residue and the C-terminus is atom
    ``O`` of the last residue.

    Args:
        gen (psfgen.PsfGen): Session containing the segment.
        segid (str): Segment ID.
        terminal (str): ``'neutral'`` (no change), ``'charged'`` (N +1.00,
            O -1.00), ``'NT'`` (N +1.00 only) or ``'CT'`` (O -1.00 only).

    Raises:
        ValueError: For any other ``terminal`` value on a protein segment.
    """
    if not segid.startswith("P"):
        return
        
    resids = gen.get_resids(segid)
    nter, cter = resids[0], resids[-1]
    
    if terminal == 'charged':
        gen.set_charge(segid, nter, "N", 1.00)
        gen.set_charge(segid, cter, "O", -1.00)
    elif terminal == 'NT':
        gen.set_charge(segid, nter, "N", 1.00)
    elif terminal == 'CT':
        gen.set_charge(segid, cter, "O", -1.00)
    elif terminal == 'neutral':
        pass
    else:
        raise ValueError("Only 'neutral', 'charged', 'NT', and 'CT' are supported.")


def at2hyres(pdb_in, pdb_out):
    """
    Convert an all-atom protein PDB to a HyRes coarse-grained PDB.

    Backbone atoms (N, H, CA, C, O) are preserved at their original positions
    (``HN``/``HT1`` are renamed H and ``OT1`` is renamed O; ``OT2``/``OXT``
    and all other hydrogens are dropped). Sidechain heavy atoms are collapsed
    into one or more beads at their geometric center, named CB, CC, CD, CE,
    CF in order: one bead for ALA, VAL, LEU, ILE, MET, ASN, ASP, GLN, GLU,
    CYS, SER, THR, PRO; two for LYS, ARG; three for HIS, PHE, TYR; five for
    TRP; none for GLY. HSD, HSE and HSP are renamed HIS. Atoms are written
    per residue as: N/H/CA (input order), sidechain beads, then C/O (input
    order). Residues are keyed by residue number only, so the input should
    be a single chain. Atom serial numbers are encoded in hybrid-36 format
    above 99,999.

    Args:
        pdb_in (str): Path to the input all-atom PDB file.
        pdb_out (str): Path to the output HyRes coarse-grained PDB file.

    Returns:
        None. Writes a CG PDB file to ``pdb_out``.

    Raises:
        SystemExit: If an unrecognized residue type is encountered.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> Convert2CG.at2hyres("protein_aa.pdb", "protein_cg.pdb")
    """
    
    def encode_serial(n):
        """Encode integer to hybrid-36 format for PDB serial number field (5 chars)."""
        if n < 100000:
            return f"{n:5d}"

        n -= 100000
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        if n < 26 * (36**4):  # uppercase range
            result = []
            for _ in range(4):
                n, remainder = divmod(n, 36)
                result.append(chars[remainder])
            result.append(chr(ord('A') + n))
            return ''.join(reversed(result))

        n -= 26 * (36**4)  # lowercase range
        chars_lower = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = []
        for _ in range(4):
            n, remainder = divmod(n, 36)
            result.append(chars_lower[remainder])
        result.append(chr(ord('a') + n))
        return ''.join(reversed(result))
    
    # Parse PDB file into residues
    residues = {}
    atom_count = 0
    
    with open(pdb_in, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
                
            atom_count += 1
            resid = int(line[22:26].strip())
            
            if resid not in residues:
                residues[resid] = {}
            
            atom_idx = len(residues[resid]) + 1
            name = line[12:16].strip()
            if name in ['HN', 'HT1', 'H']:
                name = 'H'
            elif name in ['O', 'OT1']:
                name = 'O'
            elif name in ['OT2', 'OXT']:
                continue
            elif name.startswith('H'):
                continue
            
            residues[resid][atom_idx] = {
                'record': line[:4].strip(),
                'serial': line[4:11].strip(),
                'name': name,
                'resname': line[17:20].strip(),
                'chain': line[21],
                'resid': line[22:26].strip(),
                'x': float(line[30:38].strip()),
                'y': float(line[38:46].strip()),
                'z': float(line[46:54].strip()),
                'occ': float(line[54:60].strip()) if line[54:60].strip() else 1.00,
                'bfac': float(line[60:66].strip()) if line[60:66].strip() else 0.00,
                'segid': line[72:76].strip() if len(line) > 72 else ''
            }

    num_residues = len(residues)
    print(f"Processing {atom_count} atoms / {num_residues} residues")

    # Rename histidine variants to HIS
    for resid in residues:
        first_atom = residues[resid][1]
        if first_atom['resname'] in ['HSD', 'HSE', 'HSP']:
            for atom in residues[resid].values():
                atom['resname'] = 'HIS'

    # Mapping rules for residues
    single_bead_sc = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'ASN', 'ASP', 
                      'GLN', 'GLU', 'CYS', 'SER', 'THR', 'PRO']
    
    sc_mapping = {
        'LYS': [['CB', 'CG', 'CD'], ['CE', 'NZ']],
        'ARG': [['CB', 'CG', 'CD'], ['NE', 'CZ', 'NH1', 'NH2']],
        'HIS': [['CB', 'CG'], ['CD2', 'NE2'], ['ND1', 'CE1']],
        'PHE': [['CB', 'CG', 'CD1'], ['CD2', 'CE2'], ['CE1', 'CZ']],
        'TYR': [['CB', 'CG', 'CD1'], ['CD2', 'CE2'], ['CE1', 'CZ', 'OH']],
        'TRP': [['CB', 'CG'], ['CD1', 'NE1'], ['CD2', 'CE2'], ['CZ2', 'CH2'], ['CE3', 'CZ3']]
    }
    
    bb_atoms = ['CA', 'C', 'O', 'N', 'H']
    bb_atoms_1 = ['CA', 'N', 'H']
    bb_atoms_2 = ['C', 'O']
    
    # Write CG PDB
    atom_serial = 0
    with open(pdb_out, 'w') as f:
        for resid in sorted(residues.keys()):
            res = residues[resid]
            first_atom = res[1]
            resname = first_atom['resname']
            
            # Get sidechain beads for this residue
            if resname in sc_mapping:
                sc_beads = sc_mapping[resname]
            elif resname in single_bead_sc:
                # Collect all non-backbone atoms as single sidechain bead
                sc_beads = [[atom['name'] for atom in res.values() if atom['name'] not in bb_atoms]]
            elif resname in ['HSD', 'HSE', 'HSP', 'HID', 'HIE', 'HIP']:
                resname = 'HIS'
            elif resname != 'GLY':
                print(f"Error: Unknown residue type {resname}")
                exit(1)
            else:
                sc_beads = []
            
            # Calculate sidechain bead centers
            sc_centers = []
            for bead_atoms in sc_beads:
                coords = []
                for atom in res.values():
                    if atom['name'] in bead_atoms:
                        coords.append([atom['x'], atom['y'], atom['z']])
                if coords:
                    center = np.mean(coords, axis=0)
                    sc_centers.append(center)
            
            # Write backbone atoms (first group)
            for atom in res.values():
                if atom['name'] in bb_atoms_1:
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    f.write(f"{atom['record']:4s}  {serial_str} {atom['name']:2s}   "
                           f"{resname:3s} {atom['chain']}{int(atom['resid']):4d}    "
                           f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                           f"{atom['occ']:6.2f}{atom['bfac']:6.2f}      {atom['segid']:4s}\n")
            
            # Write sidechain beads
            bead_names = ['CB', 'CC', 'CD', 'CE', 'CF']
            for i, center in enumerate(sc_centers):
                atom_serial += 1
                serial_str = encode_serial(atom_serial)
                f.write(f"{first_atom['record']:4s}  {serial_str} {bead_names[i]:2s}   "
                       f"{resname:3s} {first_atom['chain']}{int(first_atom['resid']):4d}    "
                       f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                       f"{first_atom['occ']:6.2f}{first_atom['bfac']:6.2f}      "
                       f"{first_atom['segid']:4s}\n")
            
            # Write backbone C and O atoms (second group)
            for atom in res.values():
                if atom['name'] in bb_atoms_2:
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    f.write(f"{atom['record']:4s}  {serial_str} {atom['name']:2s}   "
                           f"{resname:3s} {atom['chain']}{int(atom['resid']):4d}    "
                           f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                           f"{atom['occ']:6.2f}{atom['bfac']:6.2f}      {atom['segid']:4s}\n")
        
        f.write("END\n")
    
    print(f"At2Hyres conversion done, output written to {pdb_out}")


def at2RNA(pdb_in, pdb_out):
    """
    Convert an all-atom RNA PDB to an iConRNA coarse-grained PDB.

    Each nucleotide is mapped onto a set of coarse-grained beads:

    - **P** — phosphate group (P, O1P, O2P, O5', plus O3' of residue
      ``resid - 1`` in the same segment)
    - **C1** — sugar bead at C4'
    - **C2** — sugar bead at C1'
    - **NA/NB/NC/ND** — base beads (four for ADE/GUA, three for CYT/URA)

    Bead coordinates are computed as the geometric center of the contributing
    all-atom positions; a bead is omitted if none of its atoms are present.
    Residues are grouped by segment ID and written in sorted segid/resid
    order. Supported nucleotides: ADE, GUA, CYT, URA (other residues get only
    P/C1/C2 beads). Atom serials use hybrid-36 above 99,999.

    Args:
        pdb_in (str): Path to the input all-atom RNA PDB file.
        pdb_out (str): Path to the output iConRNA coarse-grained PDB file.

    Returns:
        None. Writes a CG PDB file to ``pdb_out``.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> Convert2CG.at2RNA("rna_aa.pdb", "rna_cg.pdb")
    """
    
    def encode_serial(n):
        """Encode integer to hybrid-36 format for PDB serial number field (5 chars)."""
        if n < 100000:
            return f"{n:5d}"

        n -= 100000
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        if n < 26 * (36**4):  # uppercase range
            result = []
            for _ in range(4):
                n, remainder = divmod(n, 36)
                result.append(chars[remainder])
            result.append(chr(ord('A') + n))
            return ''.join(reversed(result))

        n -= 26 * (36**4)  # lowercase range
        chars_lower = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = []
        for _ in range(4):
            n, remainder = divmod(n, 36)
            result.append(chars_lower[remainder])
        result.append(chr(ord('a') + n))
        return ''.join(reversed(result))
    
    # Parse PDB file
    atoms = []
    with open(pdb_in, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms.append({
                    'name': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resid': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'segid': line[72:76].strip() if len(line) > 72 else ''
                })
    
    # Group by segment and residue
    segments = {}
    for atom in atoms:
        segid = atom['segid']
        resid = atom['resid']
        if segid not in segments:
            segments[segid] = {}
        if resid not in segments[segid]:
            segments[segid][resid] = {
                'resname': atom['resname'], 
                'chain': atom['chain'], 
                'atoms': []
            }
        segments[segid][resid]['atoms'].append(atom)
    
    # Base bead mappings for each nucleotide
    base_mappings = {
        'ADE': [
            ('NA', ['N9', 'C4']),
            ('NB', ['C8', 'H8', 'N7', 'C5']),
            ('NC', ['C6', 'N1', 'N6', 'H61', 'H62']),
            ('ND', ['C2', 'H2', 'N3'])
        ],
        'GUA': [
            ('NA', ['N9', 'C4']),
            ('NB', ['C8', 'H8', 'N7', 'C5']),
            ('NC', ['C6', 'N1', 'H1', 'O6']),
            ('ND', ['C2', 'N2', 'H21', 'H22', 'N3'])
        ],
        'CYT': [
            ('NA', ['N1', 'C5', 'H5', 'C6', 'H6']),
            ('NB', ['C4', 'N4', 'H41', 'H42', 'N3']),
            ('NC', ['C2', 'O2'])
        ],
        'URA': [
            ('NA', ['N1', 'C5', 'H5', 'C6', 'H6']),
            ('NB', ['C4', 'O4', 'N3', 'H3']),
            ('NC', ['C2', 'O2'])
        ]
    }
    
    atom_serial = 0
    with open(pdb_out, 'w') as f:
        for segid in sorted(segments.keys()):
            for resid in sorted(segments[segid].keys()):
                res_data = segments[segid][resid]
                resname = res_data['resname']
                chain = res_data['chain']
                res_atoms = res_data['atoms']
                
                # P bead (phosphate group)
                p_atoms = [a for a in res_atoms if a['name'] in ["P", "O1P", "O2P", "O5'"]]
                # Add O3' from previous residue
                if resid - 1 in segments[segid]:
                    prev_atoms = segments[segid][resid - 1]['atoms']
                    p_atoms.extend([a for a in prev_atoms if a['name'] == "O3'"])
                
                if p_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in p_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    f.write(f"ATOM  {serial_str}  P   {resname:3s} {chain}{resid:4d}    "
                           f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                           f"  1.00  0.00      {segid:4s}\n")
                
                # C1 bead (C4' sugar)
                c1_atoms = [a for a in res_atoms if a['name'] == "C4'"]
                if c1_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in c1_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    f.write(f"ATOM  {serial_str}  C1  {resname:3s} {chain}{resid:4d}    "
                           f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                           f"  1.00  0.00      {segid:4s}\n")
                
                # C2 bead (C1' sugar)
                c2_atoms = [a for a in res_atoms if a['name'] == "C1'"]
                if c2_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in c2_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    f.write(f"ATOM  {serial_str}  C2  {resname:3s} {chain}{resid:4d}    "
                           f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                           f"  1.00  0.00      {segid:4s}\n")
                
                # Base beads
                if resname in base_mappings:
                    for bead_name, atom_names in base_mappings[resname]:
                        base_atoms = [a for a in res_atoms if a['name'] in atom_names]
                        if base_atoms:
                            coords = np.array([[a['x'], a['y'], a['z']] for a in base_atoms])
                            center = coords.mean(axis=0)
                            atom_serial += 1
                            serial_str = encode_serial(atom_serial)
                            f.write(f"ATOM  {serial_str}  {bead_name:2s}  {resname:3s} "
                                   f"{chain}{resid:4d}    "
                                   f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                                   f"  1.00  0.00      {segid:4s}\n")
        
        f.write('END\n')
    
    print(f'At2RNA conversion done, output written to {pdb_out}')


def at2DNA(pdb_in, pdb_out):
    """
    Convert an all-atom DNA PDB to an iConRNA-style coarse-grained PDB.

    Uses the same CG bead topology as :func:`at2RNA` (P, C1, C2, and
    NA–ND base beads, all placed at the geometric center of their
    contributing all-atom coordinates), applied to deoxyribonucleotides.

    - **P** — phosphate group (P, O1P, O2P, O5', plus O3' of residue
      ``resid - 1`` in the same segment)
    - **C1** — sugar bead at C4'
    - **C2** — sugar bead at C1'
    - **NA/NB/NC/ND** — base beads (four for DA/DG, three for DC/DT)

    Supported nucleotides: DA, DG, DC, DT, corresponding respectively to the
    RNA nucleotides ADE, GUA, CYT, URA, plus the aliases DAD, DGU, DCY and
    DTH, which share the same mappings. Thymine's 5-methyl group (C7, H71,
    H72, H73) is folded into its NB bead. Other residues get only P/C1/C2
    beads. Atom serials use hybrid-36 above 99,999.

    Args:
        pdb_in (str): Path to the input all-atom DNA PDB file.
        pdb_out (str): Path to the output coarse-grained PDB file.

    Returns:
        None. Writes a CG PDB file to ``pdb_out``.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> Convert2CG.at2DNA("dna_aa.pdb", "dna_cg.pdb")
    """
    
    def encode_serial(n):
        """Encode integer to hybrid-36 format for PDB serial number field (5 chars)."""
        if n < 100000:
            return f"{n:5d}"

        n -= 100000
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        if n < 26 * (36**4):  # uppercase range
            result = []
            for _ in range(4):
                n, remainder = divmod(n, 36)
                result.append(chars[remainder])
            result.append(chr(ord('A') + n))
            return ''.join(reversed(result))

        n -= 26 * (36**4)  # lowercase range
        chars_lower = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = []
        for _ in range(4):
            n, remainder = divmod(n, 36)
            result.append(chars_lower[remainder])
        result.append(chr(ord('a') + n))
        return ''.join(reversed(result))
    
    # Parse PDB file
    atoms = []
    with open(pdb_in, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms.append({
                    'name': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resid': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'segid': line[72:76].strip() if len(line) > 72 else ''
                })
    
    # Group by segment and residue
    segments = {}
    for atom in atoms:
        segid = atom['segid']
        resid = atom['resid']
        if segid not in segments:
            segments[segid] = {}
        if resid not in segments[segid]:
            segments[segid][resid] = {
                'resname': atom['resname'], 
                'chain': atom['chain'], 
                'atoms': []
            }
        segments[segid][resid]['atoms'].append(atom)
    
    # Base bead mappings for each deoxyribonucleotide.
    # Same topology as the RNA model (ADE->DA, GUA->DG, CYT->DC, URA->DT),
    # with DT's 5-methyl group (C7/H71/H72/H73) folded into its NB bead in
    # place of RNA's corresponding uracil H5 atom.
    base_mappings = {
        'DA': [
            ('NA', ['N9', 'C4']),
            ('NB', ['C8', 'H8', 'N7', 'C5']),
            ('NC', ['C6', 'N1', 'N6', 'H61', 'H62']),
            ('ND', ['C2', 'H2', 'N3'])
        ],
        'DG': [
            ('NA', ['N9', 'C4']),
            ('NB', ['C8', 'H8', 'N7', 'C5']),
            ('NC', ['C6', 'N1', 'H1', 'O6']),
            ('ND', ['C2', 'N2', 'H21', 'H22', 'N3'])
        ],
        'DC': [
            ('NA', ['N1', 'C5', 'H5', 'C6', 'H6']),
            ('NB', ['C4', 'N4', 'H41', 'H42', 'N3']),
            ('NC', ['C2', 'O2'])
        ],
        'DT': [
            ('NA', ['N1', 'C5', 'C6', 'H6']),
            ('NB', ['C4', 'O4', 'N3', 'H3', 'C7', 'H71', 'H72', 'H73']),
            ('NC', ['C2', 'O2'])
        ]
    }

    # CHARMM-style 3-letter DNA names share the same bead mappings.
    for _alias, _canonical in (('DAD', 'DA'), ('DGU', 'DG'),
                               ('DCY', 'DC'), ('DTH', 'DT')):
        base_mappings[_alias] = base_mappings[_canonical]
    
    atom_serial = 0
    with open(pdb_out, 'w') as f:
        for segid in sorted(segments.keys()):
            for resid in sorted(segments[segid].keys()):
                res_data = segments[segid][resid]
                resname = res_data['resname']
                chain = res_data['chain']
                res_atoms = res_data['atoms']
                
                # P bead (phosphate group)
                p_atoms = [a for a in res_atoms if a['name'] in ["P", "O1P", "O2P", "O5'"]]
                # Add O3' from previous residue
                if resid - 1 in segments[segid]:
                    prev_atoms = segments[segid][resid - 1]['atoms']
                    p_atoms.extend([a for a in prev_atoms if a['name'] == "O3'"])
                
                if p_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in p_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    f.write(f"ATOM  {serial_str}  P   {resname:3s} {chain}{resid:4d}    "
                           f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                           f"  1.00  0.00      {segid:4s}\n")
                
                # C1 bead (C4' sugar)
                c1_atoms = [a for a in res_atoms if a['name'] == "C4'"]
                if c1_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in c1_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    f.write(f"ATOM  {serial_str}  C1  {resname:3s} {chain}{resid:4d}    "
                           f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                           f"  1.00  0.00      {segid:4s}\n")
                
                # C2 bead (C1' sugar)
                c2_atoms = [a for a in res_atoms if a['name'] == "C1'"]
                if c2_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in c2_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    f.write(f"ATOM  {serial_str}  C2  {resname:3s} {chain}{resid:4d}    "
                           f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                           f"  1.00  0.00      {segid:4s}\n")
                
                # Base beads
                if resname in base_mappings:
                    for bead_name, atom_names in base_mappings[resname]:
                        base_atoms = [a for a in res_atoms if a['name'] in atom_names]
                        if base_atoms:
                            coords = np.array([[a['x'], a['y'], a['z']] for a in base_atoms])
                            center = coords.mean(axis=0)
                            atom_serial += 1
                            serial_str = encode_serial(atom_serial)
                            f.write(f"ATOM  {serial_str}  {bead_name:2s}  {resname:3s} "
                                   f"{chain}{resid:4d}    "
                                   f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                                   f"  1.00  0.00      {segid:4s}\n")
        
        f.write('END\n')
    
    print(f'At2DNA conversion done, output written to {pdb_out}')


def at2AGs(pdb_in, pdb_out):
    """
    Convert an all-atom aminoglycoside (AGs) PDB to a coarse-grained PDB.

    Each AGs residue has its own mapping; every bead is placed at the
    geometric center of its listed atoms and omitted if none are present.
    Currently mapped: kanamycin A (``KAN``, beads K1-K11). ``LLL``
    (gentamicin C1a) is a placeholder with empty atom lists, so it produces
    no beads, as do residues without a mapping. Residues are grouped by
    segment ID and written in sorted segid/resid order; atom serials use
    hybrid-36 above 99,999.

    Args:
        pdb_in (str): Path to the input all-atom AGs PDB file.
        pdb_out (str): Path to the output coarse-grained PDB file.

    Returns:
        None. Writes a CG PDB file to ``pdb_out``.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> Convert2CG.at2AGs("kan_aa.pdb", "kan_cg.pdb")
    """
    
    def encode_serial(n):
        """Encode integer to hybrid-36 format for PDB serial number field (5 chars)."""
        if n < 100000:
            return f"{n:5d}"

        n -= 100000
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        if n < 26 * (36**4):  # uppercase range
            result = []
            for _ in range(4):
                n, remainder = divmod(n, 36)
                result.append(chars[remainder])
            result.append(chr(ord('A') + n))
            return ''.join(reversed(result))

        n -= 26 * (36**4)  # lowercase range
        chars_lower = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = []
        for _ in range(4):
            n, remainder = divmod(n, 36)
            result.append(chars_lower[remainder])
        result.append(chr(ord('a') + n))
        return ''.join(reversed(result))
    
    # Parse PDB file
    atoms = []
    with open(pdb_in, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms.append({
                    'name': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resid': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'segid': line[72:76].strip() if len(line) > 72 else ''
                })
    
    # Group by segment and residue
    segments = {}
    for atom in atoms:
        segid = atom['segid']
        resid = atom['resid']
        if segid not in segments:
            segments[segid] = {}
        if resid not in segments[segid]:
            segments[segid][resid] = {
                'resname': atom['resname'], 
                'chain': atom['chain'], 
                'atoms': []
            }
        segments[segid][resid]['atoms'].append(atom)
    
    # Base bead mappings for each nucleotide
    AGs_mappings = {
        # kanamycin A
        'KAN': [
            ('K1',  ['C8', 'C9', 'C10', 'O10']),
            ('K2',  ['C11', 'C12', 'N2']),
            ('K3',  ['C7', 'C12', 'N3']),
            ('K4',  ['O11', 'C13', 'O12']),
            ('K5',  ['C14', 'C15', 'N4', 'O13']),
            ('K6',  ['C16', 'C17', 'O14']),
            ('K7',  ['C1', 'O5', 'O9']),
            ('K8',  ['C2', 'C3', 'O6', 'O7']),
            ('K9',  ['C4', 'C5', 'O8']),
            ('K10', ['C18', 'O15']),
            ('K11', ['C6', 'N1'])
        ],
        # gentamicin C1a
        'LLL': [
            ('K1',  []),
            ('K2',  []),
            ('K3',  []),
            ('K4',  []),
            ('K5',  []),
            ('K6',  []),
            ('K7',  []),
            ('K8',  []),
            ('K9',  []),
            ('K10', []),
            ('K11', []),
        ],
    }
    
    atom_serial = 0
    with open(pdb_out, 'w') as f:
        for segid in sorted(segments.keys()):
            for resid in sorted(segments[segid].keys()):
                res_data = segments[segid][resid]
                resname = res_data['resname']
                chain = res_data['chain']
                res_atoms = res_data['atoms']
                
                # Base beads
                if resname in AGs_mappings:
                    for bead_name, atom_names in AGs_mappings[resname]:
                        base_atoms = [a for a in res_atoms if a['name'] in atom_names]
                        if base_atoms:
                            coords = np.array([[a['x'], a['y'], a['z']] for a in base_atoms])
                            center = coords.mean(axis=0)
                            atom_serial += 1
                            serial_str = encode_serial(atom_serial)
                            f.write(f"ATOM  {serial_str}  {bead_name:<4s}{resname:3s} "
                                   f"{chain}{resid:4d}    "
                                   f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                                   f"  1.00  0.00      {segid:4s}\n")
        
        f.write('END\n')
    
    print(f'at2AGs conversion done, output written to {pdb_out}')

def fix_pdb_serial(pdb_file, output_file=None):
    """
    Fix PDB files where atom serial numbers exceed 99999 and have been written
    as '******' by psfgen-python. Re-numbers all ATOM/HETATM records
    sequentially from 1 using hybrid-36 encoding, so serial numbers beyond
    99999 are written as ``A0000``-``ZZZZZ``, then ``a0000``-``zzzzz``.
    Other records are copied unchanged.

    Args:
        pdb_file (str): Path to the input PDB file.
        output_file (str, optional): Path to the output fixed PDB file. If
            None, the input file is overwritten in place.

    Returns:
        str: Path to the fixed PDB file.
    """

    def _encode_serial(n):
        """Encode integer to hybrid-36 format for PDB serial number field (5 chars)."""
        if n < 100000:
            return f"{n:5d}"

        n -= 100000
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        if n < 26 * (36 ** 4):          # uppercase range: A0000 – Z9ZZZ
            result = []
            for _ in range(4):
                n, remainder = divmod(n, 36)
                result.append(chars[remainder])
            result.append(chr(ord('A') + n))
            return ''.join(reversed(result))

        n -= 26 * (36 ** 4)              # lowercase range: a0000 – z9ZZZ
        chars_lower = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = []
        for _ in range(4):
            n, remainder = divmod(n, 36)
            result.append(chars_lower[remainder])
        result.append(chr(ord('a') + n))
        return ''.join(reversed(result))

    if output_file is None:
        output_file = pdb_file

    fixed_lines = []
    serial = 0

    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            serial += 1
            # Replace columns 6–11 (0-indexed) with re-encoded serial.
            # Fixes both '******' entries and any truncated numeric serials.
            line = line[:6] + _encode_serial(serial) + line[11:]
        fixed_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)

    print(f"Fixed serial numbers for {serial} atoms. Output saved to {output_file}")
    return output_file

def at2cg(pdb_in, pdb_out, terminal='neutral', cleanup=True, renumber=False,
          charmm=False):
    """
    Convert an all-atom PDB to a coarse-grained PDB and PSF file.

    Automatically detects molecule types (protein, RNA, DNA or AGs) by chain
    (:func:`split_chains`), then applies :func:`at2hyres` for protein chains,
    :func:`at2RNA` for RNA, :func:`at2DNA` for DNA and :func:`at2AGs` for
    AGs (KAN). Each CG chain is added to psfgen (RNA, DNA, Protein and AGs
    topologies) using the segids from :func:`split_chains`; proteins use
    ``auto_angles=False``, the other types also ``auto_dihedrals=False``.
    psfgen writes ``pdb_out``, terminus charges are then set, and the PSF is
    written to ``pdb_out`` with its last four characters replaced by
    ``.psf``. Temporary ``aa2cgtmp_*.pdb`` files in the current directory are
    removed unless ``cleanup=False``. Finally ``pdb_out`` is rewritten by
    :func:`fix_pdb_serial` (hybrid-36 serials above 99,999).

    Args:
        pdb_in (str): Path to the input all-atom PDB file. May contain mixed
                      protein, RNA, and DNA chains.
        pdb_out (str): Path to the output coarse-grained PDB file.
        terminal (str, optional): Charge status of protein termini. Options:

                                  - ``'neutral'`` — uncharged termini (default)
                                  - ``'charged'`` — both termini charged
                                  - ``'NT'`` — N-terminus charged only
                                  - ``'CT'`` — C-terminus charged only

        cleanup (bool, optional): If ``True``, removes intermediate temporary
                                  PDB files after conversion. Default is ``True``.
        renumber (bool, optional): If ``True``, checks each chain/segment's
                                  starting residue number and, if it doesn't
                                  already start from 1, renumbers that
                                  segment's residues sequentially from 1.
                                  Segments already starting from 1 are left
                                  unchanged. Default is ``False``.
        charmm (bool, optional): If ``True``, treat the input as a CHARMM-style
                                  PDB in which DNA bases share the RNA residue
                                  names, and rename ADE→DA, GUA→DG,
                                  CYT→DC, THY→DT before chain splitting so
                                  those chains are detected as DNA. Only use
                                  this when the nucleic acid chains really are
                                  DNA. Default is ``False``.

    Returns:
        tuple: A 2-tuple ``(pdb_file, psf_file)`` with paths to the output
               coarse-grained PDB and PSF files.

    Raises:
        ValueError: If the input has neither chain IDs nor segment IDs, a
            chain starts with an unrecognised residue, or ``terminal`` is not
            supported (raised after ``pdb_out`` has been written).
        SystemExit: If :func:`at2hyres` meets an unknown amino acid.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> pdb, psf = Convert2CG.at2cg("system_aa.pdb", "system_cg.pdb")
        >>> pdb, psf = Convert2CG.at2cg("system_aa.pdb", "system_cg.pdb",
        ...                              terminal="charged")
        >>> pdb, psf = Convert2CG.at2cg("charmm_dna.pdb", "dna_cg.pdb",
        ...                              charmm=True)
    """
    
    # Load topology files
    RNA_topology, _ = load_ff('RNA')
    DNA_topology, _ = load_ff('DNA')
    protein_topology, _ = load_ff('Protein')
    AGs_topology, _ = load_ff('AGs')
    
    # Set up psfgen
    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(DNA_topology)
    gen.read_topology(protein_topology)
    gen.read_topology(AGs_topology)
    
    # CHARMM-style PDBs name DNA bases like RNA (ADE/GUA/CYT/THY); rename them
    # to DAD/DGU/DCY/DTH so split_chains() types these chains as DNA.
    if charmm:
        charmm_pdb = "aa2cgtmp_charmm_aa.pdb"
        fix_charmm_dna_resnames(pdb_in, charmm_pdb)
        pdb_in = charmm_pdb

    # Split chains and convert
    types, segids = split_chains(pdb_in, renumber=renumber)
    
    for i, (mol_type, segid) in enumerate(zip(types, segids)):
        tmp_pdb = f"aa2cgtmp_{i}_aa.pdb"
        tmp_cg_pdb = f"aa2cgtmp_{i}_cg.pdb"
        
        if mol_type == 'P':
            at2hyres(tmp_pdb, tmp_cg_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_cg_pdb, auto_angles=False)
            gen.read_coords(segid=segid, filename=tmp_cg_pdb)
        elif mol_type == 'R':
            at2RNA(tmp_pdb, tmp_cg_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_cg_pdb, 
                          auto_angles=False, auto_dihedrals=False)
            gen.read_coords(segid=segid, filename=tmp_cg_pdb)
        elif mol_type == 'D':
            at2DNA(tmp_pdb, tmp_cg_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_cg_pdb, 
                          auto_angles=False, auto_dihedrals=False)
            gen.read_coords(segid=segid, filename=tmp_cg_pdb)
        elif mol_type == 'A':
            at2AGs(tmp_pdb, tmp_cg_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_cg_pdb, 
                          auto_angles=False, auto_dihedrals=False)
            gen.read_coords(segid=segid, filename=tmp_cg_pdb)
        else:
            raise ValueError(f"Unsupported molecule type: {mol_type}")
    
    # Write PDB file
    gen.write_pdb(pdb_out)
    print(f"Conversion done, output written to {pdb_out}")
    
    # Set terminus charge status
    for segid in gen.get_segids():
        set_terminus(gen, segid, terminal)
    
    # Write PSF file
    psf_file = f'{pdb_out[:-4]}.psf'
    gen.write_psf(filename=psf_file)
    print(f"PSF file written to {psf_file}")
    
    # Clean up temporary files
    if cleanup:
        for file in os.listdir():
            if file.startswith("aa2cgtmp_") and file.endswith(".pdb"):
                os.remove(file)
    
    # psfgen-python cannot encode serial numbers > 99999; it writes '******'
    # for those atoms. Re-number every ATOM/HETATM record sequentially using
    # hybrid-36 so the output PDB is always valid.
    fix_pdb_serial(pdb_out, pdb_out)

    return pdb_out, psf_file

def main():
    """Command-line interface (``convert2cg`` console script).

    Usage::

        convert2cg aa.pdb cg.pdb [--hydrogen] [-t neutral|charged|NT|CT]
                   [--renumber] [--charmm]

    Runs :func:`at2cg` (with ``cleanup=True``) and writes ``cg.pdb`` and
    ``cg.psf``. ``--hydrogen`` first runs :func:`add_backbone_hydrogen`,
    writing ``<aa minus last 4 chars>_addH.pdb``, which is kept and used as
    the input. ``-t/--terminal`` defaults to ``neutral``; ``--renumber`` and
    ``--charmm`` map to the :func:`at2cg` arguments of the same name.
    ``UserWarning`` messages are suppressed.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert2CG: All-atom to HyRes/iConRNA converting'
    )
    parser.add_argument('aa', help='Input PDB file')
    parser.add_argument('cg', help='Output PDB file')
    parser.add_argument('--hydrogen', action='store_true',
                        help='add backbone amide hydrogen (H-N only), default False')
    parser.add_argument('--terminal', '-t', type=str, default='neutral',
                        help='Charge status of terminus: neutral, charged, NT, CT')
    parser.add_argument('--renumber', action='store_true',
                        help='renumber each segment\'s residues to start from 1 '
                             'if it does not already, default False')
    parser.add_argument('--charmm', action='store_true',
                        help='input is a CHARMM-style PDB whose DNA bases are '
                             'named like RNA; rename ADE->DAD, GUA->DGU, '
                             'CYT->DCY, THY->DTH before conversion so the '
                             'chains are treated as DNA, default False')

    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=UserWarning)

    if args.hydrogen:
        pdb_addH = add_backbone_hydrogen(args.aa, f'{args.aa[:-4]}_addH.pdb')
        at2cg(pdb_addH, args.cg, terminal=args.terminal,
              renumber=args.renumber, charmm=args.charmm)
    else:
        at2cg(args.aa, args.cg, terminal=args.terminal,
              renumber=args.renumber, charmm=args.charmm)

if __name__ == '__main__':
    main()