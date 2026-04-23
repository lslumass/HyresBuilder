"""
Conversion utilities for all-atom to coarse-grained (CG) structure preparation.

This module provides the tools needed to convert all-atom protein and RNA
structures into coarse-grained representations compatible with the HyRes
(protein) and iConRNA (RNA) force fields. It handles the full conversion
pipeline — from optional backbone hydrogen addition through CG bead placement,
topology generation, and PSF writing — and can process mixed protein/RNA
systems in a single call.

Conversion models
-----------------
* **HyRes (protein)** — backbone heavy atoms (N, H, CA, C, O) are retained at
  their original positions; sidechain heavy atoms are collapsed into one to five
  geometric-center beads named CB, CC, CD, CE, CF depending on residue type.
  Glycine carries no sidechain bead. Histidine variants (HSD, HSE, HSP) are
  unified under the HIS residue name (:func:`at2hyres`).
* **iConRNA (RNA)** — each nucleotide is mapped to a phosphate bead (P), two
  sugar beads (C1 at C4′, C2 at C1′), and two to four base beads (NA–ND),
  all placed at the geometric center of their contributing all-atom coordinates.
  Supported nucleotides: ADE, GUA, CYT, URA (:func:`at2icon`).

Pipeline overview
-----------------
The top-level entry point :func:`at2cg` orchestrates the full workflow:

1. Optionally add backbone amide hydrogens to the all-atom input
   (:func:`add_backbone_hydrogen`).
2. Split the input PDB into per-chain temporary files and detect molecule
   types (:func:`split_chains`).
3. Apply the appropriate CG mapping per chain (:func:`at2hyres` or
   :func:`at2icon`).
4. Build topology and write PSF via ``psfgen``, set terminus charge states
   (:func:`set_terminus`), and re-encode any atom serial numbers exceeding
   99,999 in hybrid-36 format (:func:`fix_pdb_serial`).
5. Optionally remove intermediate temporary files.

A command-line interface is exposed via :func:`main` and registered as the
``Convert2CG`` entry point.

Hybrid-36 serial encoding
--------------------------
PDB format supports a maximum atom serial of 99,999. This module encodes
larger serials in hybrid-36 (base-36 alphanumeric strings: A0000–Z9ZZZ for
atoms 100,000–1,316,735, then a0000–z9ZZZ beyond that), ensuring output files
remain valid for large systems.

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.app``, ``openmm.unit``)
* `psfgen <https://github.com/MDAnalysis/psfgen>`_ (``psfgen.PsfGen``)
* `NumPy <https://numpy.org>`_ (``numpy``)
"""

from psfgen import PsfGen
import numpy as np
import os
import warnings
from .utils import load_ff


def add_backbone_hydrogen(pdb_file, output_file):
    """
    Add backbone hydrogen atoms (H) to peptide chains in a PDB file.
    Preserves all original atoms (backbone and side chains).
    
    Parameters:
    -----------
    pdb_file : str
        Path to input PDB file
    output_file : str
        Path to output PDB file with added H atoms
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
        return (f"ATOM  {serial_str}  {atom_name:3s} {residue_name:3s} {chain_id}{resseq_str}    "
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


def split_chains(pdb):
    """Split PDB file into separate chains and identify their types."""
    aas = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    rnas = ["ADE", "GUA", "CYT", "URA", "A", "G", "C", "U"]
    dnas = ["DAD", "DGU", "DCY", "DTH", "DA", "DG", "DC", "DT"]
    ags = ["KAN"]
    counts = {'P': 0, 'R': 0, 'D': 0, 'A': 0}

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
                resname = {"A": "ADE", "G": "GUA", "C": "CYT", "U": "URA",
                           "DA": "DAD", "DG": "DGU", "DC": "DCY", "DT": "DTH"}.get(resname, resname)
                if resname in HISs:
                    resname = 'HIS'
                
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
                        raise ValueError(f'Unknown residue type: {str(resname)}')
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
        with open(f"aa2cgtmp_{i}_aa.pdb", 'w') as f:
            for line in chain:
                f.write(line)
            f.write('END\n')
    
    return types, segids


def set_terminus(gen, segid, terminal):
    """Set the charge status of protein terminus."""
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

    Backbone atoms (N, H, CA, C, O) are preserved at their original positions.
    Sidechain heavy atoms are collapsed into one or more coarse-grained beads
    by computing their geometric center, named CB, CC, CD, CE, CF in order.
    Glycine residues have no sidechain bead. Histidine variants (HSD, HSE, HSP)
    are renamed to HIS. Atom serial numbers are encoded in hybrid-36 format to
    support systems with more than 99,999 atoms.

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


def at2icon(pdb_in, pdb_out):
    """
    Convert an all-atom RNA PDB to an iConRNA coarse-grained PDB.

    Each nucleotide is mapped onto a set of coarse-grained beads:

    - **P** — phosphate group (P, O1P, O2P, O5', O3' from previous residue)
    - **C1** — sugar bead at C4'
    - **C2** — sugar bead at C1'
    - **NA/NB/NC/ND** — base beads (number depends on nucleotide type)

    Bead coordinates are computed as the geometric center of the contributing
    all-atom positions. Supported nucleotides: ADE, GUA, CYT, URA.

    Args:
        pdb_in (str): Path to the input all-atom RNA PDB file.
        pdb_out (str): Path to the output iConRNA coarse-grained PDB file.

    Returns:
        None. Writes a CG PDB file to ``pdb_out``.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> Convert2CG.at2icon("rna_aa.pdb", "rna_cg.pdb")
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
    
    print(f'At2iCon conversion done, output written to {pdb_out}')

def at2AGs(pdb_in, pdb_out):
    """
    Convert an all-atom aminoglycosides (AGs) PDB to coarse-grained PDB.

    Each AGs has its specific mapping rules

    Args:
        pdb_in (str): Path to the input all-atom RNA PDB file.
        pdb_out (str): Path to the output iConRNA coarse-grained PDB file.

    Returns:
        None. Writes a CG PDB file to ``pdb_out``.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> Convert2CG.at2AGs("rna_aa.pdb", "rna_cg.pdb")
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
    
    print(f'At2iCon conversion done, output written to {pdb_out}')

def fix_pdb_serial(pdb_file, output_file=None):
    """
    Fix PDB files where atom serial numbers exceed 99999 and have been written
    as '******' by psfgen-python. Re-numbers all ATOM/HETATM records sequentially
    using hybrid-36 encoding so serial numbers beyond 99999 are represented as
    base-36 alphanumeric strings (A0000–Z9999, then a0000–z9999).

    Parameters:
    -----------
    pdb_file : str
        Path to the input PDB file containing '******' serial fields.
    output_file : str, optional
        Path to the output fixed PDB file.
        If None, the input file is overwritten in-place.

    Returns:
    --------
    str : Path to the fixed PDB file.
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

def at2cg(pdb_in, pdb_out, terminal='neutral', cleanup=True):
    """
    Convert an all-atom PDB to a coarse-grained PDB and PSF file.

    Automatically detects molecule types (protein or RNA) by chain, then
    applies :func:`at2hyres` for protein chains and :func:`at2icon` for RNA
    chains. Topology and connectivity are handled by psfgen, which also writes
    the PSF file. Atom serial numbers exceeding 99,999 are re-encoded in
    hybrid-36 format. Temporary intermediate files are removed after conversion
    unless ``cleanup=False``.

    Args:
        pdb_in (str): Path to the input all-atom PDB file. May contain mixed
                      protein and RNA chains.
        pdb_out (str): Path to the output coarse-grained PDB file.
        terminal (str, optional): Charge status of protein termini. Options:

                                  - ``'neutral'`` — uncharged termini (default)
                                  - ``'charged'`` — both termini charged
                                  - ``'NT'`` — N-terminus charged only
                                  - ``'CT'`` — C-terminus charged only

        cleanup (bool, optional): If ``True``, removes intermediate temporary
                                  PDB files after conversion. Default is ``True``.

    Returns:
        tuple: A 2-tuple ``(pdb_file, psf_file)`` with paths to the output
               coarse-grained PDB and PSF files.

    Raises:
        ValueError: If an unsupported molecule type is detected in the PDB.

    Example:
        >>> from HyresBuilder import Convert2CG
        >>> pdb, psf = Convert2CG.at2cg("system_aa.pdb", "system_cg.pdb")
        >>> pdb, psf = Convert2CG.at2cg("system_aa.pdb", "system_cg.pdb",
        ...                              terminal="charged")
    """
    
    # Load topology files
    RNA_topology, _ = load_ff('RNA')
    protein_topology, _ = load_ff('Protein')
    
    # Set up psfgen
    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)
    
    # Split chains and convert
    types, segids = split_chains(pdb_in)
    
    for i, (mol_type, segid) in enumerate(zip(types, segids)):
        tmp_pdb = f"aa2cgtmp_{i}_aa.pdb"
        tmp_cg_pdb = f"aa2cgtmp_{i}_cg.pdb"
        
        if mol_type == 'P':
            at2hyres(tmp_pdb, tmp_cg_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_cg_pdb, auto_angles=False)
            gen.read_coords(segid=segid, filename=tmp_cg_pdb)
        elif mol_type == 'R':
            at2icon(tmp_pdb, tmp_cg_pdb)
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
    """Command-line interface for Convert2CG."""
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

    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=UserWarning)

    if args.hydrogen:
        pdb_addH = add_backbone_hydrogen(args.aa, f'{args.aa[:-4]}_addH.pdb')
        at2cg(pdb_addH, args.cg, terminal=args.terminal)
    else:
        at2cg(args.aa, args.cg, terminal=args.terminal)

if __name__ == '__main__':
    main()
