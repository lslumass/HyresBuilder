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
    rnas = ["ADE", "GUA", "CYT", "URA"]
    dnas = ["DAD", "DGU", "DCY", "DTH"]
    counts = {'P': 0, 'R': 0, 'D': 0}

    # HIS names
    HISs = ['HSD', 'HSE', 'HSP', 'HID', 'HIE', 'HIP']

    def get_type(resname):
        if resname in aas:
            return 'P'
        elif resname in rnas:
            return 'R'
        elif resname in dnas:
            return 'D'
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
    Convert all-atom protein to HyRes CG PDB.
    
    Parameters:
    -----------
    pdb_in : str
        Input all-atom PDB file
    pdb_out : str
        Output HyRes CG PDB file
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
    Convert all-atom RNA to iConRNA PDB.
    
    Parameters:
    -----------
    pdb_in : str
        Input all-atom PDB file
    pdb_out : str
        Output iConRNA PDB file
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


def at2cg(pdb_in, pdb_out, terminal='neutral', cleanup=True):
    """
    Convert all-atom PDB to CG PDB (HyRes for protein or iConRNA for RNA).
    
    Parameters:
    -----------
    pdb_in : str
        Input all-atom PDB file
    pdb_out : str
        Output CG PDB file
    terminal : str
        Charge status of protein terminus: 'neutral', 'charged', 'NT', 'CT'
    cleanup : bool
        Whether to clean up temporary files
    
    Returns:
    --------
    tuple : (pdb_file, psf_file)
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
    
    return pdb_out, psf_file


def main():
    """Command-line interface for Convert2CG."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert2CG: All-atom to HyRes/iConRNA converting'
    )
    parser.add_argument('aa', help='Input PDB file')
    parser.add_argument('cg', help='Output PDB file')
    parser.add_argument('--hydrogen', action='store_true', help='add backbone amide hydrogen (H-N only), default False')
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