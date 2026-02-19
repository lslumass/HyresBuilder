from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import os
import warnings

from psfgen import PsfGen

from .utils import load_ff


# --------------------------------------------------------------------------- #
#  Hybrid-36 helpers
# --------------------------------------------------------------------------- #

def encode_serial(n: int) -> str:
    """Encode integer to hybrid-36 format for the 5-character atom serial field."""
    if n < 100000:
        return f"{n:5d}"
    n -= 100000
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if n < 36 ** 5:
        leading = n // (36 ** 4)
        remainder = n % (36 ** 4)
        out = [chars[leading]]
        for _ in range(4):
            remainder, digit = divmod(remainder, 36)
            out.append(chars[digit])
        return ''.join(out)
    n -= 36 ** 5
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    leading = n // (36 ** 4)
    remainder = n % (36 ** 4)
    out = [chars[leading]]
    for _ in range(4):
        remainder, digit = divmod(remainder, 36)
        out.append(chars[digit])
    return ''.join(out)


def encode_resseq(n: int) -> str:
    """Encode integer to hybrid-36 format for the 4-character residue sequence field."""
    if n < 10000:
        return f"{n:4d}"
    n -= 10000
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if n < 36 ** 4:
        leading = n // (36 ** 3)
        remainder = n % (36 ** 3)
        out = [chars[leading]]
        for _ in range(3):
            remainder, digit = divmod(remainder, 36)
            out.append(chars[digit])
        return ''.join(out)
    n -= 36 ** 4
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    leading = n // (36 ** 3)
    remainder = n % (36 ** 3)
    out = [chars[leading]]
    for _ in range(3):
        remainder, digit = divmod(remainder, 36)
        out.append(chars[digit])
    return ''.join(out)


def parse_serial_field(field: str) -> int:
    """Inverse helper: decode decimal, hex, or hybrid-36 atom serial field."""
    s = field.strip()
    if not s:
        raise ValueError("Empty atom serial field")
    try:
        return int(s)
    except ValueError:
        try:  # hexadecimal overflow
            return int(s, 16)
        except ValueError:
            lead, tail = s[0], s[1:]
            value = int(tail, 36)
            if lead.isupper():
                offset = 100000
                base = ord('A')
            else:
                offset = 100000 + 26 * 36 ** 4
                base = ord('a')
            return offset + (ord(lead) - base) * 36 ** 4 + value


def parse_resseq_field(field: str) -> int:
    """Inverse helper: decode decimal or hybrid-36 residue sequence field."""
    s = field.strip()
    if not s:
        raise ValueError("Empty residue sequence field")
    try:
        return int(s)
    except ValueError:
        lead, tail = s[0], s[1:]
        value = int(tail, 36)
        if lead.isupper():
            offset = 10000
            base = ord('A')
        else:
            offset = 10000 + 26 * 36 ** 3
            base = ord('a')
        return offset + (ord(lead) - base) * 36 ** 3 + value


# --------------------------------------------------------------------------- #
#  Hydrogen builder
# --------------------------------------------------------------------------- #

def add_backbone_hydrogen(pdb_file, output_file):
    """
    Add backbone hydrogen atoms (H) to peptide chains in a PDB file.
    """

    def parse_atom_line(line):
        """Parse a PDB ATOM line and extract relevant information."""
        atom_serial = parse_serial_field(line[6:11])
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_seq = parse_resseq_field(line[22:26])
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
            'occupancy': occupancy if occupancy else "1.00",
            'temp_factor': temp_factor if temp_factor else "0.00",
            'element': element,
            'line': line
        }

    def format_atom_line(serial, atom_name, residue_name, chain_id, residue_seq,
                         coords, occupancy="1.00", temp_factor="0.00", element="H"):
        """Format an ATOM line in PDB format."""
        serial_str = encode_serial(serial)
        resseq_str = encode_resseq(residue_seq)
        atom_field = atom_name.rjust(4)
        return (
            f"ATOM  {serial_str} {atom_field} {residue_name:>3s} {chain_id}{resseq_str}"
            f"   {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
            f"{float(occupancy):6.2f}{float(temp_factor):6.2f}          {element:>2s}\n"
        )

    def calculate_h_position(n_coord, ca_coord, c_prev_coord):
        """Compute an idealized backbone amide hydrogen position."""
        v_cn = n_coord - c_prev_coord
        v_cn /= np.linalg.norm(v_cn)
        v_nca = ca_coord - n_coord
        v_nca /= np.linalg.norm(v_nca)
        bisector = v_cn - v_nca
        bisector /= np.linalg.norm(bisector)
        nh_bond_length = 1.01
        return n_coord + bisector * nh_bond_length

    with open(pdb_file, 'r') as fh:
        lines = fh.readlines()

    residue_data = []
    residue_lookup = {}
    current_segment_id = 0
    prev_chain = None
    prev_res_seq = None

    for idx, line in enumerate(lines):
        if not line.startswith('ATOM'):
            continue

        atom = parse_atom_line(line)
        chain = atom['chain']
        res_seq = atom['res_seq']
        is_new_segment = (
            prev_chain is None
            or chain != prev_chain
            or abs(res_seq - prev_res_seq) > 1
        )

        if is_new_segment and prev_chain is not None:
            current_segment_id += 1

        atom['line_idx'] = idx
        atom['segment_id'] = current_segment_id
        residue_data.append((idx, atom))

        key = (chain, res_seq, current_segment_id)
        residue_lookup.setdefault(key, {})
        residue_lookup[key][atom['name']] = atom

        prev_chain = chain
        prev_res_seq = res_seq

    output_lines = []
    current_serial = 1
    segments = defaultdict(list)

    for _, atom in residue_data:
        seg_id = atom['segment_id']
        key = (atom['chain'], atom['res_seq'], seg_id)
        if not segments[seg_id] or segments[seg_id][-1] != key:
            segments[seg_id].append(key)

    h_added = set()

    for idx, atom in residue_data:
        chain = atom['chain']
        res_seq = atom['res_seq']
        seg_id = atom['segment_id']
        key = (chain, res_seq, seg_id)

        output_lines.append(format_atom_line(
            current_serial, atom['name'], atom['residue'],
            atom['chain'], atom['res_seq'], atom['coords'],
            atom['occupancy'], atom['temp_factor'],
            atom['element'] if atom['element'] else atom['name'][0]
        ))
        current_serial += 1

        if atom['name'] == 'N' and key not in h_added:
            res_atoms = residue_lookup[key]

            if atom['residue'] == 'PRO':
                continue

            if 'CA' not in res_atoms:
                continue
            h_added.add(key)

            segment_residues = segments[seg_id]
            is_first = (key == segment_residues[0])
            c_coord = None
            if not is_first:
                res_index = segment_residues.index(key)
                prev_key = segment_residues[res_index - 1]
                prev_res_atoms = residue_lookup.get(prev_key, {})
                if 'C' in prev_res_atoms:
                    c_coord = prev_res_atoms['C']['coords']

            if c_coord is None and 'C' in res_atoms:
                c_coord = res_atoms['C']['coords']

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

    with open(output_file, 'w') as fh:
        fh.writelines(output_lines)

    print(f"Added backbone hydrogen atoms. Output saved to {output_file}")
    return output_file


# --------------------------------------------------------------------------- #
#  Chain splitter
# --------------------------------------------------------------------------- #

def split_chains(pdb):
    """Split PDB file into separate chains and identify their types."""
    aas = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
    rnas = {"ADE", "GUA", "CYT", "URA"}
    dnas = {"DAD", "DGU", "DCY", "DTH"}
    counts = {'P': 0, 'R': 0, 'D': 0}

    def get_type(resname):
        if resname in aas:
            return 'P'
        if resname in rnas:
            return 'R'
        if resname in dnas:
            return 'D'
        return None

    has_chain_id = False
    has_segid = False
    with open(pdb, 'r') as fh:
        for line in fh:
            if line.startswith('ATOM'):
                chain_id = line[21].strip()
                segid = line[72:76].strip()
                has_chain_id |= bool(chain_id)
                has_segid |= bool(segid)
                if has_chain_id and has_segid:
                    break

    if has_chain_id and has_segid:
        chain_changes = segid_changes = 0
        prev_cid = prev_sid = None
        with open(pdb, 'r') as fh:
            for line in fh:
                if not line.startswith('ATOM'):
                    continue
                cid = line[21].strip()
                sid = line[72:76].strip()
                if prev_cid is not None and cid != prev_cid:
                    chain_changes += 1
                if prev_sid is not None and sid != prev_sid:
                    segid_changes += 1
                prev_cid = cid
                prev_sid = sid
        use_chain_id = chain_changes > 0 and (segid_changes == 0 or chain_changes == segid_changes)
    elif has_chain_id:
        use_chain_id = True
    elif has_segid:
        use_chain_id = False
    else:
        raise ValueError("Neither chain ID nor segid found in PDB file")

    chains = []
    types = []
    segids = []
    chain_atoms = []
    current_identifier = None

    with open(pdb, 'r') as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            chain_id = line[21].strip()
            segid = line[72:76].strip()
            identifier = chain_id if use_chain_id else segid
            resname = line[17:20].strip()

            if identifier != current_identifier:
                if chain_atoms:
                    chains.append(chain_atoms)
                current_identifier = identifier
                mol_type = get_type(resname)
                if mol_type is None:
                    raise ValueError(f"Unknown residue type: {resname}")
                types.append(mol_type)
                new_segid = f"{mol_type}{counts[mol_type]:03d}"
                counts[mol_type] += 1
                segids.append(new_segid)
                chain_atoms = [line]
            else:
                chain_atoms.append(line)

    if chain_atoms:
        chains.append(chain_atoms)

    for i, chain in enumerate(chains):
        with open(f"aa2cgtmp_{i}_aa.pdb", 'w') as fh:
            fh.writelines(chain)
            fh.write('END\n')

    return types, segids


# --------------------------------------------------------------------------- #
#  Terminus charge adjustment
# --------------------------------------------------------------------------- #

def set_terminus(gen, segid, terminal):
    """Set the charge status of protein termini."""
    if not segid.startswith("P"):
        return
    resids = gen.get_resids(segid)
    if not resids:
        return
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


# --------------------------------------------------------------------------- #
#  AA→HyRes conversion
# --------------------------------------------------------------------------- #

def at2hyres(pdb_in, pdb_out):
    """
    Convert all-atom protein to HyRes CG PDB.
    """
    residues = {}
    atom_count = 0

    with open(pdb_in, 'r') as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_count += 1
            resid = parse_resseq_field(line[22:26])
            residues.setdefault(resid, [])
            name = line[12:16].strip()
            if name in {'HN', 'HT1', 'H'}:
                name = 'H'
            elif name in {'O', 'OT1'}:
                name = 'O'
            elif name in {'OT2', 'OXT'}:
                continue
            elif name.startswith('H'):
                continue

            residues[resid].append({
                'record': line[:4].strip(),
                'serial': parse_serial_field(line[6:11]),
                'name': name,
                'resname': line[17:20].strip(),
                'chain': line[21],
                'resid': parse_resseq_field(line[22:26]),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
                'occ': float(line[54:60]) if line[54:60].strip() else 1.00,
                'bfac': float(line[60:66]) if line[60:66].strip() else 0.00,
                'segid': line[72:76].strip() if len(line) > 72 else ''
            })

    num_residues = len(residues)
    print(f"Processing {atom_count} atoms / {num_residues} residues")

    for resid in residues:
        if residues[resid]:
            first_atom = residues[resid][0]
            if first_atom['resname'] in {'HSD', 'HSE', 'HSP'}:
                for atom in residues[resid]:
                    atom['resname'] = 'HIS'

    single_bead_sc = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'ASN', 'ASP',
                      'GLN', 'GLU', 'CYS', 'SER', 'THR', 'PRO'}

    sc_mapping = {
        'LYS': [['CB', 'CG', 'CD'], ['CE', 'NZ']],
        'ARG': [['CB', 'CG', 'CD'], ['NE', 'CZ', 'NH1', 'NH2']],
        'HIS': [['CB', 'CG'], ['CD2', 'NE2'], ['ND1', 'CE1']],
        'PHE': [['CB', 'CG', 'CD1'], ['CD2', 'CE2'], ['CE1', 'CZ']],
        'TYR': [['CB', 'CG', 'CD1'], ['CD2', 'CE2'], ['CE1', 'CZ', 'OH']],
        'TRP': [['CB', 'CG'], ['CD1', 'NE1'], ['CD2', 'CE2'], ['CZ2', 'CH2'], ['CE3', 'CZ3']]
    }

    bb_atoms = {'CA', 'C', 'O', 'N', 'H'}
    bb_atoms_first = {'CA', 'N', 'H'}
    bb_atoms_second = {'C', 'O'}

    atom_serial = 0
    with open(pdb_out, 'w') as fh:
        for resid in sorted(residues.keys()):
            res_atoms = residues[resid]
            if not res_atoms:
                continue
            resname = res_atoms[0]['resname']

            if resname in sc_mapping:
                sc_beads = sc_mapping[resname]
            elif resname in single_bead_sc:
                sc_beads = [[atom['name'] for atom in res_atoms if atom['name'] not in bb_atoms]]
            elif resname == 'GLY':
                sc_beads = []
            else:
                raise ValueError(f"Unknown residue type {resname}")

            sc_centers = []
            for bead_atoms in sc_beads:
                coords = [
                    [atom['x'], atom['y'], atom['z']]
                    for atom in res_atoms if atom['name'] in bead_atoms
                ]
                if coords:
                    sc_centers.append(np.mean(coords, axis=0))

            for atom in res_atoms:
                if atom['name'] in bb_atoms_first:
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    fh.write(
                        f"{atom['record']:<6}{serial_str:>5} {atom['name']:>4s}"
                        f" {resname:>3s} {atom['chain']}{encode_resseq(atom['resid'])}"
                        f"   {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                        f"{atom['occ']:6.2f}{atom['bfac']:6.2f}      {atom['segid']:<4s}\n"
                    )

            bead_names = ['CB', 'CC', 'CD', 'CE', 'CF']
            for i, center in enumerate(sc_centers):
                atom_serial += 1
                serial_str = encode_serial(atom_serial)
                first_atom = res_atoms[0]
                fh.write(
                    f"{first_atom['record']:<6}{serial_str:>5} {bead_names[i]:>4s}"
                    f" {resname:>3s} {first_atom['chain']}{encode_resseq(first_atom['resid'])}"
                    f"   {center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                    f"{first_atom['occ']:6.2f}{first_atom['bfac']:6.2f}      {first_atom['segid']:<4s}\n"
                )

            for atom in res_atoms:
                if atom['name'] in bb_atoms_second:
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    fh.write(
                        f"{atom['record']:<6}{serial_str:>5} {atom['name']:>4s}"
                        f" {resname:>3s} {atom['chain']}{encode_resseq(atom['resid'])}"
                        f"   {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                        f"{atom['occ']:6.2f}{atom['bfac']:6.2f}      {atom['segid']:<4s}\n"
                    )

        fh.write("END\n")

    print(f"At2Hyres conversion done, output written to {pdb_out}")


# --------------------------------------------------------------------------- #
#  AA→iCon (RNA) conversion
# --------------------------------------------------------------------------- #

def at2icon(pdb_in, pdb_out):
    """
    Convert all-atom RNA to iConRNA PDB.
    """
    atoms = []
    with open(pdb_in, 'r') as fh:
        for line in fh:
            if line.startswith('ATOM'):
                atoms.append({
                    'name': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resid': parse_resseq_field(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'segid': line[72:76].strip() if len(line) > 72 else ''
                })

    segments = defaultdict(lambda: defaultdict(lambda: {"atoms": [], "resname": "", "chain": ""}))
    for atom in atoms:
        segid = atom['segid']
        resid = atom['resid']
        seg_entry = segments[segid][resid]
        seg_entry['atoms'].append(atom)
        seg_entry['resname'] = atom['resname']
        seg_entry['chain'] = atom['chain']

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
    with open(pdb_out, 'w') as fh:
        for segid in sorted(segments.keys()):
            for resid in sorted(segments[segid].keys()):
                res_data = segments[segid][resid]
                resname = res_data['resname']
                chain = res_data['chain']
                res_atoms = res_data['atoms']

                p_atoms = [a for a in res_atoms if a['name'] in {"P", "O1P", "O2P", "O5'"}]
                if (resid - 1) in segments[segid]:
                    prev_atoms = segments[segid][resid - 1]['atoms']
                    p_atoms.extend([a for a in prev_atoms if a['name'] == "O3'"])

                if p_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in p_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    serial_str = encode_serial(atom_serial)
                    fh.write(
                        f"ATOM  {serial_str:>5}  P   {resname:>3s} {chain}{encode_resseq(resid)}   "
                        f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                        f"{1.00:6.2f}{0.00:6.2f}      {segid:<4s}\n"
                    )

                for bead_name, target_atom in (("C1", "C4'"), ("C2", "C1'")):
                    bead_atoms = [a for a in res_atoms if a['name'] == target_atom]
                    if bead_atoms:
                        coords = np.array([[a['x'], a['y'], a['z']] for a in bead_atoms])
                        center = coords.mean(axis=0)
                        atom_serial += 1
                        serial_str = encode_serial(atom_serial)
                        fh.write(
                            f"ATOM  {serial_str:>5} {bead_name:>4s} {resname:>3s} {chain}{encode_resseq(resid)}   "
                            f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                            f"{1.00:6.2f}{0.00:6.2f}      {segid:<4s}\n"
                        )

                if resname in base_mappings:
                    for bead_name, atom_names in base_mappings[resname]:
                        base_atoms = [a for a in res_atoms if a['name'] in atom_names]
                        if base_atoms:
                            coords = np.array([[a['x'], a['y'], a['z']] for a in base_atoms])
                            center = coords.mean(axis=0)
                            atom_serial += 1
                            serial_str = encode_serial(atom_serial)
                            fh.write(
                                f"ATOM  {serial_str:>5} {bead_name:>4s} {resname:>3s} {chain}{encode_resseq(resid)}   "
                                f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                                f"{1.00:6.2f}{0.00:6.2f}      {segid:<4s}\n"
                            )

        fh.write('END\n')

    print(f'At2iCon conversion done, output written to {pdb_out}')


# --------------------------------------------------------------------------- #
#  Post-process CG PDB: reuse original AA serials/resids
# --------------------------------------------------------------------------- #

def remap_to_original_indices(cg_pdb: str, aa_pdb: str, output_pdb: str | None = None) -> str:
    """
    Replace psfgen's atom serials/residue IDs with those from the original
    all-atom PDB. Each CG residue takes the first serials of its parent residue.
    """
    if output_pdb is None:
        output_pdb = cg_pdb

    aa_residues = []
    current_key = None

    with open(aa_pdb) as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            resname = line[17:20].strip()
            chain = line[21]
            segid = line[72:76] if len(line) >= 76 else "    "
            resseq_field = line[22:26]
            icode = line[26] if len(line) > 26 else " "
            key = (segid, chain, resname, resseq_field, icode)

            if key != current_key:
                resseq_int = parse_resseq_field(resseq_field)
                aa_residues.append({
                    "key": key,
                    "resseq_int": resseq_int,
                    "icode": icode if icode.strip() else " ",
                    "serials": []
                })
                current_key = key

            aa_residues[-1]["serials"].append(parse_serial_field(line[6:11]))

    if not aa_residues:
        raise ValueError(f"No ATOM/HETATM records in reference PDB '{aa_pdb}'")

    remapped_lines = []
    aa_index = 0
    serial_queue: deque[int] = deque()
    last_serial_str = None
    previous_token = None

    with open(cg_pdb) as fh:
        for line in fh:
            record = line[:6].strip()
            if record in {"ATOM", "HETATM"}:
                chain = line[21]
                segid = line[72:76] if len(line) >= 76 else "    "
                token = (segid, chain, line[17:27])
                if token != previous_token:
                    if aa_index >= len(aa_residues):
                        raise ValueError("CG PDB has more residues than AA reference.")
                    current_residue = aa_residues[aa_index]
                    serial_queue = deque(current_residue["serials"])
                    aa_index += 1
                    previous_token = token

                if not serial_queue:
                    raise ValueError(
                        f"Reference residue {current_residue['key']} "
                        "does not provide enough atom serials."
                    )

                serial_val = serial_queue.popleft()
                serial_str = encode_serial(serial_val)
                resseq_str = encode_resseq(current_residue["resseq_int"])
                icode_char = current_residue["icode"]

                line_chars = list(line.rstrip('\n'))
                if len(line_chars) < 80:
                    line_chars.extend([' '] * (80 - len(line_chars)))

                line_chars[6:11] = list(serial_str)
                line_chars[22:26] = list(resseq_str)
                line_chars[26] = icode_char

                new_line = ''.join(line_chars).rstrip() + '\n'
                remapped_lines.append(new_line)
                last_serial_str = serial_str

            elif record == "ANISOU":
                if last_serial_str is None:
                    raise ValueError("ANISOU record encountered before any ATOM record.")
                line_chars = list(line.rstrip('\n'))
                if len(line_chars) < 80:
                    line_chars.extend([' '] * (80 - len(line_chars)))
                line_chars[6:11] = list(last_serial_str)
                remapped_lines.append(''.join(line_chars).rstrip() + '\n')

            else:
                remapped_lines.append(line)

    if aa_index != len(aa_residues):
        print(
            f"[remap warning] Only {aa_index} of {len(aa_residues)} AA residues matched."
        )

    Path(output_pdb).write_text(''.join(remapped_lines))
    return output_pdb


# --------------------------------------------------------------------------- #
#  Driver: AA → CG (HyRes / iConRNA)
# --------------------------------------------------------------------------- #

def at2cg(pdb_in, pdb_out, terminal='neutral', cleanup=True):
    """
    Convert all-atom PDB to CG PDB (HyRes for protein or iConRNA for RNA).
    """
    RNA_topology, _ = load_ff('RNA')
    protein_topology, _ = load_ff('Protein')

    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)

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

    gen.write_pdb(pdb_out)
    print(f"Conversion done, output written to {pdb_out}")

    for segid in gen.get_segids():
        set_terminus(gen, segid, terminal)

    psf_file = f'{pdb_out[:-4]}.psf'
    gen.write_psf(filename=psf_file)
    print(f"PSF file written to {psf_file}")

    # Post-process the PDB to reuse original atom/residue numbering
    remap_to_original_indices(pdb_out, pdb_in)

    if cleanup:
        for file in os.listdir():
            if file.startswith("aa2cgtmp_") and file.endswith(".pdb"):
                os.remove(file)

    return pdb_out, psf_file


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert2CG: All-atom to HyRes/iConRNA converting'
    )
    parser.add_argument('aa', help='Input PDB file')
    parser.add_argument('cg', help='Output PDB file')
    parser.add_argument('--hydrogen', action='store_true',
                        help='Add backbone amide hydrogen (H-N only), default False')
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