from psfgen import PsfGen
import numpy as np
import os
import warnings
from .utils import load_ff


def split_chains(pdb):
    """Split PDB file into separate chains and identify their types."""
    aas = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    rnas = ["ADE", "GUA", "CYT", "URA"]
    dnas = ["DAD", "DGU", "DCY", "DTH"]
    counts = {'P': 0, 'R': 0, 'D': 0}

    def get_type(resname):
        if resname in aas:
            return 'P'
        elif resname in rnas:
            return 'R'
        elif resname in dnas:
            return 'D'
        return None

    current_chain = None
    chain_atoms = []
    chains = []
    types = []
    segids = []
    
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain_id = line[21]
                resname = line[17:20].strip()
                
                if chain_id != current_chain:
                    if chain_atoms:
                        chains.append(chain_atoms)
                    current_chain = chain_id
                    mol_type = get_type(resname)
                    if mol_type is None:
                        raise ValueError(f'Unknown residue type: {resname}')
                    types.append(mol_type)
                    segid = f"{mol_type}{counts[mol_type]:03d}"
                    counts[mol_type] += 1
                    segids.append(segid)
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
                    f.write(f"{atom['record']:4s}  {atom_serial:5d} {atom['name']:2s}   "
                           f"{resname:3s} {atom['chain']}{int(atom['resid']):4d}    "
                           f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                           f"{atom['occ']:6.2f}{atom['bfac']:6.2f}      {atom['segid']:4s}\n")
            
            # Write sidechain beads
            bead_names = ['CB', 'CC', 'CD', 'CE', 'CF']
            for i, center in enumerate(sc_centers):
                atom_serial += 1
                f.write(f"{first_atom['record']:4s}  {atom_serial:5d} {bead_names[i]:2s}   "
                       f"{resname:3s} {first_atom['chain']}{int(first_atom['resid']):4d}    "
                       f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                       f"{first_atom['occ']:6.2f}{first_atom['bfac']:6.2f}      "
                       f"{first_atom['segid']:4s}\n")
            
            # Write backbone C and O atoms (second group)
            for atom in res.values():
                if atom['name'] in bb_atoms_2:
                    atom_serial += 1
                    f.write(f"{atom['record']:4s}  {atom_serial:5d} {atom['name']:2s}   "
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
                    f.write(f"ATOM  {atom_serial:5d}  P   {resname:3s} {chain}{resid:4d}    "
                           f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                           f"  1.00  0.00      {segid:4s}\n")
                
                # C1 bead (C4' sugar)
                c1_atoms = [a for a in res_atoms if a['name'] == "C4'"]
                if c1_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in c1_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    f.write(f"ATOM  {atom_serial:5d}  C1  {resname:3s} {chain}{resid:4d}    "
                           f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                           f"  1.00  0.00      {segid:4s}\n")
                
                # C2 bead (C1' sugar)
                c2_atoms = [a for a in res_atoms if a['name'] == "C1'"]
                if c2_atoms:
                    coords = np.array([[a['x'], a['y'], a['z']] for a in c2_atoms])
                    center = coords.mean(axis=0)
                    atom_serial += 1
                    f.write(f"ATOM  {atom_serial:5d}  C2  {resname:3s} {chain}{resid:4d}    "
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
                            f.write(f"ATOM  {atom_serial:5d}  {bead_name:2s}  {resname:3s} "
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
    parser.add_argument('--terminal', '-t', type=str, default='neutral', 
                       help='Charge status of terminus: neutral, charged, NT, CT')
    
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=UserWarning)
    at2cg(args.aa, args.cg, terminal=args.terminal)


if __name__ == '__main__':
    main()