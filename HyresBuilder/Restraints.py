"""
This script is to help set up restraints in Hyres model
Athour: Jian Huang
Date: Nov 12, 2024
Modified: Nov 12, 2024
"""

from openmm.app import *
from openmm import *
import numpy as np
import mdtraj as md
from itertools import combinations

# helper function to get atom indices for a give pdb
def get_atom_indices_coordinates(pdb, selection):
    """
    pdb: pdb file
    selection: mdtraj selection syntax (ref: https://mdtraj.org/1.9.4/atom_selection.html)
        of note, if your 'selection' has chainid, you need to use the chainid from PDB file.
            chainid in mdtraj is forced to be numbered as 0, 1, 2, ...
            here, instead of using those sequential integers for chainid, we override it with the 
            original chain id from PDB file.
    """
    pdb_md = md.load_pdb(pdb)

    if 'chainid' in selection:
        chainid_dict = {chain.chain_id: chain.index for chain in pdb_md.topology.chains}

        new_str = []
        for substring in selection.split('and'):
            if 'chainid' in substring:
                chainid_list = substring.strip().split('chainid')[-1].strip().split()
                new_chainid_str = ' '.join([str(chainid_dict[i]) for i in chainid_list])
                final_str = 'chainid ' + new_chainid_str
                new_str.append(final_str)
            else:
                new_str.append(substring.strip())
        new_selection = ' and '.join(new_str)
    else:
        new_selection = selection
    
    indices = pdb_md.topology.select(new_selection)
    coordinates = pdb_md.xyz[0, indices, :]
    return indices, coordinates 

# Get the center of mass of a selection
def get_COM(pdb, selection):
    pdb_md = md.load_pdb(pdb)

    if 'chainid' in selection:
        chainid_dict = {chain.chain_id: chain.index for chain in pdb_md.topology.chains}
        # print(chainid_dict)

        new_str = []
        for substring in selection.split('and'):
            if 'chainid' in substring:
                chainid_list = substring.strip().split('chainid')[-1].strip().split()
                new_chainid_str = ' '.join([str(chainid_dict[i]) for i in chainid_list])
                final_str = 'chainid ' + new_chainid_str
                new_str.append(final_str)
            else:
                new_str.append(substring.strip())
        new_selection = ' and '.join(new_str)
    else:
        new_selection = selection
    # print(new_selection)
    com = md.compute_center_of_mass(pdb_md, new_selection)[0]
    return com

# Positional restraints
def positional_restraint(system, indx_pos_Kcons_list):
    """
    indx_pos_list: list of tuples. Each tuple has two elements, the first one being the atom index,
        the second one being the 3D coordinate (x, y, z) as the reference position
        
        example: [(1, (10.0, 12.0, 13.0), 400), (10, (20.0, 23.5, 5.6), 400), ...]
            restraint the atom with index 1 to be at (10.0, 12.0, 13.0) with Kcons=400 kj/mol/nm**2; and 
            restraint the atom with index 10 to be at (20.0, 23.5, 5.6) with Kcons=400 kj/mol/nm**2
    """

    # omm positional restraints
    pos_restraint = CustomExternalForce('kp*periodicdistance(x, y, z, x0, y0, z0)^2')
    pos_restraint.addPerParticleParameter('kp')
    pos_restraint.addPerParticleParameter('x0')
    pos_restraint.addPerParticleParameter('y0')
    pos_restraint.addPerParticleParameter('z0')
    for idx, pos, Kcons in indx_pos_Kcons_list:
        # add unit to Kcons
        Kcons_wU = Kcons * unit.kilojoule_per_mole/unit.nanometers**2
        pos_restraint.addParticle(idx, [Kcons_wU, *pos])
    system.addForce(pos_restraint)

    return system

def bb_positional_restraint(system, pdb_ref, Kcons=400):
    """
    add backbone restraints, given a reference PDB
    arguments:
    system: omm system object
    pdb_ref: pdb file path
    Kcons: (float) K constant. kj/mol/nm**2

    Return: system
    """
    # Load pdb
    pdb_init = PDBFile(pdb_ref)

    # add unit to Kcons
    Kcons_wU = Kcons * unit.kilojoule_per_mole/unit.nanometers**2
    
    # omm positional restraints
    pos_restraint = CustomExternalForce('kp*periodicdistance(x, y, z, x0, y0, z0)^2')
    pos_restraint.addGlobalParameter('kp', Kcons_wU)
    pos_restraint.addPerParticleParameter('x0')
    pos_restraint.addPerParticleParameter('y0')
    pos_restraint.addPerParticleParameter('z0')
    for res in pdb_init.topology.residues():
        for atom in res.atoms():
            if atom.name in [ 'CA', 'N', 'C', 'O']:
                pos_restraint.addParticle(atom.index, pdb_init.positions[atom.index])
    system.addForce(pos_restraint)
    return system

def CA_positional_restraint(system, pdb_file, domain):
    pdb_tmp = PDBFile(pdb_file)
    CA_pos_restraint = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    CA_pos_restraint.addGlobalParameter('k', 400.0*unit.kilojoule_per_mole/unit.nanometers/unit.nanometers)
    CA_pos_restraint.addPerParticleParameter('x0')
    CA_pos_restraint.addPerParticleParameter('y0')
    CA_pos_restraint.addPerParticleParameter('z0')

    CA_list = identify_folded_CA_idx(md.load_pdb(pdb_file), domain)
    for ca in CA_list:
        CA_pos_restraint.addParticle(ca, pdb_tmp.positions[ca])
    system.addForce(CA_pos_restraint)

# Center of mass positional restraints
def COM_positional_restraint(system, group_indices_pos_kcons):
    """
    system: omm system
    group_indices_pos: list of tuples
        example: [((1,2,3,4), (x0, y0, z0), 400), ...]
            1,2,3,4 are the atom indices of the group; 
            x0, y0, z0 are the x y z coordinates of reference COM position
            400 is the K constant (unit: kj/mol/nm**2) 
    return: omm system    
    """
    # omm positional restraints
    com_pos_restraint = CustomCentroidBondForce(1, 'kp*periodicdistance(x, y, z, x0, y0, z0)^2')
    com_pos_restraint.addPerBondParameter('kp')
    com_pos_restraint.addPerBondParameter('x0')
    com_pos_restraint.addPerBondParameter('y0')
    com_pos_restraint.addPerBondParameter('z0')

    for idx, (group_idx, ref_pos, Kcons) in enumerate(group_indices_pos_kcons):
        Kcons_wU = Kcons * unit.kilojoule_per_mole/unit.nanometers**2
        com_pos_restraint.addGroup(group_idx)
        com_pos_restraint.addBond([idx, ], [Kcons_wU, *ref_pos])
    system.addForce(com_pos_restraint)
    return system

# Center of mass distance restraints
def COM_relative_restraint(system, groups_dist_Kcons_list):
    """ Apply distance restraint between two atom groups
    system: omm system
    groups_dist_Kcons_list: (list of tuples)
        example: [((1,2,3), (4,5,6), 10.0, 400), ...]
            (1,2,3): atom indices of atom group1
            (4,5,6): atom indices of atom group2
            1.0: reference distance, unit of nanometer
            400: K constant, kj/mol/nm**2

            restrain distance between atom group 1 and atom group 2 to be 10 Angstrome using Kcons=400 kj/mol/nm**2

    return omm system     
    """
    COM_force = CustomCentroidBondForce(2, "0.5*kp*( (distance(g1, g2)-d0 )^2)")
    COM_force.addPerBondParameter('kp')  # Force constant
    COM_force.addPerBondParameter('d0')  # restrain distance

    i = 0
    for group1,group2,dist,Kcons in groups_dist_Kcons_list:
        Kcons_wU = Kcons * unit.kilojoule_per_mole/unit.nanometers**2
        target_dist = dist * unit.nanometers
        COM_force.addGroup(group1)  # Group 1
        COM_force.addGroup(group2)  # Group 2
        COM_force.addBond([i,  i+1], [Kcons_wU, target_dist])
        i += 2
    system.addForce(COM_force)
    return system

# Identify regions with secondary structure
def identify_folded_CA_idx(pdb, domain):
    """
    pdb: mdtraj object
    domain: 
        example: ('A', (1, 50))
    return: CA atom indices for the folded region
    """
    # get chainid dict
    chainid_dict = {chain.chain_id: chain.index for chain in pdb.topology.chains}

    # get starting residue index and ending residue index from domain definition
    chainid, (starting_resid, ending_resid) = domain

    # make sure the given chain id is included in the PDB file
    assert chainid in chainid_dict.keys(), f"chain ID '{chainid}' given in the domain definition does not exist!"

    # select domain
    selection = "chainid %s and residue %s to %s" % (chainid_dict[chainid], starting_resid, ending_resid)
    selected_domain = pdb.atom_slice(pdb.topology.select(selection))
    dssp = md.compute_dssp(selected_domain, simplified=True)
    folded = np.where(dssp!='C', 1, 0)[0]
    resid = [selected_domain.topology.residue(idx).resSeq for idx, i in enumerate(folded) if i==1 ]
    folded_CA_idx = [pdb.topology.select("chainid %s and residue %s and name CA " % \
                                         (chainid_dict[chainid], str(i) ) )[0] for i in resid]

    return folded_CA_idx

# Domain restraints
def domain_3D_restraint(system, pdb_ref, domain_ranges, Kcons=400, cutoff=1.2):
    """
    system: omm system
    domain_ranges: a list of tuples, and each tuple defines one domain range:
        ('chain_id'), (starting_resid, ending_resid))
        example: [('A', (1,50)), ('B', (75, 200)), ...] 
            # two domains: resid 1-50 of chain A and resid 75-200 of chain B
    Kcons_internal: (float) K constant for harmonic restraint. default unit: kj/mol/nm**2
    cutoff: cutoff distance, unit: nanometer
    return omm system
    """
    internal_force = HarmonicBondForce()
    Kcons_internal = Kcons * unit.kilojoule_per_mole/unit.nanometers**2

    # load pdb to mdtraj
    pdb_md = md.load_pdb(pdb_ref)

    # find all pairs
    pairs = []
    for domain in domain_ranges:
        # get C-alpha atom indices of folder region
        folded_CA_idx = identify_folded_CA_idx(pdb=pdb_md, domain=domain)
        pairs = list(combinations(folded_CA_idx, 2))
    
        pairs_num = 0
        for index in pairs:
            r1=np.squeeze(pdb_md.xyz[:,int(index[0]),:])
            r2=np.squeeze(pdb_md.xyz[:,int(index[1]),:])
            # dist0=np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)
            dist0 = np.linalg.norm(r1-r2)
            if dist0 < 1.2:
                pairs_num += 1
                internal_force.addBond(int(index[0]),int(index[1]), dist0*unit.nanometers, Kcons_internal)
        print(f"Number of internal pairs of domain {str(domain)}: {pairs_num}")
    system.addForce(internal_force)
    return system


