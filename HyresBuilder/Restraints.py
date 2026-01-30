"""
This script is to help set up restraints in Hyres model
Author: Jian Huang (jianhuang@umass.edu)
Date created: Nov 12, 2024
Modified: Jan 30, 2026

Functions:
----------
Helper Functions:
    - get_atom_indices_coordinates: Get atom indices and coordinates for a selection
    - get_COM: Calculate center of mass for a selection
    - identify_folded_CA_idx: Identify CA atoms in folded regions using DSSP
    - get_next_force_group: Get next available force group number
    - get_allforceinfo: Print all forces in the system

Positional Restraints:
    - positional_restraint: Restrain specific atoms to reference positions
    - domain_positional_restraint: Restrain CA atoms across multiple domains
    - bb_positional_restraint: Restrain backbone atoms to reference positions
    - CA_positional_restraint: Restrain CA atoms in a single domain
    - COM_positional_restraint: Restrain center of mass to reference position
    - COM_relative_restraint: Restrain distance between two groups' COMs

Domain Restraints:
    - domain_3D_restraint: Add internal harmonic restraints within domains
"""

# OpenMM imports
from openmm import (
    CustomExternalForce,
    CustomCentroidBondForce,
    HarmonicBondForce,
    CustomBondForce
)
from openmm.app import PDBFile
import openmm.unit as unit

import numpy as np
import mdtraj as md
from itertools import combinations

########################################################################################################
# helper functions for PDB, SIMULATION AND FORCES
########################################################################################################
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

def get_next_force_group(system):
    """
    Get the next available force group number from the system
    """
    existing_groups = set()
    for force in system.getForces():
        existing_groups.add(force.getForceGroup())
    
    # Find the next available group (starting from 0)
    next_group = 0
    while next_group in existing_groups:
        next_group += 1
    
    return next_group

def get_allforceinfo(system):
    print('\n# Now, the system has:')
    for force in system.getForces():
        print(f'      ForceName: {force.getName():<20}    ForceGroup: {force.getForceGroup():<20}')



########################################################################################################
# POSITIONAL RESTRAINTS
########################################################################################################
# Positional restraints based on user-defined atom index, position and K constant
# Scenarios: for only a few positional restraints
def positional_restraint(system, indx_pos_Kcons_list):
    """
    indx_pos_list: list of tuples. Each tuple has two elements, the first one being the atom index,
        the second one being the 3D coordinate (x, y, z) as the reference position
        
        example: [(1, (10.0, 12.0, 13.0), 400), (10, (20.0, 23.5, 5.6), 400), ...]
            restraint the atom with index 1 to be at (10.0, 12.0, 13.0) with Kcons=400 kj/mol/nm**2; and 
            restraint the atom with index 10 to be at (20.0, 23.5, 5.6) with Kcons=400 kj/mol/nm**2
    
    return system (openmm object)
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

# Positional restraints for the whole domain
# Scenario: uniform K constrant for all positions within a domain
def domain_positional_restraint(system, pdb_ref, domain_ranges, Kcons=400, 
                                force_name_prefix="PositionalRestraint",
                                restrain_folded_only=False):
    """
    Apply positional restraints to CA atoms in specified domains
    
    Parameters:
    -----------
    system: OpenMM system object
    pdb_ref: str 
        Path to reference PDB file 
    domain_ranges: list of tuples
        List of domain definitions [('A', (1,50)), ('B', (75, 200)), ...]
    Kcons: float or dict
        K constant for harmonic restraint (kj/mol/nm**2)
        Can be a single value or dict mapping domain to K value:
        e.g., {('A', (1,50)): 400, ('B', (75, 200)): 600}
    force_name_prefix: str
        Prefix for force name
    restrain_folded_only: bool
        If True, only restrain folded CA atoms (uses DSSP)
        If False, restrain all CA atoms in domain range
    
    Returns:
    --------
    system: OpenMM system
    restraint_info: dict
        Dictionary with restraint information for each domain
    """
    # Load pdb
    pdb_md = md.load_pdb(pdb_ref)
    # Also load for OpenMM positions
    from openmm.app import PDBFile
    pdb_omm = PDBFile(pdb_ref)
    positions = pdb_omm.positions
    
    # Handle K constant
    if isinstance(Kcons, dict):
        K_dict = Kcons
    else:
        K_dict = {domain: Kcons for domain in domain_ranges}
    
    # Get chainid dict
    chainid_dict = {chain.chain_id: chain.index for chain in pdb_md.topology.chains}
    
    # Process each domain
    for domain_idx, domain in enumerate(domain_ranges):
        chain_id, (start_res, end_res) = domain
        
        # Get K constant for this domain
        K_val = K_dict.get(domain, 400)
        K_omm = K_val * unit.kilojoule_per_mole / unit.nanometers**2
        
        # Create force for this domain
        pos_force = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')

        # Set force name
        domain_name = f"{force_name_prefix}_Chain{chain_id}_Res{start_res}to{end_res}"
        pos_force.setName(domain_name)
        
        # Set force group
        force_group = get_next_force_group(system)
        pos_force.setForceGroup(force_group)
        
        # Add parameters
        pos_force.addPerParticleParameter("x0")
        pos_force.addPerParticleParameter("y0")
        pos_force.addPerParticleParameter("z0")
        pos_force.addGlobalParameter("k", K_omm)
        
        # Get CA atom indices
        if restrain_folded_only:
            # Use DSSP to identify folded regions
            CA_indices = identify_folded_CA_idx(pdb=pdb_md, domain=domain)
        else:
            # Get all CA atoms in domain range
            assert chain_id in chainid_dict.keys(), f"Chain ID '{chain_id}' not found!"
            selection = f"chainid {chainid_dict[chain_id]} and residue {start_res} to {end_res} and name CA"
            CA_indices = pdb_md.topology.select(selection)
        
        if len(CA_indices) == 0:
            print(f"Warning: No CA atoms found for domain {domain}, skipping")
            continue
        
        # Add particles with their reference positions
        for index in CA_indices:
            pos = positions[index].value_in_unit(unit.nanometers)
            pos_force.addParticle(int(index), list(pos))
        
        # Add force to system
        system.addForce(pos_force)
        
        
        print(f"Domain {domain}: {len(CA_indices)} CA atoms restrained with K={K_val} kJ/mol/nm^2 "
              f"(Force group: {force_group}, Name: {domain_name})")
    
    return system

# Backbone positional restraint
# (Easy to customize this function to select only partial backbones of your system)
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

# C-alpha atom positional restraint for the folded region
def CA_positional_restraint(system, pdb_file, domain, Kcons=400.0, 
                           use_folded_only=True, force_name_prefix="CA_PosRestraint"):
    """
    Apply positional restraints to CA atoms in a specified domain
    
    This function adds CustomExternalForce to restrain CA atoms to their reference
    positions using a harmonic potential: E = k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)
    
    Parameters:
    -----------
    system : openmm.System
        OpenMM System object to which the positional restraint force will be added
    pdb_file : str
        Path to reference PDB file containing the reference positions for restraints
    domain : tuple
        Domain definition specifying which residues to restrain.
        Format: ('chain_id', (start_resid, end_resid))
        Example: ('A', (1, 50)) restrains residues 1-50 of chain A
    Kcons : float, optional
        Force constant for harmonic restraint in kJ/mol/nm^2.
        Default is 400.0 kJ/mol/nm^2.
        Higher values create stronger restraints.
    use_folded_only : bool, optional
        If True, only restrain CA atoms in folded regions (helices/sheets)
        identified by DSSP secondary structure analysis.
        If False, restrain all CA atoms in the specified domain range.
        Default is True.
    force_name_prefix : str, optional
        Prefix for the force name. The full name will be:
        "{prefix}_Chain{chain}_Res{start}to{end}"
        Default is "CA_PosRestraint"
    
    Returns:
    --------
    system : openmm.System
        The modified OpenMM System object with positional restraint force added
    """

    pdb_tmp = PDBFile(pdb_file)
    CA_pos_restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    CA_pos_restraint.addGlobalParameter('k', Kcons*unit.kilojoule_per_mole/unit.nanometers/unit.nanometers)
    CA_pos_restraint.addPerParticleParameter('x0')
    CA_pos_restraint.addPerParticleParameter('y0')
    CA_pos_restraint.addPerParticleParameter('z0')

    chain_id, (start_res, end_res) = domain
    force_group = get_next_force_group(system)
    force_name = f"{force_name_prefix}_Chain{chain_id}_Res{start_res}to{end_res}"
    CA_pos_restraint.setName(force_name)
    CA_pos_restraint.setForceGroup(force_group)

    md_pdb = md.load_pdb(pdb_file)

    # Get CA atom indices based on use_folded_only flag
    if use_folded_only:
        # Only restrain CA atoms in folded regions (helices/sheets)
        CA_list = identify_folded_CA_idx(md_pdb, domain)
        restrain_type = "folded"
    else:
        # Restrain all CA atoms in the domain range
        chainid_dict = {chain.chain_id: chain.index for chain in md_pdb.topology.chains}
        
        # Validate chain ID
        if chain_id not in chainid_dict:
            raise ValueError(f"Chain ID '{chain_id}' not found in PDB file. "
                           f"Available chains: {list(chainid_dict.keys())}")
        
        # Select all CA atoms in domain
        selection = f"chainid {chainid_dict[chain_id]} and residue {start_res} to {end_res} and name CA"
        CA_list = md_pdb.topology.select(selection)
        restrain_type = "all"

    # Check if we have CA atoms to restrain
    if len(CA_list) == 0:
        print(f"Warning: No CA atoms found for domain {domain}, skipping")
        return system

    for ca in CA_list:
        CA_pos_restraint.addParticle(ca, pdb_tmp.positions[ca])

    system.addForce(CA_pos_restraint)
    
    return system

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

    force_group = get_next_force_group(system)
    com_pos_restraint.setName("COM_positional_restraint")
    com_pos_restraint.setForceGroup(force_group)

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

    restrain distance between atom group 1 and atom group 2 to be 1 nm using Kcons=400 kj/mol/nm**2

    return system     
    """
    COM_force = CustomCentroidBondForce(2, "0.5*kp*( (distance(g1, g2)-d0 )^2)")

    force_group = get_next_force_group(system)
    COM_force.setName("COM_relative_positional_restraint")
    COM_force.setForceGroup(force_group)

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


# Domain restraints
def domain_3D_restraint(system, pdb_ref, domain_ranges, Kcons=400, cutoff=1.2,\
                         min_seq_sep=2, force_name_prefix="DomainRestraint"):
    """
    system: omm system
    domain_ranges: a list of tuples, and each tuple defines one domain range:
        ('chain_id'), (starting_resid, ending_resid))
        example: [('A', (1,50)), ('B', (75, 200)), ...]
            # two domains: resid 1-50 of chain A and resid 75-200 of chain B
    Kcons: (float) K constant for harmonic restraint. default unit: kj/mol/nm**2
    cutoff: cutoff distance, unit: nanometer
    force_name_prefix: prefix for the force name (will be appended with domain info)
    return omm system and a dictionary mapping domain to force group
    """
    Kcons_internal = Kcons * unit.kilojoule_per_mole/unit.nanometers**2

    # load pdb to mdtraj
    pdb_md = md.load_pdb(pdb_ref)
    
    # find all pairs
    pairs = []
    for domain in domain_ranges:
        internal_force = HarmonicBondForce()
        
        domain_name = f"{force_name_prefix}_Chain{domain[0]}_Res{domain[1][0]}to{domain[1][1]}"
        force_group = get_next_force_group(system)
        internal_force.setForceGroup(force_group)
        internal_force.setName(domain_name)
        
        # get C-alpha atom indices of folder region
        folded_CA_idx = identify_folded_CA_idx(pdb=pdb_md, domain=domain)
        if len(folded_CA_idx) < 2:
            print(f"Domain {domain}: fewer than 2 folded CAs, skipping")
            continue
        
        pairs = list(combinations(folded_CA_idx, 2))

        pairs_num = 0
        for index in pairs:
            i_atom = int(index[0])
            j_atom = int(index[1])

            seq_sep = abs(i_atom - j_atom)
            if seq_sep < min_seq_sep:
                continue
            
            r1=np.squeeze(pdb_md.xyz[:,int(index[0]),:])
            r2=np.squeeze(pdb_md.xyz[:,int(index[1]),:])
            
            dist0 = np.linalg.norm(r1-r2)
            if dist0 < cutoff:
                pairs_num += 1
                internal_force.addBond(int(index[0]),int(index[1]), dist0*unit.nanometers, Kcons_internal)
        print(f"Number of internal pairs of domain {str(domain)}: {pairs_num}")

        # Only add the force if it has at least one bond
        if pairs_num > 0:
            system.addForce(internal_force)
            print(f"Domain {str(domain)}: {pairs_num} internal pairs added to force group {force_group} ({domain_name})")
        else:
            print(f"Domain {str(domain)}: No pairs added, force not created")
    
    return system

    
     
