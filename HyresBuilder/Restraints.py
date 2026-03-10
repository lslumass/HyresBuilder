"""
| This module is used to help set up restraints in Hyres model
| Athour: Jian Huang
| Date: Nov 12, 2024
| Modified: Nov 12, 2024
"""

from openmm.app import *
from openmm import *
import numpy as np
import mdtraj as md
from itertools import combinations

# helper function to get atom indices for a give pdb
def get_atom_indices_coordinates(pdb, selection):
    """
    Get atom indices and coordinates for a selection from a PDB file.

    Wraps mdtraj's atom selection with a fix for chain IDs: mdtraj internally
    renumbers chains as 0, 1, 2, ..., but this function maps the original
    alphabetic chain IDs from the PDB file so that selections like
    'chainid A' work as expected.

    Args:
        pdb       (str): Path to the PDB file.
        selection (str): MDTraj atom selection string.
                         See https://mdtraj.org/1.9.4/atom_selection.html.
                         Chain IDs should match those in the PDB file (e.g. 'A', 'B'),
                         not mdtraj's internal sequential integers.

    Returns:
        tuple:
            - indices     (np.ndarray, shape (N,)):    Atom indices of the selection.
            - coordinates (np.ndarray, shape (N, 3)):  Atomic coordinates in nanometers
                                                       at frame 0.

    Example:
        >>> indices, coords = get_atom_indices_coordinates('system.pdb', 'name CA and chainid A')
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
    """
    Compute the center of mass (COM) of a selected group of atoms.

    Applies the same chain-ID remapping as get_atom_indices_coordinates, so
    alphabetic chain IDs from the PDB file can be used directly in the selection.

    Args:
        pdb       (str): Path to the PDB file.
        selection (str): MDTraj atom selection string.
                         Chain IDs should match those in the PDB file (e.g. 'A', 'B').

    Returns:
        np.ndarray, shape (3,): Center-of-mass coordinates in nanometers at frame 0.

    Example:
        >>> com = get_COM('system.pdb', 'resid 1 to 50 and chainid A')
    """
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
    Apply per-atom positional (harmonic) restraints to an OpenMM system.

    Each atom is restrained to an arbitrary reference position using a
    periodic-distance harmonic potential: U = kp * periodicdistance(x, y, z, x0, y0, z0)^2

    Using periodicdistance ensures the restraint works correctly under
    periodic boundary conditions.

    Args:
        system               (openmm.System): OpenMM System object to modify.
        indx_pos_Kcons_list  (list of tuples): Each tuple contains:
            - idx    (int):            Atom index (0-based).
            - pos    (array-like):     Reference position (x0, y0, z0) in nanometers.
            - Kcons  (float):          Force constant in kJ/mol/nm².

    Returns:
        openmm.System: The modified system with positional restraints added.

    Example:
        >>> restraints = [
        ...     (0,  (1.0, 1.2, 1.3), 400),   # restrain atom 0  to (1.0, 1.2, 1.3) nm
        ...     (10, (2.0, 2.3, 0.5), 400),   # restrain atom 10 to (2.0, 2.3, 0.5) nm
        ... ]
        >>> system = positional_restraint(system, restraints)
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
    Apply positional restraints to all protein backbone atoms (CA, N, C, O).

    Reference positions are taken directly from the provided PDB file.
    A single global force constant is shared across all restrained atoms.
    The harmonic potential uses periodicdistance for PBC compatibility: U = kp * periodicdistance(x, y, z, x0, y0, z0)^2

    Args:
        system  (openmm.System): OpenMM System object to modify.
        pdb_ref (str):           Path to the reference PDB file. Backbone atom
                                 positions are read from this file.
        Kcons   (float):         Force constant in kJ/mol/nm². Default: 400.

    Returns:
        openmm.System: The modified system with backbone restraints added.

    Example:
        >>> system = bb_positional_restraint(system, 'reference.pdb', Kcons=200)
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
    """
    Apply positional restraints to CA atoms within a folded domain.

    Only CA atoms identified as part of secondary structure elements (helices
    or strands) by identify_folded_CA_idx are restrained. Uses a standard
    harmonic potential (not periodic-distance): U = k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)

    Force constant is hardcoded to 400 kJ/mol/nm².

    Args:
        system   (openmm.System): OpenMM System object to modify.
        pdb_file (str):           Path to the PDB file. Used both to define the
                                  topology and as the source of reference positions.
        domain   (tuple):         Domain definition as (chain_id, (start_resid, end_resid)).
                                  Example: ('A', (1, 50))

    Returns:
        openmm.System: The modified system with CA restraints added.

    Example:
        >>> system = CA_positional_restraint(system, 'protein.pdb', ('A', (1, 50)))
    """
    pdb_tmp = PDBFile(pdb_file)
    CA_pos_restraint = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    CA_pos_restraint.addGlobalParameter('k', 400.0*unit.kilojoule_per_mole/unit.nanometers/unit.nanometers)
    CA_pos_restraint.addPerParticleParameter('x0')
    CA_pos_restraint.addPerParticleParameter('y0')
    CA_pos_restraint.addPerParticleParameter('z0')

    md_pdb = md.load_pdb(pdb_file)
    CA_list = identify_folded_CA_idx(md_pdb, domain)
    for ca in CA_list:
        CA_pos_restraint.addParticle(ca, pdb_tmp.positions[ca])
    system.addForce(CA_pos_restraint)
    return system

# Center of mass positional restraints
def COM_positional_restraint(system, group_indices_pos_kcons):
    """
    Apply positional restraints on the center of mass (COM) of atom groups.

    Each group's COM is harmonically restrained to a reference position using
    OpenMM's CustomCentroidBondForce with periodic-distance: U = kp * periodicdistance(x, y, z, x0, y0, z0)^2

    Args:
        system (openmm.System): OpenMM System object to modify.
        group_indices_pos_kcons (list of tuples): Each tuple contains:
            - group_idx  (array-like): Atom indices forming the group.
            - ref_pos    (array-like): Reference COM position (x0, y0, z0) in nanometers.
            - Kcons      (float):      Force constant in kJ/mol/nm².

    Returns:
        openmm.System: The modified system with COM positional restraints added.

    Example:
        >>> restraints = [
        ...     ([0, 1, 2, 3], (1.0, 1.5, 2.0), 400),  # restrain COM of atoms 0-3
        ...     ([4, 5, 6],    (3.0, 2.5, 1.0), 200),  # restrain COM of atoms 4-6
        ... ]
        >>> system = COM_positional_restraint(system, restraints)
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
    """
    Apply a harmonic distance restraint between the centers of mass of two atom groups.

    Uses OpenMM's CustomCentroidBondForce with potential:
        U = 0.5 * kp * (distance(g1, g2) - d0)^2

    Args:
        system                (openmm.System): OpenMM System object to modify.
        groups_dist_Kcons_list (list of tuples): Each tuple contains:
            - group1  (array-like): Atom indices of the first group.
            - group2  (array-like): Atom indices of the second group.
            - dist    (float):      Target COM–COM distance in nanometers.
            - Kcons   (float):      Force constant in kJ/mol/nm².

    Returns:
        openmm.System: The modified system with COM distance restraints added.

    Example:
        >>> restraints = [
        ...     ([0, 1, 2], [3, 4, 5], 1.0, 400),  # restrain group COM distance to 1.0 nm
        ...     ([6, 7],    [8, 9],    2.5, 200),   # restrain group COM distance to 2.5 nm
        ... ]
        >>> system = COM_relative_restraint(system, restraints)
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
    Identify CA atom indices belonging to secondary-structure elements in a domain.

    Uses MDTraj's DSSP implementation (simplified scheme) to classify each residue
    within the specified domain. Residues assigned to helix ('H') or strand ('E')
    are considered folded; coil ('C') residues are excluded.

    Args:
        pdb    (md.Trajectory): MDTraj Trajectory object (typically loaded via md.load_pdb).
        domain (tuple):         Domain definition as (chain_id, (start_resid, end_resid)).
                                - chain_id      (str): Alphabetic chain ID matching the PDB file.
                                - start_resid   (int): First residue number (inclusive, PDB numbering).
                                - end_resid     (int): Last residue number (inclusive, PDB numbering).
                                Example: ('A', (1, 50))

    Returns:
        list[int]: CA atom indices (0-based, mdtraj numbering) for all residues
                   in secondary structure elements within the domain.

    Raises:
        AssertionError: If the specified chain_id is not present in the PDB topology.

    Example:
        >>> pdb_md = md.load_pdb('protein.pdb')
        >>> ca_indices = identify_folded_CA_idx(pdb_md, ('A', (1, 50)))
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
    Restrain the internal 3D structure of folded domains using pairwise CA–CA bonds.

    For each domain, all CA atoms in secondary-structure elements are identified via
    identify_folded_CA_idx. Pairs of those CA atoms within the cutoff distance in the
    reference PDB are added as HarmonicBondForce bonds, preserving the native
    geometry of each folded region throughout the simulation.

    Bond potential: U = 0.5 * Kcons * (r - r0)^2
    where r0 is the distance between the pair in the reference PDB.

    Args:
        system        (openmm.System): OpenMM System object to modify.
        pdb_ref       (str):           Path to the reference PDB file.
        domain_ranges (list of tuples): List of domain definitions, each as
                                        (chain_id, (start_resid, end_resid)).
                                        Example: [('A', (1, 50)), ('B', (75, 200))]
        Kcons         (float):         Force constant in kJ/mol/nm². Default: 400.
        cutoff        (float):         Maximum CA–CA distance (nm) for a pair to be
                                       included as a restrained bond. Default: 1.2 nm.

    Returns:
        openmm.System: The modified system with domain structural restraints added.

    Notes:
        The number of restrained pairs per domain is printed to stdout.

    Example:
        >>> domains = [('A', (1, 50)), ('B', (75, 200))]
        >>> system = domain_3D_restraint(system, 'reference.pdb', domains, Kcons=400, cutoff=1.2)
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
