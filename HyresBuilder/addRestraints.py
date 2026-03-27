"""
| additional restraints
| Date: Oct 23, 2025
| Author: Shanlong Li
"""

from openmm.unit import *
from openmm.app import *
from openmm import *


def posres_CA(system, pdb, residue_list=None, limited_range=None):
    """
    Apply positional restraints to CA atoms selected by residue index.

    Adds a harmonic ``CustomExternalForce`` that restrains each selected CA atom
    to its reference position in the PDB file with a spring constant of
    200 kJ/mol/nm². Restraints can be further filtered to a specific atom index
    range using ``limited_range``.

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        residue_list (list of int, optional): Residue indices to restrain.
                                              If ``None``, no atoms are restrained.
        limited_range (tuple of int, optional): ``(min_index, max_index)`` atom
                                                index range. Only CA atoms within
                                                this range are restrained.
                                                If ``None``, all CA atoms in
                                                ``residue_list`` are restrained.

    Returns:
        None. Modifies ``system`` in place by adding a ``Ca_position_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.posres_CA(system, pdb, residue_list=[1, 2, 3, 4, 5])
        >>> addRestraints.posres_CA(system, pdb, residue_list=[1, 2, 3],
        ...                         limited_range=(0, 500))
    """
    
    # add restraint
    ### set position restraints CA atoms
    restraint = CustomExternalForce('kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.setName("Ca_position_restraint")
    restraint.addGlobalParameter('kpos', 200.0*kilojoule_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    
    atoms = list(pdb.topology.atoms()) 
    if limited_range:
        atom_min, atom_max = limited_range[0], limited_range[1]
    else:
        atom_min, atom_max = 0, len(atoms)
    for atom in atoms:
        resid, name = int(atom.residue.id), atom.name
        if resid in residue_list and name == 'CA':
            if atom.index > atom_min and atom.index < atom_max:
                restraint.addParticle(atom.index, pdb.positions[atom.index])
    system.addForce(restraint)

def posres_CAs(system, pdb, grp):
    """
    Apply positional restraints to CA atoms selected by atom index.

    Adds a harmonic ``CustomExternalForce`` that restrains each atom in ``grp``
    to its reference position in the PDB file with a spring constant of
    200 kJ/mol/nm². Use this function when you already know the exact atom
    indices to restrain, rather than selecting by residue ID.

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        grp (list of int): Atom indices to restrain.

    Returns:
        None. Modifies ``system`` in place by adding a ``Ca_position_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.posres_CAs(system, pdb, grp=[0, 5, 10, 15])
    """

    # add restraint
    ### set position restraints CA atoms
    restraint = CustomExternalForce('kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.setName("Ca_position_restraint")
    restraint.addGlobalParameter('kpos', 200.0*kilojoule_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    
    for idx in grp:
        restraint.addParticle(idx, pdb.positions[idx])
    system.addForce(restraint)

def posre_amyloid(system, pdb, alignment_file):
    """
    Apply CA positional restraints to the structured core of an amyloid fibril.

    Reads an alignment file (``alignment.ali``) to identify which residues are
    present in the fibril core (non-``'-'`` positions in the alignment sequence).
    Restraints are applied to CA atoms of those residues across all chains using
    :func:`posres_CAs` with a spring constant of 200 kJ/mol/nm².

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        alignment_file (str): Path to the ``alignment.ali`` file generated during
                              fibril model building. The number of sequence blocks
                              must match the number of chains in the PDB.

    Returns:
        None. Modifies ``system`` in place by adding positional restraints via
        :func:`posres_CAs`.

    Raises:
        SystemExit: If the number of chains in ``pdb`` does not match the number
                    of sequence blocks in ``alignment_file``.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("fibril.pdb")
        >>> addRestraints.posre_amyloid(system, pdb, "alignment.ali")
    """

    with open(alignment_file, 'r') as f:
        lines = f.readlines()
    blocks = [index for index, line in enumerate(lines) if line.startswith('>')]
    b1, b2 = blocks[:2]
    #nchains = b2 - b1 -3        # count the number of chains based on the lines in alignment.ali
    missings = lines[b1+2:b2-1]     # get all the sequences for each chain, "-" for missing residue

    chains = list(pdb.topology.chains())
    if len(chains) != len(missings):
        print(f"Unconsistent chain number! Found {len(chains)} in pdb file, but {len(missings)} in alignment.ali")
        exit(1)
    # get the CA index for un-missing residues
    grp = []
    for chain, sequence in zip(chains, missings):
        residues = list(chain.residues())
        nres = len(residues)
        seq = sequence[:nres]
        ca = None
        for res, s in zip(residues, seq):
            if s != '-':
                for atom in res.atoms():
                    if atom.name == 'CA':
                        ca = atom.index
                        grp.append(ca)
     
    # add position restraint
    posres_CAs(system, pdb, grp)

def freeze_amyloid(system, pdb, alignment_file):
    """
    Freeze the structured core of an amyloid fibril by setting atom masses to zero.

    Reads an ``alignment.ali`` file to identify residues present in the fibril
    core (non-``'-'`` positions in the alignment sequence). Sets the mass of
    every atom in those residues to zero, effectively freezing them during
    simulation. This is cheaper than positional restraints and guarantees no
    drift of the fibril core.

    Args:
        system (System): OpenMM ``System`` object whose particle masses will be
                         modified.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology information
                       (e.g. ``PDBFile('fibril.pdb')``).
        alignment_file (str): Path to the ``alignment.ali`` file generated during
                              fibril model building. The number of sequence blocks
                              must match the number of chains in the PDB.

    Returns:
        None. Modifies ``system`` in place by setting particle masses to zero.

    Raises:
        SystemExit: If the number of chains in ``pdb`` does not match the number
                    of sequence blocks in ``alignment_file``.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("fibril.pdb")
        >>> addRestraints.freeze_amyloid(system, pdb, "alignment.ali")
    """

    with open(alignment_file, 'r') as f:
        lines = f.readlines()
    blocks = [index for index, line in enumerate(lines) if line.startswith('>')]
    b1, b2 = blocks[:2]
    #nchains = b2 - b1 -3        # count the number of chains based on the lines in alignment.ali
    missings = lines[b1+2:b2-1]     # get all the sequences for each chain, "-" for missing residue

    chains = list(pdb.topology.chains())
    if len(chains) != len(missings):
        print(f"Unconsistent chain number! Found {len(chains)} in pdb file, but {len(missings)} in alignment.ali")
        exit(1)
    # get the CA index for un-missing residues
    grp = []
    for chain, sequence in zip(chains, missings):
        residues = list(chain.residues())
        nres = len(residues)
        seq = sequence[:nres]
        ca = None
        for res, s in zip(residues, seq):
            if s != '-':
                for atom in res.atoms():
                    system.setParticleMass(atom.index, 0.0*unit.amu)

#def freeze_residues(system, pdb, residue_list):


def comres_xyz(system, pdb, groups):
    """
    Apply a center-of-mass (COM) restraint in all three (x, y, z) dimensions.

    Computes the mass-weighted COM of the selected atoms from their reference
    positions and adds a ``CustomCentroidBondForce`` that penalizes deviation
    from that reference COM with a spring constant of 500 kJ/mol/nm². Periodic
    boundary conditions are enabled. Use this to prevent drift of a molecular
    group in all directions.

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        groups (list of int): Atom indices whose COM will be restrained.

    Returns:
        None. Modifies ``system`` in place by adding a ``COM_xyz_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.comres_xyz(system, pdb, groups=[0, 1, 2, 3, 4])
    """

    # add COM restraint
    cds = pdb.getPositions(asNumpy=True)
    def com(grp):
        cx_sum, cy_sum, cz_sum, m_sum = quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons)
        for i in grp:
            cx_sum += system.getParticleMass(i)*cds[i,0]
            cy_sum += system.getParticleMass(i)*cds[i,1]
            cz_sum += system.getParticleMass(i)*cds[i,2]
            m_sum += system.getParticleMass(i)
        cx, cy, cz = cx_sum/m_sum, cy_sum/m_sum, cz_sum/m_sum
        return [cx, cy, cz]

    print('com of selected residues:', com(groups))
    com_xyz = CustomCentroidBondForce(1, 'kxyz*((x1 - cx)^2 + (y1 - cy)^2 + (z1 - cz)^2);')
    com_xyz.setName("COM_xyz_restraint")
    com_xyz.addGroup(groups)
    com_xyz.addGlobalParameter('kxyz', 500.0*kilojoule_per_mole/(unit.nanometer**2))
    com_xyz.addPerBondParameter('cx')
    com_xyz.addPerBondParameter('cy')
    com_xyz.addPerBondParameter('cz')
    com_xyz.setUsesPeriodicBoundaryConditions(True)
    com_xyz.addBond([0], com(groups))
    system.addForce(com_xyz)


def comres_yz(system, pdb, groups):
    """
    Apply a center-of-mass (COM) restraint in the y and z dimensions only.

    Computes the mass-weighted COM of the selected atoms from their reference
    positions and adds a ``CustomCentroidBondForce`` that penalizes deviation
    in the y and z directions with a spring constant of 500 kJ/mol/nm². The
    x dimension is left unrestrained, allowing free movement along that axis.
    Useful for slab or membrane simulations where lateral diffusion along x
    should remain unhindered. Periodic boundary conditions are enabled.

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        groups (list of int): Atom indices whose COM will be restrained in y/z.

    Returns:
        None. Modifies ``system`` in place by adding a ``COM_yz_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.comres_yz(system, pdb, groups=[0, 1, 2, 3, 4])
    """


    # add COM restraint
    cds = pdb.getPositions(asNumpy=True)
    def com(grp):
        cx_sum, cy_sum, cz_sum, m_sum = quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons)
        for i in grp:
            cx_sum += system.getParticleMass(i)*cds[i,0]
            cy_sum += system.getParticleMass(i)*cds[i,1]
            cz_sum += system.getParticleMass(i)*cds[i,2]
            m_sum += system.getParticleMass(i)
        cx, cy, cz = cx_sum/m_sum, cy_sum/m_sum, cz_sum/m_sum
        return [cx, cy, cz]

    print('com of selected residues:', com(groups))
    com_yz = CustomCentroidBondForce(1, 'kyz*((y1 - cy)^2 + (z1 - cz)^2);')
    com_yz.setName("COM_yz_restraint")
    com_yz.addGroup(groups)
    com_yz.addGlobalParameter('kyz', 500.0*kilojoule_per_mole/(unit.nanometer**2))
    com_yz.addPerBondParameter('cy')
    com_yz.addPerBondParameter('cz')
    com_yz.setUsesPeriodicBoundaryConditions(True)
    com_yz.addBond([0], com(groups)[1:])
    system.addForce(com_yz)
