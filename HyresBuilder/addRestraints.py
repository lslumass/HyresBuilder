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
    CA positional resitraints based on resid list.

    Args:
        system: openmm.System
        pdb: openmm.app.PDBFile, PDBFile(pdb_file)
        residue_list: list of residue index to be restrained
        limited_range: tuple defined the index range of CA for restraint
    
    Returns:
        no returns, system will be modified.
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
    CA positional restraints based on atom index.

    Args:
        system: openmm.System
        pdb: openmm.app.PDBFile, PDBFile(pdb_file)
        grp: list of atom index of CA

    Returns:
        no return, system will be modified.
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
    Specific restraints for amyloid.

    Args:
        system: openmm.System
        pdb: openmm.app.PDBFile, PDBFile(pdb_file)
        alignment_file: alignment.ali when build fibril
    
    Returns:
        no return, system will be modified
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
    Freeze amyloid through set atom mass of fibril core to zero.

    Args:
        system: openmm.System
        pdb: openmm.app.PDBFile, PDBFile(pdb_file)
        alignment_file: alignment.ali when build fibril

    Returns:
        no return, system will be modified.
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
    Center of mass (COM) restraints at xyz dimensions.

    Args:
        system: openmm.System
        pdb: openmm.app.PDBFile, PDBFile(pdb_file)
        groups: list of atom index for com

    Returns:
        no return, add new force to system
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
    Center of mass (COM) restraints at yz dimensions.

    Args:
        system: openmm.System
        pdb: openmm.app.PDBFile, PDBFile(pdb_file)
        groups: list of atom index for com

    Returns:
        no return, add new force to system
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
