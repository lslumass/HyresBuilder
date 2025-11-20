"""
simple version of adding restraints
Date: Oct 23, 2025
Author: Shanlong Li
"""

from openmm.unit import *
from openmm.app import *
from openmm import *


def posres_CA(system, pdb, residue_list=None, limited_range=None):
    """
    system: openmm.System
    pdb: openmm.app.PDBFile
    residue_list: list of residue index to be restrained
    limited_range: tuple defined the index range of CA for restraint
    """
    # add restraint
    ### set position restraints CA atoms
    restraint = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.setName("Ca_position_restraint")
    restraint.addGlobalParameter('k', 200.0*kilojoule_per_mole/unit.nanometer)
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


def comres_xyz(system, pdb, groups):
    """
    system: openmm.System
    pdb: openmm.app.PDBFile
    groups: list of atom index for com
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
    com_xyz = CustomCentroidBondForce(1, 'k*((x1 - cx)^2 + (y1 - cy)^2 + (z1 - cz)^2);')
    com_xyz.setName("COM_xyz_restraint")
    com_xyz.addGroup(groups)
    com_xyz.addGlobalParameter('k', 500.0*kilojoule_per_mole/(unit.nanometer**2))
    com_xyz.addGlobalParameter('cx', com(groups)[0])
    com_xyz.addGlobalParameter('cy', com(groups)[1])
    com_xyz.addGlobalParameter('cz', com(groups)[2])
    com_xyz.setUsesPeriodicBoundaryConditions(True)
    com_xyz.addBond([0])
    system.addForce(com_xyz)


def comres_yz(system, pdb, groups):
    """
    system: openmm.System
    pdb: openmm.app.PDBFile
    groups: list of atom index for com
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
    com_yz = CustomCentroidBondForce(1, 'k*((y1 - cy)^2 + (z1 - cz)^2);')
    com_yz.setName("COM_yz_restraint")
    com_yz.addGroup(groups)
    com_yz.addGlobalParameter('k', 500.0*kilojoule_per_mole/(unit.nanometer**2))
    com_yz.addGlobalParameter('cy', com(groups)[1])
    com_yz.addGlobalParameter('cz', com(groups)[2])
    com_yz.setUsesPeriodicBoundaryConditions(True)
    com_yz.addBond([0])
    system.addForce(com_yz)