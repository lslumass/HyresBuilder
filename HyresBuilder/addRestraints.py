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
    com_yz = CustomCentroidBondForce(1, 'kyz*((y1 - cy1)^2 + (z1 - cz1)^2);')
    com_yz.setName("COM_yz_restraint")
    com_yz.addGroup(groups)
    com_yz.addGlobalParameter('kyz', 500.0*kilojoule_per_mole/(unit.nanometer**2))
    com_yz.addPerBondParameter('cy1')
    com_yz.addPerBondParameter('cz1')
    com_yz.setUsesPeriodicBoundaryConditions(True)
    com_yz.addBond([0], com(groups)[1:])
    system.addForce(com_yz)