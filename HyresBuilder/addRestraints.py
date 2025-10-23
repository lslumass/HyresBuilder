"""
simple version of adding restraints
Date: Oct 23, 2025
Author: Shanlong Li
system: openmm.System
pdb: openmm.app.PDBFile
residue_list: list of residue index to be restrained
"""

from openmm.unit import *
from openmm.app import *
from openmm import *


def posres_CA(system, pdb, residue_list=None):
    # add restraint
    ### set position restraints CA atoms
    restraint = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k', 200.0*kilojoule_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atom in pdb.topology.atoms():
        if atom.residue.index in residue_list and atom.name == 'CA':
            restraint.addParticle(atom.index, pdb.positions[atom.index])
    system.addForce(restraint)
