"""
Here are some templated examples for adding restraints
psf: psf = CharmmPsfFile(psf_file)
pdb: pdb = PDBFile(pdb_file)
"""

# OpenMM Imports
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np


def position_restraint(psf, pdb, system):
    pos_restraint = CustomExternalForce('kg*periodicdistance(x, y, z, x0, y0, z0)^2;')
    pos_restraint.addGlobalParameter('kg', 10000)
    pos_restraint.addPerParticleParameter('x0')
    pos_restraint.addPerParticleParameter('y0')
    pos_restraint.addPerParticleParameter('z0')
    for atom, pos in zip(psf.topology.atoms(), pdb.positions):
        if atom.residue.name == 'G':
            if atom.name in ['NA', 'NB', 'NC', 'ND']:
                pos_restraint.addParticle(atom.index, pos)
    system.addForce(pos_restraint)
    return system