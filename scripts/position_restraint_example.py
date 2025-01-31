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
import MDAnalysis as mda
from MDAnalysis.analysis.dssp import DSSP


# calculate the structure of each residue, and get the list of structured CA
def identify_structured_residue(psf, pdb):
    u=mda.Universe(psf, pdb)
    sel = u.select_atoms('segname A0')
    cas = u.select_atoms('segname A0 and name CA')
    result = DSSP(sel).run()
    strcutre_CA = []
    for ca, s in zip(cas, result):
        if s in ['E', 'H']:
            strcutre_CA.append(ca.index)
    
    return strcutre_CA


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