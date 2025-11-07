"""
define specific interactions for aging simulations
Date: Nov-07-2025
Authors: Shanlong Li
"""


from openmm.unit import *
from openmm.app import *
from openmm import *


def inRegisterHB(system, psf, age=1.0):
    """
    hydrogen bonds between same residues only for in-Register beta sheets
    Parameters:
    system: openmm system object
    psf: openmm psf object
    age: aging stength as HB force strength
    """
    for force in system.getForces():
        if force.getName() == "NonbondedForce":
            nbforce = force

    Ns, Hs, Os = [], [], []
    for atom in psf.topology.atoms():
        if atom.residue.name != 'PRO':
            if atom.name == "N":
                Ns.append(int(atom.index))
            if atom.name == "H":
                Hs.append(int(atom.index))
            if atom.name == "O":
                Os.append(int(atom.index))

    if len(Ns) != 0:
        sigma_hb = 0.29*unit.nanometer
        eps_hb = age*unit.kilocalorie_per_mole
        formula = f"""epsilon*(5*(sigma/r)^12-6*(sigma/r)^10)*step(cos3)*cos3;
                r=distance(a1,d1); cos3=-cos(phi)^3; phi=angle(a1,d2,d1);
                sigma = {sigma_hb.value_in_unit(unit.nanometer)};
                epsilon = {eps_hb.value_in_unit(unit.kilojoule_per_mole)};
                """
        inRegHB = CustomHbondForce(formula)
        inRegHB.setName('inRegister HBForce')
        inRegHB.setNonbondedMethod(nbforce.getNonbondedMethod())
        inRegHB.setCutoffDistance(0.45*unit.nanometers)
        for idx in range(len(Hs)):
            inRegHB.addDonor(Ns[idx], Hs[idx], -1)
            inRegHB.addAcceptor(Os[idx], -1, -1)
        
        system.addForce(inRegHB)
    return system

