"""
define specific interactions for aging simulations
Date: Nov-07-2025
Authors: Shanlong Li
"""


from openmm.unit import *
from openmm.app import *
from openmm import *


def inRegisterHB(system, top, res_list, age=1.0):
    """
    hydrogen bonds between same residues only for in-Register beta sheets
    Parameters:
    system: openmm system object
    res_list: list of residue indices for in-Register beta sheets
    age: aging stength as HB force strength
    """
    #for force in system.getForces():
    #    if force.getName() == "NonbondedForce":
    #        nbforce = force

    Ns, Hs, Os = [], [], []
    for atom in top.atoms():
        resid = int(atom.residue.id)
        if atom.residue.name != 'PRO' and resid in res_list:
            if atom.name == "N":
                Ns.append([int(atom.index), resid])
            if atom.name == "H":
                Hs.append([int(atom.index), resid])
            if atom.name == "O":
                Os.append([int(atom.index), resid])

    if len(Ns) != 0:
        sigma_hb = 0.29*unit.nanometer
        eps_hb = age*unit.kilocalorie_per_mole
        # cond = delta(adi); adi=di-ai; if resid is same, cond=1, else cond=0
        formula = f"""epsilon*(5*(sigma/r)^12-6*(sigma/r)^10)*step(cos3)*cos3*cond;
                r=distance(a1,d1); cos3=-cos(phi)^3; phi=angle(a1,d2,d1); cond=delta(adi); adi=di-ai;
                sigma = {sigma_hb.value_in_unit(unit.nanometer)};
                epsilon = {eps_hb.value_in_unit(unit.kilojoule_per_mole)};
                """
        inRegHB = CustomHbondForce(formula)
        inRegHB.setName('inRegister HBForce')
        #inRegHB.setNonbondedMethod(nbforce.getNonbondedMethod())
        inRegHB.setCutoffDistance(0.45*unit.nanometers)
        inRegHB.addPerDonorParameter("di")  # resid for donor
        inRegHB.addPerAcceptorParameter("ai")  # resid for acceptor
        for idx in range(len(Hs)):
            inRegHB.addDonor(Ns[idx][0], Hs[idx][0], -1, [Hs[idx][1],])
            inRegHB.addAcceptor(Os[idx][0], -1, -1, [Hs[idx][1],])
        
        system.addForce(inRegHB)
    return system

