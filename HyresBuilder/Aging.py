"""
| define specific interactions for aging simulations
| Date: Nov-07-2025
| Authors: Shanlong Li
"""


from openmm.unit import *
from openmm.app import *
from openmm import *


def inRegisterHB(system, top, res_list, age=1.0):
    """
    Add in-register backbone hydrogen bonds between identical residue positions
    across beta-sheet chains for amyloid aging simulations.

    Implements a ``CustomHbondForce`` that only forms N-H···O hydrogen bonds
    between donor and acceptor atoms that share the **same residue index**
    (``delta(di - ai) == 1``). This enforces in-register beta-sheet geometry,
    mimicking the structural locking that occurs during amyloid aging.
    Proline residues are excluded as they lack backbone NH groups. Self-pairs
    (same residue donating and accepting) are excluded via ``addExclusion``.

    The hydrogen bond potential takes the form:

    .. code-block:: text

        epsilon * (5*(sigma/r)^12 - 6*(sigma/r)^10) * step(cos3) * cos3 * delta(di-ai)

    where ``r`` is the N···O distance, ``phi`` is the N-H···O angle, and
    ``sigma = 0.29 nm``.

    Args:
        system (System): OpenMM ``System`` object to which the hydrogen bond
                         force will be added.
        top (Topology): OpenMM ``Topology`` object used to identify N, H, and O
                        atoms and their residue indices.
        res_list (list of int): Residue indices to include in the in-register
                                hydrogen bond network.
        age (float, optional): Aging strength scaling factor applied to the
                               hydrogen bond energy (in kcal/mol). A value of
                               ``1.0`` corresponds to 1 kcal/mol per bond.
                               Default is ``1.0``.

    Returns:
        System: The modified OpenMM ``System`` object with the
                ``inRegister HBForce`` added.

    Example:
        >>> from openmm.app import PDBFile, CharmmPsfFile
        >>> from HyresBuilder import Aging
        >>> psf = CharmmPsfFile("conf.psf")
        >>> res_list = list(range(10, 40))  # residues 10-39 form the fibril core
        >>> system = Aging.inRegisterHB(system, psf.topology, res_list, age=2.0)
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
            inRegHB.addExclusion(idx, idx)
        
        system.addForce(inRegHB)
    return system

