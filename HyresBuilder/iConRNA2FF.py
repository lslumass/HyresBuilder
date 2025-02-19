"""
This package is used to constructe iConRNA force field
Athours: Shanlong Li
Date: Jan 23, 2025
"""

from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np


###### for RNA System with A-U/G-C pairs ######
def iConRNA2System(psf, system, ffs):
    top = psf.topology
    # 2) constructe the force field
    print('\n################# constructe the HyRes force field ####################')
    # get nonbonded force
    for force_index, force in enumerate(system.getForces()):
        if force.getName() == "NonbondedForce":
            nbforce = force
            nbforce_index = force_index
        elif force.getName() == "HarmonicAngleForce":
            hmangle = force
            hmangle_index = force_index
    print('\n# get the NonBondedForce and HarmonicAngleForce:', nbforce.getName(), hmangle.getName())
    
    print('\n# get bondlist')
    # get bondlist
    bondlist = []
    for bond in top.bonds():
        bondlist.append([bond[0].index, bond[1].index])
    #get all atom name
    atoms = []
    for atom in psf.topology.atoms():
        atoms.append(atom.name)
    
    print('\n# replace HarmonicAngle with Restricted Bending (ReB) potential')
    # Custom Angle Force
    ReB = CustomAngleForce("kt*(theta-theta0)^2/(sin(theta)^2);")
    ReB.setName('ReBAngleForce')
    ReB.addPerAngleParameter("theta0")
    ReB.addPerAngleParameter("kt")
    for angle_idx in range(hmangle.getNumAngles()):
        ang = hmangle.getAngleParameters(angle_idx)
        ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4]])
    system.addForce(ReB)
    
    print('\n# add custom nonbondedforce')
    dh = ffs['dh']
    lmd = ffs['lmd']
    er = ffs['er']
    # add custom nonbondedforce: CNBForce, here only charge-charge interactions
    formula = f"""138.935456/er*charge1*charge2/r*exp(-r/dh)*kpmg;
                dh={dh.value_in_unit(unit.nanometer)}; er={er}; kpmg=select(lb1+lb2,1,lmd); lmd={lmd}
              """
    CNBForce = CustomNonbondedForce(formula)
    CNBForce.setName("LJ_ElecForce")
    CNBForce.setNonbondedMethod(nbforce.getNonbondedMethod())
    CNBForce.setUseSwitchingFunction(use=True)
    #CNBForce.setUseLongRangeCorrection(use=True)
    CNBForce.setCutoffDistance(1.8*unit.nanometers)
    CNBForce.setSwitchingDistance(1.6*unit.nanometers)
    CNBForce.addPerParticleParameter('charge')
    CNBForce.addPerParticleParameter('lb')
    
    for idx in range(nbforce.getNumParticles()):
        particle = nbforce.getParticleParameters(idx)
        if atoms[idx] == 'P':
            lb = 1
        elif atoms[idx] == 'MG':
            lb = -1
        else:
            lb = 2
        perP = [particle[0], lb]
        CNBForce.addParticle(perP)
    
    CNBForce.createExclusionsFromBonds(bondlist, 2)
    system.addForce(CNBForce)
    
    
    print('\n# add base stacking force')
    # base stakcing and paring
    # define relative strength of base pairing and stacking
    eps_base = ffs['eps_base']
    scales = {'AA':1.0, 'AG':1.0, 'AC':0.8, 'AU':0.8, 'GA':1.0, 'GG':1.0, 'GC':0.8, 'GU':0.8,
              'CA':0.4, 'CG':0.4, 'CC':0.2, 'CU':0.4, 'UA':0.4, 'UG':0.4, 'UC':0.2, 'UU':0.2,
              'A-U':1.2, 'C-G':1.5, 'G-U':2.0}

    # get all the groups of bases
    grps = []
    for atom in psf.topology.atoms():
        if atom.name == "NA":
            if atom.residue.name in ['A', 'G']:
                grps.append([atom.residue.name, [atom.index, atom.index+1, atom.index+2, atom.index+3]])
            elif atom.residue.name in ['C', 'U']:
                grps.append([atom.residue.name, [atom.index, atom.index+1, atom.index+2]])
    # base stacking
    Aform = CustomCentroidBondForce(2, 'eps_stack*((ra/r)^10-2*(ra/r)^5)*sr; sr=1/(1+exp(-20*(r-rs))); r=distance(g1, g2)')
    Aform.setName('IntraStackingForce')
    Aform.addPerBondParameter('eps_stack')
    Aform.addGlobalParameter('ra', 0.45*unit.nanometers)
    Aform.addGlobalParameter('rs', 0.60*unit.nanometers)    # for cutoff and switch

    for grp in grps:
        Aform.addGroup(grp[1])
    # get the stacking pairs
    sps = []
    for i in range(len(grps)-1):
        pij = grps[i][0] + grps[i+1][0]
        sps.append([[i, i+1], scales[pij]*eps_base])
    for sp in sps:
        Aform.addBond(sp[0], [sp[1]])
    print('    add ', Aform.getNumBonds(), 'Aform stacking pairs')
    system.addForce(Aform)
    

    # base pairing
    print('\n# add base pair force')
    a_b, a_c, a_d = [], [], []
    g_b, g_c, g_d = [], [], []
    c_a, c_b, c_c, u_a, u_b, u_c = [], [], [], [], [], []
    a_p, g_p, c_p, u_p = [], [], [], []
    num_A, num_G, num_C, num_U = 0, 0, 0, 0
    for atom in psf.topology.atoms():
        if atom.residue.name == 'A':
            num_A += 1
            if atom.name == 'NC':
                a_c.append(int(atom.index))
            elif atom.name == 'NB':
                a_b.append(int(atom.index))
            elif atom.name == 'ND':
                a_d.append(int(atom.index))
            elif atom.name == 'P':
                a_p.append(int(atom.index))
        elif atom.residue.name == 'G':
            num_G += 1
            if atom.name == 'NC':
                g_c.append(int(atom.index))
            elif atom.name == 'NB':
                g_b.append(int(atom.index))
            elif atom.name == 'ND':
                g_d.append(int(atom.index))
            elif atom.name == 'P':
                g_p.append(int(atom.index))
        elif atom.residue.name == 'U':
            num_U += 1
            if atom.name == 'NA':
                u_a.append(int(atom.index))
            elif atom.name == 'NB':
                u_b.append(int(atom.index))
            elif atom.name == 'NC':
                u_c.append(int(atom.index))
            elif atom.name == 'P':
                u_p.append(int(atom.index))
        elif atom.residue.name == 'C':
            num_C += 1
            if atom.name == 'NA':
                c_a.append(int(atom.index))
            elif atom.name == 'NB':
                c_b.append(int(atom.index))
            elif atom.name == 'NC':
                c_c.append(int(atom.index))
            elif atom.name == 'P':
                c_p.append(int(atom.index))
    # add A-U pair through CustomHbondForce
    eps_AU = eps_base*scales['A-U']
    r_au = 0.35*unit.nanometer
    r_au2 = 0.41*unit.nanometer
    
    if num_A != 0 and num_U != 0:
        formula = f"""eps_AU*((r_au/r)^10-2*(r_au/r)^5 + (r_au2/r2)^10-2*(r_au2/r2)^5)*step(cos1)*cos1;
                  r=distance(a1,d1); r2=distance(a3,d2); cos1=-cos(phi)^3; phi=angle(d1,a1,a2);
                  eps_AU={eps_AU.value_in_unit(unit.kilojoule_per_mole)};
                  r_au={r_au.value_in_unit(unit.nanometer)}; r_au2={r_au2.value_in_unit(unit.nanometer)}
                  """
        pairAU = CustomHbondForce(formula)
        pairAU.setName('AUpairForce')
        pairAU.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairAU.setCutoffDistance(0.65*unit.nanometer)
        for idx in range(len(a_c)):
            pairAU.addAcceptor(a_c[idx], a_b[idx], a_d[idx])
        for idx in range(len(u_b)):
            pairAU.addDonor(u_b[idx], u_c[idx], -1)
        system.addForce(pairAU)
        print(pairAU.getNumAcceptors(), pairAU.getNumDonors(), 'AU')
        
    # add C-G pair through CustomHbondForce
    eps_CG = eps_base*scales['C-G']
    r_cg = 0.35*unit.nanometer
    r_cg2 = 0.38*unit.nanometer
    
    if num_C != 0 and num_G != 0:
        formula = f"""eps_CG*((r_cg/r)^10-2*(r_cg/r)^5 + (r_cg2/r2)^10-2*(r_cg2/r2)^5)*step(cos1)*cos1;
                  r=distance(a1,d1); r2=distance(a3,d2); cos1=-cos(phi)^3; phi=angle(d1,a1,a2);
                  eps_CG={eps_CG.value_in_unit(unit.kilojoule_per_mole)};
                  r_cg={r_cg.value_in_unit(unit.nanometer)}; r_cg2={r_cg2.value_in_unit(unit.nanometer)}
                  """
        pairCG = CustomHbondForce(formula)
        pairCG.setName('CGpairForce')
        pairCG.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairCG.setCutoffDistance(0.65*unit.nanometer)
        for idx in range(len(g_c)):
            pairCG.addAcceptor(g_c[idx], g_b[idx], g_d[idx])
        for idx in range(len(c_b)):
            pairCG.addDonor(c_b[idx], c_c[idx], -1)
        system.addForce(pairCG)
        print(pairCG.getNumAcceptors(), pairCG.getNumDonors(), 'CG')

    # add G-U pair through CustomHbondForce
    eps_GU = eps_base*scales['G-U']
    r_gu = 0.35*unit.nanometer

    if num_U != 0 and num_G != 0:
        formula = f"""eps_GU*((r_gu/r)^10-2*(r_gu/r)^5)*step(cos1)*cos1;
                    r=distance(a1,d1); cos1=-cos(phi)^3; phi=angle(d1,a1,a2);
                    eps_GU={eps_GU.value_in_unit(unit.kilojoule_per_mole)}; r_gu={r_gu.value_in_unit(unit.nanometer)};
                    """
        pairGU = CustomHbondForce(formula)
        pairGU.setName('GUpairForce')
        pairGU.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairGU.setCutoffDistance(0.65*unit.nanometers)

        for idx in range(len(g_c)):
            pairGU.addAcceptor(g_c[idx], g_b[idx], -1)
        for idx in range(len(u_b)):
            pairGU.addDonor(u_c[idx], -1, -1)
        system.addForce(pairGU)
        print(pairGU.getNumAcceptors(), pairGU.getNumDonors(), 'GU')

    # delete the NonbondedForce and HarmonicAngleForce
    system.removeForce(nbforce_index)
    system.removeForce(hmangle_index)
    return system
