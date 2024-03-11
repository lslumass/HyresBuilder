"""
This package is used to constructe Hyres Force Field
Athour: Shanlong Li
Date: Mar 9, 2024
"""

from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np


def createHyresSystem(psf, params, ffs):
    T = ffs['temp']
    c_ion = ffs['c_ion']
    er = ffs['er']
    eps_base = ffs['eps_base']
    eps_gen = ffs['eps_gen']
    r_cut = ffs['r_cut']
    top = psf.topology
    system = psf.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds)
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
    # add custom nonbondedforce: CNBForce
    formula = '(4.0*epsilon*((sigma/r)^12-(sigma/r)^6)+(138.935456/eps*charge1*charge2)/r*exp(-kf*r));'+\
              'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2);'
    CNBForce = CustomNonbondedForce(formula)
    CNBForce.setName("LJ_ElecForce")
    CNBForce.setNonbondedMethod(nbforce.getNonbondedMethod())
    CNBForce.setUseSwitchingFunction(use=True)
    #CNBForce.setUseLongRangeCorrection(use=True)
    CNBForce.setCutoffDistance(r_cut)
    CNBForce.setSwitchingDistance(r_cut-0.2*unit.nanometers)
    CNBForce.addGlobalParameter('eps', er)
    CNBForce.addGlobalParameter('kf', np.sqrt(c_ion/9.480)*AngstromsPerNm)

    # perparticle variables: sigma, epsilon, charge,
    CNBForce.addPerParticleParameter('charge')
    CNBForce.addPerParticleParameter('sigma')
    CNBForce.addPerParticleParameter('epsilon')
    ## scale charge of MG through lmd
    lmd = 0.55+0.00668*(T-300)
    print('lambda: ', lmd)
    for idx in range(nbforce.getNumParticles()):
        particle = nbforce.getParticleParameters(idx)
        if atoms[idx] == 'MG':
            particle[0] = particle[0]*lmd
        perP = [particle[0], particle[1], particle[2]]
        CNBForce.addParticle(perP)

    CNBForce.createExclusionsFromBonds(bondlist, 2)
    system.addForce(CNBForce)

    print('\n# add base stacking force')
    # base stakcing and paring
    # define relative strength of base pairing and stacking
    scales = {'AA':1.0, 'AG':1.0, 'AC':0.8, 'AU':0.8, 'GA':1.0, 'GG':1.0, 'GC':1.0, 'GU':1.0,
              'CA':0.4, 'CG':0.5, 'CC':0.5, 'CU':0.3, 'UA':0.3, 'UG':0.3, 'UC':0.2, 'UU':0.0,
              'A-U':0.35, 'C-G':0.53, 'G-U':0.7}

    # get all the groups of bases
    grps = []
    for atom in psf.topology.atoms():
        if atom.name == "NA":
            if atom.residue.name in ['A', 'G']:
                grps.append([atom.residue.name, [atom.index, atom.index+1]])
                grps.append([atom.residue.name, [atom.index+2, atom.index+3]])
            elif atom.residue.name in ['C', 'U']:
                grps.append([atom.residue.name, [atom.index, atom.index+1]])
                grps.append([atom.residue.name, [atom.index+1, atom.index+2]])

    # base stacking
    fstack = CustomCentroidBondForce(2, "eps_stack*(5*(r0/r)^10-6.0*(r0/r)^6); r=distance(g1,g2);")
    fstack.setName('StackingForce')
    fstack.addPerBondParameter('eps_stack')
    fstack.addGlobalParameter('r0', 0.34*unit.nanometers)
    # add all group
    for grp in grps:
        fstack.addGroup(grp[1])
    # get the stacking pairs
    sps = []
    for i in range(0,len(grps)-2,2):
        grp = grps[i]
        pij = grps[i][0] + grps[i+2][0]
        sps.append([[i+1, i+2], scales[pij]*eps_base])
    for sp in sps:
        fstack.addBond(sp[0], [sp[1]])
    print('    add ', fstack.getNumBonds(), 'stacking pairs')
    system.addForce(fstack)

    # general hbond force
    d1, d2, a = [], [], []
    for atom in psf.topology.atoms():
        if atom.name == 'NC' and atom.residue.name in ['G', 'A']:
            d1.append(int(atom.index))
            d2.append(int(atom.index)-2)
        elif atom.name == 'ND' and atom.residue.name == 'G':
            d1.append(int(atom.index))
            d2.append(int(atom.index)-3)
        elif atom.name == 'NB' and atom.residue.name in ['U', 'C']:
            d1.append(int(atom.index))
            d2.append(int(atom.index)-1)
        elif atom.name in ['NB', 'NC', 'ND']:
            a.append(int(atom.index))

    print('\n# add general hbond between base pairs')
    formula = 'eps_gen*(5.0*(g0/r)^10-6.0*(g0/r)^6)*step(cos3)*cos3;'+\
              'r=distance(a1,d1); cos3=-2*cos(phi)^3; phi=angle(a1,d1,d2);'
    pairGen = CustomHbondForce(formula)
    pairGen.setName('GeneralPairForce')
    pairGen.setNonbondedMethod(nbforce.getNonbondedMethod())
    pairGen.addGlobalParameter('eps_gen', eps_gen)
    pairGen.addGlobalParameter('g0', 0.300*unit.nanometers)
    pairGen.setCutoffDistance(0.65*unit.nanometers)
    for idx in range(len(d1)):
        pairGen.addDonor(d1[idx], d2[idx], -1)
    for idx in range(len(a)):
        pairGen.addAcceptor(a[idx], -1, -1)
    for i in range(len(d1)):
        for j in range(len(a)):
            if d1[i] == a[j]:
                pairGen.addExclusion(i, j)
    print(pairGen.getNumAcceptors(), pairGen.getNumDonors(), 'General')
    system.addForce(pairGen)

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
    if num_A != 0 and num_U != 0:
        formula = 'eps_AU*(5.0*(r_au/r)^10-6.0*(r_au/r)^6 + 5*(r_au2/r2)^10-6.0*(r_au2/r2)^6)*step(cos3)*cos3;'+\
                  'r=distance(a1,d1); r2=distance(a3,d2); cos3=-2*cos(phi)^3; phi=angle(d1,a1,a2);'
        pairAU = CustomHbondForce(formula)
        pairAU.setName('AUpairForce')
        pairAU.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairAU.addGlobalParameter('eps_AU', eps_base*scales['A-U'])
        pairAU.addGlobalParameter('r_au', 0.305*unit.nanometers)
        pairAU.addGlobalParameter('r_au2', 0.40*unit.nanometers)
        pairAU.setCutoffDistance(0.65*unit.nanometer)

        for idx in range(len(a_c)):
            pairAU.addAcceptor(a_c[idx], a_b[idx], a_d[idx])
        for idx in range(len(u_b)):
            pairAU.addDonor(u_b[idx], u_c[idx], -1)
        system.addForce(pairAU)
        print(pairAU.getNumAcceptors(), pairAU.getNumDonors(), 'AU')

    # add C-G pair through CustomHbondForce
    if num_C != 0 and num_G != 0:
        formula = 'eps_CG*(5.0*(r_cg/r)^10-6.0*(r_cg/r)^6 + 5*(r_cg2/r2)^10-6.0*(r_cg2/r2)^6)*step(cos3)*cos3;'+\
                  'r=distance(a1,d1); r2=distance(a3,d2); cos3=-2*cos(phi)^3; phi=angle(d1,a1,a2);'
        pairCG = CustomHbondForce(formula)
        pairCG.setName('CGpairForce')
        pairCG.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairCG.addGlobalParameter('eps_CG', eps_base*scales['C-G'])
        pairCG.addGlobalParameter('r_cg', 0.305*unit.nanometers)
        pairCG.addGlobalParameter('r_cg2', 0.35*unit.nanometers)
        pairCG.setCutoffDistance(0.65*unit.nanometer)

        for idx in range(len(g_c)):
            pairCG.addAcceptor(g_c[idx], g_b[idx], g_d[idx])
        for idx in range(len(c_b)):
            pairCG.addDonor(c_b[idx], c_c[idx], -1)
        system.addForce(pairCG)
        print(pairCG.getNumAcceptors(), pairCG.getNumDonors(), 'CG')

    # add G-U pair through CustomHbondForce
    if num_U != 0 and num_G != 0:
        formula = 'eps_GU*(5.0*(r_gu/r)^10-6.0*(r_gu/r)^6)*step(cos3)*cos3;'+\
                  'r=distance(a1,d1); cos3=-2*cos(phi)^3; phi=angle(d1,a1,a2);'
        pairGU = CustomHbondForce(formula)
        pairGU.setName('GUpairForce')
        pairGU.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairGU.addPerDonorParameter('eps_GU')
        pairGU.addGlobalParameter('r_gu', 0.300*unit.nanometers)
        pairGU.setCutoffDistance(0.65*unit.nanometers)

        for idx in range(len(g_c)):
            pairGU.addAcceptor(g_c[idx], g_b[idx], -1)
        for idx in range(len(u_b)):
            pairGU.addDonor(u_b[idx], -1, -1, [eps_base*scales['G-U']])
        system.addForce(pairGU)
        print(pairGU.getNumAcceptors(), pairGU.getNumDonors(), 'GU')

    # delete the NonbondedForce and HarmonicAngleForce
    system.removeForce(nbforce_index)
    system.removeForce(hmangle_index)

    return system
