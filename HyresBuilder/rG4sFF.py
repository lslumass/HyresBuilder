"""
This package is used to constructe iConRNA force field for rG4s 
Athours: Shanlong Li
Date: Dec 1, 2024
"""

from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np

###### for RNA System with A-U/G-C/G-G pairs ######
def iConRNASystem(psf, system, ffs):
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
              'A-U':0.35, 'C-G':0.525, 'G-G': 10.0}
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
    fstack = CustomCentroidBondForce(2, 'eps_stack*(5*(r0/r)^10-6.0*(r0/r)^6); r=distance(g1, g2);')
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
    
    # base pairing
    print('\n# add base pair force')
    a_b, a_c, a_d = [], [], []
    g_b, g_c, g_d, g_a = [], [], [], []
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
            elif atom.name == 'NA':
                g_a.append(int(atom.index))
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
    r_au = 0.304*unit.nanometer
    r_au2 = 0.37*unit.nanometer
    
    if num_A != 0 and num_U != 0:
        formula = f"""eps_AU*(5.0*(r_au/r)^10-6.0*(r_au/r)^6 + 5*(r_au2/r2)^10-6.0*(r_au2/r2)^6)*step(cos3)*cos3;
                  r=distance(a1,d1); r2=distance(a3,d2); cos3=-2*cos(phi)^3; phi=angle(d1,a1,a2);
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
    r_cg = 0.304*unit.nanometer
    r_cg2 = 0.35*unit.nanometer
    
    if num_C != 0 and num_G != 0:
        formula = f"""eps_CG*(5.0*(r_cg/r)^10-6.0*(r_cg/r)^6 + 5*(r_cg2/r2)^10-6.0*(r_cg2/r2)^6)*step(cos3)*cos3;
                  r=distance(a1,d1); r2=distance(a3,d2); cos3=-2*cos(phi)^3; phi=angle(d1,a1,a2); psi=dihedral(a3,a1,d1,d2);
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


    # add G-G pair through CustomHbondForce
    eps_GG = eps_base*scales['G-G']
    r_gg = 0.395*unit.nanometer     # for NB-ND
    r_gg2 = 0.37*unit.nanometer     # for NC-NC
    
    if num_G != 0:
        formula = f"""eps_GG*(5.0*(r_gg/r)^10-6.0*(r_gg/r)^6)*step(cos3)*cos3;
                  r=distance(d2,a1); cos3=-2*cos(phi)^3; phi=angle(d1,d2,a1);
                  eps_GG={eps_GG.value_in_unit(unit.kilojoule_per_mole)};
                  r_gg={r_gg.value_in_unit(unit.nanometer)};
                  """
        pairGG1 = CustomHbondForce(formula)
        pairGG1.setName('GGpairForce')
        pairGG1.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairGG1.setCutoffDistance(0.65*unit.nanometer)
        for idx in range(len(g_a)):
            pairGG1.addDonor(g_a[idx], g_b[idx], -1)
            pairGG1.addAcceptor(g_d[idx], -1, -1)
            pairGG1.addExclusion(idx, idx)
        
        #system.addForce(pairGG1)
        print(pairGG1.getNumAcceptors(), pairGG1.getNumDonors(), 'GG1')
    
        formula = f"""eps_GG*(5*(r_gg2/r2)^10-6.0*(r_gg2/r2)^6)*step(cos3)*cos3;
                  r2=distance(d2,a1); cos3=-2*cos(phi)^3; phi=angle(d1,d2,a1);
                  eps_GG={eps_GG.value_in_unit(unit.kilojoule_per_mole)};
                  r_gg2={r_gg2.value_in_unit(unit.nanometer)}
                  """
        pairGG2 = CustomHbondForce(formula)
        pairGG2.setName('GGpairForce')
        pairGG2.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairGG2.setCutoffDistance(0.65*unit.nanometer)
        for idx in range(len(g_a)):
            pairGG2.addDonor(g_d[idx], g_c[idx], -1)
            pairGG2.addAcceptor(g_c[idx], -1, -1)
            pairGG2.addExclusion(idx, idx)
        
        #system.addForce(pairGG2)
        print(pairGG2.getNumAcceptors(), pairGG2.getNumDonors(), 'GG2')

    
    # delete the NonbondedForce and HarmonicAngleForce
    system.removeForce(nbforce_index)
    system.removeForce(hmangle_index)
    return system

