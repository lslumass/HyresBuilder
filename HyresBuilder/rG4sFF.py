"""
This function is used to constructe iConRNA force field for rG4s 
Athours: Shanlong Li
Date: Dec 1, 2024
"""

from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np

###### for RNA System with A-U/G-C/G-G pairs ######
def rG4sSystem(psf, system, ffs):
    top = psf.topology
    # 2) constructe the force field
    print('\n################# constructe the protein-RNA mixed force field ####################')
    # get nonbonded force
    for force_index, force in enumerate(system.getForces()):
        if force.getName() == "NonbondedForce":
            nbforce = force
            nbforce_index = force_index
        elif force.getName() == "HarmonicAngleForce":
            hmangle = force
            hmangle_index = force_index
        elif force.getName() == "PeriodicTorsionForce":
            dihedral = force
            dihedral_index = force_index
        elif force.getName() == "CustomNonbondedForce":
            force.setName('LJ Force w/ NBFIX')
    
    print('\n# get bondlist')
    # get bondlist
    bondlist = []
    for bond in top.bonds():
        bondlist.append([bond[0].index, bond[1].index])
    #get all atom name
    atoms = []
    ress = []    # all the resid for each atom
    for atom in psf.topology.atoms():
        atoms.append(atom.name)
        ress.append(atom.residue.index)
    
    print('\n# replace HarmonicAngle with Restricted Bending (ReB) potential')
    # Custom Angle Force
    ReB = CustomAngleForce("0.5*kt*(theta-theta0)^2/(sin(theta)^kReB);")
    ReB.setName('ReBAngleForce')
    ReB.addPerAngleParameter("theta0")
    ReB.addPerAngleParameter("kt")
    ReB.addPerAngleParameter("kReB")
    for angle_idx in range(hmangle.getNumAngles()):
        ang = hmangle.getAngleParameters(angle_idx)
        if atoms[ang[0]] in ['P', 'C1', 'C2', 'NA', 'NB', 'NC', 'ND']:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], 2*ang[4], 2])
        else:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 0])
    system.addForce(ReB)

    print('\n# add custom nonbondedforce for DH-electrostatic interaction')
    dh = ffs['dh']
    lmd = ffs['lmd']
    er = ffs['er']
    # add custom nonbondedforce: CNBForce, here only charge-charge interactions
    formula = f"""138.935456/er*charge1*charge2/r*exp(-r/dh)*kpmg;
                dh={dh.value_in_unit(unit.nanometer)}; er={er}; kpmg=select(lb1+lb2,1,lmd); lmd={lmd}
              """
    CNBForce = CustomNonbondedForce(formula)
    CNBForce.setName("DH_ElecForce")
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
    
    CNBForce.createExclusionsFromBonds(bondlist, 3)
    system.addForce(CNBForce)

    print('\n# add 1-4 nonbonded force')
    # add nonbondedforce of 1-4 interaction through custombondforece
    formula = f"""(4.0*epsilon*six*(six-1.0)+(138.935456/er*charge)/r*exp(-r/dh));
              six=(sigma/r)^6; er={er}; dh={dh.value_in_unit(unit.nanometer)}
              """
    Force14 = CustomBondForce(formula)
    Force14.setName('1-4 interaction')
    Force14.addPerBondParameter('charge')
    Force14.addPerBondParameter('sigma')
    Force14.addPerBondParameter('epsilon')
    for idx in range(nbforce.getNumExceptions()):
        ex = nbforce.getExceptionParameters(idx)
        Force14.addBond(ex[0], ex[1], [ex[2], ex[3], ex[4]])
    system.addForce(Force14)

    print('\n# add RNA base stacking force')
    # base stakcing and paring
    # define relative strength of base pairing and stacking
    eps_base = ffs['eps_base']
    scales = {'AA':1.0, 'AG':1.0, 'AC':0.8, 'AU':0.8, 'GA':1.1, 'GG':1.1, 'GC':0.8, 'GU':0.8,
              'CA':0.6, 'CG':0.6, 'CC':0.5, 'CU':0.4, 'UA':0.5, 'UG':0.5, 'UC':0.4, 'UU':0.4,
              'A-U':0.89, 'C-G':1.14}
    
    r0s = {'AA':0.35, 'AG':0.35, 'GA':0.35, 'GG':0.35, 'AC':0.38, 'AU':0.38, 'GC':0.38, 'GU':0.38,
           'CA':0.40, 'CG':0.40, 'UA':0.40, 'UG':0.40, 'CC':0.43, 'CU':0.43, 'UC':0.43, 'UU':0.43}

    # get all the groups of bases
    grps = []
    for atom in psf.topology.atoms():
        if atom.name == "NA":
            if atom.residue.name in ['A', 'G']:
                grps.append([atom.residue.name, atom.residue.chain.id, [atom.index, atom.index+1]])
                grps.append([atom.residue.name, atom.residue.chain.id, [atom.index+2, atom.index+3]])
            elif atom.residue.name in ['C', 'U']:
                grps.append([atom.residue.name, atom.residue.chain.id, [atom.index, atom.index+1, atom.index+2]])
                grps.append([atom.residue.name, atom.residue.chain.id, [atom.index, atom.index+1, atom.index+2]])
    # base stacking
    fstack = CustomCentroidBondForce(2, 'eps_stack*(5*(r0/r)^12-6.0*(r0/r)^10); r=distance(g1, g2);')
    fstack.setName('StackingForce')
    fstack.addPerBondParameter('eps_stack')
    fstack.addPerBondParameter('r0')

    # add all group
    for grp in grps:
        fstack.addGroup(grp[2])
    # get the stacking pairs
    sps = []
    for i in range(0,len(grps)-2,2):
        if grps[i][1] == grps[i+2][1]:
            pij = grps[i][0] + grps[i+2][0]
            fstack.addBond([i+1, i+2], [scales[pij]*eps_base, r0s[pij]*unit.nanometers]) 

    print('    add ', fstack.getNumBonds(), 'stacking pairs')
    system.addForce(fstack)

    # base pairing
    print('\n# add RNA base pair force')
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
    r_au = 0.35*unit.nanometer
    r_au2 = 0.40*unit.nanometer
    
    if num_A != 0 and num_U != 0:
        formula = f"""eps_AU*(5.0*(r_au/r)^12-6.0*(r_au/r)^10 + 5.0*(r_au2/r2)^12-6.0*(r_au2/r2)^10)*step_phi;
                  r=distance(a1,d1); r2=distance(a3,d2); step_phi=step(cos_phi)*cos_phi; cos_phi=-cos(phi)^5; phi=angle(d1,a1,a2);
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
            pairAU.addDonor(u_b[idx], u_c[idx], u_a[idx])
        system.addForce(pairAU)
        print(pairAU.getNumAcceptors(), pairAU.getNumDonors(), 'AU')
        
    # add C-G pair through CustomHbondForce
    eps_CG = eps_base*scales['C-G']
    r_cg = 0.35*unit.nanometer
    r_cg2 = 0.38*unit.nanometer
     
    if num_C != 0 and num_G != 0:
        formula = f"""eps_CG*(5.0*(r_cg/r)^12-6.0*(r_cg/r)^10 + 5.0*(r_cg2/r2)^12-6.0*(r_cg2/r2)^10)*step_phi;
                  r=distance(a1,d1); r2=distance(a3,d2); step_phi=step(cos_phi)*cos_phi; cos_phi=-cos(phi)^5; phi=angle(d1,a1,a2);
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
            pairCG.addDonor(c_b[idx], c_c[idx], c_a[idx])
        system.addForce(pairCG)
        print(pairCG.getNumAcceptors(), pairCG.getNumDonors(), 'CG')


    # add G-G pair through CustomHbondForce
    eps_GG = ffs['ion_type']
    r_gg1 = 0.40*unit.nanometer     # for NB-ND
    r_gg2 = 0.42*unit.nanometer     # for NC-NC
    
    if num_G != 0:
        formula = f"""eps_GG*(5.0*(r_gg1/r1)^12-6.0*(r_gg1/r1)^10)*step1*step2;
                  r1=distance(d1,a1); step1=step(psi3)*psi3; psi3=cos(psi)^1; psi=dihedral(a2,a1,d1,d2);
                  step2=step(phi3)*phi3; phi3=-cos(phi)^1; phi=angle(a3,a1,d1);
                  eps_GG={eps_GG.value_in_unit(unit.kilojoule_per_mole)};
                  r_gg1={r_gg1.value_in_unit(unit.nanometer)};
                  """
        pairGG = CustomHbondForce(formula)
        pairGG.setName('GGpairForce1')
        pairGG.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairGG.setCutoffDistance(0.65*unit.nanometer)
        for idx in range(len(g_a)):
            pairGG.addDonor(g_c[idx], g_b[idx], -1)
            pairGG.addAcceptor(g_c[idx], g_d[idx], g_b[idx])
            pairGG.addExclusion(idx, idx)

            # exclude neighboring residues
            if idx+1 < len(g_a) and ress[g_a[idx]] + 1 == ress[g_a[idx+1]]:
                pairGG.addExclusion(idx, idx+1)
                pairGG.addExclusion(idx+1, idx)

        system.addForce(pairGG)

        formula = f"""eps_GG*(5.0*(r_gg2/r2)^12-6.0*(r_gg2/r2)^10)*step1*step2;
                  r2=distance(d1,a1); step1=step(psi3)*psi3; psi3=cos(psi)^1; psi=dihedral(a1,a2,d2,d1);
                  step2=step(phi3)*phi3; phi3=-cos(phi)^1; phi=angle(d3,d1,a1);
                  eps_GG={eps_GG.value_in_unit(unit.kilojoule_per_mole)};
                  r_gg1={r_gg1.value_in_unit(unit.nanometer)}; r_gg2={r_gg2.value_in_unit(unit.nanometer)};
                  """
        pairGG2 = CustomHbondForce(formula)
        pairGG2.setName('GGpairForce2')
        pairGG2.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairGG2.setCutoffDistance(0.65*unit.nanometer)
        for idx in range(len(g_a)):
            pairGG2.addDonor(g_b[idx], g_c[idx], g_a[idx])
            pairGG2.addAcceptor(g_d[idx], g_c[idx], -1)
            pairGG2.addExclusion(idx, idx)

            # exclude neighboring residues
            if idx+1 < len(g_a) and ress[g_a[idx]] + 1 == ress[g_a[idx+1]]:
                pairGG2.addExclusion(idx, idx+1)
                pairGG2.addExclusion(idx+1, idx)

        system.addForce(pairGG2)

        print(pairGG.getNumAcceptors(), pairGG.getNumDonors(), 'GG')

    # delete the NonbondedForce and HarmonicAngleForce
    system.removeForce(nbforce_index)
    system.removeForce(hmangle_index)
    return system
 

