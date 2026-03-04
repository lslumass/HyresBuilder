"""
This package is used to constructe Hyres Force Field, iConRNA force field
Athours: Shanlong Li, Xiping Gong, Yumeng Zhang
Date: Mar 9, 2024
Modified: Sep 29, 2025 
"""

from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np



def buildSystem(psf, system, ffs, modification=None):
    """
    Constructs a protein and/or RNA force field.
    
    Args:
        psf: PSF file containing topology information
        system: OpenMM system object
        ffs: Force field parameters dictionary containing:
            - dh: Debye-Huckel screening length
            - lmd: Lambda parameter for charge-charge interactions
            - er: Relative dielectric constant
        modification: custome defined function for further modifying system
    
    Returns:
        Modified OpenMM system with protein and/or RNA force field
    
    Raises:
        ValueError: If required parameters or forces are missing
    """
    print('\n################# constructe HyRes and/or iConRNA force field ####################')
    top = psf.topology
    
    # 1. Validate force field parameters
    required_params = ['dh', 'lmd', 'er']
    missing_params = [param for param in required_params if param not in ffs]
    if missing_params:
        raise ValueError(f"Missing required force field parameters: {', '.join(missing_params)}")
    
    # 2. Get forces, bondlist, and atom names
    # get forces
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
            nbfix = force
            force.setName('LJ Force w/ NBFIX')

    # get bondlist
    bondlist = []
    for bond in top.bonds():
        bondlist.append([bond[0].index, bond[1].index])
    #get all atom name
    atoms = []
    residues = []
    for atom in psf.topology.atoms():
        atoms.append(atom.name)
        residues.append(atom.residue.name)
    
    # 3 Replace HarmonicAngle with Restricted Bending (ReB) potential
    ReB = CustomAngleForce("0.5*kt*(theta-theta0)^2/(sin(theta)^kReB);")
    ReB.setName('ReBAngleForce')
    ReB.addPerAngleParameter("theta0")
    ReB.addPerAngleParameter("kt")
    ReB.addPerAngleParameter("kReB")
    for angle_idx in range(hmangle.getNumAngles()):
        ang = hmangle.getAngleParameters(angle_idx)
        if atoms[ang[0]] in ['P', 'C1', 'C2', 'NA', 'NB', 'NC', 'ND']:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 2])
        elif atoms[ang[0]] == 'CA' and atoms[ang[1]] == 'CB':
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 2])
        else:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 0])
    system.addForce(ReB)

    # 4. Add Debye-Hückel electrostatic interactions using CustomNonbondedForce
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

    # 5 Add 1-4 nonbonded interaction through custombondforece
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

    # 6. Add the Custom hydrogen bond force for protein backbone
    Ns, Hs, Os, Cs = [], [], [], []
    for atom in psf.topology.atoms():
        if atom.name == "N" and atom.residue.name != 'PRO':
            Ns.append(int(atom.index))
        if atom.name == "H":
            Hs.append(int(atom.index))
        if atom.name == "O":
            Os.append(int(atom.index))
        if atom.name == "C":
            Cs.append(int(atom.index))
    
    if len(Ns) != 0:
        sigma_hb = 0.29*unit.nanometer
        eps_hb = 2.2*unit.kilocalorie_per_mole
        formula = f"""epsilon*(5*(sigma/r)^12-6*(sigma/r)^10)*step(cos3)*cos3;
                r=distance(a1,d1); cos3=-cos(phi)^3; phi=angle(a1,d2,d1);
                sigma = {sigma_hb.value_in_unit(unit.nanometer)}; epsilon = {eps_hb.value_in_unit(unit.kilojoule_per_mole)};
        """
        HBforce = CustomHbondForce(formula)
        HBforce.setName('N-H--O HBForce')
        HBforce.setNonbondedMethod(nbforce.getNonbondedMethod())
        HBforce.setCutoffDistance(0.45*unit.nanometers)
        for idx in range(len(Hs)):
            HBforce.addDonor(Ns[idx], Hs[idx], -1)
            HBforce.addAcceptor(Os[idx], -1, -1)
        if HBforce.getNumAcceptors() != 0 and HBforce.getNumDonors() != 0:
            system.addForce(HBforce)

    # 7. Base stacking and pairing
    eps_base = 3.4*unit.kilocalorie_per_mole
    # relative strength of base pairing and stacking
    scales = {'AA':1.0, 'AG':1.0, 'AC':0.8, 'AU':0.8, 'GA':1.1, 'GG':1.1, 'GC':0.8, 'GU':0.8,       # stacking
              'CA':0.6, 'CG':0.6, 'CC':0.5, 'CU':0.4, 'UA':0.5, 'UG':0.5, 'UC':0.4, 'UU':0.1,       # stacking
              'A-U':0.89, 'C-G':1.14, 'G-U':0.76, 'general':0.76}   # pairing
    # optimal stacking distance
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
    
    if len(grps) != 0:
        # base stacking
        fstack = CustomCentroidBondForce(2, 'eps_stack*(5*(r0/r)^12-6.0*(r0/r)^10); r=distance(g1, g2);')
        fstack.setName('StackingForce')
        fstack.addPerBondParameter('eps_stack')
        fstack.addPerBondParameter('r0')
        # add all group
        for grp in grps:
            fstack.addGroup(grp[2])
        # get the stacking pairs
        for i in range(0,len(grps)-2,2):
            if grps[i][1] == grps[i+2][1]:
                pij = grps[i][0] + grps[i+2][0]
                fstack.addBond([i+1, i+2], [scales[pij]*eps_base, r0s[pij]*unit.nanometers]) 
        print('    add ', fstack.getNumBonds(), 'stacking pairs')
        system.addForce(fstack)

        # base pairing
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

        # add G-U pair through CustomHbondForce
        eps_GU = eps_base*scales['G-U']
        r_gu = 0.35*unit.nanometer

        if num_G != 0 and num_U !=0:
            formula = f"""eps_GU*(5.0*(r_gu/r)^12-6.0*(r_gu/r)^10)*step_phi; r=distance(a1,d1);
                        step_phi=step(cos_phi)*cos_phi; cos_phi=-cos(phi)^5; phi=angle(d1,a1,a2);
                        eps_GU={eps_GU.value_in_unit(unit.kilojoule_per_mole)};
                        r_gu={r_gu.value_in_unit(unit.nanometer)}
                      """
            pairGU = CustomHbondForce(formula)
            pairGU.setName('GUpairForce')
            pairGU.setNonbondedMethod(nbforce.getNonbondedMethod())
            pairGU.setCutoffDistance(0.65*unit.nanometers)

            for idx in range(len(g_c)):
                pairGU.addAcceptor(g_c[idx], g_b[idx], -1)
            for idx in range(len(u_b)):
                pairGU.addDonor(u_b[idx], -1, -1)
            system.addForce(pairGU)
            print(pairGU.getNumAcceptors(), pairGU.getNumDonors(), 'GU')

    # further modification defined in running scripts
    if callable(modification):
        modification(system)
        
    # 8. Delete the NonbondedForce and HarmonicAngleForce
    system.removeForce(nbforce_index)
    system.removeForce(hmangle_index)

    # 9. set unique ForceGroup id for each force
    forces = system.getForces()
    for i, force in enumerate(forces):
        force.setForceGroup(i)

    return system
