"""
Legacy iConRNA force field construction (original RNA-only model).

Legacy module: it is not imported anywhere else in the package. The
maintained iConRNA builder is :func:`HyresBuilder.FFs.iConRNASystem`, which
is used by :func:`HyresBuilder.utils.iConRNA_setup`. This module is kept for
reference/backward compatibility.

:func:`iConRNASystem` replaces the CHARMM ``HarmonicAngleForce`` and
``NonbondedForce`` of an OpenMM ``System`` with the iConRNA terms: Restricted
Bending angles, Debye–Hückel electrostatics, base stacking and A-U/C-G base
pairing. No 1-4 term, protein hydrogen bonds or force-group assignment is
included.

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.app``, ``openmm.unit``)
"""
from __future__ import division, print_function
# OpenMM Imports
from openmm.unit import *
from openmm.app import *
from openmm import *


def iConRNASystem(psf, system, ffs):
    """
    Build the legacy iConRNA RNA force field into an OpenMM system.

    Forces added, in order:

    1. ``'ReBAngleForce'`` — ``kt*(theta-theta0)^2/sin(theta)^2`` for every
       angle of the ``HarmonicAngleForce``.
    2. ``'LJ_ElecForce'`` — Debye–Hückel electrostatics
       ``138.935456/er*q1*q2/r*exp(-r/dh)*kpmg`` via ``CustomNonbondedForce``
       (cutoff 1.8 nm, switching from 1.6 nm, exclusions for pairs up to
       2 bonds apart). ``kpmg = lmd`` for ``P``–``MG`` pairs, 1 otherwise.
    3. ``'StackingForce'`` — (10, 6) centroid stacking potential with global
       ``r0 = 0.34 nm`` between consecutive bases in atom order (no chain
       check), depth ``scales[pair]*eps_base``.
    4. ``'AUpairForce'`` / ``'CGpairForce'`` — (10, 6) ``CustomHbondForce``
       pairing (cutoff 0.65 nm, angular gate ``-2*cos(phi)^3``), each added
       only if both base types are present.

    The ``NonbondedForce`` and ``HarmonicAngleForce`` are then removed.

    Args:
        psf (CharmmPsfFile): Parsed PSF object providing topology and atom
                             names.
        system (System): OpenMM ``System`` created from the PSF; modified in
                         place.
        ffs (dict): Force field parameters with keys ``'dh'`` (Quantity,
                    screening length), ``'lmd'`` (float, P–Mg2+ scaling
                    factor), ``'er'`` (float, relative dielectric constant)
                    and ``'eps_base'`` (Quantity, base energy scale for
                    stacking and pairing).

    Returns:
        System: The modified OpenMM ``System``.

    Raises:
        KeyError: If a required key is missing from ``ffs``.
    """
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
    scales = {'AA':1.0, 'AG':1.0, 'AC':0.8, 'AU':0.8, 'GA':1.0, 'GG':1.0, 'GC':1.0, 'GU':1.0,
              'CA':0.4, 'CG':0.5, 'CC':0.5, 'CU':0.3, 'UA':0.3, 'UG':0.3, 'UC':0.2, 'UU':0.0,
              'A-U':0.395, 'C-G':0.545}
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
    
    # delete the NonbondedForce and HarmonicAngleForce
    system.removeForce(nbforce_index)
    system.removeForce(hmangle_index)
    return system