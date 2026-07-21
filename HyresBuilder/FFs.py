"""
HyRes protein and iConRNA RNA coarse-grained force field construction.
 
Overview
--------
This module replaces the standard OpenMM force terms generated from a CHARMM
PSF/parameter file with the custom interactions that define the HyRes protein
and iConRNA RNA coarse-grained models. All forces are added to an existing
OpenMM ``System`` object in place; the original ``NonbondedForce`` and
``HarmonicAngleForce`` are removed after substitution.
 
Three top-level builders are provided, each targeting a different molecular
context:
 
* :func:`buildSystem` — general HyRes/iConRNA mixed protein-RNA system.
* :func:`iConRNASystem` — legacy iConRNA-only RNA model (PNAS 2025).
* :func:`rG4sSystem` — RNA G-quadruplex (rG4) system with A-U, C-G, and G-G
  base pairs.
 
Force terms applied by :func:`buildSystem`
------------------------------------------
Forces are constructed and registered in the following order:
 
1. **Restricted Bending (ReB) angle force** — replaces ``HarmonicAngleForce``
   with a sine-based potential ``0.5*kt*(theta-theta0)^2 / sin(theta)^kReB``.
   RNA backbone atoms (P, C1, C2, NA–ND) and CA–CB angles use ``kReB = 2``
   to prevent numerical collapse near the planar singularity; all other angles
   use ``kReB = 0``, which recovers the ordinary harmonic form.
 
2. **Debye–Hückel electrostatics** — screened Coulomb interactions via
   ``CustomNonbondedForce``. The screening length (``dh``), relative
   dielectric constant (``er``), and a lambda factor (``lmd``) that scales
   protein–RNA cross-interactions are all user-configurable through the
   ``ffs`` dictionary.
 
3. **1-4 nonbonded interactions** — Lennard-Jones and electrostatic corrections
   for 1–4 bonded pairs via ``CustomBondForce``, sourced directly from the
   ``NonbondedForce`` exception list.
 
4. **Backbone hydrogen bonds** — N-H···O potential for protein backbone amide
   groups via ``CustomHbondForce``. Proline residues are excluded as donors
   because they lack a backbone NH.
 
5. **RNA base stacking** — centroid-distance–based potential between consecutive
   bases on the same chain via ``CustomCentroidBondForce``. Well depths and
   optimal distances are residue-pair-specific (see ``scales`` and ``r0s``
   dictionaries inside the function).
 
6. **RNA base pairing** — Watson-Crick A-U and C-G pairs and wobble G-U pairs,
   each implemented as a separate ``CustomHbondForce`` with distance- and
   angular-gating terms.
 
Force groups
------------
After all forces are added, each force is assigned a unique ``ForceGroup``
index (0, 1, 2, …) in the order they appear in ``system.getForces()``. This
allows per-force energy decomposition during analysis via
``Context.getState(getEnergy=True, groups={i})``.
 
Extensibility
-------------
Every builder accepts an optional ``modification`` callable that receives the
``System`` object after all built-in forces have been registered but before
``NonbondedForce`` is removed. Use this hook to inject positional restraints,
experimental potentials, or any other custom forces without modifying this
module directly.
 
Authors
-------
Shanlong Li, Xiping Gong, Yumeng Zhang, Xiaorong Liu, and Jianhan Chen
 
Dates
-----
Created  : Mar 09, 2024
Modified : Jun 11, 2026
 
Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.app``, ``openmm.unit``)
* `NumPy <https://numpy.org>`_ (``numpy``)
"""

from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np


# for HyRes_iConRNA System
def buildSystem(psf, system, DH_params, modification=None):
    """
    Build the HyRes protein and/or iConRNA force field into an OpenMM system.

    Replaces the standard OpenMM forces with custom force field terms tailored
    for the HyRes/iConRNA coarse-grained models. The following forces are
    constructed and added in order:

    1. **ReB angle force** — replaces ``HarmonicAngleForce`` with a Restricted
       Bending (ReB) potential for RNA and CA-CB angles.
    2. **Debye-Hückel electrostatics** — screened charge-charge interactions
       via ``CustomNonbondedForce`` with configurable screening length and
       dielectric constant.
    3. **1-4 nonbonded interactions** — short-range LJ and electrostatic terms
       for 1-4 bonded pairs via ``CustomBondForce``.
    4. **Backbone hydrogen bonds** — N-H···O hydrogen bond potential for protein
       backbone via ``CustomHbondForce`` (skipped for PRO residues).
    5. **RNA base stacking** — centroid-based stacking potential between
       consecutive bases via ``CustomCentroidBondForce``.
    6. **RNA base pairing** — A-U, C-G, and G-U Watson-Crick and wobble pair
       potentials via ``CustomHbondForce``.

    The original ``NonbondedForce`` and ``HarmonicAngleForce`` are removed after
    replacement. Each remaining force is assigned a unique force group index.

    Args:
        psf (CharmmPsfFile): Parsed PSF object containing topology and atom
                             information.
        system (System): OpenMM ``System`` object created from the PSF topology,
                         to which forces will be added.
        DH_params (dict): Debye-Hückel parameter dictionary. Required keys:

                    - ``'dh'`` (Quantity) — Debye-Hückel screening length in
                      length units (e.g. ``1.2*unit.nanometer``).
                    - ``'lmd'`` (float) — Lambda scaling factor for protein-RNA
                      charge-charge interactions.
                    - ``'er'`` (float) — Relative dielectric constant.

        modification (callable, optional): A user-defined function that accepts
                                           the ``System`` object and applies
                                           additional force modifications before
                                           the function returns. Called after all
                                           built-in forces are added but before
                                           ``NonbondedForce`` is removed.
                                           Default is ``None``.

    Returns:
        System: The modified OpenMM ``System`` object with all HyRes/iConRNA
                force field terms applied.

    Raises:
        ValueError: If any of the required keys (``'dh'``, ``'lmd'``, ``'er'``)
                    are missing from ``DH_params``.

    Example:
        >>> from openmm.app import CharmmPsfFile
        >>> from openmm import System
        >>> from HyresBuilder import HyresFF
        >>> psf = CharmmPsfFile("conf.psf")
        >>> system = psf.createSystem(...)
        >>> DH_params = {'dh': 1.2*unit.nanometer, 'lmd': 0.0, 'er': 80.0}
        >>> system = HyresFF.buildSystem(psf, system, DH_params)

        >>> # With custom modification
        >>> def my_mod(system):
        ...     pass  # add extra forces here
        >>> system = HyresFF.buildSystem(psf, system, DH_params, modification=my_mod)
    """
    
    print('\n################# constructe HyRes and/or iConRNA force field ####################')
    top = psf.topology
    
    # 1. Validate force field parameters
    required_params = ['dh', 'lmd', 'er']
    missing_params = [param for param in required_params if param not in DH_params]
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
        bead1, bead2, bead3 = atoms[ang[0]], atoms[ang[1]], atoms[ang[2]]
        backbones = ['N', 'H', 'C', 'O']
        if bead1 not in backbones and bead2 not in backbones and bead3 not in backbones:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 2])
        else:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 0])
    system.addForce(ReB)

    # 4. Add Debye-Hückel electrostatic interactions using CustomNonbondedForce
    dh = DH_params['dh']
    er = DH_params['er']
    lmd = DH_params['lmd']
    # add custom nonbondedforce: CNBForce, here only charge-charge interactions
    formula = f"""138.935456/{er}*charge1*charge2/r*exp(-r/dh)*kpmg;
                  dh={dh.value_in_unit(unit.nanometer)}; kpmg=select(lb1+lb2, 1, {lmd});
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
        if ex[4] != 0.0:
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

#        # add G-U pair through CustomHbondForce
#        eps_GU = eps_base*scales['G-U']
#        r_gu = 0.35*unit.nanometer
#
#        if num_G != 0 and num_U !=0:
#            formula = f"""eps_GU*(5.0*(r_gu/r)^12-6.0*(r_gu/r)^10)*step_phi; r=distance(a1,d1);
#                        step_phi=step(cos_phi)*cos_phi; cos_phi=-cos(phi)^5; phi=angle(d1,a1,a2);
#                        eps_GU={eps_GU.value_in_unit(unit.kilojoule_per_mole)};
#                        r_gu={r_gu.value_in_unit(unit.nanometer)}
#                      """
#            pairGU = CustomHbondForce(formula)
#            pairGU.setName('GUpairForce')
#            pairGU.setNonbondedMethod(nbforce.getNonbondedMethod())
#            pairGU.setCutoffDistance(0.65*unit.nanometers)
#
#            for idx in range(len(g_c)):
#                pairGU.addAcceptor(g_c[idx], g_b[idx], -1)
#            for idx in range(len(u_b)):
#                pairGU.addDonor(u_b[idx], -1, -1)
#            system.addForce(pairGU)
#            print(pairGU.getNumAcceptors(), pairGU.getNumDonors(), 'GU')

    # further modification defined in running scripts
    if callable(modification):
        modification(system)
        
    # 8. Delete the NonbondedForce and HarmonicAngleForce
    for idx in sorted([nbforce_index, hmangle_index], reverse=True):
        system.removeForce(idx)

    # 9. set unique ForceGroup id for each force
    forces = system.getForces()
    for i, force in enumerate(forces):
        force.setForceGroup(i)

    return system


# original iConRNA model (PNAS, 2025)
def iConRNASystem(psf, system, DH_params, modification=None):
    """
    Build the original iConRNA coarse-grained RNA force field (PNAS 2025).
 
    This is the legacy iConRNA-only builder. It applies a Restricted Bending
    angle potential, Debye–Hückel screened electrostatics, and RNA-specific
    base-stacking and A-U/C-G base-pairing forces. Unlike :func:`buildSystem`,
    this function does not include protein backbone hydrogen bonds, 1-4
    corrections, or G-U wobble pairing, and uses slightly different stacking
    potentials and exclusion radii consistent with the original publication.
 
    The ``NonbondedForce`` and ``HarmonicAngleForce`` generated by
    ``CharmmPsfFile.createSystem`` are removed before returning.
 
    Parameters
    ----------
    psf : openmm.app.CharmmPsfFile
        Parsed PSF object providing topology and per-atom metadata (name,
        residue name, residue index).
    system : openmm.System
        OpenMM ``System`` created from the PSF topology.  Modified in place.
    DH_params : dict
        Debye-Hückel parameter dictionary.  Required keys:
 
        ``'dh'`` : openmm.unit.Quantity
            Debye–Hückel screening length (e.g. ``1.2 * unit.nanometer``).
        ``'lmd'`` : float
            Lambda scaling factor for protein–RNA charge–charge interactions.
        ``'er'`` : float
            Relative dielectric constant (e.g. ``80.0``).
        ``'eps_base'`` : openmm.unit.Quantity
            Base energy scale (energy units) used to compute absolute stacking
            and pairing well depths via the ``scales`` dictionary.
 
    modification : callable, optional
        User-defined function ``modification(system)`` called after all built-in
        forces are registered but before ``NonbondedForce`` is removed.
        Default is ``None``.
 
    Returns
    -------
    openmm.System
        The modified ``System`` with iConRNA force terms applied and the
        original ``NonbondedForce`` / ``HarmonicAngleForce`` removed.
 
    Notes
    -----
    * The ReB angle potential here uses the form ``kt*(theta-theta0)^2 /
      sin(theta)^2`` (without the 0.5 prefactor and without the per-angle
      ``kReB`` exponent switch used in :func:`buildSystem`); every angle
      receives the same sine denominator regardless of atom type.
    * Exclusions for the Debye–Hückel force are created to bond-separation
      depth 2 (not 3 as in :func:`buildSystem`).
    * The stacking potential uses a (10, 6) power law rather than the (12, 10)
      form used in :func:`buildSystem`, and a single global ``r0 = 0.34 nm``
      is applied to all base pairs.
    * Base-pairing potentials also use (10, 6) powers and the angular gate
      ``-2*cos(phi)^3`` rather than ``-cos(phi)^5``.
    * G-U wobble pairing is not included in this builder.
    * No unique ``ForceGroup`` indices are assigned; force groups retain their
      OpenMM defaults.
 
    Examples
    --------
    >>> from openmm.app import CharmmPsfFile, CharmmParameterSet
    >>> from openmm import unit
    >>> from HyresBuilder import HyresFF
    >>> psf    = CharmmPsfFile("rna.psf")
    >>> params = CharmmParameterSet("rna.prm")
    >>> system = psf.createSystem(params)
    >>> DH_params    = {
    ...     'dh'       : 1.2 * unit.nanometer,
    ...     'lmd'      : 1.0,
    ...     'er'       : 20.0,
    ...     'eps_base' : 3.0 * unit.kilocalorie_per_mole,
    ... }
    >>> system = HyresFF.iConRNASystem(psf, system, DH_params)
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
    dh = DH_params['dh']
    lmd = DH_params['lmd']
    er = DH_params['er']
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
    
    CNBForce.createExclusionsFromBonds(bondlist, 3)
    system.addForce(CNBForce)

    # Add 1-4 nonbonded interaction through custombondforece
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
        if ex[4] != 0.0:
            Force14.addBond(ex[0], ex[1], [ex[2], ex[3], ex[4]])
    system.addForce(Force14)
    
    print('\n# add base stacking force')
    # base stakcing and paring
    # define relative strength of base pairing and stacking
    eps_base = 2.05*unit.kilocalorie_per_mole
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
    
    # further modification defined in running scripts
    if callable(modification):
        modification(system)
        
    # delete the NonbondedForce and HarmonicAngleForce
    for idx in sorted([nbforce_index, hmangle_index], reverse=True):
        system.removeForce(idx)

    # 9. set unique ForceGroup id for each force
    forces = system.getForces()
    for i, force in enumerate(forces):
        force.setForceGroup(i)

    return system


###### for rG4s System with A-U/G-C/G-G pairs ######
def rG4sSystem(psf, system, DH_params, modification=None):
    """
    Build the rG4 force field for RNA G-quadruplex (rG4) simulations.
 
    Constructs a mixed RNA force field that includes standard HyRes/iConRNA
    interactions (ReB angles, Debye–Hückel electrostatics, 1-4 corrections,
    base stacking, A-U and C-G Watson-Crick pairing) and additionally
    models G-G Hoogsteen pairing required for G-quadruplex tetrad formation.
    G-G pairing is implemented via two separate ``CustomHbondForce`` objects,
    each capturing a distinct donor-acceptor geometry of the Hoogsteen contact.
 
    Parameters
    ----------
    psf : openmm.app.CharmmPsfFile
        Parsed PSF object providing topology, atom names, and residue indices.
    system : openmm.System
        OpenMM ``System`` created from the PSF topology.  Modified in place.
    DH_params : dict
        Debye-Hückel parameter dictionary.  Required keys:
 
        ``'dh'`` : openmm.unit.Quantity
            Debye–Hückel screening length (e.g. ``1.2 * unit.nanometer``).
        ``'lmd'`` : float
            Lambda scaling factor for protein–RNA charge–charge interactions.
        ``'er'`` : float
            Relative dielectric constant (e.g. ``80.0``).
        ``'ion_type'`` : openmm.unit.Quantity
            G-G pairing well depth in energy units.  Represents the ion-dependent
            strength of the G-tetrad Hoogsteen contact.
 
    modification : callable, optional
        User-defined function ``modification(system)`` called after all built-in
        forces are registered but before ``NonbondedForce`` is removed.
        Default is ``None``.
 
    Returns
    -------
    openmm.System
        The modified ``System`` with rG4 force terms applied and the original
        ``NonbondedForce`` / ``HarmonicAngleForce`` removed.
 
    Notes
    -----
    * G-U wobble pairing is **not** included in this builder (unlike
      :func:`buildSystem`).
    * Two ``CustomHbondForce`` objects handle G-G pairing:
 
      - *GGpairForce1* governs the NB–ND contact (optimal distance 0.40 nm)
        with a combined dihedral gate (``psi``) and angular gate (``phi``).
      - *GGpairForce2* governs the NC–NC contact (optimal distance 0.42 nm)
        with mirrored dihedral and angular gating.
 
    * Self-exclusions (``addExclusion(i, i)``) and nearest-neighbour exclusions
      (sequential residue index) are applied to both G-G forces to suppress
      intra-strand contacts.
    * The base energy scale is hard-coded as ``3.2 kcal/mol``; the ``scales``
      and ``r0s`` dictionaries mirror :func:`buildSystem` except that the G-U
      entry is absent and ``UU`` uses ``0.4`` rather than ``0.1``.
    * No unique ``ForceGroup`` indices are assigned; groups retain OpenMM
      defaults.
 
    Raises
    ------
    KeyError
        If any required key is absent from *DH_params* (``'dh'``, ``'lmd'``, ``'er'``,
        ``'ion_type'``).
 
    Examples
    --------
    >>> from openmm.app import CharmmPsfFile, CharmmParameterSet
    >>> from openmm import unit
    >>> from HyresBuilder import HyresFF
    >>> psf    = CharmmPsfFile("rg4.psf")
    >>> params = CharmmParameterSet("rg4.prm")
    >>> system = psf.createSystem(params)
    >>> DH_params    = {
    ...     'dh'      : 1.2  * unit.nanometer,
    ...     'lmd'     : 1.0,
    ...     'er'      : 80.0,
    ...     'ion_type': 5.0  * unit.kilocalorie_per_mole,   # K+ ion strength
    ... }
    >>> system = HyresFF.rG4sSystem(psf, system, DH_params)
    """

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
    # 3 Replace HarmonicAngle with Restricted Bending (ReB) potential
    ReB = CustomAngleForce("0.5*kt*(theta-theta0)^2/(sin(theta)^kReB);")
    ReB.setName('ReBAngleForce')
    ReB.addPerAngleParameter("theta0")
    ReB.addPerAngleParameter("kt")
    ReB.addPerAngleParameter("kReB")
    for angle_idx in range(hmangle.getNumAngles()):
        ang = hmangle.getAngleParameters(angle_idx)
        bead1, bead2, bead3 = atoms[ang[0]], atoms[ang[1]], atoms[ang[2]]
        backbones = ['N', 'H', 'C', 'O']
        if bead1 not in backbones and bead2 not in backbones and bead3 not in backbones:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 2])
        else:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 0])
    system.addForce(ReB)

    # 4. Add Debye-Hückel electrostatic interactions using CustomNonbondedForce
    dh = DH_params['dh']
    er = DH_params['er']
    lmd = DH_params['lmd']
    # add custom nonbondedforce: CNBForce, here only charge-charge interactions
    formula = f"""138.935456/ker*charge1*charge2/r*exp(-r/dh)*kpmg; dh={dh.value_in_unit(unit.nanometer)};
                  ker=select(la1+la2, {er}, 20.0); kpmg=select(lb1+lb2, 1, {lmd});
              """
    CNBForce = CustomNonbondedForce(formula)
    CNBForce.setName("DH_ElecForce")
    CNBForce.setNonbondedMethod(nbforce.getNonbondedMethod())
    CNBForce.setUseSwitchingFunction(use=True)
    CNBForce.setCutoffDistance(1.8*unit.nanometers)
    CNBForce.setSwitchingDistance(1.6*unit.nanometers)
    CNBForce.addPerParticleParameter('charge')
    CNBForce.addPerParticleParameter('la')
    CNBForce.addPerParticleParameter('lb')
    
    for idx in range(nbforce.getNumParticles()):
        particle = nbforce.getParticleParameters(idx)
        if atoms[idx] == 'P':
            la, lb = 0, 1
        elif atoms[idx] in {'MG', 'CAL'}:
            la, lb = 0, -1
        else:
            la, lb = 2, 2
        perP = [particle[0], la, lb]
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
        if ex[4] != 0.0:
            Force14.addBond(ex[0], ex[1], [ex[2], ex[3], ex[4]])
    system.addForce(Force14)

    print('\n# add RNA base stacking force')
    # base stakcing and paring
    # define relative strength of base pairing and stacking
    eps_base = 3.2*unit.kilocalorie_per_mole
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
    eps_GG = DH_params['GG']*unit.kilocalorie_per_mole
    r_gg1 = 0.40*unit.nanometer     # for NB-ND
    r_gg2 = 0.42*unit.nanometer     # for NC-NC
    
    if num_G != 0:
        formula = f"""eps_GG*(5.0*(r_gg1/r1)^12-6.0*(r_gg1/r1)^10)*step1*step2;
                  r1=distance(d1,a1); step1=step(psi3)*psi3; psi3=cos(psi)^1; psi=dihedral(a2,a1,d1,d2);
                  step2=step(phi3)*phi3; phi3=-cos(phi)^1; phi=angle(a3,a1,d3);
                  eps_GG={eps_GG.value_in_unit(unit.kilojoule_per_mole)};
                  r_gg1={r_gg1.value_in_unit(unit.nanometer)};
                  """
        pairGG = CustomHbondForce(formula)
        pairGG.setName('GGpairForce1')
        pairGG.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairGG.setCutoffDistance(0.65*unit.nanometer)
        for idx in range(len(g_a)):
            pairGG.addDonor(g_c[idx], g_b[idx], g_d[idx])
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
                  r_gg2={r_gg2.value_in_unit(unit.nanometer)};
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

    # further modification defined in running scripts
    if callable(modification):
        modification(system)

    # delete the NonbondedForce and HarmonicAngleForce
    for idx in sorted([nbforce_index, hmangle_index], reverse=True):
        system.removeForce(idx)
    return system


# for HyRes_iConRNA System with Mg-RNA interactions
def buildMgSystem(psf, system, DH_params, modification=None):
    """
    similar to buildSystem, but specifically for Mg/Ca-RNA interactions.
    """
    
    print('\n################# constructe HyRes and/or iConRNA force field ####################')
    top = psf.topology
    
    # 1. Validate force field parameters
    required_params = ['dh', 'lmd', 'er']
    missing_params = [param for param in required_params if param not in DH_params]
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
        bead1, bead2, bead3 = atoms[ang[0]], atoms[ang[1]], atoms[ang[2]]
        backbones = ['N', 'H', 'C', 'O']
        if bead1 not in backbones and bead2 not in backbones and bead3 not in backbones:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 2])
        else:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 0])
    system.addForce(ReB)

    # 4. Add Debye-Hückel electrostatic interactions using CustomNonbondedForce
    dh = DH_params['dh']
    er = DH_params['er']
    lmd = DH_params['lmd']
    # add custom nonbondedforce: CNBForce, here only charge-charge interactions
    formula = f"""138.935456/ker*charge1*charge2/r*exp(-r/dh)*kpmg; dh={dh.value_in_unit(unit.nanometer)};
                  ker=select(la1+la2, {er}, 20.0); kpmg=select(lb1+lb2, 1, {lmd});
              """
    CNBForce = CustomNonbondedForce(formula)
    CNBForce.setName("DH_ElecForce")
    CNBForce.setNonbondedMethod(nbforce.getNonbondedMethod())
    CNBForce.setUseSwitchingFunction(use=True)
    CNBForce.setCutoffDistance(1.8*unit.nanometers)
    CNBForce.setSwitchingDistance(1.6*unit.nanometers)
    CNBForce.addPerParticleParameter('charge')
    CNBForce.addPerParticleParameter('la')
    CNBForce.addPerParticleParameter('lb')
    
    for idx in range(nbforce.getNumParticles()):
        particle = nbforce.getParticleParameters(idx)
        if atoms[idx] == 'P':
            la, lb = 0, 1
        elif atoms[idx] in {'P1', 'P2', 'P3'}:
            la, lb = 2, 1
        elif atoms[idx] in {'MG', 'CAL'}:
            la, lb = 0, -1
        else:
            la, lb = 2, 2
        perP = [particle[0], la, lb]
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
        if ex[4] != 0.0:
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

#        # add G-U pair through CustomHbondForce
#        eps_GU = eps_base*scales['G-U']
#        r_gu = 0.35*unit.nanometer
#
#        if num_G != 0 and num_U !=0:
#            formula = f"""eps_GU*(5.0*(r_gu/r)^12-6.0*(r_gu/r)^10)*step_phi; r=distance(a1,d1);
#                        step_phi=step(cos_phi)*cos_phi; cos_phi=-cos(phi)^5; phi=angle(d1,a1,a2);
#                        eps_GU={eps_GU.value_in_unit(unit.kilojoule_per_mole)};
#                        r_gu={r_gu.value_in_unit(unit.nanometer)}
#                      """
#            pairGU = CustomHbondForce(formula)
#            pairGU.setName('GUpairForce')
#            pairGU.setNonbondedMethod(nbforce.getNonbondedMethod())
#            pairGU.setCutoffDistance(0.65*unit.nanometers)
#
#            for idx in range(len(g_c)):
#                pairGU.addAcceptor(g_c[idx], g_b[idx], -1)
#            for idx in range(len(u_b)):
#                pairGU.addDonor(u_b[idx], -1, -1)
#            system.addForce(pairGU)
#            print(pairGU.getNumAcceptors(), pairGU.getNumDonors(), 'GU')

    # further modification defined in running scripts
    if callable(modification):
        modification(system)
        
    # 8. Delete the NonbondedForce and HarmonicAngleForce
    for idx in sorted([nbforce_index, hmangle_index], reverse=True):
        system.removeForce(idx)

    # 9. set unique ForceGroup id for each force
    forces = system.getForces()
    for i, force in enumerate(forces):
        force.setForceGroup(i)

    return system


# for HyRes_iConDNA System
def iConDNASystem(psf, system, DH_params, modification=None):
    """
    Build the HyRes protein and/or iConDNA force field into an OpenMM system.

    Replaces the standard OpenMM forces with custom force field terms tailored
    for the HyRes/iConDNA coarse-grained models. The following forces are
    constructed and added in order:

    1. **ReB angle force** — replaces ``HarmonicAngleForce`` with a Restricted
       Bending (ReB) potential for RNA and CA-CB angles.
    2. **Debye-Hückel electrostatics** — screened charge-charge interactions
       via ``CustomNonbondedForce`` with configurable screening length and
       dielectric constant.
    3. **1-4 nonbonded interactions** — short-range LJ and electrostatic terms
       for 1-4 bonded pairs via ``CustomBondForce``.
    4. **Backbone hydrogen bonds** — N-H···O hydrogen bond potential for protein
       backbone via ``CustomHbondForce`` (skipped for PRO residues).
    5. **RNA base stacking** — centroid-based stacking potential between
       consecutive bases via ``CustomCentroidBondForce``.
    6. **RNA base pairing** — A-U, C-G, and G-U Watson-Crick and wobble pair
       potentials via ``CustomHbondForce``.

    The original ``NonbondedForce`` and ``HarmonicAngleForce`` are removed after
    replacement. Each remaining force is assigned a unique force group index.

    Args:
        psf (CharmmPsfFile): Parsed PSF object containing topology and atom
                             information.
        system (System): OpenMM ``System`` object created from the PSF topology,
                         to which forces will be added.
        DH_params (dict): Debye-Hückel parameter dictionary. Required keys:

                    - ``'dh'`` (Quantity) — Debye-Hückel screening length in
                      length units (e.g. ``1.2*unit.nanometer``).
                    - ``'lmd'`` (float) — Lambda scaling factor for protein-RNA
                      charge-charge interactions.
                    - ``'er'`` (float) — Relative dielectric constant.

        modification (callable, optional): A user-defined function that accepts
                                           the ``System`` object and applies
                                           additional force modifications before
                                           the function returns. Called after all
                                           built-in forces are added but before
                                           ``NonbondedForce`` is removed.
                                           Default is ``None``.

    Returns:
        System: The modified OpenMM ``System`` object with all HyRes/iConDNA
                force field terms applied.

    Raises:
        ValueError: If any of the required keys (``'dh'``, ``'lmd'``, ``'er'``)
                    are missing from ``DH_params``.

    Example:
        >>> from openmm.app import CharmmPsfFile
        >>> from openmm import System
        >>> from HyresBuilder import HyresFF
        >>> psf = CharmmPsfFile("conf.psf")
        >>> system = psf.createSystem(...)
        >>> DH_params = {'dh': 1.2*unit.nanometer, 'lmd': 0.0, 'er': 80.0}
        >>> system = HyresFF.buildSystem(psf, system, DH_params)

        >>> # With custom modification
        >>> def my_mod(system):
        ...     pass  # add extra forces here
        >>> system = HyresFF.buildSystem(psf, system, DH_params, modification=my_mod)
    """
    
    print('\n################# constructe HyRes and/or iConDNA force field ####################')
    top = psf.topology
    
    # 1. Validate force field parameters
    required_params = ['dh', 'lmd', 'er']
    missing_params = [param for param in required_params if param not in DH_params]
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
        bead1, bead2, bead3 = atoms[ang[0]], atoms[ang[1]], atoms[ang[2]]
        backbones = ['N', 'H', 'C', 'O']
        if bead1 not in backbones and bead2 not in backbones and bead3 not in backbones:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 2])
        else:
            ReB.addAngle(ang[0], ang[1], ang[2], [ang[3], ang[4], 0])
    system.addForce(ReB)

    # 4. Add Debye-Hückel electrostatic interactions using CustomNonbondedForce
    dh = DH_params['dh']
    er = DH_params['er']
    lmd = DH_params['lmd']
    # add custom nonbondedforce: CNBForce, here only charge-charge interactions
    formula = f"""138.935456/{er}*charge1*charge2/r*exp(-r/dh)*kpmg;
                  dh={dh.value_in_unit(unit.nanometer)}; kpmg=select(lb1+lb2, 1, {lmd});
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
        if ex[4] != 0.0:
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
    # define relative strength of base pairing and stacking
    eps_base = 4.0*unit.kilocalorie_per_mole
    scales = {'DAA':1.0, 'DAG':1.0, 'DAC':0.8, 'DAT':1.0, 'DGA':1.0, 'DGG':1.0, 'DGC':1.0, 'DGT':1.0,
            'DCA':0.4, 'DCG':0.4, 'DCC':0.4, 'DCT':0.4, 'DTA':0.4, 'DTG':0.4, 'DTC':0.6, 'DTT':0.4,
            'DA-DT':1.628*0.64/1.7, 'DC-DG':1.628*0.64}
    # H bond: 'DA-T':0.9375*0.665, 'DC-G':1.628*0.665; AT is increased to 1:1.5
    # optimal stacking distance
    r0s = {'DAA':0.35, 'DAG':0.38, 'DAC':0.36, 'DAT':0.34,'DGA':0.35, 'DGG':0.37,  'DGC':0.37, 'DGT':0.35,
           'DCA':0.37, 'DCG':0.38, 'DCC':0.36, 'DCT':0.36, 'DTC':0.36, 'DTT':0.35, 'DTA':0.36, 'DTG':0.37
           }
    r1s = {'DAA':0.44, 'DAG':0.41, 'DAC':0.36, 'DAT':0.34,'DGA':0.44, 'DGG':0.42,  'DGC':0.34, 'DGT':0.33,
           'DCA':0.40, 'DCG':0.40, 'DCC':0.36, 'DCT':0.37, 'DTC':0.36, 'DTT':0.36, 'DTA':0.41, 'DTG':0.41
           }
    # get all the groups of bases
    grps = []
    for atom in psf.topology.atoms():
        if atom.name == "NA":
            if atom.residue.name in ['DA', 'DG']:
                grps.append([atom.residue.name, atom.residue.chain.id, [atom.index, atom.index+1, atom.index+2, atom.index+3]])
            elif atom.residue.name in ['DC', 'DT']:
                grps.append([atom.residue.name, atom.residue.chain.id, [atom.index, atom.index+1, atom.index+2]])
    # base stacking
    fstack = CustomCentroidBondForce(2, 'eps_stack*(5*(r0/r)^12-6.0*(r0/r)^10); r=distance(g1, g2);')
    fstack.setName('StackingForce')
    fstack.addPerBondParameter('eps_stack')
    fstack.addPerBondParameter('r0')
    # add all group
    for grp in grps:
        if grp[0] in ['DA', 'DG']:
            fstack.addGroup(grp[2], [0.25, 0.25, 0.25, 0.25]) #center
            fstack.addGroup(grp[2], [0, 1/2, 1/2, 0]) #BC
            fstack.addGroup(grp[2], [0, 0, 0, 1]) # D
            fstack.addGroup(grp[2], [0, 1, 0, 0]) #B
            fstack.addGroup(grp[2], [1/2, 0, 0, 1/2]) #AD
        if grp[0] in ['DC', 'DT']:
            fstack.addGroup(grp[2],[1/3,1/3,1/3]) #COM
            fstack.addGroup(grp[2], [0.5, 0.5, 0]) #AB
            fstack.addGroup(grp[2], [0, 1, 0]) #B
            fstack.addGroup(grp[2], [0, 0, 1]) #C
            fstack.addGroup(grp[2], [0, 0.5, 0.5]) #BC
    # get the stacking pairs
    sps = []
    for i in range(0,len(grps)-1,1): # [0, n-1)
        pij = grps[i][0] + grps[i+1][0].lstrip('D')
        # the index starts at 1; +0 means the first option; +1 second option; +2 third option; ...
        if pij in ['DAA', 'DGG', 'DAG', 'DGA']: # D to center + B to center
            sps.append([[i*5+2, (i+1)*5+0], scales[pij]*(eps_base/2), r0s[pij]*unit.nanometers])
            sps.append([[i*5+3, (i+1)*5+0], scales[pij]*(eps_base/2), r1s[pij]*unit.nanometers])
        if pij in ['DCC', 'DTT', 'DCT', 'DTC']: # C to center + center to B
            sps.append([[i*5+3, (i+1)*5+0], scales[pij]*(eps_base/2), r0s[pij]*unit.nanometers])
            sps.append([[i*5+0, (i+1)*5+2], scales[pij]*(eps_base/2), r1s[pij]*unit.nanometers])
        if pij in ['DCG', 'DTG', 'DCA', 'DTA']: # C to BC + C to AD
            sps.append([[i*5+3, (i+1)*5+1], scales[pij]*(eps_base/2), r0s[pij]*unit.nanometers])
            sps.append([[i*5+3, (i+1)*5+4], scales[pij]*(eps_base/2), r1s[pij]*unit.nanometers])
        if pij in ['DGC', 'DGT', 'DAC', 'DAT']: # center to AB + D to C
            sps.append([[i*5+0, (i+1)*5+1], scales[pij]*(eps_base/2), r0s[pij]*unit.nanometers])
            sps.append([[i*5+2, (i+1)*5+3], scales[pij]*(eps_base/2), r1s[pij]*unit.nanometers])

    for i in range(0, len(sps), 2): #[0, n*2)
        if grps[(i//2)][1] == grps[(i + 2)//2][1]:
            fstack.addBond(sps[i][0], [sps[i][1], sps[i][2]])
            fstack.addBond(sps[i+1][0], [sps[i+1][1], sps[i+1][2]])

    print('    add ', fstack.getNumBonds(), 'stacking pairs')
    system.addForce(fstack)
    
    # base pairing
    print('\n# add base pair force')
    a_b, a_c, a_d = [], [], []
    g_b, g_c, g_d = [], [], []
    c_a, c_b, c_c, t_a, t_b, t_c = [], [], [], [], [], []
    a_p, g_p, c_p, t_p = [], [], [], []
    num_A, num_G, num_C, num_T = 0, 0, 0, 0
    for atom in psf.topology.atoms():
        if atom.residue.name == 'DA':
            num_A += 1
            if atom.name == 'NC':
                a_c.append(int(atom.index))
            elif atom.name == 'NB':
                a_b.append(int(atom.index))
            elif atom.name == 'ND':
                a_d.append(int(atom.index))
            elif atom.name == 'P':
                a_p.append(int(atom.index))
        elif atom.residue.name == 'DG':
            num_G += 1
            if atom.name == 'NC':
                g_c.append(int(atom.index))
            elif atom.name == 'NB':
                g_b.append(int(atom.index))
            elif atom.name == 'ND':
                g_d.append(int(atom.index))
            elif atom.name == 'P':
                g_p.append(int(atom.index))
        elif atom.residue.name == 'DT':
            num_T += 1
            if atom.name == 'NA':
                t_a.append(int(atom.index))
            elif atom.name == 'NB':
                t_b.append(int(atom.index))
            elif atom.name == 'NC':
                t_c.append(int(atom.index))
            elif atom.name == 'P':
                t_p.append(int(atom.index))
        elif atom.residue.name == 'DC':
            num_C += 1
            if atom.name == 'NA':
                c_a.append(int(atom.index))
            elif atom.name == 'NB':
                c_b.append(int(atom.index))
            elif atom.name == 'NC':
                c_c.append(int(atom.index))
            elif atom.name == 'P':
                c_p.append(int(atom.index))
    # add A-T pair through CustomHbondForce
    eps_DAT = eps_base*scales['DA-DT']
    r_Dat = 0.327*unit.nanometer #A3-T2
    r_Dat2 = 0.43*unit.nanometer #A4-T3
    
    if num_A != 0 and num_T != 0:
        formula = f"""eps_DAT*(5.0*(r_Dat/r)^12-6.0*(r_Dat/r)^10 + 5*(r_Dat2/r2)^12-6.0*(r_Dat2/r2)^10)*step(cos5)*cos5;
                  r=distance(a1,d1); r2=distance(a3,d2); cos5=-cos(phi)^3; phi=min(min(abs(phi1),abs(phi2)),abs(phi3));
                  phi1 = dihedral(a3,a1,d2,d1); phi2 = dihedral(d1,d2,a1,a2); phi3 = dihedral(d3,d2,a1,a3);
                  eps_DAT={eps_DAT.value_in_unit(unit.kilojoule_per_mole)};
                  r_Dat={r_Dat.value_in_unit(unit.nanometer)}; r_Dat2={r_Dat2.value_in_unit(unit.nanometer)}
                  """
        pairDAT = CustomHbondForce(formula)
        pairDAT.setName('ATpairForce')
        pairDAT.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairDAT.setCutoffDistance(0.7*unit.nanometer) #switch this distance cut off for 

        for idx in range(len(a_c)):
            # pairAT.addAcceptor(a_d[idx], a_b[idx], a_c[idx])
            pairDAT.addAcceptor(a_c[idx], a_b[idx], a_d[idx]) #from iConRNA
        for idx in range(len(t_b)):
            # pairAT.addDonor(t_c[idx], t_b[idx], -1)
            pairDAT.addDonor(t_b[idx], t_c[idx], t_a[idx]) #from iConRNA
        system.addForce(pairDAT)
        print(pairDAT.getNumAcceptors(), pairDAT.getNumDonors(), 'D_AT')
        
    # add C-G pair through CustomHbondForce
    eps_DCG = eps_base*scales['DC-DG']
    r_Dcg = 0.33*unit.nanometer #G3C2 distance
    r_Dcg2 = 0.40*unit.nanometer #G4C3
    
    if num_C != 0 and num_G != 0:
        formula = f"""eps_DCG*(5.0*(r_Dcg/r)^12-6.0*(r_Dcg/r)^10 + 5*(r_Dcg2/r2)^12-6.0*(r_Dcg2/r2)^10)*step(cos5)*cos5;
                  r=distance(a1,d1); r2=distance(a3,d2); cos5=-cos(phi)^3; phi=min(min(abs(phi1),abs(phi2)),abs(phi3));
                  phi1 = dihedral(a3,a1,d2,d1); phi2 = dihedral(d1,d2,a1,a2); phi3 = dihedral(d3,d2,a1,a3);
                  eps_DCG={eps_DCG.value_in_unit(unit.kilojoule_per_mole)};
                  r_Dcg={r_Dcg.value_in_unit(unit.nanometer)}; r_Dcg2={r_Dcg2.value_in_unit(unit.nanometer)}
                  """
        pairDCG = CustomHbondForce(formula)
        pairDCG.setName('CGpairForce')
        pairDCG.setNonbondedMethod(nbforce.getNonbondedMethod())
        pairDCG.setCutoffDistance(0.7*unit.nanometer)
        for idx in range(len(g_c)):
            pairDCG.addAcceptor(g_c[idx], g_b[idx], g_d[idx])
        for idx in range(len(c_b)):
            pairDCG.addDonor(c_b[idx], c_c[idx], c_a[idx])
        system.addForce(pairDCG)
        print(pairDCG.getNumAcceptors(), pairDCG.getNumDonors(), 'D_CG')
 
    # further modification defined in running scripts
    if callable(modification):
        modification(system)
        
    # 8. Delete the NonbondedForce and HarmonicAngleForce
    for idx in sorted([nbforce_index, hmangle_index], reverse=True):
        system.removeForce(idx)

    # 9. set unique ForceGroup id for each force
    forces = system.getForces()
    for i, force in enumerate(forces):
        force.setForceGroup(i)

    return system

