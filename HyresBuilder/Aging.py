"""
Force definitions for amyloid aging simulations.

This module implements custom inter-chain interactions that model the progressive
structural consolidation of amyloid fibrils over time — a process referred to
here as "aging". As fibrils mature, backbone hydrogen bonds between adjacent
beta-strands become increasingly locked in an in-register arrangement, reducing
conformational dynamics and stiffening the fibril core.

The forces defined here are designed to be layered on top of an existing OpenMM
force field (e.g. HyRes, CHARMM36) without modifying its nonbonded terms,
and are controlled by a scalar ``age`` parameter that scales the interaction
strength to allow gradual or staged aging protocols.

Force types provided
--------------------
* **In-register backbone hydrogen bonds** — a ``CustomHbondForce`` that
  exclusively couples N-H···O donor–acceptor pairs sharing the same residue
  number across chains, enforcing parallel in-register β-sheet geometry
  (:func:`inRegisterHB`).

Conventions
-----------
* Residues are identified by ``int(atom.residue.id)``, i.e. the residue number
  from the PSF/PDB, not OpenMM's 0-based residue index.
* Proline residues are skipped entirely (no N, H or O is collected), as they
  lack a backbone NH group.
* All forces use nanometer / kilojoule-per-mole internal units; user-facing
  parameters (e.g. ``age``) are accepted in kcal/mol for convenience and
  converted internally.

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.app``, ``openmm.unit``)

Date:    Nov 07, 2025
Author:  Shanlong Li
"""


from openmm.unit import *
from openmm.app import *
from openmm import *


def inRegisterHB(system, top, res_list, age=1.0):
    """
    Add in-register backbone hydrogen bonds between identical residue positions
    across beta-sheet chains for amyloid aging simulations.

    Implements a ``CustomHbondForce`` (named ``'inRegister HBForce'``) that only
    forms N-H···O hydrogen bonds between donor and acceptor atoms that share the
    **same residue number** (``delta(di - ai) == 1``). This enforces in-register
    beta-sheet geometry, mimicking the structural locking that occurs during
    amyloid aging. Proline residues are skipped (no donor or acceptor), as they
    lack backbone NH groups. Self-pairs (the donor and acceptor of the same
    residue in the same chain) are excluded via ``addExclusion``.

    The hydrogen bond potential takes the form:

    .. code-block:: text

        epsilon * (5*(sigma/r)^12 - 6*(sigma/r)^10) * step(cos3) * cos3 * delta(di-ai)
        cos3 = -cos(phi)^3

    where ``r`` is the N···O distance, ``phi`` is the N-H···O angle (at H),
    ``sigma = 0.29 nm`` and ``epsilon = age`` kcal/mol (converted to kJ/mol).
    The minimum, ``-epsilon``, is at ``r = sigma`` with a linear N-H···O.

    Donors (N, H) and acceptors (O) are collected per residue: a selected
    residue is a donor if it has atoms named N and H, and an acceptor if it
    has an atom named O, so a residue missing one of them does not shift the
    others. Interactions are cut off at 0.45 nm, using periodic distances
    (``CutoffPeriodic``) when the system is periodic and ``CutoffNonPeriodic``
    otherwise. No force is added if there are no donors or no acceptors.

    Args:
        system (System): OpenMM ``System`` object to which the hydrogen bond
                         force will be added.
        top (Topology): OpenMM ``Topology`` object used to identify N, H, and O
                        atoms and their residue indices.
        res_list (list of int): Residue numbers (``residue.id``) to include in
                                the in-register hydrogen bond network.
        age (float, optional): Hydrogen bond well depth ``epsilon`` in kcal/mol.
                               A value of ``1.0`` corresponds to 1 kcal/mol per
                               bond. Default is ``1.0``.

    Returns:
        System: The same ``System`` object, modified in place, with the
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

    donors, acceptors = [], []      # (N, H, resid, residue index) and (O, resid, residue index)
    for residue in top.residues():
        resid = int(residue.id)
        if residue.name == 'PRO' or resid not in res_list:
            continue
        names = {atom.name: atom.index for atom in residue.atoms()}
        if 'N' in names and 'H' in names:
            donors.append((names['N'], names['H'], resid, residue.index))
        if 'O' in names:
            acceptors.append((names['O'], resid, residue.index))

    if donors and acceptors:
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
        if system.usesPeriodicBoundaryConditions():
            inRegHB.setNonbondedMethod(CustomHbondForce.CutoffPeriodic)
        else:
            inRegHB.setNonbondedMethod(CustomHbondForce.CutoffNonPeriodic)
        inRegHB.setCutoffDistance(0.45*unit.nanometers)
        inRegHB.addPerDonorParameter("di")  # resid for donor
        inRegHB.addPerAcceptorParameter("ai")  # resid for acceptor
        for n, h, resid, _ in donors:
            inRegHB.addDonor(n, h, -1, [resid])
        for o, resid, _ in acceptors:
            inRegHB.addAcceptor(o, -1, -1, [resid])
        # exclude the donor and acceptor of the same residue
        acceptor_of = {res_idx: k for k, (_, _, res_idx) in enumerate(acceptors)}
        for k, (_, _, _, res_idx) in enumerate(donors):
            if res_idx in acceptor_of:
                inRegHB.addExclusion(k, acceptor_of[res_idx])
        
        system.addForce(inRegHB)
    return system

