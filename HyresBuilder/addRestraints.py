"""
Additional restraints for HyRes+OpenMM simulations.

This module extends a standard OpenMM simulation setup with harmonic positional
and center-of-mass (COM) restraint utilities. It is designed for use in
coarse-grained and all-atom protein simulations, with specialized support for
amyloid fibril systems where structured core residues must be selectively
frozen or tethered during equilibration and production runs.

Restraint types provided
------------------------
* **Positional restraints** — harmonic springs on CA atoms selected by
  residue number (:func:`posres_CAs`), or on arbitrary atoms selected by
  atom index (:func:`posres`). These use the plain (non-periodic)
  displacement from the reference position.
* **Amyloid-aware restraints** — automatically identify the structured fibril
  core from a MODELLER ``alignment.ali`` file and apply either positional
  restraints (:func:`posre_amyloid`) or full-atom freezing via zero mass
  (:func:`freeze_amyloid`).
* **COM restraints** — restrain the center of mass of a group of atoms in all
  three dimensions (:func:`comres_xyz`) or within a user-specified 2D plane,
  leaving the remaining axis free (:func:`comres_2d`). For periodic systems
  the COM offset uses the minimum-image convention of the System's default
  (orthorhombic) box.

All forces are added directly to the provided ``openmm.System`` object in
place, so these functions must be called before the ``Context``/``Simulation``
is created (with :func:`HyresBuilder.utils.setup`, use its ``modification``
hook). Energies have no factor 1/2: U = k * |Δr|².

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.app``, ``openmm.unit``)

Date:   Oct 23, 2025
Author: Shanlong Li
"""

from openmm.unit import *
from openmm.app import *
from openmm import *


def posres_CAs(system, pdb, residue_list=None, limited_range=None, kpos=200.0):
    """
    Apply positional restraints to CA atoms selected by residue index.

    Adds a harmonic ``CustomExternalForce``
    ``kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)`` (no factor 1/2, no periodic
    wrapping) that restrains each selected CA atom to its reference position
    in the PDB file. Residues are matched by ``int(atom.residue.id)`` (the
    residue number in the file, not the 0-based residue index). Restraints can
    be further filtered to an atom index range using ``limited_range``.

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        residue_list (list of int): Residue numbers to restrain. Despite the
                                    ``None`` default it is required; ``None``
                                    raises TypeError.
        limited_range (tuple of int, optional): ``(min_index, max_index)`` atom
                                                index range. Only CA atoms with
                                                ``min_index < index < max_index``
                                                (both bounds exclusive) are
                                                restrained. If ``None``, the
                                                range is ``(0, n_atoms)``, which
                                                also excludes atom 0.
        kpos (float, optional): Spring constant (global parameter ``kpos``) in
                                kJ/mol/nm². Default is 200.0.

    Returns:
        None. Modifies ``system`` in place by adding a ``Ca_position_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.posres_CAs(system, pdb, residue_list=[1, 2, 3, 4, 5], kpos=300.0)
        >>> addRestraints.posres_CAs(system, pdb, residue_list=[1, 2, 3],
        ...                         limited_range=(0, 500))
    """
    
    # add restraint
    ### set position restraints CA atoms
    restraint = CustomExternalForce('kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.setName("Ca_position_restraint")
    restraint.addGlobalParameter('kpos', kpos*kilojoule_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    
    atoms = list(pdb.topology.atoms()) 
    if limited_range:
        atom_min, atom_max = limited_range[0], limited_range[1]
    else:
        atom_min, atom_max = 0, len(atoms)
    for atom in atoms:
        resid, name = int(atom.residue.id), atom.name
        if resid in residue_list and name == 'CA':
            if atom.index > atom_min and atom.index < atom_max:
                restraint.addParticle(atom.index, pdb.positions[atom.index])
    system.addForce(restraint)

def posres(system, pdb, grp, kpos=200.0):
    """
    Apply positional restraints to atoms selected by atom index.

    Adds a harmonic ``CustomExternalForce``
    ``kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)`` (no factor 1/2, no periodic
    wrapping) that restrains each atom in ``grp`` (any atom, not only CA) to
    its reference position in the PDB file. Use this function when you already
    know the exact atom indices to restrain, rather than selecting by residue
    ID. It shares the global parameter name ``kpos`` with :func:`posres_CAs`,
    so both may be used in one System only with the same ``kpos``.

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        grp (list of int): Atom indices to restrain.
        kpos (float, optional): Spring constant (global parameter ``kpos``) in
                                kJ/mol/nm². Default is 200.0.

    Returns:
        None. Modifies ``system`` in place by adding a ``Ca_position_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.posres(system, pdb, grp=[0, 5, 10, 15], kpos=300.0)
    """

    # add restraint
    ### set position restraints CA atoms
    restraint = CustomExternalForce('kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.setName("Ca_position_restraint")
    restraint.addGlobalParameter('kpos', kpos*kilojoule_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    
    for idx in grp:
        restraint.addParticle(idx, pdb.positions[idx])
    system.addForce(restraint)

def posre_amyloid(system, pdb, alignment_file):
    """
    Apply CA positional restraints to the structured core of an amyloid fibril.

    Reads an alignment file (``alignment.ali``) to identify which residues are
    present in the fibril core (non-``'-'`` positions in the alignment sequence).
    The sequence lines are those between the first two ``>`` headers, skipping
    the header and the following structure line and dropping the last line
    before the second header; one line per chain, compared position by
    position with the chain's residues. The CA atoms of the core residues are
    restrained by atom index with :func:`posres` (default spring constant
    200 kJ/mol/nm²).

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        alignment_file (str): Path to the ``alignment.ali`` file generated during
                              fibril model building. The number of sequence blocks
                              must match the number of chains in the PDB.

    Returns:
        None. Modifies ``system`` in place by adding positional restraints via
        :func:`posres`.

    Raises:
        SystemExit: If the number of chains in ``pdb`` does not match the number
                    of sequence lines in ``alignment_file`` (a message is
                    printed and ``exit(1)`` is called).

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("fibril.pdb")
        >>> addRestraints.posre_amyloid(system, pdb, "alignment.ali")
    """

    with open(alignment_file, 'r') as f:
        lines = f.readlines()
    blocks = [index for index, line in enumerate(lines) if line.startswith('>')]
    b1, b2 = blocks[:2]
    #nchains = b2 - b1 -3        # count the number of chains based on the lines in alignment.ali
    missings = lines[b1+2:b2-1]     # get all the sequences for each chain, "-" for missing residue

    chains = list(pdb.topology.chains())
    if len(chains) != len(missings):
        print(f"Unconsistent chain number! Found {len(chains)} in pdb file, but {len(missings)} in alignment.ali")
        exit(1)
    # get the CA index for un-missing residues
    grp = []
    for chain, sequence in zip(chains, missings):
        residues = list(chain.residues())
        nres = len(residues)
        seq = sequence[:nres]
        ca = None
        for res, s in zip(residues, seq):
            if s != '-':
                for atom in res.atoms():
                    if atom.name == 'CA':
                        ca = atom.index
                        grp.append(ca)
     
    # add position restraint, grp holds atom indices
    posres(system, pdb, grp)

def freeze_amyloid(system, pdb, alignment_file):
    """
    Freeze the structured core of an amyloid fibril by setting atom masses to zero.

    Reads an ``alignment.ali`` file to identify residues present in the fibril
    core (non-``'-'`` positions in the alignment sequence; parsed as in
    :func:`posre_amyloid`). Sets the mass of every atom in those residues to
    zero; OpenMM integrators do not move zero-mass particles, so the core is
    fixed in place. This is cheaper than positional restraints and guarantees
    no drift of the fibril core. Zero-mass particles must not take part in
    constraints, so build the System without constraints on these atoms.

    Args:
        system (System): OpenMM ``System`` object whose particle masses will be
                         modified.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology information
                       (e.g. ``PDBFile('fibril.pdb')``).
        alignment_file (str): Path to the ``alignment.ali`` file generated during
                              fibril model building. The number of sequence blocks
                              must match the number of chains in the PDB.

    Returns:
        None. Modifies ``system`` in place by setting particle masses to zero.

    Raises:
        SystemExit: If the number of chains in ``pdb`` does not match the number
                    of sequence lines in ``alignment_file`` (a message is
                    printed and ``exit(1)`` is called).

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("fibril.pdb")
        >>> addRestraints.freeze_amyloid(system, pdb, "alignment.ali")
    """

    with open(alignment_file, 'r') as f:
        lines = f.readlines()
    blocks = [index for index, line in enumerate(lines) if line.startswith('>')]
    b1, b2 = blocks[:2]
    #nchains = b2 - b1 -3        # count the number of chains based on the lines in alignment.ali
    missings = lines[b1+2:b2-1]     # get all the sequences for each chain, "-" for missing residue

    chains = list(pdb.topology.chains())
    if len(chains) != len(missings):
        print(f"Unconsistent chain number! Found {len(chains)} in pdb file, but {len(missings)} in alignment.ali")
        exit(1)
    # get the CA index for un-missing residues
    grp = []
    for chain, sequence in zip(chains, missings):
        residues = list(chain.residues())
        nres = len(residues)
        seq = sequence[:nres]
        ca = None
        for res, s in zip(residues, seq):
            if s != '-':
                for atom in res.atoms():
                    system.setParticleMass(atom.index, 0.0*unit.amu)

#def freeze_residues(system, pdb, residue_list):


def _com_offset_expr(system):
    """Energy-expression definitions of the COM offset (dx, dy, dz) from (cx, cy, cz).

    The group centre of a periodic CustomCentroidBondForce can come back in any
    periodic image (the CUDA platform wraps positions into the box), so the offset
    is taken with the minimum-image convention of the orthorhombic box; periodicdistance()
    is not available in CustomCentroidBondForce. For a non-periodic system it is the
    plain difference.

    Only the diagonal of ``system.getDefaultPeriodicBoxVectors()`` is used, and the
    box lengths are written into the expression as constants when it is built, so
    later box changes (e.g. a barostat under NPT) are not followed.

    Returns:
        tuple: ``(definitions, periodic)`` -- the expression fragment defining
        ``dx; dy; dz`` from ``x1, y1, z1`` and ``cx, cy, cz``, and whether the
        System is periodic (to pass to ``setUsesPeriodicBoundaryConditions``).
    """
    if not system.usesPeriodicBoundaryConditions():
        return "dx=x1-cx; dy=y1-cy; dz=z1-cz", False
    a, b, c = system.getDefaultPeriodicBoxVectors()
    lx, ly, lz = a[0].value_in_unit(unit.nanometer), b[1].value_in_unit(unit.nanometer), c[2].value_in_unit(unit.nanometer)
    return (f"dx=x1-cx-{lx}*floor((x1-cx)/{lx}+0.5); "
            f"dy=y1-cy-{ly}*floor((y1-cy)/{ly}+0.5); "
            f"dz=z1-cz-{lz}*floor((z1-cz)/{lz}+0.5)"), True


def comres_xyz(system, pdb, groups):
    """
    Apply a center-of-mass (COM) restraint in all three (x, y, z) dimensions.

    Computes the mass-weighted COM of the selected atoms from the PDB
    positions, using the System's current particle masses (so call it before
    anything that changes masses, e.g. rigid bodies or freezing), prints it,
    and adds a ``CustomCentroidBondForce`` named ``COM_xyz_restraint``:

        U = kxyz * (dx^2 + dy^2 + dz^2)      (no factor 1/2)

    where (dx, dy, dz) is the offset of the group's COM from the reference
    COM (per-bond parameters ``cx, cy, cz``) and ``kxyz`` is a global
    parameter fixed at 500 kJ/mol/nm². For a periodic System the offset uses
    the minimum-image convention of the default orthorhombic box (see
    :func:`_com_offset_expr`) and the force is flagged periodic; otherwise the
    plain difference is used. Use this to prevent drift of a molecular group
    in all directions.

    Caveats:
        * The restrained group should be smaller than half the box in each
          direction.
        * The box length is fixed into the expression when the force is
          created, so it is not suited to NPT (changing box).
        * Calling it more than once in one System is fine (all calls share
          ``kxyz`` = 500).

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        groups (list of int): Atom indices whose COM will be restrained.

    Returns:
        None. Modifies ``system`` in place by adding a ``COM_xyz_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.comres_xyz(system, pdb, groups=[0, 1, 2, 3, 4])
    """

    # add COM restraint
    cds = pdb.getPositions(asNumpy=True)
    def com(grp):
        cx_sum, cy_sum, cz_sum, m_sum = quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons*unit.nanometers), quantity.Quantity(0.0, unit.daltons)
        for i in grp:
            cx_sum += system.getParticleMass(i)*cds[i,0]
            cy_sum += system.getParticleMass(i)*cds[i,1]
            cz_sum += system.getParticleMass(i)*cds[i,2]
            m_sum += system.getParticleMass(i)
        cx, cy, cz = cx_sum/m_sum, cy_sum/m_sum, cz_sum/m_sum
        return [cx, cy, cz]

    print('com of selected residues:', com(groups))
    offset, periodic = _com_offset_expr(system)
    com_xyz = CustomCentroidBondForce(1, f'kxyz*(dx^2 + dy^2 + dz^2); {offset}')
    com_xyz.setName("COM_xyz_restraint")
    com_xyz.addGroup(groups)
    com_xyz.addGlobalParameter('kxyz', 500.0*kilojoule_per_mole/(unit.nanometer**2))
    com_xyz.addPerBondParameter('cx')
    com_xyz.addPerBondParameter('cy')
    com_xyz.addPerBondParameter('cz')
    com_xyz.setUsesPeriodicBoundaryConditions(periodic)
    com_xyz.addBond([0], com(groups))
    system.addForce(com_xyz)


def comres_2d(system, dimension, groups, pdb, k=1000):
    """
    Add a 2D harmonic COM restraint to the system, leaving one axis free.

    The reference COM is the mass-weighted COM of ``groups`` from the PDB
    positions, using the System's current particle masses. The added
    ``CustomCentroidBondForce`` (named ``'COM_2d_restraint'``) is, e.g. for
    ``'xy'``:

        U = k2d * (dx^2 + dy^2)      (no factor 1/2)

    with the offsets (dx, dy, dz) from the reference COM computed as in
    :func:`comres_xyz` (minimum image of the default orthorhombic box for a
    periodic System, plain difference otherwise). ``k2d`` and the reference
    ``cx, cy, cz`` are per-bond parameters (the free axis's reference is 0),
    so several calls in one System, with different groups or ``k``, do not
    interfere. To change ``k2d`` later, use ``setBondParameters`` and
    ``updateParametersInContext`` on the returned force.

    Caveats:
        * The restrained group should be smaller than half the box in each
          direction.
        * The box length is fixed into the expression when the force is
          created, so it is not suited to NPT (changing box).

    Parameters
    ----------
    system    : openmm.System — modified in place
    dimension : str — one of 'xy', 'xz', or 'yz' (case-insensitive)
                The two axes that are restrained; the third is left free.
    groups    : list[int] — atom indices forming the restrained group
    pdb       : openmm.app.PDBFile — PDB file used to compute the initial COM
    k         : float — force constant in kJ/mol/nm² (default 1000)

    Returns
    -------
    openmm.CustomCentroidBondForce — the force, already added to ``system``.

    Raises
    ------
    ValueError if ``dimension`` is not 'xy', 'xz' or 'yz'.
    """
    dimension = dimension.lower()
    if dimension not in ('xy', 'xz', 'yz'):
        raise ValueError(f"dimension must be 'xy', 'xz', or 'yz', got '{dimension}'")

    # ── Read positions from PDB ────────────────────────────────────────────────
    cds = pdb.getPositions(asNumpy=True)

    # ── Compute mass-weighted COM ──────────────────────────────────────────────
    cx_sum = quantity.Quantity(0.0, unit.dalton * unit.nanometer)
    cy_sum = quantity.Quantity(0.0, unit.dalton * unit.nanometer)
    cz_sum = quantity.Quantity(0.0, unit.dalton * unit.nanometer)
    m_sum  = quantity.Quantity(0.0, unit.dalton)

    for i in groups:
        m = system.getParticleMass(i)
        cx_sum += m * cds[i, 0]
        cy_sum += m * cds[i, 1]
        cz_sum += m * cds[i, 2]
        m_sum  += m
    cx, cy, cz = cx_sum / m_sum, cy_sum / m_sum, cz_sum / m_sum

    # ── Build energy expression ────────────────────────────────────────────────
    expr_map = {
        'yz': ('k2d * (dy^2 + dz^2)', {'cx': 0.0*unit.nanometer, 'cy': cy, 'cz': cz}),
        'xz': ('k2d * (dx^2 + dz^2)', {'cx': cx, 'cy': 0.0*unit.nanometer, 'cz': cz}),
        'xy': ('k2d * (dx^2 + dy^2)', {'cx': cx, 'cy': cy, 'cz': 0.0*unit.nanometer}),
    }
    expr, values = expr_map[dimension]

    # ── Build force ────────────────────────────────────────────────────────────
    offset, periodic = _com_offset_expr(system)
    force_2d = CustomCentroidBondForce(1, f'{expr}; {offset}')
    force_2d.setName("COM_2d_restraint")
    force_2d.addGroup(groups)
    force_2d.addPerBondParameter('k2d')
    for name in ('cx', 'cy', 'cz'):
        force_2d.addPerBondParameter(name)
    force_2d.setUsesPeriodicBoundaryConditions(periodic)
    force_2d.addBond([0], [k*kilojoule_per_mole/(unit.nanometer**2), values['cx'], values['cy'], values['cz']])

    system.addForce(force_2d)
    return force_2d