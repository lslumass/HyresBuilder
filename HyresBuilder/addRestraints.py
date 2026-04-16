"""
Additional restraints for HyRes+OpenMM simulations.

This module extends a standard OpenMM simulation setup with harmonic positional
and center-of-mass (COM) restraint utilities. It is designed for use in
coarse-grained and all-atom protein simulations, with specialized support for
amyloid fibril systems where structured core residues must be selectively
frozen or tethered during equilibration and production runs.

Restraint types provided
------------------------
* **CA positional restraints** — harmonic springs on Cα atoms, selectable by
  residue index (:func:`posres_CA`) or by explicit atom index (:func:`posres_CAs`).
* **Amyloid-aware restraints** — automatically identify the structured fibril
  core from a MODELLER ``alignment.ali`` file and apply either positional
  restraints (:func:`posre_amyloid`) or full-atom freezing via zero mass
  (:func:`freeze_amyloid`).
* **COM restraints** — restrain the center of mass of a group of atoms in all
  three dimensions (:func:`comres_xyz`) or within a user-specified 2D plane,
  leaving the remaining axis free (:func:`comres_2d`).

All forces are added directly to the provided ``openmm.System`` object in place
and are compatible with periodic boundary conditions where applicable.

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.app``, ``openmm.unit``)

Date:   Oct 23, 2025
Author: Shanlong Li
"""

from openmm.unit import *
from openmm.app import *
from openmm import *


def posres_CA(system, pdb, residue_list=None, limited_range=None):
    """
    Apply positional restraints to CA atoms selected by residue index.

    Adds a harmonic ``CustomExternalForce`` that restrains each selected CA atom
    to its reference position in the PDB file with a spring constant of
    200 kJ/mol/nm². Restraints can be further filtered to a specific atom index
    range using ``limited_range``.

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        residue_list (list of int, optional): Residue indices to restrain.
                                              If ``None``, no atoms are restrained.
        limited_range (tuple of int, optional): ``(min_index, max_index)`` atom
                                                index range. Only CA atoms within
                                                this range are restrained.
                                                If ``None``, all CA atoms in
                                                ``residue_list`` are restrained.

    Returns:
        None. Modifies ``system`` in place by adding a ``Ca_position_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.posres_CA(system, pdb, residue_list=[1, 2, 3, 4, 5])
        >>> addRestraints.posres_CA(system, pdb, residue_list=[1, 2, 3],
        ...                         limited_range=(0, 500))
    """
    
    # add restraint
    ### set position restraints CA atoms
    restraint = CustomExternalForce('kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.setName("Ca_position_restraint")
    restraint.addGlobalParameter('kpos', 200.0*kilojoule_per_mole/unit.nanometer)
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

def posres_CAs(system, pdb, grp):
    """
    Apply positional restraints to CA atoms selected by atom index.

    Adds a harmonic ``CustomExternalForce`` that restrains each atom in ``grp``
    to its reference position in the PDB file with a spring constant of
    200 kJ/mol/nm². Use this function when you already know the exact atom
    indices to restrain, rather than selecting by residue ID.

    Args:
        system (System): OpenMM ``System`` object to which the restraint force
                         will be added.
        pdb (PDBFile): OpenMM ``PDBFile`` object providing topology and reference
                       positions (e.g. ``PDBFile('conf.pdb')``).
        grp (list of int): Atom indices to restrain.

    Returns:
        None. Modifies ``system`` in place by adding a ``Ca_position_restraint``
        force.

    Example:
        >>> from openmm.app import PDBFile
        >>> from HyresBuilder import addRestraints
        >>> pdb = PDBFile("conf.pdb")
        >>> addRestraints.posres_CAs(system, pdb, grp=[0, 5, 10, 15])
    """

    # add restraint
    ### set position restraints CA atoms
    restraint = CustomExternalForce('kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.setName("Ca_position_restraint")
    restraint.addGlobalParameter('kpos', 200.0*kilojoule_per_mole/unit.nanometer)
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
    Restraints are applied to CA atoms of those residues across all chains using
    :func:`posres_CAs` with a spring constant of 200 kJ/mol/nm².

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
        :func:`posres_CAs`.

    Raises:
        SystemExit: If the number of chains in ``pdb`` does not match the number
                    of sequence blocks in ``alignment_file``.

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
     
    # add position restraint
    posres_CAs(system, pdb, grp)

def freeze_amyloid(system, pdb, alignment_file):
    """
    Freeze the structured core of an amyloid fibril by setting atom masses to zero.

    Reads an ``alignment.ali`` file to identify residues present in the fibril
    core (non-``'-'`` positions in the alignment sequence). Sets the mass of
    every atom in those residues to zero, effectively freezing them during
    simulation. This is cheaper than positional restraints and guarantees no
    drift of the fibril core.

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
                    of sequence blocks in ``alignment_file``.

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


def comres_xyz(system, pdb, groups):
    """
    Apply a center-of-mass (COM) restraint in all three (x, y, z) dimensions.

    Computes the mass-weighted COM of the selected atoms from their reference
    positions and adds a ``CustomCentroidBondForce`` that penalizes deviation
    from that reference COM with a spring constant of 500 kJ/mol/nm². Periodic
    boundary conditions are enabled. Use this to prevent drift of a molecular
    group in all directions.

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
    com_xyz = CustomCentroidBondForce(1, 'kxyz*((x1 - cx)^2 + (y1 - cy)^2 + (z1 - cz)^2);')
    com_xyz.setName("COM_xyz_restraint")
    com_xyz.addGroup(groups)
    com_xyz.addGlobalParameter('kxyz', 500.0*kilojoule_per_mole/(unit.nanometer**2))
    com_xyz.addPerBondParameter('cx')
    com_xyz.addPerBondParameter('cy')
    com_xyz.addPerBondParameter('cz')
    com_xyz.setUsesPeriodicBoundaryConditions(True)
    com_xyz.addBond([0], com(groups))
    system.addForce(com_xyz)


def comres_2d(system, dimension, groups, pdb, k=1000):
    """
    Add a 2D harmonic COM restraint to the system, leaving one axis free.

    Parameters
    ----------
    system    : openmm.System
    dimension : str — one of 'xy', 'xz', or 'yz'
                The two axes that are restrained; the third is left free.
    groups    : list[int] — atom indices forming the restrained group
    pdb       : openmm.app.PDBFile — PDB file used to compute the initial COM
    k         : force constant (default 1000 kJ/mol/nm²)

    Returns
    -------
    openmm.CustomCentroidBondForce added to system
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
        'yz': ('k2d * pointdistance(x1, y1, z1, x1, cy, cz)^2', {'cy': cy, 'cz': cz}),
        'xz': ('k2d * pointdistance(x1, y1, z1, cx, y1, cz)^2', {'cx': cx, 'cz': cz}),
        'xy': ('k2d * pointdistance(x1, y1, z1, cx, cy, z1)^2', {'cx': cx, 'cy': cy}),
    }
    expr, values = expr_map[dimension]

    # ── Build force ────────────────────────────────────────────────────────────
    force_2d = CustomCentroidBondForce(1, expr)
    force_2d.addGroup(groups)
    force_2d.addGlobalParameter('k2d', k*kilojoule_per_mole/(unit.nanometer**2))
    for name, value in values.items():
        force_2d.addGlobalParameter(name, value)
    force_2d.setUsesPeriodicBoundaryConditions(True)
    force_2d.addBond([0])

    system.addForce(force_2d)