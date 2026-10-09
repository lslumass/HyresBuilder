"""
Rigid body implementation for OpenMM molecular simulations.

This module converts arbitrary groups of atoms in an OpenMM ``System`` into
rigid bodies, allowing entire protein domains, fibril chains, or other
structured regions to be treated as internally fixed objects during simulation.
Rigid bodies reduce the number of degrees of freedom, remove the need to
integrate fast internal motions, and allow larger integration time steps for
the constrained regions.

Implementation strategy
-----------------------
For each rigid body, four atoms are designated as "real" particles whose
positions are integrated normally. All remaining atoms in the body are
converted to ``OutOfPlaneSite`` virtual sites whose positions are recomputed
analytically at every step from the four real particles. Constraints are
added between every pair of real particles to maintain their pairwise
distances. Any pre-existing constraints that couple two atoms within the same
body are automatically removed to avoid conflicts. Reference geometry is taken
from the positions passed in (normally the PDB file), so each body is frozen
in that conformation.

The four real particles and their masses are chosen so that the total mass and
center of mass of the rigid body exactly match those of the original atom set
(among the valid choices, the most mass-balanced set is preferred, because a
very light real particle is unstable at large time steps). For bodies with
fewer than five atoms, all atoms are treated as real particles and keep their
masses. The moment of inertia will be similar but not identical to the
original distribution.

The three reference atoms used to define the virtual-site frame are selected
from the real particles by maximising the norm of their cross product, ensuring
a well-conditioned out-of-plane coordinate frame.

Interface levels
----------------
Five functions are provided at increasing levels of abstraction:

* :func:`createRigidBodies` — low-level; accepts pre-built lists of atom
  indices directly.
* :func:`resolveBodiesToIndices` — mid-level; resolves PSF segment IDs and
  residue ranges (as inclusive ``(start, end)`` tuples, a list of such range
  tuples, or explicit lists) into atom index lists.
* :func:`createRigidSegments` — high-level; accepts PSF and PDB file paths or
  pre-loaded objects plus a single residue-range pattern ("27-95") and a set
  of segment-ID ranges ("P001-P080"), and applies the same residue-based
  rigid-body definition to every matching segment. Intended for
  residue-structured molecules (proteins, nucleic acids, fibrils) described
  by a PSF. By default (``loop=True``) every listed residue is rigid; with
  ``loop=False`` coil residues, as assigned by DSSP (MDTraj), are left out so
  only helix and strand residues are rigid; with ``CA=True`` only the CA atoms
  go into each body.
* :func:`createRigidCA` — high-level; identical to
  :func:`createRigidSegments` except that only atoms with selected names are
  placed in each body ("CA" by default, e.g. "P" for nucleic acids). Produces
  coarse backbone rigid bodies while leaving side-chain and other atoms freely
  mobile.
* :func:`RigidSmallMols` — high-level; for systems with many (potentially
  thousands of) small-molecule segments in a PSF, e.g. 'M001', 'M002', ...
  Accepts a PSF/PDB and a single atom-index pattern ("10-15,20-25", 1-based
  within each segment) plus a set of segment-ID ranges
  ("M001-M010,M020-M030"), and applies the same rigid-body definition to
  every matching segment in one call.

All functions modify the ``System`` in place (particle masses, constraints and
virtual sites), so they must be called before the ``Context``/``Simulation``
is created. With :func:`HyresBuilder.utils.setup`, which creates the
``Simulation`` before returning, apply them -- together with any COM
restraint, e.g. from :mod:`HyresBuilder.addRestraints` -- inside the
``modification`` hook::

    from HyresBuilder import utils
    from HyresBuilder.Rigid import createRigidSegments

    def modification(system):
        createRigidSegments(system, 'conf.psf', 'conf.pdb',
                            residues="27-95", segments="P001-P080")

    system, sim = utils.setup(params, modification=modification)

Limitations
-----------
Virtual sites are massless and cannot participate in constraints with atoms
outside their own rigid body. If such cross-body constraints exist in the
input system they will cause an exception at ``Context`` creation time and
must be removed manually before calling these functions.

This applies in particular when only part of a residue is rigidified, as with
:func:`createRigidCA`: a bond constraint between a rigidified atom that becomes
a virtual site (e.g. CA) and a neighbouring atom left out of the body (e.g. HA)
is not removed automatically, because the two atoms do not both belong to the
same body. Build the ``System`` without constraints on the rigidified atoms
(e.g. avoid ``constraints=AllBonds``) or remove those constraints yourself.

Original authors:  Peter Eastman (Stanford University / Simbios)
Modified by:       Shanlong Li

This module is derived from Peter Eastman's ``rigid.py`` example for OpenMM
(MIT licence).

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.unit``)
* `NumPy <https://numpy.org>`_ (``numpy``, ``numpy.linalg``)
* `MDTraj <https://mdtraj.org>`_ (optional; only for ``createRigidSegments``
  with ``loop=False``, to run DSSP)
"""
__author__ = "Peter Eastman"
__version__ = "1.0"


import openmm as mm
import openmm.unit as unit
import numpy as np
import numpy.linalg as lin
from itertools import combinations


def _loadPsfPdb(psf=None, pdb=None):
    """Normalise `psf`/`pdb` arguments that may be file paths or already-loaded
    objects into (CharmmPsfFile, PDBFile) objects. Either argument can be
    omitted (pass None) if a function only needs one of the two; it is then
    returned as None. Non-string arguments are returned unchanged.
    """
    if psf is not None and isinstance(psf, str):
        from openmm.app import CharmmPsfFile
        psf = CharmmPsfFile(psf)
    if pdb is not None and isinstance(pdb, str):
        from openmm.app import PDBFile
        pdb = PDBFile(pdb)
    return psf, pdb


def resolveBodiesToIndices(psf, segment_bodies):
    """Resolve segment-based body definitions into lists of atom indices.

    Parameters
    ----------
    psf : str or openmm.app.CharmmPsfFile
        Either a path to a PSF file (str) or an already-loaded CharmmPsfFile object.
    segment_bodies : list of (segid, residue_list) tuples
        Each tuple defines one rigid body:
          - segid (str): the segment ID as it appears in the PSF file.
          - residue_list: either
              * a (start, end) tuple of inclusive author residue numbers,
              * a list of (start, end) range tuples, to combine several
                contiguous stretches into one body (e.g. to exclude a loop
                region from the middle of a longer range), or
              * an explicit list of author residue numbers [27, 28, 30, ...].

        Examples::

            # range tuple – residues 27 to 95 inclusive in segment P001
            ('P001', (27, 95))

            # explicit list – only these three residues in segment P042
            ('P042', [10, 11, 50])

            # list of range tuples – residues 10-100 excluding the 40-50 loop
            ('P001', [(10, 39), (51, 100)])

            # mix all styles across multiple bodies
            [('P001', (27, 95)), ('P002', (27, 95)), ('P003', [30, 31, 32])]

    Returns
    -------
    bodies : list of list of int
        Each inner list contains the atom indices that form one rigid body,
        ready to pass directly to createRigidBodies(). The System is not
        modified.

    Raises
    ------
    ValueError
        If a segment ID is not present in the PSF.

    Notes
    -----
    Residue numbers are matched against ``residue.id`` (the PSF resid);
    residues whose ID is not a plain integer (insertion codes such as '27A')
    are skipped. A body that matches no atoms is skipped with a warning.
    """
    import warnings
    psf, _ = _loadPsfPdb(psf=psf)

    # Build a fast lookup: segid -> chain object
    chain_map = {chain.id: chain for chain in psf.topology.chains()}

    bodies = []
    for segid, residue_list in segment_bodies:
        if segid not in chain_map:
            raise ValueError(f"Segment '{segid}' not found in PSF. "
                             f"Available segments: {list(chain_map.keys())}")

        # Normalise residue_list into a set of ints for O(1) membership test
        if isinstance(residue_list, tuple) and len(residue_list) == 2:
            # single (start, end) range tuple
            res_set = set(range(int(residue_list[0]), int(residue_list[1]) + 1))
        elif (isinstance(residue_list, list) and residue_list
              and all(isinstance(r, tuple) and len(r) == 2 for r in residue_list)):
            # list of (start, end) range tuples, e.g. [(10, 39), (51, 100)]
            # lets multiple contiguous stretches (e.g. excluding a loop) form one body
            res_set = set()
            for start, end in residue_list:
                res_set.update(range(int(start), int(end) + 1))
        else:
            res_set = {int(r) for r in residue_list}

        body_atoms = []
        for residue in chain_map[segid].residues():
            try:
                res_num = int(residue.id)
            except ValueError:
                continue   # skip insertion-code residues e.g. '27A'
            if res_num in res_set:
                for atom in residue.atoms():
                    body_atoms.append(atom.index)

        if body_atoms:
            bodies.append(body_atoms)
        else:
            warnings.warn(f"No atoms found for segment '{segid}' with "
                          f"residue_list={residue_list}. Body skipped.")

    return bodies


def _dsspCodes(psf, pdb):
    """Return the simplified DSSP code of every residue in the PSF topology,
    as a list indexed by residue index: 'H' (helix), 'E' (strand), 'C' (coil),
    or 'NA' (not a protein residue).

    DSSP is run once on the whole system with MDTraj, so hydrogen bonds between
    chains (e.g. the cross-beta sheets of a fibril) are taken into account.
    Coordinates come from `pdb`; the atom order must match the PSF.

    Raises ImportError if MDTraj is not installed and ValueError if the PSF and
    PDB atom counts differ.
    """
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError("loop=False needs MDTraj to assign secondary structure. "
                          "Install it with `pip install mdtraj` or "
                          "`conda install -c conda-forge mdtraj`.")

    if psf.topology.getNumAtoms() != len(pdb.positions):
        raise ValueError(f"PSF has {psf.topology.getNumAtoms()} atoms but PDB has "
                         f"{len(pdb.positions)}; they must describe the same system.")

    top = md.Topology.from_openmm(psf.topology)
    xyz = np.array(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float32)[None]
    traj = md.Trajectory(xyz, top)
    return list(md.compute_dssp(traj, simplified=True)[0])


def createRigidSegments(system, psf, pdb, residues, segments, loop=True, CA=False):
    """Apply the same residue-range rigid-body definition to many PSF segments
    at once, e.g. every chain of a repeated fibril or multimer.

    You give one residue pattern (which author residue numbers to include from
    *each* segment) and a set of segment names/ranges to apply it to; one
    rigid body is created per matching segment.

    By default (``loop=True``, ``CA=False``) every atom of every listed residue
    goes into the body. Two options narrow what goes into each body:

    * ``loop=False`` runs DSSP (via MDTraj) on the PDB coordinates and drops
      residues assigned as coil ('C'), so only helix ('H') and strand ('E')
      residues are rigid. The coil residues stay fully flexible.
    * ``CA=True`` keeps only the CA atom of each selected residue in the body;
      all other atoms (backbone N/C/O, side chains, hydrogens) move freely.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    psf : str or openmm.app.CharmmPsfFile
        Either a path to a PSF file (str) or an already-loaded CharmmPsfFile object.
    pdb : str or openmm.app.PDBFile
        Either a path to a PDB file (str) or an already-loaded PDBFile object.
        Positions are extracted from this file.
    residues : str
        Comma-separated author residue numbers/ranges to include from *each*
        matching segment, e.g. "1-10,20-80" or "27,28,30". Applied identically
        to every segment in `segments`.
    segments : str
        Comma-separated segment-ID ranges to apply this to, e.g.
        "P001-P080" (segments P001 through P080) or an explicit list like
        "P001,P005,P010". Numeric ranges keep the zero-padding width of the
        range's start ID.
    loop : bool, optional
        If True (default), every residue in `residues` is included. If False,
        residues that DSSP assigns as coil ('C') are removed from each body;
        helix ('H') and strand ('E') residues are kept. Residues DSSP cannot
        classify ('NA', e.g. non-protein residues) are kept. The assignment is
        made per residue from the PDB coordinates, so different segments can
        end up with different residue sets. Requires MDTraj.
    CA : bool, optional
        If False (default), all atoms of each selected residue go into the
        body. If True, only the CA atoms go into the body and every other atom
        stays free. Same as :func:`createRigidCA` with ``atomNames='CA'``.

    Returns
    -------
    numBodies : int
        The number of rigid bodies (matching segments) that were created.

    Raises
    ------
    ValueError
        If no rigid body could be built, or (from :func:`createRigidBodies`)
        if a body is degenerate.
    ImportError
        If ``loop=False`` and MDTraj is not installed.

    Notes
    -----
    Each body needs at least three non-collinear atoms. Segments with fewer
    than three selected atoms (e.g. an all-coil chain with ``loop=False``) are
    skipped with a warning; segment IDs not found in the PSF are also skipped
    with a warning. Residues with non-integer IDs (insertion codes) are
    ignored. The System is modified in place (see :func:`createRigidBodies`)
    and a one-line summary is printed.

    With ``loop=False`` every helix/coil and strand/coil junction becomes a
    boundary between a rigid residue and a free one; with ``CA=True`` every CA
    is such a boundary. A bond constraint across that boundary (e.g. CA-HA
    under ``constraints=HBonds``, or C-N under ``AllBonds``) will raise an
    exception at Context creation, since virtual sites cannot be constrained.
    Build the System without those constraints, or remove them first.

    DSSP is applied to the starting structure only, so the secondary structure
    present in `pdb` is locked in for the whole simulation.

    Example
    -------
    ::

        from HyresBuilder.Rigid import createRigidSegments

        # Residues 27-95 of every chain P001 through P080, as one rigid body each.
        createRigidSegments(system, 'conf.psf', 'conf.pdb',
                             residues="27-95", segments="P001-P080")

        # Same chains, but only helix/strand residues are rigid (coils free).
        createRigidSegments(system, 'conf.psf', 'conf.pdb',
                             residues="27-95", segments="P001-P080", loop=False)

        # Only the CA atoms of the helix/strand residues are rigid.
        createRigidSegments(system, 'conf.psf', 'conf.pdb',
                             residues="27-95", segments="P001-P080",
                             loop=False, CA=True)
    """
    import warnings

    psf, pdb = _loadPsfPdb(psf=psf, pdb=pdb)
    positions = pdb.positions

    segIDs = _parseSegmentRange(segments)
    resSet = set(_parseIndexRanges(residues))

    ssCodes = None if loop else _dsspCodes(psf, pdb)

    chain_map = {chain.id: chain for chain in psf.topology.chains()}

    bodies = []
    missing = []
    tooSmall = []
    nCoil = 0
    for segid in segIDs:
        if segid not in chain_map:
            missing.append(segid)
            continue

        body_atoms = []
        for residue in chain_map[segid].residues():
            try:
                res_num = int(residue.id)
            except ValueError:
                continue   # skip insertion-code residues e.g. '27A'
            if res_num not in resSet:
                continue
            if ssCodes is not None and ssCodes[residue.index] == 'C':
                nCoil += 1
                continue   # coil residue -> left out of the body
            for atom in residue.atoms():
                if not CA or atom.name == 'CA':
                    body_atoms.append(atom.index)

        if len(body_atoms) >= 3:
            bodies.append(body_atoms)
        else:
            tooSmall.append(segid)

    if missing:
        warnings.warn(f"{len(missing)} segment(s) not found in PSF and were skipped: "
                      f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    if tooSmall:
        warnings.warn(f"{len(tooSmall)} segment(s) skipped because fewer than three "
                      f"atoms were selected (residues '{residues}', loop={loop}, "
                      f"CA={CA}): {tooSmall[:10]}{'...' if len(tooSmall) > 10 else ''}")
    if not bodies:
        raise ValueError(f"No rigid bodies were built -- check `segments`, `residues`, "
                         f"loop={loop} and CA={CA}.")

    msg = (f"[Rigid] Resolved {len(bodies)} rigid bodies from {len(segIDs)} segment(s) "
           f"with residues '{residues}'")
    if not loop:
        msg += f", {nCoil} coil residue(s) excluded by DSSP"
    if CA:
        msg += ", CA atoms only"
    print(msg + ".")
    createRigidBodies(system, positions, bodies)
    return len(bodies)

def createRigidCA(system, psf, pdb, residues, segments, atomNames='CA'):
    """Apply the same residue-range rigid-body definition to many PSF segments
    at once, but keep only selected atom names (e.g. 'CA' or 'P') in each body.

    This is the same workflow as :func:`createRigidSegments`, except that
    instead of taking *every* atom of each matching residue, only atoms whose
    name matches `atomNames` are collected. The typical use is a coarse
    backbone rigid body: alpha carbons for proteins ('CA') or phosphorus atoms
    for nucleic acids ('P'). Side-chain and other atoms keep their masses and
    are integrated normally.

    One rigid body is created per matching segment.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    psf : str or openmm.app.CharmmPsfFile
        Either a path to a PSF file (str) or an already-loaded CharmmPsfFile object.
    pdb : str or openmm.app.PDBFile
        Either a path to a PDB file (str) or an already-loaded PDBFile object.
        Positions are extracted from this file.
    residues : str
        Comma-separated author residue numbers/ranges to include from *each*
        matching segment, e.g. "1-10,20-80" or "27,28,30". Applied identically
        to every segment in `segments`.
    segments : str
        Comma-separated segment-ID ranges to apply this to, e.g.
        "P001-P080" (segments P001 through P080) or an explicit list like
        "P001,P005,P010". Numeric ranges keep the zero-padding width of the
        range's start ID.
    atomNames : str or list of str, optional
        Atom name(s) to keep from each selected residue. Defaults to 'CA'.
        May be a comma-separated string ("P,C4'") or a list (['P', "C4'"]).
        Names are matched exactly against the topology atom names.

    Returns
    -------
    numBodies : int
        The number of rigid bodies (matching segments) that were created.

    Raises
    ------
    ValueError
        If `atomNames` is empty, if no rigid body could be built, or (from
        :func:`createRigidBodies`) if a body is degenerate.

    Notes
    -----
    Each body needs at least three non-collinear selected atoms, because
    createRigidBodies() uses three real particles to define the virtual-site
    frame. Segments yielding fewer than three matching atoms, and segment IDs
    not found in the PSF, are skipped with a warning.

    If the System was built with bond constraints (e.g. ``constraints=HBonds``
    or ``AllBonds``), a constraint between a selected atom that becomes a
    virtual site (e.g. CA) and a non-selected atom (e.g. HA) will raise an
    exception at Context creation, since virtual sites cannot be constrained.
    Build the System without such constraints on the rigidified atoms, or
    remove them before calling this function.

    Example
    -------
    ::

        from HyresBuilder.Rigid import createRigidCA

        # CA atoms of residues 27-95 in every chain P001-P080, one body each.
        createRigidCA(system, 'conf.psf', 'conf.pdb',
                      residues="27-95", segments="P001-P080")

        # Phosphorus backbone of residues 1-100 in nucleic-acid segments.
        createRigidCA(system, 'conf.psf', 'conf.pdb',
                      residues="1-100", segments="N001-N010", atomNames='P')
    """
    import warnings

    psf, pdb = _loadPsfPdb(psf=psf, pdb=pdb)
    positions = pdb.positions

    # Normalise the atom-name selection into a set of exact names.
    if isinstance(atomNames, str):
        nameSet = {n.strip() for n in atomNames.split(',') if n.strip()}
    else:
        nameSet = {str(n).strip() for n in atomNames if str(n).strip()}
    if not nameSet:
        raise ValueError("`atomNames` must contain at least one atom name, "
                         "e.g. atomNames='CA'.")

    segIDs = _parseSegmentRange(segments)
    resSet = set(_parseIndexRanges(residues))

    chain_map = {chain.id: chain for chain in psf.topology.chains()}

    bodies = []
    missing = []
    empty = []
    tooSmall = []
    for segid in segIDs:
        if segid not in chain_map:
            missing.append(segid)
            continue

        body_atoms = []
        for residue in chain_map[segid].residues():
            try:
                res_num = int(residue.id)
            except ValueError:
                continue   # skip insertion-code residues e.g. '27A'
            if res_num in resSet:
                for atom in residue.atoms():
                    if atom.name in nameSet:
                        body_atoms.append(atom.index)

        if len(body_atoms) >= 3:
            bodies.append(body_atoms)
        elif body_atoms:
            tooSmall.append(segid)
        else:
            empty.append(segid)

    if missing:
        warnings.warn(f"{len(missing)} segment(s) not found in PSF and were skipped: "
                      f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    if empty:
        warnings.warn(f"{len(empty)} segment(s) had no atoms named "
                      f"{sorted(nameSet)} in residues '{residues}' and were skipped: "
                      f"{empty[:10]}{'...' if len(empty) > 10 else ''}")
    if tooSmall:
        warnings.warn(f"{len(tooSmall)} segment(s) skipped because fewer than three "
                      f"atoms named {sorted(nameSet)} were selected (at least three "
                      f"non-collinear atoms are required): "
                      f"{tooSmall[:10]}{'...' if len(tooSmall) > 10 else ''}")
    if not bodies:
        raise ValueError(f"No rigid bodies were built -- check `segments`, `residues` "
                         f"and `atomNames`={sorted(nameSet)}.")

    print(f"[Rigid] Resolved {len(bodies)} rigid bodies from {len(segIDs)} segment(s) "
          f"with residues '{residues}', atom names {sorted(nameSet)}.")
    createRigidBodies(system, positions, bodies)
    return len(bodies)

def _parseIndexRanges(spec):
    """Parse a comma-separated string of integers/ranges, e.g. "10-15,20-25,30"
    into a sorted list of unique ints: [10, 11, 12, 13, 14, 15, 20, ..., 25, 30].

    Ranges are inclusive; empty tokens are ignored. Negative numbers are not
    supported (a '-' always marks a range); malformed tokens raise ValueError.
    """
    indices = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            start, end = token.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(token))
    return sorted(set(indices))


def _parseSegmentRange(spec):
    """Parse a comma-separated string of segment IDs/ranges, e.g.
    "M001-M010,M020-M030,X5" into an ordered list of segment ID strings.
    Ranges are expanded numerically, preserving the zero-padding width of the
    range's start ID (e.g. "M001-M010" -> M001, M002, ..., M010). The prefix
    is taken from the start ID only; the end ID contributes just its number.
    Duplicates are dropped, keeping first-occurrence order. Raises ValueError
    if a range end has no numeric suffix.
    """
    import re as _re
    segids = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            start_tok, end_tok = token.split('-')
            m_start = _re.match(r'^(.*?)(\d+)$', start_tok)
            m_end = _re.match(r'^(.*?)(\d+)$', end_tok)
            if not m_start or not m_end:
                raise ValueError(f"Could not parse segment range '{token}'. "
                                  f"Expected a numeric suffix, e.g. 'M001-M010'.")
            prefix, startNumStr = m_start.groups()
            _, endNumStr = m_end.groups()
            width = len(startNumStr)
            for n in range(int(startNumStr), int(endNumStr) + 1):
                segids.append(f"{prefix}{str(n).zfill(width)}")
        else:
            segids.append(token)
    seen = set()
    ordered = []
    for s in segids:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def RigidSmallMols(system, psf, pdb, atoms=None, segments=None):
    """Build rigid bodies for many small-molecule copies at once, using PSF
    segment names -- the typical setup for systems with hundreds or thousands
    of identical small-molecule segments (e.g. 'M001', 'M002', ... 'M999').

    You give one atom-index pattern (which atoms of *each* segment to make
    rigid) and a set of segment names/ranges to apply it to; one rigid body
    is created per matching segment.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    psf : str or openmm.app.CharmmPsfFile
        Either a path to a PSF file (str) or an already-loaded CharmmPsfFile object.
    pdb : str or openmm.app.PDBFile
        Either a path to a PDB file (str) or an already-loaded PDBFile object.
        Positions are extracted from this file.
    atoms : str, optional
        Comma-separated atom indices/ranges *within each segment*, e.g.
        "10-15,20-25" or "3,7,9". These follow standard PSF/molecule atom
        numbering (1-based, in the order atoms are listed for that segment),
        the same pattern applied to every matching segment. If omitted, every
        atom in each matching segment is used (the whole small molecule is
        made rigid).
    segments : str
        Required (despite the ``None`` default). Comma-separated segment-ID
        ranges to apply this to, e.g. "M001-M010,M020-M030" (segments M001
        through M010, and M020 through M030) or an explicit list like
        "M001,M005,M010". Numeric ranges keep the zero-padding width of the
        range's start ID.

    Returns
    -------
    numBodies : int
        The number of rigid bodies (matching segments) that were created.

    Raises
    ------
    ValueError
        If `segments` is None, if no rigid body could be built, or (from
        :func:`createRigidBodies`) if a body is degenerate.

    Notes
    -----
    Segments not found in the PSF, and segments with fewer atoms than the
    largest index in `atoms`, are skipped with a warning. Selections with
    fewer than two atoms are skipped silently. :func:`createRigidBodies`
    needs three non-collinear atoms per body, so a two-atom selection raises
    ValueError there.

    Example
    -------
    ::

        from HyresBuilder.Rigid import RigidSmallMols

        # Rigidify atoms 10-15 and 20-25 (PSF numbering, within each segment)
        # of every segment M001 through M010 and M020 through M030.
        RigidSmallMols(system, 'conf.psf', 'conf.pdb',
                        atoms="10-15,20-25", segments="M001-M010,M020-M030")

        # Make the whole molecule rigid for every M001-M500 segment.
        RigidSmallMols(system, 'conf.psf', 'conf.pdb', segments="M001-M500")
    """
    import warnings

    psf, pdb = _loadPsfPdb(psf=psf, pdb=pdb)
    positions = pdb.positions

    if segments is None:
        raise ValueError("`segments` must be provided, e.g. segments=\"M001-M010\".")
    segIDs = _parseSegmentRange(segments)

    localIndices = None
    if atoms is not None:
        # `atoms` follows PSF/molecule numbering, i.e. 1-based.
        localIndices = [i - 1 for i in _parseIndexRanges(atoms)]

    chain_map = {chain.id: chain for chain in psf.topology.chains()}

    bodies = []
    missing = []
    skipped = []
    for segid in segIDs:
        if segid not in chain_map:
            missing.append(segid)
            continue
        atomIndices = [atom.index for atom in chain_map[segid].atoms()]
        if localIndices is None:
            body = atomIndices
        else:
            if max(localIndices) >= len(atomIndices):
                skipped.append(segid)
                continue
            body = [atomIndices[i] for i in localIndices]
        if len(body) >= 2:
            bodies.append(body)

    if missing:
        warnings.warn(f"{len(missing)} segment(s) not found in PSF and were skipped: "
                      f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    if skipped:
        warnings.warn(f"{len(skipped)} segment(s) skipped because `atoms` indices "
                      f"exceeded their atom count: "
                      f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    if not bodies:
        raise ValueError("No rigid bodies were built -- check `segments` and `atoms`.")

    print(f"[Rigid] Applying rigid-body definition to {len(bodies)} segment(s) "
          f"matching '{segments}'.")
    createRigidBodies(system, positions, bodies)
    return len(bodies)

def createRigidBodies(system, positions, bodies):
    """Modify a System to turn specified sets of particles into rigid bodies.

    This is the low-level interface that operates directly on pre-built atom index lists.
    For a higher-level interface that accepts PSF segment IDs and residue ranges, use
    createRigidSegments() instead.

    For every rigid body, four particles are selected as "real" particles whose positions are integrated.
    Constraints are added between them to make them move as a rigid body.  All other particles in the body
    are then turned into virtual sites whose positions are computed based on the "real" particles.

    Because virtual sites are massless, the mass properties of the rigid bodies will be slightly different
    from the corresponding sets of particles in the original system.  The masses of the non-virtual particles
    are chosen to guarantee that the total mass and center of mass of each rigid body exactly match those of
    the original particles.  The moment of inertia will be similar to that of the original particles, but
    not identical.

    Care is needed when using constraints, since virtual particles cannot participate in constraints.  If the
    input system includes any constraints, this function will automatically remove ones that connect two
    particles in the same rigid body.  But if there is a constraint between a particle in a rigid body and
    another particle not in that body, it will likely lead to an exception when you try to create a context.

    For bodies with five or more particles, candidate sets of four real particles are tried (closest to
    the body's RMS radius first) and their masses are obtained by solving for the total mass and COM; sets
    with any non-positive mass are rejected.  The search stops at the first set in which every mass is at
    least 1/8 of the body's total mass, otherwise it returns the most balanced valid set found (searching
    at most 200000 combinations once a valid set exists).  Bodies with fewer than five particles keep all
    of them as real particles with their original masses.  Virtual sites are ``OutOfPlaneSite`` objects
    defined on the three real particles with the largest cross product, and their masses are set to zero.

    Parameters
    ----------
    system : openmm.System
        The System to modify (in place). Must be called before a Context is created from it.
    positions : list of Vec3 (Quantity with length units)
        The positions of all particles in the system, e.g. ``PDBFile.positions``. These define the
        rigid geometry.
    bodies : list of list of int
        Each element defines one rigid body as a list of atom indices. Every body needs at least three
        non-collinear particles.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no set of four real particles with positive masses exists for a body, or if no three
        non-collinear real particles can be found (this includes any body with fewer than three particles).

    Example
    -------
    ::

        from HyresBuilder.Rigid import createRigidBodies
        from openmm.app import PDBFile

        pdb = PDBFile('conf.pdb')
        # bodies is a list of lists of atom indices, one per rigid body
        bodies = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        createRigidBodies(system, pdb.positions, bodies)
    """
    # Remove any constraints involving particles in rigid bodies.
    
    for i in range(system.getNumConstraints()-1, -1, -1):
        p1, p2, distance = system.getConstraintParameters(i)
        if (any(p1 in body and p2 in body for body in bodies)):
            system.removeConstraint(i)
    
    # Loop over rigid bodies and process them.
    
    for particles in bodies:
        if len(particles) < 5:
            # All the particles will be "real" particles.
            
            realParticles = particles
            realParticleMasses = [system.getParticleMass(i) for i in particles]
        else:
            # Select four particles to use as the "real" particles.  All others will be virtual sites.
            
            pos = [positions[i] for i in particles]
            mass = [system.getParticleMass(i) for i in particles]
            cm = unit.sum([p*m for p, m in zip(pos, mass)])/unit.sum(mass)
            r = [p-cm for p in pos]
            avgR = unit.sqrt(unit.sum([unit.dot(x, x) for x in r])/len(particles))
            rank = sorted(range(len(particles)), key=lambda i: abs(unit.norm(r[i])-avgR))
            realParticles = None
            # Prefer a balanced set: every real particle should carry at least half of the mean
            # mass (total/8).  Any positive set reproduces the mass and COM exactly, but a very
            # light real particle (e.g. 80 Da out of 2700 Da) takes a large share of the forces
            # and torque on the body, and that blows up to NaN at large time steps (e.g. 8 fs).
            # If no balanced set turns up, use the most balanced positive set found.
            totalMass = unit.sum(mass).value_in_unit(unit.amu)
            minWeight = 0.125*totalMass
            maxTrials = 200000
            best = None
            for trial, p in enumerate(combinations(rank, 4)):
                if best is not None and trial >= maxTrials:
                    break
                # Select masses for the "real" particles.  If any is negative, reject this set of particles
                # and keep going.

                matrix = np.zeros((4, 4))
                for i in range(4):
                    particleR = r[p[i]].value_in_unit(unit.nanometers)
                    matrix[0][i] = particleR[0]
                    matrix[1][i] = particleR[1]
                    matrix[2][i] = particleR[2]
                    matrix[3][i] = 1.0
                rhs = np.array([0.0, 0.0, 0.0, totalMass])
                try:
                    weights = lin.solve(matrix, rhs)
                except lin.LinAlgError:
                    # The four chosen particles are coplanar (or collinear) --
                    # common for planar small molecules like aromatic rings --
                    # so the mass/COM system is rank-deficient rather than
                    # having no solution.  Fall back to a least-squares solve
                    # and only accept it if it still satisfies the mass/COM
                    # constraints to a tight tolerance.
                    weights, _, _, _ = lin.lstsq(matrix, rhs, rcond=None)
                    if not np.allclose(matrix.dot(weights), rhs, atol=1e-8):
                        continue
                if all(w > 0.0 for w in weights):
                    # We have a valid set of particles.  Keep it if it is the most balanced so far.

                    if best is None or min(weights) > min(best[1]):
                        best = (p, weights)
                    if min(weights) >= minWeight:
                        break
            if best is not None:
                realParticles = [particles[i] for i in best[0]]
                realParticleMasses = [float(w) for w in best[1]]*unit.amu
            if realParticles is None:
                raise ValueError(
                    f"Could not select four 'real' particles with positive mass "
                    f"weights for rigid body {particles}. This usually means the "
                    f"atom group is planar or otherwise degenerate in a way that "
                    f"no four-particle combination can reproduce its total mass "
                    f"and center of mass with positive weights. Consider "
                    f"double-checking the atom selection for this body.")
        
        # Set particle masses.
        
        for i, m in zip(realParticles, realParticleMasses):
            system.setParticleMass(i, m)
        
        # Add constraints between the real particles.
        
        for p1, p2 in combinations(realParticles, 2):
            distance = unit.norm(positions[p1]-positions[p2])
            key = (min(p1, p2), max(p1, p2))
            system.addConstraint(p1, p2, distance)
        
        # Select which three particles to use for defining virtual sites.
        
        bestNorm = 0
        vsiteParticles = None
        for p1, p2, p3 in combinations(realParticles, 3):
            d12 = (positions[p2]-positions[p1]).value_in_unit(unit.nanometer)
            d13 = (positions[p3]-positions[p1]).value_in_unit(unit.nanometer)
            crossNorm = unit.norm((d12[1]*d13[2]-d12[2]*d13[1], d12[2]*d13[0]-d12[0]*d13[2], d12[0]*d13[1]-d12[1]*d13[0]))
            if crossNorm > bestNorm:
                bestNorm = crossNorm
                vsiteParticles = (p1, p2, p3)
        if vsiteParticles is None:
            raise ValueError(
                f"Could not find three non-collinear 'real' particles to define "
                f"the virtual-site frame for rigid body {particles}. This means "
                f"all of the real particles selected for this body lie on a "
                f"single line.")
        
        # Create virtual sites.
        
        d12 = (positions[vsiteParticles[1]]-positions[vsiteParticles[0]]).value_in_unit(unit.nanometer)
        d13 = (positions[vsiteParticles[2]]-positions[vsiteParticles[0]]).value_in_unit(unit.nanometer)
        cross = mm.Vec3(d12[1]*d13[2]-d12[2]*d13[1], d12[2]*d13[0]-d12[0]*d13[2], d12[0]*d13[1]-d12[1]*d13[0])
        matrix = np.zeros((3, 3))
        for i in range(3):
            matrix[i][0] = d12[i]
            matrix[i][1] = d13[i]
            matrix[i][2] = cross[i]
        for i in particles:
            if i not in realParticles:
                system.setParticleMass(i, 0)
                rhs = np.array((positions[i]-positions[vsiteParticles[0]]).value_in_unit(unit.nanometer))
                weights = lin.solve(matrix, rhs)
                system.setVirtualSite(i, mm.OutOfPlaneSite(vsiteParticles[0], vsiteParticles[1], vsiteParticles[2], weights[0], weights[1], weights[2]))
