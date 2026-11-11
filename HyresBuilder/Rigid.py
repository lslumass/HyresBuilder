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
For each rigid body, a few atoms are designated as "real" particles whose
positions are integrated normally. All remaining atoms in the body are
converted to ``OutOfPlaneSite`` virtual sites whose positions are recomputed
analytically at every step from three of the real particles. Constraints are
added between every pair of real particles to maintain their pairwise
distances. Any pre-existing constraints that couple two atoms within the same
body are automatically removed to avoid conflicts. Reference geometry is taken
from the positions passed in (normally the PDB file), so each body is frozen
in that conformation.

The number of real particles depends on the body's shape:

* 3D bodies use four real particles forming a tetrahedron that is not flat
  (height at least 0.25 of its longest edge where possible). Four coplanar
  real particles must be avoided: their six constraints do not hold the body
  rigid out of their plane, and such bodies crash to NaN within a few hundred
  steps even at 1 fs.
* Flat or nearly flat bodies (no tetrahedron at least 0.10 high relative to
  its longest edge, e.g. aromatic rings, and every three-atom body) use three
  real particles forming a triangle.

The masses of the real particles are chosen so that the total mass and center
of mass of the body match those of the original atom set (for a nearly flat
body the center of mass can be off by at most about the body's thickness).
Balanced masses are preferred -- at least half of the mean real mass where
possible -- because a very light real particle is unstable at large time
steps. The moment of inertia will be similar but not identical to the
original distribution.

Interface levels
----------------
Six functions are provided at increasing levels of abstraction:

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
* :func:`createRigidComplex` — high-level; same selection options as
  :func:`createRigidSegments`, but the selected residues of all listed
  segments form a *single* rigid body, e.g. two interacting domains on two
  chains held together as one complex. Residues can be the same for every
  segment ("27-95") or given per segment
  (``{"P001": "27-95", "P002": "110-180"}``).
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
  / ``createRigidComplex`` with ``loop=False``, to run DSSP)
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


def createRigidComplex(system, psf, pdb, residues, segments=None, loop=True, CA=False):
    """Make one rigid body out of selected residues of several PSF segments,
    e.g. two interacting domains on two chains held together as a complex.

    Same selection options as :func:`createRigidSegments`, but instead of one
    body per segment, the atoms selected from *all* listed segments go into a
    single rigid body. The complex is frozen in its PDB geometry, so the
    domains keep their relative pose (bound interface) for the whole
    simulation while everything outside the body stays flexible.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    psf : str or openmm.app.CharmmPsfFile
        Either a path to a PSF file (str) or an already-loaded CharmmPsfFile object.
    pdb : str or openmm.app.PDBFile
        Either a path to a PDB file (str) or an already-loaded PDBFile object.
        Positions are extracted from this file; the complex must be whole
        (not split across the periodic box) in it.
    residues : str or dict
        Which author residue numbers to take from each segment.

        * str, e.g. "27-95": the same residues from every segment in
          `segments`.
        * dict mapping segment specs to residue specs, for different
          residues on each chain, e.g.
          ``{"P001": "27-95", "P002": "110-180"}``. Keys may be segment
          ranges or lists like the `segments` string ("P001-P004" or
          "P001,P003"). `segments` must then be None.
    segments : str, optional
        Comma-separated segment-ID ranges, e.g. "P001-P002" or "P001,P005".
        Required when `residues` is a str.
    loop : bool, optional
        If True (default), every listed residue is included. If False, residues
        that DSSP (MDTraj) assigns as coil ('C') are left out. Requires MDTraj.
    CA : bool, optional
        If True, only the CA atoms of the selected residues go into the body.

    Returns
    -------
    numAtoms : int
        The number of atoms in the rigid body.

    Raises
    ------
    ValueError
        If `residues`/`segments` are inconsistent, if a segment appears twice,
        if fewer than three atoms are selected, or (from
        :func:`createRigidBodies`) if the body is degenerate.
    ImportError
        If ``loop=False`` and MDTraj is not installed.

    Notes
    -----
    Segment IDs not found in the PSF, and listed segments from which no atom
    is selected, are skipped with a warning. Call this once per complex; an
    atom must not belong to more than one rigid body, so do not also rigidify
    the same residues with :func:`createRigidSegments`. Nonbonded forces
    between atoms of the same body are still computed but have no effect.
    The constraint caveats of :func:`createRigidSegments` (``loop=False``,
    ``CA=True``) apply here as well.

    Example
    -------
    ::

        from HyresBuilder.Rigid import createRigidComplex

        # Residues 27-95 of P001 and 110-180 of P002 form one rigid complex.
        def modification(system):
            createRigidComplex(system, 'conf.psf', 'conf.pdb',
                               residues={"P001": "27-95", "P002": "110-180"})

        # Same residue range on both chains of a homodimer.
        def modification(system):
            createRigidComplex(system, 'conf.psf', 'conf.pdb',
                               residues="27-95", segments="P001-P002")
    """
    import warnings

    if isinstance(residues, dict):
        if segments is not None:
            raise ValueError("Give `segments` only when `residues` is a string; "
                             "with a dict the segments are its keys.")
        selection = [(segid, set(_parseIndexRanges(resSpec)))
                     for segSpec, resSpec in residues.items()
                     for segid in _parseSegmentRange(segSpec)]
    else:
        if segments is None:
            raise ValueError("`segments` is required when `residues` is a string.")
        resSet = set(_parseIndexRanges(residues))
        selection = [(segid, resSet) for segid in _parseSegmentRange(segments)]

    segIDs = [segid for segid, _ in selection]
    dup = sorted({s for s in segIDs if segIDs.count(s) > 1})
    if dup:
        raise ValueError(f"Segment(s) listed more than once: {dup}")

    psf, pdb = _loadPsfPdb(psf=psf, pdb=pdb)
    positions = pdb.positions
    ssCodes = None if loop else _dsspCodes(psf, pdb)
    chain_map = {chain.id: chain for chain in psf.topology.chains()}

    body_atoms = []
    missing = []
    empty = []
    used = []
    nCoil = 0
    for segid, resSet in selection:
        if segid not in chain_map:
            missing.append(segid)
            continue

        n0 = len(body_atoms)
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
        (used if len(body_atoms) > n0 else empty).append(segid)

    if missing:
        warnings.warn(f"{len(missing)} segment(s) not found in PSF and were skipped: "
                      f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    if empty:
        warnings.warn(f"No atoms selected from {len(empty)} segment(s) "
                      f"(loop={loop}, CA={CA}): {empty[:10]}{'...' if len(empty) > 10 else ''}")
    if len(body_atoms) < 3:
        raise ValueError(f"Only {len(body_atoms)} atom(s) selected for the rigid complex -- "
                         f"check `residues`, `segments`, loop={loop} and CA={CA}.")

    msg = (f"[Rigid] Built 1 rigid complex of {len(body_atoms)} atoms from "
           f"{len(used)} segment(s): {', '.join(used[:10])}{'...' if len(used) > 10 else ''}")
    if not loop:
        msg += f"; {nCoil} coil residue(s) excluded by DSSP"
    if CA:
        msg += "; CA atoms only"
    print(msg + ".")
    createRigidBodies(system, positions, [body_atoms])
    return len(body_atoms)


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

# --- selection of the real particles -------------------------------------------------
# A tetrahedron is scored by its flatness, height/longest edge (min over faces;
# 0.816 for a regular tetrahedron, 0 when the four are coplanar), and a triangle by
# its fatness, height/longest edge (0.866 equilateral, 0 collinear). Masses are
# scored by balance, the lightest real mass over the mean real mass.
_TETRA_FLAT_TARGET = 0.25    # tetrahedra at least this fat count as good
_TETRA_FLAT_MIN = 0.10       # below this no usable tetrahedron exists -> 3 real particles
_TRI_FAT_TARGET = 0.30       # triangles at least this fat count as good
_BALANCE_TARGET = 0.5        # lightest real mass >= half the mean (M/8 for 4, M/6 for 3)
_LINEAR_TOL = 1e-3           # 2nd/1st principal extent below this: atoms on a line
_POOL_RANKED = 18            # candidates closest to the RMS radius ...
_RANDOM_TRIALS = 100000      # ... and random sets over all atoms if none of those is valid


def _candidatePool(r):
    """Indices (into r) of candidate real particles: atoms closest to the body's RMS
    radius plus the two extremes along each principal axis."""
    n = len(r)
    if n <= _POOL_RANKED + 6:
        return np.arange(n)
    d = np.linalg.norm(r, axis=1)
    rms = np.sqrt(np.mean(d**2))
    ranked = list(np.argsort(np.abs(d - rms))[:_POOL_RANKED])
    _, _, vt = np.linalg.svd(r, full_matrices=False)
    proj = r @ vt.T
    extremes = [int(np.argmin(proj[:, k])) for k in range(3)] + [int(np.argmax(proj[:, k])) for k in range(3)]
    pool = []
    for i in ranked + extremes:
        if i not in pool:
            pool.append(int(i))
    return np.array(pool)


def _score(balance, shape, shape_target):
    """Higher is better: reach both targets first, then prefer balanced and fat sets."""
    return (np.minimum(1.0, balance/_BALANCE_TARGET)*np.minimum(1.0, shape/shape_target)
            + 1e-3*balance*shape)


def _bestTetrahedron(r, totalMass, combos):
    """Best set of four real particles among `combos` (array (N, 4) of indices into r).

    Returns (combo, masses, flatness) or None if no set has all masses positive.
    The masses reproduce the total mass and the centre of mass (the origin of r).
    """
    P = r[combos]                                      # (N, 4, 3)
    E = P[:, 1:] - P[:, :1]                            # edges from vertex 0
    vol = np.abs(np.linalg.det(E))/6.0
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    area = np.max([0.5*np.linalg.norm(np.cross(P[:, b] - P[:, a], P[:, c] - P[:, a]), axis=1)
                   for a, b, c in faces], axis=0)
    edge = np.max([np.linalg.norm(P[:, a] - P[:, b], axis=1)
                   for a, b in combinations(range(4), 2)], axis=0)
    flat = np.where(area > 0, 3.0*vol/np.where(area > 0, area, 1.0), 0.0)/np.where(edge > 0, edge, 1.0)
    ok = flat > 1e-6                                   # coplanar sets cannot carry the COM
    if not ok.any():
        return None
    A = np.ones((ok.sum(), 4, 4))
    A[:, :3, :] = np.transpose(P[ok], (0, 2, 1))
    w = np.linalg.solve(A, np.broadcast_to(np.array([0.0, 0.0, 0.0, totalMass]), (ok.sum(), 4))[..., None])[..., 0]
    valid = np.all(w > 0, axis=1)
    if not valid.any():
        return None
    w, f, c = w[valid], flat[ok][valid], combos[ok][valid]
    k = int(np.argmax(_score(w.min(axis=1)/(totalMass/4), f, _TETRA_FLAT_TARGET)))
    return c[k], w[k], float(f[k])


def _bestTriangle(r, totalMass, combos, axes):
    """Best set of three real particles among `combos` (array (N, 3)) for a flat body.

    The masses reproduce the total mass and the in-plane position of the centre of
    mass; `axes` (2, 3) span the body's plane. Returns (combo, masses, fatness) or None.
    """
    Q = r[combos] @ axes.T                             # (N, 3, 2) in-plane coordinates
    area = 0.5*np.abs((Q[:, 1, 0] - Q[:, 0, 0])*(Q[:, 2, 1] - Q[:, 0, 1])
                      - (Q[:, 1, 1] - Q[:, 0, 1])*(Q[:, 2, 0] - Q[:, 0, 0]))
    edge = np.max([np.linalg.norm(Q[:, a] - Q[:, b], axis=1) for a, b in ((0, 1), (0, 2), (1, 2))], axis=0)
    fat = np.where(edge > 0, 2.0*area/np.where(edge > 0, edge, 1.0)**2, 0.0)
    ok = fat > 1e-6
    if not ok.any():
        return None
    A = np.ones((ok.sum(), 3, 3))
    A[:, :2, :] = np.transpose(Q[ok], (0, 2, 1))
    w = np.linalg.solve(A, np.broadcast_to(np.array([0.0, 0.0, totalMass]), (ok.sum(), 3))[..., None])[..., 0]
    valid = np.all(w > 0, axis=1)
    if not valid.any():
        return None
    w, f, c = w[valid], fat[ok][valid], combos[ok][valid]
    k = int(np.argmax(_score(w.min(axis=1)/(totalMass/3), f, _TRI_FAT_TARGET)))
    return c[k], w[k], float(f[k])


def _combos(pool, k):
    return np.array(list(combinations(pool, k)), dtype=int).reshape(-1, k)


def _randomCombos(n, k, count, rng):
    c = np.array([rng.choice(n, size=k, replace=False) for _ in range(count)])
    return np.unique(np.sort(c, axis=1), axis=0)


def _selectRealParticles(xyz, mass):
    """Choose the real particles of one body.

    Args:
        xyz (ndarray (n, 3)): positions in nm.
        mass (ndarray (n,)): masses in amu.

    Returns:
        (local indices, masses in amu, kind) where kind is 'tetrahedron' or 'triangle'.

    Raises:
        ValueError: fewer than three atoms, atoms on a line, or no valid set.
    """
    n = len(xyz)
    if n < 3:
        raise ValueError("a rigid body needs at least three atoms")
    totalMass = float(mass.sum())
    r = xyz - (mass[:, None]*xyz).sum(axis=0)/totalMass
    sv = np.linalg.svd(r, compute_uv=False)
    if sv[0] == 0 or sv[1]/sv[0] < _LINEAR_TOL:
        raise ValueError("its atoms lie on a straight line, which cannot be made rigid this way")

    pool = _candidatePool(r)
    rng = np.random.default_rng(0)
    tet = None
    if n >= 4:
        tet = _bestTetrahedron(r, totalMass, _combos(pool, 4))
        if tet is None and n > len(pool):
            tet = _bestTetrahedron(r, totalMass, _randomCombos(n, 4, _RANDOM_TRIALS, rng))
    if tet is not None and tet[2] >= _TETRA_FLAT_MIN:
        return tet[0], tet[1], 'tetrahedron'

    # flat or nearly flat body: three real particles spanning the body's plane
    _, _, vt = np.linalg.svd(r, full_matrices=False)
    tri = _bestTriangle(r, totalMass, _combos(pool, 3), vt[:2])
    if tri is None and n > len(pool):
        tri = _bestTriangle(r, totalMass, _randomCombos(n, 3, _RANDOM_TRIALS, rng), vt[:2])
    if tri is not None:
        return tri[0], tri[1], 'triangle'
    if tet is not None:                    # no triangle carries the COM: keep the flat tetrahedron
        return tet[0], tet[1], 'tetrahedron'
    raise ValueError("no set of real particles reproduces its total mass and centre of mass "
                     "with positive masses")


def createRigidBodies(system, positions, bodies):
    """Modify a System to turn specified sets of particles into rigid bodies.

    This is the low-level interface that operates directly on pre-built atom index lists.
    For a higher-level interface that accepts PSF segment IDs and residue ranges, use
    createRigidSegments() instead.

    For every rigid body a few particles are selected as "real" particles whose positions are
    integrated, with constraints between every pair of them so that they move as a rigid body. All
    other particles in the body become massless ``OutOfPlaneSite`` virtual sites placed from three
    of the real particles. The masses of the real particles are chosen so that the total mass and
    center of mass of each body match those of the original particles; the moment of inertia is
    similar but not identical.

    How the real particles are chosen depends on the body's shape:

    * **3D bodies -- four real particles** (a tetrahedron). Candidate sets are drawn from the
      atoms closest to the body's RMS radius plus the extreme atoms along each principal axis
      (random sets over all atoms if none of those works); masses come from solving for the
      total mass and center of mass, and sets with any non-positive mass are rejected. Among the
      valid sets the one preferred reaches both targets -- every mass at least 1/8 of the body's
      total mass (a very light real particle blows up at large time steps) and a tetrahedron that
      is not flat (height at least 0.25 of its longest edge) -- and is otherwise the best
      compromise. Four (nearly) coplanar real particles must not be used: their six constraints
      do not hold the body rigid out of their plane, and such bodies crash to NaN within a few
      hundred steps even at 1 fs.
    * **Flat or nearly flat bodies -- three real particles** (a triangle). If no tetrahedron of
      the body is at least 0.10 high relative to its longest edge (e.g. aromatic rings or other
      planar molecules, and every body of three atoms), three particles whose triangle contains
      the center of mass are used instead, preferring masses of at least 1/6 of the total and a
      fat triangle. The masses reproduce the total mass and the in-plane position of the center
      of mass; for a nearly flat body the center of mass can be off by at most about the body's
      thickness.

    Care is needed when using constraints, since virtual particles cannot participate in constraints.  If the
    input system includes any constraints, this function will automatically remove ones that connect two
    particles in the same rigid body.  But if there is a constraint between a particle in a rigid body and
    another particle not in that body, it will likely lead to an exception when you try to create a context.

    Parameters
    ----------
    system : openmm.System
        The System to modify (in place). Must be called before a Context is created from it.
    positions : list of Vec3 (Quantity with length units)
        The positions of all particles in the system, e.g. ``PDBFile.positions``. These define the
        rigid geometry.
    bodies : list of list of int
        Each element defines one rigid body as a list of atom indices. Every body needs at least three
        particles that do not all lie on one line; a particle may belong to only one body.

    Returns
    -------
    dict
        Counts of the bodies built with four (``'tetrahedron'``) and three (``'triangle'``) real
        particles.

    Raises
    ------
    ValueError
        If a body has fewer than three particles, if its particles lie on a straight line, if a
        particle appears in more than one body, or if no set of real particles with positive
        masses exists.

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
    bodyOf = {}
    for b, particles in enumerate(bodies):
        for i in particles:
            if i in bodyOf and bodyOf[i] != b:
                raise ValueError(f"particle {i} belongs to more than one rigid body")
            bodyOf[i] = b

    # Remove any constraints between two particles of the same rigid body.
    for i in range(system.getNumConstraints()-1, -1, -1):
        p1, p2, distance = system.getConstraintParameters(i)
        if p1 in bodyOf and bodyOf.get(p2) == bodyOf[p1]:
            system.removeConstraint(i)

    counts = {'tetrahedron': 0, 'triangle': 0}
    for particles in bodies:
        particles = list(particles)
        xyz = np.array([positions[i].value_in_unit(unit.nanometer) for i in particles], dtype=float)
        mass = np.array([system.getParticleMass(i).value_in_unit(unit.amu) for i in particles], dtype=float)
        try:
            local, realMasses, kind = _selectRealParticles(xyz, mass)
        except ValueError as e:
            raise ValueError(f"Cannot build rigid body {particles[:10]}"
                             f"{'...' if len(particles) > 10 else ''} ({len(particles)} particles): {e}.")
        counts[kind] += 1
        realParticles = [particles[k] for k in local]
        realSet = set(realParticles)

        # Set particle masses and constrain every pair of real particles.
        for i, m in zip(realParticles, realMasses):
            system.setParticleMass(i, float(m)*unit.amu)
        for a, b in combinations(range(len(local)), 2):
            system.addConstraint(realParticles[a], realParticles[b],
                                 float(np.linalg.norm(xyz[local[a]] - xyz[local[b]]))*unit.nanometer)

        # Virtual-site frame: the three real particles with the largest cross product.
        frame = max(combinations(range(len(local)), 3),
                    key=lambda t: np.linalg.norm(np.cross(xyz[local[t[1]]] - xyz[local[t[0]]],
                                                          xyz[local[t[2]]] - xyz[local[t[0]]])))
        o, p, q = (xyz[local[k]] for k in frame)
        d12, d13 = p - o, q - o
        matrix = np.column_stack([d12, d13, np.cross(d12, d13)])
        f0, f1, f2 = (realParticles[k] for k in frame)
        for k, i in enumerate(particles):
            if i not in realSet:
                system.setParticleMass(i, 0)
                w = np.linalg.solve(matrix, xyz[k] - o)
                system.setVirtualSite(i, mm.OutOfPlaneSite(f0, f1, f2, w[0], w[1], w[2]))
    return counts
