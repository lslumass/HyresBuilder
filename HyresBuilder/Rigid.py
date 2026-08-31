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
body are automatically removed to avoid conflicts.

The four real particles and their masses are chosen so that the total mass and
center of mass of the rigid body exactly match those of the original atom set.
For bodies with fewer than five atoms, all atoms are treated as real particles.
The moment of inertia will be similar but not identical to the original
distribution.

The three reference atoms used to define the virtual-site frame are selected
from the real particles by maximising the norm of their cross product, ensuring
a well-conditioned out-of-plane coordinate frame.

Interface levels
----------------
Four functions are provided at increasing levels of abstraction:

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
  by a PSF.
* :func:`RigidSmallMols` — high-level; for systems with many (potentially
  thousands of) small-molecule segments in a PSF, e.g. 'M001', 'M002', ...
  Accepts a PSF/PDB and a single atom-index pattern ("10-15,20-25") plus a
  set of segment-ID ranges ("M001-M010,M020-M030"), and applies the same
  rigid-body definition to every matching segment in one call.

Limitations
-----------
Virtual sites are massless and cannot participate in constraints with atoms
outside their own rigid body. If such cross-body constraints exist in the
input system they will cause an exception at ``Context`` creation time and
must be removed manually before calling these functions.

Original authors:  Peter Eastman (Stanford University / Simbios)
Modified by:       Shanlong Li

This module is derived from the OpenMM toolkit and is distributed under the
MIT licence. See the licence header in the source file for the full text.

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.unit``)
* `NumPy <https://numpy.org>`_ (``numpy``, ``numpy.linalg``)
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
    omitted (pass None) if a function only needs one of the two.
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
        ready to pass directly to createRigidBodies().
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


def createRigidSegments(system, psf, pdb, residues, segments):
    """Apply the same residue-range rigid-body definition to many PSF segments
    at once, e.g. every chain of a repeated fibril or multimer.

    You give one residue pattern (which author residue numbers to include from
    *each* segment) and a set of segment names/ranges to apply it to; one
    rigid body is created per matching segment, using resolveBodiesToIndices()
    internally to turn residues into atom indices.

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

    Returns
    -------
    numBodies : int
        The number of rigid bodies (matching segments) that were created.

    Example
    -------
    ::

        from Rigid import createRigidSegments

        # Residues 27-95 of every chain P001 through P080, as one rigid body each.
        createRigidSegments(system, 'conf.psf', 'conf.pdb',
                             residues="27-95", segments="P001-P080")
    """
    psf, pdb = _loadPsfPdb(psf=psf, pdb=pdb)
    positions = pdb.positions

    segIDs = _parseSegmentRange(segments)
    resNums = _parseIndexRanges(residues)
    segment_bodies = [(segid, resNums) for segid in segIDs]

    bodies = resolveBodiesToIndices(psf, segment_bodies)
    print(f"[Rigid] Resolved {len(bodies)} rigid bodies from {len(segIDs)} segment(s) "
          f"with residues '{residues}'.")
    createRigidBodies(system, positions, bodies)
    return len(bodies)


def _parseIndexRanges(spec):
    """Parse a comma-separated string of integers/ranges, e.g. "10-15,20-25,30"
    into a sorted list of unique ints: [10, 11, 12, 13, 14, 15, 20, ..., 25, 30].
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
    range's start ID (e.g. "M001-M010" -> M001, M002, ..., M010).
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
        Comma-separated segment-ID ranges to apply this to, e.g.
        "M001-M010,M020-M030" (segments M001 through M010, and M020 through
        M030) or an explicit list like "M001,M005,M010". Numeric ranges keep
        the zero-padding width of the range's start ID.

    Returns
    -------
    numBodies : int
        The number of rigid bodies (matching segments) that were created.

    Example
    -------
    ::

        from Rigid import RigidSmallMols

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

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    positions : list
        The positions of all particles in the system.
    bodies : list of list of int
        Each element defines one rigid body as a list of atom indices.

    Example
    -------
    ::

        from Rigid import createRigidBodies
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
            for p in combinations(rank, 4):
                # Select masses for the "real" particles.  If any is negative, reject this set of particles
                # and keep going.
                
                matrix = np.zeros((4, 4))
                for i in range(4):
                    particleR = r[p[i]].value_in_unit(unit.nanometers)
                    matrix[0][i] = particleR[0]
                    matrix[1][i] = particleR[1]
                    matrix[2][i] = particleR[2]
                    matrix[3][i] = 1.0
                rhs = np.array([0.0, 0.0, 0.0, unit.sum(mass).value_in_unit(unit.amu)])
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
                    # We have a good set of particles.
                    
                    realParticles = [particles[i] for i in p]
                    realParticleMasses = [float(w) for w in weights]*unit.amu
                    break
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