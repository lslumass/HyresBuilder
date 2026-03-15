"""
rigid.py: Implements rigid bodies

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2016 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
__author__ = "Peter Eastman"
__version__ = "1.0"


import openmm as mm
import openmm.unit as unit
import numpy as np
import numpy.linalg as lin
from itertools import combinations


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
              * a (start, end) tuple of inclusive author residue numbers, or
              * an explicit list of author residue numbers [27, 28, 30, ...].

        Examples::

            # range tuple – residues 27 to 95 inclusive in segment P001
            ('P001', (27, 95))

            # explicit list – only these three residues in segment P042
            ('P042', [10, 11, 50])

            # mix both styles across multiple bodies
            [('P001', (27, 95)), ('P002', (27, 95)), ('P003', [30, 31, 32])]

    Returns
    -------
    bodies : list of list of int
        Each inner list contains the atom indices that form one rigid body,
        ready to pass directly to createRigidBodies().
    """
    from openmm.app import CharmmPsfFile
    import warnings
    if isinstance(psf, str):
        psf = CharmmPsfFile(psf)

    # Build a fast lookup: segid -> chain object
    chain_map = {chain.id: chain for chain in psf.topology.chains()}

    bodies = []
    for segid, residue_list in segment_bodies:
        if segid not in chain_map:
            raise ValueError(f"Segment '{segid}' not found in PSF. "
                             f"Available segments: {list(chain_map.keys())}")

        # Normalise residue_list into a set of ints for O(1) membership test
        if isinstance(residue_list, tuple) and len(residue_list) == 2:
            res_set = set(range(int(residue_list[0]), int(residue_list[1]) + 1))
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


def createRigidSegments(system, psf, pdb, segment_bodies):
    """Resolve segment/residue definitions from a PSF/PDB and apply rigid bodies.

    Combines resolveBodiesToIndices() and createRigidBodies() in one call.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    psf : str or openmm.app.CharmmPsfFile
        Either a path to a PSF file (str) or an already-loaded CharmmPsfFile object.
    pdb : str or openmm.app.PDBFile
        Either a path to a PDB file (str) or an already-loaded PDBFile object.
        Positions are extracted from this file.
    segment_bodies : list of (segid, residue_list) tuples
        Each tuple defines one rigid body.
        residue_list can be a (start, end) range tuple or an explicit list of residue numbers.

    Example
    -------
    ::

        from Rigid import createRigidSegments

        segment_bodies = [(f'P{i+1:03}', (27, 95)) for i in range(80)]
        createRigidSegments(system, 'conf.psf', 'conf.pdb', segment_bodies)
    """
    from openmm.app import PDBFile
    if isinstance(pdb, str):
        pdb = PDBFile(pdb)
    positions = pdb.positions

    bodies = resolveBodiesToIndices(psf, segment_bodies)
    print(f"[Rigid] Resolved {len(bodies)} rigid bodies from {len(segment_bodies)} definitions.")
    createRigidBodies(system, positions, bodies)


def createRigidBodies(system, positions, bodies):
    """Modify a System to turn specified sets of particles into rigid bodies.
    
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
    particles in the same rigid body.  But if there is a constraint beween a particle in a rigid body and
    another particle not in that body, it will likely lead to an exception when you try to create a context.
    
    Parameters:
     - system (System) the System to modify
     - positions (list) the positions of all particles in the system
     - bodies (list) each element of this list defines one rigid body.  Each element should itself be a list
       of the indices of all particles that make up that rigid body.
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
                weights = lin.solve(matrix, rhs)
                if all(w > 0.0 for w in weights):
                    # We have a good set of particles.
                    
                    realParticles = [particles[i] for i in p]
                    realParticleMasses = [float(w) for w in weights]*unit.amu
                    break
        
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
        for p1, p2, p3 in combinations(realParticles, 3):
            d12 = (positions[p2]-positions[p1]).value_in_unit(unit.nanometer)
            d13 = (positions[p3]-positions[p1]).value_in_unit(unit.nanometer)
            crossNorm = unit.norm((d12[1]*d13[2]-d12[2]*d13[1], d12[2]*d13[0]-d12[0]*d13[2], d12[0]*d13[1]-d12[1]*d13[0]))
            if crossNorm > bestNorm:
                bestNorm = crossNorm
                vsiteParticles = (p1, p2, p3)
        
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