"""
De novo coarse-grained iConRNA/iConDNA, polyphosphate and PEG chain builder.

This module writes coarse-grained PDB files directly from a sequence or a
chain length, without requiring an all-atom input structure:

* :func:`build_rna` / :func:`build_dna` -- iConRNA / iConDNA nucleic acids.
* :func:`build_polyP` -- polyphosphate, one ``PHO`` bead (atom ``P``) per unit.
* :func:`build_peg` -- PEG/PEO, one ``EO`` bead (residue ``PEG``) per unit.

All chains start at (9000, 9000, 9000) Å and are written in chain ``X``.

Nucleic acids
-------------
Reference bead coordinates are stored in the ``maps`` dictionary, keyed
by single-letter nucleotide code. Purines (A, G) carry seven beads
(P, C1, C2, NA, NB, NC, ND); pyrimidines (C, U, T) carry six
(P, C1, C2, NA, NB, NC). Residue names are ADE, GUA, CYT, URA (RNA,
segment ID ``RNA``) and DA, DG, DC, DT (DNA, segment ID ``DNA``).

Placement is translation only: each template is shifted so that its P
bead sits at the anchor point, and the next anchor is the current
residue's C1 bead shifted by +3.63 Å along z. The result is a simple
stacked, non-helical starting conformation meant to be relaxed by
simulation. The sequence is validated (:func:`_validate_sequence`) before
the file is opened.

Polymers
--------
:func:`build_polyP` and :func:`build_peg` generate self-avoiding random
chains (see their docstrings for bond lengths, angles and exclusion
distances); pass ``seed`` for reproducible coordinates.

Command line
------------
:func:`main` is registered as the ``iconbuilder`` entry point::

    iconbuilder NAME SEQ

``SEQ`` selects the molecule type (output ``NAME.pdb``):

* RNA: letters A/U/C/G (any case), e.g. ``AUCGAUCG``, or repeat shorthand
  ``<motif><count>`` such as ``A100`` or ``CAG50``. The count is the total
  number of nucleotides (the motif is repeated and truncated to that length).
* DNA: lowercase ``d`` prefix, e.g. ``dATCG``, ``dA100``.
* polyP: ``P<count>``, e.g. ``P10``.
* PEG: ``EO<count>``, e.g. ``EO20``.

All files carry an ``REMARK  iConRNA`` header line (also for DNA, polyP
and PEG) plus a ``REMARK  SEQUENCE`` line.

Reference
---------
S. Li and J. Chen, *Proc. Natl. Acad. Sci. USA*, 2025, **122**, e2504583122.

Author:     Shanlong Li
Date:       Nov 13, 2023
"""

import argparse
import re


maps = {
    'A': [
        (1,  'P', -0.129, 8.827, 16.666),
        (2, 'C1', -4.005, 8.261, 15.960),
        (3, 'C2', -4.464, 6.345, 14.718),
        (4, 'NA', -3.566, 4.736, 14.629),
        (5, 'NB', -1.574, 3.674, 14.575),
        (6, 'NC', -2.722, 1.355, 14.212),
        (7, 'ND', -5.098, 2.343, 14.218)
    ],
    'G': [
        (1,  'P', -0.129, 8.827, 16.666),
        (2, 'C1', -4.005, 8.261, 15.960),
        (3, 'C2', -4.464, 6.345, 14.718),
        (4, 'NA', -3.566, 4.736, 14.629),
        (5, 'NB', -1.574, 3.674, 14.575),
        (6, 'NC', -2.722, 1.355, 14.212),
        (7, 'ND', -5.098, 2.343, 14.218)
    ],
    'C': [
        (1,  'P', -5.442,  6.919, 19.337),
        (2, 'C1', -8.224,  4.298, 18.261),
        (3, 'C2', -7.482,  2.312, 17.284),
        (4, 'NA', -5.016,  2.586, 17.726),
        (5, 'NB', -3.505,  0.658, 17.600),
        (6, 'NC', -6.020, -0.162, 17.143)
    ],
    'U': [
        (1,  'P', -5.442,  6.919, 19.337),
        (2, 'C1', -8.224,  4.298, 18.261),
        (3, 'C2', -7.482,  2.312, 17.284),
        (4, 'NA', -5.016,  2.586, 17.726),
        (5, 'NB', -3.505,  0.658, 17.600),
        (6, 'NC', -6.020, -0.162, 17.143)
    ],
    'T': [
        (1,  'P', -5.442,  6.919, 19.337),
        (2, 'C1', -8.224,  4.298, 18.261),
        (3, 'C2', -7.482,  2.312, 17.284),
        (4, 'NA', -5.016,  2.586, 17.726),
        (5, 'NB', -3.505,  0.658, 17.600),
        (6, 'NC', -6.020, -0.162, 17.143)
    ]
}

_VALID_RNA = frozenset('AUCG')
_VALID_DNA = frozenset('ATCG')


def _validate_sequence(sequence, valid_bases, molecule):
    """Raise ValueError with a clear message if *sequence* contains unknown bases."""
    invalid = sorted(set(sequence) - valid_bases)
    if invalid:
        raise ValueError(
            f"Invalid {molecule} nucleotide(s): {', '.join(invalid)}. "
            f"Supported bases: {', '.join(sorted(valid_bases))}."
        )


def _validate_name(name):
    """Raise ValueError if *name* is empty."""
    if not name:
        raise ValueError("Output name must not be empty.")


def printcg(atoms, file):
    """Write atoms as fixed-column PDB ``ATOM`` records.

    Args:
        atoms (list[list]): Each atom is ``[record, serial, name, resname,
            chain, resid, x, y, z, occupancy, bfactor, segid]``. Atom names
            are right-aligned in two characters (columns 13-14), so names
            longer than two characters shift the following columns.
        file: Open, writable text file object.
    """
    for atom in atoms:
        file.write('{}  {:5d} {:>2}   {} {}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:<4}\n'.format(
            atom[0], int(atom[1]), atom[2], atom[3], atom[4], int(atom[5]),
            atom[6], atom[7], atom[8], atom[9], atom[10], atom[11]))


def readRNAmap(seq):
    """Return the iConRNA template beads for one RNA nucleotide.

    Args:
        seq (str): Upper-case nucleotide code, one of ``A``, ``G``, ``C``, ``U``.

    Returns:
        list[list]: Atom records in :func:`printcg` layout with residue names
        ADE/GUA/CYT/URA, chain ``X``, resid 1 and segment ID ``RNA``.

    Raises:
        KeyError: If *seq* is not a supported RNA nucleotide.
    """
    atoms = []
    nos = {'A': 'ADE', 'G': 'GUA', 'C': 'CYT', 'U': 'URA'}
    for index, name, rx, ry, rz in maps[seq]:
        atom = ['ATOM', index, name, nos[seq], 'X', 1, rx, ry, rz, 1.00, 0.00, 'RNA']
        atoms.append(atom)
    return atoms


def readDNAmap(seq):
    """Return the iConDNA template beads for one DNA nucleotide.

    Args:
        seq (str): Upper-case nucleotide code, one of ``A``, ``G``, ``C``, ``T``.

    Returns:
        list[list]: Atom records in :func:`printcg` layout with residue names
        DA/DG/DC/DT (written right-aligned as ``' DA'`` etc.), chain ``X``,
        resid 1 and segment ID ``DNA``.

    Raises:
        KeyError: If *seq* is not a supported DNA nucleotide.
    """
    atoms = []
    nos = {'A': ' DA', 'G': ' DG', 'C': ' DC', 'T': ' DT'}
    for index, name, rx, ry, rz in maps[seq]:
        atom = ['ATOM', index, name, nos[seq], 'X', 1, rx, ry, rz, 1.00, 0.00, 'DNA']
        atoms.append(atom)
    return atoms


def transform(ref, atoms):
    """Translate *atoms* in place so that the first atom (P) lies at *ref*.

    Args:
        ref (list[float]): Target (x, y, z) in Å.
        atoms (list[list]): Atom records in :func:`printcg` layout.

    Returns:
        list[list]: The same, modified *atoms* list.
    """
    refx, refy, refz = ref[0], ref[1], ref[2]
    Px, Py, Pz = atoms[0][6], atoms[0][7], atoms[0][8]
    dx, dy, dz = Px - refx, Py - refy, Pz - refz
    for atom in atoms:
        atom[6] -= dx
        atom[7] -= dy
        atom[8] -= dz
    return atoms


def _build(name, sequence, map_func, molecule):
    """Shared build core used by :func:`build_rna` and :func:`build_dna`.

    Upper-cases and validates *sequence*, then writes ``<name>.pdb`` (see the
    module docstring for the placement scheme).

    Args:
        name (str): Output file stem; must not be empty.
        sequence (str): Nucleotide sequence (any case).
        map_func (callable): :func:`readRNAmap` or :func:`readDNAmap`.
        molecule (str): ``'RNA'`` or ``'DNA'``; selects the allowed alphabet.

    Raises:
        ValueError: If *name* is empty or *sequence* contains invalid bases.
    """
    _validate_name(name)
    sequence = sequence.upper()
    valid = _VALID_RNA if molecule == 'RNA' else _VALID_DNA
    _validate_sequence(sequence, valid, molecule)

    out = f'{name}.pdb'
    with open(out, 'w') as f:
        print('REMARK  iConRNA', file=f)
        print('REMARK  CREATE BY RNABUILDER/SHANLONG LI', file=f)
        print('REMARK  Ref: S. Li and J. Chen, PNAS, 2025, 122, e2504583122.', file=f)
        print('REMARK  SEQUENCE: {}'.format(sequence), file=f)
        idx = 0
        res = 0
        ref = [9000.0, 9000.0, 9000.0]
        for seq in sequence:
            atoms = map_func(seq)
            for atom in atoms:
                atom[1] += idx
                atom[5] += res
            atoms = transform(ref, atoms)
            ref = [atoms[1][6], atoms[1][7], atoms[1][8] + 3.63]
            idx += len(atoms)
            res += 1
            printcg(atoms, f)
        print('END', file=f)


def build_rna(name, sequence):
    """
    Build an iConRNA coarse-grained RNA structure from a nucleotide sequence.

    Each residue is placed by translating a fixed template of reference beads
    (P, C1, C2, NA, NB, NC, plus ND for purines) so that its P bead sits 3.63 Å
    above (+z) the previous residue's C1 bead; the first P is placed at
    (9000, 9000, 9000) Å. No rotation is applied, so the chain is a simple
    stacked, non-helical starting model. The sequence is validated before any
    file is written. The structure is written with iConRNA REMARK headers
    (including the sequence) in chain X.

    Args:
        name (str): Stem of the output file. The PDB is written to ``<name>.pdb``.
            Must not be empty.
        sequence (str): RNA sequence in single-letter codes (e.g. ``'AUCG'``).
            Case-insensitive. Supported nucleotides: ``A``, ``U``, ``C``, ``G``.

    Returns:
        None. Writes ``<name>.pdb`` (relative to the current working directory
        unless *name* contains a path). Residue names ADE/GUA/CYT/URA, segment
        ID ``RNA``.

    Raises:
        ValueError: If *name* is empty or *sequence* contains unsupported bases.

    Example:
        >>> from HyresBuilder import iConBuilder
        >>> iConBuilder.build_rna("myrna", "AUCGAUCG")
        # output: myrna.pdb
    """
    _build(name, sequence, readRNAmap, 'RNA')


def build_dna(name, sequence):
    """
    Build an iConRNA coarse-grained DNA structure from a nucleotide sequence.

    Each residue is placed by translating a fixed template of reference beads
    (P, C1, C2, NA, NB, NC, plus ND for purines) so that its P bead sits 3.63 Å
    above (+z) the previous residue's C1 bead; the first P is placed at
    (9000, 9000, 9000) Å. No rotation is applied, so the chain is a simple
    stacked, non-helical starting model. The sequence is validated before any
    file is written. The structure is written with iConRNA REMARK headers
    (including the sequence) in chain X.

    Residue names are DA, DG, DC, DT (segment ID ``DNA``), matching
    ``top_DNA_mix.inp``.

    Args:
        name (str): Stem of the output file. The PDB is written to ``<name>.pdb``.
            Must not be empty.
        sequence (str): DNA sequence in single-letter codes (e.g. ``'ATCG'``).
            Case-insensitive. Supported nucleotides: ``A``, ``T``, ``C``, ``G``.

    Returns:
        None. Writes a PDB file to ``<name>.pdb`` in the current working directory.

    Raises:
        ValueError: If *name* is empty or *sequence* contains unsupported bases.

    Example:
        >>> from HyresBuilder import iConBuilder
        >>> iConBuilder.build_dna("mydna", "ATCGATCG")
        # output: mydna.pdb
    """
    _build(name, sequence, readDNAmap, 'DNA')


def build_polyP(name, n, seed=None):
    """
    Build a poly-phosphate (polyP) coarse-grained structure of n residues.

    Each residue is a single bead (residue PHO, atom name P, segment ID
    ``S001``), starting at (9000, 9000, 9000) Å. Beads are placed by a random
    walk with every P-P bond exactly 2.7 Å. Each step direction is a random
    unit vector, accepted only if its z component is positive (the chain
    always advances along +z) and, after the first step, if it makes an
    angle < 90° with the previous step (i.e. P-P-P angle > 90°).

    Self-avoiding constraint: a new bead is rejected if it lies within
    4.0 Å (inclusive) of any earlier bead other than its bonded predecessor.
    Up to 50 placements are tried per bead; if all fail, the whole chain is
    restarted (at most 1000 restarts). (A 5.0 Å limit is not used because
    with a 2.7 Å bond it would force all bond angles to be > 135°.)

    The PDB has ``REMARK  iConRNA``, ``REMARK  CREATE BY HyResBuilder`` and
    ``REMARK  SEQUENCE: PHO x<n>`` headers (no reference line).

    Args:
        name (str): Stem of the output file. The PDB is written to ``<name>.pdb``.
            Must not be empty.
        n (int): Number of phosphate beads (residues) in the chain. Must be >= 1.
        seed (int, optional): Random seed for reproducibility. Default is None
            (non-reproducible).

    Returns:
        None. Writes a PDB file to ``<name>.pdb`` in the current working directory.

    Raises:
        ValueError: If *name* is empty or *n* < 1.
        RuntimeError: If no collision-free chain is found within 1000 restarts.

    Example:
        >>> from HyresBuilder import iConBuilder
        >>> iConBuilder.build_polyP("polyP", 10)
        # output: polyP.pdb
        >>> iConBuilder.build_polyP("polyP_rep", 10, seed=42)  # reproducible
    """
    import math
    import random

    _validate_name(name)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    PP_DIST = 2.7   # Å, fixed P-P bond length
    MIN_DIST_SQ = 4.0 ** 2  # 4.0 Å squared for faster distance math

    rng = random.Random(seed)

    def random_unit_vector():
        """Random unit vector: (x, y) uniform in the unit disk, z = +/-sqrt(1-x^2-y^2).

        Note: this is not uniform on the sphere (density is biased toward the poles).
        """
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x * x + y * y >= 1:
                continue
            z = math.sqrt(1 - x * x - y * y) * rng.choice([-1, 1])
            return x, y, z

    def next_direction(prev_dir):
        """Draw random unit vectors until one has z > 0 and, if *prev_dir* is given, a positive dot product with it."""
        while True:
            d = random_unit_vector()
            if d[2] <= 0:                              # must go +z
                continue
            if prev_dir is not None:
                dot = d[0] * prev_dir[0] + d[1] * prev_dir[1] + d[2] * prev_dir[2]
                if dot <= 0:                           # P-P-P angle must be > 90°
                    continue
            return d

    def generate_chain():
        """Build the bead coordinates, restarting the whole chain (up to 1000 times) when a bead cannot be placed in 50 tries."""
        max_restarts = 1000
        for attempt in range(max_restarts):
            x, y, z = 9000.0, 9000.0, 9000.0
            coords = [(x, y, z)]
            prev_dir = None
            stuck = False

            for i in range(n - 1):
                placed = False
                # Try up to 50 local placements to satisfy the self-avoiding constraint
                for _ in range(50):
                    candidate_dir = next_direction(prev_dir)
                    
                    nx = coords[-1][0] + candidate_dir[0] * PP_DIST
                    ny = coords[-1][1] + candidate_dir[1] * PP_DIST
                    nz = coords[-1][2] + candidate_dir[2] * PP_DIST

                    # Excluded volume check: distance > 4.0 Å for non-adjacent beads
                    collision = False
                    for cx, cy, cz in coords[:-1]:
                        if (nx - cx)**2 + (ny - cy)**2 + (nz - cz)**2 <= MIN_DIST_SQ:
                            collision = True
                            break

                    if not collision:
                        prev_dir = candidate_dir
                        coords.append((nx, ny, nz))
                        placed = True
                        break

                if not placed:
                    stuck = True
                    break  # Chain got trapped, break out and restart the entire chain

            if not stuck:
                return coords
                
        raise RuntimeError(f"Failed to build a collision-free polyP chain after {max_restarts} attempts. Try a smaller n.")

    # Build coordinates via constrained, self-avoiding random walk
    coords = generate_chain()

    out = f"{name}.pdb"
    with open(out, "w") as f:
        print("REMARK  iConRNA", file=f)
        print("REMARK  CREATE BY HyResBuilder", file=f)
        print("REMARK  SEQUENCE: PHO x{}".format(n), file=f)
        for i, (cx, cy, cz) in enumerate(coords):
            atom = ["ATOM", i + 1, "P", "PHO", "X", i + 1,
                    cx, cy, cz, 1.00, 0.00, "S001"]
            printcg([atom], f)
        print("END", file=f)


def build_peg(name, n, seed=None):
    """
    Build a poly(ethylene glycol) (PEG) coarse-grained structure of n repeat units.

    Each repeat unit is represented by a single bead (residue PEG, atom name EO,
    segment ID ``PEG``), matching ``RESI PEG`` in ``top_polymer.inp``. The chain
    starts at (9000, 9000, 9000) Å with a random first bond direction and is
    built as a freely-rotating chain: every EO-EO bond is exactly 3.5 Å, every
    EO-EO-EO angle is exactly 123°, and each torsion is drawn uniformly from
    [0, 2π). (Note: the force-field bond b0 in ``param_polymer.inp`` is 3.60 Å;
    the angle matches its 123° theta0.)

    Self-avoiding constraint: a new bead is rejected if it lies within 5.0 Å
    (inclusive) of any earlier bead other than its bonded predecessor, so all
    non-adjacent pairs end up > 5.0 Å apart. Up to 50 torsions are tried per
    bead; if all fail, the whole chain is restarted (at most 1000 restarts).

    The PDB carries ``REMARK  iConRNA`` / ``CREATE BY RNABUILDER`` / reference
    headers and ``REMARK  SEQUENCE: PEG x<n>``.

    Args:
        name (str): Stem of the output file. The PDB is written to ``<name>.pdb``.
            Must not be empty.
        n (int): Number of EO beads (repeat units) in the chain. Must be >= 1.
        seed (int, optional): Random seed for reproducibility. Default is None
            (non-reproducible).

    Returns:
        None. Writes a PDB file to ``<name>.pdb`` in the current working directory.

    Raises:
        ValueError: If *name* is empty or *n* < 1.
        RuntimeError: If no collision-free chain is found within 1000 restarts.

    Example:
        >>> from HyresBuilder import iConBuilder
        >>> iConBuilder.build_peg("peg20", 20, seed=1)
        # output: peg20.pdb
    """
    import math
    import random

    _validate_name(name)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    EO_DIST  = 3.5                      # Å, EO-EO virtual bond length
    ANGLE    = 123.0                    # degrees, fixed EO-EO-EO bond angle
    TILT     = math.radians(180.0 - ANGLE)   # 57°
    COS_TILT = math.cos(TILT)          
    SIN_TILT = math.sin(TILT)          
    MIN_DIST_SQ = 5.0 ** 2              # 0.5 nm = 5.0 Å; squared for faster distance math

    rng = random.Random(seed)

    def random_unit_vector():
        """Random unit vector: (x, y) uniform in the unit disk, z = +/-sqrt(1-x^2-y^2).

        Note: this is not uniform on the sphere (density is biased toward the poles).
        """
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x * x + y * y >= 1:
                continue
            z = math.sqrt(1 - x * x - y * y) * rng.choice([-1, 1])
            return (x, y, z)

    def perp_vector(v):
        """Return a unit vector perpendicular to v (v x z-axis; undefined if v is parallel to z)."""
        ax = (0.0, 0.0, 1.0) if abs(v[0]) < 0.9 or abs(v[1]) < 0.9 else (1.0, 0.0, 0.0)
        cx = v[1] * ax[2] - v[2] * ax[1]
        cy = v[2] * ax[0] - v[0] * ax[2]
        cz = v[0] * ax[1] - v[1] * ax[0]
        norm = math.sqrt(cx * cx + cy * cy + cz * cz)
        return (cx / norm, cy / norm, cz / norm)

    def next_bond(prev_bond):
        """Return the next bond unit vector at 57 deg to *prev_bond* (123 deg bond angle) with a uniform random torsion."""
        p1 = perp_vector(prev_bond)
        p2 = (
            prev_bond[1] * p1[2] - prev_bond[2] * p1[1],
            prev_bond[2] * p1[0] - prev_bond[0] * p1[2],
            prev_bond[0] * p1[1] - prev_bond[1] * p1[0],
        )
        phi = rng.uniform(0.0, 2.0 * math.pi)   # random torsion angle
        cos_phi, sin_phi = math.cos(phi), math.sin(phi)
        
        nx = COS_TILT * prev_bond[0] + SIN_TILT * (cos_phi * p1[0] + sin_phi * p2[0])
        ny = COS_TILT * prev_bond[1] + SIN_TILT * (cos_phi * p1[1] + sin_phi * p2[1])
        nz = COS_TILT * prev_bond[2] + SIN_TILT * (cos_phi * p1[2] + sin_phi * p2[2])
        return (nx, ny, nz)

    def generate_chain():
        """Build the bead coordinates, restarting the whole chain (up to 1000 times) when a bead cannot be placed in 50 tries."""
        max_restarts = 1000
        for attempt in range(max_restarts):
            x, y, z = 9000.0, 9000.0, 9000.0
            coords = [(x, y, z)]
            bond = random_unit_vector()
            stuck = False

            for i in range(n - 1):
                placed = False
                # Try up to 50 random torsion angles for the current bead
                for _ in range(50):
                    if i > 0:
                        test_bond = next_bond(bond)
                    else:
                        test_bond = bond

                    nx = coords[-1][0] + test_bond[0] * EO_DIST
                    ny = coords[-1][1] + test_bond[1] * EO_DIST
                    nz = coords[-1][2] + test_bond[2] * EO_DIST

                    # Excluded volume check: distance > 5.0 Å for non-adjacent beads
                    # coords[:-1] checks all previous beads EXCEPT the immediately preceding one
                    collision = False
                    for cx, cy, cz in coords[:-1]:
                        if (nx - cx)**2 + (ny - cy)**2 + (nz - cz)**2 <= MIN_DIST_SQ:
                            collision = True
                            break

                    if not collision:
                        bond = test_bond
                        coords.append((nx, ny, nz))
                        placed = True
                        break

                if not placed:
                    stuck = True
                    break  # Chain got trapped, break out and restart the entire chain

            if not stuck:
                return coords
                
        raise RuntimeError(f"Failed to build a collision-free PEG chain after {max_restarts} attempts. Try a smaller n.")

    coords = generate_chain()

    out = f"{name}.pdb"
    with open(out, "w") as f:
        print("REMARK  iConRNA", file=f)
        print("REMARK  CREATE BY RNABUILDER/SHANLONG LI", file=f)
        print("REMARK  Ref: S. Li and J. Chen, PNAS, 2025, 122, e2504583122.", file=f)
        print(f"REMARK  SEQUENCE: PEG x{n}", file=f)
        for i, (cx, cy, cz) in enumerate(coords):
            atom = ["ATOM", i + 1, "EO", "PEG", "X", i + 1,
                    cx, cy, cz, 1.00, 0.00, "PEG"]
            printcg([atom], f)
        print("END", file=f)


def main():
    """Command-line interface (``iconbuilder`` entry point).

    Usage::

        iconbuilder NAME SEQ

    *SEQ* is interpreted in this order:

    1. Starts with lowercase ``d`` followed by a letter: DNA. The rest is
       upper-cased; ``<motif><count>`` (e.g. ``dA100``, ``dATCG20``) expands to
       *count* nucleotides by repeating the motif and truncating.
    2. ``<letters><count>``: ``P``/``p`` gives :func:`build_polyP` with *count*
       beads, ``EO`` (any case) gives :func:`build_peg` with *count* beads,
       anything else is an RNA motif repeated and truncated to *count*
       nucleotides (e.g. ``A100``, ``CAG50``).
    3. Letters only: RNA sequence (upper-cased).

    Writes ``NAME.pdb`` and prints a confirmation. Builders are called with
    their default ``seed=None``.

    Raises:
        ValueError: If *SEQ* matches none of the forms above, or the chosen
            builder rejects it (e.g. invalid nucleotide letters).
    """

    parser = argparse.ArgumentParser(description='NABuilder: build iConRNA/iConDNA from sequence')
    parser.add_argument('name', type=str, help='output name stem, produces name.pdb')
    parser.add_argument('seq', type=str, help=(
        'sequence in one-letter codes; '
        'RNA: A/U/C/G (e.g. AUCGAUCG or A100); '
        'DNA: lowercase d prefix (e.g. dATCG or dA100); '
        'polyP: P followed by count (e.g. P10); '
        'PEG: EO followed by count (e.g. EO20)'
    ))

    args = parser.parse_args()

    seq = args.seq

    # DNA mode: sequence starts with lowercase 'd' followed by letters
    # Requiring lowercase 'd' prevents ambiguity with uppercase nucleotide sequences.
    if seq.startswith('d') and len(seq) > 1 and seq[1].isalpha():
        dna_seq = seq[1:]  # strip leading 'd'

        # check for repeat shorthand: e.g. dA100 or dATCG100
        match = re.fullmatch(r'([A-Za-z]+?)(\d+)', dna_seq)
        if match:
            motif = match.group(1).upper()
            count = int(match.group(2))
            dna_seq = (motif * ((count // len(motif)) + 1))[:count]
        else:
            dna_seq = dna_seq.upper()

        build_dna(args.name, dna_seq)
        print(f"DNA structure saved to {args.name}.pdb")
        return

    # PolyP / RNA repeat shorthand: e.g. P10, A100, CAG50
    match = re.fullmatch(r'([A-Za-z]+?)(\d+)', seq)
    if match:
        motif = match.group(1)
        count = int(match.group(2))
        if motif.upper() == 'P':
            build_polyP(args.name, count)
            print(f"PolyP structure saved to {args.name}.pdb")
        elif motif.upper() == 'EO':
            build_peg(args.name, count)
            print(f"PEG structure saved to {args.name}.pdb")
        else:
            rna_seq = (motif.upper() * ((count // len(motif)) + 1))[:count]
            build_rna(args.name, rna_seq)
            print(f"RNA structure saved to {args.name}.pdb")
        return

    # Plain RNA sequence
    if seq.isalpha():
        build_rna(args.name, seq.upper())
        print(f"RNA structure saved to {args.name}.pdb")
        return

    raise ValueError(
        "Invalid sequence format.\n"
        "  RNA:   pure letter sequence, e.g. AUCGAUCG or A100\n"
        "  DNA:   lowercase d prefix, e.g. dATCGATCG or dA100\n"
        "  polyP: P followed by count, e.g. P10\n"
        "  PEG:   EO followed by count, e.g. EO20"
    )


if __name__ == '__main__':
    main()