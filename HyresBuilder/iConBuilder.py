"""
De novo iConRNA coarse-grained RNA/DNA structure generation from sequence.

This module builds iConRNA coarse-grained RNA and DNA PDB files directly from a
single-letter nucleotide sequence, without requiring an all-atom input
structure. Each nucleotide is placed sequentially using a fixed set of
reference bead coordinates that define the iConRNA bead topology, and
residues are stacked along the z-axis with a 3.63 Å rise per step,
producing a canonical A-form–like helical geometry.

Bead topology
-------------
Reference bead coordinates are stored in the ``maps`` dictionary, keyed
by single-letter nucleotide code. Purines (A, G) carry seven beads
(P, C1, C2, NA, NB, NC, ND); pyrimidines (C, U, T) carry six
(P, C1, C2, NA, NB, NC). Three-letter residue names follow the iConRNA
convention:

  RNA: ADE, GUA, CYT, URA
  DNA: DAD, DGU, DCY, DTH

Build pipeline
--------------
The top-level functions :func:`build_rna` (RNA) and :func:`build_dna` (DNA)
orchestrate the full workflow:

1. Validate the sequence against the allowed nucleotide alphabet.
2. Iterate over the sequence and load the reference bead layout for each
   nucleotide (:func:`readRNAmap` / :func:`readDNAmap`).
3. Translate the new residue so that its P bead aligns with the reference
   anchor point, which advances by 3.63 Å along z after each residue
   (:func:`transform`).
4. Accumulate all transformed beads and write the output PDB with iConRNA
   REMARK headers.

A command-line interface is exposed via :func:`main` and registered as
the ``RNABuilder`` entry point.

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
    for atom in atoms:
        file.write('{}  {:5d} {:>2}   {} {}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:<4}\n'.format(
            atom[0], int(atom[1]), atom[2], atom[3], atom[4], int(atom[5]),
            atom[6], atom[7], atom[8], atom[9], atom[10], atom[11]))


def readRNAmap(seq):
    atoms = []
    nos = {'A': 'ADE', 'G': 'GUA', 'C': 'CYT', 'U': 'URA'}
    for index, name, rx, ry, rz in maps[seq]:
        atom = ['ATOM', index, name, nos[seq], 'X', 1, rx, ry, rz, 1.00, 0.00, 'RNA']
        atoms.append(atom)
    return atoms


def readDNAmap(seq):
    atoms = []
    nos = {'A': 'DAD', 'G': 'DGU', 'C': 'DCY', 'T': 'DTH'}
    for index, name, rx, ry, rz in maps[seq]:
        atom = ['ATOM', index, name, nos[seq], 'X', 1, rx, ry, rz, 1.00, 0.00, 'DNA']
        atoms.append(atom)
    return atoms


def transform(ref, atoms):
    refx, refy, refz = ref[0], ref[1], ref[2]
    Px, Py, Pz = atoms[0][6], atoms[0][7], atoms[0][8]
    dx, dy, dz = Px - refx, Py - refy, Pz - refz
    for atom in atoms:
        atom[6] -= dx
        atom[7] -= dy
        atom[8] -= dz
    return atoms


def _build(name, sequence, map_func, molecule):
    """Shared build core used by build_rna and build_dna."""
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

    Each residue is placed sequentially by mapping its nucleotide type onto a
    set of reference bead coordinates (P, C1, C2, NA, NB, NC, and ND for purines).
    Residues are stacked along the z-axis with a 3.63 Å rise per residue.
    The sequence is validated before any file is written. The finished structure
    is written as a PDB file with iConRNA REMARK headers.

    Args:
        name (str): Stem of the output file. The PDB is written to ``<name>.pdb``.
            Must not be empty.
        sequence (str): RNA sequence in single-letter codes (e.g. ``'AUCG'``).
            Case-insensitive. Supported nucleotides: ``A``, ``U``, ``C``, ``G``.

    Returns:
        None. Writes a PDB file to ``<name>.pdb`` in the current working directory.

    Raises:
        ValueError: If *name* is empty or *sequence* contains unsupported bases.

    Example:
        >>> from HyresBuilder import RNABuilder
        >>> RNABuilder.build_rna("myrna", "AUCGAUCG")
        # output: myrna.pdb
    """
    _build(name, sequence, readRNAmap, 'RNA')


def build_dna(name, sequence):
    """
    Build an iConRNA coarse-grained DNA structure from a nucleotide sequence.

    Each residue is placed sequentially by mapping its nucleotide type onto a
    set of reference bead coordinates (P, C1, C2, NA, NB, NC, and ND for purines).
    Residues are stacked along the z-axis with a 3.63 Å rise per residue.
    The sequence is validated before any file is written. The finished structure
    is written as a PDB file with iConRNA REMARK headers.

    Three-letter residue names follow the iConRNA DNA convention:
    DAD (A), DGU (G), DCY (C), DTH (T).

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
        >>> from HyresBuilder import RNABuilder
        >>> RNABuilder.build_dna("mydna", "ATCGATCG")
        # output: mydna.pdb
    """
    _build(name, sequence, readDNAmap, 'DNA')


def build_polyP(name, n, seed=None):
    """
    Build a poly-phosphate (polyP) coarse-grained structure of n residues.

    Each residue is a single PHO bead (P). Beads are placed via a random walk
    so that every P-P bond is exactly 2.7 Å but the chain is non-linear.
    The step direction is drawn uniformly from the unit sphere, giving a
    realistic disordered conformation with a general +z propagation and 
    P-P-P angles strictly > 90°.
    
    Self-Avoiding Constraint: Excluded volume is enforced by ensuring any 
    non-adjacent bead pair maintains a distance strictly greater than 4.0 Å. 
    (Note: A 5.0 Å limit is not used here because the P-P bond is only 2.7 Å; 
    a 5.0 Å constraint would physically force all bond angles to be > 135°).

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
        RuntimeError: If a collision-free chain cannot be generated.

    Example:
        >>> from HyresBuilder import RNABuilder
        >>> RNABuilder.build_polyP("polyP", 10)
        # output: polyP.pdb
        >>> RNABuilder.build_polyP("polyP_rep", 10, seed=42)  # reproducible
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
        """Uniform random direction on the unit sphere (Marsaglia method)."""
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x * x + y * y >= 1:
                continue
            z = math.sqrt(1 - x * x - y * y) * rng.choice([-1, 1])
            return x, y, z

    def next_direction(prev_dir):
        """Return a candidate random unit vector satisfying local angle constraints."""
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
        """Generates the chain, restarting if it gets trapped in a steric clash."""
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

    Each repeat unit is represented by a single EO bead (residue name PEG,
    bead name EO). The chain is built as a freely-rotating chain (FRC): every
    EO-EO bond is exactly 3.5 Å and every EO-EO-EO bond angle is fixed at
    exactly 123°, matching the C-C-O / C-O-C backbone geometry of PEG.
    
    Self-Avoiding Constraint: Any non-adjacent bead pair is guaranteed to have 
    a distance strictly greater than 0.5 nm (5.0 Å).

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
        RuntimeError: If a collision-free chain cannot be generated.
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
        """Uniform random direction on the unit sphere (Marsaglia method)."""
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x * x + y * y >= 1:
                continue
            z = math.sqrt(1 - x * x - y * y) * rng.choice([-1, 1])
            return (x, y, z)

    def perp_vector(v):
        """Return an arbitrary unit vector perpendicular to v."""
        ax = (0.0, 0.0, 1.0) if abs(v[0]) < 0.9 or abs(v[1]) < 0.9 else (1.0, 0.0, 0.0)
        cx = v[1] * ax[2] - v[2] * ax[1]
        cy = v[2] * ax[0] - v[0] * ax[2]
        cz = v[0] * ax[1] - v[1] * ax[0]
        norm = math.sqrt(cx * cx + cy * cy + cz * cz)
        return (cx / norm, cy / norm, cz / norm)

    def next_bond(prev_bond):
        """Return a unit vector for the next bond."""
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
        """Generates the chain, restarting if it gets trapped in a steric clash."""
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
    """Command-line interface"""

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