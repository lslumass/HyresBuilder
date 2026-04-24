"""
De novo iConRNA coarse-grained RNA structure generation from sequence.

This module builds iConRNA coarse-grained RNA PDB files directly from a
single-letter nucleotide sequence, without requiring an all-atom input
structure. Each nucleotide is placed sequentially using a fixed set of
reference bead coordinates that define the iConRNA bead topology, and
residues are stacked along the z-axis with a 3.63 Å rise per step,
producing a canonical A-form–like helical geometry.

Bead topology
-------------
Reference bead coordinates are stored in the ``maps`` dictionary, keyed
by single-letter nucleotide code. Purines (A, G) carry seven beads
(P, C1, C2, NA, NB, NC, ND); pyrimidines (C, U) carry six (P, C1, C2,
NA, NB, NC). Three-letter residue names follow the iConRNA convention:
ADE, GUA, CYT, URA.

Build pipeline
--------------
The top-level function :func:`build` orchestrates the full workflow:

1. Iterate over the sequence and load the reference bead layout for each
   nucleotide (:func:`read_map`).
2. Translate the new residue so that its P bead aligns with the reference
   anchor point, which advances by 3.63 Å along z after each residue
   (:func:`transform`).
3. Accumulate all transformed beads and write the output PDB with iConRNA
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
    ]
}

def printcg(atoms, file):
    for atom in atoms:
        file.write('{}  {:5d} {:>2}   {} {}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {}\n'.format(atom[0],int(atom[1]),atom[2],atom[3],atom[4],int(atom[5]),atom[6],atom[7],atom[8],atom[9],atom[10], atom[11]))

def read_map(seq):
    atoms = []
    nos = {'A': 'ADE', 'G': 'GUA', 'C': 'CYT', 'U': 'URA'}
    for index, name, rx, ry, rz in maps[seq]:
        atom = ['ATOM', index, name, nos[seq], 'X', 1, rx, ry, rz, 1.00, 0.00, 'RNA']
        atoms.append(atom)
    return atoms

def transform(ref, atoms):
    refx, refy, refz = ref[0], ref[1], ref[2]
    Px, Py, Pz = atoms[0][6], atoms[0][7], atoms[0][8]
    dx, dy, dz = Px-refx, Py-refy, Pz-refz
    for atom in atoms:
        atom[6] -= dx
        atom[7] -= dy
        atom[8] -= dz
    return atoms

def build(name, sequence):
    """
    Build an iConRNA coarse-grained RNA structure from a nucleotide sequence.

    Each residue is placed sequentially by mapping its nucleotide type onto a
    set of reference bead coordinates (P, C1, C2, NA, NB, NC, and ND for purines).
    Residues are stacked along the z-axis with a 3.63 Å rise per residue.
    The finished structure is written as a PDB file with iConRNA REMARK headers.

    Args:
        name (str): Stem of the output file. The PDB is written to ``<name>.pdb``.
        sequence (str): RNA sequence in single-letter codes (e.g. ``'AUCG'``). Supported nucleotides: ``A``, ``U``, ``C``, ``G``.

    Returns:
        None. Writes a PDB file to ``<name>.pdb`` in the current working directory.

    Raises:
        KeyError: If any character in ``sequence`` is not one of ``A``, ``U``, ``C``, ``G``.

    Example:
        >>> from HyresBuilder import RNABuilder
        >>> RNABuilder.build("myrna", "AUCGAUCG")
        # output: myrna.pdb
    """

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
            atoms = read_map(seq)
            for atom in atoms:
                atom[1] += idx
                atom[5] += res
            atoms = transform(ref, atoms)
            ref = [atoms[1][6], atoms[1][7], atoms[1][8] + 3.63]
            idx += len(atoms)
            res += 1
            printcg(atoms, f)
        print('END', file=f)

def build_polyP(name, n, seed=None):
    """
    Build a poly-phosphate (polyP) coarse-grained structure of n residues.
 
    Each residue is a single PHO bead (P). Beads are placed via a random walk
    so that every P-P bond is exactly 2.7 Å but the chain is non-linear.
    The step direction is drawn uniformly from the unit sphere, giving a
    realistic disordered conformation. The output PDB uses residue name
    PHO and bead name P.
 
    Args:
        name (str): Stem of the output file. The PDB is written to <n>.pdb.
        n (int): Number of phosphate beads (residues) in the chain.
        seed (int, optional): Random seed for reproducibility. Default is None
            (non-reproducible).
 
    Returns:
        None. Writes a PDB file to <n>.pdb in the current working directory.
 
    Example:
        >>> from HyresBuilder import RNABuilder
        >>> RNABuilder.build_polyP("polyP", 10)
        # output: polyP.pdb
        >>> RNABuilder.build_polyP("polyP_rep", 10, seed=42)  # reproducible
    """
    import math
    import random
 
    PP_DIST = 2.7   # Å, fixed P-P bond length
 
    rng = random.Random(seed)
 
    def random_unit_vector():
        """Uniform random direction on the unit sphere (Marsaglia method)."""
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y >= 1:
                continue
            z = math.sqrt(1 - x*x - y*y) * rng.choice([-1, 1])
            return x, y, z
 
    def next_direction(prev_dir):
        """Return a random unit vector satisfying:
          - dz > 0  (z always increases along the chain)
          - dot(prev_dir, new_dir) > 0  (P-P-P angle > 90°)
        Both constraints are enforced by rejection sampling.
        prev_dir is None for the very first bond (only dz > 0 is required).
        """
        while True:
            d = random_unit_vector()
            if d[2] <= 0:                              # must go +z
                continue
            if prev_dir is not None:
                dot = d[0]*prev_dir[0] + d[1]*prev_dir[1] + d[2]*prev_dir[2]
                if dot <= 0:                           # P-P-P angle must be > 90°
                    continue
            return d
 
    # Build coordinates via constrained random walk
    x, y, z = 9000.0, 9000.0, 9000.0
    coords = [(x, y, z)]
    prev_dir = None
    for _ in range(n - 1):
        dx, dy, dz = next_direction(prev_dir)
        prev_dir = (dx, dy, dz)
        x += dx * PP_DIST
        y += dy * PP_DIST
        z += dz * PP_DIST
        coords.append((x, y, z))
 
    out = f"{name}.pdb"
    with open(out, "w") as f:
        print("REMARK  iConRNA", file=f)
        print("REMARK  CREATE BY RNABUILDER/SHANLONG LI", file=f)
        print("REMARK  Ref: S. Li and J. Chen, PNAS, 2025, 122, e2504583122.", file=f)
        print("REMARK  SEQUENCE: PHO x{}".format(n), file=f)
        for i, (cx, cy, cz) in enumerate(coords):
            atom = ["ATOM", i + 1, "P", "PHO", "X", i + 1,
                    cx, cy, cz, 1.00, 0.00, "RNAP"]
            printcg([atom], f)
        print("END", file=f)

def main():
    """Command-line interface"""
    
    parser = argparse.ArgumentParser(description='RNABuilder: build iConRNA from sequence')
    parser.add_argument('name', type=str, help='protein name, output: name.pdb')
    parser.add_argument('seq', type=str, help='sequence in one-letter, for polyP, use Pn (e.g. P10 for 10 residues)')

    args = parser.parse_args()

    seq = args.seq
    match = re.fullmatch(r'([A-Za-z])(\d+)', seq)
    if match:
        letter = match.group(1)
        count = int(match.group(2))
        if letter == 'P':
            build_polyP(args.name, count)
            print(f"PolyP structure saved to {args.name}.pdb")
        else:
            seq = letter * count
            build(args.name, seq)
            print(f"RNA structure saved to {args.name}.pdb")
    elif seq.isalpha():
        build(args.name, seq)
        print(f"RNA structure saved to {args.name}.pdb")
    else:
        raise ValueError("Invalid sequence format. Use either a pure letter sequence (e.g. AUCGAUCG) or a letter followed by a number (i.e. A100, CAG100, P10).")
    

if __name__ == '__main__':
    main()