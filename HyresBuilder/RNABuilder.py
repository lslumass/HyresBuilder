"""
this package is used to generate CG_RNA model
Athour: Shanlong Li
Date: Nov 13, 2023
"""

import sys
import argparse
import numpy as np
from pathlib import Path


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

def main():
    """Command-line interface"""
    
    parser = argparse.ArgumentParser(description='RNABuilder: build iConRNA from sequence')
    parser.add_argument('name', type=str, help='protein name, output: name.pdb')
    parser.add_argument('seq', type=str, help='sequence in one-letter')

    args = parser.parse_args()
    build(args.name, args.seq)
    print(f"RNA structure saved to {args.name}.pdb")

if __name__ == '__main__':
    main()

