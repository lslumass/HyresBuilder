"""
this package is used to generate CG_RNA model
Athour: Shanlong Li
Date: Nov 13, 2023
"""

import sys
import argparse
import numpy as np
from pathlib import Path

nos = {'A': 'ADE', 'G': 'GUA', 'C': 'CYT', 'U': 'URA'}
def printcg(atoms, file):
    for atom in atoms:
        file.write('{}  {:5d} {:>2}   {} {}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {}\n'.format(atom[0],int(atom[1]),atom[2],atom[3],atom[4],int(atom[5]),atom[6],atom[7],atom[8],atom[9],atom[10], atom[11]))

def read_map(seq):
    atoms = []
    filename = "map/"+seq+".map"
    f_map = Path(__file__).parent / filename
    data = np.genfromtxt(f_map, dtype=None, names=('index', 'name', 'rx', 'ry', 'rz'), encoding='utf-8')
    for l in data:
        atom = ['ATOM', l[0], l[1], nos[seq], 'X', 1, l[2], l[3], l[4], 1.00, 0.00, 'RNA']
        atoms.append(atom)
    return atoms

def transform(ref, atoms, theta):
    refx, refy, refz = ref[0], ref[1], ref[2]
    Px, Py, Pz = atoms[0][6], atoms[0][7], atoms[0][8]
    dx, dy, dz = Px-refx, Py-refy, Pz-refz
    for atom in atoms:
        atom[6] -= dx
        atom[7] -= dy
        atom[8] -= dz

    axis = np.array([dx, dy, dz])
    v_normal = axis / np.linalg.norm(axis)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    for atom in atoms:
        crds = np.array([atom[6], atom[7], atom[8]])
        cross, dot = np.cross(v_normal, crds), np.dot(v_normal, crds)
        new_crds = (crds * cos_theta + cross*sin_theta + v_normal*dot[:, np.newaxis]*(1-cos_theta))
        atom[6], atom[7], atom[8] = new_crds[0], new_crds[1], new_crds[2]
    return atoms

def build(seqs, out):
    with open(out, 'w') as f:
        print('REMARK  HyRes RNA', file=f)
        print('REMARK  CREATE BY RNABUILDER/SHANLONG LI', file=f)
        idx = 0
        res = 0
        ref = [9000.0, 9000.0, 9000.0]
        for seq in seqs:
            atoms = read_map(seq)
            for atom in atoms:
                atom[1] += idx
                atom[5] += res
            atoms = transform(ref, atoms, np.pi/4)
            ref = [atoms[1][6], atoms[1][7], atoms[1][8] + 3.63]
            idx += len(atoms)
            res += 1
            printcg(atoms, f)
        print('END', file=f)
