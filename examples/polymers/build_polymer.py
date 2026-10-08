"""
Build a coarse-grained methacrylate polymer chain (forcefield/top_polymer.inp).

Monomers:
    QDM  quaternized DMAEMA         BB (MMA), SC1 (MAO), SC2 (NC4)
    BZM  2-phenylethyl methacrylate BB (MMA), SC1 (MAO), SC2 (BZ1), SC3 (BZ2), SC4 (BZ2)

Beads are placed at the equilibrium geometry of the force field with random torsions,
and every non-bonded bead pair is kept further apart than MIN_DIST:
    BB-BB 2.79 A, BB-BB-BB 145 deg, BB-SC1 2.82 A
    QDM: SC1-SC2 4.00 A, BB-SC1-SC2 135 deg
    BZM: SC1-SC2 3.90 A, BB-SC1-SC2 135 deg, SC2-SC3/SC4 2.18 A, SC3-SC4 1.73 A,
         SC1-SC2-SC3/SC4 130 deg

The sequence is a '-'-separated list of residue names, each optionally followed by a
repeat count, e.g. QDM50 or QDM5-BZM-QDM5-BZM.

usage: python build_polymer.py SEQUENCE -o NAME [--seed 2026]
e.g.:  python build_polymer.py QDM50 -o qPDMAEMA50
       python build_polymer.py QDM5-BZM-QDM5-BZM-QDM5-BZM-QDM5-BZM-QDM5-BZM -o qPDMAEMA25-PBzMA5
then:  genpsf NAME.pdb NAME.psf
"""

import argparse
import re
import numpy as np

BB_BB, BB_SC1 = 2.79, 2.82                    # bond lengths, A
ANG_BB, ANG_SC = 145.0, 135.0                 # BB-BB-BB and BB-SC1-SC2 angles, deg
QDM_SC1_SC2 = 4.00
BZM_SC1_SC2, BZM_RING, BZM_RING_EE = 3.90, 2.18, 1.73
ANG_RING = 130.0                              # SC1-SC2-SC3/SC4, deg
MIN_DIST = 4.0                                # minimum distance of non-bonded beads, A

ATOMS = {'QDM': ['BB', 'SC1', 'SC2'],
         'BZM': ['BB', 'SC1', 'SC2', 'SC3', 'SC4']}


def unit(v):
    return v / np.linalg.norm(v)


def perp(v, rng):
    """Random unit vector perpendicular to v."""
    p = np.cross(v, rng.normal(size=3))
    return unit(p)


def next_dir(prev, angle, rng):
    """Unit vector making a bond angle `angle` (deg) with the previous bond `prev`, random torsion."""
    tilt = np.radians(180.0 - angle)
    return np.cos(tilt) * prev + np.sin(tilt) * perp(prev, rng)


def clash(x, coords, skip=()):
    if not coords:
        return False
    d = np.linalg.norm(np.array(coords) - x, axis=1)
    d[list(skip)] = np.inf
    return bool((d < MIN_DIST).any())


def parse_sequence(seq):
    residues = []
    for token in seq.split('-'):
        m = re.fullmatch(r'([A-Z]{3})(\d*)', token.strip())
        if not m or m.group(1) not in ATOMS:
            raise ValueError(f"Invalid monomer '{token}', supported: {', '.join(ATOMS)}")
        residues += [m.group(1)] * int(m.group(2) or 1)
    return residues


def ring(s2, d2, rng):
    """SC3 and SC4 of BZM: both at ANG_RING to SC1, forming the SC2-SC3-SC4 triangle."""
    alpha = np.radians(180.0 - ANG_RING)                      # cone angle around the SC1->SC2 bond
    gamma = 2 * np.arcsin(BZM_RING_EE / 2 / BZM_RING)         # angle SC3-SC2-SC4
    dphi = np.arccos((np.cos(gamma) - np.cos(alpha)**2) / np.sin(alpha)**2)
    e1 = perp(d2, rng)
    e2 = np.cross(d2, e1)
    return [s2 + BZM_RING * (np.cos(alpha) * d2 + np.sin(alpha) * (np.cos(phi) * e1 + np.sin(phi) * e2))
            for phi in (-dphi / 2, dphi / 2)]


def build_backbone(n, rng, tries=100):
    bb = [np.zeros(3)]
    bond = unit(rng.normal(size=3))
    for i in range(1, n):
        for _ in range(tries):
            d = bond if i == 1 else next_dir(bond, ANG_BB, rng)
            x = bb[-1] + BB_BB * d
            if not clash(x, bb, skip=[i - 1]):
                bb.append(x)
                bond = d
                break
        else:
            return None
    return bb


def add_sidechains(bb, residues, rng, tries=500):
    n = len(bb)
    beads = list(bb)                     # bead i of the backbone is beads[i]
    sc = []
    for i, res in enumerate(residues):
        # point the side chain away from the backbone neighbours
        nbr = [bb[j] for j in (i - 1, i + 1) if 0 <= j < n]
        out = unit(sum(bb[i] - x for x in nbr))
        for _ in range(tries):
            d1 = unit(out + 0.8 * perp(out, rng))
            s1 = bb[i] + BB_SC1 * d1
            if res == 'QDM':
                new = [s1, s1 + QDM_SC1_SC2 * next_dir(d1, ANG_SC, rng)]
            else:
                d2 = next_dir(d1, ANG_SC, rng)
                s2 = s1 + BZM_SC1_SC2 * d2
                new = [s1, s2] + ring(s2, d2, rng)
            if not clash(s1, beads, skip=[i]) and not any(clash(x, beads) for x in new[1:]):
                beads += new
                sc.append(new)
                break
        else:
            return None
    return sc


def build(residues, seed, restarts=200):
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        bb = build_backbone(len(residues), rng)
        if bb is None:
            continue
        sc = add_sidechains(bb, residues, rng)
        if sc is not None:
            return bb, sc
    raise RuntimeError(f"Failed to build a clash-free chain of {len(residues)} monomers.")


def write_pdb(name, seq, residues, bb, sc):
    center = np.mean(bb, axis=0)
    with open(f"{name}.pdb", "w") as f:
        f.write("REMARK  CG methacrylate polymer (forcefield/top_polymer.inp)\n")
        f.write(f"REMARK  SEQUENCE: {seq}\n")
        serial = 0
        for i, (res, b, side) in enumerate(zip(residues, bb, sc), start=1):
            for atom, x in zip(ATOMS[res], [b] + side):
                serial += 1
                x = x - center
                f.write("ATOM  {:5d} {:<4s} {:<3s} {}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:<4s}\n".format(
                    serial, " " + atom, res, "A", i, x[0], x[1], x[2], 1.0, 0.0, "S001"))
        f.write("END\n")


def main():
    parser = argparse.ArgumentParser(description="Build a CG methacrylate polymer chain")
    parser.add_argument("seq", help="sequence, e.g. QDM50 or QDM5-BZM-QDM5-BZM")
    parser.add_argument("-o", "--out", required=True, help="output name stem, writes NAME.pdb")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    args = parser.parse_args()

    residues = parse_sequence(args.seq)
    bb, sc = build(residues, args.seed)
    write_pdb(args.out, args.seq, residues, bb, sc)
    print(f"{len(residues)} monomers written to {args.out}.pdb")


if __name__ == "__main__":
    main()
