"""
Residue mutation utilities for HyRes and iConRNA coarse-grained PDB structures.

This module applies in silico point mutations to coarse-grained PDB files
produced by the HyRes (protein) and iConRNA (RNA) pipelines. Mutations are
performed via ``psfbuilder``: the mutated residues are rebuilt from the topology,
coordinates of atoms not present in the input are guessed, and the resulting
structure is written as a PDB (no PSF is written).

The whole input PDB is treated as a single segment, which must be either a
protein or an RNA chain, selected by segment ID prefix (``P`` for protein,
``R`` for RNA). The HyRes protein, iConRNA (``top_RNA_mix``), metabolite and
adduct topologies are loaded, so mutation types must be residue names defined
there (e.g. ``ALA``, ``GLY``, the dye-labelled cysteines ``A48``/``A59``/``C3B``/``C66`` for
protein; ``ADE``, ``GUA``, ``CYT``, ``URA`` for RNA). Multiple sites can be mutated in a single call, and a single mutation
type can be broadcast across all target sites.

Workflow
--------
1. Normalise sites and mutation types into matched lists (:func:`mutate`, or
   :func:`main` on the command line).
2. Load the protein and RNA topology files via ``utils.load_ff``.
3. Register the input PDB as a ``psfbuilder`` segment with the ``mutate`` table
   applied at the ``add_segment`` step (:func:`mut`).
4. Read the input coordinates, guess missing coordinates, place the phosphate
   bead of phosphorylated residues (:func:`_place_phosphate`) and the dye of
   dye-labelled cysteines (:func:`_place_adduct`), regenerate angles and
   dihedrals, and write the output PDB.

Phosphorylation
---------------
Mutating to ``SEP`` (phosphoserine), ``TPO`` (phosphothreonine) or ``PTR``
(phosphotyrosine) adds a phosphate bead (type ``PO3``) that ``guess_coords`` can only place
roughly, because the HyRes topology has no internal-coordinate tables. It is
placed at the force-field bond length and angle (``PHOSPHO_BEADS``), with
the free rotation about the anchor bond chosen to keep it as far as possible
from all other beads. The mutated site must already have the anchor and
reference beads in the input: CA and CB for SEP/TPO (e.g. from SER or THR),
CB and CD for PTR (from TYR or PHE).

Dye labels
----------
Mutating to ``A48`` (Alexa 488), ``A59`` (Alexa 594), ``C3B`` (Cy3B) or ``C66``
(CF660R) on cysteine adds 17, 23, 19 or 24 dye beads. They are placed from a CG template (``ADDUCT_TEMPLATES``): one
conformation of the dye, mapped (bead centres of mass, ``cgbuilder.ndx``) from
the all-atom A488-Cys / A594-Cys trajectories, with the frame whose
phenyl+xanthene part is closest to the trajectory average (C3B: RDKit Cy3B-Cys
ensemble; C66: FRETpredict CF660R_C2R library; frame closest to the mean linker geometry). The template is
superimposed on the N, CA and CB of the site, and of 24 rotations (every 15
deg) about the CA-CB bond the one farthest from all other beads is used. The
phenyl+xanthene geometry is therefore a proper starting point for a rigid
body. The site must have N, CA and CB in the input (any residue but GLY).

A command-line interface is exposed via :func:`main` as the ``mutate``
console script, taking the input and output PDB as positional arguments and
sites/mutations via ``-s``/``-m``.

Authors:    Shanlong Li
Date:       Feb 14, 2026

Dependencies
------------
* ``HyresBuilder.psfbuilder`` (pure-Python, psfgen-compatible ``PsfGen``)
* HyresBuilder force-field topology files, loaded via ``utils.load_ff``.
"""

from HyresBuilder.psfbuilder import PsfGen
from HyresBuilder import utils
import argparse
import sys
from pathlib import Path
import numpy as np


# Phosphate bead of phosphorylated residues: (new bead, anchor bead, reference bead,
# bond anchor-new in A, angle reference-anchor-new in deg), as in forcefield/param_hyres_mix.inp
PHOSPHO_BEADS = {
    'SEP': ('CC', 'CB', 'CA', 2.53, 132.8),   # P2S-PO3, C1E-P2S-PO3
    'TPO': ('CC', 'CB', 'CA', 3.20, 72.6),    # P2T-PO3, C1E-P2T-PO3
    'PTR': ('CE', 'CD', 'CB', 2.45, 145.0),   # P1Y-PO3, A1Y-P1Y-PO3
}

# CG templates of dye-labelled cysteines (A, CA at the origin): N, CA, CB and the dye beads, from
# one frame of the all-atom A488-Cys / A594-Cys trajectories mapped with cgbuilder.ndx (bead COM)
ADDUCT_TEMPLATES = {
    'A48': {
        'N': (0.510, -1.030, -0.880),
        'CA': (0.000, 0.000, 0.000),
        'CB': (1.479, -0.281, 1.800),
        'A01': (3.301, -0.214, 4.162),
        'A02': (3.058, -3.055, 4.452),
        'A03': (4.729, -1.613, 5.285),
        'A04': (6.156, -0.464, 7.884),
        'A05': (5.809, 2.455, 8.471),
        'A06': (5.206, 3.868, 6.149),
        'A07': (6.277, 5.736, 6.152),
        'A08': (4.948, 5.189, 4.489),
        'A09': (7.161, 7.824, 5.682),
        'A11': (4.977, 6.977, 3.023),
        'A12': (7.399, 5.558, 1.704),
        'A13': (7.508, 6.575, -0.400),
        'A14': (6.008, 8.444, -1.572),
        'A15': (4.590, 8.016, 1.478),
        'A16': (1.673, 9.683, 3.089),
        'A17': (2.308, 10.410, 0.666),
        'A18': (2.648, 8.123, 4.624),
    },
    'A59': {
        'N': (0.780, -0.540, 1.090),
        'CA': (0.000, 0.000, 0.000),
        'CB': (-0.718, -2.020, -0.927),
        'A01': (-0.829, -4.812, -1.825),
        'A02': (-3.280, -3.933, -3.082),
        'A03': (-2.804, -5.957, -2.083),
        'A04': (-3.812, -6.949, 0.629),
        'A05': (-6.830, -8.398, 2.326),
        'A06': (-8.110, -7.673, 4.566),
        'A07': (-10.126, -8.465, 4.551),
        'A08': (-9.353, -7.302, 6.254),
        'A09': (-12.314, -9.049, 4.993),
        'A10': (-11.020, -7.157, 7.847),
        'A11': (-11.959, -4.687, 6.454),
        'A12': (-12.609, -2.938, 4.888),
        'A13': (-10.184, -3.134, 3.610),
        'A14': (-13.908, -0.824, 6.896),
        'A15': (-13.895, -2.298, 8.565),
        'A16': (-12.897, -4.124, 8.395),
        'A17': (-11.908, -6.687, 9.468),
        'A18': (-11.315, -9.341, 10.724),
        'A19': (-11.337, -10.624, 12.405),
        'A20': (-10.583, -12.750, 12.161),
        'A21': (-9.712, -12.314, 9.192),
        'A22': (-7.398, -11.968, 7.806),
        'A23': (-10.373, -9.960, 8.867),
    },
    # Cy3B / CF660R C2R-maleimide (FRET/dye_mapping/new_dyes.py): C3B from the RDKit Cy3B ensemble,
    # C66 from the FRETpredict CF660R_C2R library; frame closest to the mean linker geometry
    'C3B': {
        'N': (0.979, 1.068, -0.162),
        'CA': (0.000, 0.000, 0.000),
        'CB': (0.189, -0.814, -2.215),
        'A01': (-1.987, 0.624, -2.712),
        'A02': (-3.384, -1.876, -1.948),
        'A03': (-3.911, 0.375, -1.385),
        'A04': (-5.163, 0.969, 1.335),
        'A05': (-7.941, 2.145, 1.982),
        'A06': (-8.431, 3.933, 0.367),
        'A07': (-10.149, 2.659, 0.358),
        'A08': (-11.689, 0.811, 0.816),
        'A09': (-11.695, 2.863, -0.668),
        'A10': (-12.254, 4.533, -1.913),
        'A11': (-14.361, 1.418, -1.279),
        'A12': (-14.892, 2.983, -2.400),
        'A13': (-17.827, 2.018, -2.650),
        'A14': (-17.159, 0.142, -1.798),
        'A15': (-15.940, -1.573, -0.438),
        'A16': (-18.144, -1.441, -1.452),
        'A17': (-20.143, -1.841, -2.120),
        'A18': (-18.906, -3.292, -0.928),
        'A19': (-20.453, -5.154, -0.750),
    },
    'C66': {
        'N': (1.472, 0.000, 0.000),
        'CA': (0.000, 0.000, 0.000),
        'CB': (-1.851, -0.397, 1.306),
        'A01': (-3.503, -2.355, 1.834),
        'A02': (-4.877, -1.240, 4.221),
        'A03': (-5.608, -2.924, 2.721),
        'A04': (-8.468, -2.585, 1.495),
        'A05': (-10.274, -5.191, 0.892),
        'A06': (-8.691, -7.779, 1.732),
        'A07': (-9.423, -8.824, 3.795),
        'A08': (-8.940, -10.051, 2.499),
        'A09': (-10.143, -12.718, 3.060),
        'A10': (-10.389, -14.655, 4.994),
        'A11': (-8.178, -10.440, 0.205),
        'A12': (-11.016, -9.966, -0.878),
        'A13': (-13.377, -9.541, -1.258),
        'A14': (-13.286, -7.287, 0.505),
        'A15': (-13.507, -9.969, -4.414),
        'A16': (-11.391, -11.594, -4.693),
        'A17': (-10.236, -10.812, -2.804),
        'A18': (-7.615, -11.182, -1.485),
        'A19': (-4.850, -11.740, -0.603),
        'A20': (-3.057, -13.103, -1.235),
        'A21': (-1.322, -12.166, 0.572),
        'A22': (-3.432, -11.062, 2.734),
        'A23': (-2.724, -10.575, 5.198),
        'A24': (-5.378, -10.890, 1.403),
    },
}
ADDUCT_ANCHORS = ('N', 'CA', 'CB')


def _input_atoms(pdb_in):
    """Atom names of each residue (keyed by resid string) in the input PDB."""
    atoms = {}
    with open(pdb_in) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atoms.setdefault(line[22:27].strip(), set()).add(line[12:16].strip())
    return atoms


def _place_phosphate(gen, segid, resid, resname, input_atoms):
    """
    Place the phosphate bead of a SEP, TPO or PTR residue built by psfbuilder.

    The bead is put at the bond length and angle of ``PHOSPHO_BEADS`` from
    its anchor bead; of 24 rotations (every 15 deg) about the reference-anchor
    bond, the one farthest from all other beads in the segment is used.

    Args:
        gen (PsfGen): psfbuilder session holding the mutated segment.
        segid (str): Segment ID.
        resid (str): Residue ID of the phosphorylated residue.
        resname (str): ``'SEP'``, ``'TPO'`` or ``'PTR'``.
        input_atoms (set of str): Atom names of this residue in the input PDB.

    Returns:
        float: Distance (A) from the placed bead to the closest other bead.

    Raises:
        ValueError: If the anchor or reference bead was not in the input
            residue (its position would only be guessed).
    """
    bead, anchor, ref, bond, angle = PHOSPHO_BEADS[resname]
    missing = [a for a in (ref, anchor) if a not in input_atoms]
    if missing:
        raise ValueError(f"Residue {resid} -> {resname}: bead(s) {', '.join(missing)} not in the input "
                         f"residue, so the phosphate bead cannot be placed. Mutate from a residue that "
                         f"has {ref} and {anchor} (SEP/TPO: e.g. SER or THR; PTR: TYR or PHE).")
    xyz = dict(zip(gen.get_atom_names(segid, resid), (np.array(c) for c in gen.get_coordinates(segid, resid))))
    others = np.array([c for r in gen.get_resids(segid)
                       for n, c in zip(gen.get_atom_names(segid, r), gen.get_coordinates(segid, r))
                       if not (str(r) == str(resid) and n == bead)])
    e = xyz[anchor] - xyz[ref]
    e /= np.linalg.norm(e)
    p1 = np.cross(e, [1.0, 0.0, 0.0] if abs(e[0]) < 0.9 else [0.0, 1.0, 0.0])
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(e, p1)
    tilt = np.radians(180.0 - angle)
    best, best_d = None, -1.0
    for phi in np.radians(np.arange(0, 360, 15)):
        x = xyz[anchor] + bond * (np.cos(tilt) * e + np.sin(tilt) * (np.cos(phi) * p1 + np.sin(phi) * p2))
        d = np.min(np.linalg.norm(others - x, axis=1))
        if d > best_d:
            best, best_d = x, d
    gen.set_position(segid, resid, bead, tuple(float(v) for v in best))
    return best_d


def _kabsch(P, Q):
    """Rotation matrix R minimising |P @ R - Q| for centred (n, 3) arrays P and Q."""
    V, _, W = np.linalg.svd(P.T @ Q)
    if np.linalg.det(V @ W) < 0:
        V[:, -1] *= -1
    return V @ W


def _place_adduct(gen, segid, resid, resname, input_atoms):
    """
    Place the dye beads of a dye residue (A48, A59, C3B, C66) built by psfbuilder.

    The template of ``ADDUCT_TEMPLATES`` is oriented by superimposing its N,
    CA and CB on those of the residue and anchored on the residue's CB; of 24 rotations (every 15 deg) of the dye about the CA-CB
    bond, the one farthest from all other beads of the segment is used.

    Args:
        gen (PsfGen): psfbuilder session holding the mutated segment.
        segid (str): Segment ID.
        resid (str): Residue ID of the dye-labelled residue.
        resname (str): a key of ``ADDUCT_TEMPLATES`` (``'A48'``, ``'A59'``, ``'C3B'``, ``'C66'``).
        input_atoms (set of str): Atom names of this residue in the input PDB.

    Returns:
        float: Distance (A) from the dye to the closest bead of another residue.

    Raises:
        ValueError: If N, CA or CB was not in the input residue.
    """
    missing = [a for a in ADDUCT_ANCHORS if a not in input_atoms]
    if missing:
        raise ValueError(f"Residue {resid} -> {resname}: bead(s) {', '.join(missing)} not in the input "
                         f"residue, so the dye cannot be placed. Mutate from a residue that has "
                         f"N, CA and CB (e.g. CYS; not GLY).")
    tpl = {n: np.array(x) for n, x in ADDUCT_TEMPLATES[resname].items()}
    xyz = dict(zip(gen.get_atom_names(segid, resid), (np.array(c) for c in gen.get_coordinates(segid, resid))))
    P = np.array([tpl[a] for a in ADDUCT_ANCHORS])
    Q = np.array([xyz[a] for a in ADDUCT_ANCHORS])
    R = _kabsch(P - P.mean(0), Q - Q.mean(0))
    dye = [n for n in tpl if n not in ADDUCT_ANCHORS]
    D = (np.array([tpl[n] for n in dye]) - tpl['CB']) @ R + xyz['CB']   # oriented by N/CA/CB, anchored on CB
    others = np.array([c for r in gen.get_resids(segid) if str(r) != str(resid)
                       for c in gen.get_coordinates(segid, r)]).reshape(-1, 3)
    own = np.array([xyz[n] for n in xyz if n not in dye])
    e = xyz['CB'] - xyz['CA']
    e /= np.linalg.norm(e)
    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    best, best_d = None, -1.0
    for phi in np.radians(np.arange(0, 360, 15)):
        Rot = np.eye(3) + np.sin(phi) * K + (1 - np.cos(phi)) * K @ K     # Rodrigues, about CA-CB
        X = (D - xyz['CB']) @ Rot.T + xyz['CB']
        d_other = np.min(np.linalg.norm(others[:, None] - X[None], axis=-1)) if len(others) else np.inf
        d_own = np.min(np.linalg.norm(own[:, None] - X[None], axis=-1))
        d = min(d_other, d_own + 1.0)                                     # own N/C/O/H may sit closer
        if d > best_d:
            best, best_d, best_other = X, d, d_other
    for n, x in zip(dye, best):
        gen.set_position(segid, resid, n, tuple(float(v) for v in x))
    return best_other


def mutate(pdb_in, sites, mutations, pdb_out=None, segid='P001'):
    """
    Perform mutations on a CG PDB file (convenience wrapper around :func:`mut`).

    ``sites``/``mutations`` may be scalars or sequences; a single mutation
    type is broadcast to every site. Mutation codes are upper-cased by
    :func:`mut`.

    Args:
        pdb_in  (str):              Input PDB file path.
        sites   (int | list[int]):  Residue number(s) to mutate.
        mutations (str | list[str]):Mutation type(s) as 3-letter codes (e.g. 'ALA').
                                    A single value is broadcast to all sites.
        pdb_out (str, optional):    Output PDB file path. Defaults to
                                    '<pdb_in stem>_mut.pdb' in the same directory.
        segid   (str):              Segment ID assigned to the whole input —
                                    e.g. 'P001' for protein, 'R001' for RNA.
                                    Must start with 'P' or 'R'. Default: 'P001'.

    Returns:
        str: Path to the written output PDB file.

    Raises:
        FileNotFoundError: If pdb_in does not exist.
        ValueError:        If sites/mutations lengths are mismatched, or segid is invalid.

    Examples:
        >>> from HyresBuilder.Mutator import mutate

        >>> # Single mutation
        >>> mutate('protein.pdb', 123, 'ALA')

        >>> # Multiple mutations, explicit output path
        >>> mutate('protein.pdb', [10, 20, 30], ['ALA', 'GLY', 'TRP'],
        ...        pdb_out='protein_mut.pdb')

        >>> # Broadcast one mutation type to every site
        >>> mutate('protein.pdb', [10, 20, 30], 'ALA')

        >>> # RNA mutation
        >>> mutate('rna.pdb', [10, 20], ['GUA', 'CYT'], segid='R001')
    """
    # ---------- normalise sites / mutations to lists ----------
    if isinstance(sites, int):
        sites = [sites]
    if isinstance(mutations, str):
        mutations = [mutations]

    sites     = list(sites)
    mutations = list(mutations)

    # broadcast single mutation type to all sites
    if len(mutations) == 1 and len(sites) > 1:
        mutations = mutations * len(sites)

    # ---------- default output path ----------
    if pdb_out is None:
        p = Path(pdb_in)
        pdb_out = str(p.parent / f"{p.stem}_mut{p.suffix}")

    # ---------- delegate to core function ----------
    mut(pdb_in, pdb_out, sites, mutations, segid)

    return pdb_out


def mut(pdb_in, pdb_out, sites, mutations, segid):
    """
    Perform mutations on a PDB file using psfbuilder.

    Loads the RNA and protein topologies, adds ``pdb_in`` as one segment
    with the mutation table (protein: ``auto_angles=False``; RNA:
    ``auto_angles=False, auto_dihedrals=False``), reads its coordinates,
    guesses missing coordinates, places the phosphate bead of any SEP, TPO
    or PTR mutation (:func:`_place_phosphate`) and the dye of any dye residue (``ADDUCT_TEMPLATES``)
    mutation (:func:`_place_adduct`), regenerates angles and
    dihedrals and writes ``pdb_out``. Progress is printed to stdout.

    Args:
        pdb_in (str): Input PDB file path.
        pdb_out (str): Output PDB file path (overwritten).
        sites (list): Residue numbers to mutate.
        mutations (list): Mutation types (residue names, upper-cased here);
            must have the same length as ``sites``.
        segid (str): Segment ID; must start with 'P' (protein) or 'R' (RNA).

    Raises:
        FileNotFoundError: If ``pdb_in`` does not exist.
        ValueError: If ``sites`` and ``mutations`` differ in length,
            ``segid`` does not start with 'P' or 'R', or a SEP/TPO/PTR or dye
            site lacks the beads needed to place its phosphate or dye.
    """
    # Validate inputs
    if not Path(pdb_in).exists():
        raise FileNotFoundError(f"Input PDB file not found: {pdb_in}")
    
    if len(sites) != len(mutations):
        raise ValueError(f"Number of sites ({len(sites)}) must match number of mutations ({len(mutations)})")
    
    # Load topology files
    print(f"Loading topology files...")
    RNA_topology, _ = utils.load_ff('RNA')
    protein_topology, _ = utils.load_ff('Protein')
    metabolite_topology, _ = utils.load_ff('Metabolite')
    adduct_topology, _ = utils.load_ff('Adduct')

    # Initialize the PSF builder
    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)
    gen.read_topology(metabolite_topology)   # bead types used by the adducts
    gen.read_topology(adduct_topology)
    
    # Build mutation table
    site_mut = []
    for site, mut_type in zip(sites, mutations):
        site_mut.append((str(site), mut_type.upper()))
        print(f"  Mutation: Residue {site} → {mut_type.upper()}")
    
    # Determine molecule type 
    if segid.startswith('P'):
        # Protein segment
        gen.add_segment(segid=segid, pdbfile=pdb_in, auto_angles=False, mutate=site_mut)
    elif segid.startswith('R'):
        # RNA segment
        gen.add_segment(segid=segid, pdbfile=pdb_in, auto_angles=False, auto_dihedrals=False, mutate=site_mut)
    else:
        raise ValueError(f"Error: segid must start with 'P' (Protein) or 'R' (RNA), got: {segid}")
    
    # Read coordinates
    print(f"Reading coordinates from {pdb_in}...")
    gen.read_coords(segid=segid, filename=pdb_in)

    # Guess missing coordinates
    print("Guessing missing coordinates...")
    gen.guess_coords()

    # Place the phosphate bead of phosphorylated residues (only roughly guessed without IC tables)
    input_atoms = _input_atoms(pdb_in)
    for site, mut_type in site_mut:
        if mut_type in PHOSPHO_BEADS:
            d = _place_phosphate(gen, segid, site, mut_type, input_atoms.get(site, set()))
            print(f"  Placed {PHOSPHO_BEADS[mut_type][0]} of {mut_type} {site} ({d:.2f} A from the closest bead)")
        elif mut_type in ADDUCT_TEMPLATES:
            d = _place_adduct(gen, segid, site, mut_type, input_atoms.get(site, set()))
            print(f"  Placed the dye of {mut_type} {site} ({d:.2f} A from the closest bead of another residue)")

    # Regenerate geometry
    print("Regenerating angles and dihedrals...")
    gen.regenerate_angles()
    gen.regenerate_dihedrals()

    # Write output file
    print(f"\nWriting output file...")
    print(f"  PDB: {pdb_out}")
    gen.write_pdb(pdb_out)
    
    print("\n" + "="*60)
    print("Mutation completed successfully!")
    print("="*60)
    print(f"Output PDB: {pdb_out}")
    print(f"Total mutations: {len(site_mut)}")


def main():
    """Command-line interface (``mutate`` console script).

    Usage::

        mutate input.pdb output.pdb -s 123 456 -m ALA GLY [-d P001] [-v]

    ``-s/--sites`` (ints) and ``-m/--mutations`` are required; a single
    mutation type is applied to all sites. ``-d/--segid`` defaults to
    ``P001`` (use an ``R`` prefix for RNA). Any exception is printed to
    stderr (with a traceback if ``-v/--verbose``) and the program exits with
    status 1.
    """
    parser = argparse.ArgumentParser(
        description="Perform site mutations on PDB file using psfbuilder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single mutation
  mutate input.pdb output.pdb -s 123 -m ALA
  
  # Multiple mutations
  mutate input.pdb output.pdb -s 123 456 789 -m ALA GLY TRP
  
  # Phosphorylation (SER/THR/TYR sites; the phosphate bead is placed automatically)
  mutate input.pdb output.pdb -s 23 45 67 -m SEP TPO PTR

  # Dye labels on cysteines (Alexa 488 / Alexa 594; the dye is placed from a template)
  mutate input.pdb output.pdb -s 5 120 -m A48 A59
  
  # RNA mutations
  mutate rna.pdb rna_mut.pdb -s 10 20 -m GUA CYT -d R001
  
  # Multiple sites, same mutation type
  mutate input.pdb output.pdb -s 123 456 789 -m ALA

Segment ID format:
  - Protein: P001, P002, etc.
  - RNA: R001, R002, etc.
        """
    )
    
    parser.add_argument('input_pdb', help='Input PDB file')
    parser.add_argument('output_pdb', help='Output PDB file')
    parser.add_argument('-s', '--sites', required=True, nargs='+', type=int,
                        help='Mutation sites (residue numbers). Can specify multiple sites.')
    parser.add_argument('-m', '--mutations', required=True, nargs='+',
                        help='Mutation types (3-letter codes). If single type given for multiple sites, it will be applied to all sites.')
    parser.add_argument('-d', '--segid', default='P001',
                        help='Segment ID: P001 for Protein, R001 for RNA (default: P001)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Get arguments
        inp = args.input_pdb
        out = args.output_pdb
        sites = args.sites
        mutations = args.mutations
        segid = args.segid
        
        print("="*60)
        print("PDB Mutator - HyresBuilder Edition")
        print("="*60)
        print(f"\nInput PDB: {inp}")
        print(f"Output PDB: {out}")
        print(f"Segment ID: {segid}")
        print(f"Sites: {sites}")
        print(f"Mutations: {mutations}")
        
        # Align mutation sites and types
        if len(sites) != len(mutations):
            if len(mutations) == 1:
                # Single mutation type for all sites
                print(f"\nApplying mutation {mutations[0]} to all {len(sites)} sites")
                mutations = mutations * len(sites)
            else:
                raise ValueError(
                    f"Error: Number of sites ({len(sites)}) does not match "
                    f"number of mutation types ({len(mutations)}). "
                    f"Either provide one mutation per site, or one mutation type for all sites."
                )
        
        print("\n" + "="*60)
        print("Starting mutation process...")
        print("="*60 + "\n")
        
        # Perform mutation
        mut(inp, out, sites, mutations, segid)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
