"""
Residue mutation utilities for HyRes and iConRNA coarse-grained PDB structures.

This module applies in silico point mutations to coarse-grained PDB files
produced by the HyRes (protein) and iConRNA (RNA) pipelines. Mutations are
performed via ``psfbuilder``: the mutated residues are rebuilt from the topology,
coordinates of atoms not present in the input are guessed, and the resulting
structure is written as a PDB (no PSF is written).

The whole input PDB is treated as a single segment, which must be either a
protein or an RNA chain, selected by segment ID prefix (``P`` for protein,
``R`` for RNA). Only the HyRes protein and iConRNA (``top_RNA_mix``)
topologies are loaded, so mutation types must be residue names defined there
(e.g. ``ALA``, ``GLY`` for protein; ``ADE``, ``GUA``, ``CYT``, ``URA`` for
RNA). Multiple sites can be mutated in a single call, and a single mutation
type can be broadcast across all target sites.

Workflow
--------
1. Normalise sites and mutation types into matched lists (:func:`mutate`, or
   :func:`main` on the command line).
2. Load the protein and RNA topology files via ``utils.load_ff``.
3. Register the input PDB as a ``psfbuilder`` segment with the ``mutate`` table
   applied at the ``add_segment`` step (:func:`mut`).
4. Read the input coordinates, guess missing coordinates, place the phosphate
   bead of phosphorylated residues (:func:`_place_phosphate`), regenerate
   angles and dihedrals, and write the output PDB.

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
    or PTR mutation (:func:`_place_phosphate`), regenerates angles and
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
            ``segid`` does not start with 'P' or 'R', or a SEP/TPO/PTR site
            lacks the beads needed to place its phosphate.
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
    
    # Initialize the PSF builder
    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)
    
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
