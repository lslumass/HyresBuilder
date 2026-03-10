"""
Performe mutations/modifications on a PDB file.   
Authors: Shanlong Li   
Date: Feb 14, 2026   
"""

from psfgen import PsfGen
from HyresBuilder import utils
import argparse
import sys
from pathlib import Path


def mutate(pdb_in, sites, mutations, pdb_out=None, segid='P001'):
    """
    Performe mutations on a PDB file.

    Args:
        pdb_in  (str):              Input PDB file path.
        sites   (int | list[int]):  Residue number(s) to mutate.
        mutations (str | list[str]):Mutation type(s) as 3-letter codes (e.g. 'ALA').
                                    A single value is broadcast to all sites.
        pdb_out (str, optional):    Output PDB file path. Defaults to
                                    '<pdb_in stem>_mut.pdb' in the same directory.
        segid   (str):              Segment ID — 'P001' for Protein, 'R001' for RNA.
                                    Must start with 'P' or 'R'. Default: 'P001'.

    Returns:
        str: Path to the written output PDB file.

    Raises:
        FileNotFoundError: If pdb_in does not exist.
        ValueError:        If sites/mutations lengths are mismatched, or segid is invalid.

    Examples:
        >>> from mutator import mutate

        >>> # Single mutation
        >>> mutate('protein.pdb', 123, 'ALA')

        >>> # Multiple mutations, explicit output path
        >>> mutate('protein.pdb', [10, 20, 30], ['ALA', 'GLY', 'TRP'],
        ...        pdb_out='protein_mut.pdb')

        >>> # Broadcast one mutation type to every site
        >>> mutate('protein.pdb', [10, 20, 30], 'ALA')

        >>> # RNA mutation
        >>> mutate('rna.pdb', [10, 20], ['G', 'C'], segid='R001')
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
    Perform mutations on a PDB file using psfgen

    Args:
        pdb_in (str): Input PDB file path
        pdb_out (str): Output PDB file path
        sites (list): List of residue numbers to mutate
        mutations (list): List of mutation types (3-letter codes)
        segid (str): Segment ID (P001 for Protein, R001 for RNA)
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
    
    # Initialize psfgen
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
    parser = argparse.ArgumentParser(
        description="Perform site mutations on PDB file using psfgen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single mutation
  python mutator.py input.pdb output.pdb -s 123 -m ALA
  
  # Multiple mutations
  python mutator.py input.pdb output.pdb -s 123 456 789 -m ALA GLY TRP
  
  # RNA mutations
  python mutator.py rna.pdb rna_mut.pdb -s 10 20 -m G C -d R001
  
  # Multiple sites, same mutation type
  python mutator.py input.pdb output.pdb -s 123 456 789 -m ALA

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