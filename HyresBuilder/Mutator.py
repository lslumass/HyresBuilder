"""
Mutator: module used to do mutations and PTMs
input:
    inpdb: pdb file, single chain of protein or RNA
    sites: mutation sites
    mutations: mutation type
output:
    outpdb: new pdb file after mutation

Authors: Shanlong Li
Date: Feb 14, 2026
"""

from psfgen import PsfGen
from HyresBuilder import utils
import argparse
import sys
from pathlib import Path


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
    for site, mut in zip(sites, mutations):
        site_mut.append((str(site), mut.upper()))
        print(f"  Mutation: Residue {site} → {mut.upper()}")
    
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