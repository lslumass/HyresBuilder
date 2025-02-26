import sys
from psfgen import PsfGen
from HyresBuilder import utils
import argparse
import os

# Global varibale
RNA_topology, _ = utils.load_ff('RNA_mix')
protein_topology, _ = utils.load_ff('protein_mix')

def main():
    parser = argparse.ArgumentParser(description="generate PSF for Hyres systems",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_pdb_files",  help="Hyres PDB file(s), it should be the pdb of monomer", required=True, nargs="+")
    parser.add_argument("-o", "--output_psf_file", help="output name/path for Hyres PSF", required=True)
    parser.add_argument("-n", "--num_of_chains", type=int, help="Number of copies for each pdb; it should have the same length as the given pdb list specified in the '-i' argument", default=[1,], nargs="+")
    parser.add_argument("-m", "--molecule_type", type=str, help="select from 'P', 'R', 'D' for protein, RNA, and DNA, respectively", default=['P', 'R'], nargs="+")
    parser.add_argument("-t", "--ter", choices=['neutral', 'charged'], help="Terminal charged status (choose from ['neutral', 'charged'])", default='neutral')
    args = parser.parse_args()
   
    pdb_list = args.input_pdb_files
    outpsf = args.output_psf_file
    num_list = args.num_of_chains
    print(num_list)
    assert len(pdb_list) == len(num_list), "chain number list must have the same length as the pdb list (specified by the '-n' argument)"
    type_list = args.molecule_type
    assert len(pdb_list) == len(type_list), "molecule type list must have the same length as the pdb list (specified by the '-n' argument)"
    ter = args.ter

    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)

    for idx, pdb in enumerate(pdb_list): 
        # loop through each pdb and make copies
        num = num_list[idx]
        mol_type = type_list[idx]
        if mol_type == 'P':
            for i in range(num):
                segid = f"{chr(65+idx)}{i}" 
                gen.add_segment(segid=segid, pdbfile=pdb, auto_angles=False)
        elif mol_type == 'R':
            for i in range(num):
                segid = f"{chr(65+idx)}{i}" 
                gen.add_segment(segid=segid, pdbfile=pdb, auto_angles=False, auto_dihedrals=False)
    gen.write_psf(filename=outpsf)

if __name__ == '__main__':
    main()
