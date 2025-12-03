from psfgen import PsfGen
from HyresBuilder import utils
import argparse, os, glob
import numpy as np
import MDAnalysis as mda


def set_terminus(gen, segid, charge_status):
    if segid.startswith("P"):
        nter, cter = gen.get_resids(segid)[0], gen.get_resids(segid)[-1]
        if charge_status == 'charged':
            gen.set_charge(segid, nter, "N", 1.00)
            gen.set_charge(segid, cter, "O", -1.00)
        elif charge_status == 'NT':
            gen.set_charge(segid, nter, "N", 1.00)
        elif charge_status == 'CT':
            gen.set_charge(segid, cter, "O", -1.00)
        elif charge_status == 'positive':
            gen.set_charge(segid, nter, "N", -1.00)
            gen.set_charge(segid, cter, "O", -1.00)
        else:
            print("Error: Only 'neutral', 'charged', 'NT', and 'CT' charge status are supported.")
            exit(1)

def genpsf(pdb_in, psf_out, terminal):
    # load topology files
    RNA_topology, _ = utils.load_ff('RNA')
    protein_topology, _ = utils.load_ff('Protein')

    # generate psf
    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)
    u = mda.Universe(pdb_in)

    dnas = {"DA", "DT", "DG", "DC", "DI"}
    rnas = {"A", "U", "G", "C", "I"}
    counts = {'protein': 0, 'rna': 0, 'dna': 0, 'MG': 0, 'CAL': 0}

    chains = u.atoms.split('segment')
    for chain in chains:
        resname = chain.residues[0] # get the first residue name
        if chain.select_atoms("name CA").n_atoms > 0:
            segid = f"P{counts['protein']:03d}"
            counts['protein'] += 1
            tmp_pdb = f'psftmp_{segid}.pdb'
            chain.atoms.write(tmp_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False)
        elif resname in rnas:
            segid = f"R{counts['rna']:03d}"
            counts['rna'] += 1
            tmp_pdb = f'psftmp_{segid}.pdb'
            chain.atoms.write(tmp_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False, auto_dihedrals=False)
        elif resname in dnas:
            segid = f"D{counts['dna']:03d}"
            counts['dna'] += 1
            tmp_pdb = f'psftmp_{segid}.pdb'
            chain.atoms.write(tmp_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False, auto_dihedrals=False)
        elif resname == 'MG+':
            segid = f"MG{counts['MG']:04d}"
            counts['MG'] += 1
            tmp_pdb = f'psftmp_{segid}.pdb'
            chain.atoms.write(tmp_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False, auto_dihedrals=False)
        elif resname == 'CA+':
            segid = f"CA{counts['CAL']:04d}"
            counts['CAL'] += 1
            tmp_pdb = f'psftmp_{segid}.pdb'
            chain.atoms.write(tmp_pdb)
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False, auto_dihedrals=False)
        else:
            print('There are UNKNOWN residue types')
            exit(1)

    # re-set the charge status of terminus
    for segid in gen.get_segids():
        if terminal != "neutral":
            set_terminus(gen, segid, terminal)       
            
    gen.write_psf(filename=psf_out)
    # cleanup
    for file_path in glob.glob("psftmp_*.pdb"):
        os.remove(file_path)

def main():
    parser = argparse.ArgumentParser(description="generate PSF for Hyres/iCon systems", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdb", help="CG PDB file(s)", required=True, default='conf.pdb')
    parser.add_argument("psf", help="output name/path for PSF", required=True, default='conf.psf')
    parser.add_argument("-t", "--ter", choices=['neutral', 'charged', 'NT', 'CT', 'positive'], 
                        help="Terminal charged status (choose from ['neutral', 'charged', 'NT', 'CT', 'positive'])", default='neutral')
    args = parser.parse_args()

    genpsf(args.pdb, args.psf, args.ter)
    

if __name__ == '__main__':
    main()
