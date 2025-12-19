from psfgen import PsfGen
from HyresBuilder import utils
import argparse, os, glob


def split_chains(pdb):
    aas = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                   "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    rnas = ["ADE", "GUA", "CYT", "URA"]
    dnas = ["DAD", "DGU", "DCY", "DTH"]
    mg, cal = ["MG+"], ["CA+"] 
    def get_type(resname):
        chaintype = (
            'P' if resname in aas else
            'R' if resname in rnas else
            'D' if resname in dnas else
            'M' if resname in mg else
            'C' if resname in cal else
            None
        )
        return chaintype

    currentID = None
    atoms = []
    chains = []
    types = []
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chainid = line[21]
                resname = line[17:20]
                if chainid != currentID:
                    if atoms:
                        chains.append(atoms)
                    currentID = chainid
                    types.append(get_type(resname))
                    atoms = [line]
                else:
                    atoms.append(line)
        if atoms:
            chains.append(atoms)
    # save out each chain
    for i, (t, chain) in enumerate(zip(types, chains)):
        if t in ['P', 'R', 'D']:
            tmp_pdb = f"psfgentmp_{i}.pdb"
        elif t in ['M', 'C']:
            tmp_pdb = f"psfgentmp_{t}.pdb"
        else:
            print('Unkown molecule type')
            exit(1)

        with open(tmp_pdb, 'w') as f:
            for line in chain:
                f.write(line)
            f.write('END\n')
    return types

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

    counts = {'P': 0, 'R': 0, 'D': 0, 'M': 0, 'C': 0}
    types = split_chains(pdb_in)
    for i, t in enumerate(types):
        if t in ["P", "R", "D"]:
            tmp_pdb = f"psfgentmp_{i}.pdb"
        else:
            tmp_pdb = f"psfgentmp_{t}.pdb"

        segid = f"{t}{counts[t]:03d}"
        counts[t] += 1
        if t == 'P':
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False)
        else:
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False, auto_dihedrals=False)

    # re-set the charge status of terminus
    for segid in gen.get_segids():
        if terminal != "neutral":
            set_terminus(gen, segid, terminal)       
            
    gen.write_psf(filename=psf_out)
    
    #clean up
    for file_path in glob.glob("psfgentmp_*.pdb"):
        os.remove(file_path)

def main():
    parser = argparse.ArgumentParser(description="generate PSF for Hyres/iCon systems", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdb", help="CG PDB file(s)", default='conf.pdb')
    parser.add_argument("psf", help="output name/path for PSF", default='conf.psf')
    parser.add_argument("-t", "--ter", choices=['neutral', 'charged', 'NT', 'CT', 'positive'], 
                        help="Terminal charged status (choose from ['neutral', 'charged', 'NT', 'CT', 'positive'])", default='neutral')
    args = parser.parse_args()

    genpsf(args.pdb, args.psf, args.ter)
    # cleanup
    for file_path in glob.glob("psfgentmp_*.pdb"):
        os.remove(file_path)
    

if __name__ == '__main__':
    main()
