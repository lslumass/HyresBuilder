import sys
from psfgen import PsfGen
from HyresBuilder import utils
import MDAnalysis as mda
import argparse
import os


def set_terminus(gen, segid, charge_status):
    nter, cter = gen.get_resids(segid)[0], gen.get_resids(segid)[-1]
    if charge_status == 'charged':
        gen.set_charge(segid, nter, "N", 1.00)
        gen.set_charge(segid, cter, "O", -1.00)
    elif charge_status == 'NT':
        gen.set_charge(segid, nter, "N", 1.00)
    elif charge_status == 'CT':
        gen.set_charge(segid, cter, "O", -1.00)
    else:
        print("Error: Only 'neutral', 'charged', 'NT', and 'CT' charge status are supported.")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="create phosphosrylated PDB and PSF files for Hyres systems",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', "--model", default='mix', help="simulated system: protein, RNA, or mix")
    parser.add_argument("-i", "--input_pdb_file",  help="Hyres PDB file(s), it should be the pdb of monomer", required=True)
    parser.add_argument("-o", "--output_pdb_file", help="output name/path for phosphosrylated Hyres PDB", required=True)
    parser.add_argument("-p", "--output_psf_file", help="output name/path for phosphosrylated Hyres PSF", required=True)
    parser.add_argument("-s", "--phos_sites", help="the phosphosrylated sites", nargs="+")
    parser.add_argument("-t", "--mutation", default=['SEP',], help="phosphosrylated residue name, SEP or THP", nargs="+")
    parser.add_argument("-c", "--charge", default="neutral", help="charge of the ternimus, choose from neutral, charged, NT, or CT")
    args = parser.parse_args()

    model = args.model
    inp = args.input_pdb_file
    pdb = args.output_pdb_file
    psf = args.output_psf_file
    sites = args.phos_sites
    ter = args.charge
    mutations = ['SEP',]*len(sites) if len(sites) > 1 and args.mutation == ['SEP',] else args.mutation
    assert len(sites) == len(mutations), "sites and mulation numbers are unconsistence!"

    if model in ['protein', 'RNA']:
        RNA_topology, _ = utils.load_ff('RNA')
        protein_topology, _ = utils.load_ff('protein')
    elif model == 'mix':
        RNA_topology, _ = utils.load_ff('RNA_mix')
        protein_topology, _ = utils.load_ff('protein_mix')
    else:
        print("Error: Only 'protein', 'RNA', and 'mix' models are supported.")
        exit(1)

    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)


    u = mda.Universe(inp)
    site_mut = []
    for site, mut in zip(sites, mutations):
        tmp = (str(site), mut)
        site_mut.append(tmp)

    for segment in u.segments:
        seg = segment.segid
        sel = u.select_atoms(f'segid {seg}')
        sel.write(f'{seg}.pdb')
        gen.add_segment(segid=seg, pdbfile=f'{seg}.pdb', auto_angles=False, mutate=site_mut)
        gen.read_coords(segid=seg, filename=f'{seg}.pdb')
        gen.guess_coords()
        os.system(f'rm -rf {seg}.pdb')

        # set charge for terminus
        if ter != "neutral":
            set_terminus(gen, seg, ter)

    gen.regenerate_angles()
    gen.regenerate_dihedrals()

    gen.write_pdb(filename=pdb)
    gen.write_psf(filename=psf)


if __name__ == '__main__':
    main()
    print('Done!')