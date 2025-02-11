import sys
from psfgen import PsfGen
from HyresBuilder import utils
import MDAnalysis as mda
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="create phosphosrylated PDB and PSF files for Hyres systems",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_pdb_file",  help="Hyres PDB file(s), it should be the pdb of monomer", required=True)
    parser.add_argument("-o", "--output_pdb_file", help="output name/path for phosphosrylated Hyres PDB", required=True)
    parser.add_argument("-p", "--output_psf_file", help="output name/path for phosphosrylated Hyres PSF", required=True)
    parser.add_argument("-s", "--phos_sites", help="the phosphosrylated sites", nargs="+")
    parser.add_argument("-t", "--mutation", default=['SEP',], help="phosphosrylated residue name, SEP or THP", nargs="+")
    args = parser.parse_args()

    inp = args.input_pdb_file
    pdb = args.output_pdb_file
    psf = args.output_psf_file
    sites = args.phos_sites
    mutations = ['SEP',]*len(sites) if len(sites) > 1 and args.mutation == ['SEP',] else args.mutation
    assert len(sites) == len(mutations), "sites and mulation numbers are unconsistence!"

    top_inp, param_inp = utils.load_ff('protein')

    gen = PsfGen()
    gen.read_topology(top_inp)

    u = mda.Universe(inp)
    site_mut = []
    for site, mut in zip(sites, mutations):
        tmp = (str(site), mut)
        site_mut.append(tmp)

    for segment in u.segments:
        seg = segment.segid
        sel = u.select_atoms(f'segid {seg}')
        sel.write(f'{seg}.pdb')
        gen.add_segment(segid=seg, pdbfile=f'{seg}.pdb', mutate=site_mut)
        gen.read_coords(segid=seg, filename=f'{seg}.pdb')
        gen.guess_coords()
        os.system(f'rm -rf {seg}.pdb')

    gen.regenerate_angles()
    gen.regenerate_dihedrals()

    gen.write_pdb(filename=pdb)
    gen.write_psf(filename=psf)


if __name__ == '__main__':
    main()
    print('Done!')