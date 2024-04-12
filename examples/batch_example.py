import sys
from HyresBuilder import Geometry
import HyresBuilder
import Bio.PDB

"""
Usage: python peptide_build.py seq_file
"""

# read each line for each protein
seq_file = sys.argv[1]
seqs = {}
with open(seq_file, 'r') as f:
    for line in f.readlines():
        name = line.split()[0]
        seq = line.split()[1].strip()
        seqs[name] = seq

# generate 
for name, seq in seqs.items():
    geo = Geometry.geometry(seq[0])
    structure = HyresBuilder.initialize_res(geo)
    for seq in seq[1:]:
        geo = Geometry.geometry(seq)
        HyresBuilder.add_residue(structure, geo)

    out = Bio.PDB.PDBIO()
    out.set_structure(structure)
    out.save(name+'.pdb')

exit()
