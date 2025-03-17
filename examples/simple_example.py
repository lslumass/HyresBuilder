import sys
from HyresBuilder import Geometry, HyresBuilder
import Bio.PDB

"""
Simple example script demonstrating how to use the HyresBuilder library.
Usage: python hyres_build.py seq_file output_file_name
"""

seq_file = sys.argv[1]
out_file = sys.argv[2]

with open(seq_file, 'r') as f:
    sequence = f.readlines()[0]
    sequence = sequence.strip()

geo = Geometry.geometry(sequence[0])
#geo.phi = 0
#geo.psi_im1 = 0
structure = HyresBuilder.initialize_res(geo)

for seq in sequence[1:]:
    geo = Geometry.geometry(seq)
    HyresBuilder.add_residue(structure, geo)
# add terminal oxygen (OXT)
#HyresBuilder.add_terminal_OXT(structure)

out = Bio.PDB.PDBIO()
out.set_structure(structure)
out.save(out_file)
