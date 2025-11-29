import MDAnalysis as mda
from MDAnalysis.analysis import align
from modeller import *
from modeller.automodel import *
import numpy as np
import string
import subprocess



"""
This module is used to build long amyloid fibril
"""


# download pdb and fasta form PDBBANK based on pdb id
# input: pdb id
def dwn(pdb_id):
    subprocess(f"wget www.rcsb.org/pdb/files/{pdb_id}.pdb")
    subprocess(f"wget -O {pdb_id}.fasta www.rcsb.org/fasta/entry/{pdb_id}/download")

# get the chains from top to bottom
# input: mda.Universe
def get_top_chains(u):
    chains = np.unique(u.atoms.chainIDs)
    valid_chains = [cid for cid in chains if str(cid).strip()]
    zs = []
    for chain in valid_chains:
        sel = u.select_atoms(f"chainID {chain}")
        if sel.n_atoms > 0:
            z = sel.center_of_mass()[2]
            zs.append((chain, z))
    zs.sort(key=lambda x: x[1], reverse=True)
    tops = [item[0] for item in zs]

    return tops

# build long core region
# input:
#       pdb_id: pdb id
#       nprotof: number of protofilaments in each layer, default=2
#       nlayer: how many layers, default=40
def build_long_core(pdb_id, nprotof=2, nlayer=40):
    u = mda.Universe(f"{pdb_id}.pdb")
    order_chains = get_top_chains(u)
    tops = " ".join(order_chains[:2*nprotof])     # top 2 layers
    top1 = " ".join(order_chains[:nprotof])         # first top layer
    top2 = " ".join(order_chains[nprotof:2*nprotof])    # second top layer

    sel = u.select_atoms(f"chainID {tops}")   # select top 2 layers
    ref = u.select_atoms(f"chainID {top1}")	  # select top 1 layers as reference
    mobile = sel.copy()
    
    out = f"{nlayer}layers_{pdb_id}.pdb"
    alphabet = string.ascii_uppercase
    with mda.Writer(out, multiframe=False, reindex=False) as f:
        for i in range(nlayer):
            align.alignto(mobile, ref, select=(f"chainID {top2}", f"chainID {top1}"))  #align G H I J to I J, G(H) was aligned to I(J)
            top = mobile.select_atoms(f"chainID {top1}").copy()
            old_ids = np.unique(top.chainIDs)
            n = len(old_ids)
            if n == 1:
                new_id = 'A' if i%2 == 0 else "B"
                top.chainIDs = new_id
            else:
                for k, old_id in enumerate(old_ids):
                    mask = (top.chainIDs == old_id)
                    top.atoms[mask].chainIDs = alphabet[k%26]
            f.write(top)   # write out new top layer
            ref = mobile.copy()
    return out

# add missing loops
def add_missing_loop(pdb_id, core_region, nprotof=2, nlayers=40, ncycles=10):
    with open(f"{pdb_id}.fasta", 'r') as f:
        seq = f.readlines()[1][:-1]

    nchains = nprotof*nlayers       # total number of chains
