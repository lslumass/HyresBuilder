import MDAnalysis as mda
from MDAnalysis.analysis import align
from modeller import *
from modeller.automodel import *
import gemmi
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
    subprocess(f"wget https://files.rcsb.org/download/{pdb_id}.cif")
    subprocess(f"wget -O {pdb_id}.fasta www.rcsb.org/fasta/entry/{pdb_id}/download")

# get the chains from top to bottom
# input: mda.Universe
# output: the ordered chainID from top to bottom
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
# output: the name of built pdb file
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
                new_id = 'A' if i % 2 == 0 else "B"
                top.chainIDs = new_id
            else:
                for k, old_id in enumerate(old_ids):
                    mask = (top.chainIDs == old_id)
                    top.atoms[mask].chainIDs = alphabet[k % 26]
            f.write(top)   # write out new top layer
            ref = mobile.copy()
    return out

# read missing residues
# output: dict {chainid: [missing residue list], }
def read_missing_res(pdb_id):
    doc = gemmi.cif.read_file(f"{pdb_id}.cif")
    block = doc.sole_block()
    loop = block.find_loop("_pdbx_unobs_or_zero_occ_residues.id")
    if loop is None:
        return {}
    result = {}
    for row in loop:
        chain = row["_pdbx_unobs_or_zero_occ_residues.auth_asym_id"]
        resid = int(row["_pdbx_unobs_or_zero_occ_residues.auth_seq_id"])
        if chain not in result:
            result[chain] = []
        result[chain].append(resid)
    return result

# create alignment.ali
def create_alignment_ali(pdb_id, nprotof=2, nlayers=40):
    with open(f"{pdb_id}.fasta", 'r') as f:
        seq = f.readlines()[1][:-1]

    nchains = nprotof*nlayers       # total number of chains
    ## get missing residues
    missings = read_missing_res(pdb_id)
    tops = get_top_chains()[:nprotof]

    # create alignment.ali file
    print('Creating alignment file...\n')
    with open('alignment.ali', 'w') as f:
        print('>P1;miss\nstructureX:40mer::A::::::', file=f)
        tem1 = ''
        tem2 = ''
        for l in range(nlayers):
            for n, chain in enumerate(tops[:nprotof]):
                cnt = l*nprotof + n
                res_missed = missings[chain]
                seq_miss = ''.join('-' if i+1 in res_missed else char for i, char in enumerate(seq))
                seq_fill = seq
                separator = '/\n' if cnt != nchains-1 else '*'
                tem1 += seq_miss + separator
                tem2 += seq_fill + separator
        print(tem1, file=f)
        print("\n>P1;fill\nsequence:::::::::", file=f)
        print(tem2, file=f)

# add missing loops
def add_missing_loop(pdb_id, core_region, nprotof=2, nlayers=40, ncycles=10):
    log.verbose()
    env = Environ()

    # Optimization schedule - give less weight to soft-sphere restraints
    env.schedule_scale = physical.Values(default=1.0, soft_sphere=0.7)
    env.io.atom_files_directory = ['.', './atom_files']

    # Initialize model
    a = AutoModel(env, alnfile='alignment.ali', knowns='miss', sequence='fill')
    a.starting_model = 1
    a.ending_model = 1
    # Very thorough VTFM optimization:
    a.library_schedule = autosched.slow
    a.max_var_iterations = 300
    a.md_level = refine.slow  # Options: refine.fast, refine.slow, refine.very_slow
    # Repeat the whole optimization cycle multiple times on this ONE model
    a.repeat_optimization = ncycles  # Repeat 2-5 times for thorough relaxation
    a.max_molpdf = 1e8  # Don't stop unless objective function > 1E6
    # Quality assessment
    a.assess_methods = (assess.DOPE, assess.GA341, assess.normalized_dope)
    # Generate and optimize the single model
    a.make()