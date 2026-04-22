"""
PSF file generation for HyRes and iConRNA coarse-grained systems.

This module constructs CHARMM-style PSF topology files from coarse-grained
PDB structures produced by the HyRes (protein) and iConRNA (RNA) force fields.
It handles mixed systems containing any combination of protein, RNA, DNA,
Mg²⁺, and Ca²⁺ chains in a single input PDB, automatically detecting molecule
types by chain identity and assigning structured segment IDs before invoking
``psfgen`` to build and write the topology.

Workflow
--------
1. Parse the input CG PDB and split it into per-chain temporary files,
   detecting molecule type from residue names (:func:`split_chains`).
2. Assign segment IDs following the convention below and register each chain
   with ``psfgen`` using the appropriate force-field topology (:func:`genpsf`).
3. Optionally set terminus charge states for protein chains
   (:func:`set_terminus`).
4. Write the PSF file and remove all intermediate temporary PDB files.

Segment ID convention
---------------------
Segment IDs are four characters: a single type prefix followed by a
three-character hybrid-36 counter encoded by :func:`encode_segid`.

========  ===========  ===============
Prefix    Molecule     Example IDs
========  ===========  ===============
``P``     Protein      P001, P002, …
``R``     RNA          R001, R002, …
``D``     DNA          D001, D002, …
``M``     Mg²⁺ ions    M001
``C``     Ca²⁺ ions    C001
========  ===========  ===============

The hybrid-36 counter supports up to 68,391 chains per molecule type before
overflowing.

A command-line interface is exposed via :func:`main` and registered as the
``GenPsf`` entry point.

Dependencies
------------
* `psfgen <https://github.com/MDAnalysis/psfgen>`_ (``psfgen.PsfGen``)
* HyresBuilder force-field topology files, loaded via ``utils.load_ff``.
"""

from importlib.resources import files
from psfgen import PsfGen
from HyresBuilder import utils
import argparse, os, glob


def split_chains(pdb):
    aas = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                   "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    rnas = ["ADE", "GUA", "CYT", "URA"]
    dnas = ["DAD", "DGU", "DCY", "DTH"]
    mg, cal = ["MG+"], ["CA+"]
    phos = ['PHO']
    AGs = ['KAN']

    def get_type(resname):
        chaintype = (
            'P' if resname in aas else
            'R' if resname in rnas else
            'D' if resname in dnas else
            'M' if resname in mg else
            'C' if resname in cal else
            'PHO' if resname in phos else
            'AGs' if resname in AGs else
            None
        )
        return chaintype

    currentKey = None
    atoms = []
    chains = []
    types = []
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chainid = line[21]
                segid   = line[72:76].strip()  # segID, cols 73-76
                resname = line[17:20].strip()
                key = (chainid, segid)

                if key != currentKey:
                    if atoms:
                        chains.append(atoms)
                    currentKey = key
                    types.append(get_type(resname))
                    atoms = [line]
                else:
                    atoms.append(line)
        if atoms:
            chains.append(atoms)

    # save out each chain
    pre_type = None
    for i, (t, chain) in enumerate(zip(types, chains)):
        if t in ['P', 'R', 'D', 'PHO', 'AGs']:
            tmp_pdb = f"psfgentmp_{i}.pdb"
        elif t in ['M', 'C']:
            if t == pre_type:
                continue  # skip if same ion type as previous chain
            else:
                tmp_pdb = f"psfgentmp_{t}.pdb"
        else:
            print('Unknown molecule type')
            exit(1)
        pre_type = t

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

def encode_segid(n: int) -> str:
    """Encode segment number n into a 3-char hybrid-36 segid (e.g. 000, a00)."""
    BASE36     = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    LEAD_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not 0 <= n <= 68391:
        raise ValueError(f"n={n} out of range [0, 68391]")
    if n < 1000:
        return f"{n:03d}"
    n -= 1000
    lead = LEAD_CHARS[n // 1296]
    rem  = n % 1296
    return f"{lead}{BASE36[rem // 36]}{BASE36[rem % 36]}"

def genpsf(pdb_in, psf_out, terminal='neutral', RNA='mix'):
    """
    Generate a PSF file for a HyRes protein or iConRNA coarse-grained system.

    Reads the input CG PDB, automatically detects molecule types (protein, RNA,
    DNA, Mg2+, Ca2+) by chain ID, assigns segment IDs, and uses psfgen to build
    the topology and write the PSF file. Temporary chain PDB files are removed
    after the PSF is written.

    Segment ID naming convention:

    - Protein chains: ``P001``, ``P002``, ...
    - RNA chains: ``R001``, ``R002``, ...
    - DNA chains: ``D001``, ``D002``, ...
    - Mg2+ ions: ``M001``
    - Ca2+ ions: ``C001``

    Args:
        pdb_in (str): Path to the input coarse-grained PDB file. May contain
                      mixed protein, RNA, DNA, and ion chains.
        psf_out (str): Path to the output PSF file.
        terminal (str, optional): Charge status of protein termini. Options:

                                  - ``'neutral'`` — uncharged termini (default)
                                  - ``'charged'`` — both termini charged
                                  - ``'NT'`` — N-terminus charged only
                                  - ``'CT'`` — C-terminus charged only
                                  - ``'positive'`` — both termini negatively charged
        RNA (str, optional): Which RNA topology to use. Options:
                            - ``'mix'`` (default) — use HyRes_iConRNA topologies
                            - ``'icon'`` — use iConRNA topologies instead

    Returns:
        None. Writes a PSF file to ``psf_out``.

    Example:
        >>> from HyresBuilder import GenPsf
        >>> GenPsf.genpsf("conf.pdb", "conf.psf")
        >>> GenPsf.genpsf("conf.pdb", "conf.psf", terminal="charged")
    """
    
    # load topology files
    if RNA == 'mix':
        RNA_topology, _ = utils.load_ff('RNA')
    elif RNA == 'icon':
        path1 = files("HyresBuilder") / "forcefield" / "top_RNA.inp"
        RNA_topology = path1.as_posix()
    protein_topology, _ = utils.load_ff('Protein')
    AGs_topology, _ = utils.load_ff('AGs')

    # generate psf
    gen = PsfGen()
    gen.read_topology(RNA_topology)
    gen.read_topology(protein_topology)
    gen.read_topology(AGs_topology)

    counts = {'P': 1, 'R': 1, 'D': 1, 'M': 1, 'C': 1, 'PHO': 1}
    types = split_chains(pdb_in)
    for i, t in enumerate(types):
        if t in ["P", "R", "D", "PHO"]:
            tmp_pdb = f"psfgentmp_{i}.pdb"
        else:
            tmp_pdb = f"psfgentmp_{t}.pdb"

        segid = f"{t}{encode_segid(counts[t])}"
        counts[t] += 1
        if t == 'P':
            gen.add_segment(segid=segid, pdbfile=tmp_pdb, auto_angles=False)
        elif t == 'PHO':
            gen.add_segment(segid=segid, pdbfile=tmp_pdb)
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
    parser.add_argument("--icon", action='store_true', help="Use iConRNA topologies instead of HyRes_iConRNA topologies")
    args = parser.parse_args()

    if args.icon:
        genpsf(args.pdb, args.psf, args.ter, RNA='icon')
    else:
        genpsf(args.pdb, args.psf, args.ter)
    # cleanup
    for file_path in glob.glob("psfgentmp_*.pdb"):
        os.remove(file_path)
    

if __name__ == '__main__':
    main()
