"""
Amyloid Fibril Builder

This module provides tools for building and modeling long amyloid fibril structures
from PDB files. It handles structure elongation, missing residue detection, and
loop modeling using MODELLER.

Author: Shanlong Li
Date: Nov 29, 2025
"""

import MDAnalysis as mda
from MDAnalysis.analysis import align
from modeller import *
from modeller.automodel import *
from HyresBuilder import Convert2CG
import gemmi
import numpy as np
import string
import subprocess
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AmyloidFibrilBuilder:
    """
    A class for building extended amyloid fibril structures from PDB templates.
    """
    
    def __init__(self, pdb_id: str, work_dir: str = "."):
        """
        Initialize the AmyloidFibrilBuilder.
        
        Args:
            pdb_id: PDB identifier (e.g., '6msm')
            work_dir: Working directory for input/output files
        """
        self.pdb_id = pdb_id.lower()
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.pdb_file = self.work_dir / f"{self.pdb_id}.pdb"
        self.cif_file = self.work_dir / f"{self.pdb_id}.cif"
        self.fasta_file = self.work_dir / f"{self.pdb_id}.fasta"
        
        logger.info(f"Initialized AmyloidFibrilBuilder for PDB: {self.pdb_id}")
    
    def download_files(self, force: bool = False) -> bool:
        """
        Download PDB, CIF, and FASTA files from RCSB.
        
        Args:
            force: If True, re-download even if files exist
            
        Returns:
            True if successful, False otherwise
        """
        files_to_download = [
            (f"https://files.rcsb.org/download/{self.pdb_id}.pdb", self.pdb_file),
            (f"https://files.rcsb.org/download/{self.pdb_id}.cif", self.cif_file),
            (f"https://www.rcsb.org/fasta/entry/{self.pdb_id}/download", self.fasta_file)
        ]
        
        for url, filepath in files_to_download:
            if filepath.exists() and not force:
                logger.info(f"File exists: {filepath.name}, skipping download")
                continue
            
            try:
                logger.info(f"Downloading {filepath.name}...")
                result = subprocess.run(
                    ["wget", "-q", "-O", str(filepath), url],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to download {filepath.name}: {result.stderr}")
                    return False
                    
                logger.info(f"Successfully downloaded {filepath.name}")
                
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout while downloading {filepath.name}")
                return False
            except Exception as e:
                logger.error(f"Error downloading {filepath.name}: {e}")
                return False
        
        return True
    
    def get_protof_number(self) -> int:
        """
        Get the number of protofaliment
        Returns:
            number of protofaliment
        """
        doc = gemmi.cif.read_file(f"{self.pdb_id}.cif")
        block = doc.sole_block()
        # get the strand number
        strands = block.find_values('_entity_poly.pdbx_strand_id')
        if len(strands) != 1:
            logger.error("Not a homotypic fibril")
            exit(1)
        else:
            n_strand = len(strands[0].split(','))

        # get the layer number through strand stacking
        n_layer = 1
        vals = block.find_values('_struct_sheet.number_strands')
        if vals:
            try:
                layers = [int(s) for s in vals]
                n_layer = max(layers)
            except ValueError:
                pass   # n_layer = 1
        
        # calculate the number of monomer in each layer
        nprotof = int(n_strand / n_layer)
        return nprotof
    
    def get_top_chains(self, universe: mda.Universe, nprotof=2) -> List[str]:
        """
        Get chains ordered from top to bottom based on Z-coordinate of center of mass.
        
        Args:
            universe: MDAnalysis Universe object
            
        Returns:
            List of chain IDs ordered from highest to lowest Z-coordinate
        """
        chains = np.unique(universe.atoms.chainIDs)
        valid_chains = [cid for cid in chains if str(cid).strip()]
        
        if not valid_chains:
            logger.warning("No valid chains found in structure")
            return []
        
        chain_z_coords = []
        for chain in valid_chains:
            try:
                sel = universe.select_atoms(f"chainID {chain}")
                if sel.n_atoms > 0:
                    z = sel.center_of_mass()[2]
                    chain_z_coords.append((chain, z))
                    logger.debug(f"Chain {chain}: Z-coordinate = {z:.2f}")
            except Exception as e:
                logger.warning(f"Error processing chain {chain}: {e}")
                continue
        
        chain_z_coords.sort(key=lambda x: x[1], reverse=True)
        ordered_chains = [item[0] for item in chain_z_coords]
        
        # keep the origin oder of the chains in same layer
        layer1, layer2, rest = ordered_chains[:nprotof], ordered_chains[nprotof:2*nprotof], ordered_chains[2*nprotof:]
        pos = {item: i  for i, item in enumerate(valid_chains)}
        layer1_reordered = sorted(layer1, key=lambda x: pos[x])
        layer2_reordered = sorted(layer2, key=lambda x: pos[x])
        result = layer1_reordered + layer2_reordered + rest
        logger.info(f"Chain order (top to bottom): {result}")
        return result
    
    def build_long_core(self, nprotof: int = 2, nlayer: int = 40) -> str:
        """
        Build extended fibril core by stacking layers.
        
        Args:
            nprotof: Number of protofilaments per layer
            nlayer: Number of layers to build
            
        Returns:
            Path to output PDB file
        """
        if not self.pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {self.pdb_file}")
        
        logger.info(f"Building {nlayer}-layer structure with {nprotof} protofilaments per layer")
        
        # Load structure
        u = mda.Universe(str(self.pdb_file))
        order_chains = self.get_top_chains(u)
        
        if len(order_chains) < 2 * nprotof:
            raise ValueError(
                f"Insufficient chains ({len(order_chains)}) for {nprotof} protofilaments. "
                f"Need at least {2 * nprotof} chains."
            )
        
        # Define layer selections
        tops = " ".join(order_chains[:2*nprotof])
        top1 = " ".join(order_chains[:nprotof])
        top2 = " ".join(order_chains[nprotof:2*nprotof])
        
        logger.debug(f"Top 2 layers: {tops}")
        logger.debug(f"Reference layer (top1): {top1}")
        logger.debug(f"Mobile layer (top2): {top2}")
        
        # Select atoms
        sel = u.select_atoms(f"chainID {tops}")
        ref = u.select_atoms(f"chainID {top1}")
        mobile = sel.copy()
        
        # Prepare output
        output_file = f"{self.pdb_id}_{nlayer}layers.pdb"
        out_path = self.work_dir / output_file
        
        alphabet = string.ascii_uppercase
        
        with mda.Writer(str(out_path), multiframe=False, reindex=False) as writer:
            for i in range(nlayer):
                logger.debug(f"Processing layer {i+1}/{nlayer}")
                
                # Align mobile layer to reference
                align.alignto(mobile, ref, select=(f"chainID {top2}", f"chainID {top1}"))
                ref = mobile.copy()
                
                # Extract top layer
                top_copy = mda.Merge(mobile.select_atoms(f"chainID {top1}"))
                top = top_copy.atoms
                old_ids = np.unique(top.chainIDs)
                n = len(old_ids)
                
                # Reassign chain IDs
                if n == 1:
                    new_id = 'A' if i % 2 == 0 else 'B'
                    top.chainIDs = new_id
                else:
                    for k, old_id in enumerate(old_ids):
                        mask = (top.chainIDs == old_id)
                        new_id = alphabet[k % 26]
                        top.atoms[mask].chainIDs = new_id
                
                writer.write(top)
        
        logger.info(f"Successfully built extended structure: {out_path}")
        return str(out_path)
    
    def read_missing_residues(self) -> Dict[str, List[int]]:
        """
        Read missing/unobserved residues from CIF file.
        
        Returns:
            Dictionary mapping chain IDs to lists of missing residue numbers
        """
        if not self.cif_file.exists():
            logger.warning(f"CIF file not found: {self.cif_file}")
            return {}
        
        try:
            doc = gemmi.cif.read_file(str(self.cif_file))
            block = doc.sole_block()
            loop = block.find_loop("_pdbx_unobs_or_zero_occ_residues.id")
            
            if loop is None:
                logger.info("No missing residues found in structure")
                return {}
            
            chain_loop = block.find_loop("_pdbx_unobs_or_zero_occ_residues.auth_asym_id")
            resid_loop = block.find_loop("_pdbx_unobs_or_zero_occ_residues.auth_seq_id")
            result = {}
            for chain, resid in zip(chain_loop, resid_loop):
                if chain not in result:
                    result[chain] = []
                result[chain].append(int(resid))
            
            logger.info(f"Found missing residues in {len(result)} chain(s)")
            for chain, residues in result.items():
                logger.debug(f"Chain {chain}: {len(residues)} missing residues")
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading missing residues: {e}")
            return {}
    
    def create_alignment_file(self, nprotof: int = 2, nlayer: int = 40,
                             output_file: str = "alignment.ali") -> None:
        """
        Create alignment file for MODELLER loop modeling.
        
        Args:
            nprotof: Number of protofilaments per layer
            nlayer: Number of layers
            output_file: Output alignment filename
        """
        if not self.fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_file}")
        
        # Read sequence
        with open(self.fasta_file, 'r') as f:
            lines = f.readlines()
            seq = lines[1].strip() if len(lines) > 1 else ""
        
        if not seq:
            raise ValueError("Empty sequence in FASTA file")
        
        logger.info(f"Creating alignment file for sequence of length {len(seq)}")
        
        nchains = nprotof * nlayer
        missings = self.read_missing_residues()
        
        # Get chain order
        u = mda.Universe(str(self.pdb_file))
        tops = self.get_top_chains(u)[:nprotof]
        
        # Build alignment strings
        ali_path = self.work_dir / output_file
        
        with open(ali_path, 'w') as f:
            print(f'>P1;miss\nstructureX:{self.pdb_id}_{nlayer}layers::A::::::', file=f)
            
            tem1 = ''  # Sequence with gaps for missing residues
            tem2 = ''  # Complete sequence
            
            for layer in range(nlayer):
                for n, chain in enumerate(tops):
                    cnt = layer * nprotof + n
                    res_missed = missings.get(chain, [])
                    
                    # Create gapped sequence
                    seq_miss = ''.join(
                        '-' if i+1 in res_missed else char 
                        for i, char in enumerate(seq)
                    )
                    
                    separator = '/\n' if cnt != nchains - 1 else '*'
                    tem1 += seq_miss + separator
                    tem2 += seq + separator
            
            print(tem1, file=f)
            print("\n>P1;fill\nsequence:::::::::", file=f)
            print(tem2, file=f)
        
        logger.info(f"Alignment file created: {ali_path}")
    
    def add_missing_loops(self, nprotof: int = 2, 
                         nlayers: int = 40, ncycles: int = 5,
                         alignment_file: str = "alignment.ali") -> None:
        """
        Model missing loops using MODELLER.
        
        Args:
            nprotof: Number of protofilaments per layer
            nlayers: Number of layers
            ncycles: Number of optimization cycles
            alignment_file: Alignment file for MODELLER
        """
        logger.info("Starting loop modeling with MODELLER...")
        
        ali_path = self.work_dir / alignment_file
        logger.info("Creating alignment.ali file...")
        self.create_alignment_file(nprotof, nlayers, alignment_file)
        
        # Configure MODELLER
        log.verbose()
        env = Environ()
        env.schedule_scale = physical.Values(default=1.0, soft_sphere=0.7)
        env.io.atom_files_directory = [str(self.work_dir), './atom_files']
        
        # Initialize automodel
        ## Define segments for renaming
        segments = ['A' if i % 2 == 0 else 'B' for i in range(nlayers*nprotof)]
        residues = [1] * nlayers*nprotof

        class MyAutoModel(AutoModel):
            def special_patches(self, aln):
                self.rename_segments(segment_ids=segments, renumber_residues=residues)

        a = MyAutoModel(env, alnfile=str(ali_path), knowns='miss', sequence='fill')
        
        a.starting_model = 1
        a.ending_model = 1
        a.library_schedule = autosched.slow
        a.max_var_iterations = 500
        a.md_level = refine.slow
        a.repeat_optimization = ncycles
        a.max_molpdf = 1e8
        a.deviation = 4.0
        a.assess_methods = (assess.DOPE, assess.GA341, assess.normalized_dope)
        
        logger.info(f"Running MODELLER with {ncycles} optimization cycles...")
        a.make()
        
        logger.info("Loop modeling completed successfully")

def add_backbone_hydrogen(pdb_file, output_file):
    """
    Add backbone hydrogen atoms (H) to peptide chains in a PDB file.
    Preserves all original atoms (backbone and side chains).
    
    Parameters:
    -----------
    pdb_file : str
        Path to input PDB file
    output_file : str
        Path to output PDB file with added H atoms
    """
    
    def parse_atom_line(line):
        """Parse a PDB ATOM line and extract relevant information."""
        atom_serial = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_seq = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        occupancy = line[54:60].strip() if len(line) > 54 else "1.00"
        temp_factor = line[60:66].strip() if len(line) > 60 else "0.00"
        element = line[76:78].strip() if len(line) > 76 else ""
        
        return {
            'serial': atom_serial,
            'name': atom_name,
            'residue': residue_name,
            'chain': chain_id,
            'res_seq': residue_seq,
            'coords': np.array([x, y, z]),
            'occupancy': occupancy,
            'temp_factor': temp_factor,
            'element': element,
            'line': line
        }
    
    def format_atom_line(serial, atom_name, residue_name, chain_id, residue_seq, 
                        coords, occupancy="1.00", temp_factor="0.00", element="H"):
        """Format an ATOM line in PDB format."""
        return (f"ATOM  {serial:5d}  {atom_name:3s} {residue_name:3s} {chain_id}{residue_seq:4d}    "
                f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}{occupancy:>6s}{temp_factor:>6s}"
                f"          {element:>2s}\n")
    
    def calculate_h_position(n_coord, ca_coord, c_prev_coord):
        """
        Calculate the position of backbone H atom bonded to N.
        
        The H is placed along the N-C(previous) direction with proper geometry:
        - N-H bond length: 1.01 Å
        - C-N-H angle: ~120° (sp2 hybridization)
        
        Parameters:
        -----------
        n_coord : np.array
            Coordinates of N atom
        ca_coord : np.array
            Coordinates of CA atom (current residue)
        c_prev_coord : np.array
            Coordinates of C atom from previous residue (or current for first residue)
        """
        # Vector from C(prev) to N
        v_cn = n_coord - c_prev_coord
        v_cn = v_cn / np.linalg.norm(v_cn)
        
        # Vector from N to CA
        v_nca = ca_coord - n_coord
        v_nca = v_nca / np.linalg.norm(v_nca)
        
        # Bisector direction (for ideal geometry)
        # The H should be opposite to the peptide bond direction
        # but also considering the CA position
        bisector = v_cn - v_nca
        bisector = bisector / np.linalg.norm(bisector)
        
        # N-H bond length (standard: 1.01 Å)
        nh_bond_length = 1.01
        
        # Position H atom
        h_coord = n_coord + bisector * nh_bond_length
        
        return h_coord
    
    # Read PDB file and store all lines
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    # First pass: organize atoms by residue to find N, CA, C positions
    # and detect chain segments
    residue_data = []  # List of (line_idx, atom_dict) to maintain order
    residue_lookup = {}  # For quick lookup: (chain, res_seq, segment_id) -> {atom_name: atom_dict}
    
    current_segment_id = 0
    prev_chain = None
    prev_res_seq = None
    
    for idx, line in enumerate(lines):
        if not line.startswith('ATOM'):
            continue
        
        atom = parse_atom_line(line)
        chain = atom['chain']
        res_seq = atom['res_seq']
        
        # Detect chain break: different chain ID OR non-consecutive residue numbers
        is_new_segment = False
        if prev_chain is None:
            is_new_segment = True
        elif chain != prev_chain:
            is_new_segment = True
        elif abs(res_seq - prev_res_seq) > 1:
            is_new_segment = True
        
        if is_new_segment and prev_chain is not None:
            current_segment_id += 1
        
        # Store atom with its original line index and segment info
        atom['line_idx'] = idx
        atom['segment_id'] = current_segment_id
        residue_data.append((idx, atom))
        
        # Also store in lookup dictionary
        key = (chain, res_seq, current_segment_id)
        if key not in residue_lookup:
            residue_lookup[key] = {}
        residue_lookup[key][atom['name']] = atom
        
        prev_chain = chain
        prev_res_seq = res_seq
    
    # Second pass: build output with H atoms inserted after N atoms
    output_lines = []
    current_serial = 1
    
    # Group residues by segment
    segments = {}
    for idx, atom in residue_data:
        seg_id = atom['segment_id']
        if seg_id not in segments:
            segments[seg_id] = []
        key = (atom['chain'], atom['res_seq'], seg_id)
        if key not in [s['key'] for s in segments[seg_id]]:
            segments[seg_id].append({'key': key, 'first_idx': idx})
    
    # Track which residues we've added H to
    h_added = set()
    
    # Process all original atoms in order
    for idx, atom in residue_data:
        chain = atom['chain']
        res_seq = atom['res_seq']
        seg_id = atom['segment_id']
        key = (chain, res_seq, seg_id)
        
        # Write the current atom
        output_lines.append(format_atom_line(
            current_serial, atom['name'], atom['residue'],
            atom['chain'], atom['res_seq'], atom['coords'],
            atom['occupancy'], atom['temp_factor'], 
            atom['element'] if atom['element'] else atom['name'][0]
        ))
        current_serial += 1
        
        # If this is an N atom and we haven't added H yet for this residue
        if atom['name'] == 'N' and key not in h_added:
            res_atoms = residue_lookup[key]
            
            # Skip proline (PRO) residues - they don't have backbone H
            if atom['residue'] == 'PRO':
                continue
            
            # Check if we have necessary atoms (N, CA, C)
            if 'CA' in res_atoms:
                h_added.add(key)
                
                # Find if this is the first residue in the segment
                segment_residues = [s['key'] for s in segments[seg_id]]
                is_first_in_segment = (key == segment_residues[0])
                
                c_coord = None
                if not is_first_in_segment:
                    # Try to use previous residue's C atom
                    res_index = segment_residues.index(key)
                    if res_index > 0:
                        prev_key = segment_residues[res_index - 1]
                        if prev_key in residue_lookup and 'C' in residue_lookup[prev_key]:
                            c_coord = residue_lookup[prev_key]['C']['coords']
                
                # If no previous C or first residue, use current residue's C
                if c_coord is None and 'C' in res_atoms:
                    c_coord = res_atoms['C']['coords']
                
                # Calculate and add H atom
                if c_coord is not None:
                    h_coord = calculate_h_position(
                        atom['coords'],
                        res_atoms['CA']['coords'],
                        c_coord
                    )
                    
                    output_lines.append(format_atom_line(
                        current_serial, 'H', atom['residue'],
                        atom['chain'], atom['res_seq'], h_coord,
                        atom['occupancy'], atom['temp_factor'], 'H'
                    ))
                    current_serial += 1
    
    # Write output file
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"Added backbone hydrogen atoms. Output saved to {output_file}")

def prepare_run_script(origin_script, new_script, alignment_file):
    new_section = f"""
\n## add position restraints to the core region
addRestraints.posre_amyloid(system, PDBFile(args.pdb), '{alignment_file}')
sim.context.reinitialize(preserveState=True)\n
"""
    with open(origin_script, 'r') as f:
        lines = f.readlines()
        insert_line = -1
        for line_idx, line in enumerate(lines):
            if line.startswith("### insert restaint here ###"):
                insert_line = line_idx + 1

    lines.insert(insert_line, new_section)
    with open(new_script, 'w') as f:
        f.writelines(lines)

def main():
    """
    Command-line interface for AmyloidFibrilBuilder.
    """
    parser = argparse.ArgumentParser(
        description='Build extended amyloid fibril structures from PDB templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python amyloid_builder.py 6msm --nlayers 40 --nprotof 2 --ncycles 5
  python amyloid_builder.py 6msm -n 40 -p 2 -c 5 -o my_fibril.pdb
  python amyloid_builder.py 6msm  # Use defaults: 40 layers, 5 cycles
        """
    )
    
    # Required argument
    parser.add_argument('pdb_id', type=str, help='PDB identifier (e.g., 6msm)')
    # Optional arguments
    parser.add_argument('--nlayers', '-n', type=int, default=40, help='Number of layers to build (default: 40)')
    parser.add_argument('--nprotof', '-p', type=int, default=None, help='Number of protofilaments per layer (default: None, automatically found)')
    parser.add_argument('--ncycles', '-c', type=int, default=5, help='Number of optimization cycles for loop modeling (default: 5)')
    parser.add_argument('--output', '-o', type=str, default='conf.pdb', help='Output filename for final structure if hyres true')
    parser.add_argument('--terminal', '-t', type=str, default='neutral', help='If hyres true, Charged status of terminus: neutral, charged, NT, and CT')
    parser.add_argument('--work-dir', '-w', type=str, default='.', help='Working directory for input/output files (default: current directory)')
    parser.add_argument('--nohyres', action='store_true', help='if convert to HyRes model after building, default true')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging output')
    parser.add_argument('--version', action='version', version='AmyloidFibrilBuilder 1.0.0')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("=" * 70)
        logger.info(f"AMYLOID FIBRIL BUILDER - PDB: {args.pdb_id.upper()}")
        logger.info("=" * 70)
        logger.info(f"Parameters: {args.nlayers} layers, {args.ncycles} cycles of relaxation")
        logger.info("=" * 70 + "\n")
        
        # Initialize builder
        builder = AmyloidFibrilBuilder(args.pdb_id, work_dir=args.work_dir)
        
        # Step 1: Download files
        logger.info("[Step 1/3] Downloading PDB, CIF, and FASTA files...")
        if not builder.download_files():
            logger.error("Failed to download files. Please check PDB ID and network connection.")
            sys.exit(1)
        logger.info("✓ Files downloaded successfully\n")
        
        # Step 2: Build extended structure
        logger.info(f"[Step 2/3] Building {args.nlayers}-layer core structure...")
        nprotof = args.nprotof
        if nprotof is None:
            logger.info("Getting number of protofilament from CIF...")
            nprotof = builder.get_protof_number()
            logger.info(f'Find {nprotof} protofilaments\n')
        # build core
        core_pdb = builder.build_long_core(
            nprotof=nprotof,
            nlayer=args.nlayers,
        )
        logger.info(f"✓ Core structure built: {core_pdb}\n")
        
        # Step 3: Model missing loops
        logger.info(f"[Step 3/3] Modeling missing loops ({args.ncycles} optimization cycles)...")
        logger.info("This may take several minutes depending on structure size...")
        builder.add_missing_loops(nprotof=nprotof, nlayers=args.nlayers, ncycles=args.ncycles)
        logger.info("✓ Loop modeling completed\n")
        
        # Success message
        logger.info("=" * 70)
        logger.info("SUCCESS! Final structure ready")
        logger.info("=" * 70)
        logger.info(f"Modeled structure: fill.B99990001.pdb")
        logger.info("=" * 70)
        
        # Convert to HyRes model
        if not args.nohyres:
            ## add backbone hydrogen
            tmp_file = f"{args.pdb_id}_fill_addH.pdb"
            add_backbone_hydrogen('fill.B99990001.pdb', tmp_file)
            ## convert2cg
            Convert2CG.at2cg(pdb_in=tmp_file, pdb_out=args.output, terminal=args.terminal, cleanup=True)
            ## add restraint to run script
            prepare_run_script(f'{args.work_dir}/run_latest.py', f'{args.work_dir}/run_latest_restraint.py', f"{args.work_dir}/alignment.ali")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\n\nOperation cancelled by user")
        sys.exit(130)
    except FileNotFoundError as e:
        logger.error(f"\nFile not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"\nInvalid parameter: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()