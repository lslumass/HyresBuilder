"""
HyresBuilder backmapping module.

Ultra-fast HyRes coarse-grained to all-atom backmapping with top-5 rotamer library.
"""

import sys
import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.topology.guessers import guess_types
import time
from pathlib import Path


def get_map_directory():
    """Get the path to the map directory containing ideal structures."""
    # This file is in HyresBuilder/HyresBuilder/backmap.py
    # Map directory is HyresBuilder/HyresBuilder/map/
    module_dir = Path(__file__).parent
    map_dir = module_dir / 'map'
    
    if not map_dir.exists():
        raise FileNotFoundError(
            f"Map directory not found at {map_dir}. "
            "Please ensure ideal structures are in HyresBuilder/HyresBuilder/map/"
        )
    
    return str(map_dir)


class StructureCache:
    """Cache for ideal structures and pre-computed data."""
    
    def __init__(self, hyres, map_dir=None):
        if map_dir is None:
            map_dir = get_map_directory()
        
        self.map_dir = map_dir
        self.ideal_structures = {}
        self.backbone_sel = 'name N CA C O'
        
        # Pre-load all unique residue types
        unique_resnames = set(hyres.residues.resnames)
        
        for resname in unique_resnames:
            try:
                ideal_file = os.path.join(map_dir, f'{resname}_ideal.pdb')
                self.ideal_structures[resname] = mda.Universe(ideal_file)
            except (FileNotFoundError, OSError):
                pass
        
        # Pre-compute residue metadata
        self.residue_data = []
        for res in hyres.residues:
            res_sel = hyres.select_atoms(f"resid {res.resid}", updating=False)
            
            data = {
                'resname': res.resname,
                'resid': res.resid,
                'segid': res_sel.segids[0],
                'chainID': res_sel.chainIDs[0],
                'needs_sidechain': res.resname not in ['GLY', 'PRO', 'ALA'],
                'needs_hydrogen': res.resname != 'PRO',
            }
            
            bb_sel = res_sel.select_atoms(self.backbone_sel, updating=False)
            data['bb_indices'] = bb_sel.indices
            
            if data['needs_sidechain']:
                ref_sel = res_sel.select_atoms("name CA CB CC CD CE CF", updating=False)
                data['ref_indices'] = ref_sel.indices if len(ref_sel) > 0 else None
            else:
                data['ref_indices'] = None
            
            if data['needs_hydrogen']:
                h_sel = res_sel.select_atoms("name H", updating=False)
                data['h_indices'] = h_sel.indices if len(h_sel) > 0 else None
            else:
                data['h_indices'] = None
            
            self.residue_data.append(data)


def backmap_structure(input_file, output_file, map_dir=None, verbose=True):
    """
    Backmap a single HyRes structure to all-atom representation.
    
    Parameters
    ----------
    input_file : str
        Path to input HyRes PDB file
    output_file : str
        Path to output all-atom PDB file
    map_dir : str, optional
        Directory containing ideal structures. If None, uses HyresBuilder/HyresBuilder/map/
    verbose : bool, optional
        Print progress information (default: True)
        
    Example
    -------
    >>> from HyresBuilder.backmap import backmap_structure
    >>> backmap_structure('input.pdb', 'output.pdb')
    """
    # Import rotamer module
    try:
        from .Rotamer import opt_side_chain
    except ImportError:
        from Rotamer import opt_side_chain
    
    if map_dir is None:
        map_dir = get_map_directory()
    
    if verbose:
        print('HyresBuilder - Single Structure Backmapping')
        print(f'Input: {input_file}')
        print(f'Output: {output_file}')
        print('-' * 60)
    
    # Load HyRes structure
    if verbose:
        print('Loading HyRes structure...')
    
    hyres = mda.Universe(input_file)
    guessed_eles = guess_types(hyres.atoms.names)
    hyres.add_TopologyAttr('elements', guessed_eles)
    
    if verbose:
        print(f'Loaded {len(hyres.residues)} residues, {len(hyres.atoms)} atoms')
    
    # Build cache
    if verbose:
        print('Building structure cache...')
    
    cache = StructureCache(hyres, map_dir)
    
    if verbose:
        print(f'Cached {len(cache.ideal_structures)} ideal structures')
        print('-' * 60)
        print('Processing residues...')
    
    # Process residues
    with mda.Writer(output_file, multiframe=False, reindex=False) as writer:
        atom_idx = 1
        
        for i, res_data in enumerate(cache.residue_data):
            resname = res_data['resname']
            resid = res_data['resid']
            
            if verbose and (i + 1) % 50 == 0:
                print(f'  Processed {i + 1}/{len(cache.residue_data)} residues')
            
            if resname not in cache.ideal_structures:
                if verbose:
                    print(f'  Warning: No ideal structure for {resname} at resid {resid}')
                continue
            
            # Create mobile structure
            mobile = cache.ideal_structures[resname].copy()
            mobile.residues.resids = resid
            mobile.segments.segids = res_data['segid']
            mobile.atoms.chainIDs = res_data['chainID']
            
            # Get reference atoms for alignment
            ref_atoms = hyres.atoms[res_data['bb_indices']]
            
            # Align
            align.alignto(mobile, ref_atoms, select=cache.backbone_sel, match_atoms=False)
            
            # Side chain optimization
            if res_data['needs_sidechain'] and res_data['ref_indices'] is not None:
                refs = hyres.atoms[res_data['ref_indices']]
                opt_side_chain(resname, refs, mobile)
            
            # Add hydrogens if needed
            if res_data['needs_hydrogen'] and res_data['h_indices'] is not None:
                h_atoms = hyres.atoms[res_data['h_indices']]
                h_atoms.atoms.ids = np.arange(atom_idx, atom_idx + len(h_atoms))
                atom_idx += len(h_atoms)
                writer.write(h_atoms)
            
            # Add mobile atoms
            mobile.atoms.ids = np.arange(atom_idx, atom_idx + len(mobile.atoms))
            atom_idx += len(mobile.atoms)
            writer.write(mobile.atoms)
    
    if verbose:
        print(f'  Processed {len(cache.residue_data)}/{len(cache.residue_data)} residues')
        print('-' * 60)
        print(f'Done! Output written to {output_file}')
        print(f'Total atoms: {atom_idx - 1}')


def backmap_trajectory(input_file, topology, output, map_dir=None, stride=1, verbose=True):
    """
    Backmap a HyRes trajectory to all-atom representation.
    
    Parameters
    ----------
    input_file : str
        Path to input HyRes trajectory file (xtc, dcd, trr)
    topology : str
        Path to topology PDB file
    output : str
        Path to output all-atom trajectory file
    map_dir : str, optional
        Directory containing ideal structures. If None, uses HyresBuilder/HyresBuilder/map/
    stride : int, optional
        Process every Nth frame (default: 1, all frames)
    verbose : bool, optional
        Print progress information (default: True)
        
    Example
    -------
    >>> from HyresBuilder.backmap import backmap_trajectory
    >>> backmap_trajectory('traj.xtc', 'topology.pdb', 'output.xtc', stride=10)
    """
    # Import rotamer module
    try:
        from .Rotamer import opt_side_chain
    except ImportError:
        from Rotamer import opt_side_chain
    
    if map_dir is None:
        map_dir = get_map_directory()
    
    if verbose:
        print('HyresBuilder - Trajectory Backmapping')
        print(f'Input: {input_file}')
        print(f'Topology: {topology}')
        print(f'Output: {output}')
        print(f'Stride: {stride}')
        print('-' * 60)
    
    # Load trajectory
    if verbose:
        print('Loading trajectory...')
    
    hyres = mda.Universe(topology, input_file)
    guessed_eles = guess_types(hyres.atoms.names)
    hyres.add_TopologyAttr('elements', guessed_eles)
    
    n_frames = len(hyres.trajectory)
    frames_to_process = list(range(0, n_frames, stride))
    
    if verbose:
        print(f'Loaded {n_frames} frames with {len(hyres.residues)} residues')
        print(f'Will process {len(frames_to_process)} frames')
    
    # Build cache
    if verbose:
        print('Building structure cache...')
    
    cache = StructureCache(hyres, map_dir)
    
    if verbose:
        print(f'Cached {len(cache.ideal_structures)} ideal structures')
        print('-' * 60)
        print('Processing trajectory...')
    
    start_time = time.time()
    frame_count = 0
    
    with mda.Writer(output, multiframe=True, reindex=False) as writer:
        for frame_idx in frames_to_process:
            hyres.trajectory[frame_idx]
            
            if verbose and frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = (frame_count + 1) / elapsed if elapsed > 0 else 0
                eta = (len(frames_to_process) - frame_count - 1) / fps if fps > 0 else 0
                print(f'Frame {frame_count + 1}/{len(frames_to_process)} '
                      f'({fps:.2f} fps, ETA: {eta:.1f}s)')
            
            frame_atoms = []
            atom_idx = 1
            
            for res_data in cache.residue_data:
                resname = res_data['resname']
                
                if resname not in cache.ideal_structures:
                    continue
                
                # Create mobile structure
                mobile = cache.ideal_structures[resname].copy()
                mobile.residues.resids = res_data['resid']
                mobile.segments.segids = res_data['segid']
                mobile.atoms.chainIDs = res_data['chainID']
                
                # Get reference atoms
                ref_atoms = hyres.atoms[res_data['bb_indices']]
                
                # Align
                align.alignto(mobile, ref_atoms, select=cache.backbone_sel, match_atoms=False)
                
                # Side chain optimization
                if res_data['needs_sidechain'] and res_data['ref_indices'] is not None:
                    refs = hyres.atoms[res_data['ref_indices']]
                    opt_side_chain(resname, refs, mobile)
                
                # Add hydrogens
                if res_data['needs_hydrogen'] and res_data['h_indices'] is not None:
                    h_atoms = hyres.atoms[res_data['h_indices']]
                    h_atoms.atoms.ids = np.arange(atom_idx, atom_idx + len(h_atoms))
                    atom_idx += len(h_atoms)
                    frame_atoms.append(h_atoms)
                
                # Add mobile atoms
                mobile.atoms.ids = np.arange(atom_idx, atom_idx + len(mobile.atoms))
                atom_idx += len(mobile.atoms)
                frame_atoms.append(mobile.atoms)
            
            # Write frame
            merged = mda.Merge(*frame_atoms)
            writer.write(merged.atoms)
            frame_count += 1
    
    elapsed = time.time() - start_time
    
    if verbose:
        print('-' * 60)
        print(f'Completed {frame_count} frames in {elapsed:.2f}s')
        print(f'Average speed: {frame_count/elapsed:.2f} frames/s')
        print(f'Done! Output written to {output}')


# Command-line interface
def main():
    """Command-line interface for backmapping."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Backmap: Ultra-fast HyRes to all-atom backmapping'
    )
    
    parser.add_argument('input', help='Input HyRes structure/trajectory file')
    parser.add_argument('output', help='Output all-atom file')
    parser.add_argument('--topology', '-t', help='Topology file for trajectory')
    parser.add_argument('--stride', type=int, default=1, 
                       help='Process every Nth frame (default: 1)')
    parser.add_argument('--map-dir', help='Directory with ideal structures')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Suppress output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Check if trajectory or single structure
    input_ext = os.path.splitext(args.input)[1].lower()
    is_trajectory = input_ext in ['.xtc', '.dcd', '.trr']
    
    try:
        if is_trajectory:
            if not args.topology:
                print("Error: --topology required for trajectory input")
                sys.exit(1)
            backmap_trajectory(args.input, args.topology, args.output,
                             map_dir=args.map_dir, stride=args.stride, 
                             verbose=verbose)
        else:
            backmap_structure(args.input, args.output, 
                            map_dir=args.map_dir, verbose=verbose)
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()