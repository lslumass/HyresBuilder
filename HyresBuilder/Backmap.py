"""
HyresBuilder backmapping module.

Ultra-fast HyRes coarse-grained to all-atom backmapping with top-5 rotamer library
and clash detection.
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
    module_dir = Path(__file__).parent
    map_dir = module_dir / 'map'
    
    if not map_dir.exists():
        raise FileNotFoundError(
            f"Map directory not found at {map_dir}. "
            "Please ensure ideal structures are in HyresBuilder/HyresBuilder/map/"
        )
    
    return str(map_dir)


def check_clashes(mobile, existing_atoms, clash_distance=2.0, exclude_residue=None):
    """
    Check for clashes between mobile atoms and existing atoms.
    
    Parameters
    ----------
    mobile : AtomGroup
        Atoms to check for clashes
    existing_atoms : list of AtomGroup
        List of already placed atom groups
    clash_distance : float
        Distance threshold for clash detection (Angstroms)
    exclude_residue : int, optional
        Residue ID to exclude from clash check (for neighboring residues)
        
    Returns
    -------
    bool
        True if clash detected, False otherwise
    int
        Number of clashing atom pairs
    """
    if len(existing_atoms) == 0:
        return False, 0
    
    # Get mobile heavy atoms (exclude hydrogens for clash check)
    mobile_heavy = mobile.select_atoms("not name H*")
    if len(mobile_heavy) == 0:
        return False, 0
    
    mobile_coords = mobile_heavy.positions
    clash_count = 0
    
    # Check against each existing atom group
    for existing in existing_atoms:
        # Skip if same residue or neighboring residue
        if exclude_residue is not None:
            existing_resids = existing.resids
            if len(existing_resids) > 0:
                # Exclude same residue and immediate neighbors
                if np.any(abs(existing_resids[0] - exclude_residue) <= 1):
                    continue
        
        # Get heavy atoms only
        existing_heavy = existing.select_atoms("not name H*")
        if len(existing_heavy) == 0:
            continue
            
        existing_coords = existing_heavy.positions
        
        # Calculate pairwise distances
        for mc in mobile_coords:
            distances = np.sqrt(np.sum((existing_coords - mc)**2, axis=1))
            min_dist = np.min(distances)
            
            if min_dist < clash_distance:
                clash_count += 1
                # Early return if we find any clash
                return True, clash_count
    
    return False, 0


def resolve_clashes_with_rotamers(resname, refs, mobile, existing_atoms, 
                                  clash_distance=2.0, verbose=False):
    """
    Try all rotamers and select the one with fewest clashes.
    
    Parameters
    ----------
    resname : str
        Residue name
    refs : AtomGroup
        Reference CG atoms
    mobile : Universe
        Mobile structure
    existing_atoms : list
        List of already placed atoms
    clash_distance : float
        Clash detection threshold
    verbose : bool
        Print clash information
        
    Returns
    -------
    bool
        True if clash-free rotamer found, False otherwise
    """
    # Import rotamer module
    try:
        from .Rotamer import ROTAMER_LIBRARY, set_chi_angle
    except ImportError:
        from Rotamer import ROTAMER_LIBRARY, set_chi_angle
    
    if resname not in ROTAMER_LIBRARY:
        return False
    
    rotamers = ROTAMER_LIBRARY[resname]
    
    # Get reference positions
    CA_r = refs.atoms[0].position
    ref_positions = [refs.atoms[i].position for i in range(min(len(refs.atoms), 4))]
    
    if len(ref_positions) > 1:
        ref_com = np.mean(ref_positions[1:], axis=0)
    else:
        ref_com = CA_r
    
    # Get side chain atoms
    CA = mobile.select_atoms("name CA", updating=False).atoms[0].position
    side_atoms = mobile.select_atoms("not name N CA C O HA", updating=False)
    
    if len(side_atoms) == 0:
        return False
    
    # Save original positions
    original_positions = side_atoms.atoms.positions.copy()
    
    best_score = np.inf
    best_positions = None
    best_clash_count = np.inf
    current_resid = mobile.residues[0].resid
    
    # Try each rotamer
    for rotamer_idx, (chi1, chi2, chi3, chi4, prob) in enumerate(rotamers):
        # Reset to original
        side_atoms.atoms.positions = original_positions.copy()
        
        # Apply chi angles
        if chi1 != 0:
            set_chi_angle(mobile, 1, chi1)
        if chi2 != 0:
            set_chi_angle(mobile, 2, chi2)
        if chi3 != 0:
            set_chi_angle(mobile, 3, chi3)
        if chi4 != 0:
            set_chi_angle(mobile, 4, chi4)
        
        # Check for clashes
        has_clash, clash_count = check_clashes(
            side_atoms, existing_atoms, clash_distance, exclude_residue=current_resid
        )
        
        # Calculate geometric score
        current_com = side_atoms.center_of_mass()
        dist = np.linalg.norm(current_com - ref_com)
        
        # Combined score: penalize clashes heavily, then consider geometry and probability
        if has_clash:
            score = 1000 + clash_count * 100 + dist - prob  # Heavy penalty for clashes
        else:
            score = dist - 2.0 * prob  # Prefer good geometry and high probability
        
        if verbose and has_clash:
            print(f"    Rotamer {rotamer_idx+1}: {clash_count} clashes, score={score:.2f}")
        
        # Update best
        if score < best_score or (score == best_score and clash_count < best_clash_count):
            best_score = score
            best_positions = side_atoms.atoms.positions.copy()
            best_clash_count = clash_count
            
            # If we found a clash-free rotamer with good score, we can stop
            if not has_clash and score < 0:
                break
    
    # Apply best rotamer
    if best_positions is not None:
        side_atoms.atoms.positions = best_positions
        
    # Check if we still have clashes
    final_clash, final_count = check_clashes(
        side_atoms, existing_atoms, clash_distance, exclude_residue=current_resid
    )
    
    if verbose and final_clash:
        print(f"    Warning: Best rotamer still has {final_count} clashes")
    
    return not final_clash


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


def backmap_structure(input_file, output_file, map_dir=None, verbose=True, 
                     check_clashes=True, clash_distance=2.0):
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
    check_clashes : bool, optional
        Enable clash detection and resolution (default: True)
    clash_distance : float, optional
        Distance threshold for clash detection in Angstroms (default: 2.0)
        
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
        print(f'Clash detection: {"enabled" if check_clashes else "disabled"}')
        if check_clashes:
            print(f'Clash distance: {clash_distance} Å')
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
    
    # Track statistics
    clash_count = 0
    resolved_count = 0
    unresolved_count = 0
    
    # Process residues
    with mda.Writer(output_file, multiframe=False, reindex=False) as writer:
        atom_idx = 1
        placed_atoms = []  # Track placed atoms for clash detection
        
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
            
            # Side chain optimization with clash detection
            if res_data['needs_sidechain'] and res_data['ref_indices'] is not None:
                refs = hyres.atoms[res_data['ref_indices']]
                
                if check_clashes and len(placed_atoms) > 0:
                    # Try to find clash-free rotamer
                    clash_free = resolve_clashes_with_rotamers(
                        resname, refs, mobile, placed_atoms, 
                        clash_distance, verbose=verbose and (i + 1) % 50 == 0
                    )
                    
                    if not clash_free:
                        clash_count += 1
                        unresolved_count += 1
                        if verbose:
                            print(f'  Warning: Residue {resid} ({resname}) has unresolved clashes')
                    else:
                        if clash_count > 0:  # There was a clash but we resolved it
                            resolved_count += 1
                else:
                    # Standard optimization without clash checking
                    opt_side_chain(resname, refs, mobile)
            
            # Add hydrogens if needed
            if res_data['needs_hydrogen'] and res_data['h_indices'] is not None:
                h_atoms = hyres.atoms[res_data['h_indices']]
                h_atoms.atoms.ids = np.arange(atom_idx, atom_idx + len(h_atoms))
                atom_idx += len(h_atoms)
                writer.write(h_atoms)
                placed_atoms.append(h_atoms)
            
            # Add mobile atoms
            mobile.atoms.ids = np.arange(atom_idx, atom_idx + len(mobile.atoms))
            atom_idx += len(mobile.atoms)
            writer.write(mobile.atoms)
            placed_atoms.append(mobile.atoms)
    
    if verbose:
        print(f'  Processed {len(cache.residue_data)}/{len(cache.residue_data)} residues')
        print('-' * 60)
        if check_clashes:
            print(f'Clash statistics:')
            print(f'  Clashes detected: {clash_count}')
            print(f'  Clashes resolved: {resolved_count}')
            print(f'  Clashes unresolved: {unresolved_count}')
            if unresolved_count > 0:
                print(f'  Note: Consider running energy minimization on output structure')
            print('-' * 60)
        print(f'Done! Output written to {output_file}')
        print(f'Total atoms: {atom_idx - 1}')


def backmap_trajectory(input_file, topology, output, map_dir=None, stride=1, 
                      verbose=True, check_clashes=True, clash_distance=2.0):
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
    check_clashes : bool, optional
        Enable clash detection and resolution (default: True)
    clash_distance : float, optional
        Distance threshold for clash detection in Angstroms (default: 2.0)
        
    Example
    -------
    >>> from HyresBuilder.backmap import backmap_trajectory
    >>> backmap_trajectory('traj.xtc', 'topology.pdb', 'output.xtc', stride=10)
    """
    # Import rotamer module
    try:
        from Rotamer import opt_side_chain
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
        print(f'Clash detection: {"enabled" if check_clashes else "disabled"}')
        if check_clashes:
            print(f'Clash distance: {clash_distance} Å')
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
    total_clashes = 0
    total_resolved = 0
    total_unresolved = 0
    
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
            placed_atoms = []
            frame_clashes = 0
            
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
                    
                    if check_clashes and len(placed_atoms) > 0:
                        clash_free = resolve_clashes_with_rotamers(
                            resname, refs, mobile, placed_atoms, clash_distance
                        )
                        if not clash_free:
                            frame_clashes += 1
                    else:
                        opt_side_chain(resname, refs, mobile)
                
                # Add hydrogens
                if res_data['needs_hydrogen'] and res_data['h_indices'] is not None:
                    h_atoms = hyres.atoms[res_data['h_indices']]
                    h_atoms.atoms.ids = np.arange(atom_idx, atom_idx + len(h_atoms))
                    atom_idx += len(h_atoms)
                    frame_atoms.append(h_atoms)
                    placed_atoms.append(h_atoms)
                
                # Add mobile atoms
                mobile.atoms.ids = np.arange(atom_idx, atom_idx + len(mobile.atoms))
                atom_idx += len(mobile.atoms)
                frame_atoms.append(mobile.atoms)
                placed_atoms.append(mobile.atoms)
            
            # Write frame
            merged = mda.Merge(*frame_atoms)
            writer.write(merged.atoms)
            frame_count += 1
            
            if frame_clashes > 0:
                total_clashes += frame_clashes
                total_unresolved += frame_clashes
    
    elapsed = time.time() - start_time
    
    if verbose:
        print('-' * 60)
        print(f'Completed {frame_count} frames in {elapsed:.2f}s')
        print(f'Average speed: {frame_count/elapsed:.2f} frames/s')
        if check_clashes and total_clashes > 0:
            print(f'\nClash statistics:')
            print(f'  Total clashes: {total_clashes}')
            print(f'  Average per frame: {total_clashes/frame_count:.1f}')
            print(f'  Note: Consider running energy minimization')
        print(f'Done! Output written to {output}')


# Command-line interface
def main():
    """Command-line interface for backmapping."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HyresBuilder: Ultra-fast HyRes to all-atom backmapping'
    )
    
    parser.add_argument('input', help='Input HyRes structure/trajectory file')
    parser.add_argument('output', help='Output all-atom file')
    parser.add_argument('--topology', '-t', help='Topology file for trajectory')
    parser.add_argument('--stride', type=int, default=1, 
                       help='Process every Nth frame (default: 1)')
    parser.add_argument('--map-dir', help='Directory with ideal structures')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Suppress output')
    parser.add_argument('--no-clash-check', action='store_true',
                       help='Disable clash detection (faster but may produce clashes)')
    parser.add_argument('--clash-distance', type=float, default=2.0,
                       help='Clash detection distance in Angstroms (default: 2.0)')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    check_clashes = not args.no_clash_check
    
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
                             verbose=verbose, check_clashes=check_clashes,
                             clash_distance=args.clash_distance)
        else:
            backmap_structure(args.input, args.output, 
                            map_dir=args.map_dir, verbose=verbose,
                            check_clashes=check_clashes,
                            clash_distance=args.clash_distance)
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()