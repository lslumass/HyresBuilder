"""
HyresBuilder backmapping module.

Ultra-fast HyRes coarse-grained to all-atom backmapping with advanced clash detection
using fine-grained rotamer search.
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


def check_clashes(mobile_coords, existing_coords, clash_distance=2.0):
    """
    Fast clash detection using vectorized distance calculation.
    
    Parameters
    ----------
    mobile_coords : ndarray, shape (N, 3)
        Coordinates of atoms to check
    existing_coords : ndarray, shape (M, 3)
        Coordinates of already placed atoms
    clash_distance : float
        Distance threshold for clash detection (Angstroms)
        
    Returns
    -------
    bool
        True if clash detected
    int
        Number of clashing atom pairs
    float
        Minimum distance found
    """
    if len(existing_coords) == 0:
        return False, 0, np.inf
    
    # Vectorized distance calculation
    # For each mobile atom, find minimum distance to any existing atom
    min_distances = []
    for mc in mobile_coords:
        distances = np.sqrt(np.sum((existing_coords - mc)**2, axis=1))
        min_dist = np.min(distances)
        min_distances.append(min_dist)
    
    min_distances = np.array(min_distances)
    clashes = min_distances < clash_distance
    clash_count = np.sum(clashes)
    min_overall = np.min(min_distances)
    
    return clash_count > 0, int(clash_count), float(min_overall)


def rotate_side_chain_fine_search(resname, refs, sides, existing_coords=None, 
                                   clash_distance=2.0, angle_step=15):
    """
    Optimize side chain with fine-grained rotamer search and clash avoidance.
    
    Uses original brute-force approach but with clash detection to find
    clash-free conformations.
    
    Parameters
    ----------
    resname : str
        Residue name
    refs : AtomGroup
        Reference CG atoms
    sides : Universe
        Mobile all-atom structure
    existing_coords : ndarray, optional
        Coordinates of already placed atoms for clash detection
    clash_distance : float
        Clash detection threshold in Angstroms
    angle_step : int
        Angle step size in degrees (smaller = more thorough, slower)
    """
    from .Rotamer import rotate_about_axis_fast, normalize_vector
    
    # Simple residues: SER, THR, CYS, VAL (1 chi angle)
    if resname in ['SER', 'THR', 'CYS', 'VAL']:
        CA_r = refs.atoms[0].position
        CB_r = refs.atoms[1].position
        CA = sides.select_atoms("name CA").atoms[0].position
        CB = sides.select_atoms("name CB").atoms[0].position
        axis = normalize_vector(np.array(CB) - np.array(CA))
        
        rotations = sides.select_atoms("not name N CA C O HA")
        original_positions = rotations.atoms.positions.copy()
        
        v1 = np.array(CB_r) - np.array(CA)
        min_angle = 180.0
        opt_positions = original_positions.copy()
        min_clash_count = np.inf
        
        angles = np.arange(0, 360, angle_step)
        
        for theta_deg in angles:
            theta = np.radians(theta_deg)
            rotations.atoms.positions = rotate_about_axis_fast(
                original_positions, axis, theta, np.array(CA)
            )
            
            # Check clashes if existing_coords provided
            if existing_coords is not None:
                side_heavy = rotations.select_atoms("not name H*")
                has_clash, clash_count, min_dist = check_clashes(
                    side_heavy.positions, existing_coords, clash_distance
                )
                
                # Score: prefer clash-free, then good geometry
                if has_clash:
                    score = 1000 + clash_count
                else:
                    com = rotations.center_of_mass()
                    v2 = np.array(com) - np.array(CA)
                    angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                             (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    score = np.degrees(angle)
                
                if score < min_angle or (score == min_angle and clash_count < min_clash_count):
                    min_angle = score
                    min_clash_count = clash_count
                    opt_positions = rotations.atoms.positions.copy()
            else:
                # Original geometric scoring
                com = rotations.center_of_mass()
                v2 = np.array(com) - np.array(CA)
                angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                         (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                angle_deg = np.degrees(angle)
                
                if angle_deg < min_angle:
                    min_angle = angle_deg
                    opt_positions = rotations.atoms.positions.copy()
        
        rotations.atoms.positions = opt_positions

    # Medium complexity: ASP, ASN, LEU, GLU, GLN, MET (2 chi angles)
    elif resname in ['ASP', 'ASN', 'LEU', 'GLU', 'GLN', 'MET']:
        CA_r = refs.atoms[0].position
        CB_r = refs.atoms[1].position
        CA = sides.select_atoms("name CA").atoms[0].position
        CB = sides.select_atoms("name CB").atoms[0].position
        CA_CB = normalize_vector(np.array(CB) - np.array(CA))
        
        rotations = sides.select_atoms("not name N CA C O HA")
        original_positions = rotations.atoms.positions.copy()
        part2 = sides.select_atoms("not name N CA C O HA CB HB")
        
        v1 = np.array(CB_r) - np.array(CA)
        min_score = np.inf
        opt_positions = original_positions.copy()
        
        angles = np.arange(0, 360, angle_step)
        
        for theta_deg in angles:
            theta = np.radians(theta_deg)
            temp_pos = rotate_about_axis_fast(original_positions, CA_CB, theta, np.array(CA))
            
            # Get CC position
            rot_names = [atom.name for atom in rotations.atoms]
            if 'CC' in rot_names:
                cc_idx = rot_names.index('CC')
                CC = temp_pos[cc_idx]
                CB_CC = normalize_vector(CC - np.array(CB))
                
                part2_mask = np.array([atom in part2 for atom in rotations.atoms])
                part2_positions = temp_pos[part2_mask]
                original_part2 = part2_positions.copy()
                
                for phi_deg in angles:
                    phi = np.radians(phi_deg)
                    temp_pos[part2_mask] = rotate_about_axis_fast(
                        original_part2, CB_CC, phi, np.array(CB)
                    )
                    
                    rotations.atoms.positions = temp_pos
                    
                    # Clash detection
                    if existing_coords is not None:
                        side_heavy = rotations.select_atoms("not name H*")
                        has_clash, clash_count, min_dist = check_clashes(
                            side_heavy.positions, existing_coords, clash_distance
                        )
                        
                        if has_clash:
                            score = 1000 + clash_count
                        else:
                            com = rotations.center_of_mass()
                            v2 = np.array(com) - np.array(CA)
                            angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                                     (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                            score = np.degrees(angle)
                    else:
                        com = rotations.center_of_mass()
                        v2 = np.array(com) - np.array(CA)
                        angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                                 (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                        score = np.degrees(angle)
                    
                    if score < min_score:
                        min_score = score
                        opt_positions = temp_pos.copy()
        
        rotations.atoms.positions = opt_positions

    # ILE (2 chi angles, special case)
    elif resname == 'ILE':
        CA_r = refs.atoms[0].position
        CB_r = refs.atoms[1].position
        CA = sides.select_atoms("name CA").atoms[0].position
        CB = sides.select_atoms("name CB").atoms[0].position
        CA_CB = normalize_vector(np.array(CB) - np.array(CA))
        
        rotations = sides.select_atoms("not name N CA C O HA")
        original_positions = rotations.atoms.positions.copy()
        part2 = sides.select_atoms("name CF HF")
        
        v1 = np.array(CB_r) - np.array(CA)
        min_score = np.inf
        opt_positions = original_positions.copy()
        
        angles = np.arange(0, 360, angle_step)
        
        for theta_deg in angles:
            theta = np.radians(theta_deg)
            temp_pos = rotate_about_axis_fast(original_positions, CA_CB, theta, np.array(CA))
            
            rot_names = [atom.name for atom in rotations.atoms]
            if 'CC' in rot_names:
                cc_idx = rot_names.index('CC')
                CC = temp_pos[cc_idx]
                CB_CC = normalize_vector(CC - np.array(CB))
                
                part2_mask = np.array([atom in part2 for atom in rotations.atoms])
                part2_positions = temp_pos[part2_mask]
                original_part2 = part2_positions.copy()
                
                for phi_deg in angles:
                    phi = np.radians(phi_deg)
                    temp_pos[part2_mask] = rotate_about_axis_fast(
                        original_part2, CB_CC, phi, np.array(CB)
                    )
                    
                    rotations.atoms.positions = temp_pos
                    
                    if existing_coords is not None:
                        side_heavy = rotations.select_atoms("not name H*")
                        has_clash, clash_count, min_dist = check_clashes(
                            side_heavy.positions, existing_coords, clash_distance
                        )
                        
                        if has_clash:
                            score = 1000 + clash_count
                        else:
                            com = rotations.center_of_mass()
                            v2 = np.array(com) - np.array(CA)
                            angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                                     (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                            score = np.degrees(angle)
                    else:
                        com = rotations.center_of_mass()
                        v2 = np.array(com) - np.array(CA)
                        angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                                 (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                        score = np.degrees(angle)
                    
                    if score < min_score:
                        min_score = score
                        opt_positions = temp_pos.copy()
        
        rotations.atoms.positions = opt_positions

    # For complex residues (ARG, LYS, HIS, PHE, TYR, TRP), use top-5 rotamer library
    # to avoid excessive computation time
    else:
        from .Rotamer import opt_side_chain
        opt_side_chain(resname, refs, sides)


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
                     check_clashes_enabled=True, clash_distance=2.0, angle_step=15):
    """
    Backmap a single HyRes structure to all-atom representation.
    
    Parameters
    ----------
    input_file : str
        Path to input HyRes PDB file
    output_file : str
        Path to output all-atom PDB file
    map_dir : str, optional
        Directory containing ideal structures
    verbose : bool, optional
        Print progress information (default: True)
    check_clashes_enabled : bool, optional
        Enable clash detection and avoidance (default: True)
    clash_distance : float, optional
        Distance threshold for clash detection in Angstroms (default: 2.0)
    angle_step : int, optional
        Angle step for rotamer search in degrees (default: 15)
        Smaller = more thorough but slower. Try 10 for better quality, 20 for speed.
    """
    if map_dir is None:
        map_dir = get_map_directory()
    
    if verbose:
        print('HyresBuilder - Single Structure Backmapping')
        print(f'Input: {input_file}')
        print(f'Output: {output_file}')
        print(f'Clash detection: {"enabled" if check_clashes_enabled else "disabled"}')
        if check_clashes_enabled:
            print(f'Clash distance: {clash_distance} Å')
            print(f'Angle step: {angle_step}°')
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
    
    # Track existing atom coordinates for clash detection
    existing_coords_list = []
    clash_count = 0
    
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
            
            # Side chain optimization with clash detection
            if res_data['needs_sidechain'] and res_data['ref_indices'] is not None:
                refs = hyres.atoms[res_data['ref_indices']]
                
                if check_clashes_enabled and len(existing_coords_list) > 0:
                    # Combine all existing coordinates (exclude neighboring residues)
                    existing_coords = []
                    for j, coords in enumerate(existing_coords_list):
                        # Exclude immediate neighbors (within ±2 residues)
                        if abs(j - i) > 2:
                            existing_coords.append(coords)
                    
                    if len(existing_coords) > 0:
                        existing_coords = np.vstack(existing_coords)
                        rotate_side_chain_fine_search(
                            resname, refs, mobile, existing_coords, 
                            clash_distance, angle_step
                        )
                        
                        # Check if clashes remain
                        side_heavy = mobile.select_atoms("not name N CA C O HA H*")
                        has_clash, n_clash, min_dist = check_clashes(
                            side_heavy.positions, existing_coords, clash_distance
                        )
                        if has_clash:
                            clash_count += 1
                            if verbose:
                                print(f'  Warning: {resname} {resid} has {n_clash} clashes (min dist: {min_dist:.2f} Å)')
                    else:
                        rotate_side_chain_fine_search(
                            resname, refs, mobile, None, clash_distance, angle_step
                        )
                else:
                    # No clash checking
                    rotate_side_chain_fine_search(
                        resname, refs, mobile, None, clash_distance, angle_step
                    )
            
            # Store heavy atom coordinates for future clash checks
            if check_clashes_enabled:
                heavy_atoms = mobile.select_atoms("not name H*")
                existing_coords_list.append(heavy_atoms.positions.copy())
            
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
        if check_clashes_enabled:
            print(f'Clash statistics:')
            print(f'  Residues with clashes: {clash_count}')
            clash_rate = 100 * clash_count / len(cache.residue_data)
            print(f'  Clash rate: {clash_rate:.1f}%')
            if clash_count > 0:
                print(f'  Recommendation: Run energy minimization or use smaller angle_step')
            print('-' * 60)
        print(f'Done! Output written to {output_file}')
        print(f'Total atoms: {atom_idx - 1}')


def backmap_trajectory(input_file, topology, output, map_dir=None, stride=1, 
                      verbose=True, check_clashes_enabled=False, clash_distance=2.0,
                      angle_step=20):
    """
    Backmap a HyRes trajectory to all-atom representation.
    
    Note: Clash checking is disabled by default for trajectories due to performance.
    
    Parameters
    ----------
    input_file : str
        Path to input HyRes trajectory file
    topology : str
        Path to topology PDB file
    output : str
        Path to output all-atom trajectory file
    map_dir : str, optional
        Directory containing ideal structures
    stride : int, optional
        Process every Nth frame (default: 1)
    verbose : bool, optional
        Print progress information (default: True)
    check_clashes_enabled : bool, optional
        Enable clash detection (default: False for speed)
    clash_distance : float, optional
        Clash threshold in Angstroms (default: 2.0)
    angle_step : int, optional
        Angle step in degrees (default: 20 for speed)
    """
    if map_dir is None:
        map_dir = get_map_directory()
    
    if verbose:
        print('HyresBuilder - Trajectory Backmapping')
        print(f'Input: {input_file}')
        print(f'Topology: {topology}')
        print(f'Output: {output}')
        print(f'Stride: {stride}')
        print(f'Clash detection: {"enabled" if check_clashes_enabled else "disabled"}')
        if check_clashes_enabled:
            print(f'Clash distance: {clash_distance} Å')
            print(f'Angle step: {angle_step}°')
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
            existing_coords_list = []
            
            for i, res_data in enumerate(cache.residue_data):
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
                    
                    if check_clashes_enabled and len(existing_coords_list) > 0:
                        existing_coords = []
                        for j, coords in enumerate(existing_coords_list):
                            if abs(j - i) > 2:
                                existing_coords.append(coords)
                        
                        if len(existing_coords) > 0:
                            existing_coords = np.vstack(existing_coords)
                            rotate_side_chain_fine_search(
                                resname, refs, mobile, existing_coords,
                                clash_distance, angle_step
                            )
                        else:
                            rotate_side_chain_fine_search(
                                resname, refs, mobile, None,
                                clash_distance, angle_step
                            )
                    else:
                        rotate_side_chain_fine_search(
                            resname, refs, mobile, None,
                            clash_distance, angle_step
                        )
                
                # Store coordinates
                if check_clashes_enabled:
                    heavy_atoms = mobile.select_atoms("not name H*")
                    existing_coords_list.append(heavy_atoms.positions.copy())
                
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
        description='HyresBuilder: Ultra-fast HyRes to all-atom backmapping with clash detection'
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
                       help='Disable clash detection (faster)')
    parser.add_argument('--clash-distance', type=float, default=2.0,
                       help='Clash detection distance in Angstroms (default: 2.0)')
    parser.add_argument('--angle-step', type=int, default=15,
                       help='Angle step for rotamer search in degrees (default: 15)')
    
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
                             clash_distance=args.clash_distance,
                             angle_step=args.angle_step)
        else:
            backmap_structure(args.input, args.output, 
                            map_dir=args.map_dir, verbose=verbose,
                            check_clashes=check_clashes,
                            clash_distance=args.clash_distance,
                            angle_step=args.angle_step)
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()