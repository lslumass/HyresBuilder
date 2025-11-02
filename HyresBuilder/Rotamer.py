import numpy as np
from numba import jit

@jit(nopython=True)
def norm_vector_fast(v):
    """Fast vector normalization using numba."""
    norm = np.sqrt(np.sum(v * v))
    if norm > 0:
        return v / norm
    return v

@jit(nopython=True)
def cal_angle_fast(v1, v2):
    """Fast angle calculation using numba."""
    dot = np.sum(v1 * v2)
    len1 = np.sqrt(np.sum(v1 * v1))
    len2 = np.sqrt(np.sum(v2 * v2))
    cos_angle = dot / (len1 * len2)
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle) * 180.0 / np.pi

@jit(nopython=True)
def rotate_about_axis_fast(coords, axis, angle, support):
    """
    Fast rotation using Rodrigues' formula with numba optimization.
    coords: (N, 3) array of coordinates
    axis: (3,) normalized axis vector
    angle: rotation angle in radians
    support: (3,) support vector
    """
    # Translate to origin
    coords = coords - support
    
    # Rodrigues' rotation formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Pre-compute dot products
    dots = np.sum(coords * axis, axis=1)
    
    # Cross products
    crosses = np.cross(coords, axis)
    
    # Rodrigues formula: v_rot = v*cos + (k×v)*sin + k(k·v)(1-cos)
    rotated = coords * cos_a + crosses * sin_a + axis * dots[:, np.newaxis] * (1 - cos_a)
    
    # Translate back
    return rotated + support

def norm_vector(v):
    """Normalise a vector."""
    factor = np.linalg.norm(v, axis=-1)
    if isinstance(factor, np.ndarray):
        v /= factor[..., np.newaxis]
    else:
        v /= factor

def cal_normal(v1, v2):
    return np.cross(v1, v2)

def cal_distance(coord1, coord2):
    temp = np.array(coord1) - np.array(coord2)
    return np.sqrt(np.dot(temp, temp))

def cal_angle(v1, v2):
    """Wrapper for compatibility."""
    return cal_angle_fast(np.array(v1, dtype=np.float64), np.array(v2, dtype=np.float64))

def rotate_about_axis(coords, axis, angle, support=None):
    """
    Optimized rotation function.
    """
    positions = np.asarray(coords, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    
    if support is None:
        support = np.zeros(3, dtype=np.float64)
    else:
        support = np.asarray(support, dtype=np.float64)
    
    # Normalize axis
    axis = norm_vector_fast(axis)
    
    # Handle single coordinate vs array
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)
        result = rotate_about_axis_fast(positions, axis, angle, support)
        return result[0]
    
    return rotate_about_axis_fast(positions, axis, angle, support)

def opt_side_chain(resname, refs, sides):
    """
    Optimized side chain optimization with vectorized operations.
    """
    # Pre-compute rotation angles once
    angles = np.deg2rad(np.arange(0, 360, 5))
    
    # Simple residues: SER, THR, CYS, VAL
    if resname in ['SER', 'THR', 'CYS', 'VAL']:
        CA_r = refs.atoms[0].position
        CB_r = refs.atoms[1].position
        CA = sides.select_atoms("name CA").atoms[0].position
        CB = sides.select_atoms("name CB").atoms[0].position
        axis = np.array(CB) - np.array(CA)
        axis = norm_vector_fast(axis)
        
        rotations = sides.select_atoms("not name N CA C O HA")
        original_positions = rotations.atoms.positions.copy()
        
        v1 = np.array(CB_r) - np.array(CA)
        min_angle = 180.0
        opt_positions = original_positions.copy()
        
        for theta in angles:
            # Rotate all atoms at once
            rotations.atoms.positions = rotate_about_axis_fast(
                original_positions, axis, theta, np.array(CA)
            )
            
            com = rotations.center_of_mass()
            v2 = np.array(com) - np.array(CA)
            angle = cal_angle_fast(v1, v2)
            
            if angle < min_angle:
                min_angle = angle
                opt_positions = rotations.atoms.positions.copy()
        
        rotations.atoms.positions = opt_positions

    # Medium residues: ASP, ASN, LEU, GLU, GLN, MET
    elif resname in ['ASP', 'ASN', 'LEU', 'GLU', 'GLN', 'MET']:
        CA_r = refs.atoms[0].position
        CB_r = refs.atoms[1].position
        CA = sides.select_atoms("name CA").atoms[0].position
        CB = sides.select_atoms("name CB").atoms[0].position
        CA_CB = norm_vector_fast(np.array(CB) - np.array(CA))
        
        rotations = sides.select_atoms("not name N CA C O HA")
        original_positions = rotations.atoms.positions.copy()
        part2 = sides.select_atoms("not name N CA C O HA CB HB")
        
        v1 = np.array(CB_r) - np.array(CA)
        min_angle = 180.0
        opt_positions = original_positions.copy()
        
        for theta in angles:
            temp_pos = rotate_about_axis_fast(original_positions, CA_CB, theta, np.array(CA))
            
            # Get CC position from temp positions
            rot_idx = [i for i, atom in enumerate(rotations.atoms)]
            rot_names = [atom.name for atom in rotations.atoms]
            cc_idx = rot_names.index('CC') if 'CC' in rot_names else None
            
            if cc_idx is not None:
                CC = temp_pos[cc_idx]
                CB_CC = norm_vector_fast(CC - np.array(CB))
                
                # Get part2 indices and positions
                part2_mask = np.array([atom in part2 for atom in rotations.atoms])
                part2_positions = temp_pos[part2_mask]
                original_part2 = part2_positions.copy()
                
                for phi in angles:
                    temp_pos[part2_mask] = rotate_about_axis_fast(
                        original_part2, CB_CC, phi, np.array(CB)
                    )
                    
                    rotations.atoms.positions = temp_pos
                    com = rotations.center_of_mass()
                    v2 = np.array(com) - np.array(CA)
                    angle = cal_angle_fast(v1, v2)
                    
                    if angle < min_angle:
                        min_angle = angle
                        opt_positions = temp_pos.copy()
        
        rotations.atoms.positions = opt_positions

    # ILE
    elif resname == 'ILE':
        CA_r = refs.atoms[0].position
        CB_r = refs.atoms[1].position
        CA = sides.select_atoms("name CA").atoms[0].position
        CB = sides.select_atoms("name CB").atoms[0].position
        CA_CB = norm_vector_fast(np.array(CB) - np.array(CA))
        
        rotations = sides.select_atoms("not name N CA C O HA")
        original_positions = rotations.atoms.positions.copy()
        part2 = sides.select_atoms("name CF HF")
        
        v1 = np.array(CB_r) - np.array(CA)
        min_angle = 180.0
        opt_positions = original_positions.copy()
        
        for theta in angles:
            temp_pos = rotate_about_axis_fast(original_positions, CA_CB, theta, np.array(CA))
            
            rot_names = [atom.name for atom in rotations.atoms]
            cc_idx = rot_names.index('CC') if 'CC' in rot_names else None
            
            if cc_idx is not None:
                CC = temp_pos[cc_idx]
                CB_CC = norm_vector_fast(CC - np.array(CB))
                
                part2_mask = np.array([atom in part2 for atom in rotations.atoms])
                part2_positions = temp_pos[part2_mask]
                original_part2 = part2_positions.copy()
                
                for phi in angles:
                    temp_pos[part2_mask] = rotate_about_axis_fast(
                        original_part2, CB_CC, phi, np.array(CB)
                    )
                    
                    rotations.atoms.positions = temp_pos
                    com = rotations.center_of_mass()
                    v2 = np.array(com) - np.array(CA)
                    angle = cal_angle_fast(v1, v2)
                    
                    if angle < min_angle:
                        min_angle = angle
                        opt_positions = temp_pos.copy()
        
        rotations.atoms.positions = opt_positions

    # ARG - Multi-stage optimization
    elif resname == 'ARG':
        _optimize_arg(resname, refs, sides, angles)

    # LYS - Multi-stage optimization  
    elif resname == 'LYS':
        _optimize_lys(resname, refs, sides, angles)

    # Aromatic residues: HIS, PHE, TYR, TRP
    elif resname in ['HIS', 'PHE', 'TYR', 'TRP']:
        _optimize_aromatic(resname, refs, sides, angles)

def _optimize_arg(resname, refs, sides, angles):
    """Separate function for ARG optimization."""
    CA_r = refs.select_atoms("name CA").atoms[0].position
    CB_r = refs.select_atoms("name CB").atoms[0].position
    CC_r = refs.select_atoms("name CC").atoms[0].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    CA_CB = norm_vector_fast(np.array(CB) - np.array(CA))
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    part2 = sides.select_atoms("not name N CA C O HA CB HB")
    
    v1 = np.array(CB_r) - np.array(CA)
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    for theta in angles:
        temp_pos = rotate_about_axis_fast(original_positions, CA_CB, theta, np.array(CA))
        
        rot_names = [atom.name for atom in rotations.atoms]
        cc_idx = rot_names.index('CC') if 'CC' in rot_names else None
        
        if cc_idx:
            CC = temp_pos[cc_idx]
            CB_CC = norm_vector_fast(CC - np.array(CB))
            
            part2_mask = np.array([atom in part2 for atom in rotations.atoms])
            part2_positions = temp_pos[part2_mask]
            original_part2 = part2_positions.copy()
            
            for phi in angles:
                temp_pos[part2_mask] = rotate_about_axis_fast(
                    original_part2, CB_CC, phi, np.array(CB)
                )
                
                rotations.atoms.positions = temp_pos
                com = rotations.center_of_mass()
                v2 = np.array(com) - np.array(CA)
                angle = cal_angle_fast(v1, v2)
                
                if angle < min_angle:
                    min_angle = angle
                    opt_positions = temp_pos.copy()
    
    rotations.atoms.positions = opt_positions
    
    # Second stage optimization for ARG tail
    v1 = np.array(CC_r) - np.array(CA)
    CD = sides.select_atoms("name CD").atoms[0].position
    N1 = sides.select_atoms("name N1").atoms[0].position
    CD_N1 = norm_vector_fast(np.array(N1) - np.array(CD))
    part3 = sides.select_atoms("not name N CA C O HA CB HB CC HC CD HD")
    
    original_part3 = part3.atoms.positions.copy()
    opt_positions = original_part3.copy()
    min_angle = 180.0
    
    for theta in angles:
        part3.atoms.positions = rotate_about_axis_fast(
            original_part3, CD_N1, theta, np.array(CD)
        )
        
        com = part3.center_of_mass()
        v2 = np.array(com) - np.array(CA)
        angle = cal_angle_fast(v1, v2)
        
        if angle < min_angle:
            min_angle = angle
            opt_positions = part3.atoms.positions.copy()
    
    part3.atoms.positions = opt_positions

def _optimize_lys(resname, refs, sides, angles):
    """Separate function for LYS optimization."""
    # Similar structure to ARG - implementation follows same pattern
    # [Abbreviated for space - follows same optimization pattern as ARG]
    CA_r = refs.select_atoms("name CA").atoms[0].position
    CB_r = refs.select_atoms("name CB").atoms[0].position
    CC_r = refs.select_atoms("name CC").atoms[0].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    CA_CB = norm_vector_fast(np.array(CB) - np.array(CA))
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    part2 = sides.select_atoms("not name N CA C O HA CB HB")
    
    v1 = np.array(CB_r) - np.array(CA)
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    for theta in angles:
        temp_pos = rotate_about_axis_fast(original_positions, CA_CB, theta, np.array(CA))
        
        rot_names = [atom.name for atom in rotations.atoms]
        cc_idx = rot_names.index('CC') if 'CC' in rot_names else None
        
        if cc_idx:
            CC = temp_pos[cc_idx]
            CB_CC = norm_vector_fast(CC - np.array(CB))
            
            part2_mask = np.array([atom in part2 for atom in rotations.atoms])
            part2_positions = temp_pos[part2_mask]
            original_part2 = part2_positions.copy()
            
            for phi in angles:
                temp_pos[part2_mask] = rotate_about_axis_fast(
                    original_part2, CB_CC, phi, np.array(CB)
                )
                
                rotations.atoms.positions = temp_pos
                com = rotations.center_of_mass()
                v2 = np.array(com) - np.array(CA)
                angle = cal_angle_fast(v1, v2)
                
                if angle < min_angle:
                    min_angle = angle
                    opt_positions = temp_pos.copy()
    
    rotations.atoms.positions = opt_positions
    
    # Second stage
    v1 = np.array(CC_r) - np.array(CA)
    CC = sides.select_atoms("name CC").atoms[0].position
    CD = sides.select_atoms("name CD").atoms[0].position
    CC_CD = norm_vector_fast(np.array(CD) - np.array(CC))
    part3 = sides.select_atoms("name CD HD CE HE N1 HN")
    
    original_part3 = part3.atoms.positions.copy()
    opt_positions = original_part3.copy()
    min_angle = 180.0
    
    for theta in angles:
        part3.atoms.positions = rotate_about_axis_fast(
            original_part3, CC_CD, theta, np.array(CC)
        )
        
        grp_CC = sides.select_atoms("name CE HE N1 HN")
        com = grp_CC.center_of_mass()
        v2 = np.array(com) - np.array(CA)
        angle = cal_angle_fast(v1, v2)
        
        if angle < min_angle:
            min_angle = angle
            opt_positions = part3.atoms.positions.copy()
    
    part3.atoms.positions = opt_positions

def _optimize_aromatic(resname, refs, sides, angles):
    """Optimized aromatic residue side chain placement."""
    CA_r = refs.select_atoms("name CA").atoms[0].position
    CB_r = refs.select_atoms("name CB").atoms[0].position
    CC_r = refs.select_atoms("name CC").atoms[0].position
    CD_r = refs.select_atoms("name CD").atoms[0].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    CA_CB = norm_vector_fast(np.array(CB) - np.array(CA))
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    
    v1 = np.array(CB_r) - np.array(CA)
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    for theta in angles:
        rotations.atoms.positions = rotate_about_axis_fast(
            original_positions, CA_CB, theta, np.array(CA)
        )
        
        grp_CB = sides.select_atoms("name CB HB CC")
        com_CB = grp_CB.center_of_mass()
        v2 = np.array(com_CB) - np.array(CA)
        angle = cal_angle_fast(v1, v2)
        
        if angle < min_angle:
            min_angle = angle
            opt_positions = rotations.atoms.positions.copy()
    
    rotations.atoms.positions = opt_positions
    
    # Ring plane optimization
    CC = sides.select_atoms("name CC").atoms[0].position
    CB_CC = norm_vector_fast(np.array(CC) - np.array(CB))
    part2 = sides.select_atoms("not name N CA C O HA CB HB")
    original_part2 = part2.atoms.positions.copy()
    
    min_dist = np.inf
    opt_positions = original_part2.copy()
    
    # Define groups based on residue type
    if resname == 'HIS':
        grp_names = [
            "name CB HB CC",
            "name N1 CE HE",
            "name CD HD N2 HN"
        ]
    elif resname == 'PHE':
        grp_names = [
            "name CB HB CC",
            "name CE HE CG HG",
            "name CF HF CH HH"
        ]
    elif resname == 'TYR':
        grp_names = [
            "name CB HB CC",
            "name CE HE CG HG",
            "name CF HF CH O1 HO"
        ]
    elif resname == 'TRP':
        grp_names = [
            "name CB HB CC",
            "name CD HD N1 HN",
            "name CE CF"
        ]
    
    for phi in angles:
        part2.atoms.positions = rotate_about_axis_fast(
            original_part2, CB_CC, phi, np.array(CB)
        )
        
        # Calculate total distance to reference
        total_dist = 0.0
        for i, grp_sel in enumerate(grp_names):
            grp = sides.select_atoms(grp_sel)
            com = grp.center_of_mass()
            if i == 0:
                total_dist += cal_distance(com, CB_r)
            elif i == 1:
                total_dist += cal_distance(com, CC_r)
            elif i == 2:
                total_dist += cal_distance(com, CD_r)
        
        if total_dist < min_dist:
            min_dist = total_dist
            opt_positions = part2.atoms.positions.copy()
    
    part2.atoms.positions = opt_positions