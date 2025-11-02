"""
HyresBuilder rotamer module.

Optimized geometric operations for side chain optimization with Numba acceleration.

Place this file in: HyresBuilder/HyresBuilder/rotamer.py
"""

import numpy as np
import math

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    import warnings
    warnings.warn(
        "Numba not available. Performance will be significantly reduced. "
        "Install with: pip install numba",
        ImportWarning
    )
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    prange = range


# ============================================================================
# Core Geometric Functions (Numba-accelerated)
# ============================================================================

@jit(nopython=True, cache=True)
def normalize_vector(v):
    """
    Fast vector normalization.
    
    Parameters
    ----------
    v : ndarray, shape (3,)
        Vector to normalize
        
    Returns
    -------
    ndarray
        Normalized vector
    """
    norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if norm > 1e-10:
        return v / norm
    return v.copy()

@jit(nopython=True, cache=True)
def rotation_matrix_rodrigues(axis, angle):
    """
    Compute rotation matrix using Rodrigues' formula.
    
    Parameters
    ----------
    axis : ndarray, shape (3,)
        Normalized rotation axis
    angle : float
        Rotation angle in radians
        
    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    
    x, y, z = axis[0], axis[1], axis[2]
    
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ]
    ], dtype=np.float64)

@jit(nopython=True, cache=True)
def rotate_coordinates(coords, axis, angle, center):
    """
    Rotate coordinates around axis through center point.
    
    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Coordinates to rotate
    axis : ndarray, shape (3,)
        Normalized rotation axis
    angle : float
        Rotation angle in radians
    center : ndarray, shape (3,)
        Center of rotation
        
    Returns
    -------
    ndarray, shape (N, 3)
        Rotated coordinates
    """
    R = rotation_matrix_rodrigues(axis, angle)
    centered = coords - center
    rotated = np.dot(centered, R.T)
    return rotated + center

@jit(nopython=True, cache=True)
def calculate_angle_between_vectors(v1, v2):
    """
    Calculate angle between two vectors in degrees.
    
    Parameters
    ----------
    v1, v2 : ndarray, shape (3,)
        Vectors
        
    Returns
    -------
    float
        Angle in degrees
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    len1 = np.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
    len2 = np.sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2])
    
    if len1 * len2 < 1e-10:
        return 0.0
    
    cos_angle = dot / (len1 * len2)
    # Clamp to avoid numerical errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    return np.degrees(np.arccos(cos_angle))

@jit(nopython=True, cache=True)
def calculate_center_of_mass(coords):
    """
    Calculate center of mass of coordinates.
    
    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Atomic coordinates
        
    Returns
    -------
    ndarray, shape (3,)
        Center of mass
    """
    return np.mean(coords, axis=0)


# ============================================================================
# Non-JIT wrapper functions for compatibility
# ============================================================================

def norm_vector(v):
    """
    Normalise a vector (compatibility wrapper).
    
    Parameters
    ----------
    v : ndarray
        The array containing the vector(s).
        The vectors are represented by the last axis.
    """
    factor = np.linalg.norm(v, axis=-1)
    if isinstance(factor, np.ndarray):
        v /= factor[..., np.newaxis]
    else:
        if factor > 0:
            v /= factor

def cal_normal(v1, v2):
    """Calculate normal vector (cross product)."""
    return np.cross(v1, v2)

def cal_distance(coord1, coord2):
    """Calculate Euclidean distance between two points."""
    temp = np.array(coord1) - np.array(coord2)
    euclid_dist = np.sqrt(np.dot(temp.T, temp))
    return euclid_dist

def cal_angle(v1, v2):
    """Calculate angle between two vectors in degrees."""
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    return calculate_angle_between_vectors(v1, v2)

def rotate_about_axis_fast(coords, axis, angle, support):
    """
    Fast rotation wrapper for compatibility.
    
    Parameters
    ----------
    coords : ndarray
        Coordinates to rotate
    axis : ndarray
        Rotation axis (will be normalized)
    angle : float
        Rotation angle in radians
    support : ndarray
        Center of rotation
        
    Returns
    -------
    ndarray
        Rotated coordinates
    """
    coords = np.asarray(coords, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    
    # Normalize axis
    axis = normalize_vector(axis)
    
    # Handle single coordinate vs array
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
        result = rotate_coordinates(coords, axis, angle, support)
        return result[0]
    
    return rotate_coordinates(coords, axis, angle, support)

def rotate_about_axis(coords, axis, angle, support=None):
    """
    Rotate the given atoms or coordinates about a given axis by a given angle.
    
    Parameters
    ----------
    coords : ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The coordinates to transform
    axis : array-like, length=3
        A vector representing the direction of the rotation axis
    angle : float
        The rotation angle in radians
    support : array-like, length=3, optional
        An optional support vector for the rotation axis
        By default, the center of the rotation is at (0,0,0)
    
    Returns
    -------
    transformed : ndarray
        A copy of the input coordinates, rotated about the given axis
    """
    positions = np.asarray(coords, dtype=np.float64)
    
    if support is None:
        support = np.zeros(3, dtype=np.float64)
    else:
        support = np.asarray(support, dtype=np.float64)
    
    # Normalize axis
    axis = np.asarray(axis, dtype=np.float64).copy()
    if np.linalg.norm(axis) == 0:
        raise ValueError("Length of the rotation axis is 0")
    norm_vector(axis)
    
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = positions.ndim
    orig_shape = positions.shape
    
    if orig_ndim > 2:
        positions = positions.reshape(-1, 3)
    
    # Apply rotation using fast function
    positions = rotate_about_axis_fast(positions, axis, angle, support)
    
    # Reshape back into original shape
    if orig_ndim > 2:
        positions = positions.reshape(*orig_shape)
    
    return positions


# ============================================================================
# Side Chain Optimization Functions
# ============================================================================

def opt_side_chain(resname, refs, sides):
    """
    Optimize side chain conformation using systematic search.
    
    This is the main function called from backmap.py.
    It searches through dihedral angles to find the best fit to reference geometry.
    
    Parameters
    ----------
    resname : str
        Residue name (three-letter code)
    refs : AtomGroup
        Reference atoms from coarse-grained model (CA, CB, CC, CD, etc.)
    sides : Universe
        All-atom residue structure to optimize
    """
    
    # For simple residues: SER, THR, CYS, VAL (1 chi angle)
    if resname in ['SER', 'THR', 'CYS', 'VAL']:
        _optimize_1_chi(resname, refs, sides)
    
    # For medium residues: ASP, ASN, LEU, GLU, GLN, MET (2 chi angles)
    elif resname in ['ASP', 'ASN', 'LEU', 'GLU', 'GLN', 'MET']:
        _optimize_2_chi(resname, refs, sides)
    
    # For ILE (special 2 chi case)
    elif resname == 'ILE':
        _optimize_ile(resname, refs, sides)
    
    # For ARG (4 chi angles, multi-stage)
    elif resname == 'ARG':
        _optimize_arg(resname, refs, sides)
    
    # For LYS (4 chi angles, multi-stage)
    elif resname == 'LYS':
        _optimize_lys(resname, refs, sides)
    
    # For aromatic: HIS, PHE, TYR, TRP (special cases)
    elif resname in ['HIS', 'PHE', 'TYR', 'TRP']:
        _optimize_aromatic(resname, refs, sides)


def _optimize_1_chi(resname, refs, sides):
    """Optimize residues with 1 chi angle."""
    CA_r = refs.atoms[0].position
    CB_r = refs.atoms[1].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    
    axis = np.array(CB, dtype=np.float64) - np.array(CA, dtype=np.float64)
    axis = normalize_vector(axis)
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    
    v1 = np.array(CB_r, dtype=np.float64) - np.array(CA, dtype=np.float64)
    
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    # Pre-compute angles
    angles_deg = np.arange(0, 360, 5, dtype=np.float64)
    angles_rad = np.radians(angles_deg)
    
    CA_arr = np.array(CA, dtype=np.float64)
    
    for theta in angles_rad:
        # Rotate all atoms at once
        new_positions = rotate_coordinates(original_positions, axis, theta, CA_arr)
        
        # Calculate center of mass
        com = calculate_center_of_mass(new_positions)
        v2 = com - CA_arr
        
        # Calculate angle
        angle = calculate_angle_between_vectors(v1, v2)
        
        if angle < min_angle:
            min_angle = angle
            opt_positions = new_positions.copy()
    
    rotations.atoms.positions = opt_positions


def _optimize_2_chi(resname, refs, sides):
    """Optimize residues with 2 chi angles."""
    CA_r = refs.atoms[0].position
    CB_r = refs.atoms[1].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    
    CA_CB = normalize_vector(np.array(CB, dtype=np.float64) - np.array(CA, dtype=np.float64))
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    part2 = sides.select_atoms("not name N CA C O HA CB HB")
    
    v1 = np.array(CB_r, dtype=np.float64) - np.array(CA, dtype=np.float64)
    
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    angles_deg = np.arange(0, 360, 5, dtype=np.float64)
    angles_rad = np.radians(angles_deg)
    
    CA_arr = np.array(CA, dtype=np.float64)
    CB_arr = np.array(CB, dtype=np.float64)
    
    for theta in angles_rad:
        temp_pos = rotate_coordinates(original_positions, CA_CB, theta, CA_arr)
        
        # Get CC position from rotated coordinates
        rot_names = [atom.name for atom in rotations.atoms]
        if 'CC' in rot_names:
            cc_idx = rot_names.index('CC')
            CC = temp_pos[cc_idx]
            CB_CC = normalize_vector(CC - CB_arr)
            
            # Get part2 mask and positions
            part2_mask = np.array([atom in part2 for atom in rotations.atoms])
            part2_positions = temp_pos[part2_mask]
            original_part2 = part2_positions.copy()
            
            for phi in angles_rad:
                temp_pos[part2_mask] = rotate_coordinates(original_part2, CB_CC, phi, CB_arr)
                
                # Calculate COM and angle
                com = calculate_center_of_mass(temp_pos)
                v2 = com - CA_arr
                angle = calculate_angle_between_vectors(v1, v2)
                
                if angle < min_angle:
                    min_angle = angle
                    opt_positions = temp_pos.copy()
    
    rotations.atoms.positions = opt_positions


def _optimize_ile(resname, refs, sides):
    """Optimize ILE (special 2 chi case)."""
    CA_r = refs.atoms[0].position
    CB_r = refs.atoms[1].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    
    CA_CB = normalize_vector(np.array(CB, dtype=np.float64) - np.array(CA, dtype=np.float64))
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    part2 = sides.select_atoms("name CF HF")
    
    v1 = np.array(CB_r, dtype=np.float64) - np.array(CA, dtype=np.float64)
    
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    angles_rad = np.radians(np.arange(0, 360, 5, dtype=np.float64))
    CA_arr = np.array(CA, dtype=np.float64)
    CB_arr = np.array(CB, dtype=np.float64)
    
    for theta in angles_rad:
        temp_pos = rotate_coordinates(original_positions, CA_CB, theta, CA_arr)
        
        rot_names = [atom.name for atom in rotations.atoms]
        if 'CC' in rot_names:
            cc_idx = rot_names.index('CC')
            CC = temp_pos[cc_idx]
            CB_CC = normalize_vector(CC - CB_arr)
            
            part2_mask = np.array([atom in part2 for atom in rotations.atoms])
            part2_positions = temp_pos[part2_mask]
            original_part2 = part2_positions.copy()
            
            for phi in angles_rad:
                temp_pos[part2_mask] = rotate_coordinates(original_part2, CB_CC, phi, CB_arr)
                
                com = calculate_center_of_mass(temp_pos)
                v2 = com - CA_arr
                angle = calculate_angle_between_vectors(v1, v2)
                
                if angle < min_angle:
                    min_angle = angle
                    opt_positions = temp_pos.copy()
    
    rotations.atoms.positions = opt_positions


def _optimize_arg(resname, refs, sides):
    """Optimize ARG (multi-stage optimization)."""
    # Stage 1: Optimize first two chi angles
    CA_r = refs.select_atoms("name CA").atoms[0].position
    CB_r = refs.select_atoms("name CB").atoms[0].position
    CC_r = refs.select_atoms("name CC").atoms[0].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    
    CA_CB = normalize_vector(np.array(CB, dtype=np.float64) - np.array(CA, dtype=np.float64))
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    part2 = sides.select_atoms("not name N CA C O HA CB HB")
    
    v1 = np.array(CB_r, dtype=np.float64) - np.array(CA, dtype=np.float64)
    
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    angles_rad = np.radians(np.arange(0, 360, 5, dtype=np.float64))
    CA_arr = np.array(CA, dtype=np.float64)
    CB_arr = np.array(CB, dtype=np.float64)
    
    for theta in angles_rad:
        temp_pos = rotate_coordinates(original_positions, CA_CB, theta, CA_arr)
        
        rot_names = [atom.name for atom in rotations.atoms]
        if 'CC' in rot_names:
            cc_idx = rot_names.index('CC')
            CC = temp_pos[cc_idx]
            CB_CC = normalize_vector(CC - CB_arr)
            
            part2_mask = np.array([atom in part2 for atom in rotations.atoms])
            part2_positions = temp_pos[part2_mask]
            original_part2 = part2_positions.copy()
            
            for phi in angles_rad:
                temp_pos[part2_mask] = rotate_coordinates(original_part2, CB_CC, phi, CB_arr)
                
                com = calculate_center_of_mass(temp_pos)
                v2 = com - CA_arr
                angle = calculate_angle_between_vectors(v1, v2)
                
                if angle < min_angle:
                    min_angle = angle
                    opt_positions = temp_pos.copy()
    
    rotations.atoms.positions = opt_positions
    
    # Stage 2: Optimize terminal group
    v1 = np.array(CC_r, dtype=np.float64) - CA_arr
    CD = sides.select_atoms("name CD").atoms[0].position
    N1 = sides.select_atoms("name N1").atoms[0].position
    CD_N1 = normalize_vector(np.array(N1, dtype=np.float64) - np.array(CD, dtype=np.float64))
    part3 = sides.select_atoms("not name N CA C O HA CB HB CC HC CD HD")
    
    original_part3 = part3.atoms.positions.copy()
    opt_positions = original_part3.copy()
    min_angle = 180.0
    
    CD_arr = np.array(CD, dtype=np.float64)
    
    for theta in angles_rad:
        new_pos = rotate_coordinates(original_part3, CD_N1, theta, CD_arr)
        
        com = calculate_center_of_mass(new_pos)
        v2 = com - CA_arr
        angle = calculate_angle_between_vectors(v1, v2)
        
        if angle < min_angle:
            min_angle = angle
            opt_positions = new_pos.copy()
    
    part3.atoms.positions = opt_positions


def _optimize_lys(resname, refs, sides):
    """Optimize LYS (multi-stage optimization)."""
    # Stage 1: Similar to ARG
    CA_r = refs.select_atoms("name CA").atoms[0].position
    CB_r = refs.select_atoms("name CB").atoms[0].position
    CC_r = refs.select_atoms("name CC").atoms[0].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    
    CA_CB = normalize_vector(np.array(CB, dtype=np.float64) - np.array(CA, dtype=np.float64))
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    part2 = sides.select_atoms("not name N CA C O HA CB HB")
    
    v1 = np.array(CB_r, dtype=np.float64) - np.array(CA, dtype=np.float64)
    
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    angles_rad = np.radians(np.arange(0, 360, 5, dtype=np.float64))
    CA_arr = np.array(CA, dtype=np.float64)
    CB_arr = np.array(CB, dtype=np.float64)
    
    for theta in angles_rad:
        temp_pos = rotate_coordinates(original_positions, CA_CB, theta, CA_arr)
        
        rot_names = [atom.name for atom in rotations.atoms]
        if 'CC' in rot_names:
            cc_idx = rot_names.index('CC')
            CC = temp_pos[cc_idx]
            CB_CC = normalize_vector(CC - CB_arr)
            
            part2_mask = np.array([atom in part2 for atom in rotations.atoms])
            part2_positions = temp_pos[part2_mask]
            original_part2 = part2_positions.copy()
            
            for phi in angles_rad:
                temp_pos[part2_mask] = rotate_coordinates(original_part2, CB_CC, phi, CB_arr)
                
                com = calculate_center_of_mass(temp_pos)
                v2 = com - CA_arr
                angle = calculate_angle_between_vectors(v1, v2)
                
                if angle < min_angle:
                    min_angle = angle
                    opt_positions = temp_pos.copy()
    
    rotations.atoms.positions = opt_positions
    
    # Stage 2: Terminal group
    v1 = np.array(CC_r, dtype=np.float64) - CA_arr
    CC = sides.select_atoms("name CC").atoms[0].position
    CD = sides.select_atoms("name CD").atoms[0].position
    CC_CD = normalize_vector(np.array(CD, dtype=np.float64) - np.array(CC, dtype=np.float64))
    part3 = sides.select_atoms("name CD HD CE HE N1 HN")
    
    original_part3 = part3.atoms.positions.copy()
    opt_positions = original_part3.copy()
    min_angle = 180.0
    
    CC_arr = np.array(CC, dtype=np.float64)
    
    for theta in angles_rad:
        part3.atoms.positions = rotate_coordinates(original_part3, CC_CD, theta, CC_arr)
        
        grp_CC = sides.select_atoms("name CE HE N1 HN")
        com = grp_CC.center_of_mass()
        v2 = np.array(com, dtype=np.float64) - CA_arr
        angle = calculate_angle_between_vectors(v1, v2)
        
        if angle < min_angle:
            min_angle = angle
            opt_positions = part3.atoms.positions.copy()
    
    part3.atoms.positions = opt_positions


def _optimize_aromatic(resname, refs, sides):
    """Optimize aromatic residues (HIS, PHE, TYR, TRP)."""
    CA_r = refs.select_atoms("name CA").atoms[0].position
    CB_r = refs.select_atoms("name CB").atoms[0].position
    CC_r = refs.select_atoms("name CC").atoms[0].position
    CD_r = refs.select_atoms("name CD").atoms[0].position
    CA = sides.select_atoms("name CA").atoms[0].position
    CB = sides.select_atoms("name CB").atoms[0].position
    
    CA_CB = normalize_vector(np.array(CB, dtype=np.float64) - np.array(CA, dtype=np.float64))
    
    rotations = sides.select_atoms("not name N CA C O HA")
    original_positions = rotations.atoms.positions.copy()
    
    v1 = np.array(CB_r, dtype=np.float64) - np.array(CA, dtype=np.float64)
    
    min_angle = 180.0
    opt_positions = original_positions.copy()
    
    angles_rad = np.radians(np.arange(0, 360, 5, dtype=np.float64))
    CA_arr = np.array(CA, dtype=np.float64)
    CB_arr = np.array(CB, dtype=np.float64)
    
    # Stage 1: Rotate around CA-CB
    for theta in angles_rad:
        new_pos = rotate_coordinates(original_positions, CA_CB, theta, CA_arr)
        
        grp_CB = sides.select_atoms("name CB HB CC")
        grp_CB.atoms.positions = new_pos[[i for i, atom in enumerate(rotations.atoms) if atom in grp_CB]]
        com_CB = grp_CB.center_of_mass()
        v2 = np.array(com_CB, dtype=np.float64) - CA_arr
        angle = calculate_angle_between_vectors(v1, v2)
        
        if angle < min_angle:
            min_angle = angle
            opt_positions = new_pos.copy()
    
    rotations.atoms.positions = opt_positions
    
    # Stage 2: Ring plane optimization
    CC = sides.select_atoms("name CC").atoms[0].position
    CB_CC = normalize_vector(np.array(CC, dtype=np.float64) - CB_arr)
    part2 = sides.select_atoms("not name N CA C O HA CB HB")
    
    original_part2 = part2.atoms.positions.copy()
    
    min_dist = np.inf
    opt_positions = original_part2.copy()
    
    # Define groups based on residue type
    if resname == 'HIS':
        grp_names = ["name CB HB CC", "name N1 CE HE", "name CD HD N2 HN"]
    elif resname == 'PHE':
        grp_names = ["name CB HB CC", "name CE HE CG HG", "name CF HF CH HH"]
    elif resname == 'TYR':
        grp_names = ["name CB HB CC", "name CE HE CG HG", "name CF HF CH O1 HO"]
    elif resname == 'TRP':
        grp_names = ["name CB HB CC", "name CD HD N1 HN", "name CE CF"]
    else:
        return
    
    for phi in angles_rad:
        part2.atoms.positions = rotate_coordinates(original_part2, CB_CC, phi, CB_arr)
        
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