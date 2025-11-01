"""
HyresBuilder rotamer module for backmap.

Ultra-fast rotamer library with TOP 5 most probable rotamers per residue.
Uses Dunbrack 2010 backbone-independent rotamer library.
"""

import numpy as np
from numba import jit
import functools

# Top 5 most probable rotamers from Dunbrack 2010 backbone-independent library
# Format: (chi1, chi2, chi3, chi4, probability)
ROTAMER_LIBRARY = {
    'SER': [
        (62.0, 0, 0, 0, 0.39),
        (-63.0, 0, 0, 0, 0.32),
        (180.0, 0, 0, 0, 0.29),
    ],
    'THR': [
        (62.0, 0, 0, 0, 0.38),
        (-63.0, 0, 0, 0, 0.36),
        (180.0, 0, 0, 0, 0.26),
    ],
    'CYS': [
        (-63.0, 0, 0, 0, 0.52),
        (62.0, 0, 0, 0, 0.37),
        (180.0, 0, 0, 0, 0.11),
    ],
    'VAL': [
        (175.0, 0, 0, 0, 0.47),
        (-60.0, 0, 0, 0, 0.39),
        (63.0, 0, 0, 0, 0.14),
    ],
    'ILE': [
        (-60.0, 170.0, 0, 0, 0.40),
        (-60.0, 65.0, 0, 0, 0.29),
        (175.0, 65.0, 0, 0, 0.16),
        (175.0, 170.0, 0, 0, 0.15),
    ],
    'LEU': [
        (-60.0, 175.0, 0, 0, 0.28),
        (180.0, 65.0, 0, 0, 0.22),
        (-85.0, 65.0, 0, 0, 0.19),
        (-60.0, 80.0, 0, 0, 0.17),
        (62.0, 175.0, 0, 0, 0.14),
    ],
    'ASP': [
        (-60.0, -20.0, 0, 0, 0.38),
        (-60.0, 30.0, 0, 0, 0.26),
        (180.0, -10.0, 0, 0, 0.20),
        (180.0, 65.0, 0, 0, 0.16),
    ],
    'ASN': [
        (-60.0, -20.0, 0, 0, 0.35),
        (-60.0, 30.0, 0, 0, 0.28),
        (180.0, -10.0, 0, 0, 0.21),
        (180.0, 65.0, 0, 0, 0.16),
    ],
    'GLU': [
        (-60.0, 180.0, -20.0, 0, 0.22),
        (-60.0, -80.0, -10.0, 0, 0.18),
        (-60.0, 180.0, 65.0, 0, 0.16),
        (180.0, 65.0, -10.0, 0, 0.14),
        (-70.0, -30.0, -20.0, 0, 0.12),
    ],
    'GLN': [
        (-60.0, 180.0, 20.0, 0, 0.24),
        (-60.0, -75.0, 10.0, 0, 0.19),
        (-60.0, 180.0, 65.0, 0, 0.17),
        (180.0, 65.0, 10.0, 0, 0.15),
        (-70.0, -30.0, 20.0, 0, 0.12),
    ],
    'MET': [
        (-60.0, 180.0, 70.0, 0, 0.27),
        (-60.0, -75.0, 70.0, 0, 0.21),
        (180.0, 65.0, 70.0, 0, 0.18),
        (-60.0, 180.0, 180.0, 0, 0.14),
        (180.0, 180.0, 70.0, 0, 0.10),
    ],
    'LYS': [
        (-60.0, 180.0, 68.0, 180.0, 0.19),
        (-60.0, 180.0, 180.0, 180.0, 0.17),
        (-60.0, 180.0, 180.0, 65.0, 0.13),
        (180.0, 180.0, 68.0, 180.0, 0.12),
        (-60.0, 180.0, 65.0, -85.0, 0.10),
    ],
    'ARG': [
        (-60.0, 180.0, 65.0, -85.0, 0.20),
        (-60.0, 180.0, 180.0, -85.0, 0.18),
        (-60.0, 180.0, 65.0, 175.0, 0.15),
        (180.0, 65.0, 65.0, -85.0, 0.12),
        (-60.0, -67.0, -60.0, -85.0, 0.10),
    ],
    'HIS': [
        (-60.0, -75.0, 0, 0, 0.42),
        (-60.0, 80.0, 0, 0, 0.29),
        (180.0, -75.0, 0, 0, 0.17),
        (180.0, 80.0, 0, 0, 0.12),
    ],
    'PHE': [
        (-60.0, 90.0, 0, 0, 0.49),
        (180.0, 80.0, 0, 0, 0.38),
        (-60.0, -90.0, 0, 0, 0.13),
    ],
    'TYR': [
        (-60.0, 90.0, 0, 0, 0.49),
        (180.0, 80.0, 0, 0, 0.39),
        (-60.0, -90.0, 0, 0, 0.12),
    ],
    'TRP': [
        (-60.0, -90.0, 0, 0, 0.46),
        (-60.0, 105.0, 0, 0, 0.28),
        (180.0, -90.0, 0, 0, 0.16),
        (180.0, 105.0, 0, 0, 0.10),
    ],
}

@jit(nopython=True, cache=True)
def normalize_vector(v):
    """Fast vector normalization."""
    norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if norm > 1e-10:
        return v / norm
    return v

@jit(nopython=True, cache=True)
def rotation_matrix_axis_angle(axis, angle):
    """Compute rotation matrix using Rodrigues' formula."""
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    
    x, y, z = axis[0], axis[1], axis[2]
    
    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ]
    ], dtype=np.float64)
    
    return R

@jit(nopython=True, cache=True)
def rotate_points(coords, axis, angle, center):
    """Rotate coordinates around axis through center."""
    R = rotation_matrix_axis_angle(axis, angle)
    centered = coords - center
    rotated = np.dot(centered, R.T)
    return rotated + center

@jit(nopython=True, cache=True)
def compute_dihedral(p1, p2, p3, p4):
    """Compute dihedral angle between 4 points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    b2_norm = np.sqrt(np.sum(b2 * b2))
    if b2_norm < 1e-10:
        return 0.0
    
    b2_unit = b2 / b2_norm
    
    # Project b1 and b3 onto plane perpendicular to b2
    v1 = b1 - np.dot(b1, b2_unit) * b2_unit
    v2 = b3 - np.dot(b3, b2_unit) * b2_unit
    
    # Calculate angle
    x = np.dot(v1, v2)
    y = np.dot(np.cross(b2_unit, v1), v2)
    
    return np.arctan2(y, x)

@functools.lru_cache(maxsize=128)
def get_chi_atom_names(resname, chi_num):
    """Get atom names for chi angle (cached)."""
    chi_defs = {
        'SER': {1: ('N', 'CA', 'CB', 'OG')},
        'THR': {1: ('N', 'CA', 'CB', 'OG1')},
        'CYS': {1: ('N', 'CA', 'CB', 'SG')},
        'VAL': {1: ('N', 'CA', 'CB', 'CG1')},
        'ILE': {1: ('N', 'CA', 'CB', 'CG1'), 2: ('CA', 'CB', 'CG1', 'CD1')},
        'LEU': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
        'ASP': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'OD1')},
        'ASN': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'OD1')},
        'GLU': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'OE1')},
        'GLN': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'OE1')},
        'MET': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'SD'), 3: ('CB', 'CG', 'SD', 'CE')},
        'LYS': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 
                3: ('CB', 'CG', 'CD', 'CE'), 4: ('CG', 'CD', 'CE', 'NZ')},
        'ARG': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 
                3: ('CB', 'CG', 'CD', 'NE'), 4: ('CG', 'CD', 'NE', 'CZ')},
        'HIS': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'ND1')},
        'PHE': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
        'TYR': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
        'TRP': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    }
    
    if resname in chi_defs and chi_num in chi_defs[resname]:
        return chi_defs[resname][chi_num]
    return None

def find_atom_with_alternatives(sides, name, alternatives=None):
    """Find atom by name, trying alternative names if needed."""
    if alternatives is None:
        alternatives = []
    
    # Try primary name first
    sel = sides.select_atoms(f"name {name}", updating=False)
    if len(sel) > 0:
        return sel[0]
    
    # Try alternatives
    for alt_name in alternatives:
        sel = sides.select_atoms(f"name {alt_name}", updating=False)
        if len(sel) > 0:
            return sel[0]
    
    return None

def set_chi_angle(sides, chi_num, target_angle_deg):
    """Set chi angle to target value."""
    resname = sides.residues[0].resname
    atom_names = get_chi_atom_names(resname, chi_num)
    
    if atom_names is None:
        return
    
    # Alternative atom names for different naming conventions
    name_alternatives = {
        'CG': ['CC'],
        'CD': ['CC', 'CD1'],
        'CD1': ['CC'],
        'CE': ['CD'],
        'OG': [],
        'OG1': [],
        'SG': [],
        'CG1': ['CC'],
        'SD': ['S'],
        'OD1': [],
        'OE1': [],
        'ND1': ['N1'],
        'NE': ['N1'],
        'NZ': ['N1'],
        'CZ': [],
    }
    
    # Find atoms
    atoms = []
    for name in atom_names:
        alternatives = name_alternatives.get(name, [])
        atom = find_atom_with_alternatives(sides, name, alternatives)
        if atom is None:
            return
        atoms.append(atom)
    
    # Get positions
    positions = np.array([a.position for a in atoms], dtype=np.float64)
    
    # Compute current dihedral
    current_angle = compute_dihedral(positions[0], positions[1], positions[2], positions[3])
    
    # Calculate rotation needed
    target_angle = np.radians(target_angle_deg)
    delta = target_angle - current_angle
    
    # Rotation axis (bond between atoms 2 and 3)
    axis = normalize_vector(positions[2] - positions[1])
    center = positions[1]
    
    # Select atoms to rotate based on chi number
    if chi_num == 1:
        to_rotate = sides.select_atoms("not name N CA C O HA CB HB", updating=False)
    elif chi_num == 2:
        to_rotate = sides.select_atoms(
            "not name N CA C O HA CB HB CG CC HG HC", updating=False)
    elif chi_num == 3:
        to_rotate = sides.select_atoms(
            "not name N CA C O HA CB HB CG CC HG HC CD CC HD", updating=False)
    elif chi_num == 4:
        to_rotate = sides.select_atoms(
            "name NZ HZ N1 HN CZ NH1 HH1 NH2 HH2", updating=False)
    else:
        return
    
    if len(to_rotate) == 0:
        return
    
    # Apply rotation
    coords = to_rotate.positions.copy()
    to_rotate.positions = rotate_points(coords, axis, delta, center)

def opt_side_chain(resname, refs, sides):
    """
    Optimize side chain using TOP 5 most probable rotamers.
    
    Tries all rotamers and selects the best match to reference geometry.
    
    Parameters
    ----------
    resname : str
        Residue name (three-letter code)
    refs : AtomGroup
        Reference atoms from coarse-grained model (CA, CB, CC, CD, etc.)
    sides : Universe
        All-atom residue structure to optimize
    """
    if resname not in ROTAMER_LIBRARY:
        return
    
    rotamers = ROTAMER_LIBRARY[resname]
    
    # Get reference positions for scoring
    CA_r = refs.atoms[0].position
    ref_positions = [refs.atoms[i].position for i in range(min(len(refs.atoms), 4))]
    
    # Calculate reference center of mass (excluding CA)
    if len(ref_positions) > 1:
        ref_com = np.mean(ref_positions[1:], axis=0)
    else:
        ref_com = CA_r
    
    # Get side chain atoms for scoring
    CA = sides.select_atoms("name CA", updating=False).atoms[0].position
    side_atoms = sides.select_atoms("not name N CA C O HA", updating=False)
    
    if len(side_atoms) == 0:
        return
    
    # Save original positions
    original_positions = side_atoms.atoms.positions.copy()
    
    best_score = np.inf
    best_positions = None
    
    # Try each rotamer
    for chi1, chi2, chi3, chi4, prob in rotamers:
        # Reset to original
        side_atoms.atoms.positions = original_positions.copy()
        
        # Apply chi angles
        if chi1 != 0:
            set_chi_angle(sides, 1, chi1)
        if chi2 != 0:
            set_chi_angle(sides, 2, chi2)
        if chi3 != 0:
            set_chi_angle(sides, 3, chi3)
        if chi4 != 0:
            set_chi_angle(sides, 4, chi4)
        
        # Calculate score: distance between side chain COM and reference COM
        current_com = side_atoms.center_of_mass()
        dist = np.linalg.norm(current_com - ref_com)
        
        # Weight by rotamer probability (favor high probability rotamers)
        score = dist - 2.0 * prob
        
        if score < best_score:
            best_score = score
            best_positions = side_atoms.atoms.positions.copy()
    
    # Apply best rotamer
    if best_positions is not None:
        side_atoms.atoms.positions = best_positions