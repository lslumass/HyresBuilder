"""
HyRes Protein Builder Module
Build HyRes protein structures from amino acid sequences.
"""

import numpy as np
import random

# Single letter to three letter code mapping
AA_THREE_LETTER = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
    'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
    'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
    'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
}

# Amino acid structure database
# Format: {code: [(atom_name, x, y, z), ...]}
AMINO_ACID_STRUCTURES = {
    'G': [  # Glycine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'G0': [  # Glycine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 5.077, -2.482, -0.298),
        ('N+', 3.346, -3.419,  0.765),
        ('H+', 2.545, -3.272,  1.310),
    ],
    'A0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'A': [  # Alanine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'V0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'V': [  # Valine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'I0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'I': [  # Isoleucine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'L0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'L': [  # Leucine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'M0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'M': [  # Methionine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'S0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'S': [  # Serine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'T0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'T': [  # Threonine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'N0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'N': [  # Asparagine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'Q0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'Q': [  # Glutamine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'C0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'C': [  # Cysteine
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'D0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'D': [  # Aspartic Acid
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'E0': [  # Alanine
        ('C-', 4.061, -2.364,  0.386),
        ('O-', 3.468, -3.400,  0.683),
        ( 'N', 5.169, -2.378, -0.349),
        ( 'H', 5.604, -1.525, -0.559),
        ('CA', 5.734, -3.620, -0.838),
        ('CB', 5.704, -3.564, -2.467),
        ( 'C', 7.149, -3.833, -0.322),
        ( 'O', 7.486, -3.468,  0.803),
        ('N+', 8.000, -4.434, -1.147),
        ('H+', 7.683, -4.707, -2.034),
    ],
    'E': [  # Glutamic Acid
        ('C-', 1.520,  0.000,  0.000),
        ('O-', 2.165,  0.803, -0.673),
        ( 'N', 2.116, -0.911,  0.764),
        ( 'H', 1.558, -1.528,  1.282),
        ('CA', 3.561, -1.002,  0.841),
        ('CB', 3.989, -0.704,  2.385),
        ( 'C', 4.061, -2.364,  0.386),
        ( 'O', 3.468, -3.400,  0.683),
        ('N+', 5.169, -2.378, -0.349),
        ('H+', 5.604, -1.525, -0.559),
    ],
    'R0': [  # Arginine
        ('C-', 59.174, -42.097,  -4.596),
        ('O-', 58.546, -43.153,  -4.536),
        ( 'N', 60.359, -42.020,  -5.194),
        ( 'H', 60.819, -41.155,  -5.214),
        ('CA', 60.972, -43.183,  -5.807),
        ('CB', 60.898, -42.645,  -8.309),
        ('CC', 59.783, -40.041, -10.381),
        ( 'C', 62.316, -43.506,  -5.174),
        ( 'O', 63.108, -42.621,  -4.857),
        ('N+', 62.590, -44.793,  -4.982),
        ('H+', 61.925, -45.460,  -5.252),
    ],
    'R': [  # Arginine
        ('C-', 59.174, -42.097,  -4.596),
        ('O-', 58.546, -43.153,  -4.536),
        ( 'N', 60.359, -42.020,  -5.194),
        ( 'H', 60.819, -41.155,  -5.214),
        ('CA', 60.972, -43.183,  -5.807),
        ('CB', 60.898, -42.645,  -8.309),
        ('CC', 59.783, -40.041, -10.381),
        ( 'C', 62.316, -43.506,  -5.174),
        ( 'O', 63.108, -42.621,  -4.857),
        ('N+', 62.590, -44.793,  -4.982),
        ('H+', 61.925, -45.460,  -5.252),
    ],
    'K0': [  # Lysine
        ('C-', 59.174, -42.097,  -4.596),
        ('O-', 58.546, -43.153,  -4.536),
        ( 'N', 60.359, -42.020,  -5.194),
        ( 'H', 60.819, -41.155,  -5.214),
        ('CA', 60.972, -43.183,  -5.807),
        ('CB', 60.898, -42.645,  -8.309),
        ('CC', 59.783, -40.041, -10.381),
        ( 'C', 62.316, -43.506,  -5.174),
        ( 'O', 63.108, -42.621,  -4.857),
        ('N+', 62.590, -44.793,  -4.982),
        ('H+', 61.925, -45.460,  -5.252),
    ],
    'K': [  # Lysine
        ('C-', 59.174, -42.097,  -4.596),
        ('O-', 58.546, -43.153,  -4.536),
        ( 'N', 60.359, -42.020,  -5.194),
        ( 'H', 60.819, -41.155,  -5.214),
        ('CA', 60.972, -43.183,  -5.807),
        ('CB', 60.898, -42.645,  -8.309),
        ('CC', 59.783, -40.041, -10.381),
        ( 'C', 62.316, -43.506,  -5.174),
        ( 'O', 63.108, -42.621,  -4.857),
        ('N+', 62.590, -44.793,  -4.982),
        ('H+', 61.925, -45.460,  -5.252),
    ],
    'H0': [  # Histidine
        ('C-', 34.329, -24.265, -2.457),
        ('O-', 33.695, -25.296, -2.672),
        ( 'N', 35.584, -24.115, -2.873),
        ( 'H', 36.046, -23.274, -2.677),
        ('CA', 36.267, -25.165, -3.604),
        ('CB', 36.662, -24.455, -5.365),
        ('CC', 36.150, -22.184, -6.330),
        ('CD', 37.036, -23.323, -7.504),
        ( 'C', 37.514, -25.635, -2.873),
        ( 'O', 38.262, -24.843, -2.300),
        ('N+', 37.756, -26.942, -2.884),
        ('H+', 37.129, -27.531, -3.355),
    ],
    'H': [  # Histidine
        ('C-', 34.329, -24.265, -2.457),
        ('O-', 33.695, -25.296, -2.672),
        ( 'N', 35.584, -24.115, -2.873),
        ( 'H', 36.046, -23.274, -2.677),
        ('CA', 36.267, -25.165, -3.604),
        ('CB', 36.662, -24.455, -5.365),
        ('CC', 36.150, -22.184, -6.330),
        ('CD', 37.036, -23.323, -7.504),
        ( 'C', 37.514, -25.635, -2.873),
        ( 'O', 38.262, -24.843, -2.300),
        ('N+', 37.756, -26.942, -2.884),
        ('H+', 37.129, -27.531, -3.355),
    ],
    'F0': [  # Phenylalanine
        ('C-', 18.161, -11.852, -0.844),
        ('O-', 18.459, -11.762,  0.346),
        ( 'N', 18.526, -12.898, -1.580),
        ( 'H', 18.264, -12.922, -2.524),
        ('CA', 19.296, -13.988, -1.013),
        ('CB', 17.663, -15.675, -0.913),
        ('CC', 15.435, -15.063, -1.725),
        ('CD', 15.227, -17.181, -1.093),
        ( 'C', 20.629, -14.160, -1.724),
        ( 'O', 20.727, -14.049, -2.945),
        ('N+', 21.681, -14.438, -0.959),
        ('H+', 21.550, -14.517,  0.009),
    ],
    'F': [  # Phenylalanine
        ('C-', 18.161, -11.852, -0.844),
        ('O-', 18.459, -11.762,  0.346),
        ( 'N', 18.526, -12.898, -1.580),
        ( 'H', 18.264, -12.922, -2.524),
        ('CA', 19.296, -13.988, -1.013),
        ('CB', 17.663, -15.675, -0.913),
        ('CC', 15.435, -15.063, -1.725),
        ('CD', 15.227, -17.181, -1.093),
        ( 'C', 20.629, -14.160, -1.724),
        ( 'O', 20.727, -14.049, -2.945),
        ('N+', 21.681, -14.438, -0.959),
        ('H+', 21.550, -14.517,  0.009),
    ],
    'Y0': [  # Tyrosine
        ('C-', 45.403, -32.086, -3.894),
        ('O-', 45.340, -32.198, -5.117),
        ( 'N', 46.552, -32.215, -3.237),
        ( 'H', 46.550, -32.117, -2.262),
        ('CA', 47.793, -32.495, -3.934),
        ('CB', 48.864, -30.411, -3.763),
        ('CC', 47.618, -28.666, -2.577),
        ('CD', 49.407, -27.039, -2.907),
        ( 'C', 48.414, -33.801, -3.467),
        ( 'O', 48.412, -34.128, -2.281),
        ('N+', 48.961, -34.571, -4.404),
        ('H+', 48.937, -34.265, -5.335),
    ],
    'Y': [  # Tyrosine
        ('C-', 45.403, -32.086, -3.894),
        ('O-', 45.340, -32.198, -5.117),
        ( 'N', 46.552, -32.215, -3.237),
        ( 'H', 46.550, -32.117, -2.262),
        ('CA', 47.793, -32.495, -3.934),
        ('CB', 48.864, -30.411, -3.763),
        ('CC', 47.618, -28.666, -2.577),
        ('CD', 49.407, -27.039, -2.907),
        ( 'C', 48.414, -33.801, -3.467),
        ( 'O', 48.412, -34.128, -2.281),
        ('N+', 48.961, -34.571, -4.404),
        ('H+', 48.937, -34.265, -5.335),
    ],
    'W0': [  # Tryptophan
        ('C-', 20.629, -14.160, -1.724),
        ('O-', 20.727, -14.049, -2.945),
        ( 'N', 21.681, -14.438, -0.959),
        ( 'H', 21.550, -14.517,  0.009),
        ('CA', 23.001, -14.623, -1.530),
        ('CB', 24.073, -13.094, -0.843),
        ('CC', 23.206, -11.235,  0.722),
        ('CD', 26.476, -12.365, -1.151),
        ('CE', 26.436, -10.598,  0.073),
        ('CF', 28.402, -11.409, -1.144),
        ( 'C', 23.550, -16.009, -1.232),
        ( 'O', 23.390, -16.544, -0.136),
        ('N+', 24.212, -16.612, -2.216),
        ('H+', 24.312, -16.143, -3.070),
    ],
    'W': [  # Tryptophan
        ('C-', 20.629, -14.160, -1.724),
        ('O-', 20.727, -14.049, -2.945),
        ( 'N', 21.681, -14.438, -0.959),
        ( 'H', 21.550, -14.517,  0.009),
        ('CA', 23.001, -14.623, -1.530),
        ('CB', 24.073, -13.094, -0.843),
        ('CC', 23.206, -11.235,  0.722),
        ('CD', 26.476, -12.365, -1.151),
        ('CE', 26.436, -10.598,  0.073),
        ('CF', 28.402, -11.409, -1.144),
        ( 'C', 23.550, -16.009, -1.232),
        ( 'O', 23.390, -16.544, -0.136),
        ('N+', 24.212, -16.612, -2.216),
        ('H+', 24.312, -16.143, -3.070),
    ],
    'P': [  # Proline
        ('C-',  7.149, -3.831, -0.322),
        ('O-',  7.952, -2.903, -0.241),
        ( 'N',  7.474, -5.070,  0.038),
        ('CA',  8.794, -5.388,  0.545),
        ('CB',  8.425, -5.720,  2.421),
        ( 'C',  9.495, -6.418, -0.325),
        ( 'O',  8.890, -7.376, -0.804),
        ('N+', 10.794, -6.231, -0.542),
        ('H+', 11.230, -5.452, -0.138),
    ],
    # Add more amino acids here with their structure data
    # Format: ('atom_name', x, y, z)
}

def get_amino_acid(code):
    """Get amino acid structure by single-letter code."""
    if code not in AMINO_ACID_STRUCTURES:
        raise ValueError(f"Amino acid '{code}' not found in structure database")
    
    atoms = []
    for i, (name, x, y, z) in enumerate(AMINO_ACID_STRUCTURES[code], 1):
        atoms.append({
            'index': i,
            'name': name,
            'coords': np.array([x, y, z], dtype=float)
        })
    return atoms

def get_coords(res_name, atom_names):
    """Extract coordinates from AMINO_ACID_STRUCTURES."""
    res = AMINO_ACID_STRUCTURES[res_name]
    name_to_coord = {name: np.array([x, y, z], dtype=float) for name, x, y, z in res}
    return np.vstack([name_to_coord[a] for a in atom_names])

def get_from_ref(ref, atom_names):
    """Extract coordinates from reference residue."""
    name_to_coord = {atom['name']: atom['coords'] for atom in ref}
    return np.vstack([name_to_coord[a] for a in atom_names])

def kabsch(P, Q):
    """Compute optimal rotation R and translation t using Kabsch algorithm."""
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P_centered = P - Pc
    Q_centered = Q - Qc
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # Fix improper rotation
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Qc - R @ Pc
    return R, t

def align_residues(ref, resA_name, resB_name):
    """
    Align residue B to residue A using backbone atoms.
    Handles Proline: skips H/H+ if either residue is 'P'.
    Returns list of dicts for B atoms (excluding alignment atoms).
    """
    # Default alignment atoms
    atoms_A = ["C", "O", "N+"]
    atoms_B = ["C-", "O-", "N"]

    # Include H/H+ if neither is Proline
    if resA_name != 'P' and resB_name != 'P':
        atoms_A.append("H+")
        atoms_B.append("H")

    # Extract coordinates
    QA = get_from_ref(ref, atoms_A)
    PB = get_coords(resB_name, atoms_B)

    # Compute rotation and translation
    R, t = kabsch(PB, QA)

    # Transform all atoms of B, excluding alignment atoms
    ref, B_atoms = [], []
    for i, (name, x, y, z) in enumerate(AMINO_ACID_STRUCTURES[resB_name]):
        coord = np.array([x, y, z], dtype=float)
        new_coord = R @ coord + t
        ref.append({
            'index': i,
            'name': name,
            'coords': new_coord
        })
        if name not in ['C-', 'O-', 'N+', 'H+']:
            B_atoms.append({
                'index': i,
                'name': name,
                'coords': new_coord
            })
        
    return ref, B_atoms

def write_pdb(atoms, filename):
    """Write atoms to PDB file."""
    with open(filename, 'w') as f:
        f.write("REMARK   HyRes protein\n")
        f.write("REMARK   Peptide chain generated by HyresBuilder\n")
        f.write('REMARK   Ref: Y. Zhang, S. Li, X. Gong and J. Chen, JACS, 2024, 146, 342-357.\n')
        for atom in atoms:
            # PDB format: ATOM line
            line = f"ATOM  {atom['global_index']:5d}  {atom['name']:<3s} {atom['res_name']:<3s} A{atom['res_num']:4d}    "
            line += f"{atom['coords'][0]:8.3f}{atom['coords'][1]:8.3f}{atom['coords'][2]:8.3f}"
            line += "  1.00  0.00           \n"
            f.write(line)
        f.write("END\n")

def detect_clashes(all_atoms, threshold=4.5):
    """Fast clash detection using CA atoms only. Threshold hardcoded to 3.8 Å."""
    ca_atoms = [a for a in all_atoms if a['name'] == 'CA']
    coords = np.array([a['coords'] for a in ca_atoms])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    idx = np.arange(len(dist) - 1)
    dist[idx, idx + 1] = np.inf
    dist[idx + 1, idx] = np.inf

    ii, jj = np.where((dist < threshold) & (dist > 0))
    clashes = [(i, j) for i, j in zip(ii, jj) if i < j]

    if clashes:
        print(f"  {len(clashes)} CA clashes detected (< {threshold} Å):")
        for i, j in clashes:
            a1, a2 = ca_atoms[i], ca_atoms[j]
            print(f"    {a1['res_name']}{a1['res_num']}-CA vs "
                  f"{a2['res_name']}{a2['res_num']}-CA: {dist[i,j]:.2f} Å")
    return clashes

def build_peptide(name, sequence, random_conf=True, check_clash=False, max_retries=10):
    output_file = f"{name}.pdb"

    for attempt in range(1, max_retries + 1):
        all_atoms = []
        atom_counter = 1
        ref = None

        for i, aa in enumerate(sequence):
            alt = aa + '0'
            if random_conf and alt in AMINO_ACID_STRUCTURES and (i + 1) % 4 == 0:
                res_key = random.choice([aa, alt])
            else:
                res_key = aa

            if aa not in AMINO_ACID_STRUCTURES:
                raise ValueError(f"Amino acid '{aa}' not found in structure database")

            if i == 0:
                res0 = get_amino_acid(res_key)
                ref = res0
                atoms = [{'index': a['index'], 'name': a['name'],
                        'coords': a['coords'].copy()} for a in res0[2:-2]]
                for atom in atoms:
                    atom['res_num'] = i + 1
                    atom['res_name'] = AA_THREE_LETTER.get(aa, aa)
                    atom['global_index'] = atom_counter
                    atom_counter += 1
                all_atoms.extend(atoms)
            else:
                ref, new_res = align_residues(ref, sequence[i-1], res_key)
                atoms = [{'index': a['index'], 'name': a['name'],
                        'coords': a['coords'].copy()} for a in new_res]
                for atom in atoms:
                    atom['res_num'] = i + 1
                    atom['res_name'] = AA_THREE_LETTER.get(aa, aa)
                    atom['global_index'] = atom_counter
                    atom_counter += 1
                all_atoms.extend(atoms)

        # Translate so first CA is at (5000, 5000, 5000)
        first_CA = next(a['coords'] for a in all_atoms if a['name'] == 'CA')
        offset = np.array([5000.0, 5000.0, 5000.0]) - first_CA
        for atom in all_atoms:
            atom['coords'] += offset

        # Clash check
        if not random_conf or not check_clash:
            break
        print(f"Attempt {attempt}/{max_retries}:")
        clashes = detect_clashes(all_atoms)
        if not clashes:
            print("  No clashes, build successful.")
            break
        if attempt == max_retries:
            print(f"Warning: could not resolve clashes after {max_retries} attempts, writing anyway.")

    write_pdb(all_atoms, output_file)
    print(f"Peptide chain built: {len(sequence)} residues, {len(all_atoms)} atoms")
    print(f"Output written to: {output_file}")

def main():
    "Command-line interface to build peptide from sequence."
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build a peptide chain from sequence: hyresbuilder name sequence, output: name.pdb.')
    
    parser.add_argument('name', type=str,
                        help='pdb file name, output will be name.pdb. default: hyres.pdb')
    parser.add_argument('sequence', type=str,
                        help='Amino acid sequence (single-letter codes, e.g., ACDEFG)')
    parser.add_argument('--linear', action='store_true',
                        help='Build linear chain using base conformation only')
    parser.add_argument('--check-clash', action='store_true', default=False,
                        help='Check for CA clashes and rebuild if found (default: False)')

    args = parser.parse_args()
    name = args.name
    sequence = args.sequence.upper()
    
    # Build peptide
    build_peptide(name, sequence, random_conf=not args.linear, check_clash=args.check_clash)

if __name__ == "__main__":
    main()
