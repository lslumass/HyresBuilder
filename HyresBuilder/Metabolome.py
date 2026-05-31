"""
This module is for parsing the bonded parameter file for CG metabolome.
bonded parameters are first generated based on the parameter file, and then modulated.
"""
import numpy as np


# parameter file for CG metobolome
metabolome = {
    'NCA': {
        'bonds': {
            ('P1H', 'A2W'): (150.0, 1.95),
            ('A2W', 'A2W'): (150.0, 1.95),
            ('A2W', 'A2W'): (150.0, 1.95),
            ('A2W', 'M04'): (20.0, 2.47)
        },
        'angles': {
            ('P1H', 'A2W', 'M04'): (25.0, 135.0),
            ('A2W', 'A2W', 'M04'): (0.0, 165.0)
        },
        'dihedrals': {},
        'impropers': {
            ('A2W', 'A2W', 'P1H', 'M04'): (25.0, 0, 0.0)}
    },
    '2HG': {
        'bonds': {
            ('M02', 'P1S'): (20.0, 2.53),
            ('P1S', 'C3E'): (20.0, 2.40),
            ('C3E', 'M02'): (20.0, 2.32)
        },
        'angles': {
            ('M02', 'P1S', 'C3E'): (25.0, 86.0),
            ('P1S', 'C3E', 'M02'): (15.0, 161.0)
        },
        'dihedrals': {
            ('M02', 'P1S', 'C3E', 'M02'): (0.5, 1, 180.0)
        },
        'impropers': {}
    },
}