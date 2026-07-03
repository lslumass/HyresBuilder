"""
This module is for parsing the bonded parameter file for CG metabolome.
bonded parameters are first generated based on the parameter file, and then modulated.
"""
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np


KCAL_TO_KJ: float = 4.184
"""Multiply kcal by this to obtain kJ (exact IUPAC value)."""

# ---------------------------------------------------------------------------
# Reference parameter dictionary
# ---------------------------------------------------------------------------

metabolome = {
    # ------------------------------------------------------------------
    # Each residue entry has four sub-dicts:
    #
    #   bonds     : {(atom_i, atom_j)                : (k [kcal/mol/Å²],   b0 [Å])}
    #   angles    : {(atom_i, atom_j, atom_k)         : (k [kcal/mol/rad²], theta0 [deg])}
    #   dihedrals : {(atom_i, atom_j, atom_k, atom_l) : (k [kcal/mol],      n,  phase [deg])}
    #   impropers : {(atom_i, atom_j, atom_k, atom_l) : (k [kcal/mol/rad²], _, theta0 [deg])}
    # ------------------------------------------------------------------
    'NCA': {
        'bonds': {
            ('M1', 'M2'): (150.0, 1.95),  # P1H - A2W
            ('M1', 'M3'): (150.0, 1.95),  # P1H - A2W
            ('M2', 'M3'): (150.0, 1.95),  # A2W - A2W
            ('M3', 'M4'): (10.0, 2.47),   # A2W - M04
        },
        'angles': {
            ('M1', 'M3', 'M4'): (25.0, 135.0), # P1H - A2W - M04
            ('M2', 'M3', 'M4'): (0.0, 165.0),  # A2W - A2W - M04
        },
        'dihedrals': {},
        'impropers': {
            ('M3', 'M1', 'M2', 'M4'): (25.0, 0, 0.0), # IMPH M3 M1 M2 M4
        },
    },
    '2HG': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.53),  # M02 - P1S
            ('M2', 'M3'): (20.0, 2.40),  # P1S - C3E
            ('M3', 'M4'): (20.0, 2.32),  # C3E - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (25.0, 86.0),  # M02 - P1S - C3E
            ('M2', 'M3', 'M4'): (15.0, 161.0), # P1S - C3E - M02
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 55.0), # M02 - P1S - C3E - M02
        },
        'impropers': {},
    },
    '2PG': {
        'bonds': {
            ('M1', 'M2'): (30.0, 3.02),  # M01 - P1T
            ('M2', 'M3'): (30.0, 2.79),  # P1T - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (35.0, 91.0),  # M01 - P1T - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    '3PG': {
        'bonds': {
            ('M1', 'M2'): (30.0, 3.25),  # M01 - P1T
            ('M2', 'M3'): (30.0, 2.60),  # P1T - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (35.0, 137.0), # M01 - P1T - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    'UN1': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.95),  # M02 - A2V
            ('M2', 'M3'): (20.0, 3.12),  # A2V - A3K
            ('M3', 'M4'): (20.0, 2.83),  # A3K - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 130.0),  # M02 - A2V - A3K
            ('M2', 'M3', 'M4'): (25.0, 85.0),  # A2V - A3K - M02
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.0, 1, 0.0), # M02 - A2V - A3K - M02
        },
        'impropers': {},
    },
    'AYA': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.60),  # M03 - C3E
            ('M2', 'M3'): (20.0, 3.57),  # C3E - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 90.0),   # M03 - C3E - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    'NLG': {
        'bonds': {
            ('M1', 'M2'): (20.0, 3.67),  # M03 - A2V
            ('M2', 'M3'): (20.0, 2.80),  # A2V - M02
            ('M2', 'M4'): (20.0, 2.80),  # A2V - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 82.8),   # M03 - A2V - M02
            ('M1', 'M2', 'M4'): (5.0, 82.8),   # M03 - A2V - M02
            ('M3', 'M2', 'M4'): (10.0, 134.8), # M02 - A2V - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    'AKG': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.28),  # M02 - C3E
            ('M2', 'M3'): (20.0, 2.18),  # C3E - C3E
            ('M3', 'M4'): (20.0, 2.28),  # C3E - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (30.0, 105.6), # M02 - C3E - C3E
            ('M2', 'M3', 'M4'): (30.0, 105.6), # C3E - C3E - M02
            ('M1', 'M2', 'M4'): (5.0, 164.0),  # M02 - C3E - M02
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.0, 1, 0.0), # M1 - M2 - M3 - M4
        },
        'impropers': {},
    },
    'SIN': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.34),  # M02 - C3E
            ('M2', 'M3'): (20.0, 2.34),  # C3E - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (15.0, 170.0), # M02 - C3E - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    'LMR': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.40),  # M02 - P1T
            ('M2', 'M3'): (20.0, 2.40),  # P1T - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (3.0, 125.0),  # M02 - P1T - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    'FUM': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.36),  # M02 - C3E
            ('M2', 'M3'): (20.0, 2.36),  # C3E - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (15.0, 171.0), # M02 - C3E - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    'MCT': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.63),  # M02 - P1T
            ('M2', 'M3'): (20.0, 2.92),  # P1T - C3E
            ('M3', 'M4'): (20.0, 2.51),  # C3E - M02
            ('M2', 'M5'): (20.0, 2.63),  # P1T - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (15.0, 150.0), # M02 - P1T - C3E
            ('M1', 'M2', 'M5'): (15.0, 85.0),  # M1(M02) - M2(P1T) - M5(M02) -> Order matched to topology
            ('M2', 'M3', 'M4'): (60.0, 57.5),  # P1T - C3E - M02
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.5, 1, 157.0), # M1 - M2 - M3 - M4
        },
        'impropers': {
            ('M2', 'M1', 'M3', 'M5'): (5.0, 0, 0.0),  # IMPH M2 M1 M3 M5
        },
    },
    'PAU': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.70),  # P1S - A2V
            ('M2', 'M3'): (20.0, 2.70),  # A2V - P1S
            ('M3', 'M4'): (20.0, 3.17),  # P1S - M03
            ('M4', 'M5'): (20.0, 2.80),  # M03 - QaD
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 73.0),   # P1S - A2V - P1S
            ('M2', 'M3', 'M4'): (5.0, 77.0),   # A2V - P1S - M03
            ('M3', 'M4', 'M5'): (5.0, 131.0),  # P1S - M03 - QaD
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.5, 1, 30.0),  # M1 - M2 - M3 - M4
            ('M2', 'M3', 'M4', 'M5'): (0.5, 1, -5.0),  # M2 - M3 - M4 - M5
        },
        'impropers': {},
    },
    'PEP': {
        'bonds': {
            ('M1', 'M2'): (50.0, 3.81),  # M01 - C3E
            ('M2', 'M3'): (50.0, 2.81),  # C3E - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (35.0, 62.0),   # M01 - C3E - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    '13P': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.71),  # P1S - A2V
            ('M2', 'M3'): (20.0, 3.17),  # A2V - M01
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 104.0),  # P1S - A2V - M01
        },
        'dihedrals': {},
        'impropers': {},
    },
    'G6P': {
        'bonds': {
            ('M1', 'M2'): (50.0, 3.27),  # M01 - RS1
            ('M2', 'M3'): (250.0, 3.17), # RS1 - M05
            ('M2', 'M4'): (250.0, 3.17), # RS1 - M05
            ('M3', 'M4'): (250.0, 3.13), # M05 - M05
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 150.0),  # M01 - RS1 - M05
            ('M1', 'M2', 'M4'): (5.0, 150.0),  # M01 - RS1 - M05
        },
        'dihedrals': {},
        'impropers': {
            ('M2', 'M1', 'M3', 'M4'): (5.0, 0, 0.0),  # IMPH M2 M1 M3 M4
        },
    },
    'CHT': {
        'bonds': {
            ('M1', 'M2'): (150.0, 2.93), # P1T - M06
        },
        'angles': {},
        'dihedrals': {},
        'impropers': {},
    },
    'CH5': {
        'bonds': {
            ('M1', 'M2'): (30.0, 2.57),  # P1S - P1T
            ('M2', 'M3'): (30.0, 3.53),  # P1T - RP
            ('M3', 'M4'): (30.0, 3.27),  # RP - C3E
            ('M4', 'M5'): (30.0, 2.67),  # C3E - M06
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 115.0),  # P1S - P1T - RP
            ('M2', 'M3', 'M4'): (5.0, 103.5),  # P1T - RP - C3E
            ('M3', 'M4', 'M5'): (5.0, 95.5),   # RP - C3E - M06
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.2, 2, -177.0), # M1 - M2 - M3 - M4
            ('M2', 'M3', 'M4', 'M5'): (0.2, 2, -177.0), # M2 - M3 - M4 - M5
        },
        'impropers': {},
    },
    'GSH': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.78),  # M02 - M06
            ('M2', 'M3'): (50.0, 4.14),  # M06 - M03
            ('M3', 'M4'): (50.0, 3.55),  # M03 - M03
            ('M3', 'M6'): (20.0, 2.74),  # M03 - P1C
            ('M4', 'M5'): (50.0, 2.83),  # M03 - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 115.0),  # M02 - M06 - M03
            ('M2', 'M3', 'M4'): (10.0, 132.0), # M06 - M03 - M03
            ('M2', 'M3', 'M6'): (10.0, 129.0), # M06 - M03 - P1C
            ('M3', 'M4', 'M5'): (10.0, 126.5), # M03 - M03 - M02
            ('M6', 'M3', 'M5'): (10.0, 82.7),  # P1C - M03 - M02
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, -142.0), # M1 - M2 - M3 - M4
            ('M1', 'M2', 'M3', 'M6'): (0.2, 1, -22.0),  # M1 - M2 - M3 - M6
            ('M2', 'M3', 'M4', 'M5'): (0.1, 1, -142.0), # M2 - M3 - M4 - M5
        },
        'impropers': {},
    },
    'TAU': {
        'bonds': {
            ('M1', 'M2'): (150.0, 2.98), # QdK - RS1
        },
        'angles': {},
        'dihedrals': {},
        'impropers': {},
    },
    'BET': {
        'bonds': {
            ('M1', 'M2'): (150.0, 3.27), # M06 - QaD
        },
        'angles': {},
        'dihedrals': {},
        'impropers': {},
    },
    'ABU': {
        'bonds': {
            ('M1', 'M2'): (20.0, 2.47),  # QdK - C3E
            ('M2', 'M3'): (20.0, 2.51),  # C3E - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (0.5, 106.5), # QdK - C3E - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    '152': {
        'bonds': {
            ('M1', 'M2'): (50.0, 3.50),  # M06 - P1T
            ('M2', 'M3'): (50.0, 2.50),  # P1T - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (0.5, 113.5), # M06 - P1T - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    'ACA': {
        'bonds': {
            ('M1', 'M2'): (50.0, 3.28),  # M06 - A2V
            ('M2', 'M3'): (50.0, 2.48),  # A2V - M02
            ('M2', 'M4'): (50.0, 3.64),  # A2V - M07
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 121.0),  # M06 - A2V - M02
            ('M1', 'M2', 'M4'): (5.0, 96.0),   # M06 - A2V - M07
            ('M3', 'M2', 'M4'): (5.0, 116.5),  # M02 - A2V - M07
        },
        'dihedrals': {},
        'impropers': {},
    },
    'C3C': {
        'bonds': {
            ('M1', 'M2'): (50.0, 2.52),  # C3E - M07
            ('M2', 'M3'): (50.0, 2.48),  # M07 - A2V
            ('M3', 'M5'): (50.0, 3.28),  # A2V - M06
            ('M3', 'M4'): (50.0, 2.48),  # A2V - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 115.0),  # C3E - M07 - A2V
            ('M2', 'M3', 'M4'): (5.0, 132.0),  # M07 - A2V - M02
            ('M2', 'M3', 'M5'): (5.0, 76.5),   # M07 - A2V - M06
            ('M4', 'M3', 'M5'): (5.0, 122.0),  # M02 - A2V - M06
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 68.0),  # C3E - M07 - A2V - M02
            ('M1', 'M2', 'M3', 'M5'): (0.1, 1, 54.5),  # C3E - M07 - A2V - M06
        },
        'impropers': {},
    },
    'C4C': {
        'bonds': {
            ('M1', 'M2'): (50.0, 2.48),  # A2V - M07
            ('M2', 'M3'): (50.0, 2.48),  # M07 - A2V
            ('M3', 'M5'): (50.0, 3.28),  # A2V - M06
            ('M3', 'M4'): (50.0, 2.48),  # A2V - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 148.0),  # A2V - M07 - A2V
            ('M2', 'M3', 'M4'): (5.0, 132.0),  # M07 - A2V - M02
            ('M2', 'M3', 'M5'): (5.0, 76.5),   # M07 - A2V - M06
            ('M4', 'M3', 'M5'): (5.0, 122.0),  # M02 - A2V - M06
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 68.0),  # A2V - M07 - A2V - M02
            ('M1', 'M2', 'M3', 'M5'): (0.1, 1, 54.5),  # A2V - M07 - A2V - M06
        },
        'impropers': {},
    },
    'C5C': {
        'bonds': {
            ('M1', 'M2'): (20.0, 3.49),  # A1I - M07
            ('M2', 'M3'): (50.0, 2.48),  # M07 - A2V
            ('M3', 'M5'): (50.0, 3.28),  # A2V - M06
            ('M3', 'M4'): (50.0, 2.48),  # A2V - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 148.0),  # A1I - M07 - A2V
            ('M2', 'M3', 'M4'): (5.0, 132.0),  # M07 - A2V - M02
            ('M2', 'M3', 'M5'): (5.0, 76.5),   # M07 - A2V - M06
            ('M4', 'M3', 'M5'): (5.0, 122.0),  # M02 - A2V - M06
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 68.0),  # A1I - M07 - A2V - M02
            ('M1', 'M2', 'M3', 'M5'): (0.1, 1, 54.5),  # A1I - M07 - A2V - M06
        },
        'impropers': {},
    },
    'SHR': {
        'bonds': {
            ('M1', 'M2'): (50.0, 2.84),  # M02 - QdK
            ('M2', 'M3'): (50.0, 3.20),  # QdK - A2V
            ('M3', 'M4'): (50.0, 3.20),  # A2V - QdK
            ('M4', 'M5'): (50.0, 2.94),  # QdK - C3E
            ('M4', 'M7'): (50.0, 3.17),  # QdK - M02
            ('M5', 'M6'): (150.0, 2.42), # C3E - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (30.0, 80.0),   # M02 - QdK - A2V
            ('M2', 'M3', 'M4'): (5.0, 120.0),   # QdK - A2V - QdK
            ('M3', 'M4', 'M5'): (10.0, 133.0),  # A2V - QdK - C3E
            ('M4', 'M5', 'M6'): (15.0, 150.0),  # QdK - C3E - M02
            ('M3', 'M4', 'M7'): (5.0, 147.0),   # A2V - QdK - M02
            ('M5', 'M4', 'M7'): (25.0, 75.0),   # C3E - QdK - M02
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.0, 1, 0.0),    # M02 - QdK - A2V - QdK
            ('M2', 'M3', 'M4', 'M5'): (0.0, 1, 0.0),    # QdK - A2V - QdK - C3E
            ('M3', 'M4', 'M5', 'M6'): (0.0, 1, 0.0),    # A2V - QdK - C3E - M02
            ('M2', 'M3', 'M4', 'M7'): (0.0, 1, 180.0),  # QdK - A2V - QdK - M02
            ('M6', 'M5', 'M4', 'M7'): (0.0, 1, 180.0),  # M02 - C3E - QdK - M02
        },
        'impropers': {},
    },
    'CTT': {
        'bonds': {
            ('M1', 'M2'): (50.0, 2.68),  # M02 - QdK
            ('M2', 'M3'): (50.0, 3.56),  # QdK - P1C
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 102.0),  # M02 - QdK - P1C
            ('M2', 'M3', 'M4'): (5.0, 121.5),  # QdK - P1C - QdK
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 15.0), # M02 - QdK - P1C - QdK
        },
        'impropers': {},
    },
    'ADN': {
        'bonds': {
            ('M1', 'C1'): (50.0, 2.06),  # P1S - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # P1S - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # P1S - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RA1 - RS2 - M05
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # P1S - RS1 - RS2 - RA1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # P1S - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'MTA': {
        'bonds': {
            ('M1', 'C1'): (50.0, 4.05),  # P1C - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # P1C - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # P1C - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RA1 - RS2 - M05
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # P1C - RS1 - RS2 - RA1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # P1C - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'AMP': {
        'bonds': {
            ('M1', 'C1'): (50.0, 3.97),  # M01 - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # M01 - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # M01 - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RA1 - RS2 - M05
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # M01 - RS1 - RS2 - RA1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # M01 - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'APP': {
        'bonds': {
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
            ('C3', 'M1'): (50.0, 3.04),  # M05 - M01
        },
        'angles': {
            ('C1', 'C3', 'M1'): (5.0, 141.5),  # RS1 - M05 - M01
            ('C2', 'C3', 'M1'): (5.0, 81.5),   # RS2 - M05 - M01
        },
        'dihedrals': {
            ('M1', 'C3', 'C2', 'C1'): (0.1, 1, 15.0), # M01 - M05 - RS2 - RS1
        },
        'impropers': {},
    },
    'ADP': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # M01 - PHO
            ('M1', 'C1'): (50.0, 3.83),  # PHO - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M2', 'M1', 'C1'): (5.0, 168.0),  # M01 - PHO - RS1
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # PHO - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # PHO - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RA1 - RS2 - M05
        },
        'dihedrals': {
            ('M2', 'M1', 'C1', 'C2'): (0.1, 1, 130.0), # M01 - PHO - RS1 - RS2
            ('M2', 'M1', 'C1', 'C3'): (0.1, 1, 0.0),   # M01 - PHO - RS1 - M05
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # PHO - RS1 - RS2 - RA1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # PHO - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'ATP': {
        'bonds': {
            ('P3', 'P2'): (50.0, 2.73),  # M01 - PHO
            ('P2', 'P1'): (65.0, 2.37),  # PHO - PHO
            ('P1', 'C1'): (50.0, 3.94),  # PHO - RS1
            ('C1', 'C3'): (200.0, 2.39),  # RS1 - M05
            ('C2', 'C3'): (100.0, 2.09),  # RS2 - M05
        },
        'angles': {
            ('P3', 'P2', 'P1'): (60.0, 158.0),  # M01 - PHO - PHO
            ('P2', 'P1', 'C1'): (10.0, 132.0),  # PHO - PHO - RS1
            ('P1', 'C1', 'C3'): (20.0, 105.5),  # PHO - RS1 - M05
            ('P1', 'C1', 'C2'): (30.0,  97.0),  # PHO - RS1 - RS2
            ('NA', 'C2', 'C3'): (15.0, 144.0),  # RA1 - RS2 - M05
        },
        'dihedrals': {
            ('P3', 'P2', 'P1', 'C1'): (1.5, 1, 176.0),
            ('P2', 'P1', 'C1', 'C2'): (0.8, 1, -61.0),
            ('P2', 'P1', 'C1', 'C3'): (0.5, 1, -8.0),
            ('P1', 'C1', 'C2', 'NA'): (11.0, 1, 132.0),
            ('P1', 'C1', 'C3', 'C2'): (24.0, 1, 92.0),
            ('C3', 'C2', 'NA', 'NB'): (4.0, 1, 103.0),
        },
        'impropers': {
            ('C1', 'P1', 'C2', 'C3'): (30.0, 0, -61.5),  # IMPH C1 P1 C2 C3
        },
    },
    'APR': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # PHO(M2) - PHO(M1)
            ('M2', 'M3'): (50.0, 3.83),  # PHO(M2) - RS1(M3)
            ('M3', 'M4'): (150.0, 2.51), # RS1(M3) - P1S(M4)
            ('M3', 'M5'): (150.0, 2.51), # RS1(M3) - M05(M5)
            ('M4', 'M5'): (150.0, 2.29), # P1S(M4) - M05(M5)
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 168.0),  # PHO(M1) - PHO(M2) - RS1(M3)
            ('M2', 'M3', 'M4'): (5.0, 142.0),  # PHO(M2) - RS1(M3) - P1S(M4)
            ('M2', 'M3', 'M5'): (5.0, 137.5),  # PHO(M2) - RS1(M3) - M05(M5)
            ('M4', 'M3', 'M5'): (5.0, 142.0),  # P1S(M4) - RS1(M3) - M05(M5)
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 130.0), # M1 - M2 - M3 - M4
            ('M1', 'M2', 'M3', 'M5'): (0.1, 1, 0.0),   # M1 - M2 - M3 - M5
            ('M2', 'M3', 'M5', 'M4'): (0.1, 1, 20.0),  # M2 - M3 - M5 - M4
            ('M2', 'M3', 'M4', 'M5'): (0.1, 1, 47.5),  # M2 - M3 - M4 - M5
        },
        'impropers': {},
    },
    'NAD': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # PHO(M2) - PHO(M1)
            ('M2', 'M3'): (50.0, 3.83),  # PHO(M2) - RS1(M3)
            ('M3', 'M4'): (150.0, 2.51), # RS1(M3) - RS2(M4)
            ('M3', 'M5'): (150.0, 2.51), # RS1(M3) - M05(M5)
            ('M4', 'M5'): (150.0, 2.29), # RS2(M4) - M05(M5)
            ('M4', 'M6'): (500.0, 1.47), # RS2(M4) - M06(M6)
            ('M6', 'M7'): (250.0, 2.35), # M06(M6) - A2W(M7)
            ('M7', 'M8'): (250.0, 2.39), # A2W(M7) - A2W(M8)
            ('M8', 'M9'): (50.0, 2.20),  # A2W(M8) - M04(M9)
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 168.0),  # PHO(M1) - PHO(M2) - RS1(M3)
            ('M2', 'M3', 'M4'): (5.0, 142.0),  # PHO(M2) - RS1(M3) - RS2(M4)
            ('M2', 'M3', 'M5'): (5.0, 137.5),  # PHO(M2) - RS1(M3) - M05(M5)
            ('M3', 'M4', 'M6'): (55.0, 144.0), # RS1(M3) - RS2(M4) - M06(M6)
            ('M5', 'M4', 'M6'): (55.0, 126.0), # M05(M5) - RS2(M4) - M06(M6)
            ('M4', 'M6', 'M7'): (50.0, 149.0), # RS2(M4) - M06(M6) - A2W(M7)
            ('M6', 'M7', 'M9'): (15.0, 166.7), # M06(M6) - A2W(M7) - M04(M9)
            ('M7', 'M8', 'M9'): (15.0, 134.0), # A2W(M7) - A2W(M8) - M04(M9)
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 130.0), # M1 - M2 - M3 - M4
            ('M1', 'M2', 'M3', 'M5'): (0.1, 1, 0.0),   # M1 - M2 - M3 - M5
            ('M2', 'M3', 'M5', 'M4'): (0.1, 1, 47.5),  # M2 - M3 - M5 - M4
            ('M2', 'M3', 'M4', 'M6'): (1.0, 1, -150.0), # M2 - M3 - M4 - M6
            ('M3', 'M4', 'M6', 'M7'): (1.0, 1, -165.0), # M3 - M4 - M6 - M7
            ('M4', 'M6', 'M7', 'M9'): (5.0, 1, 180.0),  # M4 - M6 - M7 - M9
            ('M9', 'M8', 'M7', 'M6'): (5.0, 1, 0.0),    # M9 - M8 - M7 - M6
        },
        'impropers': {
            ('M4', 'M3', 'M5', 'M6'): (15.0, 0, 15.0), # IMPR M4 M3 M5 M6
            ('M6', 'M4', 'M7', 'M8'): (15.0, 0, 0.0),  # IMPR M6 M4 M7 M8
        },
    },
    'NAI': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # PHO(M2) - PHO(M1)
            ('M2', 'M3'): (50.0, 3.83),  # PHO(M2) - RS1(M3)
            ('M3', 'M4'): (150.0, 2.51), # RS1(M3) - RS2(M4)
            ('M3', 'M5'): (150.0, 2.51), # RS1(M3) - M05(M5)
            ('M4', 'M5'): (150.0, 2.29), # RS2(M4) - M05(M5)
            ('M4', 'M6'): (500.0, 1.47), # RS2(M4) - P1H(M6)
            ('M6', 'M7'): (250.0, 2.35), # P1H(M6) - A2W(M7)
            ('M7', 'M8'): (250.0, 2.39), # A2W(M7) - A2W(M8)
            ('M8', 'M9'): (50.0, 2.20),  # A2W(M8) - M04(M9)
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 168.0),  # PHO(M1) - PHO(M2) - RS1(M3)
            ('M2', 'M3', 'M4'): (5.0, 142.0),  # PHO(M2) - RS1(M3) - RS2(M4)
            ('M2', 'M3', 'M5'): (5.0, 137.5),  # PHO(M2) - RS1(M3) - M05(M5)
            ('M3', 'M4', 'M6'): (55.0, 144.0), # RS1(M3) - RS2(M4) - P1H(M6)
            ('M5', 'M4', 'M6'): (55.0, 126.0), # M05(M5) - RS2(M4) - P1H(M6)
            ('M4', 'M6', 'M7'): (50.0, 149.0), # RS2(M4) - P1H(M6) - A2W(M7)
            ('M6', 'M7', 'M9'): (15.0, 166.7), # P1H(M6) - A2W(M7) - M04(M9)
            ('M7', 'M8', 'M9'): (15.0, 134.0), # A2W(M7) - A2W(M8) - M04(M9)
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 130.0), # M1 - M2 - M3 - M4
            ('M1', 'M2', 'M3', 'M5'): (0.1, 1, 0.0),   # M1 - M2 - M3 - M5
            ('M2', 'M3', 'M5', 'M4'): (0.1, 1, 47.5),  # M2 - M3 - M5 - M4
            ('M2', 'M3', 'M4', 'M6'): (1.0, 1, -150.0), # M2 - M3 - M4 - M6
            ('M3', 'M4', 'M6', 'M7'): (1.0, 1, -165.0), # M3 - M4 - M6 - M7
            ('M4', 'M6', 'M7', 'M9'): (5.0, 1, 180.0),  # M4 - M6 - M7 - M9
            ('M9', 'M8', 'M7', 'M6'): (5.0, 1, 0.0),    # M9 - M8 - M7 - M6
        },
        'impropers': {
            ('M4', 'M3', 'M5', 'M6'): (15.0, 0, 15.0), # IMPR M4 M3 M5 M6
            ('M6', 'M4', 'M7', 'M8'): (15.0, 0, 0.0),  # IMPR M6 M4 M7 M8
        },
    },
    'COA': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # PHO - PHO
            ('M2', 'M3'): (150.0, 3.97), # PHO - A2V
            ('M3', 'M4'): (150.0, 2.67), # A2V - P1T
            ('M4', 'M5'): (50.0, 3.49),  # P1T - M03
            ('M5', 'M6'): (100.0, 3.64), # M03 - M03
            ('M6', 'M7'): (50.0, 2.65),  # M03 - P1C
        },
        'angles': {
            ('M1', 'M2', 'M3'): (25.0, 122.5), # PHO(M1) - PHO(M2) - A2V(M3)
            ('M2', 'M3', 'M4'): (25.0, 83.0),  # PHO(M2) - A2V(M3) - P1T(M4)
            ('M3', 'M4', 'M5'): (5.0, 114.0),  # A2V(M3) - P1T(M4) - M03(M5)
            ('M4', 'M5', 'M6'): (10.0, 136.5), # P1T(M4) - M03(M5) - M03(M6)
            ('M5', 'M6', 'M7'): (25.0, 129.5), # M03(M5) - M03(M6) - P1C(M7)
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 0.0),    # M1 - M2 - M3 - M4
            ('M2', 'M3', 'M4', 'M5'): (2.0, 1, 123.5),  # M2 - M3 - M4 - M5
            ('M3', 'M4', 'M5', 'M6'): (2.0, 1, -58.0),  # M3 - M4 - M5 - M6
            ('M4', 'M5', 'M6', 'M7'): (1.0, 3, 0.0),    # M4 - M5 - M6 - M7
        },
        'impropers': {},
    },
    'FAD': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # PHO - PHO
            ('M2', 'M3'): (50.0, 3.46),  # PHO - P1T
            ('M3', 'M4'): (50.0, 2.57),  # P1T - P1S
            ('M3', 'M6'): (50.0, 1.95),  # P1T - P1H
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 110.0),  # PHO(M1) - PHO(M2) - P1T(M3)
            ('M2', 'M3', 'M4'): (5.0, 90.0),   # PHO(M2) - P1T(M3) - P1S(M4)
            ('M4', 'M3', 'M6'): (5.0, 145.0),  # P1S(M4) - P1T(M3) - P1H(M6)
            ('M3', 'M6', 'M7'): (25.0, 85.0),  # P1T(M3) - P1H(M6) - A1W(M7)
            ('M3', 'M6', 'M11'): (25.0, 100.0), # P1T(M3) - P1H(M6) - RG3(M11)
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (0.1, 1, 15.0),   # M1 - M2 - M3 - M4
            ('M2', 'M3', 'M4', 'M5'): (0.1, 1, 15.0),   # M2 - M3 - M4 - M5
            ('M4', 'M3', 'M6', 'M7'): (0.2, 1, 45.0),   # M4 - M3 - M6 - M7
            ('M4', 'M3', 'M6', 'M11'): (0.2, 1, -135.0), # M4 - M3 - M6 - M11
        },
        'impropers': {},
    },
    'SAM': {
        'bonds': {
            ('M1', 'M2'): (50.0, 3.55),  # M06 - QdK
            ('M2', 'M3'): (100.0, 2.31), # QdK - M02
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 105.0),  # M06 - QdK - M02
        },
        'dihedrals': {},
        'impropers': {},
    },
    'GMP': {
        'bonds': {
            ('M1', 'C1'): (50.0, 3.97),  # M01 - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # M01 - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # M01 - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RG1 - RS2 - M05
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # M01 - RS1 - RS2 - RG1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # M01 - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'NOS': {
        'bonds': {
            ('M1', 'C1'): (50.0, 2.06),  # P1S - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # P1S - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # P1S - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RA1 - RS2 - M05
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # P1S - RS1 - RS2 - RA1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # P1S - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'C5P': {
        'bonds': {
            ('M1', 'C1'): (50.0, 3.97),  # M01 - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # M01 - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # M01 - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RC1 - RS2 - M05
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # M01 - RS1 - RS2 - RC1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # M01 - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'CTN': {
        'bonds': {
            ('M1', 'C1'): (50.0, 2.06),  # P1S - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # P1S - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # P1S - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RC1 - RS2 - M05
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # P1S - RS1 - RS2 - RC1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # P1S - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'U5P': {
        'bonds': {
            ('M1', 'C1'): (50.0, 3.97),  # M01 - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # M01 - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # M01 - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RU1 - RS2 - M05
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # M01 - RS1 - RS2 - RU1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # M01 - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'UDP': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # M01 - PHO
            ('M1', 'C1'): (50.0, 3.83),  # PHO - RS1
            ('C1', 'C3'): (150.0, 2.51), # RS1 - M05
            ('C2', 'C3'): (150.0, 2.29), # RS2 - M05
        },
        'angles': {
            ('M2', 'M1', 'C1'): (5.0, 168.0),  # M01 - PHO - RS1
            ('M1', 'C1', 'C3'): (5.0, 137.5),  # PHO - RS1 - M05
            ('M1', 'C1', 'C2'): (5.0, 142.0),  # PHO - RS1 - RS2
            ('NA', 'C2', 'C3'): (5.0, 132.0),  # RU1 - RS2 - M05
        },
        'dihedrals': {
            ('M2', 'M1', 'C1', 'C2'): (0.1, 1, 130.0), # M01 - PHO - RS1 - RS2
            ('M2', 'M1', 'C1', 'C3'): (0.1, 1, 0.0),   # M01 - PHO - RS1 - M05
            ('M1', 'C1', 'C2', 'NA'): (0.1, 1, 130.0), # PHO - RS1 - RS2 - RU1
            ('M1', 'C1', 'C3', 'C2'): (0.1, 1, 45.0),  # PHO - RS1 - M05 - RS2
        },
        'impropers': {},
    },
    'UD1': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # PHO - PHO
            ('M2', 'M3'): (50.0, 3.11),  # PHO - RS1
            ('M3', 'M4'): (50.0, 3.58),  # RS1 - P1T
            ('M3', 'M5'): (250.0, 3.02), # RS1 - M05
            ('M3', 'M6'): (50.0, 3.66),  # RS1 - M03
            ('M6', 'M5'): (50.0, 5.17),  # M03 - M05
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 168.0), # PHO - PHO - RS1
            ('M2', 'M3', 'M4'): (5.0, 90.0),  # PHO - RS1 - P1T
            ('M2', 'M3', 'M5'): (5.0, 115.0), # PHO - RS1 - M05
            ('M2', 'M3', 'M6'): (5.0, 85.0),  # PHO - RS1 - M03
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M4'): (1.0, 1, -120.0), # M1 - M2 - M3 - M4
            ('M1', 'M2', 'M3', 'M5'): (0.1, 1, 130.0),  # M1 - M2 - M3 - M5
            ('M1', 'M2', 'M3', 'M6'): (1.0, 1, -55.0),  # M1 - M2 - M3 - M6
        },
        'impropers': {},
    },
    'UPG': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # PHO - PHO
            ('M2', 'M3'): (50.0, 3.11),  # PHO - RS1
            ('M3', 'M6'): (50.0, 3.15),  # RS1 - P1S
            ('M3', 'M4'): (50.0, 2.85),  # RS1 - M05
            ('M4', 'M5'): (50.0, 1.96),  # M05 - M05
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 168.0), # PHO - PHO - RS1
            ('M2', 'M3', 'M6'): (5.0, 126.0), # PHO - RS1 - P1S
            ('M2', 'M3', 'M4'): (5.0, 100.0), # PHO - RS1 - M05
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M6'): (1.0, 1, 125.0), # M1 - M2 - M3 - M6
            ('M1', 'M2', 'M3', 'M4'): (0.2, 1, -75.0), # M1 - M2 - M3 - M4
        },
        'impropers': {},
    },
    'UGA': {
        'bonds': {
            ('M2', 'M1'): (50.0, 2.59),  # PHO - PHO
            ('M2', 'M3'): (50.0, 3.11),  # PHO - RS1
            ('M3', 'M6'): (50.0, 3.15),  # RS1 - M02
            ('M3', 'M4'): (50.0, 2.85),  # RS1 - M05
            ('M4', 'M5'): (50.0, 1.96),  # M05 - M05
        },
        'angles': {
            ('M1', 'M2', 'M3'): (5.0, 168.0), # PHO - PHO - RS1
            ('M2', 'M3', 'M6'): (5.0, 126.0), # PHO - RS1 - M02
            ('M2', 'M3', 'M4'): (5.0, 100.0), # PHO - RS1 - M05
        },
        'dihedrals': {
            ('M1', 'M2', 'M3', 'M6'): (1.0, 1, 125.0), # M1 - M2 - M3 - M6
            ('M1', 'M2', 'M3', 'M4'): (0.2, 1, -75.0), # M1 - M2 - M3 - M4
        },
        'impropers': {},
    },
    'DMG': {
        'bonds': {
            ('M1', 'M2'): (15.0, 3.00),
        },
        'angles': {},
        'dihedrals': {},
        'impropers': {},
    },
    'MG7': {
        'bonds': {
            ('M1', 'C1'): (100.0, 2.00),
            ('C1', 'C2'): (130.0, 2.35),
            ('C1', 'C3'): (150.0, 2.36),
            ('C2', 'C3'): ( 70.0, 2.11),
            ('C2', 'NA'): (209.3, 1.90),
            ('NA', 'NB'): (442.1, 2.07),
            ('NA', 'ND'): (155.0, 3.06),
            ('NB', 'NC'): (113.1, 3.05),
            ('NC', 'ND'): (193.0, 2.86)
        },
        'angles': {
            ('M1', 'C1', 'C2'): (5.0, 115.0),
            ('M1', 'C1', 'C3'): (5.0, 114.0),
            ('C1', 'C2', 'NA'): (7.0, 146.0),
            ('NA', 'C2', 'C3'): (13.9, 134.2),
            ('C2', 'NA', 'NB'): (118.9, 127.3),
            ('C2', 'NA', 'ND'): (95.7, 116.5),
            ('NA', 'NB', 'NC'): (200.0, 82.0),
            ('NB', 'NC', 'ND'): (200.0, 93.7),
            ('NC', 'ND', 'NA'): (200.0, 71.0),
        },
        'dihedrals': {
            ('M1', 'C1', 'C2', 'NA'): (2.2, 1, 152.4),
            ('C1', 'C2', 'NA', 'NB'): (1.1, 1, -141.1),
        },
        'impropers': {
            ('C1', 'M1', 'C2', 'C3'): (4.5, -60.4),
            ('NA', 'C2', 'NB', 'ND'): (3.9, -8.5),
        },
    },
    'HIC': {
        'bonds': {
            ('M1', 'M2'): (),
            ('M2', 'M3'): (),
            ('M3', 'M4'): (),
            ('M3', 'M5'): (),
            ('M4', 'M5'): (),
        },
        'angles': {
            ('M1', 'M2', 'M3'): (),
            ('M2', 'M3', 'M4'): (),
            ('M2', 'M3', 'M5'): (),
        },
        'dihedrals': {
            ('M2', 'M3', 'M5', 'M4'): (),
        },
        'impropers': {},
    },
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def modify_metabolite(psf, system):
    """Overwrite bonded force parameters for CG metabolite residues in *system*.

    The function iterates over every ``Force`` object registered in the OpenMM
    *system* and, for each bonded interaction whose four (or fewer) atoms all
    belong to the same residue and that residue is listed in :data:`metabolome`,
    replaces the current parameters with the hand-tuned values from the
    dictionary.

    All dictionary values are converted from CHARMM units to OpenMM internal
    units (kJ/mol, nm, radians) before being written back.  See the module
    docstring for the full conversion table.

    Parameters
    ----------
    psf : openmm.app.CharmmPsfFile
        Loaded PSF object whose ``topology`` attribute provides atom-name and
        residue-name information.  The atom ordering in the topology must match
        the atom ordering used when *system* was created.
    system : openmm.System
        OpenMM system object, typically produced by
        ``psf.createSystem(params, ...)``.  The forces inside this object are
        modified **in place**.

    Returns
    -------
    openmm.System
        The same *system* object, returned for convenience so the call can be
        chained.

    Notes
    -----
    Force identification strategy
        ``HarmonicBondForce`` and ``PeriodicTorsionForce`` are identified by
        their Python type alone.  ``CustomAngleForce`` and
        ``CustomTorsionForce`` are additionally filtered by the name set on the
        force object (``force.getName()``).  The expected names are
        ``"ReBAngleForce"`` and ``"CustomTorsionForce"`` respectively.  These
        names **must** be assigned with ``force.setName(...)`` before calling
        this function; forces without the expected name are silently skipped.

    Parameter index layout for custom forces
        *CustomAngleForce* ("ReBAngleForce"): per-angle parameters are ordered
        ``[theta0 (rad), kt (kJ/mol/rad²)]``, i.e. index 0 = θ₀, index 1 = kₜ.

        *CustomTorsionForce* ("CustomTorsionForce"): per-torsion parameters are
        ordered ``[k (kJ/mol/rad²), theta0 (rad)]``, i.e. index 0 = k,
        index 1 = θ₀.

    Reverse-key matching
        Bond, angle, and proper dihedral keys are matched in both forward and
        reverse order.  Improper torsion keys are matched in the **forward
        order only** — reversing the atom quartet changes the out-of-plane
        center atom, yielding a physically different interaction.

    Examples
    --------
    Minimal usage after building a CHARMM system::

        import openmm as mm
        from openmm.app import CharmmPsfFile, CharmmParameterSet
        from Metabolome import modify_metabolite

        psf    = CharmmPsfFile('system.psf')
        params = CharmmParameterSet('toppar.str')
        system = psf.createSystem(params)

        for force in system.getForces():
            if isinstance(force, mm.CustomAngleForce):
                force.setName("ReBAngleForce")
            elif isinstance(force, mm.CustomTorsionForce):
                force.setName("CustomTorsionForce")

        system = modify_metabolite(psf, system)
    """
    topology = psf.topology

    print("Building atom-to-residue mapping from topology...")
    atom_map = {
        atom.index: (atom.residue.name, atom.name)
        for atom in topology.atoms()
    }
    target_resnames = set(metabolome.keys())

    counts = {"bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0}

    # Track which dictionary keys were actually matched (for validation)
    matched_keys = {
        resname: {
            'bonds': set(),
            'angles': set(),
            'dihedrals': set(),
            'impropers': set(),
        }
        for resname in metabolome
    }

    for force in system.getForces():

        # ------------------------------------------------------------------
        # 1. HARMONIC BONDS  (HarmonicBondForce)
        #
        #    Potential : E = ½ · k · (r − r₀)²
        #    Dict units: k  [kcal/mol/Å²],  r₀ [Å]
        #    OpenMM    : k  [kJ/mol/nm²],   r₀ [nm]
        #
        #    k  conversion: ×4.184 (kcal→kJ) × 100 (Å⁻²→nm⁻²) × 2 (absorb ½)
        #                 = ×836.8
        #    r₀ conversion: ×0.1
        # ------------------------------------------------------------------
        if isinstance(force, HarmonicBondForce):
            if force.getNumBonds() == 0:
                continue

            for i in range(force.getNumBonds()):
                p1, p2, _length, _k = force.getBondParameters(i)
                res1, name1 = atom_map[p1]
                res2, name2 = atom_map[p2]

                if res1 != res2 or res1 not in target_resnames:
                    continue

                for key in ((name1, name2), (name2, name1)):
                    if key in metabolome[res1]['bonds']:
                        raw_k, raw_b0 = metabolome[res1]['bonds'][key]

                        b0_new = raw_b0 * 0.1
                        # ×4.184 kcal→kJ, ×100 Å⁻²→nm⁻², ×2 absorb ½
                        k_new = raw_k * KCAL_TO_KJ * 100.0 * 2.0

                        force.setBondParameters(i, p1, p2, b0_new, k_new)
                        counts["bonds"] += 1
                        matched_keys[res1]['bonds'].add(key)
                        break

        # ------------------------------------------------------------------
        # 2. CUSTOM ANGLES  (CustomAngleForce named "ReBAngleForce")
        #
        #    Potential : E = ½ · kₜ · (θ − θ₀)²
        #    Dict units: kₜ [kcal/mol/rad²],  θ₀ [degrees]
        #    OpenMM    : kₜ [kJ/mol/rad²],    θ₀ [radians]
        #
        #    kₜ conversion: ×4.184 (kcal→kJ) × 2 (absorb ½)  = ×8.368
        #    θ₀ conversion: ×π/180
        #
        #    Per-parameter index layout: [theta0, kt]  →  idx 0 = θ₀, idx 1 = kₜ
        # ------------------------------------------------------------------
        elif isinstance(force, CustomAngleForce):
            if force.getName() != "ReBAngleForce":
                continue

            print(f"Processing CustomAngleForce '{force.getName()}'...")
            THETA_IDX, KT_IDX = 0, 1

            for i in range(force.getNumAngles()):
                p1, p2, p3, custom_params = force.getAngleParameters(i)
                res1, name1 = atom_map[p1]
                res2, name2 = atom_map[p2]
                res3, name3 = atom_map[p3]

                if not (res1 == res2 == res3) or res1 not in target_resnames:
                    continue

                for key in ((name1, name2, name3), (name3, name2, name1)):
                    if key in metabolome[res1]['angles']:
                        raw_kt, raw_theta0 = metabolome[res1]['angles'][key]

                        theta0_new = raw_theta0 * (math.pi / 180.0)
                        kt_new = raw_kt * KCAL_TO_KJ * 2.0

                        new_params = list(custom_params)
                        new_params[THETA_IDX] = theta0_new
                        new_params[KT_IDX] = kt_new

                        force.setAngleParameters(i, p1, p2, p3, tuple(new_params))
                        counts["angles"] += 1
                        matched_keys[res1]['angles'].add(key)
                        break

        # ------------------------------------------------------------------
        # 3. PROPER DIHEDRALS  (PeriodicTorsionForce)
        #
        #    Potential : E = k · (1 + cos(n · φ − φ₀))
        #    Dict units: k [kcal/mol],  φ₀ [degrees],  n [integer]
        #    OpenMM    : k [kJ/mol],    φ₀ [radians]
        #
        #    k  conversion: ×4.184  (no factor-of-2; potential has no ½)
        #    φ₀ conversion: ×π/180
        # ------------------------------------------------------------------
        elif isinstance(force, PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, _periodicity, _phase, _k = (
                    force.getTorsionParameters(i)
                )
                res1, name1 = atom_map[p1]
                res2, name2 = atom_map[p2]
                res3, name3 = atom_map[p3]
                res4, name4 = atom_map[p4]

                if not (res1 == res2 == res3 == res4) or res1 not in target_resnames:
                    continue

                fwd_key = (name1, name2, name3, name4)
                rev_key = (name4, name3, name2, name1)
                dihedral_dict = metabolome[res1]['dihedrals']

                matched_key = None
                if fwd_key in dihedral_dict:
                    matched_key = fwd_key
                elif rev_key in dihedral_dict:
                    matched_key = rev_key

                if matched_key is None:
                    continue

                params = dihedral_dict[matched_key]
                raw_k, raw_multi, raw_phase = params

                phase_new = raw_phase * (math.pi / 180.0)
                k_new = raw_k * KCAL_TO_KJ  # no ×2: periodic form has no ½

                force.setTorsionParameters(
                    i, p1, p2, p3, p4, raw_multi, phase_new, k_new
                )
                counts["dihedrals"] += 1
                matched_keys[res1]['dihedrals'].add(matched_key)

        # ------------------------------------------------------------------
        # 4. IMPROPER TORSIONS  (CustomTorsionForce named "CustomTorsionForce")
        #
        #    Potential : E = ½ · k · (θ − θ₀)²
        #    Dict units: k [kcal/mol/rad²],  θ₀ [degrees]
        #    OpenMM    : k [kJ/mol/rad²],    θ₀ [radians]
        #
        #    k  conversion: ×4.184 (kcal→kJ) × 2 (absorb ½)  = ×8.368
        #    θ₀ conversion: ×π/180
        #
        #    Per-parameter index layout: [k, theta0]  →  idx 0 = k, idx 1 = θ₀
        # ------------------------------------------------------------------
        elif isinstance(force, CustomTorsionForce):
            if force.getName() != "CustomTorsionForce":
                continue

            K_IDX, THETA_IDX = 0, 1

            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, custom_params = force.getTorsionParameters(i)
                res1, name1 = atom_map[p1]
                res2, name2 = atom_map[p2]
                res3, name3 = atom_map[p3]
                res4, name4 = atom_map[p4]

                if not (res1 == res2 == res3 == res4) or res1 not in target_resnames:
                    continue

                # Impropers: forward order only (reversing changes the
                # out-of-plane center atom)
                fwd_key = (name1, name2, name3, name4)
                improper_dict = metabolome[res1]['impropers']

                params = improper_dict.get(fwd_key)
                if params is None:
                    continue

                raw_k, _, raw_theta0 = params

                theta0_new = raw_theta0 * (math.pi / 180.0)
                k_new = raw_k * KCAL_TO_KJ * 2.0

                new_params = list(custom_params)
                new_params[K_IDX] = k_new
                new_params[THETA_IDX] = theta0_new

                force.setTorsionParameters(
                    i, p1, p2, p3, p4, tuple(new_params)
                )
                counts["impropers"] += 1
                matched_keys[res1]['impropers'].add(fwd_key)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nModification summary:")
    print(f"  -> Modified {counts['bonds']:>8d} harmonic bonds.")
    print(f"  -> Modified {counts['angles']:>8d} custom angles ('ReBAngleForce').")
    print(f"  -> Modified {counts['dihedrals']:>8d} proper dihedrals.")
    print(f"  -> Modified {counts['impropers']:>8d} improper torsions.")
    
    return system