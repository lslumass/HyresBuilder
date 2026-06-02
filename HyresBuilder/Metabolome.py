"""
This module is for parsing the bonded parameter file for CG metabolome.
bonded parameters are first generated based on the parameter file, and then modulated.
"""
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np


# parameter file for CG metabolome
metabolome = {
    'NCA': {
        'bonds': {
            ('P1H', 'A2W'): (150.0, 1.95),
            ('A2W', 'A2W'): (150.0, 1.95),
            ('A2W', 'M04'): (10.0, 2.47),
        },
        'angles': {
            ('P1H', 'A2W', 'M04'): (25.0, 135.0),
            ('A2W', 'A2W', 'M04'): (0.0, 165.0),
        },
        'dihedrals': {},
        'impropers': {
            ('A2W', 'A2W', 'P1H', 'M04'): (25.0, 0, 0.0),
        },
    },
    '2HG': {
        'bonds': {
            ('M02', 'P1S'): (20.0, 2.53),
            ('P1S', 'C3E'): (20.0, 2.40),
            ('C3E', 'M02'): (20.0, 2.32),
        },
        'angles': {
            ('M02', 'P1S', 'C3E'): (25.0, 86.0),
            ('P1S', 'C3E', 'M02'): (15.0, 161.0),
        },
        'dihedrals': {
            ('M02', 'P1S', 'C3E', 'M02'): (0.1, 1, 55.0),
        },
        'impropers': {},
    },
    '2PG': {
        'bonds': {
            ('M01', 'P1S'): (30.0, 3.02),
            ('P1S', 'M02'): (30.0, 2.79),
        },
        'angles': {
            ('M01', 'P1S', 'M02'): (35.0, 91.0),
        },
        'dihedrals': {},
        'impropers': {},
    },
    '3PG': {
        'bonds': {
            ('M01', 'P1S'): (30.0, 3.25),
            ('P1S', 'M02'): (30.0, 2.60),
        },
        'angles': {
            ('M01', 'P1S', 'M02'): (35.0, 137.0),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'UN1': {
        'bonds': {
            ('M02', 'A2V'): (20.0, 2.95),
            ('A2V', 'QdK'): (20.0, 3.12),
            ('QdK', 'M02'): (20.0, 2.83),
        },
        'angles': {
            ('M02', 'A2V', 'QdK'): (5.0, 130.0),
            ('A2V', 'QdK', 'M02'): (25.0, 85.0),
        },
        'dihedrals': {
            ('M02', 'A2V', 'QdK', 'M02'): (0.0, 1, 0.0),
        },
        'impropers': {},
    },
    'AYA': {
        'bonds': {
            ('M03', 'C3E'): (20.0, 2.60),
            ('C3E', 'M02'): (20.0, 3.57),
        },
        'angles': {
            ('M03', 'C3E', 'M02'): (5.0, 90.0),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'NLG': {
        'bonds': {
            ('M03', 'A2V'): (20.0, 3.67),
            ('A2V', 'M02'): (20.0, 2.80),
        },
        'angles': {
            ('M03', 'A2V', 'M02'): (5.0, 82.8),
            ('M02', 'A2V', 'M02'): (10.0, 134.8),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'AKG': {
        'bonds': {
            ('M02', 'C3E'): (20.0, 2.28),
            ('C3E', 'C3E'): (20.0, 2.18),
        },
        'angles': {
            ('M02', 'C3E', 'C3E'): (30.0, 105.6),
            ('M02', 'C3E', 'M02'): (5.0, 164.0),
        },
        'dihedrals': {
            ('M02', 'C3E', 'C3E', 'M02'): (0.0, 1, 0.0),
        },
        'impropers': {},
    },
    'SIN': {
        'bonds': {
            ('M02', 'C3E'): (20.0, 2.34),
        },
        'angles': {
            ('M02', 'C3E', 'M02'): (15.0, 170.0),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'LMR': {
        'bonds': {
            ('M02', 'C3E'): (20.0, 2.40),
        },
        'angles': {
            ('M02', 'C3E', 'M02'): (3.0, 125.0),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'FUM': {
        'bonds': {
            ('M02', 'C3E'): (20.0, 2.36),
        },
        'angles': {
            ('M02', 'C3E', 'M02'): (15.0, 171.0),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'MCIT': {
        'bonds': {
            ('M02', 'P1T'): (20.0, 2.63),
            ('P1T', 'C3E'): (20.0, 2.92),
            ('C3E', 'M02'): (20.0, 2.51),
        },
        'angles': {
            ('M02', 'P1T', 'C3E'): (15.0, 150.0),
            ('M02', 'P1T', 'M02'): (15.0, 85.0),
            ('P1T', 'C3E', 'M02'): (60.0, 57.5),
        },
        'dihedrals': {
            ('M02', 'P1T', 'C3E', 'M02'): (0.5, 1, 157.0),
        },
        'impropers': {
            ('P1T', 'M02', 'C3E', 'M02'): (5.0, 0, 0.0),
        },
    },
    'PAU': {
        'bonds': {
            ('P1S', 'A2V'): (20.0, 2.70),
            ('P1S', 'M03'): (20.0, 3.17),
            ('M03', 'QaD'): (20.0, 2.80),
        },
        'angles': {
            ('P1S', 'A2V', 'P1S'): (5.0, 73.0),
            ('A2V', 'P1S', 'M03'): (5.0, 77.0),
            ('P1S', 'M03', 'QaD'): (5.0, 131.0),
        },
        'dihedrals': {
            ('P1S', 'A2V', 'P1S', 'M03'): (0.5, 1, 30.0),
            ('A2V', 'P1S', 'M03', 'QaD'): (0.5, 1, -5.0),
        },
        'impropers': {},
    },
    'PEP': {
        'bonds': {
            ('M01', 'C3E'): (50.0, 3.81),
            ('C3E', 'M02'): (50.0, 2.81),
        },
        'angles': {
            ('M01', 'C3E', 'M02'): (35.0, 62.0),
        },
        'dihedrals': {},
        'impropers': {},
    },
    '13P': {
        'bonds': {
            ('P1S', 'A2V'): (20.0, 2.71),
            ('A2V', 'M01'): (20.0, 3.17),
        },
        'angles': {
            ('P1S', 'A2V', 'M01'): (5.0, 104.0),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'G6P': {
        'bonds': {
            ('M01', 'RS1'): (50.0, 3.27),
            ('RS1', 'M05'): (250.0, 3.17),
            ('M05', 'M05'): (250.0, 3.13),
        },
        'angles': {
            ('M01', 'RS1', 'M05'): (5.0, 150.0),
        },
        'dihedrals': {},
        'impropers': {
            ('RS1', 'M01', 'M05', 'M05'): (5.0, 0, 0.0),
        },
    },
    'CHT': {
        'bonds': {
            ('P1T', 'M06'): (150.0, 2.93),
        },
        'angles': {},
        'dihedrals': {},
        'impropers': {},
    },
    'CH5': {
        'bonds': {
            ('P1S', 'P1T'): (30.0, 2.57),
            ('P1T', 'RP'): (30.0, 3.53),
            ('RP', 'C3E'): (30.0, 3.27),
            ('C3E', 'M06'): (30.0, 2.67),
        },
        'angles': {
            ('P1S', 'P1T', 'RP'): (5.0, 115.0),
            ('P1T', 'RP', 'C3E'): (5.0, 103.5),
            ('RP', 'C3E', 'M06'): (5.0, 95.5),
        },
        'dihedrals': {
            ('P1S', 'P1T', 'RP', 'C3E'): (0.2, 2, -177.0),
            ('P1T', 'RP', 'C3E', 'M06'): (0.2, 2, -177.0),
        },
        'impropers': {},
    },
    'GSH': {
        'bonds': {
            ('M02', 'M06'): (20.0, 2.78),
            ('M06', 'M03'): (50.0, 4.14),
            ('M03', 'M03'): (50.0, 3.55),
            ('M03', 'P1C'): (20.0, 2.74),
            ('M03', 'M02'): (50.0, 2.83),
        },
        'angles': {
            ('M02', 'M06', 'M03'): (5.0, 115.0),
            ('M06', 'M03', 'M03'): (10.0, 132.0),
            ('M06', 'M03', 'P1C'): (10.0, 129.0),
            ('M03', 'M03', 'M02'): (10.0, 126.5),
            ('M02', 'M03', 'P1C'): (10.0, 82.7),
        },
        'dihedrals': {
            ('M02', 'M06', 'M03', 'M03'): (0.1, 1, -142.0),
            ('M02', 'M06', 'M03', 'P1C'): (0.2, 1, -22.0),
            ('M06', 'M03', 'M03', 'M02'): (0.1, 1, -142.0),
        },
        'impropers': {},
    },
    'TAU': {
        'bonds': {
            ('QdK', 'RS1'): (150.0, 2.98),
        },
        'angles': {},
        'dihedrals': {},
        'impropers': {},
    },
    'BET': {
        'bonds': {
            ('M06', 'QaD'): (150.0, 3.27),
        },
        'angles': {},
        'dihedrals': {},
        'impropers': {},
    },
    'ABU': {
        'bonds': {
            ('QdK', 'C3E'): (20.0, 2.47),
            ('C3E', 'M02'): (20.0, 2.51),
        },
        'angles': {
            ('QdK', 'C3E', 'M02'): (0.5, 106.5),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'Y52': {
        'bonds': {
            ('M06', 'P1T'): (50.0, 3.50),
            ('P1T', 'M02'): (50.0, 2.50),
        },
        'angles': {
            ('M06', 'P1T', 'M02'): (0.5, 113.5),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'ACAR': {
        'bonds': {
            ('M06', 'A2V'): (50.0, 3.28),
            ('A2V', 'M02'): (50.0, 2.48),
            ('A2V', 'M07'): (50.0, 3.64),
        },
        'angles': {
            ('M06', 'A2V', 'M02'): (5.0, 121.0),
            ('M06', 'A2V', 'M07'): (5.0, 96.0),
            ('A2V', 'M02', 'M07'): (5.0, 116.5),
        },
        'dihedrals': {},
        'impropers': {},
    },
    'CAR3': {
        'bonds': {
            ('C3E', 'M07'): (50.0, 2.52),
            ('M07', 'A2V'): (50.0, 2.48),
            ('A2V', 'M02'): (50.0, 2.48),
            ('A2V', 'M06'): (50.0, 3.28),
        },
        'angles': {
            ('C3E', 'M07', 'A2V'): (5.0, 115.0),
            ('M07', 'A2V', 'M02'): (5.0, 132.0),
            ('M07', 'A2V', 'M06'): (5.0, 76.5),
            ('M02', 'A2V', 'M06'): (5.0, 122.0),
        },
        'dihedrals': {
            ('C3E', 'M07', 'A2V', 'M02'): (0.1, 1, 68.0),
            ('C3E', 'M07', 'A2V', 'M06'): (0.1, 1, 54.5),
        },
        'impropers': {},
    },
    'CAR4': {
        'bonds': {
            ('M07', 'A2V'): (50.0, 2.48),
            ('A2V', 'M02'): (50.0, 2.48),
            ('A2V', 'M06'): (50.0, 3.28),
        },
        'angles': {
            ('A2V', 'M07', 'A2V'): (5.0, 148.0),
            ('M07', 'A2V', 'M02'): (5.0, 132.0),
            ('M07', 'A2V', 'M06'): (5.0, 76.5),
            ('M02', 'A2V', 'M06'): (5.0, 122.0),
        },
        'dihedrals': {
            ('A2V', 'M07', 'A2V', 'M02'): (0.1, 1, 68.0),
            ('A2V', 'M07', 'A2V', 'M06'): (0.1, 1, 54.5),
        },
        'impropers': {},
    },
    'CAR5': {
        'bonds': {
            ('A1I', 'M07'): (20.0, 3.49),
            ('M07', 'A2V'): (50.0, 2.48),
            ('A2V', 'M02'): (50.0, 2.48),
            ('A2V', 'M06'): (50.0, 3.28),
        },
        'angles': {
            ('A1I', 'M07', 'A2V'): (5.0, 148.0),
            ('M07', 'A2V', 'M02'): (5.0, 132.0),
            ('M07', 'A2V', 'M06'): (5.0, 76.5),
            ('M02', 'A2V', 'M06'): (5.0, 122.0),
        },
        'dihedrals': {
            ('A1I', 'M07', 'A2V', 'M02'): (0.1, 1, 68.0),
            ('A1I', 'M07', 'A2V', 'M06'): (0.1, 1, 54.5),
        },
        'impropers': {},
    },
    'SHR': {
        'bonds': {
            ('M02', 'QdK'): (50.0, 2.84),
            ('QdK', 'A2V'): (50.0, 3.20),
            ('QdK', 'C3E'): (50.0, 2.94),
            ('QdK', 'M02'): (50.0, 3.17),
            ('C3E', 'M02'): (150.0, 2.42),
        },
        'angles': {
            ('M02', 'QdK', 'A2V'): (30.0, 80.0),
            ('QdK', 'A2V', 'QdK'): (5.0, 120.0),
            ('A2V', 'QdK', 'M02'): (5.0, 147.0),
            ('A2V', 'QdK', 'C3E'): (10.0, 133.0),
            ('QdK', 'C3E', 'M02'): (15.0, 150.0),
            ('M02', 'C3E', 'M02'): (25.0, 75.0),
        },
        'dihedrals': {
            ('M02', 'QdK', 'A2V', 'QdK'): (0.0, 1, 0.0),
            ('QdK', 'A2V', 'QdK', 'C3E'): (0.0, 1, 0.0),
            ('A2V', 'QdK', 'C3E', 'M02'): (0.0, 1, 0.0),
            ('QdK', 'A2V', 'QdK', 'M02'): (0.0, 1, 180.0),
            ('M02', 'C3E', 'QdK', 'M02'): (0.0, 1, 180.0),
        },
        'impropers': {},
    },
    'CYST': {
        'bonds': {
            ('M02', 'QdK'): (50.0, 2.68),
            ('QdK', 'P1C'): (50.0, 3.56),
        },
        'angles': {
            ('M02', 'QdK', 'P1C'): (5.0, 102.0),
            ('QdK', 'P1C', 'QdK'): (5.0, 121.5),
        },
        'dihedrals': {
            ('M02', 'QdK', 'P1C', 'QdK'): (0.1, 1, 15.0),
        },
        'impropers': {},
    },
    'ADN': {
        'bonds': {
            ('P1S', 'RS1'): (50.0, 2.06),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('P1S', 'RS1', 'M05'): (5.0, 137.5),
            ('P1S', 'RS1', 'RS2'): (5.0, 142.0),
            ('RA1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('P1S', 'RS1', 'RS2', 'RA1'): (0.1, 1, 130.0),
            ('P1S', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'MTA': {
        'bonds': {
            ('P1C', 'RS1'): (50.0, 4.05),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('P1C', 'RS1', 'M05'): (5.0, 137.5),
            ('P1C', 'RS1', 'RS2'): (5.0, 142.0),
            ('RA1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('P1C', 'RS1', 'RS2', 'RA1'): (0.1, 1, 130.0),
            ('P1C', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'AMP': {
        'bonds': {
            ('M01', 'RS1'): (50.0, 3.97),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('M01', 'RS1', 'M05'): (5.0, 137.5),
            ('M01', 'RS1', 'RS2'): (5.0, 142.0),
            ('RA1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('M01', 'RS1', 'RS2', 'RA1'): (0.1, 1, 130.0),
            ('M01', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'APP': {
        'bonds': {
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
            ('M05', 'M01'): (50.0, 3.04),
        },
        'angles': {
            ('RS1', 'M05', 'M01'): (5.0, 141.5),
            ('RS2', 'M05', 'M01'): (5.0, 81.5),
        },
        'dihedrals': {
            ('M01', 'M05', 'RS2', 'RS1'): (0.1, 1, 15.0),
        },
        'impropers': {},
    },
    'ADP': {
        'bonds': {
            ('M01', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.83),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('M01', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'M05'): (5.0, 137.5),
            ('PHO', 'RS1', 'RS2'): (5.0, 142.0),
            ('RA1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('M01', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('M01', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'RS1', 'RS2', 'RA1'): (0.1, 1, 130.0),
            ('PHO', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'ATP': {
        'bonds': {
            ('M01', 'PHO'): (50.0, 2.59),
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.83),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('M01', 'PHO', 'PHO'): (50.0, 99.0),
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'M05'): (5.0, 137.5),
            ('PHO', 'RS1', 'RS2'): (5.0, 142.0),
            ('RA1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('M01', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('M01', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'RS1', 'RS2', 'RA1'): (0.1, 1, 130.0),
            ('PHO', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'APR': {
        'bonds': {
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.83),
            ('RS1', 'P1S'): (150.0, 2.51),
            ('RS1', 'M05'): (150.0, 2.51),
            ('P1S', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'P1S'): (5.0, 142.0),
            ('PHO', 'RS1', 'M05'): (5.0, 137.5),
            ('PHO', 'RS1', 'RS2'): (5.0, 142.0),
        },
        'dihedrals': {
            ('PHO', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('PHO', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'PHO', 'RS1', 'P1S'): (0.1, 1, 20.0),
            ('PHO', 'RS1', 'M05', 'RS2'): (0.1, 1, 47.5),
        },
        'impropers': {},
    },
    'NAD': {
        'bonds': {
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.83),
            ('RS1', 'RS2'): (150.0, 2.51),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
            ('RS2', 'M06'): (500.0, 1.47),
            ('M06', 'A2W'): (250.0, 2.35),
            ('A2W', 'A2W'): (250.0, 2.39),
            ('A2W', 'M04'): (50.0, 2.20),
        },
        'angles': {
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'RS2'): (5.0, 142.0),
            ('PHO', 'RS1', 'M05'): (5.0, 137.5),
            ('RS1', 'RS2', 'M06'): (55.0, 144.0),
            ('M05', 'RS2', 'M06'): (55.0, 126.0),
            ('RS2', 'M06', 'A2W'): (50.0, 149.0),
            ('M06', 'A2W', 'M04'): (15.0, 166.7),
            ('A2W', 'A2W', 'M04'): (15.0, 134.0),
        },
        'dihedrals': {
            ('PHO', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('PHO', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'RS1', 'M05', 'RS2'): (0.1, 1, 47.5),
            ('PHO', 'RS1', 'RS2', 'M06'): (1.0, 1, -150.0),
            ('RS1', 'RS2', 'M06', 'A2W'): (1.0, 1, -165.0),
            ('RS2', 'M06', 'A2W', 'M04'): (5.0, 1, 180.0),
            ('M04', 'A2W', 'A2W', 'M06'): (5.0, 1, 0.0),
        },
        'impropers': {
            ('RS2', 'RS1', 'M05', 'M06'): (15.0, 0, 15.0),
            ('M06', 'RS2', 'A2W', 'A2W'): (15.0, 0, 0.0),
        },
    },
    'NAI': {
        'bonds': {
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.83),
            ('RS1', 'RS2'): (150.0, 2.51),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
            ('RS2', 'P1H'): (500.0, 1.47),
            ('P1H', 'A2W'): (250.0, 2.35),
            ('A2W', 'A2W'): (250.0, 2.39),
            ('A2W', 'M04'): (50.0, 2.20),
        },
        'angles': {
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'RS2'): (5.0, 142.0),
            ('PHO', 'RS1', 'M05'): (5.0, 137.5),
            ('RS1', 'RS2', 'P1H'): (55.0, 144.0),
            ('M05', 'RS2', 'P1H'): (55.0, 126.0),
            ('RS2', 'P1H', 'A2W'): (50.0, 149.0),
            ('P1H', 'A2W', 'M04'): (15.0, 166.7),
            ('A2W', 'A2W', 'M04'): (15.0, 134.0),
        },
        'dihedrals': {
            ('PHO', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('PHO', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'RS1', 'M05', 'RS2'): (0.1, 1, 47.5),
            ('PHO', 'RS1', 'RS2', 'P1H'): (1.0, 1, -150.0),
            ('RS1', 'RS2', 'P1H', 'A2W'): (1.0, 1, -165.0),
            ('RS2', 'P1H', 'A2W', 'M04'): (5.0, 1, 180.0),
            ('M04', 'A2W', 'A2W', 'P1H'): (5.0, 1, 0.0),
        },
        'impropers': {
            ('RS2', 'RS1', 'M05', 'P1H'): (15.0, 0, 15.0),
            ('P1H', 'RS2', 'A2W', 'A2W'): (15.0, 0, 0.0),
        },
    },
    'COA': {
        'bonds': {
            ('PHO', 'RS1'): (50.0, 3.83),
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'A2V'): (150.0, 3.97),
            ('A2V', 'P1T'): (150.0, 2.67),
            ('P1T', 'M03'): (50.0, 3.49),
            ('M03', 'M03'): (100.0, 3.64),
            ('M03', 'P1C'): (50.0, 2.65),
        },
        'angles': {
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'RS2'): (5.0, 142.0),
            ('PHO', 'RS1', 'M05'): (5.0, 137.5),
            ('PHO', 'PHO', 'A2V'): (25.0, 122.5),
            ('PHO', 'A2V', 'P1T'): (25.0, 83.0),
            ('A2V', 'P1T', 'M03'): (5.0, 114.0),
            ('P1T', 'M03', 'M03'): (10.0, 136.5),
            ('M03', 'M03', 'P1C'): (25.0, 129.5),
        },
        'dihedrals': {
            ('PHO', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('PHO', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'RS1', 'M05', 'RS2'): (0.1, 1, 47.5),
            ('PHO', 'PHO', 'A2V', 'P1T'): (0.1, 1, 0.0),
            ('PHO', 'A2V', 'P1T', 'M03'): (2.0, 1, 123.5),
            ('A2V', 'P1T', 'M03', 'M03'): (2.0, 1, -58.0),
            ('P1T', 'M03', 'M03', 'P1C'): (1.0, 3, 0.0),
        },
        'impropers': {},
    },
    'FAD': {
        'bonds': {
            ('PHO', 'RS1'): (50.0, 3.83),
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'P1T'): (50.0, 3.46),
            ('P1T', 'P1S'): (50.0, 2.57),
            ('P1T', 'P1H'): (50.0, 1.95),
        },
        'angles': {
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'RS2'): (5.0, 142.0),
            ('PHO', 'RS1', 'M05'): (5.0, 137.5),
            ('PHO', 'PHO', 'P1T'): (5.0, 110.0),
            ('PHO', 'P1T', 'P1S'): (5.0, 90.0),
            ('P1T', 'P1S', 'P1T'): (5.0, 90.0),
            ('P1S', 'P1T', 'P1H'): (5.0, 145.0),
            ('P1T', 'P1H', 'A1W'): (25.0, 85.0),
            ('P1T', 'P1H', 'RG3'): (25.0, 100.0),
        },
        'dihedrals': {
            ('PHO', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('PHO', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'RS1', 'M05', 'RS2'): (0.1, 1, 47.5),
            ('PHO', 'PHO', 'P1T', 'P1S'): (0.1, 1, 15.0),
            ('PHO', 'P1T', 'P1S', 'P1T'): (0.1, 1, 15.0),
            ('P1T', 'P1S', 'P1T', 'P1H'): (0.2, 1, 15.0),
            ('P1S', 'P1T', 'P1H', 'A1W'): (0.2, 1, 45.0),
            ('P1S', 'P1T', 'P1H', 'RG3'): (0.2, 1, -135.0),
        },
        'impropers': {},
    },
    'SAM': {
        'bonds': {
            ('M06', 'RS1'): (50.0, 3.76),
            ('M06', 'QdK'): (50.0, 3.55),
            ('QdK', 'M02'): (100.0, 2.31),
        },
        'angles': {
            ('M06', 'RS1', 'RS2'): (5.0, 142.0),
            ('M06', 'RS1', 'M05'): (5.0, 153.5),
            ('QdK', 'M06', 'RS1'): (5.0, 145.0),
            ('M06', 'QdK', 'M02'): (5.0, 105.0),
        },
        'dihedrals': {
            ('M02', 'QdK', 'M06', 'RS1'): (0.2, 1, -50.0),
            ('QdK', 'M06', 'RS1', 'RS2'): (0.2, 1, 80.0),
        },
        'impropers': {},
    },
    'GMP': {
        'bonds': {
            ('M01', 'RS1'): (50.0, 3.97),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('M01', 'RS1', 'M05'): (5.0, 137.5),
            ('M01', 'RS1', 'RS2'): (5.0, 142.0),
            ('RG1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('M01', 'RS1', 'RS2', 'RG1'): (0.1, 1, 130.0),
            ('M01', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'NOS': {
        'bonds': {
            ('P1S', 'RS1'): (50.0, 2.06),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('P1S', 'RS1', 'M05'): (5.0, 137.5),
            ('P1S', 'RS1', 'RS2'): (5.0, 142.0),
            ('RA1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('P1S', 'RS1', 'RS2', 'RA1'): (0.1, 1, 130.0),
            ('P1S', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'C5P': {
        'bonds': {
            ('M01', 'RS1'): (50.0, 3.97),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('M01', 'RS1', 'M05'): (5.0, 137.5),
            ('M01', 'RS1', 'RS2'): (5.0, 142.0),
            ('RC1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('M01', 'RS1', 'RS2', 'RC1'): (0.1, 1, 130.0),
            ('M01', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'CTN': {
        'bonds': {
            ('P1S', 'RS1'): (50.0, 2.06),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('P1S', 'RS1', 'M05'): (5.0, 137.5),
            ('P1S', 'RS1', 'RS2'): (5.0, 142.0),
            ('RC1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('P1S', 'RS1', 'RS2', 'RC1'): (0.1, 1, 130.0),
            ('P1S', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'U5P': {
        'bonds': {
            ('M01', 'RS1'): (50.0, 3.97),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('M01', 'RS1', 'M05'): (5.0, 137.5),
            ('M01', 'RS1', 'RS2'): (5.0, 142.0),
            ('RU1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('M01', 'RS1', 'RS2', 'RU1'): (0.1, 1, 130.0),
            ('M01', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'UDP': {
        'bonds': {
            ('M01', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.83),
            ('RS1', 'M05'): (150.0, 2.51),
            ('RS2', 'M05'): (150.0, 2.29),
        },
        'angles': {
            ('M01', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'M05'): (5.0, 137.5),
            ('PHO', 'RS1', 'RS2'): (5.0, 142.0),
            ('RU1', 'RS2', 'M05'): (5.0, 132.0),
        },
        'dihedrals': {
            ('M01', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('M01', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'RS1', 'RS2', 'RU1'): (0.1, 1, 130.0),
            ('PHO', 'RS1', 'M05', 'RS2'): (0.1, 1, 45.0),
        },
        'impropers': {},
    },
    'UD1': {
        'bonds': {
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.11),
            ('RS1', 'P1T'): (50.0, 3.58),
            ('RS1', 'M05'): (250.0, 3.02),
            ('RS1', 'M03'): (50.0, 3.66),
            ('M03', 'M05'): (50.0, 5.17),
        },
        'angles': {
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'P1S'): (5.0, 90.0),
            ('PHO', 'RS1', 'M05'): (5.0, 115.0),
            ('PHO', 'RS1', 'M03'): (5.0, 85.0),
        },
        'dihedrals': {
            ('PHO', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('PHO', 'PHO', 'RS1', 'M05'): (0.1, 1, 0.0),
            ('PHO', 'PHO', 'RS1', 'P1S'): (1.0, 1, -120.0),
            ('PHO', 'PHO', 'RS1', 'M03'): (1.0, 1, -55.0),
        },
        'impropers': {},
    },
    'UPG': {
        'bonds': {
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.11),
            ('RS1', 'P1S'): (50.0, 3.15),
            ('RS1', 'M05'): (50.0, 2.85),
            ('M05', 'M05'): (50.0, 1.96),
        },
        'angles': {
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'P1S'): (5.0, 126.0),
            ('PHO', 'RS1', 'M05'): (5.0, 100.0),
        },
        'dihedrals': {
            ('PHO', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('PHO', 'PHO', 'RS1', 'M05'): (0.2, 1, -75.0),
            ('PHO', 'PHO', 'RS1', 'P1S'): (1.0, 1, 125.0),
        },
        'impropers': {},
    },
    'UGA': {
        'bonds': {
            ('PHO', 'PHO'): (50.0, 2.59),
            ('PHO', 'RS1'): (50.0, 3.11),
            ('RS1', 'M02'): (50.0, 3.15),
            ('RS1', 'M05'): (50.0, 2.85),
            ('M05', 'M05'): (50.0, 1.96),
        },
        'angles': {
            ('PHO', 'PHO', 'RS1'): (5.0, 168.0),
            ('PHO', 'RS1', 'M02'): (5.0, 126.0),
            ('PHO', 'RS1', 'M05'): (5.0, 100.0),
        },
        'dihedrals': {
            ('PHO', 'PHO', 'RS1', 'RS2'): (0.1, 1, 130.0),
            ('PHO', 'PHO', 'RS1', 'M05'): (0.2, 1, -75.0),
            ('PHO', 'PHO', 'RS1', 'M02'): (1.0, 1, 125.0),
        },
        'impropers': {},
    },
}

def modify_metabolite(psf, system):
    topology = psf.topology
    
    print("Building atom-to-residue mapping from topology...")
    atom_map = {atom.index: (atom.residue.name, atom.name) for atom in topology.atoms()}
    target_resnames = set(metabolome.keys())

    counts = {"bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0}

    # Loop through all forces in the OpenMM system
    for force in system.getForces():
        
        # 1. HARMONIC BONDS
        if isinstance(force, HarmonicBondForce):
            num_bonds = force.getNumBonds()
            if num_bonds == 0:
                continue
                
            for i in range(num_bonds):
                p1, p2, length, k = force.getBondParameters(i)
                res1, name1 = atom_map[p1]
                res2, name2 = atom_map[p2]
                
                if res1 == res2 and res1 in target_resnames:
                    bond_keys = [(name1, name2), (name2, name1)]
                    for key in bond_keys:
                        if key in metabolome[res1]['bonds']:
                            raw_k, raw_b0 = metabolome[res1]['bonds'][key]
                            
                            # Convert parameters inside the loop:
                            # length: Angstroms -> nm (* 0.1)
                            # k: multiplied by 2 due to OpenMM's 1/2*k*(b-b0)^2 functional form
                            b0_new = raw_b0 * 0.1
                            k_new = raw_k * 2.0
                            
                            force.setBondParameters(i, p1, p2, b0_new, k_new)
                            counts["bonds"] += 1
                            break

        # 2. CUSTOM ANGLES ("ReBAngleForce")
        elif isinstance(force, CustomAngleForce):
            force_name = getattr(force, 'getName', lambda: None)()
            if force_name != "ReBAngleForce":
                continue
                
            print(f"Processing CustomAngleForce '{force_name}'...")
            theta_idx, kt_idx = 0, 1 # Indices mapping to ("theta0", "kt") parameters

            for i in range(force.getNumAngles()):
                p1, p2, p3, custom_params = force.getAngleParameters(i)
                res1, name1 = atom_map[p1]
                res2, name2 = atom_map[p2]
                res3, name3 = atom_map[p3]
                
                if res1 == res2 == res3 and res1 in target_resnames:
                    angle_keys = [(name1, name2, name3), (name3, name2, name1)]
                    for key in angle_keys:
                        if key in metabolome[res1]['angles']:
                            raw_kt, raw_theta0 = metabolome[res1]['angles'][key]
                            
                            # Convert parameters inside the loop:
                            # theta0: Degrees -> Radians (* pi / 180)
                            # kt: multiplied by 2 due to OpenMM's potential math
                            theta0_new = raw_theta0 * (math.pi / 180.0)
                            kt_new = raw_kt * 2.0
                            
                            new_params = list(custom_params)
                            new_params[theta_idx] = theta0_new
                            new_params[kt_idx] = kt_new
                            
                            force.setAngleParameters(i, p1, p2, p3, tuple(new_params))
                            counts["angles"] += 1
                            break

        # 3. PROPER DIHEDRALS (PeriodicTorsionForce)
        elif isinstance(force, PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                res1, name1 = atom_map[p1]
                res2, name2 = atom_map[p2]
                res3, name3 = atom_map[p3]
                res4, name4 = atom_map[p4]
                
                if res1 == res2 == res3 == res4 and res1 in target_resnames:
                    fwd_key = (name1, name2, name3, name4)
                    rev_key = (name4, name3, name2, name1)
                    
                    dihedral_dict = metabolome[res1]['dihedrals']
                    if fwd_key in dihedral_dict or rev_key in dihedral_dict:
                        raw_k, raw_multi, raw_phase = dihedral_dict.get(fwd_key) or dihedral_dict.get(rev_key)
                        
                        # Convert parameters inside the loop:
                        # phase: Degrees -> Radians (* pi / 180)
                        phase_new = raw_phase * (math.pi / 180.0)
                        
                        force.setTorsionParameters(i, p1, p2, p3, p4, raw_multi, phase_new, raw_k)
                        counts["dihedrals"] += 1

        # 4. IMPROPER TORSIONS (CustomTorsionForce named "CustomTorsionForce")
        elif isinstance(force, CustomTorsionForce):
            force_name = getattr(force, 'getName', lambda: None)()
            if force_name != "CustomTorsionForce":
                continue
                
            # Parameter ordering layout indices: 0 = k, 1 = theta0
            k_idx, theta_idx = 0, 1 

            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, custom_params = force.getTorsionParameters(i)
                res1, name1 = atom_map[p1]
                res2, name2 = atom_map[p2]
                res3, name3 = atom_map[p3]
                res4, name4 = atom_map[p4]
                
                if res1 == res2 == res3 == res4 and res1 in target_resnames:
                    fwd_key = (name1, name2, name3, name4)
                    rev_key = (name4, name3, name2, name1)
                    
                    improper_dict = metabolome[res1]['impropers']
                    if fwd_key in improper_dict or rev_key in improper_dict:
                        raw_k, _, raw_theta0 = improper_dict.get(fwd_key) or improper_dict.get(rev_key)
                        
                        # Convert parameters inside the loop:
                        # theta0: Degrees -> Radians (* pi / 180)
                        # k: multiplied by 2 due to OpenMM's potential math
                        theta0_new = raw_theta0 * (math.pi / 180.0)
                        k_new = raw_k * 2.0
                        
                        new_params = list(custom_params)
                        new_params[k_idx] = k_new
                        new_params[theta_idx] = theta0_new
                        
                        force.setTorsionParameters(i, p1, p2, p3, p4, tuple(new_params))
                        counts["impropers"] += 1

    print(f"\nModification summary:")
    print(f" -> Modified {counts['bonds']} harmonic bonds.")
    print(f" -> Modified {counts['angles']} custom angles ('ReBAngleForce').")
    print(f" -> Modified {counts['dihedrals']} proper dihedrals.")
    print(f" -> Modified {counts['impropers']} improper torsions.")
    
    return system