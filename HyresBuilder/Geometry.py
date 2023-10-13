"""This module is part of the PeptideBuilder library,
written by Matthew Z. Tien, Dariya K. Sydykova,
Austin G. Meyer, and Claus O. Wilke, which was modified by Shanlong Li,
and named as HyresBuilder.

The Geometry module contains the default geometries of
all 20 amino acids for HyRes model. The main function to be used is the
geometry() function, which returns the default geometry
for the requested amino acid.

This file is provided to you under the MIT License."""

import random
from typing import List


class Geo:
    """Geometry base class"""

    residue_name: str

    # Geometry to bring together residue
    peptide_bond: float
    CA_C_N_angle: float
    C_N_CA_angle: float

    # Backbone coordinates
    N_CA_C_angle: float
    CA_N_length: float
    CA_C_length: float
    phi: float
    psi_im1: float
    omega: float

    # Carbonyl atom
    C_O_length: float
    CA_C_O_angle: float
    N_CA_C_O_diangle: float

    # backbone Hydrogen atom
    N_H_length: float
    CA_N_H_angle: float
    C_CA_N_H_diangle: float

    def __repr__(self) -> str:
        repr = ""
        for var in self.__dict__:
            repr += "%s = %s\n" % (var, self.__dict__[var])
        return repr


class GlyGeo(Geo):
    """Geometry of Glycine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.residue_name = "G"


class AlaGeo(Geo):
    """Geometry of Alanin"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.63
        self.C_CA_CB_angle = 111.2
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "A"


class SerGeo(Geo):
    """Geometry of Serine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.03
        self.C_CA_CB_angle = 109.4
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "S"


class CysGeo(Geo):
    """Geometry of Cystine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.40
        self.C_CA_CB_angle = 118.3
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "C"


class ValGeo(Geo):
    """Geometry of Valine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.01
        self.C_CA_CB_angle = 116.3
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "V"


class IleGeo(Geo):
    """Geometry of Isoleucine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.33
        self.C_CA_CB_angle = 117.8
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "I"


# The following function is commented out, because it is not
# recommended to randomize rotamers for isoleucine. The underlying
# reason for this recommendation is that isoleucine's beta-carbon
# is a chiral center.
##    def generateRandomRotamers(self):
##        rotamer_bins = [-60, 60, 180]
##        tempList = []
##        for i in range(0, 3):
##            tempList.append(random.choice(rotamer_bins))
##        self.inputRotamers(tempList)


class LeuGeo(Geo):
    """Geometry of Leucine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.71
        self.C_CA_CB_angle = 120.6
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "L"


class ThrGeo(Geo):
    """Geometry of Threonine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.01
        self.C_CA_CB_angle = 116.2
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "T"


class ArgGeo(Geo):
    """Geometry of Arginine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.56
        self.C_CA_CB_angle = 118.5
        self.N_C_CA_CB_diangle = 35.3

        self.CB_CC_length = 3.51
        self.CA_CB_CC_angle = 137.9
        self.N_CA_CB_CC_diangle = 0.0

        self.residue_name = "R"

    def inputRotamers(self, rotamers: List[float]) -> None:
        try:
            self.N_CA_CB_CC_diangle = rotamers[0]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CC_diangle = 0.0


class LysGeo(Geo):
    """Geometry of Lysine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.57
        self.C_CA_CB_angle = 117.0
        self.N_C_CA_CB_diangle = 35.3

        self.CB_CC_length = 3.02
        self.CA_CB_CC_angle = 144.9
        self.N_CA_CB_CC_diangle = 75.0

        self.residue_name = "K"

    def inputRotamers(self, rotamers: List[float]) -> None:
        try:
            self.N_CA_CB_CC_diangle = rotamers[0]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CC_diangle = 75.0



class AspGeo(Geo):
    """Geometry of Aspartic Acid"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.54
        self.C_CA_CB_angle = 118.5
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "D"


class AsnGeo(Geo):
    """Geometry of Asparagine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.56
        self.C_CA_CB_angle = 127.0
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "N"


class GluGeo(Geo):
    """Geometry of Glutamic Acid"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 3.14
        self.C_CA_CB_angle = 116.2
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "E"


class GlnGeo(Geo):
    """Geometry of Glutamine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 3.23
        self.C_CA_CB_angle = 119.1
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "Q"


class MetGeo(Geo):
    """Geometry of Methionine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 3.26
        self.C_CA_CB_angle = 119.8
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "M"


class HisGeo(Geo):
    """Geometry of Histidine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.94
        self.C_CA_CB_angle = 112.5
        self.N_C_CA_CB_diangle = 35.3

        self.CB_CC_length = 2.52
        self.CA_CB_CC_angle = 129.5
        self.N_CA_CB_CC_diangle = 0.0
        
        self.CC_CD_length = 1.86
        self.CB_CC_CD_angle = 66.0
        self.CA_CB_CC_CD_diangle = 180.0

        self.residue_name = "H"


class ProGeo(Geo):
    """Geometry of Proline"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.94
        self.C_CA_CB_angle = 121.7
        self.N_C_CA_CB_diangle = 35.3

        self.residue_name = "P"


class PheGeo(Geo):
    """Geometry of Phenylalanine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.35
        self.C_CA_CB_angle = 123.2
        self.N_C_CA_CB_diangle = 35.3

        self.CB_CC_length = 2.45
        self.CA_CB_CC_angle = 116.0
        self.N_CA_CB_CC_diangle = 0.00

        self.CC_CD_length = 2.22
        self.CB_CC_CD_angle = 75.66
        self.CA_CB_CC_CD_diangle = 180.00

        self.residue_name = "F"


class TyrGeo(Geo):
    """Geometry of Tyrosine"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 2.35
        self.C_CA_CB_angle = 123.6
        self.N_C_CA_CB_diangle = 35.3

        self.CB_CC_length = 2.45
        self.CA_CB_CC_angle = 115.8
        self.N_CA_CB_CC_diangle = 0.00

        self.CC_CD_length = 2.44
        self.CB_CC_CD_angle = 92.1
        self.CA_CB_CC_CD_diangle = 180.00

        self.residue_name = "Y"


class TrpGeo(Geo):
    """Geometry of Tryptophan"""

    def __init__(self):
        self.CA_N_length = 1.45
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.6

        self.C_O_length = 1.23
        self.CA_C_O_angle = 121.6
        self.N_CA_C_O_diangle = 180.0
        
        self.N_H_length = 0.98
        self.C_N_H_angle = 120.0
        self.C_CA_N_H_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.99
        self.C_CA_CB_angle = 116.0
        self.N_C_CA_CB_diangle = 35.3

        self.CB_CC_length = 2.58
        self.CA_CB_CC_angle = 125.6
        self.N_CA_CB_CC_diangle = 0.00

        self.CB_CD_length = 2.53
        self.CA_CB_CD_angle = 133.7
        self.N_CA_CB_CD_diangle = 180.00
        
        self.CD_CE_length = 2.15
        self.CB_CD_CE_angle = 98.6
        self.CA_CB_CD_CE_diangle = 180.00
        
        self.CD_CF_length = 2.15
        self.CB_CD_CF_angle = 168.1
        self.CA_CB_CD_CF_diangle = 180.00

        self.residue_name = "W"


def geometry(AA: str) -> Geo:
    """Generates the geometry of the requested amino acid.
    The amino acid needs to be specified by its single-letter
    code. If an invalid code is specified, the function
    returns the geometry of Glycine."""
    if AA == "G":
        return GlyGeo()
    elif AA == "A":
        return AlaGeo()
    elif AA == "S":
        return SerGeo()
    elif AA == "C":
        return CysGeo()
    elif AA == "V":
        return ValGeo()
    elif AA == "I":
        return IleGeo()
    elif AA == "L":
        return LeuGeo()
    elif AA == "T":
        return ThrGeo()
    elif AA == "R":
        return ArgGeo()
    elif AA == "K":
        return LysGeo()
    elif AA == "D":
        return AspGeo()
    elif AA == "E":
        return GluGeo()
    elif AA == "N":
        return AsnGeo()
    elif AA == "Q":
        return GlnGeo()
    elif AA == "M":
        return MetGeo()
    elif AA == "H":
        return HisGeo()
    elif AA == "P":
        return ProGeo()
    elif AA == "F":
        return PheGeo()
    elif AA == "Y":
        return TyrGeo()
    elif AA == "W":
        return TrpGeo()
    else:
        return GlyGeo()
