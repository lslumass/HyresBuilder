""" HyResBuilder package is based on the PeptideBuilder, which was modified for HyRes model.
``PeptideBuilder`` package for creating peptide models in PDB format based on geometrical parameters
Written by Matthew Z. Tien, Dariya K. Sydykova, Austin G. Meyer, and Claus O. Wilke.
Python modules
----------------
The package consists of the following Python modules:
* HyresBuilder
* Geometry
"""
__version__ = "1.1.0"
from .HyresBuilder import *
from .Geometry import *
