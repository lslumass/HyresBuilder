"""
HyResBuilder is for preparing HyRes protein model and iConRNA model.

Main Functions:
- build HyRes and/or iConRNA force field
- convert all-atom structures to CG ones
- backmap CG structures to all-atom ones
- construct CG model from sequence
"""

__version__ = "3.5.0"
__author__ = "Shanlong Li"
__email__ = "shanlongli@umass.edu"
__license__ = 'MIT'
__url__ = 'https://github.com/lslumass/HyresBuilder'

from .HyresBuilder import *
#from .Geometry import *
#from .RNAbuilder import *
#from .HyresFF import *
