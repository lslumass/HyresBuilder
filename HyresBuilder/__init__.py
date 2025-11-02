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

try:
    from .Convert2CG import (
        at2cg,
        at2hyres,
        at2icon,
    )
    from .Backmap import (
        backmap_structure,
        backmap_trajectory,
        get_map_directory,
        StructureCache,
    )
    from .Rotamer import (
        rotate_about_axis_fast,
        opt_side_chain,
    )
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import HyresBuilder modules: {e}\n",
        ImportWarning
    )


__all__ = [
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__url__',
    # at2cg
    'at2cg',
    'at2hyres',
    'at2icon',
    # backmap functions
    'backmap_structure',
    'backmap_trajectory',
    'get_map_directory',
    'StructureCache',
    # Rotamer functions
    'rotate_about_axis_fast',
    'opt_side_chain',
]

#from .HyresBuilder import *
#from .Geometry import *
#from .RNAbuilder import *
#from .HyresFF import *
