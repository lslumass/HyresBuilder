"""
HyresBuilder prepares coarse-grained (CG) OpenMM simulations.

Supported models: HyRes proteins, iConRNA/iConDNA nucleic acids,
metabolites, aminoglycosides (AGs, e.g. KAN), and CG polymers (QDM, BZM,
PEG/PEO).

Main functions:
- build the custom CG force fields (``HyresBuilder.FFs``) and set up complete
  simulations (``HyresBuilder.utils``: ``setup``, ``iConRNA_setup``,
  ``iConDNA_setup``, ``rG4s_setup``, ``setupMg``)
- convert all-atom structures to CG ones (``HyresBuilder.Convert2CG``)
- construct CG models from sequence (``HyresBuilder.PeptideBuilder``,
  ``HyresBuilder.iConBuilder``) and generate PSF files
  (``HyresBuilder.GenPsf``)

Submodules are not imported automatically; import them explicitly, e.g.
``from HyresBuilder import utils``. Importing the package raises
``ImportError`` if ``psfgen`` is not installed.
"""

__version__ = "4.0.0"
__author__ = "Shanlong Li"
__email__ = "shanlongli@umass.edu"
__license__ = 'MIT'
__url__ = 'https://github.com/lslumass/HyresBuilder'

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__url__',
]


try:
    import psfgen
except ImportError:
    raise ImportError(
        "psfgen is required but not installed. "
        "Install it via conda:\n\n"
        "    conda install -c conda-forge psfgen\n"
    )

