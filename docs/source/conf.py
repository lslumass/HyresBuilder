import os
import sys
import builtins
from unittest.mock import MagicMock

# 1. Setup the mocks
MOCK_MODULES = [
    'psfgen',
    'openmm',
    'openmm.app',
    'openmm.unit',
    'mdtraj',
    'numpy',
    'numpy.linalg',
    'pkg_resources',
]

for mod in MOCK_MODULES:
    sys.modules[mod] = MagicMock()

# 2. Specifically handle the 'unit' name error
# Create a mock for unit and inject it into builtins
mock_unit = MagicMock()
builtins.unit = mock_unit

# 3. Ensure the project path is correct
# Since your build log shows HyresBuilder is in the root:
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'HyresBuilder'
copyright = '2026, Shanlong Li, Jian Huang, Yumeng Zhang, Xiping Gong, Jianhan Chen'
author = 'Shanlong Li, Jian Huang, Yumeng Zhang, Xiping Gong, Jianhan Chen'
release = 'latest'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
# Make sure this directory actually exists in docs/source/
html_static_path = ['_static']
# color
html_theme_options = {
    'style_nav_header_background': '#881c1c'
}
