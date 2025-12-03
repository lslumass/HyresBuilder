from setuptools import setup
from pathlib import Path


# Read version from package
version_file = Path(__file__).parent / 'HyresBuilder' / '__init__.py'
version_info = {}
if version_file.exists():
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                exec(line, version_info)
                break
__version__ = version_info.get('__version__', '1.0.0')

# Read README
readme_file = Path(__file__).parent / 'README.md'
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = (
        'HyResBuilder is for preparing HyRes protein model and iConRNA model.'
        'Main Functions:'
        '- build HyRes and/or iConRNA force field'
        '- convert all-atom structures to CG ones'
        '- backmap CG structures to all-atom ones'
        '- construct CG model from sequence'
    )
INSTALL_REQUIRES = ["Biopython", "MDAnalysis>=2.0.0", "numpy>=1.19.0"]

TEST_REQUIRES = [
    # testing and coverage
    "pytest",
    "coverage",
    "pytest-cov",
    # to be able to run `python setup.py checkdocs`
    "collective.checkdocs",
    "pygments",
]

setup(
    name="HyresBuilder",
    version=__version__,
    author="Shanlong Li",
    author_email="shanlongli@umass.edu",
    description="Prepare HyRes protein and iConRNA simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lslumass/HyresBuilder",
    download_url="https://github.com/lslumass/HyresBuilder/releases",
    platforms="Tested on Ubuntu 22.04",
    packages=["HyresBuilder"],
    package_dir={'HyresBuilder':'HyresBuilder'},
    package_data={"HyresBuilder":["map/*.map", "forcefield/*.inp"]},
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require={"test": TEST_REQUIRES + INSTALL_REQUIRES,},

    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Intended Audience :: Science/Research",
    ],

    entry_points={
        'console_scripts': [
            'convert2cg=HyresBuilder.Convert2CG:main',
            'hyresbuilder=HyresBuilder.HyresBuilder:main',
            'genpsf=HyresBuilder.GenPsf:main',
        ]
    }
)
