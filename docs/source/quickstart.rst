Quickstart
==========

HyresBuilder prepares structures and force fields for **HyRes protein** and **iConRNA** coarse-grained simulations.

.. contents::
   :local:
   :depth: 1

----

Installation
------------

.. code-block:: bash

   git clone https://github.com/lslumass/HyresBuilder.git
   cd HyresBuilder
   pip install .

**Dependencies:**

- `OpenMM <https://openmm.org/>`_
- psfgen (install via conda):

  .. code-block:: bash

     conda install -c conda-forge psfgen

- numpy, numba, MDAnalysis

----

Step 1: Prepare Your PDB File
------------------------------

You have three options depending on your starting point:

A. Build a peptide from sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the command line:

.. code-block:: bash

   pepbuilder myprotein ACDEFGHIKLM

This outputs ``myprotein.pdb``.

From Python:

.. code-block:: python

   from HyresBuilder import PeptideBuilder
   PeptideBuilder.build_peptide("myprotein", "ACDEFGHIKLM")

----

B. Build an RNA strand from sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the command line:

.. code-block:: bash

   rnabuilder myrna AUGCAUGC

This outputs ``myrna.pdb``.

From Python:

.. code-block:: python

   from HyresBuilder import RNABuilder
   RNABuilder.build("myrna", "AUGCAUGC")

----

C. Convert atomistic structure to coarse-grained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the command line:

.. code-block:: bash

   convert2cg input.pdb output.pdb

With options:

.. code-block:: bash

   # Add backbone amide hydrogens (if missing from PDB)
   convert2cg input.pdb output.pdb --hydrogen

   # Set terminal charge status
   convert2cg input.pdb output.pdb --terminal charged

From Python:

.. code-block:: python

   from HyresBuilder import Convert2CG
   Convert2CG.at2cg("input.pdb", "output.pdb", terminal="neutral")

.. note::
   If your PDB is missing backbone amide hydrogens (H-N), use the ``--hydrogen`` flag.

.. warning::
   Make sure there are no duplicated chain IDs or seg IDs for adjacent chains in your PDB file.

.. note::
   A PSF file is automatically created when using ``Convert2CG``.

----

Step 2: Prepare Your PSF File
------------------------------

If you used ``Convert2CG`` in Step 1C, the PSF is already created — skip this step.

Otherwise, generate the PSF from your CG PDB:

From the command line:

.. code-block:: bash

   genpsf conf.pdb conf.psf

   # With terminal charge status
   genpsf conf.pdb conf.psf --ter charged

From Python:

.. code-block:: python

   from HyresBuilder import GenPsf
   GenPsf.genpsf("conf.pdb", "conf.psf", terminal="neutral")

Available terminal options: ``neutral``, ``charged``, ``NT``, ``CT``, ``positive``.

.. note::
   It is recommended to use different chain IDs for adjacent chains in your PDB,
   especially for phase separation simulations. See the
   `packmol example <https://github.com/lslumass/HyresBuilder/blob/main/examples/packmol.inp>`_
   for guidance.

----

Step 3: Run Simulation
-----------------------

Use the provided ``run_latest.py`` script:

.. code-block:: bash

   python run_latest.py -c conf.pdb -p conf.psf -o output_prefix [options]

**Common examples:**

Single chain in non-periodic system at 298 K, 150 mM NaCl:

.. code-block:: bash

   python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e non -s 150

RNA simulation with 5 mM MgCl2 in a 10 nm cubic box:

.. code-block:: bash

   python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e NVT -b 10 -s 150 -m 5

Slab simulation of condensate in a 15×15×50 nm box:

.. code-block:: bash

   python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e NVT -b 15 15 50 -s 150

**Full options:**

.. code-block:: text

   -c, --pdb     PDB file (default: conf.pdb)
   -p, --psf     PSF file (default: conf.psf)
   -o, --out     Output prefix for .xtc, .pdb, .log, .chk files
   -t, --temp    Temperature in K (default: 303 K)
   -b, --box     Box dimensions in nm, e.g. 50 50 50
   -s, --salt    NaCl concentration in mM (default: 150 mM)
   -e, --ens     Ensemble: NPT, NVT, or non (non-periodic)
   -m, --Mg      Mg2+ concentration in mM

----

Performance Benchmarks
-----------------------

Tested on **OpenMM 8.2, GTX 2080Ti**:

+---------------------------+-------------+-----------------+
| System                    | Beads       | Performance     |
+===========================+=============+=================+
| Monomer                   | 800         | 13 µs/day       |
+---------------------------+-------------+-----------------+
| GY23 LLPS                 | 29,000      | 1.5 µs/day      |
+---------------------------+-------------+-----------------+
| LAF1-RGG LLPS             | 105,000     | 550 ns/day      |
+---------------------------+-------------+-----------------+
| 0N4R Tau fibril           | 230,000     | 230 ns/day      |
+---------------------------+-------------+-----------------+

Tested on **L40S**:

+---------------------------+-------------+-----------------+
| System                    | Beads       | Performance     |
+===========================+=============+=================+
| TDP43                     | 90,000      | 1.9 µs/day      |
+---------------------------+-------------+-----------------+
| EWSLCD                    | 160,000     | 1.1 µs/day      |
+---------------------------+-------------+-----------------+
| 0N4R Tau fibril           | 230,000     | 750 ns/day      |
+---------------------------+-------------+-----------------+
