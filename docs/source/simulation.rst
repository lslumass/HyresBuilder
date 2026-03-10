Running Simulations
===================

HyresBuilder provides a ready-to-use simulation script ``run_latest.py`` for
running HyRes protein and iConRNA coarse-grained MD simulations with OpenMM.

.. contents::
   :local:
   :depth: 1

----

Command-Line Usage
------------------

.. code-block:: bash

   python run_latest.py -c <pdb> -p <psf> -o <output> [options]

**Required arguments:**

.. list-table::
   :widths: 15 15 60
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``-c``, ``--pdb``
     - ``conf.pdb``
     - Input PDB coordinate file (coarse-grained).
   * - ``-p``, ``--psf``
     - ``conf.psf``
     - Input CHARMM PSF topology file.
   * - ``-o``, ``--out``
     - ``system``
     - Prefix for all output files (``.xtc``, ``.pdb``, ``.log``, ``.chk``).
   * - ``-t``, ``--temp``
     - ``303``
     - Simulation temperature in Kelvin.
   * - ``-b``, ``--box``
     - —
     - Box dimensions in nm. One value for cubic, three for orthorhombic.
   * - ``-s``, ``--salt``
     - ``150.0``
     - Ionic strength in mM.
   * - ``-e``, ``--ens``
     - ``NVT``
     - Ensemble: ``NPT``, ``NVT``, or ``non`` (non-periodic).
   * - ``-m``, ``--Mg``
     - ``0.0``
     - Mg²⁺ concentration in mM.

----

Simulation Parameters
---------------------

The following parameters are set internally in the script and can be adjusted
by editing ``run_latest.py`` directly:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``dt_equil``
     - ``0.001 ps``
     - Time step for equilibration. Use ``0.0001 ps`` for bad initial configurations.
   * - ``dt_prod``
     - ``0.008 ps``
     - Time step for production simulation.
   * - ``prod_step``
     - ``250,000,000``
     - Number of production steps (2 µs at 0.008 ps/step).
   * - ``equil_step``
     - ``10,000``
     - Number of equilibration steps.
   * - ``log_freq``
     - ``1,250``
     - Steps between log file entries.
   * - ``traj_freq``
     - ``5,000``
     - Steps between trajectory frames (DCD/XTC).
   * - ``pdb_freq``
     - ``12,500,000``
     - Steps between PDB snapshot frames.
   * - ``chk_freq``
     - ``125,000``
     - Steps between checkpoint saves.
   * - ``pressure``
     - ``1 atm``
     - Pressure for NPT barostat.
   * - ``friction``
     - ``0.1 ps⁻¹``
     - Friction coefficient for Langevin integrator.
   * - ``er_ref``
     - ``60.0``
     - Reference dielectric constant.
   * - ``gpu_id``
     - ``"0"``
     - CUDA device index.

----

Simulation Workflow
-------------------

The script runs in four sequential stages:

**1. System Setup**

Calls :func:`utils.setup` to build the full OpenMM system including HyRes/iConRNA
force field, integrator, and CUDA simulation context. The serialized system is
saved as ``<out>.xml`` for inspection.

**2. Energy Minimization**

.. code-block:: text

   Max iterations : 500,000
   Tolerance      : 0.01 kJ/mol

Potential energy is printed before and after minimization.

**3. Equilibration**

Runs ``10,000`` steps at ``dt = 0.001 ps`` using the Langevin integrator
to relax the system before production.

**4. Production Simulation**

Switches to ``dt = 0.008 ps`` and runs ``250,000,000`` steps. Output files
are written during production:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - File
     - Contents
   * - ``<out>.dcd``
     - DCD trajectory (every 5,000 steps)
   * - ``<out>.pdb``
     - PDB snapshots (every 12,500,000 steps)
   * - ``<out>.log``
     - Step, temperature, total energy, speed
   * - ``<out>.chk``
     - Checkpoint for restarting (every 125,000 steps)
   * - ``<out>.xml``
     - Serialized OpenMM system (written at setup)

----

Examples
--------

Single chain in non-periodic system at 298 K, 150 mM NaCl:

.. code-block:: bash

   python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e non -s 150

RNA simulation with 5 mM MgCl₂ in a 10 nm cubic box at 298 K:

.. code-block:: bash

   python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e NVT -b 10 -s 150 -m 5

Slab simulation of condensate in a 15×15×50 nm box:

.. code-block:: bash

   python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e NVT -b 15 15 50 -s 150

----

Custom Force Modifications
--------------------------

To add extra forces or modify the system before simulation, define a custom
function and pass it to ``utils.setup``:

.. code-block:: python

   from openmm import CustomExternalForce

   def mod(system):
       custom_force = CustomExternalForce("k*x^2")
       custom_force.addGlobalParameter("k", 100.0)
       system.addForce(custom_force)

   system, sim = utils.setup(params, modification=mod)

After modifying the system post-setup (e.g. changing force parameters via
the context), reinitialize the context to apply changes:

.. code-block:: python

   sim.context.reinitialize(preserveState=True)

----

Output Files
------------

After a completed run, the following files will be present:

.. code-block:: text

   test.xml      ← serialized OpenMM system (inspect forces and parameters)
   test.dcd      ← DCD trajectory for analysis
   test.pdb      ← periodic PDB snapshots
   test.log      ← thermodynamic log (step, T, energy, speed)
   test.chk      ← final checkpoint (use to restart simulation)

----

.. note::
   To switch from DCD to XTC trajectory format, comment out the
   ``DCDReporter`` line and uncomment the ``XTCReporter`` line in the script.
