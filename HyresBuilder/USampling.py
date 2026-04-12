"""
Umbrella sampling and WHAM analysis
=====================================

This module provides utilities for setting up and running umbrella sampling
simulations with OpenMM, and analysing the resulting CV trajectories with the
histogram-free Weighted Histogram Analysis Method (WHAM) to produce a
potential of mean force (PMF).

Workflow
--------
1. **Define windows** – :func:`US_define_windows` returns evenly spaced CV
   target values between two endpoints.
2. **Steer to start** – :func:`US_initial_windows` uses SMD to bring the
   system to the first window target before production.
3. **Generate window structures** – :func:`US_create_windows` uses SMD to
   sweep from the start to the end CV, saving a PDB file at each window.
4. **Production sampling** – :func:`US_collect_cv` applies a harmonic bias at
   each window and records the CV time series.
5. **Generate metafile** – :func:`US_gen_metafile` writes the metafile that
   maps each CV trajectory to its restraint parameters.
6. **Compute PMF** – :class:`WHAM` reads the metafile and CV trajectories and
   returns the unbiased PMF in a single call to :meth:`WHAM.compute_pmf`.

Input file formats
------------------
**Metafile** (e.g. ``metafile.txt``) — one umbrella window per line.
Blank lines and lines starting with ``#`` are ignored::

    cv_values_window_0.txt   0.76   300
    cv_values_window_1.txt   1.01   300
    ...

Columns (whitespace-separated):

1. Path to the CV trajectory file.
2. Restraint centre :math:`r_k^0` (same units as the CV).
3. Harmonic force constant :math:`K_k` (kJ mol⁻¹ nm⁻²).

**CV trajectory file** (e.g. ``cv_values_window_0.txt``) — two columns::

    0.000000e+00   7.634050e-01
    1.000000e+00   6.905046e-01
    ...

Column 0 is the time step (ignored); column 1 is the CV value.

WHAM equations
--------------
The histogram-free WHAM self-consistent equations are:

.. math::

    f_k^{-1} = \\sum_{i,n}
        \\frac{\\exp(-\\beta U_k(r_{i,n}))}
             {\\sum_j N_j f_j \\exp(-\\beta U_j(r_{i,n}))}

where :math:`U_k(r) = \\frac{1}{2} K_k (r - r_k^0)^2` is the harmonic
restraint of window *k*, :math:`r_{i,n}` is the CV at frame *n* of
trajectory *i*, and :math:`\\beta = 1/k_\\mathrm{B}T`.  Once converged,
the unbiased PMF is:

.. math::

    A(r) = -k_\\mathrm{B} T \\ln P^0(r)

Examples
--------
**Python API:**

.. code-block:: python

    from wham import WHAM

    wham = WHAM(T=300.0, metadata='metafile.txt')
    bins, pmf = wham.compute_pmf(hmin=0.5, hmax=5.5, num_bins=100,
                                 save_pmf='pmf.txt')

**Command line:**

.. code-block:: bash

    # Generate metafile then compute the PMF
    python wham.py metafile.txt 0.5 5.5 18 300 -T 300 -b 100

    # Skip metafile generation (metafile already exists)
    python wham.py metafile.txt 0.5 5.5 18 300 --no-gen-metafile
"""

import numpy as np
import os
from openmm.unit import *
from openmm.app import *
from openmm import *


# ---------------------------------------------------------------------------
# Umbrella sampling helpers
# ---------------------------------------------------------------------------

def US_define_windows(r0, r1, window_num):
    """Define umbrella sampling window target CV values.

    Parameters
    ----------
    r0 : float
        Starting CV value (nanometers).
    r1 : float
        Ending CV value (nanometers).
    window_num : int
        Number of evenly spaced windows between *r0* and *r1* inclusive.

    Returns
    -------
    windows : numpy.ndarray, shape (window_num,)
        Evenly spaced CV target values in nanometers.
    """
    return np.linspace(r0, r1, window_num)


def US_initial_windows(system, sim, group1, group2, r0,
                        rcut=0.5, fc_pull=1000.0, v_pull=0.01,
                        total_steps=100000, increment_steps=2):
    """Steer the system to the first umbrella window via SMD.

    A harmonic bias (``CustomCVForce``) is applied to the COM distance
    between *group1* and *group2*.  The anchor is advanced at *v_pull*
    toward *r0* and clamped once it reaches the target.  The simulation
    stops early when the CV crosses ``r0 + rcut``.  The final structure
    is always written to ``init.pdb``.

    Parameters
    ----------
    system : openmm.System
        System object; the pulling force is added in place.
    sim : openmm.app.Simulation
        Running simulation whose context is reinitialized after the force
        is added.
    group1 : list[int]
        Atom indices for the first centroid group (e.g. the protein).
    group2 : list[int]
        Atom indices for the second centroid group (e.g. the ligand).
    r0 : float
        Target COM–COM distance / stopping criterion (nanometers).
    rcut : float, optional
        Tolerance around *r0* used as the stopping criterion; SMD stops
        when ``cv <= r0 + rcut``.  Default is ``0.5``.
    fc_pull : float, optional
        Harmonic force constant (kJ mol⁻¹ nm⁻²).
        Default is ``1000.0``.
    v_pull : float, optional
        Pulling velocity (nm ps⁻¹).  Default is ``0.01``.
    total_steps : int, optional
        Maximum integration steps.  Default is ``100000``.
    increment_steps : int, optional
        Steps between CV checks and anchor updates.  Default is ``2``.

    Returns
    -------
    None
        Writes the final structure to ``init.pdb``.

    Raises
    ------
    ValueError
        If *r0* is negative.

    Notes
    -----
    A stall-detection heuristic warns if the CV moves less than 5 % of
    the expected displacement over a sliding window of
    ``20000 // increment_steps`` increments.  This usually indicates a
    conflicting force on the system.
    """
    if float(r0) < 0:
        raise ValueError(f'r0 must be >= 0 nm, got {r0}')

    cv = CustomCentroidBondForce(2, 'r; r = distance(g1, g2)')
    grp1, grp2 = cv.addGroup(group1), cv.addGroup(group2)
    cv.addBond([grp1, grp2])

    fc_pull = fc_pull * kilojoule_per_mole / (unit.nanometers ** 2)
    v_pull  = v_pull  * unit.nanometers / unit.picosecond
    dt      = sim.integrator.getStepSize()
    r0_nm   = float(r0)

    pullingForce = CustomCVForce('0.5*fc_pull*(cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0_nm * unit.nanometers)
    pullingForce.addCollectiveVariable('cv', cv)
    force_index = system.addForce(pullingForce)
    sim.context.reinitialize(preserveState=True)

    current_cv_value = pullingForce.getCollectiveVariableValues(sim.context)[0]
    print(f'SMD start: r = {current_cv_value:.4f} nm  |  target r0 = {r0_nm:.4f} nm  '
          f'|  force_index = {force_index}')

    if current_cv_value <= r0_nm:
        print(f'CV already at target ({current_cv_value:.4f} nm <= {r0_nm:.4f} nm). Skipping SMD.')
    else:
        anchor_nm  = current_cv_value
        sim.context.setParameter('r0', anchor_nm * unit.nanometers)

        step_displacement_nm = (v_pull * dt * increment_steps).value_in_unit(unit.nanometers)
        log_every            = max(1, 5000  // increment_steps)
        stall_check_every    = max(1, 20000 // increment_steps)
        stall_threshold      = step_displacement_nm * stall_check_every * 0.05
        cv_at_last_check     = current_cv_value
        reached              = False

        for i in range(total_steps // increment_steps):
            sim.step(increment_steps)
            current_cv_value = pullingForce.getCollectiveVariableValues(sim.context)[0]

            anchor_nm = max(anchor_nm - step_displacement_nm, r0_nm)
            sim.context.setParameter('r0', anchor_nm * unit.nanometers)

            if i % log_every == 0:
                print(f'step {i * increment_steps:>8d}  |  '
                      f'r = {current_cv_value:.4f} nm  |  anchor = {anchor_nm:.4f} nm')

            if i > 0 and i % stall_check_every == 0:
                cv_change = abs(cv_at_last_check - current_cv_value)
                if cv_change < stall_threshold:
                    print(f'WARNING: CV moved only {cv_change:.6f} nm over the last '
                          f'{stall_check_every * increment_steps} steps '
                          f'(expected ~{step_displacement_nm * stall_check_every:.4f} nm). '
                          f'Check for conflicting forces via system.getForces().')
                cv_at_last_check = current_cv_value

            if current_cv_value <= (r0_nm + rcut):
                print(f'Target reached: r = {current_cv_value:.4f} nm '
                      f'at step {i * increment_steps}')
                reached = True
                break

        if not reached:
            print(f'WARNING: target r0 not reached after {total_steps} steps. '
                  f'Final r = {current_cv_value:.4f} nm.')

    state  = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
    with open('init.pdb', 'w') as outfile:
        PDBFile.writeFile(sim.topology, state.getPositions(), outfile)


def US_create_windows(system, sim, group1, group2, r0, r1, window_num,
                       fc_pull=1000.0, v_pull=0.01,
                       total_steps=100000, increment_steps=10):
    """Generate umbrella sampling window structures via SMD.

    A COM-distance CV is defined between *group1* and *group2*.  A
    harmonic bias sweeps the anchor from *r0* to *r1* at constant
    velocity *v_pull*.  Whenever the instantaneous CV crosses the next
    window target, the current coordinates are saved as
    ``window_<i>.pdb``.

    Parameters
    ----------
    system : openmm.System
        System object; the pulling force is added in place.
    sim : openmm.app.Simulation
        Running simulation whose context is reinitialized after the force
        is added.
    group1 : list[int]
        Atom indices for the first centroid group (e.g. the protein).
    group2 : list[int]
        Atom indices for the second centroid group (e.g. the ligand).
    r0 : float
        Starting CV value / initial anchor position (nanometers).
    r1 : float
        Ending CV value; the furthest window target (nanometers).
    window_num : int
        Number of evenly spaced windows between *r0* and *r1* inclusive.
    fc_pull : float, optional
        Harmonic force constant (kJ mol⁻¹ nm⁻²).
        Default is ``1000.0``.
    v_pull : float, optional
        Pulling velocity (nm ps⁻¹).  Default is ``0.01``.
    total_steps : int, optional
        Total integration steps during SMD.  Default is ``100000``.
    increment_steps : int, optional
        Steps between CV checks and anchor updates.  Default is ``10``.

    Returns
    -------
    windows : numpy.ndarray, shape (window_num,)
        CV target values (nanometers) for each window.

    Notes
    -----
    * The pulling force is permanently added to *system*.  To remove it
      afterward, iterate ``system.getForces()`` to locate the force by
      type and call ``system.removeForce(index)`` with its index.
    * Coordinates are saved with ``enforcePeriodicBox=False`` to keep
      molecules whole across PBC boundaries.
    * Fewer than *window_num* PDB files are written if the simulation
      ends before all targets are reached; a warning is printed in that
      case.
    """
    print('#create windows:')

    cv = CustomCentroidBondForce(2, 'r; r = distance(g1, g2)')
    grp1, grp2 = cv.addGroup(group1), cv.addGroup(group2)
    cv.addBond([grp1, grp2])

    fc_pull = fc_pull * kilojoule_per_mole / (unit.nanometers ** 2)
    v_pull  = v_pull  * unit.nanometers / unit.picosecond
    dt      = sim.integrator.getStepSize()

    r0_nm  = float(r0)
    r1_nm  = float(r1)
    r0_qty = r0_nm * unit.nanometers

    pullingForce = CustomCVForce('0.5*fc_pull*(cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0_qty)
    pullingForce.addCollectiveVariable('cv', cv)
    system.addForce(pullingForce)
    sim.context.reinitialize(preserveState=True)

    windows       = np.linspace(r0_nm, r1_nm, window_num)
    window_coords = []
    window_index  = 0

    print("SMD pulling", pullingForce.getCollectiveVariableValues(sim.context))
    log_every = max(1, 5000 // increment_steps)

    for i in range(total_steps // increment_steps):
        sim.step(increment_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(sim.context)[0]

        if i % log_every == 0:
            print(f'r0 = {r0_nm:.4f} nm  |  r = {current_cv_value:.4f} nm')

        step_displacement = v_pull * dt * increment_steps
        r0_nm  += step_displacement.value_in_unit(unit.nanometers)
        r0_qty += step_displacement
        sim.context.setParameter('r0', r0_qty)

        if window_index < len(windows) and current_cv_value >= windows[window_index]:
            print(f'saving window {window_index}: target={windows[window_index]:.4f} nm  '
                  f'r={current_cv_value:.4f} nm')
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
            window_coords.append(state.getPositions())
            window_index += 1

        if window_index >= window_num:
            print('All windows captured.')
            break

    if window_index < window_num:
        print(f'WARNING: only {window_index}/{window_num} windows captured. '
              f'Consider increasing total_steps or decreasing v_pull.')

    for i, coords in enumerate(window_coords):
        with open(f'window_{i}.pdb', 'w') as outfile:
            PDBFile.writeFile(sim.topology, coords, outfile)

    return windows


def US_collect_cv(system, sim, group1, group2, window_index, windows,
                   fc_pull=300.0, total_steps=20000000, record_steps=1000):
    """Run umbrella sampling for one window and record the CV time series.

    Applies a harmonic bias at the target CV for *window_index* and
    writes the CV values to ``cv_values_window_{window_index}.txt``.

    Parameters
    ----------
    system : openmm.System
        System object; the bias force is added in place.
    sim : openmm.app.Simulation
        Running simulation whose context is reinitialized after the force
        is added.
    group1 : list[int]
        Atom indices for the first centroid group (e.g. the protein).
    group2 : list[int]
        Atom indices for the second centroid group (e.g. the ligand).
    window_index : int
        Index into *windows* specifying which window to sample.
    windows : numpy.ndarray
        Array of window target CV values (nanometers) as returned by
        :func:`US_define_windows` or :func:`US_create_windows`.
    fc_pull : float, optional
        Harmonic force constant (kJ mol⁻¹ nm⁻²).
        Default is ``300.0``.
    total_steps : int, optional
        Total number of production integration steps.  Default is
        ``20000000``.
    record_steps : int, optional
        Steps between CV recordings.  Default is ``1000``.

    Returns
    -------
    None
        Writes ``cv_values_window_{window_index}.txt`` with two columns:
        step index and instantaneous CV value (nanometers).
    """
    print("define the CV:")
    cv = CustomCentroidBondForce(2, 'r; r = distance(g1, g2)')
    grp1, grp2 = cv.addGroup(group1), cv.addGroup(group2)
    cv.addBond([grp1, grp2])

    fc_pull = fc_pull * kilojoule_per_mole / (unit.nanometers ** 2)

    print('running window', window_index)
    r0 = windows[window_index] * unit.nanometers

    pullingForce = CustomCVForce('0.5*fc_pull*(cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0)
    pullingForce.addCollectiveVariable('cv', cv)
    system.addForce(pullingForce)
    sim.context.reinitialize(preserveState=True)
    sim.context.setParameter('r0', r0)
    sim.step(1000)

    cv_values = []
    for i in range(total_steps // record_steps):
        sim.step(record_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(sim.context)
        cv_values.append([i, current_cv_value[0]])

    np.savetxt(f'cv_values_window_{window_index}.txt', np.array(cv_values))
    print('Completed window', window_index)


def US_gen_metafile(windows, fc_pull=300.0, metafile='metafile.txt'):
    """Write a WHAM metafile from the window targets and force constant.

    Parameters
    ----------
    windows : numpy.ndarray
        Array of window target CV values (nanometers) as returned by
        :func:`US_define_windows` or :func:`US_create_windows`.
    fc_pull : float, optional
        Harmonic force constant used during production sampling
        (kJ mol⁻¹ nm⁻²).  Default is ``300.0``.
    metafile : str, optional
        Output filename.  Default is ``'metafile.txt'``.

    Returns
    -------
    None
        Writes *metafile* with one line per window formatted as::

            cv_values_window_{i}.txt   {window_target}   {fc_pull}
    """
    lines = [f'cv_values_window_{i}.txt {windows[i]} {fc_pull}\n'
             for i in range(len(windows))]
    with open(metafile, 'w') as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# WHAM
# ---------------------------------------------------------------------------

def wham():
    """Command-line entry point for the ``gfwham`` script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a WHAM metafile and compute the PMF from "
                    "umbrella sampling CV trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument('metafile', help='Path to the metafile (columns: cv_file  restraint_center  force_constant)')
    parser.add_argument('min', type=float, help='Lower CV bound for the PMF histogram')
    parser.add_argument('max', type=float, help='Upper CV bound for the PMF histogram')
    parser.add_argument('window_num', type=int, help='Number of umbrella sampling windows')
    parser.add_argument('fc_pull', type=float, help='Harmonic force constant (kJ mol-1 nm-2)')
    
    parser.add_argument('--temp', type=float, default=298.0, help='Simulation temperature in Kelvin')
    parser.add_argument('--bins', type=int, default=50, help='Number of PMF histogram bins')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for WHAM convergence')
    parser.add_argument('--pmf', default='pmf.txt', help='Output file for the PMF')
    parser.add_argument('--no-gen-metafile', action='store_true', help='Skip metafile generation (use existing metafile)')

    parser.add_argument('--MC', type=int, default=1000, help='Number of Monte Carlo steps')
    parser.add_argument('--seed', type=int, default=12345,  help='random seed for Monte Carlo sampling')

    args = parser.parse_args()

    if not args.no_gen_metafile:
        windows = US_define_windows(r0=args.pmf_min, r1=args.pmf_max, window_num=args.window_num)
        US_gen_metafile(windows, fc_pull=args.fc_pull, metafile=args.metafile)

    if args.MC > 0:
        os.system(f"wham {args.min} {args.max} {args.window_num} {args.tol} {args.temp} 0 {args.metafile} {args.pmf} {args.MC} {args.seed}")
    else:
        os.system(f"wham {args.min} {args.max} {args.window_num} {args.tol} {args.temp} 0 {args.metafile} {args.pmf}")


if __name__ == '__main__':
    wham()