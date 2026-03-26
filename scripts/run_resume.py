"""
Date: Mar 21, 2026
Latest running script for HyRes and iConRNA simulation - RESUME MODE
Author: Shanlong Li
email: shanlongli@umass.edu
"""

from __future__ import division, print_function
import argparse
import os
from HyresBuilder import utils
# OpenMM Imports
from openmm.unit import *
from openmm.app import *
from openmm import *


# input parameters
parser = argparse.ArgumentParser()
parser.add_argument('-c', "--pdb", default='conf.pdb', help="pdb file, default is conf.pdb")
parser.add_argument('-p', "--psf", default='conf.psf', help="psf file, default is conf.psf")
parser.add_argument('-o', "--out", default='system', type=str, help="the prefix name for the output files, including xtc, pdb, log, chk")
parser.add_argument('-t', "--temp", default=303, type=float, help="system temperature, default is 303 K")
parser.add_argument('-b', "--box", nargs='+', type=float, help="box dimensions in nanometer, e.g., '50 50 50' ")
parser.add_argument('-s', "--salt", default=150.0, type=float, help="salt concentration in mM, default is 150.0 mM")
parser.add_argument('-e', "--ens", default='NVT', type=str, help="simulation ensemble, NPT, NVT, or non, non is for non-periodic system")
parser.add_argument('-m', "--Mg", default=0.0, type=float, help="Mg2+ concentration in mM")
parser.add_argument('-r', "--chk", default=None, type=str, help="checkpoint file to resume from, e.g., 'system.chk'")
params = parser.parse_args()
out = params.out

# resolve checkpoint file path
chk_file = params.chk if params.chk else f'{out}.chk'
if not os.path.exists(chk_file):
    raise FileNotFoundError(f"Checkpoint file '{chk_file}' not found. Cannot resume.")

# simulation parameters
dt_prod = 0.008*unit.picoseconds                                # time step for production simulation
prod_step = 250000000                                           # production steps
log_freq = 1250                                                 # frequency of log file
traj_freq = 5000                                                # frequency of trajectory file
pdb_freq = 12500000                                             # frequency of pdb_traj file
chk_freq = 125000                                               # frequency of checkpoint file

params.dt = dt_prod                                             # resume directly at production step size
params.pressure = 1*unit.atmosphere
params.friction = 0.1/unit.picosecond
params.er_ref = 60.0
params.gpu_id = "0"

### set up system and simulation (rebuilds topology/forces, then loads state from checkpoint)
system, sim = utils.setup(params)

print(f'\n# Loading checkpoint from: {chk_file}')
sim.loadCheckpoint(chk_file)
print('# Checkpoint loaded. Resuming simulation...')
print('Potential energy: ', sim.context.getState(getEnergy=True).getPotentialEnergy())

print('\n# System forces:')
for force in system.getForces():
    print(f'      ForceName: {force.getName():<20}    ForceGroup: {force.getForceGroup():<20}')

# append reporters (trajectories and logs will continue from where they left off)
sim.reporters.append(PDBReporter(f'{out}.pdb', pdb_freq))
#sim.reporters.append(DCDReporter(f'{out}.dcd', traj_freq, append=True))   # append=True continues existing trajectory
sim.reporters.append(XTCReporter(f'{out}.xtc', traj_freq, append=True))   # append=True continues existing trajectory
sim.reporters.append(StateDataReporter(f'{out}.log', log_freq, progress=True, totalSteps=prod_step,
                                       step=True, temperature=True, totalEnergy=True, speed=True,
                                       append=True))                       # append=True continues existing log
sim.reporters.append(CheckpointReporter(f'{out}.chk', chk_freq))

print('\n# Resumed production simulation running:')
sim.integrator.setStepSize(dt_prod)
sim.step(prod_step)

sim.saveCheckpoint(f'{out}.chk')
print('\n# Finished!')