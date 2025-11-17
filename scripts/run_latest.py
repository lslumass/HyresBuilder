"""
Date: Sep 29, 2025
Latest running script for HyRes and iConRNA simulation
Aauthor: Shanlong Li
email: shanlongli@umass.edu
"""

from __future__ import division, print_function
import argparse
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
args = parser.parse_args()
out = args.out

# simulation parameters
dt_equil = 0.001*unit.picoseconds		                        # time step for equilibration, for bad configuration, use 0.0001 ps
dt_prod = 0.008*unit.picoseconds                                # time step for production simulation
prod_step = 250000000                                           # production steps
equil_step = 10000                                              # equilibration steps
log_freq = 1250                                                 # 10 ps: frequency of log file
traj_freq = 5000                                                # 40 ps: frequency of trajectory file
pdb_freq = 12500000                                             # 100 ns: frequence of dpd_traj file
chk_freq = 125000                                              # 1 ns: frequence of checkpoint file
pressure = 1*unit.atmosphere                                    # pressure in NPT
friction = 0.1/unit.picosecond                                  # friction coefficient in Langevin

### set up system and simulation
# utils.setup(model, parser, params, dt, pressure, friction, gpu_id)
# model: select from 'protein', 'RNA', 'mix'
# default set: pressure = 1*unit.atmosphere, friction = 0.1/unit.picosecond, gpu_id = "0"
system, sim = utils.setup(args=args, dt=dt_equil)

"""
if further modify the system, add this line below:
sim.context.reinitialize(preserveState=True)
"""

# print out the xml file of the system
with open(f'{out}.xml', 'w') as output:
    output.write(XmlSerializer.serialize(system))

print('\n# Now, the system has:')
for force in system.getForces():
    print('      ', force.getName())

################### Minimization, Equilibriation, Production simulation ####################'
print('# Minimization running:')
print('Potential energy before: ', sim.context.getState(getEnergy=True).getPotentialEnergy())
sim.minimizeEnergy(maxIterations=500000, tolerance=0.01)
print('Potential energy after: ', sim.context.getState(getEnergy=True).getPotentialEnergy())

print('\n# Equilibriation running:')
sim.step(equil_step)

## save a pdb traj using large step, xtc/dcd traj using small step, and log file
sim.reporters.append(PDBReporter(f'{out}.pdb', pdb_freq))
#sim.reporters.append(XTCReporter(f'{out}.xtc', traj_freq))      # xtc traj
sim.reporters.append(DCDReporter(f'{out}.dcd', traj_freq))      # dcd traj
sim.reporters.append(StateDataReporter(f'{out}.log', log_freq, progress=True, totalSteps=prod_step, step=True, temperature=True, totalEnergy=True, speed=True))
sim.reporters.append(CheckpointReporter(f'{out}.chk', chk_freq))

print('\n# Production simulation running:')
sim.integrator.setStepSize(dt_prod)
sim.step(prod_step)

sim.saveCheckpoint(f'{out}.chk')
print('\n# Finished!')
