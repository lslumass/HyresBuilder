"""
Date: Feb 16, 2025
Latest running script for HyRes and iCon simulation
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
parser.add_argument('-t', "--temp", default=303, type=float, help="system temperature, default is 303 K")
parser.add_argument('-b', "--box", nargs='+', type=float, help="box dimensions in nanometer, e.g., '50 50 50' ")
parser.add_argument('-s', "--salt", default=150.0, type=float, help="salt concentration in mM, default is 150.0 mM")
parser.add_argument('-e', "--ens", default='NVT', type=str, help="simulation ensemble, NPT, NVT, or non, non is for non-periodic system")
parser.add_argument('-m', "--Mg", default=0.0, type=float, help="Mg2+ concentration in mM")

# 0) set variables in the simulation
gpu_id = "0"
top_RNA, param_RNA = utils.load_ff('RNA_mix')
top_pro, param_pro = utils.load_ff('protein_mix')
params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)

# simulation parameters
dt_equil = 0.0001*unit.picoseconds		                        # time step for equilibration
dt_prod = 0.008*unit.picoseconds                                # time step for production simulation
prod_step = 250000000                                           # production steps
equil_step = 10000                                              # equilibration steps
log_freq = 1000                                                 # frequency of log file
traj_freq = 5000                                                # frequency of trajectory file
pdb_freq = 12500000                                             # frequence of dpd_traj file
pressure = 1*unit.atmosphere                                    # pressure in NPT
friction = 0.1/unit.picosecond                                  # friction coefficient in Langevin

### set up system and simulation
# utils.setup(model, parser, params, dt, pressure, friction, gpu_id)
# model: select from 'protein', 'RNA', 'mix'
# default set: pressure = 1*unit.atmosphere, friction = 0.1/unit.picosecond, gpu_id = "0"
system, sim = utils.setup(model='mix', parser=parser, params=params, dt=dt_equil)

with open('system.xml', 'w') as output:
    output.write(XmlSerializer.serialize(system))

print('\n# Now, the system has:')
for force in system.getForces():
    print('      ', force.getName())

################### Minimization, Equilibriation, Production simulation ####################'
print('# minimizeEnergy:')
print('before: ', sim.context.getState(getEnergy=True).getPotentialEnergy())
sim.minimizeEnergy(maxIterations=500000, tolerance=0.01)
print('after: ', sim.context.getState(getEnergy=True).getPotentialEnergy())

print('\n# Equilibriation running:')
sim.step(equil_step)

## save a pdb traj using large step, traj using small step, and log file
sim.reporters.append(PDBReporter('system.pdb', pdb_freq))
sim.reporters.append(XTCReporter('system.xtc', traj_freq))
sim.reporters.append(StateDataReporter('system.log', log_freq, progress=True, totalSteps=prod_step, temperature=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True))
#simulation.reporters.append(CheckpointReporter('system.chk', dcd_freq*10))

print('\n# Production simulation running:')
sim.integrator.setStepSize(dt_prod)
sim.step(prod_step)

sim.saveCheckpoint('system.chk')
print('\n# Finished!')
