from __future__ import division, print_function
from HyresBuilder import HyresFF, utils
# OpenMM Imports
from openmm.unit import *
from openmm.app import *
from openmm import *


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
# utils.setup(model, params, dt, pressure, friction, gpu_id)
# model: select from 'protein', 'RNA', 'mix'
# default set: pressure = 1*unit.atmosphere, friction = 0.1/unit.picosecond, gpu_id = "0"
system, sim = utils.setup(model='mix', params=params, dt=dt_equil)

with open('system.xml', 'w') as output:
    output.write(XmlSerializer.serialize(system))

print('\n# Now, the system has:')
for force in system.getForces():
    print('      ', force.getName())

print('\n################### Minimization, Equilibriation, Production simulation ####################')
print('# minimizeEnergy:')
print('before: ', sim.context.getState(getEnergy=True).getPotentialEnergy())
sim.minimizeEnergy(maxIterations=500000, tolerance=0.01)
print('after: ', sim.context.getState(getEnergy=True).getPotentialEnergy())

print('\n# Equilibriation running:')
sim.step(equil_step)

## save a pdb traj using large step, dcd traj using small step, and log file
sim.reporters.append(PDBReporter('system.pdb', pdb_freq))
sim.reporters.append(XTCReporter('system.xtc', traj_freq))
sim.reporters.append(StateDataReporter('system.log', log_freq, progress=True, totalSteps=prod_step, temperature=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True))
#simulation.reporters.append(CheckpointReporter('system.chk', dcd_freq*10))

print('\n# Production simulation running:')
sim.integrator.setStepSize(dt_prod)
sim.step(prod_step)

sim.saveCheckpoint('system.chk')
print('\n# Finished!')
