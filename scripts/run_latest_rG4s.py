"""
Date: Sep 29, 2025
Modified: Oct 22, 2026
Latest running script for rG4s-related simulations, with optional Mg2+ and Ca2+

Mg2+ and Ca2+ must be present as explicit ions (MG+ / CA+ residues) in the PDB/PSF,
e.g. added with packmol. Both share one P-ion scaling factor lmd, computed from the total
divalent concentration with utils.nMg2lmd (Mg2+ calibration), unless --lmd is given.

example: python run_latest_rG4s.py -c conf.pdb -p conf.psf -o g4 -t 298 -e NVT -b 15 -s 150 -m 2 --Ca 1
Author: Shanlong Li
email: shanlongli@umass.edu
"""

from __future__ import division, print_function
import argparse
from HyresBuilder import utils, Rigid
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
parser.add_argument('-m', "--Mg", default=0.0, type=float, help="Mg2+ concentration in mM, default is 0")
parser.add_argument("--Ca", default=0.0, type=float, help="Ca2+ concentration in mM, default is 0")
parser.add_argument("--rna", default='rA', choices=['rA', 'rU', 'CAG'], help="RNA type for the preset Mg2+ binding parameters of utils.nMg2lmd, default is rA")
parser.add_argument("--lmd", default=None, type=float, help="P-Mg2+/Ca2+ scaling factor; if given, it is used instead of the value from --Mg/--Ca")
parser.add_argument('-g', "--GG", default=3.0, type=float, help="G-G pair interaction strength in kcal/mol, default is 3.0 kcal/mol")
params = parser.parse_args()
out = params.out

# simulation parameters
dt_equil = 0.0001*unit.picoseconds		                      # time step for equilibration, for bad configuration, use 0.0001 ps
dt_prod = 0.004*unit.picoseconds                                # time step for production simulation
prod_step = 250000000                                           # production steps
equil_step = 10000                                              # equilibration steps
log_freq = 1250                                                 # frequency of log file
traj_freq = 5000                                                # frequency of trajectory file
pdb_freq = 12500000                                             # frequency of pdb_traj file
chk_freq = 125000                                               # frequency of checkpoint file

params.dt = dt_equil
params.pressure = 1*unit.atmosphere                             # pressure in NPT
params.friction = 0.1/unit.picosecond                           # friction coefficient in Langevin
params.er_ref = 60.0                                            # dielectric constant
params.gpu_id = "0"                                             # gpu_id used for simulation

# special parameters: P-Mg2+/Ca2+ scaling factor lmd, shared by both ions
c_div = params.Mg + params.Ca                                   # total divalent concentration in mM
if params.lmd is not None:
    lmd = params.lmd
elif c_div > 0:
    lmd = utils.nMg2lmd(c_div, params.temp, RNA=params.rna)       # Ca2+ uses the Mg2+ calibration
else:
    lmd = 0.0                                                    # no Mg2+/Ca2+
params.lmd = lmd
"""
for a rough estimation instead of the preset RNA types:
    lmd = utils.estimate_lmd(params.salt, c_div, RNA_length, RNA_Rg, params.temp)
"""

# check the explicit ions in the PSF against the concentrations and box volume
n_ion = {'MG': 0, 'CAL': 0}
for atom in CharmmPsfFile(params.psf).topology.atoms():
    if atom.name in n_ion:
        n_ion[atom.name] += 1
print(f'\n# Divalent ions: Mg2+ {params.Mg} mM ({n_ion["MG"]} in PSF), Ca2+ {params.Ca} mM ({n_ion["CAL"]} in PSF), lmd = {lmd:.3f}')
if params.ens == 'non' and (n_ion['MG'] or n_ion['CAL'] or lmd != 0):
    raise SystemExit('Error: Mg2+/Ca2+ need a periodic system (NVT or NPT).')
if params.ens in ('NVT', 'NPT'):
    box = params.box * 3 if len(params.box) == 1 else params.box
    volume = box[0] * box[1] * box[2]                               # nm^3
    for name, conc in (('MG', params.Mg), ('CAL', params.Ca)):
        expected = conc * volume * 6.02214076e-4                      # mM * nm^3 -> number of ions
        if abs(n_ion[name] - expected) > max(1.0, 0.1 * expected):
            print(f'  WARNING: {name}: {n_ion[name]} ions in the PSF, {expected:.1f} expected from {conc} mM in {volume:.0f} nm^3')
if lmd != 0 and not (n_ion['MG'] or n_ion['CAL']):
    print('  WARNING: lmd != 0 but the PSF has no MG/CAL ions; lmd has nothing to act on.')
if lmd == 0 and (n_ion['MG'] or n_ion['CAL']):
    print('  WARNING: the PSF has MG/CAL ions but lmd = 0, so their electrostatics with phosphate are off.')

### set up system and simulation
"""
utils.setup(params, GG=params.GG, modification=modification)
modification: custom function object
if further modify the OpenMM system, define all the changes as one function
example:
    def mod(system):
        system.addForce(customforce)
    util.setup(params, modification=mod)
"""
system, sim = utils.rG4s_setup(params, GG=params.GG)

"""
if further modify the system, add this line below:
sim.context.reinitialize(preserveState=True)
"""

# print out the xml file of the system
with open(f'{out}.xml', 'w') as output:
    output.write(XmlSerializer.serialize(system))

print('\n# Now, the system has:')
for force in system.getForces():
    print(f'      ForceName: {force.getName():<20}    ForceGroup: {force.getForceGroup():<20}')

################### Minimization, Equilibriation, Production simulation ####################'
print('\n# Minimization running:')
print('Potential energy before: ', sim.context.getState(getEnergy=True).getPotentialEnergy())
sim.minimizeEnergy(maxIterations=500000, tolerance=0.01)
print('Potential energy after: ', sim.context.getState(getEnergy=True).getPotentialEnergy())

print('\n# Equilibration running:')
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
