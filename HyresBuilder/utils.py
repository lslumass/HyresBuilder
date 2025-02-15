import pkg_resources as pkg_res
import argparse
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np
from .HyresFF import MixSystem


def load_ff(model='protein'):
    if model == 'protein':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres_GPU.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres_GPU.inp")
    elif model == 'hyres4':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres4.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres4.inp")
    elif model == 'protein_mix':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres_mix.inp")
    elif model == 'RNA':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA.inp")
    elif model == 'RNA2':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA2.inp")
    elif model == 'RNA_mix':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA_mix.inp")
    elif model == 'ATP':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_ATP.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_ATP.inp")
    else:
        print("Error: The model type {} is not supported, choose from hyres4, protein, protein_mix, RNA, RNA2, RNA_mix, ATP.".format(model))
        exit(1)
    top_inp = str(path1)
    param_inp = str(path2)

    return top_inp, param_inp

def setup(params, gpu_id, pressure, friction, dt):
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--pdb", default='conf.pdb', help="pdb file, default is conf.pdb")
    parser.add_argument('-p', "--psf", default='conf.psf', help="psf file, default is conf.psf")
    parser.add_argument('-t', "--temp", default=303, type=float, help="system temperature, default is 303 K")
    parser.add_argument('-b', "--box", nargs='+', type=float, help="box dimensions in nanometer, e.g., '50 50 50' ")
    parser.add_argument('-s', "--salt", default=150.0, type=float, help="salt concentration in mM, default is 0.0 mM")
    parser.add_argument('-e', "--ens", default='NVT', type=str, help="simulation ensemble, NPT, NVT, or non, non is for non-periodic system")
    parser.add_argument('-m', "--Mg", default=0.0, type=float, help="Mg2+ concentration in mM")
    args = parser.parse_args()

    pdb_file = args.pdb
    psf_file = args.psf
    T = args.temp
    c_ion = args.salt/1000.0                                   # concentration of ions in M
    c_Mg = args.Mg                                           # concentration of Mg in mM
    ensemble = args.ens

    ## set pbc and box vector
    if ensemble == 'non' and c_Mg != 0.0:
        print("Error: Mg ion cannot be usde in non-periodic system.")
        exit(1)
    if ensemble in ['NPT', 'NVT']:
        # pbc box length
        if len(args.box) == 1:
            lx, ly, lz = args.box[0], args.box[0], args.box[0]
        elif len(args.box) == 3:
            lx = args.box[0]
            ly = args.box[1]
            lz = args.box[2]
        else:
            print("Error: You must provide either one or three values for box.")
            exit(1)
        a = Vec3(lx, 0.0, 0.0)
        b = Vec3(0.0, ly, 0.0)
        c = Vec3(0.0, 0.0, lz)
    elif ensemble not in ['NPT', 'NVT', 'non']:
        print("Error: The ensemble must be NPT, NVT or non. The input value is {}.".format(ensemble))
        exit(1)

    # force field parameters
    Td = T-273
    temperture = T*unit.kelvin 
    er_t = 87.74-0.4008*Td+9.398*10**(-4)*Td**2-1.41*10**(-6)*Td**3
    print('relative electric constant: ', er_t*20.3/77.6)                        
    dh = 0.304/(np.sqrt(c_ion))
    print('Debye-Huckel screening length: ', dh)
    if c_Mg == 0:
        nMg = 0
        lmd0 = 0
    else:
        nMg = 0.526*(c_Mg/0.680)**(0.283)/(1+(c_Mg/0.680)**(0.283)) + 0.0012*(Td-30)                                                       
        lmd0 = 1.265*(nMg/0.172)**0.625/(1+(nMg/0.172)**0.625)
        print('lmd: ', lmd0)
    ffs = {
        'temp': T,                                                  # Temperature
        'lmd': lmd0,                                                # Charge scaling factor of P-
        'dh': dh*unit.nanometer,                                  # Debye Huckel screening length
        'ke': 138.935456,                                           # Coulomb constant, ONE_4PI_EPS0
        'er': er_t*20.3/77.6,                                         # relative dielectric constant
        'eps_hb': 1.8*unit.kilocalorie_per_mole,                    # hydrogen bond strength
        'sigma_hb': 0.29*unit.nanometer,                            # sigma of hydrogen bond
        'eps_base': 2.05*unit.kilocalorie_per_mole,                 # base stacking strength
    }

    # 1) import coordinates and topology form charmm pdb and psf
    print('\n################## load coordinates, topology and parameters ###################')
    pdb = PDBFile(pdb_file)
    psf = CharmmPsfFile(psf_file)
    top = psf.topology
    if ensemble == 'non':
        system = psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds)
    else:
        psf.setBox(lx, ly, lz)
        top.setPeriodicBoxVectors((a, b, c))
        top.setUnitCellDimensions((lx, ly,lz))
        system = psf.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds)
        system.setDefaultPeriodicBoxVectors(a, b, c)

    system = MixSystem(psf, system, ffs)
    # set simulation
    print('\n################### prepare simulation system####################')
    if ensemble == 'NPT':
        print('This is a NPT system')
        system.addForce(MonteCarloBarostat(pressure, temperture, 25))
    elif ensemble == 'NVT':
        print('This is a NVT system')
    elif ensemble == 'non':
        print('This is a non-periodic system')
    integrator = LangevinMiddleIntegrator(temperture, friction, dt)
    plat = Platform.getPlatformByName('CUDA')
    prop = {'Precision': 'mixed', 'DeviceIndex': gpu_id}
    sim = Simulation(top, system, integrator, plat, prop)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperture)
    print(f'Langevin, CUDA, {temperture}')

    return psf, system, ffs, sim
