import pkg_resources as pkg_res
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np
from .HyresFF import *


def load_ff(model='protein'):
    if model == 'protein':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres_GPU.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres_GPU.inp")
    elif model == 'protein_mix':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres_mix.inp")
    elif model == 'RNA':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA.inp")
    elif model == 'RNA_mix':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA_mix.inp")
    elif model == 'ATP':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_ATP.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_ATP.inp")
    else:
        print("Error: The model type {} is not supported, choose from hyres4, protein, protein_mix, RNA, RNA2, RNA_mix, ATP.".format(model))
        exit(1)
    top_inp, param_inp = str(path1), str(path2)

    return top_inp, param_inp

def nMg2lmd(cMg, T, F=0.0, M=0.0, n=0.0, RNA='rA'):
    if RNA == 'rA':
        F, M, n = 0.54, 0.94, 0.59
    elif RNA == 'rU':
        F, M, n = 0.48, 1.31, 0.85
    elif RNA == 'CAG':
        F, M, n = 0.53, 0.68, 0.28
    elif RNA == 'custom':
        if M == 0.0:
            print("Error: Please give F_Mg, M_1/2, and n if the RNA is custom type")
            exit(1)
    else:
        print("Error: Only rA, rU, CAG, and custom are supported RNA")
        exit(1)
    
    if cMg == 0.0:
        lmd = 0.0
    else:
        nMg = F*(cMg/M)**n/(1+(cMg/M)**n)
        nMg_T = nMg + 0.0012*(T-273-30)
        lmd = 1.265*(nMg_T/0.172)**0.625/(1+(nMg_T/0.172)**0.625)
    
    return lmd

# calculate relative dielectric constant at temperature T in K
def cal_er(T):
    Td = T-273
    er_t = 87.74-0.4008*Td+9.398*10**(-4)*Td**2-1.41*10**(-6)*Td**3
    return er_t

# calculate Debye-Huckel screening length in nm
def cal_dh(c_ion, T):
    NA = 6.02214076e23          # Avogadro's number
    er = cal_er(T)
    lB = 16710/(er*T)          # Bjerrum length in nm, 16710 = e^2/(4*pi*epsilon_0*k_B) in unit of nm*K
    dh = np.sqrt(1/(8*np.pi*lB*NA*1e-24*c_ion))   # Debye-Huckel screening length in nm
    return dh*unit.nanometer

def setup(model, args, dt, pressure=1*unit.atmosphere, friction=0.1/unit.picosecond, gpu_id="0"):
    """
    Set up the simulation system with given parameters.
    Parameters:
    -----------
    model: str
        The type of model to use ('protein', 'RNA', 'mix').
    args: argparse.Namespace
        The command line arguments containing simulation parameters.
    dt: float
        The time step for the integrator.
    pressure: unit.Quantity
        The pressure for the MonteCarloBarostat (default is 1 atm).
    friction: unit.Quantity
        The friction coefficient for the Langevin integrator (default is 0.1 / ps).
    gpu_id: str
        The GPU device index to use (default is "0").
    Returns:
    --------
    system: openmm.System
        The constructed OpenMM system.
    sim: openmm.app.Simulation
        The OpenMM simulation object.
    """

    print('\n################## set up simulation parameters ###################')
    # 1. input parameters
    pdb_file = args.pdb
    psf_file = args.psf
    T = args.temp
    c_ion = args.salt/1000.0                                   # concentration of ions in M
    c_Mg = args.Mg                                           # concentration of Mg in mM
    ensemble = args.ens
    
    # 2. set pbc and box vector
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
    
    # 3. force field parameters
    temperture = T*unit.kelvin 
    er_t = cal_er(T)                                                   # relative electric constant
    dh = cal_dh(c_ion, T)                                            # Debye-Huckel screening length in nm
    # Mg-P interaction
    lmd = nMg2lmd(c_Mg, T, RNA='rA')
    ffs = {
        'temp': T,                                                  # Temperature
        'lmd': lmd,                                                  # Charge scaling factor of P-
        'dh': dh,                                                  # Debye Huckel screening length
        'ke': 138.935456,                                           # Coulomb constant, ONE_4PI_EPS0
        'er': er_t,                                                  # relative dielectric constant
    }

    # 4. load force field files
    if model == 'protein':
        top_pro, param_pro = load_ff('protein')
        params = CharmmParameterSet(top_pro, param_pro)
    elif model == 'RNA':
        top_RNA, param_RNA = load_ff('RNA')
        params = CharmmParameterSet(top_RNA, param_RNA)
    elif model == 'mix':
        top_RNA, param_RNA = load_ff('RNA_mix')
        top_pro, param_pro = load_ff('protein_mix')
        params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)
    else:
        print("Error: Only 'protein', 'RNA', and 'mix' models are supported.")
        exit(1)

    print('\n################## load coordinates and topology ###################')
    # 5. import coordinates and topology form charmm pdb and psf
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

    # 6. construct force field
    print('\n################## build system ###################')
    if model == 'protein':
        system = HyresSystem(psf, system, ffs)
        print('This is a Hyres protein system')
    elif model == 'RNA':
        system = iConRNASystem(psf, system, ffs)
        print('This is a iConRNA system')
    elif model == 'mix':
        system = MixSystem(psf, system, ffs)
        print('This is a Hyres protein + iConRNA mixed system')
    else:
        print("Error: Only 'protein', 'RNA', and 'mix' models are supported. The input value is {}.".format(model))
        exit(1)

    # 7. set simulation
    print('\n################### prepare simulation ####################')
    if ensemble == 'NPT':
        print('This is a NPT system')
        system.addForce(MonteCarloBarostat(pressure, temperture, 25))
    elif ensemble == 'NVT':
        print('This is a NVT system')
    elif ensemble == 'non':
        print('This is a non-periodic system')
    else:
        print("Error: The ensemble must be NPT, NVT or non. The input value is {}.".format(ensemble))
        exit(1)

    integrator = LangevinMiddleIntegrator(temperture, friction, dt)
    plat = Platform.getPlatformByName('CUDA')
    prop = {'Precision': 'mixed', 'DeviceIndex': gpu_id}
    sim = Simulation(top, system, integrator, plat, prop)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperture)
    print(f'Langevin, CUDA, {temperture}')
    return system, sim
