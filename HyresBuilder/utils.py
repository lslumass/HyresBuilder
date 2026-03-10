"""
| This module is used to load force field files and set up simulation.
"""

import pkg_resources as pkg_res
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np
from .HyresFF import *
from .rG4sFF import *


def load_ff(model):
    """
    Resolve and return the topology and parameter file paths for a given HyRes force field model.

    File paths are resolved from within the installed HyresBuilder package using
    pkg_resources, so no manual path management is needed.

    Args:
        model (str): Force field model type. Supported values:
            * 'Protein' : HyRes protein force field (top_hyres_mix / param_hyres_mix)
            * 'RNA'     : HyRes RNA force field     (top_RNA_mix  / param_RNA_mix)
            * 'DNA'     : HyRes DNA force field     (top_DNA_mix  / param_DNA_mix)
            * 'rG4s'    : RNA G-quadruplex model, uses RNA topology with a dedicated parameter file (param_rG4s)
            * 'ATP'     : ATP force field            (top_ATP      / param_ATP)

    Returns:
        tuple[str, str]:
            * top_inp   (str): Absolute path to the CHARMM topology (.inp) file.
            * param_inp (str): Absolute path to the CHARMM parameter (.inp) file.

    Raises:
        SystemExit: If an unsupported model name is provided.

    Example:
        >>> top, param = load_ff('Protein')
        >>> top, param = load_ff('RNA')
    """

    if model == 'Protein':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres_mix.inp")
    elif model == 'RNA':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA_mix.inp")
    elif model == 'DNA':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_DNA_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_DNA_mix.inp")
    elif model == 'rG4s':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_rG4s.inp")
    elif model == 'ATP':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_ATP.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_ATP.inp")
    else:
        print("Error: The model type {} is not supported, only for Portein, RNA, DNA, and, ATP.".format(model))
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

#def cal_dh(c_ion):
#    dh = 0.304/np.sqrt(c_ion)   # Debye-Huckel screening length in nm at room temperature
#    return dh*unit.nanometer

def setup(params, modification=None):
    """
    Construct and initialize a HyRes OpenMM simulation system from a parameter object.

    Performs the full setup pipeline in seven stages:
    1. Parse simulation parameters from the params object.
    2. Configure periodic boundary conditions (PBC) and box vectors.
    3. Compute force field parameters: dielectric constant, Debye–Hückel screening length, and Mg²⁺–RNA charge scaling factor (lambda).
    4. Load CHARMM topology and parameter files for Protein and RNA.
    5. Import coordinates (PDB) and topology (PSF).
    6. Build the HyRes custom force field via HyresFF.buildSystem.
    7. Attach the barostat (NPT) and initialize the Langevin integrator and CUDA simulation context.

    Args:
        params (argparse.Namespace or equivalent): Object with the following attributes:
            * pdb       (str):            Path to the input PDB coordinate file.
            * psf       (str):            Path to the CHARMM PSF topology file.
            * temp      (float):          Simulation temperature in Kelvin.
            * salt      (float):          Monovalent salt concentration in mM (converted to M internally).
            * Mg        (float):          Mg²⁺ concentration in mM.
            * ens       (str):            Ensemble type: 'NPT', 'NVT', or 'non' (non-periodic).
            * dt        (unit.Quantity):  Integration time step.
            * er_ref    (float):          Reference dielectric constant used to scale the temperature-dependent er.
            * pressure  (unit.Quantity):  Pressure for NPT barostat.
            * friction  (unit.Quantity):  Friction coefficient for the Langevin integrator.
            * gpu_id    (str):            CUDA device index (e.g. '0').
            * box       (list[float]):    PBC box dimensions in nm. Provide one value for a cubic box or three values [lx, ly, lz] for an orthorhombic box.
        modification (callable, optional): A user-supplied function passed directly to HyresFF.buildSystem for custom force field modifications. Default: None.

    Returns:
        tuple:
            * system (openmm.System):          The fully constructed OpenMM System.
            * sim    (openmm.app.Simulation):  The initialized Simulation object with positions and velocities set.

    Raises:
        SystemExit: If an unsupported ensemble type is provided, if a Mg²⁺ concentration is specified for a non-periodic system, or if an invalid box dimension list is given.

    Notes:
        * The dielectric constant is computed as er = er_t(T) * er_ref / 77.6, where
          er_t(T) is the temperature-dependent pure-water dielectric from cal_er.
        * The Mg²⁺–RNA lambda parameter is computed via nMg2lmd using the 'rA' RNA type.
        * The CUDA platform is used with mixed precision.

    Example:
        >>> system, sim = setup(params)
        >>> system, sim = setup(params, modification=my_custom_force_fn)
    """
    
    print('\n################## set up simulation parameters ###################')
    # 1. input parameters
    pdb_file = params.pdb
    psf_file = params.psf
    T = params.temp
    c_ion = params.salt/1000.0                                   # concentration of ions in M
    c_Mg = params.Mg                                           # concentration of Mg in mM
    ensemble = params.ens

    dt = params.dt
    er_ref = params.er_ref
    pressure = params.pressure
    friction = params.friction
    gpu_id = params.gpu_id
    
    # 2. set pbc and box vector
    if ensemble == 'non' and c_Mg != 0.0:
        print("Error: Mg ion cannot be usde in non-periodic system.")
        exit(1)
    if ensemble in ['NPT', 'NVT']:
        # pbc box length
        if len(params.box) == 1:
            lx, ly, lz = params.box[0], params.box[0], params.box[0]
        elif len(params.box) == 3:
            lx = params.box[0]
            ly = params.box[1]
            lz = params.box[2]
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
    cutoff = 1.2*unit.nanometer                                 # nonbonded cutoff
    d_switch = 1.1*unit.nanometer                               # switch function starting distance
    temperature = T*unit.kelvin 
    er_t = cal_er(T)                                                   # relative electric constant
    er = er_t*er_ref/77.6
    dh = cal_dh(c_ion, T)                                            # Debye-Huckel screening length in nm
    # Mg-P interaction
    lmd = nMg2lmd(c_Mg, T, RNA='rA')
    print(f"dielectric constant: er = {er:.2f}")
    print(f"Debye screening length: dh = {dh.value_in_unit(unit.nanometers):.2f} nm")
    print(f'Mg-RNA interaction: lmd = {lmd:.2f}')
    ffs = {
        'temp': T,                                                  # Temperature
        'lmd': lmd,                                                  # Charge scaling factor of P-
        'dh': dh,                                                  # Debye Huckel screening length
        'ke': 138.935456,                                           # Coulomb constant, ONE_4PI_EPS0
        'er': er,                                                  # relative dielectric constant
    }

    # 4. load force field files
    top_pro, param_pro = load_ff('Protein')
    top_RNA, param_RNA = load_ff('RNA')
    #top_DNA, param_DNA = load_ff('DNA')
    #top_ATP, param_ATP = load_ff('RNA')
    params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)

    print('\n################## load coordinates and topology ###################')
    # 5. import coordinates and topology form charmm pdb and psf
    pdb = PDBFile(pdb_file)
    psf = CharmmPsfFile(psf_file)
    top = psf.topology
    print(f"coordinate file: {pdb_file}")
    print(f"topology file: {psf_file}")

    print('\n################## create system ###################')
    if ensemble == 'non':
        system = psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
    else:
        psf.setBox(lx, ly, lz)
        top.setPeriodicBoxVectors((a, b, c))
        top.setUnitCellDimensions((lx, ly,lz))
        system = psf.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
        system.setDefaultPeriodicBoxVectors(a, b, c)
    
    print(f"nonbonded cutoff: {cutoff}")
    print(f"switch distance: {d_switch}")

    # 6. construct force field
    system = buildSystem(psf, system, ffs, modification=modification)
    print("HyresFF.buildSystem for custom force field")

    # 7. set simulation
    print('\n################### prepare simulation ####################')
    if ensemble == 'NPT':
        print('This is a NPT system')
        system.addForce(MonteCarloBarostat(pressure, temperature, 25))
    elif ensemble == 'NVT':
        print('This is a NVT system')
    elif ensemble == 'non':
        print('This is a non-periodic system')
    else:
        print("Error: The ensemble must be NPT, NVT or non. The input value is {}.".format(ensemble))
        exit(1)

    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    plat = Platform.getPlatformByName('CUDA')
    prop = {'Precision': 'mixed', 'DeviceIndex': gpu_id}
    sim = Simulation(top, system, integrator, plat, prop)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature)
    print(f'Langevin, CUDA, {temperature}')
    return system, sim


def setup2(args, dt, lmd=0, pressure=1*unit.atmosphere, friction=0.1/unit.picosecond, gpu_id="0"):
    """
    Set up the simulation system with given parameters.
    Parameters:
    -----------
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
    cutoff = 1.2*unit.nanometer                                 # nonbonded cutoff
    d_switch = 1.1*unit.nanometer                               # switch function starting distance
    temperature = T*unit.kelvin 
    er_t = cal_er(T)                                                   # relative electric constant
    er = er_t*60.0/77.6
    dh = cal_dh(c_ion, T)                                            # Debye-Huckel screening length in nm
    # Mg-P interaction
    lmd = args.Mg
    print(f'er: {er}, dh: {dh}, lmd: {lmd}')
    ffs = {
        'temp': T,                                                  # Temperature
        'lmd': lmd,                                                  # Charge scaling factor of P-
        'dh': dh,                                                  # Debye Huckel screening length
        'ke': 138.935456,                                           # Coulomb constant, ONE_4PI_EPS0
        'er': er,                                                  # relative dielectric constant
    }

    # 4. load force field files
    top_pro, param_pro = load_ff('Protein')
    top_RNA, param_RNA = load_ff('RNA')
    #top_DNA, param_DNA = load_ff('DNA')
    #top_ATP, param_ATP = load_ff('RNA')
    params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)

    print('\n################## load coordinates and topology ###################')
    # 5. import coordinates and topology form charmm pdb and psf
    pdb = PDBFile(pdb_file)
    psf = CharmmPsfFile(psf_file)
    top = psf.topology
    if ensemble == 'non':
        system = psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
    else:
        psf.setBox(lx, ly, lz)
        top.setPeriodicBoxVectors((a, b, c))
        top.setUnitCellDimensions((lx, ly,lz))
        system = psf.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
        system.setDefaultPeriodicBoxVectors(a, b, c)

    # 6. construct force field
    print('\n################## build system ###################')
    system = buildSystem(psf, system, ffs)

    # 7. set simulation
    print('\n################### prepare simulation ####################')
    if ensemble == 'NPT':
        print('This is a NPT system')
        system.addForce(MonteCarloBarostat(pressure, temperature, 25))
    elif ensemble == 'NVT':
        print('This is a NVT system')
    elif ensemble == 'non':
        print('This is a non-periodic system')
    else:
        print("Error: The ensemble must be NPT, NVT or non. The input value is {}.".format(ensemble))
        exit(1)

    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    plat = Platform.getPlatformByName('CUDA')
    prop = {'Precision': 'mixed', 'DeviceIndex': gpu_id}
    sim = Simulation(top, system, integrator, plat, prop)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature)
    print(f'Langevin, CUDA, {temperature}')
    return system, sim


def rG4s_setup(args, dt, pressure=1*unit.atmosphere, friction=0.1/unit.picosecond, gpu_id="0"):
    """
    Set up the rG4s simulation system with given parameters.
    Parameters:
    -----------
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
    cutoff = 1.2*unit.nanometer                                 # nonbonded cutoff
    d_switch = 1.1*unit.nanometer                               # switch function starting distance
    temperature = T*unit.kelvin 
    er_t = cal_er(T)                                                   # relative electric constant
    er = er_t*60.0/77.6
    dh = cal_dh(c_ion, T)                                            # Debye-Huckel screening length in nm
    # Mg-P interaction
    lmd = nMg2lmd(c_Mg, T, RNA='rA')
    print(f'er: {er}, dh: {dh}, lmd: {lmd}')
    ffs = {
        'temp': T,                                                  # Temperature
        'lmd': lmd,                                                  # Charge scaling factor of P-
        'dh': dh,                                                  # Debye Huckel screening length
        'ke': 138.935456,                                           # Coulomb constant, ONE_4PI_EPS0
        'er': er,                                                  # relative dielectric constant
        'ion_type': args.ion*unit.kilocalorie_per_mole,                                      # ion type, K or Na
    }

    # 4. load force field files
    top_RNA, param_RNA = load_ff('rG4s')
    top_pro, param_pro = load_ff('Protein')
    params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)

    print('\n################## load coordinates and topology ###################')
    # 5. import coordinates and topology form charmm pdb and psf
    pdb = PDBFile(pdb_file)
    psf = CharmmPsfFile(psf_file)
    top = psf.topology
    if ensemble == 'non':
        system = psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
    else:
        psf.setBox(lx, ly, lz)
        top.setPeriodicBoxVectors((a, b, c))
        top.setUnitCellDimensions((lx, ly,lz))
        system = psf.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
        system.setDefaultPeriodicBoxVectors(a, b, c)

    # 6. construct force field
    print('\n################## build system ###################')
    system = rG4sSystem(psf, system, ffs)

    # 7. set simulation
    print('\n################### prepare simulation ####################')
    if ensemble == 'NPT':
        print('This is a NPT system')
        system.addForce(MonteCarloBarostat(pressure, temperature, 25))
    elif ensemble == 'NVT':
        print('This is a NVT system')
    elif ensemble == 'non':
        print('This is a non-periodic system')
    else:
        print("Error: The ensemble must be NPT, NVT or non. The input value is {}.".format(ensemble))
        exit(1)

    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    plat = Platform.getPlatformByName('CUDA')
    prop = {'Precision': 'mixed', 'DeviceIndex': gpu_id}
    sim = Simulation(top, system, integrator, plat, prop)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature)
    print(f'Langevin, CUDA, {temperature}')
    return system, sim
