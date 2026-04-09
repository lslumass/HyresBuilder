"""
| This module is used to load force field files and set up simulation.
"""

from importlib.resources import files
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np
from .HyresFF import *
from .rG4sFF import *


def load_ff(model: str) -> tuple[str, str]:
    """
    Return the topology and parameter file paths for a given force field model.

    File paths are resolved from within the installed HyresBuilder package using
    ``importlib.resources``, so no manual path management is needed regardless
    of where the package is installed.

    Args:
        model (str): Force field model name. Supported values:

                     - ``'Protein'`` — HyRes protein force field
                       (``top_hyres_mix`` / ``param_hyres_mix``)
                     - ``'RNA'`` — iConRNA force field
                       (``top_RNA_mix`` / ``param_RNA_mix``)
                     - ``'DNA'`` — iConDNA force field
                       (``top_DNA_mix`` / ``param_DNA_mix``)
                     - ``'rG4s'`` — RNA G-quadruplex model, uses RNA topology
                       with a dedicated parameter file (``param_rG4s``)
                     - ``'ATP'`` — ATP force field
                       (``top_ATP`` / ``param_ATP``)

    Returns:
        tuple[str, str]: A 2-tuple of absolute paths:

                         - ``top_inp``   — CHARMM topology (``.inp``) file.
                         - ``param_inp`` — CHARMM parameter (``.inp``) file.

    Raises:
        SystemExit: If an unsupported model name is provided.

    Example:
        >>> from HyresBuilder.utils import load_ff
        >>> top, param = load_ff('Protein')
        >>> top, param = load_ff('RNA')
        >>> top, param = load_ff('rG4s')
    """
    ff = files("HyresBuilder") / "forcefield"

    if model == 'Protein':
        path1 = ff / "top_hyres_mix.inp"
        path2 = ff / "param_hyres_mix.inp"
    elif model == 'RNA':
        path1 = ff / "top_RNA_mix.inp"
        path2 = ff / "param_RNA_mix.inp"
    elif model == 'DNA':
        path1 = ff / "top_DNA_mix.inp"
        path2 = ff / "param_DNA_mix.inp"
    elif model == 'rG4s':
        path1 = ff / "top_RNA_mix.inp"
        path2 = ff / "param_rG4s.inp"
    elif model == 'ATP':
        path1 = ff / "top_ATP.inp"
        path2 = ff / "param_ATP.inp"
    else:
        print("Error: The model type {} is not supported, only for Protein, RNA, DNA, rG4s, and ATP.".format(model))
        exit(1)

    top_inp, param_inp = path1.as_posix(), path2.as_posix()

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
    Build and initialize a complete HyRes/iConRNA OpenMM simulation system.

    Executes the full setup pipeline in seven stages:

    1. Parse simulation parameters from ``params``.
    2. Configure periodic boundary conditions (PBC) and box vectors.
    3. Compute force field parameters: temperature-dependent dielectric constant,
       Debye-Hückel screening length, and Mg²⁺-RNA charge scaling factor (lambda).
    4. Load CHARMM topology and parameter files for protein and RNA.
    5. Import coordinates (PDB) and topology (PSF).
    6. Build the HyRes custom force field via :func:`HyresFF.buildSystem`.
    7. Attach the barostat (NPT only), initialize the Langevin integrator, and
       create the CUDA simulation context with positions and velocities.

    Args:
        params (argparse.Namespace): Simulation parameter object with the
                                     following attributes:

                                     - ``pdb`` (str) — path to input PDB file.
                                     - ``psf`` (str) — path to CHARMM PSF file.
                                     - ``temp`` (float) — temperature in Kelvin.
                                     - ``salt`` (float) — NaCl concentration in mM.
                                     - ``Mg`` (float) — Mg²⁺ concentration in mM.
                                     - ``ens`` (str) — ensemble: ``'NPT'``,
                                       ``'NVT'``, or ``'non'`` (non-periodic).
                                     - ``box`` (list of float) — box dimensions
                                       in nm; one value for cubic, three for
                                       orthorhombic.
                                     - ``dt`` (Quantity) — integration time step.
                                     - ``er_ref`` (float) — reference dielectric
                                       used to scale the temperature-dependent er.
                                     - ``pressure`` (Quantity) — pressure for NPT
                                       barostat.
                                     - ``friction`` (Quantity) — Langevin friction
                                       coefficient.
                                     - ``gpu_id`` (str) — CUDA device index
                                       (e.g. ``'0'``).

        modification (callable, optional): User-defined function that accepts the
                                           ``System`` object and applies additional
                                           force modifications. Passed directly to
                                           :func:`HyresFF.buildSystem`. Called
                                           after all built-in forces are added.
                                           Default is ``None``.

    Returns:
        tuple:
            - ``system`` (System) — the fully constructed OpenMM ``System``.
            - ``sim`` (Simulation) — the initialized ``Simulation`` object with
              positions and velocities set.

    Raises:
        SystemExit: If an unsupported ensemble type is provided, if Mg²⁺ is
                    specified for a non-periodic system, or if an invalid box
                    dimension list is given.

    Notes:
        The dielectric constant is scaled as ``er = er_t(T) * er_ref / 77.6``,
        where ``er_t(T)`` is the temperature-dependent pure-water dielectric
        from :func:`cal_er`. The Mg²⁺-RNA lambda parameter is computed via
        :func:`nMg2lmd` using the ``'rA'`` RNA type. The CUDA platform is used
        with mixed precision.

    Example:
        >>> from HyresBuilder.utils import setup
        >>> system, sim = setup(params)

        >>> # With a custom force modification
        >>> def my_mod(system):
        ...     pass  # add or remove forces here
        >>> system, sim = setup(params, modification=my_mod)
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


## functions for umbrella sampling
def US_initial_windows(system, sim, group1, group2, r0, fc_pull=1000.0, v_pull=0.01, total_steps=100000, increment_steps=2):
    """
    SMD to steer the COM distance between two atom groups down to r0.

    The harmonic anchor starts at the current CV value and is stepped toward
    r0 at velocity v_pull. The anchor is clamped at r0 so it never overshoots.
    The simulation stops early as soon as the CV reaches or crosses r0. The
    final structure is always written to ``init.pdb`` regardless of outcome.

    Parameters
    ----------
    system : openmm.System
        The OpenMM System object to which the pulling force will be added.
    sim : openmm.app.Simulation
        The running Simulation; its context is reinitialized after the force
        is added and its integrator step size is used to advance the anchor.
    group1 : list[int]
        Atom indices for the first centroid group (e.g. the protein).
    group2 : list[int]
        Atom indices for the second centroid group (e.g. the ligand).
    r0 : float
        Target COM-COM distance / stopping criterion (nanometers). Must be
        >= 0. This is a center-of-mass distance, not an atom-atom distance —
        values well below 0.3 nm are physically meaningful depending on the
        size and geometry of the two groups.
    fc_pull : float, optional
        Harmonic force constant (kJ mol⁻¹ nm⁻², default 1000.0).
    v_pull : float, optional
        Pulling velocity (nm ps⁻¹, default 0.01).
    total_steps : int, optional
        Maximum integration steps (default 100 000).
    increment_steps : int, optional
        Steps between CV checks and anchor updates (default 2).

    Returns
    -------
    None
    """
    if float(r0) < 0:
        raise ValueError(f'r0 must be >= 0 nm, got {r0}')

    # --- Collective variable: COM distance between the two groups -----------
    cv = CustomCentroidBondForce(2, 'r; r = distance(g1, g2)')
    grp1, grp2 = cv.addGroup(group1), cv.addGroup(group2)
    cv.addBond([grp1, grp2])

    # --- Attach units --------------------------------------------------------
    fc_pull = fc_pull * kilojoule_per_mole / (unit.nanometers ** 2)
    v_pull  = v_pull  * unit.nanometers / unit.picosecond
    dt      = sim.integrator.getStepSize()

    r0_nm = float(r0)

    # --- Harmonic bias -------------------------------------------------------
    # Anchor is set to the placeholder r0 here; it is immediately corrected to
    # current_cv_value after reinitialisation, before any steps are taken.
    pullingForce = CustomCVForce('0.5*fc_pull*(cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0_nm * unit.nanometers)
    pullingForce.addCollectiveVariable('cv', cv)
    force_index = system.addForce(pullingForce)
    sim.context.reinitialize(preserveState=True)

    # --- Read current CV and set anchor to current distance -----------------
    current_cv_value = pullingForce.getCollectiveVariableValues(sim.context)[0]
    print(f'SMD start: r = {current_cv_value:.4f} nm  |  target r0 = {r0_nm:.4f} nm  '
          f'|  force_index = {force_index}')

    if current_cv_value <= r0_nm:
        print(f'CV already at target ({current_cv_value:.4f} nm <= {r0_nm:.4f} nm). Skipping SMD.')
    else:
        # Anchor starts at the current CV — not at r0 — to avoid a large
        # initial force spike from the full pulling distance.
        anchor_nm  = current_cv_value
        anchor_qty = anchor_nm * unit.nanometers
        sim.context.setParameter('r0', anchor_qty)

        # Constant per-increment displacement — computed once outside the loop.
        step_displacement_nm = (v_pull * dt * increment_steps).value_in_unit(unit.nanometers)
        log_every            = max(1, 5000  // increment_steps)
        stall_check_every    = max(1, 20000 // increment_steps)
        stall_threshold      = step_displacement_nm * stall_check_every * 0.05  # 5% of expected travel
        cv_at_last_check     = current_cv_value
        reached              = False

        for i in range(total_steps // increment_steps):
            sim.step(increment_steps)
            current_cv_value = pullingForce.getCollectiveVariableValues(sim.context)[0]

            # Advance anchor toward r0, clamped so it never overshoots.
            anchor_nm  = max(anchor_nm - step_displacement_nm, r0_nm)
            anchor_qty = anchor_nm * unit.nanometers
            sim.context.setParameter('r0', anchor_qty)

            if i % log_every == 0:
                print(f'step {i * increment_steps:>8d}  |  '
                      f'r = {current_cv_value:.4f} nm  |  anchor = {anchor_nm:.4f} nm')

            # Stall detection: if the CV has barely moved over a long window,
            # almost always means a conflicting force is cancelling the pull.
            if i > 0 and i % stall_check_every == 0:
                cv_change = abs(cv_at_last_check - current_cv_value)
                if cv_change < stall_threshold:
                    print(f'WARNING: CV moved only {cv_change:.6f} nm over the last '
                          f'{stall_check_every * increment_steps} steps '
                          f'(expected ~{step_displacement_nm * stall_check_every:.4f} nm). '
                          f'Check for conflicting forces via system.getForces().')
                cv_at_last_check = current_cv_value

            if current_cv_value <= (r0_nm + 0.5):
                print(f'Target reached: r = {current_cv_value:.4f} nm '
                      f'at step {i * increment_steps}')
                reached = True
                break

        if not reached:
            print(f'WARNING: target r0 not reached after {total_steps} steps. '
                  f'Final r = {current_cv_value:.4f} nm. '
                  f'If the CV barely moved, check for conflicting CustomCVForces '
                  f'on the system and remove them before retrying.')

    # --- Always write the final structure -----------------------------------
    state  = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
    coords = state.getPositions()
    with open('init.pdb', 'w') as outfile:
        PDBFile.writeFile(sim.topology, coords, outfile)
        

def US_create_windows(system, sim, group1, group2, r0, r1, window_num, fc_pull=1000.0, v_pull=0.01, total_steps = 100000, increment_steps = 10):
    """
    Generate umbrella sampling window structures via Steered Molecular Dynamics (SMD).

    A CustomCentroidBondForce collective variable (CV) is defined as the
    center-of-mass distance between two atom groups. A harmonic bias
    (CustomCVForce) is then applied and its anchor point r0 is advanced at
    constant velocity v_pull, steering the system from r0 to r1. Whenever
    the instantaneous CV crosses the next window target, the current
    coordinates are saved as a PDB file (window_0.pdb, window_1.pdb, …).

    Parameters
    ----------
    system : openmm.System
        The OpenMM System object to which the pulling force will be added.
    sim : openmm.app.Simulation
        The running Simulation; its context is reinitialized after the force
        is added and its integrator step size is used to advance r0.
    group1 : list[int]
        Atom indices for the first centroid group (e.g. the protein).
    group2 : list[int]
        Atom indices for the second centroid group (e.g. the ligand).
    r0 : float
        Starting CV value / initial anchor position (nanometers).
    r1 : float
        Ending CV value; the furthest window target (nanometers).
    window_num : int
        Number of evenly-spaced windows between r0 and r1 (inclusive).
    fc_pull : float, optional
        Harmonic force constant for the pulling bias
        (kJ mol⁻¹ nm⁻², default 1000.0).
    v_pull : float, optional
        Pulling velocity used to advance the anchor point
        (nm ps⁻¹, default 0.01).
    total_steps : int, optional
        Total number of integration steps to run during SMD (default 100 000).
    increment_steps : int, optional
        Steps between CV checks and anchor updates (default 10).

    Returns
    -------
        windows : numpy.ndarray
            Array of window target CV values (nanometers) corresponding to the saved structures.
        Writes window_<i>.pdb files to the current directory; one file per
        captured window. Fewer than window_num files are written if the
        simulation ends before all windows are reached.

    Notes
    -----
    * The pulling force is permanently added to `system`. Call
      ``system.removeForce(force_index)`` with the returned index if you
      intend to reuse the system object.
    * Coordinates are saved with ``enforcePeriodicBox=False`` so molecules
      are kept whole across PBC boundaries.
    * Window spacing is uniform in CV space; actual sampled windows may
      differ slightly because structures are captured when the CV first
      *crosses* each target.
    * Steps that do not divide evenly into increment_steps are silently
      dropped. Ensure ``total_steps % increment_steps == 0`` if exact step
      counts matter.
    """
    print('#create windows:')

    # --- Collective variable: COM distance between the two groups -----------
    cv = CustomCentroidBondForce(2, 'r; r = distance(g1, g2)')
    grp1, grp2 = cv.addGroup(group1), cv.addGroup(group2)
    cv.addBond([grp1, grp2])

    # --- Attach units --------------------------------------------------------
    fc_pull = fc_pull * kilojoule_per_mole / (unit.nanometers ** 2)
    v_pull  = v_pull  * unit.nanometers / unit.picosecond
    dt      = sim.integrator.getStepSize()

    r0_nm = float(r0)   
    r1_nm = float(r1)
    r0_qty = r0_nm * unit.nanometers  # Quantity used by OpenMM context parameter

    # --- Harmonic bias -------------------------------------------------------
    pullingForce = CustomCVForce('0.5*fc_pull*(cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0_qty)
    pullingForce.addCollectiveVariable('cv', cv)
    force_index = system.addForce(pullingForce)   # returned for optional cleanup
    sim.context.reinitialize(preserveState=True)

    # --- Window targets (plain floats in nm) ---------------------------------
    windows       = np.linspace(r0_nm, r1_nm, window_num)
    window_coords = []
    window_index  = 0

    # --- SMD pulling loop ----------------------------------------------------
    print("SMD pulling", pullingForce.getCollectiveVariableValues(sim.context))
    log_every = max(1, 5000 // increment_steps)
    for i in range(total_steps // increment_steps):
        sim.step(increment_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(sim.context)[0]

        if i % log_every == 0:
            print(f'r0 = {r0_nm:.4f} nm  |  r = {current_cv_value:.4f} nm')

        # Advance the anchor position
        step_displacement = v_pull * dt * increment_steps          # Quantity (nm)
        r0_nm  += step_displacement.value_in_unit(unit.nanometers) # keep float in sync
        r0_qty += step_displacement                                 # keep Quantity in sync
        sim.context.setParameter('r0', r0_qty)

        # Capture window structure when CV crosses the next target
        if window_index < len(windows) and current_cv_value >= windows[window_index]:
            print(f'saving window {window_index}: target={windows[window_index]:.4f} nm  r={current_cv_value:.4f} nm')
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
            window_coords.append(state.getPositions())
            window_index += 1
        
        if window_index >= window_num:
            print('All windows captured.')
            break

    # --- Warn if simulation ended before all windows were reached ------------
    if window_index < window_num:
        print(f'WARNING: only {window_index}/{window_num} windows captured. '
              f'Consider increasing total_steps or decreasing v_pull.')

    # --- Write window PDB files ----------------------------------------------
    for i, coords in enumerate(window_coords):
        with open(f'window_{i}.pdb', 'w') as outfile:
            PDBFile.writeFile(sim.topology, coords, outfile)
    
    return windows


def US_collect_cv(system, sim, group1, group2, window_index, windows, fc_pull=300.0, total_steps = 20000000, record_steps = 1000):
    """
    run umbrella sampling for each window and collect CV values.

    Parameters
    ----------
    system : openmm.System
        The OpenMM System object to which the pulling force will be added.
    sim : openmm.app.Simulation
        The running Simulation; its context is reinitialized after the force
        is added and its integrator step size is used to advance r0.
    group1 : list[int]
        Atom indices for the first centroid group (e.g. the protein).
    group2 : list[int]
        Atom indices for the second centroid group (e.g. the ligand).
    window_index : int
        Index of the current window to sample.
    windows : numpy.ndarray
        Array of window target CV values (nanometers).
    fc_pull : float, optional
        Harmonic force constant for the pulling bias
        (kJ mol⁻¹ nm⁻², default 300.0).
    total_steps : int, optional
        Total number of integration steps to run the production simulation for each window (default 20000000).
    record_steps : int, optional
        Steps between CV recordings (default 1000).

    Returns
    -------
    None
        Writes cv_values_window_{window_index}.txt.
    """
    print("define the CV:")
    cv = CustomCentroidBondForce(2, 'r; r = distance(g1, g2)')
    grp1, grp2 = cv.addGroup(group1), cv.addGroup(group2)
    cv.addBond([grp1, grp2])

    ### pulling force 
    fc_pull = fc_pull*kilojoule_per_mole/(unit.nanometers**2)

    ### running windows
    print('running window', window_index)
    r0 = windows[window_index]*unit.nanometers

    pullingForce = CustomCVForce('0.5*fc_pull*(cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0)
    pullingForce.addCollectiveVariable('cv', cv)
    system.addForce(pullingForce)
    sim.context.reinitialize(preserveState=True)
    sim.context.setParameter('r0', r0)
    sim.step(1000)

    # data collection
    cv_values = []
    for i in range(total_steps//record_steps):
        sim.step(record_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(sim.context)
        cv_values.append([i, current_cv_value[0]])
    np.savetxt(f'cv_values_window_{window_index}.txt', np.array(cv_values))
    print('Completed window', window_index)


def US_gen_metafile(windows, fc_pull=300.0):
    """
    Generate a metafile (metafile.txt) that lists the CV values file, window target, and force constant for each umbrella sampling window.

    Parameters
    ----------
    windows : numpy.ndarray
        Array of window target CV values (nanometers) corresponding to the saved structures.
    fc_pull : float, optional
        Harmonic force constant used in the umbrella sampling (kJ mol⁻¹ nm⁻², default 300.0).
    
    Returns
    -------
    None
        Writes metafile.txt with lines formatted as: "cv_values_window_{i}.txt {window_target} {fc_pull}" for each window index i.
    """
    metafilelines = []
    for i in range(len(windows)):
        metafileline = f'cv_values_window_{i}.txt {windows[i]} {fc_pull}\n'
        metafilelines.append(metafileline)

    with open("metafile.txt", "w") as f:
        f.writelines(metafilelines)