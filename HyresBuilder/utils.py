"""
Force field loading and simulation setup utilities for HyRes and iConRNA systems.

This module provides the shared infrastructure used across HyresBuilder to
locate force field files, compute solution-condition parameters, and assemble
complete OpenMM simulation systems. It serves as the primary entry point for
constructing production-ready simulations from a PSF/PDB pair and a parameter
namespace.

Force field file resolution
---------------------------
CHARMM topology and parameter files are bundled within the installed
HyresBuilder package and resolved at runtime via ``importlib.resources``,
requiring no manual path management. Five models are supported by
:func:`load_ff`:

========== ===============================================================
Model      Description
========== ===============================================================
Protein    HyRes coarse-grained protein (``top_hyres_mix`` / ``param_hyres_mix``)
RNA        iConRNA coarse-grained RNA (``top_RNA_mix`` / ``param_RNA_mix``)
DNA        iConDNA coarse-grained DNA (``top_DNA_mix`` / ``param_DNA_mix``)
rG4s       RNA G-quadruplex; uses RNA topology with ``param_rG4s``
ATP        ATP force field (``top_ATP`` / ``param_ATP``)
========== ===============================================================

Solution-condition parameters
------------------------------
Three helper functions translate experimental solution conditions into the
physical parameters consumed by the force field:

* :func:`cal_er` — temperature-dependent relative dielectric constant of
  water, fitted to a cubic polynomial.
* :func:`cal_dh` — Debye–Hückel screening length (nm) from ionic strength
  and temperature, using the Bjerrum length at the given dielectric.
* :func:`nMg2lmd` — Mg²⁺-to-lambda conversion: maps a Mg²⁺ concentration
  (mM) and temperature onto the charge-scaling factor ``lmd`` that modulates
  Mg²⁺–RNA phosphate interactions. Empirical Hill-function parameters are
  provided for rA, rU, and CAG RNA contexts, or can be supplied as custom
  values.

Simulation setup pipeline
--------------------------
Two setup functions share the same seven-stage pipeline — parse parameters,
configure PBC, compute force field parameters, load topology files, import
PSF/PDB, build the custom force field, attach the integrator and barostat —
and differ only in which force field they target:

* :func:`setup` — primary entry point for HyRes protein / iConRNA mixed
  systems; accepts a rich ``params`` namespace and an optional
  ``modification`` hook for injecting extra forces after the built-in
  terms are added.
* :func:`iConRNA_setup` — specialised setup for original iConRNA systems
* :func:`rG4s_setup` — specialised setup for rG4 G-quadruplex systems;
  loads the rG4s parameter file and passes an additional ``ion_type``
  energy parameter to :func:`rG4sFF.rG4sSystem`.

Both functions return a ``(system, sim)`` tuple with positions and velocities
initialised, ready for production runs. The CUDA platform is used throughout
with mixed precision.

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.app``, ``openmm.unit``)
* `NumPy <https://numpy.org>`_ (``numpy``)
* HyresBuilder submodules: ``HyresFF``, ``rG4sFF``
"""

from importlib.resources import files
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np
from .FFs import *


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
    elif model == 'AGs':
        path1 = ff / "top_AGs.inp"
        path2 = ff / "param_AGs.inp"
    else:
        print("Error: The model type {} is not supported, only for Protein, RNA, DNA, rG4s, and ATP.".format(model))
        exit(1)

    top_inp, param_inp = path1.as_posix(), path2.as_posix()

    return top_inp, param_inp

def estimate_lmd(cNa, cMg, length, Rg, T):
    # imperical estimation of nMg: doi: 10.1016/j.bpj.2010.06.029
    # convert nMg to lmd: https://doi.org/10.1073/pnas.2504583122
    N = length
    Rg0 = 0.406*N + 130/(N + 11)
    A = 0.65 + 4.2/N*(Rg/Rg0)**2
    B = 1.8 - 9.8/N*(Rg/Rg0)**2
    Na_Mg = 10**B * cMg**A
    nMg = 0.47*(Na_Mg/(Na_Mg+cNa))

    nMg_T = nMg + 0.0012*(T-273-30)
    lmd = 1.265*(nMg_T/0.172)**0.625/(1+(nMg_T/0.172)**0.625)

    return lmd

def nMg2lmd(cMg, T, F=0.0, M=0.0, n=0.0, RNA='rA'):
    if RNA == 'rA':
        F, M, n = 0.54, 0.94, 0.59
    elif RNA == 'rU':
        F, M, n = 0.48, 1.31, 0.85
    elif RNA == 'CAG':
        F, M, n = 0.53, 0.68, 0.28
    else:
        if M == 0.0:
            print("Error: Please give F_Mg, M_1/2, and n if the RNA is custom type")
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
                                     - ``lmd`` (float) — lmd for Mg²⁺-RNA interaction.
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
    lmd = getattr(params, "lmd", 0)                              # lmd for Mg²⁺-RNA interaction, if don't give, it's 0.
    ensemble = params.ens

    dt = params.dt
    er_ref = params.er_ref
    pressure = params.pressure
    friction = params.friction
    gpu_id = params.gpu_id
    
    # 2. set pbc and box vector
    if ensemble == 'non' and lmd != 0.0:
        print("Error: Mg ion cannot be run in non-periodic system.")
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
    top_AGs, param_AGs = load_ff('AGs')
    ffparams = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro, top_AGs, param_AGs)

    print('\n################## load coordinates and topology ###################')
    # 5. import coordinates and topology form charmm pdb and psf
    pdb = PDBFile(pdb_file)
    psf = CharmmPsfFile(psf_file)
    top = psf.topology
    print(f"coordinate file: {pdb_file}")
    print(f"topology file: {psf_file}")

    print('\n################## create system ###################')
    if ensemble == 'non':
        system = psf.createSystem(ffparams, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
    else:
        psf.setBox(lx, ly, lz)
        top.setPeriodicBoxVectors((a, b, c))
        top.setUnitCellDimensions((lx, ly,lz))
        system = psf.createSystem(ffparams, nonbondedMethod=CutoffPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
        system.setDefaultPeriodicBoxVectors(a, b, c)
    
    print(f"nonbonded cutoff: {cutoff}")
    print(f"switch distance: {d_switch}")

    # 6. construct force field
    system = buildSystem(psf, system, ffs, modification=modification)
    print("buildSystem for HyRes_iConRNA")

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


def iConRNA_setup(params, modification=None):
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
                                     - ``lmd`` (float) — lmd value for Mg²⁺-RNA interaction.
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
    lmd = params.lmd                                           # lmd value for Mg²⁺-RNA interaction
    ensemble = params.ens

    dt = params.dt
    er_ref = params.er_ref
    pressure = params.pressure
    friction = params.friction
    gpu_id = params.gpu_id
    
    # 2. set pbc and box vector
    if ensemble == 'non' and lmd != 0.0:
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
    cutoff = 1.8*unit.nanometer                                 # nonbonded cutoff
    d_switch = 1.6*unit.nanometer                               # switch function starting distance
    temperature = T*unit.kelvin 
    er_t = cal_er(T)                                                   # relative electric constant
    er = er_t*er_ref/77.6
    dh = cal_dh(c_ion, T)                                            # Debye-Huckel screening length in nm
    print(f"dielectric constant: er = {er:.2f}")
    print(f"Debye screening length: dh = {dh.value_in_unit(unit.nanometers):.2f} nm")
    print(f'Mg-RNA interaction: lmd = {lmd:.2f}')

    # force field parameters
    ffs = {
        'temp': T,                                                   # Temperature
        'lmd': lmd,                                                  # Charge scaling factor of P-
        'dh': dh,                                                    # Debye Huckel screening length
        'ke': 138.935456,                                            # Coulomb constant, ONE_4PI_EPS0
        'er': er,                                                    # relative dielectric constant
    }

    # 4. load force field files
    path1 = files("HyresBuilder") / "forcefield" / "top_RNA.inp"
    top_RNA = path1.as_posix()
    path2 = files("HyresBuilder") / "forcefield" / "param_RNA.inp"
    param_RNA = path2.as_posix()
    top_AGs, param_AGs = load_ff('AGs')
    top_pro, param_pro = load_ff('Protein')
    ffparams = CharmmParameterSet(top_RNA, param_RNA, top_AGs, param_AGs, top_pro, param_pro)

    print('\n################## load coordinates and topology ###################')
    # 5. import coordinates and topology form charmm pdb and psf
    pdb = PDBFile(pdb_file)
    psf = CharmmPsfFile(psf_file)
    top = psf.topology
    print(f"coordinate file: {pdb_file}")
    print(f"topology file: {psf_file}")

    print('\n################## create system ###################')
    if ensemble == 'non':
        system = psf.createSystem(ffparams, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
    else:
        psf.setBox(lx, ly, lz)
        top.setPeriodicBoxVectors((a, b, c))
        top.setUnitCellDimensions((lx, ly,lz))
        system = psf.createSystem(ffparams, nonbondedMethod=CutoffPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
        system.setDefaultPeriodicBoxVectors(a, b, c)
    
    print(f"nonbonded cutoff: {cutoff}")
    print(f"switch distance: {d_switch}")

    # 6. construct force field
    system = iConRNASystem(psf, system, ffs, modification=modification)
    print("iConRNASystem for iConRNA model")

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


def setupMg(params, RNA='rA', modification=None):
    """
    Similar to setup(), but with a focus on Mg²⁺-RNA interactions, where
    the er = 20 was specifically chosen to match the experimental Mg²⁺-RNA interactions.
    other details are the same as setup().
    """
    
    print('\nThis is special system, where er = 20.0 for Mg-P interactions, otherwise 60.0')
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
    lmd = nMg2lmd(c_Mg, T, RNA=RNA)
    print(f"dielectric constant: er = {er:.2f}")
    print(f"Debye screening length: dh = {dh.value_in_unit(unit.nanometers):.2f} nm")

    # Mg-P interaction
    match RNA:
        case str():
            lmd = nMg2lmd(c_Mg, T, RNA=RNA)
        case (F, M, n):
            lmd = nMg2lmd(c_Mg, T, F=F, M=M, n=n)
        case float():
            lmd = RNA
    print(f"dielectric constant for Mg-P interactions: 20.0")
    print(f'Mg-P interactions are determined based on {RNA}')
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
    top_AGs, param_AGs = load_ff('AGs')
    ffparams = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro, top_AGs, param_AGs)

    print('\n################## load coordinates and topology ###################')
    # 5. import coordinates and topology form charmm pdb and psf
    pdb = PDBFile(pdb_file)
    psf = CharmmPsfFile(psf_file)
    top = psf.topology
    print(f"coordinate file: {pdb_file}")
    print(f"topology file: {psf_file}")

    print('\n################## create system ###################')
    if ensemble == 'non':
        system = psf.createSystem(ffparams, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
    else:
        psf.setBox(lx, ly, lz)
        top.setPeriodicBoxVectors((a, b, c))
        top.setUnitCellDimensions((lx, ly,lz))
        system = psf.createSystem(ffparams, nonbondedMethod=CutoffPeriodic, constraints=HBonds,
                                  nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)
        system.setDefaultPeriodicBoxVectors(a, b, c)
    
    print(f"nonbonded cutoff: {cutoff}")
    print(f"switch distance: {d_switch}")

    # 6. construct force field
    system = buildMgSystem(psf, system, ffs, modification=modification)
    print("buildMgSystem for HyRes_iConRNA-Mg")

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
