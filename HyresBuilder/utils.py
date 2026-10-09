"""
Force field loading and simulation setup utilities for HyresBuilder.

This module provides the shared infrastructure used across HyresBuilder to
locate bundled force field files, convert custom molecule definitions,
compute solution-condition parameters, and assemble complete OpenMM
simulations from a CHARMM PSF/PDB pair and a parameter namespace. It covers
HyRes proteins, iConRNA/iConDNA nucleic acids, aminoglycosides (AGs),
metabolites, and coarse-grained polymers. The force field builders themselves
live in :mod:`HyresBuilder.FFs` (imported here via ``from .FFs import *``).

Force field file resolution
---------------------------
CHARMM topology and parameter files are bundled in the package's
``forcefield`` directory and resolved at runtime via ``importlib.resources``.
:func:`load_ff` supports the following model names:

========== ===============================================================
Model      Files
========== ===============================================================
Protein    HyRes protein (``top_hyres_mix`` / ``param_hyres_mix``)
RNA        iConRNA (``top_RNA_mix`` / ``param_RNA_mix``)
DNA        iConDNA (``top_DNA_mix`` / ``param_DNA_mix``)
rG4s       RNA G-quadruplex (``top_RNA_mix`` / ``param_rG4s``)
ATP        ATP (``top_ATP`` / ``param_ATP``)
AGs        aminoglycosides, e.g. KAN (``top_AGs`` / ``param_AGs``)
Metabolite metabolites (``top_metabolome`` / ``param_metabolome``)
Polymer    CG polymers: QDM (quaternized DMAEMA), BZM (2-phenylethyl
           methacrylate), PEG/PEO (``top_polymer`` / ``param_polymer``)
========== ===============================================================

Custom molecules can be converted from an ``.itp``-style definition into
CHARMM ``.top``/``.par`` files with :func:`itp2charmm`.

Solution-condition parameters
-----------------------------
* :func:`cal_er` — temperature-dependent relative dielectric constant of
  water (cubic polynomial in T - 273).
* :func:`cal_dh` — Debye–Hückel screening length (``Quantity`` in nm) from
  ionic strength (M) and temperature (K) via the Bjerrum length.
* :func:`nMg2lmd` / :func:`estimate_lmd` — map a Mg²⁺ concentration onto
  the factor ``lmd`` that scales Debye–Hückel interactions between RNA/DNA
  phosphate (``P``) beads and Mg²⁺ (``MG``) ions.

Simulation setup
----------------
All setup functions follow the same pipeline — read parameters, configure
PBC, compute ``er``/``dh``/``lmd``, load topology/parameter files, read
PSF/PDB, build the custom force field, add a 25-step ``MonteCarloBarostat``
for NPT, and create a CUDA (mixed precision) ``Simulation`` with positions
and velocities set. They differ in the builder used and the files loaded:

* :func:`setup` — general entry point; :func:`FFs.buildSystem`; loads
  Protein, RNA, DNA, AGs, Metabolite, Polymer (+ optional custom ``.itp``).
* :func:`setup2` — older argument-style variant; :func:`FFs.buildSystem`;
  loads Protein, RNA, Polymer.
* :func:`rG4s_setup` — RNA G-quadruplexes; :func:`FFs.rG4sSystem`; loads
  Protein, RNA, AGs, Polymer and passes the G–G strength ``GG``.
* :func:`iConRNA_setup` — original iConRNA model; :func:`FFs.iConRNASystem`;
  loads ``top_RNA``/``param_RNA`` plus Protein, DNA, AGs, Metabolite, Polymer
  (+ optional custom ``.itp``).
* :func:`setupMg` — explicit Mg²⁺/Ca²⁺ systems; :func:`FFs.buildMgSystem`;
  loads Protein, RNA, AGs, Metabolite, Polymer.
* :func:`iConDNA_setup` — iConDNA systems; :func:`FFs.iConDNASystem`;
  loads Protein, DNA, AGs, Metabolite, Polymer.

Each returns a ``(system, sim)`` tuple. :func:`crowding_effect` can be
applied to a built ``System`` (before creating a ``Simulation``) to scale
the LJ well depth uniformly and mimic crowding.

Dependencies
------------
* `OpenMM <https://openmm.org>`_ (``openmm``, ``openmm.app``, ``openmm.unit``)
* `NumPy <https://numpy.org>`_ (``numpy``)
* HyresBuilder submodule: ``FFs``
"""

from importlib.resources import files
from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np
import re
import os
from .FFs import *


def itp2charmm(itp):
    """
    Convert an ITP-style molecule definition into CHARMM TOP and PAR files.

    The input is split into sections by bracketed upper-case headers
    (``[ RESI ]``, ``[ ATOM ]``, ``[ BOND ]``, ``[ ANGL ]``, ``[ DIHE ]``,
    ``[ IMPR ]``); text after ``;`` is ignored and other sections are skipped.
    Expected fields per line:

    - ``RESI``: residue name (first entry is used; defaults to ``"RESI"``).
    - ``ATOM``: ``name type charge`` (a ``+`` sign in the charge is stripped).
    - ``BOND``: ``a1 a2 Kb b0``.
    - ``ANGL``: ``a1 a2 a3 Ktheta Theta0 [Kub S0]``.
    - ``DIHE``: ``a1 a2 a3 a4 Kchi n delta``.
    - ``IMPR``: ``a1 a2 a3 a4 Kpsi n psi0``.

    The TOP file contains ``RESI`` (with the summed charge), ``GROUP``,
    ``ATOM``, and ``BOND``/``ANGL``/``DIHE``/``IMPR`` connectivity entries.
    The PAR file contains BOND/THETAS/PHI/IMPHI parameters keyed by atom
    type, followed by a fixed NONBONDED and NBFIX block for the metabolite
    bead types (M01–M08, MS1, MS2, MCI, MSO, MSS, MCL, MCF, MBR, SMG).
    Entries with too few fields are skipped, and entries referencing an
    unknown atom name print a warning.

    Args:
        itp (str): Path to the ITP-style input file.

    Returns:
        None. Side effect: writes ``<RESI>.top`` and ``<RESI>.par`` to the
        current working directory (overwriting existing files).

    Raises:
        FileNotFoundError: If ``itp`` does not exist.
        ValueError: If an atom charge cannot be parsed as a float.

    Example:
        >>> from HyresBuilder.utils import itp2charmm
        >>> itp2charmm('ABC.itp')   # writes ABC.top and ABC.par if RESI is ABC
    """
    
    # Read the ITP File
    with open(itp, 'r') as f:
        itp_content = f.read()
        
    sections = {'RESI': [], 'ATOM': [], 'BOND': [], 'ANGL': [], 'DIHE': [], 'IMPR': []}
    current_section = None
    
    for line in itp_content.split('\n'):
        line = line.split(';')[0].strip()
        if not line:
            continue
            
        match = re.match(r'\[\s*([A-Z]+)\s*\]', line)
        if match:
            current_section = match.group(1)
            continue
            
        if current_section and current_section in sections:
            sections[current_section].append(line.split())

    # Extract the residue name to use for file naming and the PAR header
    resi_name = sections['RESI'][0][0] if sections['RESI'] else "RESI"
    
    atom_types = {}
    total_charge = 0.0
    for atom in sections['ATOM']:
        if len(atom) >= 3:
            name, atype = atom[0], atom[1]
            charge = float(atom[2].replace('+', ''))
            atom_types[name] = atype
            total_charge += charge
        else:
            print(f"Warning: Atom entry has fewer than 3 fields: {atom}")

    # --- 1. Generate the TOP file lines ---
    top_lines = []
    top_lines.append(f"RESI {resi_name:<8} {total_charge:>8.2f}")
    top_lines.append("GROUP")
    
    for atom in sections['ATOM']:
        if len(atom) >= 3:
            top_lines.append(f"ATOM {atom[0]:<4} {atom[1]:<6} {float(atom[2].replace('+', '')):>8.2f}")
        
    for bond in sections['BOND']:
        if len(bond) >= 4:
            top_lines.append(f"BOND {bond[0]:<4} {bond[1]:<4}")
        
    for angl in sections['ANGL']:
        if len(angl) >= 3:
            top_lines.append(f"ANGL {angl[0]:<4} {angl[1]:<4} {angl[2]:<4}")
        
    for dihe in sections['DIHE']:
        if len(dihe) >= 4:
            top_lines.append(f"DIHE {dihe[0]:<4} {dihe[1]:<4} {dihe[2]:<4} {dihe[3]:<4}")
        
    for impr in sections['IMPR']:
        if len(impr) >= 4:
            top_lines.append(f"IMPR {impr[0]:<4} {impr[1]:<4} {impr[2]:<4} {impr[3]:<4}")

    top_content = "\n".join(top_lines)

    # --- 2. Generate the PAR file lines ---
    par_lines = []
    
    # Replace {RESI name}
    par_lines.append(f"* parameter file for {resi_name}\n")
    
    par_lines.append("BOND")
    par_lines.append("!type     Kb  b0")
    for bond in sections['BOND']:
        if len(bond) >= 4:
            try:
                t1, t2 = atom_types[bond[0]], atom_types[bond[1]]
                par_lines.append(f"{t1:<5} {t2:<7} {bond[2]:>4} {bond[3]:>7}")
            except KeyError as e:
                print(f"Warning: Atom type not found for bond {bond}: {e}")
        
    par_lines.append("\nTHETAS")
    par_lines.append("!atom types         Ktheta    Theta0   Kub     S0")
    for angl in sections['ANGL']:
        if len(angl) >= 5: # Changed from 7 to allow angles without Urey-Bradley terms
            try:
                t1, t2, t3 = atom_types[angl[0]], atom_types[angl[1]], atom_types[angl[2]]
                # If Urey-Bradley terms exist (7 fields)
                if len(angl) >= 7:
                    par_lines.append(f"{t1:<5} {t2:<5} {t3:<7} {angl[3]:>5} {angl[4]:>9} {angl[5]:>5} {angl[6]:>5}")
                # If only Ktheta and Theta0 exist (5 fields)
                else:
                    par_lines.append(f"{t1:<5} {t2:<5} {t3:<7} {angl[3]:>5} {angl[4]:>9}")
            except KeyError as e:
                print(f"Warning: Atom type not found for angle {angl}: {e}")
        
    par_lines.append("\nPHI")
    par_lines.append("!atom types               Kchi    n   delta")
    for dihe in sections['DIHE']:
        if len(dihe) >= 7:
            try:
                t1, t2, t3, t4 = atom_types[dihe[0]], atom_types[dihe[1]], atom_types[dihe[2]], atom_types[dihe[3]]
                par_lines.append(f"{t1:<5} {t2:<5} {t3:<5} {t4:<7} {dihe[4]:>4} {dihe[5]:>4} {dihe[6]:>6}")
            except KeyError as e:
                print(f"Warning: Atom type not found for dihedral {dihe}: {e}")
        
    par_lines.append("\nIMPHI")
    par_lines.append("!atom types               Kpsi        psi0")
    for impr in sections['IMPR']:
        if len(impr) >= 7:
            try:
                t1, t2, t3, t4 = atom_types[impr[0]], atom_types[impr[1]], atom_types[impr[2]], atom_types[impr[3]]
                par_lines.append(f"{t1:<5} {t2:<5} {t3:<5} {t4:<7} {impr[4]:>4}    {impr[5]}    {impr[6]:>3}")
            except KeyError as e:
                print(f"Warning: Atom type not found for improper {impr}: {e}")

    # Append the comprehensive NONBONDED and NBFIX lists verbatim from the reference format
    par_lines.append("""
NONBONDED  NBXMOD 5  ATOM CDIEL SWITCH VATOM VDISTANCE VSWITCH -
     CUTNB 12.0  CTOFNB 12.0  CTONNB 11.0

M01      0.00     -0.0350      2.4340   0.00     -0.0250      2.4340
M02      0.00     -0.0150      2.0982   0.00     -0.0150      2.0982
M03      0.00     -0.0720      2.2759   0.00     -0.0720      2.2759
M04      0.00     -0.0360      2.0591   0.00     -0.0360      2.0591 !
M05      0.00     -0.0150      2.3268   0.00     -0.0150      2.3268 !
M06      0.00     -0.0150      2.5008   0.00     -0.0150      2.5008 !
M07      0.00     -0.0900      2.3689   0.00     -0.0900      2.3689 !
M08      0.00     -0.0360      2.2600   0.00     -0.0360      2.2600
MS1      0.00     -0.0592      2.2826   0.00     -0.0592      2.2826
MS2      0.00     -0.0566      2.2826   0.00     -0.0566      2.2826
MCI      0.00     -0.1594      2.4900   0.00     -0.1594      2.4900
MSO      0.00     -0.0800      2.3200   0.00     -0.0800      2.3200
MSS      0.00     -0.1200      2.2600   0.00     -0.1200      2.2600
MCL      0.00     -0.1594      2.3400   0.00     -0.1594      2.3400
MCF      0.00     -0.1594      2.1900   0.00     -0.1594      2.1900
MBR      0.00     -0.1594      2.4000   0.00     -0.1594      2.4000
SMG      0.00     -0.0200      2.0000   0.00     -0.0200      2.0000 !!

NBFIX
!                 Emin         Rmin
!                 (kcal/mol)   (A)
M01    M01       -0.0519      4.8680           !necessary, do not delete
M02    M02       -0.0150      4.1964           !necessary, do not delete
M03    M03       -0.0720      4.5518           !necessary, do not delete
M04    M04       -0.0360      4.1182           !necessary, do not delete
M05    M05       -0.0150      4.6536           !necessary, do not delete
M06    M06       -0.0150      5.0016           !necessary, do not delete
M07    M07       -0.0900      4.7378           !necessary, do not delete
M08    M08       -0.0360      4.5200           !necessary, do not delete
MS1    MS1       -0.0592      4.5652           !necessary, do not delete
MS2    MS2       -0.0566      4.5652           !necessary, do not delete
MCI    MCI       -0.1594      4.9800           !necessary, do not delete
MSO    MSO       -0.0800      4.6400           !necessary, do not delete
MSS    MSS       -0.1200      4.5200           !necessary, do not delete
MCL    MCL       -0.1594      4.6800           !necessary, do not delete
MCF    MCF       -0.1594      4.3800           !necessary, do not delete
MBR    MBR       -0.1594      4.8000           !necessary, do not delete

END""")

    par_content = "\n".join(par_lines)
    
    # --- 3. Output to Files ---
    top_filename = f"{resi_name}.top"
    par_filename = f"{resi_name}.par"
    
    with open(top_filename, 'w') as f:
        f.write(top_content)
        
    with open(par_filename, 'w') as f:
        f.write(par_content)
        
    print(f"Convert itp to {top_filename} and {par_filename}")


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
                     - ``'AGs'`` — aminoglycosides such as KAN
                       (``top_AGs`` / ``param_AGs``)
                     - ``'Metabolite'`` — metabolites
                       (``top_metabolome`` / ``param_metabolome``)
                     - ``'Polymer'`` — CG polymers: QDM (quaternized
                       DMAEMA), BZM (2-phenylethyl methacrylate), PEG/PEO
                       (``top_polymer`` / ``param_polymer``)

    Returns:
        tuple[str, str]: ``(top_inp, param_inp)``, the POSIX paths of the
        CHARMM topology and parameter ``.inp`` files. Existence of the files
        is not checked.

    Raises:
        SystemExit: If an unsupported model name is provided (prints an
                    error and calls ``exit(1)``).

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
    elif model == 'Metabolite':
        path1 = ff / "top_metabolome.inp"
        path2 = ff / "param_metabolome.inp"
    elif model == 'Polymer':
        path1 = ff / "top_polymer.inp"
        path2 = ff / "param_polymer.inp"
    else:
        print("Error: The model type {} is not supported, only for Protein, RNA, DNA, rG4s, ATP, AGs, Metabolite, and Polymer.".format(model))
        exit(1)

    top_inp, param_inp = path1.as_posix(), path2.as_posix()

    return top_inp, param_inp

def estimate_lmd(cNa, cMg, length, Rg, T):
    """
    Roughly estimate the P–Mg²⁺ scaling factor ``lmd`` for an RNA.

    The number of Mg²⁺ bound per phosphate is estimated with an empirical
    competition model (doi:10.1016/j.bpj.2010.06.029) using the RNA length
    and compactness relative to ``Rg0 = 0.406*N + 130/(N + 11)``:

        nMg = 0.47 * X / (X + cNa),  X = 10**B * cMg**A

    It is then shifted by ``0.0012*(T - 303)`` and converted to ``lmd`` with
    the Hill-type mapping ``1.265*(nMg/0.172)**0.625 / (1 + (nMg/0.172)**0.625)``
    (doi:10.1073/pnas.2504583122; calibrated for er = 20).

    Args:
        cNa (float): Monovalent salt concentration (same units as ``cMg``;
                     the example scripts pass mM).
        cMg (float): Mg²⁺ concentration.
        length (int): RNA length N (nucleotides).
        Rg (float): RNA radius of gyration, in the units of the ``Rg0``
                    reference formula above.
        T (float): Temperature in Kelvin.

    Returns:
        float: Estimated ``lmd``.

    Example:
        >>> from HyresBuilder.utils import estimate_lmd
        >>> lmd = estimate_lmd(150.0, 5.0, 40, 20.0, 303.0)
    """
    # imperical estimation of nMg: doi: 10.1016/j.bpj.2010.06.029
    # convert nMg to lmd: https://doi.org/10.1073/pnas.2504583122
    N = length
    Rg0 = 0.406*N + 130/(N + 11)
    A = 0.65 + 4.2/N*(Rg/Rg0)**2
    B = 1.8 - 9.8/N*(Rg/Rg0)**2
    Na_Mg = 10**B * cMg**A
    nMg = 0.47*(Na_Mg/(Na_Mg+cNa))

    nMg_T = nMg + 0.0012*(T-273-30)
    lmd = 1.265*(nMg_T/0.172)**0.625/(1+(nMg_T/0.172)**0.625)           # for er = 20.0
    #lmd = 1.480*(nMg_T/0.172)**0.625/(1+(nMg_T/0.172)**0.625)           # for er = 60.0

    return lmd

def nMg2lmd(cMg, T, F=0.0, M=0.0, n=0.0, RNA='rA'):
    """
    Convert a Mg²⁺ concentration into the P–Mg²⁺ scaling factor ``lmd``.

    The number of bound Mg²⁺ per phosphate follows a Hill function,
    ``nMg = F*(cMg/M)**n / (1 + (cMg/M)**n)``, is shifted by
    ``0.0012*(T - 303)``, and is converted to ``lmd`` with
    ``1.265*(nMg/0.172)**0.625 / (1 + (nMg/0.172)**0.625)``
    (doi:10.1073/pnas.2504583122; calibrated for er = 20). ``lmd`` scales
    only the Debye–Hückel interaction between phosphate (``P``) beads and
    Mg²⁺ (``MG``) ions.

    Args:
        cMg (float): Mg²⁺ concentration in mM (same units as ``M``).
        T (float): Temperature in Kelvin.
        F (float): Maximum Mg²⁺ bound per phosphate. Only used when ``RNA``
                   is not a preset. Default ``0.0``.
        M (float): Half-saturation concentration M_1/2. Only used when
                   ``RNA`` is not a preset; must be non-zero then.
                   Default ``0.0``.
        n (float): Hill coefficient. Only used when ``RNA`` is not a preset.
                   Default ``0.0``.
        RNA (str): Preset ``'rA'`` (F, M, n = 0.54, 0.94, 0.59), ``'rU'``
                   (0.48, 1.31, 0.85) or ``'CAG'`` (0.53, 0.68, 0.28); any
                   other value uses the supplied ``F``, ``M``, ``n``.
                   Presets override user-supplied values. Default ``'rA'``.

    Returns:
        float: ``lmd`` (``0.0`` when ``cMg == 0``).

    Raises:
        SystemExit: If a non-preset ``RNA`` is given with ``M == 0.0``
                    (prints an error and calls ``exit(1)``).

    Example:
        >>> from HyresBuilder.utils import nMg2lmd
        >>> lmd = nMg2lmd(10.0, 303.0, RNA='rA')
        >>> lmd = nMg2lmd(10.0, 303.0, F=0.5, M=1.0, n=0.6, RNA='custom')
    """
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
        lmd = 1.265*(nMg_T/0.172)**0.625/(1+(nMg_T/0.172)**0.625)           # for er = 20.0
        #lmd = 1.480*(nMg_T/0.172)**0.625/(1+(nMg_T/0.172)**0.625)           # for er = 60.0
    
    return lmd

# calculate relative dielectric constant at temperature T in K
def cal_er(T):
    """
    Relative dielectric constant of water at temperature ``T``.

    Uses the cubic fit ``87.74 - 0.4008*Td + 9.398e-4*Td**2 - 1.41e-6*Td**3``
    with ``Td = T - 273``.

    Args:
        T (float): Temperature in Kelvin.

    Returns:
        float: Relative dielectric constant (about 76.5 at 303 K).
    """
    Td = T-273
    er_t = 87.74-0.4008*Td+9.398*10**(-4)*Td**2-1.41*10**(-6)*Td**3
    return er_t

# calculate Debye-Huckel screening length in nm
DH_NO_SALT = 1.0e4      # nm, Debye-Huckel screening length used for salt-free systems (no screening)


def cal_dh(c_ion, T):
    """
    Debye–Hückel screening length for a 1:1 salt.

    Computes ``dh = 1/sqrt(8*pi*lB*NA*c_ion*1e-24)`` with the Bjerrum length
    ``lB = 16710/(er*T)`` nm, where ``er = cal_er(T)`` (the unscaled water
    dielectric).

    Without salt there is no screening: ``c_ion = 0`` returns ``DH_NO_SALT``
    (1e4 nm), so ``exp(-r/dh)`` stays above 0.9998 within the 1.8 nm cutoff of
    the Debye–Hückel force. A finite value is used because ``dh`` is written into
    OpenMM energy expressions as a number.

    Args:
        c_ion (float): Ionic strength (salt concentration) in M, >= 0.
        T (float): Temperature in Kelvin.

    Returns:
        openmm.unit.Quantity: Screening length in nanometers.

    Raises:
        ValueError: If ``c_ion`` is negative.

    Example:
        >>> from HyresBuilder.utils import cal_dh
        >>> dh = cal_dh(0.15, 303.0)   # ~0.8 nm
    """
    if c_ion < 0:
        raise ValueError(f"Salt concentration must be >= 0, got {c_ion} M.")
    if c_ion == 0:
        return DH_NO_SALT*unit.nanometer
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
    Build and initialize a HyRes/iConRNA/iConDNA mixed OpenMM simulation.

    General entry point for protein, nucleic acid, AG, metabolite and polymer
    systems.

    Pipeline: read ``params``; set up the periodic box (NPT/NVT); compute
    ``er`` (:func:`cal_er` scaled by ``er_ref/77.6``) and ``dh``
    (:func:`cal_dh`); load Protein, RNA, DNA, AGs, Metabolite and Polymer
    files via :func:`load_ff` plus optional custom molecules; read PDB/PSF and
    call ``psf.createSystem`` (cutoff 1.2 nm, switch 1.1 nm, ``HBonds``
    constraints, ``CutoffPeriodic`` or ``CutoffNonPeriodic`` for ``'non'``);
    build the force field with :func:`FFs.buildSystem`; add a
    ``MonteCarloBarostat`` (every 25 steps) for NPT; create a
    ``LangevinIntegrator``.

    Args:
        params (argparse.Namespace): Simulation parameters with attributes:

                                     - ``pdb`` (str) — path to input PDB file
                                       (coordinates).
                                     - ``psf`` (str) — path to CHARMM PSF file.
                                     - ``temp`` (float) — temperature in Kelvin.
                                     - ``salt`` (float) — monovalent salt
                                       concentration in mM (converted to M for
                                       :func:`cal_dh`; must be > 0).
                                     - ``lmd`` (float, optional) — scaling of
                                       the phosphate(P)–Mg²⁺ Debye–Hückel
                                       interaction (see :func:`nMg2lmd`);
                                       defaults to 0 if absent.
                                     - ``ens`` (str) — ensemble: ``'NPT'``,
                                       ``'NVT'``, or ``'non'`` (non-periodic).
                                     - ``box`` (list of float) — box lengths
                                       in nm, one value (cubic) or three
                                       (orthorhombic); required for NPT/NVT.
                                     - ``dt`` (Quantity) — integration time step.
                                     - ``er_ref`` (float) — reference dielectric;
                                       ``er = cal_er(temp) * er_ref / 77.6``.
                                     - ``pressure`` (Quantity) — barostat
                                       pressure (used for NPT only).
                                     - ``friction`` (Quantity) — Langevin friction
                                       coefficient.
                                     - ``gpu_id`` (str) — CUDA device index
                                       (e.g. ``'0'``).
                                     - ``custom`` (str or None) — required
                                       attribute; comma-separated names of
                                       custom molecules. For each name ``X``,
                                       ``X.itp`` must exist in the working
                                       directory; it is converted with
                                       :func:`itp2charmm` and ``X.top`` /
                                       ``X.par`` are added to the parameter
                                       set. Falsy values skip this.

        modification (callable, optional): Function ``modification(system)``
                                           passed to :func:`FFs.buildSystem`,
                                           which calls it after its built-in
                                           forces are added. Default ``None``.

    Returns:
        tuple: ``(system, sim)`` — the constructed OpenMM ``System`` and a
        ``Simulation`` on the CUDA platform (mixed precision) with positions
        set from the PDB and velocities drawn at ``temp``.

    Raises:
        SystemExit: Printed error and ``exit(1)`` if ``ens`` is not
                    ``'NPT'``/``'NVT'``/``'non'``, if ``ens == 'non'`` with
                    non-zero ``lmd``, or if ``box`` does not have 1 or 3
                    values. Also exits if a custom ``.itp`` file is missing.

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

    DH_params = {
        'lmd': lmd,                                                # Charge scaling factor of P-Mg interaction
        'dh': dh,                                                  # Debye Huckel screening length
        'er': er,                                                  # relative dielectric constant
    }

    # 4. load force field files
    top_pro, param_pro = load_ff('Protein')
    top_RNA, param_RNA = load_ff('RNA')
    top_DNA, param_DNA = load_ff('DNA')
    top_AGs, param_AGs = load_ff('AGs')
    top_mets, param_mets = load_ff('Metabolite')
    top_poly, param_poly = load_ff('Polymer')
    top_list = [top_pro, top_RNA, top_DNA, top_AGs, top_mets, top_poly]
    param_list = [param_pro, param_RNA, param_DNA, param_AGs, param_mets, param_poly]
    if params.custom:
        custom_list = [mol.strip() for mol in params.custom.split(',')]
        custom_tops = []
        custom_pars = []
        for mol in custom_list:
            itp_file =f'{mol}.itp'
            if not os.path.isfile(itp_file):
                print(f"Error: The custom itp file {itp_file} does not exist.")
                exit(1)
            itp2charmm(itp_file)
            custom_tops.append(f"{mol}.top")
            custom_pars.append(f"{mol}.par")

        top_list = top_list + custom_tops
        param_list = custom_pars + param_list
    ffparams = CharmmParameterSet(*top_list, *param_list)
    # ffparams = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro, top_AGs, param_AGs, top_mets, param_mets)

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
    system = buildSystem(psf, system, DH_params, modification=modification)
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

    integrator = LangevinIntegrator(temperature, friction, dt)
    plat = Platform.getPlatformByName('CUDA')
    prop = {'Precision': 'mixed', 'DeviceIndex': gpu_id}
    sim = Simulation(top, system, integrator, plat, prop)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature)
    print(f'Langevin, CUDA, {temperature}')
    return system, sim


def setup2(args, dt, lmd=0, pressure=1*unit.atmosphere, friction=0.1/unit.picosecond, gpu_id="0"):
    """
    Older argument-style variant of :func:`setup` (HyRes/iConRNA + polymers).

    Loads Protein, RNA and Polymer files only, uses a fixed reference
    dielectric (``er = cal_er(temp) * 60.0 / 77.6``), builds the force field
    with :func:`FFs.buildSystem` (no ``modification`` hook), adds a 25-step
    ``MonteCarloBarostat`` for NPT, and uses a ``LangevinMiddleIntegrator``
    on CUDA (mixed precision). Cutoff 1.2 nm, switch 1.1 nm.

    Args:
        args (argparse.Namespace): Must provide ``pdb``, ``psf``, ``temp`` (K),
            ``salt`` (mM), ``Mg``, ``ens`` (``'NPT'``/``'NVT'``/``'non'``) and,
            for NPT/NVT, ``box`` (1 or 3 lengths in nm). Note: ``args.Mg`` is
            passed directly as the P–Mg²⁺ scaling factor ``lmd``.
        dt (Quantity): Integration time step.
        lmd (float): Unused; overridden by ``args.Mg``. Default ``0``.
        pressure (Quantity): Barostat pressure for NPT. Default 1 atm.
        friction (Quantity): Langevin friction. Default 0.1/ps.
        gpu_id (str): CUDA device index. Default ``"0"``.

    Returns:
        tuple: ``(system, sim)`` — the constructed OpenMM ``System`` and a
        ``Simulation`` on the CUDA platform (mixed precision) with positions
        set from the PDB and velocities drawn at ``args.temp``.

    Raises:
        SystemExit: Printed error and ``exit(1)`` if ``ens`` is not
                    ``'NPT'``/``'NVT'``/``'non'``, if ``ens == 'non'`` with
                    non-zero ``args.Mg``, or if ``box`` does not have 1 or 3
                    values.
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
    DH_params = {
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
    top_poly, param_poly = load_ff('Polymer')
    params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro, top_poly, param_poly)

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
    system = buildSystem(psf, system, DH_params)

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


def rG4s_setup(params, GG=3.0, modification=None):
    """
    Build and initialize an RNA G-quadruplex (rG4) OpenMM simulation.

    Pipeline: read ``params``; set up the periodic box (NPT/NVT); compute
    ``er`` (:func:`cal_er` scaled by ``er_ref/77.6``) and ``dh``
    (:func:`cal_dh`); load RNA, Protein, AGs and Polymer files via
    :func:`load_ff`; read PDB/PSF and call ``psf.createSystem`` (cutoff 1.2
    nm, switch 1.1 nm, ``HBonds`` constraints, ``CutoffPeriodic`` or
    ``CutoffNonPeriodic`` for ``'non'``); build the force field with
    :func:`FFs.rG4sSystem`; add a ``MonteCarloBarostat`` (every 25 steps) for
    NPT; create a ``LangevinMiddleIntegrator``. Note that the ``'RNA'`` files
    (``param_RNA_mix``) are loaded, not ``param_rG4s``.

    Args:
        params (argparse.Namespace): Simulation parameters with attributes:

                                     - ``pdb`` (str) — path to input PDB file
                                       (coordinates).
                                     - ``psf`` (str) — path to CHARMM PSF file.
                                     - ``temp`` (float) — temperature in Kelvin.
                                     - ``salt`` (float) — monovalent salt
                                       concentration in mM (converted to M for
                                       :func:`cal_dh`; must be > 0).
                                     - ``lmd`` (float, optional) — scaling of
                                       the phosphate(P)–Mg²⁺ Debye–Hückel
                                       interaction (see :func:`nMg2lmd`);
                                       defaults to 0 if absent.
                                     - ``ens`` (str) — ensemble: ``'NPT'``,
                                       ``'NVT'``, or ``'non'`` (non-periodic).
                                     - ``box`` (list of float) — box lengths
                                       in nm, one value (cubic) or three
                                       (orthorhombic); required for NPT/NVT.
                                     - ``dt`` (Quantity) — integration time step.
                                     - ``er_ref`` (float) — reference dielectric;
                                       ``er = cal_er(temp) * er_ref / 77.6``.
                                     - ``pressure`` (Quantity) — barostat
                                       pressure (used for NPT only).
                                     - ``friction`` (Quantity) — Langevin friction
                                       coefficient.
                                     - ``gpu_id`` (str) — CUDA device index
                                       (e.g. ``'0'``).

        GG (float): G–G pair interaction strength in kcal/mol, passed to the
                    builder as ``DH_params['GG']``; tune to match experimental
                    Tm. Default ``3.0``.
        modification (callable, optional): Function ``modification(system)``
                                           passed to :func:`FFs.rG4sSystem`,
                                           which calls it after its built-in
                                           forces are added. Default ``None``.

    Returns:
        tuple: ``(system, sim)`` — the constructed OpenMM ``System`` and a
        ``Simulation`` on the CUDA platform (mixed precision) with positions
        set from the PDB and velocities drawn at ``temp``.

    Raises:
        SystemExit: Printed error and ``exit(1)`` if ``ens`` is not
                    ``'NPT'``/``'NVT'``/``'non'``, if ``ens == 'non'`` with
                    non-zero ``lmd``, or if ``box`` does not have 1 or 3
                    values.

    Example:
        >>> from HyresBuilder.utils import rG4s_setup
        >>> system, sim = rG4s_setup(params, GG=3.0)
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

    DH_params = {
        'lmd': lmd,                                                  # Charge scaling factor of P-Mg
        'dh': dh,                                                  # Debye Huckel screening length
        'er': er,                                                  # relative dielectric constant
        'GG': GG,                                                   # G-G pair interaction strength in unit.kilocalorie_per_mole
    }

    # 4. load force field files
    top_pro, param_pro = load_ff('Protein')
    top_RNA, param_RNA = load_ff('RNA')
    #top_DNA, param_DNA = load_ff('DNA')
    top_AGs, param_AGs = load_ff('AGs')
    top_poly, param_poly = load_ff('Polymer')
    ffparams = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro, top_AGs, param_AGs, top_poly, param_poly)

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
    system = rG4sSystem(psf, system, DH_params, modification=modification)
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


def iConRNA_setup(params, modification=None):
    """
    Build and initialize a simulation with the original iConRNA model.

    Pipeline: read ``params``; set up the periodic box (NPT/NVT); compute
    ``er`` (:func:`cal_er` scaled by ``er_ref/77.6``) and ``dh``
    (:func:`cal_dh`); load the original iConRNA files (``top_RNA.inp`` /
    ``param_RNA.inp``) plus Protein, DNA, AGs, Metabolite and Polymer files
    via :func:`load_ff` and optional custom molecules; read PDB/PSF and call
    ``psf.createSystem`` (cutoff 1.8 nm, switch 1.6 nm, ``HBonds``
    constraints, ``CutoffPeriodic`` or ``CutoffNonPeriodic`` for ``'non'``);
    build the force field with :func:`FFs.iConRNASystem`; add a
    ``MonteCarloBarostat`` (every 25 steps) for NPT; create a
    ``LangevinMiddleIntegrator``.

    Args:
        params (argparse.Namespace): Simulation parameters with attributes:

                                     - ``pdb`` (str) — path to input PDB file
                                       (coordinates).
                                     - ``psf`` (str) — path to CHARMM PSF file.
                                     - ``temp`` (float) — temperature in Kelvin.
                                     - ``salt`` (float) — monovalent salt
                                       concentration in mM (converted to M for
                                       :func:`cal_dh`; must be > 0).
                                     - ``lmd`` (float) — scaling of the
                                       phosphate(P)–Mg²⁺ Debye–Hückel
                                       interaction (required attribute).
                                     - ``ens`` (str) — ensemble: ``'NPT'``,
                                       ``'NVT'``, or ``'non'`` (non-periodic).
                                     - ``box`` (list of float) — box lengths
                                       in nm, one value (cubic) or three
                                       (orthorhombic); required for NPT/NVT.
                                     - ``dt`` (Quantity) — integration time step.
                                     - ``er_ref`` (float) — reference dielectric;
                                       ``er = cal_er(temp) * er_ref / 77.6``.
                                     - ``pressure`` (Quantity) — barostat
                                       pressure (used for NPT only).
                                     - ``friction`` (Quantity) — Langevin friction
                                       coefficient.
                                     - ``gpu_id`` (str) — CUDA device index
                                       (e.g. ``'0'``).
                                     - ``custom`` (str or None) — required
                                       attribute; comma-separated names of
                                       custom molecules. For each name ``X``,
                                       ``X.itp`` must exist in the working
                                       directory; it is converted with
                                       :func:`itp2charmm` and ``X.top`` /
                                       ``X.par`` are added to the parameter
                                       set. Falsy values skip this.

        modification (callable, optional): Function ``modification(system)``
                                           passed to :func:`FFs.iConRNASystem`,
                                           which calls it after its built-in
                                           forces are added. Default ``None``.

    Returns:
        tuple: ``(system, sim)`` — the constructed OpenMM ``System`` and a
        ``Simulation`` on the CUDA platform (mixed precision) with positions
        set from the PDB and velocities drawn at ``temp``.

    Raises:
        SystemExit: Printed error and ``exit(1)`` if ``ens`` is not
                    ``'NPT'``/``'NVT'``/``'non'``, if ``ens == 'non'`` with
                    non-zero ``lmd``, or if ``box`` does not have 1 or 3
                    values. Also exits if a custom ``.itp`` file is missing.

    Example:
        >>> from HyresBuilder.utils import iConRNA_setup
        >>> params.lmd = 0.0
        >>> system, sim = iConRNA_setup(params)
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

    # Debye-Hückel parameters
    DH_params = {
        'lmd': lmd,                                                  # Charge scaling factor of P-
        'dh': dh,                                                    # Debye Huckel screening length
        'er': er,                                                    # relative dielectric constant
    }

    # 4. load force field files
    path1 = files("HyresBuilder") / "forcefield" / "top_RNA.inp"
    top_RNA = path1.as_posix()
    path2 = files("HyresBuilder") / "forcefield" / "param_RNA.inp"
    param_RNA = path2.as_posix()
    top_AGs, param_AGs = load_ff('AGs')
    top_pro, param_pro = load_ff('Protein')
    top_DNA, param_DNA = load_ff('DNA')
    top_mets, param_mets = load_ff('Metabolite')
    top_poly, param_poly = load_ff('Polymer')
    top_list = [top_pro, top_RNA, top_DNA, top_AGs, top_mets, top_poly]
    param_list = [param_pro, param_RNA, param_DNA, param_AGs, param_mets, param_poly]
    if params.custom:
        custom_list = [mol.strip() for mol in params.custom.split(',')]
        custom_tops = []
        custom_pars = []
        for mol in custom_list:
            itp_file =f'{mol}.itp'
            if not os.path.isfile(itp_file):
                print(f"Error: The custom itp file {itp_file} does not exist.")
                exit(1)
            itp2charmm(itp_file)
            custom_tops.append(f"{mol}.top")
            custom_pars.append(f"{mol}.par")

        top_list = top_list + custom_tops
        param_list = custom_pars + param_list
    ffparams = CharmmParameterSet(*top_list, *param_list)

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
    system = iConRNASystem(psf, system, DH_params, modification=modification)
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


def setupMg(params, modification=None):
    """
    Build and initialize a simulation with explicit Mg²⁺/Ca²⁺ ions.

    Same as :func:`setup` except that it uses :func:`FFs.buildMgSystem`
    (which applies a fixed er = 20 to Debye–Hückel pairs among backbone ``P``
    and ``MG``/``CAL`` beads and ``er`` to all other pairs), does not support
    custom ``.itp`` molecules, and uses a ``LangevinMiddleIntegrator``.

    Pipeline: read ``params``; set up the periodic box (NPT/NVT); compute
    ``er`` (:func:`cal_er` scaled by ``er_ref/77.6``) and ``dh``
    (:func:`cal_dh`); load RNA, Protein, AGs, Metabolite and Polymer files via
    :func:`load_ff`; read PDB/PSF and call ``psf.createSystem`` (cutoff 1.2
    nm, switch 1.1 nm, ``HBonds`` constraints, ``CutoffPeriodic`` or
    ``CutoffNonPeriodic`` for ``'non'``); build the force field with
    :func:`FFs.buildMgSystem`; add a ``MonteCarloBarostat`` (every 25 steps)
    for NPT; create a ``LangevinMiddleIntegrator``.

    Args:
        params (argparse.Namespace): Simulation parameters with attributes:

                                     - ``pdb`` (str) — path to input PDB file
                                       (coordinates).
                                     - ``psf`` (str) — path to CHARMM PSF file.
                                     - ``temp`` (float) — temperature in Kelvin.
                                     - ``salt`` (float) — monovalent salt
                                       concentration in mM (converted to M for
                                       :func:`cal_dh`; must be > 0).
                                     - ``lmd`` (float, optional) — scaling of
                                       the phosphate(P)–Mg²⁺ Debye–Hückel
                                       interaction (see :func:`nMg2lmd`);
                                       defaults to 0 if absent.
                                     - ``ens`` (str) — ensemble: ``'NPT'``,
                                       ``'NVT'``, or ``'non'`` (non-periodic).
                                     - ``box`` (list of float) — box lengths
                                       in nm, one value (cubic) or three
                                       (orthorhombic); required for NPT/NVT.
                                     - ``dt`` (Quantity) — integration time step.
                                     - ``er_ref`` (float) — reference dielectric;
                                       ``er = cal_er(temp) * er_ref / 77.6``.
                                     - ``pressure`` (Quantity) — barostat
                                       pressure (used for NPT only).
                                     - ``friction`` (Quantity) — Langevin friction
                                       coefficient.
                                     - ``gpu_id`` (str) — CUDA device index
                                       (e.g. ``'0'``).

        modification (callable, optional): Function ``modification(system)``
                                           passed to :func:`FFs.buildMgSystem`,
                                           which calls it after its built-in
                                           forces are added. Default ``None``.

    Returns:
        tuple: ``(system, sim)`` — the constructed OpenMM ``System`` and a
        ``Simulation`` on the CUDA platform (mixed precision) with positions
        set from the PDB and velocities drawn at ``temp``.

    Raises:
        SystemExit: Printed error and ``exit(1)`` if ``ens`` is not
                    ``'NPT'``/``'NVT'``/``'non'``, if ``ens == 'non'`` with
                    non-zero ``lmd``, or if ``box`` does not have 1 or 3
                    values.

    Example:
        >>> from HyresBuilder.utils import setupMg
        >>> system, sim = setupMg(params)
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

    DH_params = {
        'lmd': lmd,                                                  # Charge scaling factor of P-Mg
        'dh': dh,                                                  # Debye Huckel screening length
        'er': er,                                                  # relative dielectric constant
    }

    # 4. load force field files
    top_pro, param_pro = load_ff('Protein')
    top_RNA, param_RNA = load_ff('RNA')
    top_DNA, param_DNA = load_ff('DNA')
    top_AGs, param_AGs = load_ff('AGs')
    top_mets, param_mets = load_ff('Metabolite')
    top_poly, param_poly = load_ff('Polymer')
    ffparams = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro, top_AGs, param_AGs, top_mets, param_mets, top_poly, param_poly)

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
    system = buildMgSystem(psf, system, DH_params, modification=modification)
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


def iConDNA_setup(params, modification=None):
    """
    Build and initialize a HyRes/iConDNA OpenMM simulation.

    Pipeline: read ``params``; set up the periodic box (NPT/NVT); compute
    ``er`` (:func:`cal_er` scaled by ``er_ref/77.6``) and ``dh``
    (:func:`cal_dh`); load DNA, Protein, AGs, Metabolite and Polymer files via
    :func:`load_ff` (no RNA files); read PDB/PSF and call ``psf.createSystem``
    (cutoff 1.2 nm, switch 1.1 nm, ``HBonds`` constraints, ``CutoffPeriodic``
    or ``CutoffNonPeriodic`` for ``'non'``); build the force field with
    :func:`FFs.iConDNASystem`; add a ``MonteCarloBarostat`` (every 25 steps)
    for NPT; create a ``LangevinIntegrator``.

    Args:
        params (argparse.Namespace): Simulation parameters with attributes:

                                     - ``pdb`` (str) — path to input PDB file
                                       (coordinates).
                                     - ``psf`` (str) — path to CHARMM PSF file.
                                     - ``temp`` (float) — temperature in Kelvin.
                                     - ``salt`` (float) — monovalent salt
                                       concentration in mM (converted to M for
                                       :func:`cal_dh`; must be > 0).
                                     - ``lmd`` (float, optional) — scaling of
                                       the phosphate(P)–Mg²⁺ Debye–Hückel
                                       interaction (see :func:`nMg2lmd`);
                                       defaults to 0 if absent.
                                     - ``ens`` (str) — ensemble: ``'NPT'``,
                                       ``'NVT'``, or ``'non'`` (non-periodic).
                                     - ``box`` (list of float) — box lengths
                                       in nm, one value (cubic) or three
                                       (orthorhombic); required for NPT/NVT.
                                     - ``dt`` (Quantity) — integration time step.
                                     - ``er_ref`` (float) — reference dielectric;
                                       ``er = cal_er(temp) * er_ref / 77.6``.
                                     - ``pressure`` (Quantity) — barostat
                                       pressure (used for NPT only).
                                     - ``friction`` (Quantity) — Langevin friction
                                       coefficient.
                                     - ``gpu_id`` (str) — CUDA device index
                                       (e.g. ``'0'``).

        modification (callable, optional): Function ``modification(system)``
                                           passed to :func:`FFs.iConDNASystem`,
                                           which calls it after its built-in
                                           forces are added. Default ``None``.

    Returns:
        tuple: ``(system, sim)`` — the constructed OpenMM ``System`` and a
        ``Simulation`` on the CUDA platform (mixed precision) with positions
        set from the PDB and velocities drawn at ``temp``.

    Raises:
        SystemExit: Printed error and ``exit(1)`` if ``ens`` is not
                    ``'NPT'``/``'NVT'``/``'non'``, if ``ens == 'non'`` with
                    non-zero ``lmd``, or if ``box`` does not have 1 or 3
                    values.

    Example:
        >>> from HyresBuilder.utils import iConDNA_setup
        >>> system, sim = iConDNA_setup(params)
    """
    
    print('\n################## set up simulation parameters ###################')
    # 1. input parameters
    pdb_file = params.pdb
    psf_file = params.psf
    T = params.temp
    c_ion = params.salt/1000.0                                   # concentration of ions in M
    lmd = getattr(params, "lmd", 0)                              # lmd for Mg²⁺-DNA interaction, if don't give, it's 0.
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
    print(f'Mg-DNA interaction: lmd = {lmd:.2f}')

    DH_params = {
        'lmd': lmd,                                                # Charge scaling factor of P-Mg interaction
        'dh': dh,                                                  # Debye Huckel screening length
        'er': er,                                                  # relative dielectric constant
    }

    # 4. load force field files
    top_pro, param_pro = load_ff('Protein')
    top_DNA, param_DNA = load_ff('DNA')
    top_AGs, param_AGs = load_ff('AGs')
    top_mets, param_mets = load_ff('Metabolite')
    top_poly, param_poly = load_ff('Polymer')
    ffparams = CharmmParameterSet(top_DNA, param_DNA, top_pro, param_pro, top_AGs, param_AGs, top_mets, param_mets, top_poly, param_poly)

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
    system = iConDNASystem(psf, system, DH_params, modification=modification)
    print("buildSystem for HyRes_iConDNA")

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

    integrator = LangevinIntegrator(temperature, friction, dt)
    plat = Platform.getPlatformByName('CUDA')
    prop = {'Precision': 'mixed', 'DeviceIndex': gpu_id}
    sim = Simulation(top, system, integrator, plat, prop)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature)
    print(f'Langevin, CUDA, {temperature}')
    return system, sim


def crowding_effect(system: System, crowding_factor: float = 1.0) -> None:
    """
    Scale the LJ well depth of the "LJ Force w/ NBFIX" force to mimic crowding.

    Rewrites the energy function of the ``CustomNonbondedForce`` named
    ``'LJ Force w/ NBFIX'`` (the NBFIX LJ force that the FFs builders rename;
    it exists only if the parameter set contains NBFIX terms) and adds a
    global parameter ``crowding_factor``:

    - Original energy: ``(a/r6)^2 - b/r6``
    - New energy:      ``(a*sqrt(crowding_factor)/r6)^2 - b*crowding_factor/r6``

    Since ``a`` scales as sqrt(epsilon) and ``b`` as epsilon, this multiplies
    the effective LJ epsilon of every pair by ``crowding_factor`` while
    leaving sigma unchanged. The new expression replaces the original one
    verbatim (it assumes the standard ``acoef``/``bcoef`` NBFIX form).

    Must be called BEFORE creating a ``Context``/``Simulation``; changes made
    afterwards do not affect an existing Context. The parameter can later be
    changed with ``context.setParameter('crowding_factor', value)``.

    Args:
        system (System): System containing the target force; modified in place.
        crowding_factor (float): Initial value of ``crowding_factor``
            (must be >= 0). 1.0 = original epsilon; >1 strengthens and <1
            weakens LJ interactions; 0 disables LJ. Default ``1.0``.

    Returns:
        CustomNonbondedForce: The modified force (despite the ``-> None``
        annotation).

    Raises:
        ValueError: If ``crowding_factor`` is negative, if no or more than one
                    ``CustomNonbondedForce`` named ``'LJ Force w/ NBFIX'`` is
                    found, or if the force already has a ``crowding_factor``
                    global parameter.

    Example:
        >>> from HyresBuilder.utils import setup, crowding_effect
        >>> def mod(system):
        ...     crowding_effect(system, crowding_factor=1.4)
        >>> system, sim = setup(params, modification=mod)
    """
    if crowding_factor < 0:
        raise ValueError(f"crowding_factor must be >= 0, got {crowding_factor}")

    force_name = "LJ Force w/ NBFIX"
    param_name = "crowding_factor"

    matches = [
        f for f in system.getForces()
        if isinstance(f, CustomNonbondedForce) and f.getName() == force_name
    ]

    if not matches:
        available = sorted({f.getName() for f in system.getForces()})
        raise ValueError(
            f"No CustomNonbondedForce named '{force_name}' found in System. "
            f"Forces present: {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one CustomNonbondedForce named '{force_name}', "
            f"found {len(matches)}. Force names must be unique for this to work reliably."
        )

    target_force = matches[0]

    existing_params = {
        target_force.getGlobalParameterName(i)
        for i in range(target_force.getNumGlobalParameters())
    }
    if param_name in existing_params:
        raise ValueError(
            f"Global parameter '{param_name}' already exists on '{force_name}'. "
            "crowding_effect() may have already been called on this System."
        )

    original_energy = target_force.getEnergyFunction()

    new_energy = (
        f"(a*sqrt({param_name})/r6)^2 - b*{param_name}/r6; "
        "r6=r^6; a=acoef(type1, type2); b=bcoef(type1, type2)"
    )

    target_force.setEnergyFunction(new_energy)
    target_force.addGlobalParameter(param_name, crowding_factor)

    print(
        f"[crowding_effect] '{force_name}': energy function updated, "
        f"'{param_name}' added (initial value = {crowding_factor}).\n"
        f"  Original energy: {original_energy}\n"
        f"  New energy:      {new_energy}"
    )

    return target_force