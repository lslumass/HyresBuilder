"""
HyresBuilder.Relax -- soft-core pre-relaxation of an OpenMM Simulation.

relax_simulation(sim, system) pre-relaxes the CURRENT positions of a
Simulation before the regular minimisation. A copy of the System keeps every
bonded term, constraint and virtual site; all nonbonded forces are replaced
by a soft repulsion  k (sigma - r)^2  that honours the System's own
exclusions. It stays finite when particles overlap, so it removes the
overlaps that make the real force field produce NaN. Optionally, chosen
particles are held by positional restraints. The relaxed positions are
written back into sim.context; the simulation's System is not modified.
Works for HyRes, iConRNA / iConDNA and all-atom models.

Typical use in a run script
---------------------------
    from HyresBuilder import utils, Relax
    system, sim = utils.setup(params)
    Relax.relax_simulation(sim, system)
    sim.minimizeEnergy(maxIterations=500000, tolerance=0.01)
"""

import time

import numpy as np
from scipy.spatial import cKDTree

import openmm as mm
from openmm import unit


# Forces whose energy diverges at short range (or that only make sense with
# the full nonbonded model) are replaced by the soft repulsion.
_NONBONDED_TYPES = tuple(t for t in (
    getattr(mm, n, None) for n in (
        'NonbondedForce', 'CustomNonbondedForce', 'CustomGBForce', 'GBSAOBCForce',
        'CustomHbondForce', 'CustomManyParticleForce', 'AmoebaVdwForce',
        'AmoebaMultipoleForce', 'AmoebaGeneralizedKirkwoodForce', 'AmoebaWcaDispersionForce'))
    if t is not None)
_DROP_TYPES = tuple(t for t in (
    getattr(mm, n, None) for n in (
        'CMMotionRemover', 'MonteCarloBarostat', 'MonteCarloAnisotropicBarostat',
        'MonteCarloMembraneBarostat', 'MonteCarloFlexibleBarostat', 'AndersenThermostat'))
    if t is not None)


def _exclusions(system, topology=None):
    """Sorted int64 keys i*N+j (i<j) of every pair the soft repulsion must skip:
    excluded / exception pairs of the System's nonbonded forces, plus bonded
    1-2 (bonds, constraints, topology), 1-3 (angles) and 1-4 (torsions)
    pairs, so bonded neighbours are never pushed apart even when the System
    keeps no nonbonded exclusion list."""
    n = system.getNumParticles()
    pairs = []
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            pairs += [f.getExceptionParameters(k)[:2] for k in range(f.getNumExceptions())]
        elif isinstance(f, mm.CustomNonbondedForce):
            pairs += [f.getExclusionParticles(k) for k in range(f.getNumExclusions())]
        elif isinstance(f, mm.HarmonicBondForce):
            pairs += [f.getBondParameters(k)[:2] for k in range(f.getNumBonds())]
        elif isinstance(f, mm.HarmonicAngleForce):
            pairs += [f.getAngleParameters(k)[0:3:2] for k in range(f.getNumAngles())]
        elif isinstance(f, mm.CustomAngleForce):
            pairs += [f.getAngleParameters(k)[0:3:2] for k in range(f.getNumAngles())]
        elif isinstance(f, (mm.PeriodicTorsionForce, mm.RBTorsionForce)):
            get = f.getTorsionParameters
            pairs += [get(k)[0:4:3] for k in range(f.getNumTorsions())]
        elif isinstance(f, mm.CustomTorsionForce):
            pairs += [f.getTorsionParameters(k)[0:4:3] for k in range(f.getNumTorsions())]
    pairs += [system.getConstraintParameters(k)[:2] for k in range(system.getNumConstraints())]
    if topology is not None:
        pairs += [(a.index, b.index) for a, b in topology.bonds()]
    if not pairs:
        return np.zeros(0, np.int64)
    p = np.array(pairs, np.int64).reshape(-1, 2)
    i, j = p.min(axis=1), p.max(axis=1)
    keep = i != j
    return np.unique(i[keep] * n + j[keep])


def _orthorhombic(box):
    return box is not None and abs(box[1][0]) + abs(box[2][0]) + abs(box[2][1]) < 1e-9


def _close_pairs(pos_nm, cutoff_nm, excluded, box=None):
    """Non-excluded particle pairs closer than cutoff (minimum image for an
    orthorhombic box). Returns (count, min distance in nm or None)."""
    pos_nm = np.asarray(pos_nm, float)
    n_part = len(pos_nm)
    if _orthorhombic(box):
        L = np.array([box[0][0], box[1][1], box[2][2]], float)
        tree = cKDTree(np.mod(pos_nm, L), boxsize=L)
    else:
        L = None   # non-periodic, or triclinic (contacts across the boundary not counted)
        tree = cKDTree(pos_nm)
    pairs = tree.query_pairs(cutoff_nm, output_type='ndarray')
    if not len(pairs):
        return 0, None
    i, j = pairs.min(axis=1).astype(np.int64), pairs.max(axis=1).astype(np.int64)
    keep = ~np.isin(i * n_part + j, excluded, assume_unique=False)
    i, j = i[keep], j[keep]
    if not len(i):
        return 0, None
    d = pos_nm[i] - pos_nm[j]
    if L is not None:
        d -= L * np.round(d / L)
    return len(i), float(np.sqrt((d * d).sum(axis=1)).min())


def _context_like(system, integrator, ref_context):
    """Context on the same platform / device as the running simulation."""
    plat = ref_context.getPlatform()
    props = {}
    for name in plat.getPropertyNames():
        try:
            props[name] = plat.getPropertyValue(ref_context, name)
        except Exception:
            pass
    for attempt in (props, {k: v for k, v in props.items() if 'DeviceIndex' in k}, None):
        try:
            if attempt is None:
                print(f"# WARNING: could not open a {plat.getName()} context for the relaxation; using CPU")
                return mm.Context(system, integrator, mm.Platform.getPlatformByName('CPU'))
            return mm.Context(system, integrator, plat, attempt)
        except Exception:
            integrator = mm.VerletIntegrator(0.001)
    raise RuntimeError("could not create a context for the relaxation")


RELAX_SIGMA = 0.18   # nm, soft-core radius used by relax_simulation
MAX_SIGMA = 0.5      # nm; larger values push apart normal contacts (sigma is in nm, not A)


def relax_simulation(simulation, system=None, sigma=RELAX_SIGMA, k_rep=1.0e4, restrain=None, k_restrain=1000.0,
                     max_iter=20000, tolerance=10.0, verbose=True):
    """Soft-core pre-relaxation of the current positions of `simulation`.

    Parameters
    ----------
    simulation : openmm.app.Simulation   positions are read from and written back to it
    system     : openmm.System           defaults to simulation.system
    sigma      : float, NANOMETRES (0 < sigma <= 0.5). Non-bonded pairs closer
                 than this are pushed apart. Keep it below the normal contact
                 distances of the model: 0.18 nm (default) removes the overlaps
                 that crash a simulation without disturbing hydrogen-bond or
                 bead-contact distances; up to ~0.3 nm for CG beads.
    k_rep      : soft repulsion strength, kJ/mol/nm^2  (E = k_rep (sigma - r)^2)
    restrain   : None, a list of particle indices, or a function atom -> bool
                 (openmm.app.Atom from simulation.topology) selecting particles
                 held at their start positions, e.g. the deposited structure.
    k_restrain : restraint strength, kJ/mol/nm^2
    Returns a dict with before/after counts of non-excluded pairs < 0.9 sigma.
    """
    if not 0 < sigma <= MAX_SIGMA:
        raise ValueError(f"sigma = {sigma} nm: must be in (0, {MAX_SIGMA}] nm (sigma is in nm; "
                         f"{sigma} A would be sigma={sigma / 10:g})")
    t0 = time.time()
    system = simulation.system if system is None else system
    ctx = simulation.context
    st = ctx.getState(getPositions=True)
    pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    periodic = system.usesPeriodicBoundaryConditions()
    box = st.getPeriodicBoxVectors().value_in_unit(unit.nanometer) if periodic else None
    excluded = _exclusions(system, simulation.topology)
    detect = 0.9 * sigma   # minimisation leaves pairs a hair inside sigma; report real overlaps
    n0, d0 = _close_pairs(np.asarray(pos), detect, excluded, box)

    # copy of the System: bonded terms, constraints, virtual sites kept
    relax = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    removed = []
    for k in reversed(range(relax.getNumForces())):
        f = relax.getForce(k)
        if isinstance(f, _NONBONDED_TYPES) or isinstance(f, _DROP_TYPES):
            removed.append(type(f).__name__)
            relax.removeForce(k)

    rep = mm.CustomNonbondedForce('k_rep*(s_rep - r)^2*step(s_rep - r)')
    rep.addGlobalParameter('k_rep', k_rep)
    rep.addGlobalParameter('s_rep', sigma)
    rep.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic if periodic
                           else mm.CustomNonbondedForce.CutoffNonPeriodic)
    rep.setCutoffDistance(sigma)
    for _ in range(system.getNumParticles()):
        rep.addParticle([])
    n_part = system.getNumParticles()
    for key in excluded.tolist():
        rep.addExclusion(key // n_part, key % n_part)
    relax.addForce(rep)

    n_rst = 0
    if restrain is not None:
        if callable(restrain):
            idx = [a.index for a in simulation.topology.atoms() if restrain(a)]
        else:
            idx = [int(i) for i in restrain]
        expr = ('0.5*k_rst*periodicdistance(x, y, z, x0, y0, z0)^2' if periodic
                else '0.5*k_rst*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
        rst = mm.CustomExternalForce(expr)
        rst.addGlobalParameter('k_rst', k_restrain)
        for p in ('x0', 'y0', 'z0'):
            rst.addPerParticleParameter(p)
        for i in idx:
            if not system.isVirtualSite(i):
                rst.addParticle(i, [float(v) for v in pos[i]])
        n_rst = rst.getNumParticles()
        if n_rst:
            relax.addForce(rst)

    rctx = _context_like(relax, mm.VerletIntegrator(0.001), ctx)
    if periodic:
        rctx.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
    rctx.setPositions(st.getPositions())
    e0 = rctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    mm.LocalEnergyMinimizer.minimize(rctx, tolerance, max_iter)
    rs = rctx.getState(getPositions=True, getEnergy=True)
    e1 = rs.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    new = rs.getPositions(asNumpy=True)
    if not np.isfinite(new.value_in_unit(unit.nanometer)).all():
        raise RuntimeError("soft-core relaxation produced non-finite coordinates")
    ctx.setPositions(new)
    if any(system.isVirtualSite(i) for i in range(n_part)):
        ctx.computeVirtualSites()
    n1, d1 = _close_pairs(np.asarray(new.value_in_unit(unit.nanometer)), detect, excluded, box)
    shift = np.linalg.norm(np.asarray(new.value_in_unit(unit.nanometer)) - np.asarray(pos), axis=1)
    out = {'pairs_before': n0, 'pairs_after': n1, 'min_before_nm': d0, 'min_after_nm': d1,
           'max_shift_nm': float(shift.max()), 'energy_before': e0, 'energy_after': e1}
    if verbose:
        fmt = lambda d: 'none' if d is None else f"{d * 10:.2f} A"
        print(f"# Soft-core pre-relaxation ({rctx.getPlatform().getName()}, sigma {sigma * 10:.2f} A, "
              f"replaced: {', '.join(sorted(set(removed))) or 'none'}; {n_rst} restrained particles)")
        print(f"    pairs closer than {detect * 10:.2f} A: {n0} -> {n1}   (closest: {fmt(d0)} -> {fmt(d1)})")
        print(f"    soft energy {e0:.4g} -> {e1:.4g} kJ/mol; max particle shift {shift.max() * 10:.2f} A; "
              f"{time.time() - t0:.1f} s")
    del rctx
    return out