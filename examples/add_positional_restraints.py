# add restraints
print("add restraints")
u = mda.Universe(parser.parse_args().psf, parser.parse_args().pdb)
sel = u.select_atoms('segid PA0')
cas = u.select_atoms('segid PA0 and name CA')
run = DSSP(sel).run()
result = run.results.dssp[0]

strcutre_CA = []
for ca, s in zip(cas, result):
    if s in ['E', 'H']:
        strcutre_CA.append(ca.index)

pos_restraint = CustomExternalForce('kg*((x-x0)^2 + (y-y0)^2 +(z-z0)^2);')
pos_restraint.addGlobalParameter('kg', 400*kilojoules_per_mole/unit.nanometer)
pos_restraint.addPerParticleParameter('x0')
pos_restraint.addPerParticleParameter('y0')
pos_restraint.addPerParticleParameter('z0')
crds = u.atoms.positions/10
for atom in strcutre_CA:
    pos = crds[atom]
    pos_restraint.addParticle(atom, pos)
system.addForce(pos_restraint)

sim.context.reinitialize(preserveState=True)
