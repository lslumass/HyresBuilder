"""
Pure-Python PSF builder for the HyRes / iConRNA / iConDNA topologies.

A drop-in replacement for the subset of ``psfgen.PsfGen`` that HyresBuilder
uses (``GenPsf``, ``Convert2CG``, ``Mutator``). It reads CHARMM topology
(RTF) files, builds segments from PDB files, assigns coordinates and writes
X-PLOR PSF and PDB files.

Supported topology records: ``MASS``, ``DECL``, ``DEFA``, ``AUTOGENERATE``,
``RESI``, ``GROUP``, ``ATOM``, ``BOND``/``DOUBLE``, ``ANGL``/``THET``,
``DIHE``/``PHI``, ``IMPR``/``IMPH``, ``CMAP``, ``DONO``, ``ACCE``. ``IC``
lines and ``PRES`` patches are read but not applied (the HyresBuilder
topologies do not use them). Inter-residue atoms are written ``-NAME`` and
``+NAME``; terms referring to a residue beyond the segment ends are dropped.

Example:
    >>> from HyresBuilder.psfbuilder import PsfBuilder
    >>> gen = PsfBuilder()
    >>> gen.read_topology('top_hyres_mix.inp')
    >>> gen.add_segment(segid='P001', pdbfile='protein.pdb', auto_angles=False)
    >>> gen.read_coords(segid='P001', filename='protein.pdb')
    >>> gen.write_psf('protein.psf')
    >>> gen.write_pdb('protein_out.pdb')
"""

import numpy as np

__all__ = ['PsfBuilder', 'PsfGen']


class _ResidueDef:
    """One RESI (or PRES) entry of a topology file."""

    def __init__(self, name, charge, is_patch=False):
        self.name = name
        self.charge = charge
        self.is_patch = is_patch
        self.atoms = []          # [name, type, charge]
        self.bonds = []          # tuples of atom names, possibly '-X' / '+X'
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.cmaps = []
        self.donors = []
        self.acceptors = []


class _Atom:
    __slots__ = ('name', 'type', 'charge', 'mass', 'pos', 'occupancy', 'element')

    def __init__(self, name, type_, charge, mass):
        self.name = name
        self.type = type_
        self.charge = charge
        self.mass = mass
        self.pos = None
        self.occupancy = -1.0    # PDB occupancy: 1 read/set, 0 guessed, -1 unknown
        self.element = ''


class _Residue:
    __slots__ = ('resid', 'resname', 'chain', 'atoms')

    def __init__(self, resid, resname, chain):
        self.resid = resid
        self.resname = resname
        self.chain = chain
        self.atoms = {}          # name -> _Atom, in topology order


class _Segment:
    def __init__(self, segid, auto_angles, auto_dihedrals):
        self.segid = segid
        self.auto_angles = auto_angles
        self.auto_dihedrals = auto_dihedrals
        self.residues = []       # list of _Residue
        self.bonds = []          # tuples of _Atom
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.cmaps = []
        self.donors = []
        self.acceptors = []
        self.order = {}          # attr -> (residue position, topology position) of each topology term
        self.generated = set()   # attrs generated from the bond graph instead of the topology

    def residue(self, resid):
        for res in self.residues:
            if res.resid == resid:
                return res
        raise KeyError(f"no residue {resid} in segment {self.segid}")


def _serial(n):
    """PDB atom serial in 5 columns; hybrid-36 (A0000..., then a0000...) beyond 99999."""
    if n < 100000:
        return f"{n:5d}"
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    n -= 100000
    if n >= 26 * 36**4:
        n -= 26 * 36**4
        digits = digits.lower()
    n += 10 * 36**4
    out = ''
    for _ in range(5):
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out


def _read_pdb_atoms(filename):
    """Yield (name, resname, chain, resid, xyz, element) for ATOM/HETATM records."""
    with open(filename) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                yield (line[12:16].strip(), line[17:21].strip(), line[21:22],
                       line[22:27].strip(),
                       (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                       line[76:78].strip())


class PsfBuilder:
    """Build CHARMM/X-PLOR PSF files from topology files and PDB coordinates.

    The method names and arguments follow ``psfgen.PsfGen`` so existing code
    only needs a different import.
    """

    def __init__(self, output=None):
        # ``output`` is accepted for psfgen compatibility and ignored
        self.masses = {}
        self.residue_defs = {}
        self.segments = []
        self.topology_files = []

    # ------------------------------------------------------------------
    # topology
    # ------------------------------------------------------------------
    def read_topology(self, filename):
        """Read a CHARMM topology (RTF) file.

        Duplicates follow psfgen: the first definition of a residue wins
        (later ones are ignored), while a later ``MASS`` entry for an atom
        type replaces the earlier one, also for residues read before it.
        """
        self.topology_files.append(str(filename))
        current = None
        with open(filename) as f:
            for raw in f:
                line = raw.split('!')[0].strip()
                if not line or line.startswith('*'):
                    continue
                tok = line.split()
                key = tok[0].upper()[:4]
                if key == 'MASS':
                    self.masses[tok[2].upper()] = float(tok[3])
                elif key in ('RESI', 'PRES'):
                    charge = float(tok[2]) if len(tok) > 2 else 0.0
                    current = _ResidueDef(tok[1].upper(), charge, is_patch=(key == 'PRES'))
                    if key == 'RESI':
                        self.residue_defs.setdefault(current.name, current)
                elif key == 'END':
                    current = None
                elif current is None:
                    continue            # DECL, DEFA, AUTO, header lines
                elif key == 'ATOM':
                    current.atoms.append([tok[1].upper(), tok[2].upper(), float(tok[3])])
                elif key in ('BOND', 'DOUB', 'TRIP'):
                    current.bonds += self._groups(tok[1:], 2)
                elif key in ('ANGL', 'THET'):
                    current.angles += self._groups(tok[1:], 3)
                elif key in ('DIHE', 'PHI'):
                    current.dihedrals += self._groups(tok[1:], 4)
                elif key in ('IMPR', 'IMPH'):
                    current.impropers += self._groups(tok[1:], 4)
                elif key == 'CMAP':
                    current.cmaps += self._groups(tok[1:], 8)
                elif key == 'DONO':
                    current.donors += self._groups(tok[1:], 2)
                elif key == 'ACCE':
                    current.acceptors += self._groups(tok[1:], 2)
                # GROUP, IC, DELE, PATC, BILD, LONE: ignored

    @staticmethod
    def _groups(names, n):
        names = [x.upper() for x in names]
        if len(names) % n:
            raise ValueError(f"topology entry needs a multiple of {n} atoms: {' '.join(names)}")
        return [tuple(names[i:i + n]) for i in range(0, len(names), n)]

    def alias_residue(self, top_resname, pdb_resname):
        """Use the topology residue ``top_resname`` for PDB residues named ``pdb_resname``."""
        self.residue_defs[pdb_resname.upper()] = self.residue_defs[top_resname.upper()]

    # ------------------------------------------------------------------
    # segments
    # ------------------------------------------------------------------
    def add_segment(self, segid, first='none', last='none', pdbfile=None,
                    auto_angles=True, auto_dihedrals=True, residues=None, mutate=None):
        """Build a segment from the residues of ``pdbfile`` (or a ``residues`` list).

        Args:
            segid (str): Segment ID.
            first, last (str): Terminal patches; only ``'none'`` is supported.
            pdbfile (str): PDB file; residues are taken in order of first
                appearance, keyed by resid. Coordinates are not read here
                (use :meth:`read_coords`).
            auto_angles (bool): Generate all angles from the bond graph instead
                of the topology ``ANGL`` entries.
            auto_dihedrals (bool): Generate dihedrals from pairs of angles
                (topology or generated ones) instead of the topology ``DIHE``
                entries.
            residues (list of (resid, resname)): Used when ``pdbfile`` is None.
            mutate (list of (resid, resname)): Residues to rename before building.
        """
        for p in (first, last):
            if p is not None and str(p).lower() != 'none':
                raise NotImplementedError(f"terminal patch '{p}' is not supported")
        if any(s.segid == segid for s in self.segments):
            raise ValueError(f"duplicate segment {segid}")

        reslist = []
        if pdbfile is not None:
            seen = set()
            for name, resname, chain, resid, xyz, _ in _read_pdb_atoms(pdbfile):
                if resid not in seen:
                    seen.add(resid)
                    reslist.append([resid, resname.upper(), chain])
        elif residues is not None:
            reslist = [[str(r), n.upper(), ' '] for r, n in residues]
        else:
            raise ValueError("add_segment needs pdbfile or residues")

        if mutate:
            new = {str(r): n.upper() for r, n in mutate}
            for item in reslist:
                if item[0] in new:
                    item[1] = new.pop(item[0])
            if new:
                raise ValueError(f"mutated residues not in segment {segid}: {list(new)}")

        seg = _Segment(segid, auto_angles, auto_dihedrals)
        for resid, resname, chain in reslist:
            rdef = self.residue_defs.get(resname)
            if rdef is None:
                raise KeyError(f"unknown residue type {resname} (segment {segid}, resid {resid})")
            res = _Residue(resid, resname, chain)
            for aname, atype, charge in rdef.atoms:
                if atype not in self.masses:
                    raise KeyError(f"no MASS entry for atom type {atype} ({resname} {aname})")
                res.atoms[aname] = _Atom(aname, atype, charge, self.masses[atype])
            seg.residues.append(res)

        # resolve the topology terms residue by residue
        for i, res in enumerate(seg.residues):
            rdef = self.residue_defs[res.resname]
            for attr in ('bonds', 'angles', 'dihedrals', 'impropers', 'cmaps', 'donors', 'acceptors'):
                if attr == 'angles' and auto_angles:
                    continue
                if attr == 'dihedrals' and auto_dihedrals:
                    continue
                out = getattr(seg, attr)
                order = seg.order.setdefault(attr, [])
                for pos, names in enumerate(getattr(rdef, attr)):
                    atoms = [self._resolve(seg, i, n) for n in names]
                    if all(a is not None for a in atoms):
                        out.append(tuple(atoms))
                        order.append((i, pos))

        self.segments.append(seg)
        if auto_angles:
            seg.angles = self._gen_angles(seg)
            seg.generated.add('angles')
        if auto_dihedrals:
            seg.dihedrals = self._gen_dihedrals(seg.angles)
            seg.generated.add('dihedrals')

    @staticmethod
    def _resolve(seg, i, name):
        """Atom for a topology name in residue i; '-'/'+' refer to the neighbours."""
        if name.startswith('-'):
            i, name = i - 1, name[1:]
        elif name.startswith('+'):
            i, name = i + 1, name[1:]
        if i < 0 or i >= len(seg.residues):
            return None
        return seg.residues[i].atoms.get(name)

    # ------------------------------------------------------------------
    # angle / dihedral generation from bonds
    # ------------------------------------------------------------------
    @staticmethod
    def _neighbours(bonds):
        nbr = {}
        for a, b in bonds:
            if a is b:
                continue
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
        for k in nbr:                       # drop duplicate bonds, keep order
            seen, uniq = set(), []
            for x in nbr[k]:
                if id(x) not in seen:
                    seen.add(id(x))
                    uniq.append(x)
            nbr[k] = uniq
        return nbr

    @staticmethod
    def _partners(seg):
        """Bonded partners of every atom (keyed by id) in psfgen's storage order.

        Bonds are stored in psfgen's insertion order (residue by residue,
        each residue's topology bonds in reverse) and every atom keeps its
        partners most recent first; duplicates are dropped.
        """
        bonds = seg.bonds
        order = seg.order.get('bonds')
        if order is None or len(order) != len(bonds):
            order = [(0, -k) for k in range(len(bonds))]
        partners = {}
        for k in sorted(range(len(bonds)), key=lambda k: (order[k][0], -order[k][1])):
            a, b = bonds[k]
            if a is b:
                continue
            partners.setdefault(id(a), []).insert(0, b)
            partners.setdefault(id(b), []).insert(0, a)
        for key, ns in partners.items():
            seen, uniq = set(), []
            for x in ns:
                if id(x) not in seen:
                    seen.add(id(x))
                    uniq.append(x)
            partners[key] = uniq
        return partners

    @classmethod
    def _gen_angles(cls, seg):
        """All angles of the segment's bond graph, generated as psfgen does.

        Going through the centre atoms in order, each pair of bonded partners
        (in the order of :meth:`_partners`) gives one angle.
        """
        partners = cls._partners(seg)
        angles, seen = [], set()
        for res in seg.residues:
            for b in res.atoms.values():
                ns = partners.get(id(b), [])
                for x in range(len(ns)):
                    for y in range(x + 1, len(ns)):
                        key = frozenset((id(ns[x]), id(ns[y]))), id(b)
                        if key not in seen:
                            seen.add(key)
                            angles.append((ns[x], b, ns[y]))
        return angles

    @staticmethod
    def _gen_dihedrals(angles):
        """Dihedrals a-b-c-d for every pair of angles a-b-c and b-c-d (as psfgen does).

        Built from the angle list, so with explicit (topology) angles only the
        dihedrals spanned by two existing angles are generated.
        """
        ends = {}                       # (b, c) -> atoms d with an angle b-c-d
        for x, y, z in angles:
            ends.setdefault((id(x), id(y)), []).append(z)
            ends.setdefault((id(z), id(y)), []).append(x)
        dihedrals, seen = [], set()
        for x, y, z in angles:
            for a, b, c in ((x, y, z), (z, y, x)):
                for d in ends.get((id(b), id(c)), []):
                    if d is a or d is b:
                        continue
                    key = (id(a), id(b), id(c), id(d))
                    if key in seen or key[::-1] in seen:
                        continue
                    seen.add(key)
                    dihedrals.append((a, b, c, d))
        return dihedrals

    def regenerate_angles(self):
        """Replace the angles of every segment by all angles of its bond graph."""
        for seg in self.segments:
            seg.angles = self._gen_angles(seg)
            seg.generated.add('angles')

    def regenerate_dihedrals(self):
        """Replace the dihedrals of every segment by those spanned by its angles."""
        for seg in self.segments:
            seg.dihedrals = self._gen_dihedrals(seg.angles)
            seg.generated.add('dihedrals')

    # ------------------------------------------------------------------
    # coordinates
    # ------------------------------------------------------------------
    def read_coords(self, filename, segid):
        """Set coordinates of segment ``segid`` from a PDB, matching resid and atom name.

        Residue names are not checked, so the backbone of a mutated residue
        keeps its input coordinates. Atoms absent from the topology are ignored.
        The PDB element column is kept for :meth:`write_pdb`.
        """
        seg = self._segment(segid)
        lookup = {res.resid: res for res in seg.residues}
        for name, _, _, resid, xyz, elem in _read_pdb_atoms(filename):
            res = lookup.get(resid)
            if res is not None and name.upper() in res.atoms:
                atom = res.atoms[name.upper()]
                atom.pos = np.array(xyz, dtype=float)
                atom.occupancy = 1.0
                atom.element = elem

    def guess_coords(self, distance=2.0):
        """Place atoms without coordinates next to a bonded atom that has them.

        There are no IC tables in the HyRes topologies, so a missing atom is
        put ``distance`` A from a placed bonded neighbour, pointing away from
        that neighbour's other placed partners and, among a few tilted
        directions, as far as possible from every placed atom of the segment
        (so two new atoms never overlap). Repeated until nothing more can be
        placed; atoms with no placed neighbour stay unset. Guessed atoms get
        occupancy 0 in :meth:`write_pdb`, as in psfgen. The geometry is only a
        starting point for minimization.
        """
        rng = np.random.default_rng(0)
        for seg in self.segments:
            nbr = self._neighbours(seg.bonds)
            atoms = [a for r in seg.residues for a in r.atoms.values()]
            missing = [a for a in atoms if a.pos is None]
            while missing:
                progress = []
                for atom in missing:
                    anchors = [x for x in nbr.get(atom, []) if x.pos is not None]
                    if not anchors:
                        continue
                    anchor = anchors[0]
                    others = [x.pos for x in nbr.get(anchor, []) if x.pos is not None and x is not atom]
                    v = anchor.pos - np.mean(others, axis=0) if others else rng.normal(size=3)
                    if np.linalg.norm(v) < 1e-6:
                        v = rng.normal(size=3)
                    v = v / np.linalg.norm(v)
                    placed = np.array([a.pos for a in atoms if a.pos is not None])
                    best, best_d = None, -1.0
                    for k in range(30):
                        u = v if k == 0 else v + 0.8 * rng.normal(size=3)
                        p = anchor.pos + distance * u / np.linalg.norm(u)
                        d = np.min(np.linalg.norm(placed - p, axis=1))
                        if d > best_d:
                            best, best_d = p, d
                        if d >= 0.9 * distance:
                            break
                    atom.pos = best
                    atom.occupancy = 0.0
                    progress.append(atom)
                if not progress:
                    break
                missing = [a for a in missing if a.pos is None]

    # ------------------------------------------------------------------
    # queries and edits
    # ------------------------------------------------------------------
    def _segment(self, segid):
        for seg in self.segments:
            if seg.segid == segid:
                return seg
        raise KeyError(f"no segment {segid}")

    def get_segids(self):
        return [seg.segid for seg in self.segments]

    def get_resids(self, segid):
        return [res.resid for res in self._segment(segid).residues]

    def get_resname(self, segid, resid):
        return self._segment(segid).residue(str(resid)).resname

    def get_atom_names(self, segid, resid):
        return list(self._segment(segid).residue(str(resid)).atoms)

    def get_charges(self, segid, resid):
        return [a.charge for a in self._segment(segid).residue(str(resid)).atoms.values()]

    def get_masses(self, segid, resid):
        return [a.mass for a in self._segment(segid).residue(str(resid)).atoms.values()]

    def get_coordinates(self, segid, resid):
        out = []
        for a in self._segment(segid).residue(str(resid)).atoms.values():
            p = a.pos if a.pos is not None else np.zeros(3)
            out.append(tuple(float(v) for v in p))
        return out

    def set_charge(self, segid, resid, atomname, charge):
        self._segment(segid).residue(str(resid)).atoms[atomname.upper()].charge = float(charge)

    def set_position(self, segid, resid, atomname, position):
        atom = self._segment(segid).residue(str(resid)).atoms[atomname.upper()]
        atom.pos = np.array(position, dtype=float)
        atom.occupancy = 1.0

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _all_atoms(self):
        for seg in self.segments:
            for res in seg.residues:
                for atom in res.atoms.values():
                    yield seg, res, atom

    def write_psf(self, filename, type='x-plor'):
        """Write an X-PLOR PSF (atom types as names), as psfgen does."""
        if type != 'x-plor':
            raise NotImplementedError("only x-plor PSF files are supported")
        index = {}
        rows = []
        for seg, res, atom in self._all_atoms():
            index[id(atom)] = len(index) + 1
            rows.append((seg.segid, res.resid, res.resname, atom))

        ext = (len(rows) > 99999999 or
               any(len(s) > 4 or len(r) > 4 or len(n) > 4 or len(a.name) > 4 or len(a.type) > 4
                   for s, r, n, a in rows))

        def terms(attr):
            out = []
            for seg in self.segments:
                if attr == 'dihedrals' and attr in seg.generated:
                    out += self._generated_dihedral_order(seg, index)
                    continue
                ts = [tuple(index[id(a)] for a in t) for t in getattr(seg, attr)]
                out += self._psfgen_order(ts, seg.order.get(attr), attr if attr in seg.generated else None)
            return out

        bonds, angles = terms('bonds'), terms('angles')
        dihedrals, impropers = terms('dihedrals'), terms('impropers')
        donors, acceptors, cmaps = terms('donors'), terms('acceptors'), terms('cmaps')

        flags = (['EXT'] if ext else []) + (['CMAP'] if cmaps else [])
        width = 10 if ext else 8
        title = [' REMARKS original generated structure x-plor psf file']
        title += [f' REMARKS topology {t} ' for t in self.topology_files]
        for seg in self.segments:
            auto = f"{'angles' if seg.auto_angles else 'none'} {'dihedrals' if seg.auto_dihedrals else ''}"
            title.append(f' REMARKS segment {seg.segid} {{ first none; last none; auto {auto} }}')

        with open(filename, 'w') as out:
            out.write('PSF' + ''.join(' ' + f for f in flags) + '\n\n')
            out.write(f"{len(title):>{width}d} !NTITLE\n")
            out.write('\n'.join(title) + '\n\n')
            out.write(f"{len(rows):>{width}d} !NATOM\n")
            for i, (segid, resid, resname, atom) in enumerate(rows, start=1):
                if ext:
                    out.write(f"{i:10d} {segid:<8s} {resid:<8s} {resname:<8s} {atom.name:<8s} "
                              f"{atom.type:<6s} {atom.charge:10.6f}    {atom.mass:10.4f}  {0:10d}\n")
                else:
                    out.write(f"{i:8d} {segid:<4s} {resid:<4s} {resname:<4s} {atom.name:<4s} "
                              f"{atom.type:<4s} {atom.charge:10.6f} {atom.mass:13.4f} {0:11d}\n")
            out.write('\n')
            self._section(out, bonds, 'NBOND: bonds', 4, width)
            self._section(out, angles, 'NTHETA: angles', 3, width)
            self._section(out, dihedrals, 'NPHI: dihedrals', 2, width)
            self._section(out, impropers, 'NIMPHI: impropers', 2, width)
            self._section(out, donors, 'NDON: donors', 4, width)
            self._section(out, acceptors, 'NACC: acceptors', 4, width)
            out.write(f"{0:>{width}d} !NNB\n\n")
            zeros = [0] * len(rows)
            for k in range(0, len(zeros), 8):
                out.write(''.join(f"{v:>{width}d}" for v in zeros[k:k + 8]) + '\n')
            out.write('\n')
            out.write(f"{1:>{width}d} {0:>{width - 1}d} !NGRP\n")
            out.write(''.join(f"{v:>{width}d}" for v in (0, 0, 0)) + '\n\n')
            if cmaps:
                self._section(out, cmaps, 'NCRTERM: cross-terms', 1, width)

    @staticmethod
    def _psfgen_order(terms, order, generated):
        """Sort terms as psfgen writes them, so PSF files can be diffed.

        psfgen writes each term under its first atom, in atom order, the most
        recently added first. Topology terms are added residue by residue,
        each residue's in reverse topology order, so for one first atom those
        of a later residue come first and those of one residue keep the
        topology order. Generated angles keep the generation orientation and
        come latest generated first (generated dihedrals: see
        :meth:`_generated_dihedral_order`).

        Args:
            terms (list of tuple): Terms as 1-based atom indices.
            order (list): (residue position, topology position) of each
                topology term, or None.
            generated (str or None): 'angles' if the terms were generated
                from the bond graph.
        """
        if generated == 'angles':
            idx = sorted(range(len(terms)), key=lambda k: (terms[k][0], -k))
            return [terms[k] for k in idx]
        keys = order if order is not None and len(order) == len(terms) else [(0, k) for k in range(len(terms))]
        idx = sorted(range(len(terms)), key=lambda k: (terms[k][0], -keys[k][0], keys[k][1]))
        return [terms[k] for k in idx]

    @classmethod
    def _generated_dihedral_order(cls, seg, index):
        """Generated dihedrals in psfgen's order, as 1-based index tuples.

        Each dihedral is oriented with the lower index first; they are sorted
        by the first atom and then by the rank of each next atom in the
        partner list (:meth:`_partners`) of the previous one.
        """
        partners = cls._partners(seg)
        rank = {key: {id(x): r for r, x in enumerate(ns)} for key, ns in partners.items()}
        rows = []
        for t in seg.dihedrals:
            if index[id(t[0])] > index[id(t[-1])]:
                t = t[::-1]
            key = (index[id(t[0])],) + tuple(rank.get(id(t[k]), {}).get(id(t[k + 1]), 0) for k in range(3))
            rows.append((key, tuple(index[id(a)] for a in t)))
        rows.sort(key=lambda r: r[0])
        return [r[1] for r in rows]

    @staticmethod
    def _section(out, entries, label, per_line, width):
        out.write(f"{len(entries):>{width}d} !{label}\n")
        for k in range(0, len(entries), per_line):
            out.write(''.join(f"{v:>{width}d}" for t in entries[k:k + per_line] for v in t) + '\n')
        if not entries:
            out.write('\n')            # psfgen writes an empty line for an empty section
        out.write('\n')

    def write_pdb(self, filename):
        """Write all atoms; occupancy 1 (read or set), 0 (guessed) or -1 (unknown, at 0,0,0).

        Serials beyond 99999 are written in hybrid-36, where psfgen writes '******'.
        """
        with open(filename, 'w') as out:
            out.write('REMARK original generated coordinate pdb file\n')
            for i, (seg, res, atom) in enumerate(self._all_atoms(), start=1):
                xyz = atom.pos if atom.pos is not None else (0.0, 0.0, 0.0)
                name = atom.name if len(atom.name) == 4 else ' ' + atom.name
                resid = res.resid
                if resid[-1:].isalpha():
                    resseq, icode = resid[:-1], resid[-1]
                else:
                    resseq, icode = resid, ' '
                out.write(f"ATOM  {_serial(i)} {name:<4s} {res.resname:<4s}{res.chain:1s}"
                          f"{resseq:>4s}{icode:1s}   {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                          f"{atom.occupancy:6.2f}{0.0:6.2f}      {seg.segid:<4s}{atom.element:>2s}\n")
            out.write('END\n')


# psfgen-compatible name
PsfGen = PsfBuilder
