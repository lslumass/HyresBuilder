"""
Tests for ``HyresBuilder.psfbuilder``, the pure-Python replacement of psfgen.

The parity tests build each system twice through the real ``GenPsf`` code,
once with psfgen and once with psfbuilder, and require byte-identical PSF
files (same atoms, terms and term order). They are skipped when psfgen is
not installed. The other tests need only HyresBuilder (and OpenMM for the
loading test).

Run with:  pytest tests/test_psfbuilder.py
"""

import contextlib
import filecmp
import io
from pathlib import Path

import numpy as np
import pytest

from HyresBuilder import GenPsf, Mutator, iConBuilder
from HyresBuilder.psfbuilder import PsfBuilder, _serial

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'

GENPSF_CASES = [
    ('peptide', 'test.pdb', {}),
    ('tdp43', 'tdp-43_hyres.pdb', {}),
    ('tdp43_charged', 'tdp-43_hyres.pdb', {'terminal': 'charged'}),
    ('monomer', 'benchmark_system/monomer/conf.pdb', {}),
    ('llps2', 'benchmark_system/LLPS2/conf.pdb', {}),
    ('qPDMAEMA', 'polymers/pc.pdb', {}),
    ('pBzMA', 'polymers/pBzMA.pdb', {}),
    ('metabolite', 'metabolite_pdb/AD0.pdb', {}),
    ('mg', 'Mg.pdb', {}),
]


def _real_psfgen():
    """psfgen's PsfGen with its console output silenced, or skip the test."""
    psfgen = pytest.importorskip('psfgen')

    class QuietPsfGen(psfgen.PsfGen):
        def __init__(self, output=None):
            super().__init__(output='/dev/null')
    return QuietPsfGen


def _run_genpsf(builder, workdir, monkeypatch, fast=False, **kwargs):
    """Run GenPsf with ``builder`` as its PsfGen inside ``workdir``; return the PSF path."""
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(GenPsf, 'PsfGen', builder)
    with contextlib.redirect_stdout(io.StringIO()):
        if fast:
            GenPsf.custom_genpsf_fast(psf_out='out.psf', verbose=False, **kwargs)
        else:
            GenPsf.genpsf(psf_out='out.psf', **kwargs)
    return workdir / 'out.psf'


def _assert_same_psf(tmp_path, monkeypatch, fast=False, **kwargs):
    real = _real_psfgen()
    a = _run_genpsf(real, tmp_path / 'psfgen', monkeypatch, fast=fast, **kwargs)
    b = _run_genpsf(PsfBuilder, tmp_path / 'builder', monkeypatch, fast=fast, **kwargs)
    assert filecmp.cmp(a, b, shallow=False), f"PSF files differ: diff {a} {b}"


def _build_nucleic(tmp_path, kind, sequence):
    tmp_path.mkdir(parents=True, exist_ok=True)
    stem = str(tmp_path / kind)
    with contextlib.redirect_stdout(io.StringIO()):
        if kind == 'rna':
            iConBuilder.build_rna(stem, sequence)
        else:
            iConBuilder.build_dna(stem, sequence)
    return stem + '.pdb'


# ----------------------------------------------------------------------
# parity with psfgen
# ----------------------------------------------------------------------
@pytest.mark.parametrize('name,pdb,options', GENPSF_CASES, ids=[c[0] for c in GENPSF_CASES])
def test_genpsf_matches_psfgen(name, pdb, options, tmp_path, monkeypatch):
    _assert_same_psf(tmp_path, monkeypatch, pdb_in=str(EXAMPLES / pdb), **options)


@pytest.mark.parametrize('kind,sequence', [('rna', 'UUAGGGUUAGGGUUAGGGUUAGGG'), ('dna', 'ATGCGCATTAGC')])
def test_nucleic_acids_match_psfgen(kind, sequence, tmp_path, monkeypatch):
    pdb = _build_nucleic(tmp_path / 'input', kind, sequence)
    _assert_same_psf(tmp_path, monkeypatch, pdb_in=pdb)


def test_fast_path_matches_psfgen(tmp_path, monkeypatch):
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        iConBuilder.build_polyP('polyP', 30, seed=1)        # PHO uses automatic angles and dihedrals
    pdbs = [str(EXAMPLES / 'test.pdb'), str(EXAMPLES / 'polymers' / 'pc.pdb'),
            str(EXAMPLES / 'Mg.pdb'), str(tmp_path / 'polyP.pdb')]
    _assert_same_psf(tmp_path, monkeypatch, fast=True, pdb_list=pdbs, num_list=[20, 5, 30, 4])


# ----------------------------------------------------------------------
# psfbuilder on its own
# ----------------------------------------------------------------------
def test_psf_loads_in_openmm(tmp_path, monkeypatch):
    app = pytest.importorskip('openmm.app')
    rna = _build_nucleic(tmp_path / 'input', 'rna', 'GGGAUUCCC')
    for pdb, natoms, nbonds in ((str(EXAMPLES / 'test.pdb'), 56, 59), (rna, 58, 66)):
        psf = _run_genpsf(PsfBuilder, tmp_path / Path(pdb).stem, monkeypatch, pdb_in=pdb)
        top = app.CharmmPsfFile(str(psf)).topology
        assert top.getNumAtoms() == natoms
        assert top.getNumBonds() == nbonds


def test_mutator_places_every_bead(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = EXAMPLES / 'test.pdb'
    with contextlib.redirect_stdout(io.StringIO()):
        Mutator.mut(str(src), 'mut.pdb', [2, 3, 5, 6], ['SER', 'TRP', 'SEP', 'PTR'], 'P001')

    def atoms(fn):
        out = {}
        for line in open(fn):
            if line.startswith('ATOM'):
                key = (line[22:27].strip(), line[12:16].strip())
                out[key] = (np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                            float(line[54:60]))
        return out

    before, after = atoms(src), atoms('mut.pdb')
    assert all(occ >= 0 for _, occ in after.values()), "a bead was left without coordinates"
    # each newly built bead is at least 1 A from every other bead
    xyz = {key: pos for key, (pos, _) in after.items()}
    for key in set(after) - set(before):
        d = min(np.linalg.norm(xyz[key] - pos) for other, pos in xyz.items() if other != key)
        assert d > 1.0, f"new bead {key} overlaps another bead"
    # every bead of the input keeps its coordinates (incl. the mutated residues' backbone)
    for key, (pos, _) in after.items():
        if key in before:
            assert np.allclose(pos, before[key][0], atol=1e-3), key


def test_duplicate_definitions_follow_psfgen(tmp_path):
    first = tmp_path / 'a.top'
    first.write_text("MASS 1 XA 10.0\nRESI ABC 0.0\nATOM A1 XA 0.5\nEND\n")
    second = tmp_path / 'b.top'
    second.write_text("MASS 2 XA 20.0\nRESI ABC 0.0\nATOM B1 XA -0.5\nEND\n")
    gen = PsfBuilder()
    gen.read_topology(first)
    gen.read_topology(second)
    gen.add_segment(segid='X001', residues=[('1', 'ABC')])
    assert gen.get_atom_names('X001', '1') == ['A1']        # first RESI wins
    assert gen.get_masses('X001', '1') == [20.0]            # last MASS wins
    assert gen.get_charges('X001', '1') == [0.5]


def test_hybrid36_serials():
    assert [_serial(n) for n in (1, 99999, 100000, 100001, 43770015, 43770016)] == \
        ['    1', '99999', 'A0000', 'A0001', 'ZZZZZ', 'a0000']
