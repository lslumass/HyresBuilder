#!/usr/bin/env python3
"""
add_missing_residues_assembly.py
=================================

End-to-end pipeline that takes a PDB ID and produces a complete structural
model of its BIOLOGICAL ASSEMBLY (not just the deposited asymmetric unit)
with all missing residues/loops built in, using MODELLER.

Pipeline
--------
1. Download the asymmetric-unit mmCIF from RCSB (it contains both the
   observed coordinates AND the full entity sequence in entity_poly_seq,
   plus the symmetry operators needed to build the assembly).
2. Record each asymmetric-unit chain's FULL entity sequence *before*
   expanding the assembly (entity/subchain bookkeeping is only reliable on
   the un-transformed structure).
3. Use gemmi to expand the asymmetric unit into the requested biological
   assembly, then give every resulting chain a GLOBALLY UNIQUE
   single-character chain ID.
4. For every polymer chain in the assembly, align the full entity sequence
   against the observed (possibly gappy) coordinates with
   gemmi.align_sequence_to_polymer(), which tells us exactly which
   residues are missing and where.
5. Write a MODELLER PIR alignment (.ali): the "structureX" sequence
   (what's actually resolved, with '-' for gaps) vs. the "sequence"
   (the complete target sequence), chains separated by '/'.
6. Validate that the template PDB and the alignment agree on chain count,
   chain order and per-chain residue count, then run MODELLER's automodel.
7. Save the completed assembly as <PDBID>_assembly_filled.pdb and report
   the DOPE score.

Two things that will silently break a multi-chain run
-----------------------------------------------------
* Chain IDs must be globally unique. Both the PDB format and MODELLER
  identify a chain by its ID alone, so reusing an ID for symmetry copies
  (A, B, A, B, ...) does not create distinct chains -- MODELLER reads up to
  the first chain matching the header's end ID and stops, truncating the
  template (a 12-chain / 5232-residue assembly collapses to 872 residues).
  Hence unique IDs plus an unrestricted FIRST:@ / END:@ header.
* PIR header lines need exactly 10 colon-separated fields. One colon too
  many or too few aborts parsing with "Invalid PIR file header line", so
  headers here are assembled from a 10-element list by _pir_header().

Requirements
------------
    pip install "gemmi>=0.5" requests
    MODELLER must be installed and licensed, e.g.:
        conda install -c salilab modeller
    and either export KEY_MODELLER=<your academic key>, or have it set in
    modeller's config (~/.modeller.ini / Lib/modeller/config.py).

Usage
-----
    python add_missing_residues_assembly.py 1ABC
    python add_missing_residues_assembly.py 1ABC --assembly-id 1 --outdir ./out
    python add_missing_residues_assembly.py 1ABC --keep-hetatm
    python add_missing_residues_assembly.py 1ABC --refine very_fast

Notes / limitations
--------------------
* By default HETATM records (ligands, ions, waters) are dropped. With
  --keep-hetatm they are retained as rigid, un-modeled blocks and
  represented as '.' placeholders in the alignment; missing atoms *within*
  a ligand are never modeled. This path is best-effort: it assumes het
  residues follow the polymer within each chain.
* Only polymer entities (protein/nucleic acid) are modeled. A chain that is
  skipped for the alignment is also removed from the template, so the two
  can never disagree.
* If a chain has zero observed residues it cannot serve as its own
  structural template and is skipped with a warning.
* Assemblies with more than 62 chains cannot be represented in classic PDB
  format and are rejected rather than silently mangled.
"""

import argparse
import os
import string
import sys
import urllib.request

import gemmi

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

ONE_LETTER_FALLBACK = "X"

# Pool of legal single-character chain IDs for classic PDB output. Every
# chain in the written file gets a DISTINCT character from this pool, which
# caps a writable assembly at len(CHAIN_ID_POOL) chains.
CHAIN_ID_POOL = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

# A PIR header line must have exactly this many colon-separated fields:
# type:code:start_res:start_chain:end_res:end_chain:name:source:resolution:rfactor
PIR_HEADER_FIELDS = 10


# ---------------------------------------------------------------------------
# download / small helpers
# ---------------------------------------------------------------------------

def download_cif(pdb_id: str, outdir: str) -> str:
    """Download the asymmetric-unit mmCIF for pdb_id into outdir, return path."""
    pdb_id = pdb_id.lower()
    path = os.path.join(outdir, f"{pdb_id}.cif")
    if os.path.exists(path):
        print(f"[download] using cached {path}")
        return path
    url = RCSB_CIF_URL.format(pdb_id=pdb_id)
    print(f"[download] fetching {url}")
    urllib.request.urlretrieve(url, path)
    return path


def three_to_one(code: str) -> str:
    """Map a 3(+)-letter monomer code to a 1-letter code, 'X' if unknown."""
    info = gemmi.find_tabulated_residue(code)
    if info is not None and info.one_letter_code.isalpha():
        return info.one_letter_code.upper()
    return ONE_LETTER_FALLBACK


def _wrap(seq: str, width: int = 75) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def _pir_header(fields) -> str:
    """Join exactly 10 PIR header fields into one header line.

    MODELLER's parser counts fields, not colons, and aborts with
    "Invalid PIR file header line ... contains N fields" on any other count.
    Building the line from a list makes an off-by-one colon impossible.
    """
    fields = list(fields)
    if len(fields) != PIR_HEADER_FIELDS:
        raise ValueError(f"PIR header needs {PIR_HEADER_FIELDS} fields, "
                         f"got {len(fields)}: {fields}")
    return ":".join(fields)


def _polymer_residues(chain: gemmi.Chain):
    """Polymer residues of a chain, with a safe fallback if subchain/entity
    bookkeeping was lost during assembly expansion."""
    pol = chain.get_polymer()
    if len(pol):
        return list(pol)
    out = []
    for r in chain:
        info = gemmi.find_tabulated_residue(r.name)
        if info is not None and info.is_amino_acid():
            out.append(r)
    return out


def _het_residues(chain: gemmi.Chain):
    """Non-polymer residues of a chain (ligands, ions, waters)."""
    polymer_ids = {id(r) for r in _polymer_residues(chain)}
    return [r for r in chain if id(r) not in polymer_ids]


# ---------------------------------------------------------------------------
# assembly handling
# ---------------------------------------------------------------------------

def describe_assembly(assembly: gemmi.Assembly) -> str:
    """One-line human-readable summary of a gemmi.Assembly."""
    chain_ids = sorted({c for gen in assembly.generators for c in gen.chains}
                       | {s for gen in assembly.generators for s in gen.subchains})
    n_ops = sum(len(gen.operators) for gen in assembly.generators)
    source = "author-defined" if assembly.author_determined else (
        "software-defined" if assembly.software_determined else "unspecified")
    details = assembly.oligomeric_details or "n/a"
    return (f"assembly '{assembly.name}': {details} ({source}), "
            f"{n_ops} symmetry operator(s), chains involved: "
            f"{', '.join(chain_ids) if chain_ids else 'n/a'}")


def list_assemblies(st: gemmi.Structure):
    """Print a summary of every biological assembly defined for this entry."""
    if not st.assemblies:
        print("[assembly] this entry defines no biological assemblies; "
              "only the deposited asymmetric unit is available.")
        return
    print(f"[assembly] {len(st.assemblies)} assembly(ies) defined for "
          f"{st.name or 'this entry'}:")
    for a in st.assemblies:
        print(f"  - {describe_assembly(a)}")


def choose_default_assembly(st: gemmi.Structure) -> str:
    """Prefer an author-curated assembly over a software-inferred one."""
    author_defined = [a.name for a in st.assemblies if a.author_determined]
    if author_defined:
        return author_defined[0]
    return st.assemblies[0].name


def collect_full_sequences(st: gemmi.Structure):
    """Map asymmetric-unit chain name -> (full_sequence, polymer_type).

    Must be called on the UN-transformed structure: after
    transform_to_assembly() the subchain labels of generated copies no
    longer resolve cleanly back to their entity, so get_entity_of() becomes
    unreliable. Every assembly copy inherits its original chain's sequence.
    """
    table = {}
    for chain in st[0]:
        polymer = chain.get_polymer()
        if not len(polymer):
            continue
        entity = st.get_entity_of(polymer)
        if entity is None or not entity.full_sequence:
            print(f"[entity] chain {chain.name}: no full sequence in the file "
                  f"(entity_poly_seq missing); it cannot be gap-filled.")
            continue
        table[chain.name] = (list(entity.full_sequence), entity.polymer_type)
        print(f"[entity] chain {chain.name}: full sequence has "
              f"{len(entity.full_sequence)} residues")
    return table


def assign_unique_chain_ids(model: gemmi.Model):
    """Give every chain a unique, legal, single-character PDB chain ID.

    Returns {new_chain_id: original_chain_name} so callers can still look up
    per-entity information (full sequence, polymer type) for each copy.

    The first chain to claim a legal, unused name keeps it; every later
    chain (including every symmetry duplicate, and any multi-character
    auth_asym_id that will not fit the PDB format) is given the next free
    character from CHAIN_ID_POOL.
    """
    origin = {}
    used = set()

    for chain in model:
        original = chain.name
        legal = len(original) == 1 and original in CHAIN_ID_POOL
        if legal and original not in used:
            used.add(original)
            origin[original] = original
            continue

        new_name = next((c for c in CHAIN_ID_POOL if c not in used), None)
        if new_name is None:
            raise RuntimeError(
                f"this assembly has more than {len(CHAIN_ID_POOL)} chains; it "
                "cannot be represented in classic PDB format (which MODELLER "
                "requires here). Model a smaller sub-assembly instead.")
        reason = "duplicate" if legal else "not a legal single-character ID"
        print(f"[chain-id] '{original}' -> '{new_name}' ({reason}; chain IDs "
              f"must be globally unique or MODELLER truncates the template)")
        chain.name = new_name
        used.add(new_name)
        origin[new_name] = original

    return origin


def load_assembly(cif_path: str, assembly_id):
    """Read the asymmetric unit and expand it into the requested assembly.

    Returns (structure, seq_table, origin):
        seq_table  original chain name -> (full_sequence, polymer_type)
        origin     assembly chain ID   -> original chain name
    """
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.assign_label_seq_id()

    # Capture sequences BEFORE transforming (see collect_full_sequences).
    seq_table = collect_full_sequences(st)

    available = [a.name for a in st.assemblies]
    if not available:
        print("[assembly] no assembly records found in the file; "
              "falling back to the asymmetric unit as-is.")
        origin = assign_unique_chain_ids(st[0])
        return st, seq_table, origin

    if len(available) > 1:
        list_assemblies(st)

    if assembly_id is None:
        assembly_id = choose_default_assembly(st)
        if len(available) > 1:
            print(f"[assembly] no --assembly-id given and multiple assemblies "
                  f"exist ({available}); defaulting to '{assembly_id}'. Pass "
                  f"--assembly-id to pick a different one.")
    elif assembly_id not in available:
        print(f"[assembly] requested assembly '{assembly_id}' not found "
              f"(available: {available}); using '{available[0]}' instead.")
        assembly_id = available[0]

    print(f"[assembly] building biological assembly '{assembly_id}' "
          f"(available: {available})")

    # Dup keeps each generated copy under its ORIGINAL chain ID, which makes it
    # trivial to map every copy back to its entity. Those duplicate IDs are then
    # immediately resolved into unique ones by assign_unique_chain_ids() --
    # duplicates must never reach the PDB file.
    st.transform_to_assembly(assembly_id, gemmi.HowToNameCopiedChain.Dup)
    origin = assign_unique_chain_ids(st[0])

    print(f"[assembly] expanded to {len(st[0])} chains with unique IDs: "
          f"{', '.join(c.name for c in st[0])}")
    return st, seq_table, origin


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

def build_alignment(st, seq_table, origin, keep_hetatm):
    """Build parallel gap-aligned sequences for every modelable chain.

    Returns (chain_order, observed_seqs, full_seqs):
        observed_seqs[chain] -> resolved residues, '-' for gaps ('.' for het)
        full_seqs[chain]     -> complete target sequence, same length
    """
    model = st[0]
    scoring = gemmi.AlignmentScoring()  # default protein/NA scoring

    chain_order = []
    observed_seqs = {}
    full_seqs = {}

    for chain in model:
        residues = _polymer_residues(chain)
        if not residues:
            continue  # pure ligand/water chain: handled by keep_hetatm below

        key = origin.get(chain.name, chain.name)
        if key not in seq_table:
            print(f"[align] chain {chain.name}: no entity/full sequence info, "
                  f"skipping (its coordinates are dropped from the template).")
            continue

        full_sequence, polymer_type = seq_table[key]
        polymer = chain.get_polymer()
        target = polymer if len(polymer) else residues

        result = gemmi.align_sequence_to_polymer(
            full_sequence, polymer, polymer_type, scoring
        )
        full_gapped = result.add_gaps(
            "".join(three_to_one(c) for c in full_sequence), 1
        )
        observed_gapped = result.add_gaps(
            "".join(three_to_one(r.name) for r in target), 2
        )

        if len(full_gapped) != len(observed_gapped):
            print(f"[align] WARNING chain {chain.name}: gapped length mismatch "
                  f"({len(full_gapped)} vs {len(observed_gapped)}); skipping.")
            continue
        if observed_gapped.replace("-", "") == "":
            print(f"[align] WARNING chain {chain.name}: no observed residues, "
                  f"nothing can template it; skipping.")
            continue

        # Retained het residues become '.' placeholders in BOTH lines, so the
        # two entries stay the same length and MODELLER's residue count for the
        # chain matches the file.
        het_pad = ""
        if keep_hetatm:
            n_het = len(_het_residues(chain))
            het_pad = "." * n_het
            if n_het:
                print(f"[align] chain {chain.name}: {n_het} het residue(s) "
                      f"kept as rigid '.' blocks")

        n_missing = observed_gapped.count("-")
        print(f"[align] chain {chain.name}: {len(full_gapped)} residues total, "
              f"{n_missing} missing (identity vs template: "
              f"{result.calculate_identity():.1f}%)")

        chain_order.append(chain.name)
        observed_seqs[chain.name] = observed_gapped + het_pad
        full_seqs[chain.name] = full_gapped + het_pad

    return chain_order, observed_seqs, full_seqs


def write_pir_alignment(pdb_id, chain_order, observed_seqs, full_seqs, ali_path):
    """Write a MODELLER PIR (.ali) alignment with one 'structureX' and one
    'sequence' entry, chains joined by '/', gaps as '-'."""
    struct_code = f"{pdb_id}_template"
    target_code = f"{pdb_id}_filled"

    observed_joined = "/".join(observed_seqs[c] for c in chain_order) + "*"
    full_joined = "/".join(full_seqs[c] for c in chain_order) + "*"

    # FIRST:@ / END:@ is MODELLER's unrestricted model_segment: read the WHOLE
    # atom file. Naming explicit start/end chain IDs here would make MODELLER
    # stop at the first chain matching the end ID, truncating the template.
    template_header = _pir_header([
        "structureX",   # 1  type
        struct_code,    # 2  code
        "FIRST",        # 3  start residue
        "@",            # 4  start chain
        "END",          # 5  end residue
        "@",            # 6  end chain
        "",             # 7  name
        "",             # 8  source
        "",             # 9  resolution
        "",             # 10 R-factor
    ])
    target_header = _pir_header([
        "sequence",     # 1  type
        target_code,    # 2  code
        "",             # 3  start residue
        "",             # 4  start chain
        "",             # 5  end residue
        "",             # 6  end chain
        "",             # 7  name
        "",             # 8  source
        "",             # 9  resolution
        "",             # 10 R-factor
    ])

    with open(ali_path, "w") as fh:
        fh.write(f">P1;{struct_code}\n")
        fh.write(template_header + "\n")
        fh.write(_wrap(observed_joined) + "\n\n")

        fh.write(f">P1;{target_code}\n")
        fh.write(target_header + "\n")
        fh.write(_wrap(full_joined) + "\n\n")

    n_seg = len(chain_order)
    print(f"[alignment] wrote PIR alignment -> {ali_path} "
          f"({n_seg} segments, {n_seg - 1} chain breaks, "
          f"{len(observed_joined) - n_seg} alignment positions)")
    return struct_code, target_code


# ---------------------------------------------------------------------------
# template coordinates
# ---------------------------------------------------------------------------

def write_template_pdb(st, keep_hetatm, path, chain_order):
    """Write the template coordinates MODELLER will read, containing exactly
    the chains the alignment declares, in the same order.

    Chain IDs are already globally unique, so plain per-chain 1..N residue
    numbering is safe -- no global renumbering trickery is needed (and the
    final model therefore comes out with sane numbering by itself).
    """
    st_out = st.clone()
    if not keep_hetatm:
        st_out.remove_ligands_and_waters()

    keep = set(chain_order)
    for name in [c.name for c in st_out[0] if c.name not in keep]:
        print(f"[template] dropping chain {name} (not present in the alignment)")
        st_out[0].remove_chain(name)
    st_out.remove_empty_chains()

    # NOTE: deliberately NO setup_entities() here. It merges same-named chain
    # parts, which would change the chain partitioning the alignment was built
    # from. IDs are unique now, so there is nothing to gain from it.

    for chain in st_out[0]:
        for i, res in enumerate(chain, start=1):
            res.seqid = gemmi.SeqId(i, " ")
            res.label_seq = None

    options = gemmi.PdbWriteOptions()
    st_out.write_pdb(path, options)

    total = sum(len(c) for c in st_out[0])
    print(f"[template] wrote {path}: {len(st_out[0])} chains, {total} residues "
          f"-> {', '.join(f'{c.name}:{len(c)}' for c in st_out[0])}")
    return st_out


def validate_template_vs_alignment(st_out, chain_order, observed_seqs,
                                   keep_hetatm):
    """Fail loudly *before* MODELLER runs if the file and the .ali disagree.

    Catches exactly the class of error that otherwise surfaces as
    'Number of residues in the alignment and pdb files are different'.
    """
    pdb_chains = [c.name for c in st_out[0]]

    if len(set(pdb_chains)) != len(pdb_chains):
        raise RuntimeError(f"duplicate chain IDs in template: {pdb_chains}. "
                           "MODELLER would read only the first occurrence.")
    if pdb_chains != list(chain_order):
        raise RuntimeError(f"chain order mismatch: template has {pdb_chains}, "
                           f"alignment declares {list(chain_order)}")

    total_pdb = 0
    for chain in st_out[0]:
        n_pol = len(_polymer_residues(chain))
        n_het = len(_het_residues(chain)) if keep_hetatm else 0
        seq = observed_seqs[chain.name]
        n_ali = len(seq) - seq.count("-")
        if n_pol + n_het != n_ali:
            raise RuntimeError(
                f"chain {chain.name}: template has {n_pol} polymer + {n_het} "
                f"het residues, alignment declares {n_ali} non-gap positions")
        total_pdb += n_pol + n_het

    print(f"[validate] OK: {len(pdb_chains)} chains, {total_pdb} template "
          f"residues, {len(pdb_chains) - 1} chain breaks -- consistent with "
          f"the alignment")


# ---------------------------------------------------------------------------
# MODELLER
# ---------------------------------------------------------------------------

def run_modeller(workdir, ali_path, struct_code, target_code,
                 keep_hetatm=False, refine_level="fast"):
    """Run MODELLER automodel to build the complete assembly model."""
    from modeller import Environ, log
    from modeller.automodel import AutoModel, assess, refine

    refine_levels = {
        "none": None,
        "very_fast": refine.very_fast,
        "fast": refine.fast,
        "slow": refine.slow,
        "very_slow": refine.very_slow,
    }
    if refine_level not in refine_levels:
        raise ValueError(f"unknown refine level '{refine_level}'; "
                         f"choose from {sorted(refine_levels)}")

    log.verbose()
    env = Environ()
    env.io.atom_files_directory = [str(os.path.abspath(workdir)), "."]
    # Must agree with the alignment: '.' placeholders are only read when hetatm
    # input is enabled, and enabling it without placeholders shifts every
    # residue count.
    env.io.hetatm = keep_hetatm

    a = AutoModel(
        env,
        alnfile=ali_path,
        knowns=struct_code,
        sequence=target_code,
        assess_methods=(assess.DOPE,),
    )
    a.starting_model = 1
    a.ending_model = 1
    a.md_level = refine_levels[refine_level]
    print(f"[modeller] refinement level: {refine_level}")
    a.make()

    successful = [m for m in a.outputs if m["failure"] is None]
    if not successful:
        raise RuntimeError("MODELLER did not produce a successful model.")
    best = min(successful, key=lambda m: m["DOPE score"])
    print(f"[modeller] best model: {best['name']} "
          f"(DOPE score = {best['DOPE score']:.2f})")
    return best["name"]


def report_model_chains(path: str):
    """Print the chain/residue inventory of the finished model."""
    st = gemmi.read_structure(path)
    chains = [(c.name, len(c)) for c in st[0]]
    total = sum(n for _, n in chains)
    print(f"[postprocess] final model: {len(chains)} chains, {total} residues "
          f"-> {', '.join(f'{n}:{k}' for n, k in chains)}")
    print("[postprocess] residues are numbered 1..N within each chain; the "
          "original author/SEQRES numbering is not preserved.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb_id", help="4-character PDB ID, e.g. 1ABC")
    parser.add_argument("--assembly-id", default=None,
                        help="Biological assembly identifier as given in the "
                             "mmCIF, e.g. '1' or '2'. If omitted, an "
                             "author-defined assembly is used; see "
                             "--list-assemblies to inspect the options.")
    parser.add_argument("--list-assemblies", action="store_true",
                        help="Print the biological assemblies defined for this "
                             "entry and exit without modeling.")
    parser.add_argument("--outdir", default=None,
                        help="Working/output directory. Default: ./<pdbid>_work")
    parser.add_argument("--keep-hetatm", action="store_true",
                        help="Keep ligands/ions/waters as rigid HETATM blocks "
                             "in the template (represented as '.' in the "
                             "alignment; not modeled themselves).")
    parser.add_argument("--refine", default="fast",
                        choices=["none", "very_fast", "fast", "slow",
                                 "very_slow"],
                        help="MODELLER MD refinement level. Large assemblies "
                             "are slow; 'very_fast' or 'none' is usually "
                             "plenty when only short gaps are being filled. "
                             "Default: fast")
    args = parser.parse_args()

    pdb_id = args.pdb_id.lower()
    outdir = args.outdir or f"{pdb_id}_work"
    os.makedirs(outdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(outdir)

    try:
        cif_path = download_cif(pdb_id, ".")

        if args.list_assemblies:
            st_raw = gemmi.read_structure(cif_path)
            st_raw.setup_entities()
            list_assemblies(st_raw)
            return

        st, seq_table, origin = load_assembly(cif_path, args.assembly_id)

        chain_order, observed_seqs, full_seqs = build_alignment(
            st, seq_table, origin, args.keep_hetatm
        )
        if not chain_order:
            sys.exit("No polymer chains with usable sequence info were found; "
                     "nothing to model.")

        # struct_code must equal the template file's basename, because MODELLER
        # looks for <known>.pdb in atom_files_directory.
        struct_code = f"{pdb_id}_template"
        template_pdb = f"{struct_code}.pdb"
        st_out = write_template_pdb(st, args.keep_hetatm, template_pdb,
                                    chain_order)
        validate_template_vs_alignment(st_out, chain_order, observed_seqs,
                                       args.keep_hetatm)

        ali_path = f"{pdb_id}_alignment.ali"
        struct_code, target_code = write_pir_alignment(
            pdb_id, chain_order, observed_seqs, full_seqs, ali_path
        )

        model_name = run_modeller(".", ali_path, struct_code, target_code,
                                  keep_hetatm=args.keep_hetatm,
                                  refine_level=args.refine)

        final_path = f"{pdb_id}_assembly_filled.pdb"
        os.replace(model_name, final_path)
        report_model_chains(final_path)
        print(f"\nDone. Completed assembly model: "
              f"{os.path.join(outdir, final_path)}")

    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()