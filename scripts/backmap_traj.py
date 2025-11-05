import os
import subprocess
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from psfgen import PsfGen

os.system('export PYTORCH_ALLOC_CONF=expandable_segments:True')

# ------------------------------
# CONFIGURATION
# ------------------------------
top = "../conf.psf"
pdb0 = "../conf.pdb"
pdb = "conf.pdb"  # conf.pdb after rename the Chain ID
traj = "../system.xtc"
num = 10
nframe = 100

## reanem ChainID in conf.pdb
u = mda.Universe(pdb0)
chain_ids = ["A", "B"]
for i, seg in enumerate(u.segments):
    seg.atoms.chainIDs = chain_ids[i % len(chain_ids)]
u.atoms.write(pdb)

# ------------------------------
# Split trajectory
# ------------------------------
u = mda.Universe(top, traj)
total_frames = u.trajectory.n_frames
step = int(total_frames/(num * nframe))

for i in range(num):
    start = i * nframe * step
    stop = (i + 1) * nframe * step
    out_file = f"chunk_{i:02d}.xtc"
    
    with XTCWriter(out_file, n_atoms=u.atoms.n_atoms) as w:
        for ts in u.trajectory[start:stop:step]:
            w.write(u)
print(f"Trajectory split into {num} parts, each with {nframe} frames in the step of {step}.")

# ------------------------------
# Backmap reference structure
# ------------------------------
cmd = ["srun", "-c", "4",
       "-G", "1", 
       "-p", "gpu-preempt",
       "convert_cg2all",
       "-p", pdb,
       "-o", "conf_aa.pdb",
       "--cg", "BB",
       "--device", "cuda",
       "--batch", "2",
       "--chain-break-cutoff", '4']
subprocess.run(cmd, check=True)
print("conf_aa.pdb created.")

##monomer_aa.pdb
cmd = ["srun", "-c", "4",
       "-G", "1",
       "-p", "gpu-preempt",
       "convert_cg2all",
       "-p", "monomer.pdb",
       "-o", "monomer_aa.pdb",
       "--cg", "MC",
       "--device", "cuda",
       "--batch", "2",
       "--chain-break-cutoff", '4']
subprocess.run(cmd, check=True)
print("monomer_aa.pdb created.")

# ------------------------------
# Create PSF for all-atom system
# ------------------------------

gen = PsfGen()
gen.read_topology("/scratch3/workspace/shanlongli_umass_edu-amyloid/top_all36_prot.rtf")
gen.alias_residue(top_resname="HSE", pdb_resname="HIS")
for seg in u.segments:
    gen.add_segment(segid=seg.segid, pdbfile="monomer_aa.pdb", first="NTER", last="CTER", auto_angles=False, auto_dihedrals=False)
gen.write_psf("conf_aa.psf")
print("conf_aa.psf created.")

# ------------------------------
# Backmap trajectory chunks
# ------------------------------
print("Starting parallel backmapping with srun...")
processes = []
for i in range(num):
    cmd = [
        "srun",
        "-c", "4",
        "-G", "1",
        "-p", "gpu-preempt",
        "convert_cg2all",
        "-p", pdb,
        "-d", f"chunk_{i:02d}.xtc",
        "-o", f"chunk_{i:02d}_backmapped.xtc",
        "--cg", "MC",
        "--device", "cuda",
        "--chain-break-cutoff", '4',
    ]
    # Start process in background
    proc = subprocess.Popen(cmd)
    processes.append(proc)
    print(f"Started backmapping chunk {i}")

# Wait for all processes to complete
print("Waiting for all backmapping jobs to complete...")
for i, proc in enumerate(processes):
    proc.wait()
    if proc.returncode != 0:
        print(f"ERROR: Chunk {i} failed with return code {proc.returncode}")
    else:
        print(f"Chunk {i} completed successfully")

# ------------------------------
# Recombine backmapped parts
# ------------------------------
print("Recombining trajectory...")
out = "aa.xtc"
u0 = mda.Universe("conf_aa.psf", "conf_aa.pdb")
with XTCWriter(out, n_atoms=u0.atoms.n_atoms) as w:
    for i in range(num):
        in_file = f"chunk_{i:02d}_backmapped.xtc"
        u_part = mda.Universe("conf_aa.psf", in_file)
        for ts in u_part.trajectory:
            w.write(u_part)
print(f"Recombined all-atom trajectory saved to {out}.")

# ------------------------------
# Clean up intermediate files
# ------------------------------
print("Clean up...")
for i in range(num):
    os.remove(f"chunk_{i:02d}.xtc")
    os.remove(f"chunk_{i:02d}_backmapped.xtc")

print("Done!")
