# HyresBuilder: for running HyRes and iCon simulation.   
This package is built for the simulation of HyRes protein and iConRNA (iConDNA later) simulation.    
**Main functions:**
1. Construct HyRes peptide structure from sequence or convert atomistic structure into HyRes model;   
2. Construct iConRNA model from sequence or convert atomistic structure into iConRNA model;   
3. Set up HyRes and/or iConRNA force fields;   
4. Backmap CG structures to all-atom ones.   


## Dependencies:
1. [OpenMM](https://openmm.org/)
2. [CHARMM-GUI](https://www.charmm-gui.org/) (registration required)
3. [psfgen-python](https://psfgen.robinbetz.com/) (install through "conda install -c conda-forge psfgen", python < 3.10)
4. basis: numpy, numba, MDAnalysis, Biopython

## Installation: 
1. git clone https://github.com/lslumass/HyresBuilder.git   
2. cd into the download folder   
3. python setup.py install   


## Prepare PDB file:   
### A. Construct HyRes peptides structure from sequence   
Please follow the [examples](examples) for details of the script and sequence files.   

1. for a single idp:      
`python simple_example.py tdp-43-lcd.seq tdp-43_hyres.pdb`     
2. for a set of idps:   
To quickly build a series of peptides, one can use:   
`python bactch_example.py idps.seq`   

### B. Convert atomistic structure into HyRes model
Follow the detailed steps [here](./scripts/at2hyres/README.md)   

### C. Construct iConRNA coil strand from sequence
Follow the example [build_CG_RNA.py](./examples/build_CG_RNA.py)
`python build_CG_RNA.py out.pdb sequence`

### D. Convert atomistic RNA structure into iConRNA model
Follow the detailed steps [here](./scripts/at2iCon/README.md)

## Prepare PSF file:  
[psfgen_latest.py](./scripts/psfgen_latest.py) is used to generate psf file for all kinds of scenarios.   
```
usage: psfgen_latest.py [-h] [-d MODEL] -i INPUT_PDB_FILES [INPUT_PDB_FILES ...] -o OUTPUT_PSF_FILE [-n NUM_OF_CHAINS [NUM_OF_CHAINS ...]]
                        [-m MOLECULE_TYPE [MOLECULE_TYPE ...]] [-t {neutral,charged,NT,CT}]

generate PSF for Hyres systems

options:
  -h, --help            show this help message and exit
  -d MODEL, --model MODEL
                        simulated system: protein, RNA, or mix (default: mix)
  -i INPUT_PDB_FILES [INPUT_PDB_FILES ...], --input_pdb_files INPUT_PDB_FILES [INPUT_PDB_FILES ...]
                        Hyres PDB file(s), it should be the pdb of monomer (default: None)
  -o OUTPUT_PSF_FILE, --output_psf_file OUTPUT_PSF_FILE
                        output name/path for Hyres PSF (default: None)
  -n NUM_OF_CHAINS [NUM_OF_CHAINS ...], --num_of_chains NUM_OF_CHAINS [NUM_OF_CHAINS ...]
                        Number of copies for each pdb; it should have the same length as the given pdb list specified in the '-i' argument (default: [1])
  -m MOLECULE_TYPE [MOLECULE_TYPE ...], --molecule_type MOLECULE_TYPE [MOLECULE_TYPE ...]
                        select from 'P', 'R', 'D', 'C' for protein, RNA, DNA, complex, respectively (default: ['P'])
  -t {neutral,charged,NT,CT}, --ter {neutral,charged,NT,CT}
                        Terminal charged status (choose from ['neutral', 'charged', 'NT', 'CT']) (default: neutral)
```
**arguments:**
```
-d model: use "protein" for HyRes_GPU model; "RNA" for iConRNA model; 'mix' for the latest Hyres_iCon integration
-t ter: Terminal charged status for peptides. 'neutral' for neutral; 'charged' for charged N/C-terminus; 'NT' for charged N-terminus but neutral C-terminus; 'CT' for charged C-terminus but neutral N-terminus.
```
**Here are some examples:**
1. generate PSF for a single-chain protein
```
python psfgen_latest.py -i protein.pdb -o protein.psf
python psfgen_latest.py -i protein.pdb -o protein.psf -ter charged
```
2. generate PSF for a multi-chain protein
```
python psfgen_latest.py -i complex.pdb -n 1 -m C -o complex.psf
```
3. generate PSF for LLPS simulation of one kind of protein
```
python psfgen_latest.py -i chainA.pdb -n 100 -o llps.psf
```
4. generate PSF for LLPS simulation of multiple kinds of protein
```
# 20 copies of chain A + 30 copies of chain B
python psfgen_latest.py -i chainA.pdb chainB.pdb -n 20 30 -m P P -o llps.psf   
```

## Run simulation:
[run_latest.py](./scripts/run_latest.py) is used for running simulation. 
```
usage: run_latest.py [-h] [-d MODEL] [-c PDB] [-p PSF] [-o OUT] [-t TEMP] [-b BOX [BOX ...]] [-s SALT] [-e ENS] [-m MG]

options:
  -h, --help            show this help message and exit
  -d MODEL, --model MODEL
                        simulated system: protein, RNA, or mix
  -c PDB, --pdb PDB     pdb file, default is conf.pdb
  -p PSF, --psf PSF     psf file, default is conf.psf
  -o OUT, --out OUT     the prefix name for the output files, including xtc, pdb, log, chk
  -t TEMP, --temp TEMP  system temperature, default is 303 K
  -b BOX [BOX ...], --box BOX [BOX ...]
                        box dimensions in nanometer, e.g., '50 50 50'
  -s SALT, --salt SALT  salt concentration in mM, default is 150.0 mM
  -e ENS, --ens ENS     simulation ensemble, NPT, NVT, or non, non is for non-periodic system
  -m MG, --Mg MG        Mg2+ concentration in mM
```

**Simple examples:**
1. simulate single chain in non-periodic system under 150 mM NaCl at 298 K:   
```python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e non -s 150```
2. simulate RNA chain with 5 mM MgCl2 in a 10 nm cubic box at 298 K:   
```python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e NVT -b 10 -s 150 -m 5```
3. slab simulation of condensate at 15*15*50 nm rectangle box:   
```python run_latest.py -c conf.pdb -p conf.psf -o test -t 298 -e NVT -b 15 15 50 -s 150```