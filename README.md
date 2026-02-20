# HyresBuilder: for running HyRes and iCon simulation.   
This package is built for the simulation of HyRes protein and iConRNA (iConDNA later) simulation.    
**Main functions:**
1. Construct HyRes peptide structure from sequence or convert atomistic structure into HyRes model;   
2. Construct iConRNA model from sequence or convert atomistic structure into iConRNA model;   
3. Set up HyRes and/or iConRNA force fields;   

## Dependencies:
1. [OpenMM](https://openmm.org/)
2. [CHARMM-GUI](https://www.charmm-gui.org/) (registration required)
3. [psfgen-python](https://psfgen.robinbetz.com/) (install through "conda install -c conda-forge psfgen")
4. basis: numpy, numba, MDAnalysis

## Installation: 
1. git clone https://github.com/lslumass/HyresBuilder.git   
2. cd into the download folder   
3. pip install .   


## Prepare PDB file:   
### A. Construct HyRes peptides structure from sequence   
1. from command line, use ```hyresbuilder```:   
```
usage: hyresbuilder [-h] name sequence

Build a peptide chain from sequence: hyresbuilder name sequence, output: name.pdb.

positional arguments:
  name        pdb file name, output will be name.pdb. default: hyres.pdb
  sequence    Amino acid sequence (single-letter codes, e.g., ACDEFG)

options:
  -h, --help  show this help message and exit
```
2. from script, use module of ```HyresBuilder.build_peptide```:
```
from HyresBuilder import HyresBuilder
HyresBuilder.build_peptide("name", "the sequence")
```
then name.pdb will be created.   

### B. Construct iConRNA coil strand from sequence
1. from comman line, use ```rnabuilder```:   
```
usage: rnabuilder [-h] name seq

RNABuilder: build iConRNA from sequence

positional arguments:
  name        protein name, output: name.pdb
  seq         sequence in one-letter

options:
  -h, --help  show this help message and exit
```
2. from script, use ```RNABuilder.build```:   
```
from HyresBuilder import RNABuilder
RNABuilder.build("name", "sequence")
```

### C. Convert atomistic structure into HyRes model
1. from command line, use ```convert2cg```:   
```
usage: convert2cg [-h] [--hydrogen] [--terminal TERMINAL] aa cg

Convert2CG: All-atom to HyRes/iConRNA converting

positional arguments:
  aa                    Input PDB file
  cg                    Output PDB file

options:
  -h, --help            show this help message and exit
  --hydrogen            add backbone amide hydrogen (HN only), default False
  --terminal TERMINAL, -t TERMINAL
                        Charge status of terminus: neutral, charged, NT, CT
```
**Note:** if your pdb doesn't have backbone amide hydrogen (H-N), use ```--hydrogen``` to add   
**Note:** ternimal is for setting the charged status of peptides   
**Warning:** make sure no duplicated chainID or segID for adjacent chains in pdb file    

2. from script, use the module of ```Convert2CG.at2cg```:   
```
from HyresBuilder import Convert2CG
Convert2CG.at2cg(AA_pdb, CG_pdb, terminal="neutral")
```
3. psf file will be automatically created!   

## Prepare PSF file:  
**Note:** It's better to have different chain ID for adjacent chain ID in pdb file.   
```changechains``` in packmol is usefull shen preparing phase separation simulation, follow [packmol.inp](./examples/packmol.inp) for an example.   

1. If converted using "Convert2CG" (Part C), psf file will be automatically created.   
2. Create psf from CG PDB:   
**from command line, use ```genpsf```:**     
```
usage: genpsf [-h] [-t {neutral,charged,NT,CT,positive}] pdb psf

generate PSF for Hyres/iCon systems

positional arguments:
  pdb                   CG PDB file(s)
  psf                   output name/path for PSF

options:
  -h, --help            show this help message and exit
  -t {neutral,charged,NT,CT,positive}, --ter {neutral,charged,NT,CT,positive}
                        Terminal charged status (choose from ['neutral', 'charged', 'NT', 'CT', 'positive']) (default: neutral)
```
**from script, use ```GenPsf.genpsf```:**   
```
from HyresBuilder import GenPsf
GenPsf.genpsf("input_pdb", "output_psf", terminal="neutral")
```

## Run simulation:
[run_latest.py](./scripts/run_latest.py) is used for running simulation. 
```
usage: run_latest.py [-h] [-c PDB] [-p PSF] [-o OUT] [-t TEMP] [-b BOX [BOX ...]] [-s SALT] [-e ENS] [-m MG]

options:
  -h, --help            show this help message and exit
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


## Banchmark:    
**platform:** OpenMM 8.2, GTX 2080Ti   
1. [monomer](./examples/benchmark_system/monomer/) system (800 beads): 13 us/day    
2. [GY23](./examples/benchmark_system/LLPS2) LLPS system (29K beads): 1.5 us/day   
3. [LAF1-RGG](./examples/benchmark_system/LLPS1/) LLPS system (105K beads): 550 ns/day   

**@ L40S:**
1. TDP43 (90K beads): 1.9 us/day   
2. EWSLCD (160K beads): 1.1 us/day   

