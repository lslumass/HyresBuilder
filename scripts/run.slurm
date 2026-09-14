#!/bin/bash
#SBATCH -c 1                                      # Number of Cores per Task
#SBATCH --mem=16GB                                # Requested Memory
#SBATCH -t 14-00:00:00
#SBATCH -p ials-gpu                                       # Partition
#SBATCH -G 1                                      # Number of GPUs
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --mail-type=FAIL
#SBATCH --job-name=PRL

python -u run_latest_Mg.py -e NVT -b 40 -t 303 -s 25 -m 5 -r "PRL"
