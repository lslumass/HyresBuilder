#!/bin/bash
#SBATCH -c 1                                      # Number of Cores per Task
#SBATCH --mem=16GB                                # Requested Memory
#SBATCH -t 7-00:00:00
#SBATCH -p gpupod-l40s                                       # Partition
#SBATCH -q gpu-quota-16
#SBATCH -G 1                                      # Number of GPUs
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --mail-type=FAIL
#SBATCH --job-name=PRL

python -u run_latest_Mg.py -e NVT -b 255 -t 303 -s 25 -m 10 --custom "PRL"
