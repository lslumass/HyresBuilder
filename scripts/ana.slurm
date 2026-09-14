#!/bin/bash
#SBATCH -c 40                                      # Number of Cores per Task
#SBATCH --nodes=1
#SBATCH --mem=16GB                                # Requested Memory
#SBATCH -t 2-00:00:00
#SBATCH -p cpu,cpu-preempt                                       # Partition
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --job-name=analysis

scical aggr --top conf.psf --traj system.dcd --ref P --rcut 10 --stride 10 --nproc 40 --sel R001-R100
