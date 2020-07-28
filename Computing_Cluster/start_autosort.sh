#!/bin/bash
#SBATCH --job-name=Autosort_%j
#SBATCH --output=Autosort_%j.log
#SBATCH --error=Autosort_error_%j.log
#SBATCH --partition=Standard
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 8
#SBATCH --mem=48G
module purge > /dev/null 2>&1
cd /data/home/dmarshal/Autosort
module add miniconda
python3 Cluster_Autosort_Final.py
mv Autosort_${SLURM_JOBID}.log Results
mv Autosort_error_${SLURM_JOBID}.log Results