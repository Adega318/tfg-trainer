#!/bin/bash
#----------------------------------------------------
# Example SLURM job script with SBATCH
#----------------------------------------------------
#SBATCH -J mjob            # Job name
#SBATCH -o mjob_%j.out       # Name of stdout output file(%j expands to jobId)
#SBATCH -e mjob_%j.err       # Name of stderr output file(%j expands to jobId)
#SBATCH -c 32         	    # Cores per task requested
#SBATCH -t 16:00:00         # Run time (hh:mm:ss) - 10 min
#SBATCH --mem-per-cpu=3G    # Memory per core demandes (24 GB = 3GB * 8 cores)
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end 
#SBATCH --mail-type=fail 
#SBATCH --mail-user=eadegafernandez@gmail.com

module load cesga/2020 miniconda3/4.11.0

conda init bash 
source activate xtts

./train_gpt_xtts.sh
