#!/bin/bash
#
# Define a job name.
#SBATCH -J Optim
#SBATCH -o CPU_myjob..o # Name of the stdout output file
#SBATCH -e CPU_myjob..e # Name of stderr error file
#SBATCH -p compute3 # Queue (partition) name
#SBATCH -N 1 # Total # of nodes 
#SBATCH -n 1 # Total # of mpi tasks
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH --mail-type=ALL
#SBATCH --mail-user juan.navarro@my.utsa.edu
# USER PARAMETERS
# ==== ==========

module load anaconda3
# %j
# module load cudatoolkit/12.8.0_570.86.10
# module load cuda/cudnn/9.2.0.82

# module list

# conda activate CUDA_GP

# export XLA_PYTHON_CLIENT_PREALLOCATE=false   # avoids OOM on shared nodes
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80   # use 75% of GPU memory

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export CUDA_VISIBLE_DEVICES=0

# python MyOptimization.py
python NewPlot.py