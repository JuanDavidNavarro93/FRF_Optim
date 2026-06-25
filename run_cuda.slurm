#!/bin/bash
#
# Define a job name.
#SBATCH -J MYCUDA_GP
#SBATCH -o GPU_myjob.o # Name of the stdout output file
#SBATCH -e GPU_myjob.e # Name of stderr error file
#SBATCH -p gpu1v100 # Queue (partition) name
#SBATCH -N 1 # Total # of nodes 
#SBATCH -n 1 # Total # of mpi tasks
#SBATCH -c 40 # Total # of gpu tasks
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH --mail-type=ALL
#SBATCH --mail-user juan.navarro@my.utsa.edu
# USER PARAMETERS
# ==== ==========

module load anaconda3
module load cudatoolkit/12.8.0_570.86.10
module load cuda12/cudnn/9.2.0.82

module list

conda activate CUDA_GP

export XLA_PYTHON_CLIENT_PREALLOCATE=false   # avoids OOM on shared nodes
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99   # use 75% of GPU memory

ulimit -s unlimited

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

# python MyCuda_GP.py
python Bivar_Cuda_GP.py
# JDN  my comment SBATCH --gres=gpu:v100:1 