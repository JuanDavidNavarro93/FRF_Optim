#!/bin/bash
#SBATCH -J MYCUDA_GP
#SBATCH -o GPU_myjob.o
#SBATCH -e GPU_myjob.e
#SBATCH -p gpu1v100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=juan.navarro@my.utsa.edu

module load anaconda3
module load cudatoolkit/12.8.0_570.86.10
module load cuda12/cudnn/9.2.0.82

# source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate CUDA_GP

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

ulimit -s unlimited

# python Bivar_Cuda_GP.py
python Sparse_GP.py

