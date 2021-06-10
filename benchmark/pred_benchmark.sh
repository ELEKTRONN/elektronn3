#!/bin/bash -l

#SBATCH -o ./pred-out.%j
#SBATCH -e ./pred-err.%j
#SBATCH -D ./
#SBATCH -J E3_PRED_GPU_TEST
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

module purge
module load anaconda/3/2020.02

conda activate e3

export CUDA_VISIBLE_DEVICES=0
srun python3 ./pred_benchmark.py
