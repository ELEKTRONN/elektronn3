#!/bin/bash -l

#SBATCH -o ./outputs/pred-out.%j
#SBATCH -e ./outputs/pred-err.%j
#SBATCH -D ./
#SBATCH -J E3_PRED_GPU_TEST
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

module purge
module load anaconda/3/2020.02

conda activate e3

srun python3 ./pred_benchmark.py
