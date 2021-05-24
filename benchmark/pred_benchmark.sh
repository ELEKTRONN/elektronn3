#!/bin/bash -l

#SBATCH -o ./outputs/out.%j
#SBATCH -e ./outputs/err.%j
#SBATCH -D ./
#SBATCH -J GPU_TEST
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

module purge
# module load cuda/10.2
module load anaconda/3/2020.02
# module load pytorch/gpu/1.7.0

# export OMP_NUM_THREADS=1

conda activate e3

srun python3 ./pred_benchmark.py
