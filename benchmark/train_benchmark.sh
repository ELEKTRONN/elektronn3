#!/bin/bash -l

#SBATCH -o ./outputs/train-out.%j
#SBATCH -e ./outputs/train-err.%j
#SBATCH -D ./
#SBATCH -J E3_TRAIN_GPU_TEST
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

module purge
module load anaconda/3/2020.02

conda activate e3

## Single GPU:
# Train without mixed precision
srun python3 ./train_benchmark.py --jit=disabled
# Train with mixed precision (hopefully using Tensor Cores)
srun python3 ./train_benchmark.py --jit=disabled --amp

## Multi GPU data parallel:
# Train without mixed precision
srun python3 ./train_benchmark.py --jit=disabled --dp
# Train with mixed precision (hopefully using Tensor Cores)
srun python3 ./train_benchmark.py --jit=disabled --dp --amp
