#!/bin/bash -l
#SBATCH -p booster
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --account=training2203
#SBATCH --reservation=gpuhack22-30

module load Stages/2022 GCC/11.2.0 OpenMPI/4.1.2 Horovod/0.24.2-Python-3.9.6
cd /p/home/jusers/starovoitovs1/juwels/projects/makers

env > env.txt
srun python fbsde.py
