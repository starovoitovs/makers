#!/bin/bash -l
#SBATCH -p booster
#SBATCH -t 02:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --account=training2203

module load Stages/2022 GCC/11.2.0 OpenMPI/4.1.2 Horovod/0.24.2-Python-3.9.6 Nsight-Systems
cd /p/home/jusers/$USER/juwels/projects/makers

#srun nsys profile -t mpi,cuda,nvtx --mpi-impl=openmpi -s none -f true -o /p/scratch/training2203/makers/profiler/profiler.%q{SLURM_PROCID}.qdrep python fbsde.py
srun python fbsde.py --n_paths=262144

