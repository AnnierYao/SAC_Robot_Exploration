#!/bin/bash

#SBATCH --job SAC
#SBATCH --partition=gpu3090
#SBATCH --qos 2gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --output=%j.out
#SBATCH --error=%j.err


module load cuda/11.8
conda init
source ~/.bashrc
cd $SLURM_SUBMIT_DIR
conda activate pytorch201
python main.py

