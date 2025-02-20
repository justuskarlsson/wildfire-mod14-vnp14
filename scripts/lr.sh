#!/bin/bash
#SBATCH --array=0-7
#SBATCH --time=72:00:00
#SBATCH --job-name=wildfire_lr
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --mem=128000
#SBATCH --cpus-per-task 16
#SBATCH -o /proj/cvl/users/x_juska/slurm_logs/wildfire_lr_%a.out
#SBATCH -e /proj/cvl/users/x_juska/slurm_logs/wildfire_lr_%a.err

options=(
    "--lr=0.0005 --is_modis=True --pos_weight=2.0"
    "--lr=0.001 --is_modis=True --pos_weight=2.0"
    "--lr=0.005 --is_modis=True --pos_weight=2.0"
    "--lr=0.01 --is_modis=True --pos_weight=2.0"
    "--lr=0.0005 --pos_weight=3.0"
    "--lr=0.001 --pos_weight=3.0"
    "--lr=0.005 --pos_weight=3.0"
    "--lr=0.01 --pos_weight=3.0"
)

# Execute the command corresponding to the SLURM_ARRAY_TASK_ID
python -u wildfire/run_train.py finetune --name="${options[$SLURM_ARRAY_TASK_ID]}" ${options[$SLURM_ARRAY_TASK_ID]} 
