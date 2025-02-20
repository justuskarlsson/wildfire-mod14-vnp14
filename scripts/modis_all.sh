#!/bin/bash
#SBATCH --array=0-3
#SBATCH --time=72:00:00
#SBATCH --job-name=modis_all
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --mem=128000
#SBATCH --cpus-per-task 16
#SBATCH -o /proj/cvl/users/x_juska/slurm_logs/modis_all_%a.out
#SBATCH -e /proj/cvl/users/x_juska/slurm_logs/modis_all_%a.err

options=(
    "--is_modis=True --swap_train_test=True"
    "--is_modis=True --other_target=True --swap_train_test=True"
    "--is_modis=False --swap_train_test=True"
    "--is_modis=False --other_target=True --swap_train_test=True"
)

names=(
    "modis_mod14_vnp14"
    "modis_vnp14_mod14"
    "viirs_vnp14_mod14"
    "viirs_mod14_vnp14"
)

# Execute the command corresponding to the SLURM_ARRAY_TASK_ID
python -u wildfire/run_train.py finetune --name="${names[$SLURM_ARRAY_TASK_ID]}" --num_runs=5 ${options[$SLURM_ARRAY_TASK_ID]} 