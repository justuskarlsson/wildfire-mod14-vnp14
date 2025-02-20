#!/bin/bash
#SBATCH --array=0-3
#SBATCH --time=72:00:00
#SBATCH --job-name=wildfire_main
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --mem=128000
#SBATCH --cpus-per-task 16
#SBATCH -o /proj/cvl/users/x_juska/slurm_logs/wildfire_main_%a.out
#SBATCH -e /proj/cvl/users/x_juska/slurm_logs/wildfire_main_%a.err

options=(
    "--is_modis=True"
    "--is_modis=True --other_target=True"
    "--is_modis=False"
    "--is_modis=False --other_target=True"
)

names=(
    "modis_modis"
    "modis_viirs"
    "viirs_viirs"
    "viirs_modis"
)

# Execute the command corresponding to the SLURM_ARRAY_TASK_ID
python -u wildfire/run_train.py finetune --name="${names[$SLURM_ARRAY_TASK_ID]}" --num_runs=5 --test_save_all=True ${options[$SLURM_ARRAY_TASK_ID]} 