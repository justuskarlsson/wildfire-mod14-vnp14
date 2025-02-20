#!/bin/bash
#SBATCH --array=0-1
#SBATCH --time=72:00:00
#SBATCH --job-name=wildfire_padding
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --mem=128000
#SBATCH --cpus-per-task 16
#SBATCH -o /proj/cvl/users/x_juska/slurm_logs/wildfire_padding_%a.out
#SBATCH -e /proj/cvl/users/x_juska/slurm_logs/wildfire_padding_%a.err

options=(
    "--is_modis=True --loss_pixel_padding=0"
    "--loss_pixel_padding=0"
)

names=(
    "modis_pad"
    "viirs_pad"
)

# Execute the command corresponding to the SLURM_ARRAY_TASK_ID
python -u wildfire/run_train.py finetune  --name=${names[$SLURM_ARRAY_TASK_ID]} ${options[$SLURM_ARRAY_TASK_ID]} 