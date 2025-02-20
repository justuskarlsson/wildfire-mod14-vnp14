#!/bin/bash
#SBATCH --array=0-9
#SBATCH --time=72:00:00
#SBATCH --job-name=wildfire_pos_weight
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --mem=128000
#SBATCH --cpus-per-task 16
#SBATCH -o /proj/cvl/users/x_juska/slurm_logs/wildfire_pos_weight_%a.out
#SBATCH -e /proj/cvl/users/x_juska/slurm_logs/wildfire_pos_weight_%a.err

options=(
    "--pos_weight=1.0"
    "--pos_weight=2.0"
    "--pos_weight=3.0"
    "--pos_weight=5.0"
    "--pos_weight=10.0"
    "--pos_weight=1.0 --is_modis=True"
    "--pos_weight=2.0 --is_modis=True"
    "--pos_weight=3.0 --is_modis=True"
    "--pos_weight=5.0 --is_modis=True"
    "--pos_weight=10.0 --is_modis=True"
)

# Execute the command corresponding to the SLURM_ARRAY_TASK_ID
python -u wildfire/run_train.py finetune --name="${options[$SLURM_ARRAY_TASK_ID]}" --num_runs=5 ${options[$SLURM_ARRAY_TASK_ID]} 
