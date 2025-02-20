#!/bin/bash
#SBATCH --array=0-1
#SBATCH --time=72:00:00
#SBATCH --job-name=wildfire_download
#SBATCH --gpus=0
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH -o /proj/cvl/users/x_juska/slurm_logs/wildfire_download_v2_%a.out
#SBATCH -e /proj/cvl/users/x_juska/slurm_logs/wildfire_download_v2_%a.err

options=(
    # "--start_date 2019-10-01 --end_date 2020-01-31"
    # "--start_date 2023-10-01 --end_date 2024-01-31 --test=True"
    # "--start_date 2019-10-01 --end_date 2020-01-31 --is_modis=True"
    # "--start_date 2023-10-01 --end_date 2024-01-31 --is_modis=True --test=True"

)

# python -u wildfire/main.py download ${options[$SLURM_ARRAY_TASK_ID]}
