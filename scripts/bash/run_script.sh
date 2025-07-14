#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --mem=78G
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=script
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/nowcasting/results/logs/%x/%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/mrobben/nowcasting/results/logs/%x/%j.err

# Change to the project's root directory
cd /vol/knmimo-nobackup/users/mrobben/nowcasting

# Activate the Python virtual environment
source ../.venv/bin/activate

# Execute the Python script
python scripts/compute_stats.py