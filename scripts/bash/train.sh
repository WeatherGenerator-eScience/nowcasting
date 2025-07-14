#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-preempt
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:rtx_a5000:4
#SBATCH --time=48:00:00
#SBATCH --job-name=ldcast_nowcast
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/nowcasting/results/logs/%x/%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/mrobben/nowcasting/results/logs/%x/%j.err

# Change to the project's root directory
cd /vol/knmimo-nobackup/users/mrobben/nowcasting

# Activate the Python virtual environment
source ../.venv/bin/activate

# Execute the Python training script
# Only one `--config` or `--version_path` line should be uncommented at a time.

# Some examples:
python scripts/train.py --config="./config/ldcast_nowcast.yaml"
# python scripts/train.py --config="./config/ldcast_autoenc.yaml"
# python scripts/train.py --config="./config/earthformer_sat_l1p5.yaml"
# python scripts/train.py --version_path="./results/tb_logs/earthformer/version_59"