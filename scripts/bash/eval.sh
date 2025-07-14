#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-preempt
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=Evaluation
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/nowcasting/results/logs/%x/%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/mrobben/nowcasting/results/logs/%x/%j.err

# Change to the project's root directory
cd /vol/knmimo-nobackup/users/mrobben/nowcasting

# Activate the Python virtual environment
source ../.venv/bin/activate

# Execute the Python evaluation script
# The `--config` argument specifies the YAML configuration file for the evaluation.
# Uncomment the lines that should be run for evaluation. 
# Note: PySTEPS does not require a GPU, while the others benifit greatly from it. 

# Examples:
python scripts/eval.py --config="./config/pysteps.yaml"
# python scripts/eval.py --config="./results/tb_logs/earthformer/version_58/earthformer.yaml"
# python scripts/eval.py --config="./results/tb_logs/ldcast_nowcast/version_10/ldcast_nowcast.yaml"
# python scripts/eval.py --config="./config/dgmr_uk.yaml"