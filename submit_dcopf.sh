#!/bin/bash
#SBATCH --job-name=scdcopf_data
#SBATCH --output=logs/array_%A_task_%a.out  # Standard output log
#SBATCH --error=logs/array_%A_task_%a.err   # Standard error log
#SBATCH --time=02:00:00                     # Max time per node (e.g., 2 hours)
#SBATCH --mem=4G                            # RAM required per node
#SBATCH --cpus-per-task=1                   # CPU cores required per node

# =========================================================
# THIS IS THE MAGIC LINE THAT SPLITS IT INTO 140 NODES
# =========================================================
#SBATCH --array=0-139                       

# 1. Load your Python/Conda environment
module load anaconda3  # (Change this to match your cluster's module name)
source activate gur

# 2. Execute the Python script
python run_experiments.py --case 118_ieee