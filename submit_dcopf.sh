#!/bin/bash
#SBATCH --job-name=scdcopf_gen
#SBATCH --output=logs/dcopf_%A_%a.out
#SBATCH --error=logs/dcopf_%A_%a.err
#SBATCH --time=04:00:00                 # Max 4 hours per chunk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Give Gurobi 4 cores per job
#SBATCH --mem=8G                        # 8GB RAM per node
#SBATCH --array=0-139                   # 140 total tasks (140 * 100 = 14000 instances)

# Load your Anaconda/Miniconda module and activate the environment
module load anaconda3
source activate your_optimization_env

# Execute the wrapper script
python run_experiments.py