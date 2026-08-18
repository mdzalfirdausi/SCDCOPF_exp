#!/bin/bash
#SBATCH --job-name=scdcopf_data
#SBATCH --output=logs/array_%A_task_%a.out  # Standard output log
#SBATCH --error=logs/array_%A_task_%a.err   # Standard error log
#SBATCH --time=02:00:00                     # Max time per node (e.g., 2 hours)
#SBATCH --mem=4G                            # RAM required per node
#SBATCH --cpus-per-task=1                   # CPU cores required per node
#SBATCH --array=0-139                       # Splitting into 140 nodes

# =========================================================
# 1. ARGUMENT PARSING
# =========================================================
# Default case if none is provided
CASE_NAME="pglib_opf_case118_ieee"

# Parse arguments passed from the sbatch command
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --case) CASE_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Target Case: $CASE_NAME"

# =========================================================
# 2. CROSS-HPC ENVIRONMENT LOADER
# =========================================================
# Check the hostname of the current cluster node to load the right modules
HOSTNAME=$(hostname)

if [[ "$HOSTNAME" == *"hpc1_keyword"* ]]; then
    # Settings for your first HPC
    echo "Detected HPC 1..."
    source activate pytorch

elif [[ "$HOSTNAME" == *"hpc2_keyword"* ]]; then
    # Settings for your second HPC
    echo "Detected HPC 2..."
    conda activate pytorch

else
    # Fallback if hostname isn't recognized (assumes conda is in ~/.bashrc)
    echo "Unknown cluster node: $HOSTNAME. Using fallback Conda initialization..."
    source ~/.bashrc
    conda activate pytorch
fi

# =========================================================
# 3. EXECUTE THE PYTHON SCRIPT
# =========================================================
python run_experiments.py --case $CASE_NAME