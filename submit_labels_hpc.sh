#!/bin/bash

#SBATCH --job-name=scdcopf
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032

# SAVE YOUR LOGS! This writes them to your data/labels folder
#SBATCH --output=data/labels/log_%A_%a.out
#SBATCH --error=data/labels/log_%A_%a.err

#SBATCH --time=2-00:00:00

#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# 1000 total scenarios / 100 scenarios per task = 10 tasks
#SBATCH --array=0-9

# =========================================================
# 1. ARGUMENT PARSING (Supports both --case name and plain name)
# =========================================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --case) CASE_NAME="$2"; shift ;;
        *) CASE_NAME="$1" ;;
    esac
    shift
done

if [ -z "$CASE_NAME" ]; then
    echo "ERROR: You must provide a case name."
    echo "Usage: sbatch submit_labels_hpc.sh --case <case_name>  OR  sbatch submit_labels_hpc.sh <case_name>"
    exit 1
fi

echo "Running Ground-Truth Generation for case: $CASE_NAME"

# Ensure the output directory exists
mkdir -p data/labels

# =========================================================
# 2. ARRAY MATH
# =========================================================
# 1000 total scenarios divided into 10 chunks of 100
CHUNK_SIZE=100
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing scenarios $START_IDX to $END_IDX"

# =========================================================
# 3. ENVIRONMENT
# =========================================================
# Load HPC Gurobi configuration
module purge
module load gurobi/13.0.1

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

# Verify Gurobi before starting experiments
python -c "
import gurobipy as gp
print('gurobipy version:', gp.gurobi.version())
m = gp.Model()
print('Gurobi license: OK')
" || {
    echo "ERROR: Gurobi license check failed."
    exit 1
}

echo "============================================"
echo "Environment Ready"
echo "============================================"

# Pass the dynamic variable to your python script
python generate_ground_truth.py --case $CASE_NAME --start_idx $START_IDX --end_idx $END_IDX