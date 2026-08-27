#!/bin/bash

#SBATCH --job-name=erdos_train
#SBATCH --partition=main

# Disable default slurm logs to use our custom exec routing
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Training takes a bit longer, so we increase the time limit
#SBATCH --time=4:00:00

# PyTorch Geometric trains faster with a few extra CPU cores for data loading
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# =========================================================
# 0. CUSTOM LOG ROUTING (erdos_ prefix)
# =========================================================
LOG_DIR="logs/erdos_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

exec > "${LOG_DIR}/erdos_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/erdos_${SLURM_JOB_ID}.err"

# =========================================================
# 1. CASE / ARGUMENT PARSING
# =========================================================

CASE_NAME=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --case)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --case requires a value."
                exit 1
            fi
            CASE_NAME="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# --case is mandatory
if [[ -z "$CASE_NAME" ]]; then
    echo "ERROR: --case is required."
    echo "Usage: sbatch submit_erdos.sh --case <case_name>"
    exit 1
fi

# =========================================================
# 2. JOB INFORMATION
# =========================================================

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Case:          $CASE_NAME"
echo "Start time:    $(date)"
echo "============================================"

# =========================================================
# 3. CONDA ENVIRONMENT
# =========================================================

# Initialize Conda explicitly for non-interactive Slurm shells
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate pytorch

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
python --version

# =========================================================
# 4. PREVENT EACH JOB FROM SPAWNING EXTRA CPU THREADS
# =========================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# =========================================================
# 5. RUN
# =========================================================

echo "Starting Unsupervised Erdős-GNN Training..."

python gnn_erdos.py \
    --case "$CASE_NAME"

EXIT_CODE=$?

echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE