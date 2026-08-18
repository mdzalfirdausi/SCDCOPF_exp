#!/bin/bash

#SBATCH --job-name=scdcopf
#SBATCH --partition=cpu_x440

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Maximum allowed by the main partition
#SBATCH --time=7-00:00:00

# One CPU per independent Python process
#SBATCH --cpus-per-task=1

# Start conservatively; increase after measuring actual usage
#SBATCH --mem=8G

#SBATCH --array=0-9

LOG_DIR="logs/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$LOG_DIR"

exec > "${LOG_DIR}/${SLURM_ARRAY_TASK_ID}.out" \
     2> "${LOG_DIR}/${SLURM_ARRAY_TASK_ID}.err"

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
    echo "Usage: sbatch submit_iss.sh --case <case_name>"
    exit 1
fi

# =========================================================
# 2. JOB INFORMATION
# =========================================================

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Job ID:  $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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

python run_experiments.py \
    --case "$CASE_NAME"

EXIT_CODE=$?

echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE