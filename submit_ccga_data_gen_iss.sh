#!/bin/bash

#SBATCH --job-name=ccga_data
#SBATCH --partition=main
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

#SBATCH --time=08:00:00

# Allocate 16 cores per task to accelerate Gurobi's solving
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# 1000 scenarios / 10 scenarios per task = 100 tasks
#SBATCH --array=0-99

LOG_DIR="logs/ccga_${SLURM_ARRAY_JOB_ID}"

mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to the log directory
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
    echo "Usage: sbatch submit_ccga_gen.sh --case <case_name>"
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
# 3. ENVIRONMENT
# =========================================================

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

echo "Python: $(which python)"
python --version

echo "Waiting for Gurobi license..."
for i in {1..20}; do
    python -c "import gurobipy as gp; m = gp.Model()" 2>/dev/null && break
    echo "Attempt $i: Token server full or busy. Sleeping 60s..."
    sleep 60
    if [ $i -eq 20 ]; then
        echo "ERROR: Could not get Gurobi license after 20 minutes."
        exit 1
    fi
done
echo "Gurobi license: OK"

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

# Pass ONLY the case name; the python script handles task_id via os.environ automatically
python generate_ccga_dataset.py --case "$CASE_NAME"

EXIT_CODE=$?

echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE