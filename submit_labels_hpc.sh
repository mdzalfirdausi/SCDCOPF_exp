#!/bin/bash

#SBATCH --job-name=scdcopf
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

#SBATCH --time=2-00:00:00

#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# 1000 total scenarios / 100 scenarios per task = 10 tasks
#SBATCH --array=0-9

# =========================================================
# 0. LOG REDIRECTION
# =========================================================
LOG_DIR="logs/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$LOG_DIR"

exec > "${LOG_DIR}/ccga_${SLURM_ARRAY_TASK_ID}.out" \
     2> "${LOG_DIR}/ccga_${SLURM_ARRAY_TASK_ID}.err"

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
            CASE_NAME="$1"
            shift
            ;;
    esac
done

if [[ -z "$CASE_NAME" ]]; then
    echo "ERROR: You must provide a case name."
    echo "Usage: sbatch submit_labels_hpc.sh --case <case_name>  OR  sbatch submit_labels_hpc.sh <case_name>"
    exit 1
fi

# Ensure data directory exists
mkdir -p data/labels

# =========================================================
# 2. ARRAY MATH & JOB INFO
# =========================================================
CHUNK_SIZE=100
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Job ID:  $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node:          $(hostname)"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Case:          $CASE_NAME"
echo "Range:         Scenarios $START_IDX to $END_IDX"
echo "Start time:    $(date)"
echo "============================================"

# =========================================================
# 3. ENVIRONMENT & GUROBI SETUP
# =========================================================
module purge
module load gurobi/13.0.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

# Thread limits to prevent CPU oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
echo "Environment: Conda ($CONDA_DEFAULT_ENV) | Python ($(which python))"
echo "============================================"

# =========================================================
# 4. RUN
# =========================================================
python generate_ground_truth.py \
    --case "$CASE_NAME" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX"

EXIT_CODE=$?

echo "============================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE