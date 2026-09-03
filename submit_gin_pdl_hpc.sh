#!/bin/bash

#SBATCH --job-name=gin_pdl_hpc
#SBATCH --partition=gpu_x450
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu0002

# Disable default slurm logs to use custom exec routing
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# =========================================================
# 0. LOG REDIRECTION
# =========================================================
LOG_DIR="logs/gin_pdl_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR" data/models

exec > "${LOG_DIR}/train.out" \
     2> "${LOG_DIR}/train.err"

# =========================================================
# 1. ARGUMENT CONFIGURATION
# =========================================================
CASE_NAME="${1:-pglib_opf_case300_ieee}"
OUTER_K=20
INNER_L=50
BATCH_SIZE=32

echo "============================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Host Node:      $(hostname)"
echo "Partition:      $SLURM_JOB_PARTITION"
echo "CPUs Allocated: $SLURM_CPUS_PER_TASK"
echo "Memory:         $SLURM_MEM_PER_NODE"
echo "Case Study:     $CASE_NAME"
echo "Start Time:     $(date)"
echo "============================================"

# =========================================================
# 2. ENVIRONMENT & THREAD SETUP
# =========================================================
# Source modules properly to avoid "module: command not found" in batch mode
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null
module purge
module load gurobi/13.0.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify CUDA recognition on allocated GPU node
python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Active Device:', torch.cuda.get_device_name(0))
" || {
    echo "ERROR: CUDA device verification failed."
    exit 1
}

# =========================================================
# 3. EXECUTE GIN-PDL SOLVER
# =========================================================
python gin_pdl_scopf.py \
    --case "$CASE_NAME" \
    --outer_K "$OUTER_K" \
    --inner_L "$INNER_L" \
    --batch_size "$BATCH_SIZE"

EXIT_CODE=$?

echo "============================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE