#!/bin/bash
#SBATCH --job-name=era5-download
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --propagate=NONE

set -euo pipefail

# ========= Settings ============
PROJECT_DIR="$SLURM_SUBMIT_DIR"
CONDA_ENV="data-era5"
DATA_DIR="$PROJECT_DIR/data/example"
LOG_DIR="$PROJECT_DIR/logs"
MAX_WORKERS="$SLURM_CPUS_PER_TASK"   # each worker downloads one time block of one variable/level

START="2020-07-01"
END="2020-07-02"             # inclusive
REGION=(24 50 -125 -66)      # LAT_MIN LAT_MAX LON_MIN LON_MAX
CHUNKS=(720 35 79)           # time (hours), latitude, longitude

VARS=(
  --vars t2m single_level 2m_temperature
  --vars t pressure_level temperature 500 850
)
# ================================

cd "$PROJECT_DIR"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "Job $SLURM_JOB_ID on $(hostname): $START to $END"
python download.py \
  --start "$START" --end "$END" \
  --bbox "${REGION[@]}" \
  --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" \
  --chunks "${CHUNKS[@]}" --max-workers "$MAX_WORKERS" \
  "${VARS[@]}"
