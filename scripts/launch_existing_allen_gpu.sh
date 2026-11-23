#!/usr/bin/env bash
# Lightweight controller only; Python runs inside the supplied allocation.
set -euo pipefail
allocation_id=${1:?Pass an existing GPU job ID}
run_output=${2:?Pass the run output directory}
resume_checkpoint=${3:-}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
python_bin=${NUCLR_PYTHON:-$project_root/.venv/bin/python}
runner_args=(--output "$run_output")
if [[ -n "$resume_checkpoint" ]]; then
    runner_args+=(--resume "$resume_checkpoint")
fi
srun --jobid="$allocation_id" --partition=b200-mig45 --overlap --exact \
    --nodes=1 --ntasks=1 --cpus-per-task=6 --gres=gpu:1 --immediate=30 \
    /usr/bin/env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$project_root" \
    "$python_bin" -u scripts/run_allen_pretraining.py "${runner_args[@]}"
