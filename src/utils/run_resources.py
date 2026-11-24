"""Fail-fast resource checks for explicitly authorized local training."""

import csv
import math
from pathlib import Path
import shutil
import subprocess


def check_free_storage(path, minimum_gib):
    if not math.isfinite(minimum_gib) or minimum_gib < 0:
        raise ValueError("Minimum free storage must be finite and nonnegative")
    existing = Path(path).resolve()
    while not existing.exists():
        existing = existing.parent
    free_gib = shutil.disk_usage(existing).free / 2 ** 30
    if free_gib < minimum_gib:
        raise RuntimeError(f"Insufficient storage: {free_gib:.1f} GiB free; require {minimum_gib:.1f} GiB")
    return free_gib


def require_idle_gpu(gpu):
    if not gpu or "," in gpu:
        raise ValueError("Idle-GPU check requires one explicit CUDA_VISIBLE_DEVICES index or UUID")
    base = ["nvidia-smi", "-i", gpu]
    processes = subprocess.check_output(
        [*base, "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    if processes:
        raise RuntimeError(f"GPU {gpu} already has compute processes; refusing to launch: {processes}")
    raw = subprocess.check_output(
        [*base, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
    values = list(csv.reader(raw.splitlines()))
    if len(values) != 1:
        raise RuntimeError("Could not establish the selected GPU's resource state")
    utilization, memory_mib = map(float, values[0])
    if not all(math.isfinite(value) for value in (utilization, memory_mib)):
        raise RuntimeError("GPU utilization or memory is unavailable")
    # Permit idle desktop display allocations, but reject substantial graphics use too.
    if utilization > 5 or memory_mib > 1024:
        raise RuntimeError(f"GPU {gpu} is busy: {utilization:g}% utilization, {memory_mib:g} MiB used")
    return {"gpu": gpu, "utilization_percent": utilization, "memory_used_mib": memory_mib}
