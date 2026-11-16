# --------------------------------------------------
# Acknowledgments
# Adapted from code by: Vinam Arora, Divyansh Lachi
# --------------------------------------------------

import subprocess
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch


def save_ckpt(filepath: Path, **kwargs) -> Path:
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        git_hash = None
    checkpoint = {"git_hash": git_hash}

    for k, v in kwargs.items():
        if isinstance(v, torch.nn.parallel.DistributedDataParallel):
            checkpoint[f"{k}_state_dict"] = v.module.state_dict()
        elif hasattr(v, "state_dict"):
            checkpoint[f"{k}_state_dict"] = v.state_dict()
        else:
            checkpoint[k] = v

    filepath.parent.mkdir(exist_ok=True, parents=True)
    # Keep the previous rolling checkpoint intact if saving is interrupted.
    fd, temporary = tempfile.mkstemp(prefix="." + filepath.name + "-", suffix=".tmp", dir=filepath.parent)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, filepath)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return filepath
