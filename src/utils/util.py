import logging
import os
import sys
from pathlib import Path
from typing import Any
import copy
from omegaconf import OmegaConf, DictConfig

import numpy as np
import torch
import torch.distributed as dist


class Precision:
    def __init__(self, precision_str: str):
        self.precision_str = precision_str

        if precision_str == "fp32":
            self.dtype = torch.float32
        elif precision_str == "fp16":
            self.dtype = torch.float16
        elif precision_str == "bf16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Invalid precision string: {precision_str}")

    def __repr__(self):
        return self.precision_str

    def __str__(self):
        return self.precision_str


def move_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    elif isinstance(data, (str, int, float, bool, type(None), np.ndarray)):
        # Metadata/scalars do not need device transfer.
        return data
    else:
        raise TypeError(f"Unknown data type {type(data)}")


def seed_everything(seed):
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_zero_only(func):
    def wrapper(self, *args, **kwargs):
        if self.rank != 0:
            return
        return func(self, *args, **kwargs)

    return wrapper


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _count_params(module: torch.nn.Module, trainable_only: bool = False):
    total = 0
    skipped = []
    for name, p in module.named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        try:
            total += p.numel()
        except ValueError:
            skipped.append(name)
    return total, skipped


def get_num_params(module: torch.nn.Module, return_skipped: bool = True):
    total, skipped = _count_params(module, trainable_only=False)
    if return_skipped:
        return total, skipped
    return total


def get_num_trainable_params(module: torch.nn.Module, return_skipped: bool = True):
    total, skipped = _count_params(module, trainable_only=True)
    if return_skipped:
        return total, skipped
    return total


def human_readable(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def validate_weights(module: torch.nn.Module):
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module but got {type(module)}")

    nan_weights = []
    for k, v in module.named_parameters():
        if not v.isfinite().all():
            nan_weights.append(k)
    if len(nan_weights) != 0:
        raise RuntimeError(f"Found NaN weights in layers: {nan_weights}")


def get_cli_logger(all_ranks=False):
    # Custom filter to convert absolute paths to relative paths
    class RelativePathFilter(logging.Filter):
        def filter(self, record):
            try:
                record.relpath = os.path.relpath(record.pathname)
            except ValueError:
                # pathname may be on a different drive/root than the cwd
                record.relpath = record.pathname
            return True

    LOGGING_FORMAT = (
        "\033[1;33m%(levelname)s\033[0m | "
        "\033[1;36m%(asctime)s\033[0m | "
        "\033[1;32m%(relpath)s:%(lineno)-4d\033[0m | "
        "%(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_FORMAT,
        force=True,
        stream=sys.stdout,
    )
    logger = logging.getLogger()
    # Attach to the handlers, not the logger: a logger-level filter only sees
    # records logged directly through it, so records propagated from library
    # loggers (numexpr, etc.) would reach the formatter without `relpath`.
    relpath_filter = RelativePathFilter()
    for handler in logger.handlers:
        handler.addFilter(relpath_filter)
    if not all_ranks:
        if dist.is_initialized() and not dist.get_rank() == 0:
            logger.setLevel(logging.ERROR)
    return logger


def is_divisible(x: float, y: float, tol: float = 1e-6) -> bool:
    """Returns true if x is divisible by y"""
    return abs(x / y - round(x / y)) < tol


def unpad_sequences(padded_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Remove padding from a padded tensor without sorting.

    Args:
        padded_tensor: Tensor of shape [batch_size, max_seq_len, ...]
        lengths: Actual sequence lengths [batch_size]

    Returns:
        Concatenated tensor of shape [sum(lengths), ...]
    """
    sequences = [padded_tensor[i, : lengths[i]] for i in range(padded_tensor.size(0))]
    return torch.cat(sequences, dim=0)


def unchain_sequences(
    chained_tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Reconstruct padded tensor from a concatenated (unchained) tensor.

    Args:
        chained_tensor: Concatenated tensor [sum(lengths), ...]
        lengths: Length of each sequence [batch_size]

    Returns:
        Padded tensor of shape [batch_size, max_seq_len, ...]
    """
    sequences = torch.split(chained_tensor, lengths.tolist(), dim=0)
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)


def instantiate(dictconfig: DictConfig | dict, *args, **kwargs) -> Any:
    """
    Instantiate a class from a configuration dictionary.

    Args:
        cfg: Configuration with _target_ field specifying the class path
        **kwargs: Additional arguments to pass to the class constructor

    Returns:
        An instance of the specified class
    """
    if isinstance(dictconfig, DictConfig):
        cfg_dict = OmegaConf.to_container(dictconfig)
    else:
        cfg_dict = copy.deepcopy(dictconfig)

    assert isinstance(cfg_dict, dict)

    target = cfg_dict.pop("_target_", None)
    assert target is not None
    cls = get_class_from_path(target)
    return cls(*args, **cfg_dict, **kwargs)  # ty:ignore[invalid-argument-type]


def get_class_from_path(path: str):
    import importlib

    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls
