import math
from typing import Dict, Iterator, List

import torch
import torch.distributed as dist
from torch_brain.data import Interval
from torch_brain.datasets import DatasetIndex


class DistributedSamplerWrapper(torch.utils.data.Sampler):
    """Wrapper for distributing any sampler across multiple processes.

    This wrapper takes an existing sampler and distributes its indices across
    multiple replicas (processes) in a distributed training setup. It ensures
    each replica gets a unique subset of the data by splitting the indices
    evenly across replicas.

    Note: This wrapper supports samplers that may return a different number of
    samples each epoch, as it recomputes the indices on each iteration.

    Note: If the length of the sampler is not a multiple of num_replicas, some
    samples will be dropped to ensure equal distribution across replicas.

    Args:
        sampler: The base sampler to wrap
        num_replicas: Number of processes in distributed training
        rank: Rank of the current process
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        drop_last: bool = False,
    ) -> None:

        if dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas, rank = 1, 0

        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

    def _get_indices(self):
        indices = list(self.sampler)

        if not self.drop_last:
            return indices[self.rank :: self.num_replicas]

        # Force len(indices) to be a multiple of num_replicas
        num_samples_per_rank = math.floor(len(indices) / self.num_replicas)
        total_size = num_samples_per_rank * self.num_replicas
        indices = indices[:total_size]
        return indices[self.rank : total_size : self.num_replicas]

    def __len__(self):
        total = len(self.sampler)
        if self.drop_last:
            return total // self.num_replicas
        return max(0, (total - self.rank + self.num_replicas - 1) // self.num_replicas)

    def __iter__(self):
        indices = self._get_indices()
        return iter(indices)

    def set_epoch(self, epoch: int):
        """Pass the epoch to the underlying sampler if it supports it."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)  # ty:ignore[call-non-callable]
