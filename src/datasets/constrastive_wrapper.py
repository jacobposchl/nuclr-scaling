from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch_brain.data import Data, Interval
from torch_brain.batching import chain
from torch_brain.datasets import Dataset, DatasetIndex, SpikingDatasetMixin
from torch_brain.utils import np_string_prefix


class ContrastiveDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Callable):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self._two_view = False

    def __getattr__(self, name: str):
        return self.dataset.__getattribute__(name)

    def enable_two_views(self, max_distance: float | None, seed: int):
        """Prepare the dataset to return two views on every __getitem__

        Args:
            max_distance: Max time delta between the two views
            seed: integer seed for numpy RNG
        """
        # Two view mode needs to keep sampling_intervals around to able to generate
        # valid start/end times for the second view. So, we load them once and cache it
        self._sampling_intervals = self.get_sampling_intervals()
        self._two_view = True
        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance must be nonnegative or None")
        self.max_distance = float("inf") if max_distance is None else max_distance
        self._rng = np.random.default_rng(seed)

    def get_sampling_intervals(self, *_args, **_kwargs) -> dict[str, Interval]:
        if self._two_view:
            return self._sampling_intervals

        return self.dataset.get_sampling_intervals()

    def _get_second_view(self, v: DatasetIndex):
        if self.max_distance == 0.0:
            return v

        dur = v.end - v.start
        intrvl = self._sampling_intervals[v.recording_id]

        # Sample only starts that fit inside an actual interval. Different
        # datasets may have gaps; the enclosing span is not a valid domain.
        lower = np.maximum(intrvl.start, v.start - self.max_distance)
        upper = np.minimum(intrvl.end - dur, v.start + self.max_distance)
        eligible = upper >= lower
        lower, upper = lower[eligible], upper[eligible]
        if len(lower) == 0:
            raise ValueError("No valid second-view window within max_distance")
        widths = upper - lower
        if len(lower) == 1:
            start = self._rng.uniform(lower[0], upper[0])
        elif widths.sum() == 0:
            start = lower[self._rng.integers(len(lower))]
        else:
            draw = self._rng.uniform(0, widths.sum())
            cumulative = widths.cumsum()
            chosen = min(np.searchsorted(cumulative, draw, side="right"), len(lower) - 1)
            previous = cumulative[chosen - 1] if chosen else 0.0
            start = lower[chosen] + draw - previous
        end = start + dur

        assert start >= intrvl.start[0]
        assert end <= intrvl.end[-1]

        ans = deepcopy(v)
        ans.start = start
        ans.end = end
        return ans

    def _get_one_item(self, index: DatasetIndex):
        data = self.dataset[index]
        model_inputs = self.tokenizer(data)
        if model_inputs is None:
            return None

        X = {
            "model_inputs": model_inputs,
            "unit_seqlen": len(data.units),
            "unit_ids": data.units.id.astype(str).tolist(),
            "probe_ids": data.units.probe_id.astype(str).tolist(),
        }
        return X

    def __getitem__(self, index: DatasetIndex):
        X1 = self._get_one_item(index)
        if not self._two_view:
            return X1

        if X1 is None:
            return None

        index2 = self._get_second_view(index)
        X2 = self._get_one_item(index2)

        if X2 is None:
            return None

        return X1, X2

    @staticmethod
    def collate(data: list[dict | None], two_view: bool = False):
        import itertools
        from torch_brain.batching import collate as tb_collate

        X = [x for x in data if x is not None]

        if two_view:
            # flatten the list of tuples, i.e. the two views are _interleaved_
            X = list(itertools.chain(*X))

        if len(X) == 0:
            return None

        out = {
            "model_inputs": tb_collate([x["model_inputs"] for x in X]),
            "unit_seqlen": tb_collate([x["unit_seqlen"] for x in X]),
            "unit_ids": np.concatenate([x["unit_ids"] for x in X]).astype(str),
            "probe_ids": np.concatenate([x["probe_ids"] for x in X]).astype(str),
        }
        return out
