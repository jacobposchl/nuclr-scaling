import copy
from typing import Callable, Literal, TypeAlias

import torch
import numpy as np
from torch_brain.data import Data, Interval
from torch_brain.datasets import Dataset, DatasetIndex

TaskType: TypeAlias = Literal["regression", "classification"]


class BehaviorDatasetMixin:
    def enable_behavior(self):
        raise NotImplementedError()

    def get_target(self, data: Data) -> dict:
        raise NotImplementedError()

    def get_behavior_sampling_intervals(self) -> dict[str, Interval]:
        raise NotImplementedError()

    @property
    def task_type(self) -> TaskType:
        raise NotImplementedError()

    @property
    def loss_name(self) -> str:
        raise NotImplementedError()

    @property
    def metric_names(self) -> list[str]:
        raise NotImplementedError()


class _DatasetWithBehavior(BehaviorDatasetMixin, Dataset): ...


class BehaviorDatasetWrapper(torch.utils.data.Dataset):
    """A wrapper that allows for the NLE+behavior-decoder framework

    It emits data as an (X,Y) tuple, where X is the inputs needed
    for the decoder, and Y contains target information.

    If :meth:``connect_unit_encoder`` has been called, then
    the dataset will emit X["encoder_input"]

    Args:
        dataset: The dataset that you want to wrap.
            Should have at least the following two parents:
            ``BehaviorDatasetMixin`` and ``torch_brain.datasets.Dataset``
        allow_none_sample: If True, a ``None`` sample return will raise an error.
            (default True)
    """

    _has_unit_encoder: bool = False
    _decoder_connected: bool = False

    def __init__(
        self,
        dataset: _DatasetWithBehavior,
        allow_none_sample: bool = False,
    ):
        self.dataset = dataset
        self.dataset.enable_behavior()
        self.allow_none_sample = allow_none_sample

    def __getattr__(self, name: str):
        return self.dataset.__getattribute__(name)

    def connect_decoder(self, decoder):
        self.decoder_tokenizer: Callable = decoder.tokenize
        self._decoder_connected = True

    def connect_unit_encoder(self, unit_encoder, window_gap: float = 0.0):
        self._has_unit_encoder = True
        self.unit_enc_ctx: float = unit_encoder.ctx_duration
        self.unit_enc_tokenizer: Callable = unit_encoder.tokenize
        self.unit_enc_window_gap = window_gap

    def get_sampling_intervals(self):
        return self.dataset.get_behavior_sampling_intervals()

    def __getitem__(self, index: DatasetIndex):
        if not self._decoder_connected:
            raise ValueError(
                "Please call ``connect_decoder()`` for this dataset to be usable"
            )

        def return_none():
            if self.allow_none_sample:
                return None
            else:
                raise RuntimeError("Got None sample")

        recording = self.dataset.get_recording(index.recording_id)
        target_slice = recording.slice(index.start, index.end)
        Y = self.dataset.get_target(target_slice)

        if not self._has_unit_encoder:
            if self.dataset.transform is not None:
                target_slice = self.dataset.transform(target_slice)
            X = {"decoder_input": self.decoder_tokenizer(target_slice)}
            if X["decoder_input"] is None:
                return return_none()

            return X, Y

        gap = self.unit_enc_window_gap

        big_index = copy.deepcopy(index)
        big_index.start = big_index.start - self.unit_enc_ctx - gap
        big_slice = self.dataset[big_index]

        # Ensure we got a slice that is long enough
        big_slice_duration = big_slice.end - big_slice.start
        expected_slice_duration = index.end - index.start + self.unit_enc_ctx + gap
        if not np.isclose(big_slice_duration, expected_slice_duration):
            return return_none()

        # Filter out inactive units
        big_slice = filter_inactive_units_(big_slice)

        X = {}
        offset = index.end - big_slice.end

        # Encoder
        unit_enc_slice = big_slice.slice(
            start=index.start - self.unit_enc_ctx - offset - gap,
            end=index.start - offset - gap,
        )
        X["encoder_input"] = self.unit_enc_tokenizer(unit_enc_slice)
        if X["encoder_input"] is None:
            return return_none()

        # Decoder
        dec_slice = big_slice.slice(index.start - offset, index.end - offset)
        X["decoder_input"] = self.decoder_tokenizer(dec_slice)
        if X["decoder_input"] is None:
            return return_none()

        return X, Y

    @staticmethod
    def collate_fn(data: list):
        from torch_brain.batching import collate as tb_collate

        safe_data = [x for x in data if x is not None]
        X, Y = list(zip(*safe_data))
        X = tb_collate(X)
        Y = tb_collate(Y)
        return X, Y


def filter_inactive_units_(data: Data) -> Data:
    from src.utils.data import apply_unit_mask_

    ts = data.spikes.timestamps  # to handle a bug in temporaldata
    # must read timestamps before unit_index to allow for a re-slice
    unit_idx = data.spikes.unit_index

    active_unit_idx = np.unique(unit_idx)
    mask = np.zeros(len(data.units), dtype=bool)
    mask[active_unit_idx] = True
    if not mask.all():
        apply_unit_mask_(data, mask)

    assert len(data.units) == len(np.unique(data.spikes.unit_index))
    return data
