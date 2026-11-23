"""Equal recording quotas drawn from the existing jittered window sampler."""

import math

import torch
from torch_brain.samplers import RandomFixedWindowSampler


class RecordingBalancedWindowSampler(torch.utils.data.Sampler):
    def __init__(self, *, sampling_intervals, window_length, samples_per_recording, seed):
        if (not isinstance(samples_per_recording, int)
                or isinstance(samples_per_recording, bool) or samples_per_recording <= 0):
            raise ValueError("samples_per_recording must be a positive integer")
        if not math.isfinite(window_length) or window_length <= 0:
            raise ValueError("window_length must be positive and finite")
        if not sampling_intervals:
            raise ValueError("Recording-balanced sampling requires at least one recording")
        self.sampling_intervals = dict(sorted(sampling_intervals.items()))
        self.window_length = window_length
        self.samples_per_recording = samples_per_recording
        self.seed = seed
        self.epoch = 0
        for recording, intervals in self.sampling_intervals.items():
            capacity = sum(math.floor((end - start) / window_length)
                           for start, end in zip(intervals.start, intervals.end, strict=True))
            if capacity < samples_per_recording:
                raise ValueError(
                    f"Recording {recording} has {capacity} available windows, "
                    f"fewer than the requested quota of {samples_per_recording}"
                )

    def __len__(self):
        return len(self.sampling_intervals) * self.samples_per_recording

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # Epoch-derived randomness makes resumed and uninterrupted selection agree.
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        candidates = {recording: [] for recording in self.sampling_intervals}
        for index in RandomFixedWindowSampler(
            sampling_intervals=self.sampling_intervals,
            window_length=self.window_length,
            generator=generator,
        ):
            candidates[index.recording_id].append(index)
        selected = []
        for recording, indices in candidates.items():
            if len(indices) < self.samples_per_recording:
                raise RuntimeError(f"Window sampler produced too few candidates for {recording}")
            positions = torch.randperm(len(indices), generator=generator)[:self.samples_per_recording]
            selected.extend(indices[position] for position in positions.tolist())
        order = torch.randperm(len(selected), generator=generator).tolist()
        return iter(selected[position] for position in order)
