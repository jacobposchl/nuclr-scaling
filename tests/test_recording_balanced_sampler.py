from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_brain.data import Interval

from src.samplers import DistributedSamplerWrapper, RecordingBalancedWindowSampler
from src.trainers.nuclr_pretrainer import NuCLRPreTrainer


def window_keys(sampler):
    return [(index.recording_id, index.start, index.end) for index in sampler]


class RecordingBalancedSamplerTests(unittest.TestCase):
    def setUp(self):
        self.intervals = {
            "short": Interval(start=np.array([0., 100.]), end=np.array([61., 141.])),
            "long": Interval(start=np.array([0.]), end=np.array([2000.])),
            "middle": Interval(start=np.array([50.]), end=np.array([250.])),
        }

    def sampler(self, **kwargs):
        options = dict(sampling_intervals=self.intervals, window_length=20.,
                       samples_per_recording=3, seed=42)
        return RecordingBalancedWindowSampler(**(options | kwargs))

    def test_equal_quotas_unique_windows_and_allowed_bounds(self):
        sampler = self.sampler()
        selected = window_keys(sampler)
        self.assertEqual(len(selected), len(sampler))
        self.assertEqual(Counter(rid for rid, a, b in selected),
                         {"short": 3, "long": 3, "middle": 3})
        self.assertEqual(len(set(selected)), len(selected))
        for rid, start, end in selected:
            self.assertAlmostEqual(end - start, 20.)
            allowed = self.intervals[rid]
            self.assertTrue(any(a <= start and end <= b for a, b in zip(allowed.start, allowed.end)))

    def test_epoch_redraw_and_resume_reproducibility(self):
        sampler = self.sampler()
        first = window_keys(sampler)
        sampler.set_epoch(1)
        second = window_keys(sampler)
        self.assertNotEqual(first, second)
        resumed = self.sampler()
        resumed.set_epoch(1)
        self.assertEqual(second, window_keys(resumed))
        self.assertEqual(second, window_keys(sampler))

    def test_quota_cannot_silently_oversample_short_recordings(self):
        with self.assertRaisesRegex(ValueError, "short has 5 available windows"):
            self.sampler(samples_per_recording=6)
        for value in [0, -1, 1.5, True]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                self.sampler(samples_per_recording=value)

    def test_distributed_length_never_draws_samples_and_matches_shards(self):
        sampler = self.sampler()
        for drop_last in [False, True]:
            for rank in range(4):
                wrapper = DistributedSamplerWrapper(sampler, drop_last=drop_last)
                wrapper.num_replicas, wrapper.rank = 4, rank
                with patch.object(RecordingBalancedWindowSampler, "__iter__", side_effect=AssertionError):
                    count = len(wrapper)
                self.assertEqual(count, len(list(wrapper)))

    def test_trainer_keeps_partial_batch_and_records_sampling_policy(self):
        base = SimpleNamespace(
            provenance={"sampling": "uniform_available_windows_no_dataset_reweighting"},
            recording_ids=list(self.intervals),
            get_sampling_intervals=lambda: self.intervals,
            get_unit_ids=lambda: np.array(["unit"]),
        )
        trainer = SimpleNamespace(
            cfg=OmegaConf.create(dict(batch_size=4, samples_per_recording=3,
                                      dataset={}, data_root="unused", train_transform={},
                                      views={"max_distance": 120.}, seed=42, num_workers=0,
                                      pin_memory=False, persistent_workers=False)),
            world_size=1,
            model=SimpleNamespace(ctx_duration=20., tokenize=lambda data: data),
            logger=SimpleNamespace(info=lambda message: None),
        )
        with patch("src.trainers.nuclr_pretrainer.instantiate", side_effect=[None, base]):
            NuCLRPreTrainer.setup_train_loader(trainer)
        batches = list(trainer.train_loader.batch_sampler)
        self.assertEqual([len(batch) for batch in batches], [4, 4, 1])
        self.assertEqual(Counter(index.recording_id for batch in batches for index in batch),
                         {"short": 3, "long": 3, "middle": 3})
        self.assertEqual(trainer.cfg.data_provenance.samples_per_recording, 3)
        self.assertEqual(trainer.cfg.data_provenance.sampling,
                         "fixed_samples_per_recording_without_replacement")
        trainer.cfg.micro_batch_size = 2
        trainer.logger.warn = MagicMock()
        with patch("src.trainers.nuclr_pretrainer.instantiate", side_effect=[None, base]):
            NuCLRPreTrainer.setup_train_loader(trainer)
        batches = list(trainer.train_loader.batch_sampler)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 2, 2, 1])
        self.assertEqual(trainer.accumulation_steps, 2)
        self.assertEqual(Counter(index.recording_id for batch in batches for index in batch),
                         {"short": 3, "long": 3, "middle": 3})
        trainer.logger.warn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
