"""Metadata-only checks for excluding entire optotagging blocks."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from src.datasets.allen_sessions import AllenRegistry, windows


class OptotaggingIntervalsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.path = root / "1.h5"
        with h5py.File(self.path, "w") as f:
            f.create_group("subject").attrs["id"] = "mouse"
            f["units/id"] = [1]
            f["units/probe_id"] = [1]
            f["units/area"] = np.array(["VISp"], dtype="S")
            f["spikes/timestamps"] = []
            f["spikes/domain/start"], f["spikes/domain/end"] = [0.], [120.]
            f["domain/start"], f["domain/end"] = [0., 80.], [60., 120.]
            f["intervals/blocks/optotagging/start"] = [20., 90.]
            f["intervals/blocks/optotagging/end"] = [40., 100.]
        config = {"schema_version": 1, "root": str(root), "datasets": {
            "test": {"directory": ".", "domain": "domain",
                     "region_key": "area", "probe_key": "probe_id",
                     "optotagging": {"interval_path": "intervals/blocks/optotagging"}}}}
        registry_path = root / "registry.json"
        registry_path.write_text(json.dumps(config))
        self.registry = AllenRegistry(registry_path)
        self.spec = self.registry.spec("test")

    def test_excludes_whole_blocks_and_preserves_recording_gaps(self):
        desc = self.registry.describe("test", "1")
        self.assertEqual(desc["intervals"], [(0., 20.), (40., 60.), (80., 90.), (100., 120.)])
        self.assertEqual(list(windows(desc["intervals"], 20., 20.)),
                         [(0., 20.), (40., 60.), (100., 120.)])
        for start, end in [(10., 30.), (20., 40.), (50., 90.), (85., 105.)]:
            with self.assertRaisesRegex(ValueError, "gap"):
                self.registry.window("test", "1", start, end, desc)

    def test_session_metadata_excludes_overlapping_and_outside_blocks(self):
        self.spec["optotagging"] = {"intervals_by_session": {
            "1": [[-10., 10.], [20., 40.], [30., 50.], [100., 140.]]}}
        self.assertEqual(self.registry.describe("test", "1")["intervals"],
                         [(10., 20.), (50., 60.), (80., 100.)])

    def test_missing_metadata_fails_closed_but_verified_empty_is_allowed(self):
        self.spec["optotagging"] = {"interval_path": "missing"}
        with self.assertRaisesRegex(ValueError, "Missing optotagging"):
            self.registry.describe("test", "1")
        self.spec["optotagging"] = {"intervals_by_session": {}}
        with self.assertRaisesRegex(ValueError, "Missing optotagging"):
            self.registry.describe("test", "1")
        self.spec["optotagging"]["intervals_by_session"]["1"] = []
        self.assertEqual(self.registry.describe("test", "1")["intervals"],
                         [(0., 60.), (80., 120.)])

    def test_invalid_or_ambiguous_metadata_is_rejected(self):
        for intervals in ([[20., 10.]], [[float("nan"), 20.]], [[10., 20., 30.]]):
            with self.subTest(intervals=intervals):
                self.spec["optotagging"] = {"intervals_by_session": {"1": intervals}}
                with self.assertRaisesRegex(ValueError, "finite, increasing"):
                    self.registry.describe("test", "1")
        self.spec["optotagging"]["interval_path"] = "intervals/blocks/optotagging"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.registry.describe("test", "1")

    def test_explicit_missing_timing_fallback_discards_only_the_recording_tail(self):
        with h5py.File(self.path, "r+") as f:
            f["stimulus_epochs/end"] = [30., 105.]
        self.spec["optotagging"] = {"intervals_by_session": {},
                                   "end_at_last_stimulus_sessions": ["1"]}
        desc = self.registry.describe("test", "1")
        self.assertEqual(desc["intervals"], [(0., 60.), (80., 105.)])
        with self.assertRaisesRegex(ValueError, "gap"):
            self.registry.window("test", "1", 90., 110., desc)
        self.spec["optotagging"]["intervals_by_session"]["1"] = []
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.registry.describe("test", "1")
        del self.spec["optotagging"]["intervals_by_session"]["1"]
        with h5py.File(self.path, "r+") as f:
            f["stimulus_epochs/end"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite stimulus end"):
            self.registry.describe("test", "1")

    def test_both_contrastive_views_stay_outside_tagging(self):
        from torch_brain.data import Interval
        from torch_brain.datasets import DatasetIndex
        from src.datasets.constrastive_wrapper import ContrastiveDatasetWrapper

        selected = self.registry.describe("test", "1")["intervals"]
        intervals = {"test/1": Interval(start=np.array([a for a, b in selected]),
                                        end=np.array([b for a, b in selected]))}
        dataset = SimpleNamespace(get_sampling_intervals=lambda: intervals)
        wrapper = ContrastiveDatasetWrapper(dataset, tokenizer=None)
        wrapper.enable_two_views(max_distance=120., seed=42)
        for start, end in windows(selected, 20., 20.):
            first = DatasetIndex(recording_id="test/1", start=start, end=end)
            for _ in range(30):
                second = wrapper._get_second_view(first)
                self.assertTrue(any(a <= second.start and second.end <= b for a, b in selected))


if __name__ == "__main__":
    unittest.main()
