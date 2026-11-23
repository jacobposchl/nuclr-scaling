"""Small synthetic checks for source timing recovery and add-only HDF5 writes."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from src.datasets.allen_metadata import (
    local_stimulus_digest, read_stimulus_epochs, restore_stimulus_epochs, validate_epochs,
)


class AllenMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "123.h5"
        self.source_path = Path(temporary.name) / "source.nwb"
        with h5py.File(self.path, "w") as local:
            local.create_group("session").attrs["id"] = "123"
            local["domain/start"], local["domain/end"] = [0.], [100.]
            local["spikes/timestamps"] = [1., 2., 3.]
            local["spikes/unit_index"] = [0, 1, 0]
            local["units/id"] = [10, 20]
            # The source has an extra presentation within the first block.
            local["gabors/start"] = [10., 30.]
            local["gabors/end"] = [11., 31.]
            local["gabors/timestamps"] = [10., 30.]
        with h5py.File(self.source_path, "w") as source:
            source["identifier"] = b"123"
            source["intervals/gabors_presentations/start_time"] = [10., 12., 30.]
            source["intervals/gabors_presentations/stop_time"] = [11., 13., 31.]
            source["intervals/gabors_presentations/stimulus_block"] = [0., 0., 2.]
            source["intervals/gabors_presentations/stimulus_name"] = np.array(["gabors"] * 3, dtype="S")
            source["intervals/spontaneous_presentations/start_time"] = [0., 13.]
            source["intervals/spontaneous_presentations/stop_time"] = [10., 30.]
            source["intervals/spontaneous_presentations/stimulus_name"] = np.array(["spontaneous"] * 2, dtype="S")
            source["intervals/invalid_times/start_time"] = [11.]
            source["intervals/invalid_times/stop_time"] = [12.]
            source["intervals/invalid_times/tags"] = np.array(["EcephysSession", "123", "stimulus"], dtype="S")
            source["intervals/invalid_times/tags_index"] = [3]

    def record(self):
        with h5py.File(self.source_path, "r") as source, h5py.File(self.path, "r") as local:
            epochs, checked, anomalies = read_stimulus_epochs(source, local, "123")
            return {"session_id": "123", "source_url": "https://example.test/source.nwb",
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "local_stimulus_sha256": local_stimulus_digest(local),
                    "epochs": epochs, "verified_stimulus_tables": checked,
                    "source_timing_anomalies": anomalies}

    def test_blocks_keep_separate_repeats_and_source_rows_missing_locally(self):
        record = self.record()
        self.assertEqual(record["verified_stimulus_tables"], ["gabors"])
        self.assertEqual([(e["start"], e["end"], e["type"]) for e in record["epochs"]],
                         [(0., 10., "spontaneous"), (10., 13., "gabors"),
                          (13., 30., "spontaneous"), (30., 31., "gabors")])

    def test_session_and_timestamp_mismatches_fail(self):
        with h5py.File(self.source_path, "r+") as source:
            source["identifier"][()] = b"999"
        with self.assertRaisesRegex(ValueError, "identifiers"):
            self.record()
        with h5py.File(self.source_path, "r+") as source:
            source["identifier"][()] = b"123"
        with h5py.File(self.path, "r+") as local:
            local["gabors/end"][0] = 11.5
        with self.assertRaisesRegex(ValueError, "timestamps differ"):
            self.record()

    def test_functional_connectivity_gratings_keep_the_source_label(self):
        with h5py.File(self.path, "r+") as local:
            local.move("gabors", "drifting_gratings")
        with h5py.File(self.source_path, "r+") as source:
            source.move("intervals/gabors_presentations", "intervals/drifting_gratings_75_repeats_presentations")
            group = source["intervals/drifting_gratings_75_repeats_presentations"]
            del group["stimulus_name"]
            group["stimulus_name"] = np.array(["drifting_gratings_75_repeats"] * 3, dtype="S")
        record = self.record()
        self.assertEqual(record["verified_stimulus_tables"], ["drifting_gratings"])
        self.assertEqual(record["epochs"][-1]["type"], "drifting_gratings_75_repeats")

    def test_invalid_epoch_bounds_fail(self):
        for epochs in ([], [{"start": 1., "end": 0., "type": "gabors"}],
                       [{"start": float("nan"), "end": 2., "type": "gabors"}],
                       [{"start": 0., "end": 2., "type": "gabors"},
                        {"start": 1., "end": 3., "type": "spontaneous"}]):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                validate_epochs(epochs)

    def test_source_marked_frame_anomalies_preserve_the_block_envelope(self):
        with h5py.File(self.source_path, "r+") as source:
            source["intervals/natural_movie_one_presentations/start_time"] = [40., 42., 41., 43.]
            source["intervals/natural_movie_one_presentations/stop_time"] = [42., 41., 43., 44.]
            source["intervals/natural_movie_one_presentations/stimulus_block"] = [3.] * 4
            source["intervals/natural_movie_one_presentations/stimulus_name"] = np.array(["natural_movie_one"] * 4, dtype="S")
        with self.assertRaisesRegex(ValueError, "Unexplained source timing anomaly"):
            self.record()
        with h5py.File(self.source_path, "r+") as source:
            source["intervals/invalid_times/start_time"][0] = 40.
            source["intervals/invalid_times/stop_time"][0] = 44.
        record = self.record()
        self.assertEqual(record["epochs"][-1], {"start": 40., "end": 44., "type": "natural_movie_one", "stimulus_block": 3.})
        self.assertEqual(record["source_timing_anomalies"][0]["rows"], [1, 2])

    def test_add_only_restoration_roundtrips_and_is_idempotent(self):
        from torch_brain.data import Interval

        before = {}
        with h5py.File(self.path, "r") as local:
            local.visititems(lambda name, item: before.update({name: item[:]})
                             if isinstance(item, h5py.Dataset) else None)
        record = self.record()
        self.assertTrue(restore_stimulus_epochs(self.path, record))
        self.assertFalse(restore_stimulus_epochs(self.path, record))
        with h5py.File(self.path, "r") as local:
            for name, values in before.items():
                np.testing.assert_array_equal(local[name][:], values)
            loaded = Interval.from_hdf5(local["stimulus_epochs"])
            np.testing.assert_array_equal(loaded.type, ["spontaneous", "gabors", "spontaneous", "gabors"])
            np.testing.assert_array_equal(loaded.end, [10., 13., 30., 31.])
        with h5py.File(self.path, "r+") as local:
            local["stimulus_epochs/end"][-1] = 32.
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            restore_stimulus_epochs(self.path, record)

    def test_changed_local_data_rejects_cached_source_epochs(self):
        record = self.record()
        with h5py.File(self.path, "r+") as local:
            local["gabors/start"][0] += .1
        with self.assertRaisesRegex(ValueError, "changed after source verification"):
            restore_stimulus_epochs(self.path, record)
        with h5py.File(self.path, "r") as local:
            self.assertNotIn("stimulus_epochs", local)


if __name__ == "__main__":
    unittest.main()
