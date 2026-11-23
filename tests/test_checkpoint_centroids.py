import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from src.artifacts.checkpoint_centroids import analyze_checkpoint_centroids, centroid_distance
from src.artifacts.store import ArtifactStore, write_json


class CheckpointCentroidTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0., 0.], [2., 0.], [4., 4.], [4., 2.]])
        self.datasets = ["allen_visual_behavior"] * 2 + ["allen_visual_coding"] * 2

    def test_distance_between_means_is_not_mean_distance_to_a_centroid(self):
        result = centroid_distance(self.x, self.datasets)
        self.assertAlmostEqual(result["centroid_distance"], np.sqrt(18))
        self.assertEqual(result["visual_behavior_neurons"], 2)
        rotation = np.array([[0., -1.], [1., 0.]])
        transformed = centroid_distance(self.x @ rotation + 100, self.datasets)
        self.assertAlmostEqual(result["centroid_distance"], transformed["centroid_distance"])
        repeated = centroid_distance(np.repeat(self.x, 3, axis=0), np.repeat(self.datasets, 3))
        self.assertAlmostEqual(result["centroid_distance"], repeated["centroid_distance"])

    def test_missing_dataset_or_nonfinite_embedding_is_rejected(self):
        for x, groups in [(self.x, self.datasets[:2]), (self.x[:2], self.datasets[:2]),
                          (self.x * np.nan, self.datasets)]:
            with self.assertRaisesRegex(ValueError, "finite embeddings"):
                centroid_distance(x, groups)

    def collection(self, store, epoch, scale=1., window_start=0.):
        recipe = {"operation": "extract_v1", "checkpoint_id": str(epoch), "sources": [], "selection": [],
                  "window": {"duration_seconds": 20}, "population": "all_source_units_grouped_by_probe_no_dropout",
                  "representation": "encoder", "normalization": "none", "device": "cpu", "dtype": "float32"}
        # Window content is also part of recipe identity in this synthetic fixture.
        recipe["fixture_window_start"] = window_start
        with store.build("collections", recipe) as (stage, cid):
            entries = []
            for i, dataset in enumerate(sorted(set(self.datasets))):
                shard = stage / f"session-{i}"
                shard.mkdir()
                mask = np.array(self.datasets) == dataset
                rows = [{"unit_uid": f"{dataset}/{j}", "dataset_id": dataset}
                        for j in range(int(mask.sum()))]
                write_json(shard / "units.json", rows)
                write_json(shard / "windows.json", [{"start": window_start, "end": window_start + 20}])
                with h5py.File(shard / "embeddings.h5", "w") as f:
                    f["unit_mean"] = self.x[mask] * scale
                entries.append({"dataset_id": dataset, "session_id": "1", "shard": shard.name})
            write_json(stage / "index.json", {"sessions": entries, "dimension": 2})
        return cid

    def test_fixed_cohort_epoch_plot_and_window_mismatch_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(Path(temporary))
            first = self.collection(store, 1)
            second = self.collection(store, 2, scale=2)
            analysis = analyze_checkpoint_centroids(store, {2: second, 1: first})
            folder = store.path("analyses", analysis)
            with (folder / "centroid_distances.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["epoch"] for row in rows], ["1", "2"])
            self.assertAlmostEqual(float(rows[1]["centroid_distance"]), 2 * np.sqrt(18))
            self.assertTrue((folder / "dataset_centroid_distance.png").is_file())
            mismatch = self.collection(store, 3, window_start=1)
            with self.assertRaisesRegex(ValueError, "different neurons, windows"):
                analyze_checkpoint_centroids(store, {1: first, 3: mismatch})


if __name__ == "__main__":
    unittest.main()
