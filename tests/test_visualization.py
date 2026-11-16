"""Run only in an allocated CPU job; fixtures contain no real recordings."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import joblib
import numpy as np
from sklearn.decomposition import PCA

from src.artifacts.analyses import analyze
from src.artifacts.store import ArtifactStore, write_json


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ArtifactStore(Path(self.temp.name))
        self.x = np.random.default_rng(19).normal(size=(32, 6)).astype("float32")
        self.x[16:] += 20  # Display-only outliers must not affect fitting/scaling.
        with self.store.build("collections", {"checkpoint_id": "synthetic"}) as (stage, cid):
            shard = stage / "session-00000"
            shard.mkdir()
            rows = [dict(unit_uid=f"unit/{i}", subject_uid="allen/a" if i < 16 else "allen/b",
                         brain_region="VISp" if i % 2 else "CA1", cell_type="sst" if i % 3 else None,
                         firing_rate_hz=float(i)) for i in range(32)]
            write_json(shard / "units.json", rows)
            with h5py.File(shard / "embeddings.h5", "w") as f:
                f["unit_mean"] = self.x
            write_json(stage / "index.json", {"sessions": [{"shard": shard.name}]})
        self.cid = cid

    def run_analysis(self, **options):
        cfg = dict(kind="visualization", methods=["pca"],
                   fit_selection={"subject_uid": ["allen/a"]}, show_selection={}, standardize=True,
                   colors=[dict(field="brain_region", type="categorical", palette="tab10")])
        cfg.update(options)
        aid = analyze(self.store, [self.cid], cfg)
        path = self.store.path("analyses", aid)
        index = json.loads((path / "index.json").read_text())
        return aid, path, index

    def test_fit_scope_and_recolor_reuses_projection(self):
        aid, path, index = self.run_analysis()
        projection = self.store.path("analyses", index["projection_id"])
        saved = joblib.load(projection / "models.joblib")
        scaler = saved["scaler"]
        np.testing.assert_allclose(scaler.mean_, self.x[:16].mean(0), atol=1e-7)
        expected = PCA(n_components=2, svd_solver="full").fit(scaler.transform(self.x[:16]))
        np.testing.assert_allclose(saved["models"]["pca"].components_, expected.components_)
        with (path / "pca/coordinates.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["in_fit"] for r in rows], ["True"] * 16 + ["False"] * 16)
        actual = np.array([[r["coordinate_1"], r["coordinate_2"]] for r in rows], dtype=float)
        np.testing.assert_allclose(actual, expected.transform(scaler.transform(self.x)), atol=2e-6)
        second, recolored, other = self.run_analysis(colors=[dict(field="firing_rate_hz", type="continuous", palette="magma")])
        self.assertNotEqual(aid, second)
        self.assertEqual(index["projection_id"], other["projection_id"])
        self.assertEqual((path / "pca/coordinates.csv").read_bytes(), (recolored / "pca/coordinates.csv").read_bytes())
        self.assertEqual(second, self.run_analysis(colors=[dict(field="firing_rate_hz", type="continuous", palette="magma")])[0])
        _, _, other = self.run_analysis(show_selection={"subject_uid": ["allen/b"]})
        self.assertEqual(index["projection_id"], other["projection_id"])
        _, _, other = self.run_analysis(fit_selection={"subject_uid": ["allen/b"]})
        self.assertNotEqual(index["projection_id"], other["projection_id"])

    def test_grouped_umap_transform_and_unknown_colors(self):
        _, path, index = self.run_analysis(methods=["pca", "umap"],
                    umap=dict(n_neighbors=4, n_epochs=30), colors=[
                        dict(field="cell_type", type="categorical"),
                        dict(field="firing_rate_hz", type="continuous")])
        self.assertEqual(set(index["methods"]), {"pca", "umap"})
        with (path / "umap/coordinates.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        xy = np.array([[r["coordinate_1"], r["coordinate_2"]] for r in rows], dtype=float)
        self.assertTrue(np.isfinite(xy).all())
        projection = self.store.path("analyses", index["projection_id"])
        with np.load(projection / "fit_coordinates.npz") as fit:
            np.testing.assert_array_equal(xy[:16], fit["umap"])
        for method in index["methods"].values():
            self.assertEqual(method["plots"][0]["unknown_rows"], 11)
            self.assertEqual(method["plots"][1]["min"], 0)
            self.assertEqual(method["plots"][1]["max"], 31)
            self.assertTrue(all((path / plot["file"]).is_file() for plot in method["plots"]))

    def test_cohort_limit_and_invalid_metadata(self):
        _, path, _ = self.run_analysis(show_selection=None, max_rows=16)
        self.assertEqual(json.loads((path / "metrics.json").read_text())["shown_rows"], 16)
        with self.assertRaisesRegex(ValueError, "max_rows"):
            self.run_analysis(max_rows=16)
        with self.assertRaisesRegex(ValueError, "Unknown selection"):
            self.run_analysis(fit_selection={"subject_uid": ["typo"]})
        with self.assertRaisesRegex(ValueError, "Missing metadata"):
            self.run_analysis(colors=[dict(field="not_a_label")])
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.run_analysis(colors=[dict(field="brain_region", type="continuous")])
        with self.assertRaisesRegex(ValueError, "n_neighbors"):
            self.run_analysis(methods=["umap"], umap=dict(n_neighbors=16))


if __name__ == "__main__":
    unittest.main()
