import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from src.artifacts.store import ArtifactStore, write_json

from src.artifacts.direction_removed_probes import remaining_coordinates
from src.artifacts.sequential_rate_directions import analyze_sequential_directions, fit_sequential_directions


class SequentialRateDirectionTests(unittest.TestCase):
    def test_reduced_problem_matches_direct_refits_and_energy_accounting(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=(500, 8)) @ rng.normal(size=(8, 8))
        rates = 20 + x[:, 0] + .3 * rng.normal(size=500)
        result = fit_sequential_directions(x[:400], rates[:400], max_directions=5)
        remaining = x.copy()
        for step in range(5):
            model = LinearRegression().fit(remaining[:400], rates[:400])
            actual = ((x[400:] - result["mean"]) @ result["directions"][step]
                      * result["slopes"][step] + result["rate_mean"])
            np.testing.assert_allclose(actual, model.predict(remaining[400:]), atol=1e-9)
            remaining = remaining_coordinates(remaining, remaining[:400].mean(axis=0), model.coef_)
        centered = x - result["mean"]
        residual = centered - (centered @ result["directions"].T) @ result["directions"]
        np.testing.assert_allclose(np.sum(residual**2, axis=1), np.sum(remaining**2, axis=1), atol=1e-9)

    def test_direct_collection_uses_training_only_animal_folds(self):
        rng = np.random.default_rng(82)
        x = rng.normal(size=(60, 5)) @ rng.normal(size=(5, 5))
        rates = 100 + x[:, 0] + .3 * rng.normal(size=60)
        groups = np.array([f"animal_{i // 10}" for i in range(60)])
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(temporary)
            with store.build("collections", {"checkpoint_id": "synthetic", "sources": []}) as (stage, cid):
                shard = stage / "session-00000"
                shard.mkdir()
                write_json(shard / "units.json", [dict(unit_uid=str(i), subject_uid=groups[i], firing_rate_hz=rates[i])
                                                  for i in range(60)])
                with h5py.File(shard / "embeddings.h5", "w") as out:
                    out["unit_mean"] = x
                write_json(stage / "index.json", {"sessions": [{"shard": shard.name, "units": 60}], "dimension": 5})
            aid = analyze_sequential_directions(store, collection_id=cid)
            folder = store.path("analyses", aid)
            index = json.loads((folder / "index.json").read_text())
            self.assertEqual(index["animals"], 6)
            with np.load(folder / index["weights"]) as weights:
                for fold, (train, test) in enumerate(GroupKFold(5).split(x, groups=groups)):
                    self.assertFalse(set(groups[train]) & set(groups[test]))
                    np.testing.assert_allclose(weights[f"fold_{fold}_mean"], x[train].mean(0))
                    direction = weights[f"fold_{fold}_directions"][0]
                    predicted = ((x[test] - weights[f"fold_{fold}_mean"]) @ direction *
                                 weights[f"fold_{fold}_slopes"][0] + weights[f"fold_{fold}_rate_mean"])
                    np.testing.assert_allclose(predicted, LinearRegression().fit(x[train], rates[train]).predict(x[test]))
            with (folder / index["metrics"]).open() as stream:
                records = list(csv.DictReader(stream))
            retained = [float(row["embedding_variance_retained"]) for row in records]
            self.assertTrue(np.all(np.diff(retained) <= 1e-12))
            self.assertTrue((folder / index["plots"][0]).exists())

    def test_stops_when_remaining_training_signal_vanishes(self):
        x, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(100, 4)))
        x -= x.mean(axis=0)
        x, _ = np.linalg.qr(x)
        result = fit_sequential_directions(x, 10 + x[:, 0])
        self.assertEqual(len(result["directions"]), 1)


if __name__ == "__main__":
    unittest.main()
