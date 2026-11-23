import unittest

import numpy as np

from src.artifacts.dataset_rate_probes import compare_dataset_probes, DATASETS


class DatasetProbeTests(unittest.TestCase):
    def test_same_direction_different_calibration(self):
        x = np.random.default_rng(4).normal(size=(400, 4))
        groups = np.repeat(np.arange(20), 20)
        datasets = np.repeat(DATASETS, 200)
        rates = 20 + x[:, 0]
        rates[200:] = 25 + 2 * x[200:, 0]
        result = compare_dataset_probes(x, rates, datasets, groups)
        self.assertAlmostEqual(result["similarity"]["cosine"], 1)
        for metric in result["metrics"]:
            if metric["train_dataset"] == metric["test_dataset"]:
                self.assertAlmostEqual(metric["r2"], 1)
            else:
                self.assertLess(metric["r2"], 0)
        for group in set(groups):
            self.assertEqual(len(set(result["fold_ids"][groups == group])), 1)

    def test_shared_animals_are_rejected(self):
        x = np.random.default_rng(4).normal(size=(400, 4))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            compare_dataset_probes(x, 20 + x[:, 0], np.repeat(DATASETS, 200),
                                   np.tile(np.repeat(np.arange(10), 20), 2))


if __name__ == "__main__":
    unittest.main()
