import unittest

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

from src.artifacts.ward_clustering import dataset_composition


class WardDatasetTests(unittest.TestCase):
    def test_separated_datasets_and_leaf_order_match_cluster_numbers(self):
        x = np.array([[20., 0], [0., 0], [40., 0], [20.1, 0], [0.2, 0], [40.3, 0]])
        datasets = np.array(["allen_visual_coding", "allen_visual_behavior", "ibl_brain_wide_bench_2026"] * 2)
        tree = linkage(x, method="ward")
        order = dendrogram(tree, no_plot=True, count_sort="descending")["leaves"]
        labels, counts, metrics = dataset_composition(tree, datasets, order)
        self.assertEqual(list(dict.fromkeys(labels[order])), [1, 2, 3])
        self.assertAlmostEqual(metrics["adjusted_rand_index"], 1.)
        self.assertAlmostEqual(metrics["majority_dataset_fraction"], 1.)
        self.assertAlmostEqual(metrics["largest_dataset_baseline"], 1 / 3)
        self.assertEqual(sum(row["neurons"] for row in counts), len(x))
        for cluster in range(1, 4):
            self.assertAlmostEqual(sum(row["fraction_of_cluster"] for row in counts if row["cluster"] == cluster), 1.)

    def test_mixed_clusters_do_not_appear_dataset_pure(self):
        x = np.array([[0., 0], [.1, 0], [20., 0], [20.2, 0], [40., 0], [40.3, 0]])
        datasets = ["allen_visual_behavior", "allen_visual_coding"] * 3
        tree = linkage(x, method="ward")
        order = dendrogram(tree, no_plot=True)["leaves"]
        _, counts, metrics = dataset_composition(tree, datasets, order)
        self.assertAlmostEqual(metrics["majority_dataset_fraction"], .5)
        self.assertLess(metrics["adjusted_rand_index"], 0.)
        self.assertTrue(all(row["neurons"] == 0 and row["fraction_of_dataset"] == 0
                            for row in counts if row["dataset_id"] == "ibl_brain_wide_bench_2026"))


if __name__ == "__main__":
    unittest.main()
