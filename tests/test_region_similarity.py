import unittest

import numpy as np

from src.artifacts.region_similarity import regional_means


class RegionalSimilarityTests(unittest.TestCase):
    def test_layers_merge_with_neuron_weighting_but_datasets_stay_separate(self):
        rows = [dict(dataset_id=dataset, brain_region=area, subject_uid="mouse", session_uid="session")
                for dataset, area in [("ibl_brain_wide_bench_2026", "VISp2/3"),
                                      ("ibl_brain_wide_bench_2026", "VISp5"),
                                      ("ibl_brain_wide_bench_2026", "VISp5"),
                                      ("allen_visual_coding", "VISp"),
                                      ("allen_visual_coding", "VISrll")]]
        x = np.array([[9., 0], [0., 3], [0., 3], [0., 2], [-2., 0]])
        means, groups, similarity = regional_means(x, rows)
        keys = [(row["dataset_id"], row["brain_region"]) for row in groups]
        ibl = keys.index(("ibl_brain_wide_bench_2026", "VISp"))
        vc = keys.index(("allen_visual_coding", "VISp"))
        np.testing.assert_allclose(means[ibl], [3., 2.])
        self.assertEqual(groups[ibl]["neurons"], 3)
        self.assertEqual(sum(row["neurons"] for row in groups), 5)
        self.assertIn(("allen_visual_coding", "VISrll"), keys)
        self.assertAlmostEqual(similarity[ibl, vc], 2 / np.sqrt(13))
        np.testing.assert_allclose(similarity, similarity.T)
        np.testing.assert_allclose(np.diag(similarity), 1.)

    def test_visceral_cortex_and_zero_mean_are_rejected(self):
        row = dict(dataset_id="ibl_brain_wide_bench_2026", brain_region="VISC5",
                   subject_uid="mouse", session_uid="session")
        with self.assertRaisesRegex(ValueError, "visceral"):
            regional_means([[1., 0]], [row])
        row["brain_region"] = "VISp6b"
        with self.assertRaisesRegex(ValueError, "zero norm"):
            regional_means([[1., 0], [-1., 0]], [row, row])


if __name__ == "__main__":
    unittest.main()
