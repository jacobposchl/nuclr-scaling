import unittest

import numpy as np

from src.artifacts.rate_residualization import residualize_rates


class RateResidualizationTests(unittest.TestCase):
    def test_removes_rate_component_and_preserves_orthogonal_signal(self):
        rates = np.array([[1., 2.], [2., 1.], [3., 4.], [4., 3.], [5., 7.], [7., 6.]])
        weights = np.array([1., 1., .5, .5, .5, .5])
        design = np.column_stack([np.ones(len(rates)), rates])
        signal = np.array([[2., 1.], [3., 5.], [7., 2.], [4., 8.], [6., 3.], [9., 4.]])
        root_weight = np.sqrt(weights)[:, None]
        signal -= design @ np.linalg.lstsq(design * root_weight, signal * root_weight, rcond=None)[0]
        embeddings = signal + design @ np.array([[10., 20.], [3., -2.], [-4., 7.]])
        residuals, _, metrics = residualize_rates(embeddings, rates, weights)
        np.testing.assert_allclose(residuals, signal, atol=1e-6)
        self.assertLess(metrics["maximum_weighted_design_residual_product"], 1e-10)

    def test_duplicating_recording_neurons_does_not_change_weighted_fit(self):
        rates = np.array([[1.], [2.], [4.], [6.]])
        embeddings = np.array([[2., 9.], [5., 3.], [6., 4.], [11., 7.]])
        _, model, _ = residualize_rates(embeddings, rates, np.full(4, .5))
        duplicated = np.array([0, 0, 1, 1, 2, 3])
        _, repeated_model, _ = residualize_rates(
            embeddings[duplicated], rates[duplicated], np.array([.25, .25, .25, .25, .5, .5])
        )
        for key in model:
            np.testing.assert_allclose(model[key], repeated_model[key], atol=1e-12)

    def test_rejects_dependent_predictors(self):
        rates = np.column_stack([np.arange(6.), 2 * np.arange(6.)])
        with self.assertRaisesRegex(ValueError, "linearly dependent"):
            residualize_rates(np.ones((6, 3)), rates, np.ones(6))


if __name__ == "__main__":
    unittest.main()
