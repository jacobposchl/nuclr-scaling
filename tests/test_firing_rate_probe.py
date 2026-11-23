import unittest

import numpy as np

from src.artifacts.firing_rate_probe import fit_firing_rate_probes


class FiringRateProbeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(42).uniform(-1, 1, (120, 4))
        self.rates = 20 + 3 * self.x[:, 0]
        self.animals = np.repeat(np.arange(12), 10)

    def test_known_direction_energy_and_grouped_predictions(self):
        result = fit_firing_rate_probes(self.x, self.rates, self.animals, n_splits=3)
        centered = self.x - self.x.mean(axis=0)
        energy = centered[:, 0] ** 2
        norm = (centered ** 2).sum(axis=1)
        np.testing.assert_allclose(result["model"]["coefficients"][0], [3, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(result["scores"][:, 0], centered[:, 0], atol=1e-12)
        np.testing.assert_allclose(result["fractions"][:, 0], energy / norm, atol=1e-12)
        np.testing.assert_allclose(result["held_out"][:, 0], self.rates, atol=1e-12)
        self.assertAlmostEqual(result["metrics"]["raw_hz"]["embedding_variance_fraction_along_direction"],
                               energy.sum() / norm.sum())
        for animal in np.unique(self.animals):
            self.assertEqual(len(np.unique(result["fold_ids"][self.animals == animal])), 1)

    def test_direction_fractions_are_translation_and_rotation_invariant(self):
        rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(4, 4)))
        original = fit_firing_rate_probes(self.x, self.rates, self.animals, n_splits=3)
        transformed = fit_firing_rate_probes(self.x @ rotation + 10, self.rates, self.animals, n_splits=3)
        np.testing.assert_allclose(original["fractions"], transformed["fractions"], atol=1e-11)
        np.testing.assert_allclose(original["held_out"], transformed["held_out"], atol=1e-11)

    def test_invalid_rate_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            fit_firing_rate_probes(self.x, np.zeros(120), self.animals)
        with self.assertRaisesRegex(ValueError, "must vary"):
            fit_firing_rate_probes(self.x, np.ones(120), self.animals)


if __name__ == "__main__":
    unittest.main()
