import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from src.artifacts.direction_removed_probes import remaining_coordinates
from src.artifacts.sequential_rate_directions import fit_sequential_directions


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

    def test_stops_when_remaining_training_signal_vanishes(self):
        x, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(100, 4)))
        x -= x.mean(axis=0)
        x, _ = np.linalg.qr(x)
        result = fit_sequential_directions(x, 10 + x[:, 0])
        self.assertEqual(len(result["directions"]), 1)


if __name__ == "__main__":
    unittest.main()
