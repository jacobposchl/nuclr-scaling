import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from src.artifacts.direction_removed_probes import remaining_coordinates, fit_two_direction_removal


class DirectionRemovedProbeTests(unittest.TestCase):
    def test_two_removals_remove_both_correlated_rate_coordinates(self):
        rng = np.random.default_rng(24)
        x = rng.normal(size=(1000, 5))
        # Whiten the sample so unrelated dimensions have exactly zero covariance.
        x -= x.mean(axis=0)
        x, _ = np.linalg.qr(x)
        rates = 20 + x[:, 0]
        x[:, 1] += x[:, 0]
        mean, basis, directions = fit_two_direction_removal(x, rates)
        remaining = (x - mean) @ basis
        self.assertEqual(remaining.shape, (1000, 3))
        np.testing.assert_allclose(remaining.T @ (rates - rates.mean()), 0, atol=1e-11)
        np.testing.assert_allclose(remaining @ basis.T,
                                   (x - mean) - ((x - mean) @ directions.T) @ directions, atol=1e-12)
        changed = x + rng.normal(size=(1000, 2)) @ directions
        np.testing.assert_allclose((changed - mean) @ basis, remaining, atol=1e-12)

    def test_removed_coordinate_cannot_be_recovered_from_roundoff(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(500, 5))
        direction = rng.normal(size=5)
        direction /= np.linalg.norm(direction)
        remaining = remaining_coordinates(x, x.mean(axis=0), direction)
        self.assertEqual(remaining.shape, (500, 4))
        # Arbitrarily changing only the removed coordinate leaves features unchanged.
        changed = x + rng.normal(size=(500, 1)) * direction
        np.testing.assert_allclose(remaining_coordinates(changed, x.mean(axis=0), direction),
                                   remaining, atol=1e-12)
        centered = x - x.mean(axis=0)
        np.testing.assert_allclose(np.sum(remaining**2, axis=1),
                                   np.sum(centered**2, axis=1) - (centered @ direction)**2,
                                   atol=1e-11)

    def test_correlated_remaining_direction_retains_linear_information(self):
        rng = np.random.default_rng(7)
        rate = rng.normal(size=1000)
        x = np.column_stack([rate, rate + .1 * rng.normal(size=1000),
                             rng.normal(size=1000)])
        original = LinearRegression().fit(x[:800], rate[:800])
        remaining = remaining_coordinates(x, x[:800].mean(axis=0), original.coef_)
        new_probe = LinearRegression().fit(remaining[:800], rate[:800])
        self.assertGreater(new_probe.score(remaining[800:], rate[800:]), .98)


if __name__ == "__main__":
    unittest.main()
