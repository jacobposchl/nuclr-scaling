import unittest

import numpy as np

from src.artifacts.rate_variant_comparison import grouped_decoding, matched_order


class RateVariantComparisonTests(unittest.TestCase):
    def test_matched_order_checks_labels_and_rates(self):
        rows = [dict(unit_uid=str(i), firing_rate_hz=float(i), dataset_id='a', subject_uid='mouse',
                     session_uid='session', brain_region='VISp') for i in range(4)]
        np.testing.assert_array_equal(matched_order(rows, rows[::-1]), [3, 2, 1, 0])
        with self.assertRaisesRegex(ValueError, 'unique neurons'):
            matched_order(rows, rows[:-1])
        with self.assertRaisesRegex(ValueError, 'different metadata'):
            matched_order(rows, [dict(row, firing_rate_hz=2.) for row in rows])

    def test_grouped_probes_recover_controlled_signals(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(180, 5))
        groups = np.repeat(np.arange(18), 10)
        datasets = np.repeat(np.arange(3), 60)
        x[:, 1] = datasets * 10 + rng.normal(size=180) * .01
        rates = 10 + x[:, 0]
        metrics, folds = grouped_decoding(x, rates, datasets, groups)
        self.assertAlmostEqual(metrics['firing_rate_r2'], 1.)
        self.assertAlmostEqual(metrics['dataset_balanced_accuracy'], 1.)
        self.assertEqual(sum(row['test_neurons'] for row in folds), len(x))


if __name__ == '__main__':
    unittest.main()
