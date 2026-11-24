import unittest
import numpy as np
from src.artifacts.biological_neighbors import (
    SCORES, balanced_queries, candidate_pool, neighbor_agreement, summarize,
)


class BiologicalNeighborsTests(unittest.TestCase):
    def test_matching_excludes_animal_and_enforces_both_rates(self):
        subjects = np.array(['a', 'a', 'b', 'c', 'd', 'e'])
        rates = np.array([10., 10., 10.5, 11.1, 10.1, 9.])
        population = np.array([5., 5., 5.5, 5., 7., 5.])
        targets = np.arange(1, 6)
        np.testing.assert_array_equal(candidate_pool(0, targets, subjects, rates, population, 'Unmatched'), [2, 3, 4, 5])
        np.testing.assert_array_equal(candidate_pool(0, targets, subjects, rates, population, 'Matched FR'), [2, 4])
        np.testing.assert_array_equal(candidate_pool(0, targets, subjects, rates, population, 'Matched FR + population FR'), [2])

    def test_neighbors_and_scale_invariance(self):
        x = np.array([[1., 0.], [1., .01], [0., 1.], [-1., 0.]])
        y = np.array(['A', 'A', 'B', 'B'])
        expected = [1., 1.]
        self.assertEqual(neighbor_agreement(x, 0, np.arange(1, 4), y, 1), expected)
        self.assertEqual(neighbor_agreement(10 * x, 0, np.arange(1, 4), y, 1), expected)

    def test_sampling_and_macro_average_do_not_favor_large_animals(self):
        subjects = np.array(['a'] * 20 + ['b'] * 3)
        q = balanced_queries(np.arange(23), subjects, np.random.default_rng(42), maximum=6)
        self.assertEqual(sum(subjects[q] == 'a'), 3)
        self.assertEqual(len(set(q)), 6)
        records = []
        for region in ['X', 'Y']:
            for animal, count, value in [('a', 20, 0.), ('b', 2, 1.)]:
                for _ in range(count):
                    row = dict(label=region, subject_uid=animal, **{s: value for s in SCORES})
                    row['variant_euclidean'] = row['baseline_euclidean'] + .1
                    records.append(row)
        result, _ = summarize(records, repeats=100)
        self.assertAlmostEqual(result['baseline_euclidean'], .5)
        self.assertAlmostEqual(result['delta_euclidean'], .1)
        self.assertAlmostEqual(result['delta_euclidean_low'], .1)
        self.assertAlmostEqual(result['delta_euclidean_high'], .1)


if __name__ == '__main__':
    unittest.main()
