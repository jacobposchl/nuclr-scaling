import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.artifacts.brain_region_decoding import (
    animal_folds, beryl_gray_mapping, paired_animal_bootstrap, region_cohorts, region_probe,
)


class BrainRegionDecodingTests(unittest.TestCase):
    def test_mapping_uses_nearest_selected_gray_ancestor(self):
        rows = [(0, 'void', ''), (997, 'root', '/997/'), (8, 'grey', '/997/8/'),
                (10, 'VISp', '/997/8/10/'), (11, 'VISp6a', '/997/8/10/11/'),
                (12, 'tract', '/997/12/')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (path / 'allen_structure_tree.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'acronym', 'structure_id_path'])
                writer.writerows(rows)
            np.save(path / 'beryl.npy', np.array([[997], [10], [12]]))
            mapping = beryl_gray_mapping(path)
        self.assertEqual(mapping['VISp6a'], 'VISp')
        self.assertEqual(mapping['VISp'], 'VISp')
        self.assertTrue(all(mapping[name] is None for name in ('root', 'void', 'grey', 'tract')))

    def test_region_support_and_positive_rate_filter(self):
        rows = [dict(dataset_id='d', subject_uid=f'a{i % 3}', brain_region=region, firing_rate_hz=1.)
                for region in ('A', 'B', 'rare') for i in range(3 if region != 'rare' else 1)]
        rows.append(dict(rows[0], firing_rate_hz=0.))
        cohorts, coverage = region_cohorts(rows, {'A': 'A', 'B': 'B', 'rare': 'rare'}, 2, 2)
        np.testing.assert_array_equal(cohorts['d'], np.arange(6))
        self.assertFalse(next(row['included'] for row in coverage if row['region'] == 'rare'))
        with self.assertRaisesRegex(ValueError, 'Unknown atlas'):
            region_cohorts(rows, {'A': 'A'})

    def test_grouped_probe_and_paired_bootstrap(self):
        rng = np.random.default_rng(9)
        groups = np.repeat(np.arange(15), 12)
        y = np.tile(np.repeat(np.arange(3), 4), 15)
        x = np.eye(3)[y] * 10 + rng.normal(size=(len(y), 3)) * .1
        folds = animal_folds(y, groups)
        self.assertTrue(all(not set(groups[a]) & set(groups[b]) for a, b in folds))
        predicted, assignments, records = region_probe(x, y, folds)
        np.testing.assert_array_equal(predicted, y)
        self.assertEqual(set(assignments), set(range(5)))
        self.assertEqual(sum(row['test_neurons'] for row in records), len(y))
        np.testing.assert_array_equal(paired_animal_bootstrap(y, groups, predicted, predicted, repeats=100), [0, 0])
        np.testing.assert_allclose(paired_animal_bootstrap(y, groups, np.zeros_like(y), predicted, repeats=100), [2/3, 2/3])


if __name__ == '__main__':
    unittest.main()
