import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from src.artifacts.vc_region_decoding import verify_reference_cohort, verify_pooling


class VCRegionDecodingTests(unittest.TestCase):
    def test_reference_alignment_and_rejection_of_fold_or_label_changes(self):
        rows = [dict(unit_uid=f'u{i}', subject_uid=f'a{i}') for i in range(3)]
        labels = np.array(['A', 'B', 'A'])
        folds = [(np.array([j for j in range(3) if i != j]), np.array([i])) for i in range(3)]
        saved = [dict(unit_uid=f'u{i}', animal=f'a{i}', region=labels[i], fold=str(i)) for i in reversed(range(3))]
        ordered = verify_reference_cohort(rows, np.arange(3), labels, folds, saved)
        self.assertEqual([r['unit_uid'] for r in ordered], ['u0', 'u1', 'u2'])
        with self.assertRaisesRegex(ValueError, 'cohorts differ'):
            verify_reference_cohort(rows, np.arange(3), labels, folds, saved[:-1])
        saved[0]['fold'] = '0'
        with self.assertRaisesRegex(ValueError, 'folds differ'):
            verify_reference_cohort(rows, np.arange(3), labels, folds, saved)

    def test_pooling_compares_exact_window_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['left', 'right']:
                p = root / name
                (p / 'session').mkdir(parents=True)
                (p / 'index.json').write_text(json.dumps({'sessions': [dict(dataset_id='allen_visual_coding',
                    session_id='1', units=4, windows=1, shard='session')]}))
                (p / 'session/windows.json').write_text('[{"start":0,"end":20}]')
            store = SimpleNamespace(path=lambda kind, aid: root / aid)
            verify_pooling(store, 'left', 'right')
            (root / 'right/session/windows.json').write_text('[{"start":20,"end":40}]')
            with self.assertRaisesRegex(ValueError, 'pooling windows'):
                verify_pooling(store, 'left', 'right')


if __name__ == '__main__':
    unittest.main()
