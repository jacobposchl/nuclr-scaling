import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.utils.pretraining_queue import queue_pretraining


class PretrainingQueueTests(unittest.TestCase):
    def test_waits_for_predecessor_storage_and_gpu_before_one_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predecessor = root / 'previous'
            predecessor.mkdir()
            source = predecessor / 'status.json'
            source.write_text(json.dumps(dict(state='running', completed_epochs=5, target_epochs=10)))
            output = root / 'continuation'
            child = Mock(pid=456)
            def finish():
                output.mkdir()
                (output / 'status.json').write_text(json.dumps(dict(state='complete', completed_epochs=20)))
                return 0
            child.wait.side_effect = finish
            with patch('src.utils.pretraining_queue.subprocess.Popen', return_value=child) as launch, \
                 patch('src.utils.pretraining_queue.check_free_storage', side_effect=[RuntimeError('low'), 80., 80.]), \
                 patch('src.utils.pretraining_queue.require_idle_gpu', side_effect=[RuntimeError('busy'), {}]), \
                 patch('src.utils.pretraining_queue.time.sleep') as sleep, \
                 patch('builtins.print'):
                def advance(_):
                    launch.assert_not_called()
                    source.write_text(json.dumps(dict(state='complete', completed_epochs=10, target_epochs=10)))
                sleep.side_effect = advance
                queue_pretraining(predecessor, 'train_nuclr_allen_ibl_rate_separated_continue', output)
                self.assertEqual(sleep.call_count, 3)
                launch.assert_called_once()
                self.assertIn('--require-idle-gpu', launch.call_args.args[0])
                self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
            status = json.loads(output.with_suffix('.queue.json').read_text())
            self.assertEqual(status['state'], 'complete')
            self.assertEqual(status['completed_epochs'], 20)

    def test_failed_predecessor_never_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predecessor = root / 'previous'
            predecessor.mkdir()
            (predecessor / 'status.json').write_text(json.dumps(dict(state='failed')))
            with patch('src.utils.pretraining_queue.subprocess.Popen') as launch, patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'not launched'):
                    queue_pretraining(predecessor, 'train_nuclr_allen_ibl_rate_separated_continue', root / 'next')
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
