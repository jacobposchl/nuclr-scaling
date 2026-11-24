from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.utils.run_resources import check_free_storage, require_idle_gpu


class ResourceChecksTests(unittest.TestCase):
    def test_busy_gpu_is_rejected_before_utilization_query(self):
        with patch("src.utils.run_resources.subprocess.check_output", return_value="123, python\n") as query:
            with self.assertRaisesRegex(RuntimeError, "compute processes"):
                require_idle_gpu("0")
            self.assertEqual(query.call_count, 1)

    def test_desktop_only_allowed_but_graphics_workload_rejected(self):
        with patch("src.utils.run_resources.subprocess.check_output", side_effect=["", "0, 153\n"]):
            self.assertEqual(require_idle_gpu("0")["memory_used_mib"], 153.)
        for values in ("50, 153\n", "0, 2048\n", "nan, 153\n"):
            with patch("src.utils.run_resources.subprocess.check_output", side_effect=["", values]):
                with self.assertRaises(RuntimeError):
                    require_idle_gpu("0")
        with self.assertRaises(ValueError):
            require_idle_gpu("0,1")

    def test_free_space_gate_works_for_new_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new/run"
            with patch("src.utils.run_resources.shutil.disk_usage", return_value=SimpleNamespace(free=20 * 2 ** 30)):
                with self.assertRaisesRegex(RuntimeError, "Insufficient storage"):
                    check_free_storage(output, 25.)
                self.assertEqual(check_free_storage(output, 15.), 20.)
            self.assertFalse(output.exists())

    def test_runner_refuses_busy_gpu_before_cuda_or_output_creation(self):
        import scripts.run_allen_pretraining as runner
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch("sys.argv", ["run", "--local", "--require-idle-gpu", "--output", str(output)]), \
                    patch.object(runner, "require_idle_gpu", side_effect=RuntimeError("GPU busy")), \
                    patch.object(runner.torch.cuda, "set_device") as initialize:
                with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                    runner.main()
            initialize.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
