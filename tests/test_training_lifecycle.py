"""Run in an allocated job: checkpoint cadence, atomic saves, and all-data mode."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from src.artifacts.store import write_json
from src.datasets.allen_multi import AllenMultiDataset
from src.trainers.base import Trainer
from src.utils.checkpoint import save_ckpt
from src.utils.logger import Logger
from test_artifacts import fixture


class TrainingLifecycleTests(unittest.TestCase):
    def test_logging_preserves_shell_expressions_in_git_diff(self):
        cfg = OmegaConf.create({"seed": 42})
        diff = '+python_bin=${NUCLR_PYTHON:-$project_root/.venv/bin/python}\n'
        with patch("src.utils.logger.git.Repo") as repository, patch("src.utils.logger.wandb.config") as logged:
            repository.return_value.head.commit.hexsha = "example-commit"
            repository.return_value.git.diff.return_value = diff
            Logger.save_config(SimpleNamespace(dummy=False), cfg)
        logged.update.assert_called_once_with(
            {"seed": 42, "git": {"commit_id": "example-commit", "diff": diff}},
            allow_val_change=True,
        )

    def test_logging_without_git_metadata(self):
        cfg = OmegaConf.create({"seed": 42})
        with patch("src.utils.logger.git.Repo", side_effect=RuntimeError("no repository")), patch("src.utils.logger.wandb.config") as logged:
            Logger.save_config(SimpleNamespace(dummy=False), cfg)
        logged.update.assert_called_once_with({"seed": 42}, allow_val_change=True)

    def test_checkpoint_cadence_uses_completed_epochs(self):
        fake = SimpleNamespace(rank=0, cfg=OmegaConf.create({"ckpt": {"every_n_epochs": 10}}),
                               ckpt_dir=Path("unused"), train_step=0,
                               logger=SimpleNamespace(run_id="fixture"), to_ckpt={})
        with patch("src.trainers.base.save_ckpt") as save:
            for epoch in range(50):
                fake.epoch = epoch
                Trainer._save_checkpoint(fake)
        self.assertEqual([call.kwargs["filepath"].name for call in save.call_args_list],
                         [f"epoch_{epoch}.pt" for epoch in (10, 20, 30, 40, 50)])
        self.assertEqual([call.kwargs["epoch"] for call in save.call_args_list], [9, 19, 29, 39, 49])

    def test_failed_checkpoint_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_ckpt(path, epoch=4)
            original = path.read_bytes()
            def fail(value, destination):
                Path(destination).write_bytes(b"incomplete")
                raise OSError("simulated write failure")
            with patch("src.utils.checkpoint.torch.save", side_effect=fail):
                with self.assertRaises(OSError):
                    save_ckpt(path, epoch=5)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            save_ckpt(path, epoch=5)
            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 5)

    def test_all_sessions_with_no_evaluation_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, registry_path = fixture(root)
            manifest = root / "split.json"
            write_json(manifest, {"schema_version": 1, "partitions": {
                "pretrain": [{"dataset_id": "behavior", "sessions": ["1", "2"]},
                             {"dataset_id": "coding", "sessions": ["3", "4"]}], "eval": []}})
            dataset = AllenMultiDataset(root, "pretrain", "train+val", registry_path, manifest)
            self.assertEqual(len(dataset.recording_ids), 4)
            with self.assertRaisesRegex(ValueError, "evaluation requires"):
                AllenMultiDataset(root, "eval", "test", registry_path, manifest)


if __name__ == "__main__":
    unittest.main()
