"""Compare accumulated NuCLR updates with the same combined-batch objective."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.losses.nuclr_loss import NuCLRLoss
from src.trainers.gradient_accumulation import pretraining_batch_settings
from src.trainers.nuclr_pretrainer import NuCLRPreTrainer
from src.utils.checkpoint import save_ckpt


def example_pairs():
    generator = torch.Generator().manual_seed(12)
    pairs = []
    for index, count in enumerate([2, 5, 3, 4, 2, 6]):
        ids = [f"{index}/unit{unit}" for unit in range(count)]
        # The first pair has no shared neurons; other pairs have unequal weights.
        second_ids = [f"other/{uid}" for uid in ids] if index == 0 else ids
        pairs.append({"model_inputs": {"input": torch.randn(2 * count, 5, generator=generator, dtype=torch.float64)},
                      "unit_seqlen": torch.tensor([count, count]),
                      "unit_ids": np.array(ids + second_ids),
                      "probe_ids": np.array([f"probe{index}"] * (2 * count))})
    return pairs


def combine(pairs):
    return {"model_inputs": {"input": torch.cat([p["model_inputs"]["input"] for p in pairs])},
            "unit_seqlen": torch.cat([p["unit_seqlen"] for p in pairs]),
            "unit_ids": np.concatenate([p["unit_ids"] for p in pairs]),
            "probe_ids": np.concatenate([p["probe_ids"] for p in pairs])}


class BatchList(list):
    sampler = SimpleNamespace(set_epoch=lambda epoch: None)


def make_trainer(batches, *, micro_batch_size=None, batch_size=4, world_size=1):
    torch.manual_seed(42)
    trainer = NuCLRPreTrainer.__new__(NuCLRPreTrainer)
    trainer.cfg = OmegaConf.create({"batch_size": batch_size, "micro_batch_size": micro_batch_size,
        "wd": .01, "num_epochs": 1, "lr": {"peak": .001, "rise": .3, "min_factor": .01},
        "ckpt": {"resume": True}})
    trainer.micro_batch_size, trainer.accumulation_steps = pretraining_batch_settings(trainer.cfg, world_size)
    trainer.model = torch.nn.Linear(5, 8).double()
    trainer.ssl_loss = NuCLRLoss(8, .2, True).double()
    trainer.optimizer = torch.optim.AdamW(list(trainer.model.parameters()) + list(trainer.ssl_loss.parameters()), lr=.001)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.)
    trainer.train_loader = BatchList(batches)
    trainer.device = torch.device("cpu")
    trainer.precision = SimpleNamespace(dtype=torch.bfloat16)
    trainer.doing_eval = False
    trainer.epoch = trainer.train_step = trainer.rank = 0
    trainer.world_size = world_size
    trainer.to_ckpt = {}
    trainer.logger = MagicMock()
    trainer.logger.get_pbar.side_effect = lambda iterable, **kw: iterable
    trainer.log_lr = lambda: None
    trainer.push_logs = lambda: None
    return trainer


def run_epoch(trainer):
    # Float64 without autocast isolates accumulation arithmetic from roundoff.
    with patch("src.trainers.nuclr_pretrainer.torch.autocast", side_effect=lambda **kw: nullcontext()):
        trainer.train_epoch()


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        pairs = example_pairs()
        trainer = make_trainer([pairs[i] for i in range(rank, 6, 2)], micro_batch_size=1, world_size=2)
        trainer.model = DDP(trainer.model)
        trainer.ssl_loss = DDP(trainer.ssl_loss)
        run_epoch(trainer)
        torch.save({"model": trainer.model.module.state_dict(), "loss": trainer.ssl_loss.module.state_dict(),
                    "steps": trainer.train_step}, Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class AccumulationTests(unittest.TestCase):
    def assert_same_parameters(self, first, second):
        for a, b in zip(list(first.model.parameters()) + list(first.ssl_loss.parameters()),
                        list(second.model.parameters()) + list(second.ssl_loss.parameters())):
            # NuCLR computes its integer-count ratios in float32, even with
            # float64 embeddings, so grouping changes introduce float32 roundoff.
            torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)

    def test_only_explicit_micro_batch_enables_accumulation(self):
        for config in ({"batch_size": 24}, {"batch_size": 24, "micro_batch_size": None}):
            self.assertEqual(pretraining_batch_settings(OmegaConf.create(config), 2), (12, 1))
        config = OmegaConf.create({"batch_size": 24, "micro_batch_size": 3})
        self.assertEqual(pretraining_batch_settings(config, 1), (3, 8))
        self.assertEqual(pretraining_batch_settings(config, 2), (3, 4))

    def test_invalid_settings_fail_before_model_or_data_setup(self):
        settings = [(32, 3, 1), (3, 6, 1), (9, 3, 2), (3, None, 2)]
        settings += [(24, value, 1) for value in (0, -1, 1.5, False, True, "3")]
        settings += [(value, None, 1) for value in (0, -1, 1.5, False, True)]
        for batch, micro, world in settings:
            with self.subTest(batch=batch, micro=micro, world=world):
                trainer = NuCLRPreTrainer.__new__(NuCLRPreTrainer)
                trainer.cfg = OmegaConf.create({"batch_size": batch, "micro_batch_size": micro})
                trainer.world_size = world
                trainer.setup_model = MagicMock()
                trainer.setup_train_loader = MagicMock()
                with self.assertRaises(ValueError):
                    trainer.setup(None)
                trainer.setup_model.assert_not_called()
                trainer.setup_train_loader.assert_not_called()

    def test_weighted_updates_match_large_batches_including_final_partial_group(self):
        pairs = example_pairs()[:5]
        full = make_trainer([combine(pairs[:4]), pairs[4]])
        accumulated = make_trainer([combine(pairs[:2]), combine(pairs[2:4]), pairs[4]], micro_batch_size=2)
        run_epoch(full)
        run_epoch(accumulated)
        self.assert_same_parameters(full, accumulated)
        self.assertEqual(accumulated.train_step, 2)
        self.assertEqual(accumulated.lr_scheduler.last_epoch, 2)
        self.assertTrue(all(p.grad is None for group in accumulated.optimizer.param_groups for p in group["params"]))
        for direct, micro in zip(full.optimizer.state.values(), accumulated.optimizer.state.values()):
            for key in direct:
                torch.testing.assert_close(direct[key], micro[key], rtol=1e-6, atol=1e-9)

    def test_missing_micro_batches_do_not_discard_pending_gradients(self):
        a, b = example_pairs()[1:3]
        full = make_trainer([a, b], batch_size=2)
        accumulated = make_trainer([a, None, None, b, None], micro_batch_size=1, batch_size=2)
        run_epoch(full)
        run_epoch(accumulated)
        self.assert_same_parameters(full, accumulated)
        self.assertEqual(accumulated.train_step, 2)
        self.assertEqual(accumulated.lr_scheduler.last_epoch, 2)

    def test_zero_weight_group_skips_update_only_when_accumulation_is_enabled(self):
        pair = example_pairs()[0]
        accumulated = make_trainer([pair, pair], micro_batch_size=1, batch_size=2)
        before = deepcopy(accumulated.model.state_dict())
        run_epoch(accumulated)
        self.assertEqual(accumulated.train_step, 0)
        self.assertEqual(accumulated.lr_scheduler.last_epoch, 0)
        for key, value in accumulated.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        legacy = make_trainer([pair], batch_size=1)
        run_epoch(legacy)
        self.assertEqual(legacy.train_step, 1)

    def test_schedule_and_checkpoint_track_optimizer_updates(self):
        trainer = make_trainer([example_pairs()[1]] * 7, micro_batch_size=1, batch_size=3)
        trainer.setup_optimizer(None)
        self.assertEqual(trainer.optimizer_steps_per_epoch, 3)
        trainer.logger.warn.assert_called_once()
        run_epoch(trainer)
        self.assertEqual(trainer.train_step, 3)
        self.assertEqual(trainer.lr_scheduler.last_epoch, 3)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], .00001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch_1.pt"
            save_ckpt(path, model=trainer.model, ssl_loss=trainer.ssl_loss, optimizer=trainer.optimizer,
                      lr_scheduler=trainer.lr_scheduler, cfg=trainer.cfg, epoch=0, train_step=trainer.train_step)
            checkpoint = torch.load(path, weights_only=False)
            restored = make_trainer(list(trainer.train_loader), micro_batch_size=1, batch_size=3)
            restored.setup_optimizer(checkpoint)
            self.assertEqual(checkpoint["train_step"], 3)
            self.assertEqual(restored.lr_scheduler.state_dict(), trainer.lr_scheduler.state_dict())
            for state in restored.optimizer.state.values():
                self.assertEqual(int(state["step"]), 3)

    def test_resume_rejects_changed_batch_settings(self):
        trainer = make_trainer([], micro_batch_size=1, batch_size=4)
        old_cfg = deepcopy(trainer.cfg)
        old_cfg.micro_batch_size = 2
        with self.assertRaisesRegex(ValueError, "Cannot change micro_batch_size"):
            trainer.setup({"cfg": old_cfg})

    def test_runner_records_optimizer_updates_separately_from_micro_batches(self):
        import scripts.run_allen_pretraining as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("{}")
            trainer = make_trainer([example_pairs()[1]] * 7, micro_batch_size=1, batch_size=3)
            cfg = trainer.cfg
            cfg.dataset = {"registry": str(source), "split_manifest": str(source)}
            cfg.trainer = {}
            cfg.wandb = {"mode": "disabled", "dir": str(root / "wandb")}
            cfg.ckpt.enable = cfg.ckpt.save_last = True
            cfg.ckpt.load_from = None
            trainer.train_dataset = SimpleNamespace(recording_ids=["1/session"])
            trainer.train_loader.sampler = MagicMock()
            trainer.train_loader.sampler.__len__.return_value = 7
            trainer.logger.run_id = "test"
            trainer.setup_optimizer(None)
            trainer.add_checkpoint_items(model=trainer.model, ssl_loss=trainer.ssl_loss)
            trainer.ckpt_dir = root / "run/checkpoints/test"
            trainer.train = lambda: (run_epoch(trainer), trainer._save_last_checkpoint())
            with patch.object(runner, "initialize_config_dir", return_value=nullcontext()), \
                 patch.object(runner, "compose", return_value=cfg), \
                 patch.object(runner, "instantiate", return_value=trainer), \
                 patch.object(runner, "implementation_stamp", return_value={}), \
                 patch.object(runner.signal, "signal"), \
                 patch("sys.argv", ["run", "--local", "--output", str(root / "run")]), \
                 patch.multiple(torch.cuda, device_count=lambda: 1, set_device=lambda value: None,
                                get_device_name=lambda value: "CPU test", reset_peak_memory_stats=lambda: None,
                                max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0):
                runner.main()
            status = json.loads((root / "run/status.json").read_text())
            records = [json.loads(line) for line in (root / "run/steps.jsonl").read_text().splitlines()]
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["steps_per_epoch"], 3)
            self.assertEqual(status["micro_batches_per_epoch"], 7)
            self.assertEqual(status["last_epoch"]["micro_batches"], 7)
            self.assertEqual(status["last_epoch"]["train_step"], 3)
            self.assertEqual([r["train_step"] for r in records], [1, 2, 3])
            self.assertEqual([r["micro_batches"] for r in records], [3, 6, 7])

    def test_two_rank_accumulation_matches_combined_global_batches(self):
        pairs = example_pairs()
        full = make_trainer([combine(pairs[:4]), combine(pairs[4:])])
        run_epoch(full)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)
            for rank in range(2):
                result = torch.load(Path(directory) / f"rank{rank}.pt", weights_only=True)
                self.assertEqual(result["steps"], 2)
                for name, expected in (("model", full.model), ("loss", full.ssl_loss)):
                    for key, value in expected.state_dict().items():
                        torch.testing.assert_close(result[name][key], value, rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
