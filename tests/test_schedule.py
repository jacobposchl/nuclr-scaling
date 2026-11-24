"""Check the actual training schedule for fractional-epoch warmup budgets."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf
import torch

from src.trainers.nuclr_pretrainer import NuCLRPreTrainer


class ScheduleTests(unittest.TestCase):
    def test_fractional_warmup_reaches_peak_and_final_learning_rates(self):
        trainer = NuCLRPreTrainer.__new__(NuCLRPreTrainer)
        trainer.cfg = OmegaConf.create({"wd": 0.01, "num_epochs": 1,
                                       "lr": {"peak": 5e-4, "rise": 0.3, "min_factor": 0.01}})
        trainer.model = torch.nn.Linear(2, 2)
        trainer.ssl_loss = torch.nn.Linear(2, 2)
        trainer.train_loader = range(7)  # 0.3 * 7 = 2.1, rounded to two steps.
        trainer.accumulation_steps = 1
        trainer.rank = 0
        trainer.to_ckpt = {}
        trainer.logger = SimpleNamespace(info=lambda *args: None)
        trainer.setup_optimizer(None)
        rates = [trainer.optimizer.param_groups[0]["lr"]]
        for _ in range(7):
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            rates.append(trainer.optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(rates[0], 5e-6)
        self.assertAlmostEqual(rates[2], 5e-4)
        self.assertAlmostEqual(max(rates), 5e-4)
        self.assertAlmostEqual(rates[-1], 5e-6)

    def make_trainer(self, epochs, cycle_start=0):
        trainer = NuCLRPreTrainer.__new__(NuCLRPreTrainer)
        trainer.cfg = OmegaConf.create({"wd": .01, "num_epochs": epochs,
            "lr": {"peak": 5e-4, "rise": 1, "min_factor": .01, "cycle_start_epoch": cycle_start},
            "ckpt": {"resume": True}})
        trainer.model = torch.nn.Linear(2, 2)
        trainer.ssl_loss = torch.nn.Linear(2, 2)
        trainer.train_loader = range(4)
        trainer.accumulation_steps = 1
        trainer.rank = 0
        trainer.to_ckpt = {}
        trainer.logger = SimpleNamespace(info=lambda *args: None)
        return trainer

    def step(self, trainer):
        trainer.optimizer.zero_grad()
        trainer.model(torch.ones(1, 2)).sum().backward()
        trainer.optimizer.step()
        trainer.lr_scheduler.step()

    def checkpoint(self, trainer, epoch):
        return deepcopy(dict(cfg=OmegaConf.to_container(trainer.cfg), epoch=epoch,
            optimizer_state_dict=trainer.optimizer.state_dict(),
            lr_scheduler_state_dict=trainer.lr_scheduler.state_dict()))

    def test_extension_preserves_adam_and_resumes_new_cycle(self):
        original = self.make_trainer(10)
        original.setup_optimizer(None)
        for _ in range(40):
            self.step(original)
        checkpoint = self.checkpoint(original, 9)
        continuation = self.make_trainer(20, cycle_start=10)
        continuation.setup_optimizer(checkpoint)
        states = list(continuation.optimizer.state.values())
        previous = list(original.optimizer.state.values())
        for old, new in zip(previous, states):
            for key in ['step', 'exp_avg', 'exp_avg_sq']:
                torch.testing.assert_close(old[key], new[key])
        self.assertAlmostEqual(continuation.optimizer.param_groups[0]['lr'], 5e-6)
        for _ in range(4):
            self.step(continuation)
        self.assertAlmostEqual(continuation.optimizer.param_groups[0]['lr'], 5e-4)
        for _ in range(8):
            self.step(continuation)
        resumed = self.make_trainer(20, cycle_start=10)
        resumed.setup_optimizer(self.checkpoint(continuation, 12))
        self.assertEqual(resumed.lr_scheduler.state_dict(), continuation.lr_scheduler.state_dict())
        for _ in range(28):
            self.step(continuation)
            self.step(resumed)
            self.assertAlmostEqual(resumed.optimizer.param_groups[0]['lr'], continuation.optimizer.param_groups[0]['lr'])
        self.assertAlmostEqual(continuation.optimizer.param_groups[0]['lr'], 5e-6)

    def test_extension_requires_explicit_boundary(self):
        original = self.make_trainer(10)
        original.setup_optimizer(None)
        checkpoint = self.checkpoint(original, 9)
        with self.assertRaisesRegex(ValueError, 'explicit new'):
            self.make_trainer(20).setup_optimizer(checkpoint)
        with self.assertRaisesRegex(ValueError, 'checkpoint boundary'):
            self.make_trainer(20, cycle_start=9).setup_optimizer(checkpoint)


if __name__ == "__main__":
    unittest.main()
