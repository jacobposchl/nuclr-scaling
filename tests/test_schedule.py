"""Check the actual training schedule for fractional-epoch warmup budgets."""

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


if __name__ == "__main__":
    unittest.main()
