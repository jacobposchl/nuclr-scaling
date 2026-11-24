from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import numpy as np
import torch

from src.losses.rate_separated import RateSeparatedNuCLRLoss, mean_squared_rate_correlation
from src.models.unit_encoders.nuclr import NuCLR
from test_gradient_accumulation import BatchList, make_trainer, run_epoch


MODEL = dict(ctx_duration=2., latent_step=1., bin_size=.25, dim=16,
             self_heads=2, dim_head=8, lin_dropout=0., t_layers=1, st_layers=1,
             rate_separation=True)


def batch():
    bins = torch.arange(64).reshape(8, 8) % 4
    return {"model_inputs": {"bins": bins, "spax_seqlen": torch.tensor([4, 4])},
            "unit_seqlen": torch.tensor([4, 4]), "unit_ids": np.array(["a", "b", "c", "d"] * 2),
            "probe_ids": np.array(["probe"] * 8)}


class RateSeparationTests(unittest.TestCase):
    def test_early_rate_branch_does_not_depend_on_later_layers(self):
        model = NuCLR(**MODEL)
        x = batch()["model_inputs"]
        z = model(**x)
        self.assertEqual(z.shape, (8, 16))
        self.assertTrue((z[:, 0] >= 0).all())
        z[:, 0].sum().backward()
        self.assertIsNotNone(model.rate_head.weight.grad)
        self.assertIsNotNone(model.read_in.weight.grad)
        self.assertTrue(all(p.grad is None or not p.grad.count_nonzero()
                            for p in model.st_blocks.parameters()))
        self.assertTrue(model.content_head.weight.grad is None or
                        not model.content_head.weight.grad.count_nonzero())
        before = z[:, 0].detach()
        with torch.no_grad():
            for p in model.st_blocks.parameters():
                p.add_(.5)
        torch.testing.assert_close(before, model(**x)[:, 0])

    def test_contrastive_loss_cannot_use_rate_coordinate(self):
        data = batch()
        loss = RateSeparatedNuCLRLoss(16, .2, True, rate_supervision_weight=0., rate_correlation_weight=0.)
        z = torch.randn(8, 16, requires_grad=True)
        args = (data["unit_seqlen"], data["unit_ids"], data["probe_ids"])
        a, _ = loss(z, *args, firing_rate_hz=torch.arange(8).float())
        changed = z.detach().clone()
        changed[:, 0] += 1000
        b, _ = loss(changed, *args, firing_rate_hz=torch.arange(8).float())
        torch.testing.assert_close(a, b)
        a.backward()
        torch.testing.assert_close(z.grad[:, 0], torch.zeros(8))
        self.assertGreater(z.grad[:, 1:].abs().sum().item(), 0.)

    def test_correlation_penalty_and_constant_rate_gradients(self):
        rate = torch.tensor([0., 1., 2., 3., 4.])
        x = torch.stack([rate, (rate - 2).square()], dim=1).requires_grad_()
        self.assertAlmostEqual(mean_squared_rate_correlation(x, rate).item(), .5, places=6)
        constant = mean_squared_rate_correlation(x, torch.ones(5))
        self.assertEqual(constant.item(), 0.)
        constant.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_training_targets_accumulation_and_checkpoint_reload(self):
        trainer = make_trainer([], micro_batch_size=1, batch_size=2)
        trainer.cfg.views = {"duration": 2.}
        trainer.model = NuCLR(**MODEL)
        trainer.ssl_loss = RateSeparatedNuCLRLoss(16, .2, True)
        trainer.rate_separation = True
        trainer.optimizer = torch.optim.AdamW([*trainer.model.parameters(), *trainer.ssl_loss.parameters()], lr=.001)
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.)
        data = batch()
        # Distinct rates also exercise the correlation term in a real backward pass.
        data["model_inputs"]["bins"] *= torch.arange(1, 9)[:, None]
        trainer.train_loader = BatchList([data] * 3)
        seen, components = [], []
        hook = trainer.ssl_loss.register_forward_hook(
            lambda module, args, kwargs, result: (seen.append(kwargs["firing_rate_hz"].clone()),
                                                components.append(result[1])) and None,
            with_kwargs=True)
        before = trainer.model.rate_head.weight.detach().clone()
        run_epoch(trainer)
        hook.remove()
        self.assertEqual(trainer.train_step, 2)
        self.assertEqual(len(seen), 3)
        for rates in seen:
            torch.testing.assert_close(rates, data["model_inputs"]["bins"].float().sum(1) / 2.)
        self.assertFalse(torch.equal(before, trainer.model.rate_head.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in trainer.model.parameters()))
        self.assertTrue(all("train/rate_correlation_penalty" in logs for logs in components))
        trainer.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"cfg": OmegaConf.create({"model": {"_target_": "src.models.unit_encoders.nuclr.NuCLR", **MODEL}}),
                        "model_state_dict": trainer.model.state_dict()}, path)
            restored = NuCLR.from_pretrained(path).eval()
            torch.testing.assert_close(trainer.model(**data["model_inputs"]), restored(**data["model_inputs"]))

    def test_experiment_configuration(self):
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.2"):
            cfg = compose(config_name="train_nuclr_allen_ibl_rate_separated")
        self.assertTrue(cfg.model.rate_separation)
        self.assertEqual((cfg.batch_size, cfg.micro_batch_size, cfg.num_epochs), (81, 3, 10))
        self.assertEqual(cfg.ckpt.every_n_epochs, 1)
        self.assertIsNone(cfg.ckpt.load_from)
        self.assertEqual(cfg.loss.rate_correlation_weight, .1)
        self.assertEqual(cfg.views.duration, 20.)


if __name__ == "__main__":
    unittest.main()
