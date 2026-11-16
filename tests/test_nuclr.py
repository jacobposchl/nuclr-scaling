"""Regression checks for the Zen port. Run only in an allocated compute job.

Run explicitly with: python -m unittest discover -s tests -v
ZEN_SOURCE_ROOT enables comparisons against a separate Zen source checkout.
"""

import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_brain.batching import collate

from src.datasets.constrastive_wrapper import ContrastiveDatasetWrapper
from src.losses.nuclr_loss import NuCLRLoss
from src.models.unit_encoders.nuclr import NuCLR


ROOT = Path(__file__).resolve().parents[1]
MODEL_KWARGS = dict(
    ctx_duration=2.0, latent_step=1.0, bin_size=0.25,
    dim=32, self_heads=2, dim_head=32, lin_dropout=0.0,
    t_layers=1, st_layers=1,
)


class Units(SimpleNamespace):
    def __len__(self):
        return len(self.id)


def example_data(probe_ids=("a", "a", "b")):
    return SimpleNamespace(
        units=Units(id=np.array(["u0", "u1", "u2"]), probe_id=np.array(probe_ids)),
        spikes=SimpleNamespace(
            # The endpoint spike must be discarded, not added to the last bin.
            timestamps=np.array([0.0, 0.24, 0.25, 1.99, 2.0]),
            unit_index=np.array([0, 0, 1, 2, 2]),
        ),
    )


def reference_loss(x, lengths, unit_ids, probe_ids, tau, dcl):
    """Explicit per-anchor objective, independent of the vectorized masks."""
    z = F.normalize(x, dim=-1)
    weighted = []
    total_matches = 0
    offset = 0
    for a, b in zip(lengths[::2], lengths[1::2]):
        n = a + b
        positives, denominators = [], []
        for i in range(offset, offset + n):
            negatives = []
            for j in range(offset, offset + n):
                if i == j:
                    continue
                similarity = torch.dot(z[i], z[j]) / tau
                if unit_ids[i] == unit_ids[j]:
                    positives.append(similarity)
                if probe_ids[i] == probe_ids[j] and (
                    not dcl or unit_ids[i] != unit_ids[j]
                ):
                    negatives.append(similarity)
            if negatives:
                denominators.append(torch.stack(negatives).logsumexp(0))
        offset += n
        if positives and denominators:
            count = len(positives)
            weighted.append(count * (
                -torch.stack(positives).mean() + torch.stack(denominators).mean()
            ))
            total_matches += count
    return sum(weighted, x.sum() * 0.0) / max(total_matches, 1)


class LossTests(unittest.TestCase):
    def test_matches_explicit_objective_and_gradients(self):
        # Unequal view sizes, different overlap counts, and multiple probes.
        ids = np.array(["a", "b", "c", "a", "b", "d", "e", "f", "d", "e", "f"])
        probes = np.array(["p", "p", "q", "p", "p"] + ["r"] * 6)
        lengths = [3, 2, 3, 3]
        for dcl in (True, False):
            with self.subTest(dcl=dcl):
                torch.manual_seed(42)
                x = torch.randn(len(ids), 8, requires_grad=True)
                expected_x = x.detach().clone().requires_grad_(True)
                actual, _ = NuCLRLoss(8, 0.2, dcl, projector=False)(
                    x, torch.tensor(lengths), ids, probes
                )
                expected = reference_loss(expected_x, lengths, ids, probes, 0.2, dcl)
                torch.testing.assert_close(actual, expected)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(x.grad, expected_x.grad)

    def test_no_overlap_does_not_reuse_previous_sample(self):
        x = torch.randn(8, 8, requires_grad=True)
        ids = np.array(["a", "b", "c", "d", "e", "f", "e", "f"])
        probes = np.array(["p"] * 8)
        objective = NuCLRLoss(8, 0.2, True, projector=False)
        actual, _ = objective(x, torch.tensor([2, 2, 2, 2]), ids, probes)
        expected, _ = objective(x[4:], torch.tensor([2, 2]), ids[4:], probes[4:])
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(x.grad[:4], torch.zeros_like(x.grad[:4]))

    def test_all_unusable_pairs_have_finite_zero_gradients(self):
        for ids, probes in (
            (["a", "b", "c", "d"], ["p"] * 4),  # no positives
            (["a", "b", "a", "b"], ["p", "q", "p", "q"]),  # no negatives
        ):
            with self.subTest(ids=ids, probes=probes):
                x = torch.randn(4, 8, requires_grad=True)
                objective = NuCLRLoss(8, 0.2, True)
                loss, _ = objective(x, torch.tensor([2, 2]), np.array(ids), np.array(probes))
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                torch.testing.assert_close(x.grad, torch.zeros_like(x.grad))
                for parameter in objective.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_singleton_probe_does_not_poison_other_gradients(self):
        x = torch.randn(6, 8, requires_grad=True)
        loss, _ = NuCLRLoss(8, 0.2, True, projector=False)(
            x, torch.tensor([3, 3]), np.array(["a", "b", "c"] * 2),
            np.array(["p", "q", "q"] * 2),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class TokenizerAndConfigTests(unittest.TestCase):
    def test_binning_and_probe_order(self):
        model = NuCLR(**MODEL_KWARGS)
        tokens = collate([model.tokenize(example_data(("z", "z", "a")))])
        expected = torch.zeros(3, 8, dtype=torch.int16)
        expected[0, 0], expected[1, 1], expected[2, 7] = 2, 1, 1
        torch.testing.assert_close(tokens["bins"], expected)
        torch.testing.assert_close(tokens["spax_seqlen"], torch.tensor([2, 1]))

    def test_interleaved_probes_are_rejected(self):
        with self.assertRaises(AssertionError):
            NuCLR(**MODEL_KWARGS).tokenize(example_data(("a", "b", "a")))

    def test_collation_keeps_view_pairs_adjacent(self):
        model = NuCLR(**MODEL_KWARGS)

        def item(prefix):
            data = example_data()
            return dict(model_inputs=model.tokenize(data), unit_seqlen=3,
                        unit_ids=[prefix + x for x in data.units.id],
                        probe_ids=data.units.probe_id.tolist())

        batch = ContrastiveDatasetWrapper.collate(
            [(item("a/"), item("b/")), None, (item("c/"), item("d/"))],
            two_view=True,
        )
        self.assertEqual(batch["unit_ids"][::3].tolist(), ["a/u0", "b/u0", "c/u0", "d/u0"])
        self.assertEqual(batch["unit_seqlen"].tolist(), [3, 3, 3, 3])
        self.assertIsNone(ContrastiveDatasetWrapper.collate([None], two_view=True))

    def test_zero_view_distance_means_same_window(self):
        dataset = SimpleNamespace(get_sampling_intervals=lambda: {})
        wrapper = ContrastiveDatasetWrapper(dataset, tokenizer=lambda x: x)
        wrapper.enable_two_views(max_distance=0, seed=42)
        index = SimpleNamespace(recording_id="s", start=3.0, end=5.0)
        self.assertIs(wrapper._get_second_view(index), index)

    def test_configs_compose_without_data_or_training(self):
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.2"):
            for name, batch_size in (("train_nuclr", 80), ("train_nuclr_allen_vbn", 16)):
                cfg = compose(config_name=name)
                self.assertEqual(cfg.batch_size, batch_size)
                self.assertEqual(cfg.views.duration, cfg.model.ctx_duration)
                self.assertFalse(cfg.ckpt.model_only)

    def test_checkpoint_format_round_trip(self):
        model = NuCLR(**MODEL_KWARGS)
        cfg = OmegaConf.create({"model": {"_target_": "src.models.unit_encoders.nuclr.NuCLR", **MODEL_KWARGS}})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"cfg": cfg, "model_state_dict": model.state_dict()}, path)
            restored = NuCLR.from_pretrained(path)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name])


def load_reference(relative_path, name):
    path = Path(os.environ["ZEN_SOURCE_ROOT"]) / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get("ZEN_SOURCE_ROOT"), "Set ZEN_SOURCE_ROOT for source parity checks")
class ZenParityTests(unittest.TestCase):
    def test_loss_and_projector_gradient_parity(self):
        reference = load_reference("src/losses/nuclr_loss.py", "zen_loss_reference")
        old = reference.NuCLRLoss(8, 0.2, True)
        new = NuCLRLoss(8, 0.2, True)
        new.load_state_dict(old.state_dict(), strict=True)
        inputs = torch.randn(10, 8)
        metadata = (torch.tensor([3, 3, 2, 2]),
                    np.array(["a", "b", "c", "a", "b", "c", "d", "e", "d", "e"]),
                    np.array(["p"] * 6 + ["q"] * 4))
        old_x = inputs.clone().requires_grad_(True)
        new_x = inputs.clone().requires_grad_(True)
        old_loss, _ = old(old_x, *metadata)
        new_loss, _ = new(new_x, *metadata)
        torch.testing.assert_close(old_loss, new_loss)
        old_loss.backward()
        new_loss.backward()
        torch.testing.assert_close(old_x.grad, new_x.grad)
        for a, b in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(a.grad, b.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "Zen spatial attention requires CUDA")
    def test_encoder_forward_and_backward_parity(self):
        reference = load_reference("src/models/unit_encoders/nuclr.py", "zen_encoder_reference")
        reference_attention = load_reference("src/nn/rotary_attention.py", "zen_attention_reference")
        reference.RotarySelfAttention = reference_attention.RotarySelfAttention
        old = reference.NuCLR(**MODEL_KWARGS).cuda()
        new = NuCLR(**MODEL_KWARGS).cuda()
        new.load_state_dict(old.state_dict(), strict=True)
        bins = torch.randint(0, 4, (10, 8), device="cuda")
        seqlen = torch.tensor([3, 3, 2, 2], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            old_out, new_out = old(bins, seqlen), new(bins, seqlen)
            old_loss, new_loss = old_out.square().mean(), new_out.square().mean()
        torch.testing.assert_close(old_out, new_out)
        old_loss.backward()
        new_loss.backward()
        for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, msg=name)


if __name__ == "__main__":
    unittest.main()
