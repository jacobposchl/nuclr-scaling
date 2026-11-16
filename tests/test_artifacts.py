"""CPU scaffold tests. Run only within an explicitly provided compute allocation."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.artifacts.analyses import analyze
from src.artifacts.collections import derive_means, extract, load_unit_means
from src.artifacts.store import ArtifactStore, write_json
from src.datasets.allen_multi import AllenMultiDataset
from src.datasets.allen_sessions import AllenRegistry, windows
from src.datasets.constrastive_wrapper import ContrastiveDatasetWrapper
from src.models.unit_encoders.nuclr import NuCLR
from src.nn.rotary_attention import _attn_varlen


def checkpoint(path):
    torch.manual_seed(7)
    cfg = dict(ctx_duration=2.0, latent_step=1.0, bin_size=0.25, dim=16,
               self_heads=2, dim_head=8, lin_dropout=0.0, t_layers=1, st_layers=1)
    model = NuCLR(**cfg)
    torch.save({"cfg": OmegaConf.create({"model": cfg}), "model_state_dict": model.state_dict()}, path)


def fixture(root):
    specs = {}
    for name, sessions in (("behavior", ["1", "2"]), ("coding", ["3", "4"])):
        folder = root / name
        folder.mkdir()
        specs[name] = {"directory": name, "domain": "domain", "region_key": "area_name",
                       "probe_key": "probe_id" if name == "behavior" else "unit_id_prefix"}
        for session in sessions:
            with h5py.File(folder / (session + ".h5"), "w") as f:
                f.create_group("subject").attrs["id"] = "mouse" + session
                f["domain/start"] = [0.0, 8.0]
                f["domain/end"] = [6.0, 12.0]
                f["spikes/domain/start"], f["spikes/domain/end"] = [0.0], [12.0]
                events = sorted((t, i) for i in range(4) for t in np.arange(i * .03, 12, .13 + i * .07))
                f["spikes/timestamps"] = [t for t, i in events]
                f["spikes/unit_index"] = [i for t, i in events]
                f["units/id"] = (np.array(["9/0/0", "8/1/1", "9/2/2", "8/3/3"], dtype="S")
                                 if name == "coding" else np.arange(4))
                f["units/probe_id"] = [9, 8, 9, 8]
                f["units/area_name"] = np.array(["A", "B", "A", "B"], dtype="S")
    registry_path = root / "registry.json"
    write_json(registry_path, {"schema_version": 1, "root": str(root), "datasets": specs})
    return AllenRegistry(registry_path), registry_path


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ArtifactStore(Path(self.temp.name) / "artifacts")

    def test_atomic_failure_and_cache_integrity(self):
        recipe = {"operation": "test", "a": 1}
        with self.assertRaisesRegex(RuntimeError, "producer"):
            with self.store.build("collections", recipe) as (stage, aid):
                (stage / "partial.txt").write_text("incomplete")
                raise RuntimeError("producer failed")
        self.assertEqual(list(self.store.list()), [])
        with self.store.build("collections", recipe, "first label") as (stage, aid):
            (stage / "values.txt").write_text("complete")
        with self.store.build("collections", {"a": 1, "operation": "test"}, "another label") as (stage, reused):
            self.assertIsNone(stage)
            self.assertEqual(reused, aid)
        self.assertEqual(len(list(self.store.list())), 1)
        (self.store.path("collections", aid) / "values.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.store.find("collections", recipe)

    def test_checkpoint_mutation_and_identity(self):
        path = Path(self.temp.name) / "checkpoint.pt"
        path.write_bytes(b"original bytes")
        aid = self.store.register_checkpoint(path, {"status": "unknown"})
        self.assertEqual(self.store.checkpoint_path(aid), path)
        path.write_bytes(b"new bytes")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.checkpoint_path(aid)

    def test_lock_and_path_traversal_are_rejected(self):
        recipe = {"operation": "concurrent"}
        with self.store.build("analyses", recipe) as (stage, aid):
            with self.assertRaisesRegex(RuntimeError, "lock"):
                with self.store.build("analyses", recipe):
                    pass
            write_json(stage / "result.json", {})
        with self.assertRaises(ValueError):
            self.store.path("collections", "../escape")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry, self.registry_path = fixture(self.root)
        self.store = ArtifactStore(self.root / "artifacts")
        path = self.root / "test.pt"
        checkpoint(path)
        self.checkpoint_id = self.store.register_checkpoint(path, {"status": "untrained_test_fixture"})
        self.config = {"datasets": [{"dataset_id": "behavior", "sessions": ["1", "2"]},
                                    {"dataset_id": "coding", "sessions": ["3", "4"]}],
                       "step_seconds": 2.0, "max_windows_per_session": 2, "output": "both"}

    def test_window_read_namespace_order_and_gap(self):
        desc = self.registry.describe("coding", "3")
        data = self.registry.window("coding", "3", 0, 2, desc)
        self.assertEqual([r["probe_id"] for r in desc["rows"]], ["8", "8", "9", "9"])
        self.assertEqual(desc["order"].tolist(), [1, 3, 0, 2])
        self.assertTrue(all(u.startswith("coding/3/") for u in data.units.id))
        self.assertTrue((data.spikes.timestamps >= 0).all())
        self.assertTrue((data.spikes.timestamps < 2).all())
        self.assertEqual(data.spikes.unit_index[0], 2)  # original unit 0 remaps to position 2
        with self.assertRaisesRegex(ValueError, "gap"):
            self.registry.window("coding", "3", 5, 9)
        self.assertEqual(list(windows(desc["intervals"], 2, 2)), [(0., 2.), (2., 4.), (4., 6.), (8., 10.), (10., 12.)])

    def test_full_extraction_cache_derived_means_and_analyses(self):
        cid = extract(self.store, self.checkpoint_id, self.registry, self.config)
        self.assertEqual(cid, extract(self.store, self.checkpoint_id, self.registry, self.config))
        derived = derive_means(self.store, cid)
        x, rows = load_unit_means(self.store, [cid])
        means, other_rows = load_unit_means(self.store, [derived])
        np.testing.assert_array_equal(x, means)
        self.assertEqual(x.shape, (16, 16))
        self.assertEqual(len({r["unit_uid"] for r in rows}), 16)
        one_window = derive_means(self.store, cid, [0])
        self.assertNotEqual(one_window, derived)
        _, window_rows = load_unit_means(self.store, [one_window])
        for item, derived_row, window_row in zip(rows, other_rows, window_rows):
            desc = self.registry.describe(item["dataset_id"], item["session_id"])
            unit_index = [r["unit_uid"] for r in desc["rows"]].index(item["unit_uid"])
            data = self.registry.window(item["dataset_id"], item["session_id"], 0, 4, desc)
            expected = np.count_nonzero(data.spikes.unit_index == unit_index) / 4
            self.assertEqual(item["firing_rate_hz"], expected)
            self.assertEqual(derived_row["firing_rate_hz"], expected)
            first = self.registry.window(item["dataset_id"], item["session_id"], 0, 2, desc)
            self.assertEqual(window_row["firing_rate_hz"], np.count_nonzero(first.spikes.unit_index == unit_index) / 2)
        pca = analyze(self.store, [cid], {"kind": "pca"})
        self.assertTrue((self.store.path("analyses", pca) / "pca.png").is_file())
        config = {"kind": "linear_probe", "test_groups": ["allen/mouse2", "allen/mouse4"]}
        probe = analyze(self.store, [cid], config)
        self.assertEqual(probe, analyze(self.store, [cid], config))
        path = self.store.path("analyses", probe)
        split = json.loads((path / "split.json").read_text())
        self.assertFalse(set(split["train_groups"]) & set(split["test_groups"]))
        model = np.load(path / "linear_probe.npz")
        train = np.array([r["unit_uid"] in split["train_unit_uids"] for r in rows])
        np.testing.assert_allclose(model["mean"], x[train].mean(0), atol=1e-6)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            load_unit_means(self.store, [cid, derived])
        with self.assertRaisesRegex(ValueError, "max_rows"):
            load_unit_means(self.store, [cid], max_rows=1)
        before = self.store.read("collections", cid)["recipe"]
        with h5py.File(self.root / "behavior/1.h5", "r+") as f:
            f.attrs["new_preprocessing"] = "changed"
        after = self.registry.fingerprint("behavior", ["1", "2"])
        self.assertNotEqual(before["sources"][0], after)

    def test_mixed_dataset_and_subject_leakage(self):
        manifest = {"schema_version": 1, "partitions": {
            "pretrain": [{"dataset_id": "behavior", "sessions": ["1"]}, {"dataset_id": "coding", "sessions": ["3"]}],
            "eval": [{"dataset_id": "behavior", "sessions": ["2"]}, {"dataset_id": "coding", "sessions": ["4"]}],
        }}
        path = self.root / "splits.json"
        write_json(path, manifest)
        ds = AllenMultiDataset(self.root, "pretrain", "train+val", self.registry_path, path)
        self.assertEqual(ds.recording_ids, ["behavior/1", "coding/3"])
        self.assertEqual(len(ds.get_unit_ids()), 8)
        from torch_brain.datasets import DatasetIndex
        wrapper = ContrastiveDatasetWrapper(ds, NuCLR.from_pretrained(self.root / "test.pt").tokenize)
        wrapper.enable_two_views(max_distance=None, seed=42)
        index = DatasetIndex(recording_id="coding/3", start=4., end=6.)
        for _ in range(30):
            second = wrapper._get_second_view(index)
            self.assertTrue(second.end <= 6 or second.start >= 8)
        behavior_index = DatasetIndex(recording_id="behavior/1", start=0., end=2.)
        batch = ContrastiveDatasetWrapper.collate([wrapper[index], wrapper[behavior_index]], two_view=True)
        model = NuCLR.from_pretrained(self.root / "test.pt")
        from src.losses.nuclr_loss import NuCLRLoss
        z = model(**batch["model_inputs"])
        loss, _ = NuCLRLoss(16, .2, True)(z, batch["unit_seqlen"], batch["unit_ids"], batch["probe_ids"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with h5py.File(self.root / "behavior/2.h5", "r+") as f:
            f["subject"].attrs["id"] = "mouse1"
        with self.assertRaisesRegex(ValueError, "Subject"):
            AllenMultiDataset(self.root, "pretrain", "train+val", self.registry_path, path)

    def test_cpu_segmented_attention_matches_block_mask_and_gradients(self):
        torch.manual_seed(8)
        q, k, v = [torch.randn(5, 2, 8, requires_grad=True) for _ in range(3)]
        lengths = torch.tensor([2, 3])
        actual = _attn_varlen(q, k, v, lengths)
        mask = torch.zeros(5, 5, dtype=torch.bool)
        mask[:2, :2] = True
        mask[2:, 2:] = True
        expected = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask).transpose(0, 1)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), (q, k, v))
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)


@unittest.skipUnless(os.environ.get("NUCLR_REAL_DATA_SMOKE") == "1", "Opt-in real-data CPU smoke test")
class RealDataSmokeTests(unittest.TestCase):
    def test_one_window_from_each_allen_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "diagnostic_untrained.pt"
            checkpoint(path)
            store = ArtifactStore(root / "artifacts")
            checkpoint_id = store.register_checkpoint(path, {"status": "diagnostic_untrained"})
            registry = AllenRegistry()
            config = {"datasets": [{"dataset_id": dataset, "sessions": [registry.sessions(dataset)[0]]}
                                    for dataset in registry.config["datasets"]],
                      "max_windows_per_session": 1, "output": "both"}
            cid = extract(store, checkpoint_id, registry, config)
            values, rows = load_unit_means(store, [cid])
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(len(set(r["dataset_id"] for r in rows)), 2)
            self.assertGreater(len(rows), 100)
            print(f"Real-data diagnostic: {len(rows)} units, 2 sessions, CPU, untrained small model; temporary artifacts removed")


if __name__ == "__main__":
    unittest.main()
