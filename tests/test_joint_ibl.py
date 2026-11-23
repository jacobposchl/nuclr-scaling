"""IBL source discovery, identity handling, and joint contrastive batches."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch_brain.datasets import DatasetIndex

from src.artifacts.store import write_json
from src.datasets.allen_multi import AllenMultiDataset
from src.datasets.allen_sessions import AllenRegistry
from src.datasets.constrastive_wrapper import ContrastiveDatasetWrapper
from src.losses.nuclr_loss import NuCLRLoss
from src.models.unit_encoders.nuclr import NuCLR
from test_artifacts import fixture


class JointIBLTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        registry, self.registry_path = fixture(self.root)
        self.sessions = ["11111111-1111-4111-8111-111111111111",
                         "22222222-2222-4222-8222-222222222222"]
        for folder, session in zip(["pretrain", "eval"], self.sessions):
            path = self.root / "ibl" / folder / (session + ".h5")
            path.parent.mkdir(parents=True)
            shutil.copyfile(self.root / "behavior/1.h5", path)
            with h5py.File(path, "r+") as f:
                del f["units/id"], f["units/probe_id"]
                f["units/id"] = np.array([f"unit-{i}" for i in range(4)], dtype="S")
                f["units/probe_id"] = np.array(["probe-b", "probe-a", "probe-b", "probe-a"], dtype="S")
                f["units/qc_neural"] = np.array(["FAIL"] * 4, dtype="S")
        registry.config["datasets"]["ibl"] = {
            "directory": "ibl", "subdirectories": ["pretrain", "eval"],
            "session_id_format": "uuid", "subject_namespace": "ibl",
            "domain": "domain", "region_key": "area_name", "probe_key": "probe_id"}
        write_json(self.registry_path, registry.config)
        self.registry = AllenRegistry(self.registry_path)
        self.manifest = self.root / "split.json"
        write_json(self.manifest, {"schema_version": 1, "partitions": {
            "pretrain": [{"dataset_id": "behavior", "sessions": ["1"]},
                         {"dataset_id": "coding", "sessions": ["3"]},
                         {"dataset_id": "ibl", "sessions": self.sessions}], "eval": []}})

    def dataset(self, regime="pretrain"):
        return AllenMultiDataset(self.root, regime, "train+val", self.registry_path, self.manifest)

    def test_both_folders_and_lazy_window_preserve_source_units(self):
        self.assertEqual(self.registry.sessions("ibl"), self.sessions)
        self.assertEqual(self.registry.path("ibl", self.sessions[1]).parent.name, "eval")
        dataset = self.dataset()
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.get_unit_ids()), 16)
        desc = self.registry.describe("ibl", self.sessions[1])
        self.assertEqual(desc["subject_uid"], "ibl/mouse1")
        self.assertEqual(desc["order"].tolist(), [1, 3, 0, 2])
        data = self.registry.window("ibl", self.sessions[1], 0, 2, desc)
        self.assertEqual(len(data.units), 4)
        self.assertEqual(data.spikes.unit_index[0], 2)
        self.assertTrue((data.spikes.timestamps < 2).all())
        self.assertEqual(len(set(data.units.probe_id)), 2)
        self.assertTrue(all(row["cell_type"] is None for row in desc["rows"]))

    def test_source_ambiguity_and_invalid_ids_rejected(self):
        for value in ("../escape", "123", "" , self.sessions[0].replace("-", "")):
            with self.assertRaisesRegex(ValueError, "UUID"):
                self.registry.path("ibl", value)
        with self.assertRaises(FileNotFoundError):
            self.registry.path("ibl", "33333333-3333-4333-8333-333333333333")
        shutil.copyfile(self.registry.path("ibl", self.sessions[0]),
                        self.root / "ibl/eval" / (self.sessions[0] + ".h5"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.registry.sessions("ibl")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.registry.path("ibl", self.sessions[0])

    def test_subject_split_guard_uses_dataset_family(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["partitions"]["eval"] = [manifest["partitions"]["pretrain"].pop()]
        write_json(self.manifest, manifest)
        self.assertEqual(len(self.dataset()), 2)  # Allen mouse1 != IBL mouse1.
        manifest["partitions"]["pretrain"].append({"dataset_id": "ibl", "sessions": self.sessions[:1]})
        manifest["partitions"]["eval"][0]["sessions"] = self.sessions[1:]
        write_json(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "Subject appears"):
            self.dataset()

    def test_three_dataset_batch_forward_and_backward(self):
        model = NuCLR(ctx_duration=2., latent_step=1., bin_size=.25, dim=16,
                      self_heads=2, dim_head=8, lin_dropout=0., t_layers=1, st_layers=1)
        wrapper = ContrastiveDatasetWrapper(self.dataset(), model.tokenize)
        wrapper.enable_two_views(max_distance=0., seed=42)
        batch = wrapper.collate([wrapper[DatasetIndex(rid, 0., 2.)]
                                 for rid in wrapper.recording_ids], two_view=True)
        self.assertEqual(set(uid.split("/")[0] for uid in batch["unit_ids"]),
                         {"behavior", "coding", "ibl"})
        loss, _ = NuCLRLoss(16, .2, True)(model(**batch["model_inputs"]),
                    batch["unit_seqlen"], batch["unit_ids"], batch["probe_ids"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))


if __name__ == "__main__":
    unittest.main()
