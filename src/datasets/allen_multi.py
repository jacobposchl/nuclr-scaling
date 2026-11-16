"""One pretraining dataset spanning explicit Allen session selections.

The existing fixed-window sampler samples available recording time. Dataset
balancing is deliberately not implied: its policy is recorded as window-based.
"""

import json
from pathlib import Path

import numpy as np

from .allen_sessions import AllenRegistry, namespace


class AllenMultiDataset:
    evaluator_cls = None

    def __init__(self, root, regime, split, registry, split_manifest, transform=None):
        if regime not in ("pretrain", "eval") or split not in ("train", "val", "train+val", "test"):
            raise ValueError("Unsupported regime or split")
        self.registry = AllenRegistry(registry, root)
        manifest = json.loads(Path(split_manifest).read_text())
        if manifest.get("schema_version") != 1 or set(manifest["partitions"]) != {"pretrain", "eval"}:
            raise ValueError("Split manifest must define pretrain and eval partitions")
        self.transform = transform
        descriptions, source_sessions, selected_keys = {}, {}, {"pretrain": [], "eval": []}
        subject_roles = {}
        for role, entries in manifest["partitions"].items():
            if not entries and (role == "pretrain" or regime == "eval"):
                raise ValueError(f"Empty {role} partition; evaluation requires explicitly held-out sessions")
            for entry in entries:
                dataset = entry["dataset_id"]
                sessions = entry["sessions"]
                if not isinstance(sessions, list) or not sessions:
                    raise ValueError("Pretraining splits require explicit session lists")
                for session in map(str, sessions):
                    rid = namespace(dataset, session)
                    if rid in descriptions:
                        raise ValueError(f"Duplicate or overlapping split session: {rid}")
                    desc = self.registry.describe(dataset, session)
                    subject = namespace("allen", desc["subject_id"])
                    if subject in subject_roles and subject_roles[subject] != role:
                        raise ValueError(f"Subject appears in pretrain and eval: {subject}")
                    subject_roles[subject] = role
                    descriptions[rid] = (dataset, session, desc)
                    selected_keys[role].append(rid)
                    source_sessions.setdefault(dataset, []).append(session)
        self.recording_ids = sorted(selected_keys[regime])
        self._recordings = {rid: descriptions[rid] for rid in self.recording_ids}
        self.provenance = {
            "split_manifest": manifest, "sampling": "uniform_available_windows_no_dataset_reweighting",
            "sources": [self.registry.fingerprint(dataset, sessions) for dataset, sessions in sorted(source_sessions.items())],
        }

    def get_sampling_intervals(self):
        from torch_brain.data import Interval
        return {rid: Interval(start=np.array([a for a, b in desc["intervals"]]),
                              end=np.array([b for a, b in desc["intervals"]]))
                for rid, (_, _, desc) in self._recordings.items()}

    def get_unit_ids(self):
        return np.array([row["unit_uid"] for _, _, desc in self._recordings.values() for row in desc["rows"]])

    def __getitem__(self, index):
        dataset, session, desc = self._recordings[index.recording_id]
        data = self.registry.window(dataset, session, index.start, index.end, desc)
        return self.transform(data) if self.transform is not None else data

    def __len__(self):
        return len(self.recording_ids)
