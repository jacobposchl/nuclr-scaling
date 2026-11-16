"""Read small windows from either Allen HDF5 release without loading a session.

Shared data are opened read-only. IDs include dataset and session namespaces;
local IDs and raw metadata remain available for joins. No filtering or session
split is silently inferred from the file's location.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import h5py
import numpy as np

from src.artifacts.store import file_digest


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def namespace(*parts):
    return "/".join(quote(text(x), safe="") for x in parts)


@dataclass
class UnitTable:
    id: np.ndarray
    probe_id: np.ndarray

    def __len__(self):
        return len(self.id)

    def select_by_mask(self, mask):
        return UnitTable(self.id[mask], self.probe_id[mask])


@dataclass
class SpikeTable:
    timestamps: np.ndarray
    unit_index: np.ndarray

    def select_by_mask(self, mask):
        return SpikeTable(self.timestamps[mask], self.unit_index[mask])


def lower_bound(dataset, value):
    """Binary search sorted HDF5 timestamps; never materialize the full array."""
    left, right = 0, len(dataset)
    while left < right:
        middle = (left + right) // 2
        if dataset[middle] < value:
            left = middle + 1
        else:
            right = middle
    return left


class AllenRegistry:
    def __init__(self, path="configs/data_registry.json", root=None):
        self.config = json.loads(Path(path).read_text())
        if self.config["schema_version"] != 1:
            raise ValueError("Unsupported dataset registry version")
        self.root = Path(root or self.config["root"]).expanduser().resolve()

    def spec(self, dataset_id):
        return self.config["datasets"][dataset_id]

    def directory(self, dataset_id):
        return (self.root / self.spec(dataset_id)["directory"]).resolve()

    def sessions(self, dataset_id):
        return sorted(p.stem for p in self.directory(dataset_id).glob("*.h5"))

    def path(self, dataset_id, session_id):
        if not str(session_id).isdigit():
            raise ValueError("Allen session IDs must be numeric")
        return self.directory(dataset_id) / (str(session_id) + ".h5")

    def fingerprint(self, dataset_id, sessions, mode="stat"):
        if mode not in ("stat", "sha256"):
            raise ValueError("fingerprint must be stat or sha256")
        files = []
        for session in sorted(sessions):
            path = self.path(dataset_id, session).resolve(strict=True)
            st = path.stat()
            record = {"session_id": session, "path": str(path), "size": st.st_size,
                      "mtime_ns": st.st_mtime_ns}
            if mode == "sha256":
                record["sha256"] = file_digest(path)
            files.append(record)
        return {"dataset_id": dataset_id, "spec": self.spec(dataset_id),
                "fingerprint_mode": mode, "files": files}

    def describe(self, dataset_id, session_id):
        spec = self.spec(dataset_id)
        with h5py.File(self.path(dataset_id, session_id), "r") as f:
            local_ids = np.array([text(x) for x in f["units/id"][:]])
            if len(set(local_ids)) != len(local_ids):
                raise ValueError("Duplicate local unit IDs")
            if spec["probe_key"] == "unit_id_prefix":
                pieces = [x.split("/") for x in local_ids]
                if not all(len(x) == 3 and all(y.isdigit() for y in x) for x in pieces):
                    raise ValueError("Expected Visual Coding probe/channel/unit IDs")
                probes = np.array([x[0] for x in pieces])
            else:
                probes = np.array([text(x) for x in f["units/" + spec["probe_key"]][:]])
            regions = [text(x) for x in f["units/" + spec["region_key"]][:]]
            if len(probes) != len(local_ids) or len(regions) != len(local_ids):
                raise ValueError("Unit metadata lengths do not match")
            extra_metadata = {
                key: f["units/" + key][:]
                for key in ("optotagged_cell_type", "optotagging_cre_positive", "quality", "type")
                if "units/" + key in f
            }
            subject_id = text(f["subject"].attrs["id"])
            # A stable sort preserves original order within each probe.
            order = np.argsort(probes, kind="stable")
            rows = []
            for i in order:
                row = {"dataset_id": dataset_id, "session_id": str(session_id),
                       "session_uid": namespace(dataset_id, session_id),
                       "subject_id": subject_id, "subject_uid": namespace("allen", subject_id),
                       "unit_id": local_ids[i], "unit_uid": namespace(dataset_id, session_id, local_ids[i]),
                       "probe_id": probes[i], "probe_uid": namespace(dataset_id, session_id, probes[i]),
                       "brain_region": regions[i]}
                for key, values in extra_metadata.items():
                    row[key] = text(values[i])
                # Keep the original annotation alongside the display label.
                # 'wt' is a fallback in this preprocessing, not a cell type.
                annotation = row.get("optotagged_cell_type")
                row["cell_type"] = None if annotation in (None, "", "wt") else annotation
                rows.append(row)
            starts, ends = f[spec["domain"] + "/start"][:], f[spec["domain"] + "/end"][:]
            spike_start = float(f["spikes/domain/start"][0])
            spike_end = float(f["spikes/domain/end"][-1])
            intervals = sorted((max(float(a), spike_start), min(float(b), spike_end))
                               for a, b in zip(starts, ends) if max(a, spike_start) < min(b, spike_end))
            if any(a < previous_end for (_, previous_end), (a, _) in zip(intervals, intervals[1:])):
                raise ValueError("Sampling intervals overlap")
            return {"rows": rows, "order": order, "intervals": intervals,
                    "subject_id": subject_id, "num_spikes": len(f["spikes/timestamps"])}

    def window(self, dataset_id, session_id, start, end, description=None):
        desc = description or self.describe(dataset_id, session_id)
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            raise ValueError("Window bounds must be finite and increasing")
        if not any(a <= start and end <= b for a, b in desc["intervals"]):
            raise ValueError("Window crosses a gap or leaves the selected domain")
        with h5py.File(self.path(dataset_id, session_id), "r") as f:
            ts = f["spikes/timestamps"]
            lo, hi = lower_bound(ts, start), lower_bound(ts, end)
            timestamps = ts[lo:hi] - start
            original_indices = f["spikes/unit_index"][lo:hi].astype(np.int64)
        if len(original_indices) and (original_indices.min() < 0 or original_indices.max() >= len(desc["rows"])):
            raise ValueError("Spike unit index is outside the unit table")
        inverse = np.empty(len(desc["order"]), dtype=np.int64)
        inverse[desc["order"]] = np.arange(len(inverse))
        return SimpleNamespace(
            units=UnitTable(np.array([r["unit_uid"] for r in desc["rows"]]),
                            np.array([r["probe_uid"] for r in desc["rows"]])),
            spikes=SpikeTable(timestamps, inverse[original_indices]),
        )


def windows(intervals, duration, step, limit=None):
    if not (np.isfinite(duration) and np.isfinite(step) and duration > 0 and step > 0):
        raise ValueError("Window duration and step must be positive and finite")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("Window limit must be a positive integer or null")
    count = 0
    for start, end in intervals:
        for i in range(max(0, int(np.floor((end - start - duration) / step)) + 1)):
            a = start + i * step
            yield a, a + duration
            count += 1
            if limit is not None and count >= limit:
                return
