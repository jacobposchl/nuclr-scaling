"""Read small windows from the Allen and IBL HDF5 releases lazily.

Shared data are opened read-only. IDs include dataset and session namespaces;
local IDs and raw metadata remain available for joins. No filtering or session
split is silently inferred from the file's location.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from uuid import UUID

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
        directory = self.directory(dataset_id)
        sessions = [p.stem for folder in self.spec(dataset_id).get("subdirectories", ["."])
                    for p in (directory / folder).glob("*.h5")]
        if len(set(sessions)) != len(sessions):
            raise ValueError(f"Duplicate session IDs across source folders: {dataset_id}")
        return sorted(sessions)

    def path(self, dataset_id, session_id):
        session_id = str(session_id)
        spec = self.spec(dataset_id)
        id_format = spec.get("session_id_format", "numeric")
        if id_format == "uuid":
            try:
                valid = str(UUID(session_id)) == session_id
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("IBL session IDs must be canonical UUIDs")
        elif id_format == "numeric":
            if not session_id.isdigit():
                raise ValueError("Allen session IDs must be numeric")
        else:
            raise ValueError(f"Unsupported session ID format: {id_format}")
        directory = self.directory(dataset_id)
        if "subdirectories" not in spec:
            return directory / (session_id + ".h5")
        candidates = [directory / folder / (session_id + ".h5")
                      for folder in spec["subdirectories"]]
        matches = [path for path in candidates if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"Missing {dataset_id} session: {session_id}")
        if len(matches) != 1:
            raise ValueError(f"Duplicate session across source folders: {session_id}")
        return matches[0]

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
            subject_uid = namespace(spec.get("subject_namespace", "allen"), subject_id)
            # A stable sort preserves original order within each probe.
            order = np.argsort(probes, kind="stable")
            rows = []
            for i in order:
                row = {"dataset_id": dataset_id, "session_id": str(session_id),
                       "session_uid": namespace(dataset_id, session_id),
                       "subject_id": subject_id, "subject_uid": subject_uid,
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
            if "stimulus_types" in spec:
                stimulus_types = spec["stimulus_types"]
                if (not isinstance(stimulus_types, list) or not stimulus_types
                        or any(not isinstance(value, str) or not value for value in stimulus_types)):
                    raise ValueError("stimulus_types must be a nonempty list of stimulus labels")
                epochs = f["stimulus_epochs"]
                stimulus_intervals = [(float(a), float(b))
                                      for a, b, label in zip(epochs["start"][:], epochs["end"][:], epochs["type"][:])
                                      if text(label) in stimulus_types]
                # Keep recording gaps and stimulus boundaries: a window must fit in both.
                intervals = sorted((max(a, c), min(b, d))
                                   for a, b in intervals for c, d in stimulus_intervals
                                   if max(a, c) < min(b, d))
            if any(a < previous_end for (_, previous_end), (a, _) in zip(intervals, intervals[1:])):
                raise ValueError("Sampling intervals overlap")
            if "optotagging" in spec:
                exclusion = spec["optotagging"]
                sources = {"interval_path", "intervals_by_session"} & exclusion.keys()
                if len(sources) != 1:
                    raise ValueError("optotagging requires exactly one interval source")
                if "interval_path" in sources:
                    path = exclusion["interval_path"]
                    if path not in f:
                        raise ValueError(f"Missing optotagging intervals: {path}")
                    starts, ends = f[path + "/start"][:], f[path + "/end"][:]
                    if starts.ndim != 1 or ends.shape != starts.shape:
                        raise ValueError("Invalid optotagging interval arrays")
                    excluded = np.column_stack((starts, ends))
                else:
                    by_session = exclusion["intervals_by_session"]
                    if str(session_id) in exclusion.get("end_at_last_stimulus_sessions", []):
                        if str(session_id) in by_session:
                            raise ValueError("Conflicting optotagging timing and stimulus-end fallback")
                        # Allen tagging follows visual stimulation. For explicitly listed
                        # sessions without timings, discard the uncertain recording tail.
                        stimulus_ends = f["stimulus_epochs/end"][:]
                        if not len(stimulus_ends) or not np.isfinite(stimulus_ends).all():
                            raise ValueError("Missing finite stimulus end for optotagging exclusion")
                        cutoff = float(stimulus_ends.max())
                        intervals = [(a, min(b, cutoff)) for a, b in intervals if a < cutoff]
                        excluded = np.empty((0, 2))
                    else:
                        if str(session_id) not in by_session:
                            raise ValueError(f"Missing optotagging intervals for session {session_id}")
                        excluded = np.asarray(by_session[str(session_id)], dtype=float)
                        if excluded.shape == (0,):
                            excluded = excluded.reshape(0, 2)
                if (excluded.ndim != 2 or excluded.shape[1] != 2
                        or not np.isfinite(excluded).all()
                        or np.any(excluded[:, 1] <= excluded[:, 0])):
                    raise ValueError("Optotagging intervals must have finite, increasing bounds")
                # Exclude the complete tagging blocks, including time between pulses.
                # Splitting the domain also constrains the second contrastive view.
                for start, end in excluded:
                    remaining = []
                    for a, b in intervals:
                        if b <= start or a >= end:
                            remaining.append((a, b))
                        else:
                            if a < start:
                                remaining.append((a, float(start)))
                            if end < b:
                                remaining.append((float(end), b))
                    intervals = remaining
            return {"rows": rows, "order": order, "intervals": intervals,
                    "subject_id": subject_id, "subject_uid": subject_uid,
                    "num_spikes": len(f["spikes/timestamps"])}

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
