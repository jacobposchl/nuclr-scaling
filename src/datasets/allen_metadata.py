"""Restore Visual Coding stimulus epochs from metadata in Allen's public NWBs."""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .allen_sessions import text


class RemoteMetadata(io.RawIOBase):
    """Seekable, bounded HTTP range reader; never download an entire NWB."""

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.http = requests.Session()
        self.http.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))
        self.position = 0
        self.blocks = {}
        self.block_size = 262144
        self.transferred = 0
        self.size = None
        self._block(0)

    def _block(self, index):
        if index not in self.blocks:
            if self.transferred >= 32 * 1024 * 1024:
                raise RuntimeError("Metadata-only download limit exceeded")
            start = index * self.block_size
            end = start + self.block_size - 1
            if self.size is not None:
                end = min(end, self.size - 1)
            with self.http.get(self.url, headers={"Range": f"bytes={start}-{end}",
                                                 "Accept-Encoding": "identity"},
                               timeout=30, stream=True) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(f"bytes {start}-"):
                    raise RuntimeError("Server did not honor metadata-only byte range")
                self.size = int(content_range.split("/")[-1])
                end = min(end, self.size - 1)
                data = response.raw.read(end - start + 1)
                if len(data) != end - start + 1:
                    raise RuntimeError("Truncated metadata byte range")
            self.transferred += len(data)
            self.blocks[index] = data
        return self.blocks[index]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def readinto(self, target):
        count = max(0, min(len(target), self.size - self.position))
        done = 0
        while done < count:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            n = min(count - done, len(block) - offset)
            target[done:done + n] = block[offset:offset + n]
            done += n
            self.position += n
        return done

    def close(self):
        self.http.close()
        super().close()


def source_urls(sessions):
    criteria = ("model::WellKnownFile,rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
                "[attachable_type$eq'EcephysSession'][attachable_id$in" + ",".join(sessions)
                + "],rma::options[num_rows$eq'all']")
    response = requests.get("https://api.brain-map.org/api/v2/data/query.json",
                            params={"criteria": criteria}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError("Allen source metadata query failed")
    result = {str(row["attachable_id"]): "https://api.brain-map.org" + row["download_link"]
              for row in payload["msg"]}
    if set(result) != set(sessions) or len(payload["msg"]) != len(sessions):
        raise ValueError("Expected exactly one source NWB for every local session")
    return result


def local_stimulus_digest(local):
    """Tie cached epochs to the existing local stimulus timestamps."""
    digest = hashlib.sha256()
    for name in sorted(local):
        group = local[name]
        if isinstance(group, h5py.Group) and "timestamps" in group and "start" in group and "end" in group:
            digest.update(name.encode())
            for key in ("start", "end"):
                values = np.asarray(group[key][:], dtype="<f8")
                digest.update(key.encode())
                digest.update(str(values.shape).encode())
                digest.update(values.tobytes())
    return digest.hexdigest()


def validate_epochs(epochs):
    if not epochs:
        raise ValueError("Missing stimulus epochs")
    previous_end = -np.inf
    for epoch in epochs:
        start, end, label = epoch["start"], epoch["end"], epoch["type"]
        if (not np.isfinite([start, end]).all() or end <= start or start < previous_end - 1e-8
                or not isinstance(label, str) or not label):
            raise ValueError("Stimulus epochs must be finite, ordered, and nonoverlapping")
        previous_end = end


def read_stimulus_epochs(source, local, session_id):
    """Aggregate native stimulus blocks, preserving each spontaneous interval.

    These describe the scheduled stimuli, including source-marked invalid
    presentations. They do not change the local recording/sampling domain.
    """
    if text(source["identifier"][()]) != session_id or text(local["session"].attrs["id"]) != session_id:
        raise ValueError("Source and local session identifiers do not match")
    invalid_stimulus_intervals = []
    if "invalid_times" in source["intervals"]:
        invalid = source["intervals/invalid_times"]
        tags = [text(value) for value in invalid["tags"][:]]
        offsets = np.r_[0, invalid["tags_index"][:]]
        for i, (start, end) in enumerate(zip(invalid["start_time"][:], invalid["stop_time"][:])):
            row_tags = set(tags[int(offsets[i]):int(offsets[i + 1])])
            if {"EcephysSession", session_id, "stimulus"} <= row_tags:
                invalid_stimulus_intervals.append((start, end))
    epochs, checked, anomalies = [], set(), []
    for name, table in source["intervals"].items():
        if not name.endswith("_presentations"):
            continue
        label = name.removesuffix("_presentations")
        starts, ends = table["start_time"][:], table["stop_time"][:]
        labels = np.array([text(value) for value in table["stimulus_name"][:]])
        if (starts.ndim != 1 or starts.shape != ends.shape or starts.shape != labels.shape
                or not len(starts) or not np.isfinite(starts).all() or not np.isfinite(ends).all()
                or np.any(labels != label)):
            raise ValueError(f"Invalid source presentation table: {name}")
        bad_rows = np.flatnonzero((ends <= starts) | np.r_[False, np.diff(starts) < 0])
        if len(bad_rows):
            for i in bad_rows:
                if not any(a <= min(starts[i], ends[i]) and max(starts[i], ends[i]) <= b
                           for a, b in invalid_stimulus_intervals):
                    raise ValueError(f"Unexplained source timing anomaly in {name}, row {i}")
            anomalies.append({"stimulus": label, "rows": bad_rows.tolist(),
                              "reason": "Source-marked invalid stimulus timing; block envelope retained"})
        local_label = "drifting_gratings" if label == "drifting_gratings_75_repeats" else label
        if local_label in local and "start" in local[local_label] and "end" in local[local_label]:
            # Preprocessing omits some presentations. Every retained local row
            # must match a source start/end pair; epoch bounds use the full source.
            existing_starts, existing_ends = local[local_label]["start"][:], local[local_label]["end"][:]
            order = np.argsort(starts, kind="stable")
            positions = np.searchsorted(starts[order], existing_starts)
            if np.any(positions >= len(starts)):
                raise ValueError(f"Source/local stimulus timestamps differ: {label}")
            indices = order[positions]
            if (existing_starts.shape != existing_ends.shape or np.any(indices >= len(starts))
                    or not np.allclose(existing_starts, starts[indices], rtol=0, atol=1e-8)
                    or not np.allclose(existing_ends, ends[indices], rtol=0, atol=1e-8)):
                raise ValueError(f"Source/local stimulus timestamps differ: {label}")
            checked.add(local_label)
        if "stimulus_block" in table:
            blocks = table["stimulus_block"][:]
            if blocks.shape != starts.shape or not np.isfinite(blocks).all():
                raise ValueError(f"Invalid source block IDs: {name}")
            boundaries = np.r_[0, np.flatnonzero(np.diff(blocks) != 0) + 1, len(blocks)]
        elif label == "spontaneous":
            blocks = np.full(len(starts), -1.)
            boundaries = np.arange(len(starts) + 1)
        else:
            raise ValueError(f"Missing source block IDs: {name}")
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            epochs.append({"start": float(starts[left:right].min()), "end": float(ends[left:right].max()),
                           "type": label, "stimulus_block": float(blocks[left])})
    for name in ("gabors", "drifting_gratings", "static_gratings", "natural_scenes"):
        if name in local and len(local[name]["start"]) and name not in checked:
            raise ValueError(f"Local stimulus has no checked source table: {name}")
    if not checked:
        raise ValueError("No local stimulus timestamps were verified against the source")
    epochs.sort(key=lambda epoch: epoch["start"])
    validate_epochs(epochs)
    return epochs, sorted(checked), anomalies


def prepare_stimulus_epochs(path, url, cache_directory):
    path, cache_directory = Path(path), Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"{path.stem}.json"
    with h5py.File(path, "r") as local:
        digest = local_stimulus_digest(local)
        if cache_path.exists():
            record = json.loads(cache_path.read_text())
            if (record["session_id"] != path.stem or record["source_url"] != url
                    or record["local_stimulus_sha256"] != digest):
                raise ValueError(f"Cached stimulus metadata does not match {path.name}")
            validate_epochs(record["epochs"])
            return record
        with RemoteMetadata(url) as remote, h5py.File(remote, "r") as source:
            epochs, checked, anomalies = read_stimulus_epochs(source, local, path.stem)
            record = {"session_id": path.stem, "source_url": url,
                      "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                      "source_nwb_bytes": remote.size, "downloaded_metadata_bytes": remote.transferred,
                      "local_stimulus_sha256": digest, "verified_stimulus_tables": checked,
                      "epochs": epochs, "source_timing_anomalies": anomalies}
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(cache_path)
    return record


def restore_stimulus_epochs(path, record):
    """Add only the missing epoch group; refuse to replace existing annotations."""
    path = Path(path)
    epochs = record["epochs"]
    validate_epochs(epochs)
    arrays = {key: np.array([row[key] for row in epochs], dtype="S" if key == "type" else float)
              for key in ("start", "end", "type", "stimulus_block")}
    with h5py.File(path, "r+") as local:
        if (record["session_id"] != path.stem or text(local["session"].attrs["id"]) != path.stem
                or local_stimulus_digest(local) != record["local_stimulus_sha256"]):
            raise ValueError("Local stimulus metadata changed after source verification")
        if "stimulus_epochs" in local:
            group = local["stimulus_epochs"]
            if (group.attrs.get("source_url") != record["source_url"]
                    or any(key not in group or not np.array_equal(group[key][:], values)
                           for key, values in arrays.items())):
                raise ValueError("Refusing to replace existing stimulus epochs")
            return False
        pending = "_restoring_stimulus_epochs"
        if pending in local:
            raise ValueError("An incomplete epoch restoration needs inspection")
        group = local.create_group(pending)
        try:
            for key, values in arrays.items():
                group.create_dataset(key, data=values)
            group.attrs.update({"object": "Interval", "timekeys": np.array(["start", "end"], dtype="S"),
                                "_unicode_keys": np.array(["type"], dtype="S"),
                                "allow_split_mask_overlap": False,
                                "source_url": record["source_url"], "source_session_id": path.stem,
                                "retrieved_utc": record["retrieved_utc"],
                                "local_stimulus_sha256": record["local_stimulus_sha256"],
                                "method": "Native NWB stimulus blocks: minimum start to maximum stop; "
                                          "each spontaneous interval retained; no invalid-time masking; "
                                          "stimulus_block=-1 for spontaneous periods without a source block ID."})
            if record.get("source_timing_anomalies"):
                group.attrs["source_timing_anomalies"] = json.dumps(record["source_timing_anomalies"])
            local.flush()
            local.move(pending, "stimulus_epochs")
            local.flush()
        except Exception:
            if pending in local:
                del local[pending]
            raise
    return True
