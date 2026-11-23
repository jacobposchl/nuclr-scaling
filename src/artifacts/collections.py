"""Window-level extraction and reusable unit means on CPU or an explicit GPU."""

import json
from pathlib import Path
import time

import h5py
import numpy as np

from .store import implementation_stamp, write_json
from src.datasets.allen_sessions import windows


def expand_selection(registry, selection):
    result, seen = [], set()
    for entry in selection:
        dataset = entry["dataset_id"]
        sessions = registry.sessions(dataset) if entry["sessions"] == "all" else entry["sessions"]
        if not isinstance(sessions, list) or not sessions:
            raise ValueError("Each dataset needs an explicit nonempty session list or 'all'")
        for session in sorted(str(s) for s in sessions):
            key = dataset, session
            if key in seen:
                raise ValueError(f"Duplicate session selection: {key}")
            seen.add(key)
            registry.path(*key).resolve(strict=True)
            result.append({"dataset_id": dataset, "session_id": session,
                           "split": entry.get("split", "unspecified")})
    if not result:
        raise ValueError("Empty extraction selection")
    return sorted(result, key=lambda r: (r["dataset_id"], r["session_id"]))


def extract(store, checkpoint_id, registry, config, label=""):
    """Use float32 on CPU by default, or bf16 inference with device='cuda'."""
    import torch
    from torch_brain.batching import collate
    from src.models.unit_encoders.nuclr import NuCLR

    allowed = {"datasets", "step_seconds", "max_windows_per_session", "output", "fingerprint", "device"}
    if set(config) - allowed:
        raise ValueError(f"Unknown extraction options: {set(config) - allowed}")
    mode = config.get("output", "unit_mean")
    if mode not in ("unit_mean", "windows", "both"):
        raise ValueError("output must be unit_mean, windows, or both")
    device = config.get("device", "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; select a GPU with CUDA_VISIBLE_DEVICES")
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA extraction requires an available GPU with bf16 support")
    checkpoint_path = store.checkpoint_path(checkpoint_id)
    model = NuCLR.from_pretrained(checkpoint_path).to(device).float().eval()
    selection = expand_selection(registry, config["datasets"])
    sources = [registry.fingerprint(dataset, [s["session_id"] for s in selection if s["dataset_id"] == dataset],
                                    config.get("fingerprint", "stat"))
               for dataset in sorted({s["dataset_id"] for s in selection})]
    step = config.get("step_seconds", model.ctx_duration)
    limit = config.get("max_windows_per_session")
    # Validate even when a cache hit would otherwise avoid window enumeration.
    list(windows([], model.ctx_duration, step, limit))
    recipe = {"operation": "extract_v1", "checkpoint_id": checkpoint_id,
              "sources": sources, "selection": selection,
              "window": {"duration_seconds": model.ctx_duration, "step_seconds": step,
                         "max_windows_per_session": limit, "order": "chronological"},
              "population": "all_source_units_grouped_by_probe_no_dropout",
              "representation": "encoder", "normalization": "none", "output": mode,
              "device": device, "dtype": "bfloat16" if device == "cuda" else "float32",
              "implementation": implementation_stamp()}
    with store.build("collections", recipe, label, [{"kind": "checkpoints", "id": checkpoint_id}]) as (stage, artifact_id):
        if stage is None:
            return artifact_id
        summary = []
        started = time.monotonic()
        for session_number, selected in enumerate(selection):
            dataset, session = selected["dataset_id"], selected["session_id"]
            desc = registry.describe(dataset, session)
            if len(desc["rows"]) < 2:
                raise ValueError(f"Fewer than two units in {dataset}/{session}")
            bounds = list(windows(desc["intervals"], model.ctx_duration, step, limit))
            if not bounds:
                raise ValueError(f"No complete extraction windows in {dataset}/{session}")
            shard = stage / f"session-{session_number:05d}"
            shard.mkdir()
            rows = [dict(row, split=selected["split"]) for row in desc["rows"]]
            write_json(shard / "windows.json", [{"index": i, "start": a, "end": b}
                                               for i, (a, b) in enumerate(bounds)])
            total = np.zeros((len(rows), model.emb_dim), dtype=np.float64)
            spike_counts = np.zeros(len(rows), dtype=np.int64)
            with h5py.File(shard / "embeddings.h5", "w") as out:
                if mode in ("windows", "both"):
                    window_array = out.create_dataset(
                        "windows", shape=(len(bounds), len(rows), model.emb_dim), dtype="float32",
                        chunks=(1, min(len(rows), 256), model.emb_dim), compression="lzf")
                    window_rates = out.create_dataset("window_firing_rate_hz", shape=(len(bounds), len(rows)), dtype="float64")
                for index, (start, end) in enumerate(bounds):
                    data = registry.window(dataset, session, start, end, desc)
                    counts = np.bincount(data.spikes.unit_index, minlength=len(rows))
                    spike_counts += counts
                    tokens = collate([model.tokenize(data)])
                    tokens = {key: value.to(device) for key, value in tokens.items()}
                    with torch.inference_mode(), torch.autocast(
                            device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                        z = model(**tokens).float().cpu().numpy()
                    if not np.isfinite(z).all():
                        raise ValueError(f"Nonfinite embeddings in {dataset}/{session}")
                    total += z
                    if mode in ("windows", "both"):
                        window_array[index] = z
                        window_rates[index] = counts / (end - start)
                if mode in ("unit_mean", "both"):
                    out.create_dataset("unit_mean", data=(total / len(bounds)).astype(np.float32))
                out.create_dataset("observation_count", data=np.full(len(rows), len(bounds), dtype=np.int64))
            for row, count in zip(rows, spike_counts):
                row["firing_rate_hz"] = float(count / (len(bounds) * model.ctx_duration))
            write_json(shard / "units.json", rows)
            summary.append({**selected, "shard": shard.name, "units": len(rows), "windows": len(bounds)})
            print(f"EXTRACT session={session_number + 1}/{len(selection)} {dataset}/{session} "
                  f"units={len(rows)} windows={len(bounds)} elapsed_minutes={(time.monotonic() - started) / 60:.1f}",
                  flush=True)
        # Do not publish if source files were replaced during extraction.
        after = [registry.fingerprint(s["dataset_id"], [f["session_id"] for f in s["files"]], s["fingerprint_mode"])
                 for s in sources]
        if sources != after:
            raise RuntimeError("Source data changed during extraction")
        if store.checkpoint_path(checkpoint_id) != checkpoint_path:
            raise RuntimeError("Checkpoint reference changed during extraction")
        write_json(stage / "index.json", {"sessions": summary, "dimension": model.emb_dim})
    return artifact_id


def load_unit_means(store, collection_ids, max_rows=100000, row_filter=None):
    """Load a bounded analysis matrix. Large analyses must choose a smaller cohort."""
    arrays, rows, keys = [], [], set()
    checkpoints = set()
    for collection_id in collection_ids:
        manifest = store.read("collections", collection_id)
        lineage = manifest["recipe"]
        while "parent_collection" in lineage:
            lineage = store.read("collections", lineage["parent_collection"])["recipe"]
        checkpoints.add(lineage["checkpoint_id"])
        if len(checkpoints) > 1:
            raise ValueError("Do not pool coordinate spaces from different checkpoints; run separate analyses")
        folder = store.path("collections", collection_id)
        index = json.loads((folder / "index.json").read_text())
        for item in index["sessions"]:
            shard = folder / item["shard"]
            metadata = json.loads((shard / "units.json").read_text())
            mask = np.ones(len(metadata), dtype=bool) if row_filter is None else np.asarray([row_filter(row) for row in metadata], dtype=bool)
            if mask.shape != (len(metadata),):
                raise ValueError("Row filter must return one boolean per metadata row")
            selected = np.flatnonzero(mask)
            if len(rows) + len(selected) > max_rows:
                raise ValueError("Analysis exceeds max_rows; use a smaller collection or raise the explicit limit")
            with h5py.File(shard / "embeddings.h5", "r") as f:
                if "unit_mean" not in f:
                    raise ValueError("Collection contains windows only; derive unit means first")
                if f["unit_mean"].shape[0] != len(metadata):
                    raise ValueError("Embedding matrix and metadata lengths differ")
                if not len(selected):
                    continue
                values = f["unit_mean"][:] if mask.all() else f["unit_mean"][selected]
            metadata = [metadata[i] for i in selected]
            if values.shape[0] != len(metadata) or not np.isfinite(values).all():
                raise ValueError("Invalid embedding matrix or row metadata")
            for row in metadata:
                if row["unit_uid"] in keys:
                    raise ValueError("Overlapping unit IDs across collections; compare models in separate analyses")
                keys.add(row["unit_uid"])
                rows.append(dict(row, collection_id=collection_id))
            arrays.append(values)
    if not arrays:
        raise ValueError("No embeddings selected")
    if len({a.shape[1] for a in arrays}) != 1:
        raise ValueError("Collections have different embedding dimensions")
    return np.concatenate(arrays), rows


def derive_means(store, collection_id, window_indices=None, label=""):
    store.read("collections", collection_id)
    folder = store.path("collections", collection_id)
    index = json.loads((folder / "index.json").read_text())
    recipe = {"operation": "mean_v1", "parent_collection": collection_id,
              "window_indices": window_indices, "implementation": implementation_stamp()}
    if window_indices is not None:
        if not window_indices or len(set(window_indices)) != len(window_indices) or any(
                not isinstance(i, int) or i < 0 for i in window_indices):
            raise ValueError("Window indices must be unique nonnegative integers")
        window_indices = sorted(window_indices)
        recipe["window_indices"] = window_indices
    with store.build("collections", recipe, label, [{"kind": "collections", "id": collection_id}]) as (stage, artifact_id):
        if stage is None:
            return artifact_id
        summary = []
        for item in index["sessions"]:
            source = folder / item["shard"]
            destination = stage / item["shard"]
            destination.mkdir()
            metadata = json.loads((source / "units.json").read_text())
            bounds = json.loads((source / "windows.json").read_text())
            with h5py.File(source / "embeddings.h5", "r") as f:
                if "windows" not in f:
                    raise ValueError("Parent collection did not retain window embeddings")
                selected = list(range(len(bounds))) if window_indices is None else window_indices
                if max(selected) >= len(bounds):
                    raise ValueError("Requested window is not present in every selected session")
                total = np.zeros(f["windows"].shape[1:], dtype=np.float64)
                rates = np.zeros(len(metadata), dtype=np.float64) if "window_firing_rate_hz" in f else None
                for i in selected:
                    total += f["windows"][i]
                    if rates is not None:
                        rates += f["window_firing_rate_hz"][i]
            with h5py.File(destination / "embeddings.h5", "w") as f:
                f.create_dataset("unit_mean", data=(total / len(selected)).astype(np.float32))
                f.create_dataset("observation_count", data=np.full(len(metadata), len(selected), dtype=np.int64))
            for i, row in enumerate(metadata):
                # Legacy collections have no per-window rates: never retain a
                # rate computed over a different selection of windows.
                row["firing_rate_hz"] = None if rates is None else float(rates[i] / len(selected))
            write_json(destination / "units.json", metadata)
            write_json(destination / "windows.json", [bounds[i] for i in selected])
            summary.append(dict(item, windows=len(selected)))
        write_json(stage / "index.json", dict(index, sessions=summary))
    return artifact_id
