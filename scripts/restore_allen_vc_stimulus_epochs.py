"""Recover missing Visual Coding stimulus epochs without downloading spike data."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/data_registry.json")
    parser.add_argument("--cache", type=Path, default=Path("results/cache/allen_vc_stimulus_epochs"))
    parser.add_argument("--apply", action="store_true", help="Add verified epochs to the local HDF5 files")
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.datasets.allen_metadata import prepare_stimulus_epochs, restore_stimulus_epochs, source_urls
    from src.datasets.allen_sessions import AllenRegistry, windows

    registry = AllenRegistry(args.registry)
    dataset = "allen_visual_coding"
    sessions = registry.sessions(dataset)
    if not sessions:
        raise ValueError("No local Visual Coding sessions found")
    urls = source_urls(sessions)
    records, errors = {}, []
    # Concurrent network I/O; numerical work stays on one CPU thread.
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(prepare_stimulus_epochs, registry.path(dataset, sid), urls[sid], args.cache): sid
                   for sid in sessions}
        for future in as_completed(pending):
            sid = pending[future]
            try:
                records[sid] = future.result()
                print(f"Verified source metadata: {len(records)}/{len(sessions)} ({sid})", flush=True)
            except Exception as error:
                errors.append(f"{sid}: {error}")
                print(f"Failed: {errors[-1]}", flush=True)
    if errors:
        raise RuntimeError("No HDF5 files modified; source verification failed:\n" + "\n".join(errors))
    if not args.apply:
        print(f"All {len(records)} sessions verified and cached. Use --apply to restore the epoch groups.")
        return
    rows, restored = [], 0
    for sid in sessions:
        record = records[sid]
        restored += restore_stimulus_epochs(registry.path(dataset, sid), record)
        desc = registry.describe(dataset, sid)
        rows.append({"session_id": sid, "epochs": len(record["epochs"]),
                     "stimulus_types": ";".join(sorted({e["type"] for e in record["epochs"]})),
                     "available_seconds": sum(b - a for a, b in desc["intervals"]),
                     "available_20s_windows": len(list(windows(desc["intervals"], 20, 20))),
                     "opto_policy": "last_visual_stimulus" if sid in registry.spec(dataset)["optotagging"]
                                    .get("end_at_last_stimulus_sessions", []) else "verified_opto_block",
                     "source_url": record["source_url"]})
    output = Path("results/data_validation/allen_all_regions/restored_vc_stimulus_epochs.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Restored {restored} files; all {len(sessions)} VC sessions load. Audit: {output}")


if __name__ == "__main__":
    main()
