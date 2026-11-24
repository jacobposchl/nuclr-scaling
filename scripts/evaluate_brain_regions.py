"""Evaluate matched baseline/variant brain-region probes without rerunning extraction."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="artifacts")
    parser.add_argument("--source-comparison", required=True)
    parser.add_argument("--atlas-directory", default="results/cache/ibl_atlas_regions")
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.brain_region_decoding import analyze_brain_regions
    from src.artifacts.store import ArtifactStore
    from src.utils.run_resources import check_free_storage

    check_free_storage(args.store, 25.)
    store = ArtifactStore(args.store)
    aid = analyze_brain_regions(store, args.source_comparison, args.atlas_directory)
    print(f"COMPLETED {store.path('analyses', aid)}", flush=True)


if __name__ == "__main__":
    main()
