"""Compare a baseline with two rate-separated epochs after excluding zero-rate neurons."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="artifacts")
    parser.add_argument("--baseline-overview", required=True)
    parser.add_argument("--epoch5-overview", required=True)
    parser.add_argument("--epoch10-overview", required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.rate_variant_comparison import analyze_rate_variant_comparison
    from src.artifacts.store import ArtifactStore
    from src.utils.run_resources import check_free_storage

    check_free_storage(args.store, 25.)
    store = ArtifactStore(args.store)
    aid = analyze_rate_variant_comparison(store, {"Baseline epoch 10": args.baseline_overview,
        "Variant epoch 5": args.epoch5_overview, "Variant epoch 10": args.epoch10_overview})
    print(f"COMPLETED {store.path('analyses', aid)}", flush=True)


if __name__ == "__main__":
    main()
