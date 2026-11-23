"""Plot exact Ward clustering with a dataset strip for a visual-cortex overview."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("artifacts"))
    parser.add_argument("--source-analysis", required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.store import ArtifactStore
    from src.artifacts.ward_clustering import analyze_ward_clustering

    store = ArtifactStore(args.store)
    analysis = analyze_ward_clustering(store, args.source_analysis)
    print(f"Saved Ward clustering: {store.path('analyses', analysis)}", flush=True)


if __name__ == "__main__":
    main()
