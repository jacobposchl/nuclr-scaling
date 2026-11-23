"""Remove firing-rate effects from saved neuron means and compare UMAPs."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("artifacts"))
    parser.add_argument("--source-analysis", required=True, help="Existing complete-cohort UMAP analysis ID")
    args = parser.parse_args()

    # Keep this CPU analysis bounded even when the caller has a multithreaded BLAS default.
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.rate_residualization import analyze_rate_residualization
    from src.artifacts.store import ArtifactStore

    store = ArtifactStore(args.store)
    analysis_id = analyze_rate_residualization(store, args.source_analysis)
    print(f"Saved figures and coordinates: {store.path('analyses', analysis_id)}", flush=True)


if __name__ == "__main__":
    main()
