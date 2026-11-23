"""Plot cosine similarity between dataset-by-region mean neuron embeddings."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("artifacts"))
    parser.add_argument("--source-analysis", required=True, help="Visual-cortex Ward analysis ID")
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.region_similarity import analyze_region_similarity
    from src.artifacts.store import ArtifactStore

    store = ArtifactStore(args.store)
    analysis = analyze_region_similarity(store, args.source_analysis)
    print(f"Saved regional similarity: {store.path('analyses', analysis)}", flush=True)


if __name__ == "__main__":
    main()
