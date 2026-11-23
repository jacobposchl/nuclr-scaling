"""Create UMAP or combined PCA figures with multiple metadata panels."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("artifacts"))
    parser.add_argument("--source-analysis", required=True, help="Joint UMAP ID, or saved overview ID for recoloring/PCA")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--coding-labels", type=Path, help="LOLCAT annotation CSV; reuses saved overview coordinates")
    mode.add_argument("--pca", action="store_true", help="Create combined PC1/PC2 panels using the overview's annotations")
    mode.add_argument("--visual-cortex", action="store_true", help="Refit joint UMAP using only visual-cortex neurons")
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.embedding_overview import (
        analyze_combined_pca_overview, analyze_embedding_overview, recolor_embedding_overview,
        analyze_visual_cortex_overview,
    )
    from src.artifacts.store import ArtifactStore

    store = ArtifactStore(args.store)
    if args.visual_cortex:
        analysis_id = analyze_visual_cortex_overview(store, args.source_analysis)
    elif args.pca:
        analysis_id = analyze_combined_pca_overview(store, args.source_analysis)
    elif args.coding_labels:
        analysis_id = recolor_embedding_overview(store, args.source_analysis, args.coding_labels)
    else:
        analysis_id = analyze_embedding_overview(store, args.source_analysis)
    print(f"Saved figures and coordinates: {store.path('analyses', analysis_id)}", flush=True)


if __name__ == "__main__":
    main()
