"""Fit firing-rate probes and quantify the direction fraction for every neuron."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("artifacts"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collection", help="Pooled collection for --sequential-directions")
    source.add_argument("--source-analysis",
                        help="PCA analysis for probes; probe analysis for --remove-direction; removed UMAP analysis for --removed-pca")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare-datasets", action="store_true",
                      help="Compare separately fitted VB/VC directions; source is the original probe analysis")
    mode.add_argument("--sequential-directions", action="store_true",
                      help="Plot sequential individual-direction R2 and variance retained; source is the two-direction comparison")
    mode.add_argument("--remove-second-direction", action="store_true",
                      help="Remove two sequential directions; source is a one-direction probe comparison")
    mode.add_argument("--reprobe-removed", action="store_true",
                      help="Compare linear and nonlinear raw-Hz probes after direction removal; source is a probe analysis")
    mode.add_argument("--remove-direction", action="store_true",
                        help="Remove the saved raw-Hz direction and fit joint and dataset UMAPs")
    mode.add_argument("--removed-pca", action="store_true",
                      help="Fit six PCs after the saved direction removal, with PC1/2, PC3/4, and PC5/6 panels")
    args = parser.parse_args()
    if args.collection and not args.sequential_directions:
        parser.error("--collection requires --sequential-directions")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.firing_rate_probe import (
        analyze_firing_rate_probes, analyze_direction_removed_umaps, analyze_direction_removed_pca,
    )
    from src.artifacts.store import ArtifactStore
    from src.artifacts.direction_removed_probes import analyze_removed_probes, analyze_second_direction_removal
    from src.artifacts.sequential_rate_directions import analyze_sequential_directions
    from src.artifacts.dataset_rate_probes import analyze_dataset_probes

    store = ArtifactStore(args.store)
    analyze = (analyze_dataset_probes if args.compare_datasets else
               analyze_sequential_directions if args.sequential_directions else
               analyze_second_direction_removal if args.remove_second_direction else
               analyze_removed_probes if args.reprobe_removed else
               analyze_direction_removed_pca if args.removed_pca else
               analyze_direction_removed_umaps if args.remove_direction else analyze_firing_rate_probes)
    analysis_id = (analyze_sequential_directions(store, collection_id=args.collection) if args.collection
                   else analyze(store, args.source_analysis))
    print(f"Saved firing-rate direction analysis: {store.path('analyses', analysis_id)}", flush=True)


if __name__ == "__main__":
    main()
