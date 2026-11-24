"""Compare cross-dataset biological neighborhoods using saved embeddings."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', default='artifacts')
    parser.add_argument('--source-comparison', required=True)
    parser.add_argument('--atlas-directory', default='results/cache/ibl_atlas_regions')
    parser.add_argument('--variant-epoch', type=int, choices=[5, 10], default=10)
    args = parser.parse_args()
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.biological_neighbors import analyze_biological_neighbors
    from src.artifacts.store import ArtifactStore
    from src.utils.run_resources import check_free_storage
    check_free_storage(args.store, 25.)
    store = ArtifactStore(args.store)
    aid = analyze_biological_neighbors(store, args.source_comparison, args.atlas_directory, variant_epoch=args.variant_epoch)
    print(f"COMPLETED {store.path('analyses', aid)}", flush=True)


if __name__ == '__main__':
    main()
