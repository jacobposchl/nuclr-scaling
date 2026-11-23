"""Compare full non-optotagging sampling domains and neurons per probe."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allen-registry", type=Path, default=Path("configs/data_registry.json"))
    parser.add_argument("--joint-registry", type=Path,
                        default=Path("configs/registries/allen_passive_natural_scenes_ibl.json"),
                        help="Used only for the IBL source paths and supplied session partitions")
    parser.add_argument("--window-seconds", type=float, default=20.)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.artifacts.sampling_distributions import collect_sampling_metadata, save_sampling_distributions

    recordings, probes = collect_sampling_metadata(args.allen_registry, args.joint_registry,
                                                    args.window_seconds)
    summary = save_sampling_distributions(recordings, probes, args.output_dir)
    for row in summary:
        print(f"{row['dataset_id']} {row['measurement']}: n={row['count']}, "
              f"mean={row['mean']:.2f}, median={row['median']:.2f}, "
              f"IQR={row['q25']:.2f}–{row['q75']:.2f}, "
              f"range={row['minimum']:.2f}–{row['maximum']:.2f}")
    print(f"Saved figure and CSV tables: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
