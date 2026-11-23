"""Pool the final checkpoint over its full training domains and plot neuron UMAPs."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("artifacts"))
    parser.add_argument("--coding-labels", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from omegaconf import OmegaConf
    from src.artifacts.collections import extract
    from src.artifacts.embedding_overview import analyze_embedding_overview
    from src.artifacts.store import ArtifactStore, write_json
    from src.artifacts.visualization import visualize
    from src.datasets.allen_sessions import AllenRegistry

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.run.resolve(strict=True)
    training = json.loads((run / "status.json").read_text())
    if training["state"] != "complete":
        raise ValueError("The requested run must have finished training")
    labels = args.coding_labels.resolve(strict=True)
    epoch = training["completed_epochs"]
    checkpoint_path = Path(training["checkpoint_dir"]) / f"epoch_{epoch}.pt"
    cfg = OmegaConf.to_container(OmegaConf.load(run / "resolved_config.yaml"), resolve=True)
    manifest = json.loads((run / "split_manifest.json").read_text())
    registry = AllenRegistry(run / "registry.json")
    config = {"datasets": [dict(entry, split="pretrain") for entry in manifest["partitions"]["pretrain"]],
              "device": "cuda", "fingerprint": "stat", "output": "unit_mean",
              "step_seconds": cfg["views"]["duration"], "max_windows_per_session": None}
    output = run / "embedding_overviews"
    output.mkdir(exist_ok=True)
    write_json(output / "collection_config.json", config)
    store = ArtifactStore(args.store)
    started = time.monotonic()
    status = {"run_id": training["run_id"], "epoch": epoch}

    def update(state, **values):
        status.update(state=state, elapsed_seconds=round(time.monotonic() - started, 2), **values)
        temporary = output / ".status.tmp"
        write_json(temporary, status)
        temporary.replace(output / "status.json")
        print(json.dumps(status), flush=True)

    try:
        checkpoint = store.register_checkpoint(checkpoint_path,
            {"run_id": training["run_id"], "epoch": epoch, "data_provenance": cfg["data_provenance"]},
            f"{training['run_id']}: epoch {epoch}")
        update("extracting", checkpoint_id=checkpoint)
        collection = extract(store, checkpoint, registry, config,
                             f"{training['run_id']} epoch {epoch}: full non-optotagging domains")
        index = json.loads((store.path("collections", collection) / "index.json").read_text())
        total_neurons = sum(item["units"] for item in index["sessions"])
        visualization_config = {"methods": ["umap"], "fit_selection": {}, "show_selection": {},
            "standardize": False, "seed": 42, "max_rows": total_neurons,
            "umap": {"n_neighbors": 15, "min_dist": .1, "metric": "euclidean"},
            "colors": [{"field": "dataset_id", "type": "categorical", "palette": "Set1"}]}
        write_json(output / "visualization_config.json", visualization_config)
        update("fitting_joint_umap", collection_id=collection, neurons=total_neurons)
        joint = visualize(store, [collection], visualization_config,
                          f"{training['run_id']} epoch {epoch}: combined UMAP")
        update("fitting_dataset_umaps", joint_analysis_id=joint)
        analysis = analyze_embedding_overview(store, joint, labels_path=labels)
        update("complete", overview_analysis_id=analysis,
               output_directory=str(store.path("analyses", analysis).resolve()))
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
