"""Extract a fixed neuron cohort at each saved epoch and plot VB–VC centroid distance."""

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
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from omegaconf import OmegaConf
    from src.artifacts.checkpoint_centroids import analyze_checkpoint_centroids
    from src.artifacts.collections import extract
    from src.artifacts.store import ArtifactStore, write_json
    from src.datasets.allen_sessions import AllenRegistry

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.run.resolve(strict=True)
    training = json.loads((run / "status.json").read_text())
    if training["state"] != "complete":
        raise ValueError("The requested run must have finished training")
    checkpoints = Path(training["checkpoint_dir"])
    epochs = list(range(1, training["completed_epochs"] + 1))
    for epoch in epochs:
        (checkpoints / f"epoch_{epoch}.pt").resolve(strict=True)
    cfg = OmegaConf.to_container(OmegaConf.load(run / "resolved_config.yaml"), resolve=True)
    manifest = json.loads((run / "split_manifest.json").read_text())
    registry = AllenRegistry(run / "registry.json")
    config = {"datasets": [dict(entry, split="pretrain") for entry in manifest["partitions"]["pretrain"]],
              "device": "cuda", "fingerprint": "stat", "output": "unit_mean",
              "step_seconds": cfg["views"]["duration"], "max_windows_per_session": None}
    output = run / "checkpoint_embeddings"
    output.mkdir(exist_ok=True)
    write_json(output / "collection_config.json", config)
    store = ArtifactStore(args.store)
    started, collections = time.monotonic(), {}
    status = {"run_id": training["run_id"], "epochs": epochs, "collections": {}}

    def update(state, **values):
        status.update(state=state, elapsed_seconds=round(time.monotonic() - started, 2), **values)
        temporary = output / ".status.tmp"
        write_json(temporary, status)
        temporary.replace(output / "status.json")
        print(json.dumps(status), flush=True)

    try:
        # Extract the final checkpoint first so its UMAPs can be fitted while
        # the GPU continues through the earlier checkpoints.
        for epoch in [epochs[-1], *epochs[:-1]]:
            provenance = {"run_id": training["run_id"], "epoch": epoch,
                          "data_provenance": cfg["data_provenance"]}
            checkpoint = store.register_checkpoint(checkpoints / f"epoch_{epoch}.pt", provenance,
                                                   f"{training['run_id']}: epoch {epoch}")
            update("extracting", current_epoch=epoch)
            collection = extract(store, checkpoint, registry, config,
                                 f"Epoch {epoch}: all regions, VC natural scenes and VB passive replay")
            collections[epoch] = collection
            status["collections"][str(epoch)] = collection
            update("extracted", current_epoch=epoch)
        update("plotting_centroid_distance")
        analysis = analyze_checkpoint_centroids(store, collections)
        update("complete", centroid_analysis_id=analysis,
               output_directory=str(store.path("analyses", analysis)))
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
