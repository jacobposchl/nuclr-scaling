"""Compare dataset centroids across checkpoints using a fixed pooling cohort."""

import csv
import json

import numpy as np

from .collections import load_unit_means
from .store import digest, file_digest, implementation_stamp, write_json


def centroid_distance(embeddings, datasets):
    """Euclidean distance between neuron-weighted VB and VC mean embeddings."""
    embeddings = np.asarray(embeddings)
    datasets = np.asarray(datasets)
    expected = {"allen_visual_behavior", "allen_visual_coding"}
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(datasets)
            or embeddings.shape[1] == 0 or set(datasets) != expected
            or not np.isfinite(embeddings).all()):
        raise ValueError("Expected finite embeddings with matching VB and VC dataset labels")
    centroids = {dataset: embeddings[datasets == dataset].mean(axis=0, dtype=np.float64)
                 for dataset in sorted(expected)}
    vb, vc = centroids["allen_visual_behavior"], centroids["allen_visual_coding"]
    return {"centroid_distance": float(np.linalg.norm(vb - vc)),
            "visual_behavior_neurons": int(np.sum(datasets == "allen_visual_behavior")),
            "visual_coding_neurons": int(np.sum(datasets == "allen_visual_coding"))}


def pooling_identity(store, collection_id, rows):
    """Compare actual windows, source populations, and extraction settings."""
    manifest = store.read("collections", collection_id, verify=False)
    recipe = manifest["recipe"]
    if recipe["operation"] != "extract_v1":
        raise ValueError("Centroid trajectories require directly extracted pooled collections")
    folder = store.path("collections", collection_id)
    index = json.loads((folder / "index.json").read_text())
    return digest({
        "extraction": {key: recipe[key] for key in ("sources", "selection", "window", "population",
                                                   "representation", "normalization", "device", "dtype")},
        "units": [(row["unit_uid"], row["dataset_id"]) for row in rows],
        "windows": [(item["dataset_id"], item["session_id"],
                     file_digest(folder / item["shard"] / "windows.json")) for item in index["sessions"]],
    })


def analyze_checkpoint_centroids(store, epoch_collections):
    epochs = sorted(epoch_collections)
    if not epochs or any(not isinstance(epoch, int) or epoch < 1 for epoch in epochs):
        raise ValueError("Provide positive integer epochs and their extracted collections")
    if len(set(epoch_collections.values())) != len(epochs):
        raise ValueError("Each epoch must use a distinct checkpoint collection")
    recipe = {"operation": "dataset_centroid_trajectory_v1",
              "epochs": [{"epoch": epoch, "collection_id": epoch_collections[epoch]} for epoch in epochs],
              "metric": "euclidean_distance_between_dataset_mean_embeddings",
              "weighting": "equal_per_neuron_within_dataset", "normalization": "none",
              "space": "original_encoder_embedding", "implementation": implementation_stamp()}
    parents = [{"kind": "collections", "id": epoch_collections[epoch]} for epoch in epochs]
    with store.build("analyses", recipe, "VB–VC centroid distance across epochs", parents) as (stage, aid):
        if stage is None:
            return aid
        measurements, reference = [], None
        for epoch in epochs:
            collection_id = epoch_collections[epoch]
            embeddings, rows = load_unit_means(store, [collection_id])
            identity = pooling_identity(store, collection_id, rows)
            if reference is not None and identity != reference:
                raise ValueError("Epochs use different neurons, windows, or extraction settings")
            reference = identity
            measurement = centroid_distance(embeddings, [row["dataset_id"] for row in rows])
            measurements.append({"epoch": epoch, **measurement, "collection_id": collection_id})
            print(f"CENTROIDS epoch={epoch} distance={measurement['centroid_distance']:.6f}", flush=True)
        with (stage / "centroid_distances.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
            writer.writeheader()
            writer.writerows(measurements)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.plot(epochs, [row["centroid_distance"] for row in measurements],
                color="#4557b2", linewidth=2.3, marker="o", markersize=6)
        ax.set(xlabel="Epoch", ylabel="Euclidean distance", title="VB–VC embedding centroid distance",
               xticks=epochs, ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e8ed", linewidth=.7)
        ax.set_axisbelow(True)
        fig.savefig(stage / "dataset_centroid_distance.png", dpi=200, facecolor="white")
        plt.close(fig)
        write_json(stage / "index.json", {"plots": ["dataset_centroid_distance.png"],
                   "measurements": "centroid_distances.csv", "pooling_identity": reference,
                   "embedding_dimension": int(embeddings.shape[1]), "epochs": epochs})
    return aid
