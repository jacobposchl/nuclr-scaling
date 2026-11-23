"""Exact Ward clustering of a saved neuron cohort, with dataset annotations."""

from collections import Counter
import csv
import json
import time

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, is_valid_linkage, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from .collections import load_unit_means
from .embedding_overview import DATASET_LABELS, categorical_palettes
from .store import implementation_stamp, write_json


def dataset_composition(tree, dataset_ids, leaf_order, n_clusters=3):
    """Summarize a tree cut, numbering clusters from left to right in the figure."""
    datasets = np.asarray(dataset_ids)
    raw_labels = fcluster(tree, t=n_clusters, criterion="maxclust")
    mapping = {label: i + 1 for i, label in enumerate(dict.fromkeys(raw_labels[leaf_order]))}
    labels = np.array([mapping[label] for label in raw_labels])
    counts = []
    for cluster in sorted(set(labels)):
        selected = labels == cluster
        for dataset in DATASET_LABELS:
            count = int(np.count_nonzero(selected & (datasets == dataset)))
            counts.append({"cluster": int(cluster), "dataset_id": dataset, "neurons": count,
                           "cluster_neurons": int(selected.sum()),
                           "fraction_of_cluster": count / int(selected.sum()),
                           "fraction_of_dataset": count / int(np.count_nonzero(datasets == dataset))
                           if np.any(datasets == dataset) else 0.})
    metrics = {"requested_clusters": n_clusters, "actual_clusters": len(mapping),
               "adjusted_rand_index": float(adjusted_rand_score(datasets, labels)),
               "majority_dataset_fraction": sum(max(row["neurons"] for row in counts if row["cluster"] == cluster)
                                                for cluster in sorted(set(labels))) / len(labels),
               "largest_dataset_baseline": max(Counter(dataset_ids).values()) / len(labels)}
    return labels, counts, metrics


def plot_ward_datasets(path, tree, rows, leaf_order, labels):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    palette = categorical_palettes(rows)["dataset_id"]
    fig, (tree_axis, marker_axis) = plt.subplots(
        2, 1, figsize=(20, 9), gridspec_kw={"height_ratios": [12, 1], "hspace": .06})
    fig.subplots_adjust(left=.07, right=.99, top=.87, bottom=.12)
    fig.suptitle(f"Visual cortex · Ward clustering · {len(rows):,} neurons", y=.98, fontsize=21)
    plotted = dendrogram(tree, ax=tree_axis, no_labels=True, count_sort="descending",
                         link_color_func=lambda _: "#505b68")
    if plotted["leaves"] != list(leaf_order):
        raise ValueError("Dendrogram and dataset annotation orders differ")
    for collection in tree_axis.collections:
        collection.set_linewidth(.55)
    tree_axis.set_ylabel("Ward linkage distance", fontsize=13)
    tree_axis.spines[["top", "right", "bottom"]].set_visible(False)
    tree_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    n = len(rows)
    ordered_labels = labels[leaf_order]
    ordered_colors = np.array([to_rgb(palette[rows[i]["dataset_id"]]) for i in leaf_order])
    marker_axis.imshow(ordered_colors[None, :, :], aspect="auto", interpolation="nearest",
                       extent=(0, 10 * n, 0, 1))
    marker_axis.set_yticks([.5], ["Dataset"], fontsize=12)
    marker_axis.tick_params(axis="y", length=0)
    positions, ticks = [], []
    for cluster in sorted(set(labels)):
        indices = np.flatnonzero(ordered_labels == cluster)
        positions.append(10 * (indices[0] + indices[-1] + 1) / 2)
        ticks.append(f"Cluster {cluster}\n{len(indices):,} neurons")
        if indices[0] > 0:
            marker_axis.axvline(10 * indices[0], color="white", linewidth=1.3)
    marker_axis.set_xticks(positions, ticks, fontsize=11)
    marker_axis.set_xlabel("Neurons in dendrogram order", fontsize=12, labelpad=10)
    marker_axis.set_xlim(tree_axis.get_xlim())
    marker_axis.spines[:].set_visible(False)
    k = len(set(labels))
    if 1 < k < n and tree[-k, 2] < tree[-k + 1, 2]:
        threshold = (tree[-k, 2] + tree[-k + 1, 2]) / 2
        tree_axis.axhline(threshold, color="#929aa4", linestyle="--", linewidth=1)
    counts = Counter(row["dataset_id"] for row in rows)
    handles = [Patch(facecolor=palette[dataset], label=f"{DATASET_LABELS[dataset]} ({count:,})")
               for dataset, count in counts.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncols=3,
               frameon=False, fontsize=12)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def analyze_ward_clustering(store, source_analysis):
    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "visual_cortex_metadata_overview_v1":
        raise ValueError("Use a saved visual-cortex overview to identify the clustering cohort")
    folder = store.path("analyses", source_analysis)
    source_index = json.loads((folder / "index.json").read_text())
    with (folder / source_index["coordinates"]).open(newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    by_uid = {row["unit_uid"]: row for row in saved_rows}
    if len(by_uid) != len(saved_rows) or len(saved_rows) < 4:
        raise ValueError("Expected at least four uniquely identified neurons")
    recipe = {"operation": "visual_cortex_ward_clustering_v1", "source_analysis": source_analysis,
              "collections": source["recipe"]["collections"], "space": "original_unit_mean_embeddings",
              "method": "ward", "metric": "euclidean", "standardize": False,
              "subsample": False, "optimal_ordering": False, "display_count_sort": "descending",
              "summary_clusters": 3, "implementation": implementation_stamp()}
    parents = [{"kind": "analyses", "id": source_analysis}]
    with store.build("analyses", recipe, "Visual-cortex Ward dendrogram by dataset", parents) as (stage, aid):
        if stage is None:
            return aid
        started = time.monotonic()
        embeddings, embedding_rows = load_unit_means(
            store, recipe["collections"], max_rows=len(saved_rows),
            row_filter=lambda row: row["unit_uid"] in by_uid)
        if len(embedding_rows) != len(by_uid) or {row["unit_uid"] for row in embedding_rows} != set(by_uid):
            raise ValueError("Clustering embeddings and saved plotting cohort differ")
        rows = [by_uid[row["unit_uid"]] for row in embedding_rows]
        dimensions = embeddings.shape[1]
        print(f"Computing Euclidean distances: {len(rows):,} neurons × {dimensions} dimensions, one CPU thread", flush=True)
        distances = pdist(embeddings, metric="euclidean")
        del embeddings
        print(f"Distances ready after {time.monotonic() - started:.1f}s; fitting exact Ward linkage", flush=True)
        tree = linkage(distances, method="ward", optimal_ordering=False)
        del distances
        is_valid_linkage(tree, throw=True)
        if not np.isfinite(tree).all():
            raise ValueError("Nonfinite Ward linkage")
        np.save(stage / "linkage.npy", tree)
        print(f"Linkage ready after {time.monotonic() - started:.1f}s; rendering dendrogram", flush=True)
        leaf_order = np.array(dendrogram(tree, no_plot=True, count_sort="descending")["leaves"])
        labels, composition, metrics = dataset_composition(
            tree, [row["dataset_id"] for row in rows], leaf_order, recipe["summary_clusters"])
        plot_ward_datasets(stage / "ward_dataset_dendrogram.png", tree, rows, leaf_order, labels)
        with (stage / "leaf_order.csv").open("w", newline="") as stream:
            fields = ["leaf_position", "embedding_row", "unit_uid", "dataset_id", "session_uid", "subject_uid",
                      "brain_region", "cluster"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for position, row_id in enumerate(leaf_order):
                writer.writerow({**{field: rows[row_id][field] for field in fields[2:-1]},
                                 "leaf_position": position, "embedding_row": row_id, "cluster": labels[row_id]})
        with (stage / "cluster_composition.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(composition[0]))
            writer.writeheader()
            writer.writerows(composition)
        write_json(stage / "index.json", {
            "plots": ["ward_dataset_dendrogram.png"], "linkage": "linkage.npy", "leaf_order": "leaf_order.csv",
            "cluster_composition": "cluster_composition.csv", "neurons": len(rows), "dimensions": dimensions,
            "datasets": dict(Counter(row["dataset_id"] for row in rows)), "metrics": metrics,
            "elapsed_seconds": time.monotonic() - started, "pooling": source_index["pooling"],
            "source_analysis": source_analysis})
        print(f"Three-cluster summary: {metrics}", flush=True)
    return aid
