"""Dataset-by-region mean embeddings and their cosine similarity matrix."""

from collections import defaultdict
import csv
import json
import re

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

from .collections import load_unit_means
from .embedding_overview import DATASET_LABELS
from .store import implementation_stamp, write_json


DATASET_SHORT_LABELS = {"allen_visual_behavior": "VB", "allen_visual_coding": "VC",
                        "ibl_brain_wide_bench_2026": "IBL"}


def regional_means(embeddings, rows):
    """Average equally over neurons within dataset/area, merging cortical layers."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(rows) or not len(rows) or not np.isfinite(x).all():
        raise ValueError("Expected one finite embedding per annotated neuron")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        region = row["brain_region"]
        if not region.startswith("VIS") or region.startswith("VISC"):
            raise ValueError("Expected the visual-cortex cohort, excluding visceral cortex")
        area = re.sub(r"^(VIS[a-z]*)(?:1|2/3|4|5|6a|6b)$", r"\1", region)
        if row["dataset_id"] not in DATASET_SHORT_LABELS:
            raise ValueError("Unsupported dataset")
        groups[(row["dataset_id"], area)].append(i)
    means, metadata = [], []
    for (dataset, area), indices in sorted(groups.items()):
        means.append(x[indices].mean(axis=0))
        metadata.append({"dataset_id": dataset, "brain_region": area, "neurons": len(indices),
                         "animals": len({rows[i]["subject_uid"] for i in indices}),
                         "recordings": len({rows[i]["session_uid"] for i in indices}),
                         "source_regions": ";".join(sorted({rows[i]["brain_region"] for i in indices}))})
    means = np.stack(means)
    norms = np.linalg.norm(means, axis=1)
    if np.any(norms == 0):
        raise ValueError("A regional mean has zero norm; cosine similarity is undefined")
    normalized = means / norms[:, None]
    similarity = np.clip(normalized @ normalized.T, -1., 1.)
    return means, metadata, similarity


def plot_region_similarity(path, tree, groups, similarity, palette):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(15, 14))
    matrix_axis = fig.add_axes([.245, .16, .58, .6214286])
    top_marker = fig.add_axes([.245, .788, .58, .0143])
    left_marker = fig.add_axes([.225, .16, .01333, .6214286])
    tree_axis = fig.add_axes([.245, .815, .58, .12])
    color_axis = fig.add_axes([.85, .16, .018, .6214286])
    plotted = dendrogram(tree, ax=tree_axis, no_labels=True, count_sort="descending",
                         link_color_func=lambda _: "#505b68")
    order = np.array(plotted["leaves"])
    tree_axis.set_ylabel("Ward distance", fontsize=10)
    tree_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    tree_axis.tick_params(axis="y", labelsize=9)
    tree_axis.spines[["top", "right", "bottom"]].set_visible(False)
    ordered = [groups[i] for i in order]
    matrix = similarity[np.ix_(order, order)]
    lower = max(-1., float(np.floor(matrix.min() * 100) / 100))
    points = matrix_axis.imshow(matrix, vmin=lower, vmax=1., cmap="viridis", interpolation="nearest")
    labels = [f"{DATASET_SHORT_LABELS[row['dataset_id']]} · {row['brain_region']}" for row in ordered]
    matrix_axis.set_xticks(range(len(groups)), labels, rotation=90, fontsize=10)
    matrix_axis.set_yticks(range(len(groups)),
                          [f"{label}  (n={row['neurons']:,})" for label, row in zip(labels, ordered)], fontsize=10)
    matrix_axis.tick_params(axis="y", length=0, pad=25)
    matrix_axis.tick_params(axis="x", length=0, pad=7)
    matrix_axis.spines[:].set_visible(False)
    colors = np.array([to_rgb(palette[row["dataset_id"]]) for row in ordered])
    top_marker.imshow(colors[None], aspect="auto", interpolation="nearest")
    left_marker.imshow(colors[:, None], aspect="auto", interpolation="nearest")
    top_marker.set_axis_off()
    left_marker.set_axis_off()
    bar = fig.colorbar(points, cax=color_axis)
    bar.set_label("Cosine similarity", fontsize=13)
    bar.ax.tick_params(labelsize=10)
    fig.suptitle("Visual cortex · similarity of regional mean embeddings", y=.99, fontsize=21)
    fig.legend(handles=[Patch(facecolor=palette[key], label=label) for key, label in DATASET_LABELS.items()
                        if key in palette], loc="upper center", bbox_to_anchor=(.53, .965), ncols=3,
               frameon=False, fontsize=12)
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
    return order


def analyze_region_similarity(store, source_analysis):
    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "visual_cortex_ward_clustering_v1":
        raise ValueError("Use the saved visual-cortex Ward analysis to identify the cohort")
    folder = store.path("analyses", source_analysis)
    source_index = json.loads((folder / "index.json").read_text())
    with (folder / source_index["leaf_order"]).open(newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    by_uid = {row["unit_uid"]: row for row in saved_rows}
    if len(by_uid) != len(saved_rows):
        raise ValueError("Duplicate neuron IDs in the source cohort")
    recipe = {"operation": "visual_cortex_region_similarity_v1", "source_analysis": source_analysis,
              "collections": source["recipe"]["collections"], "grouping": "dataset_and_parent_visual_area",
              "layer_merging": "remove_terminal_1_2/3_4_5_6a_6b_from_VIS_area_acronyms",
              "averaging": "arithmetic_mean_of_raw_embeddings_equal_weight_per_neuron",
              "similarity": "cosine_between_regional_mean_embeddings", "center": False,
              "standardize": False, "ordering": "Ward_on_raw_regional_mean_embeddings_equal_weight_per_group",
              "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "Visual-cortex regional mean similarity",
                     [{"kind": "analyses", "id": source_analysis}]) as (stage, aid):
        if stage is None:
            return aid
        x, rows = load_unit_means(store, recipe["collections"], max_rows=len(saved_rows),
                                 row_filter=lambda row: row["unit_uid"] in by_uid)
        if len(rows) != len(by_uid) or {row["unit_uid"] for row in rows} != set(by_uid):
            raise ValueError("Embeddings and source clustering cohort differ")
        means, groups, similarity = regional_means(x, rows)
        if len(groups) < 2:
            raise ValueError("Need at least two dataset/region groups")
        tree = linkage(means, method="ward", metric="euclidean", optimal_ordering=False)
        overview_id = source["recipe"]["source_analysis"]
        store.read("analyses", overview_id)
        with (store.path("analyses", overview_id) / "category_colors.csv").open(newline="") as stream:
            palette = {row["category"]: row["color"] for row in csv.DictReader(stream) if row["field"] == "dataset_id"}
        order = plot_region_similarity(stage / "region_similarity.png", tree, groups, similarity, palette)
        np.savez(stage / "regional_embeddings.npz", means=means, linkage=tree, leaf_order=order)
        labels = [f"{DATASET_SHORT_LABELS[groups[i]['dataset_id']]}|{groups[i]['brain_region']}" for i in order]
        with (stage / "similarity.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset_region", *labels])
            writer.writerows([label, *values] for label, values in zip(labels, similarity[np.ix_(order, order)]))
        with (stage / "regions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["leaf_position", "embedding_row", *groups[0]])
            writer.writeheader()
            writer.writerows(dict(leaf_position=position, embedding_row=int(i), **groups[i])
                             for position, i in enumerate(order))
        write_json(stage / "index.json", {"plots": ["region_similarity.png"], "regions": "regions.csv",
                   "similarity": "similarity.csv", "embeddings": "regional_embeddings.npz",
                   "neurons": len(rows), "groups": len(groups), "dimensions": means.shape[1],
                   "minimum_cosine": float(similarity.min()), "source_analysis": source_analysis})
        print(f"Saved {len(groups)} dataset/region means from {len(rows):,} neurons; "
              f"cosine range {similarity.min():.4f} to {similarity.max():.4f}", flush=True)
    return aid
