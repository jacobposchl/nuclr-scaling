"""Matched-neuron comparisons of dataset separation and firing-rate decoding."""

from collections import Counter
import csv
import json

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold

from .collections import load_unit_means
from .embedding_overview import analyze_embedding_overview, DATASET_LABELS
from .store import implementation_stamp, write_json


def matched_order(reference, rows):
    by_uid = {row["unit_uid"]: i for i, row in enumerate(rows)}
    if len(by_uid) != len(rows) or set(by_uid) != {r["unit_uid"] for r in reference}:
        raise ValueError("Models must contain exactly the same unique neurons")
    order = np.array([by_uid[row["unit_uid"]] for row in reference])
    for original, i in zip(reference, order):
        if any(original[field] != rows[i][field] for field in
               ("dataset_id", "subject_uid", "session_uid", "brain_region", "firing_rate_hz")):
            raise ValueError("Matched neurons have different metadata or firing-rate pooling")
    return order


def grouped_decoding(x, rates, datasets, groups):
    """All regression/classification parameters are learned within each training fold."""
    predictions = np.empty(len(rates))
    dataset_predictions = np.empty(len(rates), dtype=datasets.dtype)
    folds = []
    for fold, (train, test) in enumerate(GroupKFold(5).split(x, groups=groups)):
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Animal leakage")
        regression = LinearRegression().fit(x[train], rates[train])
        predictions[test] = regression.predict(x[test])
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(3, 1 / 3))
        classifier.fit(x[train], datasets[train])
        dataset_predictions[test] = classifier.predict(x[test])
        folds.append({"fold": fold, "train_neurons": len(train), "test_neurons": len(test),
                      "firing_rate_r2": float(r2_score(rates[test], predictions[test]))})
    return {"firing_rate_r2": float(r2_score(rates, predictions)),
            "dataset_balanced_accuracy": float(balanced_accuracy_score(datasets, dataset_predictions))}, folds


def plot_matched_umaps(path, models, overviews, store):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 10), layout="constrained", squeeze=False)
    palette = {"allen_visual_behavior": "#3b4cc0", "allen_visual_coding": "#b40426",
               "ibl_brain_wide_bench_2026": "#13866e"}
    for column, (label, aid) in enumerate(zip(models, overviews)):
        with (store.path("analyses", aid) / "coordinates.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        xy = np.array([[float(r["joint_umap_1"]), float(r["joint_umap_2"])] for r in rows])
        rates = np.array([float(r["firing_rate_hz"]) for r in rows])
        order = np.random.default_rng(42).permutation(len(rows))
        colors = np.array([palette[r["dataset_id"]] for r in rows])
        axes[0, column].scatter(*xy[order].T, c=colors[order], s=2, alpha=.7, linewidths=0)
        axes[0, column].set_title(label, fontsize=16)
        axes[0, column].legend(handles=[Line2D([], [], color=color, marker="o", linestyle="", label=DATASET_LABELS[key])
                                       for key, color in palette.items()], frameon=False, fontsize=9)
        points = axes[1, column].scatter(*xy[order].T, c=rates[order], s=2, alpha=.7, linewidths=0,
                                         cmap="viridis", norm=LogNorm(rates.min(), rates.max()))
        fig.colorbar(points, ax=axes[1, column], label="Firing rate (Hz)", shrink=.7)
        for ax in axes[:, column]:
            ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
            ax.set_aspect("equal", adjustable="box")
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def analyze_rate_variant_comparison(store, models):
    if len(models) != 3:
        raise ValueError("Provide baseline epoch 10 and variant epochs 5 and 10")
    recipe = {"operation": "matched_rate_variant_comparison_v1", "models": models,
              "selection": "firing_rate_hz > 0", "validation": "five_animal_folds",
              "rate_probe": "OLS_with_intercept_raw_hz_no_scaling",
              "dataset_probe": "linear_discriminant_auto_shrinkage_equal_class_priors",
              "geometry": "unscaled_original_embedding_space",
              "silhouette": "same_1000_neurons_per_dataset_seed42",
              "implementation": implementation_stamp()}
    parents = [{"kind": "analyses", "id": aid} for aid in models.values()]
    filtered = []
    for label, aid in models.items():
        source = store.read("analyses", aid)
        annotation = source["recipe"].get("coding_annotations", {})
        print(f"Refitting zero-excluded UMAPs: {label}", flush=True)
        filtered.append(analyze_embedding_overview(store, source["recipe"]["source_analysis"],
                        labels_path=annotation.get("path"), positive_rate_only=True))
    with store.build("analyses", recipe, "Baseline versus rate-separated content, positive firing rates",
                     parents + [{"kind": "analyses", "id": aid} for aid in filtered]) as (stage, result_id):
        if stage is None:
            return result_id
        reference, metrics, fold_metrics, selected_sample = None, [], [], None
        for label, overview_id in models.items():
            source = store.read("analyses", overview_id)
            index = json.loads((store.path("analyses", overview_id) / "index.json").read_text())
            x, rows = load_unit_means(store, source["recipe"]["collections"], max_rows=index["neurons"])
            if reference is None:
                reference = rows
            else:
                x = x[matched_order(reference, rows)]
            rates = np.array([row["firing_rate_hz"] for row in reference])
            keep = rates > 0
            x = x[keep].astype(np.float64)
            kept_rows = [row for row, yes in zip(reference, keep) if yes]
            rates = rates[keep]
            datasets = np.array([row["dataset_id"] for row in kept_rows])
            groups = np.array([row["subject_uid"] for row in kept_rows])
            if selected_sample is None:
                rng = np.random.default_rng(42)
                selected_sample = np.concatenate([rng.choice(np.flatnonzero(datasets == name), 1000, replace=False)
                                                  for name in sorted(set(datasets))])
            print(f"Matched probes: {label}, {len(x):,} neurons, {x.shape[1]} dimensions", flush=True)
            scores, folds = grouped_decoding(x, rates, datasets, groups)
            total = np.sum((x - x.mean(0)) ** 2)
            between = sum(np.count_nonzero(datasets == name) * np.sum((x[datasets == name].mean(0) - x.mean(0)) ** 2)
                          for name in set(datasets))
            scores.update(model=label, neurons=len(x), dimensions=x.shape[1],
                          dataset_centroid_variance_fraction=float(between / total),
                          dataset_silhouette=float(silhouette_score(x[selected_sample], datasets[selected_sample])))
            metrics.append(scores)
            fold_metrics.extend(dict(model=label, **row) for row in folds)
            print(json.dumps(scores), flush=True)
        for name, records in (("metrics.csv", metrics), ("fold_metrics.csv", fold_metrics)):
            with (stage / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        plot_matched_umaps(stage / "positive_rate_umap_comparison.png", models, filtered, store)
        write_json(stage / "index.json", {"metrics": "metrics.csv", "fold_metrics": "fold_metrics.csv",
                   "plots": ["positive_rate_umap_comparison.png"],
                   "filtered_overviews": dict(zip(models, filtered)), "neurons": len(kept_rows),
                   "animals": len(set(groups)), "removed_zero_rate_by_dataset": dict(Counter(
                       row["dataset_id"] for row in reference if row["firing_rate_hz"] == 0)),
                   "caveat": "Animals held out from probes, not encoder pretraining. Dataset mixing does not establish biological utility."})
    return result_id
