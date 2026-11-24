"""Sequential orthogonal raw-Hz decoding directions with animal-held-out evaluation."""

import csv
import json

import numpy as np
from scipy.linalg import null_space
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from .collections import load_unit_means
from .store import implementation_stamp, write_json


def fit_sequential_directions(x, rates, max_directions=None):
    """OLS in shrinking complements, using one data SVD rather than repeated data fits.

    The reduced least-squares problem preserves the original objective without
    squaring its condition number. Only training arrays belong in this function.
    """
    x = np.asarray(x, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    mean, rate_mean = x.mean(axis=0), rates.mean()
    centered = x - mean
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    design = singular[:, None] * right
    target = left.T @ (rates - rate_mean)
    total_rate_energy = np.sum((rates - rate_mean)**2)
    del left, centered
    basis = np.eye(x.shape[1])
    directions, slopes, training_r2 = [], [], []
    for _ in range(min(max_directions or x.shape[1], x.shape[1])):
        reduced = design @ basis
        coefficient = np.linalg.lstsq(reduced, target,
                                      rcond=np.finfo(float).eps * max(x.shape))[0]
        fitted_energy = np.sum((reduced @ coefficient)**2)
        explained = fitted_energy / total_rate_energy if total_rate_energy > 0 else 0.
        # A vanishing coefficient has no scientifically meaningful direction.
        if explained < 1e-12:
            break
        slope = np.linalg.norm(coefficient)
        local_direction = coefficient / slope
        directions.append(basis @ local_direction)
        slopes.append(slope)
        training_r2.append(explained)
        basis = basis @ null_space(local_direction[None, :])
    directions = np.asarray(directions).reshape(-1, x.shape[1])
    np.testing.assert_allclose(directions @ directions.T, np.eye(len(directions)), atol=1e-10)
    return {"mean": mean, "rate_mean": rate_mean, "directions": directions,
            "slopes": np.asarray(slopes), "training_r2": np.asarray(training_r2)}


def plot_sequential_curves(path, records):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    series = records if isinstance(records, dict) else {"": records}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    minimum_r2 = 0.
    for color, (label, table) in zip(["#3265a8", "#d16a30"], series.items()):
        steps = np.array([row["direction"] for row in table])
        r2 = np.array([row["held_animal_r2"] for row in table])
        retained = np.array([row["embedding_variance_retained"] for row in table]) * 100
        minimum_r2 = min(minimum_r2, r2.min())
        axes[0].plot(steps, r2, color=color, label=label, marker="o", markersize=3, linewidth=1.8)
        axes[1].plot(np.r_[0, steps], np.r_[100, retained], color=color, label=label,
                     marker="o", markersize=3, linewidth=1.8)
    axes[0].axhline(0, color="#888888", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Direction fitted", ylabel="Individual direction · held-animal R²",
                title="Firing-rate prediction", ylim=(min(-.03, minimum_r2 - .02), 1.02))
    axes[1].set(xlabel="Directions removed", ylabel="Original embedding variance retained (%)",
                title="Cumulative variance retained", ylim=(-2, 102))
    if any(series):
        for ax in axes:
            ax.legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=.15)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.15, facecolor="white")
    plt.close(fig)


def analyze_sequential_directions(store, source_analysis=None, *, collection_id=None):
    if (source_analysis is None) == (collection_id is None):
        raise ValueError("Provide either a prior probe analysis or a pooled collection")
    saved = None
    if collection_id is not None:
        collection = store.read("collections", collection_id)
        collection_ids = [collection_id]
        pooling = {item["dataset_id"]: item["spec"] for item in collection["recipe"]["sources"]}
        parents = [{"kind": "collections", "id": collection_id}]
    else:
        source = store.read("analyses", source_analysis)
        if source["recipe"]["operation"] != "two_direction_removed_probe_comparison_v1":
            raise ValueError("Use the two-direction probe comparison as the source")
        collection_ids = source["recipe"]["collections"]
        folder = store.path("analyses", source_analysis)
        previous = json.loads((folder / "index.json").read_text())
        pooling = previous["pooling"]
        with (folder / previous["predictions"]).open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        parents = [{"kind": "analyses", "id": source_analysis}]
    recipe = {"operation": "sequential_raw_hz_direction_probes_v1", "source_analysis": source_analysis,
              "collections": collection_ids, "target": "raw_hz",
              "validation": "five_animal_folds", "fit": "sequential_OLS_with_intercept",
              "max_directions": "embedding_dimension", "training_r2_stop_threshold": 1e-12,
              "features": "unscaled_embeddings_in_orthonormal_complements",
              "variance": "sum_of_within_test_fold_centered_squares_relative_to_original",
              "weighting": "equal_per_neuron", "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "Sequential firing-rate directions and variance retained", parents) as (stage, aid):
        if stage is None:
            return aid
        max_rows = sum(item["units"] for cid in collection_ids for item in
                       json.loads((store.path("collections", cid) / "index.json").read_text())["sessions"])
        embeddings, original_rows = load_unit_means(store, collection_ids, max_rows=max_rows)
        rows = original_rows
        folds = None
        if saved is not None:
            by_uid = {row["unit_uid"]: row for row in saved}
            if len(by_uid) != len(saved) or set(by_uid) != {r["unit_uid"] for r in original_rows}:
                raise ValueError("Probe and embedding cohorts differ")
            rows = [by_uid[row["unit_uid"]] for row in original_rows]
            folds = np.array([int(row["probe_validation_fold"]) for row in rows])
        x = embeddings.astype(np.float64)
        rates = np.array([float(row["firing_rate_hz"]) for row in rows])
        if not np.isfinite(rates).all() or np.any(rates < 0) or np.var(rates) == 0:
            raise ValueError("Need finite, nonnegative, varying firing rates")
        np.testing.assert_array_equal(rates, [r["firing_rate_hz"] for r in original_rows])
        groups = np.array([row["subject_uid"] for row in rows])
        if len(set(groups)) < 5:
            raise ValueError("Need at least five animals for grouped validation")
        predictions = np.full(x.shape, np.nan)
        fold_results, fold_models = [], []
        print(f"Sequential probes: {len(rows):,} neurons, {len(set(groups))} animals, {x.shape[1]} dimensions", flush=True)
        for fold, (train, test) in enumerate(GroupKFold(5).split(x, groups=groups)):
            if folds is not None:
                np.testing.assert_array_equal(folds[test], fold)
            if set(groups[train]) & set(groups[test]):
                raise ValueError("Animal leakage")
            model = fit_sequential_directions(x[train], rates[train])
            count = len(model["directions"])
            if count == 0:
                raise ValueError(f"No measurable training rate direction in fold {fold}; cannot plot a sequential curve")
            predicted = ((x[test] - model["mean"]) @ model["directions"].T
                         * model["slopes"] + model["rate_mean"])
            predictions[test, :count] = predicted
            if saved is not None:
                for step, key in enumerate(("original_linear", "training_only_direction_linear", "two_directions_linear")):
                    np.testing.assert_allclose(predicted[:, step], [float(rows[i][key]) for i in test], atol=1e-7)
            # Center within the held-out cloud to measure variance, not mean offset.
            test_centered = x[test] - x[test].mean(axis=0)
            total = float(np.sum(test_centered**2))
            energy = np.sum((test_centered @ model["directions"].T)**2, axis=0)
            remaining = total - np.cumsum(energy)
            if remaining.min() < -1e-9 * total or np.any(np.diff(remaining) > 1e-9 * total):
                raise ValueError("Invalid cumulative variance accounting")
            for step in range(count):
                fold_results.append({"fold": fold, "direction": step + 1,
                                     "held_animal_r2": float(r2_score(rates[test], predicted[:, step])),
                                     "training_r2": float(model["training_r2"][step]),
                                     "original_embedding_variance_sum": total,
                                     "direction_embedding_variance_sum": float(energy[step]),
                                     "remaining_embedding_variance_sum": float(max(0, remaining[step]))})
            fold_models.append(model)
            print(f"Fold {fold + 1}/5: {count} directions; last training R2={model['training_r2'][-1]:.3g}", flush=True)
        count = min(len(model["directions"]) for model in fold_models)
        records = []
        for step in range(count):
            selected = [row for row in fold_results if row["direction"] == step + 1]
            total = sum(row["original_embedding_variance_sum"] for row in selected)
            records.append({"direction": step + 1,
                            "held_animal_r2": float(r2_score(rates, predictions[:, step])),
                            "fold_r2_min": min(row["held_animal_r2"] for row in selected),
                            "fold_r2_max": max(row["held_animal_r2"] for row in selected),
                            "direction_embedding_variance_fraction": sum(row["direction_embedding_variance_sum"] for row in selected) / total,
                            "embedding_variance_retained": sum(row["remaining_embedding_variance_sum"] for row in selected) / total})
        for filename, table in (("sequential_direction_metrics.csv", records), ("fold_metrics.csv", fold_results)):
            with (stage / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        # Save fitted directions for reproducing transformations without refitting.
        weights = {f"fold_{fold}_{key}": value for fold, model in enumerate(fold_models)
                   for key, value in model.items()}
        np.savez(stage / "sequential_directions.npz", **weights)
        plot_sequential_curves(stage / "sequential_direction_curves.png", records)
        write_json(stage / "index.json", {"neurons": len(rows), "animals": len(set(groups)),
                   "dimensions": x.shape[1], "directions_in_all_folds": count,
                   "directions_per_fold": [len(model["directions"]) for model in fold_models],
                   "metrics": "sequential_direction_metrics.csv", "fold_metrics": "fold_metrics.csv",
                   "weights": "sequential_directions.npz", "plots": ["sequential_direction_curves.png"],
                   "pooling": pooling,
                   "interpretation": "Direction k predicts raw Hz alone after k-1 orthogonal removals. "
                   "Its R2 equals that of the full linear probe fitted on the remaining space at that step. "
                   "Directions are trained separately within each training fold; no test labels are used. "
                   "Variance retained is cumulative after k removals, pooled over within-test-fold centered clouds. "
                   "Only directions fitted in all five folds enter the aggregate curve. "
                   "Animals were not held out of encoder pretraining."})
        for row in records:
            if row["direction"] <= 5 or row["direction"] % 10 == 0 or row == records[-1]:
                print(f"Direction {row['direction']}: R2={row['held_animal_r2']:.5f}; "
                      f"variance retained={row['embedding_variance_retained']:.3%}", flush=True)
    return aid
