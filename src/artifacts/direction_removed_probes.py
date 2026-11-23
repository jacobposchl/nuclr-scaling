"""Animal-held-out rate decoding after removing an OLS decoder direction."""

import csv
import json

import numpy as np
from scipy.linalg import null_space
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from .collections import load_unit_means
from .store import implementation_stamp, write_json


def remaining_coordinates(embeddings, mean, direction):
    """Represent the orthogonal remainder in D-1 dimensions to exclude roundoff leakage."""
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    basis = null_space(direction[None, :])
    np.testing.assert_allclose(direction @ basis, 0, atol=1e-12)
    return (np.asarray(embeddings, dtype=np.float64) - mean) @ basis


def fit_two_direction_removal(embeddings, rates):
    """Learn two sequential, orthogonal decoder directions from training data only."""
    x = np.asarray(embeddings, dtype=np.float64)
    mean = x.mean(axis=0)
    first = LinearRegression().fit(x, rates).coef_
    first /= np.linalg.norm(first)
    first_basis = null_space(first[None, :])
    first_remaining = (x - mean) @ first_basis
    second = LinearRegression().fit(first_remaining, rates).coef_
    second /= np.linalg.norm(second)
    directions = np.stack([first, first_basis @ second])
    basis = first_basis @ null_space(second[None, :])
    np.testing.assert_allclose(directions @ directions.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(directions @ basis, 0, atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(x.shape[1] - 2), atol=1e-12)
    return mean, basis, directions


def compare_removed_probes(embeddings, rates, groups, saved_mean, saved_direction,
                           saved_folds, nonlinear_parameters):
    x = np.asarray(embeddings, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    groups = np.asarray(groups)
    fixed = remaining_coordinates(x, saved_mean, saved_direction)
    predictions = {"original_linear": np.empty(len(x))}
    for removal in ("saved_direction", "training_only_direction"):
        for model in ("linear", "nonlinear"):
            predictions[f"{removal}_{model}"] = np.empty(len(x))
    fold_metrics = []
    for fold, (train, test) in enumerate(GroupKFold(5).split(x, groups=groups)):
        np.testing.assert_array_equal(saved_folds[test], fold)
        if set(groups[train]) & set(groups[test]):
            raise ValueError("An animal appears in both training and test")
        original = LinearRegression().fit(x[train], rates[train])
        predictions["original_linear"][test] = original.predict(x[test])
        train_mean = x[train].mean(axis=0)
        training_only = remaining_coordinates(x, train_mean, original.coef_)
        for removal, coordinates in (("saved_direction", fixed),
                                     ("training_only_direction", training_only)):
            # Both models see exactly the same 255 remaining coordinates.
            for name, model in (("linear", LinearRegression()),
                                ("nonlinear", HistGradientBoostingRegressor(**nonlinear_parameters))):
                print(f"Fold {fold + 1}/5: {removal}, {name}", flush=True)
                model.fit(coordinates[train], rates[train])
                prediction = model.predict(coordinates[test])
                if not np.isfinite(prediction).all():
                    raise ValueError("Non-finite probe predictions")
                key = f"{removal}_{name}"
                predictions[key][test] = prediction
                metrics = {"fold": fold, "model": key, "test_neurons": len(test),
                           "test_animals": len(np.unique(groups[test])),
                           "r2": float(r2_score(rates[test], prediction)),
                           "mae_hz": float(mean_absolute_error(rates[test], prediction))}
                fold_metrics.append(metrics)
                print(f"  held-animal R2 = {metrics['r2']:.5f}", flush=True)
    metrics = {}
    for key, prediction in predictions.items():
        metrics[key] = {"r2": float(r2_score(rates, prediction)),
                        "mae_hz": float(mean_absolute_error(rates, prediction))}
    return predictions, metrics, fold_metrics


def plot_comparison(path, rates, predictions, metrics, removal):
    import matplotlib.pyplot as plt

    names = ["original_linear", f"{removal}_linear", f"{removal}_nonlinear"]
    titles = ["Original embeddings · linear", "Direction removed · linear",
              "Direction removed · nonlinear"]
    bounds = [min(0, *(predictions[name].min() for name in names)),
              max(rates.max(), *(predictions[name].max() for name in names))]
    order = np.random.default_rng(42).permutation(len(rates))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, name, title in zip(axes, names, titles):
        ax.scatter(predictions[name][order], rates[order], s=3, alpha=.16,
                   color="#3265a8", linewidths=0, rasterized=True)
        ax.plot(bounds, bounds, color="#555555", linestyle="--", linewidth=1)
        ax.text(.05, .94, f"Held-animal R² = {metrics[name]['r2']:.4f}",
                transform=ax.transAxes, va="top", fontsize=12)
        ax.set(title=title, xlabel="Predicted firing rate (Hz)",
               ylabel="True firing rate (Hz)", xlim=bounds, ylim=bounds)
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.15, facecolor="white")
    plt.close(fig)


def analyze_removed_probes(store, source_analysis):
    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "pooled_neuron_firing_rate_probe_v1":
        raise ValueError("Use a saved firing-rate probe analysis")
    parameters = {"loss": "squared_error", "learning_rate": .1, "max_iter": 200,
                  "max_leaf_nodes": 31, "min_samples_leaf": 30,
                  "l2_regularization": 1.0, "early_stopping": False, "random_state": 42}
    recipe = {"operation": "direction_removed_probe_comparison_v1",
              "source_analysis": source_analysis, "collections": source["recipe"]["collections"],
              "target": "raw_hz", "linear": "OLS_with_intercept",
              "nonlinear": {"estimator": "HistGradientBoostingRegressor", **parameters},
              "validation": "same_five_animal_folds_as_source",
              "removal": ["saved_full_cohort_direction_descriptive",
                          "direction_refit_using_training_animals_only"],
              "features": "orthonormal_basis_of_direction_complement_no_standardization",
              "weighting": "equal_per_neuron", "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "Firing-rate probes after direction removal",
                     [{"kind": "analyses", "id": source_analysis}]) as (stage, aid):
        if stage is None:
            return aid
        folder = store.path("analyses", source_analysis)
        source_index = json.loads((folder / "index.json").read_text())
        with (folder / source_index["per_neuron_table"]).open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        by_uid = {row["unit_uid"]: row for row in saved}
        embeddings, original_rows = load_unit_means(store, recipe["collections"])
        if len(by_uid) != len(saved) or set(by_uid) != {r["unit_uid"] for r in original_rows}:
            raise ValueError("Probe and embedding cohorts differ")
        rows = [by_uid[row["unit_uid"]] for row in original_rows]
        rates = np.array([float(row["firing_rate_hz"]) for row in rows])
        np.testing.assert_array_equal(rates, [row["firing_rate_hz"] for row in original_rows])
        groups = [row["subject_uid"] for row in rows]
        folds = np.array([int(row["probe_validation_fold"]) for row in rows])
        with np.load(folder / source_index["weights"]) as weights:
            mean = weights["embedding_mean"]
            direction = weights["unit_directions"][source["recipe"]["targets"].index("raw_hz")]
        print(f"Comparing probes on {len(rows):,} neurons, {len(set(groups))} animals", flush=True)
        predictions, metrics, fold_metrics = compare_removed_probes(
            embeddings, rates, groups, mean, direction, folds, parameters)
        np.testing.assert_allclose(predictions["original_linear"],
                                   [float(r["raw_hz_held_animal_prediction"]) for r in rows], atol=1e-8)
        for key, values in metrics.items():
            print(f"{key}: R2={values['r2']:.6f}, MAE={values['mae_hz']:.4f} Hz", flush=True)
        fields = ["unit_uid", "dataset_id", "subject_uid", "firing_rate_hz", "probe_validation_fold"]
        with (stage / "held_animal_predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + list(predictions))
            writer.writeheader()
            for i, row in enumerate(rows):
                record = {key: row[key] for key in fields}
                record.update({key: float(value[i]) for key, value in predictions.items()})
                writer.writerow(record)
        plots = []
        for removal in ("saved_direction", "training_only_direction"):
            name = f"{removal}_probe_comparison.png"
            plot_comparison(stage / name, rates, predictions, metrics, removal)
            plots.append(name)
        datasets = np.array([r["dataset_id"] for r in rows])
        dataset_metrics = {dataset: {key: float(r2_score(rates[datasets == dataset],
                                                       prediction[datasets == dataset]))
                                    for key, prediction in predictions.items()}
                           for dataset in sorted(set(datasets))}
        write_json(stage / "index.json", {"neurons": len(rows), "animals": len(set(groups)),
                   "metrics": metrics, "fold_metrics": fold_metrics,
                   "dataset_r2": dataset_metrics, "plots": plots,
                   "predictions": "held_animal_predictions.csv", "pooling": source_index["pooling"],
                   "interpretation": "Training-only direction is the leakage-controlled evaluation. "
                   "Saved-direction results describe the plotted cloud, whose removal used all animals' labels. "
                   "Animals are held out of probe fitting, not encoder pretraining. "
                   "Nonlinear settings were fixed before evaluation; no test-fold tuning."})
    return aid


def analyze_second_direction_removal(store, source_analysis):
    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "direction_removed_probe_comparison_v1":
        raise ValueError("Use the previous one-direction probe comparison")
    parameters = {key: value for key, value in source["recipe"]["nonlinear"].items()
                  if key != "estimator"}
    recipe = {"operation": "two_direction_removed_probe_comparison_v1",
              "source_analysis": source_analysis, "collections": source["recipe"]["collections"],
              "target": "raw_hz", "removal": "two_sequential_OLS_directions_fit_on_training_animals_only",
              "validation": "same_five_animal_folds_as_source", "weighting": "equal_per_neuron",
              "nonlinear": source["recipe"]["nonlinear"], "linear": "OLS_with_intercept",
              "features": "254_orthonormal_complement_coordinates_no_standardization",
              "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "Firing-rate probes after two direction removals",
                     [{"kind": "analyses", "id": source_analysis}]) as (stage, aid):
        if stage is None:
            return aid
        folder = store.path("analyses", source_analysis)
        previous = json.loads((folder / "index.json").read_text())
        with (folder / previous["predictions"]).open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        by_uid = {row["unit_uid"]: row for row in saved}
        embeddings, original_rows = load_unit_means(store, recipe["collections"])
        if len(by_uid) != len(saved) or set(by_uid) != {r["unit_uid"] for r in original_rows}:
            raise ValueError("Probe and embedding cohorts differ")
        x = embeddings.astype(np.float64)
        rows = [by_uid[row["unit_uid"]] for row in original_rows]
        rates = np.array([float(row["firing_rate_hz"]) for row in rows])
        np.testing.assert_array_equal(rates, [r["firing_rate_hz"] for r in original_rows])
        groups = np.array([row["subject_uid"] for row in rows])
        folds = np.array([int(row["probe_validation_fold"]) for row in rows])
        predictions = {key: np.array([float(row[key]) for row in rows]) for key in
                       ("original_linear", "training_only_direction_linear", "training_only_direction_nonlinear")}
        predictions.update({f"two_directions_{name}": np.empty(len(rows)) for name in ("linear", "nonlinear")})
        fold_metrics, fold_means, fold_directions = [], [], []
        for fold, (train, test) in enumerate(GroupKFold(5).split(x, groups=groups)):
            np.testing.assert_array_equal(folds[test], fold)
            if set(groups[train]) & set(groups[test]):
                raise ValueError("An animal appears in both training and test")
            mean, basis, directions = fit_two_direction_removal(x[train], rates[train])
            # Verify that the first removal reproduces the preceding experiment.
            one_removed = remaining_coordinates(x, mean, directions[0])
            first_probe = LinearRegression().fit(one_removed[train], rates[train])
            np.testing.assert_allclose(first_probe.predict(one_removed[test]),
                                       predictions["training_only_direction_linear"][test], atol=1e-8)
            remaining = (x - mean) @ basis
            fold_means.append(mean)
            fold_directions.append(directions)
            for name, model in (("linear", LinearRegression()),
                                ("nonlinear", HistGradientBoostingRegressor(**parameters))):
                print(f"Fold {fold + 1}/5: two directions removed, {name}", flush=True)
                model.fit(remaining[train], rates[train])
                prediction = model.predict(remaining[test])
                if not np.isfinite(prediction).all():
                    raise ValueError("Non-finite probe predictions")
                predictions[f"two_directions_{name}"][test] = prediction
                score = float(r2_score(rates[test], prediction))
                fold_metrics.append({"fold": fold, "model": name, "r2": score,
                                     "mae_hz": float(mean_absolute_error(rates[test], prediction)),
                                     "test_animals": len(set(groups[test])), "test_neurons": len(test)})
                print(f"  held-animal R2 = {score:.5f}", flush=True)
        metrics = {key: {"r2": float(r2_score(rates, prediction)),
                         "mae_hz": float(mean_absolute_error(rates, prediction))}
                   for key, prediction in predictions.items()}
        for key, values in metrics.items():
            print(f"{key}: R2={values['r2']:.6f}, MAE={values['mae_hz']:.4f} Hz", flush=True)
        fields = ["unit_uid", "dataset_id", "subject_uid", "firing_rate_hz", "probe_validation_fold"]
        with (stage / "held_animal_predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + list(predictions))
            writer.writeheader()
            for i, row in enumerate(rows):
                record = {key: row[key] for key in fields}
                record.update({key: float(value[i]) for key, value in predictions.items()})
                writer.writerow(record)
        np.savez(stage / "removed_directions.npz", fold_means=fold_means, fold_directions=fold_directions)
        datasets = np.array([r["dataset_id"] for r in rows])
        dataset_metrics = {dataset: {key: float(r2_score(rates[datasets == dataset],
                                                       prediction[datasets == dataset]))
                                    for key, prediction in predictions.items()}
                           for dataset in sorted(set(datasets))}
        plot_second_removal(stage / "two_direction_probe_comparison.png", rates, predictions, metrics)
        write_json(stage / "index.json", {"neurons": len(rows), "animals": len(set(groups)),
                   "metrics": metrics, "fold_metrics": fold_metrics, "dataset_r2": dataset_metrics,
                   "predictions": "held_animal_predictions.csv", "directions": "removed_directions.npz",
                   "plots": ["two_direction_probe_comparison.png"], "pooling": previous["pooling"],
                   "interpretation": "Both removed directions and both new probes use training animals only. "
                   "The second direction is learned in the first direction's orthogonal complement. "
                   "Zero- and one-removal predictions are reused from the source analysis. "
                   "Animals were not held out of encoder pretraining."})
    return aid


def plot_second_removal(path, rates, predictions, metrics):
    import matplotlib.pyplot as plt

    keys = ["training_only_direction_linear", "two_directions_linear", "two_directions_nonlinear"]
    titles = ["One direction removed · linear", "Two directions removed · linear",
              "Two directions removed · nonlinear"]
    bounds = [min(0, *(predictions[key].min() for key in keys)),
              max(rates.max(), *(predictions[key].max() for key in keys))]
    order = np.random.default_rng(42).permutation(len(rates))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, key, title in zip(axes, keys, titles):
        ax.scatter(predictions[key][order], rates[order], s=3, alpha=.16,
                   color="#3265a8", linewidths=0)
        ax.plot(bounds, bounds, color="#555555", linestyle="--", linewidth=1)
        ax.text(.05, .94, f"Held-animal R² = {metrics[key]['r2']:.4f}",
                transform=ax.transAxes, va="top", fontsize=12)
        ax.set(title=title, xlabel="Predicted firing rate (Hz)", ylabel="True firing rate (Hz)",
               xlim=bounds, ylim=bounds)
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.15, facecolor="white")
    plt.close(fig)
