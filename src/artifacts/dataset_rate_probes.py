"""Compare raw-Hz decoder directions learned separately in VB and VC."""

import csv
import json

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from .collections import load_unit_means
from .store import implementation_stamp, write_json


DATASETS = ("allen_visual_behavior", "allen_visual_coding")
LABELS = ("VB", "VC")


def compare_dataset_probes(embeddings, rates, dataset_ids, groups):
    x = np.asarray(embeddings, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    datasets, groups = np.asarray(dataset_ids), np.asarray(groups)
    masks = [datasets == dataset for dataset in DATASETS]
    if set(datasets) != set(DATASETS) or any(len(set(groups[mask])) < 5 for mask in masks):
        raise ValueError("Need VB and VC with at least five animals each")
    if set(groups[masks[0]]) & set(groups[masks[1]]):
        raise ValueError("Cross-dataset evaluation requires disjoint animal groups")
    models, fold_weights, held_out, fold_ids = [], [], np.empty(len(x)), np.empty(len(x), dtype=int)
    fold_metrics = []
    for dataset, mask in zip(DATASETS, masks):
        indices = np.flatnonzero(mask)
        model = LinearRegression().fit(x[mask], rates[mask])
        models.append(model)
        weights = []
        for fold, (train, test) in enumerate(GroupKFold(5).split(x[mask], groups=groups[mask])):
            train, test = indices[train], indices[test]
            if set(groups[train]) & set(groups[test]):
                raise ValueError("Animal leakage")
            fitted = LinearRegression().fit(x[train], rates[train])
            held_out[test] = fitted.predict(x[test])
            fold_ids[test] = fold
            weights.append(fitted.coef_)
            fold_metrics.append({"dataset": dataset, "fold": fold,
                                 "held_animal_r2": float(r2_score(rates[test], held_out[test])),
                                 "test_animals": len(set(groups[test])), "test_neurons": len(test)})
        fold_weights.append(weights)
    coefficients = np.stack([model.coef_ for model in models])
    directions = coefficients / np.linalg.norm(coefficients, axis=1)[:, None]
    cosine = float(np.clip(directions[0] @ directions[1], -1, 1))
    fold_weights = np.asarray(fold_weights)
    fold_directions = fold_weights / np.linalg.norm(fold_weights, axis=2)[:, :, None]
    cross_fold_cosines = fold_directions[0] @ fold_directions[1].T
    metrics, prediction_matrix = [], np.empty((len(x), 2))
    for source, model in enumerate(models):
        prediction_matrix[:, source] = model.predict(x)
        # Diagonal entries must use held-out predictions rather than training fits.
        prediction_matrix[masks[source], source] = held_out[masks[source]]
        for target, mask in enumerate(masks):
            predicted = prediction_matrix[mask, source]
            metrics.append({"train_dataset": LABELS[source], "test_dataset": LABELS[target],
                            "evaluation": "five_fold_held_animal" if source == target else "cross_dataset_no_refit",
                            "r2": float(r2_score(rates[mask], predicted)),
                            "mae_hz": float(mean_absolute_error(rates[mask], predicted))})
    return {"coefficients": coefficients, "directions": directions,
            "intercepts": np.array([model.intercept_ for model in models]),
            "fold_coefficients": fold_weights, "fold_ids": fold_ids,
            "predictions": prediction_matrix, "metrics": metrics, "fold_metrics": fold_metrics,
            "similarity": {"cosine": cosine, "angle_degrees": float(np.degrees(np.arccos(cosine))),
                           "cross_fold_cosine_min": float(cross_fold_cosines.min()),
                           "cross_fold_cosine_max": float(cross_fold_cosines.max()),
                           "within_dataset_fold_to_full_cosines": {
                               label: (fold_directions[i] @ directions[i]).tolist()
                               for i, label in enumerate(LABELS)}}}


def plot_dataset_comparison(path, result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    directions = result["directions"]
    limit = np.max(np.abs(directions)) * 1.08
    axes[0].scatter(*directions, s=18, alpha=.65, color="#3265a8", linewidths=0)
    axes[0].plot([-limit, limit], [-limit, limit], linestyle="--", color="#888888", linewidth=1)
    axes[0].set(xlabel="VB direction weight", ylabel="VC direction weight",
                title="Normalized firing-rate directions", xlim=(-limit, limit), ylim=(-limit, limit))
    axes[0].set_aspect("equal", adjustable="box")
    similarity = result["similarity"]
    axes[0].text(.04, .96, f"Cosine = {similarity['cosine']:.4f}\nAngle = {similarity['angle_degrees']:.1f}°",
                 transform=axes[0].transAxes, va="top", fontsize=12)
    values = np.array([metric["r2"] for metric in result["metrics"]]).reshape(2, 2)
    heatmap = axes[1].imshow(values, cmap="Blues", vmin=min(0, values.min()), vmax=1)
    axes[1].set(xticks=[0, 1], xticklabels=LABELS, yticks=[0, 1], yticklabels=LABELS,
                xlabel="Test dataset", ylabel="Probe training dataset", title="Firing-rate prediction · R²")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{values[i, j]:.4f}", ha="center", va="center", fontsize=18,
                         color="white" if heatmap.norm(values[i, j]) > .55 else "black")
    fig.colorbar(heatmap, ax=axes[1], label="R²", shrink=.8)
    axes[0].spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.15, facecolor="white")
    plt.close(fig)


def analyze_dataset_probes(store, source_analysis):
    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "pooled_neuron_firing_rate_probe_v1":
        raise ValueError("Use the original firing-rate probe analysis")
    recipe = {"operation": "dataset_specific_raw_hz_probes_v1", "source_analysis": source_analysis,
              "collections": source["recipe"]["collections"], "datasets": list(DATASETS),
              "target": "raw_hz", "fit": "OLS_with_intercept_unscaled_original_embeddings",
              "weighting": "equal_per_neuron", "within_validation": "five_fold_by_animal_per_dataset",
              "cross_validation": "full_source_dataset_model_on_other_dataset_no_refit",
              "similarity": "signed_cosine_of_normalized_decoder_coefficients",
              "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "VB and VC firing-rate direction comparison",
                     [{"kind": "analyses", "id": source_analysis}]) as (stage, aid):
        if stage is None:
            return aid
        x, rows = load_unit_means(store, recipe["collections"])
        rates = np.array([row["firing_rate_hz"] for row in rows])
        print(f"Fitting separate dataset probes for {len(rows):,} neurons", flush=True)
        result = compare_dataset_probes(x, rates, [row["dataset_id"] for row in rows],
                                        [row["subject_uid"] for row in rows])
        np.savez(stage / "probe_weights.npz", **{key: result[key] for key in
                 ("coefficients", "directions", "intercepts", "fold_coefficients")})
        fields = ["unit_uid", "dataset_id", "subject_uid", "firing_rate_hz", "fold", "vb_prediction", "vc_prediction"]
        with (stage / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, row in enumerate(rows):
                record = {key: row[key] for key in fields[:4]}
                record.update(fold=int(result["fold_ids"][i]), vb_prediction=result["predictions"][i, 0],
                              vc_prediction=result["predictions"][i, 1])
                writer.writerow(record)
        with (stage / "prediction_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["metrics"][0]))
            writer.writeheader()
            writer.writerows(result["metrics"])
        plot_dataset_comparison(stage / "dataset_direction_comparison.png", result)
        previous = json.loads((store.path("analyses", source_analysis) / "index.json").read_text())
        counts = {dataset: {"neurons": sum(r["dataset_id"] == dataset for r in rows),
                            "animals": len({r["subject_uid"] for r in rows if r["dataset_id"] == dataset})}
                  for dataset in DATASETS}
        write_json(stage / "index.json", {"counts": counts, "similarity": result["similarity"],
                   "metrics": result["metrics"], "fold_metrics": result["fold_metrics"],
                   "pooling": previous["pooling"], "weights": "probe_weights.npz", "predictions": "predictions.csv",
                   "plots": ["dataset_direction_comparison.png"],
                   "interpretation": "Full-dataset coefficients define descriptive directions. Diagonal prediction "
                   "scores are animal-held-out within each dataset; off-diagonal scores transfer the entire source "
                   "model, including intercept and scale, without target calibration. Fold cosines describe "
                   "stability, not confidence intervals. Encoder pretraining included both datasets."})
        print(json.dumps({"similarity": result["similarity"], "metrics": result["metrics"]}, indent=2), flush=True)
    return aid
