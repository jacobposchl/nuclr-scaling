"""Compare neuron UMAPs before and after pooled linear firing-rate adjustment."""

import csv
import json

import numpy as np

from .store import implementation_stamp, write_json


def residualize_rates(embeddings, predictors, weights):
    """Remove an intercept and rate slopes, using the supplied observation weights.

    Dataset labels are intentionally absent. Each embedding dimension is a separate
    response, with the same pooled regression design and weights.
    """
    values = np.asarray(embeddings, dtype=np.float64)
    rates = np.asarray(predictors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (values.ndim != 2 or rates.ndim != 2 or weights.shape != (len(values),)
            or len(rates) != len(values) or len(values) <= rates.shape[1] + 1):
        raise ValueError("Expected aligned embedding rows, predictor rows, and weights")
    if (not np.isfinite(values).all() or not np.isfinite(rates).all()
            or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Regression inputs must be finite and weights positive")
    weights = weights / weights.sum()
    center = np.average(rates, axis=0, weights=weights)
    scale = np.sqrt(np.average((rates - center) ** 2, axis=0, weights=weights))
    if (scale <= 0).any():
        raise ValueError("Firing-rate predictors must vary across neurons")
    design = np.column_stack([np.ones(len(values)), (rates - center) / scale])
    root_weight = np.sqrt(weights)[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * root_weight, values * root_weight, rcond=None
    )
    if rank != design.shape[1]:
        raise ValueError("Firing-rate predictors are linearly dependent")
    residuals = values - design @ coefficients
    original_variance = np.sum(weights[:, None] * (
        values - np.average(values, axis=0, weights=weights)
    ) ** 2)
    if original_variance <= 0:
        raise ValueError("Embeddings must have nonzero variance")
    residual_variance = np.sum(weights[:, None] * residuals ** 2)
    orthogonality_error = float(np.max(np.abs(design.T @ (weights[:, None] * residuals))))
    if orthogonality_error > 1e-9 * max(1., float(np.sqrt(original_variance))):
        raise ValueError("Residuals retain a fitted linear firing-rate component")
    model = dict(coefficients=coefficients, predictor_center=center, predictor_scale=scale)
    metrics = dict(removed_variance_fraction=float(1 - residual_variance / original_variance),
                   maximum_weighted_design_residual_product=orthogonality_error)
    return residuals.astype(np.float32), model, metrics


def plot_comparisons(folder, projections, rows, neuron_rates, population_rates, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D

    dataset_colors = {"allen_visual_behavior": "#3b4cc0", "allen_visual_coding": "#b40426"}
    dataset_labels = {"allen_visual_behavior": "Visual Behavior", "allen_visual_coding": "Visual Coding"}
    colors = [dataset_colors[row["dataset_id"]] for row in rows]
    order = np.random.default_rng(seed).permutation(len(rows))
    titles = ["Original", "Population FR removed", "Population + neuron FR removed"]
    colorings = [
        ("dataset", np.asarray(colors), None, None),
        ("neuron_firing_rate", neuron_rates, LogNorm(neuron_rates.min(), neuron_rates.max()), "Neuron firing rate (Hz)"),
        ("population_firing_rate", population_rates, Normalize(population_rates.min(), population_rates.max()), "Mean population firing rate (Hz/neuron)"),
    ]
    for name, values, norm, label in colorings:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), layout="constrained")
        for ax, title, coordinates in zip(axes, titles, projections.values()):
            kwargs = {} if norm is None else {"cmap": "viridis", "norm": norm}
            points = ax.scatter(*coordinates[order].T, c=values[order], s=3, alpha=.8,
                                linewidths=0, **kwargs)
            ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
            ax.spines[["top", "right"]].set_visible(False)
        if norm is None:
            handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                              label=dataset_labels[dataset])
                       for dataset, color in dataset_colors.items()]
            axes[0].legend(handles=handles, fontsize=9, loc="best")
        else:
            fig.colorbar(points, ax=axes, label=label, fraction=.02, pad=.015)
        fig.savefig(folder / f"{name}_comparison.png", dpi=200, facecolor="white")
        plt.close(fig)


def analyze_rate_residualization(store, source_analysis):
    from .collections import load_unit_means

    source_manifest = store.read("analyses", source_analysis)
    source_recipe = source_manifest["recipe"]
    config = source_recipe["config"]
    if (source_recipe["operation"] != "visualization_v1" or config["standardize"]
            or config["fit_selection"] or config["show_selection"]):
        raise ValueError("Use an unstandardized baseline UMAP of the complete collection")
    collection_ids = source_recipe["collections"]
    if len(collection_ids) != 1:
        raise ValueError("Use one complete collection so population rates cover each recording")
    source_folder = store.path("analyses", source_analysis)
    source_index = json.loads((source_folder / "index.json").read_text())
    recipe = {
        "operation": "linear_firing_rate_residualization_umap_v1",
        "source_analysis": source_analysis,
        "collections": collection_ids,
        "predictors": {"population": ["population_mean_rate_hz"],
                       "population_and_neuron": ["population_mean_rate_hz", "firing_rate_hz"]},
        "regression": "pooled_weighted_least_squares_with_intercept",
        "predictor_units": "raw_hz",
        "observation_weight": "inverse_recording_population_size",
        "fit_scope": "all_neurons_descriptive_only_no_dataset_labels",
        "umap": config["umap"], "seed": config["seed"],
        "implementation": implementation_stamp(),
    }
    parents = [{"kind": "analyses", "id": source_analysis},
               {"kind": "collections", "id": collection_ids[0]}]
    with store.build("analyses", recipe, "Firing-rate residualized neuron UMAPs", parents) as (stage, aid):
        if stage is None:
            return aid
        print("Loading the existing neuron means and checking the baseline cohort", flush=True)
        embeddings, rows = load_unit_means(store, collection_ids, config["max_rows"])
        with (source_folder / source_index["methods"]["umap"]["coordinates"]).open() as stream:
            baseline_rows = list(csv.DictReader(stream))
        baseline_by_uid = {row["unit_uid"]: row for row in baseline_rows}
        if (len(baseline_by_uid) != len(baseline_rows)
                or set(baseline_by_uid) != {row["unit_uid"] for row in rows}):
            raise ValueError("Baseline coordinates and original embeddings must contain identical neurons")
        baseline_rows = [baseline_by_uid[row["unit_uid"]] for row in rows]
        baseline = np.array([[float(row["coordinate_1"]), float(row["coordinate_2"])]
                             for row in baseline_rows])
        neuron_rates = np.array([float(row["firing_rate_hz"]) for row in rows])
        np.testing.assert_array_equal(neuron_rates, [float(row["firing_rate_hz"]) for row in baseline_rows])
        if not np.isfinite(neuron_rates).all() or (neuron_rates <= 0).any() or not np.isfinite(baseline).all():
            raise ValueError("Expected finite coordinates and strictly positive measured firing rates")
        if {row["dataset_id"] for row in rows} != {"allen_visual_behavior", "allen_visual_coding"}:
            raise ValueError("Expected the two Allen datasets")
        sessions, session_indices, sizes = np.unique(
            [row["session_uid"] for row in rows], return_inverse=True, return_counts=True
        )
        mean_rates = np.bincount(session_indices, weights=neuron_rates) / sizes
        population_rates = mean_rates[session_indices]
        weights = 1. / sizes[session_indices]
        projections = {"original": baseline}
        regression_metrics = []
        from umap import UMAP

        for variant, predictors in [
            ("population", population_rates[:, None]),
            ("population_and_neuron", np.column_stack([population_rates, neuron_rates])),
        ]:
            print(f"Removing linear firing-rate effects: {variant}", flush=True)
            residuals, model, metrics = residualize_rates(embeddings, predictors, weights)
            np.savez(stage / f"{variant}_regression.npz", **model)
            regression_metrics.append(dict(variant=variant, **metrics))
            print(f"Fitting UMAP: {variant} ({len(rows):,} neurons, one CPU thread)", flush=True)
            coordinates = UMAP(n_components=2, random_state=config["seed"],
                               transform_seed=config["seed"], n_jobs=1, init="random",
                               **config["umap"]).fit_transform(residuals)
            if not np.isfinite(coordinates).all():
                raise ValueError("Nonfinite residualized UMAP coordinates")
            projections[variant] = coordinates
        print("Saving aligned coordinates and comparison figures", flush=True)
        with (stage / "coordinates.csv").open("w", newline="") as stream:
            fields = ["unit_uid", "session_uid", "subject_uid", "dataset_id", "firing_rate_hz",
                      "population_mean_rate_hz"] + [f"{variant}_umap_{axis}"
                      for variant in projections for axis in (1, 2)]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows):
                record = {key: row[key] for key in fields[:5]}
                record["population_mean_rate_hz"] = float(population_rates[index])
                for variant, coordinates in projections.items():
                    record.update({f"{variant}_umap_{axis+1}": float(value)
                                   for axis, value in enumerate(coordinates[index])})
                writer.writerow(record)
        with (stage / "regression_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(regression_metrics[0]))
            writer.writeheader()
            writer.writerows(regression_metrics)
        plot_comparisons(stage, projections, rows, neuron_rates, population_rates, config["seed"])
        write_json(stage / "index.json", {
            "neurons": len(rows), "recordings": len(sessions), "embedding_dimensions": embeddings.shape[1],
            "source_analysis": source_analysis, "baseline_coordinates_reused": True,
            "coordinates": "coordinates.csv", "regression_metrics": "regression_metrics.csv",
            "plots": [f"{color}_comparison.png" for color in
                      ("dataset", "neuron_firing_rate", "population_firing_rate")],
            "interpretation": "Descriptive pooled linear adjustment; nonlinear rate dependence may remain. "
                              "Each recording has equal total regression weight. UMAPs are separate fits.",
        })
    return aid
