"""Linear firing-rate decoding and per-neuron projections onto decoder weights."""

import csv
import json

import numpy as np

from .collections import load_unit_means
from .embedding_overview import DATASET_LABELS
from .store import implementation_stamp, write_json


TARGETS = ("raw_hz", "log10_hz")


def fit_firing_rate_probes(embeddings, rates, animal_ids, n_splits=5):
    """OLS with an intercept; unscaled features, equal neuron weights, animal-held CV.

    The geometric fraction for neuron i is (centered_z_i @ unit_weight)^2 /
    ||centered_z_i||^2. It is not a within-neuron temporal R-squared.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    x = np.asarray(embeddings, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    groups = np.asarray(animal_ids)
    if (x.ndim != 2 or rates.shape != (len(x),) or groups.shape != (len(x),)
            or len(x) <= x.shape[1] or len(np.unique(groups)) < n_splits):
        raise ValueError("Need aligned neurons, rates, and sufficient animal groups")
    if not np.isfinite(x).all() or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Embeddings must be finite and rates strictly positive")
    targets = np.column_stack([rates, np.log10(rates)])
    if np.any(targets.std(axis=0) == 0):
        raise ValueError("Firing rates must vary")
    mean = x.mean(axis=0)
    centered = x - mean
    squared_norm = np.einsum("ij,ij->i", centered, centered)
    if squared_norm.sum() <= 0:
        raise ValueError("Embeddings must vary")
    probe = LinearRegression().fit(x, targets)
    lengths = np.linalg.norm(probe.coef_, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        raise ValueError("Fitted probe has no defined weight direction")
    directions = probe.coef_ / lengths[:, None]
    scores = centered @ directions.T
    energy = scores ** 2
    fractions = np.divide(energy, squared_norm[:, None],
                          out=np.full_like(energy, np.nan), where=squared_norm[:, None] > 0)
    if np.nanmin(fractions) < -1e-12 or np.nanmax(fractions) > 1 + 1e-12:
        raise ValueError("Projection fractions outside [0, 1]")
    fractions = np.clip(fractions, 0, 1)
    predicted = probe.predict(x)
    held_out = np.empty_like(targets)
    fold_ids = np.full(len(x), -1, dtype=int)
    fold_coefficients, fold_intercepts, fold_metrics = [], [], []
    for fold, (train, test) in enumerate(GroupKFold(n_splits).split(x, groups=groups)):
        fitted = LinearRegression().fit(x[train], targets[train])
        held_out[test] = fitted.predict(x[test])
        fold_ids[test] = fold
        fold_coefficients.append(fitted.coef_)
        fold_intercepts.append(fitted.intercept_)
        fold_metrics.append({"fold": fold, "test_animals": len(np.unique(groups[test])),
                             "test_neurons": len(test),
                             "r2": r2_score(targets[test], held_out[test], multioutput="raw_values").tolist()})
    training_r2 = r2_score(targets, predicted, multioutput="raw_values")
    held_r2 = r2_score(targets, held_out, multioutput="raw_values")
    metrics = {}
    for i, name in enumerate(TARGETS):
        fold_directions = np.array(fold_coefficients)[:, i, :]
        fold_directions /= np.linalg.norm(fold_directions, axis=1)[:, None]
        metrics[name] = {
            "training_r2": float(training_r2[i]), "held_animal_r2": float(held_r2[i]),
            "embedding_variance_fraction_along_direction": float(energy[:, i].sum() / squared_norm.sum()),
            "per_neuron_fraction_quantiles": dict(zip(
                ["p05", "p25", "median", "p75", "p95"],
                np.nanquantile(fractions[:, i], [.05, .25, .5, .75, .95]).tolist())),
            "fold_weight_cosine_to_full_fit": (fold_directions @ directions[i]).tolist(),
        }
    return {
        "model": {"embedding_mean": mean, "coefficients": probe.coef_, "intercepts": probe.intercept_,
                  "unit_directions": directions, "fold_coefficients": np.array(fold_coefficients),
                  "fold_intercepts": np.array(fold_intercepts)},
        "metrics": metrics, "fold_metrics": fold_metrics, "squared_norm": squared_norm,
        "targets": targets, "predicted": predicted, "held_out": held_out, "fold_ids": fold_ids,
        "scores": scores, "energy": energy, "fractions": fractions,
    }


def plot_firing_rate_probes(path, result, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    coordinates = np.array([[float(row["pc1"]), float(row["pc2"])] for row in rows])
    datasets = np.array([row["dataset_id"] for row in rows])
    colors = {"allen_visual_behavior": "#3b4cc0", "allen_visual_coding": "#b40426"}
    order = np.random.default_rng(42).permutation(len(rows))
    fractions = result["fractions"] * 100
    maximum = float(np.nanmax(fractions))
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), layout="constrained")
    for i, (target, label) in enumerate(zip(TARGETS, ["Raw-Hz probe", "Log-rate probe"])):
        ax = axes[i, 0]
        scatter = ax.scatter(*coordinates[order].T, c=fractions[order, i], s=3,
                             cmap="viridis", norm=Normalize(0, maximum), linewidths=0)
        ax.set(title=f"{label} · per-neuron projection", xlabel="PC1", ylabel="PC2")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=ax, label="Centered embedding energy along direction (%)", shrink=.8)
        ax = axes[i, 1]
        for dataset, color in colors.items():
            mask = datasets == dataset
            ax.hist(fractions[mask, i], bins=np.linspace(0, maximum, 61), density=True,
                    histtype="step", linewidth=1.5, color=color, label=DATASET_LABELS[dataset])
        ax.set(title=f"{label} · fraction distribution", xlabel="Centered embedding energy along direction (%)",
               ylabel="Density", yscale="log")
        ax.legend(frameon=False)
        ax = axes[i, 2]
        for dataset, color in colors.items():
            selected = order[datasets[order] == dataset]
            ax.scatter(result["targets"][selected, i], result["held_out"][selected, i], s=2,
                       alpha=.35, color=color, linewidths=0)
        bounds = [min(result["targets"][:, i].min(), result["held_out"][:, i].min()),
                  max(result["targets"][:, i].max(), result["held_out"][:, i].max())]
        ax.plot(bounds, bounds, color="#333333", linewidth=1, linestyle="--")
        unit = "Hz" if i == 0 else "log10 Hz"
        ax.set(title=f"{label} · held-animal R² = {result['metrics'][target]['held_animal_r2']:.3f}",
               xlabel=f"Measured firing rate ({unit})", ylabel=f"Predicted firing rate ({unit})")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.15, facecolor="white")
    plt.close(fig)


def analyze_firing_rate_probes(store, source_analysis):
    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "combined_pca_metadata_overview_v1":
        raise ValueError("Use the annotated combined PCA analysis")
    collection_ids = source["recipe"]["collections"]
    recipe = {"operation": "pooled_neuron_firing_rate_probe_v1", "source_analysis": source_analysis,
              "collections": collection_ids, "targets": list(TARGETS), "fit": "OLS_with_intercept",
              "standardize": False, "weighting": "equal_per_neuron", "validation": "5_fold_grouped_by_animal",
              "direction": "normalized_full_fit_decoder_weights", "centering": "pooled_neuron_mean",
              "per_neuron_fraction": "squared_projection / squared_centered_embedding_norm",
              "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "Firing-rate probes and per-neuron direction fractions",
                     [{"kind": "analyses", "id": source_analysis}]) as (stage, aid):
        if stage is None:
            return aid
        folder = store.path("analyses", source_analysis)
        source_index = json.loads((folder / "index.json").read_text())
        with (folder / source_index["coordinates"]).open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        by_uid = {row["unit_uid"]: row for row in saved}
        embeddings, embedding_rows = load_unit_means(store, collection_ids)
        if len(by_uid) != len(saved) or set(by_uid) != {r["unit_uid"] for r in embedding_rows}:
            raise ValueError("Embeddings and annotated neurons do not match")
        rows = [by_uid[r["unit_uid"]] for r in embedding_rows]
        rates = np.array([float(r["firing_rate_hz"]) for r in rows])
        np.testing.assert_array_equal(rates, [r["firing_rate_hz"] for r in embedding_rows])
        print(f"Fitting raw-Hz and log-rate probes for {len(rows):,} neurons on one CPU thread", flush=True)
        result = fit_firing_rate_probes(embeddings, rates, [r["subject_uid"] for r in rows])
        for target, metrics in result["metrics"].items():
            print(f"{target}: held-animal R2={metrics['held_animal_r2']:.4f}; "
                  f"direction variance share={metrics['embedding_variance_fraction_along_direction']:.4%}", flush=True)
        np.savez(stage / "probe_weights.npz", **result["model"])
        metadata = ["unit_uid", "dataset_id", "session_uid", "subject_uid", "probe_uid", "cell_type",
                    "cell_type_source", "brain_region", "firing_rate_hz", "pc1", "pc2"]
        values = {"prediction": result["predicted"], "held_animal_prediction": result["held_out"],
                  "direction_coordinate": result["scores"], "squared_projection": result["energy"],
                  "direction_energy_fraction": result["fractions"],
                  "total_variance_contribution": result["energy"] / result["squared_norm"].sum()}
        fields = metadata + ["probe_validation_fold", "centered_embedding_squared_norm"]
        fields += [f"{target}_{name}" for target in TARGETS for name in values]
        with (stage / "neuron_direction_fractions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, row in enumerate(rows):
                record = {key: row[key] for key in metadata}
                record.update(probe_validation_fold=int(result["fold_ids"][i]),
                              centered_embedding_squared_norm=float(result["squared_norm"][i]))
                record.update({f"{target}_{name}": float(array[i, j])
                               for j, target in enumerate(TARGETS) for name, array in values.items()})
                writer.writerow(record)
        plot_firing_rate_probes(stage / "firing_rate_direction.png", result, rows)
        write_json(stage / "index.json", {"neurons": len(rows), "embedding_dimensions": embeddings.shape[1],
                   "metrics": result["metrics"], "fold_metrics": result["fold_metrics"],
                   "pooling": source_index["pooling"], "plots": ["firing_rate_direction.png"],
                   "per_neuron_table": "neuron_direction_fractions.csv", "weights": "probe_weights.npz",
                   "interpretation": "Per-neuron fractions are geometric energy shares of pooled centered embeddings, "
                   "not temporal variance or per-neuron prediction R2. Validation holds animals out of the probe, "
                   "not out of encoder pretraining. Full-fit decoder directions define the geometric projections."})
    return aid


def analyze_direction_removed_pca(store, source_analysis):
    """Fit the first six PCs of the same cloud used for direction-removed UMAPs."""
    from sklearn.decomposition import PCA
    from .embedding_overview import categorical_palettes, plot_overview

    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "raw_hz_decoder_direction_removed_umaps_v1":
        raise ValueError("Use a saved raw-Hz direction-removed UMAP analysis")
    collection_ids = source["recipe"]["collections"]
    probe_id = source["recipe"]["source_analysis"]
    probe = store.read("analyses", probe_id)
    recipe = {"operation": "raw_hz_decoder_direction_removed_pca_v1", "source_analysis": source_analysis,
              "collections": collection_ids, "probe_analysis": probe_id,
              "pca": {"n_components": 6, "svd_solver": "full", "standardize": False,
                      "weighting": "equal_per_neuron", "variance_denominator": "direction_removed_cloud"},
              "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "Raw-Hz direction removed: first six PCs",
                     [{"kind": "analyses", "id": source_analysis}]) as (stage, aid):
        if stage is None:
            return aid
        folder = store.path("analyses", source_analysis)
        source_index = json.loads((folder / "index.json").read_text())
        with (folder / source_index["coordinates"]).open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        by_uid = {r["unit_uid"]: r for r in saved}
        embeddings, original_rows = load_unit_means(store, collection_ids)
        if len(by_uid) != len(saved) or set(by_uid) != {r["unit_uid"] for r in original_rows}:
            raise ValueError("PCA and direction-removed UMAP neurons do not match")
        rows = [by_uid[r["unit_uid"]] for r in original_rows]
        np.testing.assert_array_equal([float(r["firing_rate_hz"]) for r in rows],
                                      [r["firing_rate_hz"] for r in original_rows])
        with np.load(store.path("analyses", probe_id) / "probe_weights.npz") as weights:
            mean = weights["embedding_mean"]
            direction = weights["unit_directions"][probe["recipe"]["targets"].index("raw_hz")]
        centered = embeddings.astype(np.float64) - mean
        scores = centered @ direction
        residuals = centered - scores[:, None] * direction
        np.testing.assert_allclose(residuals @ direction, 0, atol=1e-10)
        removed_fraction = float(np.sum(scores**2) / np.sum(centered**2))
        np.testing.assert_allclose(removed_fraction, source_index["removed_variance_fraction"], atol=1e-12)
        print(f"Fitting six PCs: {len(rows):,} neurons; saved direction removal verified", flush=True)
        model = PCA(n_components=6, svd_solver="full")
        coordinates = model.fit_transform(residuals)
        variance = model.explained_variance_ratio_
        if not np.isfinite(coordinates).all() or not np.isfinite(variance).all():
            raise ValueError("Nonfinite PCA result")
        np.testing.assert_allclose(model.components_ @ direction, 0, atol=1e-10)
        np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(6), atol=1e-10)
        np.testing.assert_allclose(coordinates.var(axis=0, ddof=1), model.explained_variance_, rtol=1e-10)
        palettes = categorical_palettes(rows)
        plots = []
        for first in (0, 2, 4):
            name = f"combined_pc{first + 1}_pc{first + 2}_panels.png"
            plots.append(name)
            plot_overview(stage / name, coordinates[:, first:first + 2], rows, palettes, rows,
                          "Both datasets · FR direction removed", True,
                          axis_labels=tuple(f"PC{i + 1} ({variance[i]:.2%} variance)"
                                            for i in (first, first + 1)))
            print(f"Rendered PC{first + 1}/PC{first + 2}", flush=True)
        fields = [field for field in rows[0] if "umap" not in field]
        pc_fields = [f"pc{i + 1}" for i in range(6)]
        with (stage / "coordinates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + pc_fields)
            writer.writeheader()
            for row, coordinate in zip(rows, coordinates):
                writer.writerow({**{field: row[field] for field in fields}, **dict(zip(pc_fields, coordinate))})
        np.savez(stage / "pca_basis.npz", components=model.components_, mean=model.mean_,
                 explained_variance_ratio=variance, explained_variance=model.explained_variance_,
                 original_embedding_mean=mean, removed_direction=direction)
        write_json(stage / "index.json", {"plots": plots, "coordinates": "coordinates.csv",
                   "basis": "pca_basis.npz", "source_analysis": source_analysis, "neurons": len(rows),
                   "pooling": source_index["pooling"], "removed_variance_fraction": removed_fraction,
                   "explained_variance_ratio": variance.tolist(), "pca": recipe["pca"]})
        print("Explained variance: " + ", ".join(f"PC{i + 1} {v:.2%}" for i, v in enumerate(variance)), flush=True)
    return aid


def analyze_direction_removed_umaps(store, source_analysis):
    """Remove the saved pooled raw-Hz decoder direction before three UMAP fits."""
    from umap import UMAP
    from .embedding_overview import add_population_metadata, categorical_palettes, plot_overview

    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "pooled_neuron_firing_rate_probe_v1":
        raise ValueError("Use a saved pooled firing-rate probe analysis")
    collection_ids = source["recipe"]["collections"]
    baseline = store.read("analyses", source["recipe"]["source_analysis"])
    while baseline["recipe"]["operation"] != "embedding_metadata_overview_v1":
        baseline = store.read("analyses", baseline["recipe"]["source_analysis"])
    seed, umap_config = baseline["recipe"]["seed"], baseline["recipe"]["umap"]
    recipe = {"operation": "raw_hz_decoder_direction_removed_umaps_v1",
              "source_analysis": source_analysis, "collections": collection_ids,
              "centering": "saved_pooled_neuron_mean", "direction": "saved_full_fit_raw_hz_unit_direction",
              "transformation": "X_centered - outer(X_centered @ u, u)",
              "standardize": False, "umap": umap_config, "seed": seed,
              "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "Raw-Hz probe direction removed: joint and dataset UMAPs",
                     [{"kind": "analyses", "id": source_analysis}]) as (stage, aid):
        if stage is None:
            return aid
        folder = store.path("analyses", source_analysis)
        source_index = json.loads((folder / "index.json").read_text())
        with (folder / source_index["per_neuron_table"]).open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        by_uid = {row["unit_uid"]: row for row in saved}
        embeddings, original_rows = load_unit_means(store, collection_ids)
        if len(by_uid) != len(saved) or set(by_uid) != {r["unit_uid"] for r in original_rows}:
            raise ValueError("Probe and embedding neurons do not match")
        rows = add_population_metadata([by_uid[r["unit_uid"]] for r in original_rows])
        np.testing.assert_array_equal([r["firing_rate_hz"] for r in rows],
                                      [r["firing_rate_hz"] for r in original_rows])
        target = source["recipe"]["targets"].index("raw_hz")
        with np.load(folder / source_index["weights"]) as weights:
            mean = weights["embedding_mean"]
            direction = weights["unit_directions"][target]
        centered = embeddings.astype(np.float64) - mean
        np.testing.assert_allclose(embeddings.astype(np.float64).mean(axis=0), mean, atol=1e-12)
        np.testing.assert_allclose(direction @ direction, 1, atol=1e-12)
        scores = centered @ direction
        residuals = centered - scores[:, None] * direction
        norm2 = np.einsum("ij,ij->i", centered, centered)
        residual_norm2 = np.einsum("ij,ij->i", residuals, residuals)
        np.testing.assert_allclose(scores**2 / norm2,
                                  [float(r["raw_hz_direction_energy_fraction"]) for r in rows], atol=1e-12)
        np.testing.assert_allclose(residuals @ direction, 0, atol=1e-10)
        np.testing.assert_allclose(norm2, residual_norm2 + scores**2, rtol=1e-12)
        removed_fraction = float(np.sum(scores**2) / np.sum(norm2))
        np.testing.assert_allclose(removed_fraction,
            source_index["metrics"]["raw_hz"]["embedding_variance_fraction_along_direction"], atol=1e-12)
        print(f"Removed {removed_fraction:.4%} of variance; perpendicularity and saved fractions verified", flush=True)
        palettes = categorical_palettes(rows)
        joint = np.empty((len(rows), 2))
        within = np.empty_like(joint)
        plots = ["joint_umap_panels.png", "visual_behavior_umap_panels.png", "visual_coding_umap_panels.png"]
        selections = [(None, "Both datasets"), *DATASET_LABELS.items()]
        for plot, (dataset, label) in zip(plots, selections):
            mask = np.ones(len(rows), dtype=bool) if dataset is None else np.array(
                [r["dataset_id"] == dataset for r in rows])
            print(f"Fitting {label}: {mask.sum():,} neurons, one CPU thread", flush=True)
            coordinates = UMAP(n_components=2, random_state=seed, transform_seed=seed,
                               n_jobs=1, init="random", **umap_config).fit_transform(residuals[mask])
            if not np.isfinite(coordinates).all():
                raise ValueError("Nonfinite direction-removed UMAP coordinates")
            if dataset is None:
                joint[:] = coordinates
            else:
                within[mask] = coordinates
            plot_overview(stage / plot, coordinates, [r for r, yes in zip(rows, mask) if yes],
                          palettes, rows, label + " · FR direction removed", dataset is None)
            print(f"Rendered {label}", flush=True)
        fields = ["unit_uid", "dataset_id", "session_uid", "subject_uid", "probe_uid", "cell_type",
                  "cell_type_source", "brain_region", "firing_rate_hz", "population_mean_firing_rate_hz",
                  "population_size", "probe_population_size"]
        coordinate_fields = ["joint_umap_1", "joint_umap_2", "within_dataset_umap_1", "within_dataset_umap_2"]
        with (stage / "coordinates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + coordinate_fields)
            writer.writeheader()
            for row, xy, within_xy in zip(rows, joint, within):
                record = {field: row[field] for field in fields}
                record.update(zip(coordinate_fields, [*xy, *within_xy]))
                writer.writerow(record)
        write_json(stage / "index.json", {"plots": plots, "coordinates": "coordinates.csv",
                   "neurons": len(rows), "removed_variance_fraction": removed_fraction,
                   "remaining_variance_fraction": float(residual_norm2.sum() / norm2.sum()),
                   "maximum_remaining_direction_coordinate": float(np.abs(residuals @ direction).max()),
                   "source_analysis": source_analysis, "pooling": source_index["pooling"],
                   "interpretation": "The same full-fit raw-Hz decoder direction is removed for all neurons. "
                   "Each UMAP is separately fitted. Colors retain measured firing rates and original labels. "
                   "Removing this direction does not establish firing-rate independence."})
    return aid
