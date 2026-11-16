"""Grouped CPU visualizations with reusable fits and explicit display cohorts."""

import csv
import json
import re

import numpy as np

from .collections import load_unit_means
from .store import implementation_stamp, write_json


SELECTORS = {"dataset_id", "subject_uid", "subject_id", "session_uid", "session_id",
             "brain_region", "cell_type", "split"}


def _selection(value):
    if not isinstance(value, dict) or set(value) - SELECTORS:
        raise ValueError(f"Selections must map metadata fields to lists; fields: {sorted(SELECTORS)}")
    for field, values in value.items():
        if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
            raise ValueError(f"Selection {field} must be a nonempty list of strings")
    return {key: sorted(set(values)) for key, values in value.items()}


def _matches(row, selection):
    return all(str(row.get(field)) in values for field, values in selection.items())


def _normalize(config):
    allowed = {"kind", "methods", "fit_selection", "show_selection", "colors",
               "standardize", "seed", "umap", "max_rows"}
    if set(config) - allowed:
        raise ValueError("Unknown visualization options")
    cfg = dict(config)
    methods = cfg.setdefault("methods", ["pca", "umap"])
    if not isinstance(methods, list) or not methods or set(methods) - {"pca", "umap"} or len(set(methods)) != len(methods):
        raise ValueError("methods must select pca and/or umap without duplicates")
    cfg["methods"] = sorted(methods)
    cfg["fit_selection"] = _selection(cfg.get("fit_selection", {}))
    show = cfg.get("show_selection")
    cfg["show_selection"] = _selection(cfg["fit_selection"] if show is None else show)
    cfg.setdefault("standardize", False)
    cfg.setdefault("seed", 42)
    cfg.setdefault("max_rows", 100000)
    if not isinstance(cfg["standardize"], bool) or not isinstance(cfg["seed"], int):
        raise ValueError("standardize must be boolean and seed must be an integer")
    if not isinstance(cfg["max_rows"], int) or cfg["max_rows"] < 1:
        raise ValueError("max_rows must be a positive integer")
    options = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}
    supplied = cfg.get("umap", {})
    if not isinstance(supplied, dict) or set(supplied) - {"n_neighbors", "min_dist", "metric", "n_epochs"}:
        raise ValueError("Unknown UMAP options")
    options.update(supplied)
    if not isinstance(options["n_neighbors"], int) or options["n_neighbors"] < 2:
        raise ValueError("UMAP n_neighbors must be an integer >= 2")
    if not 0 <= options["min_dist"] <= 1:
        raise ValueError("UMAP min_dist must lie between 0 and 1")
    if "n_epochs" in options and (not isinstance(options["n_epochs"], int) or options["n_epochs"] < 10):
        raise ValueError("UMAP n_epochs must be an integer >= 10")
    cfg["umap"] = options if "umap" in methods else {}
    colors = cfg.get("colors", [{"field": "brain_region", "type": "categorical", "palette": "tab20"}])
    if not isinstance(colors, list) or not colors:
        raise ValueError("colors must be a nonempty list")
    cfg["colors"] = []
    for color in colors:
        if not isinstance(color, dict) or set(color) - {"field", "type", "palette"}:
            raise ValueError("Each color accepts field, type, and palette")
        if not isinstance(color.get("field"), str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", color["field"]):
            raise ValueError("Each color needs a metadata field name")
        kind = color.get("type", "categorical")
        if kind not in {"categorical", "continuous"}:
            raise ValueError("Color type must be categorical or continuous")
        cfg["colors"].append(dict(field=color["field"], type=kind,
                                  palette=color.get("palette", "viridis" if kind == "continuous" else "tab20")))
    return cfg


def _plot(path, coordinates, rows, color, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(color["palette"])
    values = [r.get(color["field"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    legend = {}
    try:
        if color["type"] == "continuous":
            try:
                values = np.array([np.nan if v is None or v == "" else float(v) for v in values])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Continuous color {color['field']} requires numeric metadata") from error
            known = np.isfinite(values)
            if known.any():
                dots = ax.scatter(*coordinates[known].T, c=values[known], cmap=cmap, s=6, alpha=.7)
                fig.colorbar(dots, ax=ax, label=color["field"])
                legend.update(min=float(values[known].min()), max=float(values[known].max()))
        else:
            known = np.array([v is not None and str(v) not in ("", "unknown") for v in values])
            categories = sorted({str(v) for v, yes in zip(values, known) if yes})
            labels = np.array([str(v) for v in values])
            from matplotlib.colors import to_hex
            palette = cmap.resampled(max(1, len(categories)))
            legend["category_colors"] = {}
            for i, category in enumerate(categories):
                mask = known & (labels == category)
                rgba = palette(i)
                ax.scatter(*coordinates[mask].T, color=rgba, s=6, alpha=.7, label=category)
                legend["category_colors"][category] = to_hex(rgba)
        if (~known).any():
            ax.scatter(*coordinates[~known].T, color="#aaaaaa", s=6, alpha=.7, label="unknown")
        legend["unknown_rows"] = int((~known).sum())
        if color["type"] == "categorical" or (~known).any():
            ax.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        prefix = "PC" if method == "pca" else "UMAP"
        ax.set(xlabel=f"{prefix} 1", ylabel=f"{prefix} 2", title=f"{method.upper()} — {color['field']}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    return legend


def visualize(store, collection_ids, config, label=""):
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cfg = _normalize(config)
    collection_ids = sorted(collection_ids)
    if not collection_ids or len(set(collection_ids)) != len(collection_ids):
        raise ValueError("Provide nonempty, unique input collections")
    for cid in collection_ids:
        store.read("collections", cid)
    stamp = implementation_stamp()
    parents = [{"kind": "collections", "id": cid} for cid in collection_ids]
    fit_recipe = {"operation": "visualization_projection_v1", "collections": collection_ids,
                  "config": {key: cfg[key] for key in ("methods", "fit_selection", "standardize", "seed", "umap")},
                  "implementation": stamp}
    projection_id = store.identity("analyses", fit_recipe)
    recipe = {"operation": "visualization_v1", "collections": collection_ids,
              "projection": projection_id, "config": cfg, "implementation": stamp}
    # Check selectors against all metadata while loading only the fit/show union.
    observed = {field: set() for selection in (cfg["fit_selection"], cfg["show_selection"]) for field in selection}

    def selected(row):
        for field in observed:
            if field in row:
                observed[field].add(str(row[field]))
        return _matches(row, cfg["fit_selection"]) or _matches(row, cfg["show_selection"])

    with store.build("analyses", recipe, label, parents + [{"kind": "analyses", "id": projection_id}]) as (stage, aid):
        if stage is None:
            store.read("analyses", projection_id)
            return aid
        x, rows = load_unit_means(store, collection_ids, cfg["max_rows"], row_filter=selected)
        for selection in (cfg["fit_selection"], cfg["show_selection"]):
            for field, values in selection.items():
                missing = set(values) - observed[field]
                if missing:
                    raise ValueError(f"Unknown selection values for {field}: {sorted(missing)}")
        fit = np.array([_matches(r, cfg["fit_selection"]) for r in rows])
        show = np.array([_matches(r, cfg["show_selection"]) for r in rows])
        if fit.sum() < 2 or not show.any() or x.shape[1] < 2:
            raise ValueError("Need at least two fit rows, two dimensions, and one displayed row")
        if "umap" in cfg["methods"] and cfg["umap"]["n_neighbors"] >= fit.sum():
            raise ValueError("UMAP n_neighbors must be smaller than the number of fit rows")
        shown_rows = [r for r, yes in zip(rows, show) if yes]
        for color in cfg["colors"]:
            if not any(color["field"] in r for r in shown_rows):
                raise ValueError(f"Missing metadata field {color['field']}; re-extract older collections if needed")
        fit_uids = [r["unit_uid"] for r, yes in zip(rows, fit) if yes]
        with store.build("analyses", fit_recipe, "Visualization projection", parents) as (fit_stage, _):
            if fit_stage is not None:
                scaler = StandardScaler().fit(x[fit]) if cfg["standardize"] else None
                fit_x = scaler.transform(x[fit]) if scaler is not None else x[fit]
                models, coordinates, metrics = {}, {}, {}
                for method in cfg["methods"]:
                    if method == "pca":
                        model = PCA(n_components=2, svd_solver="full")
                    else:
                        from umap import UMAP
                        model = UMAP(n_components=2, random_state=cfg["seed"], transform_seed=cfg["seed"],
                                     n_jobs=1, init="random", **cfg["umap"])
                    coordinates[method] = model.fit_transform(fit_x)
                    if not np.isfinite(coordinates[method]).all():
                        raise ValueError(f"Nonfinite {method} fit coordinates")
                    models[method] = model
                    if method == "pca":
                        metrics[method] = {"explained_variance_ratio": [float(v) if np.isfinite(v) else None
                                                                       for v in model.explained_variance_ratio_]}
                        np.savez(fit_stage / "pca_basis.npz", components=model.components_, mean=model.mean_)
                joblib.dump({"models": models, "scaler": scaler}, fit_stage / "models.joblib")
                np.savez(fit_stage / "fit_coordinates.npz", **coordinates)
                write_json(fit_stage / "fit_unit_uids.json", fit_uids)
                write_json(fit_stage / "metrics.json", metrics)
        # Models are internal artifacts; checksum validation precedes deserialization.
        store.read("analyses", projection_id)
        folder = store.path("analyses", projection_id)
        if json.loads((folder / "fit_unit_uids.json").read_text()) != fit_uids:
            raise ValueError("Cached projection cohort does not match requested fit")
        payload = joblib.load(folder / "models.joblib")
        scaler = payload["scaler"]
        display_x = scaler.transform(x[show]) if scaler is not None else x[show]
        positions = {uid: i for i, uid in enumerate(fit_uids)}
        in_fit = np.array([r["unit_uid"] in positions for r in shown_rows])
        fit_positions = [positions[r["unit_uid"]] for r in shown_rows if r["unit_uid"] in positions]
        index = {"projection_id": projection_id, "methods": {}}
        with np.load(folder / "fit_coordinates.npz") as fit_coordinates:
            for method, model in payload["models"].items():
                output = stage / method
                output.mkdir()
                coordinates = np.empty((len(shown_rows), 2))
                coordinates[in_fit] = fit_coordinates[method][fit_positions]
                if (~in_fit).any():
                    coordinates[~in_fit] = model.transform(display_x[~in_fit])
                if not np.isfinite(coordinates).all():
                    raise ValueError(f"Nonfinite {method} display coordinates")
                table = [dict(row, coordinate_1=float(v[0]), coordinate_2=float(v[1]), in_fit=bool(yes))
                         for row, v, yes in zip(shown_rows, coordinates, in_fit)]
                with (output / "coordinates.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted({key for row in table for key in row}))
                    writer.writeheader()
                    writer.writerows(table)
                plots = []
                for i, color in enumerate(cfg["colors"]):
                    filename = f"{i:02d}_{color['field']}.png"
                    legend = _plot(output / filename, coordinates, shown_rows, color, method)
                    plots.append(dict(color, file=f"{method}/{filename}", **legend))
                index["methods"][method] = {"coordinates": f"{method}/coordinates.csv", "plots": plots}
        write_json(stage / "index.json", index)
        write_json(stage / "selection.json", {"fit_unit_uids": fit_uids,
                   "shown_unit_uids": [r["unit_uid"] for r in shown_rows],
                   "fit_selection": cfg["fit_selection"], "show_selection": cfg["show_selection"]})
        write_json(stage / "metrics.json", {"fit_rows": int(fit.sum()), "shown_rows": int(show.sum()),
                   "shown_rows_outside_fit": int((~in_fit).sum()),
                   "projection": json.loads((folder / "metrics.json").read_text()),
                   "interpretation": "Descriptive visualization; fitting separately does not establish pretraining holdout"})
    return aid
