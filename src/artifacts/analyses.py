"""Reusable CPU analyses. Probe preprocessing is fitted on training rows only."""

import csv
import json

import numpy as np

from .collections import load_unit_means
from .store import implementation_stamp, write_json


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(store, collection_ids, config, label=""):
    kind = config["kind"]
    if kind == "visualization":
        from .visualization import visualize
        return visualize(store, collection_ids, config, label)
    allowed = {
        "pca": {"kind", "color_by", "max_rows"},
        "linear_probe": {"kind", "target", "group_by", "test_fraction", "seed", "C",
                         "max_iter", "exclude_labels", "max_rows", "test_groups"},
    }
    if kind not in allowed or set(config) - allowed[kind]:
        raise ValueError("Unknown analysis kind or options")
    collection_ids = sorted(collection_ids)
    if len(set(collection_ids)) != len(collection_ids):
        raise ValueError("Duplicate input collections")
    for cid in collection_ids:
        store.read("collections", cid)
    recipe = {"operation": kind + "_v1", "collections": collection_ids,
              "config": config, "implementation": implementation_stamp()}
    parents = [{"kind": "collections", "id": cid} for cid in collection_ids]
    with store.build("analyses", recipe, label, parents) as (stage, artifact_id):
        if stage is None:
            return artifact_id
        x, rows = load_unit_means(store, collection_ids, config.get("max_rows", 100000))
        if kind == "pca":
            from sklearn.decomposition import PCA
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if min(x.shape) < 2:
                raise ValueError("PCA requires at least two rows and two embedding dimensions")
            color = config.get("color_by", "dataset_id")
            colors = np.array([str(r[color]) for r in rows])
            pca = PCA(n_components=2, svd_solver="full")
            coordinates = pca.fit_transform(x)
            np.savez(stage / "pca.npz", coordinates=coordinates, components=pca.components_, mean=pca.mean_)
            write_rows(stage / "coordinates.csv", [dict(r, pc1=float(v[0]), pc2=float(v[1])) for r, v in zip(rows, coordinates)])
            ratios = [float(v) if np.isfinite(v) else None for v in pca.explained_variance_ratio_]
            write_json(stage / "metrics.json", {"rows": len(rows), "explained_variance_ratio": ratios,
                                               "fit_scope": "all_selected_rows_descriptive_only"})
            fig, ax = plt.subplots(figsize=(7, 5))
            for value in sorted(set(colors)):
                selected = colors == value
                ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=5, alpha=0.6, label=value)
            ax.set(xlabel="PC 1", ylabel="PC 2", title="NuCLR unit embeddings")
            if len(set(colors)) <= 20:
                ax.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.savefig(stage / "pca.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
            from sklearn.model_selection import GroupShuffleSplit
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            target = config.get("target", "brain_region")
            group_by = config.get("group_by", "subject_uid")
            if group_by not in ("subject_uid", "session_uid"):
                raise ValueError("Probe holdouts must be grouped by subject_uid or session_uid")
            excluded = config.get("exclude_labels", ["", "root", "void"])
            keep = np.array([str(r.get(target, "")) not in excluded for r in rows])
            excluded_count = int((~keep).sum())
            x, rows = x[keep], [r for r, yes in zip(rows, keep) if yes]
            y = np.array([str(r[target]) for r in rows])
            groups = np.array([r[group_by] for r in rows])
            if len(set(groups)) < 2:
                raise ValueError("A held-out probe needs at least two independent groups")
            seed = config.get("seed", 42)
            if "test_groups" in config:
                requested = set(config["test_groups"])
                if not requested or requested - set(groups):
                    raise ValueError("test_groups must be a nonempty subset of the selected groups")
                test = np.flatnonzero(np.isin(groups, list(requested)))
                train = np.flatnonzero(~np.isin(groups, list(requested)))
            else:
                splitter = GroupShuffleSplit(n_splits=1, test_size=config.get("test_fraction", 0.2), random_state=seed)
                train, test = next(splitter.split(x, y, groups))
            if not len(train) or not len(test) or len(set(y[train])) < 2:
                raise ValueError("Need nonempty train/test groups and at least two training classes")
            pipeline = make_pipeline(StandardScaler(), LogisticRegression(
                C=config.get("C", 1.0), max_iter=config.get("max_iter", 1000),
                random_state=seed, class_weight="balanced"))
            pipeline.fit(x[train], y[train])
            predicted = pipeline.predict(x[test])
            model, scaler = pipeline[-1], pipeline[0]
            classes, counts = np.unique(y[train], return_counts=True)
            majority = classes[counts.argmax()]
            write_json(stage / "metrics.json", {
                "accuracy": float(accuracy_score(y[test], predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
                "macro_f1": float(f1_score(y[test], predicted, average="macro", zero_division=0)),
                "majority_accuracy": float(np.mean(y[test] == majority)),
                "train_rows": len(train), "test_rows": len(test), "excluded_rows": excluded_count,
                "unseen_test_labels": sorted(set(y[test]) - set(y[train])),
                "reached_iteration_limit": bool((model.n_iter_ >= model.max_iter).any()),
                "pretraining_exposure": "See checkpoint lineage; probe holdout alone does not establish pretraining holdout",
            })
            write_json(stage / "split.json", {
                "group_by": group_by, "seed": seed,
                "train_groups": sorted(set(groups[train])), "test_groups": sorted(set(groups[test])),
                "train_unit_uids": [rows[i]["unit_uid"] for i in train],
                "test_unit_uids": [rows[i]["unit_uid"] for i in test],
            })
            np.savez(stage / "linear_probe.npz", coefficients=model.coef_, intercept=model.intercept_,
                     classes=model.classes_, mean=scaler.mean_, scale=scaler.scale_)
            write_rows(stage / "predictions.csv", [dict(rows[i], target=str(y[i]), prediction=str(p))
                                                    for i, p in zip(test, predicted)])
    return artifact_id
