"""Aligned metadata panels for joint and separately fitted neuron UMAPs."""

from collections import Counter, defaultdict
import colorsys
import csv
import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .collections import load_unit_means
from .store import file_digest, implementation_stamp, write_json


DATASET_LABELS = {"allen_visual_behavior": "Visual Behavior", "allen_visual_coding": "Visual Coding",
                  "ibl_brain_wide_bench_2026": "IBL"}
CATEGORICAL_FIELDS = ["dataset_id", "subject_uid", "cell_type", "brain_region", "probe_uid"]
PANELS = [
    ("dataset_id", "Dataset"),
    ("firing_rate_hz", "Neuron firing rate"),
    ("subject_uid", "Animal"),
    ("cell_type", "Cell type"),
    ("brain_region", "Brain region"),
    ("population_mean_firing_rate_hz", "Population mean firing rate"),
    ("population_size", "Recording population size"),
    ("probe_uid", "Probe"),
    ("probe_population_size", "Neurons in the same probe"),
]


def overview_datasets(rows):
    present = {row["dataset_id"] for row in rows}
    if not present or present - DATASET_LABELS.keys():
        raise ValueError("Expected supported Allen/IBL datasets")
    return {dataset: label for dataset, label in DATASET_LABELS.items() if dataset in present}


def add_population_metadata(rows):
    """Population means/counts use all retained units in each recording/probe."""
    rows = [dict(row) for row in rows]
    if len({row["unit_uid"] for row in rows}) != len(rows):
        raise ValueError("Duplicate neuron IDs in the plotting cohort")
    sizes = Counter(row["session_uid"] for row in rows)
    probe_sizes = Counter(row["probe_uid"] for row in rows)
    rate_sums, probe_sessions = defaultdict(float), {}
    for row in rows:
        rate = float(row["firing_rate_hz"])
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("Neuron firing rates must be finite and nonnegative")
        row["firing_rate_hz"] = rate
        rate_sums[row["session_uid"]] += rate
        session = probe_sessions.setdefault(row["probe_uid"], row["session_uid"])
        if session != row["session_uid"]:
            raise ValueError("Probe IDs must be namespaced by recording")
    for row in rows:
        row["population_size"] = sizes[row["session_uid"]]
        row["population_mean_firing_rate_hz"] = rate_sums[row["session_uid"]] / row["population_size"]
        row["probe_population_size"] = probe_sizes[row["probe_uid"]]
        row["cell_type"] = row.get("cell_type") or "Unknown"
    return rows


def categorical_palettes(rows):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    palettes = {}
    for field in CATEGORICAL_FIELDS:
        categories = sorted({str(row[field]) for row in rows})
        if field == "dataset_id":
            palette = {"allen_visual_behavior": "#3b4cc0", "allen_visual_coding": "#b40426",
                       "ibl_brain_wide_bench_2026": "#13866e"}
        elif field == "cell_type":
            palette = {"Unknown": "#c5cad0", "Unlabeled": "#c5cad0", "Sst": "#1565c0",
                       "Vip": "#d81b60", "Pvalb": "#2e7d32"}
            if set(categories) - palette.keys():
                raise ValueError("Unrecognized cell-type label; specify its display color")
        elif field == "brain_region" and len(categories) <= 20:
            palette = {value: to_hex(plt.get_cmap("tab20")(i)) for i, value in enumerate(categories)}
        else:
            # Nonmonotonic hues avoid implying an order in animal or probe IDs.
            palette = {value: to_hex(colorsys.hsv_to_rgb(
                (i * 0.618033988749895) % 1,
                [0.72, 0.90, 0.57, 0.82][i % 4],
                [0.80, 0.63, 0.96, 0.91][(i // 4) % 4],
            )) for i, value in enumerate(categories)}
        palettes[field] = {value: palette[value] for value in categories}
    return palettes


def plot_overview(path, coordinates, rows, palettes, all_rows, title, include_dataset, axis_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm
    from matplotlib.lines import Line2D

    panels = PANELS if include_dataset else PANELS[1:]
    shape = (3, 3) if include_dataset else (2, 4)
    figsize = (22, 19) if include_dataset else (28, 13)
    fig, axes = plt.subplots(*shape, figsize=figsize, layout="constrained")
    fig.suptitle(f"{title}  ·  {len(rows):,} neurons", fontsize=22, fontweight="medium")
    order = np.random.default_rng(42).permutation(len(rows))
    minimum, maximum = coordinates.min(axis=0), coordinates.max(axis=0)
    padding = np.maximum(maximum - minimum, 1.) * .05
    for ax, (field, panel_title) in zip(axes.flat, panels):
        values = np.array([row[field] for row in rows])
        if field in palettes:
            colors = np.array([palettes[field][str(value)] for value in values])
            categories = sorted(set(values))
            if field == "cell_type":
                unknown = np.isin(values, ["Unknown", "Unlabeled"])
                ax.scatter(*coordinates[unknown].T, c=colors[unknown], s=2.5, alpha=.55, linewidths=0)
                known = order[~unknown[order]]
                ax.scatter(*coordinates[known].T, c=colors[known], s=14, alpha=.95, linewidths=0)
            else:
                ax.scatter(*coordinates[order].T, c=colors[order], s=3, alpha=.85, linewidths=0)
            if field in {"subject_uid", "probe_uid"} or (field == "brain_region" and len(categories) > 65):
                # Detailed IBL region labels are too numerous for an inline legend;
                # the complete, shared color mapping is saved with the coordinates.
                panel_title += f" ({len(categories):,})"
            else:
                handles = [Line2D([], [], marker="o", linestyle="none", markersize=6,
                                  color=palettes[field][value],
                                  label=DATASET_LABELS.get(value, value)) for value in categories]
                columns = max(1, int(np.ceil(len(categories) / 22))) if field == "brain_region" else 1
                ax.legend(handles=handles, frameon=False, fontsize=7 if columns > 1 else 9, ncols=columns,
                          columnspacing=.7, handletextpad=.3, labelspacing=.35,
                          loc="upper left", bbox_to_anchor=(1.01, 1.), borderaxespad=0.)
        else:
            global_values = np.array([row[field] for row in all_rows], dtype=float)
            if field == "firing_rate_hz":
                if (global_values == 0).any():
                    norm = SymLogNorm(linthresh=.01, vmin=0, vmax=max(global_values.max(), .01))
                    label = "Hz (linear below 0.01; log above)"
                else:
                    norm = LogNorm(global_values.min(), global_values.max())
                    label = "Hz (log scale)"
                cmap = "viridis"
            else:
                norm = Normalize(global_values.min(), global_values.max())
                label = "Hz / neuron" if field == "population_mean_firing_rate_hz" else "Neurons"
                cmap = "viridis" if field == "population_mean_firing_rate_hz" else "cividis"
            points = ax.scatter(*coordinates[order].T, c=values[order].astype(float),
                                cmap=cmap, norm=norm, s=3, alpha=.85, linewidths=0)
            bar = fig.colorbar(points, ax=ax, fraction=.035, pad=.025, shrink=.82)
            bar.set_label(label, fontsize=10)
            bar.ax.tick_params(labelsize=9)
        xlabel, ylabel = axis_labels or ("UMAP 1", "UMAP 2")
        ax.set(title=panel_title, xlabel=xlabel, ylabel=ylabel,
               xlim=(minimum[0] - padding[0], maximum[0] + padding[0]),
               ylim=(minimum[1] - padding[1], maximum[1] + padding[1]))
        ax.title.set_fontsize(15)
        ax.set_aspect("equal", adjustable="box")
        if axis_labels is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#b8bec6")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def analyze_embedding_overview(store, source_analysis, labels_path=None):
    source = store.read("analyses", source_analysis)
    cfg = source["recipe"]["config"]
    collection_ids = source["recipe"]["collections"]
    if (source["recipe"]["operation"] != "visualization_v1" or len(collection_ids) != 1
            or cfg["fit_selection"] or cfg["show_selection"] or cfg["standardize"]):
        raise ValueError("Use the original joint UMAP of one complete, unstandardized collection")
    recipe = {"operation": "embedding_metadata_overview_v1", "source_analysis": source_analysis,
              "collections": collection_ids, "joint_coordinates": "reused",
              "dataset_coordinates": "separate_umap_fits", "umap": cfg["umap"], "seed": cfg["seed"],
              "metadata": "population=all_retained_recording_units;probe=unique_recording_probe",
              "implementation": implementation_stamp()}
    annotation_source = None
    if labels_path is not None:
        labels_path = Path(labels_path).expanduser().resolve(strict=True)
        annotation_source = {"path": str(labels_path), "sha256": file_digest(labels_path),
                             "column": "lolcat_celltype", "wt_display": "Unlabeled"}
        recipe["coding_annotations"] = annotation_source
    parents = [{"kind": "analyses", "id": source_analysis},
               {"kind": "collections", "id": collection_ids[0]}]
    with store.build("analyses", recipe, "Joint and within-dataset UMAP panels", parents) as (stage, aid):
        if stage is None:
            return aid
        embeddings, original_rows = load_unit_means(store, collection_ids, max_rows=cfg["max_rows"])
        rows = add_population_metadata(original_rows)
        if labels_path is not None:
            rows = add_coding_labels(rows, labels_path)
            if file_digest(labels_path) != annotation_source["sha256"]:
                raise ValueError("Coding annotations changed while being read")
        datasets = overview_datasets(rows)
        source_folder = store.path("analyses", source_analysis)
        source_index = json.loads((source_folder / "index.json").read_text())
        with (source_folder / source_index["methods"]["umap"]["coordinates"]).open() as stream:
            saved = list(csv.DictReader(stream))
        by_uid = {row["unit_uid"]: row for row in saved}
        if len(by_uid) != len(saved) or set(by_uid) != {row["unit_uid"] for row in rows}:
            raise ValueError("Joint coordinates and embedding neurons do not match")
        joint = np.array([[float(by_uid[row["unit_uid"]][f"coordinate_{axis}"]) for axis in (1, 2)] for row in rows])
        np.testing.assert_array_equal([row["firing_rate_hz"] for row in rows],
                                      [float(by_uid[row["unit_uid"]]["firing_rate_hz"]) for row in rows])
        if not np.isfinite(joint).all():
            raise ValueError("Nonfinite joint coordinates")
        within = np.empty_like(joint)
        palettes = categorical_palettes(rows)
        plots = ["joint_umap_panels.png"] + [f"{dataset.removeprefix('allen_')}_umap_panels.png"
                                          for dataset in datasets]
        plot_overview(stage / plots[0], joint, rows, palettes, rows, "Combined datasets", True)
        print(f"Rendered joint UMAP: {len(rows):,} neurons, original coordinates", flush=True)
        from umap import UMAP

        for plot_name, (dataset, label) in zip(plots[1:], datasets.items()):
            selected = np.array([row["dataset_id"] == dataset for row in rows])
            print(f"Fitting {label} UMAP: {selected.sum():,} neurons, one CPU thread", flush=True)
            coordinates = UMAP(n_components=2, random_state=cfg["seed"], transform_seed=cfg["seed"],
                               n_jobs=1, init="random", **cfg["umap"]).fit_transform(embeddings[selected])
            if not np.isfinite(coordinates).all():
                raise ValueError("Nonfinite within-dataset UMAP coordinates")
            within[selected] = coordinates
            plot_overview(stage / plot_name, coordinates, [row for row, yes in zip(rows, selected) if yes],
                          palettes, rows, label, False)
            print(f"Rendered {label} panels", flush=True)
        metadata_fields = ["unit_uid", "dataset_id", "session_uid", "subject_uid", "probe_uid", "cell_type",
                  "brain_region", "firing_rate_hz", "population_mean_firing_rate_hz", "population_size",
                  "probe_population_size"]
        if annotation_source is not None:
            metadata_fields += ["lolcat_celltype", "cell_type_source"]
        fields = metadata_fields + ["joint_umap_1", "joint_umap_2", "within_dataset_umap_1", "within_dataset_umap_2"]
        with (stage / "coordinates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row, joint_xy, within_xy in zip(rows, joint, within):
                record = {field: row[field] for field in metadata_fields}
                record.update(joint_umap_1=joint_xy[0], joint_umap_2=joint_xy[1],
                              within_dataset_umap_1=within_xy[0], within_dataset_umap_2=within_xy[1])
                writer.writerow(record)
        with (stage / "category_colors.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["field", "category", "color"])
            writer.writeheader()
            writer.writerows(dict(field=field, category=category, color=color)
                             for field, palette in palettes.items() for category, color in palette.items())
        write_json(stage / "index.json", {"plots": plots, "coordinates": "coordinates.csv",
                   "category_colors": "category_colors.csv", "neurons": len(rows),
                   "datasets": dict(Counter(row["dataset_id"] for row in rows)),
                   "cell_types": dict(Counter(row["cell_type"] for row in rows)),
                   "coding_annotations": annotation_source,
                   "cell_types_by_dataset": {dataset: dict(Counter(row["cell_type"] for row in rows
                                                                     if row["dataset_id"] == dataset))
                                             for dataset in datasets},
                   "population_definition": "Retained neurons in one recording, across its probes",
                   "probe_definition": "Unique recording/probe identity; no reuse of probe slots across sessions",
                   "pooling": {source["dataset_id"]: source["spec"] for source in
                               store.read("collections", collection_ids[0], verify=False)["recipe"]["sources"]},
                   "source_analysis": source_analysis})
    return aid


def add_coding_labels(rows, labels_path):
    """Join explicit LOLCAT annotations by session/unit; preserve Behavior labels."""
    with Path(labels_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"session_id", "unit_id", "probe_id", "ecephys_structure_acronym", "lolcat_celltype"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Coding annotation CSV is missing required fields")
        annotations = {}
        for annotation in reader:
            key = (annotation["session_id"], annotation["unit_id"])
            if key in annotations:
                raise ValueError(f"Duplicate annotation for session/unit {key}")
            annotations[key] = annotation
    result = []
    for original in rows:
        row = dict(original)
        row["lolcat_celltype"] = ""
        row["cell_type_source"] = ("not_available" if row["dataset_id"] == "ibl_brain_wide_bench_2026"
                                   else "source_collection_optotagging")
        if row["cell_type"] in (None, "", "Unknown"):
            row["cell_type"] = "Unlabeled"
        if row["dataset_id"] == "allen_visual_coding":
            local_id = unquote(row["unit_uid"].rsplit("/", 1)[-1])
            parts = local_id.split("/")
            if len(parts) == 3 and all(part.isdigit() for part in parts):
                if parts[0] != unquote(row["probe_uid"].rsplit("/", 1)[-1]):
                    raise ValueError("Coding unit-ID probe prefix does not match probe metadata")
                local_id = parts[-1]
            elif not local_id.isdigit():
                raise ValueError("Expected a numeric Coding unit ID or probe/channel/unit ID")
            key = (unquote(row["session_uid"].rsplit("/", 1)[-1]),
                   local_id)
            if key not in annotations:
                raise ValueError(f"Missing Coding annotation for session/unit {key}")
            annotation = annotations[key]
            if (annotation["probe_id"] != unquote(row["probe_uid"].rsplit("/", 1)[-1])
                    or annotation["ecephys_structure_acronym"] != row["brain_region"]):
                raise ValueError(f"Coding annotation probe/region mismatch for {key}")
            label = annotation["lolcat_celltype"]
            if label not in {"wt", "Sst", "Vip", "Pvalb"}:
                raise ValueError(f"Unrecognized LOLCAT label: {label!r}")
            row["lolcat_celltype"] = label
            row["cell_type"] = "Unlabeled" if label == "wt" else label
            row["cell_type_source"] = "coding_csv_lolcat_celltype"
        result.append(row)
    return result


def recolor_embedding_overview(store, source_analysis, labels_path):
    """Render revised annotations using the saved joint and dataset coordinates."""
    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "embedding_metadata_overview_v1":
        raise ValueError("Recoloring requires a saved joint/within-dataset overview")
    labels_path = Path(labels_path).expanduser().resolve(strict=True)
    annotation_source = {"path": str(labels_path), "sha256": file_digest(labels_path),
                         "column": "lolcat_celltype", "wt_display": "Unlabeled"}
    recipe = {"operation": "embedding_metadata_overview_recolor_v1", "source_analysis": source_analysis,
              "collections": source["recipe"]["collections"], "coordinates": "reused_without_refitting",
              "coding_annotations": annotation_source, "implementation": implementation_stamp()}
    parents = [{"kind": "analyses", "id": source_analysis}]
    with store.build("analyses", recipe, "UMAP panels with Coding LOLCAT labels", parents) as (stage, aid):
        if stage is None:
            return aid
        folder = store.path("analyses", source_analysis)
        index = json.loads((folder / "index.json").read_text())
        with (folder / index["coordinates"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len({row["unit_uid"] for row in rows}) != len(rows):
            raise ValueError("Duplicate neuron IDs in the saved coordinates")
        datasets = overview_datasets(rows)
        rows = add_coding_labels(rows, labels_path)
        if file_digest(labels_path) != annotation_source["sha256"]:
            raise ValueError("Coding annotations changed while being read")
        joint = np.array([[float(row[f"joint_umap_{axis}"]) for axis in (1, 2)] for row in rows])
        within = np.array([[float(row[f"within_dataset_umap_{axis}"]) for axis in (1, 2)] for row in rows])
        if not np.isfinite(joint).all() or not np.isfinite(within).all():
            raise ValueError("Nonfinite saved UMAP coordinates")
        palettes = categorical_palettes(rows)
        plots = index["plots"]
        plot_overview(stage / plots[0], joint, rows, palettes, rows, "Combined datasets", True)
        print("Rendered joint panels with Coding LOLCAT labels", flush=True)
        for plot_name, (dataset, label) in zip(plots[1:], datasets.items()):
            selected = np.array([row["dataset_id"] == dataset for row in rows])
            plot_overview(stage / plot_name, within[selected],
                          [row for row, yes in zip(rows, selected) if yes], palettes, rows, label, False)
            print(f"Rendered {label} panels using saved coordinates", flush=True)
        with (stage / "coordinates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (stage / "category_colors.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["field", "category", "color"])
            writer.writeheader()
            writer.writerows(dict(field=field, category=category, color=color)
                             for field, palette in palettes.items() for category, color in palette.items())
        index.update(source_analysis=source_analysis, coding_annotations=annotation_source,
                     cell_types=dict(Counter(row["cell_type"] for row in rows)),
                     cell_types_by_dataset={dataset: dict(Counter(row["cell_type"] for row in rows
                                                                  if row["dataset_id"] == dataset))
                                            for dataset in datasets})
        write_json(stage / "index.json", index)
    return aid


def analyze_combined_pca_overview(store, source_analysis):
    """Fit centered, unscaled PCA to pooled embeddings with the overview's labels."""
    from sklearn.decomposition import PCA

    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] not in {
            "embedding_metadata_overview_v1", "embedding_metadata_overview_recolor_v1"}:
        raise ValueError("Combined PCA requires an existing metadata overview")
    collection_ids = source["recipe"]["collections"]
    recipe = {"operation": "combined_pca_metadata_overview_v1", "source_analysis": source_analysis,
              "collections": collection_ids, "pca": {"n_components": 2, "svd_solver": "full",
              "center": True, "standardize": False, "weighting": "equal_per_neuron"},
              "implementation": implementation_stamp()}
    parents = [{"kind": "analyses", "id": source_analysis}]
    with store.build("analyses", recipe, "Combined PC1/PC2 with metadata panels", parents) as (stage, aid):
        if stage is None:
            return aid
        folder = store.path("analyses", source_analysis)
        source_index = json.loads((folder / "index.json").read_text())
        with (folder / source_index["coordinates"]).open(newline="") as stream:
            saved_rows = list(csv.DictReader(stream))
        by_uid = {row["unit_uid"]: row for row in saved_rows}
        embeddings, embedding_rows = load_unit_means(store, collection_ids, max_rows=source_index["neurons"])
        if len(by_uid) != len(saved_rows) or set(by_uid) != {r["unit_uid"] for r in embedding_rows}:
            raise ValueError("PCA embeddings and annotated neurons do not match")
        rows = [by_uid[row["unit_uid"]] for row in embedding_rows]
        print(f"Fitting combined PCA: {len(rows):,} neurons, {embeddings.shape[1]} dimensions, one CPU thread", flush=True)
        model = PCA(n_components=2, svd_solver="full")
        coordinates = model.fit_transform(embeddings.astype(np.float64))
        variance = model.explained_variance_ratio_
        if not np.isfinite(coordinates).all() or not np.isfinite(variance).all():
            raise ValueError("Nonfinite PCA result")
        palettes = categorical_palettes(rows)
        plot_name = "combined_pc1_pc2_panels.png"
        plot_overview(stage / plot_name, coordinates, rows, palettes, rows, "Combined datasets · PCA", True,
                      axis_labels=tuple(f"PC{i + 1} ({value:.1%} variance)" for i, value in enumerate(variance)))
        fields = [field for field in rows[0] if "umap" not in field] + ["pc1", "pc2"]
        with (stage / "coordinates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row, xy in zip(rows, coordinates):
                writer.writerow({**{field: row[field] for field in fields[:-2]}, "pc1": xy[0], "pc2": xy[1]})
        np.savez(stage / "pca_basis.npz", components=model.components_, mean=model.mean_,
                 explained_variance_ratio=variance)
        (stage / "category_colors.csv").write_bytes((folder / source_index["category_colors"]).read_bytes())
        index = {key: value for key, value in source_index.items() if key not in {"plots", "source_analysis"}}
        index.update(plots=[plot_name], source_analysis=source_analysis,
                     explained_variance_ratio=variance.tolist(), pca=recipe["pca"])
        write_json(stage / "index.json", index)
        print(f"Rendered combined PCA: PC1 {variance[0]:.2%}, PC2 {variance[1]:.2%} variance", flush=True)
    return aid


def analyze_visual_cortex_overview(store, source_analysis):
    """Refit joint UMAP on visual cortical units, retaining full-population metadata."""
    from umap import UMAP

    source = store.read("analyses", source_analysis)
    if source["recipe"]["operation"] != "embedding_metadata_overview_v1":
        raise ValueError("Visual-cortex UMAP requires an existing metadata overview")
    folder = store.path("analyses", source_analysis)
    source_index = json.loads((folder / "index.json").read_text())
    with (folder / source_index["coordinates"]).open(newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    # VISC denotes visceral cortex, including its layer-specific IBL labels.
    regions = sorted({row["brain_region"] for row in saved_rows
                      if row["brain_region"].startswith("VIS") and not row["brain_region"].startswith("VISC")})
    selected = [row for row in saved_rows if row["brain_region"] in regions]
    if len(selected) <= source["recipe"]["umap"]["n_neighbors"]:
        raise ValueError("Not enough visual-cortex neurons for the requested UMAP")
    by_uid = {row["unit_uid"]: row for row in selected}
    if len(by_uid) != len(selected):
        raise ValueError("Duplicate visual-cortex neuron IDs")
    collection_ids = source["recipe"]["collections"]
    recipe = {"operation": "visual_cortex_metadata_overview_v1", "source_analysis": source_analysis,
              "collections": collection_ids, "brain_regions": regions,
              "selection": "VIS-prefixed labels excluding VISC (visceral cortex)",
              "umap": source["recipe"]["umap"], "seed": source["recipe"]["seed"],
              "standardize": False, "coordinates": "refit_on_selected_neurons",
              "metadata": "preserved_from_full_population_overview", "implementation": implementation_stamp()}
    parents = [{"kind": "analyses", "id": source_analysis}]
    with store.build("analyses", recipe, "Visual cortex across datasets", parents) as (stage, aid):
        if stage is None:
            return aid
        embeddings, embedding_rows = load_unit_means(
            store, collection_ids, max_rows=source_index["neurons"],
            row_filter=lambda row: row["brain_region"] in regions)
        if (len(embedding_rows) != len(by_uid)
                or {row["unit_uid"] for row in embedding_rows} != set(by_uid)):
            raise ValueError("Selected embeddings and annotated visual-cortex neurons do not match")
        rows = [by_uid[row["unit_uid"]] for row in embedding_rows]
        print(f"Fitting visual-cortex UMAP: {len(rows):,} neurons, one CPU thread", flush=True)
        model = UMAP(n_components=2, random_state=recipe["seed"], transform_seed=recipe["seed"],
                     n_jobs=1, init="random", **recipe["umap"])
        coordinates = model.fit_transform(embeddings)
        if not np.isfinite(coordinates).all():
            raise ValueError("Nonfinite visual-cortex UMAP coordinates")
        # Keep colors and continuous scales comparable to the complete overview.
        palettes = {field: {} for field in CATEGORICAL_FIELDS}
        with (folder / source_index["category_colors"]).open(newline="") as stream:
            for color in csv.DictReader(stream):
                palettes[color["field"]][color["category"]] = color["color"]
        plot_name = "visual_cortex_umap_panels.png"
        plot_overview(stage / plot_name, coordinates, rows, palettes, saved_rows,
                      "Visual cortex · combined datasets", True)
        fields = [field for field in rows[0] if "umap" not in field] + ["joint_umap_1", "joint_umap_2"]
        with (stage / "coordinates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row, xy in zip(rows, coordinates):
                writer.writerow({**{field: row[field] for field in fields[:-2]},
                                 "joint_umap_1": xy[0], "joint_umap_2": xy[1]})
        (stage / "category_colors.csv").write_bytes((folder / source_index["category_colors"]).read_bytes())
        write_json(stage / "index.json", {
            "plots": [plot_name], "coordinates": "coordinates.csv", "category_colors": "category_colors.csv",
            "neurons": len(rows), "datasets": dict(Counter(row["dataset_id"] for row in rows)),
            "regions_by_dataset": {dataset: dict(Counter(row["brain_region"] for row in rows
                                                         if row["dataset_id"] == dataset))
                                   for dataset in overview_datasets(rows)},
            "population_definition": source_index["population_definition"],
            "probe_definition": source_index["probe_definition"], "pooling": source_index["pooling"],
            "source_analysis": source_analysis})
        print(f"Rendered visual-cortex panels: {dict(Counter(row['dataset_id'] for row in rows))}", flush=True)
    return aid
