"""Matched linear region probes with animal-held-out, within-dataset evaluation."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from .collections import load_unit_means
from .rate_variant_comparison import matched_order
from .store import file_digest, implementation_stamp, write_json


def beryl_gray_mapping(atlas_directory):
    folder = Path(atlas_directory)
    with (folder / "allen_structure_tree.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    by_id = {int(row["id"]): row for row in rows}
    gray_id = next(int(row["id"]) for row in rows if row["acronym"] == "grey")
    root_id = next(int(row["id"]) for row in rows if row["acronym"] == "root")
    selected = set(map(int, np.load(folder / "beryl.npy", allow_pickle=False).ravel()))
    mapping = {}
    for row in rows:
        ancestors = [int(value) for value in row["structure_id_path"].strip("/").split("/") if value]
        candidates = [value for value in ancestors if value in selected]
        mapping[row["acronym"]] = (by_id[candidates[-1]]["acronym"] if candidates and gray_id in ancestors
                                   and candidates[-1] not in {gray_id, root_id} else None)
    return mapping


def region_cohorts(rows, mapping, min_neurons=50, min_animals=5):
    unknown = {row["brain_region"] for row in rows} - mapping.keys()
    if unknown:
        raise ValueError(f"Unknown atlas acronyms: {sorted(unknown)}")
    cohorts, coverage = {}, []
    for dataset in sorted({row["dataset_id"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["dataset_id"] == dataset
                   and row["firing_rate_hz"] > 0 and mapping[row["brain_region"]] is not None]
        counts = Counter(mapping[rows[i]["brain_region"]] for i in indices)
        animals = defaultdict(set)
        for i in indices:
            animals[mapping[rows[i]["brain_region"]]].add(rows[i]["subject_uid"])
        allowed = {region for region in counts if counts[region] >= min_neurons and len(animals[region]) >= min_animals}
        if len(allowed) < 2:
            raise ValueError(f"Too few supported brain regions in {dataset}")
        cohorts[dataset] = np.array([i for i in indices if mapping[rows[i]["brain_region"]] in allowed])
        coverage.extend(dict(dataset=dataset, region=region, neurons=counts[region], animals=len(animals[region]),
                             included=region in allowed) for region in sorted(counts))
    return cohorts, coverage


def animal_folds(labels, groups):
    classes = set(labels)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(labels)), labels, groups))
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Animal leakage")
        if set(labels[train]) != classes:
            raise ValueError("A training fold lacks a region; increase animal support before fitting")
    return folds


def region_probe(x, labels, folds):
    predictions = np.empty(len(labels), dtype=labels.dtype)
    fold_ids = np.empty(len(labels), dtype=int)
    fold_records = []
    priors = np.full(len(set(labels)), 1 / len(set(labels)))
    for fold, (train, test) in enumerate(folds):
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
        model.fit(x[train], labels[train])
        predictions[test] = model.predict(x[test])
        fold_ids[test] = fold
        fold_records.append(dict(fold=fold, test_neurons=len(test), test_regions=len(set(labels[test])),
                                 balanced_accuracy=float(balanced_accuracy_score(labels[test], predictions[test]))))
    return predictions, fold_ids, fold_records


def paired_animal_bootstrap(labels, groups, baseline, candidate, repeats=2000):
    """Resample animals jointly; compare macro recall over classes present in each draw."""
    classes, y = np.unique(labels, return_inverse=True)
    subjects, g = np.unique(groups, return_inverse=True)
    total = np.zeros((len(subjects), len(classes)))
    gain = np.zeros_like(total)
    np.add.at(total, (g, y), 1)
    np.add.at(gain, (g, y), (candidate == labels).astype(float) - (baseline == labels).astype(float))
    weights = np.random.default_rng(42).multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), repeats)
    counts = weights @ total
    differences = np.divide(weights @ gain, counts, out=np.full_like(counts, np.nan), where=counts > 0)
    sampled = np.nanmean(differences, axis=1)
    return np.quantile(sampled, [.025, .975])


def plot_region_scores(path, metrics, datasets):
    import matplotlib.pyplot as plt

    models = ["Firing rate only", "Baseline epoch 10", "Variant epoch 5", "Variant epoch 10"]
    colors = ["#999999", "#3265a8", "#d6a145", "#238a74"]
    names = {"allen_visual_behavior": "Visual Behavior", "allen_visual_coding": "Visual Coding",
             "ibl_brain_wide_bench_2026": "IBL"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5), layout="constrained", sharey=True)
    for ax, dataset in zip(axes, datasets):
        table = {row["model"]: row for row in metrics if row["dataset"] == dataset}
        scores = [table[name]["balanced_accuracy"] * 100 for name in models]
        bars = ax.bar(np.arange(4), scores, color=colors, width=.7)
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
        ax.axhline(100 / table[models[0]]["regions"], color="#777777", linestyle="--", linewidth=1, label="Chance")
        ax.set(title=f"{names[dataset]} · {table[models[0]]['regions']} regions",
               xticks=np.arange(4), xticklabels=["Rate only", "Baseline\nepoch 10", "Variant\nepoch 5", "Variant\nepoch 10"])
        ax.set_ylim(0, max(row["balanced_accuracy"] for row in metrics) * 120)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Held-out-animal balanced accuracy (%)")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def analyze_brain_regions(store, source_comparison, atlas_directory):
    source = store.read("analyses", source_comparison)
    models = source["recipe"]["models"]
    order = ["Baseline epoch 10", "Variant epoch 5", "Variant epoch 10"]
    atlas = Path(atlas_directory).resolve()
    mapping = beryl_gray_mapping(atlas)
    recipe = {"operation": "within_dataset_brain_region_decoding_v1", "source_comparison": source_comparison,
              "models": models, "atlas_source": (atlas / "source.txt").read_text().strip(),
              "atlas_files": {name: file_digest(atlas / name) for name in ("allen_structure_tree.csv", "beryl.npy")},
              "labels": "Beryl; gray matter descendants; exclude root and grey",
              "selection": "positive firing rate; >=50 neurons and >=5 animals per dataset/region",
              "validation": "five_stratified_animal_folds_within_each_dataset_seed42",
              "probe": "linear_discriminant_auto_shrinkage_equal_class_priors_no_tuning",
              "control": "same_linear_classifier_with_raw_firing_rate_only",
              "uncertainty": "2000_paired_animal_bootstrap_draws_on_fixed_OOF_predictions; present_classes_per_draw",
              "implementation": implementation_stamp()}
    with store.build("analyses", recipe, "Held-out-animal brain region decoding",
                     [{"kind": "analyses", "id": source_comparison}]) as (stage, aid):
        if stage is None:
            return aid
        metrics, fold_metrics, per_region, differences = [], [], [], []
        reference, cohorts, targets, groups, folds, predictions, assignments = None, None, {}, {}, {}, {}, {}
        for label in order:
            overview = store.read("analyses", models[label])
            index = json.loads((store.path("analyses", models[label]) / "index.json").read_text())
            x, rows = load_unit_means(store, overview["recipe"]["collections"], max_rows=index["neurons"])
            if reference is None:
                reference = rows
                cohorts, coverage = region_cohorts(rows, mapping)
                for dataset, selected in cohorts.items():
                    targets[dataset] = np.array([mapping[rows[i]["brain_region"]] for i in selected])
                    groups[dataset] = np.array([rows[i]["subject_uid"] for i in selected])
                    folds[dataset] = animal_folds(targets[dataset], groups[dataset])
                    predictions[dataset] = {}
            else:
                x = x[matched_order(reference, rows)]
            for dataset, selected in cohorts.items():
                y = targets[dataset]
                features = [(label, x[selected].astype(np.float64))]
                if label == order[0]:
                    features.insert(0, ("Firing rate only", np.array([[reference[i]["firing_rate_hz"]] for i in selected])))
                for name, values in features:
                    print(f"FIT {name} {dataset}: {len(selected):,} neurons, {len(set(y))} regions, {len(set(groups[dataset]))} animals", flush=True)
                    predicted, fold_ids, fold_stats = region_probe(values, y, folds[dataset])
                    predictions[dataset][name] = predicted
                    assignments[dataset] = fold_ids
                    record = dict(dataset=dataset, model=name, neurons=len(y), regions=len(set(y)),
                                  animals=len(set(groups[dataset])), balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                                  macro_f1=float(f1_score(y, predicted, average="macro")), chance=1 / len(set(y)))
                    metrics.append(record)
                    fold_metrics.extend(dict(dataset=dataset, model=name, **row) for row in fold_stats)
                    classes = np.unique(y)
                    for region, recall in zip(classes, recall_score(y, predicted, labels=classes, average=None, zero_division=0)):
                        per_region.append(dict(dataset=dataset, model=name, region=region, recall=float(recall),
                                               neurons=int(np.sum(y == region)), animals=len(set(groups[dataset][y == region]))))
                    print(json.dumps(record), flush=True)
        for dataset in cohorts:
            baseline = predictions[dataset][order[0]]
            for name in order[1:]:
                candidate = predictions[dataset][name]
                low, high = paired_animal_bootstrap(targets[dataset], groups[dataset], baseline, candidate)
                differences.append(dict(dataset=dataset, model=name,
                    balanced_accuracy_difference=float(balanced_accuracy_score(targets[dataset], candidate) -
                                                       balanced_accuracy_score(targets[dataset], baseline)),
                    bootstrap_low=float(low), bootstrap_high=float(high)))
        for filename, table in (("metrics.csv", metrics), ("fold_metrics.csv", fold_metrics),
                                ("region_recall.csv", per_region), ("region_coverage.csv", coverage),
                                ("paired_differences.csv", differences)):
            with (stage / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        with (stage / "predictions.csv").open("w", newline="") as stream:
            fields = ["unit_uid", "dataset", "animal", "source_region", "region", "fold", "Firing rate only", *order]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for dataset, selected in cohorts.items():
                for j, i in enumerate(selected):
                    writer.writerow(dict(unit_uid=reference[i]["unit_uid"], dataset=dataset, animal=groups[dataset][j],
                        source_region=reference[i]["brain_region"], region=targets[dataset][j], fold=assignments[dataset][j],
                        **{name: predicted[j] for name, predicted in predictions[dataset].items()}))
        plot_region_scores(stage / "brain_region_decoding.png", metrics, list(cohorts))
        write_json(stage / "index.json", {"plots": ["brain_region_decoding.png"], "metrics": "metrics.csv",
            "paired_differences": "paired_differences.csv", "region_recall": "region_recall.csv",
            "predictions": "predictions.csv", "region_coverage": "region_coverage.csv",
            "caveat": "Animals held out from probe fitting only; all were included in encoder pretraining. "
                      "Bootstrap conditions on trained encoders and fitted OOF probes; no between-seed uncertainty. "
                      "Datasets have different supported region sets, so compare models within each dataset."})
    return aid
