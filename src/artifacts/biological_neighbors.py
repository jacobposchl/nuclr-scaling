"""Cross-dataset biological neighborhood agreement with firing-rate matching."""

from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .brain_region_decoding import beryl_gray_mapping, region_cohorts
from .collections import load_unit_means
from .rate_variant_comparison import matched_order
from .store import file_digest, implementation_stamp, write_json


DATASETS = {'allen_visual_behavior': 'VB', 'allen_visual_coding': 'VC',
            'ibl_brain_wide_bench_2026': 'IBL'}
CONDITIONS = ['Unmatched', 'Matched FR', 'Matched FR + population FR']
SCORES = ['random', 'rate_only', 'baseline_euclidean', 'variant_euclidean',
          'baseline_cosine', 'variant_cosine']


def balanced_queries(indices, subjects, rng, maximum=100):
    """Sample across animals in round-robin order, then within animal."""
    buckets = [list(rng.permutation(indices[subjects[indices] == animal]))
               for animal in sorted(set(subjects[indices]))]
    result = []
    while buckets and len(result) < maximum:
        for j in rng.permutation(len(buckets)):
            if len(result) == maximum:
                break
            result.append(buckets[j].pop())
        buckets = [bucket for bucket in buckets if bucket]
    return np.asarray(result, dtype=int)


def candidate_pool(query, targets, subjects, rates, population_rates, condition):
    keep = subjects[targets] != subjects[query]
    if condition != 'Unmatched':
        keep &= np.abs(np.log(rates[targets] / rates[query])) <= np.log(1.1)
    if condition == 'Matched FR + population FR':
        keep &= np.abs(np.log(population_rates[targets] / population_rates[query])) <= np.log(1.25)
    return targets[keep]


def neighbor_agreement(x, query, candidates, labels, k=10):
    values, anchor = x[candidates], x[query]
    dot = values @ anchor
    squared_norms = np.einsum('ij,ij->i', values, values)
    distances = squared_norms + anchor @ anchor - 2 * dot
    cosine_distance = 1 - dot / np.maximum(np.sqrt(squared_norms * (anchor @ anchor)), 1e-20)
    return [float(np.mean(labels[candidates[np.argpartition(d, k - 1)[:k]]] == labels[query]))
            for d in (distances, cosine_distance)]


def summarize(records, repeats=1000):
    """Macro-average regions, then animals within each region; paired query-animal CI."""
    regions = sorted({row['label'] for row in records})
    animals = sorted({row['subject_uid'] for row in records})
    ri, ai = {r: i for i, r in enumerate(regions)}, {a: i for i, a in enumerate(animals)}
    totals = np.zeros((len(regions), len(animals), len(SCORES)))
    counts = np.zeros((len(regions), len(animals)))
    for row in records:
        r, a = ri[row['label']], ai[row['subject_uid']]
        totals[r, a] += [row[key] for key in SCORES]
        counts[r, a] += 1
    present = counts > 0
    means = totals / np.maximum(counts[..., None], 1)
    region_means = means.sum(1) / present.sum(1)[:, None]
    result = dict(zip(SCORES, map(float, region_means.mean(0))))
    rng = np.random.default_rng(42)
    weights = rng.multinomial(len(animals), np.full(len(animals), 1 / len(animals)), repeats)
    denominator = present @ weights.T
    for metric in ('euclidean', 'cosine'):
        differences = means[:, :, SCORES.index('variant_' + metric)] - means[:, :, SCORES.index('baseline_' + metric)]
        numerator = differences @ weights.T
        # Keep all regions represented; an absent region is not silently dropped.
        valid = np.all(denominator > 0, axis=0)
        draws = (numerator[:, valid] / denominator[:, valid]).mean(0)
        result['delta_' + metric] = result['variant_' + metric] - result['baseline_' + metric]
        low, high = np.quantile(draws, [.025, .975]) if len(draws) else (np.nan, np.nan)
        result['delta_' + metric + '_low'] = float(low)
        result['delta_' + metric + '_high'] = float(high)
        result['bootstrap_draws_' + metric] = len(draws)
    result.update(queries=len(records), regions=len(regions), animals=len(animals))
    return result, [dict(label=region, **dict(zip(SCORES, map(float, score))))
                    for region, score in zip(regions, region_means)]


def plot_results(path, summary, analysis="region", variant_epoch=10):
    import matplotlib.pyplot as plt
    pairs = [('VB', 'VC'), ('VB', 'IBL'), ('VC', 'IBL')]
    colors = ['#b5b5b5', '#d9a247', '#386cb0', '#1b9e77']
    keys = SCORES[:4]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), layout='constrained', sharey='row')
    for i, condition in enumerate(CONDITIONS):
        row_max = max(float(r[key]) for r in summary if r['analysis'] == analysis and r['condition'] == condition for key in keys)
        for j, pair in enumerate(pairs):
            rows = [r for r in summary if r['analysis'] == analysis and r['condition'] == condition
                    and {r['source'], r['target']} == set(pair)]
            values = 100 * np.mean([[r[key] for key in keys] for r in rows], axis=0)
            ax = axes[i, j]
            bars = ax.bar(range(4), values, color=colors)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)
            ax.set(xticks=range(4), xticklabels=['Random', 'FR only', 'Original E10', f'Variant E{variant_epoch}'],
                   title=f'{pair[0]} ↔ {pair[1]} · {condition}')
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(0, row_max * 120)
    for ax in axes[:, 0]:
        ax.set_ylabel('Same-region neighbors (%)')
    fig.savefig(path, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_controlled_results(path, summary, variant_epoch=10):
    import matplotlib.pyplot as plt
    pairs = [('VB', 'VC'), ('VB', 'IBL'), ('VC', 'IBL')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained', sharey='row')
    for i, analysis in enumerate(['region', 'visual_region']):
        selected = [r for r in summary if r['analysis'] == analysis
                    and r['condition'] == 'Matched FR + population FR']
        upper = max(r[key] for r in selected for key in SCORES[:4]) * 120
        for j, pair in enumerate(pairs):
            rows = [r for r in selected if {r['source'], r['target']} == set(pair)]
            values = 100 * np.mean([[r[key] for key in SCORES[:4]] for r in rows], axis=0)
            ax = axes[i, j]
            bars = ax.bar(range(4), values, color=['#b5b5b5', '#d9a247', '#386cb0', '#1b9e77'])
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=11)
            region_label = 'all shared regions' if i == 0 else 'visual subregions'
            ax.set(title=f'{pair[0]} ↔ {pair[1]} · {region_label}', ylim=(0, upper),
                   xticks=range(4), xticklabels=['Random', 'FR only', 'Original E10', f'Variant E{variant_epoch}'])
            ax.spines[['top', 'right']].set_visible(False)
        axes[i, 0].set_ylabel('Same-region neighbors (%)')
    fig.savefig(path, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def analyze_biological_neighbors(store, source_comparison, atlas_directory, variant_epoch=10):
    if variant_epoch not in (5, 10):
        raise ValueError("Variant epoch must be 5 or 10")
    variant_label = f"Variant epoch {variant_epoch}"
    source = store.read('analyses', source_comparison)
    overviews = source['recipe']['models']
    atlas = Path(atlas_directory)
    recipe = {'operation': 'cross_dataset_biological_neighbors_v1', 'source_comparison': source_comparison,
              'models': {key: overviews[key] for key in ['Baseline epoch 10', variant_label]},
              'atlas_files': {name: file_digest(atlas / name) for name in ['allen_structure_tree.csv', 'beryl.npy']},
              'region_support': '>=50 positive-rate neurons and >=5 animals in each dataset; pairwise shared Beryl gray regions',
              'queries': 'up to 100 per region per direction, animal-balanced round-robin, seed42',
              'separate_cohorts': ['shared gray regions', 'shared VIS regions only'],
              'candidate_selection': 'same pools for models; other dataset and different animal; uniformly cap at512; minimum50; k10',
              'matching': 'FR ratio <=1.1; population mean FR ratio <=1.25; no embedding-dependent selection',
              'comparisons': CONDITIONS, 'geometry': 'original coordinates; Euclidean primary, uncentered cosine sensitivity',
              'aggregation': 'equal regions, equal query animals within region, equal directions for plot',
              'uncertainty': '1000 paired query-animal bootstrap draws, fixed candidate pools and trained models',
              'cell_type': 'exploratory VB/VC Sst vs Vip only; same Beryl region; FR ratio<=1.25; min10 candidates and >=2 of each type; k5',
              'implementation': implementation_stamp()}
    with store.build('analyses', recipe, 'Cross-dataset biological neighbors beyond firing rate',
                     [{'kind': 'analyses', 'id': source_comparison}]) as (stage, aid):
        if stage is None:
            return aid
        matrices, reference = {}, None
        for model, label in [('baseline', 'Baseline epoch 10'), ('variant', variant_label)]:
            overview = store.read('analyses', overviews[label])
            x, rows = load_unit_means(store, overview['recipe']['collections'], max_rows=200000)
            if reference is None:
                reference = rows
            else:
                x = x[matched_order(reference, rows)]
            matrices[model] = x.astype(np.float32)
        with (store.path('analyses', overviews['Baseline epoch 10']) / 'coordinates.csv').open() as stream:
            annotation = {r['unit_uid']: r for r in csv.DictReader(stream)}
        rows = [dict(r, cell_type=annotation[r['unit_uid']]['cell_type'],
                     population_mean_firing_rate_hz=float(annotation[r['unit_uid']]['population_mean_firing_rate_hz']))
                for r in reference]
        mapping = beryl_gray_mapping(atlas)
        cohorts, coverage = region_cohorts(rows, mapping)
        regions = np.array([mapping[r['brain_region']] or '' for r in rows])
        subjects = np.array([r['subject_uid'] for r in rows])
        rates = np.array([r['firing_rate_hz'] for r in rows])
        population_rates = np.array([r['population_mean_firing_rate_hz'] for r in rows])
        types = np.array([r['cell_type'] for r in rows])
        rng = np.random.default_rng(42)
        records, coverage_records = [], []

        def score(query, candidates, labels, k, analysis, source_name, target_name, condition):
            if len(candidates) > 512:
                candidates = rng.choice(candidates, 512, replace=False)
            row = dict(analysis=analysis, source=DATASETS[source_name], target=DATASETS[target_name],
                       condition=condition, unit_uid=rows[query]['unit_uid'], subject_uid=subjects[query],
                       label=labels[query], brain_region=regions[query], candidates=len(candidates),
                       random=float(np.mean(labels[candidates] == labels[query])))
            rate_dist = np.abs(np.log(rates[candidates] / rates[query]))
            near = candidates[np.argpartition(rate_dist, k - 1)[:k]]
            row['rate_only'] = float(np.mean(labels[near] == labels[query]))
            for model, x in matrices.items():
                row[model + '_euclidean'], row[model + '_cosine'] = neighbor_agreement(x, query, candidates, labels, k)
            records.append(row)

        for analysis in ['region', 'visual_region']:
            for source_name, target_name in itertools.permutations(DATASETS, 2):
                shared = sorted(set(regions[cohorts[source_name]]) & set(regions[cohorts[target_name]]))
                if analysis == 'visual_region':
                    shared = [region for region in shared if region.startswith('VIS')]
                targets = cohorts[target_name][np.isin(regions[cohorts[target_name]], shared)]
                queries = np.concatenate([balanced_queries(cohorts[source_name][regions[cohorts[source_name]] == region],
                                                           subjects, rng) for region in shared])
                print(f"{analysis.upper()} {DATASETS[source_name]} -> {DATASETS[target_name]}: {len(shared)} regions, {len(queries)} queries", flush=True)
                for condition in CONDITIONS:
                    start = len(records)
                    for query in queries:
                        candidates = candidate_pool(query, targets, subjects, rates, population_rates, condition)
                        if len(candidates) < 50:
                            continue
                        score(query, candidates, regions, 10, analysis, source_name, target_name, condition)
                    current = records[start:]
                    retained = {r['unit_uid'] for r in current}
                    for region in shared:
                        q = queries[regions[queries] == region]
                        coverage_records.append(dict(analysis=analysis, source=DATASETS[source_name], target=DATASETS[target_name],
                            condition=condition, region=region, proposed_queries=len(q),
                            retained_queries=sum(rows[i]['unit_uid'] in retained for i in q)))
                    print(f"  {condition}: {len(current)} retained", flush=True)
        # Cell types are a separate exploratory comparison: unknown is not a negative label.
        for source_name, target_name in [('allen_visual_behavior', 'allen_visual_coding'),
                                          ('allen_visual_coding', 'allen_visual_behavior')]:
            source_indices = np.array([i for i, r in enumerate(rows) if r['dataset_id'] == source_name
                                       and types[i] in {'Sst', 'Vip'} and rates[i] > 0 and regions[i]])
            targets = np.array([i for i, r in enumerate(rows) if r['dataset_id'] == target_name
                               and types[i] in {'Sst', 'Vip'} and rates[i] > 0 and regions[i]])
            for query in source_indices:
                candidates = targets[(regions[targets] == regions[query]) & (subjects[targets] != subjects[query])
                                     & (np.abs(np.log(rates[targets] / rates[query])) <= np.log(1.25))]
                if len(candidates) >= 10 and all(np.sum(types[candidates] == label) >= 2 for label in ['Sst', 'Vip']):
                    score(query, candidates, types, 5, 'cell_type', source_name, target_name, 'Same region + matched FR')
        grouped = defaultdict(list)
        for row in records:
            grouped[(row['analysis'], row['source'], row['target'], row['condition'])].append(row)
        summary, per_label = [], []
        for (analysis, source_name, target_name, condition), group in grouped.items():
            metadata = dict(analysis=analysis, source=source_name, target=target_name, condition=condition)
            values, label_values = summarize(group)
            summary.append(dict(**metadata, **values))
            per_label.extend(dict(**metadata, **v) for v in label_values)
            print(json.dumps(summary[-1]), flush=True)
        for name, data in [('queries.csv', records), ('summary.csv', summary), ('region_scores.csv', per_label),
                           ('matching_coverage.csv', coverage_records), ('region_coverage.csv', coverage)]:
            with (stage / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        plot_results(stage / 'cross_dataset_region_neighbors.png', summary, variant_epoch=variant_epoch)
        plot_results(stage / 'visual_region_neighbors.png', summary, 'visual_region', variant_epoch=variant_epoch)
        plot_controlled_results(stage / 'rate_matched_biological_neighbors.png', summary, variant_epoch=variant_epoch)
        write_json(stage / 'index.json', {'summary': 'summary.csv', 'queries': 'queries.csv',
                   'plots': ['rate_matched_biological_neighbors.png', 'cross_dataset_region_neighbors.png', 'visual_region_neighbors.png'], 'source_comparison': source_comparison,
                   'limitations': ['Region is a proxy for biological comparability, not functional ground truth.',
                    'Rate matching controls measured pooled rates, not all firing statistics or other confounders.',
                    'Cell types: VB source optotagging, VC LOLCAT predictions; IBL unavailable.',
                    'Bootstrap conditions on reference neurons and trained model seeds; no held-out decoding.']})
    return aid
