"""Compare VC-only and joint-pretrained encoders using identical VC probe folds."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from .brain_region_decoding import (
    animal_folds, beryl_gray_mapping, paired_animal_bootstrap, region_cohorts, region_probe,
)
from .collections import load_unit_means
from .rate_variant_comparison import matched_order
from .store import file_digest, implementation_stamp, write_json


DATASET = 'allen_visual_coding'


def verify_reference_cohort(rows, selected, labels, folds, saved):
    by_uid = {r['unit_uid']: r for r in saved}
    if len(by_uid) != len(saved) or set(by_uid) != {rows[i]['unit_uid'] for i in selected}:
        raise ValueError('Reference and VC-only probe cohorts differ')
    ordered = [by_uid[rows[i]['unit_uid']] for i in selected]
    expected_folds = np.empty(len(selected), dtype=int)
    for fold, (_, test) in enumerate(folds):
        expected_folds[test] = fold
    for j, (i, old) in enumerate(zip(selected, ordered)):
        if (old['region'] != labels[j] or old['animal'] != rows[i]['subject_uid']
                or int(old['fold']) != expected_folds[j]):
            raise ValueError('Reference labels, animals, or held-out folds differ')
    return ordered


def verify_pooling(store, left, right):
    sessions = []
    for aid in (left, right):
        root = store.path('collections', aid)
        index = json.loads((root / 'index.json').read_text())
        sessions.append({r['session_id']: (r['units'], r['windows'], file_digest(root / r['shard'] / 'windows.json'))
                         for r in index['sessions'] if r['dataset_id'] == DATASET})
    if sessions[0] != sessions[1]:
        raise ValueError('VC-only and reference embeddings use different sessions or pooling windows')


def analyze_vc_region_decoding(store, collection, reference_decoding, atlas_directory):
    source = store.read('analyses', reference_decoding)
    original_overview = store.read('analyses', source['recipe']['models']['Baseline epoch 10'])
    original_collection = original_overview['recipe']['collections'][0]
    verify_pooling(store, collection, original_collection)
    _, reference_rows = load_unit_means(store, [original_collection], max_rows=50000,
                                       row_filter=lambda row: row['dataset_id'] == DATASET)
    x, rows = load_unit_means(store, [collection], max_rows=50000)
    if {r['dataset_id'] for r in rows} != {DATASET}:
        raise ValueError('Expected a VC-only collection')
    x = x[matched_order(reference_rows, rows)]
    rows = reference_rows
    atlas = Path(atlas_directory)
    for filename, checksum in source['recipe']['atlas_files'].items():
        if file_digest(atlas / filename) != checksum:
            raise ValueError('Atlas differs from the existing decoding comparison')
    mapping = beryl_gray_mapping(atlas)
    cohorts, coverage = region_cohorts(rows, mapping)
    selected = cohorts[DATASET]
    labels = np.array([mapping[rows[i]['brain_region']] for i in selected])
    subjects = np.array([rows[i]['subject_uid'] for i in selected])
    folds = animal_folds(labels, subjects)
    with (store.path('analyses', reference_decoding) / 'predictions.csv').open() as stream:
        cached = [r for r in csv.DictReader(stream) if r['dataset'] == DATASET]
    cached = verify_reference_cohort(rows, selected, labels, folds, cached)
    recipe = {'operation': 'vc_only_vs_joint_region_decoding_v1', 'vc_collection': collection,
              'reference_decoding': reference_decoding, 'atlas_files': source['recipe']['atlas_files'],
              'selection': source['recipe']['selection'], 'validation': source['recipe']['validation'],
              'probe': source['recipe']['probe'], 'variant_representation': 'content coordinates only',
              'reuse': 'reference OOF predictions after exact unit/label/animal/fold/pooling checks',
              'implementation': implementation_stamp()}
    with store.build('analyses', recipe, 'VC-only versus joint brain-region decoding',
                     [{'kind': 'collections', 'id': collection}, {'kind': 'analyses', 'id': reference_decoding}]) as (stage, aid):
        if stage is None:
            return aid
        print(f'FIT VC-only: {len(selected)} neurons, {len(set(labels))} regions, {len(set(subjects))} animals', flush=True)
        predicted, fold_ids, fold_metrics = region_probe(x[selected].astype(np.float64), labels, folds)
        names = {'Firing rate only': 'Firing rate only', 'Three datasets original E10': 'Baseline epoch 10',
                 'Three datasets variant E10': 'Variant epoch 10'}
        predictions = {name: np.array([r[key] for r in cached]) for name, key in names.items()}
        predictions['VC only E10'] = predicted
        metrics, recalls, differences = [], [], []
        classes = np.unique(labels)
        for name, values in predictions.items():
            record = dict(model=name, neurons=len(labels), animals=len(set(subjects)), regions=len(classes),
                          balanced_accuracy=float(balanced_accuracy_score(labels, values)),
                          macro_f1=float(f1_score(labels, values, average='macro')), chance=1 / len(classes))
            metrics.append(record)
            print(json.dumps(record), flush=True)
            for region, score in zip(classes, recall_score(labels, values, labels=classes, average=None, zero_division=0)):
                recalls.append(dict(model=name, region=region, recall=float(score), neurons=int(sum(labels == region))))
        for name in ['Three datasets original E10', 'Three datasets variant E10']:
            low, high = paired_animal_bootstrap(labels, subjects, predictions[name], predicted)
            differences.append(dict(reference=name, candidate='VC only E10',
                balanced_accuracy_difference=float(balanced_accuracy_score(labels, predicted) - balanced_accuracy_score(labels, predictions[name])),
                bootstrap_low=float(low), bootstrap_high=float(high)))
        for filename, table in [('metrics.csv', metrics), ('region_recall.csv', recalls), ('paired_differences.csv', differences),
                                ('vc_fold_metrics.csv', fold_metrics), ('region_coverage.csv', coverage)]:
            with (stage / filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        with (stage / 'predictions.csv').open('w', newline='') as stream:
            fields = ['unit_uid', 'animal', 'region', 'fold', *predictions]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for j, i in enumerate(selected):
                writer.writerow(dict(unit_uid=rows[i]['unit_uid'], animal=subjects[j], region=labels[j], fold=int(fold_ids[j]),
                                     **{name: values[j] for name, values in predictions.items()}))
        import matplotlib.pyplot as plt
        order = ['Firing rate only', 'VC only E10', 'Three datasets original E10', 'Three datasets variant E10']
        table = {r['model']: r for r in metrics}
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        bars = ax.bar(range(4), [100 * table[name]['balanced_accuracy'] for name in order],
                      color=['#999999', '#8d65b3', '#386cb0', '#1b9e77'])
        ax.bar_label(bars, fmt='%.1f%%', padding=4)
        ax.axhline(100 / len(classes), linestyle='--', color='#777777', label='Chance')
        ax.set(xticks=range(4), xticklabels=['FR only', 'VC only\nepoch 10', 'Joint original\nepoch 10', 'Joint variant\nepoch 10'],
               ylabel='Held-out-animal balanced accuracy (%)', title=f'VC brain-region decoding · {len(classes)} regions',
               ylim=(0, max(r['balanced_accuracy'] for r in metrics) * 120))
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
        fig.savefig(stage / 'vc_region_decoding.png', dpi=180, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        write_json(stage / 'index.json', {'plots': ['vc_region_decoding.png'], 'metrics': 'metrics.csv',
                   'paired_differences': 'paired_differences.csv', 'predictions': 'predictions.csv',
                   'caveats': ['Animals held out from probe fitting, not encoder pretraining.',
                               'Equal epochs and VC exposure; joint pretraining has more total optimizer updates.',
                               'Paired animal intervals condition on trained encoders and fitted probes; single seed.']})
    return aid
