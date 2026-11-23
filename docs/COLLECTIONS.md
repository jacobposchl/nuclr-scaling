# Collections and analyses

The workflow separates three reusable products:

1. **Checkpoint:** a trained model plus declared training provenance.
2. **Collection:** a specific checkpoint applied to particular data and windows.
3. **Analysis:** a particular question and protocol applied to one or more collections.

The scaffold currently supports CPU or explicit GPU extraction, unit averages, window embeddings,
grouped PCA/UMAP visualization, and a linear classification probe with held-out subjects or
sessions. Temporal behavior decoding, dataset balancing, and a comparison dashboard
are future extensions. The existing GPU training entry point remains separate.

## Dataset choices

The registry is `configs/data_registry.json`. Paths are relative to
`/home/jposchl/data/processed`; `--root` can override this location.

| Registry ID | Selected directory | Observed sessions | Sampling domain |
| --- | --- | --- | --- |
| `allen_visual_behavior` | `allen_vbn_2022_vis_only` | 103 | Full available recording, excluding optotagging |
| `allen_visual_coding` | `allen_vc_2019_vis` | 58 | Valid recording intervals, excluding optotagging and uncertain tails |

The local files contain the same session IDs as the all-session manifest, but
retain visual-area populations. They are distinct derivatives from the full
populations used in the historical cluster checks in `VALIDATION.md`. The registry
records local release names so the earlier population and throughput measurements
are not implicitly attributed to these files. Behavior retains active/passive
intervals, detailed unit QC, brain-region names, and optotagging-related metadata.

The default registry now permits every task period, including active behavior,
passive replay, spontaneous activity, gratings, and movies where available. All
intervals are intersected with stored spike bounds; windows cannot cross gaps or
optotagging blocks, including the second view of a contrastive training pair.
Behavior uses its stored `intervals/blocks/optotagging` boundaries. These files
already end five minutes before tagging, so changing the domain cannot recover
the missing end of passive replay.

Coding's processed files lack optotagging timings. The registry embeds the full
first-pulse to last-pulse blocks recovered from 54 original Allen NWBs on
2026-11-17. Those 54 processed recordings already end at least five minutes before
tagging. Four original NWBs also lack the timing table: `732592105`, `737581020`,
`739448407`, and `743475441`. For those explicitly listed sessions, sampling stops
at the final visual stimulus, conservatively excluding the uncertain tail.
This uses Allen's documented protocol that
[optotagging occurs after visual stimulation](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_optotagging.html).
Missing timing metadata for any other configured session is an error, not an
implicit claim that tagging was absent. The recovered boundaries live inside the
registry so training snapshots remain self-contained and need no network access.

The per-session counts and timing sources are in
`results/sampling/allen_full_recording_window_counts.csv`. Window counts sum
`floor(interval_seconds / 20)` over allowed intervals, matching the current
uncapped epoch sampler. A sample produces two views, not two optimizer updates.
The full training preset now selects exactly 300 of those candidate windows per
recording per epoch, without replacement. New jitter and selections are seeded
by epoch. Partial batches are retained to preserve the quota. Sessions with
fewer than 300 candidates are rejected, not oversampled. The available-window
counts in the CSV describe capacity before applying this quota.
The completed 10-epoch run retains its frozen active-only Behavior registry and
is unaffected by these new defaults.

The new adapters retain all units already present in each processed source.
They do not claim identical upstream QC across releases. Both local datasets use
their explicit `units/probe_id` field; local Coding unit IDs do not encode probes.
Units are grouped by probe with spike indices remapped.
Local identifiers are preserved alongside globally namespaced unit, session,
and probe IDs. Subject grouping uses Allen subject IDs across both datasets.

`optotagged_cell_type=wt` is retained as source metadata, not interpreted as a
confirmed biological cell type or a verified negative optotagging label. Coding's
numeric `type` field is likewise preserved without assigning a new meaning.
The visualization field `cell_type` copies available optotagging annotations;
absent labels and `wt` become unknown. It is not a comprehensive cell taxonomy.

The source files remain read-only. The adapter reads only the requested spike
window using binary search of sorted timestamps; it does not load whole recordings.

For task-specific pooling, use a separate registry passed with `--registry`.
Behavior passive replay uses `"domain": "intervals/blocks/passive"`. Coding
natural scenes uses `"domain": "domain"` with `"stimulus_types": ["natural_scenes"]`;
this intersects the labeled stimulus blocks with valid recording intervals and
spike bounds. Select only sessions containing the requested stimulus (32 local
Coding sessions have natural scenes). Complete windows stay within each resulting
interval. These extraction settings do not change the training configuration.

## Storage and cache behavior

```text
artifacts/
  checkpoints/<id>/
    manifest.json
    reference.json
  collections/<id>/
    manifest.json
    index.json
    session-00000/
      units.json
      windows.json
      embeddings.h5
  analyses/<id>/
    manifest.json
    metrics.json
    ... figures, coordinates, splits, predictions, fitted parameters ...
```

Every ID is a SHA256 digest of a canonical recipe. Display labels are for humans
and do not change the ID. Manifests record parent artifacts, creation time,
the recipe, and checksums of every payload. Source code and numeric-library
versions are included in collection and analysis recipes. Changing implementation
or environment deliberately invalidates reuse, even if filenames stay the same.

Collection identity includes the exact checkpoint digest; selected dataset
release, sessions and split annotations; source fingerprints; window duration,
stride and limits; population policy; representation; precision; and output form.
Extraction uses the encoder output with no normalization, no unit dropout,
and the full retained population for each probe. The checkpoint determines the
context duration. Projector embeddings are not implemented yet.

`fingerprint: stat` records resolved source paths, byte sizes, and nanosecond
modification times. This is fast but **is not a content checksum**: it cannot
detect an in-place edit that deliberately preserves both size and mtime. Use
`fingerprint: sha256` for byte-level dataset identity; it reads each selected
session file in full and should only run in an allocated compute job. Artifact
payloads and checkpoints always use byte-level checksums. Sources are checked
again before publishing an extraction.

Writes use private staging directories, a per-ID writer lock, and an atomic rename.
Failures do not appear as completed collections. Reuse verifies saved payload
checksums. Published artifacts are never overwritten by the API; filesystem
permissions are not a substitute for this convention. After a forcibly killed
writer, inspect its `.partial-*` directory and `.<id>.lock` before manually
removing them. Automatic stale-lock deletion is deliberately avoided.

Checkpoint registration records a reference to the existing checkpoint, rather
than copying a large file. Keep that file at its registered path and retain it;
deletion or mutation makes subsequent extraction fail. Exported collections and
analyses remain readable independently of the checkpoint file.

Window arrays have shape `[window, unit, dimension]`; unit means have shape
`[unit, dimension]`. Absolute window bounds and row metadata are stored separately.
Both forms retain observation counts. Averages weight windows equally, including
when windows overlap. No population subsampling occurs. Windows with no spikes
still produce an encoder representation for their retained units.
New collections also store each unit's `firing_rate_hz`: counted spikes divided
by total extraction-window duration. Overlapping time contributes once per window,
matching the embedding average. Saved windows include per-window rates, so deriving
a subset average also recomputes its rate. Legacy windows lacking these counts
produce an unknown rate, rather than retaining a rate from different windows.

## Run commands only in an allocated CPU job

Do not run Python, extraction, tests, hashing of datasets, or analyses on the
login node. The following commands assume a correctly installed environment
**inside an already allocated CPU job**; they do not request an allocation.
Extraction defaults to float32 on CPU. In an authorized GPU environment, set
`"device": "cuda"` in the collection recipe and select the GPU with
`CUDA_VISIBLE_DEVICES`. This uses bf16 inference, saves float32 embeddings, and
accumulates session means in float64. Device and inference precision are recorded
in the collection identity. Analyses still run on CPU.

Inspect headers without loading all spike arrays:

```bash
python -m src.artifacts.cli inventory --samples 1
```

Register a checkpoint and its provenance:

```bash
python -m src.artifacts.cli register-checkpoint /path/to/model.pt \
  --provenance training_provenance.json --label "Allen baseline, epoch 49"
```

The provenance JSON should identify the training run, datasets, session split,
and any checkpoint-selection rule. For old checkpoints with incomplete records,
use an explicit declaration such as `{"status": "unknown"}` instead of inventing
which sessions were held out. New mixed-Allen training checkpoints capture their
resolved selections in `cfg.data_provenance`; include that snapshot when registering.

Extract a deliberately small selection:

```bash
python -m src.artifacts.cli extract CHECKPOINT_ID \
  --config configs/collections/allen_example.json --label "Allen two-session check"
```

The example uses two real session IDs, two chronological windows per session,
and saves both forms. It does **not** define a scientific evaluation cohort.
Set explicit session lists for experiments, or use `"sessions": "all"` when a
full release is intentionally wanted. `"max_windows_per_session": null` removes
the window cap. A `split` annotation is recorded verbatim and defaults to
`unspecified`; an annotation alone does not prove that pretraining excluded it.

Build averages from saved windows without running NuCLR again:

```bash
python -m src.artifacts.cli derive-means COLLECTION_ID
python -m src.artifacts.cli derive-means COLLECTION_ID --windows 0 1
```

The same window-index selection is applied to every session and must exist in
each. Metadata retains original absolute times. This is a new derived collection,
linked to its parent, so it cannot silently replace a previous average.

Run analyses and browse the catalog:

```bash
python -m src.artifacts.cli analyze COLLECTION_ID --config configs/analyses/visualization.json
python -m src.artifacts.cli analyze COLLECTION_ID --config configs/analyses/brain_region_probe.json
python -m src.artifacts.cli list
python -m src.artifacts.cli show collections COLLECTION_ID
```

Pass `--store /path/to/artifacts` **before** the subcommand to change the destination.
Analyses can combine non-overlapping collections from the same checkpoint.
Different checkpoints should receive separate analyses: their coordinate spaces
cannot simply be pooled, even if their embedding dimensions match.

### Grouped visualization

`configs/analyses/visualization.json` produces both PCA and UMAP, with separate
figures colored by brain region, subject, cell-type annotation, and firing rate.
One dot is one unit's mean embedding. Each method uses the same coordinates for
all its colorings. The config chooses methods, metadata fields, categorical or
continuous coloring, and Matplotlib palette names (such as `tab20`, `viridis`,
or `magma`). Missing values appear gray; continuous values receive a colorbar.
Exact category colors and continuous ranges are recorded with each figure.
Colors are assigned within each displayed cohort, not globally across analyses.

Fit and display cohorts are separate. For example, replace the example IDs with
subject IDs from collection metadata:

```json
{
  "fit_selection": {"subject_uid": ["allen/MOUSE_A", "allen/MOUSE_B"]},
  "show_selection": {"subject_uid": ["allen/MOUSE_A", "allen/MOUSE_C"]}
}
```

This fits on A and B, displays A using its fitted coordinates, and transforms C
using that saved fit. C does not affect PCA, UMAP, or optional standardization.
`show_selection: null` displays the fit cohort; an explicit `{}` displays all
input units. The default fit selection `{}` fits all input units. Selectors also
support `subject_id`, `dataset_id`, `session_uid`, `session_id`, `brain_region`,
`cell_type`, and `split`. Values within one field are alternatives; different
fields are intersected. Animal and subject refer to the same grouping here.
Namespaced `subject_uid` is preferred. `max_rows` limits the union of fit and
display rows, after filtering, without silently sampling units.

The visualization links to a cached projection artifact containing the fitted
models, fit-unit IDs, coordinates, optional scaler, and PCA explained variance.
Changing a palette or display cohort reuses the fit. Changing the fit cohort,
projection parameters, checkpoint/collections, or implementation creates a new
projection. UMAP uses a fixed seed and one worker. Its held-out coordinates can
depend on the transform batch, so use the same display cohort for comparisons.
Only load saved model files from your trusted artifact store.

Each grouped result contains:

```text
analyses/<visualization-id>/
  index.json            # links projection, coordinates, figures, color legends
  selection.json        # exact fit and displayed unit IDs
  metrics.json
  pca/coordinates.csv
  pca/00_brain_region.png
  pca/...               # remaining colorings
  umap/coordinates.csv
  umap/00_brain_region.png
  umap/...
```

These are descriptive plots, not held-out performance estimates. UMAP needs at
least three fit units and `n_neighbors` smaller than the fit population. Existing
collections lacking requested metadata must be re-extracted or visualized using
available fields; published collections are not modified. The old `pca` analysis
remains available for compatibility. This scaffold is config-driven, not a GUI.

### Linear classification probe

The linear probe fits its scaler and classifier on training groups only.
It saves group and unit assignments, held-out predictions, metrics, and the fitted
scaler/classifier arrays. Default grouping is by subject; session grouping is an
explicit alternative. Both classes and majority-class baseline performance are
reported, and unseen held-out labels are not silently removed.

For reproducible comparison across models, specify the same `test_groups` in the
probe config (take the exact IDs from an earlier analysis's `split.json`) and keep
the unit cohort fixed. The probe's group holdout is separate from pretraining
exposure. Both must be checked before interpreting a result as across-subject
generalization. The reader refuses duplicate units and enforces an explicit
analysis row limit rather than silently subsampling a scientific cohort.

## Multi-dataset pretraining connection

Training uses `train.py` with Hydra YAML presets; post-training collection and
analysis commands use JSON recipes. The training presets currently are:

| Preset | Intended use | Batch size | Checkpoints by default |
| --- | --- | --- | --- |
| `train_nuclr` | Original IBL benchmark baseline | 80 | Off |
| `train_nuclr_allen_vbn` | Original Zen Allen Behavior adapter and splits | 16 | Off |
| `train_nuclr_allen_multi` | New registry-backed Behavior 2022 and/or Coding 2019 | 6 | On |
| `train_nuclr_allen_smoke` | One-epoch diagnostic, two training sessions | 2 | On |
| `train_nuclr_allen_full` | All 161 sessions, 300 samples/recording/epoch, 10 epochs | 8 | Epoch 10 + rolling last |

The legacy Allen preset does not automatically adopt the new registry or the
chosen 2022 directory. Use `train_nuclr_allen_multi` for the current datasets,
including a Behavior-only or Coding-only baseline through its split manifest.
The selected combined cohort is `configs/splits/allen_all_sessions.json`: all 103
Behavior 2022 and 58 Coding 2019 sessions, with an empty evaluation partition.
The full preset uses 10 epochs, 20-second views, peak learning rate `5e-4`, and
the 256-dimensional encoder. The smoke preset uses one epoch. YAML options can
be overridden at launch.

`configs/train_nuclr_allen_multi.yaml` connects the new Allen adapters to the
existing two-view pretrainer. It can include one or both datasets. It requires a
session split manifest and checks that no Allen subject spans pretrain and eval.
The shape of the file is:

```json
{
  "schema_version": 1,
  "partitions": {
    "pretrain": [
      {"dataset_id": "allen_visual_behavior", "sessions": ["CHOSEN_SESSION_ID"]},
      {"dataset_id": "allen_visual_coding", "sessions": ["CHOSEN_SESSION_ID"]}
    ],
    "eval": [
      {"dataset_id": "allen_visual_behavior", "sessions": ["HELD_OUT_SESSION_ID"]},
      {"dataset_id": "allen_visual_coding", "sessions": ["HELD_OUT_SESSION_ID"]}
    ]
  }
}
```

These placeholders illustrate an optional held-out cohort. The full run instead
uses explicit lists of every session and `"eval": []`, as requested. An empty
evaluation partition is supported for pretraining; attempting evaluation with it
raises an error. `pretrain` and `eval` are session partitions; the trainer's
`train+val` argument does not mix them. Source within-session masks are not used
by this configuration.

The full preset uses `samples_per_recording: 300` for equal recording quotas,
giving 48,300 samples and 6,038 updates per epoch, including a final batch of four.
It is **not dataset-balanced**: Behavior contributes 30,900 samples and Coding
17,400. Other presets default to `samples_per_recording: null`, which samples
all available fixed windows. Second-view sampling respects gaps and optotagging
exclusions in the selected domains. This configuration saves
checkpoints and disables the old automatic `.pt` embedding dump; collection
extraction is the separate post-training stage described above.

Actual pretraining uses the CUDA trainer. A one-epoch diagnostic on one training
session from each Allen dataset has now run on a single GPU, including checks of
finite gradients and exact checkpoint state restoration. Full scaling experiments
and multi-GPU/DDP training remain unvalidated. The CPU adapter/contrastive-batch
path and artifact workflow are independently testable without running that trainer.

After choosing the manifest, an example launch **inside an allocated GPU job** is:

```bash
python train.py --config-name train_nuclr_allen_multi \
  dataset.split_manifest=/path/to/chosen_splits.json
```

Before a full experiment, validate the dedicated GPU environment and a short
training/checkpoint/resume run. Also review inherited worker count, WandB account,
and batch size for that allocation. Completing visualization does not establish
GPU readiness or require changing the pretraining objective.
