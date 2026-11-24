# NuCLR scaling

This project starts with a port of Zen's NuCLR pretraining implementation.
The project includes a mixed-Allen data scaffold and an artifact workflow for
checkpoint registration, embedding collections, and reusable analyses. See
[Collections and analyses](docs/COLLECTIONS.md) for dataset choices and usage.
The all-session pretraining preset uses the local visual-area derivatives of all
103 Visual Behavior 2022 sessions and 58 Visual Coding 2019 sessions, with no
subjects held out. A diagnostic mixed-Allen
GPU training epoch and a batch-size benchmark have run; scientific scaling
benefits have not been evaluated.

The source is `/vast/projects/dyer1/lab/user/jacob/zen/repo`, at commit
`5090787b6fe099109dddbd8dd484d4db109ed22a`. The encoder, attention, loss,
dataset splits, and training defaults come from that checkout. See
[PORTING.md](PORTING.md) for the implementation details and deliberate changes.

**Status:** CPU regression and artifact-workflow checks have run, followed by a
single-GPU mixed-Allen diagnostic epoch, Zen encoder/gradient parity checks, and
exact restoration of model/loss/optimizer/scheduler checkpoint state. Distributed
training, continued optimization after resume, and scientific scaling improvements
remain unverified. See [validation notes](docs/VALIDATION.md) for the precise scope.

## Included

- NuCLR encoder: binned spikes, temporal rotary attention, probe-restricted
  spatial attention, and time-averaged unit representations.
- Two-view contrastive training with unit dropout, QC filtering, and Zen's
  decoupled contrastive loss and projection head.
- IBL Brain Wide Benchmark and Allen Visual Behavior Neuropixels loaders,
  including their existing session lists and IBL QC metadata.
- AdamW, warmup/cosine learning rate schedule, checkpoint loading and saving,
  multi-GPU launch, and final unit-embedding export.
- Zen's IBL brain-region evaluator. Allen's online evaluator remains disabled,
  as in the source, but final embedding export is available.

The `src.*` module paths and model parameter names are retained for compatibility
with Zen configs and checkpoint dictionaries. This project does not import code
from the original checkout during normal use.

## Setup and commands for later use

Use Python 3.11+ (the local `.venv` uses Python 3.12) and a CUDA environment
compatible with the PyTorch 2.10 attention
backend. Training still requires a supported GPU. A segmented CPU attention
implementation supports the separate collection-extraction workflow and small
validation runs. Run installation and Python commands only in an allocated
compute job, never on the login node.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency versions follow Zen, including its moving `torchfoo@dev`
dependency. A fully locked environment is still needed for reproducible runs.

The Allen presets default to `/home/jposchl/data/processed`; `DATA_ROOT` or the
`data_root` config option can override it. The local layout is:

```text
/home/jposchl/data/
  processed/
    allen_visual_behavior_neuropixels_2022/processed/
    allen_visual_coding_neuropixels_2019/
```

All 161 sessions in the full-run manifest are present in those directories.
These imports retain visual and nonvisual regions. Behavior provides explicit
probe IDs; Coding encodes probe/channel/unit in each unit ID. Both datasets sample
their available recording domains with optotagging excluded.
Preprocessing pipelines and datasets are not copied here. The IBL baseline
separately requires `ibl_brain_wide_bench_2026/pretrain` and `eval` under its data
root. An optional local `.env` can supply `DATA_ROOT` and `CKPT_DIR`.

IBL baseline:

```bash
python train.py --config-name=train_nuclr \
  data_root=/path/to/data wandb.mode=disabled ckpt.enable=true
```

Allen VBN baseline:

```bash
python train.py --config-name=train_nuclr_allen_vbn \
  wandb.mode=disabled ckpt.enable=true
```

The default config retains Zen's online W&B setting; the examples explicitly
disable it. Checkpointing is disabled by default in Zen and enabled explicitly
above. `batch_size` is global and must be divisible by the number of visible
GPUs. The entry point launches one worker per visible GPU itself; do not wrap it
in a second multi-process launcher. `CUDA_VISIBLE_DEVICES` selects the GPUs.

Resume model, loss projector, optimizer, scheduler, and epoch counters:

```bash
python train.py --config-name=train_nuclr \
  data_root=/path/to/data wandb.mode=disabled \
  ckpt.load_from=/path/to/last.pt ckpt.resume=true ckpt.enable=true
```

For encoder-only initialization, use `ckpt.model_only=true ckpt.resume=false`.
Keep the original dataset, batch size, epoch budget, and schedule when resuming.
As in Zen, RNG and dataloader state are not checkpointed, so resume is not a
bitwise continuation of the uninterrupted sample sequence.

With `save_embs.enable=true`, the completed run exports `train_embs`,
`train_uids`, `eval_embs`, and `eval_uids`. The default destination is
`embs/<run_id>.pt`; `save_embs.path=/path/to/file.pt` specifies a file. Disable
export with `save_embs.enable=false`. Missing unit observations remain NaN and
are reported by the original collector.

## Validation

```bash
python -m unittest discover -s tests -v
```

To include comparisons against the original source:

```bash
ZEN_SOURCE_ROOT=/vast/projects/dyer1/lab/user/jacob/zen/repo \
  python -m unittest discover -s tests -v
```

Run these commands only inside an existing compute allocation. The tests cover
the mathematical loss and gradients, zero-overlap views,
singleton probes, spike bin boundaries, probe ordering, view collation, config
composition, and checkpoint format. Optional source comparisons check the loss
and projector gradients, plus encoder outputs and gradients on CUDA. These are
synthetic checks; they do not replace a real-data training and resume test or
multi-GPU validation.

`tests/test_artifacts.py` also covers cache identity/integrity, failed writes,
the Allen adapters, CPU attention, extraction, derived averages, firing rates, PCA, grouped
probing, and subject-split checks. Set `NUCLR_REAL_DATA_SMOKE=1` to include a
one-window-per-dataset check with a small untrained diagnostic model; its
temporary artifacts are removed afterward.

`configs/analyses/visualization.json` groups PCA and UMAP with configurable fit
and display cohorts and multiple metadata colorings. `tests/test_visualization.py`
checks projection reuse, fit-only preprocessing, UMAP transforms, and metadata
validation. See [Collections and analyses](docs/COLLECTIONS.md) for recipes and
the available pretraining presets.

## One-epoch GPU diagnostic

`configs/train_nuclr_allen_smoke.yaml` selects one Behavior 2022 session and one
Coding 2019 session for a full training epoch, with two other subjects held out.
It retains the standard 256-dimensional encoder and 20-second windows, uses batch
size 2 and two loader workers, enables checkpoints, and disables WandB uploads.
Its split is a diagnostic fixture, not a scientific benchmark.

Inside an existing single-GPU allocation, from the repository root:

```bash
PYTHONPATH="$PWD" ZEN_SOURCE_ROOT=/path/to/zen/repo \
  python scripts/validate_allen_gpu.py --output .validation-output/unique-run-name
```

The runner checks Zen encoder/loss parity when `ZEN_SOURCE_ROOT` is provided,
trains one epoch through the production trainer, checks finite gradients and
contributions from both datasets, evaluates fixed held-out windows before/after,
and verifies checkpoint/model/optimizer/scheduler restoration. It does not train
a second epoch. The output directory must be new; it contains the resolved
configuration, step losses, checkpoints, and a machine-readable report.

## Full all-session pretraining

`configs/train_nuclr_allen_full.yaml` selects all 161 sessions with no subject
holdout, 10 epochs, batch size 8, and two loader workers. Each recording contributes
exactly 300 samples per epoch, drawn without replacement from its available
jittered 20-second windows. Windows are redrawn each epoch. The final partial
batch is retained: 48,300 samples produce 6,038 optimizer updates per epoch.
This balances recordings; datasets with more recordings still contribute more
samples. Checkpoints retain epoch 10; `last.pt` is atomically replaced after every
completed epoch and includes the model, loss projector, optimizer, and scheduler.

Run from the repository root using an **existing** compatible GPU allocation:

```bash
bash scripts/launch_existing_allen_gpu.sh EXISTING_JOB_ID runs/unique-run-name
```

The launcher does not request an allocation. It currently targets the B200 MIG
45 GB partition and uses this repository's `.venv`; `NUCLR_PYTHON` can override
the interpreter. The runner freezes the session manifest and registry, saves a resolved
configuration, and records `status.json`, `steps.jsonl`, and `epochs.jsonl` in
the run directory. Keep the controller alive or detach it from the terminal.

On an explicitly authorized local GPU, use `--local` instead of the Slurm
launcher. For a single validation epoch, including a shorter warmup:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$PWD" \
  .venv/bin/python scripts/run_allen_pretraining.py --local \
  --output runs/unique-one-epoch-check \
  --override num_epochs=1 --override lr.rise=0.1
```

The default Allen registry now selects the full available recording in both
datasets, excluding optotagging and preserving existing valid-time gaps. Four
Coding sessions lack tagging times even in the original NWBs; their uncertain
tails after the last visual stimulus are excluded. See
[`docs/COLLECTIONS.md`](docs/COLLECTIONS.md) for the timing sources and per-session
window counts. A recording with fewer than the configured quota of available
windows causes an error rather than being oversampled. Existing runs retain
their frozen registries and sampling configuration.

### Joint Allen + IBL integration run

`configs/train_nuclr_allen_ibl_smoke.yaml` runs one fresh epoch at batch size 3
using 103 VB passive-replay sessions, 32 VC natural-scenes sessions, and all 452
IBL sessions (423 supplied pretrain + 29 supplied eval). IBL uses its full stored
recording domain. All source-retained units participate; unit dropout remains
50–100%, and Allen optotagging is excluded. This configuration intentionally has
no held-out evaluation partition.

There is no recording quota or dataset reweighting: 20-second windows contribute
in proportion to available recording time. The current files provide 127,591
window pairs (17,070 VB, 2,368 VC, 108,153 IBL), giving 42,530 batches after dropping
the final incomplete batch. Views with fewer than two units after dropout are
skipped by the existing loader. `epochs.jsonl` records usable pairs per dataset.

On an explicitly authorized local GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" \
  .venv/bin/python scripts/run_allen_pretraining.py --local \
  --config train_nuclr_allen_ibl_smoke --output runs/unique-allen-ibl-check
```

The run saves `epoch_1.pt` and `last.pt`, including optimizer and scheduler state.
Gradient accumulation is not enabled.

### Full-domain joint Allen + IBL pretraining

`configs/train_nuclr_allen_ibl_full.yaml` prepares a fresh 10-epoch run using all
103 VB, 58 VC, and 452 IBL recordings (179,540 source-retained neurons across all
regions). Both supplied IBL partitions are included in pretraining; no sessions
are held out for evaluation. Allen uses every available task period, with whole
optotagging blocks excluded and the existing conservative tail exclusion for the
four VC sessions without exact tagging times. IBL uses its full stored domain.

The preset retains duration-based sampling without recording quotas or dataset/
probe reweighting, 20-second views, a 120-second maximum view separation, and
50–100% unit retention per view. It targets 81 window pairs per optimizer update
with `micro_batch_size: 3` (27 accumulation passes on one GPU). The learning
rate remains `5e-4`, with one epoch of warmup, followed by the existing cosine
schedule. Model dropout remains 0.3 for linear layers and 0 for attention.

The current metadata provide 179,306 window pairs per epoch: 43,978 VB, 27,175 VC,
and 108,153 IBL. On one GPU, the loader drops the final two-window partial batch,
giving 59,768 micro-batches and 2,214 planned optimizer updates per epoch. The final
update uses the remaining 17 micro-batches (51 window pairs before any skips),
normalized by their actual loss weight and announced during setup. Existing
post-dropout skips can reduce the usable pairs in an update. Batch settings must
divide exactly; incompatible values fail before training.

Select this preset with `--config train_nuclr_allen_ibl_full` in
`scripts/run_allen_pretraining.py`, and supply a new `--output` run directory.
The runner saves the resolved config, frozen registry/session list, local metrics,
and checkpoints at every epoch plus `last.pt`. This preset disables automatic
embedding extraction and external W&B logging. Creating the config does not
launch training.

### Rate-separated experiment

`train_nuclr_allen_ibl_rate_separated` starts a fresh 10-epoch run with the same
three-dataset full non-optotagging sampling, effective batch 81, and micro-batch 3.
It saves every epoch plus `last.pt`. The 256-dimensional output contains a
nonnegative raw-Hz prediction in coordinate 0, supervised from the early temporal
layers, and 255 content coordinates from the later layers. Contrastive matching
uses only the content coordinates.

The total loss is contrastive loss plus `0.1 * MSE(rate / 10 Hz)` plus `0.1`
times the mean squared correlation between each content coordinate and the true
rate. Targets are each neuron's spike count divided by its 20-second view
length. Correlations are computed across neurons and views within each
micro-batch, not across the entire accumulated batch. This discourages linear
rate leakage but does not guarantee rate independence; pooled content should
later be evaluated with fresh held-out-animal probes. Content-only analyses
must select coordinates 1 through 255. This experimental mode supports one GPU.

For an authorized local run, the launcher can enforce resource checks:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -u scripts/run_allen_pretraining.py \
  --config train_nuclr_allen_ibl_rate_separated \
  --output runs/unique-rate-separated-run --local --require-idle-gpu --min-free-gib 25
```

The GPU check rejects existing compute processes or substantial graphics activity.
Storage is checked before launch, every 100 optimizer updates, and around epoch
checkpoint saves. Failed checks stop the run without deleting any files.

To pool and plot a saved epoch over the same full non-optotagging domains:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/plot_pretraining_umaps.py \
  --run runs/your-run --epoch 5 \
  --coding-labels /home/jposchl/data/processed/allen_visual_coding_neuropixels_2019/allen_vc_2019.csv
```

For rate-separated models, extraction retains all 256 coordinates and derives a
255-dimensional content collection for the joint and separate dataset UMAPs.
The rate coordinate is excluded from every UMAP. Epoch-specific progress and
artifact links are recorded under `embedding_overviews/epoch_N/status.json`.
Omitting `--epoch` selects the final checkpoint. Extraction requires an idle GPU
and at least 25 GiB free; UMAP fits use one CPU thread.

### Continuing runs and comparing representations

`train_nuclr_vc_full_baseline` trains the original encoder on the 58 VC sessions.
`train_nuclr_allen_ibl_rate_separated_continue` resumes the joint variant from
epoch 10 through epoch 20, preserving Adam state and starting a new learning-rate
cycle at `lr.cycle_start_epoch: 10`. The cycle uses one warmup epoch and nine cosine
epochs. Resuming a checkpoint within that cycle restores its scheduler state.
Continuation configs reference local checkpoint and frozen registry paths; update
those paths when moving to another machine. Checkpoints and run outputs are not
tracked in Git.

To wait for a local run to finish before launching the next authorized run:

```bash
.venv/bin/python scripts/queue_pretraining.py \
  --after-run runs/preceding-run \
  --config train_nuclr_allen_ibl_rate_separated_continue \
  --output runs/new-continuation --gpu 0 --min-free-gib 25
```

The watcher waits for successful completion, an idle GPU, and sufficient storage.
Its state is saved beside the output as `new-continuation.queue.json`.

Analysis runners use saved embeddings and one CPU thread:

- `scripts/compare_rate_separated.py`: matched firing-rate and dataset comparisons,
  excluding zero-rate neurons.
- `scripts/evaluate_brain_regions.py`: within-dataset Beryl region decoding with
  five held-out-animal folds and paired animal bootstrap comparisons.
- `scripts/analyze_biological_neighbors.py --variant-epoch 5`: cross-dataset region
  agreement among nearest neighbors, controlling neuron and population firing
  rates. Epoch 10 is the default. The cell-type comparison is exploratory because
  VC labels are LOLCAT predictions and IBL has no corresponding labels.
- `scripts/compare_vc_pretraining.py`: waits for the run supplied through
  `--wait-for-run`, extracts the completed VC-only epoch-10 checkpoint, and compares
  its region decoding with saved joint-model predictions. It checks identical
  pooling windows, neuron identities, labels, and animal folds before comparison.

Each runner exposes its required artifact IDs and paths through `--help`. The
Beryl mapping inputs and pinned source are in `results/cache/ibl_atlas_regions`.
Animal holdouts apply to probe fitting; these animals participated in encoder
pretraining. Equal epoch counts do not equalize total updates across dataset mixes.

### Gradient accumulation

Accumulation is opt-in through `micro_batch_size`. Missing or `null` retains the
existing behavior: `batch_size` window pairs across all GPUs in one optimizer
update. For example:

```yaml
batch_size: 24       # Total window pairs per optimizer update, across all GPUs.
micro_batch_size: 3  # Window pairs per GPU per forward/backward pass.
```

On one GPU this accumulates eight micro-batches; on two GPUs it accumulates four
micro-batches per GPU. Both sizes must be positive integers, and `batch_size`
must divide exactly by `micro_batch_size * number_of_GPUs`. Invalid combinations
raise an error before model/data setup; values are never rounded. With accumulation
disabled, `batch_size` must still divide exactly by the number of GPUs.

Gradients are normalized using the loss's actual matching-neuron-pair counts,
preserving its weighting across populations of different sizes. The learning-rate
schedule, training-step counter, and `steps.jsonl` advance on optimizer updates.
The run reports micro-batches separately. Any incomplete accumulation group at
the end of an epoch is announced during setup and flushed before checkpointing;
skipped views are handled using the remaining contributions. Resuming optimizer
state requires the same batch settings as the checkpoint.

If a Visual Coding import lacks `stimulus_epochs`, recover the coarse blocks from
Allen's public NWB metadata with:

```bash
.venv/bin/python scripts/restore_allen_vc_stimulus_epochs.py --apply
```

The repair verifies local stimulus timestamps against the source, caches the
small metadata tables under `results/cache/allen_vc_stimulus_epochs`, and adds only
the missing epoch group. It preserves source stimulus labels, including separate
Functional Connectivity protocol variants and spontaneous intervals. Source-marked
frame timing faults are recorded without changing the sampling domain. Omitting
`--apply` prepares and verifies the metadata without modifying the HDF5 files.

For a batch-size check on that GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$PWD" \
  .venv/bin/python scripts/benchmark_allen_batches.py --local \
  --output .validation-output/unique-batch-check --batch-sizes 2 4 6 7 8 12
```

Each candidate performs a largest-population stress step and a two-session
diagnostic epoch; benchmark weights are discarded. The full-data run is separate.
New runs accept repeated `--override` options. Resume uses the frozen run
configuration and rejects overrides.

To continue in a later existing allocation, pass the same run directory and its
`last_completed_checkpoint` from `status.json`:

```bash
bash scripts/launch_existing_allen_gpu.sh EXISTING_JOB_ID runs/unique-run-name /absolute/path/to/last.pt
```

Resume restores training state and continues toward the saved epoch target.
An interrupted partial epoch is replayed. Recording-quota window selection is
reproducible from the seed and epoch; worker augmentation and second-view random
states are not saved, so the complete resumed run is not bitwise identical. See
[validation notes](docs/VALIDATION.md) for the batch benchmark and runtime limits.
# nuclr-scaling
