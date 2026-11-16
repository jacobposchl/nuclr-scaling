# NuCLR scaling

This project starts with a port of Zen's NuCLR pretraining implementation.
The project includes a mixed-Allen data scaffold and an artifact workflow for
checkpoint registration, embedding collections, and reusable analyses. See
[Collections and analyses](docs/COLLECTIONS.md) for dataset choices and usage.
The all-session pretraining preset uses all 103 Visual Behavior 2022 sessions and
58 Visual Coding 2019 sessions, with no subjects held out. A diagnostic mixed-Allen
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

These are instructions, not commands that have been executed for this port.
Use Python 3.10+ and a CUDA environment compatible with the PyTorch 2.10 attention
backend. Training still requires a supported GPU. A segmented CPU attention
implementation supports the separate collection-extraction workflow and small
validation runs. Run installation and Python commands only in an allocated
compute job, never on the login node.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency versions follow Zen, including its moving `torchfoo@dev`
dependency. A fully locked environment is still needed for reproducible runs.

Set `DATA_ROOT` to a parent directory with the existing, preprocessed data:

```text
DATA_ROOT/
  ibl_brain_wide_bench_2026/
    pretrain/
    eval/
  allen_vbn_2022/
```

Preprocessing pipelines and datasets are not copied here. IBL uses its neural-QC
session lists. Allen samples the `intervals.blocks.active` domain. Both use
`task: null` for pretraining. An optional local `.env` can supply `DATA_ROOT` and
`CKPT_DIR`; no credentials or environment file were copied from Zen.

IBL baseline:

```bash
python train.py --config-name=train_nuclr \
  data_root=/path/to/data wandb.mode=disabled ckpt.enable=true
```

Allen VBN baseline:

```bash
python train.py --config-name=train_nuclr_allen_vbn \
  data_root=/path/to/data wandb.mode=disabled ckpt.enable=true
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
holdout, 50 epochs, batch size 4, and two loader workers. Sampling follows available
20-second windows without dataset balancing. Checkpoints retain completed epochs
10, 20, 30, 40, and 50; `last.pt` is atomically replaced after every completed
epoch. It includes the model, loss projector, optimizer, and scheduler.

Run from the repository root using an **existing** compatible GPU allocation:

```bash
bash scripts/launch_existing_allen_gpu.sh EXISTING_JOB_ID runs/unique-run-name
```

The launcher does not request an allocation. It currently targets the B200 MIG
45 GB partition and the validated Zen environment with the local dependency
overlay. The runner freezes the session manifest and registry, saves a resolved
configuration, and records `status.json`, `steps.jsonl`, and `epochs.jsonl` in
the run directory. Keep the controller alive or detach it from the terminal.

To continue in a later existing allocation, pass the same run directory and its
`last_completed_checkpoint` from `status.json`:

```bash
bash scripts/launch_existing_allen_gpu.sh EXISTING_JOB_ID runs/unique-run-name /absolute/path/to/last.pt
```

Resume restores training state and continues toward the original 50-epoch target.
An interrupted partial epoch is replayed; data-loader random state is not saved,
so resumed sampling is not guaranteed bitwise identical. See
[validation notes](docs/VALIDATION.md) for the batch benchmark and runtime limits.
# nuclr-scaling
