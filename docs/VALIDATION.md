# CPU validation — November 12, 2026

This section records the November 12 CPU stage. Later GPU results are recorded
below; the limitations in this section describe what the CPU checks alone establish.

All Python execution, package preparation, and data inspection for this stage
ran in the user-provided Slurm allocation **10497173**, on **epyc-6-4**. Steps used
two CPUs with `CUDA_VISIBLE_DEVICES` empty and numerical-library thread limits of
two. No new allocation was created, no GPU workload was run, and no pretraining
experiment was launched. Login-node actions were limited to file inspection,
editing, and scheduler interaction.

## Results

The initial scaffold run discovered **20 tests: 19 passed and 1 was skipped**.
The skip was the original encoder's CUDA forward/backward parity check.

After adding grouped visualization, the full suite discovered **23 tests:
21 passed and 2 were skipped**. This run included the Zen source comparisons;
GPU parity and the opt-in real-data extraction were skipped. The earlier real-data
results below were not repeated. The new checks covered:

- PCA and optional scaling fitted exclusively on the selected fit subjects,
  despite large outliers in display-only subjects.
- Recoloring and changing displayed subjects reusing the projection artifact;
  changing fit subjects producing a different projection.
- Actual UMAP fitting and transformation of display-only units, finite coordinates,
  and preservation of fitted units' original coordinates.
- Grouped PCA/UMAP figures with categorical, unknown, and continuous metadata;
  invalid fields/selections and row limits applied after cohort selection.
- Firing rates matching independent spike counts over extraction windows and
  following the selected windows when deriving new unit averages.
- The existing linear classification probe and all prior CPU regressions.

Checks that passed include:

- Cache identity, reuse, payload integrity, checkpoint mutation detection,
  exclusive writers, and non-publication after a failed write.
- Namespaced neuron/probe IDs, spike-index remapping, sparse window reads,
  sampling domains with gaps, and mixed-dataset subject-split rejection.
- A synthetic mixed-Allen batch through tokenization, the encoder, contrastive
  loss, and backward, with finite gradients.
- CPU segmented attention outputs and gradients against independent block-masked
  scaled dot-product attention.
- Checkpoint registration → window/mean extraction → cached reuse → derived
  means → PCA → grouped linear probe, using synthetic data and a small untrained
  diagnostic checkpoint.
- Agreement between directly accumulated unit means and means derived from
  saved windows; explicit analysis row limits; rejection of duplicate units.
- Probe group separation and agreement between its stored scaler mean and the
  training rows alone.
- The original port's CPU loss, binning, collation, configuration, and checkpoint
  format checks. Its loss and projector gradients also matched the original
  Zen source on the tested ordinary input.

The real-data smoke test extracted one two-second window from session
`1044385384` (Visual Behavior) and session `715093703` (Visual Coding), with the
full retained population in each. It produced finite 16-dimensional vectors for
**1,451 units** in total: 567 Behavior and 884 Coding units. The model was a small
**untrained diagnostic model**, not a pretrained scientific baseline. Temporary
checkpoints, collections, and analyses created by tests were removed afterward.

The inventory command read unit metadata, subject IDs, probe grouping, and
sampling domains from **all 103 selected Behavior files and all 58 Coding files**.
It did not read all spike arrays or extract full-recording embeddings. Its local
report is `.validation-output/allen_inventory.json` (ignored output).

## Environment

The existing Zen environment contained torch-brain
`0.1.1.dev39+g6dcb06dda`, which lacks the API required by the port. Validation used
its Python 3.10 interpreter and installed packages, with **torch-brain v0.2.0**
(`bd46a574237e6618b8091d722c434ea8e6aef66a`) installed in the project-local
`.validation-deps` directory and placed first on `PYTHONPATH`. The existing Zen
environment was not modified.

Other observed versions were PyTorch 2.10.0, NumPy 2.2.6, h5py 3.16.0,
scikit-learn 1.7.2, and torchfoo `0.0.4.dev7+g03045994e`. This is a validation
environment, **not a lock or clean installation of the full requirements file**;
the requirements retain Zen's scikit-learn 1.8.0 pin. Before production runs,
create and validate a dedicated environment in an allocated compute job.
The grouped visualization checks used umap-learn 0.5.12, numba 0.67.0, and
pynndescent 0.6.0 from that existing environment, with two Numba threads permitted
and UMAP itself configured with `n_jobs=1` and a fixed seed.

The initial scaffold suite ran with `NUCLR_REAL_DATA_SMOKE=1` and
`ZEN_SOURCE_ROOT=/vast/projects/dyer1/lab/user/jacob/zen/repo`. A representative
invocation, while allocation 10497173 remains available, is:

```bash
srun --jobid=10497173 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --immediate=30 \
  /usr/bin/env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/.validation-deps:$PWD" \
  MPLCONFIGDIR="$PWD/.validation-output/matplotlib" \
  NUCLR_REAL_DATA_SMOKE=1 \
  ZEN_SOURCE_ROOT=/vast/projects/dyer1/lab/user/jacob/zen/repo \
  /vast/projects/dyer1/lab/user/jacob/zen/repo/.venv/bin/python \
  -m unittest discover -s tests -v
```

This attaches a step to the existing allocation. Do not omit `--jobid` or create
a replacement allocation without user authorization.
For the later visualization run, `NUCLR_REAL_DATA_SMOKE` was omitted, and
`NUMBA_NUM_THREADS=2` plus
`NUMBA_CACHE_DIR="$PWD/.validation-output/numba"` were added to the environment.

## Not established by these checks

- GPU numerical parity, GPU/DDP training, or full optimizer/scheduler resume.
- Scientific quality of pretrained embeddings or improvements from combining
  datasets. No trained-model evaluation or full dataset extraction was performed.
- Identical preprocessing/QC between Allen releases, or agreement between all
  Behavior derivatives. The derivative choice is explicit in the registry.
- A final training/evaluation cohort or sampling balance. Neither is selected
  automatically by the scaffold.
- A bitwise-reproducible environment or protection against changes that preserve
  source size/mtime when the fast `stat` fingerprint mode is used.

The prototype is a validated CPU workflow scaffold, not a completed scaling study.

## GPU diagnostic — November 15, 2026

The user explicitly authorized connecting to existing GPU allocation **10494778**
and running one epoch. All execution ran on **dgx028**, with one visible NVIDIA
B200 MIG 2g.45gb device (45,312 MiB), six allocated CPU threads, and numerical
thread limits of two. No new GPU allocation was requested. The existing Zen
Python environment and project-local torch-brain 0.2.0 overlay were reused:
PyTorch 2.10.0+cu128 and xformers 0.0.34.

The reusable setup is `configs/train_nuclr_allen_smoke.yaml`,
`configs/splits/allen_gpu_smoke.json`, and `scripts/validate_allen_gpu.py`.
It uses the original 10,534,656-parameter encoder, 256 dimensions, 20-second
windows, bf16, batch size 2, two loader workers, and a one-epoch learning-rate
schedule with 10% warmup. The fractional warmup budget is rounded to whole
optimizer steps so the warmup/cosine transition reaches the intended peak.
`tests/test_schedule.py` checks both the peak and final learning rate on a
nonintegral warmup budget; it passed in the allocation.

| Partition | Source | Session | Subject |
| --- | --- | --- | --- |
| Training | Behavior 2022 | 1044385384 | 524761 |
| Training | Coding 2019 | 715093703 | 699733581 |
| Held out | Behavior 2022 | 1047969464 | 509808 |
| Held out | Coding 2019 | 719161530 | 703279284 |

Training includes all valid fixed windows from these two sessions, with 1,451
retained source units and unit dropout during training. The held-out check uses
one fixed two-view example per held-out session, with no unit dropout. It is a
small diagnostic of finite outputs, not a held-out dataset-wide score or probe.
Both training datasets contribute under the existing recording-time-based
sampling policy. No balanced-dataset sampling or multi-GPU/DDP check was added.

The runner compares the original Zen encoder's CUDA outputs and parameter
gradients, plus loss/projector gradients; both parity tests passed. It then
checks every training loss and gradient, verifies parameter updates, saves a
checkpoint with data provenance, and restores it through the production resume
path. Restoration compares model, loss, optimizer moments, scheduler state,
epoch/step counters, and held-out embeddings exactly. It does not perform an
additional optimizer step after resuming. WandB uploads and automatic full
embedding extraction are disabled for this diagnostic.

Final run: `.validation-output/allen-gpu-20261115-run3/report.json` (**passed**).
Its log is `.validation-output/allen-gpu-20261115-run3.log`; its resolved config
and per-step losses are beside the report. Results:

- 319 optimizer steps; 55.26 seconds training, 71.63 seconds for the diagnostic
  runner including parity/setup/reload (excluding Python startup).
- Behavior appeared in 150 batches and Coding in 289; mixed batches count in both.
- All training losses, gradients, parameter updates and held-out outputs finite.
- Mean first/last 20 training losses: **4.862 → 3.830**.
- Fixed held-out window loss: **5.443 → 4.178**.
- Peak allocated GPU memory: **9.934 GiB**.
- Learning rate reached **5e-4** at the warmup transition and **5e-6** at epoch end.
- Exact checkpoint/model/loss/optimizer/scheduler restoration passed.
- Final checkpoint:
  `.validation-output/allen-gpu-20261115-run3/checkpoints/kcpzykkt/last.pt`.

This establishes that the mixed-data pipeline can train the standard model on
one GPU at batch size 2. It does not establish improvement over single-dataset
pretraining, full-release throughput, the default batch size 16, DDP correctness,
or downstream probing quality. No embedding collection was extracted in this run.

## All-session batch benchmark — November 15, 2026

The user selected all sessions with no subject holdouts. The recorded inventory
contains 103 Behavior 2022 sessions and 58 Coding 2019 sessions, giving 46,810
available 20-second windows per epoch. The explicit full-run manifest has an empty
evaluation partition. This does not alter the separate diagnostic fixture above.

`scripts/benchmark_allen_batches.py` ran inside existing allocation 10494778 on
the B200 MIG 45 GB slice. Each candidate first performed a complete forward,
backward, and AdamW step with all 1,307 neurons from the largest retained
population (Behavior session 1119946360), repeated across the batch with no unit
dropout. Successful candidates then trained a fresh model for one complete epoch
on the two diagnostic training sessions. Benchmark weights are discarded.

| Batch | Diagnostic epoch | Windows/second | Largest-population peak | Result |
| --- | --- | --- | --- | --- |
| 2 | 54.95 s | 11.61 | 15.84 GiB | Passed |
| 4 | 52.67 s | 12.08 | 31.55 GiB | Selected |
| 6 | Not run | — | Exhausted memory | Rejected |

Larger candidates were skipped after memory exhaustion. Batch 4 is the fastest
tested size with at least 15% headroom in the population stress test (28.7%
measured). This is a two-session throughput comparison, not a guarantee of
optimality over the full cohort. Different batches also change contrastive
negatives and optimizer updates; loss values are not a quality comparison.
Raw results: `.validation-output/batch-sweep-20261115/summary.json`.

Extrapolation gives approximately 64.6 minutes per full epoch and 53.8 hours for
50 epochs. Full-data I/O and population sizes can change that estimate. The
current allocation expires November 15 at 20:27 EST, so completion requires
resuming in a later allocation. No additional allocation was requested.

The full preset retains snapshots at completed epochs 10/20/30/40/50 and replaces
`last.pt` after every completed epoch using an atomic rename. The previous file
survives a failed save. The run directory records configuration, manifest,
registry, per-step losses, epoch timing, and the last completed checkpoint.
No partial-epoch checkpoint or automatic requeue is implemented.

Before launch, 11 tests passed and one opt-in real-data test was skipped using
`test_training_lifecycle`, `test_artifacts`, and `test_schedule` inside the existing
allocation. These cover all-session pretraining with an empty evaluation split,
continued rejection of subject leakage, completed-epoch checkpoint numbering,
preservation of the old checkpoint on write failure, and the learning-rate
schedule. The live run uses `scripts/run_allen_pretraining.py`; its current state
is recorded in `runs/allen-full-20261115/status.json`.

The full run started at 14:44 EST as Slurm step `10494778.7`, run ID `5ncqtv72`.
Initialization confirmed 161 sessions, 112,380 retained unit rows, and 11,702
batches per epoch. The first 100 steps completed with finite losses; their early
timing suggests roughly 56 minutes per epoch (still preliminary). No full epoch
or checkpoint from this run had completed at this check. The detached controller
continues inside the existing allocation; finishing 50 epochs remains pending.

## Local RTX 5090 validation — November 16, 2026

The user authorized a local batch-size benchmark and one validation epoch on
`nerdslab-ws02`, using its otherwise idle RTX 5090. No Slurm allocation was opened.
The runners now accept explicit `--local` execution, while retaining the allocation
requirement otherwise. Python 3.12.3 in this repository's `.venv` supplies the
installed requirements, including PyTorch 2.10.0, torch-brain 0.2.0, xformers
0.0.34, and scikit-learn 1.8.0. The runs used bf16, two loader workers, two numerical
threads, 20-second views, and the unchanged 10,534,656-parameter encoder.

The local files are visual-area derivatives under `/home/jposchl/data/processed`:
103 Behavior sessions with 29,122 units and 58 Coding sessions with 17,784 units.
Both datasets provide explicit probe IDs. These populations differ from the
earlier full-population cluster validation. There are 44,263 complete available
20-second windows; the sampler reports 52.69 seconds in shorter intervals that
cannot supply a full window.

Startup exposed a configuration-logging bug: shell `${...}` expressions in a
Git diff were parsed as OmegaConf interpolation. Git metadata now remains a plain
dictionary. Both focused regression tests passed, covering literal shell expressions
and unavailable Git metadata.

### Batch-size comparison

Each candidate ran a forward/backward/AdamW stress step with all 580 retained
neurons from the largest local population, Behavior session `1119946360`, in
both views with no unit dropout. A fresh model then trained for one epoch on
the two diagnostic sessions. Candidate weights were discarded. GPU memory below
is peak tensor allocation during the stress step, not allocator reservation.

| Batch | Stress peak GiB | Diagnostic window pairs/second | Result |
| --- | --- | --- | --- |
| 2 | 7.16 | 75.6 | Passed |
| 4 | 14.16 | 81.5 | Passed |
| 6 | 21.09 | 85.7 | Selected |
| 7 | 24.60 | 84.4 | Passed |
| 8 | 28.08 | 85.2 | Passed, below the 15% headroom criterion |
| 12 | — | — | Out of memory |

Batch 6 retained 32.7% of the CUDA-reported 31.32 GiB capacity in the stress
test, with 22.40 GiB peak allocator reservation. Throughput differences among
6/7/8 were small in this short benchmark; 6 provides greater memory headroom.
Sizes 9–11 were not tested, so this does not establish an exact maximum batch.
The mixed and full Allen presets now default to 6. Results are in
`.validation-output/rtx5090-batch-sweep-20261116-run2/summary.json`.

### Full one-epoch result

`runs/allen-rtx5090-one-epoch-20261116` completed one epoch across all 161 sessions
at batch 6, with 0.1 epoch of warmup. It did not start the normal 50-epoch run.

- 7,377 optimizer steps and 44,262 sampled window pairs; the final incomplete
  batch dropped one window.
- Epoch time, including the rolling checkpoint save: **623.86 seconds (10m 24s)**.
- Mean first/last 100 training losses: **3.6581 → 2.0835**; all losses finite.
- Peak allocated GPU memory: **12.46 GiB**; peak reserved memory: **28.15 GiB**.
  Variable population sizes increased allocator reservation beyond the stress
  test's reservation. No out-of-memory error occurred during the full epoch.
- Peak/final learning rates: **5e-4 / 5e-6**.
- Saved model and optimizer tensors were finite. All 112 model state tensors
  differed from the seeded initialization.
- Production resume restored model, projector, optimizer, scheduler, and progress
  exactly. No additional optimizer step was taken after restoring.

The report is `runs/allen-rtx5090-one-epoch-20261116/validation.json`; the checkpoint
is `runs/allen-rtx5090-one-epoch-20261116/checkpoints/cxn358x8/last.pt`.
This validates local training, memory capacity, and checkpoint restoration. All
sessions participated in pretraining; no held-out scientific score was measured.
