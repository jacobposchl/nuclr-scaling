# Zen NuCLR port notes

Source checkout: `/vast/projects/dyer1/lab/user/jacob/zen/repo`

Source commit: `5090787b6fe099109dddbd8dd484d4db109ed22a`

Copied code retains Zen's authorship and existing acknowledgments. This is a
focused extraction of the pretraining dependency path, not a new implementation
of the scientific method. The source checkout has not been edited.

## Baseline behavior retained

| Component | Zen behavior |
| --- | --- |
| Input | 20-second windows, 20-ms spike bins, 1-second latent patches |
| Encoder | Width 256; four heads of width 64; two temporal blocks; four spatial/temporal blocks |
| Position and pooling | Rotary temporal embeddings, temporal value rotation, mean over time |
| Spatial attention | Separate attention segments for each probe, view, and time patch |
| Dropout | Linear/residual dropout 0.3; attention dropout 0 |
| Views | Two windows from the same session, starts within 120 seconds; independently transformed |
| Augmentation | Retain a random integer unit count from floor(0.5 × units) through all units, then drop failed neural-QC probes |
| Positives | Matching unit IDs across the two views |
| Denominator | Exclude self-pairs, positive pairs in DCL mode, and pairs across probes |
| Aggregation | Compute loss within each sample's pair of views; weight samples by directed positive-pair count |
| Projection head | LayerNorm → Linear → ReLU → Linear, output normalized before similarity |
| Temperature | 0.2 |
| Optimizer | AdamW, peak LR 5e-4, weight decay 1e-2 except norm/bias parameters |
| Schedule | 50 epochs, one-epoch linear warmup, cosine decay to 1% of peak LR |
| Precision | bfloat16 autocast |
| Batch size | 80 for IBL, 16 for Allen, globally across GPUs |
| Evaluation | IBL brain-region linear probing; Allen online evaluator disabled |

The original port copied the rotary attention implementation unchanged; the
collection scaffold now adds segmented CPU attention for CPU extraction, keeping
the CUDA kernel path intact. The encoder network,
forward pass, and parameter names are unchanged. Source dataset adapters retain
their behavior-related methods because those are part of the original adapter
classes; no behavior decoders or decoder training loops are included.

## Deliberate changes from the source

1. **Entry point:** default to `train_nuclr`; report missing CUDA explicitly;
   join all spawned GPU workers and propagate their failures.
2. **Probe boundaries:** compute attention segment lengths in the existing unit
   order. Zen's sorted `np.unique` counts can assign the wrong lengths when
   contiguous probes appear in a different order with unequal unit counts. Units
   must still be contiguous within each probe; no reordering is introduced.
3. **Loss edge cases:** advance the sample offset even when no positive matches
   exist; return differentiable zero when no valid pairs exist; avoid computing
   `logsumexp` on all-negative-infinity rows, whose gradients are undefined.
   Preserve Zen's distributed mean reduction and positive-pair weighting for
   usable samples. These changes intentionally differ on invalid/degenerate
   inputs; ordinary objective equivalence still needs the prepared tests.
4. **View distance:** `max_distance=0` means the same window. Zen's `or inf`
   expression accidentally converted zero into unlimited distance. The later
   mixed-dataset scaffold also restricts second views to actual intervals, so
   views cannot cross interior gaps.
5. **Trainer setup:** resolve the evaluator from the dataset, replacing an
   undefined local variable. Instantiate it only on rank zero. Supply the missing
   `ckpt.model_only=false` config field and reject model-only resume.
6. **Empty batches:** if QC removes every view on any rank, skip that batch on
   all ranks before distributed forward/backward. Report an empty training
   loader, invalid global batch size, or invalid warmup duration explicitly.
7. **Validation and export:** populate training embeddings for `val.first`,
   flush evaluation metrics, invoke the existing final-export method after
   training, and allow Allen export without an online evaluator. Sequential
   pretraining extraction uses `train+val` (Allen has no pretrain `test` split),
   retains the last partial batch, and uses the unwrapped model/tokenizer to
   avoid DDP attribute errors and collectives on uneven evaluation shards.
8. **Output paths/checkpoints:** create embedding directories even if absent;
   retain checkpoint writing when Git metadata is unavailable; store checkpoint
   paths as config-compatible strings. RNG states remain outside the checkpoint.
9. **Dependencies:** remove the encoder's unused direct xFormers import; retain
   Zen's xFormers version in requirements for environment compatibility. Omit
   unrelated decoder/tooling dependencies and use the `python-dotenv` package
   name for the existing `dotenv` import.

`full_denom` is accepted and stored but unused in Zen's loss; it remains unused
here. No new interpretation has been assigned to this flag. Attention dropout
also retains Zen's behavior: nonzero values print a warning but are not applied.

## Scope and outstanding validation

The initial port was not executed. Subsequently, the user supplied an existing
CPU allocation and authorized validation of the collection scaffold. CPU checks
and small real-data diagnostic extraction have now run; see
[validation notes](docs/VALIDATION.md). No training job or GPU validation has run,
and no claim of full numerical or training-result parity is made.

Known inherited constraints to revisit before scaling:

- The IBL evaluator expects fixed full-benchmark metadata counts. Subset/scaling
  experiments will need a deliberate evaluation policy.
- The second-view sampler now excludes interior gaps. Its NumPy RNG is still
  copied into dataloader workers, as in Zen; worker/rank-specific sampling
  and reproducible resume should be addressed deliberately before new datasets.
- The distributed sampler computes its length by materializing the underlying
  sampler. This retains Zen's RNG-consumption behavior.
- The collector includes units with no observations as NaN vectors. Real-data
  checks should confirm which units are excluded by QC and window coverage.
- Multi-GPU loss scaling, empty-rank handling, export synchronization, GPU
  attention compatibility, real-data loading, and checkpoint resume need runtime
  validation. Synthetic CPU tests cannot establish those properties.
- The package versions are not a full environment lock, particularly
  `torchfoo@dev` and unpinned dependencies.

The mixed-Allen adapter and grouped analysis protocols now provide a scaffold.
Dataset balancing, final experimental splits, and scaling sweeps remain deferred.
