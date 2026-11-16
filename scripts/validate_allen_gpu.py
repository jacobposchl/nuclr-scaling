"""One real-data epoch through the production trainer, in an existing GPU job.

Run from the repository root. The output directory must not already exist.
The diagnostic split is not a scientific benchmark or a scaling comparison.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.artifacts.store import implementation_stamp, write_json
from src.datasets.constrastive_wrapper import ContrastiveDatasetWrapper
from src.utils import move_to_device, instantiate as instantiate_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="train_nuclr_allen_smoke")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("Run only inside an existing allocation with exactly one visible GPU")
    torch.cuda.set_device(0)
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        raise RuntimeError("Run from the repository root")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "job_id": os.environ["SLURM_JOB_ID"],
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "implementation": implementation_stamp(), "purpose": "diagnostic_only"}
    write_json(output / "report.json", report)
    try:
        if os.environ.get("ZEN_SOURCE_ROOT"):
            sys.path.insert(0, str(root / "tests"))
            suite = unittest.defaultTestLoader.loadTestsFromName("test_nuclr.ZenParityTests")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError("Zen parity checks failed or were skipped")
            report["zen_parity_tests_passed"] = result.testsRun

        with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.2"):
            cfg = compose(config_name=args.config)
        if cfg.num_epochs != 1:
            raise ValueError("This diagnostic expects exactly one epoch")
        cfg.ckpt.dir = str(output / "checkpoints")
        cfg.wandb.dir = str(output / "wandb")
        cfg.wandb.mode = "disabled"
        OmegaConf.update(cfg, "disable_pbar", True, force_add=True)
        trainer = instantiate_trainer(cfg.trainer, cfg=cfg)
        OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
        initial = {name: value.detach().cpu().clone() for name, value in trainer.model.named_parameters()}
        expected_steps = len(trainer.train_loader)
        losses, gradients, learning_rates, datasets = [], [], [], Counter()

        # A fixed two-view window pair from each held-out session. The production
        # Allen trainer has no online evaluator, so this check is explicit here.
        from torch_brain.datasets import DatasetIndex
        eval_ds = instantiate(cfg.dataset, root=cfg.data_root, regime="eval", split="test",
                              transform=instantiate(cfg.eval_transform))
        wrapper = ContrastiveDatasetWrapper(eval_ds, trainer.model.tokenize)
        wrapper.enable_two_views(max_distance=cfg.views.max_distance, seed=cfg.seed)
        examples = []
        for rid, interval in wrapper.get_sampling_intervals().items():
            begin = next(float(a) for a, b in zip(interval.start, interval.end)
                         if b - a >= cfg.views.duration)
            examples.append(wrapper[DatasetIndex(recording_id=rid, start=begin, end=begin + cfg.views.duration)])
        heldout = wrapper.collate(examples, two_view=True)
        if heldout is None:
            raise RuntimeError("Empty held-out diagnostic batch")

        def evaluate(model, criterion):
            model.eval()
            criterion.eval()
            batch = move_to_device(heldout, trainer.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=trainer.precision.dtype):
                embeddings = model(**batch["model_inputs"])
                loss, _ = criterion(embeddings, batch["unit_seqlen"], batch["unit_ids"], batch["probe_ids"])
            if not torch.isfinite(embeddings).all() or not torch.isfinite(loss):
                raise RuntimeError("Nonfinite held-out output")
            return float(loss), embeddings.float().cpu()

        report["heldout_loss_before"], _ = evaluate(trainer.model, trainer.ssl_loss)
        report["heldout_sessions"] = eval_ds.recording_ids

        def capture_loss(module, positional, kwargs, result):
            if kwargs.get("prefix") != "train/":
                return
            value = float(result[0].detach())
            if not np.isfinite(value):
                raise RuntimeError("Nonfinite training loss")
            losses.append(value)
            learning_rates.append(trainer.optimizer.param_groups[0]["lr"])
            datasets.update({str(uid).split("/")[0] for uid in kwargs["unit_ids"]})
            record = {"step": len(losses), "loss": value, "lr": trainer.optimizer.param_groups[0]["lr"]}
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if len(losses) == 1 or len(losses) % 20 == 0:
                print(f"TRAIN {len(losses)}/{expected_steps}: loss={value:.5f}, elapsed={time.monotonic()-started:.1f}s", flush=True)

        def check_gradients(optimizer, positional, kwargs):
            grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            norms = torch.stack([g.detach().float().norm() for g in grads])
            if not torch.isfinite(norms).all():
                raise RuntimeError("Nonfinite gradient before optimizer step")
            gradients.append(float(norms.norm()))

        loss_hook = trainer.ssl_loss.register_forward_hook(capture_loss, with_kwargs=True)
        gradient_hook = trainer.optimizer.register_step_pre_hook(check_gradients)
        torch.cuda.reset_peak_memory_stats()
        train_start = time.monotonic()
        trainer.train()
        torch.cuda.synchronize()
        report["training_seconds"] = time.monotonic() - train_start
        loss_hook.remove()
        gradient_hook.remove()
        if len(losses) != expected_steps or trainer.train_step != expected_steps:
            raise RuntimeError("Epoch did not execute every expected optimizer step")
        if not np.isclose(max(learning_rates), cfg.lr.peak, rtol=1e-6, atol=0):
            raise RuntimeError("Learning rate did not reach the configured peak")
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        if not np.isclose(final_lr, cfg.lr.peak * cfg.lr.min_factor, rtol=1e-6, atol=0):
            raise RuntimeError("Learning rate did not reach the configured final value")
        expected_datasets = {rid.split("/")[0] for rid in trainer.train_dataset.recording_ids}
        if set(datasets) != expected_datasets:
            raise RuntimeError("Not all training datasets contributed to this epoch")
        report["heldout_loss_after"], trained_embeddings = evaluate(trainer.model, trainer.ssl_loss)
        changes = [float((p.detach().cpu() - initial[name]).norm()) for name, p in trainer.model.named_parameters()]
        if not all(np.isfinite(changes)) or max(changes) == 0:
            raise RuntimeError("Model parameters did not change finitely")
        checkpoint = Path(trainer.last_ckpt_path)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["epoch"] != 0 or saved["train_step"] != expected_steps:
            raise RuntimeError("Incorrect checkpoint progress")
        if "data_provenance" not in saved["cfg"]:
            raise RuntimeError("Checkpoint is missing dataset provenance")
        # Verify model, loss, optimizer, scheduler and progress restoration through
        # the production resume path; do not launch a second epoch.
        trainer.logger.wandb_run.finish()
        resumed_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        resumed_cfg.ckpt.load_from = str(checkpoint)
        resumed_cfg.ckpt.resume = True
        resumed_cfg.ckpt.dir = str(output / "resume_check")
        restored = instantiate_trainer(resumed_cfg.trainer, cfg=resumed_cfg)
        if restored.epoch != 1 or restored.train_step != expected_steps:
            raise RuntimeError("Resume progress mismatch")
        if restored.lr_scheduler.state_dict() != trainer.lr_scheduler.state_dict():
            raise RuntimeError("Scheduler resume mismatch")
        def equal_state(a, b):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
            elif isinstance(a, dict):
                if a.keys() != b.keys():
                    raise AssertionError("State keys differ")
                for key in a:
                    equal_state(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                if len(a) != len(b):
                    raise AssertionError("State lengths differ")
                for first, second in zip(a, b):
                    equal_state(first, second)
            elif a != b:
                raise AssertionError("State values differ")
        equal_state(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
        equal_state(trainer.model.state_dict(), restored.model.state_dict())
        equal_state(trainer.ssl_loss.state_dict(), restored.ssl_loss.state_dict())
        _, restored_embeddings = evaluate(restored.model, restored.ssl_loss)
        torch.testing.assert_close(trained_embeddings, restored_embeddings, rtol=0, atol=0)
        restored.logger.wandb_run.finish()
        report.update(status="passed", optimizer_steps=expected_steps,
                      dataset_batch_counts=dict(datasets),
                      first_loss=losses[0], last_loss=losses[-1], mean_loss=float(np.mean(losses)),
                      mean_first_20_losses=float(np.mean(losses[:20])), mean_last_20_losses=float(np.mean(losses[-20:])),
                      gradient_norm_range=[min(gradients), max(gradients)],
                      peak_learning_rate=max(learning_rates), final_learning_rate=final_lr,
                      peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                      max_parameter_change_norm=max(changes), checkpoint=str(checkpoint),
                      checkpoint_reload_and_resume_state="exact_match", elapsed_seconds=time.monotonic()-started)
        write_json(output / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic()-started)
        write_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
