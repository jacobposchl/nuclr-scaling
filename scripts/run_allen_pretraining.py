"""Run or resume a recorded full-data experiment inside an existing GPU job."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import time
from types import MethodType

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.artifacts.store import implementation_stamp, write_json
from src.utils import instantiate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="train_nuclr_allen_full")
    parser.add_argument("--resume", help="Path to last.pt; uses the run's saved configuration")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires an existing allocation with exactly one visible GPU")
    torch.cuda.set_device(0)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    status_path = output / "status.json"
    if args.resume:
        cfg = OmegaConf.load(output / "resolved_config.yaml")
        cfg.ckpt.load_from = str(Path(args.resume).resolve(strict=True))
        cfg.ckpt.resume = True
    else:
        with initialize_config_dir(config_dir=str(Path.cwd() / "configs"), version_base="1.2"):
            cfg = compose(config_name=args.config)
        # Freeze the explicit session list and registry for future resumes.
        for key in ("split_manifest", "registry"):
            source = Path(cfg.dataset[key])
            destination = output / (key + ".json")
            destination.write_bytes(source.read_bytes())
            cfg.dataset[key] = str(destination)
    cfg.ckpt.dir = str(output / "checkpoints")
    cfg.wandb.dir = str(output / "wandb")
    cfg.wandb.mode = "disabled"
    OmegaConf.update(cfg, "disable_pbar", True, force_add=True)
    status = {"state": "starting", "job_id": os.environ["SLURM_JOB_ID"],
              "target_epochs": cfg.num_epochs, "batch_size": cfg.batch_size,
              "implementation": implementation_stamp(), "resume_from": args.resume}
    def publish(**updates):
        status.update(updates, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        temporary = output / ".status.tmp"
        write_json(temporary, status)
        os.replace(temporary, status_path)
    def interrupted(signum, frame):
        publish(state="interrupted", signal=signum,
                resume_note="Resume from last_completed_checkpoint; the partial epoch is not checkpointed")
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    publish()
    try:
        trainer = instantiate(cfg.trainer, cfg=cfg)
        # Include the actual source/session provenance populated by setup.
        OmegaConf.save(trainer.cfg, output / "resolved_config.yaml", resolve=True)
        steps = len(trainer.train_loader)
        registry_sessions = Counter(rid.split("/")[0] for rid in trainer.train_dataset.recording_ids)
        publish(state="running", run_id=trainer.logger.run_id, steps_per_epoch=steps,
                sessions=dict(registry_sessions), completed_epochs=trainer.epoch,
                last_completed_checkpoint=args.resume, checkpoint_dir=str(trainer.ckpt_dir))
        original_epoch = trainer.train_epoch
        original_last = trainer._save_last_checkpoint
        epoch_losses = []
        epoch_start = time.monotonic()
        overall_start = time.monotonic()
        start_epoch = trainer.epoch
        def train_epoch(self):
            nonlocal epoch_start
            epoch_losses.clear()
            epoch_start = time.monotonic()
            publish(current_epoch=self.epoch+1, epoch_step=0)
            original_epoch()
        trainer.train_epoch = MethodType(train_epoch, trainer)
        def checkpoint_complete(self):
            original_last()
            elapsed = time.monotonic() - epoch_start
            completed = self.epoch+1
            record = {"epoch": completed, "steps": len(epoch_losses),
                      "mean_loss": float(np.mean(epoch_losses)), "seconds": elapsed,
                      "train_step": self.train_step, "checkpoint": str(self.last_ckpt_path)}
            with (output / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(record)+"\n")
            publish(completed_epochs=completed, last_completed_checkpoint=str(self.last_ckpt_path),
                    last_epoch=record, estimated_remaining_hours=(cfg.num_epochs-completed)*
                    (time.monotonic()-overall_start)/(completed-start_epoch)/3600)
            print("EPOCH_COMPLETE " + json.dumps(record), flush=True)
        trainer._save_last_checkpoint = MethodType(checkpoint_complete, trainer)
        def loss_record(module, positional, kwargs, result):
            if kwargs.get("prefix") != "train/":
                return
            value = float(result[0].detach())
            if not np.isfinite(value):
                raise RuntimeError("Nonfinite training loss")
            epoch_losses.append(value)
            record = {"epoch": trainer.epoch+1, "epoch_step": len(epoch_losses),
                      "train_step": trainer.train_step+1, "loss": value,
                      "lr": trainer.optimizer.param_groups[0]["lr"]}
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record)+"\n")
            if len(epoch_losses) == 1 or len(epoch_losses) % 100 == 0:
                elapsed = time.monotonic()-epoch_start
                eta = (steps-len(epoch_losses))*elapsed/len(epoch_losses)
                publish(epoch_step=len(epoch_losses), loss=value, epoch_remaining_minutes=eta/60)
                print(f"TRAIN epoch={trainer.epoch+1}/{cfg.num_epochs} step={len(epoch_losses)}/{steps} "
                      f"loss={value:.5f} epoch_eta_minutes={eta/60:.1f}", flush=True)
        hook = trainer.ssl_loss.register_forward_hook(loss_record, with_kwargs=True)
        trainer.train()
        hook.remove()
        publish(state="complete", completed_epochs=cfg.num_epochs,
                last_completed_checkpoint=str(trainer.last_ckpt_path))
        trainer.logger.wandb_run.finish()
    except Exception as error:
        publish(state="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
