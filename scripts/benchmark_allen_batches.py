"""Benchmark one diagnostic epoch per batch size, plus a worst-population step.

Use an existing GPU allocation, or explicitly select an authorized local GPU.
Benchmark weights are discarded; the full run starts fresh.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.artifacts.store import implementation_stamp, write_json
from src.datasets.allen_sessions import AllenRegistry
from src.datasets.constrastive_wrapper import ContrastiveDatasetWrapper
from src.utils import instantiate, move_to_device


def candidate(output, batch_size):
    result_path = output / f"batch_{batch_size}.json"
    result = {"batch_size": batch_size, "status": "running"}
    try:
        with initialize_config_dir(config_dir=str(Path.cwd() / "configs"), version_base="1.2"):
            cfg = compose(config_name="train_nuclr_allen_smoke")
        cfg.batch_size = batch_size
        cfg.ckpt.enable = False
        cfg.wandb.dir = str(output / "wandb")
        cfg.wandb.mode = "disabled"
        OmegaConf.update(cfg, "disable_pbar", True, force_add=True)
        trainer = instantiate(cfg.trainer, cfg=cfg)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Use the largest retained population, with unit dropout disabled. All
        # examples keep all neurons in both views, conservatively bounding the
        # randomly sampled unit counts seen in normal training.
        inventory = json.loads((output / "inventory.json").read_text())
        largest = max(inventory, key=lambda row: row["units"])
        registry = AllenRegistry()
        desc = registry.describe(largest["dataset_id"], largest["session_id"])
        begin = next(a for a, b in desc["intervals"] if b-a >= cfg.views.duration)
        data = registry.window(largest["dataset_id"], largest["session_id"], begin, begin+cfg.views.duration, desc)
        item = {"model_inputs": trainer.model.tokenize(data), "unit_seqlen": len(data.units),
                "unit_ids": data.units.id.tolist(), "probe_ids": data.units.probe_id.tolist()}
        stress_cpu = ContrastiveDatasetWrapper.collate([(item, item)] * batch_size, two_view=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stress = move_to_device(stress_cpu, trainer.device)
        with torch.autocast("cuda", dtype=trainer.precision.dtype):
            z = trainer.model(**stress["model_inputs"])
            loss, _ = trainer.ssl_loss(z, stress["unit_seqlen"], stress["unit_ids"], stress["probe_ids"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite stress loss")
        loss.backward()
        trainer.optimizer.step()  # Include AdamW moment allocation in the bound.
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        result.update(stress_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                      stress_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      stress_population=largest, total_gpu_gib=total_memory/2**30)
        del stress, stress_cpu, item, data, z, loss, trainer
        gc.collect()
        torch.cuda.empty_cache()
        # Fresh initialization and seed for the actual diagnostic epoch.
        import wandb
        wandb.finish()
        trainer = instantiate(cfg.trainer, cfg=cfg)
        losses = []
        samples = 0
        def record(module, args, kwargs, result_tuple):
            nonlocal samples
            value = float(result_tuple[0].detach())
            if not np.isfinite(value):
                raise RuntimeError("Nonfinite training loss")
            losses.append(value)
            samples += len(kwargs["seqlen"]) // 2
        hook = trainer.ssl_loss.register_forward_hook(record, with_kwargs=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        trainer.train()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        hook.remove()
        if not losses or trainer.train_step != len(trainer.train_loader):
            raise RuntimeError("Incomplete diagnostic epoch")
        headroom = 1 - result["stress_peak_gib"] / result["total_gpu_gib"]
        result.update(status="passed", epoch_seconds=elapsed, samples=samples,
                      samples_per_second=samples/elapsed, steps=len(losses),
                      epoch_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                      mean_loss=float(np.mean(losses)), worst_case_headroom=headroom,
                      eligible=headroom >= 0.15)
        trainer.logger.wandb_run.finish()
    except torch.OutOfMemoryError as error:
        result.update(status="out_of_memory", eligible=False, error=str(error))
    except Exception as error:
        result.update(status="failed", eligible=False, error=f"{type(error).__name__}: {error}")
        write_json(result_path, result)
        raise
    write_json(result_path, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 6, 8, 16])
    parser.add_argument("--local", action="store_true", help="Use an explicitly authorized local GPU without Slurm")
    args = parser.parse_args()
    if not args.local and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Requires an existing allocation or explicit --local execution")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    if any(size < 1 for size in args.batch_sizes) or (args.batch_size is not None and args.batch_size < 1):
        parser.error("Batch sizes must be positive")
    torch.cuda.set_device(0)
    output = Path(args.output).resolve()
    if args.batch_size is not None:
        return candidate(output, args.batch_size)
    output.mkdir(parents=True, exist_ok=False)
    registry = AllenRegistry()
    inventory, entries = [], []
    for dataset in ("allen_visual_behavior", "allen_visual_coding"):
        sessions = registry.sessions(dataset)
        if not sessions:
            raise ValueError(f"No sessions in {dataset}")
        entries.append({"dataset_id": dataset, "sessions": sessions})
        for session in sessions:
            desc = registry.describe(dataset, session)
            inventory.append({"dataset_id": dataset, "session_id": session,
                              "subject_id": desc["subject_id"], "units": len(desc["rows"]),
                              "windows_20s": sum(math.floor((b-a)/20) for a,b in desc["intervals"])})
    write_json(output / "inventory.json", inventory)
    write_json(output / "all_sessions.json", {"schema_version": 1,
               "purpose": "Pretrain on all sessions, no held-out subjects, as requested",
               "partitions": {"pretrain": entries, "eval": []}})
    print(f"ALL-SESSIONS: {len(inventory)} sessions; {sum(row['windows_20s'] for row in inventory)} windows per epoch", flush=True)
    results = []
    for size in sorted(set(args.batch_sizes)):
        print(f"BENCHMARK batch_size={size}", flush=True)
        command = [sys.executable, "-u", __file__, "--output", str(output), "--batch-size", str(size)]
        if args.local:
            command.append("--local")
        subprocess.run(command, check=True)
        result = json.loads((output / f"batch_{size}.json").read_text())
        results.append(result)
        if result["status"] == "out_of_memory":
            print("Larger sizes skipped after worst-case memory exhaustion.", flush=True)
            break
    candidates = [r for r in results if r.get("eligible")]
    if not candidates:
        raise RuntimeError("No tested batch size retained 15% worst-case memory headroom")
    best = max(candidates, key=lambda r: r["samples_per_second"])
    windows = sum(row["windows_20s"] for row in inventory)
    summary = {"implementation": implementation_stamp(), "results": results,
               "selected_batch_size": best["batch_size"], "selection": "fastest_tested_with_15_percent_worst_case_headroom",
               "sessions": len(inventory), "windows_per_epoch": windows,
               "estimated_epoch_seconds": windows/best["samples_per_second"],
               "estimated_50_epoch_hours": 50*windows/best["samples_per_second"]/3600,
               "estimate_caveat": "Extrapolation from two diagnostic sessions; full-data I/O and population sizes differ",
               "split_manifest": str(output / "all_sessions.json")}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
