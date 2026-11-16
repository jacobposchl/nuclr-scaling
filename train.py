import os, sys, signal
import hydra
import omegaconf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.utils import instantiate
from src.utils.logger import get_cli_logger
from src.utils.distributed import get_open_port, cleanup_ddp, setup_ddp


def run(cfg: omegaconf.DictConfig, rank: int, world_size: int):
    try:
        torch.cuda.set_device(rank)  # avoid putting things on GPU0 by default
        setup_ddp(rank, world_size, cfg.ddp)

        trainer = instantiate(cfg.trainer, cfg=cfg)
        trainer.train()
    finally:
        cleanup_ddp()


@hydra.main(version_base="1.2", config_path="./configs", config_name="train_nuclr")
def main(cfg: omegaconf.DictConfig):
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("NuCLR pretraining requires a CUDA GPU (Zen attention backend).")

    if world_size > 1 or cfg.ddp.force:
        if cfg.ddp.master_port is None:
            cfg.ddp.master_port = get_open_port()
            get_cli_logger().info(f"DDP MASTER_PORT = {cfg.ddp.master_port}")

    if world_size > 1:
        # Join every worker and propagate failures to the launching process.
        return mp.spawn(_run_worker, args=(cfg, world_size), nprocs=world_size, join=True)
    return run(cfg, 0, world_size)


def _run_worker(rank: int, cfg: omegaconf.DictConfig, world_size: int):
    return run(cfg, rank, world_size)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
