import os
import subprocess
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.utils.checkpoint import save_ckpt
from src.utils.logger import EPOCH_PBAR_FMT, Logger
from src.utils.util import Precision, expand_path, rank_zero_only, seed_everything

import torchfoo as tfoo


class Trainer(ABC):
    def __init__(self, cfg: DictConfig):
        seed_everything(cfg.seed)

        self.cfg = cfg
        self.precision = Precision(cfg.precision)
        self.rank = tfoo.dist.get_rank()
        self.world_size = tfoo.dist.get_world_size()
        self.is_distributed = dist.is_initialized()
        self.device = torch.device(self.rank)
        self.train_step = 0
        self.epoch = 0
        self.to_ckpt = {}

        if self.cfg.debug is not None:
            self.cfg.wandb.mode = "disabled"
            self.cfg.ckpt.enable = False

        if self.cfg.debug == "tokenizer":
            self.cfg.num_workers = 0

        self._setup_logger()
        self.logger.save_config(self.cfg)

        self.logger.info(f"data_root = {self.cfg.data_root}")
        self.logger.info(f"ckpt.dir = {self.cfg.ckpt.dir}")

        ckpt = self._load_ckpt()
        self.setup(ckpt)
        self._setup_ckpt()

    def train(self):
        if self.cfg.val.first:
            self.val_epoch()

        epoch_pbar = tqdm(
            range(self.epoch, self.cfg.num_epochs),
            initial=self.epoch,
            total=self.cfg.num_epochs,
            bar_format=EPOCH_PBAR_FMT,
            dynamic_ncols=True,
            disable=(self.rank != 0) or self.cfg.get("disable_pbar", False),
        )
        self.logger.set_epoch_pbar(epoch_pbar)

        for epoch in epoch_pbar:
            self.epoch = epoch
            self.barrier()
            self.train_epoch()
            self._save_checkpoint()
            self._save_last_checkpoint()

            user_asked_for_val = self.epoch % self.cfg.val.every_n_epochs == 0
            last_epoch = self.epoch == self.cfg.num_epochs - 1
            if user_asked_for_val or last_epoch:
                early_stop = self.val_epoch()

                if early_stop:
                    self.logger.info(f"Early stopping at the end of epoch {self.epoch}")
                    break

        return self.__dict__.get("return_value", None)

    @abstractmethod
    def train_epoch(self):
        """
        Perform a training epoch.
        """
        ...

    @abstractmethod
    def val_epoch(self):
        """
        Perform a validation epoch. This method is called every `val.every_n_epochs` epochs or at the end of training.

        Returns:
            early_stop : bool, True if early stopping should be triggered, False (or falsey) otherwise.
        """
        ...

    @abstractmethod
    def setup(self, ckpt: dict | None):
        """Set up the trainer.

        This method is called after setting up distributed, loading the checkpoint, and setting up logging.

        Args:
            ckpt: The checkpoint dictionary, if any.
        """
        ...

    def _setup_logger(self):
        self.logger = Logger(
            self.rank, disable_pbar=self.cfg.get("disable_pbar", False)
        )
        self.logger.init_wandb(self.cfg.wandb)
        self.logger.info(f"Run ID: {self.logger.run_id}")
        if self.cfg.wandb.mode != "online":
            self.logger.warn(f"W&B mode: {self.cfg.wandb.mode}")

    @rank_zero_only
    def add_checkpoint_items(self, **kwargs):
        """
        Add items to the checkpoint dictionary.

        Args:
            kwargs: The items to add to the checkpoint dictionary.
        """
        self.to_ckpt.update(kwargs)

    @rank_zero_only
    def _setup_ckpt(self):
        if not self.cfg.ckpt.enable:
            self.logger.warn("Checkpointing disabled")
            self.ckpt_dir = None
            return

        # Create checkpoint directory
        self.ckpt_dir = (expand_path(self.cfg.ckpt.dir) / self.logger.run_id).resolve()
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Checkpoint directory: {self.ckpt_dir}")

        # A copied source tree need not have a Git checkout to save checkpoints.
        try:
            diff = subprocess.check_output(["git", "diff", "HEAD"], stderr=subprocess.DEVNULL)
            commit_id = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except (OSError, subprocess.CalledProcessError):
            self.logger.warn("Git metadata unavailable; checkpoints will still be saved")
        else:
            diff_path = self.ckpt_dir / "diff.patch"
            diff_path.write_bytes(diff)
            self.logger.wandb_run.log_artifact(diff_path)
            (self.ckpt_dir / "commit_id.txt").write_text(commit_id)

        # Tell user what will be checkpointed
        if len(self.to_ckpt) > 0:
            self.logger.info(f"Checkpoints will track: {list(self.to_ckpt.keys())}")
        else:
            self.logger.warn(f"No items specified for checkpointing")

    @rank_zero_only
    def _save_checkpoint(self):
        if self.ckpt_dir is None:
            return

        every_n_epochs = self.cfg.ckpt.every_n_epochs
        if every_n_epochs is None:
            return
        if not isinstance(every_n_epochs, int) or every_n_epochs < 1:
            raise ValueError("ckpt.every_n_epochs must be a positive integer or null")
        if (self.epoch + 1) % every_n_epochs != 0:
            return

        self.last_ckpt_path = save_ckpt(
            filepath=self.ckpt_dir / f"epoch_{self.epoch + 1}.pt",
            train_step=self.train_step,
            epoch=self.epoch,
            run_id=self.logger.run_id,
            cfg=self.cfg,
            **self.to_ckpt,
        )

    @rank_zero_only
    def _save_last_checkpoint(self):
        if self.ckpt_dir is None:
            return
        if not self.cfg.ckpt.get("save_last", True):
            return

        self.last_ckpt_path = save_ckpt(
            filepath=self.ckpt_dir / "last.pt",
            train_step=self.train_step,
            epoch=self.epoch,
            run_id=self.logger.run_id,
            cfg=self.cfg,
            **self.to_ckpt,
        )

    @rank_zero_only
    def _save_best_checkpoint(self):
        if self.ckpt_dir is None:
            return

        self.best_ckpt_path = save_ckpt(
            filepath=self.ckpt_dir / f"best.pt",
            train_step=self.train_step,
            epoch=self.epoch,
            run_id=self.logger.run_id,
            cfg=self.cfg,
            **self.to_ckpt,
        )

    def _load_ckpt(self) -> dict | None:
        if self.cfg.ckpt.load_from is None:
            return

        if self.cfg.ckpt.get("model_only", False) and self.cfg.ckpt.resume:
            raise ValueError("Use ckpt.resume=false when ckpt.model_only=true")

        ckpt_path = expand_path(self.cfg.ckpt.load_from)
        if not ckpt_path.exists():
            # try this other path
            ckpt_path = expand_path(self.cfg.ckpt.dir) / self.cfg.ckpt.load_from
            if not ckpt_path.exists():
                raise ValueError(
                    f"Checkpoint not found at {self.cfg.ckpt.load_from} or {ckpt_path}"
                )
        self.cfg.ckpt.load_from = str(ckpt_path)

        self.logger.info(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=False,
        )

        if self.cfg.ckpt.resume:
            self.epoch = ckpt["epoch"] + 1
            self.train_step = ckpt["train_step"]
            self.logger.info(
                f"Resuming from: epoch = {self.epoch}, train_step = {self.train_step}"
            )

        return ckpt

    def make_ddp(self, module: torch.nn.Module, find_unused_parameters: bool = False):
        if self.is_distributed:
            ret = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module.to(self.rank))
            ret = DDP(
                ret,
                device_ids=[self.rank],
                find_unused_parameters=find_unused_parameters,
            )
            return ret
        else:
            return module

    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def log_lr(self):
        assert hasattr(self, "optimizer")
        assert isinstance(self.optimizer, torch.optim.Optimizer)
        optim = self.optimizer

        for i, param_group in enumerate(optim.param_groups):
            name = param_group["name"] if "name" in param_group else f"group_{i}"
            self.logger.log(f"lr/{name}", param_group["lr"])

    def log_weight_stats(self):
        if not self.cfg.log_weights:
            return

        for name, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                self.logger.log_dict(
                    {
                        f"{name}_weight_norm/{k}": v.norm()
                        for k, v in module.named_parameters()
                    }
                )
                self.logger.log_dict(
                    {
                        f"{name}_weight_max/{k}": v.max()
                        for k, v in module.named_parameters()
                    }
                )
                self.logger.log_dict(
                    {
                        f"{name}_weight_min/{k}": v.min()
                        for k, v in module.named_parameters()
                    }
                )

    def log_grad_stats(self):
        if not self.cfg.log_grads:
            return

        for name, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                # Individual norms
                param_grad_norms = {
                    f"{name}_grad_norm/{k}": v.grad.norm()
                    for k, v in module.named_parameters()
                    if v.grad is not None
                }
                self.logger.log_dict(param_grad_norms)
                # Total norms
                if param_grad_norms:
                    total_grad_norm = torch.norm(
                        torch.stack(list(param_grad_norms.values()))
                    )
                    self.logger.log(f"{name}_grad_norm/total", total_grad_norm)
                # Max
                self.logger.log_dict(
                    {
                        f"{name}_grad_max/{k}": v.grad.max()
                        for k, v in module.named_parameters()
                        if v.grad is not None
                    }
                )
                # Min
                self.logger.log_dict(
                    {
                        f"{name}_grad_min/{k}": v.grad.min()
                        for k, v in module.named_parameters()
                        if v.grad is not None
                    }
                )

    def push_logs(self):
        self.logger.log("train/epoch", self.epoch)
        self.logger.log("train/step", self.train_step)
        self.logger.push()
