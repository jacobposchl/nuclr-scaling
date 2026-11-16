from typing import Literal
from omegaconf import DictConfig
from functools import partial
import multiprocessing as mp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.utils import instantiate

import torchfoo as tfoo
from torch_brain.samplers import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)

from .base import Trainer
from src.utils import (
    move_to_device,
    get_num_params,
    human_readable,
    expand_path,
)
from src.samplers import DistributedSamplerWrapper

from src.datasets.constrastive_wrapper import ContrastiveDatasetWrapper
from src.utils.emb_evaluator import EmbeddingCollector


class NuCLRPreTrainer(Trainer):

    def train(self):
        result = super().train()
        self.test()
        return result

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")

        if ckpt is not None:
            self.ckpt_cfg = DictConfig(ckpt["cfg"])

        self.setup_model(ckpt)
        self.setup_ssl_loss(ckpt)
        self.setup_train_loader()
        self.doing_eval = self.train_dataset.evaluator_cls is not None
        self.setup_eval_loader()
        self.setup_evaluator()
        self.setup_optimizer(ckpt)

        self.model = self.make_ddp(self.model)
        self.ssl_loss = self.make_ddp(self.ssl_loss)

    def train_epoch(self):
        self.model.train()
        self.ssl_loss.train()
        self.train_loader.sampler.set_epoch(self.epoch)  # type: ignore
        if self.doing_eval:
            self.train_collector.reset()

        for X in self.logger.get_pbar(self.train_loader, prefix="train"):
            # All ranks must agree before entering DDP forward/backward.
            usable = torch.tensor(int(X is not None), device=self.device)
            if dist.is_initialized():
                dist.all_reduce(usable, op=dist.ReduceOp.MIN)
            if not usable.item():
                self.logger.warn("Skipping batch: at least one rank has no usable views")
                continue
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                X = move_to_device(X, self.device)
                z = self.model(**X["model_inputs"])
                assert torch.isfinite(z).all()

                loss, loss_dict = self.ssl_loss(
                    x=z,
                    seqlen=X["unit_seqlen"],
                    unit_ids=X["unit_ids"],
                    probe_ids=X["probe_ids"],
                    prefix="train/",
                )
                assert torch.isfinite(loss).all()

            if self.doing_eval:
                self.train_collector.update(z, X["unit_ids"])

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            self.logger.log_dict(loss_dict)
            self.logger.log("Loss", loss.item(), pbar=True, wandb=False)
            self.log_lr()
            self.push_logs()
            self.train_step += 1

    @torch.inference_mode()
    def val_epoch(self):
        if not self.doing_eval:
            return

        assert self.train_collector is not None
        assert self.eval_collector is not None

        self.model.eval()
        self.ssl_loss.eval()
        if self.train_step == 0:
            # val.first has no training-epoch embeddings to evaluate yet.
            self.train_collector.reset()
            _, loader = self.get_sequential_one_view_loader("pretrain")
            self.collect_embs(loader, self.train_collector, "eval-pretrain")
        self.eval_collector.reset()
        self.collect_embs(self.eval_loader, self.eval_collector, "eval")

        train_embs, train_uids = self.train_collector.compute()
        eval_embs, eval_uids = self.eval_collector.compute()

        if self.rank == 0:
            assert self.evaluator is not None
            metrics = self.evaluator(
                train_embs=train_embs,
                train_uids=train_uids,
                eval_embs=eval_embs,
                eval_uids=eval_uids,
            )
            self.logger.log_dict(metrics)
            self.push_logs()

    @torch.inference_mode()
    def test(self):
        """Compute and save the final unit representations here"""
        if not self.cfg.save_embs.enable:
            self.logger.info("Final embeddings not saved (save_embs.enable=false)")
            return

        assert self.train_collector is not None
        assert self.eval_collector is not None
        self.logger.info("Computing final embeddings (save_embs.enable=true)")

        if self.rank == 0:
            save_path = expand_path(self.cfg.save_embs.path)
            if save_path.is_dir() or save_path.suffix != ".pt":
                save_path = save_path / f"{self.logger.run_id}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Will save in {save_path}")

        self.model.eval()
        self.train_collector.reset()
        self.eval_collector.reset()
        _, sequential_train_loader = self.get_sequential_one_view_loader("pretrain")
        self.collect_embs(sequential_train_loader, self.train_collector, "emb-pretrain")
        self.collect_embs(self.eval_loader, self.eval_collector, "emb-eval")

        train_embs, train_uids = self.train_collector.compute()
        eval_embs, eval_uids = self.eval_collector.compute()

        if self.rank == 0:
            torch.save(
                dict(
                    train_embs=train_embs.cpu(),
                    train_uids=train_uids,
                    eval_embs=eval_embs.cpu(),
                    eval_uids=eval_uids,
                ),
                save_path,
            )
            self.logger.info(f"Embeddings saved at {save_path}")

    def collect_embs(self, loader, collector, prefix: str):
        # Sequential shards can have different lengths. Avoid DDP's per-forward
        # collectives; the collector reduces the completed sums across ranks.
        model = self.model.module if isinstance(self.model, DDP) else self.model
        for X in self.logger.get_pbar(loader, prefix=prefix):
            if X is None:
                # Some sessions have all bad neurons. Combine this with
                # the fact that here we are doing sequential sampling,
                # we could end up with batches containing only bad sessions.
                continue
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                X = move_to_device(X, self.device)
                z = model(**X["model_inputs"])
                assert torch.isfinite(z).all()

            collector.update(z, X["unit_ids"])

    def setup_model(self, ckpt: dict | None):
        self.logger.info(f"Precision: {self.precision}")

        if ckpt is not None:
            self.cfg.model = self.ckpt_cfg.model
            self.logger.info("Model: Overriding config from ckpt")
            self.logger.info(self.cfg.model)
        else:
            self.cfg.model.ctx_duration = self.cfg.views.duration
        self.model: torch.nn.Module = instantiate(self.cfg.model)

        num_params, _ = get_num_params(self.model)
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Model: params - {human_readable(num_params)}")

        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.logger.info("Model: loaded state from checkpoint")

        self.model = self.model.to(self.device)
        self.model.train()
        self.add_checkpoint_items(model=self.model)

    def setup_ssl_loss(self, ckpt: dict | None):
        if ckpt is not None:
            if not self.cfg.ckpt.model_only:
                self.cfg.loss = self.ckpt_cfg.loss
                self.logger.info("Loss: Overriding config from ckpt")
                self.logger.info(self.cfg.loss)

        self.ssl_loss: torch.nn.Module = instantiate(
            self.cfg.loss,
            dim_in=self.model.emb_dim,
        )
        self.logger.info(f"Loss: {self.ssl_loss.__class__}")

        num_params, _ = get_num_params(self.ssl_loss)
        self.logger.info(f"Loss: params - {human_readable(num_params)}")

        if ckpt is not None:
            if not self.cfg.ckpt.model_only:
                self.ssl_loss.load_state_dict(ckpt["ssl_loss_state_dict"])
                self.logger.info("Loss: loaded state from checkpoint")
            else:
                self.logger.info("Loss: checkpoint not loaded due to ckpt.model_only")

        self.ssl_loss = self.ssl_loss.to(self.device)
        self.ssl_loss.train()
        self.add_checkpoint_items(ssl_loss=self.ssl_loss)

    def setup_optimizer(self, ckpt: dict | None):
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        decay = []
        no_decay = []

        def update_params(module: torch.nn.Module):
            for name, param in module.named_parameters():
                if "norm" in name or "bias" in name:
                    no_decay.append(param)
                else:
                    decay.append(param)

        update_params(self.model)
        update_params(self.ssl_loss)

        params = [
            {"params": decay, "weight_decay": self.cfg.wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr.peak)

        steps_per_epoch = len(self.train_loader)
        if steps_per_epoch == 0:
            raise ValueError("Training loader is empty; reduce batch_size or check data windows")
        # Fractional epochs are useful for short diagnostic runs. SequentialLR
        # switches/reset schedulers at integer optimizer-step milestones.
        warmup_steps = round(self.cfg.lr.rise * steps_per_epoch)
        total_steps = steps_per_epoch * self.cfg.num_epochs
        if not 0 < warmup_steps < total_steps:
            raise ValueError("Warmup must round to at least one step and leave at least one cosine step")
        s1 = LinearLR(
            self.optimizer,
            start_factor=self.cfg.lr.min_factor,
            total_iters=warmup_steps,
        )
        s2 = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.cfg.lr.peak * self.cfg.lr.min_factor,
        )
        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[s1, s2],
            milestones=[warmup_steps],
        )

        if ckpt is not None:
            if self.cfg.ckpt.resume:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
                self.logger.info("Optimizer & Scheduler: loaded state from checkpoint")
            else:
                self.logger.info(
                    "Optim & Scheduler: ckpt not loaded due to ckpt.resume=false"
                )

        # both optim and scheduler states needed for resuming
        self.add_checkpoint_items(
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

        self.logger.info(f"Optimizer: {self.optimizer.__class__}")
        self.logger.info(f"Num params: {tfoo.module.count_parameters(self.model):,}")
        self.logger.info(
            f"Num trainable params: "
            f"{tfoo.module.count_parameters(self.model, 'trainable'):,}"
        )
        self.logger.info(f"LR Scheduler: {self.lr_scheduler}")

    def setup_train_loader(self):
        if self.cfg.batch_size < self.world_size or self.cfg.batch_size % self.world_size:
            raise ValueError("batch_size must be positive and divisible by world_size")
        base_ds = instantiate(
            self.cfg.dataset,
            root=self.cfg.data_root,
            regime="pretrain",
            split="train+val",
            transform=instantiate(self.cfg.train_transform),
        )
        if getattr(base_ds, "provenance", None) is not None:
            from omegaconf import OmegaConf
            # Snapshot the actual selections, not just a mutable manifest path.
            OmegaConf.update(self.cfg, "data_provenance", base_ds.provenance, force_add=True)
        ds = ContrastiveDatasetWrapper(
            dataset=base_ds,
            tokenizer=self.model.tokenize,  # ty:ignore[invalid-argument-type]
        )
        ds.enable_two_views(
            max_distance=self.cfg.views.max_distance,
            seed=self.cfg.seed,
        )

        sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=ds.get_sampling_intervals(),
                window_length=self.model.ctx_duration,  # ty:ignore[invalid-argument-type]
                generator=torch.Generator().manual_seed(self.cfg.seed),
            ),
            drop_last=True,
        )

        has_workers = self.cfg.num_workers > 0
        loader = torch.utils.data.DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=partial(ds.collate, two_view=True),
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )
        self.logger.info(
            f"Pretrain dataset: {len(ds.recording_ids)} sessions"
            f", {len(ds.get_unit_ids())} units"
            f", {len(loader)} batches per epoch."
        )

        self.train_dataset = ds
        self.train_loader = loader

    def setup_eval_loader(self):
        if not (self.doing_eval or self.cfg.save_embs.enable):
            return

        ds, loader = self.get_sequential_one_view_loader("eval")
        self.logger.info(
            f"Eval dataset: {len(ds.recording_ids)} sessions"
            f", {len(ds.get_unit_ids())} units"
            f", {len(loader)} batches per epoch."
        )
        self.eval_dataset = ds
        self.eval_loader = loader

    def setup_evaluator(self):
        self.train_collector = None
        self.eval_collector = None
        self.evaluator = None
        if not (self.doing_eval or self.cfg.save_embs.enable):
            return

        self.train_collector = EmbeddingCollector(
            unit_ids=self.train_dataset.get_unit_ids(),
            dim=self.model.emb_dim,  # ty:ignore[invalid-argument-type]
            device=self.device,
        )
        self.eval_collector = EmbeddingCollector(
            unit_ids=self.eval_dataset.get_unit_ids(),
            dim=self.model.emb_dim,  # ty:ignore[invalid-argument-type]
            device=self.device,
        )
        if self.doing_eval and self.rank == 0:
            self.evaluator = self.train_dataset.evaluator_cls(
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                world_size=self.world_size,
            )
        elif not self.doing_eval:
            self.logger.info("No online evaluator; final embedding export remains available.")

    def get_sequential_one_view_loader(self, regime: Literal["pretrain", "eval"]):
        """Get dataset and loader used for computing final embeddings used for BR classification"""
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = self.model
        if isinstance(model, DDP):
            model = model.module

        base_ds = instantiate(
            self.cfg.dataset,
            root=self.cfg.data_root,
            regime=regime,
            split="train+val" if regime == "pretrain" else "test",
            transform=instantiate(self.cfg.eval_transform),
        )
        ds = ContrastiveDatasetWrapper(
            dataset=base_ds,
            tokenizer=model.tokenize,
        )

        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=ds.get_sampling_intervals(),  # ty:ignore[invalid-argument-type]
                window_length=model.ctx_duration,  # ty:ignore[invalid-argument-type]
                step=0.5 * model.ctx_duration,  # ty:ignore[unsupported-operator]
                drop_short=True,
            )
        )

        has_workers = self.cfg.num_workers > 0
        loader = torch.utils.data.DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=partial(ds.collate, two_view=False),
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )

        return ds, loader
