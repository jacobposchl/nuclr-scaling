import os
import copy
from typing import Any, Literal, TypeAlias, get_args
from tqdm import tqdm

import numpy as np
from torch_brain.data import Data, Interval
from torch_brain.datasets import Dataset, SpikingDatasetMixin
from src.datasets.behavior_wrapper import BehaviorDatasetMixin

from torch_brain.batching import chain

Split: TypeAlias = Literal["train", "val", "test", "train+val"]
BenchmarkRegime: TypeAlias = Literal["pretrain", "eval"]
BehaviorTask: TypeAlias = Literal["running_speed", "reward"]

TASK_CFG = {
    "running_speed": dict(
        valkey="behavior.running_speed.speed",
        timekey="behavior.running_speed.timestamps",
        type="regression",
        loss_name="mse",
        metric_names=("r2", "pearson", "pearson_sq"),
    ),
    "reward": dict(
        type="classification",
        loss_name="bce",
        metric_names=("binary_bacc", "auroc", "binary_macrof1", "binary_macro_aurpc"),
    ),
}


def get_recording_list(regime: BenchmarkRegime, split: Split) -> list[str]:
    def load_list(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    base_dir = os.path.dirname(__file__)

    if regime == "eval":
        path = os.path.join(base_dir, "eval_ids.txt")
        return load_list(path)
    elif regime == "pretrain":
        if split == "train":
            path = os.path.join(base_dir, "pretrain_train_ids.txt")
            return load_list(path)
        elif split == "val":
            path = os.path.join(base_dir, "pretrain_val_ids.txt")
            return load_list(path)
        elif split == "train+val":
            train_path = os.path.join(base_dir, "pretrain_train_ids.txt")
            val_path = os.path.join(base_dir, "pretrain_val_ids.txt")
            return load_list(train_path) + load_list(val_path)
        else:
            raise ValueError(f"pretrain regime has no {split} split.")
    else:
        raise ValueError("Invalid regime and/or split selected.")


class AllenVisualBehaviorBenchmark(BehaviorDatasetMixin, SpikingDatasetMixin, Dataset):
    """
    Dataset for the Allen Institute Visual Behavior Neuropixels Dataset.

    Args:
        root: The root directory of the dataset.
        dirname: The name of the dataset (and the directory containing its data).
        recording_ids: The recording ids to include in the dataset. If None, all
                       recordings in the dataset related to the regime are included.
        transform: The transform(s) to apply to the data.
        split: The split of the dataset (train, val, test).
        regime: The regime of the dataset (pretrain, eval).
        exclude_blocks: Names under ``recording.intervals.blocks`` to remove from
                       all sampling domains.
        return_target: Whether to return the target(s) during __getitem__. This is automatically set
                       to True for the eval regime.
        split_pretrain_across_trials: If True, within "pretrain" regime, the train/val splits
            will be across trials (instead of subjects/sessions).
    """

    spiking_dataset_mixin_uniquify_unit_ids = True
    _behavior_enabled = False
    # evaluator_cls = AllenVBBenchmarkEvaluator
    evaluator_cls = None

    def __init__(
        self,
        root: str,
        regime: BenchmarkRegime,
        split: Split,
        task: BehaviorTask | None,
        context_window: float = 0.8,
        behavior_sfreq: float = 60,
        dirname: str = "allen_vbn_2022",
        recording_ids: list[str] | None = None,
        transform: Any | None = None,
        normalize_behavior: bool = True,
        return_target: bool = False,
        exclude_blocks: list[str] | tuple[str, ...] | None = None,
        split_pretrain_across_trials: bool = False,
    ):

        # Verification
        if regime not in get_args(BenchmarkRegime):
            raise ValueError(
                f"regime={regime} not well defined should be ('pretrain', 'eval')"
            )
        if split not in get_args(Split):
            raise ValueError(
                f"split={split} not well define should be ('train', 'val', 'test')"
            )
        if regime == "pretrain" and split == "test":
            raise ValueError("pretrain regime has no test split")
        if task is not None:
            assert task in get_args(BehaviorTask)

        if regime == "pretrain" and split_pretrain_across_trials:
            regime_rids = get_recording_list(regime, "train+val")
        else:
            regime_rids = get_recording_list(regime, split)

        if recording_ids is None:
            recording_ids = regime_rids
        else:
            recording_ids = [str(x) for x in recording_ids]
            illegal = set(recording_ids) - set(regime_rids)
            assert len(illegal) == 0, f"RIDs not in regime: {illegal}"

        super().__init__(
            # Doesn't expect regime folders. Just pass in dataset_dir as preprocessed folder
            dataset_dir=os.path.join(root, dirname),
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=[
                "session.id",
                "subject.id",
                "units.id",
            ],
        )

        self.CONTEXT_WINDOW = context_window
        self.BEHAVIOR_SFREQ = behavior_sfreq

        self.split_pretrain_across_trials = split_pretrain_across_trials
        self.regime = regime
        self.split = split
        self.task = task
        self.return_target = return_target
        self.exclude_blocks = tuple(exclude_blocks or ())
        if self.task is not None:
            self.task_cfg = TASK_CFG[self.task]

    def get_sampling_intervals(self):
        ans = {}
        for rid in self.recording_ids:
            data = self.get_recording(rid)
            ans[rid] = data.intervals.blocks.active.materialize()
        return ans

    @property
    def task_type(self) -> str:
        return self.task_cfg["type"]  # type: ignore

    @property
    def loss_name(self) -> str:
        return self.task_cfg["loss_name"]  # type: ignore

    @property
    def metric_names(self) -> list[str]:
        return list(self.task_cfg["metric_names"])

    def enable_behavior(self):
        assert self.task in get_args(BehaviorTask)
        self._behavior_enabled = True
        self.task_cfg = TASK_CFG[self.task]

        self.normalize_behavior = self.task_cfg["type"] == "regression"
        if self.normalize_behavior:
            self._set_normalization_coeffs()

    def _set_normalization_coeffs(self):
        self.normalization_coeffs = {}

        desc = "Computing Normalization Coeffs"
        for rid in tqdm(self.recording_ids, leave=False, desc=desc):
            rec = self.get_recording(rid)
            domain = self._get_behavior_sampling_domain(rec, "train")
            if domain is None:
                continue

            vals = []
            for start, end in zip(domain.start, domain.end):
                slc = rec.slice(start, end)
                valkey = self.task_cfg["valkey"]
                vals.append(slc.get_nested_attribute(valkey))
            vals = np.concat(vals, axis=0)

            match self.task:
                case "running_speed":
                    ans = (vals.mean(axis=0), vals.std(axis=0))
                case _:
                    raise NotImplementedError(f"Not impl for {self.task}")

            self.normalization_coeffs[rid] = ans

    def get_target(self, data: Data):
        rid: str = data.session.id
        task_cfg = self.task_cfg

        match self.task:
            case "running_speed":
                key = task_cfg["valkey"]
                assert isinstance(key, str)
                values = data.get_nested_attribute(key)

                if self.normalize_behavior:
                    mean, std = self.normalization_coeffs[rid]
                    values = (values - mean) / std
            case "reward":
                values = np.isfinite(data.behavior.trials.reward_time[:1])
            case _:
                raise NotImplementedError(f"Not impl for {self.task}")

        values = values[..., None].astype(np.float32)
        if task_cfg["type"] == "regression":
            key = task_cfg["timekey"]
            assert isinstance(key, str)
            ts = data.get_nested_attribute(key)
        else:
            # some frame-level classification task
            ts = self.CONTEXT_WINDOW / 2.0

        return {
            "values": chain(values),
            "ts": chain(ts),
            "seqlen": len(values),
            "session_id": data.session.id,
            "absolute_time": data._absolute_start,
        }

    def get_behavior_sampling_intervals(self):
        """
        Task-specific behavior windows for decoder training
        """
        ret = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            domain = self._get_behavior_sampling_domain(rec, self.split)
            if domain is not None:
                ret[rid] = domain
        return ret

    def _get_behavior_sampling_domain(
        self, recording: Data, split: str
    ) -> Interval | None:
        """
        For behavior domain, pretraining regime uses all windows for the selected task,
        while eval regime uses a pre-selected subset of those windows.
        """

        match self.task:
            case "reward" | "running_speed":
                trials = copy.deepcopy(recording.behavior.trials)
                mask = np.isfinite(trials.change_time)
                trials = trials.select_by_mask(mask)
                ans = Interval(
                    start=trials.change_time,
                    end=trials.change_time + self.CONTEXT_WINDOW,
                )
            case _:
                raise NotImplementedError()

        active_period = recording.intervals.blocks.active
        ans = ans & active_period
        assert np.isclose(ans.end - ans.start, self.CONTEXT_WINDOW).all()

        ans = self._split_intervals(ans, split)
        return ans

        benchmark = recording.intervals.benchmark
        task_windows = getattr(benchmark, self.task)

        if self.regime == "pretrain":
            domain = task_windows.all
        elif self.regime == "eval":
            if not hasattr(task_windows, split):
                raise ValueError(f"Unknown split: {split}")
            domain = getattr(task_windows, split)
        else:
            raise ValueError(f"Unknown regime: {self.regime}")
        if len(domain) == 0:
            return None
        domain = self._remove_blocks_from_domain(recording, domain, self.exclude_blocks)
        if len(domain) == 0:
            return None
        return domain

    def _remove_blocks_from_domain(
        self,
        recording: Data,
        domain: Interval,
        block_names: tuple[str, ...],
    ) -> Interval:
        for block_name in block_names:
            if not hasattr(recording.intervals.blocks, block_name):
                raise ValueError(f"Unknown exclusion block: {block_name}")
            domain = domain.difference(getattr(recording.intervals.blocks, block_name))
        return domain

    def _split_intervals(self, x: Interval, split: Split | None = None):
        if split is None:
            split = self.split

        if self.regime == "pretrain" and not self.split_pretrain_across_trials:
            return x

        n = len(x)
        if self.regime == "eval":
            n_train, n_test = int(0.4 * n), int(0.4 * n)
        else:
            n_train, n_test = int(0.9 * n), 0

        n_val = n - n_train - n_test
        assert n_val > 0

        match split:
            case "train":
                mask = np.zeros(len(x), dtype=bool)
                mask[:n_train] = True
                return x.select_by_mask(mask)
            case "val":
                mask = np.zeros(len(x), dtype=bool)
                mask[n_train : n_train + n_val] = True
                return x.select_by_mask(mask)
            case "test":
                mask = np.zeros(len(x), dtype=bool)
                mask[-n_test:] = True
                return x.select_by_mask(mask)
            case "train+val":
                mask = np.zeros(len(x), dtype=bool)
                mask[: n_train + n_val] = True
                return x.select_by_mask(mask)

        raise ValueError(f"split {split} does not make sense")
