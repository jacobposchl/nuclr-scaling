import copy
from typing_extensions import Self
from collections import defaultdict
import logging
import os
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args, Callable, TypedDict
from tqdm import tqdm

import numpy as np
import pandas as pd
from torch_brain.data import Data, Interval
from torch_brain.datasets import Dataset, SpikingDatasetMixin, DatasetIndex
from torch_brain.batching import chain

from src.utils import get_cli_logger
from src.datasets.behavior_wrapper import BehaviorDatasetMixin, TaskType
from .evaluator import IBLBenchmarkBrainRegionEvaluator

Split: TypeAlias = Literal["train", "val", "test", "train+val"]
BenchmarkRegime: TypeAlias = Literal["pretrain", "eval"]

BehaviorTask: TypeAlias = Literal[
    # "block_prior",
    # "choice",
    "reward",
    # "stimulus_side",
    # "stimulus_contrast",
    "whisker",
    "wheel_speed",
    "wheel_vel",
    # "right_paw_speed",
    # "left_paw_speed",
    # "licking_rate",
]


class TaskCfg(TypedDict):
    valkey: str
    timekey: str | None
    type: TaskType
    loss_name: str
    metric_names: list[str]


TASK_CFG = {
    "whisker": TaskCfg(
        valkey="whisker.motion_energy",
        timekey="whisker.timestamps",
        type="regression",
        loss_name="mse",
        metric_names=["r2", "pearson", "pearson_sq"],
    ),
    "wheel_speed": TaskCfg(
        valkey="wheel.speed",
        timekey="wheel.timestamps",
        type="regression",
        loss_name="mse",
        metric_names=["r2", "pearson", "pearson_sq"],
    ),
    "wheel_vel": TaskCfg(
        valkey="wheel.vel",
        timekey="wheel.timestamps",
        type="regression",
        loss_name="mse",
        metric_names=["pearson", "r2", "pearson_sq"],
    ),
    "reward": TaskCfg(
        valkey="task_aligned_intervals.reward.reward",
        timekey=None,
        type="classification",
        loss_name="bce",
        metric_names=["binary_bacc", "binary_macro_aurpc", "auroc", "binary_macrof1"],
    ),
}


def _get_behavior_qc_recording_list(regime: BenchmarkRegime, split: Split) -> list[str]:
    from .zs_behavior_rid_split import ZS_TRAIN_RIDS, ZS_VAL_RIDS, ZS_TEST_RIDS

    if regime == "pretrain" and split == "train":
        return ZS_TRAIN_RIDS

    if regime == "pretrain" and split == "val":
        return ZS_VAL_RIDS

    if regime == "pretrain" and split == "train+val":
        return ZS_TRAIN_RIDS + ZS_VAL_RIDS

    if regime == "eval":
        return ZS_TEST_RIDS

    raise ValueError(f"Unknown configuration {regime=} {split=}")


def _get_neural_qc_recording_list(regime: BenchmarkRegime) -> list[str]:
    with open(Path(__file__).parent / f"{regime}_eids.txt") as f:
        ret = [line.strip() for line in f if line.strip()]

    # Remove sessions for with all probes have qc_neural == FAIL
    csv = _get_qc()
    csv = csv[np.isin(csv.eid, ret)]
    csv = csv.groupby("eid").qc_neural.apply(lambda x: (x != "FAIL").any())
    ret = csv[csv].index.tolist()

    return ret


def _subset_pretrain_subjects(rids: list[str], factor: float):
    csv = _get_qc()
    csv = csv[np.isin(csv.eid, rids)]

    subjects = sorted(csv.subject_id.unique().tolist())
    rng = np.random.default_rng(seed=0)
    rng.shuffle(subjects)

    num_subjects = len(subjects)
    num_keep = int(num_subjects * factor)
    subjects_keep = subjects[:num_keep]
    csv_keep = csv[np.isin(csv.subject_id, subjects_keep)]
    eids_keep = sorted(csv_keep.eid.unique().tolist())

    get_cli_logger().info(
        f"Keeping {num_keep}/{num_subjects} subjects"
        f", {len(eids_keep)}/{len(csv)} sessions"
    )

    return eids_keep


class IBLBrainWideBench2026(BehaviorDatasetMixin, SpikingDatasetMixin, Dataset):
    """
    Dataset for the IBL BrainWideBench benchmark.

    Args:
        root: The root directory of the dataset.
        dirname: The name of the dataset (and the directory containing its data).
        recording_ids: The recording ids to include in the dataset. If None, all
                       recordings in the dataset related to the regime are included.
        transform: The transform(s) to apply to the data.
        split: The split of the dataset (train, val, test).
        regime: The regime of the dataset (pretrain, eval).
        split_type: The type of split to use (causal, non-causal, causal-60).
        return_target: Whether to return the target(s) during __getitem__. This is automatically set
                       to True for the eval regime.
    """

    spiking_dataset_mixin_uniquify_unit_ids = True
    _behavior_enabled = False
    evaluator_cls = IBLBenchmarkBrainRegionEvaluator

    CONTEXT_WINDOW = 1.0  # seconds
    BEHAVIOR_SFREQ = 50  # Hz

    def __init__(
        self,
        root: str,
        regime: BenchmarkRegime,
        split: Split,
        task: BehaviorTask | None,
        recording_level_qc: Literal["behavior", "neural"],
        pretrain_subject_factor: float | None = None,
        train_trial_factor: float | None = None,
        dirname: str = "ibl_brain_wide_bench_2026",
        recording_ids: list[str] | None = None,
        transform: Any | None = None,
        split_pretrain_across_trials: bool = False,
    ):

        assert regime in get_args(BenchmarkRegime)
        assert split in get_args(Split)

        if recording_level_qc == "behavior":
            if regime == "pretrain" and split_pretrain_across_trials:
                legal_rids = _get_behavior_qc_recording_list(regime, "train+val")
            else:
                legal_rids = _get_behavior_qc_recording_list(regime, split)
        elif recording_level_qc == "neural":
            legal_rids = _get_neural_qc_recording_list(regime)

        if regime == "pretrain" and split in ("train", "train+val"):
            if pretrain_subject_factor is not None:
                legal_rids = _subset_pretrain_subjects(
                    rids=legal_rids,
                    factor=pretrain_subject_factor,
                )
        if recording_ids is None:
            recording_ids = legal_rids
        else:
            illegal = set(recording_ids) - set(legal_rids)
            assert len(illegal) == 0, f"RIDs not in regime: {illegal}"

        super().__init__(
            dataset_dir=os.path.join(root, dirname, regime),
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
        )

        self.regime = regime
        self.split = split
        self.train_trial_factor = train_trial_factor
        self.task = task
        self.split_pretrain_across_trials = split_pretrain_across_trials

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

            match self.task:
                case "whisker":
                    stats = rec.normalize.whisker.motion_energy
                    ans = (stats.mean, stats.std)

                case "wheel_speed" | "wheel_vel":
                    vals = []
                    for start, end in zip(domain.start, domain.end):
                        slc = rec.slice(start, end)
                        valkey = self.task_cfg["valkey"]
                        vals.append(slc.get_nested_attribute(valkey))
                    vals = np.concat(vals, axis=0)

                    ans = (vals.mean(axis=0), vals.std(axis=0))
                    # rms = np.sqrt(np.mean(np.square(vals), axis=0))
                    # ans = (0, rms)
                case _:
                    raise NotImplementedError(f"Not impl for {self.task}")

            self.normalization_coeffs[rid] = ans

    def get_target(self, data: Data):
        rid: str = data.session.id
        task_cfg = self.task_cfg
        values = data.get_nested_attribute(task_cfg["valkey"])

        if self.normalize_behavior:
            match self.task:
                case "wheel_speed" | "wheel_vel" | "whisker":
                    mean, std = self.normalization_coeffs[rid]
                    values = (values - mean) / std
                case _:
                    raise NotImplementedError(f"Not impl for {self.task}")

        values = values[..., None].astype(np.float32)
        if task_cfg["type"] == "regression":
            assert task_cfg["timekey"] is not None
            ts = data.get_nested_attribute(task_cfg["timekey"])
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

    @property
    def task_type(self) -> str:
        assert self._behavior_enabled
        return self.task_cfg["type"]

    @property
    def loss_name(self) -> str:
        assert self._behavior_enabled
        return self.task_cfg["loss_name"]

    @property
    def metric_names(self) -> list[str]:
        assert self._behavior_enabled
        return self.task_cfg["metric_names"]

    def get_behavior_sampling_intervals(self):
        trial_ratio = None
        if self.split == "train":
            trial_ratio = self.train_trial_factor

        ret = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            domain = self._get_behavior_sampling_domain(rec, self.split, trial_ratio)
            if domain is not None:
                ret[rid] = domain
        return ret

    def _get_behavior_sampling_domain(
        self,
        recording: Data,
        split: str,
        trial_factor: float | None = None,
    ) -> Interval | None:
        domain_key = f"{split}_domain"
        if self.regime == "pretrain" and not self.split_pretrain_across_trials:
            # we are using entire recordings in pretrain
            domain_key = "domain"

        match self.task:
            case "wheel_speed" | "wheel_vel":
                interval = (
                    recording.task_aligned_intervals.movement_intervals
                    & recording.wheel.domain
                    & recording.get_nested_attribute(domain_key)
                )
            case "whisker":
                if "whisker" not in recording:
                    return None
                interval = (
                    recording.task_aligned_intervals.movement_intervals
                    & recording.whisker.domain
                    & recording.get_nested_attribute(domain_key)
                )
            case "reward":
                interval = (
                    recording.task_aligned_intervals.reward
                    & recording.get_nested_attribute(domain_key)
                )
            case _:
                raise ValueError(f"Unkown task: {self.task}")

        if trial_factor is None:
            return interval

        total = len(interval)
        to_keep = int(total * trial_factor)
        return Interval(start=interval.start[-to_keep:], end=interval.end[-to_keep:])


def _get_qc():
    qc_path = os.path.join(os.path.dirname(__file__), f"bwm_qc.csv")
    csv = pd.read_csv(qc_path)
    return csv


def _generate_recording_list(regime: BenchmarkRegime, split: Split) -> list[str]:
    path = os.path.join(os.path.dirname(__file__), f"{regime}_eids.txt")
    with open(path) as f:
        ret = [line.strip() for line in f if line.strip()]

    if regime == "eval":
        return ret

    csv = _get_qc()
    csv = csv[np.isin(csv.eid, ret)]
    # want: qc_behavior == PASS and qc_neural != FAIL
    # assuming: qc_overall == PASS for all eval probes
    mask_behav = csv.groupby("eid").qc_behavior.apply(lambda x: (x == "PASS").all())
    mask_neural = csv.groupby("eid").qc_neural.apply(lambda x: (x != "FAIL").any())
    new_df = mask_behav & mask_neural
    pretrain_rids = new_df[new_df].index.tolist()

    csv = csv[csv.eid.isin(pretrain_rids)]

    subjects = sorted(csv.subject_id.unique().tolist())
    rng = np.random.default_rng(seed=0)
    rng.shuffle(subjects)

    val_subjects = subjects[:12]
    train_subjects = subjects[12:]
    train_csv = csv[csv.subject_id.isin(train_subjects)]
    val_csv = csv[csv.subject_id.isin(val_subjects)]

    if split == "train":
        return sorted(train_csv.eid.unique().tolist())
    elif split == "val":
        return sorted(val_csv.eid.unique().tolist())
    else:
        raise ValueError("You should not be here")
