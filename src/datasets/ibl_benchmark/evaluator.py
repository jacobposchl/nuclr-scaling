from pathlib import Path
from typing import List, Dict, Any, Literal
import os
import pickle
import numpy as np
import pandas as pd
import torch
import ray
import matplotlib.pyplot as plt
import hashlib

from torch_brain.datasets import Dataset
from src.utils.logger import get_cli_logger

from src.utils.emb_evaluator import (
    kfold_assignment,
    rankme,
    KFoldClassifier,
    ZeroshotClassifier,
    LeaveOneOutClassifier,
)

logger = get_cli_logger()


class IBLBenchmarkBrainRegionEvaluator:
    """An evaluator for linear-probe brain-region performance

    This, in conjunction with the Collector class, can be used to provide an
    "online" performance measure.

    Args:
        train_dataset: the multi-session training dataset
        eval_dataset: the multi-session eval dataset
        world_size: the number of parallel CPU workers for linear probing will be
            set to ``world_size * 7``
    """

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        world_size: int,
    ):

        self.train_md = _get_md_from_dataset(train_dataset, "pretrain")
        self.eval_md = _get_md_from_dataset(eval_dataset, "eval")

        logger.info(f"train_md size: {len(self.train_md)}")
        logger.info(f"eval_md size: {len(self.eval_md)}")

        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        ray.init(address="local", num_cpus=world_size * 7)

    def __call__(
        self,
        train_embs: torch.Tensor,
        train_uids: list[str] | np.ndarray,
        eval_embs: torch.Tensor,
        eval_uids: list[str] | np.ndarray,
    ) -> Dict[str, Any]:

        train_embs_on_device = train_embs
        train_embs = train_embs.detach().cpu().numpy()
        train_uids = np.array(train_uids).astype(str)

        eval_embs_on_device = eval_embs
        eval_embs = eval_embs.detach().cpu().numpy()
        eval_uids = np.array(eval_uids).astype(str)

        # Train masking
        # mask what is finite
        mask = np.all(np.isfinite(train_embs), axis=-1)
        if not mask.all():
            logger.info(f"{(~mask).sum()} train_embs are not finite")
        train_embs, train_uids = train_embs[mask], train_uids[mask]
        # mask what we can use
        mask = pd.Index(train_uids).isin(self.train_md.index)  # faster than np.isin
        train_embs, train_uids = train_embs[mask], train_uids[mask]

        # Val masking
        # mask what is finite
        mask = np.all(np.isfinite(eval_embs), axis=-1)
        if not mask.all():
            logger.info(f"{(~mask).sum()} eval_embs are not finite")
        eval_embs, eval_uids = eval_embs[mask], eval_uids[mask]
        # mask what we can use
        mask = pd.Index(eval_uids).isin(self.eval_md.index)  # faster than np.isin
        eval_embs, eval_uids = eval_embs[mask], eval_uids[mask]

        train_md = self.train_md.loc[train_uids]
        eval_md = self.eval_md.loc[eval_uids]

        if len(eval_md) != len(self.eval_md):
            raise ValueError(
                "Some eval embeddings are missing."
                f" Got {len(eval_md)}, expected {len(self.eval_md)}"
            )

        self.futures = []

        # Subject-wise K-fold on train embeddings
        train_br_classifier = LeaveOneOutClassifier("train/l1subout_br", True)
        train_br_classifier.launch(
            X=train_embs,
            y=train_md.brain_region.values,
            s=kfold_assignment(train_md.subject_id.values, 5),
        )
        self.futures.append(train_br_classifier)

        # Zero-shot classification
        eval_br_classifier = ZeroshotClassifier("eval/zs_br", True)
        eval_br_classifier.launch(
            X_train=train_embs,
            y_train=train_md.brain_region.values,
            X_test=eval_embs,
            y_test=eval_md.brain_region.values,
        )
        self.futures.append(eval_br_classifier)

        # Rank me (not parallel, because on GPU)
        rankme_train = rankme(train_embs_on_device)
        rankme_val = rankme(eval_embs_on_device)

        metrics = {
            "train/rankme": rankme_train,
            "eval/rankme": rankme_val,
        }

        # Finalize all futures
        for future in self.futures:
            _metrics = future.finalize()
            metrics.update(_metrics)

        plt.close()

        return metrics


# Expected neurons after unit and probe-level QC
_EXPECTED_MD_SIZE = {
    "pretrain": 41834,
    "eval": 3209,
}


def _get_md_from_dataset(ds: Dataset, regime: Literal["pretrain", "eval"]):

    unit_ids = []
    regions = []
    session_ids = []
    subject_ids = []
    depths = []
    probe_ids = []
    qc_neural_alignments = []
    qc_neurals = []
    firing_rates = []
    unit_qc_labels = []

    for rid in ds.recording_ids:
        rec = ds.get_recording(rid)
        unit_ids.append(rec.units.id.astype(str))
        regions.append(rec.units.region_cosmos.astype(str))
        session_ids.extend([rid] * len(rec.units))
        subject_ids.extend([rec.subject.id] * len(rec.units))
        depths.extend(rec.units.depths)
        probe_ids.extend(rec.units.probe_id.astype(str))
        qc_neural_alignments.extend(rec.units.qc_neural_alignment.astype(str))
        qc_neurals.extend(rec.units.qc_neural.astype(str))
        firing_rates.extend(rec.units.firing_rate)
        unit_qc_labels.extend(rec.units.label)

    ans = pd.DataFrame(
        {
            "unit_id": np.concatenate(unit_ids),
            "brain_region": np.concatenate(regions),
            "session_id": session_ids,
            "subject_id": subject_ids,
            "depths": depths,
            "probe_ids": probe_ids,
            "qc_neural_alignment": qc_neural_alignments,
            "qc_neural": qc_neurals,
            "firing_rate": firing_rates,
            "unit_qc_label": unit_qc_labels,
        }
    ).set_index("unit_id")
    # Apply all QC metrics needed for TS3
    if regime == "eval":
        ans = ans[ans.qc_neural.values == "PASS"]
    else:
        ans = ans[ans.qc_neural.values != "FAIL"]
    ans = ans[ans.qc_neural_alignment.values == "PASS"]
    ans = ans[ans.unit_qc_label == 1.0]
    # ans = ans[ans.firing_rate > 1.0]
    ans = ans[~np.isin(ans.brain_region.values, ["void", "root"])]

    exp_size = _EXPECTED_MD_SIZE[regime]
    if len(ans) != exp_size:
        raise RuntimeError(
            f"Number of candidate {regime} units doesn't match expectation."
            f"Expected: {exp_size}, Got: {len(ans)}"
        )

    return ans
