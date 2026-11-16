from __future__ import annotations
from typing import Union, Dict, Tuple
import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import Tensor

from src.utils.distributed import all_reduce

NEG_INF = float("-inf")
N_VIEWS = 2


class NuCLRLoss(nn.Module):
    def __init__(
        self,
        dim_in: int,
        tau: float,
        dcl: bool,
        projector: bool = True,
        full_denom: bool = False,
    ):
        super().__init__()
        self.tau: float = tau
        self.dcl = dcl
        self.full_denom = full_denom

        self.tau_inv = 1.0 / self.tau

        if projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(dim_in),
                nn.Linear(dim_in, dim_in),
                nn.ReLU(),
                nn.Linear(dim_in, dim_in),
            )
        else:
            self.projector = nn.Identity()

    def forward(
        self,
        x: Tensor,
        seqlen: Tensor,
        unit_ids: np.ndarray,
        probe_ids: np.ndarray,
        prefix: str = "",
    ) -> tuple[Tensor, dict]:

        dev = x.device
        x = F.normalize(self.projector(x), dim=-1)
        # Convert uids/pids into unique indices
        # it is faster to create masks on gpu using these
        # rather than using the raw string arrays
        uidx = torch.tensor(np.unique(unit_ids, return_inverse=True)[1], device=dev)
        pidx = torch.tensor(np.unique(probe_ids, return_inverse=True)[1], device=dev)

        # seqlen here = num neurons in all views for each sample
        seqlen = seqlen.view(-1, N_VIEWS).sum(1)
        bs = len(seqlen)

        loss_numers, loss_denoms = x.new_zeros(bs), x.new_zeros(bs)
        num_matches = x.new_zeros(bs, dtype=torch.long)
        ptr: int = 0
        for b in range(bs):
            n = seqlen[b]  # num neurons in all views
            _x = x[ptr : ptr + n]  # all views concatenated
            _pidx = pidx[ptr : ptr + n]
            _uidx = uidx[ptr : ptr + n]
            ptr += n  # Advance even when this pair of views has no common units.

            sim = _x @ _x.T * self.tau_inv

            # -- NUMERATOR --

            match = _uidx[:, None] == _uidx[None, :]
            match.fill_diagonal_(False)
            num_matches[b] = match.sum()
            if num_matches[b] == 0:
                continue
            # loss_numers[b] = 1 - sim[match].mean() * self.tau
            loss_numers[b] = -sim[match].mean()
            assert loss_numers[b].isfinite().all()

            # -- DENOMENATOR --

            if self.dcl:
                # No positive-pairs in the denominator
                sim[match] = NEG_INF
            # No repetitions in the denominator
            sim.fill_diagonal_(NEG_INF)
            # No negatives across probes
            sim[_pidx[:, None] != _pidx[None, :]] = NEG_INF

            # Exclude empty denominators before logsumexp: evaluating an all
            # -inf row produces undefined gradients even if masked afterward.
            valid_rows = torch.isfinite(sim).any(1)
            if not valid_rows.any():
                loss_numers[b] = 0
                num_matches[b] = 0
                continue
            sim_lse = sim[valid_rows].logsumexp(1)
            # loss_denoms[b] = sim_lse[sim_lse.isfinite()].mean() * self.tau
            loss_denoms[b] = sim_lse[sim_lse.isfinite()].mean()

            if not (
                loss_numers[b].isfinite().all() and loss_denoms[b].isfinite().all()
            ):
                loss_numers[b] = 0
                loss_denoms[b] = 0
                num_matches[b] = 0

        # Compute loss weight as num_pos_pairs_in_view / total_pos_pairs_in_all_views
        total_matches = all_reduce(num_matches.sum())
        # all_reduce returns a rank mean, which may be between zero and one.
        safe_total = torch.where(total_matches > 0, total_matches, torch.ones_like(total_matches))
        loss_weights = num_matches / safe_total

        # Compute weighted-averaged loss
        loss_numer = all_reduce((loss_numers * loss_weights).sum())
        loss_denom = all_reduce((loss_denoms * loss_weights).sum())
        # Retain a zero-gradient path through the projector on ranks with no
        # usable positive pairs so distributed backward still participates.
        loss = loss_numer + loss_denom + x.sum() * 0.0

        log_dict = {
            f"{prefix}loss_numer": loss_numer.item(),
            f"{prefix}loss_denom": loss_denom.item(),
            f"{prefix}loss": loss.item(),
        }
        return loss, log_dict
