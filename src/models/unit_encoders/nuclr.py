from typing import Iterable
from pathlib import Path
from einops import rearrange, repeat
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from torch_brain.batching import chain

from src.utils import get_cli_logger
from src.utils.util import is_divisible, expand_path

from torch_brain.nn import RotaryTimeEmbedding
from src.nn.rotary_attention import RotarySelfAttention


class NuCLR(nn.Module):
    def __init__(
        self,
        ctx_duration: float,
        latent_step: float,
        bin_size: float,
        dim: int,
        self_heads: int,
        dim_head: int,
        lin_dropout: float,
        t_layers: int,
        st_layers: int,
        atn_dropout: float = 0.0,
        rot_ratio: float = 0.5,
        rotate_value: bool = True,
        attn_biases: bool = False,
        rate_separation: bool = False,
    ):
        super().__init__()

        assert is_divisible(latent_step, bin_size)
        assert is_divisible(ctx_duration, latent_step)
        self.ctx_duration = ctx_duration
        self.latent_step = latent_step
        self.bin_size = bin_size
        self.emb_dim = self.dim = dim
        if not isinstance(rate_separation, bool) or (rate_separation and dim < 3):
            raise ValueError("rate_separation must be boolean and requires dim >= 3")
        self.rate_separation = rate_separation

        if atn_dropout != 0.0:
            print(f"{atn_dropout=} but this impl only supports atn_dropout==0")

        # derived params
        self.bins_per_latent = int(latent_step / bin_size)  # num of bins per latent
        self.num_latents = int(np.ceil(ctx_duration / latent_step))

        t_min, t_max = 1.0, 8.0 * self.num_latents
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.read_in = nn.Linear(self.bins_per_latent, dim)

        self.t_blocks: list[list[nn.Module]] = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dim_head=dim_head,
                            rotate_value=rotate_value,
                            to_qkv_bias=attn_biases,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                    ]
                )
                for _ in range(t_layers)
            ]
        )  # ty:ignore[invalid-assignment]

        self.st_blocks: list[list[nn.Module]] = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dim_head=dim_head,
                            to_qkv_bias=attn_biases,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dim_head=dim_head,
                            rotate_value=rotate_value,
                            to_qkv_bias=attn_biases,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                    ]
                )
                for _ in range(st_layers)
            ]
        )  # ty:ignore[invalid-assignment]

        self.dp = nn.Dropout(lin_dropout)
        if rate_separation:
            self.rate_head = nn.Linear(dim, 1)
            self.content_head = nn.Linear(dim, dim - 1)

    def forward(self, bins: Tensor, spax_seqlen: Tensor) -> Tensor:
        r"""
        Args:
            bins: Chained tensor of binned spikes (*num_neurons, num_bins)
            spax_seqlen: seqlen to constrain spatial attention
        """
        device = bins.device
        num_units = int(spax_seqlen.sum())
        spax_seqlen = spax_seqlen.repeat(self.num_latents)

        x = rearrange(
            bins,
            "U (L B) -> U L B",
            L=self.num_latents,
            B=self.bins_per_latent,
        )

        x = self.read_in(x.float())
        t = torch.arange(self.num_latents, device=device).repeat(num_units, 1).float()
        x_rot = self.rotary_emb(t)

        # Temporal attention layers
        for t_attn, t_ffn in self.t_blocks:
            x = x + self.dp(t_attn(x, rotary=x_rot))
            x = x + self.dp(t_ffn(x))

        if self.rate_separation:
            rate_hz = torch.nn.functional.softplus(self.rate_head(x.mean(1)))

        # Spatio-temporal attention layers
        for s_attn, s_ffn, t_attn, t_ffn in self.st_blocks:
            # Spatial
            x = rearrange(x, "U L D -> (L U) D", L=self.num_latents, U=num_units)
            x = x + self.dp(s_attn(x, seqlen=spax_seqlen))
            x = x + self.dp(s_ffn(x))
            x = rearrange(x, "(L U) D -> U L D", U=num_units, L=self.num_latents)

            # Temporal
            x = x + self.dp(t_attn(x, rotary=x_rot))
            x = x + self.dp(t_ffn(x))

        y = x.mean(1)
        if self.rate_separation:
            return torch.cat((rate_hz, self.content_head(y)), dim=-1)
        return y

    def tokenize(self, data) -> dict | None:
        if len(data.units) < 2:
            return None  # cannot work with a single unit

        spike_times = torch.tensor(data.spikes.timestamps, dtype=torch.float32)
        spike_units = torch.tensor(data.spikes.unit_index, dtype=torch.long)
        num_units = len(data.units.id)

        # bin spikes
        num_bins = int(self.ctx_duration / self.bin_size)
        rate = 1 / self.bin_size
        bins = torch.zeros((num_units, num_bins + 1), dtype=torch.int16)
        bins.index_put_(
            indices=(spike_units, torch.floor(spike_times * rate).long()),
            values=torch.ones(len(spike_times), dtype=torch.int16),
            accumulate=True,
        )
        bins = bins[:, : self.num_latents * self.bins_per_latent]

        pids = data.units.probe_id.astype(str)
        probe_boundary_idx = np.where(pids[1:] != pids[:-1])[0]
        # Attention segments must follow unit order, not np.unique's sorted IDs.
        probe_counts = np.diff(np.r_[0, probe_boundary_idx + 1, num_units])
        assert len(probe_counts) == len(np.unique(pids)), "Probe units must be contiguous"

        return {
            "bins": chain(bins),
            "spax_seqlen": chain(probe_counts),
        }

    @classmethod
    def from_pretrained(cls, ckpt_path: str | Path):
        from omegaconf import OmegaConf

        ckpt = torch.load(
            expand_path(ckpt_path),
            weights_only=False,
            map_location="cpu",
        )
        hparams: dict = OmegaConf.to_container(ckpt["cfg"]["model"])  # ty:ignore
        hparams.pop("_target_", None)

        model = cls(**hparams)
        model.load_state_dict(ckpt["model_state_dict"])
        get_cli_logger().info(f"Loaded NuCLR from {ckpt_path}")
        return model


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float, pre_norm: bool):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.in_proj = nn.Linear(dim, 2 * dim * mult)
        self.out_proj = nn.Linear(dim * mult, dim)
        self.dp = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.norm(x)
        y, gate = self.in_proj(y).chunk(2, dim=-1)
        y = self.act(gate) * y
        y = self.dp(y)
        return self.out_proj(y)
