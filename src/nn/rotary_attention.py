from typing import Optional
from functools import partial

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.attention.varlen import varlen_attn

from torch_brain.nn.position_embeddings import RotaryTimeEmbedding

_MAX_FA_BS = 2**16 - 1  # FA seems to fail with batch sizes bigger than this


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ctx_dim: int | None = None,
        dim_head: Optional[int] = None,
        rotate_value: bool = False,
        to_kv_bias: bool = False,
        to_q_bias: bool = False,
        to_out_bias: bool = True,
        pre_norm_q: bool = True,
        pre_norm_kv: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.rotate_value = rotate_value
        self.dim_head = dim_head = dim_head or (dim // heads)

        # build networks
        ctx_dim = ctx_dim or dim
        self.norm = nn.LayerNorm(dim) if pre_norm_q else nn.Identity()
        self.norm_context = nn.LayerNorm(ctx_dim) if pre_norm_kv else nn.Identity()

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=to_q_bias)
        self.to_kv = nn.Linear(ctx_dim, inner_dim * 2, bias=to_kv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x_q: torch.Tensor,
        x_ctx: torch.Tensor,
        rotary_q: Optional[torch.Tensor] = None,
        rotary_ctx: Optional[torch.Tensor] = None,
        seqlen_q: torch.Tensor | None = None,
        seqlen_ctx: torch.Tensor | None = None,
    ):

        q = self.to_q(self.norm(x_q))
        k, v = self.to_kv(self.norm_context(x_ctx)).chunk(2, dim=-1)

        if seqlen_q is not None:
            assert seqlen_ctx is not None
            n_q, h, d = q.size(0), self.heads, self.dim_head
            n_kv = k.size(0)
            q = q.view(n_q, h, d)
            k = k.view(n_kv, h, d)
            v = v.view(n_kv, h, d)
            attn_func = partial(_attn_varlen, seqlen_q=seqlen_q, seqlen_kv=seqlen_ctx)
        else:
            raise NotImplementedError

        if rotary_q is not None:
            assert rotary_ctx is not None
            q = RotaryTimeEmbedding.rotate(q, rotary_q, 1)
            k = RotaryTimeEmbedding.rotate(k, rotary_ctx, 1)
            if self.rotate_value:
                v = RotaryTimeEmbedding.rotate(v, rotary_ctx, 1)

        out = attn_func(q, k, v)

        if rotary_q is not None and self.rotate_value:
            rotary_q_inv = RotaryTimeEmbedding.invert(rotary_q)
            out = RotaryTimeEmbedding.rotate(out, rotary_q_inv, 1)

        out = out.view(n_q, h * d)
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: Optional[int] = None,
        rotate_value: bool = False,
        to_qkv_bias: bool = True,
        to_out_bias: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.heads = heads
        # self.atn_dropout = atn_dropout
        self.rotate_value = rotate_value
        self.dim_head = dim_head = dim_head or (dim // heads)

        # build networks
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=to_qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x: torch.Tensor,
        rotary: Optional[torch.Tensor] = None,
        seqlen: torch.Tensor | None = None,
    ):

        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)

        if seqlen is not None:
            assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2
            n, h, d = q.size(0), self.heads, self.dim_head
            q = q.view(n, h, d)
            k = k.view(n, h, d)
            v = v.view(n, h, d)
            attn_func = partial(_attn_varlen, seqlen_q=seqlen)
        else:
            assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
            b, n, h, d = q.size(0), q.size(1), self.heads, self.dim_head
            q = q.view(b, n, h, d).permute(0, 2, 1, 3)
            k = k.view(b, n, h, d).permute(0, 2, 1, 3)
            v = v.view(b, n, h, d).permute(0, 2, 1, 3)
            attn_func = _attn_fixedlen

        if rotary is not None:
            q = RotaryTimeEmbedding.rotate(q, rotary, 1)
            k = RotaryTimeEmbedding.rotate(k, rotary, 1)
            if self.rotate_value:
                v = RotaryTimeEmbedding.rotate(v, rotary, 1)

        out = attn_func(q, k, v)

        if rotary is not None and self.rotate_value:
            rotary_inv = RotaryTimeEmbedding.invert(rotary)
            out = RotaryTimeEmbedding.rotate(out, rotary_inv, 1)

        if seqlen is not None:
            # out is n,h,d
            out = out.view(n, h * d)
        else:
            # out is b,h,n,d
            out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)

        out = self.to_out(out)
        return out


def _attn_fixedlen(q: Tensor, k: Tensor, v: Tensor):
    bs = len(q)
    if bs <= _MAX_FA_BS:
        return F.scaled_dot_product_attention(q, k, v)

    # BS is too big for FA. Need to split it and combine at the end
    q_split = q.split(_MAX_FA_BS)
    k_split = k.split(_MAX_FA_BS)
    v_split = v.split(_MAX_FA_BS)

    out = torch.empty_like(q)
    ptr = 0
    for _q, _k, _v in zip(q_split, k_split, v_split):
        _bs = len(_q)
        out[ptr : ptr + _bs] = _attn_fixedlen(_q, _k, _v)
        ptr += _bs

    return out


def _attn_varlen(
    q: Tensor, k: Tensor, v: Tensor, seqlen_q: Tensor, seqlen_kv: Tensor | None = None
):
    if q.device.type == "cpu":
        # The same independent attention segments as the CUDA kernel, evaluated
        # one at a time for CPU extraction and small validation runs.
        q_lengths = seqlen_q.tolist()
        kv_lengths = (seqlen_q if seqlen_kv is None else seqlen_kv).tolist()
        if (len(q_lengths) != len(kv_lengths) or any(n <= 0 for n in q_lengths + kv_lengths)
                or sum(q_lengths) != len(q) or sum(kv_lengths) != len(k)):
            raise ValueError("Invalid attention segment lengths")
        outputs = []
        for qs, ks, vs in zip(q.split(q_lengths), k.split(kv_lengths), v.split(kv_lengths)):
            result = F.scaled_dot_product_attention(
                qs.transpose(0, 1).unsqueeze(0), ks.transpose(0, 1).unsqueeze(0),
                vs.transpose(0, 1).unsqueeze(0),
            )
            outputs.append(result.squeeze(0).transpose(0, 1))
        return torch.cat(outputs).contiguous()
    cuseq_q, seqmax_q = _seqlen_to_cuseq(seqlen_q)
    cuseq_kv, seqmax_kv = cuseq_q, seqmax_q
    if seqlen_kv is not None:
        cuseq_kv, seqmax_kv = _seqlen_to_cuseq(seqlen_kv)

    return varlen_attn(q, k, v, cuseq_q, cuseq_kv, seqmax_q, seqmax_kv)


def _seqlen_to_cuseq(seqlen: Tensor) -> tuple[Tensor, int]:
    batch_size = len(seqlen)
    cu_seq = torch.zeros(batch_size + 1, device=seqlen.device, dtype=torch.int32)
    cu_seq[1:] = seqlen.cumsum(0)
    max_len: int = seqlen.max().item()  # ty:ignore[invalid-assignment]
    return cu_seq, max_len
