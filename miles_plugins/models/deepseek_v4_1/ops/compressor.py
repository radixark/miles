from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles_plugins.models.deepseek_v4.ops.norm import RMSNorm
from miles_plugins.models.deepseek_v4.ops.thd_utils import (
    CompressorInputCompact,
    ThdLayout,
    batch_of_row,
    compact_group_capacity,
    compressed_cu_seqlens,
    compressed_rank_layout,
    exchange_cp_boundary_hidden,
    get_q_positions_thd,
)


class DeepSeekV41Compressor(nn.Module):
    def __init__(self, config: TransformerConfig, head_dim: int, compress_ratio: int):
        super().__init__()
        assert compress_ratio in (1, 2)
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        dim = config.hidden_size
        self.linear_wkv = nn.Linear(dim, head_dim, bias=False, dtype=torch.bfloat16)
        if compress_ratio > 1:
            self.linear_wgate = nn.Linear(dim, head_dim, bias=False, dtype=torch.bfloat16)
        self.norm = RMSNorm(head_dim, config.layernorm_epsilon)
        mark_keep_in_fp32(self.norm.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen, bsz, _ = x.shape
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(F.linear(x, self.linear_wkv.weight))
        groups = seqlen // ratio
        x = x[: groups * ratio].float()
        kv = F.linear(x, self.linear_wkv.weight.float())
        score = F.linear(x, self.linear_wgate.weight.float())
        kv = kv.view(groups, ratio, bsz, self.head_dim)
        score = score.view(groups, ratio, bsz, self.head_dim)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        return self.norm(kv.to(torch.bfloat16))


@dataclass
class CompressorRows:
    """What the compressor reads under THD and where its output rows end up.

    ``hidden`` holds each group's tokens back to back, the layout the compressor groups by.
    ``positions`` is the in-segment position of each group's first token, for the rotary.
    ``seq_rows`` reorders the CP all-gather of the output into sequence-major rows, the layout
    every later layer that reuses the latent and index keys reads; None when the gather is
    already in that order.
    """

    hidden: Tensor
    positions: Tensor
    seq_rows: Tensor | None


def compressor_rows_thd(hidden: Tensor, thd: ThdLayout, *, ratio: int, cp_group) -> CompressorRows:
    """Group this rank's packed rows ``[s, b, d]`` within their segments.

    Ratio 1 groups nothing. Otherwise a segment's tail shorter than ``ratio`` gets no group, as
    in the BSHD path, and under CP a group that straddles the split is built by the rank holding
    its last token, from rows its left neighbour sends.
    """
    seqlen = hidden.size(0)
    if ratio == 1:
        return CompressorRows(hidden, get_q_positions_thd(thd.cu_seqlens, seqlen, thd.global_start), None)
    cp_size = cp_group.size() if cp_group is not None else 1
    n_groups = sum(n // ratio for n in thd.seq_lens)
    if cp_size == 1:
        cu_compressed = compressed_cu_seqlens(thd.cu_seqlens, ratio)
        seg = batch_of_row(cu_compressed, n_groups)
        group_ids = torch.arange(n_groups, device=hidden.device) - cu_compressed[seg]
        heads = thd.cu_seqlens[seg] + group_ids * ratio
        rows = (heads.unsqueeze(1) + torch.arange(ratio, device=hidden.device)).flatten()
        return CompressorRows(hidden.index_select(0, rows.long()), group_ids * ratio, None)
    boundary = exchange_cp_boundary_hidden(hidden, ratio=ratio, cp_group=cp_group)
    c_cap = compact_group_capacity(seqlen, ratio)
    compact, group_ids = CompressorInputCompact.apply(hidden, boundary, thd.cu_seqlens, thd.global_start, ratio, c_cap)
    seq_rows = compressed_rank_layout(
        thd.cu_seqlens,
        compressed_cu_seqlens(thd.cu_seqlens, ratio),
        l_local=seqlen,
        cp_size=cp_size,
        ratio=ratio,
        c_cap=c_cap,
    )
    # Rows past the last group are -1; dropping them leaves the same row count as without CP.
    return CompressorRows(compact, group_ids.clamp(min=0) * ratio, seq_rows[:n_groups].long())
