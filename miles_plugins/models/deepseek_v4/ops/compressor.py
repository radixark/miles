import os
from types import SimpleNamespace

import einops
import torch
import torch.nn as nn
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import Linear

from .cp_utils import all_gather_cp, get_freqs_cis_for_cp
from .qat import fp8_simulate_qat
from .ref_model import Compressor as CompressorRef
from .ref_model import RMSNorm, apply_rotary_emb, rotate_activation
from .utils import wrapped_precompute_freqs_cis


class DeepSeekV4Compressor(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        head_dim: int,
        compress_ratio: int,
        rotate: bool,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ):
        super().__init__()

        dim = config.hidden_size
        rope_head_dim = config.qk_pos_emb_head_dim
        norm_eps = config.layernorm_epsilon

        assert dim == 4096
        assert head_dim in {128, 512}
        assert rope_head_dim == 64
        assert compress_ratio in {4, 128}
        assert norm_eps == 1e-6

        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1
        self.cp_rank = cp_group.rank() if cp_group is not None else 0

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, norm_eps)

        for p in [self.ape, self.wkv.weight, self.wgate.weight]:
            p._keep_fp32 = True

        base = config.dsv4_compress_rope_theta
        assert rope_head_dim == 64
        assert base == 160000
        freqs_cis = wrapped_precompute_freqs_cis(config, rope_head_dim=rope_head_dim, base=base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def overlap_transform_raw(self, tensor: torch.Tensor, value=0):
        """Raw overlap transform without CP handling."""
        return CompressorRef.overlap_transform(
            self=SimpleNamespace(compress_ratio=self.compress_ratio, head_dim=self.head_dim),
            tensor=tensor,
            value=value,
        )

    def overlap_transform_with_cp(self, tensor: torch.Tensor, value=0) -> torch.Tensor:
        """
        Overlap transform with CP support.

        Args:
            tensor: [bsz, G_local, ratio, coff*d]
            value: Fill value for overlap transform (0 for kv, -inf for score)

        Returns:
            [bsz, G_local, ratio, coff*d]
        """
        if self.cp_size == 1:
            return self.overlap_transform_raw(tensor, value)

        tensor = all_gather_cp(tensor, dim=1, cp_group=self.cp_group)

        tensor = self.overlap_transform_raw(tensor, value)

        G_local = tensor.shape[1] // self.cp_size
        start = self.cp_rank * G_local
        return tensor[:, start : start + G_local, :, :]

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        assert self.ape.dtype == torch.float32
        assert self.wkv.weight.dtype == torch.float32
        assert self.wgate.weight.dtype == torch.float32

        bsz, seqlen_local, _ = x.size()
        ratio, overlap, _ = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype

        assert (seqlen_local >= ratio) and (seqlen_local % ratio == 0), f"{seqlen_local=} {ratio=}"
        if self.cp_size > 1:
            assert seqlen_local % (ratio * 2) == 0

        x_fp32 = x.float()
        kv = self.wkv(x_fp32)
        score = self.wgate(x_fp32)

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            kv = self.overlap_transform_with_cp(kv, 0)
            score = self.overlap_transform_with_cp(score, float("-inf"))

        score_softmax = score.softmax(dim=2)
        kv = (kv * score_softmax).sum(dim=2)

        kv = self.norm(kv.to(dtype))

        freqs_cis = get_freqs_cis_for_cp(self.freqs_cis, seqlen_local, self.cp_size, self.cp_group, stride=ratio)

        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)
            if os.environ.get("MEGATRON_USE_KV_QAT", "0") == "1":
                kv = fp8_simulate_qat(kv, 128)
        else:
            if os.environ.get("MEGATRON_USE_KV_QAT", "0") == "1":
                kv = kv.clone()
                kv[..., : self.nope_head_dim] = fp8_simulate_qat(kv[..., : self.nope_head_dim], 64)
            else:
                pass

        return kv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seqlen, batch, dim] SBHD layout (Megatron standard)
        Returns:
            k: [seqlen // compress_ratio, batch, head_dim] SBHD layout
        """
        x_bshd = einops.rearrange(x, "s b d -> b s d")
        k_bshd = self.forward_raw(x_bshd)
        k = einops.rearrange(k_bshd, "b sc d -> sc b d")
        return k
