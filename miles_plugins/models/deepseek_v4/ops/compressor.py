import einops
import torch
import torch.nn as nn
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import Linear

from miles_plugins.models.deepseek_v4.ops.cp_utils import all_gather_cp, get_freqs_cis_for_cp
from miles_plugins.models.deepseek_v4.ops.kernel.precision_aligned_ops import linear_bf16_fp32
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v4.ops.rope import (
    apply_rotary_emb,
    apply_rotary_emb_thd,
    wrapped_precompute_freqs_cis,
)
from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout, batch_of_row, compressed_cu_seqlens
from miles_plugins.models.deepseek_v4.ops.utils import rotate_activation


class RMSNorm(nn.Module):
    """
    Kept in pure PyTorch with FP32 weights to match SGLang's compressor norm.

    Args:
        dim: Dimension of the input tensor.
        eps: Epsilon for numerical stability. Defaults to ``1e-6``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def _overlap_transform(tensor: torch.Tensor, *, compress_ratio: int, head_dim: int, value=0) -> torch.Tensor:
    """Overlap-transform for compress_ratio=4: for each token group of size ``ratio``,
    split into (first_half, second_half) halves along ``head_dim`` and re-arrange
    them across a doubled ratio axis (`2 * ratio`), shifting the first half by one
    group so that adjacent groups overlap by ``ratio`` positions.
    """
    b, s, _, _ = tensor.size()
    new_tensor = tensor.new_full((b, s, 2 * compress_ratio, head_dim), value)
    new_tensor[:, :, compress_ratio:] = tensor[:, :, :, head_dim:]
    new_tensor[:, 1:, :compress_ratio] = tensor[:, :-1, :, :head_dim]
    return new_tensor


def _overlap_transform_thd(
    tensor: torch.Tensor, *, compress_ratio: int, head_dim: int, is_first: torch.Tensor, value=0
) -> torch.Tensor:
    """Overlap-transform for packed THD groups.

    Same rearrangement as :func:`_overlap_transform`, but groups are flat along dim 0
    and a segment's first group has no predecessor to pull from.

    Args:
        tensor: ``[total_comp, ratio, batch, 2 * head_dim]``
        is_first: ``[total_comp]`` bool, True for the first group of each segment.
    """
    total_comp, ratio, bsz, _ = tensor.size()
    new_tensor = tensor.new_full((total_comp, 2 * ratio, bsz, head_dim), value)
    new_tensor[:, ratio:] = tensor[:, :, :, head_dim:]
    prev = torch.roll(tensor[:, :, :, :head_dim], shifts=1, dims=0)
    prev[is_first] = value
    new_tensor[:, :ratio] = prev
    return new_tensor


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
        self.use_fp8_qat = config.fp8 is not None

        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1
        self.cp_rank = cp_group.rank() if cp_group is not None else 0

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.wgate = Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.norm = RMSNorm(self.head_dim, norm_eps)

        self.ape._keep_fp32 = True

        base = config.dsv4_compress_rope_theta
        assert rope_head_dim == 64
        assert base == 160000

    def overlap_transform_raw(self, tensor: torch.Tensor, value=0):
        """Raw overlap transform without CP handling."""
        return _overlap_transform(tensor, compress_ratio=self.compress_ratio, head_dim=self.head_dim, value=value)

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
        assert self.wkv.weight.dtype == torch.bfloat16
        assert self.wgate.weight.dtype == torch.bfloat16

        bsz, seqlen_local, _ = x.size()
        ratio, overlap, _ = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype

        assert (seqlen_local >= ratio) and (seqlen_local % ratio == 0), f"{seqlen_local=} {ratio=}"
        if self.cp_size > 1:
            assert seqlen_local % (ratio * 2) == 0

        kv = linear_bf16_fp32(x, self.wkv.weight)
        score = linear_bf16_fp32(x, self.wgate.weight)

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            kv = self.overlap_transform_with_cp(kv, 0)
            score = self.overlap_transform_with_cp(score, float("-inf"))

        score_softmax = score.softmax(dim=2)
        kv = (kv * score_softmax).sum(dim=2)

        kv = self.norm(kv.to(dtype))

        freqs_cis = wrapped_precompute_freqs_cis(
            self.config,
            self.rope_head_dim,
            self.config.dsv4_compress_rope_theta,
            False,
            seqlen_local * self.cp_size,
            x.device,
        )
        freqs_cis = get_freqs_cis_for_cp(freqs_cis, seqlen_local, self.cp_size, self.cp_group, stride=ratio)

        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)
            if self.use_fp8_qat:
                kv = fp8_simulate_qat(kv, 128)
        else:
            if self.use_fp8_qat:
                kv = kv.clone()
                kv[..., : self.nope_head_dim] = fp8_simulate_qat(kv[..., : self.nope_head_dim], 64)

        return kv

    def _forward_thd(self, x: torch.Tensor, thd_layout: ThdLayout):
        """Compress a THD-packed stream, grouping tokens within each segment.

        The trailing ``seqlen % compress_ratio`` tokens of a segment get no compressed
        entry and rely on the sliding window instead. This matches inference, where a
        decode token sitting in an incomplete buffer has no compressed entry either, and
        avoids the train/inference mismatch that padding to compress_ratio would introduce.

        Args:
            x: [total, batch, dim] packed SBHD layout, or [c_cap * ratio, batch, dim] already
                grouped when the layout carries compressed_group_ids.
            thd_layout: packed-stream layout.
        Returns:
            (k, cu_seqlens_compressed) with k [total_comp, batch, head_dim], or
            (None, cu_seqlens_compressed) when no segment reaches compress_ratio.
        """
        cu_seqlens = thd_layout.cu_seqlens
        compressed_group_ids = thd_layout.compressed_group_ids
        max_seqlen = thd_layout.max_seqlen

        assert self.ape.dtype == torch.float32
        assert self.wkv.weight.dtype == torch.bfloat16
        assert self.wgate.weight.dtype == torch.bfloat16

        total_tokens = x.size(0)
        ratio, overlap = self.compress_ratio, self.overlap
        dtype = x.dtype

        pre_grouped = compressed_group_ids is not None
        if pre_grouped:
            if max_seqlen is None:
                raise ValueError(
                    "Pre-grouped compressor input needs max_seqlen: its group ids address "
                    "positions past the compacted row count."
                )
            cu_seqlens_compressed = None
            local_pos = compressed_group_ids
            total_comp = local_pos.size(0)
        else:
            cu_seqlens_compressed = compressed_cu_seqlens(cu_seqlens, ratio)
            total_comp = int(cu_seqlens_compressed[-1])
            if total_comp == 0:
                return None, cu_seqlens_compressed

        kv = linear_bf16_fp32(x, self.wkv.weight)
        score = linear_bf16_fp32(x, self.wgate.weight)

        if pre_grouped:
            # Compaction already laid rows out as [g * ratio, (g + 1) * ratio); no gather.
            kv = kv.unflatten(0, (total_comp, ratio))
            score = score.unflatten(0, (total_comp, ratio)) + self.ape.view(1, ratio, 1, -1)
        else:
            batch_ids = batch_of_row(cu_seqlens_compressed, total_comp)
            local_pos = torch.arange(total_comp, device=x.device) - cu_seqlens_compressed[batch_ids]
            gather_idx = (cu_seqlens[batch_ids] + local_pos * ratio).unsqueeze(1) + torch.arange(
                ratio, device=x.device
            )
            kv = kv[gather_idx]
            score = score[gather_idx] + self.ape.view(1, ratio, 1, -1)

        if overlap:
            is_first = local_pos == 0
            kv = _overlap_transform_thd(kv, compress_ratio=ratio, head_dim=self.head_dim, is_first=is_first, value=0)
            score = _overlap_transform_thd(
                score,
                compress_ratio=ratio,
                head_dim=self.head_dim,
                is_first=is_first,
                value=float("-inf"),
            )

        score_softmax = score.softmax(dim=1)
        kv = (kv * score_softmax).sum(dim=1)

        kv = self.norm(kv.to(dtype))

        freqs_cis = wrapped_precompute_freqs_cis(
            self.config,
            self.rope_head_dim,
            self.config.dsv4_compress_rope_theta,
            False,
            max_seqlen if pre_grouped else total_tokens,
            x.device,
        )
        # Capacity padding carries -1; clamp keeps the gather in range and those rows are
        # dropped by seq_to_rank_row anyway.
        rope_positions = local_pos.clamp(min=0) * ratio if pre_grouped else local_pos * ratio
        apply_rotary_emb_thd(kv[..., -self.rope_head_dim :], freqs_cis.index_select(0, rope_positions))

        if self.rotate:
            kv = rotate_activation(kv)
            if self.use_fp8_qat:
                kv = fp8_simulate_qat(kv, 128)
        else:
            if self.use_fp8_qat:
                kv = kv.clone()
                kv[..., : self.nope_head_dim] = fp8_simulate_qat(kv[..., : self.nope_head_dim], 64)

        return kv, cu_seqlens_compressed

    def forward(self, x: torch.Tensor, thd_layout: ThdLayout | None = None):
        """
        Args:
            x: [seqlen, batch, dim] SBHD layout (Megatron standard); [total, batch, dim]
                when thd_layout is given.
            thd_layout: packed-stream layout, or None when unpacked.
        Returns:
            k: [seqlen // compress_ratio, batch, head_dim] SBHD layout, or
                (k, cu_seqlens_compressed) for THD packing.
        """
        if thd_layout is not None:
            return self._forward_thd(x, thd_layout)
        x_bshd = einops.rearrange(x, "s b d -> b s d")
        k_bshd = self.forward_raw(x_bshd)
        k = einops.rearrange(k_bshd, "b sc d -> sc b d")
        return k
