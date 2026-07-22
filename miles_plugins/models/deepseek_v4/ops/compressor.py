import einops
import torch
import torch.nn as nn
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import Linear

from miles_plugins.models.deepseek_v4.ops.cp_utils import (
    all_gather_cp,
    get_compress_cu_seqlens_for_packed,
    get_freqs_cis_for_cp,
    get_q_positions_for_packed_cp,
    get_seq_ids_and_offsets_from_cu_seqlens,
    is_packed_thd_contiguous_cp,
)
from miles_plugins.models.deepseek_v4.ops.kernel.precision_aligned_ops import linear_bf16_fp32
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb, wrapped_precompute_freqs_cis
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

    def overlap_transform_packed(self, tensor: torch.Tensor, cu_seqlens: torch.Tensor, value=0) -> torch.Tensor:
        """Apply overlap transform independently inside each packed sample."""
        if self.cp_size > 1:
            tensor = all_gather_cp(tensor, dim=1, cp_group=self.cp_group)

        comp_cu_seqlens = get_compress_cu_seqlens_for_packed(cu_seqlens, ratio=self.compress_ratio)
        expected_groups = int(comp_cu_seqlens[-1].item())
        if tensor.size(1) != expected_groups:
            raise ValueError(
                "Packed DeepSeek-V4 compressor group count mismatch: "
                f"tensor={tensor.size(1)}, boundaries={expected_groups}"
            )
        group_boundaries = comp_cu_seqlens.tolist()
        pieces = [
            self.overlap_transform_raw(tensor[:, start:end], value)
            for start, end in zip(group_boundaries[:-1], group_boundaries[1:], strict=True)
        ]
        tensor = torch.cat(pieces, dim=1) if pieces else tensor[:, :0]

        if self.cp_size == 1:
            return tensor
        if tensor.shape[1] % self.cp_size != 0:
            raise ValueError(
                f"Packed compressed length {tensor.shape[1]} must be divisible by CP size {self.cp_size}"
            )
        G_local = tensor.shape[1] // self.cp_size
        start = self.cp_rank * G_local
        return tensor[:, start : start + G_local, :, :]

    def forward_raw(self, x: torch.Tensor, packed_seq_params=None) -> torch.Tensor:
        assert self.ape.dtype == torch.float32
        assert self.wkv.weight.dtype == torch.bfloat16
        assert self.wgate.weight.dtype == torch.bfloat16

        bsz, seqlen_local, _ = x.size()
        ratio, overlap, _ = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype

        assert (seqlen_local >= ratio) and (seqlen_local % ratio == 0), f"{seqlen_local=} {ratio=}"
        if self.cp_size > 1:
            assert seqlen_local % (ratio * 2) == 0

        packed_seq = is_packed_thd_contiguous_cp(packed_seq_params, self.cp_size)
        cu_seqlens = None
        if packed_seq:
            cu_seqlens = packed_seq_params.cu_seqlens_q.to(device=x.device, dtype=torch.long)
            get_compress_cu_seqlens_for_packed(cu_seqlens, ratio=ratio)
            expected_global = seqlen_local * self.cp_size
            if int(cu_seqlens[-1].item()) != expected_global:
                raise ValueError(
                    "Packed DeepSeek-V4 compressor requires contiguous allgather CP: "
                    f"cu_seqlens[-1]={int(cu_seqlens[-1].item())}, local={seqlen_local}, cp={self.cp_size}"
                )

        kv = linear_bf16_fp32(x, self.wkv.weight)
        score = linear_bf16_fp32(x, self.wgate.weight)

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            if packed_seq:
                kv = self.overlap_transform_packed(kv, cu_seqlens, 0)
                score = self.overlap_transform_packed(score, cu_seqlens, float("-inf"))
            else:
                kv = self.overlap_transform_with_cp(kv, 0)
                score = self.overlap_transform_with_cp(score, float("-inf"))

        score_softmax = score.softmax(dim=2)
        kv = (kv * score_softmax).sum(dim=2)

        kv = self.norm(kv.to(dtype))

        if packed_seq:
            max_seq_len = int(packed_seq_params.max_seqlen_q)
            q_positions = get_q_positions_for_packed_cp(seqlen_local, self.cp_size, self.cp_group, x.device)
            _, offsets, _, _ = get_seq_ids_and_offsets_from_cu_seqlens(cu_seqlens, q_positions)
            if int(q_positions[0].item()) % ratio != 0:
                raise ValueError(
                    f"Packed CP chunk must start on a compression-group boundary: {q_positions[0].item()=}, {ratio=}"
                )
            group_offsets = offsets[::ratio] // ratio
            if group_offsets.numel() != kv.size(1):
                raise ValueError(
                    "Packed DeepSeek-V4 compressor RoPE/group mismatch: "
                    f"offsets={group_offsets.numel()}, compressed_kv={kv.size(1)}"
                )
            max_comp_seq_len = (max_seq_len + ratio - 1) // ratio
            freqs_cis = wrapped_precompute_freqs_cis(
                self.config,
                self.rope_head_dim,
                self.config.dsv4_compress_rope_theta,
                False,
                max_comp_seq_len * ratio,
                x.device,
            )
            freqs_cis = freqs_cis[::ratio][group_offsets]
        else:
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

    def forward(self, x: torch.Tensor, packed_seq_params=None) -> torch.Tensor:
        """
        Args:
            x: [seqlen, batch, dim] SBHD layout (Megatron standard)
        Returns:
            k: [seqlen // compress_ratio, batch, head_dim] SBHD layout
        """
        x_bshd = einops.rearrange(x, "s b d -> b s d")
        k_bshd = self.forward_raw(x_bshd, packed_seq_params=packed_seq_params)
        k = einops.rearrange(k_bshd, "b sc d -> sc b d")
        return k
