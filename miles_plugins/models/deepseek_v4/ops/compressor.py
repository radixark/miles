import einops
import torch
import torch.nn as nn
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import Linear

from miles_plugins.models.deepseek_v4.ops.cp_utils import all_gather_cp, get_freqs_cis_for_cp
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


def _compress_c4_reference(
    x: torch.Tensor,
    wkv_weight: torch.Tensor,
    wgate_weight: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    head_dim: int,
    rope_head_dim: int,
    norm_eps: float,
) -> torch.Tensor:
    """Differentiable C4 compressor used as the exact kernel's backward."""
    ratio = 4
    dtype = x.dtype

    kv = linear_bf16_fp32(x, wkv_weight).to(torch.bfloat16).float()
    score = linear_bf16_fp32(x, wgate_weight).to(torch.bfloat16).float()
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + ape
    kv = _overlap_transform(kv, compress_ratio=ratio, head_dim=head_dim, value=0)
    score = _overlap_transform(
        score,
        compress_ratio=ratio,
        head_dim=head_dim,
        value=float("-inf"),
    )
    kv = (kv * score.softmax(dim=2)).sum(dim=2)

    kv_fp32 = kv.float()
    variance = kv_fp32.square().mean(-1, keepdim=True)
    kv = (norm_weight * kv_fp32 * torch.rsqrt(variance + norm_eps)).to(dtype)
    apply_rotary_emb(kv[..., -rope_head_dim:], freqs_cis[: x.shape[1] : ratio])
    return kv


def _compress_c4_sglang(
    x: torch.Tensor,
    wkv_weight: torch.Tensor,
    wgate_weight: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    head_dim: int,
    norm_eps: float,
) -> torch.Tensor:
    """Run SGLang's exact C4 prefill compressor for a single sequence."""
    from sglang.jit_kernel.dsv4 import (
        CompressorPrefillPlan,
        compress_forward,
        compress_norm_rope_store,
    )

    kv = linear_bf16_fp32(x, wkv_weight).to(torch.bfloat16).float()
    score = linear_bf16_fp32(x, wgate_weight).to(torch.bfloat16).float()
    kernel_input = torch.cat((kv, score), dim=-1).squeeze(0).contiguous()
    kernel_ape = ape.view(4, -1, head_dim).transpose(0, 1).reshape(-1, head_dim).contiguous()

    plan = CompressorPrefillPlan.generate_legacy(
        compress_ratio=4,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=x.device),
        seq_lens=torch.tensor([kernel_input.shape[0]], dtype=torch.int64),
        extend_lens=torch.tensor([kernel_input.shape[0]], dtype=torch.int64),
        num_q_tokens=kernel_input.shape[0],
        device=x.device,
    )
    plan_c = plan.plan_c.clone()
    plan_raw = plan_c.view(torch.int32)
    plan_raw[:, 1] &= 0xFFFF
    plan_raw[:, 2:4] = 0
    plan = CompressorPrefillPlan(
        4,
        plan_c,
        torch.empty((0, 8), dtype=torch.uint8, device=x.device),
        None,
    )

    buffer = torch.empty(
        (1, 4, kernel_input.shape[-1]),
        dtype=kernel_input.dtype,
        device=x.device,
    )
    compressed = compress_forward(
        buffer,
        kernel_input,
        kernel_ape,
        plan,
        head_dim=head_dim,
        compress_ratio=4,
    )
    output = torch.empty(
        (compressed.shape[0], head_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )
    out_loc = (
        torch.arange(
            kernel_input.shape[0],
            dtype=torch.int64,
            device=x.device,
        )
        // 4
    )
    compress_norm_rope_store(
        compressed,
        plan,
        norm_weight=norm_weight.to(compressed.dtype).contiguous(),
        norm_eps=norm_eps,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=output.view(torch.uint8),
        page_size=1,
        bf16_store=True,
    )
    return output.unsqueeze(0)


class _DeepSeekV4C4CompressorFunction(torch.autograd.Function):
    """Exact SGLang forward with a differentiable PyTorch backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        wkv_weight: torch.Tensor,
        wgate_weight: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        freqs_cis: torch.Tensor,
        head_dim: int,
        rope_head_dim: int,
        norm_eps: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            x,
            wkv_weight,
            wgate_weight,
            ape,
            norm_weight,
            freqs_cis,
        )
        ctx.head_dim = head_dim
        ctx.rope_head_dim = rope_head_dim
        ctx.norm_eps = norm_eps
        return _compress_c4_sglang(
            x,
            wkv_weight,
            wgate_weight,
            ape,
            norm_weight,
            freqs_cis,
            head_dim=head_dim,
            norm_eps=norm_eps,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        inputs = [tensor.detach().requires_grad_(True) for tensor in saved[:5]]
        freqs_cis = saved[5]
        with torch.enable_grad():
            output = _compress_c4_reference(
                *inputs,
                freqs_cis,
                head_dim=ctx.head_dim,
                rope_head_dim=ctx.rope_head_dim,
                norm_eps=ctx.norm_eps,
            )
        grads = torch.autograd.grad(
            output,
            inputs,
            grad_output,
            allow_unused=True,
        )
        grads = tuple(grad if needed else None for grad, needed in zip(grads, ctx.needs_input_grad[:5]))
        return (*grads, None, None, None, None)


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

        if getattr(self.config, "batch_invariant_mode", False) and ratio == 4 and self.head_dim == 512:
            if self.cp_size != 1:
                raise RuntimeError("DSV4 TOP exact C4 compressor currently requires context parallel size 1.")
            if bsz != 1:
                raise RuntimeError("DSV4 TOP exact C4 compressor currently requires micro-batch size 1.")
            freqs_cis = wrapped_precompute_freqs_cis(
                self.config,
                self.rope_head_dim,
                self.config.dsv4_compress_rope_theta,
                False,
                seqlen_local,
                x.device,
            )
            kv = _DeepSeekV4C4CompressorFunction.apply(
                x,
                self.wkv.weight,
                self.wgate.weight,
                self.ape,
                self.norm.weight,
                freqs_cis,
                self.head_dim,
                self.rope_head_dim,
                self.norm.eps,
            )
            if self.rotate:
                kv = rotate_activation(kv)
                if self.use_fp8_qat:
                    kv = fp8_simulate_qat(kv, 128)
            elif self.use_fp8_qat:
                kv = kv.clone()
                kv[..., : self.nope_head_dim] = fp8_simulate_qat(
                    kv[..., : self.nope_head_dim],
                    64,
                )
            return kv

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
