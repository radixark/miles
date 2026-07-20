"""GDN packed-document correctness patch for qwen3_5.

torchtitan's qwen3_5 GatedDeltaNet (the 24-of-32 GDN layers) has no notion of document
boundaries within a packed multi-document training batch: ``GatedDeltaNet.forward``
takes only ``x`` (no positions), its causal conv1d runs over the whole packed row with
no boundary awareness, and its fla kernel call never passes ``cu_seqlens`` even though
``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` already accepts it natively. Without
this patch, every GDN layer would leak recurrent state and depthwise-conv context across
document boundaries in a packed RL training batch — the 8 full-attention layers are
unaffected (they already get proper per-document flex-attention masks from
``model.get_attention_masks(positions)``).

This monkeypatches 4 methods on torchtitan's actual classes (not vendored/copied) rather
than editing the pip-installed package, matching this codebase's existing precedent
(``compat.py``'s shims, ``batch_invariance.py``'s copy-not-import elsewhere in the design).
Call ``apply()`` once at import time; ``models.py`` does this automatically for qwen3_5.

Assumes a single packed row per micro-batch (``positions.shape[0] == 1``) — miles' RL
training convention (one packed sequence per data-parallel rank's micro-batch), matching
``fla``'s own varlen convention where ``cu_seqlens``-using calls expect ``B=1``.
"""

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.qwen3_5.model import (
    GatedDeltaKernel,
    GatedDeltaNet,
    Qwen35TransformerBlock,
    _fla_chunk_gated_delta_rule,
    _fla_fused_recurrent_gated_delta_rule,
    _torch_native_gated_delta,
)

_APPLIED = False


def _cu_seqlens_from_positions(positions: torch.Tensor | None) -> torch.Tensor | None:
    if positions is None:
        return None
    assert positions.shape[0] == 1, (
        "qwen3_5 GDN packing patch assumes one packed row per micro-batch "
        f"(miles' RL convention); got batch_size={positions.shape[0]}."
    )
    return create_varlen_metadata_for_document(positions).cu_seq_q


def _patched_block_forward(self, x, attention_masks, positions=None):
    """Same as upstream Qwen35TransformerBlock.forward, except the GDN branch also
    receives ``positions`` (upstream drops it: ``h = self.attn(h)``)."""
    h = self.attention_norm(x)
    if self.full_attn:
        h = self.attn(h, attention_masks, positions)
    else:
        h = self.attn(h, positions)
    x = x + h

    h = self.ffn_norm(x)
    if self.moe_enabled:
        x = x + self.moe(h)
    else:
        x = x + self.feed_forward(h)
    return x


def _patched_causal_conv(self, x, conv, cu_seqlens=None):
    """Same as upstream _causal_conv when cu_seqlens is None (kept for callers that
    don't have positions, e.g. any future non-packed forward). When cu_seqlens is given,
    delegates to fla's own causal_conv1d, which is cu_seqlens-aware (resets at each
    document boundary instead of sliding the kernel window across it) — this is the
    exact tested kernel Mamba/DeltaNet architectures already rely on for packed
    training, not a hand-rolled boundary mask.
    """
    if cu_seqlens is None:
        x = F.pad(x.transpose(1, 2), [self.conv_kernel_size - 1, 0])
        x = conv(x)
        return F.silu(x).transpose(1, 2)

    from fla.modules.conv.causal_conv1d import causal_conv1d

    weight = conv.weight.squeeze(1)  # nn.Conv1d depthwise: (C, 1, K) -> fla's (C, K)
    y, _ = causal_conv1d(x, weight=weight, bias=conv.bias, activation="silu", cu_seqlens=cu_seqlens)
    return y


def _patched_gdn_forward(self, x, positions=None):
    """Same as upstream GatedDeltaNet.forward, with positions threaded into the conv
    and kernel calls so both can be cu_seqlens-aware for packed documents."""
    bs, seqlen, _ = x.shape
    cu_seqlens = _cu_seqlens_from_positions(positions)

    xq = self._causal_conv(self.in_proj_q(x), self.conv_q, cu_seqlens=cu_seqlens)
    xk = self._causal_conv(self.in_proj_k(x), self.conv_k, cu_seqlens=cu_seqlens)
    xv = self._causal_conv(self.in_proj_v(x), self.conv_v, cu_seqlens=cu_seqlens)
    xz = self.in_proj_z(x)
    xa = self.in_proj_a(x)
    xb = self.in_proj_b(x)

    xq = xq.view(bs, seqlen, -1, self.key_head_dim)
    xk = xk.view(bs, seqlen, -1, self.key_head_dim)
    xv = xv.view(bs, seqlen, -1, self.value_head_dim)

    g = -torch.exp(self.A_log.float()) * F.softplus(xa.float() + self.dt_bias)
    beta = torch.sigmoid(xb)

    output = self.kernel(xq, xk, xv, g, beta, cu_seqlens=cu_seqlens)

    xz = xz.view(bs, seqlen, -1, self.value_head_dim)
    output = self.norm(output, xz)
    output = output.reshape(bs, seqlen, -1)
    return self.out_proj(output)


def _patched_kernel_forward(self, q, k, v, g, beta, cu_seqlens=None):
    """Same as upstream GatedDeltaKernel.forward, with cu_seqlens threaded into the fla
    kernel call — fla.ops.gated_delta_rule.chunk_gated_delta_rule already resets the
    recurrent state at each document boundary when given cu_seqlens; upstream just never
    passes it."""
    if q.shape[2] != v.shape[2]:
        assert v.shape[2] % q.shape[2] == 0
        repeat = v.shape[2] // q.shape[2]
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)

    if self.backend == "torch_native":
        assert cu_seqlens is None, "torch_native GDN backend has no packed-document support"
        return _torch_native_gated_delta(q, k, v, g, beta)

    if self.backend == "fla_chunked":
        result = _fla_chunk_gated_delta_rule(
            q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens
        )
    elif self.backend == "fla_fused_recurrent":
        result = _fla_fused_recurrent_gated_delta_rule(
            q, k, v, g, beta=beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens
        )
    else:
        raise ValueError(f"Unknown fla_backend {self.backend!r}.")
    return result[0]


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    Qwen35TransformerBlock.forward = _patched_block_forward
    GatedDeltaNet._causal_conv = _patched_causal_conv
    GatedDeltaNet.forward = _patched_gdn_forward
    GatedDeltaKernel.forward = _patched_kernel_forward
    _APPLIED = True
