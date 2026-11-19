"""Inkling2 numerics, each mirroring the serving kernel it must match: fp32 compute with
the same rounding points (the engine's bf16 stores), so train and rollout logprobs agree.
"""

import math

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask

from miles.utils.mxfp8 import mxfp8_quantize
from miles.utils.nvfp4 import FP4_E2M1_MAX, nvfp4_quantize_1d, nvfp4_weight_e4m3_max
from miles_plugins.mbridge.inkling2 import dequantize_mxfp8, dequantize_nvfp4
from miles_plugins.models.inkling.ops import flex_compiled


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float, out_dtype: torch.dtype) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(out_dtype)


def attn_res_mix(
    sources: list[torch.Tensor],
    queries: torch.Tensor,
    *,
    normalize_values: bool,
    partial_addend: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Block attention residual: per token, softmax attention across depth.

    ``sources`` are the completed blocks followed by the current partial block;
    ``partial_addend`` is added to the partial block in fp32 (the attention output of a
    non-boundary layer, which the engine does not round into the partial). Each row of
    ``queries`` ([n_queries, hidden]) yields one fp32 mixture.
    """
    values = torch.stack([s.float() for s in sources])
    if partial_addend is not None:
        values = torch.cat((values[:-1], (values[-1] + partial_addend.float()).unsqueeze(0)))
    inv_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-6)
    scaled_queries = queries.float() * queries.shape[-1] ** -0.5
    logits = torch.einsum("n...d,qd,n...->q...n", values, scaled_queries, inv_rms)
    probabilities = torch.softmax(logits, dim=-1)
    if normalize_values:
        values = values * inv_rms.unsqueeze(-1)
    return list(torch.einsum("q...n,n...d->q...d", probabilities, values).unbind(0))


def attn_res_partial(partial: torch.Tensor, addend: torch.Tensor, *, normalize_values: bool) -> torch.Tensor:
    """The mixture when no block is complete: the fp32 partial sum itself."""
    value = partial.float() + addend.float()
    if normalize_values:
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
    return value


def positions_in_sequence(seqlens: list[int], device: torch.device) -> torch.Tensor:
    lengths = torch.tensor(seqlens, device=device)
    starts = torch.cumsum(lengths, 0) - lengths
    return torch.arange(int(lengths.sum()), device=device) - torch.repeat_interleave(starts, lengths)


def short_conv(
    x: torch.Tensor,
    positions: torch.Tensor,
    static_weight: torch.Tensor,
    *,
    dynamic_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Residual causal short convolution over packed sequences ([T, C] -> [T, C]).

    ``static_weight`` is [W, C] indexed by lag (lag 0 is the current token).
    ``dynamic_weight`` is [T, heads, W]; it is shared by the channels of a head and added
    to the static weight. The output is scaled by (W * (2 if dynamic else 1) + 1) ** -0.5.
    """
    width = static_weight.shape[0]
    weight = static_weight.float().unsqueeze(0)
    if dynamic_weight is not None:
        head_dim = x.shape[-1] // dynamic_weight.shape[1]
        dynamic = dynamic_weight.float().repeat_interleave(head_dim, dim=1).transpose(1, 2)
        weight = weight + dynamic
    xf = x.float()
    out = xf
    for lag in range(width):
        shifted = F.pad(xf, (0, 0, lag, 0))[: xf.shape[0]]
        shifted = shifted * (positions >= lag).unsqueeze(-1)
        out = out + weight[:, lag] * shifted
    scale = (width * (2 if dynamic_weight is not None else 1) + 1) ** -0.5
    return (out * scale).to(x.dtype)


def l2_clip(
    x: torch.Tensor,
    threshold: float,
    *,
    scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip each head vector ([T, heads, dim]) to L2 norm ``threshold`` after an optional
    learned per-head scale; ``output_scale`` ([T]) multiplies the rounded result."""
    value = x.float()
    if scale is not None:
        value = value * scale.float()
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    out = (value * (threshold / (norm + eps)).clamp(max=1.0)).to(x.dtype)
    if output_scale is not None:
        out = (out.float() * output_scale.view(-1, 1, 1)).to(x.dtype)
    return out


def log_scaling_tau(positions: torch.Tensor, n_floor: float, alpha: float) -> torch.Tensor:
    return 1 + alpha * torch.log(torch.clamp((positions.float() + 1) / n_floor, min=1.0))


# Rows per chunk: bounds capped_swiglu's fp32 temporaries (a hot expert rank can hold ~1M rows).
CAPPED_SWIGLU_CHUNK_ROWS = 32768


def _capped_swiglu_rows(fc1_out: torch.Tensor, probs: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    gate, up = torch.chunk(fc1_out.float(), 2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    if gate_cap > 0:
        gate = gate_cap * torch.tanh(gate / gate_cap)
    if up_cap > 0:
        up = up_cap * torch.tanh(up / up_cap)
    return (gate * sigmoid * up * probs.float()).to(fc1_out.dtype)


class _CappedSwiGLU(torch.autograd.Function):
    """Row-chunked capped SwiGLU that saves only its inputs: the unfused fp32 graph kept about
    five [rows, ffn] fp32 tensors alive, which ran a hot EP rank out of memory."""

    @staticmethod
    def forward(ctx, fc1_out, probs, gate_cap, up_cap):
        ctx.save_for_backward(fc1_out, probs)
        ctx.caps = (gate_cap, up_cap)
        out = fc1_out.new_empty((*fc1_out.shape[:-1], fc1_out.shape[-1] // 2))
        for start in range(0, fc1_out.shape[0], CAPPED_SWIGLU_CHUNK_ROWS):
            rows = slice(start, start + CAPPED_SWIGLU_CHUNK_ROWS)
            out[rows] = _capped_swiglu_rows(fc1_out[rows], probs[rows], gate_cap, up_cap)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        fc1_out, probs = ctx.saved_tensors
        gate_cap, up_cap = ctx.caps
        grad_fc1 = torch.empty_like(fc1_out) if ctx.needs_input_grad[0] else None
        grad_probs = torch.empty_like(probs) if ctx.needs_input_grad[1] else None
        # Replays each chunk's forward graph, so the gradients equal the unfused graph's bit for bit.
        for start in range(0, fc1_out.shape[0], CAPPED_SWIGLU_CHUNK_ROWS):
            rows = slice(start, start + CAPPED_SWIGLU_CHUNK_ROWS)
            with torch.enable_grad():
                x, p = fc1_out[rows].detach().requires_grad_(), probs[rows].detach().requires_grad_()
                dx, dp = torch.autograd.grad(_capped_swiglu_rows(x, p, gate_cap, up_cap), (x, p), grad_out[rows])
            if grad_fc1 is not None:
                grad_fc1[rows] = dx
            if grad_probs is not None:
                grad_probs[rows] = dp
        return grad_fc1, grad_probs, None, None


def capped_swiglu(fc1_out: torch.Tensor, probs: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    """``softcap(g) * sigmoid(g) * softcap(u) * prob`` in fp32 with one rounding.

    The sigmoid reads the uncapped gate, as the serving kernel does; fc1 rows are
    contiguous ``[gate; up]``.
    """
    if fc1_out.dim() != 2 or probs.shape[0] != fc1_out.shape[0]:
        return _capped_swiglu_rows(fc1_out, probs, gate_cap, up_cap)
    return _CappedSwiGLU.apply(fc1_out, probs, gate_cap, up_cap)


def fake_quantize_expert_weight(weight: torch.Tensor, fmt: str) -> torch.Tensor:
    """One expert weight as the engine serves it after a weight sync, with a straight-through
    gradient: encoded by the sync's quantizers (TE NVFP4 / MXFP8), then dequantized."""
    with torch.no_grad():
        if fmt == "nvfp4":
            packed, scales, decode = nvfp4_quantize_1d(weight)
            denominator = torch.tensor(float(nvfp4_weight_e4m3_max()) * FP4_E2M1_MAX, device=weight.device)
            served = dequantize_nvfp4(packed, scales, decode * denominator, denominator)
        else:
            assert fmt == "mxfp8", fmt
            served = dequantize_mxfp8(*mxfp8_quantize(weight))
    return weight + (served.to(weight.dtype) - weight).detach()


# Rows per backward chunk: bounds the softcap gradient's one fp32 temporary (~0.8 GiB at a 201k vocab).
SOFTCAP_LOGITS_CHUNK_ROWS = 1024


class _SoftcapMaskedLogits(torch.autograd.Function):
    """fp32 ``cap * tanh(logits / cap)`` with ``padded`` vocab columns at the fp32 minimum, in place.

    The unfused graph peaked at three [tokens, vocab] fp32 tensors (~74 GiB at 32k tokens and a
    201k vocab) and kept two for backward. This overwrites fp32 logits (the LM head's own output),
    keeps only the result for backward, and writes the gradient into the incoming one.
    """

    @staticmethod
    def forward(ctx, logits, cap, padded):
        if logits.dtype == torch.float32:
            out = logits
            ctx.mark_dirty(out)
        else:
            out = logits.to(torch.float32)
        out.div_(cap).tanh_().mul_(cap)
        out.masked_fill_(padded, torch.finfo(out.dtype).min)
        ctx.save_for_backward(out, padded)
        ctx.cap, ctx.input_dtype = cap, logits.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out, padded = ctx.saved_tensors
        vocab = out.shape[-1]
        rows = out.view(-1, vocab)
        if grad_out.dtype == torch.float32 and grad_out.is_contiguous():
            grad = grad_out  # the loss's gradient buffer, consumed only here
        else:
            grad = grad_out.float().contiguous()
        grad_rows = grad.view(-1, vocab)
        for start in range(0, rows.shape[0], SOFTCAP_LOGITS_CHUNK_ROWS):
            chunk = slice(start, start + SOFTCAP_LOGITS_CHUNK_ROWS)
            # d/dx cap * tanh(x / cap) = 1 - tanh^2, with tanh = out / cap
            t = rows[chunk] / ctx.cap
            grad_rows[chunk].mul_(t.mul_(t).neg_().add_(1)).masked_fill_(padded, 0)
        return grad.view(out.shape).to(ctx.input_dtype), None, None


def softcap_masked_logits(logits: torch.Tensor, cap: float, padded: torch.Tensor) -> torch.Tensor:
    """fp32 ``cap * tanh(logits / cap)`` with ``padded`` (a [vocab] mask) at the fp32 minimum."""
    return _SoftcapMaskedLogits.apply(logits, cap, padded)


def relative_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_logits: torch.Tensor,
    seqlens: list[int],
    *,
    window: int | None,
    block_mask_cache: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal GQA attention with a learned relative bias, returning (out, lse).

    q: [T, heads, dim]; k, v: [T, kv_heads, dim]; rel_logits: [T, heads, extent] where
    ``rel_logits[i, h, d]`` is added for key distance ``d = i - j < extent``. ``window``
    keeps keys with ``i - j <= window`` (local layers); None is full causal attention.
    Packed sequences never attend across boundaries. lse is the natural-log normalizer.
    """
    total = q.shape[0]
    extent = rel_logits.shape[-1]
    segment = torch.repeat_interleave(
        torch.arange(len(seqlens), device=q.device), torch.tensor(seqlens, device=q.device)
    )
    bias = rel_logits.permute(1, 0, 2).float().contiguous()

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        rel = bias[h, q_idx, torch.clamp(distance, 0, extent - 1)]
        return torch.where(distance < extent, score + rel, score)

    def mask_mod(b, h, q_idx, kv_idx):
        keep = (q_idx >= kv_idx) & (segment[q_idx] == segment[kv_idx])
        if window is not None:
            keep = keep & (q_idx - kv_idx <= window)
        return keep

    key = (tuple(seqlens), window)
    block_mask = block_mask_cache.get(key)
    if block_mask is None:
        block_mask_cache.clear()
        block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=q.device)
        block_mask_cache[key] = block_mask
    out, aux = flex_compiled()(
        q.permute(1, 0, 2).unsqueeze(0),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        score_mod=score_mod,
        block_mask=block_mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        enable_gqa=True,
        return_aux=AuxRequest(lse=True),
    )
    return out.squeeze(0).permute(1, 0, 2), aux.lse.squeeze(0).permute(1, 0)
