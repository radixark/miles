from __future__ import annotations

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

_USE_SGLANG_ROPE = False


def enable_sglang_rope() -> None:
    """Enable the SGLang-compatible RoPE path used by dense true-on-policy."""

    global _USE_SGLANG_ROPE
    _USE_SGLANG_ROPE = True


def disable_sglang_rope() -> None:
    global _USE_SGLANG_ROPE
    _USE_SGLANG_ROPE = False


def is_sglang_rope_enabled() -> bool:
    return _USE_SGLANG_ROPE


def sglang_apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor, is_neox_style: bool = True) -> Tensor:
    if cos.dim() == 2:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float()
    sin = sin.float()

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim < x.shape[-1]:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = sglang_apply_rotary_pos_emb(x_rot, cos, sin, is_neox_style)
        return torch.cat((x_rot, x_pass), dim=-1).to(orig_dtype)

    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1).to(orig_dtype)

    return torch.stack((o1, o2), dim=-1).flatten(-2).to(orig_dtype)


def sglang_apply_rotary_pos_emb_with_freqs(
    x: Tensor, freqs: Tensor, config: TransformerConfig, layer_number: int | None = None
) -> Tensor:
    del layer_number

    x_seq_len = x.shape[0]
    freqs_seq_len = freqs.shape[0]

    freqs_flat = freqs.squeeze(1).squeeze(1)
    head_dim = x.shape[-1]
    raw_angles = freqs_flat[..., : head_dim // 2]
    cos = torch.cos(raw_angles)
    sin = torch.sin(raw_angles)
    is_neox_style = not getattr(config, "rotary_interleaved", False)

    if x_seq_len == freqs_seq_len:
        return sglang_apply_rotary_pos_emb(x, cos, sin, is_neox_style)
    if freqs_seq_len < x_seq_len:
        x_valid = x[:freqs_seq_len]
        x_valid = sglang_apply_rotary_pos_emb(x_valid, cos, sin, is_neox_style)
        return torch.cat([x_valid, x[freqs_seq_len:]], dim=0)

    cos = cos[:x_seq_len]
    sin = sin[:x_seq_len]
    return sglang_apply_rotary_pos_emb(x, cos, sin, is_neox_style)
