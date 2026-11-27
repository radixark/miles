"""Bind rotary operations by variant; adapt layouts at the framework seam.

Ordinary full-dimensional Neox rotation delegates to the serving object. No
model-name dispatch is used here.
"""

import torch


def _uses_shared_rotary(config, hf):
    parameters = (
        getattr(hf, "rope_parameters", None) or {}
    )  # config-access-exempt: HF model families differ in optional rope_parameters metadata
    scaling = (
        getattr(hf, "rope_scaling", None) or {}
    )  # config-access-exempt: HF model families differ in optional rope_scaling metadata
    ordinary_scaling = (
        set(scaling) <= {"rope_type", "type", "rope_theta"}
        and scaling.get("rope_type", scaling.get("type", "default")) == "default"
    )
    return (
        not getattr(
            config, "multi_latent_attention", False
        )  # config-access-exempt: TransformerConfig versions and model families differ in optional multi_latent_attention
        and not getattr(config, "rotary_interleaved", False)
        and getattr(config, "mrope_section", None) is None
        and getattr(config, "rotary_percent", 1.0) == 1.0
        and getattr(hf, "partial_rotary_factor", 1.0) == 1.0
        and ordinary_scaling
        and parameters.get("rope_type", parameters.get("type", "default")) == "default"
        and not getattr(config, "apply_rope_scaling", False)
        and getattr(config, "rotary_scaling_factor", 1.0) == 1.0
        and getattr(config, "seq_len_interpolation_factor", None) in (None, 1.0)
    )


def enforce_fused_rope(args, config) -> str:
    """Install the rotary binding and report the implementation selected here."""
    # Construction-only optional dependencies: importing the hook needs no CUDA.
    from miles.utils.hf_utils.config import load_hf_config
    from miles_plugins.top.install import install_fused_rope

    hf = load_hf_config(args.hf_checkpoint)
    if not _uses_shared_rotary(config, hf):
        raise NotImplementedError("[top] only ordinary full-dimensional Neox RoPE has a TOP program")
    if getattr(
        config, "fused_single_qkv_rope", False
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional fused_single_qkv_rope
        raise NotImplementedError("[top] shared rotary adapter requires separate Q/K calls")
    parameters = (
        getattr(hf, "rope_parameters", None) or {}
    )  # config-access-exempt: HF model families differ in optional rope_parameters metadata
    base = parameters.get(
        "rope_theta", getattr(hf, "rope_theta", None)
    )  # config-access-exempt: HF model families differ in optional rope_theta metadata
    # GPT configs carry no rotary_base; the model takes it from the launch args.
    trainer_base = (
        getattr(config, "rotary_base", None) or args.backend.rotary_base
    )  # config-access-exempt: TransformerConfig versions and model families differ in optional rotary_base
    if base is None or trainer_base != base:
        raise ValueError("[top] trainer and shared rotary base must match")
    if config.params_dtype != torch.bfloat16:
        raise NotImplementedError("[top] shared rotary adapter requires BF16 tensors")
    # Only the delegated variant needs the serving rotary implementation.
    from sglang.srt.layers.rotary_embedding import get_rope

    with torch.device("cuda"):
        operation = get_rope(
            config.kv_channels,
            rotary_dim=config.kv_channels,
            max_position=hf.max_position_embeddings,
            base=base,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
    if not operation._force_native:
        raise RuntimeError("[top] publish the shared TOP runtime before constructing rotary ops")
    config._top_rope = operation
    config.apply_rope_fusion = True
    install_fused_rope()
    return "delegate: SGLang ordinary rotary"


def _packed_frequency_indices(cu_seqlens, rows, device):
    """Packed lengths -> per-sequence position indices."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    torch._assert_async(torch.all(lengths >= 0), "packed lengths must be monotonic")
    torch._assert_async(cu_seqlens[0] == 0, "packed lengths must start at zero")
    torch._assert_async(cu_seqlens[-1] == rows, "packed lengths do not cover the rows")
    flat = torch.arange(rows, device=device, dtype=torch.int64)
    sequence = torch.searchsorted(cu_seqlens[1:], flat, right=True)
    return flat - cu_seqlens[sequence]


class _SharedRotary(torch.autograd.Function):
    """Serving rotation forward; backward rotates the gradient by the inverse angle."""

    @staticmethod
    def forward(ctx, rows, positions, operation):
        ctx.save_for_backward(positions)
        ctx.operation = operation
        empty_key = rows.new_empty((rows.shape[0], 0, rows.shape[-1]))
        # The serving kernel rotates in place; keep the caller's tensor intact.
        result, _ = operation(positions, rows.clone(), empty_key)
        return result

    @staticmethod
    def backward(ctx, grad):
        (positions,) = ctx.saved_tensors
        # Shared rotary admits only full-dimensional Neox rotation.
        cos, sin = ctx.operation.cos_sin_cache.index_select(0, positions).float().chunk(2, dim=-1)
        cos, sin = cos[:, None, :], sin[:, None, :]
        g = grad.float()
        half = g.shape[-1] // 2
        g1, g2 = g[..., :half], g[..., half:]
        return torch.cat((g1 * cos + g2 * sin, g2 * cos - g1 * sin), dim=-1).to(grad.dtype), None, None


def apply_fused_rope(
    t,
    freqs,
    config,
    cu_seqlens=None,
    mscale=1.0,
    cp_group=None,
    mla_rotary_interleaved=False,
    inverse=False,
    mla_output_remove_interleaving=False,
    max_seqlen=None,
):
    operation = config._top_rope
    if not config.apply_rope_fusion:
        raise RuntimeError("[top] rotary binding disabled after construction")
    if mscale != 1.0 or inverse or mla_rotary_interleaved or mla_output_remove_interleaving:
        raise NotImplementedError("[top] unsupported transformation for ordinary shared rotary")
    if t.dtype != torch.bfloat16 or t.shape[-1] != config.kv_channels:
        raise ValueError("[top] shared rotary tensor dtype/head dimension mismatch")
    if freqs.ndim != 4 or freqs.shape[1:3] != (1, 1) or freqs.shape[-1] != config.kv_channels:
        raise NotImplementedError("[top] ordinary rotary requires a full-dimensional 1D frequency table")
    shape = t.shape
    # Ordinary, unscaled RoPE's first frequency has inv_freq=1. Read it rather
    # than assuming zero-based positions: this preserves supplied offsets. No GPU->CPU conversion.
    position_table = freqs[:, 0, 0, 0]
    torch._assert_async(
        torch.all(position_table == position_table.round()),
        "non-integral rotary positions",
    )
    position_table = position_table.to(torch.int64)
    if cu_seqlens is None:
        if t.ndim != 4 or freqs.shape[0] != shape[0]:
            raise ValueError("[top] SBHD rotary requires one supplied position per sequence row")
        positions = position_table.repeat(shape[1])
        rows = t.transpose(0, 1).reshape(-1, shape[-2], shape[-1])
    else:
        if t.ndim not in (3, 4) or (t.ndim == 4 and shape[1] != 1):
            raise ValueError("[top] packed rotary requires THD or T1HD tensors")
        rows = t.reshape(-1, shape[-2], shape[-1])
        indices = _packed_frequency_indices(cu_seqlens, rows.shape[0], t.device)
        torch._assert_async(torch.all(indices < freqs.shape[0]), "rotary frequency table too short")
        positions = position_table[indices]
    torch._assert_async(torch.all(positions >= 0), "negative rotary positions")
    torch._assert_async(
        torch.all(positions < operation.max_position_embeddings),
        "rotary positions exceed shared cache",
    )
    result = _SharedRotary.apply(rows, positions, operation)
    if cu_seqlens is None:
        return result.reshape(shape[1], shape[0], shape[-2], shape[-1]).transpose(0, 1).contiguous()
    return result.reshape(shape)
