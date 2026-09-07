"""Patches for ops with no seam. Each one named, guarded, and asserted to have applied."""

from __future__ import annotations

_INSTALLED: dict[str, bool] = {}


def install_tree_tp_reduce() -> bool:
    """Route row-parallel linears' all-reduce through sglang's deterministic tree.

    The reduce is in RowParallelLinear.forward, not _forward_impl, so there is no seam to bind.
    """
    if _INSTALLED.get("tree_tp_reduce"):
        return True
    from megatron.core.tensor_parallel import layers as _layers

    from miles_plugins.top.ops import tree_reduce_from_tp_region

    if not hasattr(_layers, "reduce_from_tensor_model_parallel_region"):
        raise RuntimeError("[top] tree-reduce patch: symbol missing; megatron changed shape")
    _layers.reduce_from_tensor_model_parallel_region = tree_reduce_from_tp_region
    _INSTALLED["tree_tp_reduce"] = True
    return True


def installed() -> dict[str, bool]:
    return dict(_INSTALLED)


def install_fused_rope() -> bool:
    """Bind the function SelfAttention actually calls, without replacing its forward."""
    from megatron.core.models.common.embeddings import rope_utils
    from megatron.core.transformer import attention

    from miles_plugins.top.rope import apply_fused_rope

    current = getattr(attention, "apply_rotary_pos_emb", None)
    if current is not apply_fused_rope:
        if current is None or current is not rope_utils.apply_rotary_pos_emb:
            raise RuntimeError("[top] fused-RoPE patch: unexpected SelfAttention call site")
        attention.apply_rotary_pos_emb = apply_fused_rope
    if attention.apply_rotary_pos_emb is not apply_fused_rope:
        raise RuntimeError("[top] fused-RoPE patch did not bind SelfAttention")
    _INSTALLED["fused_rope"] = True
    return True
