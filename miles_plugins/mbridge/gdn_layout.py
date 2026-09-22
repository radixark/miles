"""Shared helper for the GDN bridges: which ``linear_attn`` tensors the Megatron model stores head-interleaved."""

# Keep in sync with miles_plugins.models.gdn_attention.HEAD_INTERLEAVED_PARAMS (imported lazily there
# because the bridges must stay importable without the model plugins' CUDA-side dependencies).
_HEAD_INTERLEAVED_PARAMS = {
    "qwen3_5": ("in_proj_qkv.weight", "conv1d.weight", "conv1d.bias"),
    "qwen3_next": ("conv1d.weight", "conv1d.bias"),
}


def _head_interleaved_linear_attn_param(mcore_weights_name: str, hf_layout: str) -> str | None:
    """``linear_attn.<name>`` suffix when ``mcore_weights_name`` is one of the head-interleaved tensors, else ``None``."""
    marker = "self_attention.linear_attn."
    if marker not in mcore_weights_name:
        return None
    name = mcore_weights_name.split(marker, 1)[1]
    return name if name in _HEAD_INTERLEAVED_PARAMS[hf_layout] else None
