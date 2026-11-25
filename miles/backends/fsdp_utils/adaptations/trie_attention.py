"""Trie-packed attention for the FSDP backend (CUDA only: it calls flash-attn).

A trie row stores the tokens that samples share once (``training_utils/data/trie_packing.py``). Each segment's
queries attend to their ancestors plus the segment itself, which ``flash_attn_varlen_func`` computes over keys
gathered per segment, with its causal diagonal aligned bottom-right (as it is for ``seqlen_q < seqlen_k``). The
function is registered with HF's ``AttentionInterface``, so it serves every architecture whose attention
dispatches through it; HF RoPE reads the row's logical ``position_ids``, so positions need nothing here.
"""

import functools
import importlib

import torch
from transformers import AttentionInterface, PretrainedConfig
from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask

from miles.backends.fsdp_utils.adaptations.trie_gather import gather_rows
from miles.backends.fsdp_utils.adaptations.trie_linear_attention import TrieLinearSchedule
from miles.backends.training_utils.data.trie_packing import TrieLayout, TrieModality, TrieModelSpec

TRIE_ATTENTION = "miles_trie"

# The --attn-implementation values trie attention runs on, and the module each takes flash_attn_varlen_func from.
# All three align the causal diagonal bottom-right and take FA2's keyword arguments.
FLASH_ATTN_MODULES = {
    "flash_attention_2": "flash_attn",
    "flash_attention_3": "flash_attn_interface",
    "flash_attention_4": "flash_attn.cute",
}

# Layer types whose token mixing runs over a trie row; any other would mix across samples.
TRIE_LAYER_TYPES = frozenset(
    {
        "full_attention",
        "sliding_attention",
        "linear_attention",
    }
)
# Attention arguments trie attention does not pass on: attention sinks (gpt-oss) and logit soft-capping (Gemma 2).
_UNSUPPORTED_ATTENTION_KWARGS = ("s_aux", "softcap")

# The config attribute that names the short convolution's kernel size, by linear-attention architecture.
_CONV_KERNEL_SIZE_NAMES = ("linear_conv_kernel_dim",)  # GatedDeltaNet


class TrieAttentionArgs:
    """One trie row's segment boundaries and key gathers, on the device; passed to the model as
    ``miles_trie_attn_args``."""

    def __init__(self, layout: TrieLayout, device: torch.device) -> None:
        window = layout.window_kv
        self.cu_seqlens_q = layout.cu_seqlens_q.to(device=device, dtype=torch.int32)
        self.max_seqlen_q = layout.max_seqlen_q
        self.kv_index = layout.kv.index.to(device)
        self.kv_cu_seqlens = layout.kv.cu_seqlens.to(device=device, dtype=torch.int32)
        self.kv_max_seqlen = layout.kv.max_seqlen
        self.window_kv_index = None if window is None else window.index.to(device)
        self.window_kv_cu_seqlens = None if window is None else window.cu_seqlens.to(device=device, dtype=torch.int32)
        self.window_kv_max_seqlen = None if window is None else window.max_seqlen
        # the convolution and recurrence schedule, for linear-attention layers
        self.linear = None if layout.conv_kv is None else TrieLinearSchedule(layout, device)


def _layer_types(text_config: PretrainedConfig) -> tuple[str, ...]:
    return tuple(
        getattr(text_config, "layer_types", None) or ()
    )  # config-access-exempt: architectures without per-layer attention types omit layer_types


def trie_model_spec(text_config: PretrainedConfig, media: tuple[TrieModality, ...] = ()) -> TrieModelSpec:
    """What this model's trie layouts need, read from its config:

    - a window gather, if any layer slides
    - a gather for the short convolution of linear-attention layers, if any
    - ``media``, the image and video modalities of its processor output (``trie_media``)
    """
    layer_types = _layer_types(text_config)
    if unsupported := sorted(set(layer_types) - TRIE_LAYER_TYPES):
        raise ValueError(f"--trie-packing does not implement layer types {unsupported}")
    window = getattr(
        text_config, "sliding_window", None
    )  # config-access-exempt: architectures without sliding-window attention omit sliding_window
    if window is not None and "sliding_attention" not in layer_types:
        # such models' layers may pass it anyway (Mistral, Phi-3, StarCoder2), but the rows would lack its gather
        raise ValueError(f"--trie-packing: sliding_window={window} is set, but no layer type is sliding_attention")
    conv_kernel_size = _conv_kernel_size(text_config) if "linear_attention" in layer_types else None
    return TrieModelSpec(
        sliding_window_size=window,
        conv_kernel_size=conv_kernel_size,
        media=media,
    )


def trie_media(config: PretrainedConfig) -> tuple[TrieModality, ...]:
    """The image and video modalities of a Qwen-VL-style model (Qwen3-VL, Qwen3.5); none for any other."""
    vision = getattr(config, "vision_config", None)  # config-access-exempt: text-only configs omit vision_config
    names = ("image_token_id", "video_token_id")
    if vision is None or not all(
        hasattr(config, name) for name in names
    ):  # config-access-exempt: only Qwen-VL-style configs name both placeholder tokens
        return ()
    merge = vision.spatial_merge_size
    return (
        TrieModality(config.image_token_id, "pixel_values", "image_grid_thw", merge),
        TrieModality(config.video_token_id, "pixel_values_videos", "video_grid_thw", merge),
    )


def trie_attn_implementation(config: PretrainedConfig, tower_attn_implementation: str) -> str | dict[str, str]:
    """``attn_implementation`` for a trie-packed model: ``miles_trie`` for the language model, while other
    sub-models (a vision tower) keep ``tower_attn_implementation``, as their inputs are not the trie row."""
    towers = [key for key in config.sub_configs if key != "text_config"]
    if not towers:
        return TRIE_ATTENTION
    return {"": TRIE_ATTENTION, "text_config": TRIE_ATTENTION} | {key: tower_attn_implementation for key in towers}


def _conv_kernel_size(text_config: PretrainedConfig) -> int:
    """The short convolution's kernel size, under the name its architecture's config gives it."""
    return next(
        getattr(text_config, name)
        for name in _CONV_KERNEL_SIZE_NAMES
        if hasattr(text_config, name)  # config-access-exempt: linear-attention configs name it differently
    )


def trie_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    flash_attn_module: str = FLASH_ATTN_MODULES["flash_attention_2"],
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """HF attention over one trie row: ``[1, H, T, D]`` inputs, ``[1, T, H, D]`` output, on the
    ``flash_attn_varlen_func`` of ``flash_attn_module``."""
    flash_attn_varlen_func = importlib.import_module(flash_attn_module).flash_attn_varlen_func

    trie: TrieAttentionArgs | None = kwargs.get("miles_trie_attn_args")
    # No fallback: attention over the row as one sequence would let samples attend to each other.
    if trie is None:
        raise ValueError(f"{TRIE_ATTENTION} attention needs the trie row passed as `miles_trie_attn_args`")
    if dropout:
        raise ValueError("trie-packed attention requires attention dropout 0")
    if unsupported := [name for name in _UNSUPPORTED_ATTENTION_KWARGS if kwargs.get(name) is not None]:
        raise ValueError(f"trie-packed attention does not implement {unsupported}")
    if sliding_window is None:
        rows, cu_seqlens_k, max_seqlen_k, window_size = trie.kv_index, trie.kv_cu_seqlens, trie.kv_max_seqlen, (-1, -1)
    else:
        rows, cu_seqlens_k, max_seqlen_k = trie.window_kv_index, trie.window_kv_cu_seqlens, trie.window_kv_max_seqlen
        window_size = (sliding_window - 1, 0)
    output = flash_attn_varlen_func(
        query[0].transpose(0, 1),
        gather_rows(key[0].transpose(0, 1), rows),
        gather_rows(value[0].transpose(0, 1), rows),
        cu_seqlens_q=trie.cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=trie.max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True,
        window_size=window_size,
        softmax_scale=scaling,
    )
    if isinstance(output, tuple):  # FA4 also returns the log-sum-exp
        output = output[0]
    return output.unsqueeze(0), None


def register_trie_attention(attn_implementation: str) -> None:
    """Make ``attn_implementation="miles_trie"`` resolve to ``trie_attention_forward`` on the flash-attn that
    ``attn_implementation`` names; registering again replaces it.

    Its mask is flash-attn's, which is no mask for an unpadded row: an implementation HF does not know
    would otherwise get a dense ``[T, T]`` mask built for every forward.
    """
    forward = functools.partial(trie_attention_forward, flash_attn_module=FLASH_ATTN_MODULES[attn_implementation])
    AttentionInterface.register(TRIE_ATTENTION, forward)
    AttentionMaskInterface.register(TRIE_ATTENTION, flash_attention_mask)
