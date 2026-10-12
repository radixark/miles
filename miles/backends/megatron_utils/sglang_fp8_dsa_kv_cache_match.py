"""Make Megatron DSA attention read the K/V values SGLang's FP8 KV cache stores.

SGLang serving a DSA model with ``--kv-cache-dtype fp8_e4m3`` stores each token's
512-wide compressed latent as E4M3 with one FP32 scale per 128 values and keeps the
64-wide RoPE part in BF16; its attention reads those stored values back as BF16. The
trainer otherwise attends over the unquantized BF16 latent. With
``--match-sglang-fp8-dsa-kv-cache-values``, a forward pre-hook on every Megatron
``AbsorbedMLASelfAttention.core_attention`` replaces the key with SGLang's
quantize-then-dequantize result, computed by SGLang's own cache kernels so the values
are bit-identical to the sampler's cache. The gradient passes through unchanged.
"""

import logging
from collections.abc import Sequence

import torch
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import AbsorbedMLASelfAttention
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import TransformerLayer

logger = logging.getLogger(__name__)

KV_LORA_RANK = 512
QK_POS_EMB_HEAD_DIM = 64
_KV_COMPRESSED_DIM = KV_LORA_RANK + QK_POS_EMB_HEAD_DIM
_HOOK_HANDLE_ATTR = "_miles_sglang_fp8_dsa_kv_cache_match_hook"


def sglang_fp8_dsa_kv_cache_value(kv_compressed: torch.Tensor) -> torch.Tensor:
    """Return the BF16 values SGLang's FP8 DSA KV cache stores for ``kv_compressed``.

    ``kv_compressed`` is the absorbed-MLA key ``[..., 1, 576]``: the normalized latent
    followed by the RoPE part, exactly what SGLang writes into its cache.
    """
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache

    if kv_compressed.dtype != torch.bfloat16:
        raise ValueError(f"SGLang's FP8 KV cache quantizes BF16 keys, got {kv_compressed.dtype}")
    if kv_compressed.dim() < 2 or kv_compressed.shape[-2:] != (1, _KV_COMPRESSED_DIM):
        raise ValueError(
            f"Expected an absorbed-MLA key of shape [..., 1, {_KV_COMPRESSED_DIM}], got {tuple(kv_compressed.shape)}"
        )
    if not kv_compressed.is_contiguous():
        raise ValueError("Expected a contiguous absorbed-MLA key")

    num_cache_rows = kv_compressed.numel() // _KV_COMPRESSED_DIM
    if num_cache_rows == 0:
        return torch.empty_like(kv_compressed)
    cache_rows = kv_compressed.view(num_cache_rows, 1, 1, _KV_COMPRESSED_DIM)
    return dequantize_k_cache(quantize_k_cache(cache_rows)).view(kv_compressed.shape)


class _MatchSGLangFP8DSAKVCacheValue(torch.autograd.Function):
    """Forward returns the cached value itself; backward is the identity (straight-through)."""

    @staticmethod
    def forward(ctx, kv_compressed: torch.Tensor) -> torch.Tensor:
        return sglang_fp8_dsa_kv_cache_value(kv_compressed)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def match_sglang_fp8_dsa_kv_cache_value(kv_compressed: torch.Tensor) -> torch.Tensor:
    """SGLang's FP8 KV cache value in the forward pass, with a straight-through gradient."""
    return _MatchSGLangFP8DSAKVCacheValue.apply(kv_compressed)


def _feed_core_attention_sglang_fp8_dsa_kv_cache_value(
    module: torch.nn.Module, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    # AbsorbedMLASelfAttention passes (query, key, ...) positionally on every core_attention call.
    if len(args) < 2:
        raise RuntimeError(
            "--match-sglang-fp8-dsa-kv-cache-values expects AbsorbedMLASelfAttention to pass the key as the second "
            f"positional argument of core_attention; got {len(args)} positional arguments"
        )
    return (args[0], match_sglang_fp8_dsa_kv_cache_value(args[1]), *args[2:]), kwargs


def register_sglang_fp8_dsa_kv_cache_match_hook(attention: AbsorbedMLASelfAttention) -> None:
    """Make one absorbed-MLA DSA attention module read SGLang's cache-represented values."""
    if not isinstance(attention.core_attention, DSAttention):
        raise ValueError(
            "--match-sglang-fp8-dsa-kv-cache-values supports only DSAttention core attention, "
            f"got {type(attention.core_attention).__name__}"
        )
    config = attention.config
    if config.kv_lora_rank != KV_LORA_RANK or config.qk_pos_emb_head_dim != QK_POS_EMB_HEAD_DIM:
        raise ValueError(
            f"SGLang's FP8 DSA KV cache stores a {KV_LORA_RANK}-wide latent and a {QK_POS_EMB_HEAD_DIM}-wide "
            f"RoPE part; got kv_lora_rank={config.kv_lora_rank}, qk_pos_emb_head_dim={config.qk_pos_emb_head_dim}"
        )
    # A second hook would quantize the already-dequantized key again.
    if hasattr(attention.core_attention, _HOOK_HANDLE_ATTR):
        raise RuntimeError("--match-sglang-fp8-dsa-kv-cache-values hook is already registered on this attention")
    handle = attention.core_attention.register_forward_pre_hook(
        _feed_core_attention_sglang_fp8_dsa_kv_cache_value, with_kwargs=True
    )
    setattr(attention.core_attention, _HOOK_HANDLE_ATTR, handle)


def register_sglang_fp8_dsa_kv_cache_match_hooks(model_chunks: Sequence[torch.nn.Module]) -> int:
    """Make every transformer layer's attention read SGLang's cache-represented values.

    Applies to every forward of these layers, whichever weights are loaded, including
    activation recomputation. Fails on any layer whose attention it cannot cover, so no
    layer silently stays in BF16.
    """
    num_hooked = 0
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, TransformerLayer) or isinstance(module.self_attention, IdentityOp):
                continue
            if not isinstance(module.self_attention, AbsorbedMLASelfAttention):
                raise ValueError(
                    "--match-sglang-fp8-dsa-kv-cache-values supports only Megatron's absorbed-MLA DSA attention "
                    f"(--dsa-impl megatron); layer {module.layer_number} uses {type(module.self_attention).__name__}"
                )
            register_sglang_fp8_dsa_kv_cache_match_hook(module.self_attention)
            num_hooked += 1
    logger.info(f"--match-sglang-fp8-dsa-kv-cache-values: hooked {num_hooked} absorbed-MLA DSA attention modules")
    return num_hooked
