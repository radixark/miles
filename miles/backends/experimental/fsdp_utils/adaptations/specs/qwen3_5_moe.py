"""Qwen3.5/3.6/Qwen3-Next (GatedDeltaNet) adaptations.

Two hooks:

* packing — a config-time packed-doc reset that feeds cu_seqlens to fla
  chunk/recurrent_gated_delta_rule and seq_idx to causal_conv1d_fn per packed document. Patches the
  DecoderLayer/GatedDeltaNet class forwards; kernel logic lives in ``models/qwen3_5_moe.py``.
* precision — pin the token embedding to the compute dtype whenever gather and compute disagree.
"""

from dataclasses import replace

from ..packing.registry import PackingPatch, register_packing_patch
from ..precision import ModuleSel, PrecisionPolicyHook, PrecisionSpec, Rule, dtype_name, register_precision_policy


def _applies(hf_config) -> bool:
    """True for GatedDeltaNet archs (Qwen3.5/3.6, Qwen3-Next): a linear_attention layer type or qwen3_5."""
    if hf_config is None:
        return False
    model_type = str(getattr(hf_config, "model_type", "") or "")
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return (layer_types is not None and "linear_attention" in layer_types) or "qwen3_5" in model_type


def _apply():
    from ...models.qwen3_5_moe import apply_gateddeltanet_packing_patch

    return apply_gateddeltanet_packing_patch()


def _precision_applies(hf_config, args) -> bool:
    return _applies(hf_config)


def _resolve_precision(base_policy, hf_config, args):
    """Gather the token embedding at the compute dtype when it differs from the gather dtype.

    ``F.embedding`` is not an autocast-covered op, so the embedding output carries the *gathered*
    weight dtype, and it seeds the residual stream. ``Qwen3_5RMSNorm`` ends in ``output.type_as(x)``
    and every residual add promotes, so that one dtype propagates through the whole activation path:
    an fp32-gather run computes its norms and residual adds in fp32 while autocast runs the matmuls
    at the compute dtype, which is exactly the train/rollout mismatch this pins shut. Matmul weights
    keep the run's gather dtype — only the embedding is moved.

    No-op under the default policy, where compute is the gather dtype and nothing disagrees.
    """
    compute_dtype = base_policy.autocast_dtype
    if compute_dtype is None or compute_dtype == base_policy.param_dtype:
        return base_policy
    rule = Rule(ModuleSel(fqn="*embed_tokens"), gather=dtype_name(compute_dtype))
    return replace(base_policy, precision_spec=PrecisionSpec(rules=base_policy.precision_spec.rules + (rule,)))


register_packing_patch(PackingPatch("gated_deltanet_packing", _applies, "config", _apply))
register_precision_policy(PrecisionPolicyHook("gated_deltanet_embed_gather", _precision_applies, _resolve_precision))
