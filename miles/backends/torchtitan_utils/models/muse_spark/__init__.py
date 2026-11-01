"""Muse Spark Open on torchtitan: the text decoder, sized from the checkpoint's config.json.

torchtitan's own flavors hardcode the released 62-layer sizes and place the global-attention layers by
counting forward, which a sliced checkpoint does not follow; sizing from the config, with the attention and
RoPE schedules read per layer, lets a slice and the full model share this path. The constants the config
does not carry (post-norm gain centers, final-norm gain center, embedding multiplier, route-norm epsilon,
the metap router multiplier) stay at the released flavor's values, and the ones it does carry are checked
against torchtitan's before the model is built.

Every flavor builds the vision and audio encoders, as torchtitan requires; with no dataset modalities they
never run, and the checkpoint's tower weights load into them unchanged.
"""

import math
from dataclasses import dataclass

import torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform.base import convert_config_type
from torchtitan.models.muse_spark import MuseSparkModel
from torchtitan.models.muse_spark import flavors as titan_flavors
from torchtitan.models.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter

from miles.utils.hf_utils.config import hf_text_config

__all__ = ["MilesMuseSparkModel", "build_model_config"]

_METAP_BASE_WIDTH = 256.0
_ROUTE_NORM_EPSILON = 1e-15
# auxiliary-loss-free balancing: the buffer must exist for the checkpoint's bias to load; miles freezes it
_LOAD_BALANCE_COEFF = 1e-3


class _MuseSparkStateDictAdapter(MuseSparkStateDictAdapter):
    """Converts the parts miles streams on their own.

    miles exports the model in parts (the dense weights, then each grouped expert tensor, and under LoRA
    only the adapters), so a part is converted without the rest of its layer. A load still refuses an
    incomplete checkpoint: it converts the model's own tensors first, which validates the checkpoint's
    index against every expected key, and ``from_hf`` checks the whole key set of a checkpoint that
    carries its towers. A LoRA factor shares its projection's rows and goes through the same row split
    and RoPE permute as the weight, so only the rows are checked; a load still fails on any other shape
    in ``load_state_dict``.
    """

    @staticmethod
    def _validate_pipeline_local_key_set(provided, expected_by_component, *, missing_label, unexpected_label):
        unexpected = provided - set().union(*expected_by_component.values())
        if unexpected:
            raise ValueError(f"{unexpected_label}: " + ", ".join(sorted(unexpected)))

    @staticmethod
    def _validate_tensor(key, tensor, metadata):
        if tuple(tensor.shape[:1]) != tuple(metadata.shape[:1]):
            raise ValueError(f"row count mismatch for {key}: expected {metadata.shape}, got {tuple(tensor.shape)}")


class MilesMuseSparkModel(MuseSparkModel):
    state_dict_adapter_cls = _MuseSparkStateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(MuseSparkModel.Config):
        pass


def build_model_config(
    flavor: str, *, parallelism: ParallelismConfig, seq_len: int, attn_backend: str = "flex", hf_config
) -> MilesMuseSparkModel.Config:
    """Sized from the checkpoint's config.json, as Megatron is; ``flavor`` only names the run."""
    text = hf_text_config(hf_config)
    if seq_len > text.max_position_embeddings:
        raise ValueError(f"--seq-length {seq_len} exceeds max_position_embeddings {text.max_position_embeddings}")
    _require_torchtitan_constants(text)
    dim = text.hidden_size
    vision_encoder, vision_adapter = titan_flavors.muse_spark_navit_configs(
        titan_flavors._MUSE_SPARK_816B_VISION_ADAPTER_DIM
    )
    config = titan_flavors._muse_spark_config(
        dim=dim,
        num_layers=text.num_hidden_layers,
        vocab_size=text.vocab_size,
        num_heads=text.num_attention_heads,
        num_kv_heads=text.num_key_value_heads,
        head_dim=text.head_dim,
        max_context_length=seq_len,
        global_attention_layers=_global_attention_layers(text),
        sliding_window_size=text.sliding_window,
        compressed_dim=text.moe_hidden_size,
        expert_hidden_dim=text.moe_intermediate_size,
        num_experts=text.num_local_experts,
        top_k=text.num_experts_per_tok,
        position_temperature_floor_scale=None,
        position_temperature_query_scale=None,
        query_scale=text.qk_scale_factor,
        post_norm_gain_centers=(1.0,) * text.num_hidden_layers,
        post_attention_norm_has_weight=False,
        final_norm_gain_center=0.0,
        embedding_multiplier=1.0,
        output_multiplier=text.output_multiplier,
        output_soft_cap_temp=text.final_logit_softcapping,
        final_norm_input_dtype=torch.bfloat16,
        route_scale=text.routed_scaling_factor,
        router_logit_multiplier=text.output_multiplier,
        route_norm_epsilon=_ROUTE_NORM_EPSILON,
        load_balance_coeff=_LOAD_BALANCE_COEFF,
        expert_bias_init=0.0,
        attn_backend=attn_backend,
        moe_comm_backend="standard",
        dataset_modalities=(),
        audio_encoder=titan_flavors.muse_spark_audio_encoder_config(output_size=dim),
        vision_encoder=vision_encoder,
        vision_adapter=vision_adapter,
        parallelism=parallelism,
    )
    # torchtitan compiles the norm, q/k and residual regions with recompile limits sized for its trainer's one
    # grad mode and fixed shapes; RL runs the model without grad for log-probs and with grad for training, over
    # micro-batches of varying token counts, and the hard limit (fullgraph) then fails the first train step.
    config.local_compile_regions = []
    return convert_config_type(config, MilesMuseSparkModel)


def _global_attention_layers(text) -> set[int]:
    """Global layers from the per-layer schedule; torchtitan keys RoPE off the same set (global = NoPE)."""
    global_layers = set()
    for layer_id, (kind, theta) in enumerate(zip(text.layer_types, text.layer_rope_theta, strict=True)):
        if kind not in ("sliding_attention", "full_attention"):
            raise ValueError(f"layer {layer_id} is {kind}, which Muse Spark on torchtitan does not have")
        if (kind == "full_attention") != (theta == 0):
            raise ValueError(f"layer {layer_id} is {kind} with rope theta {theta}; global layers have no RoPE")
        if kind == "full_attention":
            global_layers.add(layer_id)
        elif not math.isclose(theta, titan_flavors._ROPE_THETA):
            raise ValueError(
                f"layer {layer_id} rope theta {theta} differs from torchtitan's {titan_flavors._ROPE_THETA}"
            )
    return global_layers


def _require_torchtitan_constants(text) -> None:
    """What torchtitan fixes for this architecture; a different checkpoint would build the wrong model silently."""
    checks = {
        "rms_norm_eps": (text.rms_norm_eps, titan_flavors._NORM_EPS),
        "post_norm_eps": (text.post_norm_eps, titan_flavors._POST_NORM_EPS),
        "residual_gate_temperature": (text.residual_gate_temperature, titan_flavors._RESIDUAL_GATE_TEMPERATURE),
        "qk_scale_factor": (text.qk_scale_factor, titan_flavors._SCALE_QUERY_NUMERATOR / math.sqrt(text.head_dim)),
        # metap: the output and router multipliers are both the inverse square root of the width multiplier
        "output_multiplier": (text.output_multiplier, (text.hidden_size / _METAP_BASE_WIDTH) ** -0.5),
    }
    mismatched = {
        name: (actual, expected)
        for name, (actual, expected) in checks.items()
        if not math.isclose(actual, expected, rel_tol=1e-3)
    }
    if mismatched:
        raise ValueError(f"the checkpoint differs from torchtitan's Muse Spark (got, expected): {mismatched}")
    if text.num_shared_experts:
        raise ValueError("Muse Spark on torchtitan has no shared experts")
    if any(kind != "sparse" for kind in text.mlp_layer_types):
        raise ValueError("Muse Spark on torchtitan routes every layer; the checkpoint has dense layers")
