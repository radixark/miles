"""Immutable text-model semantics from the Muse Spark checkpoint configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch.nn.functional as F

from miles_plugins.models.muse_spark.ops.norms import calculate_gain_center

_TEXT_METADATA = frozenset(
    {
        "_name_or_path",
        "architectures",
        "model_type",
        "transformers_version",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "audio_token_id",
        "image_token_id",
        "video_token_id",
        "vision_config",
        "audio_perception_modality_args",
        "use_audio_perception",
        "gated_delta_net",
        "allow_kv_head_padding",
        "u_proj_freq_embed_size",
    }
)


@dataclass(frozen=True)
class MuseSpark1xConfig:
    hidden_size: int = 3072
    num_hidden_layers: int = 24
    vocab_size: int = 202048
    num_attention_heads: int = 48
    num_key_value_heads: int = 6
    head_dim: int = 64
    max_position_embeddings: int = 16384
    ffn_dim_multiplier: float = 2.0
    non_linearity: str = "swiglu"
    parameter_dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-5
    rms_post_norm_eps: float = 1e-8
    norm_affine: bool = True
    post_norm_affine: bool = True
    norm_zero_centered_gamma: bool = False
    post_norm_gain_center_type: str = "one"
    output_norm_gain_center_type: str = "zero"
    normalize_tok_embeddings: bool = False
    use_ffn_learnable_scales: bool = False
    use_residual_gate: bool = False
    residual_dtype: str = "fp32"
    use_qk_norm: bool = True
    use_attn_o_gate: bool = False
    use_attn_o_norm: bool = False
    use_sqrt_attn_scaling: bool = False
    use_rope: bool = True
    rope_theta: float = 500000.0
    rope_use_fp32_in_outer_product: bool = True
    # Per-layer RoPE schedule (1 = RoPE, 0 = NoPE) from ``layer_rope_theta``; None = every layer uses RoPE.
    rope_layer_schedule: tuple[int, ...] | None = None
    # Per-layer attention window (0 = global) from ``layer_types`` / ``sliding_window``; None = all global.
    global_attn_cfg: tuple[int, ...] | None = None
    attn_temperature_tuning_floor_scale: float | None = None
    attn_temperature_tuning_layers: str | None = None
    attn_temperature_tuning_q_scale_constant: float | None = None
    output_soft_cap_temp: float | None = None
    use_fp32_for_output: bool = False
    output_norm_in_fp32: bool = False
    tie_word_embeddings: bool = False
    use_u_proj: bool = False
    moe_is_enabled: bool = False
    num_local_experts: int = 0
    num_experts_per_tok: int = 1
    moe_compression_factor: int = 1
    moe_override_routed_hidden_dim: int | None = None
    moe_interleave_moe_layer_step: int = 1
    moe_n_dense_layers_before_moe: int = 0
    moe_route_scale: float = 1.0
    moe_router_in_fp32: bool = False
    # Checkpoint metadata; MCore owns the routing normalization epsilon.
    moe_router_score_division_eps: float | None = None
    moe_router_score_normalization: str = "sigmoid"
    moe_use_pre_expert_norm: bool = False
    moe_use_post_expert_norm: bool = False
    moe_use_shared_expert: bool = False
    metap_enabled: bool = False
    metap_base_width: float = 256.0
    metap_m_emb: float = 1.0
    metap_mode: str = "sp"
    metap_router_has_one_inf_dim: bool = False
    metap_use_output_multiplier_without_metap: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "hidden_size",
            "num_hidden_layers",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "moe_compression_factor",
        ):
            field_value = getattr(self, field_name)
            if type(field_value) is not int or field_value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_size % self.moe_compression_factor:
            raise ValueError("hidden_size must be divisible by moe_compression_factor")
        if self.use_rope and self.head_dim % 2:
            raise ValueError("interleaved RoPE requires an even head_dim")
        if self.rope_layer_schedule is not None:
            if (
                not isinstance(self.rope_layer_schedule, tuple)
                or len(self.rope_layer_schedule) != self.num_hidden_layers
                or any(entry not in (0, 1) for entry in self.rope_layer_schedule)
            ):
                raise ValueError("rope_layer_schedule must hold one 0/1 entry per layer")
        if self.global_attn_cfg is not None:
            if not isinstance(self.global_attn_cfg, tuple) or len(self.global_attn_cfg) != self.num_hidden_layers:
                raise ValueError("global_attn_cfg must hold one window per layer or be None")
            if any(type(window) is not int or window < 0 for window in self.global_attn_cfg):
                raise ValueError("global_attn_cfg entries must be nonnegative integers")
            if 1 in self.global_attn_cfg:
                raise ValueError("window 1 is unsupported: the reference aliases its zero extent to global attention")
        for field_name in (
            "ffn_dim_multiplier",
            "rms_norm_eps",
            "rms_post_norm_eps",
            "rope_theta",
            "metap_base_width",
        ):
            field_value = getattr(self, field_name)
            if not math.isfinite(field_value) or field_value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.dense_intermediate_size < 1:
            raise ValueError("ffn_dim_multiplier must produce a nonempty intermediate dimension")
        calculate_gain_center(self.post_norm_gain_center_type, 1)
        calculate_gain_center(self.output_norm_gain_center_type, -1)
        if self.non_linearity != "swiglu":
            raise ValueError("only non_linearity=swiglu is supported")
        if self.parameter_dtype != "bfloat16":
            raise ValueError("parameter_dtype must be bfloat16")
        if self.residual_dtype not in {"fp32", "bf16"}:
            raise ValueError("residual_dtype must be fp32 or bf16")
        if not self.rope_use_fp32_in_outer_product:
            raise ValueError("the reference requires FP32 RoPE cache construction")
        if self.use_u_proj or self.tie_word_embeddings:
            raise ValueError("u_proj and tied embeddings are outside the supported checkpoint contract")
        self._validate_moe()
        self._validate_scales()

    def _validate_moe(self) -> None:
        if self.moe_use_shared_expert:
            raise ValueError("shared experts are unsupported by the text reference")
        if self.moe_router_score_normalization != "sigmoid":
            raise ValueError("only sigmoid router score normalization is supported")
        if type(self.moe_interleave_moe_layer_step) is not int or self.moe_interleave_moe_layer_step < 0:
            raise ValueError("moe_interleave_moe_layer_step must be a nonnegative integer")
        if type(self.moe_n_dense_layers_before_moe) is not int or self.moe_n_dense_layers_before_moe < 0:
            raise ValueError("moe_n_dense_layers_before_moe must be a nonnegative integer")
        if self.moe_override_routed_hidden_dim is not None and (
            type(self.moe_override_routed_hidden_dim) is not int or self.moe_override_routed_hidden_dim <= 0
        ):
            raise ValueError("moe_override_routed_hidden_dim must be a positive integer or None")
        if self.moe_is_enabled:
            if type(self.num_local_experts) is not int or self.num_local_experts <= 0:
                raise ValueError("enabled MoE requires positive num_local_experts")
            if (
                type(self.num_experts_per_tok) is not int
                or not 1 <= self.num_experts_per_tok <= self.num_local_experts
            ):
                raise ValueError("num_experts_per_tok must be between 1 and num_local_experts")
        if self.moe_router_score_division_eps is not None and (
            not math.isfinite(self.moe_router_score_division_eps) or self.moe_router_score_division_eps < 0
        ):
            raise ValueError("moe_router_score_division_eps must be nonnegative or None")

    def _validate_scales(self) -> None:
        if self.metap_mode not in {"sp", "mup"}:
            raise ValueError("metap_mode must be sp or mup")
        if self.metap_use_output_multiplier_without_metap:
            raise ValueError("use_metap_output_multiplier_without_metap is not implemented by the reference")
        for field_name in ("metap_m_emb", "moe_route_scale"):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if self.output_soft_cap_temp is not None and (
            not math.isfinite(self.output_soft_cap_temp) or self.output_soft_cap_temp <= 0
        ):
            raise ValueError("output_soft_cap_temp must be finite and positive or None")
        temperature_options = (
            self.attn_temperature_tuning_floor_scale,
            self.attn_temperature_tuning_layers,
            self.attn_temperature_tuning_q_scale_constant,
        )
        if any(setting is not None for setting in temperature_options):
            raise ValueError("position temperature is not supported")

    @property
    def dense_intermediate_size(self) -> int:
        return int(self.hidden_size * self.ffn_dim_multiplier)

    @property
    def moe_intermediate_size(self) -> int:
        return self.moe_override_routed_hidden_dim or self.dense_intermediate_size

    @property
    def expert_hidden_size(self) -> int:
        return self.hidden_size // self.moe_compression_factor

    @property
    def metap_width_mult(self) -> float:
        return self.hidden_size / self.metap_base_width

    @property
    def metap_router_output_multiplier(self) -> float | None:
        if not self.metap_enabled or not self.metap_router_has_one_inf_dim:
            return None
        return self.metap_width_mult ** (-0.5 if self.metap_mode == "sp" else -1)

    @property
    def metap_output_multiplier(self) -> float:
        return self.metap_router_output_multiplier or 1.0

    @property
    def embedding_multiplier(self) -> float:
        return self.metap_m_emb if self.metap_enabled and self.normalize_tok_embeddings else 1.0

    @property
    def query_multiplier(self) -> float:
        return 3.87 if self.use_sqrt_attn_scaling else 43.7840518911 / math.sqrt(self.head_dim)

    def layer_uses_rope(self, layer_idx: int) -> bool:
        if not self.use_rope:
            return False
        return self.rope_layer_schedule is None or bool(self.rope_layer_schedule[layer_idx])

    def layer_window_size(self, layer_idx: int) -> int | None:
        """Return the kernel's left extent, excluding the current token."""
        if self.global_attn_cfg is None:
            return None
        window = self.global_attn_cfg[layer_idx]
        return window - 1 if window > 0 else None

    def layer_is_moe(self, layer_idx: int) -> bool:
        return (
            self.moe_is_enabled
            and layer_idx >= self.moe_n_dense_layers_before_moe
            and (self.moe_interleave_moe_layer_step == 0 or (layer_idx + 1) % self.moe_interleave_moe_layer_step == 0)
        )

    @property
    def no_rope_freq(self) -> tuple[int, ...]:
        return tuple(int(not self.layer_uses_rope(index)) for index in range(self.num_hidden_layers))

    @property
    def moe_layer_freq(self) -> tuple[int, ...]:
        return tuple(int(self.layer_is_moe(index)) for index in range(self.num_hidden_layers))


_MOE_KEYS = {
    "is_enabled": "moe_is_enabled",
    "num_experts": "num_local_experts",
    "topk": "num_experts_per_tok",
    "compression_factor": "moe_compression_factor",
    "override_routed_hidden_dim": "moe_override_routed_hidden_dim",
    "interleave_moe_layer_step": "moe_interleave_moe_layer_step",
    "n_dense_layers_before_moe": "moe_n_dense_layers_before_moe",
    "route_scale": "moe_route_scale",
    "router_in_fp32": "moe_router_in_fp32",
    "router_score_division_eps": "moe_router_score_division_eps",
    "router_score_normalization": "moe_router_score_normalization",
    "use_pre_expert_norm": "moe_use_pre_expert_norm",
    "use_post_expert_norm": "moe_use_post_expert_norm",
    "use_shared_expert": "moe_use_shared_expert",
}
_METAP_KEYS = {
    "use_metap": "metap_enabled",
    "base_width": "metap_base_width",
    "m_emb": "metap_m_emb",
    "metap_mode": "metap_mode",
    "router_has_one_inf_dim": "metap_router_has_one_inf_dim",
    "use_metap_output_multiplier_without_metap": "metap_use_output_multiplier_without_metap",
}
_FLAT_CONFIG_FIELDS = frozenset(
    {field.name for field in fields(MuseSpark1xConfig)}
    - {"output_norm_in_fp32", "parameter_dtype"}
    - set(_MOE_KEYS.values())
    - set(_METAP_KEYS.values())
)
_ALLOWED_CONFIG_FIELDS = (
    _FLAT_CONFIG_FIELDS | _TEXT_METADATA | {"moe", "metap", "numerics_args", "dtype", "torch_dtype"}
)


def _parse_nested(source: Mapping[str, Any], section_name: str, field_names: Mapping[str, str]) -> dict[str, Any]:
    section = source.get(section_name) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{section_name} must be a mapping")
    unsupported_fields = set(section) - set(field_names)
    if section_name == "moe" and section.get("override_shared_hidden_dim") in (None, 0):
        unsupported_fields.discard("override_shared_hidden_dim")
    if unsupported_fields:
        raise ValueError(f"unsupported {section_name} fields: {sorted(unsupported_fields)}")
    return {
        field_names[field_name]: field_value
        for field_name, field_value in section.items()
        if field_name in field_names
    }


def _parse_numerics(source: Mapping[str, Any]) -> dict[str, bool]:
    numerics = source.get("numerics_args") or {}
    if not isinstance(numerics, Mapping) or set(numerics) - {
        "use_fp32_for_output",
        "out_dtype_per_module_overrides",
    }:
        raise ValueError("unsupported numerics_args")
    output_options = {
        "use_fp32_for_output": bool(source.get("use_fp32_for_output") or numerics.get("use_fp32_for_output"))
    }
    for override in (numerics.get("out_dtype_per_module_overrides") or "").split(","):
        if not override.strip():
            continue
        override_parts = override.strip().split(":")
        if (
            len(override_parts) != 3
            or override_parts[0] not in {"output", "output.norm"}
            or override_parts[2].lower() != "fp32"
        ):
            raise ValueError(f"unsupported output dtype override: {override}")
        output_options["use_fp32_for_output" if override_parts[0] == "output" else "output_norm_in_fp32"] = True
    return output_options


ARCHITECTURE = "MuseSparkForConditionalGeneration"
MODEL_TYPE = "muse_spark"

# Constants the Miles model hard-codes (ops/residual.py, config.query_multiplier). A checkpoint config that disagrees
# describes a different model and is rejected instead of silently mapped.
_RESIDUAL_GATE_TEMPERATURE = 0.3
_QUERY_MULTIPLIER_NUMERATOR = 43.7840518911
_ROUTER_SCORE_DIVISION_EPS = 1e-15
_METAP_BASE_WIDTH = 256.0

# Architecture flags the checkpoint's modeling code implies (Miles' flat config spells them out).
_ARCHITECTURE_FLAGS: dict[str, Any] = {
    "non_linearity": "swiglu",
    "norm_affine": True,
    "use_ffn_learnable_scales": True,
    "norm_zero_centered_gamma": True,
    "normalize_tok_embeddings": True,
    "post_norm_affine": False,
    "post_norm_gain_center_type": "one",
    "output_norm_gain_center_type": "zero",
    "rope_use_fp32_in_outer_product": True,
    "use_qk_norm": True,
    "use_residual_gate": True,
    "use_sqrt_attn_scaling": False,
    "use_attn_o_gate": True,
    "use_attn_o_norm": True,
    "residual_dtype": "fp32",
}
_METAP: dict[str, Any] = {
    "use_metap": True,
    "use_metap_output_multiplier_without_metap": False,
    "metap_mode": "sp",
    "base_width": _METAP_BASE_WIDTH,
    "router_has_one_inf_dim": True,
    "m_emb": 1.0,
}

# Keys the text_config may carry (SGLang ``MuseSpark1xTextConfig`` + HF bookkeeping). Anything else is new
# semantics the Miles model does not implement, so it is rejected.
_TEXT_KEYS = frozenset(
    {
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "final_logit_softcapping",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "layer_rope_theta",
        "layer_types",
        "max_position_embeddings",
        "mlp_layer_types",
        "model_type",
        "moe_hidden_size",
        "moe_intermediate_size",
        "num_attention_heads",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_key_value_heads",
        "num_local_experts",
        "num_shared_experts",
        "output_multiplier",
        "post_norm_eps",
        "qk_scale_factor",
        "residual_gate_temperature",
        "rms_norm_eps",
        "rope_parameters",
        "routed_scaling_factor",
        "sliding_window",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
        "dtype",
        "torch_dtype",
        "transformers_version",
        "_name_or_path",
    }
)
_TOP_KEYS = frozenset(
    {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "audio_config",
        "dtype",
        "torch_dtype",
        "transformers_version",
        "_name_or_path",
        "image_token_id",
        "video_token_id",
        "audio_token_id",
        "out_hidden_size",
        "projector_hidden_size",
        "projector_hidden_act",
        "tie_word_embeddings",
        "use_cache",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    }
)


def text_fields_from_checkpoint_config(hf_config: Mapping[str, Any]) -> dict[str, Any]:
    """Flat text fields (what ``parse_muse_spark_1x_config`` consumes) of a ``MuseSparkForConditionalGeneration`` config.

    Per-layer schedules become ``rope_layer_schedule`` (1 = RoPE, 0 = NoPE) and a full-length ``global_attn_cfg``
    (window per layer, 0 = global).
    """
    unknown = set(hf_config) - _TOP_KEYS
    if unknown:
        raise ValueError(f"unsupported Spark config fields: {sorted(unknown)}")
    text = hf_config.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError("Muse Spark config needs a text_config mapping (MuseSparkForConditionalGeneration export)")
    unknown = set(text) - _TEXT_KEYS
    if unknown:
        raise ValueError(f"unsupported text_config fields: {sorted(unknown)}")
    if text.get("model_type", "muse_spark_text") != "muse_spark_text":
        raise ValueError(f"unexpected text_config.model_type {text.get('model_type')!r}")

    layers = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    head_dim = int(text["head_dim"])
    rope_parameters = text.get("rope_parameters") or {}
    if rope_parameters.get("rope_type", "default") != "default":
        raise ValueError("Muse Spark text attention supports default RoPE only")
    default_theta = rope_parameters.get("rope_theta")
    layer_types = list(text.get("layer_types") or ["full_attention"] * layers)
    rope_thetas = list(text.get("layer_rope_theta") or [default_theta] * layers)
    mlp_types = list(text.get("mlp_layer_types") or ["sparse"] * layers)
    for name, values in (
        ("layer_types", layer_types),
        ("layer_rope_theta", rope_thetas),
        ("mlp_layer_types", mlp_types),
    ):
        if len(values) != layers:
            raise ValueError(f"{name} must have num_hidden_layers entries")
    if set(layer_types) - {"sliding_attention", "full_attention"}:
        raise ValueError(f"unsupported text layer_types {sorted(set(layer_types))}")
    if set(mlp_types) - {"sparse", "dense"}:
        raise ValueError(f"unsupported mlp_layer_types {sorted(set(mlp_types))}")
    if text.get("attention_bias") or text.get("hidden_act", "silu") != "silu" or text.get("num_shared_experts", 0):
        raise ValueError("Muse Spark is bias-free, silu, routed experts only")
    if any(theta is None for theta in rope_thetas):
        raise ValueError("rope_parameters.rope_theta or layer_rope_theta is required")
    thetas = {float(theta) for theta in rope_thetas if theta}
    if len(thetas) > 1:
        raise ValueError(f"expected one RoPE theta across RoPE layers, got {sorted(thetas)}")
    # Miles expresses dense layers as a prefix (n_dense_layers_before_moe) followed by MoE layers.
    dense = [index for index, kind in enumerate(mlp_types) if kind == "dense"]
    if dense != list(range(len(dense))):
        raise ValueError(f"dense layers must form a prefix, got mlp_layer_types {mlp_types}")
    if len(dense) == layers:
        raise ValueError("a Muse Spark checkpoint without MoE layers is outside the supported contract")
    sliding_window = int(text.get("sliding_window") or 0)
    if "sliding_attention" in layer_types and sliding_window <= 1:
        raise ValueError("sliding_attention layers need sliding_window > 1")
    windows = [sliding_window if kind == "sliding_attention" else 0 for kind in layer_types]

    expected_multiplier = (hidden / _METAP_BASE_WIDTH) ** -0.5
    if not math.isclose(float(text["output_multiplier"]), expected_multiplier, rel_tol=1e-9):
        raise ValueError(
            f"output_multiplier {text['output_multiplier']} is not the metaP sp multiplier {expected_multiplier}"
        )
    expected_qk_scale = _QUERY_MULTIPLIER_NUMERATOR / math.sqrt(head_dim)
    if not math.isclose(float(text["qk_scale_factor"]), expected_qk_scale, rel_tol=1e-9):
        raise ValueError(f"qk_scale_factor {text['qk_scale_factor']} differs from Miles' {expected_qk_scale}")
    gate_temperature = float(text.get("residual_gate_temperature", _RESIDUAL_GATE_TEMPERATURE))
    if not math.isclose(gate_temperature, _RESIDUAL_GATE_TEMPERATURE):
        raise ValueError("residual_gate_temperature differs from Miles' constant 0.3")
    # moe_hidden_size is the latent (expert input) width = hidden / compression_factor;
    # moe_intermediate_size is the expert FFN width (Miles' override_routed_hidden_dim).
    latent = int(text["moe_hidden_size"])
    if latent <= 0 or hidden % latent:
        raise ValueError("hidden_size must be a positive multiple of moe_hidden_size")
    intermediate = int(text["intermediate_size"])
    if intermediate <= 0 or intermediate % hidden:
        raise ValueError("intermediate_size must be a positive multiple of hidden_size (ffn_dim_multiplier)")

    flat: dict[str, Any] = {
        "architectures": list(hf_config.get("architectures") or [ARCHITECTURE]),
        "model_type": hf_config.get("model_type", MODEL_TYPE),
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "vocab_size": int(text["vocab_size"]),
        "num_attention_heads": int(text["num_attention_heads"]),
        "num_key_value_heads": int(text["num_key_value_heads"]),
        "head_dim": head_dim,
        "max_position_embeddings": int(text["max_position_embeddings"]),
        "ffn_dim_multiplier": intermediate / hidden,
        "rms_norm_eps": float(text["rms_norm_eps"]),
        "rms_post_norm_eps": float(text["post_norm_eps"]),
        "output_soft_cap_temp": text.get("final_logit_softcapping"),
        "tie_word_embeddings": bool(text.get("tie_word_embeddings", False)),
        "use_rope": bool(thetas),
        "rope_theta": next(iter(thetas)) if thetas else float(default_theta or 500000.0),
        "rope_layer_schedule": [int(bool(theta)) for theta in rope_thetas],
        "global_attn_cfg": windows if any(windows) else None,
        "moe": {
            "is_enabled": True,
            "topk": int(text["num_experts_per_tok"]),
            "num_experts": int(text["num_local_experts"]),
            "use_shared_expert": False,
            "router_in_fp32": True,
            "route_scale": float(text.get("routed_scaling_factor", 1.0)),
            "override_routed_hidden_dim": int(text["moe_intermediate_size"]),
            "interleave_moe_layer_step": 1,
            "n_dense_layers_before_moe": len(dense),
            "router_score_normalization": "sigmoid",
            "router_score_division_eps": _ROUTER_SCORE_DIVISION_EPS,
            "compression_factor": hidden // latent,
            "use_pre_expert_norm": True,
            "use_post_expert_norm": True,
        },
        "metap": dict(_METAP),
        "dtype": hf_config.get("dtype") or hf_config.get("torch_dtype") or text.get("dtype") or "bfloat16",
        **_ARCHITECTURE_FLAGS,
    }
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = text.get(key, hf_config.get(key))
        if value is not None:
            flat[key] = value
    for key in ("image_token_id", "video_token_id", "audio_token_id"):
        if hf_config.get(key) is not None:
            flat[key] = hf_config[key]
    return flat


def parse_muse_spark_1x_config(source: Mapping[str, Any]) -> MuseSpark1xConfig:
    """Parse the text semantics of a ``MuseSparkForConditionalGeneration`` ``config.json``."""
    if not isinstance(source, Mapping):
        raise ValueError("Spark configuration must be a mapping")
    source = text_fields_from_checkpoint_config(source)
    unsupported_fields = set(source) - _ALLOWED_CONFIG_FIELDS
    if unsupported_fields:
        raise ValueError(f"unsupported Spark config fields: {sorted(unsupported_fields)}")
    config_fields = {field_name: source[field_name] for field_name in _FLAT_CONFIG_FIELDS if field_name in source}
    config_fields.update(_parse_nested(source, "moe", _MOE_KEYS))
    config_fields.update(_parse_nested(source, "metap", _METAP_KEYS))
    config_fields.update(_parse_numerics(source))
    if config_fields.get("moe_compression_factor") is None:
        config_fields["moe_compression_factor"] = 1
    attention_schedule = config_fields.get("global_attn_cfg")
    if attention_schedule is None or attention_schedule == "all" or attention_schedule == []:
        config_fields["global_attn_cfg"] = None
    elif isinstance(attention_schedule, (list, tuple)):
        config_fields["global_attn_cfg"] = tuple(attention_schedule)
    else:
        raise ValueError("global_attn_cfg must be a list, tuple, all, or None")
    rope_schedule = config_fields.get("rope_layer_schedule")
    if isinstance(rope_schedule, list):
        config_fields["rope_layer_schedule"] = tuple(rope_schedule)
    parameter_dtype = source.get("dtype") or source.get("torch_dtype") or "bfloat16"
    config_fields["parameter_dtype"] = {"bf16": "bfloat16"}.get(parameter_dtype, parameter_dtype)
    return MuseSpark1xConfig(**config_fields)


def map_to_megatron_fields(config: MuseSpark1xConfig) -> dict[str, Any]:
    """Map checkpoint semantics to native TransformerConfig fields."""
    layer_windows = [config.layer_window_size(index) for index in range(config.num_hidden_layers)]
    local_windows = {window for window in layer_windows if window is not None}
    if len(local_windows) > 1:
        raise ValueError("MCore requires a single sliding window size; global_attn_cfg has multiple sizes")
    window = next(iter(local_windows), None)
    return {
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_query_groups": config.num_key_value_heads,
        "kv_channels": config.head_dim,
        "ffn_hidden_size": config.dense_intermediate_size,
        "normalization": "RMSNorm",
        "layernorm_epsilon": config.rms_norm_eps,
        "add_bias_linear": False,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "rotary_interleaved": True,
        "no_rope_freq": list(config.no_rope_freq),
        "window_size": (window, 0) if window is not None else None,
        "window_attn_skip_freq": [int(value is not None) for value in layer_windows] if local_windows else None,
        "attention_output_gate": config.use_attn_o_gate,
        "post_self_attn_layernorm": True,
        "post_mlp_layernorm": True,
        "fp32_residual_connection": config.residual_dtype == "fp32",
        "num_moe_experts": config.num_local_experts if config.moe_is_enabled else None,
        "moe_router_topk": config.num_experts_per_tok,
        "moe_ffn_hidden_size": (config.moe_intermediate_size if config.moe_is_enabled else None),
        "moe_latent_size": (
            config.expert_hidden_size if config.moe_is_enabled and config.moe_compression_factor > 1 else None
        ),
        "moe_layer_freq": list(config.moe_layer_freq),
        "moe_router_score_function": "sigmoid",
        "moe_router_enable_expert_bias": config.moe_is_enabled,
        "freeze_e_score_correction_bias": True,
        "moe_router_topk_scaling_factor": config.moe_route_scale,
        "moe_router_dtype": "fp32" if config.moe_router_in_fp32 else None,
        # only the unfused AlltoAll FP32 combine applies probs after post_expert_norm
        "moe_token_dispatcher_type": "alltoall",
        "moe_permute_fusion": False,
        "moe_combine_in_fp32": config.moe_is_enabled,
        "expert_tensor_parallel_size": 1,
        "use_mup": False,
    }
