"""Muse Glimmer on torchtitan: the text decoder, sized from the checkpoint's config.json.

torchtitan's own flavors hardcode the released sizes; sizing from the config lets a truncated
checkpoint and the full one share this path. torchtitan derives the architecture's constants
(FFN width, norm eps, RoPE phase, logit soft-cap) from the layer count, so the config is checked
against them before the model is built.
"""

import math
from dataclasses import dataclass

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform.base import convert_config_type
from torchtitan.models.muse_glimmer import MuseGlimmerModel
from torchtitan.models.muse_glimmer import flavors as titan_flavors
from torchtitan.models.muse_glimmer.state_dict_adapter import MuseGlimmerStateDictAdapter

from miles.utils.hf_utils.config import hf_text_config

__all__ = ["MilesMuseGlimmerModel", "build_model_config"]

_SOFT_CAP = 20.0


class _MuseGlimmerStateDictAdapter(MuseGlimmerStateDictAdapter):
    """Permutes q/k rows by the tensor's own shape, so a rank-r ``lora_B`` exports like its weight."""

    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        return super()._permute(w, n_heads_arg)

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        return super()._reverse_permute(w, n_heads_arg)


class MilesMuseGlimmerModel(MuseGlimmerModel):
    state_dict_adapter_cls = _MuseGlimmerStateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(MuseGlimmerModel.Config):
        pass


def build_model_config(
    flavor: str, *, parallelism: ParallelismConfig, seq_len: int, attn_backend: str = "flex", hf_config
) -> MilesMuseGlimmerModel.Config:
    """Sized from the checkpoint's config.json, as Megatron is; ``flavor`` only names the run."""
    text = hf_text_config(hf_config)
    if seq_len > text.max_position_embeddings:
        raise ValueError(f"--seq-length {seq_len} exceeds max_position_embeddings {text.max_position_embeddings}")
    _require_torchtitan_constants(text)
    config = titan_flavors._muse_glimmer_config(
        dim=text.hidden_size,
        n_layers=text.num_hidden_layers,
        n_heads=text.num_attention_heads,
        n_kv_heads=text.num_key_value_heads,
        head_dim=text.head_dim,
        vocab_size=text.vocab_size,
        max_context_length=seq_len,
        window_pattern=_window_pattern(text),
        output_multiplier=text.output_multiplier,
        attn_backend=attn_backend,
        enable_sp=parallelism.enable_sequence_parallel,
    )
    return convert_config_type(config, MilesMuseGlimmerModel)


def _window_pattern(text) -> list[int]:
    """The cyclic window pattern torchtitan aligns to the last layer, checked against every HF layer."""
    n_layers = text.num_hidden_layers
    tail = text.layer_types[-titan_flavors._EVERY_N_LAYERS_NOPE :]
    pattern = [text.sliding_window if kind == "sliding_attention" else 0 for kind in tail]
    for layer_id, (kind, theta) in enumerate(zip(text.layer_types, text.layer_rope_theta, strict=True)):
        window = titan_flavors._layer_window_size(layer_id, n_layers, pattern)
        if (kind == "sliding_attention") != (window is not None):
            raise ValueError(f"layer {layer_id} is {kind}, which torchtitan's cyclic window pattern cannot place")
        expected_theta = titan_flavors._ROPE_THETA if titan_flavors._layer_use_rope(layer_id, n_layers) else 0.0
        if not math.isclose(theta, expected_theta, rel_tol=1e-3):
            raise ValueError(
                f"layer {layer_id} has rope theta {theta}, not the {expected_theta} of torchtitan's every-fourth "
                "NoPE phase"
            )
    return pattern


def _require_torchtitan_constants(text) -> None:
    """What torchtitan fixes for this architecture; a different checkpoint would build the wrong model silently."""
    checks = {
        "intermediate_size": (text.intermediate_size, int(titan_flavors._FFN_DIM_MULTIPLIER * text.hidden_size)),
        "rms_norm_eps": (text.rms_norm_eps, titan_flavors._NORM_EPS),
        "post_norm_eps": (text.post_norm_eps, titan_flavors._POST_NORM_EPS),
        "final_logit_softcapping": (text.final_logit_softcapping, _SOFT_CAP),
        "rope_theta": (text.rope_parameters["rope_theta"], titan_flavors._ROPE_THETA),
        "qk_scale_factor": (text.qk_scale_factor, titan_flavors._SCALE_QUERY_NUMERATOR / math.sqrt(text.head_dim)),
    }
    mismatched = {
        name: (actual, expected)
        for name, (actual, expected) in checks.items()
        if not math.isclose(actual, expected, rel_tol=1e-3)
    }
    if mismatched:
        raise ValueError(f"the checkpoint differs from torchtitan's Muse Glimmer (got, expected): {mismatched}")
    if text.rope_parameters["rope_type"] != "default":
        raise ValueError(f"Muse Glimmer on torchtitan has unscaled RoPE, not {text.rope_parameters['rope_type']!r}")
