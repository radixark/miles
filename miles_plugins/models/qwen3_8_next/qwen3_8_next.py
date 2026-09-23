"""Qwen3.8-Next block spec: the GPT decoder with Qwen3.8-Next hyper-connections.

Every block layernorm is dropped: the checkpoint has none (each HC's ``hc_norm`` is the
pre-block norm), and a leftover norm would load at init values.
"""

import copy

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.hf_config import load_hf_config, register_hf_config_aliases
from miles_plugins.models.qwen3_5 import Attention as Qwen35LinearAttention
from miles_plugins.models.qwen3_5 import _get_text_config
from miles_plugins.models.qwen3_8_next.hyper_connection import (
    Qwen38NextHCHeadContraction,
    Qwen38NextHyperConnection,
    Qwen38NextPLEHyperConnection,
)
from miles_plugins.models.qwen3_8_next.ops.attention import Qwen38NextAttention


def _layer_types(text_config):
    """Per-layer ``linear_attention`` / ``full_attention`` labels, with Qwen3.5's fallback."""
    if hasattr(text_config, "layer_types") and text_config.layer_types:
        return list(text_config.layer_types)
    interval = getattr(text_config, "full_attention_interval", 4)
    n = text_config.num_hidden_layers
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)]


def _hc_spec(config, *, with_ple: bool = False):
    """The attention-site HC on the PLE layer also owns the PLE module.

    PLE's increment lands on the widened residual before the read gate, and the HC
    state is PLE's query.
    """
    return ModuleSpec(module=Qwen38NextPLEHyperConnection if with_ple else Qwen38NextHyperConnection)


def _strip_block_layernorms(layer_spec, config):
    """Replace the fused-layernorm qkv with a plain TE linear, and drop pre_mlp_layernorm."""
    submodules = layer_spec.submodules
    attn = submodules.self_attention
    if getattr(attn, "submodules", None) is not None and hasattr(attn.submodules, "linear_qkv"):
        attn.submodules.linear_qkv = TEColumnParallelLinear
    submodules.input_layernorm = IdentityOp
    submodules.pre_mlp_layernorm = IdentityOp


class Qwen38NextLinearAttention(Qwen35LinearAttention):
    """Qwen3.5's gated-delta-net wrapper with its input layernorm removed.

    The attention HC's ``hc_norm`` already normalises the GDN input. Identity rather
    than deletion keeps ``hf_forward`` inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layernorm = torch.nn.Identity()


def _apply_qwen3_8_next_config(config, text_config) -> None:
    """Put the Qwen3.8-Next fields on a TransformerConfig built from argparse.

    Megatron has no CLI flags for them; mirrors ``Qwen38NextBridge._build_config``.
    """
    config.enable_hyper_connections = True
    config.num_residual_streams = getattr(text_config, "hc_count", 4)
    config.qwen3_8_next_hc_lowrank = getattr(text_config, "hc_lowrank", 320)

    config.qwen3_8_next_ple_layer_ids = sorted({int(i) - 1 for i in getattr(text_config, "ple_layer_ids", None) or []})
    config.qwen3_8_next_ple_embed_dim = getattr(text_config, "ple_embed_dim", 2560)
    config.qwen3_8_next_ngram_size = getattr(text_config, "ngram_size", 3)
    config.qwen3_8_next_heads_per_ngram = getattr(text_config, "heads_per_ngram", 8)
    config.qwen3_8_next_ngram_vocab_size_base = getattr(text_config, "ngram_vocab_size_base", 20000000)
    config.qwen3_8_next_split_ngram_parts = getattr(text_config, "split_ngram_parts", 128)
    config.qwen3_8_next_ple_conv_kernel_size = getattr(text_config, "ple_conv_kernel_size", 4)
    config.qwen3_8_next_ple_conv_dilation = (
        getattr(text_config, "ple_conv_dilation", None) or config.qwen3_8_next_ngram_size
    )
    config.qwen3_8_next_eos_token_id = getattr(text_config, "eos_token_id", 0)

    config.qwen3_8_next_indexer_budget = getattr(text_config, "indexer_budget", 2048)
    config.qwen3_8_next_indexer_compress_ratio = getattr(text_config, "indexer_compress_ratio", 4)
    config.qwen3_8_next_indexer_n_heads = getattr(text_config, "indexer_n_heads", 4)
    config.qwen3_8_next_indexer_head_dim = getattr(text_config, "indexer_head_dim", 128)
    config.qwen3_8_next_indexer_kv_heads = getattr(text_config, "indexer_kv_heads", 1)


def get_qwen3_8_next_spec(args, config, vp_stage=None):
    """Transformer block spec for Qwen3.8-Next."""
    register_hf_config_aliases()
    hf_config = load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    _apply_qwen3_8_next_config(config, text_config)
    config.qwen3_8_next_hf_checkpoint = args.hf_checkpoint

    if getattr(config, "virtual_pipeline_model_parallel_size", None):
        raise NotImplementedError(
            "Qwen3.8-Next + interleaved pipeline parallelism is unverified: "
            "megatron/core/pipeline_parallel/schedules.py widens every intermediate "
            "P2P buffer uniformly on the VPP path and flags its own logic as "
            "simplified. Run with --num-layers-per-virtual-pipeline-stage unset."
        )

    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    layer_types = _layer_types(text_config)

    ple_here = [i for i in config.qwen3_8_next_ple_layer_ids if offset <= i < offset + num_layers_to_build]
    if ple_here and offset > 0:
        raise NotImplementedError(
            f"PLE layers {ple_here} landed on pipeline stage starting at layer {offset}, "
            "not the first stage. PLE hashes input token ids, which are only available "
            "where the embedding is; a later stage has hidden states and nothing to hash."
        )

    for layer_id in range(num_layers_to_build):
        global_layer_id = layer_id + offset
        layer_spec = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])

        with_ple = global_layer_id in config.qwen3_8_next_ple_layer_ids
        layer_spec.submodules.self_attention_hyper_connection = _hc_spec(config, with_ple=with_ple)
        layer_spec.submodules.mlp_hyper_connection = _hc_spec(config)

        if layer_types[global_layer_id] == "linear_attention":
            layer_spec.submodules.self_attention = ModuleSpec(
                module=Qwen38NextLinearAttention,
                params={"args": args},
            )
        else:
            layer_spec.submodules.self_attention = ModuleSpec(
                module=Qwen38NextAttention,
                params=dict(layer_spec.submodules.self_attention.params or {}),
                submodules=layer_spec.submodules.self_attention.submodules,
            )

        _strip_block_layernorms(layer_spec, config)
        transformer_layer_spec.layer_specs[layer_id] = layer_spec

    transformer_layer_spec.hc_head_contraction = ModuleSpec(module=Qwen38NextHCHeadContraction)

    transformer_layer_spec.layer_norm = IdentityOp

    return transformer_layer_spec
