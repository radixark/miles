"""MiniMax-M3 block spec: the GPT decoder with MSA sparse self-attention on every layer.

The pre-attention norm is kept as its own module (not fused into ``linear_qkv``) because
the indexer reads the normed hidden states too.
"""

import copy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from transformers import PretrainedConfig

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.minimax_m3.attention import MiniMaxM3Attention, MiniMaxM3HeadNorm


def text_config_of(hf_config):
    """The checkpoint's remote config class keeps ``text_config`` as a plain dict."""
    text_config = getattr(hf_config, "text_config", None) or hf_config
    if isinstance(text_config, dict):
        text_config = PretrainedConfig.from_dict(text_config)
    return text_config


def apply_minimax_m3_config(config, text_config) -> None:
    sparse = text_config.sparse_attention_config
    if not (
        sparse["use_sparse_attention"]
        and all(sparse["sparse_attention_freq"])
        and all(sparse["sparse_disable_index_value"])
    ):
        raise NotImplementedError(
            "only MiniMax-M3.1 is supported: sparse attention without an index value branch on every layer; "
            "MiniMax-M3 has dense-attention layers"
        )
    assert sparse["sparse_score_type"] == "max"
    config.minimax_index_n_heads = sparse["sparse_num_index_heads"]
    config.minimax_index_head_dim = sparse["sparse_index_dim"]
    config.minimax_index_topk_blocks = sparse["sparse_topk_blocks"]
    config.minimax_index_block_size = sparse["sparse_block_size"]
    config.minimax_index_local_blocks = sparse["sparse_local_block"]
    config.minimax_index_init_blocks = sparse["sparse_init_block"]
    assert config.minimax_index_n_heads == config.num_query_groups, "index head h selects for kv head h"
    # swigluoai: clamp(g, max=7) * sigmoid(1.702 g) * (clamp(u, -7, 7) + 1)
    config.glu_linear_offset = 1.0
    config.activation_func_clamp_value = text_config.swiglu_limit


def get_minimax_m3_spec(args, config, vp_stage=None):
    text_config = text_config_of(load_hf_config(args.hf_checkpoint))
    apply_minimax_m3_config(config, text_config)

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    block_spec = get_gpt_decoder_block_spec(config, **kwargs)

    for layer_id in range(get_num_layers_to_build(config, vp_stage=vp_stage)):
        layer_spec = copy.deepcopy(block_spec.layer_specs[layer_id])
        attn = layer_spec.submodules.self_attention
        attn.submodules.linear_qkv = TEColumnParallelLinear
        attn.submodules.q_layernorm = MiniMaxM3HeadNorm
        attn.submodules.k_layernorm = MiniMaxM3HeadNorm
        layer_spec.submodules.input_layernorm = TENorm
        layer_spec.submodules.self_attention = ModuleSpec(
            module=MiniMaxM3Attention,
            params=dict(attn.params or {}),
            submodules=attn.submodules,
        )
        block_spec.layer_specs[layer_id] = layer_spec
    return block_spec
