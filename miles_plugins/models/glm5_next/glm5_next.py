"""GLM-5.3-Flash block spec: Megatron-native mHC + per-layer KDA/DSA dispatch.

``get_gpt_decoder_block_spec`` with ``enable_hyper_connections=True`` emits
``HyperConnectionTransformerLayer`` with both HC ModuleSpec slots filled by
Megatron's ``HyperConnectionModule`` (same math and per-layer weight layout as
the checkpoint's ``hc_{attn,ffn}_{fn,base,scale}``). Per ``layer_types`` the
attention slot is then swapped to the KDA wrapper or the no-rope kpool DSA.

Config fields without Megatron CLI flags (hyper-connection, indexer, kpool,
gate bound) are patched onto the argparse-built config here, mirroring
``Glm5NextBridge._build_config`` -- ``convert_hf_to_torch_dist`` builds its
config from ``parse_args``, so without this the converted model would silently
have no hyper-connections (the qwen3.8-next lesson).

GLM-5.3 contracts the residual streams with a plain mean and ships no usable
``hc_head_*`` tensors (sglang skips them on load), while Megatron's
``TransformerBlock`` unconditionally applies ``learned_output_contract`` with
its own ``hc_head_*`` parameters. ``_patch_mean_output_contract`` swaps the
contraction for ``HyperConnectionModule.output_contract`` (mean) and demotes
the orphan ``hc_head_*`` parameters to plain tensors so they never reach DDP,
the distributed optimizer, checkpoints, weight sync, or the bridge's
name-mapping audit. MTP is dropped for training (rollout-side EAGLE only,
same decision as glm5.x).
"""

import copy

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import transformer_block
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.hf_config import load_hf_config
from miles_plugins.models.glm5.glm5 import DSASelfAttentionSubmodules
from miles_plugins.models.glm5_next.dsa import Glm5NextDSAAttention
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention, _get_text_config

_MHC_EPS = 1e-6

_HC_HEAD_PARAM_NAMES = ("hc_head_fn", "hc_head_base", "hc_head_scale")


def full_attn_layers(text_config) -> list[int]:
    """0-based DSA layer indices; the legacy ``linear_attn_config`` dict wins
    verbatim over ``layer_types`` (sglang precedence), with the ``i % 4 == 3``
    fallback when neither is present."""
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get("full_attn_layers") is not None:
        return sorted(int(i) for i in linear_attn_config["full_attn_layers"])
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return [i for i, layer_type in enumerate(layer_types) if layer_type != "linear_attention"]
    return [i for i in range(text_config.num_hidden_layers) if i % 4 == 3]


def _apply_glm5_next_config(config, text_config) -> None:
    """Put GLM-5.3 fields lacking CLI flags on a TransformerConfig built from
    argparse. Mirrors ``Glm5NextBridge._build_base_config`` so both paths agree."""
    hc_eps = getattr(text_config, "hc_eps", _MHC_EPS)
    assert hc_eps == _MHC_EPS, (
        f"GLM-5.3 hc_eps={hc_eps} but Megatron's mHC sinkhorn/compute-h eps are the "
        f"constants _MHC_SINKHORN_EPS=_MHC_COMPUTE_H_EPS={_MHC_EPS}"
    )
    config.enable_hyper_connections = True
    config.num_residual_streams = getattr(text_config, "hc_mult", 4)
    config.mhc_sinkhorn_iterations = getattr(text_config, "hc_sinkhorn_iters", 20)
    config.use_fused_mhc = False

    config.index_num_attention_heads = text_config.index_n_heads
    config.index_head_dim = text_config.index_head_dim
    config.index_topk = text_config.index_topk
    config.index_kpool = text_config.index_kpool
    assert (
        text_config.index_kpool > 1
        and getattr(text_config, "index_kpool_compress", False)
        and getattr(text_config, "index_kpool_always_select_tail", False)
    ), "GLM-5.3 kpool indexer expects index_kpool>1 with compress and always_select_tail"

    config.glm5_next_full_attn_layers = full_attn_layers(text_config)


def _patch_mean_output_contract() -> None:
    if getattr(transformer_block, "_glm5_next_mean_contract_patched", False):
        return

    def _mean_output_contract(hidden_states, head_fn, base, scale, n, eps):
        return HyperConnectionModule.output_contract(hidden_states, n)

    transformer_block.learned_output_contract = _mean_output_contract

    original_init = TransformerBlock.__init__

    def _init_and_demote_hc_head(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for param_name in _HC_HEAD_PARAM_NAMES:
            param = self._parameters.pop(param_name, None)
            if param is not None:
                setattr(self, param_name, param.data)

    TransformerBlock.__init__ = _init_and_demote_hc_head
    transformer_block._glm5_next_mean_contract_patched = True


def get_glm5_next_spec(args, config, vp_stage=None):
    """Transformer block spec for GLM-5.3-Flash."""
    hf_config = load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    _apply_glm5_next_config(config, text_config)
    config.freeze_indexer = getattr(args, "freeze_indexer", False)
    _patch_mean_output_contract()

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    backend = TESpecProvider()
    dsa_module_spec = ModuleSpec(
        module=Glm5NextDSAAttention,
        params={
            "attn_mask_type": AttnMaskType.causal,
            "topk_backend": args.miles_dsa_topk_backend,
        },
        submodules=DSASelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_layer_norm_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
            linear_v_up_proj=IdentityOp,
            wq_b=backend.linear(),
            wk=backend.linear(),
            k_norm=backend.layer_norm(),
            weights_proj=backend.linear(),
        ),
    )

    dsa_layers = set(config.glm5_next_full_attn_layers)
    for layer_id in range(num_layers_to_build):
        layer_spec = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
        if (layer_id + offset) in dsa_layers:
            layer_spec.submodules.self_attention = dsa_module_spec
        else:
            layer_spec.submodules.self_attention = ModuleSpec(
                module=Glm5NextKDAAttention,
                params={"args": args},
            )
        transformer_layer_spec.layer_specs[layer_id] = layer_spec

    return transformer_layer_spec
