import copy

import torch

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import hyper_connection, transformer_block
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.hf_config import load_hf_config
from miles_plugins.models.glm5.glm5 import DSASelfAttentionSubmodules
from miles_plugins.models.glm5_next.dsa import Glm5NextDSAAttention
from miles_plugins.models.glm5_next.hf_compat import register_glm5_next_config
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention, _get_text_config

_MHC_EPS = 1e-6

_HC_HEAD_PARAM_NAMES = ("hc_head_fn", "hc_head_base", "hc_head_scale")


def full_attn_layers(text_config) -> list[int]:
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get("full_attn_layers") is not None:
        return sorted(int(i) for i in linear_attn_config["full_attn_layers"])
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return [i for i, layer_type in enumerate(layer_types) if layer_type != "linear_attention"]
    return [i for i in range(text_config.num_hidden_layers) if i % 4 == 3]


def _apply_glm5_next_config(config, text_config) -> None:
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


def _reference_proj_rms(x, weight, eps):
    proj = torch.matmul(x, weight.t())
    r = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return proj, r


def _patch_reference_proj_rms() -> None:
    if getattr(hyper_connection, "_glm5_next_reference_proj_rms_patched", False):
        return

    original_init = HyperConnectionModule.__init__

    def _init_with_reference_proj_rms(self, config, layer_number):
        original_init(self, config, layer_number)
        assert not config.use_fused_mhc, "GLM-5.3 mHC requires the native (unfused) proj_rms path"
        self.norm_eps = config.layernorm_epsilon
        self._proj_rms_op = torch.compile(_reference_proj_rms)

    HyperConnectionModule.__init__ = _init_with_reference_proj_rms
    hyper_connection._glm5_next_reference_proj_rms_patched = True


def get_glm5_next_spec(args, config, vp_stage=None):
    register_glm5_next_config()
    hf_config = load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    _apply_glm5_next_config(config, text_config)
    config.freeze_indexer = getattr(args, "freeze_indexer", False)
    _patch_mean_output_contract()
    _patch_reference_proj_rms()

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "pipeline_model_parallel_layout is not supported"

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
