import copy

import torch

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import hyper_connection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.glm5.glm5 import DSASelfAttentionSubmodules
from miles_plugins.models.glm5_next.dsa import Glm5NextDSAAttention
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention, _get_text_config

_MHC_EPS = 1e-6


def full_attn_layers(text_config) -> list[int]:
    """DSA layers; the native config labels the KDA layers ``linear_attention``."""
    return [i for i, layer_type in enumerate(text_config.layer_types) if layer_type != "linear_attention"]


def _apply_glm5_next_config(config, text_config) -> None:
    hc_eps = text_config.hc_eps
    assert hc_eps == _MHC_EPS, (
        f"GLM-5.3 hc_eps={hc_eps} but Megatron's mHC sinkhorn/compute-h eps are the "
        f"constants _MHC_SINKHORN_EPS=_MHC_COMPUTE_H_EPS={_MHC_EPS}"
    )
    config.enable_hyper_connections = True
    config.num_residual_streams = text_config.hc_mult
    config.mhc_sinkhorn_iterations = text_config.hc_sinkhorn_iters
    config.use_fused_mhc = False

    config.index_num_attention_heads = text_config.index_n_heads
    config.index_head_dim = text_config.index_head_dim
    config.index_topk = text_config.index_topk
    config.index_kpool = text_config.index_kpool
    # index_kpool_compress is a checkpoint field the native config does not declare
    assert (
        text_config.index_kpool > 1
        and getattr(text_config, "index_kpool_compress", False)
        and text_config.index_kpool_always_select_tail
    ), "GLM-5.3 kpool indexer expects index_kpool>1 with compress and always_select_tail"

    config.glm5_next_full_attn_layers = full_attn_layers(text_config)


class Glm5NextMeanOutputContraction(torch.nn.Module):
    """Plain mean over the residual streams, with no learned head weights."""

    def __init__(self, config):
        super().__init__()
        self.num_streams = config.num_residual_streams

    def forward(self, hidden_states):
        return HyperConnectionModule.output_contract(hidden_states, self.num_streams)


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
    hf_config = load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    _apply_glm5_next_config(config, text_config)
    _patch_reference_proj_rms()

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
    transformer_layer_spec.hc_head_contraction = ModuleSpec(module=Glm5NextMeanOutputContraction)

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
